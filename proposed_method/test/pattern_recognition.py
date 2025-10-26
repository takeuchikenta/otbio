import wfdb
import os
import glob
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, cheb2ord, cheby2, firwin, find_peaks
from sklearn.decomposition import FastICA
from scipy.interpolate import RectBivariateSpline
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import welch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import matplotlib.cm as cm
from IPython.display import HTML
from pyclustering.cluster.gmeans import gmeans
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric
from sklearn.svm import SVC
import itertools
import json
from pathlib import Path
from collections.abc import Mapping
from typing import List, Sequence, Any, Optional
import hdbscan
import warnings
np.warnings = warnings
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -*- coding: utf-8 -*-
"""
8×8→中央6×6サブセットの学習 & 再装着時の既知ずれ(dx,dy,theta)でサブセットをずらし、
スプライン補間で取得 → 特徴抽出 → (SVM / 任意sklearn / 3D-CNN) で分類。

- モデル差し替え: ModelSpec(type='svm'|'rf'|'mlp'|'sklearn'|'cnn3d', **kwargs)
- 特徴差し替え: FeatureSpec(kind='rms'|'raw', window=..., hop=...)
  * 'rms' = 従来どおり窓RMS 36次元
  * 'raw' = 窓内の (window, 6, 6) をそのまま利用（cnn3d向け）

PyTorchが未インストールでも、sklearn系は動作します（cnn3dはスキップされます）。
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional, Any, Dict, Callable, List
from scipy.interpolate import RectBivariateSpline

# ===== 可搬な import（PyTorchは任意） =====
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


import numpy as np
from scipy.spatial import cKDTree


# ---------- 角度ユーティリティ ----------
def wrap_half_pi(theta):
    """角度を[-pi/2, pi/2]へ正規化（θ≡θ+π を同一視）"""
    t = (theta + np.pi/2) % np.pi - np.pi/2
    return t

def angdiff_half_pi(a, b):
    """d = a - b を [-pi/2, pi/2] へ折り返し"""
    return wrap_half_pi(a - b)

# ---------- 初期回転（角度ヒストグラム相関） ----------
def init_rotation_from_angles(alpha, beta, nbins=90):
    # mod π に折る
    a = (alpha % np.pi)
    b = (beta  % np.pi)
    hist_a, edges = np.histogram(a, bins=nbins, range=(0, np.pi), density=False)
    hist_b, _     = np.histogram(b, bins=nbins, range=(0, np.pi), density=False)
    # 円環相関：シフトで最大の相関を与える bin を探す
    best_shift, best_score = 0, -1
    for s in range(nbins):
        score = np.dot(hist_a, np.roll(hist_b, s))
        if score > best_score:
            best_score, best_shift = score, s
    # bin → 角度（中心）へ
    bin_width = np.pi / nbins
    dtheta0 = best_shift * bin_width
    # [-pi/2, pi/2] に折り返して返す
    return wrap_half_pi(dtheta0)

# ---------- 重み付き並進の閉形式 ----------
def weighted_translation(P_rot, Q_match, w):
    wp = np.sum(w) + 1e-12
    muP = np.sum(P_rot * w[:, None], axis=0) / wp
    muQ = np.sum(Q_match * w[:, None], axis=0) / wp
    return muQ - muP

# ---------- コスト計算 ----------
def compute_cost(P_rot, Q_match, alpha_rot, beta_match, w, lam=1.0, huber_delta=None):
    pos_err = np.linalg.norm(P_rot - Q_match, axis=1)
    ang_err = np.abs(angdiff_half_pi(alpha_rot, beta_match))
    resid2 = pos_err**2 + lam * ang_err**2

    if huber_delta is not None:
        # Huber損失（位置と角度を合わせた合成残差に適用）
        r = np.sqrt(resid2 + 1e-12)
        quad = r <= huber_delta
        cost = np.sum(w[quad] * (r[quad]**2))
        cost += np.sum(w[~quad] * (2*huber_delta*r[~quad] - huber_delta**2))
        return cost
    else:
        return np.sum(w * resid2)

# ---------- 角度つきICP（Δθ 1次元探索 + t閉形式） ----------
def estimate_shift_rotation(
    P, alpha, Q, beta,
    w=None, lam=1.0,
    max_iter=30, tol=1e-5,
    coarse_deg=2.0, refine_halfwidth_deg=5.0,
    huber_delta=None, outlier_quantile=0.90
):
    """
    P,Q: (N,2) の点群（71×71座標系）
    alpha,beta: (N,) 方向（ラジアン, θ≡θ+π）
    w: (N,) 基準点の重み（Noneなら1）
    lam: 角度項の重み（0.5〜3あたりで調整）
    huber_delta: Huber閾値（座標単位, 例 2.0〜5.0）。Noneなら二乗誤差
    outlier_quantile: 対応後の合成残差の上位何割を外れ値として無視するか（0.90=上位10%除外）
    """

    N = P.shape[0]
    if w is None:
        w = np.ones(N, dtype=float)
    w = w / (np.sum(w) + 1e-12)

    # 角度正規化
    alpha = wrap_half_pi(alpha)
    beta  = wrap_half_pi(beta)

    # 初期回転
    dtheta = init_rotation_from_angles(alpha, beta)

    # KDTree
    tree = cKDTree(Q)

    # 反復
    last_cost = np.inf
    t = np.zeros(2, dtype=float)

    for it in range(max_iter):
        # ---- 1) Δθ の近傍を粗→細に 1次元探索 ----
        # 粗いグリッド
        deg2rad = np.pi / 180.0
        grid = np.arange(dtheta - refine_halfwidth_deg*deg2rad,
                         dtheta + refine_halfwidth_deg*deg2rad + 1e-12,
                         coarse_deg*deg2rad)

        best = (None, None, None, None)  # cost, θ, t, matches
        for th in grid:
            # 回転
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s],[s, c]])
            P_rot = P @ R.T
            alpha_rot = wrap_half_pi(alpha + th)

            # 最近傍対応
            dists, idx = tree.query(P_rot + t, k=1)
            Q_match = Q[idx]
            beta_match = beta[idx]

            # 外れ値のための一時重み
            pos_err = np.linalg.norm(P_rot + t - Q_match, axis=1)
            ang_err = np.abs(angdiff_half_pi(alpha_rot, beta_match))
            mix_err = np.sqrt(pos_err**2 + lam * ang_err**2)
            cutoff = np.quantile(mix_err, outlier_quantile)
            inlier = mix_err <= cutoff
            if np.sum(inlier) < max(5, 0.2*N):
                inlier = np.argsort(mix_err)[:max(5, int(0.2*N))]  # 最低限確保

            # 並進の閉形式更新（inlierのみ）
            t_new = weighted_translation(P_rot[inlier], Q_match[inlier], w[inlier])
            cost = compute_cost(P_rot[inlier]+t_new, Q_match[inlier],
                                alpha_rot[inlier], beta_match[inlier],
                                w[inlier], lam=lam, huber_delta=huber_delta)

            if best[0] is None or cost < best[0]:
                best = (cost, th, t_new, (idx, inlier))

        cost, dtheta, t, (idx, inlier) = best

        # 収束判定
        if np.abs(last_cost - cost) < tol:
            break
        last_cost = cost

    # 出力をまとめる
    # 並進は71格子座標系のピクセル単位。8x8→71x71 は 10倍補間なので：
    # 1 px ≈ 0.1 電極間距離。電極間距離が10 mmなら 1 px = 1 mm。
    px_per_electrode = (71 - 1) / (8 - 1)  # = 10
    dx_px, dy_px = t
    dx_elec = dx_px / px_per_electrode
    dy_elec = dy_px / px_per_electrode

    return {
        "dtheta_rad": dtheta,
        "dtheta_deg": float(dtheta * 180/np.pi),
        "dx_px": float(dx_px),
        "dy_px": float(dy_px),
        "dx_electrode_units": float(dx_elec),
        "dy_electrode_units": float(dy_elec),
        "matches_inlier_ratio": float(np.mean(inlier)),
        "inlier_count": int(np.sum(inlier)),
        "N": int(N)
    }




# -*- coding: utf-8 -*-
"""
HD-sEMG 8×8 → 中央6×6サブセット学習 & 再装着時に既知の位置ずれ(並進+回転)で
サブセットをずらしてテスト時のEMGを空間スプライン補間で取得するユーティリティ。

【想定/前提】
- 入力EMGは shape = (n_samples, 8, 8)。n_samplesは時間サンプル数（例: 4秒×Fs）。
- 電極の空間サンプリングは正方格子、電極間ピッチ = spacing（既定=1.0。1ならグリッド座標は0..7）。
- 学習は“元の位置”の中央6×6（行1..6, 列1..6）で行う。
- テストでは既知の位置ずれ (dx, dy [電極間距離単位], theta [radian, 反時計回り]) を
  中央6×6サブセットの各座標に適用して、ずらした位置の時系列を
  2次元スプライン補間(RectBivariateSpline, bicubic相当)で取得する。
- 回転の中心は 8×8のグリッド中心（(7/2, 7/2) * spacing）。必要なら自由に変更可。
- 補間の境界外は2通り:
    mode='extrapolate' → スプラインの外挿（デフォルト）
    mode='clip'        → 領域外を境界にクリップして擬似的な最近傍に相当（外れ値を抑えたい場合）

【提供関数/クラス】
- extract_center_6x6(emg): central 6×6の切り出し（学習用そのまま使える）
- GridSubsetMapper: 
    - transform(emg, dx, dy, theta, mode='extrapolate'): 
        既知のずれを適用した“ずらし6×6”のEMG時系列を補間で返す (n_samples, 6, 6)
- 使い方例はファイル末尾の __main__ ブロック参照
"""

ArrayLike = np.ndarray

def extract_center_6x6(emg: ArrayLike) -> ArrayLike:
    """
    元の8×8から中央6×6(行1..6, 列1..6)を抽出。
    emg: (n_samples, 8, 8)
    return: (n_samples, 6, 6)
    """
    if emg.ndim != 3 or emg.shape[1:] != (8, 8):
        raise ValueError("emg must have shape (n_samples, 8, 8)")
    return emg[:, 1:7, 1:7]


@dataclass
class GridSubsetMapper:
    """
    8×8グリッドのEMGから、既知の並進・回転だけずらした“中央6×6サブセット”位置の
    EMG時系列を、空間スプライン補間で取得するクラス。
    """
    n_rows: int = 8
    n_cols: int = 8
    spacing: float = 1.0  # 電極間距離の空間単位（mmでも1でも可）
    rotate_about: str = "grid_center"  # 'grid_center'（推奨）/ 'subset_center'
    # 将来: 'custom' にして任意中心にしたい場合は center=(cx,cy) を追加実装

    def _grid_coords(self) -> Tuple[ArrayLike, ArrayLike]:
        """8×8グリッドの座標（行y, 列x）。"""
        y = np.arange(self.n_rows) * self.spacing  # 行インデックス方向
        x = np.arange(self.n_cols) * self.spacing  # 列インデックス方向
        return y, x

    def _subset_coords(self) -> Tuple[ArrayLike, ArrayLike]:
        """中央6×6サブセットの“元の”座標（行y, 列x）を (6,6) で返す。"""
        y_full, x_full = self._grid_coords()
        y6 = y_full[1:7]  # 行1..6
        x6 = x_full[1:7]  # 列1..6
        Y6, X6 = np.meshgrid(y6, x6, indexing="ij")  # (6,6)
        return Y6, X6

    def _rotation_center(self) -> Tuple[float, float]:
        """回転中心座標 (cy, cx) を返す。"""
        if self.rotate_about == "grid_center":
            cy = (self.n_rows - 1) * 0.5 * self.spacing
            cx = (self.n_cols - 1) * 0.5 * self.spacing
        elif self.rotate_about == "subset_center":
            # 中央6×6の中心（行/列1..6の中心）= 3.5 → グリッド座標では index=1..6 の中心は (3.5, 3.5)
            cy = 3.5 * self.spacing
            cx = 3.5 * self.spacing
        else:
            raise ValueError("rotate_about must be 'grid_center' or 'subset_center'")
        return cy, cx

    @staticmethod
    def _apply_se2(Y: ArrayLike, X: ArrayLike, cy: float, cx: float,
                   dy: float, dx: float, theta: float) -> Tuple[ArrayLike, ArrayLike]:
        """
        2Dの相似変換（回転+並進、スケール=1）。
        入出力: (6,6)の座標格子。回転は (cx,cy) を中心。dx,dy は並進（x右+, y下+）。
        """
        # 平行移動して原点を回転中心へ
        Y0 = Y - cy
        X0 = X - cx
        c, s = np.cos(theta), np.sin(theta)
        # 反時計回り回転
        Xr =  c * X0 - s * Y0
        Yr =  s * X0 + c * Y0
        # 元の位置に戻して並進
        Xp = Xr + cx + dx
        Yp = Yr + cy + dy
        return Yp, Xp

    def _build_spline(self, frame: ArrayLike) -> RectBivariateSpline:
        """
        1フレーム(8×8)からRectBivariateSplineを作成。
        RectBivariateSplineの引数順に注意: (y, x, z)
        """
        if frame.shape != (self.n_rows, self.n_cols):
            raise ValueError("frame must be (8,8)")
        y, x = self._grid_coords()
        # kx=3, ky=3 → bicubic。s=0で補間（平滑化なし）
        return RectBivariateSpline(y, x, frame, kx=3, ky=3, s=0)

    def transform(
        self,
        emg: ArrayLike,
        dx: float,
        dy: float,
        theta: float,
        mode: Literal["extrapolate", "clip"] = "extrapolate",
    ) -> ArrayLike:
        """
        既知の位置ずれ (dx,dy,theta) を中央6×6サブセットに適用し、
        その“ずらした位置”でのEMG時系列をスプライン補間で取得する。

        emg : (n_samples, 8, 8)
        dx, dy : 並進。+xは列方向(右)、+yは行方向(下)。単位は self.spacing に合わせる。
        theta : 反時計回りの回転 [radian]。
        mode : 'extrapolate'（外挿許容） or 'clip'（境界にクリップ）

        return : (n_samples, 6, 6)
        """
        if emg.ndim != 3 or emg.shape[1:] != (self.n_rows, self.n_cols):
            raise ValueError("emg must have shape (n_samples, 8, 8)")

        n_samples = emg.shape[0]
        Y6, X6 = self._subset_coords()           # (6,6) 元のサブセット座標
        cy, cx = self._rotation_center()

        # サブセット座標に対してずれ(回転+並進)を適用
        Yt, Xt = self._apply_se2(Y6, X6, cy, cx, dy, dx, theta)  # (6,6)

        # クリップ指定なら境界内に丸め込む
        if mode == "clip":
            y0, y1 = 0.0, (self.n_rows - 1) * self.spacing
            x0, x1 = 0.0, (self.n_cols - 1) * self.spacing
            Yq = np.clip(Yt, y0, y1)
            Xq = np.clip(Xt, x0, x1)
        elif mode == "extrapolate":
            Yq, Xq = Yt, Xt
        else:
            raise ValueError("mode must be 'extrapolate' or 'clip'")

        # (n_samples, 6, 6) を格納
        out = np.empty((n_samples, 6, 6), dtype=emg.dtype)

        # 各時間サンプルごとに2Dスプラインで補間
        # RectBivariateSpline.ev はベクトル化可能：ev(Y.flatten(), X.flatten()) → reshape
        Y_flat = Yq.ravel()
        X_flat = Xq.ravel()
        for i in range(n_samples):
            spl = self._build_spline(emg[i])         # frame=(8,8)
            zi = spl.ev(Y_flat, X_flat).reshape(6, 6)
            out[i] = zi

        return out
    

# 1) 追加インポート
from dataclasses import dataclass, field
from typing import Literal, Dict, Any, Optional
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# PyTorch は任意（2D/3D CNN 用）
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 一時的にGPUを無効化してCPUのみで実行
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# torch.backends.cudnn.enabled = False


# ========= 特徴抽出と学習/推論パイプライン =========
def segment_time_series(emg_6x6: np.ndarray, window: int, hop: int) -> np.ndarray:
    """
    (n_samples, 6, 6) を窓分割して (n_windows, window, 6, 6) へ。
    """
    n, _, _ = emg_6x6.shape
    idx_starts = np.arange(0, max(1, n - window + 1), hop)
    segs = np.stack([emg_6x6[s:s+window] for s in idx_starts if s + window <= n], axis=0)
    # segs: (n_windows, window, 6, 6)
    return segs

# 2) 追加：特徴量仕様と複数特徴の実装
@dataclass
class FeatureSpec:
    """
    kind:
      - 'rms'      : 各窓のRMSマップ (6x6) → flattenで36次元（sklearn向け）
      - 'arv'      : 各窓のARV(平均絶対値) → 36次元
      - 'var'      : 各窓の分散 → 36次元
      - 'zc'       : Zero-Crossing数（閾値なしの符号反転カウント）→ 36次元
      - 'rms_map'  : 各窓のRMSマップをそのまま (6,6)（2D-CNN向け）
      - 'raw'      : 生の (window,6,6)（3D-CNN向け）
    """
    kind: Literal['rms', 'arv', 'var', 'zc', 'rms_map', 'raw'] = 'rms'
    window: int = 200
    hop: int = 50

def window_maps(segs: np.ndarray, reducer: Literal['rms','arv','var']) -> np.ndarray:
    # segs: (n_w, T, 6, 6) -> (n_w, 6, 6)
    if reducer == 'rms':
        return np.sqrt(np.mean(segs**2, axis=1))
    if reducer == 'arv':
        return np.mean(np.abs(segs), axis=1)
    if reducer == 'var':
        return np.var(segs, axis=1, ddof=0)
    raise ValueError("unknown reducer")

def window_zc(segs: np.ndarray) -> np.ndarray:
    # ゼロクロッシング（符号変化数）。(n_w, T, 6,6) -> (n_w, 6,6)
    s = np.sign(segs)                  # -1,0,1
    s[s==0] = 1                        # 0を非反転扱いに
    zc = (np.diff(s, axis=1) != 0).sum(axis=1)
    return zc.astype(np.float32)

def extract_features(emg_6x6: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    segs = segment_time_series(emg_6x6, window=spec.window, hop=spec.hop)  # (n_w, T, 6,6)
    if spec.kind in ('rms','arv','var'):
        maps = window_maps(segs, 'rms' if spec.kind=='rms' else spec.kind)  # (n_w,6,6)
        return maps.reshape(maps.shape[0], -1)                               # (n_w,36)
    if spec.kind == 'zc':
        maps = window_zc(segs)                                              # (n_w,6,6)
        return maps.reshape(maps.shape[0], -1)                               # (n_w,36)
    if spec.kind == 'rms_map':
        return window_maps(segs, 'rms')                                      # (n_w,6,6)  # 2D-CNN
    if spec.kind == 'raw':
        return segs                                                          # (n_w,T,6,6) # 3D-CNN
    raise ValueError("Unsupported FeatureSpec.kind")


# def window_rms(segs: np.ndarray) -> np.ndarray:
#     """
#     segs: (n_windows, window, 6, 6) → 各チャネルのRMSを窓内で計算 → (n_windows, 36)
#     """
#     # 時間方向に RMS
#     rms = np.sqrt(np.mean(segs**2, axis=1))       # (n_windows, 6, 6)
#     feats = rms.reshape(rms.shape[0], -1)         # (n_windows, 36)
#     return feats

# def features_from_emg(emg_6x6: np.ndarray, window: int, hop: int) -> np.ndarray:
#     """中央6×6の時系列から、窓RMS(36次元)の列を返す。"""
#     segs = segment_time_series(emg_6x6, window=window, hop=hop)
#     X = window_rms(segs)
#     return X  # (n_windows, 36)

# 3) 追加：モデル仕様と sklearn/CNN ラッパ
@dataclass
class ModelSpec:
    """
    type:
      - 'svm'   : SVC
      - 'lda'   : 線形判別分析
      - 'rf'    : RandomForest
      - 'mlp'   : MLPClassifier
      - 'sklearn': 任意の sklearn 推定器（kwargs に estimator を渡す）
      - 'cnn2d' : 2D-CNN（FeatureSpec.kind='rms_map' を要求）
      - 'cnn3d' : 3D-CNN（FeatureSpec.kind='raw' を要求）
    kwargs: 各モデル固有のハイパラ（例：C, n_estimators, hidden_layer_sizes, epochs 等）
    """
    type: Literal['svm','lda','rf','mlp','sklearn','cnn2d','cnn3d'] = 'svm'
    kwargs: Dict[str, Any] = field(default_factory=dict)

class SklearnClassifier:
    def __init__(self, model_spec: ModelSpec, use_scaler: bool = True):
        self.model_spec = model_spec
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None
        self.clf = self._build()

    def _build(self):
        t = self.model_spec.type
        kw = dict(self.model_spec.kwargs)
        if t == 'svm':
            return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, **kw)
        if t == 'lda':
            return LDA(**kw)
        if t == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=300, random_state=0, **kw)
        if t == 'mlp':
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, random_state=0, **kw)
        if t == 'sklearn':
            if 'estimator' not in kw:
                raise ValueError("For type='sklearn', pass estimator=... in kwargs")
            return kw['estimator']
        raise ValueError("Unsupported sklearn model type")

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.fit_transform(X) if self.scaler is not None else X
        self.clf.fit(Xs, y); return self

    def predict(self, X: np.ndarray):
        Xs = self.scaler.transform(X) if self.scaler is not None else X
        y = self.clf.predict(Xs)
        proba = None
        try: proba = self.clf.predict_proba(Xs)
        except Exception: pass
        return y, proba

# ---- 2D CNN（入力: (N, 6, 6) → (N,1,6,6)）----
class CNN2DClassifier:
    def __init__(self, n_classes: int, epochs: int=10, batch_size:int=64, lr:float=1e-3,
                 device: Optional[str]=None, verbose: bool=True):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for cnn2d")
        self.n_classes=n_classes; self.epochs=epochs; self.batch_size=batch_size
        self.lr=lr; self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose=verbose
        self.model = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16,32,kernel_size=3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(32,n_classes)
        ).to(self.device)

    def _to_tensor(self, X_map: np.ndarray, y: Optional[np.ndarray]=None):
        # X_map: (N,6,6) -> (N,1,6,6)
        Xt = torch.from_numpy(X_map.astype(np.float32)).unsqueeze(1).to(self.device)
        if y is None: return Xt, None
        yt = torch.from_numpy(y.astype(np.int64)).to(self.device)
        return Xt, yt

    def fit(self, X_map: np.ndarray, y: np.ndarray):
        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        X,Y = self._to_tensor(X_map, y)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,Y),
                                         batch_size=self.batch_size, shuffle=True)
        for ep in range(1,self.epochs+1):
            tot, corr, loss_sum = 0,0,0.0
            for xb,yb in dl:
                opt.zero_grad(); logits=self.model(xb); loss=crit(logits,yb)
                loss.backward(); opt.step()
                loss_sum += loss.item()*xb.size(0)
                corr += (logits.argmax(1)==yb).sum().item(); tot += xb.size(0)
            if self.verbose: print(f"[cnn2d] epoch {ep}/{self.epochs} loss={loss_sum/tot:.4f} acc={corr/tot:.3f}")
        return self

    @torch.no_grad()
    def predict(self, X_map: np.ndarray):
        self.model.eval()
        X,_ = self._to_tensor(X_map, None)
        logits = []
        for i in range(0, X.size(0), self.batch_size):
            logits.append(self.model(X[i:i+self.batch_size]).cpu().numpy())
        logits = np.concatenate(logits, axis=0)
        y_pred = logits.argmax(1)
        proba = np.exp(logits - logits.max(1, keepdims=True)) ; proba /= proba.sum(1, keepdims=True)
        return y_pred, proba

# ---- 3D CNN（入力: (N,T,6,6) → (N,1,T,6,6)）----
class Simple3DCNN(nn.Module):
    def __init__(self, n_classes:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1,16,(5,3,3),stride=(2,1,1),padding=(2,1,1)), nn.BatchNorm3d(16), nn.ReLU(True),
            nn.Conv3d(16,32,(5,3,3),stride=(2,1,1),padding=(2,1,1)), nn.BatchNorm3d(32), nn.ReLU(True),
            nn.Conv3d(32,64,(3,3,3),stride=(2,1,1),padding=(1,1,1)), nn.BatchNorm3d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.head = nn.Linear(64, n_classes)
    def forward(self,x):
        x=self.net(x).flatten(1); return self.head(x)

class CNN3DClassifier:
    def __init__(self, n_classes:int, epochs:int=10, batch_size:int=32, lr:float=1e-3,
                 device: Optional[str]=None, verbose: bool=True):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for cnn3d")
        self.n_classes=n_classes; self.epochs=epochs; self.batch_size=batch_size
        self.lr=lr; self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose=verbose; self.model=Simple3DCNN(n_classes).to(self.device)

    def _to_tensor(self, X_raw: np.ndarray, y: Optional[np.ndarray]=None):
        Xt = torch.from_numpy(X_raw.astype(np.float32)).unsqueeze(1).to(self.device)
        if y is None: return Xt,None
        yt = torch.from_numpy(y.astype(np.int64)).to(self.device)
        return Xt, yt

    def fit(self, X_raw: np.ndarray, y: np.ndarray):
        self.model.train(); opt=optim.Adam(self.model.parameters(), lr=self.lr)
        crit=nn.CrossEntropyLoss()
        X,Y = self._to_tensor(X_raw, y)
        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X,Y),
                                         batch_size=self.batch_size, shuffle=True)
        for ep in range(1,self.epochs+1):
            tot,corr,loss_sum=0,0,0.0
            for xb,yb in dl:
                opt.zero_grad(); logits=self.model(xb); loss=crit(logits,yb)
                loss.backward(); opt.step()
                loss_sum += loss.item()*xb.size(0)
                corr += (logits.argmax(1)==yb).sum().item(); tot += xb.size(0)
            if self.verbose: print(f"[cnn3d] epoch {ep}/{self.epochs} loss={loss_sum/tot:.4f} acc={corr/tot:.3f}")
        return self

    @torch.no_grad()
    def predict(self, X_raw: np.ndarray):
        self.model.eval(); X,_ = self._to_tensor(X_raw, None)
        logits=[]
        for i in range(0, X.size(0), self.batch_size):
            logits.append(self.model(X[i:i+self.batch_size]).cpu().numpy())
        logits=np.concatenate(logits, axis=0); y_pred=logits.argmax(1)
        proba = np.exp(logits - logits.max(1, keepdims=True)) ; proba /= proba.sum(1, keepdims=True)
        return y_pred, proba


# （置換）GestureClassifier を「特徴量は外部で作る」設計に変更
class GestureClassifier:
    """
    役割：
      - fit(X, y):  前処理/特徴抽出済みの特徴量で学習
      - predict(X): 前処理/特徴抽出済みの特徴量で推論
      - evaluate(X, y): 推論＋精度
    注意：
      - sklearn系モデルは use_scaler=True の場合、内部で標準化（trainでfit→testでtransform）
      - CNN系は標準化なし（入力形状の整合のみユーザ側で担保）
    """
    def __init__(self,
                 model_spec: ModelSpec = ModelSpec(type='svm'),
                 use_scaler: bool = True):
        self.model_spec = model_spec
        self.use_scaler = use_scaler
        self._clf = None
        self._n_classes = None
        self._fitted = False

    def _build_clf(self, n_classes: int):
        t = self.model_spec.type
        if t in ('svm','lda','rf','mlp','sklearn'):
            return SklearnClassifier(self.model_spec, use_scaler=self.use_scaler)
        if t == 'cnn2d':
            return CNN2DClassifier(n_classes=n_classes, **self.model_spec.kwargs)
        if t == 'cnn3d':
            return CNN3DClassifier(n_classes=n_classes, **self.model_spec.kwargs)
        raise ValueError("Unsupported model type")

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(y) != len(X):
            raise ValueError(f"y length ({len(y)}) must match X length ({len(X)})")
        self._n_classes = int(len(np.unique(y)))
        self._clf = self._build_clf(n_classes=self._n_classes)
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        y_pred, proba = self._clf.predict(X_test)
        return y_pred, proba

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        y_pred, proba = self.predict(X_test)
        if len(y_test) != len(y_pred):
            raise ValueError(f"y_test length ({len(y_test)}) != #predictions ({len(y_pred)})")
        acc = float(np.mean(y_pred == y_test))
        return y_pred, proba, acc


# === ヘルパ ===
def _json_load_if_needed(x):
    """セル文字列がJSONならPythonオブジェクトに戻す。空欄/NaNはNone。"""
    if pd.isna(x):
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith('[') or s.startswith('{') or s.startswith('"'):
            try:
                return json.loads(s)
            except Exception:
                return x  # JSONでなければそのまま
    return x

def load_records_from_csv(path, json_cols=None, numpy_cols=None, encoding='utf-8-sig'):
    """
    path: CSVパス（.gzなど圧縮もOK：compressionは自動推定）
    json_cols: JSON文字列として保存したカラム名のリスト（例: 配列/辞書）
    numpy_cols: その中でNumPy配列に戻したいカラム名（リスト→np.arrayへ）

    return: (df, records_list_of_dicts)
    """
    df = pd.read_csv(path, encoding=encoding)  # 圧縮は自動推定されます（.gzなど）
    json_cols = json_cols or []
    numpy_cols = numpy_cols or []

    # JSON復元
    for c in json_cols:
        if c in df.columns:
            df[c] = df[c].apply(_json_load_if_needed)

    # NumPy配列化
    for c in numpy_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda v: np.array(v) if isinstance(v, list) else v)

    return df, df.to_dict(orient='records')

# === 使い方1: Googleドライブに保存したCSVを読む ===
csv_path = 'output/features/test1_zc_gmm_features.csv'       # または 'records.csv.gz'
json_cols=['virtual_bipolars', 'labels', 'center_direction', 'features']
numpy_cols=['virtual_bipolars', 'center_direction', 'features']
df, records = load_records_from_csv(csv_path,json_cols=json_cols,numpy_cols=numpy_cols)
print('rows:', len(records))
# print(type(records[0]['emg_data']))  # <class 'numpy.ndarray'> を想定
# df.head()

# === 使い方2: 手元PCからアップロードして読む ===
# from google.colab import files
# uploaded = files.upload()  # ダイアログでCSVを選択
# for name in uploaded.keys():
#     df, records = load_records_from_csv(
#         name,
#         json_cols=['emg_data', 'virtual_bipolars', 'labels', 'center_direction'],
#         numpy_cols=['emg_data', 'virtual_bipolars', 'center_direction']
#     )
#     print(name, 'rows:', len(records))
#     break  # 最初の1個だけ読みたい場合

vpolars = records

file_name1 = 'pr_dataset/subject01_session1/maintenance_preprocess_sample1'
file_name2 = 'pr_dataset/subject01_session2/maintenance_preprocess_sample1'
electrode_place = 'FD'  # 'ED', 'EP', 'FD', 'FP'

vpolar_session1 = [vpolar for vpolar in vpolars if vpolar['file_name'] == file_name1 and vpolar['electrode_place'] == electrode_place][0]
vpolar_session2 = [vpolar for vpolar in vpolars if vpolar['file_name'] == file_name2 and vpolar['electrode_place'] == electrode_place][0]

record_session1 = wfdb.rdrecord(file_name1)
record_session2 = wfdb.rdrecord(file_name2)

if vpolar_session1['electrode_place'] == 'ED':
    emg_data_session1 = record_session1.p_signal[:,:64] #Extensor Distal
    emg_data_session2 = record_session2.p_signal[:,:64] #Extensor Distal
elif vpolar_session1['electrode_place'] == 'EP':
    emg_data_session1 = record_session1.p_signal[:,64:128] #Extensor Proximal
    emg_data_session2 = record_session2.p_signal[:,64:128] #Extensor Proximal
elif vpolar_session1['electrode_place'] == 'FD': 
    emg_data_session1 = record_session1.p_signal[:,128:192] #Flexor Distal
    emg_data_session2 = record_session2.p_signal[:,128:192] #Flexor Distal
elif vpolar_session1['electrode_place'] == 'FP':
    emg_data_session1 = record_session1.p_signal[:,192:256] #Flexor Proximal
    emg_data_session2 = record_session2.p_signal[:,192:256] #Flexor Proximal

# virtual_bipolars_session1 = vpolar_session1['virtual_bipolars']
# virtual_bipolars_session2 = vpolar_session2['virtual_bipolars']
# labels_session1 = vpolar_session1['labels']
# labels_session2 = vpolar_session2['labels']
# center_direction_session1 = vpolar_session1['center_direction']
# center_direction_session2 = vpolar_session2['center_direction']

# if vpolar_session1['n_virtual_bipolars'] == 1:
#     virtual_bipolars_session1.append([0,0,0,0])
#     labels_session1.append(999)
#     center_direction_session1.append([0,0,0])

# virtual_emg_session1 = get_virtual_emg(emg_data_session1, virtual_bipolars_session1)
P_ref = []
a_ref = []
for i in range(vpolar_session1['features'].shape[0]):
    P_ref.append((vpolar_session1['features'][i,0]*10, vpolar_session1['features'][i,1]*10))
    a_ref.append(np.radians(vpolar_session1['features'][i,2]))
P_ref = np.array(P_ref)
a_ref = np.array(a_ref)

# virtual_emg_session2 = get_virtual_emg(emg_data_session2, virtual_bipolars_session2)
Q_test = []
b_test = []
for i in range(vpolar_session2['features'].shape[0]):
    Q_test.append((vpolar_session2['features'][i,0]*10, vpolar_session2['features'][i,1]*10))
    b_test.append(np.radians(vpolar_session2['features'][i,2]))
Q_test = np.array(Q_test)
b_test = np.array(b_test)

# P_ref:(N,2), a_ref:(N,), Q_test:(M,2), b_test:(M,)
res = estimate_shift_rotation(
    P_ref, a_ref, Q_test, b_test, w=None,
    lam=1.0, max_iter=40, coarse_deg=2.0,
    refine_halfwidth_deg=6.0, huber_delta=3.0,
    outlier_quantile=0.90
)
print(res)

subject_No = 1


# 学習データ（68試行 = 34ジェスチャ×2施行など）
emg_list_train = []  # 各要素: shape=(T,8,8)
y_list_train   = []  # ファイル名から抽出したラベル（長さ = 試行数）
# 推論データ（同一装着条件で複数試行）
emg_list_test = []
y_list_test   = []

i = subject_No-1
for j in range(2):
    lines = []
    gestures= open('pr_dataset/subject{:02}'.format(i+1) + '_session' + str(j+1) + '/label_maintenance.txt', 'r')
    line = gestures.read()
    for l in line.split(','):
        lines.append(l.strip())
    gestures.close()
    gesture = 0
    for k, line in enumerate(lines): #for k, line in enumerate(lines):
        if gesture == line:
            trial = 2
        else:
            trial = 1
        gesture = line
        record_name = 'pr_dataset/subject{:02}'.format(i+1) + '_session' + str(j+1) + '/maintenance_preprocess_sample'+str(k+1)
        # lists.append({'file_name': record_name, 'gesture': gesture, 'trial': trial, 'subject': i+1, 'session': j+1})
        print(record_name) # ファイル名
        print(gesture)
        try:
            record = wfdb.rdrecord(record_name)
            
            if electrode_place == 'ED':
                emg_data = record.p_signal[:,:64] #Extensor Distal
            elif electrode_place == 'EP':
                emg_data = record.p_signal[:,64:128] #Extensor Proximal
            elif electrode_place == 'FD': 
                emg_data = record.p_signal[:,128:192] #Flexor Distal
            elif electrode_place == 'FP':
                emg_data = record.p_signal[:,192:256] #Flexor Proximal
            
            if j ==0: #train
                emg_list_train.append(emg_data.reshape(-1,8,8))
                y_list_train.append(gesture)
            elif j ==1: #test
                emg_list_test.append(emg_data.reshape(-1,8,8))
                y_list_test.append(gesture)
        except FileNotFoundError:
            pass


# ===== 使い方の例（ダミーデータでのデモ） =====
# if __name__ == "__main__":
np.random.seed(0)

# --- 窓パラメータ（要データ合わせ） ---
window = 200   # サンプル幅（例：100ms @ 2kHz）
hop    = 50    # ホップ（例：25ms）
kind = 'rms'  # 'rms' or 'mav' or 'waveform_length' etc.
# X_train = []
# y_train = []
for i, (emg_train_8x8, label) in enumerate(zip(emg_list_train, y_list_train)):
    # --- ラベル（学習用）：あなたのラベリングに置き換えてください ---
    # 中央6×6から特徴抽出した個数に合わせる必要があります。
    tmp_X = extract_features(extract_center_6x6(emg_train_8x8), FeatureSpec(kind=kind, window=window, hop=hop))  # (n_windows, 36)
    n_windows = len(tmp_X)
    # 例：ダミーの 3 クラスを周回（実際はジェスチャーIDに差し替え）
    tmp_y = [int(label)-1 for i in range(n_windows)]

    if len(tmp_y) != len(tmp_X):
        raise ValueError(f"y_train length ({len(tmp_y)}) must match number of windows ({len(tmp_X)})")
    
    if i == 0:
        X_train = tmp_X
        y_train = tmp_y
    else:
        X_train = np.vstack([X_train, tmp_X])
        y_train = np.hstack([y_train, tmp_y])

# print(y_train)
# print(np.unique(y_train))
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
model = GestureClassifier(model_spec=ModelSpec(type='rf'))
# 学習直前（あなたの学習スクリプト側）
# sanity_check_labels(y_train, name="train")
# torch.backends.cudnn.enabled = False
model.fit(X_train, y_train)


# --- テスト用：再装着時の既知ずれを適用して、ずらした位置の6×6を取得 ---
# 例) 並進 dx=+0.4, dy=-0.2 (電極間距離=1.0の単位) , 回転 theta=+10度
dx, dy = res['dx_electrode_units'], res['dy_electrode_units']
theta = res['dtheta_rad']
spacing = 1.0
rotate_about = "grid_center"  # or "top_left"
mode = "extrapolate"          # or "clip"

# X_test = []
# y_test = []
for i, (emg_test_8x8, label) in enumerate(zip(emg_list_test, y_list_test)):
    # --- ラベル（学習用）：あなたのラベリングに置き換えてください ---
    # 中央6×6から特徴抽出した個数に合わせる必要があります。
    mapper = GridSubsetMapper(spacing=spacing, rotate_about=rotate_about)
    emg_shifted_6x6 = mapper.transform(emg_test_8x8, dx=dx, dy=dy, theta=theta, mode=mode)  # (n,6,6)

    tmp_X = extract_features(emg_shifted_6x6, FeatureSpec(kind=kind, window=window, hop=hop))  # (m,36)
    n_windows = len(tmp_X)
    # 例：ダミーの 3 クラスを周回（実際はジェスチャーIDに差し替え）
    tmp_y = [int(label)-1 for i in range(n_windows)]

    if len(tmp_y) != len(tmp_X):
        raise ValueError(f"y_train length ({len(y_train)}) must match number of windows ({len(X)})")
    
    if i ==0:
        X_test = tmp_X
        y_test = tmp_y
    else:
        X_test = np.vstack([X_test, tmp_X])
        y_test = np.hstack([y_test, tmp_y])

y_pred, proba, acc = model.evaluate(X_test, y_test)
print(f"y_pred shape = {y_pred.shape}, accuracy = {acc}")