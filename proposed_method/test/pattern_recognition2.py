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

#---------- w付きでかつ、初期回転のあら推定ありで、小角近似ありの位置合わせ ----------
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


#---------- w付きでかつ、初期回転のあら推定無しで、小角近似なしの位置合わせ ----------
def wrap_half_pi(th):
    return (th + np.pi/2) % np.pi - np.pi/2

def estimate_w_noinit_nosmallangle(P, alpha, Q, beta, w=None,
                                   lam=1.0, max_iter=30,
                                   deg_step=1.0, huber_delta=None,
                                   trim_q=0.90):
    # 正規化
    alpha = wrap_half_pi(alpha)
    beta  = wrap_half_pi(beta)
    N = P.shape[0]
    if w is None:
        w = np.ones(N, dtype=float)
    w = np.asarray(w, float); w /= (w.sum() + 1e-12)

    tree = cKDTree(Q)
    dtheta, t = 0.0, np.zeros(2)
    last_cost = np.inf
    rad_step = np.deg2rad(deg_step)
    angle_grid = np.arange(-np.pi/2, np.pi/2 + 1e-12, rad_step)

    for _ in range(max_iter):
        # 対応（現時点の姿勢で）
        c, s = np.cos(dtheta), np.sin(dtheta)
        Rk = np.array([[c, -s],[s,  c]])
        Pk = P @ Rk.T
        dists, idx = tree.query(Pk + t, k=1)
        Qm = Q[idx]; betam = beta[idx]

        best = (None, None, None, None)  # cost, theta, t_new, inlier_mask

        for th in angle_grid:
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s],[s, c]])
            Pr = P @ R.T
            alr = wrap_half_pi(alpha + th)

            # 暫定並進（重み付き重心）
            muP = np.sum(Pr * w[:,None], axis=0)
            muQ = np.sum(Qm * w[:,None], axis=0)
            t_new = muQ - muP

            # 合成残差
            pos = np.linalg.norm(Pr + t_new - Qm, axis=1)
            ang = 1.0 - np.cos(2.0 * wrap_half_pi(alr - betam))  # 小角近似なし
            r = np.sqrt(pos**2 + lam * ang)

            # トリム
            cutoff = np.quantile(r, trim_q)
            inlier = r <= cutoff
            if inlier.sum() < max(5, int(0.2*len(r))):
                inlier = np.argsort(r)[:max(5, int(0.2*len(r)))]

            # Huber（任意）
            if huber_delta is not None:
                rho = np.ones_like(r)
                mask = r > huber_delta
                rho[mask] = (2*huber_delta*r[mask] - huber_delta**2) / (r[mask]**2 + 1e-12)
                weff = w * rho
                weff /= (weff.sum() + 1e-12)
            else:
                weff = w

            # 重み付き重心を inlier で作り直し
            t_new = (np.sum((Pr[inlier])*weff[inlier,None], axis=0) *
                     0 + np.sum(Qm[inlier]*weff[inlier,None], axis=0) -
                     np.sum(Pr[inlier]*weff[inlier,None], axis=0))

            # コスト（inlier・weff）
            pos_i = np.linalg.norm(Pr[inlier] + t_new - Qm[inlier], axis=1)
            ang_i = 1.0 - np.cos(2.0 * wrap_half_pi(alr[inlier] - betam[inlier]))
            cost = float(np.sum(weff[inlier] * (pos_i**2 + lam * ang_i)))

            if best[0] is None or cost < best[0]:
                best = (cost, th, t_new, inlier)

        cost, dtheta, t, inlier = best
        if abs(last_cost - cost) < 1e-6:
            break
        last_cost = cost

    # 出力（71×71座標→電極単位への換算も添付）
    px_per_elec = (71 - 1) / (8 - 1)  # =10
    return {
        "dtheta_rad": float(dtheta),
        "dtheta_deg": float(dtheta*180/np.pi),
        "dx_px": float(t[0]), "dy_px": float(t[1]),
        "dx_electrode_units": float(t[0]/px_per_elec),
        "dy_electrode_units": float(t[1]/px_per_elec),
        "inlier_count": int(np.sum(inlier))
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


# ========= TD-PSD特徴量 =========
def waveform_length(signal):
    return np.sum(np.abs(np.diff(signal, axis=0)), axis=0)

def rms_feat(x):
    return np.sqrt(np.mean(np.square(x), axis=0))

def wl_feat(x):
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

def zero_crossings(x, thresh):
    # count sign changes where both samples exceed threshold in magnitude
    x1 = x[:-1, :]
    x2 = x[1:, :]
    cond = (np.abs(x1) > thresh) & (np.abs(x2) > thresh) & (np.sign(x1) != np.sign(x2))
    return np.sum(cond, axis=0).astype(float)

def slope_sign_changes(x, thresh):
    # count SSC where middle sample forms a slope sign change and all magnitudes exceed threshold
    x_prev = x[:-2, :]
    x_mid  = x[1:-1, :]
    x_next = x[2:, :]
    cond_mag = (np.abs(x_prev) > thresh) & (np.abs(x_mid) > thresh) & (np.abs(x_next) > thresh)
    cond_ssc = (((x_mid - x_prev) * (x_mid - x_next)) > 0)
    return np.sum(cond_mag & cond_ssc, axis=0).astype(float)

# ---- WAMP (Willison Amplitude) ----
def wamp_feat(x, thresh):
    """
    Willison Amplitude (WAMP)
    x: (N, C) 窓データ
    thresh: (C,) しきい値（各チャネル）
    return: (C,) 各チャネルのWAMP
    """
    dx = np.abs(np.diff(x, axis=0))      # (N-1, C)
    return np.sum(dx > thresh, axis=0).astype(float)

# === TD-PSD特徴量 ===

_EPS = 1e-12

def _hoyer_sparsity(x):
    """0〜1の疎度。0=等値ベクトル、1=スパース"""
    x = np.asarray(x)
    n = x.size
    l1 = np.linalg.norm(x, 1)
    l2 = np.linalg.norm(x, 2) + _EPS
    return (np.sqrt(n) - (l1 / l2)) / (np.sqrt(n) - 1 + _EPS)

def _moments_and_derivatives(sig, alpha=0.1):
    """0,1,2,4次微分に基づくモーメント（L2ノルム）を計算し、alpha乗で正規化。"""
    # 微分
    dx1 = np.gradient(sig)
    dx2 = np.gradient(dx1)
    dx4 = np.gradient(np.gradient(dx2))

    # Root-squared moments (式(4)〜(6))
    m0 = np.sqrt(np.sum(sig**2))
    m2 = np.sqrt(np.sum(dx1**2))
    m4 = np.sqrt(np.sum(dx2**2))

    # ノイズ感度低減のための power 変換（式(7)）
    m0 = np.power(m0 + _EPS, alpha)
    m2 = np.power(m2 + _EPS, alpha)
    m4 = np.power(m4 + _EPS, alpha)

    return m0, m2, m4, dx1, dx2

def _six_base_features(sig, alpha=0.1):
    """
    基底6特徴（式(8)〜(11)）:
      f1 = m0
      f2 = m2/m0
      f3 = m4/m2
      f4 = Sparseness（Hoyer型）
      f5 = Irregularity Factor（momentsのみで表現）
      f6 = Waveform Length Ratio = WL(dx1)/WL(dx2)
    """
    m0, m2, m4, dx1, dx2 = _moments_and_derivatives(sig, alpha=alpha)

    # (8) 系: 比特徴
    f1 = m0
    f2 = m2 / (m0 + _EPS)
    f3 = m4 / (m2 + _EPS)

    # (9) 疎度：信号の振幅分布に対するHoyer疎度
    f4 = _hoyer_sparsity(np.abs(sig))

    # (10) IF：ゼロ交差頻度/ピーク頻度 ≈ sqrt( (m2^2)/(m0*m4) )
    f5 = np.sqrt((m2**2) / (m0 * m4 + _EPS))

    # (11) WL比
    wl1 = np.sum(np.abs(dx1))
    wl2 = np.sum(np.abs(dx2)) + _EPS
    f6 = wl1 / wl2

    return np.array([f1, f2, f3, f4, f5, f6], dtype=float)

def _log_map(x):
    """対数スケール（符号は保持）。"""
    return np.sign(x) * np.log1p(np.abs(x))

def td_psd_multichannel(
    window, 
    fs=2000, 
    alpha=0.1, 
    mode="vector",
    m0_channel_norm=False
):
    """
    TD-PSD（式(4)〜(12)）を複数チャネル窓データに適用。

    Parameters
    ----------
    window : ndarray, shape (n_samples, n_channels)
        1窓ぶんのEMG。チャネルは列方向。
    fs : float
        サンプリング周波数（式自体はfsに依存しない）
    alpha : float
        式(7)のpower正規化指数（論文は 0.1）
    mode : {"scalar", "vector"}
        "scalar": 各ch 1次元（cosθ）
        "vector": 各ch 6次元（正規化f,gの要素積。総和はcosθ）
    m0_channel_norm : bool
        式(4)注記の「0次モーメントの全チャネル正規化」を有効化

    Returns
    -------
    feats : ndarray, shape (n_channels, D)
        D = 1（scalar）または 6（vector）
    """
    X = np.asarray(window)
    assert X.ndim == 2, "window must be (n_samples, n_channels)"
    n, C = X.shape

    # 平均除去
    Xz = X - X.mean(axis=0, keepdims=True)
    Xlog = _log_map(Xz)

    # まず各chの基底6特徴（原）と（log）を作る
    F = np.zeros((C, 6), dtype=float)
    G = np.zeros((C, 6), dtype=float)
    for ch in range(C):
        F[ch] = _six_base_features(Xz[:, ch], alpha=alpha)
        G[ch] = _six_base_features(Xlog[:, ch], alpha=alpha)

    # オプション：0次モーメントの全チャネル正規化（式(4)注）
    if m0_channel_norm:
        sum_m0_F = np.sum(F[:, 0]) + _EPS
        sum_m0_G = np.sum(G[:, 0]) + _EPS
        F[:, 0] = F[:, 0] / sum_m0_F
        G[:, 0] = G[:, 0] / sum_m0_G

    # （式12）“方向”の抽出：f,gを正規化し、cosθ = <f̂,ĝ>
    F_norm = F / (np.linalg.norm(F, axis=1, keepdims=True) + _EPS)
    G_norm = G / (np.linalg.norm(G, axis=1, keepdims=True) + _EPS)

    if mode == "scalar":
        # 各chの cosθ を1値で返す
        cos_theta = np.sum(F_norm * G_norm, axis=1, keepdims=True)
        return cos_theta  # (C,1)

    elif mode == "vector":
        # “方向ベクトル成分”：要素ごとの積（和をとると cosθ ）
        orient_vec = F_norm * G_norm  # (C,6)
        return orient_vec

    else:
        raise ValueError("mode must be 'scalar' or 'vector'")

# --- 使い方例 ---
# emg_win: shape (n_samples, n_channels) の1窓データ
# feats_scalar = td_psd_multichannel(emg_win, mode="scalar")  # -> (C,1)
# feats_vector = td_psd_multichannel(emg_win, mode="vector")  # -> (C,6)

# メディアンフィルタ＋特徴量の連結
from scipy.ndimage import median_filter

def medianfilter_and_hstack(features_list, kernel_size=3, shape=6):
    for i, features in enumerate(features_list):
        y = median_filter(np.array(features).reshape(-1,shape,shape), size=(1,kernel_size,kernel_size)).reshape(-1,shape*shape)
        if i == 0:
            Y = y
        else:
            Y = np.hstack([Y, y])
    return np.array(Y)

# == PyTorch版シンプル2D-CNNを sklearn 風に ===
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class _SimpleCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),    nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),    nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),            # (N,64,1,1)
        )
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.net(x).flatten(1)          # (N,64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)                  # logits

class CNNClassifierTorch:
    """
    sklearn風の2D-CNN分類器（PyTorch版）
      - fit(X, y, validation_data=None)
      - predict(X) -> y_pred (int)
      - predict_proba(X) -> softmax確率
      - score(X, y) -> accuracy
    """
    def __init__(
        self,
        input_shape,           # (H, W) or (1, H, W)
        n_classes: int,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 64,
        verbose: int = 0,
        seed: int = 42,
        device: str = "auto",  # "auto" / "cpu" / "cuda"
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if len(input_shape) == 2:
            in_ch, H, W = 1, input_shape[0], input_shape[1]
        elif len(input_shape) == 3:
            in_ch, H, W = input_shape
        else:
            raise ValueError("input_shape must be (H,W) or (C,H,W)")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.n_classes = n_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.model = _SimpleCNN(in_ch=in_ch, n_classes=n_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    # --------- helpers ----------
    def _prep_X(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 3:  # (N,H,W) -> (N,1,H,W)
            X = X[:, None, :, :]
        elif X.ndim == 4:
            pass
        else:
            raise ValueError("X must be (N,H,W) or (N,1,H,W)")
        return torch.from_numpy(X)

    def _prep_y(self, y):
        y = np.asarray(y, dtype=np.int64)
        return torch.from_numpy(y)

    def _make_loader(self, X, y=None, shuffle=False):
        X_t = self._prep_X(X)
        if y is None:
            ds = TensorDataset(X_t)
        else:
            y_t = self._prep_y(y)
            ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    # --------- sklearn-like API ----------
    def fit(self, X, y, validation_data=None):
        train_loader = self._make_loader(X, y, shuffle=True)
        if validation_data is not None:
            Xv, yv = validation_data
            val_loader = self._make_loader(Xv, yv, shuffle=False)
        else:
            val_loader = None

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total_correct = 0
            total_seen = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item()) * xb.size(0)
                total_correct += (logits.argmax(1) == yb).sum().item()
                total_seen += xb.size(0)

            if self.verbose:
                msg = f"[{epoch}/{self.epochs}] loss={total_loss/total_seen:.4f} acc={total_correct/total_seen:.3f}"
                if val_loader is not None:
                    val_acc = self._evaluate_loader(val_loader)
                    msg += f" | val_acc={val_acc:.3f}"
                print(msg)
        return self

    @torch.no_grad()
    def _evaluate_loader(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            logits = self.model(xb)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
        return correct / max(1, total)

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        loader = self._make_loader(X, y=None, shuffle=False)
        preds = []
        for (xb,) in loader:
            xb = xb.to(self.device, non_blocking=True)
            logits = self.model(xb)
            preds.append(logits.argmax(1).cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.int64)

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        loader = self._make_loader(X, y=None, shuffle=False)
        probs = []
        for (xb,) in loader:
            xb = xb.to(self.device, non_blocking=True)
            logits = self.model(xb)
            p = F.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
        return np.concatenate(probs, axis=0)

    def score(self, X, y):
        loader = self._make_loader(X, y, shuffle=False)
        return self._evaluate_loader(loader)

# ===PyTorch版CNNのデータオーグメント===

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# --- 単品オペレータ（形状は(1,8,8)想定。2Dならx[None,None]で扱う） ---
def _translate_map(x, dx, dy, pad_value=0.0):
    X = x.clone()
    Y = torch.zeros_like(X) + pad_value
    x_min = max(0, dx); x_max = min(6, 6+dx)
    y_min = max(0, dy); y_max = min(6, 6+dy)
    Y[..., y_min:y_max, x_min:x_max] = X[..., y_min-dy:y_max-dy, x_min-dx:x_max-dx]
    return Y

def _rotate_map(x, deg):
    theta = torch.tensor([[
        [ np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg)), 0.0 ],
        [ np.sin(np.deg2rad(deg)),  np.cos(np.deg2rad(deg)), 0.0 ],
    ]], dtype=torch.float32, device=x.device)
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

def _gaussian_kernel2d(ks=3, sigma=0.5, device="cpu"):
    ax = torch.arange(ks, device=device) - (ks-1)/2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    K = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
    K /= K.sum()
    return K.view(1,1,ks,ks)

def _blur_map(x, sigma=0.5):
    K = _gaussian_kernel2d(3, sigma, device=x.device)
    return F.conv2d(x, K, padding=1)

def _add_white_noise(x, snr_db=25.0):
    sig_pow = (x**2).mean()
    noise_pow = sig_pow / (10**(snr_db/10))
    noise = torch.randn_like(x) * torch.sqrt(noise_pow + 1e-12)
    return x + noise

# --- まとめて呼べるTransform ---
class MapAugment:
    def __init__(self,
                 p_translate=0.5, max_shift=1,      # 平行移動 ±1セル
                 p_rotate=0.5,   max_deg=10,        # 小回転 ±10°
                 p_drop=0.3,     max_drop=2,        # 電極0化 1〜2点
                 p_blur=0.3,     sigma=0.5,         # 弱Gaussian blur
                 p_noise=0.5,    snr_db=25.0):      # 白色ノイズ
        self.p_translate = p_translate; self.max_shift = max_shift
        self.p_rotate = p_rotate;       self.max_deg = max_deg
        self.p_drop = p_drop;           self.max_drop = max_drop
        self.p_blur = p_blur;           self.sigma = sigma
        self.p_noise = p_noise;         self.snr_db = snr_db

    def __call__(self, x):
        # x: (1,8,8) torch.float32
        assert x.ndim == 3 and x.shape[-2:] == (6,6)
        X = x.clone()

        if np.random.rand() < self.p_translate:
            dx = np.random.randint(-self.max_shift, self.max_shift+1)
            dy = np.random.randint(-self.max_shift, self.max_shift+1)
            X = _translate_map(X, dx, dy)

        if np.random.rand() < self.p_rotate:
            deg = np.random.uniform(-self.max_deg, self.max_deg)
            X = _rotate_map(X[None], deg).squeeze(0)

        if np.random.rand() < self.p_drop:
            k = np.random.randint(1, self.max_drop+1)
            idx = np.random.choice(36, size=k, replace=False)
            for i in idx:
                r, c = divmod(i, 6); X[..., r, c] = 0.0

        if np.random.rand() < self.p_blur:
            X = _blur_map(X[None], self.sigma).squeeze(0)

        if np.random.rand() < self.p_noise:
            X = _add_white_noise(X, self.snr_db)

        return X

# --- オーグメントを適用するDatasetラッパ ---
class AugDataset(Dataset):
    def __init__(self, X_torch, y_torch, transform=None):
        self.X = X_torch  # (N,1,8,8) torch.Tensor
        self.y = y_torch  # (N,)     torch.LongTensor
        self.tf = transform

    def __len__(self): return self.X.size(0)

    def __getitem__(self, i):
        xi = self.X[i]
        if self.tf is not None:
            xi = self.tf(xi)
        if self.y is None:
            return xi
        return xi, self.y[i]



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
# w付きでかつ、初期回転のあら推定ありで、小角近似ありの位置合わせ
# res = estimate_shift_rotation(
#     P_ref, a_ref, Q_test, b_test, w=None,
#     lam=1.0, max_iter=40, coarse_deg=2.0,
#     refine_halfwidth_deg=6.0, huber_delta=3.0,
#     outlier_quantile=0.90
# )

# w付きでかつ、初期回転のあら推定無しで、小角近似なしの位置合わせ
res = estimate_w_noinit_nosmallangle(
    P_ref, a_ref, Q_test, b_test, w=None,
    lam=1.0, max_iter=500, deg_step=1.0, huber_delta=None, trim_q=0.90
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
# X_train = np.array(emg_list_train)
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],-1)
# X_test = np.array(emg_list_test)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],-1)

def segment_time_series(emg_6x6: np.ndarray, window: int, hop: int) -> np.ndarray:
    """
    (n_samples, 6, 6) を窓分割して (n_windows, window, 6, 6) へ。
    """
    n, _, _ = emg_6x6.shape
    idx_starts = np.arange(0, max(1, n - window + 1), hop)
    segs = np.stack([emg_6x6[s:s+window] for s in idx_starts if s + window <= n], axis=0)
    # segs: (n_windows, window, 6, 6)
    return segs

window = 200   # サンプル幅（例：100ms @ 2kHz）
hop    = 50    # ホップ（例：25ms）
kind = 'rms'  # 'rms' or 'mav' or 'waveform_length' etc.
# X_train = []
# y_train = []
sizes_te = []
threshold = 0
threshold2 = 0.0013
for i, (emg_train_8x8, label) in enumerate(zip(emg_list_train, y_list_train)):
    # --- ラベル（学習用）：あなたのラベリングに置き換えてください ---
    # 中央6×6から特徴抽出した個数に合わせる必要があります。
    tmp_X = extract_center_6x6(emg_train_8x8)  # (n_samples, 6, 6)
    tmp_X = segment_time_series(tmp_X, window=window, hop=hop)  # (n_windows, window, 6, 6)
    # tmp_X = tmp_X.reshape(tmp_X.shape[0], tmp_X.shape[1], 36)  # (n_windows, window, 36) #
    rms = [rms_feat(x) for x in tmp_X]
    wl = [waveform_length(x) for x in tmp_X]
    zc = [zero_crossings(x, threshold) for x in tmp_X]
    ssc = [slope_sign_changes(x, threshold) for x in tmp_X]
    wamp = [wamp_feat(x, threshold2) for x in tmp_X]
    # td_psd = [td_psd_multichannel(x, fs=2048, mode="vector") for x in tmp_X]
    # td_psd = np.array(td_psd)
    # f1 = td_psd[:,:,0]
    # f2 = td_psd[:,:,1]
    # f3 = td_psd[:,:,2]
    # f4 = td_psd[:,:,3]
    # f5 = td_psd[:,:,4]
    # f6 = td_psd[:,:,5]
    # td_psd = td_psd.reshape(td_psd.shape[0], -1)
    # tmp_X = medianfilter_and_hstack([wl, f1, f6], kernel_size=2, shape=6)
    # tmp_X = np.hstack([rms, wl, zc]) # tmp_X = np.hstack([rms, wl, zc, ssc]) #
    tmp_X = rms
    # tmp_X = extract_features(emg_train_8x8, FeatureSpec(kind=kind, window=window, hop=hop))  # (n_windows, 36)
    n_windows = len(tmp_X)
    # 例：ダミーの 3 クラスを周回（実際はジェスチャーIDに差し替え）
    tmp_y = [int(label)-1 for i in range(n_windows)]
    sizes_te.append(n_windows)

    if len(tmp_y) != len(tmp_X):
        raise ValueError(f"y_train length ({len(tmp_y)}) must match number of windows ({len(tmp_X)})")
    
    if i == 0:
        X_train = tmp_X
        y_train = tmp_y
    else:
        X_train = np.vstack([X_train, tmp_X])
        y_train = np.hstack([y_train, tmp_y])

PCA_N_COMPONENTS = 0.98  # float: 分散説明率、または int: 次元数
RANDOM_STATE = 0

# パイプライン（標準化→PCA→LDA）
components = PCA_N_COMPONENTS
if isinstance(components, float):
    pca = PCA(n_components=components, svd_solver='full', random_state=RANDOM_STATE)
else:
    pca = PCA(n_components=components, random_state=RANDOM_STATE)

# clf = Pipeline([
#     ('scaler', StandardScaler()),
#     ('pca', pca),
#     ('lda', LDA())
# ])
# clf = Pipeline([
#     ('scaler', StandardScaler()),
#     # ('pca', PCA(n_components=120, random_state=0)),
#     ('clf', RandomForestClassifier(n_estimators=300, max_depth=50, random_state=0))
# ])
# clf = RandomForestClassifier(n_estimators=300, random_state=0)
# clf = LDA()
# clf = CNNClassifier(input_shape=(6, 6), n_classes=34, epochs=20, batch_size=128, verbose=1)
# clf = CNNClassifierTorch(input_shape=(6, 6), n_classes=34, epochs=20, batch_size=128)
# clf.fit(X_train, y_train)

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
    tmp_X = mapper.transform(emg_test_8x8, dx=dx, dy=dy, theta=theta, mode=mode)  # (n,6,6)

    tmp_X = segment_time_series(tmp_X, window=window, hop=hop)  # (n_windows, window, 6, 6)
    # tmp_X = tmp_X.reshape(tmp_X.shape[0], tmp_X.shape[1], 36)  # (n_windows, window, 36) #
    rms = [rms_feat(x) for x in tmp_X]
    wl = [waveform_length(x) for x in tmp_X]
    zc = [zero_crossings(x, threshold) for x in tmp_X]
    ssc = [slope_sign_changes(x, threshold) for x in tmp_X]
    wamp = [wamp_feat(x, threshold2) for x in tmp_X]
    # td_psd = [td_psd_multichannel(x, fs=2048, mode="vector") for x in tmp_X]
    # td_psd = np.array(td_psd)
    # f1 = td_psd[:,:,0]
    # f2 = td_psd[:,:,1]
    # f3 = td_psd[:,:,2]
    # f4 = td_psd[:,:,3]
    # f5 = td_psd[:,:,4]
    # f6 = td_psd[:,:,5]
    # td_psd = td_psd.reshape(td_psd.shape[0], -1)
    # tmp_X = medianfilter_and_hstack([wl, f1, f6], kernel_size=2, shape=6)
    # tmp_X = np.hstack([rms, wl, zc]) # tmp_X = np.hstack([rms, wl, zc, ssc]) #
    tmp_X = rms
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

lr = 1.09e-2
batch_size = 128 #128
clf = CNNClassifierTorch(input_shape=(6, 6), n_classes=34, epochs=50, batch_size=batch_size, lr=lr, verbose=1)
clf.fit(X_train, y_train, validation_data=(X_test, y_test))
y_pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)
score = clf.score(X_test, y_test)
print(f"y_pred shape = {y_pred.shape}, accuracy = {score}")
print(f'lr={lr}, batch_size={batch_size}')