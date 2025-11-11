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
import cv2
from scipy.stats import pearsonr
import pingouin as pg  # for ICC
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

import evaluation_across_trials
import evaluation_across_sessions

# evaluation_across_trials.main(feature_name_for_filename = "ptp")
# evaluation_across_trials.main(feature_name_for_filename = "rms")
# evaluation_across_trials.main(feature_name_for_filename = "zc")
# evaluation_across_trials.main(feature_name_for_filename = "waveformlength")

evaluation_across_sessions.main(feature_name_for_filename = "ptp")
evaluation_across_sessions.main(feature_name_for_filename = "rms")
evaluation_across_sessions.main(feature_name_for_filename = "zc")
evaluation_across_sessions.main(feature_name_for_filename = "waveformlength")