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

import muscle_activity_information_gaussianfitting_meanfrequency, muscle_activity_information_gaussianfitting_ptp, muscle_activity_information_gaussianfitting_rms, muscle_activity_information_gaussianfitting_zc, muscle_activity_information_gaussianfitting_waveformlength, muscle_activity_information_gaussianfitting_medianfrequency, muscle_activity_information_gaussianfitting_peakfrequency, muscle_activity_information_gaussianfitting_spectralentropy
import muscle_activity_information_gmm_meanfrequency, muscle_activity_information_gmm_ptp, muscle_activity_information_gmm_rms, muscle_activity_information_gmm_zc, muscle_activity_information_gmm_waveformlength, muscle_activity_information_gmm_medianfrequency, muscle_activity_information_gmm_peakfrequency, muscle_activity_information_gmm_spectralentropy

muscle_activity_information_gmm_ptp.run()
muscle_activity_information_gmm_rms.run()
muscle_activity_information_gmm_zc.run()
muscle_activity_information_gmm_waveformlength.run()
muscle_activity_information_gmm_meanfrequency.run()
muscle_activity_information_gmm_medianfrequency.run()
muscle_activity_information_gmm_peakfrequency.run()
muscle_activity_information_gmm_spectralentropy.run()