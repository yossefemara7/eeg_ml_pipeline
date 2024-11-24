import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import antropy as ent
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def mean(x):
    x = np.nan_to_num(x, nan=0)
    return np.mean(x, axis=-1)

def std(x):
    x = np.nan_to_num(x, nan=0)
    return np.std(x, axis=-1)
def ptp(x):
    x = np.nan_to_num(x, nan=0)
    return np.ptp(x, axis=-1)
def var(x):
    x = np.nan_to_num(x, nan=0)
    return np.var(x, axis=-1)
def min(x):
    x = np.nan_to_num(x, nan=0)
    return np.min(x, axis=-1)
def max(x):
    x = np.nan_to_num(x, nan=0)
    return np.max(x, axis=-1)
def argmin(x):
    x = np.nan_to_num(x, nan=0)
    return np.argmin(x, axis=-1)
def argmax(x):
    x = np.nan_to_num(x, nan=0)
    return np.argmax(x, axis=-1)
def rms(x):
    x = np.nan_to_num(x, nan=0)
    return np.sqrt(np.mean(x**2, axis=-1))
def abs_diff_signal(x):
    x = np.nan_to_num(x, nan=0)
    return np.sum(np.abs(np.diff(x, axis=-1)), axis=-1)
def skewness(x):
    x = np.nan_to_num(x, nan=0)
    return stats.skew(x, axis=-1)
def kurtosis(x):
    x = np.nan_to_num(x, nan=0)
    return stats.kurtosis(x, axis=-1)
def signal_energy(x):
    x = np.nan_to_num(x, nan=0)
    return np.sum(x**2, axis=-1)

def hjorth_mobility(x):
    mobilities = []
    for row in x:
        mobility = np.round(ent.hjorth_params(row, axis=-1)[0], 4)
        mobilities.append(mobility)
    return np.array(mobilities)
def hjorth_complexity(x):
    complexities = []
    for row in x:
        complexity = np.round(ent.hjorth_params(row, axis=-1)[1], 4)
        complexities.append(complexity)
    return np.array(complexities)
