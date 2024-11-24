import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from .time_features import *
from .frequency_features import *
from .entropy_features import *
from .wavelet_features import *
from .fractal_dimensions_features import *

#VISUALIZATION CONSTANTS

SPECTOGRAM_WINDOW_LENGTH = 20
SPECTOGRAM_STD_DEV = 12
SPECTOGRAM_HOP_SIZE = 2
SPECTOGRAM_MFFT = 800
SPECTROGRAM_MAX_FREQUENCY = 60
#MODEL CONSTANTS

SOLE_MODEL_ACCURACY_THRESHOLD = 0.5
SOLE_MODEL_TESTING = 0.2
SAMPLING_FREQUENCY = 250

GAF_CMAP = "binary_r"
#MODEL TRAINING

MODELS = {
    'svm': SVC(probability=True),
    'knn': KNeighborsClassifier(),
    'extra_trees': ExtraTreesClassifier(),
    'random_forest': RandomForestClassifier(),
    'decision_tree': DecisionTreeClassifier(),
    # 'gradient_boosting': GradientBoostingClassifier(),
    'ada_boost': AdaBoostClassifier(),
    'naive_bayes': GaussianNB(),
    # 'logistic_regression': LogisticRegression(),
}

PARAM_GRIDS = {
    'svm': {
        'C': np.logspace(-1, 10, 5),
        'kernel': ['rbf'],
    },
    'knn': {
        'n_neighbors': [1, 3, 5, 10, 20, 50, 100],
        'weights': ['uniform', 'distance'],
    },
    'extra_trees': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 50, 100, 150],
        'min_samples_split': [2, 5, 10, 20], 
        'min_samples_leaf': [1, 2, 5, 10], 
    },
    'random_forest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 50, 100, 150],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10], 
    },
    'decision_tree': {
        'max_depth': [None, 10, 50, 100, 150],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
    },
    'gradient_boosting': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7, 9],
    },
    'naive_bayes': {
        # GaussianNB has no hyperparameters to tune
    },
    'logistic_regression': {
        'C': np.logspace(-3, 3, 7),
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],  
        'solver': ['saga', 'lbfgs', 'liblinear'],  
    },
}

#MODEL TRAINING CONSTANTS

SEARCH_VERBOSE = 1
SCORING_METHOD = "accuracy"
SEARCH_ITERATIONS = 40
CROSS_VALIDATION_NUM = 4


#FEATURES

TIME_FEATURES_ARRAY = [
    '01_std',
    '01_ptp',
    '01_var',
    '01_min',
    '01_max',
    '01_argmin',
    '01_argmax',
    '01_rms',
    '01_abs_diff_signal',
    '01_skewness',
    '01_kurtosis',
    '01_signal_energy',
    '01_mean',
    '01_hjorth_complexity',
    '01_hjorth_mobility',
]

FREQUENCY_FEATURES_ARRAY = [
    '01_theta_channels',
    '01_alpha_channels',
    '01_beta_channels',
    # '01_gamma_channels',
    # '01_delta_channels'
]

# ENTROPY_FEATURES_ARRAY = [
#     'wavelet_entropy',
#     'spectral_entropy_channels'
#     'sample_entropy',
#     'wavelet_entropy',
#     'fuzzy_entropy',
#     'spectral_fuzzy_entropy'
# ]

FRACTAL_DIMENSIONS_FEATURES_ARRAY = [
    '01_katz',
    '01_petrosian',
]

WAVELET_FEATURES_ARRAY = [
    '24_mean_and_std_wavelet_coefficients'
]

ALL_FEATURES_ARRAY = TIME_FEATURES_ARRAY + FREQUENCY_FEATURES_ARRAY + WAVELET_FEATURES_ARRAY + FRACTAL_DIMENSIONS_FEATURES_ARRAY

FEATURE_FUNCTIONS_DICT = {

    # Time Based Features
    '01_std': std,
    '01_ptp': ptp,
    '01_var': var,
    '01_min': min,
    '01_max': max,
    '01_argmin': argmin,
    '01_argmax': argmax,
    '01_rms': rms,
    '01_abs_diff_signal': abs_diff_signal,
    '01_skewness': skewness,
    '01_kurtosis': kurtosis,
    '01_signal_energy': signal_energy,
    '01_mean': mean,
    '01_hjorth_complexity': hjorth_complexity,
    '01_hjorth_mobility': hjorth_mobility,

    #Frequnecy Based Features
    '01_theta_channels': theta_band,
    '01_alpha_channels': alpha_band,
    '01_beta_channels' : beta_band,
    # 'delta_channels': delta_band,

    #Fractal Dimension Features

    '01_katz' : katz_fd,
    '01_petrosian' : petrosian_fd,

    #Wavelet Features

    '24_mean_and_std_wavelet_coefficients' : get_wavelet_features

    }

#FEATURE CONSTANTS

WAVELET_FREQUENCIES = [61.115, 43.59, 31.09, 22.17, 15.81, 11.28, 8.04, 5.74, 4.09, 2.92, 2.08, 1.48]