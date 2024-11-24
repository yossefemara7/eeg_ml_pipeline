import numpy as np
from tqdm import tqdm
from .config import *
import time
import warnings

# Suppress specific UserWarning from scipy.signal
warnings.filterwarnings("ignore", message="nperseg = .* is greater than input length")

def extract_features(data_array, chosen_features, display_progress = True):
    '''
    Purpose: Concatenates desired feature arrays from a data_array based on a chosen_features.
    Parameters:
    - data_array: Array of data where each element contains features to concatenate.
    - chosen_features: List of features to concatenate. Can contain specific features or 'all' to select all features.
    Return Value:
    - features_array: Concatenated feature array.
    '''
    
    def concat_features(x, chosen_features):
        functions = FEATURE_FUNCTIONS_DICT
        
        # selected_functions = [
        #     functions[key](x) for key in functions if key in chosen_features or 'all' in chosen_features
        # ]
        selected_functions = []

        for key in functions:
            if key in chosen_features or 'all' in chosen_features:
                features = functions[key](x)
                features = np.array(features)
                if len(features.shape) == 1:
                    selected_functions.append(features)
                else:
                    for nested_feature in features:
                        selected_functions.append(nested_feature)

        concatenated_features = np.concatenate(selected_functions, axis=-1)
        return concatenated_features

    features = []

    if display_progress:
        print("Extracting features...")
        for d in tqdm(data_array):
            concatenated_features = concat_features(d, chosen_features)
            features.append(concatenated_features)
    else:
        for d in data_array:
            concatenated_features = concat_features(d, chosen_features)
            features.append(concatenated_features)
    
    features_array = np.array(features)
    return features_array

def extract_single_epoch_features(data, chosen_features):
    def concat_features(x, chosen_features):
        functions = FEATURE_FUNCTIONS_DICT
        
        selected_functions = [
            functions[key](x) for key in functions if key in chosen_features or 'all' in chosen_features
        ]
        concatenated_features = np.concatenate(selected_functions, axis=-1)
        return concatenated_features
    
    feature_array = concat_features(data, chosen_features)
    return feature_array


def estimate_feature_extraction_time(data_array):
    '''
    Purpose: Estimates the time taken to extract certain features from a data_array.
    Parameters:
    - data_array: Array of data where each element contains features to concatenate.
    Return Value:
    - elapsed_time_array: An array pf the expected time taken to extract features from
    time, frequency, entropy, and all.
    '''
    num_epochs = data_array.shape[0]
    factor = int(num_epochs/10)

    time_start = time.time()
    extract_features(data_array[:10], TIME_FEATURES_ARRAY, display_progress = False)
    time_end = time.time()
    
    frequency_start = time.time()
    extract_features(data_array[:10], FREQUENCY_FEATURES_ARRAY, display_progress = False)
    frequency_end = time.time()

    entropy_start = time.time()
    extract_features(data_array[:10], FREQUENCY_FEATURES_ARRAY, display_progress = False)
    entropy_end = time.time()

    time_elapsed = (time_end - time_start)*factor
    frequnecy_elapsed = (frequency_end - frequency_start)*factor
    entropy_elapsed = (entropy_end - entropy_start)*factor
    
    all_elpased = time_elapsed + entropy_elapsed + frequnecy_elapsed
    elapsed_time_array = {
        "time" : time_elapsed,
        "frequency" :frequnecy_elapsed,
        "entropy" : entropy_elapsed,
        "all" :all_elpased
        }
    
    return elapsed_time_array

