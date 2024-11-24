import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import seaborn as sns
from tqdm import tqdm
from .utils import *
from .visualization import *

def ensure_list(value):
    """Ensure that the input value is wrapped in a list if it's not already a list."""
    if not isinstance(value, list):
        return [value]
    return value


def compute_spectrogram(eeg_signal, fs, nperseg=256, overlap_fraction=0.5, cmap='viridis'):
    """
    Calculate and plot the spectrogram of an EEG signal with adaptive overlap.

    Parameters:
    - eeg_signal: 1D numpy array of EEG signal data
    - fs: Sampling frequency of the EEG signal (default is 1000 Hz)
    - nperseg: Length of each segment for STFT (default is 256)
    - overlap_fraction: Fraction of `nperseg` to use for overlap (default is 0.5)
    - cmap: Colormap for the spectrogram plot (default is 'viridis')

    Returns:
    - None (plots the spectrogram)
    """
    noverlap = int(nperseg * overlap_fraction)
    
    frequencies, times, Sxx = spectrogram(eeg_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap=cmap)
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.title("Spectrogram of EEG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, 60)

    plt.tight_layout()
    plt.show()

def feature_selection(feature_array, label_array, all_features):
    from featurewiz import featurewiz
    from statistics import mode

    df = pd.DataFrame(feature_array, columns=all_features)
    df['diagnosis'] = label_array
    output = featurewiz(df, target='diagnosis', skip_sulov = True, corr_limit = 2)

    array1 = np.array(output[0])
    array2 = np.array(output[1])
    remove_array = []
    for data in array2:
        temp = []
        for i, number in enumerate(data):
            if number == 1 or number == 0:
                remove_array.append(i)

    remove = mode(remove_array)
    feature_array = []
    for data in array2:
        temp = []
        for i, number in enumerate(data):
            if i == remove:
                pass
            else:
                temp.append(number)
        feature_array.append(temp)
    feature_array = np.array(feature_array)
    return feature_array, array1

def evaluate_model_performance(classifier, X_test, y_test):
    """
    Evaluate the performance of a classifier with various metrics.

    Args:
    - classifier: The trained model (e.g., classifier)
    - X_test: Features of the test set
    - y_test: True labels for the test set

    Returns:
    - None: Prints various performance metrics to the console.
    """
    #Make your own predict function
    preds = classifier.predict(X_test)

    print(f"New Feature Array Shape: {X_test.shape}")
    
    print(f'Preds Length: {len(preds)}')

    final_accuracy = accuracy_score(y_test, preds)
    print(f'Final Accuracy: {final_accuracy}')
    
    f1 = f1_score(y_test, preds)
    print(f'F1 Score: {f1}')
    
    precision = precision_score(y_test, preds)
    print(f'Precision: {precision}')
    
    recall = recall_score(y_test, preds)
    print(f'Recall: {recall}')
    
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, probs)
        print(f'ROC AUC: {auc_score}')
    
    conf_matrix = confusion_matrix(y_test, preds)
    print(f'Confusion Matrix:\n{conf_matrix}')
    
    class_report = classification_report(y_test, preds)
    print(f'Classification Report:\n{class_report}')


def apply_cmap(arr, cmap_name='viridis'):
    """
    Converts a 2D array into an RGB image using the specified colormap.

    Parameters:
    - arr (np.array): A 2D array with shape (256, 256), values in any range.
    - cmap_name (str): The name of the colormap to use (e.g., 'viridis', 'plasma', 'inferno', etc.).

    Returns:
    - rgb_image (np.array): A 3D RGB image with shape (256, 256, 3).
    """
    arr_normalized = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    cmap = plt.get_cmap(cmap_name)

    rgba_image = cmap(arr_normalized)

    rgb_image = rgba_image[..., :3]

    return rgb_image

def display_image(image):
    """
    Function to display an image array.
    
    Parameters:
    image (numpy.ndarray): A 3D array of shape (X, X, 3) representing the image.
    """
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def graph_training_history(history):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def display_image_classifier_performance_metrics(results_dict):
    """
    Prints and plots the performance metrics from the results dictionary.
    
    Parameters:
    - results_dict (dict): Dictionary containing the model's performance metrics.
    """
    # Print the metrics
    print(f"Test Accuracy: {results_dict['test_accuracy']:.4f}")
    print(f"Precision: {results_dict['precision']:.4f}")
    print(f"Recall: {results_dict['recall']:.4f}")
    print(f"F1 Score: {results_dict['f1_score']:.4f}")
    
    # Confusion Matrix
    cm = results_dict['confusion_matrix']
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(cm.shape[0]), yticklabels=np.arange(cm.shape[0]))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Plot training history (Accuracy and Loss)
    history = results_dict['history']
    
    # Plot Accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def get_channel_feature_names(channel_names, all_features):
    channel_feature_names = []
    for channel in channel_names:
        for feature in all_features:
            number = int(feature[0] + feature[1])
            for num in range(number):
                channel_feature_names.append(f"{channel}_{feature}_{num}")
    return channel_feature_names

def get_gaf_image_array(data, max, min, gaf_scale, gaf_image_preview = True):
    images_array = []
    for epoch in tqdm(data):
        image = []
        for channel in epoch:
            one_channel_gaf = compute_gaf(channel, max, min, scale = gaf_scale)
            
            if gaf_image_preview:
                display_image(apply_cmap(one_channel_gaf, GAF_CMAP))
                gaf_image_preview = False 

            image.append(one_channel_gaf)
        images_array.append(image)
    images_array = np.array(images_array)
    A, B, C, D = images_array.shape 
    return images_array.reshape((A, C, D, B)) 