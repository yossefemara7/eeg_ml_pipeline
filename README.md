# EEG Machine Learning Analysis Package

The **EEG Machine Learning Pipeline Package** provides a comprehensive framework for analyzing EEG data using machine learning techniques. The package includes support for feature extraction and selection, model training, and advance visualization methods, all integrated into the `EEGPipeline` class.

---

## **Quickstart**

### **Example Workflow**

```python
# Import your data
from my_eeg_data import data_array, label_array, channel_array, SAMPLING_RATE  # Replace with your data loader

pipeline = EEGPipeline(
    data_array=data_array,
    label_array=label_array,
    sampling_rate=SAMPLING_RATE
)

pipeline.train_model(
    enable_feature_selection=True,
    frequency_features=True,
    fractal_dimensions_features=True,
    selected_classifiers=["tabnet", "ml_ensemble", "tabpfn"],
    ml_baseline_accuracy=0.9
)

from my_eeg_data import test_data_array
predictions = pipeline.predict_input(test_data_array)
print(predictions)
```

---

## **Getting Started**

### **Installation**

Install the package via pip:
```bash
pip install eeg_ml_pipeline
```

### **Requirements**

- Python 3.8+
- NumPy, Pandas
- scikit-learn
- TensorFlow
- PyTorch
- pyts

---

## **EEGPipeline**

The `EEGPipeline` class is the primary interface for performing all your EEG data analysis. It handles feature extraction and selection, model training, and prediction in a streamlined workflow.

### **Initialization**

```python
pipeline = EEGPipeline(data_array, label_array, sampling_rate, channel_names=None)
```

#### **Parameters**
- **`data_array`** (`numpy.ndarray`): EEG data array of shape `(epochs, channels, timepoints)`.
- **`label_array`** (`numpy.ndarray`): Corresponding labels for each epoch.
- **`sampling_rate`** (`int`): Sampling frequency of the EEG data in Hz.
- **`channel_names`** (`list`, optional): Names of EEG channels. Defaults to `['channel_0', ..., 'channel_n']`.

---

### **Core Method: `train_model`**

```python
pipeline.train_model(
    initial_test_size=0.2,
    enable_feature_selection=False,
    time_features=True,
    frequency_features=False,
    wavelet_features=False,
    fractal_dimensions_features=False,
    voting_type="hard",
    ml_baseline_accuracy=0.8,
    tabnet_training_epochs=200,
    selected_classifiers=['tabpfn']
)
```

#### **Description**
Trains machine learning models using the specified features and classifiers.

#### **Parameters**
- **`initial_test_size`** (`float`): Proportion of data for testing (default: `0.2`).
- **`enable_feature_selection`** (`bool`): Enable feature selection (default: `False`).
- **`time_features`**, **`frequency_features`**, **`wavelet_features`**, **`fractal_dimensions_features`** (`bool`): Toggle feature domains.
- **`voting_type`** (`str`): Voting method for ensemble classifiers (`"hard"` or `"soft"`).
- **`ml_baseline_accuracy`** (`float`): Minimum accuracy threshold for hyperparameter tuning (default: `0.8`).
- **`tabnet_training_epochs`** (`int`): Training epochs for TabNet classifier (default: `200`).
- **`selected_classifiers`** (`list`, optional): Classifiers to use (e.g., `['tabnet', 'ml_ensemble']`).

---

### **Prediction**

#### **Method: `predict_input`**

```python
predictions = pipeline.predict_input(input_data)
```

Predicts the class of new input data using the trained model.

- **`input_data`** (`numpy.ndarray`): New EEG data array of shape `(epochs, channels, timepoints)`.
- **Returns**: Predicted labels.

---

## **Supported Classifiers**

- **`ml_ensemble`**: Traditional ML ensemble classifier.
- **`tabpfn`**: TabPFN Pre-Trained classifier (Only works on <1000 samples and works best at <100 features).
- **`tabnet`**: TabNet Pre-Trained classifier.
- **`gaf`**: GAF-based CNN (Coming Soon).
- **`rp`**: Recurrence-Plot based CNN (Coming Soon).

---


## **License**
This project is licensed under the MIT License.

---

## **Acknowledgments**
This package was developed as part of research initiatives to classify EEG data efficiently and effectively.

