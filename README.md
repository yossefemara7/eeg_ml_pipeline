EEG Machine Learning Analysis Package

The EEG Machine Learning Analysis Package provides a comprehensive and modular framework for analyzing EEG data using state-of-the-art machine learning techniques. The package includes support for feature extraction, model training, and advanced visualization methods, all seamlessly integrated into the EEGPipeline class.

Quickstart

Example Workflow

# Import your data
from my_eeg_data import arrays  # Replace with your data loader

# Initialize the pipeline
pipeline = EEGPipeline(
    data_array=arrays["ASD"]["Data"],
    label_array=arrays["ASD"]["Label"],
    sampling_rate=256
)

# Train the model
pipeline.train_model(
    enable_feature_selection=True,
    frequency_features=True,
    fractal_dimensions_features=True,
    selected_classifiers=["tabnet", "ml_ensemble", "tabpfn"],
    ml_baseline_accuracy=0.9
)

# Predict on new data
new_data = np.random.rand(10, 32, 256)  # Example input data
predictions = pipeline.predict_input(new_data)
print(predictions)

Getting Started

Installation

Clone the repository and install the required dependencies:

git clone https://github.com/your-repo/eeg-analysis.git
cd eeg-analysis
pip install -r requirements.txt

Requirements

Python 3.8+

NumPy, Pandas

scikit-learn

TensorFlow

PyTorch (for TabNet)

pyts (for GAF visualizations)

EEGPipeline

The EEGPipeline class is the primary interface for performing EEG data analysis. It handles feature extraction, model training, and prediction in a streamlined workflow.

Initialization

pipeline = EEGPipeline(data_array, label_array, sampling_rate, channel_names=None)

Parameters

data_array (numpy.ndarray): EEG data array of shape (epochs, channels, timepoints).

label_array (numpy.ndarray): Corresponding labels for each epoch.

sampling_rate (int): Sampling frequency of the EEG data in Hz.

channel_names (list, optional): Names of EEG channels. Defaults to ['channel_0', ..., 'channel_n'].

Core Method: train_model

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
    selected_classifiers=None
)

Description

Trains machine learning models using the specified features and classifiers.

Parameters

initial_test_size (float): Proportion of data for testing (default: 0.2).

enable_feature_selection (bool): Enable feature selection (default: False).

time_features, frequency_features, wavelet_features, fractal_dimensions_features (bool): Toggle feature domains.

voting_type (str): Voting method for ensemble classifiers ("hard" or "soft").

ml_baseline_accuracy (float): Minimum accuracy threshold for hyperparameter tuning (default: 0.8).

tabnet_training_epochs (int): Training epochs for TabNet classifier (default: 200).

selected_classifiers (list, optional): Classifiers to use (e.g., ['tabnet', 'ml_ensemble']).

Prediction

Method: predict_input

predictions = pipeline.predict_input(input_data)

Predicts the class of new input data using the trained model.

input_data (numpy.ndarray): New EEG data array of shape (epochs, channels, timepoints).

Returns: Predicted labels.

Feature Extraction

Method: extract_features_from_data

features = pipeline.extract_features_from_data(data_array)

Extracts features from raw EEG data based on the configured feature set.

data_array (numpy.ndarray): Raw EEG data array.

Returns: Extracted feature array.

Visualization

Methods: plot_gaf, compute_spectrogram

pipeline.plot_gaf(epoch_to_plot=0, channel_to_plot="Fz")

Generates visualizations like Gramian Angular Field (GAF) images or spectrograms for EEG signals.

Supported Classifiers

gaf: GAF-based CNN.

tabpfn: TabPFN classifier.

tabnet: TabNet classifier.

ml_ensemble: Traditional ML ensemble classifier.

Contributing

Fork the repository.

Create a feature branch.

Submit a pull request with detailed explanations.

License

This project is licensed under the MIT License.

Acknowledgments

This package was developed as part of a BCI Initiative to classify EEG data efficiently and effectively.
