print("Importing Modules Please wait...")
import numpy as np
import pandas as pd
from .config import *
from .feature_extraction import extract_features, estimate_feature_extraction_time, extract_single_epoch_features
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from .utils import *
from .model_training import *
from .visualization import compute_gaf, compute_spectrogram, plot_gaf, plot_spectrogram



class EEGPipeline:
    def __init__(self, data_array, label_array, sampling_rate, channel_names = None):
        self.data_array = data_array
        self.label_array = label_array
        self.channel_names = channel_names
        self.sf = sampling_rate
        self.num_classes = len(set(self.label_array))
        self.classifier = None
        self.feature_extractor = None
        SAMPLING_FREQUENCY = sampling_rate
        if channel_names is None:
            self.channel_names = ["channel_" + str(i) for i in range(data_array.shape[1])]
        if len(data_array.shape) != 3:
            raise ValueError(f"Data array must have 3 dimensions, but got {len(data_array.shape)}")
        if data_array.shape[0] != len(label_array):
            raise ValueError(f"Number of samples in data array does not match number of labels: {data_array.shape[0]} != {len(label_array)}")
        
        if len(self.channel_names) != data_array.shape[1]:
            raise ValueError(f"Number of channels in channel_names_array does not match the second dimension of data_array: {len(channel_names)} != {data_array.shape[1]}")
        
        # times_dict = estimate_feature_extraction_time(self.data_array)
        print("EEG Pipeline Initialized:")
        print(f"  Data shape: {data_array.shape} (Epochs, Channels, Timepoints)")
        print(f"  Number of Epochs and Labels: {len(label_array)}")
        print(f"  Number of classes: {self.num_classes}")
        print(f"  Number of channels: {len(self.channel_names)}")
        print(f"  Sampling rate: {sampling_rate} Hz")
        # encoder = LabelEncoder()
        # self.label_array = encoder.fit_transform(self.label_array)

        # print("Estimated Feature Extraction Times:")
        # print(f"  Time Domain: {times_dict['time']} seconds")
        # print(f"  Frequency Domain: {times_dict['frequency']} seconds")
        # print(f"  Entropy Domain: {times_dict['entropy']} seconds")
        # print(f"  Combined Domains: {times_dict['all']} seconds\n")

    class FeatureExtractor:
        def __init__(self, channel_feature_names, all_feature_names, filtered_features):
            self.channel_feature_names = channel_feature_names
            self.all_feature_names = all_feature_names
            self.filtered_features = filtered_features

        def extract_features(self, data):
            all_feature_array = np.array([extract_single_epoch_features(data, self.all_feature_names)])
            all_feature_df = pd.DataFrame(all_feature_array, columns = self.channel_feature_names)
            filtered_feature_df = all_feature_df[self.filtered_features]
            filtered_features = filtered_feature_df.to_numpy()
            filtered_features = np.nan_to_num(filtered_features, nan = 0)
            return filtered_features
        
    def train_model(
        self,
        initial_test_size: float = 0.2,
        enable_feature_selection: bool = False,
        time_features: bool = True,
        frequency_features: bool = False,
        # entropy_features: bool = False,
        wavelet_features: bool = False,
        fractal_dimensions_features: bool = False,
        voting_type: str = "hard",
        ml_baseline_accuracy: float = 0.8,
        tabnet_training_epochs: int = 200,
        gaf_training_epochs: int = 100,
        gaf_image_preview: bool = True,
        gaf_training_batch: int = 32,
        selected_classifiers: list = None, 
    ) -> None:
        """
        A configurable pipeline for feature extraction, model training, and prediction.

        Parameters:
        - initial_test_size: Test split ratio.
        - enable_feature_selection: Whether to perform feature selection.
        - time_features, frequency_features, etc.: Toggle different feature types.
        - voting_type: Voting method for ensemble classifiers.
        - tabnet_training_epochs, gaf_training_epochs: Training epochs for respective models.
        - selected_classifiers: List of classifiers to include ['gaf', 'tabpfn', 'tabnet', 'ml_ensemble'].
        """
        if selected_classifiers is None:
            selected_classifiers = ["gaf", "tabpfn", "tabnet", "ml_ensemble"]
        
        SOLE_MODEL_ACCURACY_THRESHOLD = ml_baseline_accuracy
        
        # Step 1: Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.data_array, self.label_array, test_size=initial_test_size, random_state=42
        )
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        # Step 2: Feature extraction
        all_feature_names = []
        if time_features:
            all_feature_names.extend(TIME_FEATURES_ARRAY)
        if frequency_features:
            all_feature_names.extend(FREQUENCY_FEATURES_ARRAY)
        if fractal_dimensions_features:
            all_feature_names.extend(FRACTAL_DIMENSIONS_FEATURES_ARRAY)
        if wavelet_features:
            all_feature_names.extend(WAVELET_FEATURES_ARRAY)

        self.all_feature_names = all_feature_names
        channel_feature_names = get_channel_feature_names(self.channel_names, all_feature_names)
        self.channel_feature_names = channel_feature_names

        feature_array = extract_features(X_train, all_feature_names)
        features_df = pd.DataFrame(feature_array, columns=channel_feature_names)

        if enable_feature_selection:
            filtered_feature_array, filtered_features = feature_selection(
                features_df.to_numpy(), y_train, channel_feature_names
            )
        else:
            filtered_feature_array = feature_array
            filtered_features = channel_feature_names

        self.filtered_features = filtered_features
        filtered_feature_array = np.nan_to_num(filtered_feature_array, nan=0)
        test_feature_array = self.get_test_features_array()


        # Step 3: Train classifiers
        classifiers = {}

        if "gaf" in selected_classifiers:
            gaf_classifier, gaf_val_accuracy, gaf_preds = self.train_model_gaf(
                gaf_training_epochs, gaf_image_preview, gaf_training_batch
            )
            classifiers["gaf_classifier"] = {
                "Classifier": gaf_classifier,
                "Val Accuracy": gaf_val_accuracy,
                "Predictions": gaf_preds,
            }

        if "tabpfn" in selected_classifiers:
            tabpfn_classifier, tabpfn_val_accuracy, tabpfn_preds = train_tabpfn_classifier(
                filtered_feature_array, y_train, test_feature_array
            )
            classifiers["tabpfn_classifier"] = {
                "Classifier": tabpfn_classifier,
                "Val Accuracy": tabpfn_val_accuracy,
                "Predictions": tabpfn_preds,
            }

        if "tabnet" in selected_classifiers:
            tabnet_classifier, tabnet_val_accuracy, tabnet_preds = train_and_test_tabnet(
                filtered_feature_array, y_train, test_feature_array, tabnet_training_epochs
            )
            classifiers["tabnet_classifier"] = {
                "Classifier": tabnet_classifier,
                "Val Accuracy": tabnet_val_accuracy,
                "Predictions": tabnet_preds,
            }

        if "ml_ensemble" in selected_classifiers:
            sole_model_info, _ = sole_models_hp_tuning(
                filtered_feature_array, y_train, SOLE_MODEL_TESTING, SOLE_MODEL_ACCURACY_THRESHOLD
            )
            best_models = [
                model_info["model"] for model_info in sole_model_info["best_model_info"]["models"]
            ]
            best_model_params = [
                model_info["best_params"] for model_info in sole_model_info["best_model_info"]["models"]
            ]
            ml_ensemble_classifier, ml_ensemble_val_accuracy, ml_ensemble_preds = create_and_fit_voting_classifier(
                best_models, best_model_params, filtered_feature_array, y_train, test_feature_array, voting=voting_type
            )
            classifiers["ml_ensemble_classifier"] = {
                "Classifier": ml_ensemble_classifier,
                "Val Accuracy": ml_ensemble_val_accuracy,
                "Predictions": ml_ensemble_preds,
            }

        best_classifier = None
        best_classifier_name = None
        best_test_accuracy = 0
        for clf_name, clf_info in classifiers.items():
            val_acc = clf_info["Val Accuracy"]
            test_pred = clf_info["Predictions"]
            test_acc = accuracy_score(y_test, test_pred)
            print(f"{clf_name} Validation Accuracy: {val_acc}")
            print(f"{clf_name} Testing Accuracy: {test_acc}")
            if test_acc > best_test_accuracy:
                best_test_accuracy = test_acc
                best_classifier = clf_info["Classifier"]
                best_classifier_name = clf_name

        self.classifiers = classifiers
        self.best_classifier = best_classifier
        print(f"Final Accuracy: {best_test_accuracy} achieved by {best_classifier_name}")
        self.feature_extractor = self.FeatureExtractor(self.channel_feature_names, self.all_feature_names, self.filtered_features)

        
    def extract_features_from_data(self, data_array):
        extracted_features = extract_features(data_array, self.all_feature_names)
        test_features_df = pd.DataFrame(extract_features, columns = self.channel_feature_names)
        filtered_test_feature_df = test_features_df[self.filtered_features]
        filtered_test_feature_array = filtered_test_feature_df.to_numpy()
        print(f"Test Filtered feature array Shape : {filtered_test_feature_array.shape}")
        filtered_test_feature_array = np.nan_to_num(filtered_test_feature_array, nan = 0)
        return extracted_features
    


    def get_test_features_array(self):
        test_features_array = extract_features(self.X_test, self.all_feature_names)
        print(test_features_array.shape)
        test_features_df = pd.DataFrame(test_features_array, columns = self.channel_feature_names)
        filtered_test_feature_df = test_features_df[self.filtered_features]
        print(self.channel_feature_names)
        filtered_test_feature_array = filtered_test_feature_df.to_numpy()
        print(f"Test Filtered feature array Shape : {filtered_test_feature_array.shape}")
        filtered_test_feature_array = np.nan_to_num(filtered_test_feature_array, nan = 0)
        return filtered_test_feature_array

    def predict_input(self, input_data):
        if self.classifier is None:
            raise ValueError("No model has not been trained yet to classify new input_data.")
        
        input_features_array = extract_features(input_data, self.all_feature_names)
        input_features_df = pd.DataFrame(input_features_array, columns = self.channel_feature_names)
        filtered_input_feature_df = input_features_df[self.filtered_features]
        filtered_input_feature_array = filtered_input_feature_df.to_numpy()
        filtered_input_feature_array = np.nan_to_num(filtered_input_feature_array, nan = 0)
        predictions = self.classifier.predict(filtered_input_feature_array)

        return predictions
    
    def get_feature_extractor_and_classifier(self):
        return self.feature_extractor, self.classifier

    def display_model_info(self):
        #Needs work
        if self.classifier is None:
            print("The model has not been trained yet.")
            return
        
        print("\nEEG Pipeline Model Information:")
        print("--------------------------------------------------")
        
        print("Selected Features for Training:")
        print(f"  All features: {self.all_feature_names}")
        # print(f"  Filtered features after selection: {self.filtered_features}")
        
        print("\nBest Models and Parameters:")
        best_models_info = [
            (model_name, params)
            for model_name, params in zip(self.best_model_names, self.best_model_params)
        ]
        for model_name, params in best_models_info:
            print(f"  Model: {model_name}")
            print(f"    Parameters: {params}")
        
        print("\nVoting Classifier Information:")
        print(f"  Voting Type: {self.classifier.voting}")
        print("  Base Models:")
        for estimator_name, estimator in self.classifier.named_estimators_.items():
            print(f"    {estimator_name}: {estimator}")

    ##Visualization

    def plot_gaf(self, epoch_to_plot = 0, channel_to_plot = None):
        if channel_to_plot is None:
            channel = 0
        else:
            channel = self.channel_names.index(channel_to_plot)
        eeg_signal = self.data_array[epoch_to_plot][channel]
        gaf = compute_gaf(eeg_signal)
        plot_gaf(gaf)


    #DEEP LEARNING USING IMAGES GENERATED FROM EEG_SIGNALS
    def train_model_gaf(self, training_epochs = 100, training_batch_size = 32):
        max_memory_usage = 0.01  # GB
        image_area = self.data_array.shape[0] ** 2
        n_images = self.data_array.shape[0]

        gaf_scale = ((max_memory_usage * 10**9) / (image_area * n_images))**-1
        print(f"Calculated GAF Scale: {gaf_scale}")

        self.eeg_max = np.max(self.X_train)
        self.eeg_min = np.min(self.X_train)
        print(self.eeg_max)
        print(self.eeg_min)
        gaf_train_images_array = get_gaf_image_array(self.X_train, self.eeg_max, self.eeg_min, gaf_scale=gaf_scale)
        gaf_test_images_array = get_gaf_image_array(self.X_test, self.eeg_max, self.eeg_min, gaf_scale=gaf_scale)
        print(f"Gaf Images Array Shape : {gaf_train_images_array.shape}")
        gaf_classifier, gaf_performance_metrics, val_accuracy, gaf_preds = train_and_evaluate_image_classifier(
            gaf_train_images_array,
            gaf_test_images_array,
            self.y_train,
            num_classes=self.num_classes,
            epochs=training_epochs,
            batch_size=training_batch_size
        )
        self.gaf_classifier = gaf_classifier
        display_image_classifier_performance_metrics(gaf_performance_metrics)
        return gaf_classifier, val_accuracy, gaf_preds

