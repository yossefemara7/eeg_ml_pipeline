
from .config import *
from .utils import *
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.utils import all_estimators
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from tabpfn import TabPFNClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter

def create_and_fit_voting_classifier(models, param_grids, X_train, y_train, X_test, voting='hard'):
    """
    Create and fit a Voting Classifier based on given models and parameters.

    Parameters:
    - models (list): List of model instances (e.g., classifiers like LogisticRegression, SVM, etc.).
    - param_grids (list): List of dictionaries containing hyperparameter grids for each model.
    - X (array-like): Feature data.
    - y (array-like): Label data.
    - voting (str): 'hard' for majority voting, 'soft' for probability averaging.

    Returns:
    - fitted_voting_classifier (VotingClassifier): The trained VotingClassifier.
    """

    tuned_models = []
    
    for model, param_grid in zip(models, param_grids):
        param_grid = {key: ensure_list(value) for key, value in param_grid.items()}
        
        grid_search = GridSearchCV(model, param_grid, cv=5)
        tuned_models.append(('best_' + model.__class__.__name__, grid_search))
    
    voting_classifier = VotingClassifier(estimators=tuned_models, voting=voting)
    
    voting_classifier.fit(X_train, y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    val_preds = voting_classifier.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy : {val_accuracy}")
    
    test_pred = voting_classifier.predict(X_test)
    return voting_classifier, val_accuracy, test_pred

def sole_models_hp_tuning(features_array, label_array, test_size, accuracy_threshold=0.9):
    """
    Perform conditional hyperparameter tuning for multiple models.
    If a model's baseline accuracy is below the threshold, skip tuning.
    """
    X_train, X_test, y_train, y_test = train_test_split(features_array, label_array, test_size=test_size)
    models = MODELS.copy()
    param_grids = PARAM_GRIDS.copy()

    accuracy_scores = {}
    best_params = {}
    best_model_info = {"models": []}
    max_accuracy = 0

    for model_name, model in models.items():
        # Step 1: Evaluate baseline accuracy with default parameters
        model.fit(X_train, y_train)
        baseline_pred = model.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, baseline_pred)

        print(f"{model_name} baseline accuracy: {baseline_accuracy:.4f}")

        # Step 2: Skip hyperparameter tuning if baseline accuracy is below the threshold
        if baseline_accuracy < accuracy_threshold:
            print(f"Skipping hyperparameter tuning for {model_name} (baseline below threshold)")
            accuracy_scores[model_name] = baseline_accuracy
            continue

        # Step 3: Perform hyperparameter tuning
        param_grid = param_grids.get(model_name, {})
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            cv=CROSS_VALIDATION_NUM,
            scoring=SCORING_METHOD,
            n_iter=SEARCH_ITERATIONS,
            verbose=SEARCH_VERBOSE,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)

        # Best model and accuracy after tuning
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        accuracy_scores[model_name] = accuracy
        best_params[model_name] = grid_search.best_params_

        # Step 4: Add to results if accuracy meets the threshold
        if accuracy >= accuracy_threshold:
            best_model_info["models"].append({
                "model_name": model_name,
                "model": best_model,
                "best_params": grid_search.best_params_,
                "accuracy": accuracy,
            })
            print(f"{model_name} tuned accuracy: {accuracy:.4f}")
            max_accuracy = max([max_accuracy, accuracy])

    return {"best_model_info": best_model_info, "accuracy_scores": accuracy_scores}, max_accuracy



def train_and_evaluate_image_classifier(data, test_data_real, labels, num_classes, epochs=10, batch_size=32):
    """
    Splits the data into training and testing sets, builds, compiles, and trains a CNN model.
    Evaluates the model on the test set using the best validation accuracy achieved during training.
    
    Parameters:
    - data (numpy.ndarray): Input data with shape (samples, 51, 51, 63)
    - labels (numpy.ndarray): Labels with shape (samples,)
    - num_classes (int): Number of classes for classification
    - epochs (int): Number of epochs to train the model
    - batch_size (int): Batch size for training
    
    Returns:
    - model: Trained TensorFlow model with the best weights
    - history: Training history object
    - test_accuracy: Accuracy of the model on the test set
    
    """
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels, num_classes=num_classes)
    
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    def create_model(input_shape, num_classes):
        """
        Creates a CNN model for classification tasks with input shape.
        
        Parameters:
        - input_shape (tuple): Shape of the input data (height, width, channels).
        - num_classes (int): Number of output classes.
        
        Returns:
        - model (tf.keras.Model): Compiled CNN model.
        """
        model = models.Sequential()
        
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(0.02), input_shape=input_shape))

        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(0.01)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        
        model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(0.05)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.2))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
        model.add(layers.Dropout(0.2))
        
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    input_shape = data.shape[1:]
    model = create_model(input_shape=input_shape, num_classes=num_classes)
    best_val_accuracy = 0.0
    
    def update_best_val_accuracy(epoch, logs):
        nonlocal best_val_accuracy
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

    checkpoint = ModelCheckpoint(
        filepath='best_model.keras',       
        monitor='val_accuracy',           
        save_best_only=True,               
        mode='max',                        
        verbose=1                       
    )
    best_accuracy_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: update_best_val_accuracy(epoch, logs)
    )
    history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2, callbacks=[checkpoint, best_accuracy_callback])

    model.load_weights('best_model.keras')
    
    test_loss, test_accuracy = model.evaluate(test_data, test_labels, verbose=0)
    
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    
    cm = confusion_matrix(true_labels, predicted_labels)
    
    results_dict = {
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'history': history.history
    }
    test_pred = model.predict(test_data_real)

    return model, results_dict, best_val_accuracy, test_pred



def train_and_test_tabnet(X_train, y_train, X_test, epochs):
    
    model = TabNetClassifier()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model.fit(X_train, y_train, max_epochs=epochs, patience=10, batch_size=128, virtual_batch_size=128)
    
    y_pred_val = model.predict(X_val)
    
    val_accuracy = accuracy_score(y_val, y_pred_val)

    print(f"Validation Accuracy: {val_accuracy}")

    test_pred = model.predict(X_test)
    
    return model, val_accuracy, test_pred


def preprocess_features_for_tabpfn(feature_array, label_array, max_samples=1024, max_features=99):
    if len(feature_array) > max_samples:
        feature_array, label_array = resample(
            feature_array, label_array, n_samples=max_samples, stratify=label_array, random_state=42
        )

    return feature_array, label_array

def train_tabpfn_classifier(X_train, y_train, X_test):
    X_train, y_train = preprocess_features_for_tabpfn(X_train[:, :99], y_train)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)
    classifier.fit(X_train, y_train)

    val_pred, p_eval = classifier.predict(X_val, return_winning_probability=True)
    val_accuracy = accuracy_score(y_val, val_pred)
    print('Validation Accuracy:', val_accuracy)

    test_pred = classifier.predict(X_test[:, :99])

    return classifier, val_accuracy, test_pred
