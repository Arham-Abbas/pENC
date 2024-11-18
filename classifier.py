import os
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from shortcut import resolve_shortcut
from mfcc_extractor import extract_mfcc

# Resolve the path to the dataset folder using the shortcut
shortcut_path = 'dataset.lnk'
dataset_dir = resolve_shortcut(shortcut_path)
print(f"Resolved dataset directory: {dataset_dir}")

# Path to the classifier CSV file
csv_file = os.path.join(dataset_dir, 'Classifier.csv')
print(f"Classifier CSV path: {csv_file}")

audio_dir = os.path.join(dataset_dir, 'classifier')

# Function to extract features from an audio file using the compiled Cython module
def extract_features(file_path):
    signal, sample_rate = sf.read(file_path)
    mfccs = extract_mfcc(signal, sample_rate)
    return np.mean(mfccs, axis=0)

# Function to load and split data from the CSV file
def load_and_split_data(csv_path, test_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    X = []
    y = []
    missing_files = 0
    for _, row in df.iterrows():
        file_name = f"{row['fname']}.wav"
        file_path = os.path.join(audio_dir, file_name)
        if os.path.exists(file_path):
            features = extract_features(file_path)
            X.append(features)
            y.append(row['label'])
        else:
            print(f"File not found: {file_path}")
            missing_files += 1
    print(f"Total missing files: {missing_files}")
    
    # Convert lists to arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Load and split the data
X_train, X_test, y_train, y_test = load_and_split_data(csv_file)

# Ensure the data is not empty and has the correct shape
print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}")

if X_train.size == 0 or y_train.size == 0:
    raise ValueError("Training data is empty. Please check the CSV file and audio files.")

if X_train.ndim == 1:
    X_train = X_train.reshape(-1, 1)

if X_test.ndim == 1:
    X_test = X_test.reshape(-1, 1)

# Train a classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=2, max_features='sqrt')
model.fit(X_train, y_train)

# Test the model
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Save the trained model
joblib.dump(model, 'audio_classifier_model.pkl')

print("Model training and evaluation complete.")
