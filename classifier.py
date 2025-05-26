import os
import sys
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from shortcut import resolve_shortcut
import ctypes
import concurrent.futures


# Determine build type: Debug (F5) or Release (Ctrl+F5)
if sys.gettrace() is not None:
    build_type = "Debug"
else:
    build_type = "Release"

dll_dir = os.path.join(os.getcwd(), "build", "bin", build_type)
os.add_dll_directory(dll_dir)
os.add_dll_directory(r"C:\Windows\System32")
os.add_dll_directory(r"C:\Program Files\clFFT\bin")

dll_path = os.path.join(dll_dir, "mfcc_extractor.dll")
mfcc_lib = ctypes.CDLL(dll_path)

# Define the return type and argument types of the C++ function
mfcc_lib.extract_mfcc.restype = None
mfcc_lib.extract_mfcc.argtypes = [
    ctypes.POINTER(ctypes.c_double), 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.c_int, 
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_char_p
]

# Function to extract features from an audio file
def extract_mfcc(signal, sample_rate, num_cepstra=13, file_name=None):
    signal = np.array(signal, dtype=np.float64)
    num_frames = len(signal) // 512
    mfccs = np.zeros((num_frames, num_cepstra), dtype=np.float64)
    signal_ctypes = signal.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    mfccs_ctypes = mfccs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    mfcc_lib.extract_mfcc(signal_ctypes, signal.size, sample_rate, num_cepstra, mfccs_ctypes, file_name.encode('utf-8'))
    return mfccs

# Resolve the path to the dataset folder using the shortcut
shortcut_path = 'dataset.lnk'
dataset_dir = resolve_shortcut(shortcut_path)

# Path to the classifier CSV file
csv_file = os.path.join(dataset_dir, 'classifier.csv')
audio_dir = os.path.join(dataset_dir, 'classifier')

# Path to save the extracted features
features_file = 'extracted_features.pkl'

# Function to extract features from an audio file
def extract_features(file_path):
    signal, sample_rate = sf.read(file_path)
    try:
        mfccs = extract_mfcc(signal, sample_rate, file_name=os.path.basename(file_path))
        return np.mean(mfccs, axis=0)
    except Exception:
        return None

# Function to process a single file and extract features
def process_file(row):
    file_name = f"{row['fname']}.wav"
    file_path = os.path.join(audio_dir, file_name)
    if os.path.exists(file_path):
        features = extract_features(file_path)
        if features is not None:
            return features, row['label']
    return None, None

# Function to load and split data from the CSV file
def load_and_split_data(csv_path, test_size=0.2, random_state=42):
    if os.path.exists(features_file):
        # Load features from file
        data = joblib.load(features_file)
        features = data['features']
        labels = data['labels']
    else:
        # Extract features and save to file
        df = pd.read_csv(csv_path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            results = list(executor.map(process_file, df.to_dict('records')))
        
        # Filter out any None results from missing files or errors
        results = [res for res in results if res[0] is not None]
        
        features, labels = zip(*results)
        
        # Convert lists to arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Save features to file
        joblib.dump({'features': features, 'labels': labels}, features_file)
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test

# Load and split the data
try:
    x_train, x_test, y_train, y_test = load_and_split_data(csv_file)
except Exception as e:
    print(f"Critical error loading and splitting data: {e}")
    exit(1)

# Ensure the data is not empty and has the correct shape
if x_train.size == 0 or y_train.size == 0:
    raise ValueError("Training data is empty. Please check the CSV file and audio files.")

if x_train.ndim == 1:
    x_train = x_train.reshape(-1, 1)

if x_test.ndim == 1:
    x_test = x_test.reshape(-1, 1)

# Train a classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_leaf=2, max_features='sqrt')
model.fit(x_train, y_train)

# Test the model
test_predictions = model.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy:.2f}')

# Save the trained model
joblib.dump(model, 'audio_classifier_model.pkl')

print("Model training and evaluation complete.")
