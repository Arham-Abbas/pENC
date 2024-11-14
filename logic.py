import pandas as pd
import soundfile as sf
import numpy as np
import os
from inverse_wave import invert_wave  # Ensure this import points to the compiled Cython module

# Paths to directories
dataset_dir = 'dataset'
csv_path = os.path.join(dataset_dir, 'metadata.csv')
audio_dir = os.path.join(dataset_dir, 'original')
inverted_dir = os.path.join(dataset_dir, 'inverted')
merged_dir = os.path.join(dataset_dir, 'merged')

# Read the CSV file
df = pd.read_csv(csv_path)

# Ensure the output directories exist
os.makedirs(inverted_dir, exist_ok=True)
os.makedirs(merged_dir, exist_ok=True)

# Function to process audio files
def process_audio_files():
    all_data = []

    for _, row in df.iterrows():
        audio_file = row['Archivo']
        audio_path = os.path.join(audio_dir, audio_file)
        inverted_audio_path = os.path.join(inverted_dir, f'inverted_{audio_file}')
        merged_audio_path = os.path.join(merged_dir, f'merged_{audio_file}')

        # Load original audio file
        audio_data, sample_rate = sf.read(audio_path)
        
        if audio_data is None:
            print(f"Error loading {audio_path}")
            continue

        # Invert the audio waveform using Cython
        inverted_audio = invert_wave(audio_data)
        
        if inverted_audio is None:
            print(f"Error inverting {audio_file}")
            continue

        # Ensure the inverted audio is properly formatted
        inverted_audio = np.array(inverted_audio)
        print(f"Original audio data: {audio_data[:10]}")
        print(f"Inverted audio data: {inverted_audio[:10]}")

        # Save the inverted audio to a new file
        sf.write(inverted_audio_path, inverted_audio, sample_rate)
        print(f"Inverted audio saved to {inverted_audio_path}")

        # Merge original and inverted audio streams
        if len(audio_data) == len(inverted_audio):
            merged_audio = audio_data + inverted_audio
        else:
            min_length = min(len(audio_data), len(inverted_audio))
            merged_audio = audio_data[:min_length] + inverted_audio[:min_length]

        # Normalize merged audio to prevent clipping
        max_val = np.max(np.abs(merged_audio))
        if max_val > 0:
            merged_audio = merged_audio / max_val
        
        print(f"Merged audio data: {merged_audio[:10]}")

        # Save the merged audio to a new file
        sf.write(merged_audio_path, merged_audio, sample_rate)
        print(f"Merged audio saved to {merged_audio_path}")

        # Append data for plotting later
        all_data.append((audio_file, audio_data, inverted_audio, merged_audio))

    return all_data

if __name__ == "__main__":
    all_data = process_audio_files()
