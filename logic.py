import pandas as pd
import soundfile as sf
import numpy as np
import os
from shortcut import resolve_shortcut
from inverse_wave import invert_wave
from concurrent.futures import ThreadPoolExecutor, as_completed

# Resolve the path to the dataset folder using the shortcut
shortcut_path = 'dataset.lnk'
dataset_dir = resolve_shortcut(shortcut_path)
csv_path = os.path.join(dataset_dir, 'metadata.csv')
audio_dir = os.path.join(dataset_dir, 'original')
inverted_dir = os.path.join(dataset_dir, 'inverted')
mixed_dir = os.path.join(dataset_dir, 'mixed')

# Read the CSV file
df = pd.read_csv(csv_path)

# Ensure the output directories exist
os.makedirs(inverted_dir, exist_ok=True)
os.makedirs(mixed_dir, exist_ok=True)

# Function to process a single audio file
def process_audio_file(row):
    audio_file = row['Archivo']
    audio_path = os.path.join(audio_dir, audio_file)
    inverted_audio_path = os.path.join(inverted_dir, f'inverted_{audio_file}')
    mixed_audio_path = os.path.join(mixed_dir, f'mixed_{audio_file}')

    # Load original audio file
    audio_data, sample_rate = sf.read(audio_path)
    if audio_data is None:
        return f"Error loading {audio_path}"

    # Invert the audio waveform using Cython
    inverted_audio = invert_wave(audio_data)
    if inverted_audio is None:
        return f"Error inverting {audio_file}"

    # Ensure the inverted audio is properly formatted
    inverted_audio = np.array(inverted_audio)

    # Save the inverted audio to a new file
    sf.write(inverted_audio_path, inverted_audio, sample_rate)

    # Create stereo audio with original as left and inverted as right
    if len(audio_data.shape) == 1:  # Mono to Stereo
        stereo_audio = np.column_stack((audio_data, inverted_audio))
    elif len(audio_data.shape) == 2:  # Stereo
        stereo_audio = np.column_stack((audio_data[:, 0], inverted_audio[:, 0]))  # Assuming both are stereo

    # Normalize mixed audio to prevent clipping
    max_val = np.max(np.abs(stereo_audio))
    if max_val > 1:
        stereo_audio = stereo_audio / max_val

    # Save the mixed audio to a new file
    sf.write(mixed_audio_path, stereo_audio, sample_rate)

    # Return the data needed for all_data
    return audio_file, audio_data, inverted_audio, stereo_audio

# Function to process audio files with multithreading
def process_audio_files_multithreaded():
    all_data = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_row = {executor.submit(process_audio_file, row): row for _, row in df.iterrows()}
        for future in as_completed(future_to_row):
            try:
                result = future.result()
                if isinstance(result, str):
                    print(result)
                else:
                    all_data.append(result)
            except Exception as exc:
                print(f'Generated an exception: {exc}')
    return all_data

if __name__ == "__main__":
    all_data = process_audio_files_multithreaded()
