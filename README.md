# pENC: Parallel Audio Feature Extraction and Classification

![OpenCL](https://img.shields.io/badge/OpenCL-Enabled-green) ![Cython](https://img.shields.io/badge/Cython-Optimized-blue) ![Python](https://img.shields.io/badge/Python-3.13-blue) ![License](https://img.shields.io/badge/license-Restricted-red)

## Overview

**pENC** is a high-performance, hybrid C++/Python audio processing toolkit designed for large-scale, parallel feature extraction and classification of audio files. It leverages OpenCL for GPU acceleration, Cython for CPU parallelism, and Python for orchestration and machine learning, making it ideal for research and production environments where speed and flexibility are critical.

## Features

- **Blazing Fast MFCC Extraction**: Utilizes clFFT and OpenCL for GPU-accelerated FFT and power spectrum computation.
- **Cython and C++ Hybrid**: Seamlessly switches between Cython (CPU) and C++/OpenCL (GPU) for optimal performance.
- **Batch Audio Processing**: Efficiently processes thousands of audio files with minimal CPU and GPU idle time.
- **Custom OpenCL Kernels**: Includes custom kernels for Hamming window, mel filterbank, and DCT operations.
- **Python Orchestration**: Easy-to-use Python interface for feature extraction, model training, and evaluation.
- **Machine Learning Ready**: Integrates with scikit-learn for training and evaluating classifiers.
- **Flexible Build System**: CMake-based build with Visual Studio and CMake Tools extension support.
- **Extensive Dataset Support**: Designed to handle large datasets with robust error handling and caching.

## Project Structure

pseudocode

pENC/
  ├── classifier.py           # Main Python script for feature extraction and classification
  ├── mfcc_extractor.cpp      # C++/OpenCL backend for MFCC extraction
  ├── mfcc_kernel.cl          # OpenCL kernels for DSP operations
  ├── logic.py, shortcut.py   # Supporting Python modules
  ├── setup.py                # Cython build script
  ├── CMakeLists.txt          # CMake build configuration
  ├── dataset/                # Audio dataset and metadata
  ├── build/                  # Build artifacts
  ├── LICENSE                 # License file
  └── README.md               # This file

## Requirements

- **Python 3.13**
- **Cython**
- **OpenCL 2.0+ compatible GPU** (tested on Intel UHD Graphics 600)
- **clFFT library** (for GPU FFT)
- **Visual Studio 2022** (for C++/CMake build)
- **scikit-learn, numpy, pandas, soundfile, joblib** (Python dependencies)

## Installation & Build

1. **Install Python 3.13 and dependencies:**

   ```sh
   pip install cython numpy pandas scikit-learn soundfile joblib
   ```

2. **Install clFFT:**
   - Download from [clFFT GitHub](https://github.com/clMathLibraries/clFFT/releases)
   - Extract and add `clFFT/bin` to your PATH or use `os.add_dll_directory` in Python (already handled in `classifier.py`)
3. **Build C++/Cython extensions:**
   - Use the CMake Tools extension in VS Code or run:

     ```sh
     cmake -S . -B build
     cmake --build build --config Release
     ```

## Usage

- Place your audio files and metadata in the `dataset/` directory.
- Run the main pipeline:
  
  ```sh
  python classifier.py
  ```
  
- The script will extract MFCC features, train a classifier, and report accuracy.

## Customization

- **Switch between Debug/Release DLLs**: The Python code automatically selects the correct DLL based on how you run (F5 for Debug, Ctrl+F5 for Release).
- **Add new features**: Extend `mfcc_extractor.cpp` or `mfcc_kernel.cl` for custom DSP operations.
- **Dataset**: Update `dataset/Classifier.csv` and place audio files in `dataset/classifier/`.

## License

This project is licensed to Sayed Arham Abbas Rizvi, 2025. See [LICENSE](LICENSE) for details.

---

*Developed by Sayed Arham Abbas Rizvi. For research and private use only.*
