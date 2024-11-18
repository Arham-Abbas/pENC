# Import the required libraries
import numpy as np
cimport numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile

# Set the language level to 3 (Python 3)
# cython: language_level=3

def extract_mfcc(np.ndarray[np.float64_t, ndim=1] signal, int sample_rate, int num_cepstra=13):
    cdef int n_fft = 2048
    cdef int hop_length = 512
    cdef int n_mels = 128
    cdef int n_filters = 40
    
    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    # Framing
    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    # Windowing
    frames *= np.hamming(frame_length)
    
    # FFT and Power Spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    
    # Filter Banks
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate).astype(np.int32)
    
    fbank = np.zeros((n_mels, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, n_mels + 1):
        f_m_minus = bin[m - 1]   # Left
        f_m = bin[m]             # Center
        f_m_plus = bin[m + 1]    # Right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
            
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    
    # Mel-frequency Cepstral Coefficients (MFCCs)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_cepstra + 1)]  # Keep 2-13
    
    return mfcc
