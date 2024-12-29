# distutils: language=c++
# cython: language_level=3

# Import the required libraries
import numpy as np
cimport numpy as np

# Use the correct NumPy API
np.import_array()

def invert_wave(np.ndarray[np.float64_t, ndim=1] audio_data):
    cdef int i
    cdef int n = len(audio_data)
    cdef np.ndarray[np.float64_t, ndim=1] inverted_audio = np.zeros(n, dtype=np.double)

    for i in range(n):
        inverted_audio[i] = -audio_data[i]
    
    return inverted_audio