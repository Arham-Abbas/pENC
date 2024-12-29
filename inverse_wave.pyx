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

    cdef double* audio_ptr = &audio_data[0]
    cdef double* inverted_ptr = &inverted_audio[0]

    for i in range(0, n, 4):
        inverted_ptr[i] = -audio_ptr[i]
        if i + 1 < n:
            inverted_ptr[i + 1] = -audio_ptr[i + 1]
        if i + 2 < n:
            inverted_ptr[i + 2] = -audio_ptr[i + 2]
        if i + 3 < n:
            inverted_ptr[i + 3] = -audio_ptr[i + 3]
    
    return inverted_audio
