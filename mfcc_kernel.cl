__kernel void hamming_window(__global double* window, int N) {
    int n = get_global_id(0);
    if (n < N) {
        window[n] = 0.54 - 0.46 * cos(2.0 * M_PI * n / (N - 1));
    }
}

__kernel void fft(__global double* input, __global double2* output, int N) {
    int k = get_global_id(0);
    if (k < N) {
        double real = 0.0;
        double imag = 0.0;
        for (int n = 0; n < N; n++) {
            double angle = -2.0 * M_PI * k * n / N;
            real += input[n] * cos(angle);
            imag += input[n] * sin(angle);
        }
        output[k] = (double2)(real, imag);
    }
}

__kernel void mel_filter_kernel(__global double* pow_frames, __global double* fbank, __global double* filter_banks, int num_frames, int n_mels, int n_fft) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < num_frames && j < n_mels) {
        double sum = 0.0;
        for (int k = 0; k < n_fft / 2 + 1; k++) {
            sum += pow_frames[i * (n_fft / 2 + 1) + k] * fbank[j * (n_fft / 2 + 1) + k];
        }
        filter_banks[i * n_mels + j] = sum > 0 ? 20 * log10(sum) : 1e-10;
    }
}

__kernel void dct_kernel(__global double* filter_banks, __global double* mfcc, int num_frames, int num_cepstra, int n_mels) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i < num_frames && j < num_cepstra) {
        double sum = 0.0;
        for (int k = 0; k < n_mels; k++) {
            sum += filter_banks[i * n_mels + k] * cos(M_PI * j * (k + 0.5) / n_mels);
        }
        mfcc[i * num_cepstra + j] = sum;
    }
}
