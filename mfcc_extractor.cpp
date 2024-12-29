#include <CL/opencl.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void load_kernel_code(const std::string& filename, std::string& kernel_code) {
    std::ifstream file(filename);
    kernel_code.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

extern "C" {
    __declspec(dllexport) void extract_mfcc(double* signal, int signal_size, int sample_rate, int num_cepstra, double* mfcc, const char* audio_file_name) {
        try {
            std::cout << "Processing audio file: " << audio_file_name << std::endl;
            int n_fft = 2048;
            int hop_length = 512;
            int n_mels = 128;

            std::cout << "Starting pre-emphasis..." << std::endl;
            // Pre-emphasis
            std::vector<double> emphasized_signal(signal_size);
            emphasized_signal[0] = signal[0];
            for (size_t i = 1; i < signal_size; i++) {
                emphasized_signal[i] = signal[i] - 0.97 * signal[i - 1];
            }
            std::cout << "Pre-emphasis completed." << std::endl;

            std::cout << "Starting framing..." << std::endl;
            // Framing
            double frame_size = 0.025;
            double frame_stride = 0.01;
            int frame_length = static_cast<int>(std::round(frame_size * sample_rate));
            int frame_step = static_cast<int>(std::round(frame_stride * sample_rate));
            int num_frames = static_cast<int>(std::ceil((signal_size - frame_length) / static_cast<double>(frame_step)));

            std::vector<double> pad_signal = emphasized_signal;
            pad_signal.resize(num_frames * frame_step + frame_length, 0);

            std::vector<std::vector<double>> frames(num_frames, std::vector<double>(frame_length));
            std::vector<double> window(frame_length);
            std::cout << "Framing completed." << std::endl;

            std::cout << "Starting Hamming window computation..." << std::endl;
            // Hamming window computation using OpenCL
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            cl::Platform platform = platforms.front();
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            cl::Device device = devices.front();
            std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
            cl::Context context(device);
            cl::CommandQueue queue(context, device);

            std::string kernel_code;
            load_kernel_code("mfcc_kernel.cl", kernel_code);

            cl::Program::Sources sources;
            sources.push_back({kernel_code.c_str(), kernel_code.length()});
            cl::Program program(context, sources);
            program.build({device});

            cl::Kernel hamming_kernel(program, "hamming_window");
            cl::Buffer window_buffer(context, CL_MEM_READ_WRITE, sizeof(double) * frame_length);
            queue.enqueueWriteBuffer(window_buffer, CL_TRUE, 0, sizeof(double) * frame_length, window.data());
            hamming_kernel.setArg(0, window_buffer);
            hamming_kernel.setArg(1, frame_length);
            queue.enqueueNDRangeKernel(hamming_kernel, cl::NullRange, cl::NDRange(frame_length));
            queue.enqueueReadBuffer(window_buffer, CL_TRUE, 0, sizeof(double) * frame_length, window.data());
            std::cout << "Hamming window computation completed." << std::endl;

            std::cout << "Starting frame multiplication with Hamming window..." << std::endl;
            for (int i = 0; i < num_frames; i++) {
                for (int j = 0; j < frame_length; j++) {
                    frames[i][j] = pad_signal[i * frame_step + j] * window[j];
                }
            }
            std::cout << "Frame multiplication with Hamming window completed." << std::endl;

            std::cout << "Starting FFT and Power Spectrum computation..." << std::endl;
            // FFT and Power Spectrum using OpenCL
            std::vector<std::vector<double>> pow_frames(num_frames, std::vector<double>(n_fft / 2 + 1));
            std::vector<std::complex<double>> fft_output(n_fft / 2 + 1);

            cl::Kernel fft_kernel(program, "fft");
            cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, sizeof(double) * frame_length);
            cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_double2) * (n_fft / 2 + 1));

            for (int i = 0; i < num_frames; i++) {
                queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, sizeof(double) * frame_length, frames[i].data());
                fft_kernel.setArg(0, input_buffer);
                fft_kernel.setArg(1, output_buffer);
                fft_kernel.setArg(2, frame_length);
                queue.enqueueNDRangeKernel(fft_kernel, cl::NullRange, cl::NDRange(frame_length));
                queue.finish();
                std::vector<cl_double2> output(n_fft / 2 + 1);
                queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(cl_double2) * (n_fft / 2 + 1), output.data());

                for (size_t j = 0; j < output.size(); ++j) {
                    fft_output[j] = std::complex<double>(output[j].s[0], output[j].s[1]);
                }

                for (int j = 0; j < n_fft / 2 + 1; ++j) {
                    double mag = std::abs(fft_output[j]);
                    pow_frames[i][j] = (1.0 / n_fft) * (mag * mag);
                }
            }
            std::cout << "FFT and Power Spectrum computation completed." << std::endl;

            std::cout << "Starting filter bank computation..." << std::endl;
            // Compute filter banks
            double low_freq_mel = 0;
            double high_freq_mel = (2595 * std::log10(1.0 + (sample_rate / 2.0) / 700.0));
            std::vector<double> mel_points(n_mels + 2);
            std::vector<double> hz_points(n_mels + 2);
            std::vector<int> bin(n_mels + 2);

            for (int i = 0; i < n_mels + 2; i++) {
                mel_points[i] = low_freq_mel + i * (high_freq_mel - low_freq_mel) / (n_mels + 1);
                hz_points[i] = 700 * (std::pow(10.0, mel_points[i] / 2595.0) - 1.0);
                bin[i] = static_cast<int>(std::floor((n_fft + 1) * hz_points[i] / static_cast<double>(sample_rate)));
            }

            std::vector<std::vector<double>> fbank(n_mels, std::vector<double>(n_fft / 2 + 1));
            for (int m = 1; m <= n_mels; m++) {
                int f_m_minus = bin[m - 1];
                int f_m = bin[m];
                int f_m_plus = bin[m + 1];

                for (int k = f_m_minus; k < f_m; k++) {
                    fbank[m - 1][k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1]);
                }
                for (int k = f_m; k < f_m_plus; k++) {
                    fbank[m - 1][k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m]);
                }
            }
            std::cout << "Filter bank computation completed." << std::endl;

            std::cout << "Starting mel filterbank application..." << std::endl;
            std::vector<std::vector<double>> filter_banks(num_frames, std::vector<double>(n_mels));

            // Flatten pow_frames and fbank for OpenCL buffers
            std::vector<double> flat_pow_frames(num_frames * (n_fft / 2 + 1));
            std::vector<double> flat_fbank(n_mels * (n_fft / 2 + 1));
            std::vector<double> flat_filter_banks(num_frames * n_mels);

            for (int i = 0; i < num_frames; ++i) {
                for (int j = 0; j < (n_fft / 2 + 1); ++j) {
                    flat_pow_frames[i * (n_fft / 2 + 1) + j] = pow_frames[i][j];
                }
            }

            for (int i = 0; i < n_mels; ++i) {
                for (int j = 0; j < (n_fft / 2 + 1); ++j) {
                    flat_fbank[i * (n_fft / 2 + 1) + j] = fbank[i][j];
                }
            }

            // Apply mel filterbanks using OpenCL
            cl::Kernel mel_filter_kernel(program, "mel_filter_kernel");
            cl::Buffer pow_frames_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * flat_pow_frames.size(), flat_pow_frames.data());
            cl::Buffer fbank_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * flat_fbank.size(), flat_fbank.data());
            cl::Buffer filter_banks_buffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * flat_filter_banks.size());

            queue.enqueueWriteBuffer(pow_frames_buffer, CL_TRUE, 0, sizeof(double) * flat_pow_frames.size(), flat_pow_frames.data());
            queue.enqueueWriteBuffer(fbank_buffer, CL_TRUE, 0, sizeof(double) * flat_fbank.size(), flat_fbank.data());

            mel_filter_kernel.setArg(0, pow_frames_buffer);
            mel_filter_kernel.setArg(1, fbank_buffer);
            mel_filter_kernel.setArg(2, filter_banks_buffer);
            mel_filter_kernel.setArg(3, num_frames);
            mel_filter_kernel.setArg(4, n_mels);
            mel_filter_kernel.setArg(5, n_fft);

            std::cout << "Starting mel filterbank kernel execution..." << std::endl;
            cl_int err = queue.enqueueNDRangeKernel(mel_filter_kernel, cl::NullRange, cl::NDRange(num_frames, n_mels));
            if (err != CL_SUCCESS) {
                std::cerr << "Error during mel filterbank kernel execution: " << err << std::endl;
                abort();
            }
            queue.finish();
            queue.enqueueReadBuffer(filter_banks_buffer, CL_TRUE, 0, sizeof(double) * flat_filter_banks.size(), flat_filter_banks.data());
            std::cout << "Mel filterbank kernel execution completed." << std::endl;

            // Unflatten filter_banks
            for (int i = 0; i < num_frames; ++i) {
                for (int j = 0; j < n_mels; ++j) {
                    filter_banks[i][j] = flat_filter_banks[i * n_mels + j];
                }
            }

            std::cout << "Starting DCT computation..." << std::endl;
            // Compute DCT of filter banks using OpenCL
            cl::Kernel dct_kernel(program, "dct_kernel");
            cl::Buffer filter_banks_cl_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(double) * flat_filter_banks.size(), flat_filter_banks.data());
            cl::Buffer mfcc_buffer(context, CL_MEM_WRITE_ONLY, sizeof(double) * num_frames * num_cepstra);

            queue.enqueueWriteBuffer(filter_banks_cl_buffer, CL_TRUE, 0, sizeof(double) * flat_filter_banks.size(), flat_filter_banks.data());
            dct_kernel.setArg(0, filter_banks_cl_buffer);
            dct_kernel.setArg(1, mfcc_buffer);
            dct_kernel.setArg(2, num_frames);
            dct_kernel.setArg(3, num_cepstra);
            dct_kernel.setArg(4, n_mels);

            std::cout << "Starting DCT kernel execution..." << std::endl;
            err = queue.enqueueNDRangeKernel(dct_kernel, cl::NullRange, cl::NDRange(num_frames, num_cepstra));
            if (err != CL_SUCCESS) {
                std::cerr << "Error during DCT kernel execution: " << err << std::endl;
                abort();
            }
            queue.finish();
            queue.enqueueReadBuffer(mfcc_buffer, CL_TRUE, 0, sizeof(double) * num_frames * num_cepstra, mfcc);
            std::cout << "DCT kernel execution completed." << std::endl;

            // Clean up OpenCL resources
            std::cout << "Cleaning up OpenCL resources..." << std::endl;
            clReleaseMemObject(window_buffer());
            clReleaseMemObject(input_buffer());
            clReleaseMemObject(output_buffer());
            clReleaseMemObject(pow_frames_buffer());
            clReleaseMemObject(fbank_buffer());
            clReleaseMemObject(filter_banks_buffer());
            clReleaseMemObject(filter_banks_cl_buffer());
            clReleaseMemObject(mfcc_buffer());
            clReleaseCommandQueue(queue());
            clReleaseContext(context());
            std::cout << "OpenCL resources cleaned up." << std::endl;

            std::cout << "Completed extract_mfcc function" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Exception caught in extract_mfcc: " << e.what() << std::endl;
            abort();
        } catch (...) {
            std::cerr << "Unknown exception caught in extract_mfcc" << std::endl;
            abort();
        }
    }
}
