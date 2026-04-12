#include "librosa/core/audio.hpp"
#include "librosa/core/convert.hpp"
#include "librosa/util/exceptions.hpp"
#include "librosa/util/utils.hpp"
#ifdef LIBROSA_HAS_AUDIOTOOLBOX
#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>
#endif
#ifdef LIBROSA_HAS_SNDFILE
#include <sndfile.h>
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "../internal/fft.hpp"

namespace librosa {

namespace {
#ifdef LIBROSA_HAS_AUDIOTOOLBOX
    [[noreturn]] void throw_audio_toolbox_error(const std::string& operation, OSStatus status) {
        throw ParameterError(operation + " failed with OSStatus " +
                             std::to_string(static_cast<long>(status)));
    }

    void check_audio_toolbox_status(OSStatus status, const std::string& operation) {
        if (status != noErr) {
            throw_audio_toolbox_error(operation, status);
        }
    }

    CFURLRef create_audio_file_url(const std::string& path) {
        CFURLRef url = CFURLCreateFromFileSystemRepresentation(
            kCFAllocatorDefault,
            reinterpret_cast<const UInt8*>(path.c_str()),
            static_cast<CFIndex>(path.size()),
            false);

        if (!url) {
            throw ParameterError("Failed to create file URL for audio path: " + path);
        }

        return url;
    }

    class ExtAudioFileHandle {
    public:
        explicit ExtAudioFileHandle(const std::string& path) {
            CFURLRef url = create_audio_file_url(path);
            OSStatus status = ExtAudioFileOpenURL(url, &file_);
            CFRelease(url);

            if (status != noErr) {
                throw ParameterError("Failed to open audio file: " + path +
                                     " (OSStatus " +
                                     std::to_string(static_cast<long>(status)) + ")");
            }
        }

        ~ExtAudioFileHandle() {
            if (file_) {
                ExtAudioFileDispose(file_);
            }
        }

        ExtAudioFileRef get() const { return file_; }

        ExtAudioFileHandle(const ExtAudioFileHandle&) = delete;
        ExtAudioFileHandle& operator=(const ExtAudioFileHandle&) = delete;

    private:
        ExtAudioFileRef file_ = nullptr;
    };

    AudioStreamBasicDescription read_file_data_format(ExtAudioFileRef file) {
        AudioStreamBasicDescription format{};
        UInt32 size = sizeof(format);
        check_audio_toolbox_status(
            ExtAudioFileGetProperty(file, kExtAudioFileProperty_FileDataFormat,
                                    &size, &format),
            "ExtAudioFileGetProperty(FileDataFormat)");
        return format;
    }

    SInt64 read_file_length_frames(ExtAudioFileRef file) {
        SInt64 frames = 0;
        UInt32 size = sizeof(frames);
        check_audio_toolbox_status(
            ExtAudioFileGetProperty(file, kExtAudioFileProperty_FileLengthFrames,
                                    &size, &frames),
            "ExtAudioFileGetProperty(FileLengthFrames)");
        return frames;
    }

    AudioFileInfo read_audio_info(const std::string& path) {
        ExtAudioFileHandle file(path);
        AudioStreamBasicDescription file_format = read_file_data_format(file.get());
        SInt64 frames = read_file_length_frames(file.get());

        if (file_format.mSampleRate <= 0 || file_format.mChannelsPerFrame == 0) {
            throw ParameterError("Invalid audio file metadata for: " + path);
        }

        return AudioFileInfo{
            static_cast<Eigen::Index>(frames),
            static_cast<Real>(file_format.mSampleRate),
            static_cast<int>(file_format.mChannelsPerFrame),
            static_cast<Real>(frames) / static_cast<Real>(file_format.mSampleRate)
        };
    }

    void set_double_interleaved_client_format(ExtAudioFileRef file,
                                              double sample_rate,
                                              UInt32 channels) {
        AudioStreamBasicDescription client_format{};
        client_format.mSampleRate = sample_rate;
        client_format.mFormatID = kAudioFormatLinearPCM;
        client_format.mFormatFlags =
            kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked |
            kAudioFormatFlagsNativeEndian;
        client_format.mBytesPerPacket = sizeof(double) * channels;
        client_format.mFramesPerPacket = 1;
        client_format.mBytesPerFrame = sizeof(double) * channels;
        client_format.mChannelsPerFrame = channels;
        client_format.mBitsPerChannel = static_cast<UInt32>(sizeof(double) * 8);

        check_audio_toolbox_status(
            ExtAudioFileSetProperty(file, kExtAudioFileProperty_ClientDataFormat,
                                    sizeof(client_format), &client_format),
            "ExtAudioFileSetProperty(ClientDataFormat)");
    }

    AudioData read_audio_toolbox(const std::string& path,
                                 bool mono,
                                 Real offset,
                                 std::optional<Real> duration) {
        ExtAudioFileHandle file(path);
        AudioStreamBasicDescription file_format = read_file_data_format(file.get());
        SInt64 file_frames = read_file_length_frames(file.get());

        if (file_format.mSampleRate <= 0 || file_format.mChannelsPerFrame == 0) {
            throw ParameterError("Invalid audio file metadata for: " + path);
        }

        UInt32 channels = file_format.mChannelsPerFrame;
        set_double_interleaved_client_format(file.get(), file_format.mSampleRate,
                                             channels);

        SInt64 start_frame = static_cast<SInt64>(offset * file_format.mSampleRate);
        if (start_frame > file_frames) {
            throw ParameterError("offset exceeds audio duration for: " + path);
        }

        SInt64 frame_count = file_frames - start_frame;
        if (duration) {
            SInt64 requested_frames =
                static_cast<SInt64>(*duration * file_format.mSampleRate);
            frame_count = std::min(frame_count, requested_frames);
        }

        if (start_frame > 0) {
            check_audio_toolbox_status(ExtAudioFileSeek(file.get(), start_frame),
                                       "ExtAudioFileSeek");
        }

        AudioData result;
        result.sample_rate = static_cast<Real>(file_format.mSampleRate);
        result.channels = mono && channels > 1 ? 1 : static_cast<int>(channels);

        if (frame_count == 0) {
            result.samples.resize(result.channels, 0);
            return result;
        }

        std::vector<double> interleaved(static_cast<size_t>(frame_count) * channels);
        SInt64 frames_remaining = frame_count;
        SInt64 frames_read_total = 0;

        while (frames_remaining > 0) {
            UInt32 frames_this_read = static_cast<UInt32>(
                std::min<SInt64>(frames_remaining, 65536));

            AudioBufferList buffers{};
            buffers.mNumberBuffers = 1;
            buffers.mBuffers[0].mNumberChannels = channels;
            buffers.mBuffers[0].mDataByteSize =
                frames_this_read * channels * static_cast<UInt32>(sizeof(double));
            buffers.mBuffers[0].mData =
                interleaved.data() + static_cast<size_t>(frames_read_total) * channels;

            UInt32 actual_frames = frames_this_read;
            check_audio_toolbox_status(ExtAudioFileRead(file.get(), &actual_frames,
                                                        &buffers),
                                       "ExtAudioFileRead");

            if (actual_frames == 0) {
                break;
            }

            frames_read_total += actual_frames;
            frames_remaining -= actual_frames;
        }

        if (frames_read_total <= 0) {
            throw ParameterError("Failed to read audio data from: " + path);
        }

        interleaved.resize(static_cast<size_t>(frames_read_total) * channels);

        ArrayXXr samples(channels, frames_read_total);
        for (SInt64 frame = 0; frame < frames_read_total; ++frame) {
            for (UInt32 channel = 0; channel < channels; ++channel) {
                samples(channel, frame) =
                    interleaved[static_cast<size_t>(frame) * channels + channel];
            }
        }

        if (mono && channels > 1) {
            ArrayXr mono_samples = samples.colwise().mean();
            result.samples.resize(1, mono_samples.size());
            result.samples.row(0) = mono_samples.transpose();
            result.channels = 1;
        } else {
            result.samples = samples;
            result.channels = static_cast<int>(channels);
        }

        return result;
    }
#elif defined(LIBROSA_HAS_SNDFILE)
    AudioFileInfo read_audio_info(const std::string& path) {
        SF_INFO sfinfo;
        sfinfo.format = 0;

        SNDFILE* sndfile = sf_open(path.c_str(), SFM_READ, &sfinfo);
        if (!sndfile) {
            throw ParameterError("Failed to open audio file: " + path + " - " + sf_strerror(nullptr));
        }

        AudioFileInfo info{
            static_cast<Eigen::Index>(sfinfo.frames),
            static_cast<Real>(sfinfo.samplerate),
            sfinfo.channels,
            static_cast<Real>(sfinfo.frames) / sfinfo.samplerate
        };

        sf_close(sndfile);
        return info;
    }
#else
    [[noreturn]] void throw_no_audio_io(const char* fn) {
        throw ParameterError(std::string(fn) +
            " requires an audio I/O backend; this build was configured without "
            "Apple AudioToolbox or libsndfile.");
    }
#endif

    // Helper to compute next power of 2
    int next_power_of_2(int n) {
        int p = 1;
        while (p < n) p *= 2;
        return p;
    }

    // Helper for fast FFT size (simple version)
    int next_fast_fft_size(int n) {
        // For simplicity, use next power of 2
        // A more sophisticated version would find sizes that factor into 2, 3, 5
        return next_power_of_2(n);
    }

    struct SincResamplerSpec {
        int zero_crossings;
        Real beta;
        Real rolloff;
    };

    bool is_kaiser_resampler(const std::string& res_type) {
        return res_type == "kaiser" ||
               res_type == "kaiser_vhq" ||
               res_type == "kaiser_hq" ||
               res_type == "kaiser_mq" ||
               res_type == "kaiser_lq" ||
               res_type == "kaiser_fast";
    }

    SincResamplerSpec kaiser_resampler_spec(const std::string& res_type) {
        if (res_type == "kaiser_vhq") {
            return {96, 16.0, 0.975};
        }
        if (res_type == "kaiser_mq") {
            return {32, 10.0, 0.92};
        }
        if (res_type == "kaiser_lq") {
            return {16, 8.0, 0.86};
        }
        if (res_type == "kaiser_fast") {
            return {8, 6.0, 0.80};
        }

        // Match librosa's default high-quality resample path with a conservative
        // band-limited sinc interpolator.
        return {64, 14.0, 0.95};
    }

    Real sinc(Real x) {
        if (std::abs(x) < 1e-12) {
            return 1.0;
        }
        Real pix = constants::PI * x;
        return std::sin(pix) / pix;
    }

    Real modified_bessel_i0(Real x) {
        Real y = (x * x) * 0.25;
        Real term = 1.0;
        Real sum = 1.0;

        for (int k = 1; k < 80; ++k) {
            term *= y / static_cast<Real>(k * k);
            sum += term;
            if (term <= sum * std::numeric_limits<Real>::epsilon()) {
                break;
            }
        }

        return sum;
    }

    Real kaiser_window(Real distance, Real radius, Real beta, Real i0_beta) {
        Real x = distance / radius;
        if (std::abs(x) > 1.0) {
            return 0.0;
        }
        Real arg = beta * std::sqrt(std::max<Real>(0.0, 1.0 - x * x));
        return modified_bessel_i0(arg) / i0_beta;
    }

    ArrayXr kaiser_sinc_resample(const ArrayXr& y, Real ratio, Eigen::Index n_samples,
                                  const SincResamplerSpec& spec) {
        ArrayXr y_hat(n_samples);
        if (n_samples == 0) {
            return y_hat;
        }

        Real cutoff = std::min<Real>(1.0, ratio) * spec.rolloff;
        cutoff = std::min<Real>(1.0, std::max<Real>(cutoff, 1e-8));

        Eigen::Index radius = static_cast<Eigen::Index>(
            std::ceil(static_cast<Real>(spec.zero_crossings) / cutoff));
        radius = std::max<Eigen::Index>(radius, 1);

        Real i0_beta = modified_bessel_i0(spec.beta);

        for (Eigen::Index out_idx = 0; out_idx < n_samples; ++out_idx) {
            Real input_pos = static_cast<Real>(out_idx) / ratio;
            Eigen::Index center = static_cast<Eigen::Index>(std::floor(input_pos));
            Eigen::Index start = std::max<Eigen::Index>(0, center - radius);
            Eigen::Index stop = std::min<Eigen::Index>(y.size() - 1, center + radius);

            Real sample = 0.0;
            for (Eigen::Index in_idx = start; in_idx <= stop; ++in_idx) {
                Real distance = input_pos - static_cast<Real>(in_idx);
                Real window = kaiser_window(distance, static_cast<Real>(radius),
                                            spec.beta, i0_beta);
                Real weight = cutoff * sinc(cutoff * distance) * window;
                sample += y(in_idx) * weight;
            }

            y_hat(out_idx) = sample;
        }

        return y_hat;
    }
}

// ============================================================================
// Audio I/O
// ============================================================================

AudioData load(const std::string& path,
               std::optional<Real> sr,
               bool mono,
               Real offset,
               std::optional<Real> duration) {
#if !defined(LIBROSA_HAS_AUDIOTOOLBOX) && !defined(LIBROSA_HAS_SNDFILE)
    (void)path; (void)sr; (void)mono; (void)offset; (void)duration;
    throw_no_audio_io("load");
#else
    if (offset < 0) {
        throw ParameterError("offset must be non-negative");
    }

    if (duration && *duration < 0) {
        throw ParameterError("duration must be non-negative");
    }

#ifdef LIBROSA_HAS_AUDIOTOOLBOX
    AudioData result = read_audio_toolbox(path, mono, offset, duration);

    // Resample if requested
    if (sr && *sr != result.sample_rate) {
        if (result.num_samples() == 0) {
            result.sample_rate = *sr;
            return result;
        }
        if (result.channels == 1) {
            ArrayXr resampled = resample(result.samples.row(0), result.sample_rate, *sr);
            result.samples.resize(1, resampled.size());
            result.samples.row(0) = resampled.transpose();
        } else {
            ArrayXXr resampled(result.channels, 0);
            for (int c = 0; c < result.channels; ++c) {
                ArrayXr channel_resampled = resample(result.samples.row(c), result.sample_rate, *sr);
                if (c == 0) {
                    resampled.resize(result.channels, channel_resampled.size());
                }
                resampled.row(c) = channel_resampled.transpose();
            }
            result.samples = resampled;
        }
        result.sample_rate = *sr;
    }

    return result;
#else
    SF_INFO sfinfo;
    sfinfo.format = 0;

    SNDFILE* sndfile = sf_open(path.c_str(), SFM_READ, &sfinfo);
    if (!sndfile) {
        throw ParameterError("Failed to open audio file: " + path + " - " + sf_strerror(nullptr));
    }

    // Calculate start frame and frame count
    sf_count_t start_frame = static_cast<sf_count_t>(offset * sfinfo.samplerate);
    if (start_frame > sfinfo.frames) {
        sf_close(sndfile);
        throw ParameterError("offset exceeds audio duration for: " + path);
    }

    sf_count_t frame_count = sfinfo.frames - start_frame;

    if (duration) {
        sf_count_t requested_frames = static_cast<sf_count_t>(*duration * sfinfo.samplerate);
        frame_count = std::min(frame_count, requested_frames);
    }

    // Seek to start position
    if (start_frame > 0) {
        sf_seek(sndfile, start_frame, SEEK_SET);
    }

    AudioData result;
    result.sample_rate = static_cast<Real>(sfinfo.samplerate);
    result.channels = mono && sfinfo.channels > 1 ? 1 : sfinfo.channels;

    if (frame_count == 0) {
        result.samples.resize(result.channels, 0);
        sf_close(sndfile);
        if (sr && *sr != result.sample_rate) {
            result.sample_rate = *sr;
        }
        return result;
    }

    // Read audio data
    std::vector<double> buffer(frame_count * sfinfo.channels);
    sf_count_t frames_read = sf_readf_double(sndfile, buffer.data(), frame_count);

    sf_close(sndfile);

    if (frames_read <= 0) {
        throw ParameterError("Failed to read audio data from: " + path);
    }

    // Convert to our format (channels x samples)
    ArrayXXr samples(sfinfo.channels, frames_read);
    for (sf_count_t i = 0; i < frames_read; ++i) {
        for (int c = 0; c < sfinfo.channels; ++c) {
            samples(c, i) = buffer[i * sfinfo.channels + c];
        }
    }

    result.channels = sfinfo.channels;

    // Convert to mono if requested
    if (mono && sfinfo.channels > 1) {
        ArrayXr mono_samples = samples.colwise().mean();
        result.samples.resize(1, mono_samples.size());
        result.samples.row(0) = mono_samples.transpose();
        result.channels = 1;
    } else {
        result.samples = samples;
    }

    // Resample if requested
    if (sr && *sr != result.sample_rate) {
        if (result.channels == 1) {
            ArrayXr resampled = resample(result.samples.row(0), result.sample_rate, *sr);
            result.samples.resize(1, resampled.size());
            result.samples.row(0) = resampled.transpose();
        } else {
            ArrayXXr resampled(result.channels, 0);
            for (int c = 0; c < result.channels; ++c) {
                ArrayXr channel_resampled = resample(result.samples.row(c), result.sample_rate, *sr);
                if (c == 0) {
                    resampled.resize(result.channels, channel_resampled.size());
                }
                resampled.row(c) = channel_resampled.transpose();
            }
            result.samples = resampled;
        }
        result.sample_rate = *sr;
    }

    return result;
#endif
#endif
}

Real get_duration(const std::string& path) {
#if !defined(LIBROSA_HAS_AUDIOTOOLBOX) && !defined(LIBROSA_HAS_SNDFILE)
    (void)path;
    throw_no_audio_io("get_duration");
#else
    return get_audio_info(path).duration;
#endif
}

AudioFileInfo get_audio_info(const std::string& path) {
#if !defined(LIBROSA_HAS_AUDIOTOOLBOX) && !defined(LIBROSA_HAS_SNDFILE)
    (void)path;
    throw_no_audio_io("get_audio_info");
#else
    return read_audio_info(path);
#endif
}

Real get_duration(const ArrayXr& y, Real sr) {
    return static_cast<Real>(y.size()) / sr;
}

Real get_duration(const ArrayXXr& S, Real sr, int hop_length, int n_fft, bool center) {
    Eigen::Index n_frames = S.cols();
    Eigen::Index n_samples;

    if (center) {
        n_samples = (n_frames - 1) * hop_length;
    } else {
        n_samples = (n_frames - 1) * hop_length + n_fft;
    }

    return static_cast<Real>(n_samples) / sr;
}

Real get_samplerate(const std::string& path) {
#if !defined(LIBROSA_HAS_AUDIOTOOLBOX) && !defined(LIBROSA_HAS_SNDFILE)
    (void)path;
    throw_no_audio_io("get_samplerate");
#else
    return get_audio_info(path).sample_rate;
#endif
}

// ============================================================================
// Audio Processing
// ============================================================================

ArrayXr to_mono(const ArrayXXr& y) {
    util::valid_audio(y);

    if (y.rows() == 1) {
        return y.row(0);
    }

    return y.colwise().mean();
}

ArrayXr to_mono(const ArrayXr& y) {
    util::valid_audio(y);
    return y;
}

ArrayXr resample(const ArrayXr& y, Real orig_sr, Real target_sr,
                 const std::string& res_type,
                 bool fix, bool scale) {
    util::valid_audio(y);

    if (orig_sr <= 0.0 || target_sr <= 0.0) {
        throw ParameterError("orig_sr and target_sr must be positive");
    }

    if (orig_sr == target_sr) {
        return y;
    }

    Real ratio = target_sr / orig_sr;
    Eigen::Index n_samples = static_cast<Eigen::Index>(std::ceil(y.size() * ratio));

    ArrayXr y_hat;

    if (is_kaiser_resampler(res_type)) {
        y_hat = kaiser_sinc_resample(y, ratio, n_samples, kaiser_resampler_spec(res_type));
    } else if (res_type == "fft" || res_type == "scipy") {
        int n_fft = y.size();
        int n_out = n_samples;

        int n_freq_in = n_fft / 2 + 1;
        int n_freq_out = n_out / 2 + 1;

        std::vector<Complex> in_fft(n_freq_in);
        std::vector<Complex> out_fft(n_freq_out, Complex(0, 0));

        internal::RealFft forward_fft(n_fft);
        forward_fft.forward(y.data(), in_fft.data());

        int n_copy = std::min(n_freq_in, n_freq_out);
        for (int i = 0; i < n_copy; ++i) {
            out_fft[i] = in_fft[i];
        }

        y_hat.resize(n_out);
        internal::RealFft inverse_fft(n_out);
        inverse_fft.inverse(out_fft.data(), y_hat.data());

        y_hat /= static_cast<Real>(n_fft);
    } else if (res_type == "linear") {
        // Simple linear interpolation
        y_hat.resize(n_samples);
        for (Eigen::Index i = 0; i < n_samples; ++i) {
            Real idx = static_cast<Real>(i) / ratio;
            Eigen::Index idx0 = static_cast<Eigen::Index>(idx);
            Eigen::Index idx1 = std::min(idx0 + 1, y.size() - 1);
            Real frac = idx - idx0;
            y_hat(i) = (1.0 - frac) * y(idx0) + frac * y(idx1);
        }
    } else {
        throw ParameterError("Unknown resampling type: " + res_type);
    }

    if (fix) {
        y_hat = util::fix_length(y_hat, n_samples);
    }

    if (scale) {
        y_hat /= std::sqrt(ratio);
    }

    return y_hat;
}

ArrayXr autocorrelate(const ArrayXr& y, std::optional<int> max_size) {
    Eigen::Index n = y.size();
    Eigen::Index max_lag = max_size.value_or(n);
    max_lag = std::min(max_lag, n);

    int n_fft = next_fast_fft_size(2 * n - 1);

    if (n_fft < 32) {
        ArrayXr result(max_lag);
        for (Eigen::Index lag = 0; lag < max_lag; ++lag) {
            Real sum = 0.0;
            for (Eigen::Index i = 0; i + lag < n; ++i) {
                sum += y(i) * y(i + lag);
            }
            result(lag) = sum;
        }
        return result;
    }

    std::vector<Real> padded(n_fft, 0.0);
    std::copy(y.data(), y.data() + n, padded.begin());

    int n_freq = n_fft / 2 + 1;
    std::vector<Complex> freq(n_freq);

    internal::RealFft fft(n_fft);
    fft.forward(padded.data(), freq.data());

    for (int i = 0; i < n_freq; ++i) {
        Real re = freq[i].real();
        Real im = freq[i].imag();
        freq[i] = Complex(re * re + im * im, 0);
    }

    std::vector<Real> autocorr(n_fft);
    fft.inverse(freq.data(), autocorr.data());

    ArrayXr result(max_lag);
    for (Eigen::Index i = 0; i < max_lag; ++i) {
        result(i) = autocorr[i] / n_fft;
    }

    return result;
}

ArrayXXr autocorrelate(const ArrayXXr& y, std::optional<int> max_size, int axis) {
    if (axis < 0) axis = 2 + axis;

    if (axis == 0) {
        // Along rows
        Eigen::Index n = y.rows();
        Eigen::Index max_lag = max_size.value_or(n);
        max_lag = std::min(max_lag, n);

        ArrayXXr result(max_lag, y.cols());
        for (Eigen::Index j = 0; j < y.cols(); ++j) {
            ArrayXr col = y.col(j);
            result.col(j) = autocorrelate(col, static_cast<int>(max_lag));
        }
        return result;
    } else {
        // Along columns
        Eigen::Index n = y.cols();
        Eigen::Index max_lag = max_size.value_or(n);
        max_lag = std::min(max_lag, n);

        ArrayXXr result(y.rows(), max_lag);
        for (Eigen::Index i = 0; i < y.rows(); ++i) {
            ArrayXr row = y.row(i).transpose();
            result.row(i) = autocorrelate(row, static_cast<int>(max_lag)).transpose();
        }
        return result;
    }
}

ArrayXr lpc(const ArrayXr& y, int order) {
    if (order < 1) {
        throw ParameterError("LPC order must be > 0");
    }
    util::valid_audio(y);

    Eigen::Index n = y.size();
    ArrayXr ar_coeffs = ArrayXr::Zero(order + 1);
    ar_coeffs(0) = 1.0;

    ArrayXr ar_coeffs_prev = ar_coeffs;

    // Forward and backward prediction errors
    ArrayXr fwd_pred_error = y.segment(1, n - 1);
    ArrayXr bwd_pred_error = y.segment(0, n - 1);

    // Denominator for reflection coefficient
    Real den = (fwd_pred_error.square() + bwd_pred_error.square()).sum();
    Real epsilon = util::tiny<Real>();

    for (int i = 0; i < order; ++i) {
        // Compute reflection coefficient
        Real reflect_coeff = -2.0 * (bwd_pred_error * fwd_pred_error).sum() / (den + epsilon);

        // Update AR coefficients using Levinson-Durbin recursion
        ar_coeffs_prev = ar_coeffs;
        for (int j = 1; j <= i + 1; ++j) {
            ar_coeffs(j) = ar_coeffs_prev(j) + reflect_coeff * ar_coeffs_prev(i - j + 1);
        }

        // Update prediction errors
        ArrayXr fwd_pred_error_tmp = fwd_pred_error;
        fwd_pred_error = fwd_pred_error + reflect_coeff * bwd_pred_error;
        bwd_pred_error = bwd_pred_error + reflect_coeff * fwd_pred_error_tmp;

        // Update denominator
        Real q = 1.0 - reflect_coeff * reflect_coeff;
        den = q * den - bwd_pred_error(bwd_pred_error.size() - 1) * bwd_pred_error(bwd_pred_error.size() - 1)
              - fwd_pred_error(0) * fwd_pred_error(0);

        // Shift prediction errors
        fwd_pred_error = fwd_pred_error.segment(1, fwd_pred_error.size() - 1).eval();
        bwd_pred_error = bwd_pred_error.segment(0, bwd_pred_error.size() - 1).eval();
    }

    return ar_coeffs;
}

ArrayXXr lpc(const ArrayXXr& y, int order, int axis) {
    if (axis < 0) axis = 2 + axis;

    if (axis == 0) {
        ArrayXXr result(order + 1, y.cols());
        for (Eigen::Index j = 0; j < y.cols(); ++j) {
            ArrayXr col = y.col(j);
            result.col(j) = lpc(col, order);
        }
        return result;
    } else {
        ArrayXXr result(y.rows(), order + 1);
        for (Eigen::Index i = 0; i < y.rows(); ++i) {
            ArrayXr row = y.row(i).transpose();
            result.row(i) = lpc(row, order).transpose();
        }
        return result;
    }
}

Eigen::Array<bool, Eigen::Dynamic, 1>
zero_crossings(const ArrayXr& y,
               Real threshold,
               std::optional<Real> ref_magnitude,
               bool pad,
               bool zero_pos) {
    Real thresh = threshold;
    if (ref_magnitude) {
        thresh *= *ref_magnitude;
    }

    Eigen::Array<bool, Eigen::Dynamic, 1> result(y.size());
    result(0) = pad;

    for (Eigen::Index i = 1; i < y.size(); ++i) {
        Real x0 = y(i);
        Real x1 = y(i - 1);

        // Apply threshold
        if (-thresh <= x0 && x0 <= thresh) x0 = 0;
        if (-thresh <= x1 && x1 <= thresh) x1 = 0;

        if (zero_pos) {
            result(i) = std::signbit(x0) != std::signbit(x1);
        } else {
            int sign0 = (x0 > 0) ? 1 : ((x0 < 0) ? -1 : 0);
            int sign1 = (x1 > 0) ? 1 : ((x1 < 0) ? -1 : 0);
            result(i) = sign0 != sign1;
        }
    }

    return result;
}

Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
zero_crossings(const ArrayXXr& y,
               Real threshold,
               std::optional<Real> ref_magnitude,
               bool pad,
               bool zero_pos,
               int axis) {
    if (axis < 0) axis = 2 + axis;

    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(y.rows(), y.cols());

    if (axis == 0) {
        for (Eigen::Index j = 0; j < y.cols(); ++j) {
            ArrayXr col = y.col(j);
            result.col(j) = zero_crossings(col, threshold, ref_magnitude, pad, zero_pos);
        }
    } else {
        for (Eigen::Index i = 0; i < y.rows(); ++i) {
            ArrayXr row = y.row(i).transpose();
            Eigen::Array<bool, Eigen::Dynamic, 1> row_result =
                zero_crossings(row, threshold, ref_magnitude, pad, zero_pos);
            result.row(i) = row_result.transpose();
        }
    }

    return result;
}

// ============================================================================
// Signal Generation
// ============================================================================

ArrayXr clicks(const ArrayXr& times, Real sr,
               Real click_freq, Real click_duration,
               std::optional<int> length) {
    // Generate default click
    int click_length = static_cast<int>(click_duration * sr);
    ArrayXr click(click_length);
    ArrayXr envelope = ArrayXr::LinSpaced(click_length, 1.0, 0.0).square();

    for (int i = 0; i < click_length; ++i) {
        Real t = static_cast<Real>(i) / sr;
        click(i) = envelope(i) * std::sin(2.0 * constants::PI * click_freq * t);
    }

    // Convert times to positions
    std::vector<Eigen::Index> positions;
    for (Eigen::Index i = 0; i < times.size(); ++i) {
        positions.push_back(static_cast<Eigen::Index>(times(i) * sr));
    }

    // Determine output length
    Eigen::Index out_length;
    if (length) {
        out_length = *length;
    } else {
        out_length = positions.back() + click_length;
    }

    // Place clicks
    ArrayXr result = ArrayXr::Zero(out_length);
    for (auto start : positions) {
        if (start >= out_length) continue;

        Eigen::Index end = std::min(start + click_length, out_length);
        Eigen::Index click_end = end - start;
        result.segment(start, click_end) += click.head(click_end);
    }

    return result;
}

ArrayXr clicks_frames(const std::vector<Eigen::Index>& frames, Real sr,
                      int hop_length,
                      Real click_freq, Real click_duration,
                      std::optional<int> length) {
    ArrayXr times(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
        times(i) = static_cast<Real>(frames[i] * hop_length) / sr;
    }
    return clicks(times, sr, click_freq, click_duration, length);
}

ArrayXr tone(Real frequency, Real sr,
             std::optional<int> length,
             std::optional<Real> duration,
             std::optional<Real> phi) {
    Eigen::Index n;
    if (length) {
        n = *length;
    } else if (duration) {
        n = static_cast<Eigen::Index>(*duration * sr);
    } else {
        throw ParameterError("Either length or duration must be provided");
    }

    Real phase = phi.value_or(-constants::PI * 0.5);

    ArrayXr result(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        result(i) = std::cos(2.0 * constants::PI * frequency * i / sr + phase);
    }

    return result;
}

ArrayXr chirp(Real fmin, Real fmax, Real sr,
              std::optional<int> length,
              std::optional<Real> duration,
              bool linear,
              std::optional<Real> phi) {
    Real dur;
    if (length) {
        dur = static_cast<Real>(*length) / sr;
    } else if (duration) {
        dur = *duration;
    } else {
        throw ParameterError("Either length or duration must be provided");
    }

    Eigen::Index n = static_cast<Eigen::Index>(dur * sr);
    Real phase = phi.value_or(-constants::PI * 0.5);

    ArrayXr result(n);

    if (linear) {
        // Linear chirp: f(t) = fmin + (fmax - fmin) * t / dur
        Real k = (fmax - fmin) / dur;
        for (Eigen::Index i = 0; i < n; ++i) {
            Real t = static_cast<Real>(i) / sr;
            Real inst_phase = 2.0 * constants::PI * (fmin * t + 0.5 * k * t * t) + phase;
            result(i) = std::cos(inst_phase);
        }
    } else {
        // Exponential (logarithmic) chirp: f(t) = fmin * (fmax/fmin)^(t/dur)
        Real log_ratio = std::log(fmax / fmin);
        for (Eigen::Index i = 0; i < n; ++i) {
            Real t = static_cast<Real>(i) / sr;
            Real inst_phase = 2.0 * constants::PI * fmin * dur / log_ratio *
                             (std::exp(log_ratio * t / dur) - 1.0) + phase;
            result(i) = std::cos(inst_phase);
        }
    }

    return result;
}

// ============================================================================
// Mu-law Compression/Expansion
// ============================================================================

ArrayXr mu_compress(const ArrayXr& x, Real mu, bool quantize) {
    if (mu <= 0) {
        throw ParameterError("mu must be strictly positive");
    }
    if ((x < -1.0).any() || (x > 1.0).any()) {
        throw ParameterError("Input must be in range [-1, 1]");
    }

    ArrayXr x_comp = x.sign() * (1.0 + mu * x.abs()).log() / std::log(1.0 + mu);

    if (quantize) {
        ArrayXr result(x.size());
        int n_levels = static_cast<int>(1 + mu);
        int half_levels = n_levels / 2;

        for (Eigen::Index i = 0; i < x.size(); ++i) {
            // Map from [-1, 1] to [0, n_levels-1], then center around 0
            int level = static_cast<int>((x_comp(i) + 1.0) * 0.5 * (n_levels - 1) + 0.5);
            result(i) = level - half_levels;
        }
        return result;
    }

    return x_comp;
}

ArrayXr mu_expand(const ArrayXr& x, Real mu, bool quantize) {
    if (mu <= 0) {
        throw ParameterError("mu must be strictly positive");
    }

    ArrayXr x_norm = x;
    if (quantize) {
        x_norm = x * 2.0 / (1.0 + mu);
    }

    if ((x_norm < -1.0).any() || (x_norm > 1.0).any()) {
        throw ParameterError("Input must be in range [-1, 1]");
    }

    return x_norm.sign() / mu * (Eigen::pow(1.0 + mu, x_norm.abs()) - 1.0);
}

// Implement mono() method for AudioData
ArrayXr AudioData::mono() const {
    if (channels == 1) {
        return samples.row(0);
    }
    return samples.colwise().mean();
}

} // namespace librosa
