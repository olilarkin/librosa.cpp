#include <gtest/gtest.h>
#include <librosa/core/audio.hpp>
#include <librosa/util/exceptions.hpp>
#ifdef LIBROSA_HAS_AUDIOTOOLBOX
#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>
#else
#include <sndfile.h>
#endif
#include <cmath>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using namespace librosa;

namespace {

// Random number utility
std::mt19937 rng(628318530);

ArrayXr random_array(Eigen::Index size) {
    std::normal_distribution<Real> dist(0.0, 1.0);
    ArrayXr result(size);
    for (Eigen::Index i = 0; i < size; ++i) {
        result(i) = dist(rng);
    }
    return result;
}

class TempAudioFile {
public:
    TempAudioFile(int sample_rate, int channels, int64_t frames) {
        char path_template[] = "/tmp/librosa-audio-XXXXXX.wav";
        int fd = mkstemps(path_template, 4);
        if (fd == -1) {
            throw std::runtime_error("Failed to create temporary audio path");
        }

        ::close(fd);
        path_ = path_template;

        std::vector<double> buffer(static_cast<size_t>(frames) * channels);
        for (int64_t frame = 0; frame < frames; ++frame) {
            for (int channel = 0; channel < channels; ++channel) {
                buffer[static_cast<size_t>(frame) * channels + channel] =
                    static_cast<double>(channel + 1) * (static_cast<double>(frame + 1) / frames);
            }
        }

#ifdef LIBROSA_HAS_AUDIOTOOLBOX
        CFURLRef url = CFURLCreateFromFileSystemRepresentation(
            kCFAllocatorDefault,
            reinterpret_cast<const UInt8*>(path_.c_str()),
            static_cast<CFIndex>(path_.size()),
            false);
        if (!url) {
            throw std::runtime_error("Failed to create temporary audio file URL");
        }

        AudioStreamBasicDescription file_format{};
        file_format.mSampleRate = sample_rate;
        file_format.mFormatID = kAudioFormatLinearPCM;
        file_format.mFormatFlags =
            kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked |
            kAudioFormatFlagsNativeEndian;
        file_format.mBytesPerPacket = sizeof(float) * channels;
        file_format.mFramesPerPacket = 1;
        file_format.mBytesPerFrame = sizeof(float) * channels;
        file_format.mChannelsPerFrame = static_cast<UInt32>(channels);
        file_format.mBitsPerChannel = static_cast<UInt32>(sizeof(float) * 8);

        ExtAudioFileRef audio_file = nullptr;
        OSStatus status = ExtAudioFileCreateWithURL(
            url, kAudioFileWAVEType, &file_format, nullptr,
            kAudioFileFlags_EraseFile, &audio_file);
        CFRelease(url);
        if (status != noErr || !audio_file) {
            throw std::runtime_error("Failed to open temporary audio file for writing");
        }

        AudioStreamBasicDescription client_format{};
        client_format.mSampleRate = sample_rate;
        client_format.mFormatID = kAudioFormatLinearPCM;
        client_format.mFormatFlags =
            kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked |
            kAudioFormatFlagsNativeEndian;
        client_format.mBytesPerPacket = sizeof(double) * channels;
        client_format.mFramesPerPacket = 1;
        client_format.mBytesPerFrame = sizeof(double) * channels;
        client_format.mChannelsPerFrame = static_cast<UInt32>(channels);
        client_format.mBitsPerChannel = static_cast<UInt32>(sizeof(double) * 8);

        status = ExtAudioFileSetProperty(audio_file,
                                         kExtAudioFileProperty_ClientDataFormat,
                                         sizeof(client_format),
                                         &client_format);
        if (status == noErr) {
            AudioBufferList buffers{};
            buffers.mNumberBuffers = 1;
            buffers.mBuffers[0].mNumberChannels = static_cast<UInt32>(channels);
            buffers.mBuffers[0].mDataByteSize =
                static_cast<UInt32>(buffer.size() * sizeof(double));
            buffers.mBuffers[0].mData = buffer.data();
            status = ExtAudioFileWrite(audio_file, static_cast<UInt32>(frames), &buffers);
        }

        ExtAudioFileDispose(audio_file);

        if (status != noErr) {
            throw std::runtime_error("Failed to write temporary audio file");
        }
#else
        SF_INFO sfinfo;
        sfinfo.frames = frames;
        sfinfo.samplerate = sample_rate;
        sfinfo.channels = channels;
        sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
        sfinfo.sections = 0;
        sfinfo.seekable = 0;

        SNDFILE* sndfile = sf_open(path_.c_str(), SFM_WRITE, &sfinfo);
        if (!sndfile) {
            throw std::runtime_error("Failed to open temporary audio file for writing");
        }

        sf_count_t written = sf_writef_double(sndfile, buffer.data(), frames);
        sf_close(sndfile);

        if (written != frames) {
            throw std::runtime_error("Failed to write temporary audio file");
        }
#endif
    }

    ~TempAudioFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    const std::string& path() const { return path_; }

private:
    std::string path_;
};

} // anonymous namespace

// ============================================================================
// To Mono Tests
// ============================================================================

TEST(ToMonoTest, StereoToMono) {
    // Create stereo signal with different channels
    ArrayXXr stereo(2, 100);
    stereo.row(0).setConstant(1.0);
    stereo.row(1).setConstant(-1.0);

    ArrayXr mono = to_mono(stereo);

    EXPECT_EQ(mono.size(), 100);

    // Mono should be average of channels
    for (Eigen::Index i = 0; i < mono.size(); ++i) {
        EXPECT_NEAR(mono(i), 0.0, 1e-10);
    }
}

TEST(ToMonoTest, MonoPassthrough) {
    ArrayXr input = ArrayXr::Ones(100);
    ArrayXr output = to_mono(input);

    EXPECT_EQ(output.size(), input.size());
    for (Eigen::Index i = 0; i < input.size(); ++i) {
        EXPECT_NEAR(output(i), input(i), 1e-10);
    }
}

// ============================================================================
// Resample Tests
// ============================================================================

class ResampleTest : public ::testing::TestWithParam<std::tuple<Real, Real, std::string>> {};

TEST_P(ResampleTest, BasicResample) {
    auto [orig_sr, target_sr, res_type] = GetParam();

    // Create a test signal
    ArrayXr y = ArrayXr::Random(static_cast<Eigen::Index>(orig_sr));

    ArrayXr y_resampled = resample(y, orig_sr, target_sr, res_type, true, false);

    // Check output length is approximately correct
    Eigen::Index expected_length = static_cast<Eigen::Index>(y.size() * target_sr / orig_sr);
    EXPECT_NEAR(y_resampled.size(), expected_length, 2);

    // If same sample rate, should be unchanged
    if (std::abs(orig_sr - target_sr) < 1e-10) {
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            EXPECT_NEAR(y_resampled(i), y(i), 1e-10);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(ResampleTests, ResampleTest,
    ::testing::Values(
#ifdef LIBROSA_TEST_FFT_BACKEND_PFFFT
        std::make_tuple(24000.0, 8000.0, "fft"),
        std::make_tuple(24000.0, 24000.0, "fft"),
        std::make_tuple(48000.0, 24000.0, "fft"),
#else
        std::make_tuple(22050.0, 8000.0, "fft"),
        std::make_tuple(22050.0, 22050.0, "fft"),
        std::make_tuple(44100.0, 22050.0, "fft"),
#endif
        std::make_tuple(22050.0, 8000.0, "linear"),
        std::make_tuple(44100.0, 22050.0, "linear"),
        std::make_tuple(22050.0, 8000.0, "kaiser_hq"),
        std::make_tuple(44100.0, 22050.0, "kaiser_hq")
    ));

TEST(ResampleTest, ScalePreservesEnergyDensity) {
#ifdef LIBROSA_TEST_FFT_BACKEND_PFFFT
    Real orig_sr = 24000;
    Real target_sr = 12000;

    ArrayXr y = random_array(24000);
#else
    Real orig_sr = 22050;
    Real target_sr = 11025;

    ArrayXr y = random_array(22050);
#endif

    ArrayXr y_resampled = resample(y, orig_sr, target_sr, "fft", true, true);

    // Scale preserves energy density (mean power per sample), not total energy
    Real orig_mean_power = (y * y).mean();
    Real resamp_mean_power = (y_resampled * y_resampled).mean();

    EXPECT_NEAR(orig_mean_power, resamp_mean_power, orig_mean_power * 0.1);
}

TEST(ResampleTest, KaiserHqPreservesToneAmplitude) {
    Real orig_sr = 22050;
    Real target_sr = 11025;
    Eigen::Index n = static_cast<Eigen::Index>(orig_sr);

    ArrayXr t = ArrayXr::LinSpaced(n, 0, static_cast<Real>(n - 1) / orig_sr);
    ArrayXr y = (2.0 * constants::PI * 440.0 * t).sin();

    ArrayXr y_resampled = resample(y, orig_sr, target_sr, "kaiser_hq", true, false);

    EXPECT_EQ(y_resampled.size(), static_cast<Eigen::Index>(target_sr));
    EXPECT_GT(y_resampled.abs().maxCoeff(), 0.8);
}

TEST(ResampleTest, InvalidSampleRatesThrow) {
    ArrayXr y = ArrayXr::Ones(32);

    EXPECT_THROW(resample(y, 0.0, 22050.0), ParameterError);
    EXPECT_THROW(resample(y, 22050.0, -1.0), ParameterError);
}

// ============================================================================
// Autocorrelate Tests
// ============================================================================

TEST(AutocorrelateTest, Basic1D) {
    ArrayXr y = random_array(100);

    ArrayXr ac = autocorrelate(y, std::nullopt);

    // Autocorrelation at lag 0 should be maximum
    Real max_val = ac.maxCoeff();
    EXPECT_NEAR(ac(0), max_val, 1e-10);
}

TEST(AutocorrelateTest, WithMaxSize) {
    ArrayXr y = random_array(100);

    ArrayXr ac = autocorrelate(y, 50);

    EXPECT_EQ(ac.size(), 50);
}

TEST(AutocorrelateTest, PureFrequency) {
    // Create a pure tone
    Real freq = 100.0;
    Real sr = 1000.0;
    Eigen::Index n = 1000;

    ArrayXr t = ArrayXr::LinSpaced(n, 0, static_cast<Real>(n - 1) / sr);
    ArrayXr y = (2.0 * constants::PI * freq * t).cos();

    ArrayXr ac = autocorrelate(y, std::nullopt);

    // Autocorrelation should be periodic at the signal frequency
    // Period in samples = sr / freq = 10
    Eigen::Index period = static_cast<Eigen::Index>(sr / freq);

    // Check that autocorrelation peaks at multiples of the period
    if (ac.size() > 2 * period) {
        EXPECT_GT(ac(period), ac(period / 2));
    }
}

// ============================================================================
// LPC Tests
// ============================================================================

TEST(LPCTest, BasicLPC) {
    ArrayXr y = random_array(1000);
    int order = 10;

    ArrayXr coeffs = lpc(y, order);

    // Should return order + 1 coefficients
    EXPECT_EQ(coeffs.size(), order + 1);

    // First coefficient should be 1
    EXPECT_NEAR(coeffs(0), 1.0, 1e-10);
}

TEST(LPCTest, SineWave) {
    // LPC of a sine wave should show periodicity
    Real freq = 100.0;
    Real sr = 8000.0;
    Eigen::Index n = 1000;

    ArrayXr t = ArrayXr::LinSpaced(n, 0, static_cast<Real>(n - 1) / sr);
    ArrayXr y = (2.0 * constants::PI * freq * t).sin();

    ArrayXr coeffs = lpc(y, 4);

    EXPECT_EQ(coeffs.size(), 5);
    EXPECT_NEAR(coeffs(0), 1.0, 1e-10);
}

// ============================================================================
// Zero Crossings Tests
// ============================================================================

TEST(ZeroCrossingsTest, BasicCrossings) {
    ArrayXr y(5);
    y << 1.0, -1.0, 1.0, -1.0, 1.0;

    auto zc = zero_crossings(y, 0.0, std::nullopt, false, true);

    EXPECT_EQ(zc.size(), y.size());

    // Should have crossings at indices 1, 2, 3, 4
    EXPECT_FALSE(zc(0));
    EXPECT_TRUE(zc(1));
    EXPECT_TRUE(zc(2));
    EXPECT_TRUE(zc(3));
    EXPECT_TRUE(zc(4));
}

TEST(ZeroCrossingsTest, NoCrossings) {
    ArrayXr y = ArrayXr::Ones(10);

    auto zc = zero_crossings(y, 0.0, std::nullopt, false, true);

    // Should have no crossings (all positive)
    int count = 0;
    for (Eigen::Index i = 0; i < zc.size(); ++i) {
        if (zc(i)) count++;
    }
    EXPECT_EQ(count, 0);
}

TEST(ZeroCrossingsTest, WithPad) {
    ArrayXr y(5);
    y << 1.0, -1.0, 1.0, -1.0, 1.0;

    auto zc = zero_crossings(y, 0.0, std::nullopt, true, true);

    // With pad=true, first sample is considered a crossing
    EXPECT_TRUE(zc(0));
}

// ============================================================================
// Signal Generation Tests
// ============================================================================

TEST(ToneTest, BasicTone) {
    Real freq = 440.0;
    Real sr = 22050;
    Eigen::Index length = 22050;

    ArrayXr y = tone(freq, sr, length, std::nullopt, std::nullopt);

    EXPECT_EQ(y.size(), length);

    // Should be bounded by [-1, 1]
    EXPECT_LE(y.maxCoeff(), 1.0 + 1e-10);
    EXPECT_GE(y.minCoeff(), -1.0 - 1e-10);
}

TEST(ToneTest, WithDuration) {
    Real freq = 440.0;
    Real sr = 22050;
    Real duration = 1.0;

    ArrayXr y = tone(freq, sr, std::nullopt, duration, std::nullopt);

    EXPECT_EQ(y.size(), static_cast<Eigen::Index>(sr * duration));
}

TEST(ToneTest, WithPhase) {
    Real freq = 440.0;
    Real sr = 22050;
    Eigen::Index length = 100;
    Real phi = constants::PI / 2;  // Start at maximum

    ArrayXr y = tone(freq, sr, length, std::nullopt, phi);

    // With phi = pi/2, cos(0 + pi/2) = 0, so first sample should be near 0
    // Actually cos(pi/2) = 0
    EXPECT_NEAR(y(0), 0.0, 1e-10);
}

TEST(ChirpTest, LinearChirp) {
    Real fmin = 100.0;
    Real fmax = 1000.0;
    Real sr = 22050;
    Eigen::Index length = 22050;

    ArrayXr y = chirp(fmin, fmax, sr, length, std::nullopt, true, std::nullopt);

    EXPECT_EQ(y.size(), length);

    // Should be bounded
    EXPECT_LE(y.maxCoeff(), 1.0 + 1e-10);
    EXPECT_GE(y.minCoeff(), -1.0 - 1e-10);
}

TEST(ChirpTest, ExponentialChirp) {
    Real fmin = 100.0;
    Real fmax = 1000.0;
    Real sr = 22050;
    Eigen::Index length = 22050;

    ArrayXr y = chirp(fmin, fmax, sr, length, std::nullopt, false, std::nullopt);

    EXPECT_EQ(y.size(), length);

    // Should be bounded
    EXPECT_LE(y.maxCoeff(), 1.0 + 1e-10);
    EXPECT_GE(y.minCoeff(), -1.0 - 1e-10);
}

TEST(ClicksTest, BasicClicks) {
    ArrayXr times(3);
    times << 0.0, 0.5, 1.0;
    Real sr = 22050;

    ArrayXr y = clicks(times, sr, 1000.0, 0.1, std::nullopt);

    // Should have length to contain all clicks
    EXPECT_GE(y.size(), static_cast<Eigen::Index>(sr));
}

TEST(ClicksTest, WithLength) {
    ArrayXr times(2);
    times << 0.0, 0.5;
    Real sr = 22050;
    Eigen::Index length = 44100;

    ArrayXr y = clicks(times, sr, 1000.0, 0.1, length);

    EXPECT_EQ(y.size(), length);
}

// ============================================================================
// Mu-law Tests
// ============================================================================

TEST(MuLawTest, CompressExpand) {
    ArrayXr x = ArrayXr::LinSpaced(101, -1.0, 1.0);
    Real mu = 255;

    ArrayXr compressed = mu_compress(x, mu, false);
    ArrayXr expanded = mu_expand(compressed, mu, false);

    // Should round-trip
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(expanded(i), x(i), 1e-5);
    }
}

TEST(MuLawTest, QuantizedCompressExpand) {
    ArrayXr x = ArrayXr::LinSpaced(101, -1.0, 1.0);
    Real mu = 255;

    ArrayXr compressed = mu_compress(x, mu, true);
    ArrayXr expanded = mu_expand(compressed, mu, true);

    // Quantized version has more error but should still be close
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(expanded(i), x(i), 0.05);
    }
}

TEST(MuLawTest, CompressBounds) {
    ArrayXr x = ArrayXr::LinSpaced(101, -1.0, 1.0);
    Real mu = 255;

    ArrayXr compressed = mu_compress(x, mu, false);

    // Compressed values should be bounded
    EXPECT_LE(compressed.maxCoeff(), 1.0 + 1e-10);
    EXPECT_GE(compressed.minCoeff(), -1.0 - 1e-10);
}

// ============================================================================
// Duration Tests
// ============================================================================

TEST(DurationTest, FromSamples) {
    ArrayXr y = ArrayXr::Random(22050);
    Real sr = 22050;

    Real duration = get_duration(y, sr);

    EXPECT_NEAR(duration, 1.0, 1e-5);
}

TEST(DurationTest, FromSpectrogram) {
    // 129 freq bins, 100 frames
    ArrayXXr S = ArrayXXr::Random(129, 100);
    Real sr = 22050;
    int hop_length = 512;
    int n_fft = 256;

    Real duration = get_duration(S, sr, hop_length, n_fft, true);

    // Expected duration based on number of frames
    Real expected = static_cast<Real>(S.cols() * hop_length) / sr;
    EXPECT_NEAR(duration, expected, 0.1);
}

// ============================================================================
// Audio Validation Tests
// Note: These tests require audio files which may not be present
// ============================================================================

// Test that we can handle missing files gracefully
TEST(AudioLoadTest, MissingFileThrows) {
    EXPECT_THROW(load("/nonexistent/path/to/audio.wav"), std::runtime_error);
}

// Test audio parameter validation
TEST(AudioLoadTest, InvalidOffset) {
    TempAudioFile file(22050, 1, 128);
    EXPECT_THROW(load(file.path(), 22050, true, -1.0), ParameterError);
}

TEST(AudioLoadTest, InvalidDuration) {
    TempAudioFile file(22050, 1, 128);
    EXPECT_THROW(load(file.path(), 22050, true, 0.0, -1.0), ParameterError);
}

TEST(AudioLoadTest, OffsetPastEndThrows) {
    TempAudioFile file(1000, 1, 100);
    EXPECT_THROW(load(file.path(), std::nullopt, true, 0.101), ParameterError);
}

TEST(AudioInfoTest, ReportsNativeMetadata) {
    TempAudioFile file(48000, 2, 9600);

    AudioFileInfo info = get_audio_info(file.path());

    EXPECT_EQ(info.samples, 9600);
    EXPECT_EQ(info.channels, 2);
    EXPECT_EQ(info.sample_rate, 48000);
    EXPECT_NEAR(info.duration, 0.2, 1e-12);

    AudioData transformed = load(file.path(), 22050, true);
    EXPECT_EQ(transformed.num_channels(), 1);
    EXPECT_NE(transformed.num_samples(), info.samples);
}
