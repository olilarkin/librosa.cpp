#include "../common.hpp"
#include "../formatter.hpp"
#include <CLI/CLI.hpp>
#include <librosa/effects.hpp>
#ifdef LIBROSA_HAS_AUDIOTOOLBOX
#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>
#elif defined(LIBROSA_HAS_SNDFILE)
#include <sndfile.h>
#endif
#include <stdexcept>
#include <iostream>
#include <memory>
#include <vector>

namespace cli {

static void write_wav(const std::string& path, const librosa::ArrayXr& y, double sr) {
#ifdef LIBROSA_HAS_AUDIOTOOLBOX
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault,
        reinterpret_cast<const UInt8*>(path.c_str()),
        static_cast<CFIndex>(path.size()),
        false);
    if (!url) {
        throw std::runtime_error("Failed to create output file URL: " + path);
    }

    AudioStreamBasicDescription file_format{};
    file_format.mSampleRate = sr;
    file_format.mFormatID = kAudioFormatLinearPCM;
    file_format.mFormatFlags =
        kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked |
        kAudioFormatFlagsNativeEndian;
    file_format.mBytesPerPacket = sizeof(float);
    file_format.mFramesPerPacket = 1;
    file_format.mBytesPerFrame = sizeof(float);
    file_format.mChannelsPerFrame = 1;
    file_format.mBitsPerChannel = static_cast<UInt32>(sizeof(float) * 8);

    ExtAudioFileRef audio_file = nullptr;
    OSStatus status = ExtAudioFileCreateWithURL(
        url, kAudioFileWAVEType, &file_format, nullptr,
        kAudioFileFlags_EraseFile, &audio_file);
    CFRelease(url);
    if (status != noErr || !audio_file) {
        throw std::runtime_error("Failed to open output file: " + path);
    }

    AudioStreamBasicDescription client_format{};
    client_format.mSampleRate = sr;
    client_format.mFormatID = kAudioFormatLinearPCM;
    client_format.mFormatFlags =
        kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked |
        kAudioFormatFlagsNativeEndian;
    client_format.mBytesPerPacket = sizeof(double);
    client_format.mFramesPerPacket = 1;
    client_format.mBytesPerFrame = sizeof(double);
    client_format.mChannelsPerFrame = 1;
    client_format.mBitsPerChannel = static_cast<UInt32>(sizeof(double) * 8);

    status = ExtAudioFileSetProperty(audio_file,
                                     kExtAudioFileProperty_ClientDataFormat,
                                     sizeof(client_format),
                                     &client_format);
    if (status == noErr) {
        std::vector<double> buffer(y.size());
        Eigen::Map<Eigen::VectorXd>(buffer.data(), y.size()) =
            y.matrix().cast<double>();

        AudioBufferList buffers{};
        buffers.mNumberBuffers = 1;
        buffers.mBuffers[0].mNumberChannels = 1;
        buffers.mBuffers[0].mDataByteSize =
            static_cast<UInt32>(buffer.size() * sizeof(double));
        buffers.mBuffers[0].mData = buffer.data();
        status = ExtAudioFileWrite(audio_file, static_cast<UInt32>(y.size()), &buffers);
    }

    ExtAudioFileDispose(audio_file);

    if (status != noErr) {
        throw std::runtime_error("Failed to write output file: " + path);
    }
#elif defined(LIBROSA_HAS_SNDFILE)
    SF_INFO sfinfo;
    sfinfo.samplerate = static_cast<int>(sr);
    sfinfo.channels = 1;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* sndfile = sf_open(path.c_str(), SFM_WRITE, &sfinfo);
    if (!sndfile) {
        throw std::runtime_error("Failed to open output file: " + path + " - " + sf_strerror(nullptr));
    }

    // Convert to double buffer for sf_writef_double
    std::vector<double> buffer(y.size());
    Eigen::Map<Eigen::VectorXd>(buffer.data(), y.size()) = y.matrix().cast<double>();

    sf_writef_double(sndfile, buffer.data(), y.size());
    sf_close(sndfile);
#else
    (void)path; (void)y; (void)sr;
    throw std::runtime_error("WAV output requires Apple AudioToolbox or libsndfile");
#endif
}

void register_hpss(CLI::App& app, CommonOptions& opts) {
    auto* sub = app.add_subcommand("hpss", "Harmonic/percussive source separation (writes WAV)");

    auto output_path = std::make_shared<std::string>();
    auto component = std::make_shared<std::string>("harmonic");
    sub->add_option("--output,-o", *output_path, "Output WAV file path")->required();
    sub->add_option("--component", *component, "Component: harmonic or percussive")->default_val("harmonic");

    sub->callback([&opts, output_path, component]() {
        auto audio = load_audio(opts);
        auto y = audio.mono();
        auto sr = audio.sample_rate;

        librosa::ArrayXr result;
        if (*component == "percussive") {
            result = librosa::effects::percussive(y);
        } else {
            result = librosa::effects::harmonic(y);
        }

        write_wav(*output_path, result, sr);
        std::cerr << "Wrote " << *component << " component to " << *output_path << "\n";
    });
}

} // namespace cli
