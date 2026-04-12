#include <gtest/gtest.h>
#include "../cli/common.hpp"
#include "../cli/formatter.hpp"
#include <librosa/util/exceptions.hpp>
#ifdef LIBROSA_HAS_AUDIOTOOLBOX
#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>
#else
#include <sndfile.h>
#endif
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using namespace cli;

namespace {

class TempAudioFile {
public:
    TempAudioFile(int sample_rate, int channels, int64_t frames) {
        char path_template[] = "/tmp/librosa-cli-XXXXXX.wav";
        int fd = mkstemps(path_template, 4);
        if (fd == -1) {
            throw std::runtime_error("Failed to create temporary audio path");
        }

        ::close(fd);
        path_ = path_template;

        std::vector<double> buffer(static_cast<size_t>(frames) * channels, 0.5);

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

} // namespace

TEST(OutputFormatterTest, JsonEscapesStringFields) {
    OutputFormatter formatter("json", 6, false, "info\"cmd", "C:\\tmp\\\"a\"\n.wav");

    testing::internal::CaptureStdout();
    formatter.key_value({
        {"filename", "C:\\tmp\\\"a\"\n.wav"},
        {"note", "123abc"},
        {"channels", "2"}
    });
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("\"file\": \"C:\\\\tmp\\\\\\\"a\\\"\\n.wav\""), std::string::npos);
    EXPECT_NE(output.find("\"command\": \"info\\\"cmd\""), std::string::npos);
    EXPECT_NE(output.find("\"filename\": \"C:\\\\tmp\\\\\\\"a\\\"\\n.wav\""), std::string::npos);
    EXPECT_NE(output.find("\"note\": \"123abc\""), std::string::npos);
    EXPECT_NE(output.find("\"channels\": 2"), std::string::npos);
}

TEST(OutputFormatterTest, JsonUsesNullForNonFiniteScalars) {
    OutputFormatter formatter("json", 6, false, "tempo", "file.wav");

    testing::internal::CaptureStdout();
    formatter.scalar("tempo", std::numeric_limits<double>::quiet_NaN());
    std::string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("\"tempo\": null"), std::string::npos);
}

TEST(CommonOptionsTest, RejectsNegativeDurationExceptSentinel) {
    TempAudioFile file(22050, 1, 128);

    CommonOptions opts;
    opts.input_file = file.path();
    opts.duration = -0.5;

    EXPECT_THROW(load_audio(opts), librosa::ParameterError);
}
