#pragma once

#include <librosa/types.hpp>
#include <string>
#include <vector>
#include <utility>
#include <iostream>

namespace cli {

class OutputFormatter {
public:
    OutputFormatter(const std::string& format, int precision, bool no_time,
                    const std::string& command, const std::string& file);

    // Scalar output (tempo, tuning)
    void scalar(const std::string& label, double value);

    // Key-value pairs (info, trim)
    void key_value(const std::vector<std::pair<std::string, std::string>>& pairs);

    // 1D vector with optional time axis (onset times, beat times)
    void vector(const librosa::ArrayXr& data, const std::string& label = "value");

    // Per-frame 1D with time axis (rms, zcr, spectral-centroid)
    void per_frame(const librosa::ArrayXr& times, const librosa::ArrayXXr& data,
                   const std::vector<std::string>& labels);

    // Matrix with time axis (mfcc, melspectrogram, chroma)
    void matrix(const librosa::ArrayXr& times, const librosa::ArrayXXr& data,
                const std::vector<std::string>& col_labels);

    // Pitch-specific: time, f0, voiced_prob with NaN handling
    void pitch_output(const librosa::ArrayXr& times, const librosa::ArrayXr& f0,
                      const librosa::ArrayXr& voiced_prob);

private:
    std::string format_;
    int precision_;
    bool no_time_;
    std::string command_;
    std::string file_;

    void json_header(std::ostream& os, int rows, int cols,
                     const std::vector<std::pair<std::string, std::string>>& params = {});
    void json_footer(std::ostream& os);
    std::string fmt(double v) const;
    std::string json_number(double v) const;
    std::string json_string(const std::string& value) const;
};

} // namespace cli
