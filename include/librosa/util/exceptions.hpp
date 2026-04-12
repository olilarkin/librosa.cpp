#pragma once

#include <stdexcept>
#include <string>

namespace librosa {

/// The root librosa exception class
class LibrosaError : public std::runtime_error {
public:
    explicit LibrosaError(const std::string& message)
        : std::runtime_error(message) {}
};

/// Exception class for mal-formed inputs
class ParameterError : public LibrosaError {
public:
    explicit ParameterError(const std::string& message)
        : LibrosaError(message) {}
};

} // namespace librosa
