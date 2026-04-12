#pragma once

#include "../types.hpp"
#include <map>
#include <string>
#include <vector>

namespace librosa {

// ============================================================================
// Notation — Indian Music Systems
// ============================================================================

/// Construct svara indices (degrees) for a given thaat
/// @param thaat Name of the thaat (e.g., "bilaval", "todi")
/// @return Array of 7 svara indices (starting from 0=Sa)
std::vector<int> thaat_to_degrees(const std::string& thaat);

/// Construct svara indices (degrees) for a given melakarta raga
/// @param mela Numerical index [1, 72] of the melakarta raga
/// @return Array of 7 svara indices (starting from 0=Sa)
std::vector<int> mela_to_degrees(int mela);

/// Construct svara indices by raga name
std::vector<int> mela_to_degrees(const std::string& mela);

/// Spell Carnatic svara names for a given melakarta raga
/// @param mela Numerical index [1, 72]
/// @param abbr If true, use single-letter names (S, R, G, ...)
/// @param unicode If true, use Unicode subscripts
/// @return 12 svara names for each pitch class
std::vector<std::string> mela_to_svara(int mela, bool abbr = true, bool unicode = true);

/// Spell Carnatic svara names by raga name
std::vector<std::string> mela_to_svara(const std::string& mela, bool abbr = true, bool unicode = true);

/// List melakarta ragas by name and index
/// @return Map of raga name -> index (1-72)
std::map<std::string, int> list_mela();

/// List supported thaats by name
/// @return List of thaat names
std::vector<std::string> list_thaat();

// ============================================================================
// Notation — Western Music Theory
// ============================================================================

/// Construct the diatonic scale degrees for a given key
/// @param key Key string (e.g., "C:maj", "A:min", "D:dor")
/// @return Array of 7 semitone numbers (0=C, 1=C#, ... 11=B)
std::vector<int> key_to_degrees(const std::string& key);

/// List all 12 note names in chromatic scale as spelled by a given key
/// @param key Key string (e.g., "C:maj", "A:min")
/// @param unicode If true, use Unicode accidentals
/// @return 12 note names starting from C
std::vector<std::string> key_to_notes(const std::string& key, bool unicode = true);

/// Calculate note name for a given number of perfect fifths from unison
/// @param unison Starting note name (e.g., "C", "Bb")
/// @param fifths Number of perfect fifths to deviate
/// @param unicode If true, use Unicode accidentals
/// @return Note name
std::string fifths_to_note(const std::string& unison, int fifths, bool unicode = true);

// ============================================================================
// Notation — Functional Just System (FJS)
// ============================================================================

/// Convert a just intonation interval to Functional Just System notation
/// @param interval A positive interval ratio (e.g., 3.0/2.0 for perfect fifth)
/// @param unison Name of the unison note (default "C")
/// @param tolerance Tolerance for Pythagorean approximation (default 65/63)
/// @param use_unicode Use Unicode accidentals and super/subscripts
/// @return Note name in FJS notation
std::string interval_to_fjs(Real interval, const std::string& unison = "C",
                             Real tolerance = 65.0 / 63.0, bool use_unicode = true);

} // namespace librosa
