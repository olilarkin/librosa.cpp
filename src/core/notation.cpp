#include <librosa/core/notation.hpp>
#include <librosa/util/exceptions.hpp>
#include <algorithm>
#include <cmath>
#include <regex>

namespace librosa {

// ============================================================================
// Internal Data
// ============================================================================

namespace {

// Thaat definitions (Hindustani music)
const std::map<std::string, std::vector<int>> THAAT_MAP = {
    {"bilaval",  {0, 2, 4, 5, 7, 9, 11}},
    {"khamaj",   {0, 2, 4, 5, 7, 9, 10}},
    {"kafi",     {0, 2, 3, 5, 7, 9, 10}},
    {"asavari",  {0, 2, 3, 5, 7, 8, 10}},
    {"bhairavi", {0, 1, 3, 5, 7, 8, 10}},
    {"kalyan",   {0, 2, 4, 6, 7, 9, 11}},
    {"marva",    {0, 1, 4, 6, 7, 9, 11}},
    {"poorvi",   {0, 1, 4, 6, 7, 8, 11}},
    {"todi",     {0, 1, 3, 6, 7, 8, 11}},
    {"bhairav",  {0, 1, 4, 5, 7, 8, 11}},
};

// Melakarta raga names, indexed from 1
const std::vector<std::string> MELAKARTA_NAMES = {
    "kanakangi", "ratnangi", "ganamurthi", "vanaspathi", "manavathi", "tanarupi",
    "senavathi", "hanumathodi", "dhenuka", "natakapriya", "kokilapriya", "rupavathi",
    "gayakapriya", "vakulabharanam", "mayamalavagaula", "chakravakom", "suryakantham",
    "hatakambari", "jhankaradhwani", "natabhairavi", "keeravani", "kharaharapriya",
    "gaurimanohari", "varunapriya", "mararanjini", "charukesi", "sarasangi",
    "harikambhoji", "dheerasankarabharanam", "naganandini", "yagapriya", "ragavardhini",
    "gangeyabhushani", "vagadheeswari", "sulini", "chalanatta", "salagam", "jalarnavam",
    "jhalavarali", "navaneetham", "pavani", "raghupriya", "gavambodhi", "bhavapriya",
    "subhapanthuvarali", "shadvidhamargini", "suvarnangi", "divyamani", "dhavalambari",
    "namanarayani", "kamavardhini", "ramapriya", "gamanasrama", "viswambhari",
    "syamalangi", "shanmukhapriya", "simhendramadhyamam", "hemavathi", "dharmavathi",
    "neethimathi", "kanthamani", "rishabhapriya", "latangi", "vachaspathi",
    "mechakalyani", "chitrambari", "sucharitra", "jyotisvarupini", "dhatuvardhini",
    "nasikabhushani", "kosalam", "rasikapriya",
};

// Pitch map for note letters
const std::map<char, int> PITCH_MAP = {
    {'C', 0}, {'D', 2}, {'E', 4}, {'F', 5}, {'G', 7}, {'A', 9}, {'B', 11}
};

// Accidental map (single-character)
int acc_value(char c) {
    // Handle ASCII accidentals
    if (c == '#') return 1;
    if (c == 'b' || c == '!') return -1;
    if (c == 'n') return 0;
    // Handle UTF-8 accidentals - these are multi-byte, handled separately
    return 0;
}

// Parse accidental offset from a string
// Supports: # b ! n and UTF-8: ♯ ♭ ♮ 𝄪 𝄫
int parse_accidentals(const std::string& acc) {
    int offset = 0;
    size_t i = 0;
    while (i < acc.size()) {
        unsigned char c = acc[i];
        if (c == '#') { offset += 1; i += 1; }
        else if (c == 'b' || c == '!') { offset -= 1; i += 1; }
        else if (c == 'n') { i += 1; }  // natural = 0
        // UTF-8 multi-byte: ♯ = E2 99 AF, ♭ = E2 99 AD, ♮ = E2 99 AE
        else if (c == 0xE2 && i + 2 < acc.size()) {
            unsigned char c1 = acc[i + 1];
            unsigned char c2 = acc[i + 2];
            if (c1 == 0x99 && c2 == 0xAF) { offset += 1; i += 3; }  // ♯
            else if (c1 == 0x99 && c2 == 0xAD) { offset -= 1; i += 3; }  // ♭
            else if (c1 == 0x99 && c2 == 0xAE) { i += 3; }  // ♮
            else { i += 1; }
        }
        // 𝄪 = F0 9D 84 AA (double sharp), 𝄫 = F0 9D 84 AB (double flat)
        else if (c == 0xF0 && i + 3 < acc.size()) {
            unsigned char c1 = acc[i + 1];
            unsigned char c2 = acc[i + 2];
            unsigned char c3 = acc[i + 3];
            if (c1 == 0x9D && c2 == 0x84 && c3 == 0xAA) { offset += 2; i += 4; }  // 𝄪
            else if (c1 == 0x9D && c2 == 0x84 && c3 == 0xAB) { offset -= 2; i += 4; }  // 𝄫
            else { i += 1; }
        }
        else { i += 1; }
    }
    return offset;
}

// Parse a key string like "C:maj", "A#:min", "D:dor"
// Returns: tonic letter (uppercase), accidental string, scale/mode string
struct KeyParts {
    char tonic;
    std::string accidental;
    std::string scale;   // "maj" or "min" (after mode resolution)
    bool is_mode;
    std::string mode;    // original mode: "ion", "dor", "phr", "lyd", "mix", "aeo", "loc"
};

// Mode offsets from the major key (how many degrees to rotate)
const std::map<std::string, int> MODE_OFFSETS = {
    {"ion", 0}, {"dor", 1}, {"phr", 2}, {"lyd", 3}, {"mix", 4}, {"aeo", 5}, {"loc", 6}
};

// Mode-to-major mapping: for each mode, which major key does each tonic map to
// MAJOR_DICT[mode][tonic_letter] = major key tonic name
const std::map<std::string, std::map<char, std::string>> MAJOR_DICT = {
    {"ion", {{'C',"C"}, {'D',"D"}, {'E',"E"}, {'F',"F"}, {'G',"G"}, {'A',"A"}, {'B',"B"}}},
    {"dor", {{'C',std::string("B") + "\xe2\x99\xad"}, {'D',"C"}, {'E',"D"}, {'F',std::string("E") + "\xe2\x99\xad"}, {'G',"F"}, {'A',"G"}, {'B',"A"}}},
    {"phr", {{'C',std::string("A") + "\xe2\x99\xad"}, {'D',std::string("B") + "\xe2\x99\xad"}, {'E',"C"}, {'F',std::string("D") + "\xe2\x99\xad"}, {'G',std::string("E") + "\xe2\x99\xad"}, {'A',"F"}, {'B',"G"}}},
    {"lyd", {{'C',"G"}, {'D',"A"}, {'E',"B"}, {'F',"C"}, {'G',"D"}, {'A',"E"}, {'B',std::string("F") + "\xe2\x99\xaf"}}},
    {"mix", {{'C',"F"}, {'D',"G"}, {'E',"A"}, {'F',std::string("B") + "\xe2\x99\xad"}, {'G',"C"}, {'A',"D"}, {'B',"E"}}},
    {"aeo", {{'C',std::string("E") + "\xe2\x99\xad"}, {'D',"F"}, {'E',"G"}, {'F',std::string("A") + "\xe2\x99\xad"}, {'G',std::string("B") + "\xe2\x99\xad"}, {'A',"C"}, {'B',"D"}}},
    {"loc", {{'C',std::string("D") + "\xe2\x99\xad"}, {'D',std::string("E") + "\xe2\x99\xad"}, {'E',"F"}, {'F',std::string("G") + "\xe2\x99\xad"}, {'G',std::string("A") + "\xe2\x99\xad"}, {'A',std::string("B") + "\xe2\x99\xad"}, {'B',"C"}}},
};

// Normalize a mode abbreviation to 3-letter form
std::string normalize_mode(const std::string& mode) {
    std::string m = mode;
    std::transform(m.begin(), m.end(), m.begin(), ::tolower);
    if (m.substr(0, 3) == "ion" || m == "ionian") return "ion";
    if (m.substr(0, 3) == "dor" || m == "dorian") return "dor";
    if (m.substr(0, 3) == "phr" || m == "phrygian" || m == "phryg" || m == "phrygian") return "phr";
    if (m.substr(0, 3) == "lyd" || m == "lydian") return "lyd";
    if (m.substr(0, 3) == "mix" || m == "mixolydian" || m == "mixolyd") return "mix";
    if (m.substr(0, 3) == "aeo" || m == "aeolian") return "aeo";
    if (m.substr(0, 3) == "loc" || m == "locrian") return "loc";
    return m.substr(0, 3);
}

// Parse a key string and resolve mode to its equivalent major key
KeyParts parse_key(const std::string& key) {
    // regex: TONIC + ACCIDENTALS + ":" + (SCALE or MODE)
    std::regex key_re(R"(^([A-Ga-g])([#b!\xe2\xf0]*(?:[\x80-\xbf]*)*):(maj(?:or)?|min(?:or)?|ion(?:ian)?|dor(?:ian)?|phr(?:yg(?:ian)?)?|lyd(?:ian)?|mix(?:olyd(?:ian)?)?|aeo(?:l(?:ian)?)?|loc(?:r(?:ian)?)?)$)");

    // Since std::regex with UTF-8 is tricky, parse manually
    auto colon_pos = key.find(':');
    if (colon_pos == std::string::npos || colon_pos < 1) {
        throw ParameterError("Improper key format: " + key);
    }

    char tonic = std::toupper(key[0]);
    if (tonic < 'A' || tonic > 'G') {
        throw ParameterError("Improper key format: " + key);
    }

    std::string accidental = key.substr(1, colon_pos - 1);
    std::string scale_or_mode = key.substr(colon_pos + 1);

    // Normalize
    std::string lower_scale = scale_or_mode;
    std::transform(lower_scale.begin(), lower_scale.end(), lower_scale.begin(), ::tolower);

    KeyParts parts;
    parts.tonic = tonic;
    parts.accidental = accidental;

    // Check if it's a scale
    if (lower_scale.substr(0, 3) == "maj") {
        parts.scale = "maj";
        parts.is_mode = false;
        parts.mode = "";
    } else if (lower_scale.substr(0, 3) == "min") {
        parts.scale = "min";
        parts.is_mode = false;
        parts.mode = "";
    } else {
        // It's a mode
        parts.mode = normalize_mode(lower_scale);
        parts.is_mode = true;
        parts.scale = "maj";  // will be resolved
    }

    return parts;
}

// Build accidental string from offset
std::string build_accidental(int offset, bool unicode) {
    std::string result;
    if (offset > 0) {
        if (unicode) {
            int doubles = offset / 2;
            int singles = offset % 2;
            for (int i = 0; i < doubles; ++i) result += "\xf0\x9d\x84\xaa";  // 𝄪
            for (int i = 0; i < singles; ++i) result += "\xe2\x99\xaf";  // ♯
        } else {
            for (int i = 0; i < offset; ++i) result += "#";
        }
    } else if (offset < 0) {
        int abs_off = -offset;
        if (unicode) {
            int doubles = abs_off / 2;
            int singles = abs_off % 2;
            for (int i = 0; i < doubles; ++i) result += "\xf0\x9d\x84\xab";  // 𝄫
            for (int i = 0; i < singles; ++i) result += "\xe2\x99\xad";  // ♭
        } else {
            for (int i = 0; i < abs_off; ++i) result += "b";
        }
    }
    return result;
}

// Circle of fifths note order starting at F
const char COFMAP[] = "FCGDAEB";

} // anonymous namespace

// ============================================================================
// Indian Music — Thaat
// ============================================================================

std::vector<int> thaat_to_degrees(const std::string& thaat) {
    std::string lower = thaat;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    auto it = THAAT_MAP.find(lower);
    if (it == THAAT_MAP.end()) {
        throw ParameterError("Unknown thaat: " + thaat);
    }
    return it->second;
}

std::vector<std::string> list_thaat() {
    std::vector<std::string> result;
    for (const auto& [name, _] : THAAT_MAP) {
        result.push_back(name);
    }
    return result;
}

// ============================================================================
// Indian Music — Melakarta
// ============================================================================

std::map<std::string, int> list_mela() {
    std::map<std::string, int> result;
    for (size_t i = 0; i < MELAKARTA_NAMES.size(); ++i) {
        result[MELAKARTA_NAMES[i]] = static_cast<int>(i) + 1;
    }
    return result;
}

static int resolve_mela_index(int mela) {
    if (mela < 1 || mela > 72) {
        throw ParameterError("mela must be in range [1, 72]");
    }
    return mela - 1;
}

static int resolve_mela_index(const std::string& mela) {
    std::string lower = mela;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    for (size_t i = 0; i < MELAKARTA_NAMES.size(); ++i) {
        if (MELAKARTA_NAMES[i] == lower) {
            return static_cast<int>(i);
        }
    }
    throw ParameterError("Unknown melakarta raga: " + mela);
}

std::vector<int> mela_to_degrees(int mela) {
    int index = resolve_mela_index(mela);

    std::vector<int> degrees = {0};

    // Fill in Ri and Ga
    int lower = index % 36;
    if (lower < 6)       degrees.insert(degrees.end(), {1, 2});
    else if (lower < 12) degrees.insert(degrees.end(), {1, 3});
    else if (lower < 18) degrees.insert(degrees.end(), {1, 4});
    else if (lower < 24) degrees.insert(degrees.end(), {2, 3});
    else if (lower < 30) degrees.insert(degrees.end(), {2, 4});
    else                 degrees.insert(degrees.end(), {3, 4});

    // Ma
    degrees.push_back(index < 36 ? 5 : 6);

    // Pa
    degrees.push_back(7);

    // Dha and Ni
    int upper = index % 6;
    if (upper == 0)      degrees.insert(degrees.end(), {8, 9});
    else if (upper == 1) degrees.insert(degrees.end(), {8, 10});
    else if (upper == 2) degrees.insert(degrees.end(), {8, 11});
    else if (upper == 3) degrees.insert(degrees.end(), {9, 10});
    else if (upper == 4) degrees.insert(degrees.end(), {9, 11});
    else                 degrees.insert(degrees.end(), {10, 11});

    return degrees;
}

std::vector<int> mela_to_degrees(const std::string& mela) {
    int index = resolve_mela_index(mela);
    return mela_to_degrees(index + 1);
}

std::vector<std::string> mela_to_svara(int mela, bool abbr, bool unicode) {
    int mela_idx = resolve_mela_index(mela);

    // Unicode subscripts
    const std::string sub1 = unicode ? "\xe2\x82\x81" : "1";  // ₁
    const std::string sub2 = unicode ? "\xe2\x82\x82" : "2";  // ₂
    const std::string sub3 = unicode ? "\xe2\x82\x83" : "3";  // ₃

    std::vector<std::string> svara_map(12);
    svara_map[0]  = abbr ? "S"  : "Sa";
    svara_map[1]  = (abbr ? "R" : "Ri") + sub1;
    // [2] and [3] are ambiguous — resolved below
    svara_map[4]  = (abbr ? "G" : "Ga") + sub3;
    svara_map[5]  = (abbr ? "M" : "Ma") + sub1;
    svara_map[6]  = (abbr ? "M" : "Ma") + sub2;
    svara_map[7]  = abbr ? "P"  : "Pa";
    svara_map[8]  = (abbr ? "D" : "Dha") + sub1;
    // [9] and [10] are ambiguous — resolved below
    svara_map[11] = (abbr ? "N" : "Ni") + sub3;

    // Determine Ri2/Ga1 at position 2
    int lower = mela_idx % 36;
    if (lower < 6) {
        svara_map[2] = (abbr ? "G" : "Ga") + sub1;
    } else {
        svara_map[2] = (abbr ? "R" : "Ri") + sub2;
    }

    // Determine Ri3/Ga2 at position 3
    if (lower < 30) {
        svara_map[3] = (abbr ? "G" : "Ga") + sub2;
    } else {
        svara_map[3] = (abbr ? "R" : "Ri") + sub3;
    }

    // Determine Dha2/Ni1 at position 9
    int upper = mela_idx % 6;
    if (upper == 0) {
        svara_map[9] = (abbr ? "N" : "Ni") + sub1;
    } else {
        svara_map[9] = (abbr ? "D" : "Dha") + sub2;
    }

    // Determine Dha3/Ni2 at position 10
    if (upper == 5) {
        svara_map[10] = (abbr ? "D" : "Dha") + sub3;
    } else {
        svara_map[10] = (abbr ? "N" : "Ni") + sub2;
    }

    return svara_map;
}

std::vector<std::string> mela_to_svara(const std::string& mela, bool abbr, bool unicode) {
    int index = resolve_mela_index(mela);
    return mela_to_svara(index + 1, abbr, unicode);
}

// ============================================================================
// Western Music — Key/Scale Functions
// ============================================================================

std::vector<int> key_to_degrees(const std::string& key) {
    KeyParts parts = parse_key(key);

    // Major and minor scale patterns
    const std::vector<int> major_degrees = {0, 2, 4, 5, 7, 9, 11};
    const std::vector<int> minor_degrees = {0, 2, 3, 5, 7, 8, 10};

    if (parts.is_mode) {
        // Resolve mode to equivalent major key, then rotate
        auto it = MAJOR_DICT.find(parts.mode);
        if (it == MAJOR_DICT.end()) {
            throw ParameterError("Unknown mode: " + parts.mode);
        }

        auto tonic_it = it->second.find(parts.tonic);
        if (tonic_it == it->second.end()) {
            throw ParameterError("Unknown tonic for mode: " + std::string(1, parts.tonic));
        }

        // Get the equivalent major key tonic and add any accidentals
        std::string equiv_tonic = tonic_it->second;

        // Parse the accidental offset from our original tonic
        int orig_offset = parse_accidentals(parts.accidental);

        // Build equiv key string (tonic name may have accidentals already)
        // The accidental from the original key is added to the major tonic
        std::string equiv_key = equiv_tonic;
        if (orig_offset > 0) {
            for (int i = 0; i < orig_offset; ++i) equiv_key += "#";
        } else if (orig_offset < 0) {
            for (int i = 0; i < -orig_offset; ++i) equiv_key += "b";
        }
        equiv_key += ":maj";

        // Get the major degrees, then rotate by mode offset
        auto degrees = key_to_degrees(equiv_key);
        int mode_offset = MODE_OFFSETS.at(parts.mode);
        std::rotate(degrees.begin(), degrees.begin() + mode_offset, degrees.end());
        return degrees;
    }

    // Simple major/minor key
    int tonic_semitone = PITCH_MAP.at(parts.tonic);
    int offset = parse_accidentals(parts.accidental);

    const auto& pattern = (parts.scale == "maj") ? major_degrees : minor_degrees;

    std::vector<int> degrees(7);
    for (int i = 0; i < 7; ++i) {
        degrees[i] = (pattern[i] + tonic_semitone + offset) % 12;
    }

    return degrees;
}

std::vector<std::string> key_to_notes(const std::string& key, bool unicode) {
    KeyParts parts = parse_key(key);

    if (parts.is_mode) {
        // Resolve mode to equivalent major key
        auto it = MAJOR_DICT.find(parts.mode);
        if (it == MAJOR_DICT.end()) {
            throw ParameterError("Unknown mode: " + parts.mode);
        }
        auto tonic_it = it->second.find(parts.tonic);
        if (tonic_it == it->second.end()) {
            throw ParameterError("Unknown tonic for mode");
        }
        std::string equiv_tonic = tonic_it->second;
        int orig_offset = parse_accidentals(parts.accidental);
        std::string equiv_key = equiv_tonic;
        if (orig_offset > 0) {
            for (int i = 0; i < orig_offset; ++i) equiv_key += "#";
        } else if (orig_offset < 0) {
            for (int i = 0; i < -orig_offset; ++i) equiv_key += "b";
        }
        equiv_key += ":maj";
        return key_to_notes(equiv_key, unicode);
    }

    bool major = (parts.scale == "maj");
    int offset = parse_accidentals(parts.accidental);

    // Calculate how many clockwise steps on circle of fifths (= # sharps)
    int tonic_number;
    if (major) {
        tonic_number = ((PITCH_MAP.at(parts.tonic) + offset) * 7) % 12;
    } else {
        tonic_number = ((PITCH_MAP.at(parts.tonic) + offset) * 7 + 9) % 12;
    }
    // Handle negative modulo
    tonic_number = ((tonic_number % 12) + 12) % 12;

    // Decide sharps vs flats
    bool use_sharps;
    if (offset < 0) {
        use_sharps = false;
    } else if (offset > 0) {
        use_sharps = true;
    } else if (tonic_number < 6) {
        use_sharps = true;
    } else if (tonic_number > 6) {
        use_sharps = false;
    } else {
        use_sharps = major;  // tie-break
    }

    // Accidental symbols
    std::string sharp = unicode ? "\xe2\x99\xaf" : "#";
    std::string flat = unicode ? "\xe2\x99\xad" : "b";
    std::string dbl_sharp = unicode ? "\xf0\x9d\x84\xaa" : "##";
    std::string dbl_flat = unicode ? "\xf0\x9d\x84\xab" : "bb";

    std::vector<std::string> notes_sharp = {"C", "C" + sharp, "D", "D" + sharp, "E", "F", "F" + sharp, "G", "G" + sharp, "A", "A" + sharp, "B"};
    std::vector<std::string> notes_flat = {"C", "D" + flat, "D", "E" + flat, "E", "F", "G" + flat, "G", "A" + flat, "A", "B" + flat, "B"};

    // Sharp corrections for keys with >= 6 sharps
    struct Correction { int index; std::string name; };
    std::vector<Correction> sharp_corrections = {
        {5, "E" + sharp}, {0, "B" + sharp}, {7, "F" + dbl_sharp},
        {2, "C" + dbl_sharp}, {9, "G" + dbl_sharp}, {4, "D" + dbl_sharp}, {11, "A" + dbl_sharp}
    };
    std::vector<Correction> flat_corrections = {
        {11, "C" + flat}, {4, "F" + flat}, {9, "B" + dbl_flat},
        {2, "E" + dbl_flat}, {7, "A" + dbl_flat}, {0, "D" + dbl_flat}
    };

    // Apply mod-12 correction for B#:maj
    int n_sharps = tonic_number;
    if (tonic_number == 0 && parts.tonic == 'B') {
        n_sharps = 12;
    }

    std::vector<std::string> notes;
    if (use_sharps) {
        for (int n = 0; n <= n_sharps - 6; ++n) {
            if (n < static_cast<int>(sharp_corrections.size())) {
                notes_sharp[sharp_corrections[n].index] = sharp_corrections[n].name;
            }
        }
        notes = notes_sharp;
    } else {
        int n_flats = (12 - tonic_number) % 12;
        for (int n = 0; n <= n_flats - 6; ++n) {
            if (n < static_cast<int>(flat_corrections.size())) {
                notes_flat[flat_corrections[n].index] = flat_corrections[n].name;
            }
        }
        notes = notes_flat;
    }

    return notes;
}

std::string fifths_to_note(const std::string& unison, int fifths, bool unicode) {
    // Parse unison note
    if (unison.empty()) {
        throw ParameterError("Improper note format: empty string");
    }

    char pitch = std::toupper(unison[0]);
    if (pitch < 'A' || pitch > 'G') {
        throw ParameterError("Improper note format: " + unison);
    }

    // Parse accidental offset
    int offset = 0;
    if (unison.size() > 1) {
        offset = parse_accidentals(unison.substr(1));
    }

    // Find the note in the circle of fifths
    const char* cofmap = COFMAP;
    int circle_idx = -1;
    for (int i = 0; i < 7; ++i) {
        if (cofmap[i] == pitch) {
            circle_idx = i;
            break;
        }
    }
    if (circle_idx < 0) {
        throw ParameterError("Improper note format: " + unison);
    }

    // Raw target note
    int target_idx = ((circle_idx + fifths) % 7 + 7) % 7;
    char raw_output = cofmap[target_idx];

    // Accrue accidentals: each time we cross B-F boundary
    int acc_index = offset + (circle_idx + fifths) / 7;
    // Handle negative fifths correctly
    if ((circle_idx + fifths) < 0 && (circle_idx + fifths) % 7 != 0) {
        acc_index = offset + (circle_idx + fifths) / 7 - 1;
    }

    std::string acc_str = build_accidental(acc_index, unicode);

    return std::string(1, raw_output) + acc_str;
}

// ============================================================================
// FJS Notation
// ============================================================================

namespace {

// Octave-fold: map interval to [1, 2)
double o_fold(double d) {
    return d * std::pow(2.0, -std::floor(std::log2(d)));
}

// Balanced octave-fold: map interval to [sqrt(2)/2, sqrt(2))
double bo_fold(double d) {
    return d * std::pow(2.0, -std::round(std::log2(d)));
}

// Search for the number of fifths to get within tolerance of interval
int fifth_search(double interval, double tolerance) {
    double log_tolerance = std::abs(std::log2(tolerance));
    for (int power = 0; power < 32; ++power) {
        for (int sign : {1, -1}) {
            if (std::abs(std::log2(bo_fold(interval / std::pow(3.0, power * sign)))) <= log_tolerance) {
                return power * sign;
            }
        }
    }
    return 32;
}

// Unicode superscript digits
std::string to_superscript(int n) {
    // Unicode superscript digits: ⁰¹²³⁴⁵⁶⁷⁸⁹
    static const char* super_digits[] = {
        "\xe2\x81\xb0", "\xc2\xb9", "\xc2\xb2", "\xc2\xb3",
        "\xe2\x81\xb4", "\xe2\x81\xb5", "\xe2\x81\xb6",
        "\xe2\x81\xb7", "\xe2\x81\xb8", "\xe2\x81\xb9"
    };
    std::string s = std::to_string(n);
    std::string result;
    for (char c : s) {
        result += super_digits[c - '0'];
    }
    return result;
}

// Unicode subscript digits
std::string to_subscript(int n) {
    // Unicode subscript digits: ₀₁₂₃₄₅₆₇₈₉
    static const char* sub_digits[] = {
        "\xe2\x82\x80", "\xe2\x82\x81", "\xe2\x82\x82", "\xe2\x82\x83",
        "\xe2\x82\x84", "\xe2\x82\x85", "\xe2\x82\x86",
        "\xe2\x82\x87", "\xe2\x82\x88", "\xe2\x82\x89"
    };
    std::string s = std::to_string(n);
    std::string result;
    for (char c : s) {
        result += sub_digits[c - '0'];
    }
    return result;
}

// Interval table: maps octave-folded interval (rounded to 6 decimals)
// to prime factorization {prime -> exponent}
// Auto-generated from librosa/core/intervals.msgpack (333 entries)
struct IntervalEntry {
    double key;
    std::map<int,int> powers;
};

const IntervalEntry INTERVAL_TABLE[] = {
    {1.000000, {}},
    {1.002090, {{2, -84}, {3, 53}}},
    {1.003922, {{2, 8}, {3, -1}, {5, -1}, {17, -1}}},
    {1.012500, {{2, -4}, {3, 4}, {5, -1}}},
    {1.013643, {{2, -19}, {3, 12}}},
    {1.015625, {{2, -6}, {5, 1}, {13, 1}}},
    {1.015762, {{2, -103}, {3, 65}}},
    {1.015873, {{2, 6}, {3, -2}, {7, -1}}},
    {1.020833, {{2, -4}, {3, -1}, {7, 2}}},
    {1.024000, {{2, 7}, {5, -3}}},
    {1.025329, {{2, 46}, {3, -29}}},
    {1.025391, {{2, -9}, {3, 1}, {5, 2}, {7, 1}}},
    {1.025641, {{2, 3}, {3, -1}, {5, 1}, {13, -1}}},
    {1.027473, {{2, -38}, {3, 24}}},
    {1.028571, {{2, 2}, {3, 2}, {5, -1}, {7, -1}}},
    {1.031250, {{2, -5}, {3, 1}, {11, 1}}},
    {1.037037, {{2, 2}, {3, -3}, {7, 1}}},
    {1.039318, {{2, 27}, {3, -17}}},
    {1.041491, {{2, -57}, {3, 36}}},
    {1.041667, {{2, -3}, {3, -1}, {5, 2}}},
    {1.043478, {{2, 3}, {3, 1}, {23, -1}}},
    {1.047619, {{2, 1}, {3, -1}, {7, -1}, {11, 1}}},
    {1.050000, {{2, -2}, {3, 1}, {5, -1}, {7, 1}}},
    {1.052632, {{2, 2}, {5, 1}, {19, -1}}},
    {1.053498, {{2, 8}, {3, -5}}},
    {1.054687, {{2, -7}, {3, 3}, {5, 1}}},
    {1.054945, {{2, 5}, {3, 1}, {7, -1}, {13, -1}}},
    {1.055700, {{2, -76}, {3, 48}}},
    {1.058824, {{2, 1}, {3, 2}, {17, -1}}},
    {1.062500, {{2, -4}, {17, 1}}},
    {1.066406, {{2, -8}, {3, 1}, {7, 1}, {13, 1}}},
    {1.066667, {{2, 4}, {3, -1}, {5, -1}}},
    {1.067871, {{2, -11}, {3, 7}}},
    {1.070103, {{2, -95}, {3, 60}}},
    {1.071429, {{2, -1}, {3, 1}, {5, 1}, {7, -1}}},
    {1.075630, {{2, 7}, {7, -1}, {17, -1}}},
    {1.076923, {{2, 1}, {7, 1}, {13, -1}}},
    {1.078125, {{2, -6}, {3, 1}, {23, 1}}},
    {1.080000, {{3, 3}, {5, -2}}},
    {1.082440, {{2, -30}, {3, 19}}},
    {1.083333, {{2, -2}, {3, -1}, {13, 1}}},
    {1.090909, {{2, 2}, {3, 1}, {11, -1}}},
    {1.093750, {{2, -5}, {5, 1}, {7, 1}}},
    {1.094017, {{2, 7}, {3, -2}, {13, -1}}},
    {1.094920, {{2, 35}, {3, -22}}},
    {1.097208, {{2, -49}, {3, 31}}},
    {1.098633, {{2, -10}, {3, 2}, {5, 3}}},
    {1.100000, {{2, -1}, {5, -1}, {11, 1}}},
    {1.108225, {{2, 8}, {3, -1}, {7, -1}, {11, -1}}},
    {1.109858, {{2, 16}, {3, -10}}},
    {1.111111, {{2, 1}, {3, -2}, {5, 1}}},
    {1.112178, {{2, -68}, {3, 43}}},
    {1.112366, {{2, -14}, {3, 6}, {5, 2}}},
    {1.120000, {{2, 2}, {5, -2}, {7, 1}}},
    {1.122807, {{2, 6}, {3, -1}, {19, -1}}},
    {1.125000, {{2, -3}, {3, 2}}},
    {1.127352, {{2, -87}, {3, 55}}},
    {1.129412, {{2, 5}, {3, 1}, {5, -1}, {17, -1}}},
    {1.133333, {{3, -1}, {5, -1}, {17, 1}}},
    {1.137778, {{2, 8}, {3, -2}, {5, -2}}},
    {1.139062, {{2, -7}, {3, 6}, {5, -1}}},
    {1.140349, {{2, -22}, {3, 14}}},
    {1.142732, {{2, -106}, {3, 67}}},
    {1.142857, {{2, 3}, {7, -1}}},
    {1.145833, {{2, -4}, {3, -1}, {5, 1}, {11, 1}}},
    {1.148438, {{2, -7}, {3, 1}, {7, 2}}},
    {1.152000, {{2, 4}, {3, 2}, {5, -3}}},
    {1.153496, {{2, 43}, {3, -27}}},
    {1.153846, {{3, 1}, {5, 1}, {13, -1}}},
    {1.155907, {{2, -41}, {3, 26}}},
    {1.157407, {{2, -2}, {3, -3}, {5, 3}}},
    {1.163636, {{2, 6}, {5, -1}, {11, -1}}},
    {1.166667, {{2, -1}, {3, -1}, {7, 1}}},
    {1.169233, {{2, 24}, {3, -15}}},
    {1.171677, {{2, -60}, {3, 38}}},
    {1.171875, {{2, -6}, {3, 1}, {5, 2}}},
    {1.176471, {{2, 2}, {5, 1}, {17, -1}}},
    {1.178571, {{2, -2}, {3, 1}, {7, -1}, {11, 1}}},
    {1.185185, {{2, 5}, {3, -3}}},
    {1.186523, {{2, -10}, {3, 5}, {5, 1}}},
    {1.187500, {{2, -4}, {19, 1}}},
    {1.187663, {{2, -79}, {3, 50}}},
    {1.190476, {{3, -1}, {5, 2}, {7, -1}}},
    {1.195312, {{2, -7}, {3, 2}, {17, 1}}},
    {1.200000, {{2, 1}, {3, 1}, {5, -1}}},
    {1.201355, {{2, -14}, {3, 9}}},
    {1.203125, {{2, -6}, {7, 1}, {11, 1}}},
    {1.203866, {{2, -98}, {3, 62}}},
    {1.212121, {{2, 3}, {3, -1}, {5, 1}, {11, -1}}},
    {1.214286, {{2, -1}, {7, -1}, {17, 1}}},
    {1.215000, {{2, -3}, {3, 5}, {5, -2}}},
    {1.215205, {{2, 51}, {3, -32}}},
    {1.217745, {{2, -33}, {3, 21}}},
    {1.218750, {{2, -5}, {3, 1}, {13, 1}}},
    {1.219048, {{2, 7}, {3, -1}, {5, -1}, {7, -1}}},
    {1.220703, {{2, -9}, {5, 4}}},
    {1.222222, {{3, -2}, {11, 1}}},
    {1.225000, {{2, -3}, {5, -1}, {7, 2}}},
    {1.230469, {{2, -8}, {3, 2}, {5, 1}, {7, 1}}},
    {1.230769, {{2, 4}, {13, -1}}},
    {1.231785, {{2, 32}, {3, -20}}},
    {1.234359, {{2, -52}, {3, 33}}},
    {1.234568, {{2, 2}, {3, -4}, {5, 2}}},
    {1.235962, {{2, -13}, {3, 4}, {5, 3}}},
    {1.238095, {{2, 1}, {3, -1}, {7, -1}, {13, 1}}},
    {1.244444, {{2, 3}, {3, -2}, {5, -1}, {7, 1}}},
    {1.246753, {{2, 5}, {3, 1}, {7, -1}, {11, -1}}},
    {1.248590, {{2, 13}, {3, -8}}},
    {1.250000, {{2, -2}, {5, 1}}},
    {1.251200, {{2, -71}, {3, 45}}},
    {1.254902, {{2, 6}, {3, -1}, {17, -1}}},
    {1.263158, {{2, 3}, {3, 1}, {19, -1}}},
    {1.264198, {{2, 9}, {3, -4}, {5, -1}}},
    {1.265625, {{2, -6}, {3, 4}}},
    {1.268271, {{2, -90}, {3, 57}}},
    {1.269841, {{2, 4}, {3, -2}, {5, 1}, {7, -1}}},
    {1.272727, {{2, 1}, {7, 1}, {11, -1}}},
    {1.275000, {{2, -3}, {3, 1}, {5, -1}, {17, 1}}},
    {1.276042, {{2, -6}, {3, -1}, {5, 1}, {7, 2}}},
    {1.280000, {{2, 5}, {5, -2}}},
    {1.282892, {{2, -25}, {3, 16}}},
    {1.285574, {{2, -109}, {3, 69}}},
    {1.285714, {{3, 2}, {7, -1}}},
    {1.289062, {{2, -7}, {3, 1}, {5, 1}, {11, 1}}},
    {1.292929, {{2, 7}, {3, -2}, {11, -1}}},
    {1.296000, {{2, 1}, {3, 4}, {5, -3}}},
    {1.297683, {{2, 40}, {3, -25}}},
    {1.300000, {{2, -1}, {5, -1}, {13, 1}}},
    {1.300395, {{2, -44}, {3, 28}}},
    {1.302083, {{2, -5}, {3, -1}, {5, 3}}},
    {1.306122, {{2, 6}, {7, -2}}},
    {1.312500, {{2, -4}, {3, 1}, {7, 1}}},
    {1.312821, {{2, 8}, {3, -1}, {5, -1}, {13, -1}}},
    {1.315387, {{2, 21}, {3, -13}}},
    {1.318137, {{2, -63}, {3, 40}}},
    {1.318359, {{2, -9}, {3, 3}, {5, 2}}},
    {1.328125, {{2, -6}, {5, 1}, {17, 1}}},
    {1.333333, {{2, 2}, {3, -1}}},
    {1.336120, {{2, -82}, {3, 52}}},
    {1.347368, {{2, 7}, {5, -1}, {19, -1}}},
    {1.350000, {{2, -2}, {3, 3}, {5, -1}}},
    {1.351524, {{2, -17}, {3, 11}}},
    {1.354167, {{2, -4}, {3, -1}, {5, 1}, {13, 1}}},
    {1.354349, {{2, -101}, {3, 64}}},
    {1.354497, {{2, 8}, {3, -3}, {7, -1}}},
    {1.361111, {{2, -2}, {3, -2}, {7, 2}}},
    {1.363636, {{3, 1}, {5, 1}, {11, -1}}},
    {1.365333, {{2, 9}, {3, -1}, {5, -3}}},
    {1.367106, {{2, 48}, {3, -30}}},
    {1.367187, {{2, -7}, {5, 2}, {7, 1}}},
    {1.369964, {{2, -36}, {3, 23}}},
    {1.371429, {{2, 4}, {3, 1}, {5, -1}, {7, -1}}},
    {1.373291, {{2, -12}, {3, 2}, {5, 4}}},
    {1.375000, {{2, -3}, {11, 1}}},
    {1.384615, {{2, 1}, {3, 2}, {13, -1}}},
    {1.385758, {{2, 29}, {3, -18}}},
    {1.388654, {{2, -55}, {3, 35}}},
    {1.388889, {{2, -1}, {3, -2}, {5, 2}}},
    {1.391304, {{2, 5}, {23, -1}}},
    {1.392857, {{2, -2}, {3, 1}, {7, -1}, {13, 1}}},
    {1.400000, {{5, -1}, {7, 1}}},
    {1.404664, {{2, 10}, {3, -6}}},
    {1.406250, {{2, -5}, {3, 2}, {5, 1}}},
    {1.406593, {{2, 7}, {7, -1}, {13, -1}}},
    {1.407600, {{2, -74}, {3, 47}}},
    {1.411765, {{2, 3}, {3, 1}, {17, -1}}},
    {1.416667, {{2, -2}, {3, -1}, {17, 1}}},
    {1.421875, {{2, -6}, {7, 1}, {13, 1}}},
    {1.422222, {{2, 6}, {3, -2}, {5, -1}}},
    {1.423828, {{2, -9}, {3, 6}}},
    {1.426804, {{2, -93}, {3, 59}}},
    {1.428571, {{2, 1}, {5, 1}, {7, -1}}},
    {1.435547, {{2, -9}, {3, 1}, {5, 1}, {7, 2}}},
    {1.435897, {{2, 3}, {3, -1}, {7, 1}, {13, -1}}},
    {1.437500, {{2, -4}, {23, 1}}},
    {1.440000, {{2, 2}, {3, 2}, {5, -2}}},
    {1.443254, {{2, -28}, {3, 18}}},
    {1.444444, {{3, -2}, {13, 1}}},
    {1.446271, {{2, -112}, {3, 71}}},
    {1.454545, {{2, 4}, {11, -1}}},
    {1.458333, {{2, -3}, {3, -1}, {5, 1}, {7, 1}}},
    {1.459893, {{2, 37}, {3, -23}}},
    {1.462857, {{2, 8}, {5, -2}, {7, -1}}},
    {1.462944, {{2, -47}, {3, 30}}},
    {1.464844, {{2, -8}, {3, 1}, {5, 3}}},
    {1.466667, {{2, 1}, {3, -1}, {5, -1}, {11, 1}}},
    {1.476562, {{2, -7}, {3, 3}, {7, 1}}},
    {1.476923, {{2, 5}, {3, 1}, {5, -1}, {13, -1}}},
    {1.479811, {{2, 18}, {3, -11}}},
    {1.481481, {{2, 3}, {3, -3}, {5, 1}}},
    {1.482904, {{2, -66}, {3, 42}}},
    {1.483154, {{2, -12}, {3, 5}, {5, 2}}},
    {1.484375, {{2, -6}, {5, 1}, {19, 1}}},
    {1.493333, {{2, 4}, {3, -1}, {5, -2}, {7, 1}}},
    {1.500000, {{2, -1}, {3, 1}}},
    {1.503135, {{2, -85}, {3, 54}}},
    {1.505882, {{2, 7}, {5, -1}, {17, -1}}},
    {1.517037, {{2, 10}, {3, -3}, {5, -2}}},
    {1.518750, {{2, -5}, {3, 5}, {5, -1}}},
    {1.520465, {{2, -20}, {3, 13}}},
    {1.523437, {{2, -7}, {3, 1}, {5, 1}, {13, 1}}},
    {1.523643, {{2, -104}, {3, 66}}},
    {1.523810, {{2, 5}, {3, -1}, {7, -1}}},
    {1.531250, {{2, -5}, {7, 2}}},
    {1.536000, {{2, 6}, {3, 1}, {5, -3}}},
    {1.537994, {{2, 45}, {3, -28}}},
    {1.538462, {{2, 2}, {5, 1}, {13, -1}}},
    {1.541209, {{2, -39}, {3, 25}}},
    {1.546875, {{2, -6}, {3, 2}, {11, 1}}},
    {1.551515, {{2, 8}, {3, -1}, {5, -1}, {11, -1}}},
    {1.555556, {{2, 1}, {3, -2}, {7, 1}}},
    {1.558977, {{2, 26}, {3, -16}}},
    {1.562236, {{2, -58}, {3, 37}}},
    {1.562500, {{2, -4}, {5, 2}}},
    {1.568627, {{2, 4}, {3, -1}, {5, 1}, {17, -1}}},
    {1.571429, {{7, -1}, {11, 1}}},
    {1.575000, {{2, -3}, {3, 2}, {5, -1}, {7, 1}}},
    {1.580247, {{2, 7}, {3, -4}}},
    {1.582031, {{2, -8}, {3, 4}, {5, 1}}},
    {1.583333, {{2, -2}, {3, -1}, {19, 1}}},
    {1.583550, {{2, -77}, {3, 49}}},
    {1.593750, {{2, -5}, {3, 1}, {17, 1}}},
    {1.600000, {{2, 3}, {5, -1}}},
    {1.601807, {{2, -12}, {3, 8}}},
    {1.604167, {{2, -4}, {3, -1}, {7, 1}, {11, 1}}},
    {1.605155, {{2, -96}, {3, 61}}},
    {1.607143, {{2, -2}, {3, 2}, {5, 1}, {7, -1}}},
    {1.615385, {{3, 1}, {7, 1}, {13, -1}}},
    {1.620000, {{2, -1}, {3, 4}, {5, -2}}},
    {1.620274, {{2, 53}, {3, -33}}},
    {1.623661, {{2, -31}, {3, 20}}},
    {1.625000, {{2, -3}, {13, 1}}},
    {1.625397, {{2, 9}, {3, -2}, {5, -1}, {7, -1}}},
    {1.627604, {{2, -7}, {3, -1}, {5, 4}}},
    {1.633333, {{2, -1}, {3, -1}, {5, -1}, {7, 2}}},
    {1.636364, {{2, 1}, {3, 2}, {11, -1}}},
    {1.640625, {{2, -6}, {3, 1}, {5, 1}, {7, 1}}},
    {1.641026, {{2, 6}, {3, -1}, {13, -1}}},
    {1.642379, {{2, 34}, {3, -21}}},
    {1.645813, {{2, -50}, {3, 32}}},
    {1.647059, {{2, 2}, {7, 1}, {17, -1}}},
    {1.647949, {{2, -11}, {3, 3}, {5, 3}}},
    {1.650000, {{2, -2}, {3, 1}, {5, -1}, {11, 1}}},
    {1.662338, {{2, 7}, {7, -1}, {11, -1}}},
    {1.664787, {{2, 15}, {3, -9}}},
    {1.666667, {{3, -1}, {5, 1}}},
    {1.668267, {{2, -69}, {3, 44}}},
    {1.673203, {{2, 8}, {3, -2}, {17, -1}}},
    {1.680000, {{2, 1}, {3, 1}, {5, -2}, {7, 1}}},
    {1.684211, {{2, 5}, {19, -1}}},
    {1.687500, {{2, -4}, {3, 3}}},
    {1.691027, {{2, -88}, {3, 56}}},
    {1.696970, {{2, 3}, {3, -1}, {7, 1}, {11, -1}}},
    {1.700000, {{2, -1}, {5, -1}, {17, 1}}},
    {1.706667, {{2, 7}, {3, -1}, {5, -2}}},
    {1.710523, {{2, -23}, {3, 15}}},
    {1.714099, {{2, -107}, {3, 68}}},
    {1.714286, {{2, 2}, {3, 1}, {7, -1}}},
    {1.718750, {{2, -5}, {5, 1}, {11, 1}}},
    {1.722656, {{2, -8}, {3, 2}, {7, 2}}},
    {1.728000, {{2, 3}, {3, 3}, {5, -3}}},
    {1.730243, {{2, 42}, {3, -26}}},
    {1.733333, {{2, 1}, {3, -1}, {5, -1}, {13, 1}}},
    {1.733860, {{2, -42}, {3, 27}}},
    {1.736111, {{2, -3}, {3, -2}, {5, 3}}},
    {1.741497, {{2, 8}, {3, -1}, {7, -2}}},
    {1.745455, {{2, 5}, {3, 1}, {5, -1}, {11, -1}}},
    {1.750000, {{2, -2}, {7, 1}}},
    {1.753850, {{2, 23}, {3, -14}}},
    {1.757516, {{2, -61}, {3, 39}}},
    {1.757812, {{2, -7}, {3, 2}, {5, 2}}},
    {1.764706, {{2, 1}, {3, 1}, {5, 1}, {17, -1}}},
    {1.770833, {{2, -4}, {3, -1}, {5, 1}, {17, 1}}},
    {1.777778, {{2, 4}, {3, -2}}},
    {1.779785, {{2, -11}, {3, 6}, {5, 1}}},
    {1.781250, {{2, -5}, {3, 1}, {19, 1}}},
    {1.781494, {{2, -80}, {3, 51}}},
    {1.785714, {{2, -1}, {5, 2}, {7, -1}}},
    {1.800000, {{3, 2}, {5, -1}}},
    {1.802032, {{2, -15}, {3, 10}}},
    {1.804688, {{2, -7}, {3, 1}, {7, 1}, {11, 1}}},
    {1.805799, {{2, -99}, {3, 63}}},
    {1.818182, {{2, 2}, {5, 1}, {11, -1}}},
    {1.820444, {{2, 11}, {3, -2}, {5, -3}}},
    {1.822808, {{2, 50}, {3, -31}}},
    {1.822917, {{2, -5}, {3, -1}, {5, 2}, {7, 1}}},
    {1.826618, {{2, -34}, {3, 22}}},
    {1.828125, {{2, -6}, {3, 2}, {13, 1}}},
    {1.828571, {{2, 6}, {5, -1}, {7, -1}}},
    {1.831055, {{2, -10}, {3, 1}, {5, 4}}},
    {1.833333, {{2, -1}, {3, -1}, {11, 1}}},
    {1.837500, {{2, -4}, {3, 1}, {5, -1}, {7, 2}}},
    {1.846154, {{2, 3}, {3, 1}, {13, -1}}},
    {1.847677, {{2, 31}, {3, -19}}},
    {1.851539, {{2, -53}, {3, 34}}},
    {1.851852, {{2, 1}, {3, -3}, {5, 2}}},
    {1.855072, {{2, 7}, {3, -1}, {23, -1}}},
    {1.857143, {{7, -1}, {13, 1}}},
    {1.859375, {{2, -6}, {7, 1}, {17, 1}}},
    {1.866667, {{2, 2}, {3, -1}, {5, -1}, {7, 1}}},
    {1.872885, {{2, 12}, {3, -7}}},
    {1.875000, {{2, -3}, {3, 1}, {5, 1}}},
    {1.875458, {{2, 9}, {3, -1}, {7, -1}, {13, -1}}},
    {1.876800, {{2, -72}, {3, 46}}},
    {1.882353, {{2, 5}, {17, -1}}},
    {1.888889, {{3, -2}, {17, 1}}},
    {1.895833, {{2, -4}, {3, -1}, {7, 1}, {13, 1}}},
    {1.896296, {{2, 8}, {3, -3}, {5, -1}}},
    {1.898437, {{2, -7}, {3, 5}}},
    {1.900000, {{2, -1}, {5, -1}, {19, 1}}},
    {1.902406, {{2, -91}, {3, 58}}},
    {1.904762, {{2, 3}, {3, -1}, {5, 1}, {7, -1}}},
    {1.909091, {{3, 1}, {7, 1}, {11, -1}}},
    {1.914062, {{2, -7}, {5, 1}, {7, 2}}},
    {1.916667, {{2, -2}, {3, -1}, {23, 1}}},
    {1.920000, {{2, 4}, {3, 1}, {5, -2}}},
    {1.924338, {{2, -26}, {3, 17}}},
    {1.928361, {{2, -110}, {3, 70}}},
    {1.928571, {{2, -1}, {3, 3}, {7, -1}}},
    {1.939394, {{2, 6}, {3, -1}, {11, -1}}},
    {1.944444, {{2, -1}, {3, -2}, {5, 1}, {7, 1}}},
    {1.946524, {{2, 39}, {3, -24}}},
    {1.950000, {{2, -2}, {3, 1}, {5, -1}, {13, 1}}},
    {1.950593, {{2, -45}, {3, 29}}},
    {1.953125, {{2, -6}, {5, 3}}},
    {1.959184, {{2, 5}, {3, 1}, {7, -2}}},
    {1.968750, {{2, -5}, {3, 2}, {7, 1}}},
    {1.969231, {{2, 7}, {5, -1}, {13, -1}}},
    {1.973081, {{2, 20}, {3, -12}}},
    {1.975309, {{2, 5}, {3, -4}, {5, 1}}},
    {1.977205, {{2, -64}, {3, 41}}},
    {1.977539, {{2, -10}, {3, 4}, {5, 2}}},
    {1.992187, {{2, -7}, {3, 1}, {5, 1}, {17, 1}}},
};
static const int INTERVAL_TABLE_SIZE = 333;

// Lookup interval in the table (returns nullptr if not found)
const std::map<int,int>* lookup_interval(double interval_folded) {
    double key = std::round(interval_folded * 1e6) / 1e6;
    for (int i = 0; i < INTERVAL_TABLE_SIZE; ++i) {
        if (std::abs(INTERVAL_TABLE[i].key - key) < 5e-7) {
            return &INTERVAL_TABLE[i].powers;
        }
    }
    return nullptr;
}

} // anonymous namespace

std::string interval_to_fjs(Real interval, const std::string& unison,
                             Real tolerance, bool use_unicode) {
    if (interval <= 0) {
        throw ParameterError("Interval must be strictly positive");
    }

    // Find Pythagorean approximation
    int fifths = fifth_search(interval, tolerance);

    // Determine base note name
    std::string note_name = fifths_to_note(unison, fifths, use_unicode);

    // Look up prime factorization
    double interval_b = o_fold(interval);
    const auto* powers = lookup_interval(interval_b);
    if (!powers) {
        throw ParameterError("Unknown interval: " + std::to_string(interval));
    }

    // Collect non-Pythagorean prime factors (primes > 3)
    int otonal = 1;
    int utonal = 1;
    for (const auto& [p, e] : *powers) {
        if (p <= 3) continue;
        if (e > 0) {
            for (int i = 0; i < e; ++i) otonal *= p;
        } else if (e < 0) {
            for (int i = 0; i < -e; ++i) utonal *= p;
        }
    }

    std::string suffix;
    if (otonal > 1) {
        if (use_unicode) {
            suffix += to_superscript(otonal);
        } else {
            suffix += "^" + std::to_string(otonal);
        }
    }
    if (utonal > 1) {
        if (use_unicode) {
            suffix += to_subscript(utonal);
        } else {
            suffix += "_" + std::to_string(utonal);
        }
    }

    return note_name + suffix;
}

} // namespace librosa
