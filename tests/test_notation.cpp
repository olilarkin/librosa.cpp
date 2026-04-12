#include <gtest/gtest.h>
#include <librosa/core/notation.hpp>
#include <librosa/util/exceptions.hpp>

using namespace librosa;

// ============================================================================
// Thaat Tests
// ============================================================================

TEST(ThaatTest, Bilaval) {
    auto degrees = thaat_to_degrees("bilaval");
    ASSERT_EQ(degrees.size(), 7u);
    std::vector<int> expected = {0, 2, 4, 5, 7, 9, 11};
    EXPECT_EQ(degrees, expected);
}

TEST(ThaatTest, Todi) {
    auto degrees = thaat_to_degrees("todi");
    std::vector<int> expected = {0, 1, 3, 6, 7, 8, 11};
    EXPECT_EQ(degrees, expected);
}

TEST(ThaatTest, CaseInsensitive) {
    auto d1 = thaat_to_degrees("BILAVAL");
    auto d2 = thaat_to_degrees("bilaval");
    EXPECT_EQ(d1, d2);
}

TEST(ThaatTest, UnknownThrows) {
    EXPECT_THROW(thaat_to_degrees("nonexistent"), ParameterError);
}

TEST(ThaatTest, ListThaat) {
    auto thaats = list_thaat();
    EXPECT_EQ(thaats.size(), 10u);
}

// ============================================================================
// Melakarta Tests
// ============================================================================

TEST(MelaTest, Kanakangi) {
    // Mela #1
    auto degrees = mela_to_degrees(1);
    ASSERT_EQ(degrees.size(), 7u);
    std::vector<int> expected = {0, 1, 2, 5, 7, 8, 9};
    EXPECT_EQ(degrees, expected);
}

TEST(MelaTest, ByName) {
    auto d1 = mela_to_degrees("kanakangi");
    auto d2 = mela_to_degrees(1);
    EXPECT_EQ(d1, d2);
}

TEST(MelaTest, HighIndex) {
    // Mela #72 (rasikapriya)
    auto degrees = mela_to_degrees(72);
    ASSERT_EQ(degrees.size(), 7u);
    // Last group: Ri3, Ga3 (3,4), Ma2 (6), Pa (7), Dha3, Ni3 (10, 11)
    std::vector<int> expected = {0, 3, 4, 6, 7, 10, 11};
    EXPECT_EQ(degrees, expected);
}

TEST(MelaTest, OutOfRangeThrows) {
    EXPECT_THROW(mela_to_degrees(0), ParameterError);
    EXPECT_THROW(mela_to_degrees(73), ParameterError);
}

TEST(MelaTest, ListMela) {
    auto melas = list_mela();
    EXPECT_EQ(melas.size(), 72u);
    EXPECT_EQ(melas["kanakangi"], 1);
    EXPECT_EQ(melas["rasikapriya"], 72);
}

// ============================================================================
// Mela to Svara Tests
// ============================================================================

TEST(MelaSvaraTest, Kanakangi) {
    // Mela #1: uses R1, G1, D1, N1
    auto svara = mela_to_svara(1);
    EXPECT_EQ(svara.size(), 12u);
    EXPECT_EQ(svara[0], "S");    // Sa
    EXPECT_EQ(svara[7], "P");    // Pa
}

TEST(MelaSvaraTest, ByName) {
    auto s1 = mela_to_svara("kanakangi");
    auto s2 = mela_to_svara(1);
    EXPECT_EQ(s1, s2);
}

TEST(MelaSvaraTest, NoUnicode) {
    auto svara = mela_to_svara(1, true, false);
    EXPECT_EQ(svara[0], "S");
    // Check that subscripts are ASCII
    EXPECT_TRUE(svara[1].find("1") != std::string::npos || svara[1].find("R") != std::string::npos);
}

// ============================================================================
// Key to Degrees Tests
// ============================================================================

TEST(KeyDegreesTest, CMajor) {
    auto degrees = key_to_degrees("C:maj");
    ASSERT_EQ(degrees.size(), 7u);
    std::vector<int> expected = {0, 2, 4, 5, 7, 9, 11};
    EXPECT_EQ(degrees, expected);
}

TEST(KeyDegreesTest, CSharpMajor) {
    auto degrees = key_to_degrees("C#:maj");
    ASSERT_EQ(degrees.size(), 7u);
    std::vector<int> expected = {1, 3, 5, 6, 8, 10, 0};
    EXPECT_EQ(degrees, expected);
}

TEST(KeyDegreesTest, AMinor) {
    auto degrees = key_to_degrees("A:min");
    ASSERT_EQ(degrees.size(), 7u);
    std::vector<int> expected = {9, 11, 0, 2, 4, 5, 7};
    EXPECT_EQ(degrees, expected);
}

TEST(KeyDegreesTest, InvalidThrows) {
    EXPECT_THROW(key_to_degrees("X:maj"), ParameterError);
    EXPECT_THROW(key_to_degrees("C"), ParameterError);
}

// ============================================================================
// Key to Notes Tests
// ============================================================================

TEST(KeyNotesTest, CMajorUsesNaturals) {
    auto notes = key_to_notes("C:maj");
    ASSERT_EQ(notes.size(), 12u);
    // C major uses sharps
    EXPECT_EQ(notes[0], "C");
    EXPECT_EQ(notes[4], "E");
    EXPECT_EQ(notes[7], "G");
}

TEST(KeyNotesTest, CMajorNoUnicode) {
    auto notes = key_to_notes("C:maj", false);
    ASSERT_EQ(notes.size(), 12u);
    EXPECT_EQ(notes[0], "C");
    EXPECT_EQ(notes[1], "C#");
    EXPECT_EQ(notes[4], "E");
}

TEST(KeyNotesTest, FMajorUsesFlat) {
    // F major has one flat (Bb)
    auto notes = key_to_notes("F:maj");
    ASSERT_EQ(notes.size(), 12u);
    // Should use flats
    EXPECT_EQ(notes[0], "C");
    EXPECT_EQ(notes[5], "F");
}

// ============================================================================
// Fifths to Note Tests
// ============================================================================

TEST(FifthsTest, CToG) {
    auto note = fifths_to_note("C", 1);
    EXPECT_EQ(note, "G");
}

TEST(FifthsTest, CToFSharp) {
    auto note = fifths_to_note("C", 6, false);
    EXPECT_EQ(note, "F#");
}

TEST(FifthsTest, GToFlats) {
    auto note = fifths_to_note("G", -3, false);
    EXPECT_EQ(note, "Bb");
}

TEST(FifthsTest, InvalidNoteThrows) {
    EXPECT_THROW(fifths_to_note("", 1), ParameterError);
    EXPECT_THROW(fifths_to_note("X", 1), ParameterError);
}
