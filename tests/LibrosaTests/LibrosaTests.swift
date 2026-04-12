import Foundation
import XCTest
@testable import Librosa

final class LibrosaTests: XCTestCase {
    func testBasicConversionsAndTone() throws {
        XCTAssertEqual(try Librosa.midiToHz(69), 440, accuracy: 1e-12)
        XCTAssertEqual(try Librosa.hzToMidi(440), 69, accuracy: 1e-12)
        XCTAssertEqual(try Librosa.noteToHz("A4"), 440, accuracy: 1e-12)

        let tone = try Librosa.tone(frequency: 440, sampleRate: 8_000, duration: 0.25)
        XCTAssertEqual(tone.count, 2_000)
        XCTAssertEqual(tone[0], 0, accuracy: 1e-12)
        XCTAssertGreaterThan(tone.map(abs).max() ?? 0, 0.9)
    }

    func testAudioToolboxLoadReadsPCM_Wav() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("librosa-swift-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        try writeWav(url: url, sampleRate: 8_000, channels: 2, frames: [
            [0.0, 0.25],
            [0.5, -0.5],
            [-0.25, 0.75],
            [1.0, -1.0],
        ])

        let info = try Librosa.audioInfo(path: url.path)
        XCTAssertEqual(info.samples, 4)
        XCTAssertEqual(info.sampleRate, 8_000)
        XCTAssertEqual(info.channels, 2)
        XCTAssertEqual(info.duration, 0.0005, accuracy: 1e-12)

        let audio = try Librosa.load(path: url.path, sampleRate: nil, mono: false)
        XCTAssertEqual(audio.channels, 2)
        XCTAssertEqual(audio.samplesPerChannel, 4)
        XCTAssertEqual(audio.sampleRate, 8_000)
        XCTAssertEqual(audio.samples[0, 1], 0.5, accuracy: 1e-4)
        XCTAssertEqual(audio.samples[1, 1], -0.5, accuracy: 1e-4)
    }

    func testSwiftMFCCMatchesPythonReferenceData() throws {
        let expected = try referenceArray("feature_mfcc")

        let actual = try Librosa.mfcc(Self.featureTestSignal(), sampleRate: 22_050, nMFCC: 13)

        XCTAssertEqual(actual.rows, expected.shape[0])
        XCTAssertEqual(actual.columns, expected.shape[1])
        assertRelativeNear(actual.values, expected.data, tolerance: 1e-3)
    }

    func testSwiftMelSpectrogramMatchesPythonReferenceData() throws {
        let expected = try referenceArray("feature_melspectrogram")

        let actual = try Librosa.melspectrogram(Self.featureTestSignal(), sampleRate: 22_050)

        XCTAssertEqual(actual.rows, expected.shape[0])
        XCTAssertEqual(actual.columns, expected.shape[1])
        assertRelativeNear(actual.values, expected.data, tolerance: 1e-3)
    }

    func testSTFTMagnitudeAmplitudeToDBAndHPSS() throws {
        let audio = Self.clickTrack(sampleRate: 22_050, bpm: 120, duration: 2.0)

        let spectrum = try Librosa.stftMagnitude(audio)
        XCTAssertEqual(spectrum.rows, 1025)
        XCTAssertGreaterThan(spectrum.columns, 0)
        XCTAssertGreaterThan(spectrum.values.max() ?? 0, 0)

        let reference = spectrum.values.max() ?? 1
        let db = try Librosa.amplitudeToDB(spectrum, ref: reference)
        XCTAssertEqual(db.rows, spectrum.rows)
        XCTAssertEqual(db.columns, spectrum.columns)
        XCTAssertTrue(db.values.allSatisfy(\.isFinite))
        XCTAssertLessThanOrEqual(db.values.max() ?? 1, 1e-9)

        let split = try Librosa.hpss(spectrum, margin: 2)
        XCTAssertEqual(split.harmonic.rows, spectrum.rows)
        XCTAssertEqual(split.harmonic.columns, spectrum.columns)
        XCTAssertEqual(split.percussive.rows, spectrum.rows)
        XCTAssertEqual(split.percussive.columns, spectrum.columns)
        XCTAssertTrue(split.harmonic.values.allSatisfy { $0 >= 0 && $0.isFinite })
        XCTAssertTrue(split.percussive.values.allSatisfy { $0 >= 0 && $0.isFinite })
    }

    func testOnsetStrengthAndDetectFromSpectrogram() throws {
        let sampleRate = 22_050.0
        let hopLength = 512
        let audio = Self.clickTrack(sampleRate: sampleRate, bpm: 120, duration: 4.0)
        let mel = try Librosa.melspectrogram(audio, sampleRate: sampleRate, hopLength: hopLength)
        let melDB = try Librosa.powerToDB(mel, ref: mel.values.max() ?? 1)

        let envelope = try Librosa.onsetStrength(spectrogram: melDB,
                                                 sampleRate: sampleRate,
                                                 hopLength: hopLength)
        let onsets = try Librosa.onsetDetect(onsetEnvelope: envelope,
                                             sampleRate: sampleRate,
                                             hopLength: hopLength)

        XCTAssertEqual(envelope.count, mel.columns)
        XCTAssertFalse(onsets.isEmpty)
    }

    func testChromaCQTAndFilteringAPIs() throws {
        let sampleRate = 22_050.0
        let audio = Self.featureTestSignal()

        let cqt = try Librosa.cqtMagnitude(audio,
                                           sampleRate: sampleRate,
                                           nBins: 7 * 12,
                                           binsPerOctave: 12)
        XCTAssertEqual(cqt.rows, 84)
        XCTAssertGreaterThan(cqt.columns, 0)

        let chroma = try Librosa.chromaCQT(audio,
                                           sampleRate: sampleRate,
                                           binsPerOctave: 36)
        XCTAssertEqual(chroma.rows, 12)
        XCTAssertGreaterThan(chroma.columns, 0)

        let harmonic = try Librosa.harmonic(audio, margin: 2)
        XCTAssertFalse(harmonic.isEmpty)

        let filtered = try Librosa.nnFilter(chroma,
                                            metric: "cosine",
                                            aggregateMedian: true)
        XCTAssertEqual(filtered.rows, chroma.rows)
        XCTAssertEqual(filtered.columns, chroma.columns)

        let mask = try Librosa.softmask(
            LibrosaMatrix(rows: 1, columns: 3, values: [3, 0, 2]),
            xRef: LibrosaMatrix(rows: 1, columns: 3, values: [1, 0, 2]),
            power: 1,
            splitZeros: true
        )
        XCTAssertEqual(mask.values[0], 0.75, accuracy: 1e-12)
        XCTAssertEqual(mask.values[1], 0.5, accuracy: 1e-12)
        XCTAssertEqual(mask.values[2], 0.5, accuracy: 1e-12)

        let x = LibrosaMatrix(rows: 2, columns: 3, values: [
            1, 0, 1,
            0, 1, 0,
        ])
        let y = LibrosaMatrix(rows: 2, columns: 3, values: [
            1, 0, 1,
            0, 1, 0,
        ])
        let alignment = try Librosa.dtw(x, y: y, metric: "cosine")
        XCTAssertEqual(alignment.cost.rows, 3)
        XCTAssertEqual(alignment.cost.columns, 3)
        XCTAssertEqual(alignment.path.first, LibrosaDTWPathPoint(firstFrame: 0, secondFrame: 0))
        XCTAssertEqual(alignment.path.last, LibrosaDTWPathPoint(firstFrame: 2, secondFrame: 2))

        let smoothed = try Librosa.medianFilter(filtered,
                                                sizeRows: 1,
                                                sizeColumns: 3)
        XCTAssertEqual(smoothed.rows, chroma.rows)
        XCTAssertEqual(smoothed.columns, chroma.columns)

        let syncInput = LibrosaMatrix(rows: 2, columns: 5, values: [
            1, 2, 3, 4, 5,
            10, 20, 30, 40, 50,
        ])
        let synced = try Librosa.sync(syncInput, indices: [0, 2, 5])
        XCTAssertEqual(synced.rows, 2)
        XCTAssertEqual(synced.columns, 2)
        XCTAssertEqual(synced.values, [1.5, 4.0, 15.0, 40.0])

        let recInput = LibrosaMatrix(rows: 2, columns: 8, values: [
            0, 1, 2, 3, 3, 2, 1, 0,
            1, 1, 1, 1, 0, 0, 0, 0,
        ])
        let recurrence = try Librosa.recurrenceMatrix(recInput,
                                                      k: 2,
                                                      width: 1,
                                                      sym: true,
                                                      mode: "affinity")
        XCTAssertEqual(recurrence.rows, 8)
        XCTAssertEqual(recurrence.columns, 8)
        XCTAssertTrue(recurrence.values.allSatisfy(\.isFinite))

        let filteredRecurrence = try Librosa.timelagMedianFilter(recurrence,
                                                                 sizeRows: 1,
                                                                 sizeColumns: 3)
        XCTAssertEqual(filteredRecurrence.rows, recurrence.rows)
        XCTAssertEqual(filteredRecurrence.columns, recurrence.columns)

        let graph = LibrosaMatrix(rows: 4, columns: 4, values: [
            0, 1, 0, 0,
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
        ])
        let components = try Librosa.laplacianComponents(graph,
                                                         components: 2,
                                                         medianFilterRows: 1)
        XCTAssertEqual(components.rows, 4)
        XCTAssertEqual(components.columns, 2)
        XCTAssertTrue(components.values.allSatisfy(\.isFinite))
    }

    func testTempoDetects120BPMOnsetEnvelope() throws {
        let sampleRate = 24_000.0
        let hopLength = 500
        let framesPerBeat = 24
        let onsetEnvelope = Self.pulseTrain(frames: framesPerBeat * 32,
                                            period: framesPerBeat)

        let tempo = try Librosa.tempo(onsetEnvelope: onsetEnvelope,
                                      sampleRate: sampleRate,
                                      hopLength: hopLength)

        XCTAssertEqual(tempo, 120.0, accuracy: 0.001)
    }

    func testTempoDetects120BPMAudioClickLoop() throws {
        let sampleRate = 24_000.0
        let bpm = 120.0
        let audio = Self.clickTrack(sampleRate: sampleRate,
                                    bpm: bpm,
                                    duration: 12.0)

        let tempo = try Librosa.tempo(audio,
                                      sampleRate: sampleRate,
                                      hopLength: 500)

        XCTAssertEqual(tempo, bpm, accuracy: 0.001)
    }

    func testBeatTrackDetectsAudioClickLoop() throws {
        let sampleRate = 24_000.0
        let bpm = 120.0
        let audio = Self.clickTrack(sampleRate: sampleRate,
                                    bpm: bpm,
                                    duration: 12.0)

        let result = try Librosa.beatTrack(audio,
                                          sampleRate: sampleRate,
                                          hopLength: 500,
                                          trim: false)

        XCTAssertEqual(result.tempo, bpm, accuracy: 0.001)
        XCTAssertFalse(result.beats.isEmpty)
    }

    func testTempoDetectsDefaultGrid120BPMAudioClickLoop() throws {
        let sampleRate = 22_050.0
        let hopLength = 512
        let bpm = 120.0
        let audio = Self.clickTrack(sampleRate: sampleRate,
                                    bpm: bpm,
                                    duration: 12.0)

        let tempo = try Librosa.tempo(audio,
                                      sampleRate: sampleRate,
                                      hopLength: hopLength)
        let expectedQuantizedTempo = 60.0 * sampleRate / Double(hopLength * 22)

        XCTAssertEqual(tempo, expectedQuantizedTempo, accuracy: 0.001)
        XCTAssertEqual(tempo, bpm, accuracy: 3.0)
    }

    func testTempoDetectsLoaded120BPMWavLoop() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("librosa-tempo-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }

        try writeMonoWav(url: url,
                         sampleRate: 44_100,
                         samples: Self.clickTrack(sampleRate: 44_100,
                                                  bpm: 120,
                                                  duration: 12.0))

        let loaded = try Librosa.load(path: url.path, sampleRate: nil, mono: true)
        XCTAssertEqual(loaded.sampleRate, 44_100)

        let onsetEnvelope = try Librosa.onsetStrength(loaded.mono,
                                                      sampleRate: loaded.sampleRate)
        let tempo = try Librosa.tempo(onsetEnvelope: onsetEnvelope,
                                      sampleRate: loaded.sampleRate)
        let expectedQuantizedTempo = 60.0 * loaded.sampleRate / Double(512 * 43)

        XCTAssertEqual(tempo, expectedQuantizedTempo, accuracy: 0.001)
        XCTAssertEqual(tempo, 120.0, accuracy: 3.0)
    }
}

private extension LibrosaTests {
    static func featureTestSignal() -> [Double] {
        let sampleRate = 22_050.0
        return (0..<Int(sampleRate)).map { index in
            let t = Double(index) / sampleRate
            let sample = 0.5 * sin(2 * Double.pi * 440 * t)
                + 0.3 * sin(2 * Double.pi * 880 * t)
                + 0.2 * sin(2 * Double.pi * 1_320 * t)
            return Double(Float(sample))
        }
    }

    static func pulseTrain(frames: Int, period: Int) -> [Double] {
        (0..<frames).map { frame in
            frame % period == 0 ? 1.0 : 0.0
        }
    }

    static func clickTrack(sampleRate: Double, bpm: Double, duration: Double) -> [Double] {
        let sampleCount = Int(sampleRate * duration)
        let samplesPerBeat = Int(sampleRate * 60.0 / bpm)
        var audio = Array(repeating: 0.0, count: sampleCount)

        for sample in stride(from: 0, to: sampleCount, by: samplesPerBeat) {
            for offset in 0..<64 where sample + offset < sampleCount {
                let decay = 1.0 - Double(offset) / 64.0
                audio[sample + offset] = decay
            }
        }

        return audio
    }
}

private struct ReferenceArray: Decodable {
    let shape: [Int]
    let data: [Double]
}

private func referenceArray(_ name: String) throws -> ReferenceArray {
    guard let url = Bundle.module.url(forResource: name,
                                      withExtension: "json",
                                      subdirectory: "ReferenceData") else {
        throw XCTSkip("Swift cross-validation reference data not found: \(name)")
    }
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(ReferenceArray.self, from: data)
}

private func assertRelativeNear(_ actual: [Double],
                                _ expected: [Double],
                                tolerance: Double,
                                file: StaticString = #filePath,
                                line: UInt = #line) {
    XCTAssertEqual(actual.count, expected.count, file: file, line: line)
    for (index, pair) in zip(actual, expected).enumerated() {
        let scale = max(abs(pair.1), 1.0)
        XCTAssertEqual(pair.0, pair.1, accuracy: tolerance * scale,
                       "index \(index)", file: file, line: line)
    }
}

private func writeWav(url: URL,
                      sampleRate: UInt32,
                      channels: UInt16,
                      frames: [[Double]]) throws {
    var data = Data()
    let bytesPerSample: UInt16 = 2
    let dataSize = UInt32(frames.count) * UInt32(channels) * UInt32(bytesPerSample)

    data.appendASCII("RIFF")
    data.appendLittleEndian(UInt32(36) + dataSize)
    data.appendASCII("WAVE")
    data.appendASCII("fmt ")
    data.appendLittleEndian(UInt32(16))
    data.appendLittleEndian(UInt16(1))
    data.appendLittleEndian(channels)
    data.appendLittleEndian(sampleRate)
    data.appendLittleEndian(sampleRate * UInt32(channels) * UInt32(bytesPerSample))
    data.appendLittleEndian(channels * bytesPerSample)
    data.appendLittleEndian(UInt16(16))
    data.appendASCII("data")
    data.appendLittleEndian(dataSize)

    for frame in frames {
        precondition(frame.count == Int(channels))
        for sample in frame {
            let clipped = min(1.0, max(-1.0, sample))
            data.appendLittleEndian(Int16(clipped * Double(Int16.max)))
        }
    }

    try data.write(to: url)
}

private func writeMonoWav(url: URL,
                          sampleRate: UInt32,
                          samples: [Double]) throws {
    var data = Data()
    let bytesPerSample: UInt16 = 2
    let dataSize = UInt32(samples.count) * UInt32(bytesPerSample)

    data.appendASCII("RIFF")
    data.appendLittleEndian(UInt32(36) + dataSize)
    data.appendASCII("WAVE")
    data.appendASCII("fmt ")
    data.appendLittleEndian(UInt32(16))
    data.appendLittleEndian(UInt16(1))
    data.appendLittleEndian(UInt16(1))
    data.appendLittleEndian(sampleRate)
    data.appendLittleEndian(sampleRate * UInt32(bytesPerSample))
    data.appendLittleEndian(bytesPerSample)
    data.appendLittleEndian(UInt16(16))
    data.appendASCII("data")
    data.appendLittleEndian(dataSize)

    for sample in samples {
        let clipped = min(1.0, max(-1.0, sample))
        data.appendLittleEndian(Int16(clipped * Double(Int16.max)))
    }

    try data.write(to: url)
}

private extension Data {
    mutating func appendASCII(_ value: String) {
        append(value.data(using: .ascii)!)
    }

    mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var littleEndian = value.littleEndian
        Swift.withUnsafeBytes(of: &littleEndian) { bytes in
            append(contentsOf: bytes)
        }
    }
}
