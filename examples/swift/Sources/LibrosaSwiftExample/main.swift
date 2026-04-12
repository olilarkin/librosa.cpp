import Foundation
import Librosa

@main
struct LibrosaSwiftExample {
    static func main() throws {
        let inputPath = CommandLine.arguments.dropFirst().first
        let sampleRate = 22_050.0

        let audio: [Double]
        let source: String

        if let inputPath {
            let loaded = try Librosa.load(path: inputPath, sampleRate: sampleRate, mono: true)
            audio = loaded.mono
            source = URL(fileURLWithPath: inputPath).lastPathComponent
        } else {
            audio = try Librosa.tone(frequency: 440, sampleRate: sampleRate, duration: 1.0)
            source = "generated 440 Hz tone"
        }

        let duration = Double(audio.count) / sampleRate
        let mfcc = try Librosa.mfcc(audio, sampleRate: sampleRate, nMFCC: 13)
        let mel = try Librosa.melspectrogram(audio, sampleRate: sampleRate)
        let onsetEnvelope = try Librosa.onsetStrength(audio, sampleRate: sampleRate)
        let tempo = try Librosa.tempo(onsetEnvelope: onsetEnvelope, sampleRate: sampleRate)
        let centroid = try Librosa.spectralCentroid(audio, sampleRate: sampleRate)
        let rolloff = try Librosa.spectralRolloff(audio, sampleRate: sampleRate)

        print("Librosa Swift example")
        print("source: \(source)")
        print("samples: \(audio.count)")
        print("duration: \(format(duration)) s")
        print("mfcc: \(mfcc.rows)x\(mfcc.columns)")
        print("mel spectrogram: \(mel.rows)x\(mel.columns)")
        print("onset envelope frames: \(onsetEnvelope.count)")
        print("tempo: \(format(tempo)) BPM")
        print("spectral centroid mean: \(format(mean(centroid.values))) Hz")
        print("spectral rolloff mean: \(format(mean(rolloff.values))) Hz")
        print("first MFCC frame: \(preview(Array(mfcc.values.prefix(mfcc.rows))))")
    }

    private static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else {
            return 0
        }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }

    private static func preview(_ values: [Double], maxCount: Int = 6) -> String {
        let shown = values.prefix(maxCount).map(format)
        if values.count > maxCount {
            return "[\(shown.joined(separator: ", ")), ...]"
        }
        return "[\(shown.joined(separator: ", "))]"
    }
}
