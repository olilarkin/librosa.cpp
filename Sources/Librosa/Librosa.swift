import CLibrosa
import Foundation

public enum LibrosaError: Error, CustomStringConvertible, Equatable {
    case operationFailed(String)

    public var description: String {
        switch self {
        case .operationFailed(let message):
            return message
        }
    }
}

public struct LibrosaMatrix: Equatable {
    public let rows: Int
    public let columns: Int
    public let values: [Double]

    public init(rows: Int, columns: Int, values: [Double]) {
        precondition(rows >= 0)
        precondition(columns >= 0)
        precondition(values.count == rows * columns)
        self.rows = rows
        self.columns = columns
        self.values = values
    }

    public subscript(row: Int, column: Int) -> Double {
        values[row * columns + column]
    }

    public func row(_ index: Int) -> ArraySlice<Double> {
        let start = index * columns
        return values[start..<(start + columns)]
    }
}

public struct LibrosaAudioData: Equatable {
    public let channels: Int
    public let samplesPerChannel: Int
    public let sampleRate: Double
    public let samples: LibrosaMatrix

    public var mono: [Double] {
        if channels == 1 {
            return Array(samples.row(0))
        }

        var result = Array(repeating: 0.0, count: samplesPerChannel)
        for channel in 0..<channels {
            for sample in 0..<samplesPerChannel {
                result[sample] += samples[channel, sample]
            }
        }
        return result.map { $0 / Double(channels) }
    }
}

public struct LibrosaAudioFileInfo: Equatable {
    public let samples: Int
    public let sampleRate: Double
    public let channels: Int
    public let duration: Double
}

public struct LibrosaBeatTrackResult: Equatable {
    public let tempo: Double
    public let beats: [Int]
}

public struct LibrosaTrimResult: Equatable {
    public let audio: [Double]
    public let interval: Range<Int>
}

public struct LibrosaHPSSResult: Equatable {
    public let harmonic: LibrosaMatrix
    public let percussive: LibrosaMatrix
}

public struct LibrosaDTWPathPoint: Equatable {
    public let firstFrame: Int
    public let secondFrame: Int
}

public struct LibrosaDTWResult: Equatable {
    public let cost: LibrosaMatrix
    public let path: [LibrosaDTWPathPoint]
}

public enum Librosa {
    public static func audioInfo(path: String) throws -> LibrosaAudioFileInfo {
        var raw = CLibrosa.LibrosaAudioFileInfo()
        try path.withCString { cPath in
            try check(librosa_audio_info(cPath, &raw))
        }
        return LibrosaAudioFileInfo(
            samples: Int(raw.samples),
            sampleRate: raw.sample_rate,
            channels: Int(raw.channels),
            duration: raw.duration
        )
    }

    public static func load(path: String,
                            sampleRate: Double? = 22_050,
                            mono: Bool = true,
                            offset: Double = 0,
                            duration: Double? = nil) throws -> LibrosaAudioData {
        var raw = CLibrosa.LibrosaAudioData()
        try path.withCString { cPath in
            try check(librosa_load(cPath,
                                   flag(sampleRate != nil),
                                   sampleRate ?? 0,
                                   flag(mono),
                                   offset,
                                   flag(duration != nil),
                                   duration ?? 0,
                                   &raw))
        }
        defer { librosa_audio_data_free(&raw) }

        let matrix = LibrosaMatrix(
            rows: Int(raw.channels),
            columns: Int(raw.samples),
            values: doubles(raw.data, count: Int(raw.channels * raw.samples))
        )
        return LibrosaAudioData(
            channels: Int(raw.channels),
            samplesPerChannel: Int(raw.samples),
            sampleRate: raw.sample_rate,
            samples: matrix
        )
    }

    public static func midiToHz(_ midi: Double) throws -> Double {
        try scalar(midi, librosa_midi_to_hz)
    }

    public static func hzToMidi(_ hz: Double) throws -> Double {
        try scalar(hz, librosa_hz_to_midi)
    }

    public static func hzToMel(_ hz: Double, htk: Bool = false) throws -> Double {
        var output = 0.0
        try check(librosa_hz_to_mel(hz, flag(htk), &output))
        return output
    }

    public static func melToHz(_ mel: Double, htk: Bool = false) throws -> Double {
        var output = 0.0
        try check(librosa_mel_to_hz(mel, flag(htk), &output))
        return output
    }

    public static func noteToMidi(_ note: String, round: Bool = true) throws -> Double {
        var output = 0.0
        try note.withCString { cNote in
            try check(librosa_note_to_midi(cNote, flag(round), &output))
        }
        return output
    }

    public static func noteToHz(_ note: String) throws -> Double {
        var output = 0.0
        try note.withCString { cNote in
            try check(librosa_note_to_hz(cNote, &output))
        }
        return output
    }

    public static func fftFrequencies(sampleRate: Double = 22_050,
                                      nFFT: Int = 2048) throws -> [Double] {
        try vector { out in
            librosa_fft_frequencies(sampleRate, CInt(nFFT), out)
        }
    }

    public static func melFrequencies(nMels: Int = 128,
                                      fmin: Double = 0,
                                      fmax: Double = 11_025,
                                      htk: Bool = false) throws -> [Double] {
        try vector { out in
            librosa_mel_frequencies(CInt(nMels), fmin, fmax, flag(htk), out)
        }
    }

    public static func tone(frequency: Double,
                            sampleRate: Double = 22_050,
                            length: Int? = nil,
                            duration: Double? = nil,
                            phi: Double? = nil) throws -> [Double] {
        try vector { out in
            librosa_tone(frequency,
                         sampleRate,
                         flag(length != nil),
                         Int64(length ?? 0),
                         flag(duration != nil),
                         duration ?? 0,
                         flag(phi != nil),
                         phi ?? 0,
                         out)
        }
    }

    public static func chirp(fmin: Double,
                             fmax: Double,
                             sampleRate: Double = 22_050,
                             length: Int? = nil,
                             duration: Double? = nil,
                             linear: Bool = false,
                             phi: Double? = nil) throws -> [Double] {
        try vector { out in
            librosa_chirp(fmin,
                          fmax,
                          sampleRate,
                          flag(length != nil),
                          Int64(length ?? 0),
                          flag(duration != nil),
                          duration ?? 0,
                          flag(linear),
                          flag(phi != nil),
                          phi ?? 0,
                          out)
        }
    }

    public static func resample(_ y: [Double],
                                originalSampleRate: Double,
                                targetSampleRate: Double,
                                resType: String = "kaiser_hq",
                                fix: Bool = true,
                                scale: Bool = false) throws -> [Double] {
        try y.withUnsafeBufferPointer { buffer in
            try resType.withCString { cResType in
                try vector { out in
                    librosa_resample(buffer.baseAddress,
                                     Int64(buffer.count),
                                     originalSampleRate,
                                     targetSampleRate,
                                     cResType,
                                     flag(fix),
                                     flag(scale),
                                     out)
                }
            }
        }
    }

    public static func stftMagnitude(_ y: [Double],
                                     nFFT: Int = 2048,
                                     hopLength: Int = 512) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_stft_magnitude(buffer.baseAddress,
                                   Int64(buffer.count),
                                   CInt(nFFT),
                                   CInt(hopLength),
                                   out)
        }
    }

    public static func amplitudeToDB(_ spectrogram: LibrosaMatrix,
                                     ref: Double = 1.0,
                                     amin: Double = 1e-5,
                                     topDB: Double? = 80) throws -> LibrosaMatrix {
        try matrix(from: spectrogram) { buffer, out in
            librosa_amplitude_to_db(buffer.baseAddress,
                                    Int64(spectrogram.rows),
                                    Int64(spectrogram.columns),
                                    ref,
                                    amin,
                                    flag(topDB != nil),
                                    topDB ?? 0,
                                    out)
        }
    }

    public static func powerToDB(_ spectrogram: LibrosaMatrix,
                                 ref: Double = 1.0,
                                 amin: Double = 1e-10,
                                 topDB: Double? = 80) throws -> LibrosaMatrix {
        try matrix(from: spectrogram) { buffer, out in
            librosa_power_to_db(buffer.baseAddress,
                                Int64(spectrogram.rows),
                                Int64(spectrogram.columns),
                                ref,
                                amin,
                                flag(topDB != nil),
                                topDB ?? 0,
                                out)
        }
    }

    public static func hpss(_ spectrogram: LibrosaMatrix,
                            kernelSize: Int = 31,
                            power: Double = 2,
                            mask: Bool = false,
                            margin: Double = 1) throws -> LibrosaHPSSResult {
        try matrixPair(from: spectrogram) { buffer, harmonic, percussive in
            librosa_hpss(buffer.baseAddress,
                         Int64(spectrogram.rows),
                         Int64(spectrogram.columns),
                         CInt(kernelSize),
                         power,
                         flag(mask),
                         margin,
                         harmonic,
                         percussive)
        }
    }

    public static func cqtMagnitude(_ y: [Double],
                                    sampleRate: Double = 22_050,
                                    hopLength: Int = 512,
                                    fmin: Double? = nil,
                                    nBins: Int = 84,
                                    binsPerOctave: Int = 12) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_cqt_magnitude(buffer.baseAddress,
                                  Int64(buffer.count),
                                  sampleRate,
                                  CInt(hopLength),
                                  flag(fmin != nil),
                                  fmin ?? 0,
                                  CInt(nBins),
                                  CInt(binsPerOctave),
                                  out)
        }
    }

    public static func chromaCQT(_ y: [Double],
                                 sampleRate: Double = 22_050,
                                 hopLength: Int = 512,
                                 fmin: Double? = nil,
                                 nChroma: Int = 12,
                                 nOctaves: Int = 7,
                                 binsPerOctave: Int = 36) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_chroma_cqt(buffer.baseAddress,
                               Int64(buffer.count),
                               sampleRate,
                               CInt(hopLength),
                               flag(fmin != nil),
                               fmin ?? 0,
                               CInt(nChroma),
                               CInt(nOctaves),
                               CInt(binsPerOctave),
                               out)
        }
    }

    public static func harmonic(_ y: [Double],
                                kernelSize: Int = 31,
                                power: Double = 2,
                                mask: Bool = false,
                                margin: Double = 1,
                                nFFT: Int = 2048,
                                hopLength: Int = 512) throws -> [Double] {
        try y.withUnsafeBufferPointer { buffer in
            try vector { out in
                librosa_harmonic_effect(buffer.baseAddress,
                                        Int64(buffer.count),
                                        CInt(kernelSize),
                                        power,
                                        flag(mask),
                                        margin,
                                        CInt(nFFT),
                                        CInt(hopLength),
                                        out)
            }
        }
    }

    public static func nnFilter(_ matrix: LibrosaMatrix,
                                metric: String = "euclidean",
                                aggregateMedian: Bool = false,
                                k: Int = 0,
                                width: Int = 1) throws -> LibrosaMatrix {
        try matrix.values.withUnsafeBufferPointer { buffer in
            try metric.withCString { cMetric in
                var raw = CLibrosa.LibrosaMatrix()
                try check(librosa_nn_filter(buffer.baseAddress,
                                            Int64(matrix.rows),
                                            Int64(matrix.columns),
                                            cMetric,
                                            flag(aggregateMedian),
                                            CInt(k),
                                            CInt(width),
                                            &raw))
                defer { librosa_matrix_free(&raw) }
                return LibrosaMatrix(
                    rows: Int(raw.rows),
                    columns: Int(raw.columns),
                    values: doubles(raw.data, count: Int(raw.rows * raw.columns))
                )
            }
        }
    }

    public static func softmask(_ x: LibrosaMatrix,
                                xRef: LibrosaMatrix,
                                power: Double = 1,
                                splitZeros: Bool = false) throws -> LibrosaMatrix {
        precondition(x.rows == xRef.rows)
        precondition(x.columns == xRef.columns)
        return try x.values.withUnsafeBufferPointer { xBuffer in
            try xRef.values.withUnsafeBufferPointer { refBuffer in
                var raw = CLibrosa.LibrosaMatrix()
                try check(librosa_softmask(xBuffer.baseAddress,
                                           refBuffer.baseAddress,
                                           Int64(x.rows),
                                           Int64(x.columns),
                                           power,
                                           flag(splitZeros),
                                           &raw))
                defer { librosa_matrix_free(&raw) }
                return LibrosaMatrix(
                    rows: Int(raw.rows),
                    columns: Int(raw.columns),
                    values: doubles(raw.data, count: Int(raw.rows * raw.columns))
                )
            }
        }
    }

    public static func medianFilter(_ matrix: LibrosaMatrix,
                                    sizeRows: Int,
                                    sizeColumns: Int) throws -> LibrosaMatrix {
        try self.matrix(from: matrix) { buffer, out in
            librosa_median_filter(buffer.baseAddress,
                                  Int64(matrix.rows),
                                  Int64(matrix.columns),
                                  CInt(sizeRows),
                                  CInt(sizeColumns),
                                  out)
        }
    }

    public static func sync(_ matrix: LibrosaMatrix,
                            indices: [Int],
                            aggregateMedian: Bool = false,
                            pad: Bool = true,
                            axis: Int = -1) throws -> LibrosaMatrix {
        let rawIndices = indices.map(Int64.init)
        return try matrix.values.withUnsafeBufferPointer { buffer in
            try rawIndices.withUnsafeBufferPointer { indexBuffer in
                var raw = CLibrosa.LibrosaMatrix()
                try check(librosa_sync(buffer.baseAddress,
                                       Int64(matrix.rows),
                                       Int64(matrix.columns),
                                       indexBuffer.baseAddress,
                                       Int64(indexBuffer.count),
                                       flag(aggregateMedian),
                                       flag(pad),
                                       CInt(axis),
                                       &raw))
                defer { librosa_matrix_free(&raw) }
                return LibrosaMatrix(
                    rows: Int(raw.rows),
                    columns: Int(raw.columns),
                    values: doubles(raw.data, count: Int(raw.rows * raw.columns))
                )
            }
        }
    }

    public static func recurrenceMatrix(_ matrix: LibrosaMatrix,
                                        k: Int = 0,
                                        width: Int = 1,
                                        metric: String = "euclidean",
                                        sym: Bool = false,
                                        mode: String = "connectivity",
                                        bandwidth: Double = 0,
                                        selfEdges: Bool = false) throws -> LibrosaMatrix {
        try matrix.values.withUnsafeBufferPointer { buffer in
            try metric.withCString { cMetric in
                try mode.withCString { cMode in
                    var raw = CLibrosa.LibrosaMatrix()
                    try check(librosa_recurrence_matrix(buffer.baseAddress,
                                                        Int64(matrix.rows),
                                                        Int64(matrix.columns),
                                                        CInt(k),
                                                        CInt(width),
                                                        cMetric,
                                                        flag(sym),
                                                        cMode,
                                                        bandwidth,
                                                        flag(selfEdges),
                                                        &raw))
                    defer { librosa_matrix_free(&raw) }
                    return LibrosaMatrix(
                        rows: Int(raw.rows),
                        columns: Int(raw.columns),
                        values: doubles(raw.data, count: Int(raw.rows * raw.columns))
                    )
                }
            }
        }
    }

    public static func timelagMedianFilter(_ matrix: LibrosaMatrix,
                                           sizeRows: Int,
                                           sizeColumns: Int,
                                           pad: Bool = true) throws -> LibrosaMatrix {
        try self.matrix(from: matrix) { buffer, out in
            librosa_timelag_median_filter(buffer.baseAddress,
                                          Int64(matrix.rows),
                                          Int64(matrix.columns),
                                          CInt(sizeRows),
                                          CInt(sizeColumns),
                                          flag(pad),
                                          out)
        }
    }

    public static func laplacianComponents(_ matrix: LibrosaMatrix,
                                           components: Int,
                                           medianFilterRows: Int = 9) throws -> LibrosaMatrix {
        try self.matrix(from: matrix) { buffer, out in
            librosa_laplacian_components(buffer.baseAddress,
                                         Int64(matrix.rows),
                                         Int64(matrix.columns),
                                         CInt(components),
                                         CInt(medianFilterRows),
                                         out)
        }
    }

    public static func melspectrogram(_ y: [Double],
                                      sampleRate: Double = 22_050,
                                      nFFT: Int = 2048,
                                      hopLength: Int = 512,
                                      nMels: Int = 128,
                                      fmin: Double = 0,
                                      fmax: Double? = nil,
                                      htk: Bool = false,
                                      normSlaney: Bool = true) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_melspectrogram(buffer.baseAddress,
                                   Int64(buffer.count),
                                   sampleRate,
                                   CInt(nFFT),
                                   CInt(hopLength),
                                   CInt(nMels),
                                   fmin,
                                   flag(fmax != nil),
                                   fmax ?? 0,
                                   flag(htk),
                                   flag(normSlaney),
                                   out)
        }
    }

    public static func mfcc(_ y: [Double],
                            sampleRate: Double = 22_050,
                            nMFCC: Int = 20,
                            nFFT: Int = 2048,
                            hopLength: Int = 512,
                            nMels: Int = 128,
                            fmin: Double = 0,
                            fmax: Double? = nil,
                            htk: Bool = false) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_mfcc(buffer.baseAddress,
                         Int64(buffer.count),
                         sampleRate,
                         CInt(nMFCC),
                         CInt(nFFT),
                         CInt(hopLength),
                         CInt(nMels),
                         fmin,
                         flag(fmax != nil),
                         fmax ?? 0,
                         flag(htk),
                         out)
        }
    }

    public static func chromaSTFT(_ y: [Double],
                                  sampleRate: Double = 22_050,
                                  nFFT: Int = 2048,
                                  hopLength: Int = 512,
                                  nChroma: Int = 12,
                                  tuning: Double? = nil,
                                  norm: Double = .infinity) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_chroma_stft_options(buffer.baseAddress,
                                        Int64(buffer.count),
                                        sampleRate,
                                        CInt(nFFT),
                                        CInt(hopLength),
                                        CInt(nChroma),
                                        flag(tuning != nil),
                                        tuning ?? 0,
                                        norm,
                                        out)
        }
    }

    public static func dtw(_ x: LibrosaMatrix,
                           y: LibrosaMatrix,
                           metric: String = "euclidean",
                           subseq: Bool = false) throws -> LibrosaDTWResult {
        precondition(x.rows == y.rows)
        return try x.values.withUnsafeBufferPointer { xBuffer in
            try y.values.withUnsafeBufferPointer { yBuffer in
                try metric.withCString { cMetric in
                    var raw = CLibrosa.LibrosaDTWResult()
                    try check(librosa_dtw(xBuffer.baseAddress,
                                          Int64(x.rows),
                                          Int64(x.columns),
                                          yBuffer.baseAddress,
                                          Int64(y.rows),
                                          Int64(y.columns),
                                          cMetric,
                                          flag(subseq),
                                          &raw))
                    defer { librosa_dtw_result_free(&raw) }

                    let cost = LibrosaMatrix(
                        rows: Int(raw.cost.rows),
                        columns: Int(raw.cost.columns),
                        values: doubles(raw.cost.data, count: Int(raw.cost.rows * raw.cost.columns))
                    )
                    let pathData = ints(raw.path.data, count: Int(raw.path.count))
                    let path = stride(from: 0, to: pathData.count, by: 2).compactMap { index -> LibrosaDTWPathPoint? in
                        guard index + 1 < pathData.count else {
                            return nil
                        }
                        return LibrosaDTWPathPoint(
                            firstFrame: pathData[index],
                            secondFrame: pathData[index + 1]
                        )
                    }
                    return LibrosaDTWResult(cost: cost, path: path)
                }
            }
        }
    }

    public static func spectralCentroid(_ y: [Double],
                                        sampleRate: Double = 22_050,
                                        nFFT: Int = 2048,
                                        hopLength: Int = 512) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_spectral_centroid(buffer.baseAddress,
                                      Int64(buffer.count),
                                      sampleRate,
                                      CInt(nFFT),
                                      CInt(hopLength),
                                      out)
        }
    }

    public static func spectralBandwidth(_ y: [Double],
                                         sampleRate: Double = 22_050,
                                         nFFT: Int = 2048,
                                         hopLength: Int = 512,
                                         p: Double = 2,
                                         norm: Bool = true) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_spectral_bandwidth(buffer.baseAddress,
                                       Int64(buffer.count),
                                       sampleRate,
                                       CInt(nFFT),
                                       CInt(hopLength),
                                       p,
                                       flag(norm),
                                       out)
        }
    }

    public static func spectralRolloff(_ y: [Double],
                                       sampleRate: Double = 22_050,
                                       nFFT: Int = 2048,
                                       hopLength: Int = 512,
                                       rollPercent: Double = 0.85) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_spectral_rolloff(buffer.baseAddress,
                                     Int64(buffer.count),
                                     sampleRate,
                                     CInt(nFFT),
                                     CInt(hopLength),
                                     rollPercent,
                                     out)
        }
    }

    public static func spectralFlatness(_ y: [Double],
                                        nFFT: Int = 2048,
                                        hopLength: Int = 512) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_spectral_flatness(buffer.baseAddress,
                                      Int64(buffer.count),
                                      CInt(nFFT),
                                      CInt(hopLength),
                                      out)
        }
    }

    public static func spectralContrast(_ y: [Double],
                                        sampleRate: Double = 22_050,
                                        nFFT: Int = 2048,
                                        hopLength: Int = 512,
                                        fmin: Double = 200,
                                        nBands: Int = 6,
                                        quantile: Double = 0.02,
                                        linear: Bool = false) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_spectral_contrast(buffer.baseAddress,
                                      Int64(buffer.count),
                                      sampleRate,
                                      CInt(nFFT),
                                      CInt(hopLength),
                                      fmin,
                                      CInt(nBands),
                                      quantile,
                                      flag(linear),
                                      out)
        }
    }

    public static func rms(_ y: [Double],
                           frameLength: Int = 2048,
                           hopLength: Int = 512,
                           center: Bool = true) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_rms(buffer.baseAddress,
                        Int64(buffer.count),
                        CInt(frameLength),
                        CInt(hopLength),
                        flag(center),
                        out)
        }
    }

    public static func zeroCrossingRate(_ y: [Double],
                                        frameLength: Int = 2048,
                                        hopLength: Int = 512,
                                        center: Bool = true,
                                        threshold: Double = 0) throws -> LibrosaMatrix {
        try matrix(from: y) { buffer, out in
            librosa_zero_crossing_rate(buffer.baseAddress,
                                       Int64(buffer.count),
                                       CInt(frameLength),
                                       CInt(hopLength),
                                       flag(center),
                                       threshold,
                                       out)
        }
    }

    public static func onsetStrength(_ y: [Double],
                                     sampleRate: Double = 22_050,
                                     nFFT: Int = 2048,
                                     hopLength: Int = 512,
                                     lag: Int = 1,
                                     maxSize: Int = 1,
                                     detrend: Bool = false,
                                     center: Bool = true) throws -> [Double] {
        try y.withUnsafeBufferPointer { buffer in
            try vector { out in
                librosa_onset_strength(buffer.baseAddress,
                                       Int64(buffer.count),
                                       sampleRate,
                                       CInt(nFFT),
                                       CInt(hopLength),
                                       CInt(lag),
                                       CInt(maxSize),
                                       flag(detrend),
                                       flag(center),
                                       out)
            }
        }
    }

    public static func onsetStrength(spectrogram: LibrosaMatrix,
                                     sampleRate: Double = 22_050,
                                     nFFT: Int = 2048,
                                     hopLength: Int = 512,
                                     lag: Int = 1,
                                     maxSize: Int = 1,
                                     detrend: Bool = false,
                                     center: Bool = true) throws -> [Double] {
        try spectrogram.values.withUnsafeBufferPointer { buffer in
            try vector { out in
                librosa_onset_strength_spectrogram(buffer.baseAddress,
                                                   Int64(spectrogram.rows),
                                                   Int64(spectrogram.columns),
                                                   sampleRate,
                                                   CInt(nFFT),
                                                   CInt(hopLength),
                                                   CInt(lag),
                                                   CInt(maxSize),
                                                   flag(detrend),
                                                   flag(center),
                                                   out)
            }
        }
    }

    public static func onsetDetect(_ y: [Double],
                                   sampleRate: Double = 22_050,
                                   hopLength: Int = 512,
                                   backtrack: Bool = false,
                                   normalize: Bool = true) throws -> [Int] {
        try indices(from: y) { buffer, out in
            librosa_onset_detect(buffer.baseAddress,
                                 Int64(buffer.count),
                                 sampleRate,
                                 CInt(hopLength),
                                 flag(backtrack),
                                 flag(normalize),
                                 out)
        }
    }

    public static func onsetDetect(onsetEnvelope: [Double],
                                   sampleRate: Double = 22_050,
                                   hopLength: Int = 512,
                                   backtrack: Bool = false,
                                   normalize: Bool = true) throws -> [Int] {
        try indices(from: onsetEnvelope) { buffer, out in
            librosa_onset_detect_envelope(buffer.baseAddress,
                                          Int64(buffer.count),
                                          sampleRate,
                                          CInt(hopLength),
                                          flag(backtrack),
                                          flag(normalize),
                                          out)
        }
    }

    public static func tempo(onsetEnvelope: [Double],
                             sampleRate: Double = 22_050,
                             hopLength: Int = 512,
                             startBPM: Double = 120,
                             stdBPM: Double = 1,
                             acSize: Double = 8,
                             maxTempo: Double? = 320) throws -> Double {
        try onsetEnvelope.withUnsafeBufferPointer { buffer in
            var output = 0.0
            try check(librosa_tempo(buffer.baseAddress,
                                    Int64(buffer.count),
                                    sampleRate,
                                    CInt(hopLength),
                                    startBPM,
                                    stdBPM,
                                    acSize,
                                    flag(maxTempo != nil),
                                    maxTempo ?? 0,
                                    &output))
            return output
        }
    }

    public static func tempo(_ y: [Double],
                             sampleRate: Double = 22_050,
                             hopLength: Int = 512,
                             startBPM: Double = 120,
                             stdBPM: Double = 1,
                             acSize: Double = 8,
                             maxTempo: Double? = 320) throws -> Double {
        try y.withUnsafeBufferPointer { buffer in
            var output = 0.0
            try check(librosa_tempo_audio(buffer.baseAddress,
                                          Int64(buffer.count),
                                          sampleRate,
                                          CInt(hopLength),
                                          startBPM,
                                          stdBPM,
                                          acSize,
                                          flag(maxTempo != nil),
                                          maxTempo ?? 0,
                                          &output))
            return output
        }
    }

    public static func beatTrack(onsetEnvelope: [Double],
                                 sampleRate: Double = 22_050,
                                 hopLength: Int = 512,
                                 startBPM: Double = 120,
                                 tightness: Double = 100,
                                 trim: Bool = true,
                                 bpm: Double? = nil) throws -> LibrosaBeatTrackResult {
        try onsetEnvelope.withUnsafeBufferPointer { buffer in
            var raw = CLibrosa.LibrosaBeatTrackResult()
            try check(librosa_beat_track(buffer.baseAddress,
                                         Int64(buffer.count),
                                         sampleRate,
                                         CInt(hopLength),
                                         startBPM,
                                         tightness,
                                         flag(trim),
                                         flag(bpm != nil),
                                         bpm ?? 0,
                                         &raw))
            defer { librosa_beat_track_result_free(&raw) }
            return LibrosaBeatTrackResult(
                tempo: raw.tempo,
                beats: ints(raw.beats.data, count: Int(raw.beats.count))
            )
        }
    }

    public static func beatTrack(_ y: [Double],
                                 sampleRate: Double = 22_050,
                                 hopLength: Int = 512,
                                 startBPM: Double = 120,
                                 tightness: Double = 100,
                                 trim: Bool = true,
                                 bpm: Double? = nil) throws -> LibrosaBeatTrackResult {
        try y.withUnsafeBufferPointer { buffer in
            var raw = CLibrosa.LibrosaBeatTrackResult()
            try check(librosa_beat_track_audio(buffer.baseAddress,
                                               Int64(buffer.count),
                                               sampleRate,
                                               CInt(hopLength),
                                               startBPM,
                                               tightness,
                                               flag(trim),
                                               flag(bpm != nil),
                                               bpm ?? 0,
                                               &raw))
            defer { librosa_beat_track_result_free(&raw) }
            return LibrosaBeatTrackResult(
                tempo: raw.tempo,
                beats: ints(raw.beats.data, count: Int(raw.beats.count))
            )
        }
    }

    public static func timeStretch(_ y: [Double],
                                   rate: Double,
                                   nFFT: Int = 2048,
                                   hopLength: Int = 512) throws -> [Double] {
        try y.withUnsafeBufferPointer { buffer in
            try vector { out in
                librosa_time_stretch(buffer.baseAddress,
                                     Int64(buffer.count),
                                     rate,
                                     CInt(nFFT),
                                     CInt(hopLength),
                                     out)
            }
        }
    }

    public static func pitchShift(_ y: [Double],
                                  sampleRate: Double,
                                  steps: Double,
                                  binsPerOctave: Int = 12,
                                  resType: String = "kaiser_hq",
                                  nFFT: Int = 2048,
                                  hopLength: Int = 512) throws -> [Double] {
        try y.withUnsafeBufferPointer { buffer in
            try resType.withCString { cResType in
                try vector { out in
                    librosa_pitch_shift(buffer.baseAddress,
                                        Int64(buffer.count),
                                        sampleRate,
                                        steps,
                                        CInt(binsPerOctave),
                                        cResType,
                                        CInt(nFFT),
                                        CInt(hopLength),
                                        out)
                }
            }
        }
    }

    public static func trim(_ y: [Double],
                            topDB: Double = 60,
                            frameLength: Int = 2048,
                            hopLength: Int = 512) throws -> LibrosaTrimResult {
        try y.withUnsafeBufferPointer { buffer in
            var raw = CLibrosa.LibrosaTrimResult()
            try check(librosa_trim(buffer.baseAddress,
                                   Int64(buffer.count),
                                   topDB,
                                   CInt(frameLength),
                                   CInt(hopLength),
                                   &raw))
            defer { librosa_trim_result_free(&raw) }
            return LibrosaTrimResult(
                audio: doubles(raw.audio.data, count: Int(raw.audio.count)),
                interval: Int(raw.start)..<Int(raw.end)
            )
        }
    }
}

private extension Librosa {
    static func flag(_ value: Bool) -> CInt {
        value ? 1 : 0
    }

    static func check(_ status: CInt) throws {
        guard status == LIBROSA_STATUS_OK else {
            let message = String(cString: librosa_last_error_message())
            throw LibrosaError.operationFailed(message)
        }
    }

    static func scalar(_ input: Double,
                       _ function: (Double, UnsafeMutablePointer<Double>?) -> CInt) throws -> Double {
        var output = 0.0
        try check(function(input, &output))
        return output
    }

    static func vector(_ body: (UnsafeMutablePointer<CLibrosa.LibrosaVector>?) -> CInt) throws -> [Double] {
        var raw = CLibrosa.LibrosaVector()
        try check(body(&raw))
        defer { librosa_vector_free(&raw) }
        return doubles(raw.data, count: Int(raw.count))
    }

    static func matrix(from y: [Double],
                       _ body: (UnsafeBufferPointer<Double>, UnsafeMutablePointer<CLibrosa.LibrosaMatrix>?) -> CInt) throws -> LibrosaMatrix {
        try y.withUnsafeBufferPointer { buffer in
            var raw = CLibrosa.LibrosaMatrix()
            try check(body(buffer, &raw))
            defer { librosa_matrix_free(&raw) }
            return LibrosaMatrix(
                rows: Int(raw.rows),
                columns: Int(raw.columns),
                values: doubles(raw.data, count: Int(raw.rows * raw.columns))
            )
        }
    }

    static func matrix(from matrix: LibrosaMatrix,
                       _ body: (UnsafeBufferPointer<Double>, UnsafeMutablePointer<CLibrosa.LibrosaMatrix>?) -> CInt) throws -> LibrosaMatrix {
        try matrix.values.withUnsafeBufferPointer { buffer in
            var raw = CLibrosa.LibrosaMatrix()
            try check(body(buffer, &raw))
            defer { librosa_matrix_free(&raw) }
            return LibrosaMatrix(
                rows: Int(raw.rows),
                columns: Int(raw.columns),
                values: doubles(raw.data, count: Int(raw.rows * raw.columns))
            )
        }
    }

    static func matrixPair(from matrix: LibrosaMatrix,
                           _ body: (UnsafeBufferPointer<Double>, UnsafeMutablePointer<CLibrosa.LibrosaMatrix>?, UnsafeMutablePointer<CLibrosa.LibrosaMatrix>?) -> CInt) throws -> LibrosaHPSSResult {
        try matrix.values.withUnsafeBufferPointer { buffer in
            var harmonic = CLibrosa.LibrosaMatrix()
            var percussive = CLibrosa.LibrosaMatrix()
            try check(body(buffer, &harmonic, &percussive))
            defer {
                librosa_matrix_free(&harmonic)
                librosa_matrix_free(&percussive)
            }
            return LibrosaHPSSResult(
                harmonic: LibrosaMatrix(
                    rows: Int(harmonic.rows),
                    columns: Int(harmonic.columns),
                    values: doubles(harmonic.data, count: Int(harmonic.rows * harmonic.columns))
                ),
                percussive: LibrosaMatrix(
                    rows: Int(percussive.rows),
                    columns: Int(percussive.columns),
                    values: doubles(percussive.data, count: Int(percussive.rows * percussive.columns))
                )
            )
        }
    }

    static func indices(from y: [Double],
                        _ body: (UnsafeBufferPointer<Double>, UnsafeMutablePointer<CLibrosa.LibrosaIndexVector>?) -> CInt) throws -> [Int] {
        try y.withUnsafeBufferPointer { buffer in
            var raw = CLibrosa.LibrosaIndexVector()
            try check(body(buffer, &raw))
            defer { librosa_index_vector_free(&raw) }
            return ints(raw.data, count: Int(raw.count))
        }
    }

    static func doubles(_ data: UnsafeMutablePointer<Double>?, count: Int) -> [Double] {
        guard count > 0, let data else {
            return []
        }
        return Array(UnsafeBufferPointer(start: data, count: count))
    }

    static func ints(_ data: UnsafeMutablePointer<Int64>?, count: Int) -> [Int] {
        guard count > 0, let data else {
            return []
        }
        return UnsafeBufferPointer(start: data, count: count).map(Int.init)
    }
}
