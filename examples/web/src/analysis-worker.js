import { createLibrosa } from "../../../packages/librosa-wasm/dist/src/index.js";

let librosa = null;

const runners = {
  presets: runPresets,
  superflux: runSuperflux,
  hpss: runHpss,
  vocal: runVocalSeparation,
  sync: runMusicSync,
  chroma: runEnhancedChroma,
  segmentation: runSegmentation
};

self.addEventListener("message", async (event) => {
  const { id, type, payload = {} } = event.data ?? {};
  try {
    let result;
    if (type === "init") {
      librosa = await createLibrosa();
      result = librosa.buildInfo();
    } else if (type === "prepareSource") {
      result = prepareSource(payload);
    } else if (type === "run") {
      if (!librosa) throw new Error("WASM runtime is not ready");
      const runner = runners[payload.exampleId];
      if (!runner) throw new Error(`Unknown example: ${payload.exampleId}`);
      result = await runner(librosa, payload.source);
    } else {
      throw new Error(`Unknown worker request: ${type}`);
    }
    self.postMessage({ id, ok: true, payload: result });
  } catch (error) {
    self.postMessage({
      id,
      ok: false,
      error: error instanceof Error ? error.message : String(error)
    });
  }
});

function prepareSource({ y, sr, targetSr, maxSeconds }) {
  let audio = y;
  let outputSr = sr;
  if (outputSr !== targetSr) {
    audio = librosa.resample(audio, {
      origSr: outputSr,
      targetSr,
      resType: "linear"
    });
    outputSr = targetSr;
  }
  audio = normalizeAudio(firstSeconds(audio, outputSr, maxSeconds));
  return {
    y: audio,
    sr: outputSr,
    samples: audio.length
  };
}

function runStep(label, fn) {
  try {
    return fn();
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`${label} failed: ${detail}`);
  }
}

async function runPresets(librosa, source) {
  const clip = firstSeconds(source.y, source.sr, 5);
  const y441 = ensureSampleRate(librosa, clip, source.sr, 44100);
  const y11025 = ensureSampleRate(librosa, y441, 44100, 11025);

  const base = librosa.melspectrogram(y441, {
    sr: 44100,
    nFft: 4096,
    hopLength: 1024,
    nMels: 80
  });
  const highres = librosa.melspectrogram(y441, {
    sr: 44100,
    nFft: 4096,
    hopLength: 512,
    nMels: 80
  });
  const low = librosa.melspectrogram(y11025, {
    sr: 11025,
    nFft: 4096,
    hopLength: 1024,
    nMels: 80
  });

  return {
    metrics: [
      { label: "44100 / 1024", value: shape(base) },
      { label: "44100 / 512", value: shape(highres) },
      { label: "11025 / 1024", value: shape(low) },
      { label: "n_fft", value: "4096" }
    ],
    plots: [
      heatmapPlot("44100 / 1024 / 4096", "mel dB", relativeDb(librosa.powerToDb(base, { topDb: null }))),
      heatmapPlot("44100 / 512 / 4096", "mel dB", relativeDb(librosa.powerToDb(highres, { topDb: null }))),
      heatmapPlot("11025 / 1024 / 4096", "mel dB", relativeDb(librosa.powerToDb(low, { topDb: null })), { wide: true })
    ]
  };
}

async function runSuperflux(librosa, source) {
  const y = firstSeconds(source.y, source.sr, 7);
  const sr = source.sr;
  const nFft = 1024;
  const hopLength = Math.max(64, Math.round(sr / 200));
  const fmax = Math.min(8000, sr / 2 - 100);
  const mel = runStep("Superflux mel spectrogram", () => librosa.melspectrogram(y, {
    sr,
    nFft,
    hopLength,
    fmin: 27.5,
    fmax,
    nMels: 96
  }));
  const melDb = relativeDb(librosa.powerToDb(mel, { topDb: null }));
  const spectralFlux = runStep("Spectral flux", () => librosa.onsetStrength(y, { sr, nFft, hopLength }));
  const defaultEvents = runStep("Default onset detection", () => librosa.onsetDetect(spectralFlux, {
    envelope: true,
    sr,
    hopLength,
    preMax: 3,
    postMax: 3,
    preAvg: 3,
    postAvg: 5,
    delta: 0.05,
    wait: 4
  }));
  const superflux = runStep("Superflux onset strength", () => librosa.onsetStrength(melDb, {
    sr,
    nFft,
    hopLength,
    lag: 2,
    maxSize: 3
  }));
  const superEvents = runStep("Superflux onset detection", () => librosa.onsetDetect(superflux, {
    envelope: true,
    sr,
    hopLength,
    preMax: 3,
    postMax: 3,
    preAvg: 3,
    postAvg: 5,
    delta: 0.05,
    wait: 4
  }));

  return {
    metrics: [
      { label: "Mel spectrogram", value: shape(mel) },
      { label: "Default onsets", value: `${defaultEvents.length}` },
      { label: "Superflux onsets", value: `${superEvents.length}` },
      { label: "Hop", value: `${hopLength}` }
    ],
    plots: [
      heatmapPlot("Mel spectrogram", "power to dB", melDb, { wide: true }),
      linePlot("Onset envelopes", "spectral flux vs Superflux", [
        { label: "Spectral flux", values: spectralFlux, color: "#f39a2f" },
        { label: "Superflux", values: superflux, color: "#d13d8c" }
      ], {
        wide: true,
        markers: [
          ...Array.from(defaultEvents, (frame) => ({ x: (frame * hopLength) / sr, maxX: y.length / sr, color: "rgba(243, 154, 47, 0.45)" })),
          ...Array.from(superEvents, (frame) => ({ x: (frame * hopLength) / sr, maxX: y.length / sr, color: "rgba(209, 61, 140, 0.55)" }))
        ]
      })
    ]
  };
}

async function runHpss(librosa, source) {
  const y = firstSeconds(source.y, source.sr, 10);
  const nFft = 2048;
  const hopLength = 512;
  const d = librosa.stft(y, { nFft, hopLength });
  const mag = librosa.magnitude(d);
  const base = librosa.hpss(mag, { kernelSize: 31, margin: 1 });
  const margin8 = librosa.hpss(mag, { kernelSize: 31, margin: 8 });

  return {
    metrics: [
      { label: "STFT", value: shape(mag) },
      { label: "Margin 1 harmonic", value: `${percentEnergy(base.harmonic, mag)}%` },
      { label: "Margin 1 percussive", value: `${percentEnergy(base.percussive, mag)}%` },
      { label: "Margin 8 harmonic", value: `${percentEnergy(margin8.harmonic, mag)}%` }
    ],
    plots: [
      heatmapPlot("Full spectrogram", "amplitude dB", relativeDb(librosa.amplitudeToDb(mag, { topDb: null }))),
      heatmapPlot("Harmonic spectrogram", "margin 1", relativeDb(librosa.amplitudeToDb(base.harmonic, { topDb: null }))),
      heatmapPlot("Percussive spectrogram", "margin 1", relativeDb(librosa.amplitudeToDb(base.percussive, { topDb: null }))),
      heatmapPlot("Harmonic vs percussive", "margin 8", matrixDifference(
        relativeDb(librosa.amplitudeToDb(margin8.harmonic, { topDb: null })),
        relativeDb(librosa.amplitudeToDb(margin8.percussive, { topDb: null }))
      ))
    ]
  };
}

async function runVocalSeparation(librosa, source) {
  const y = firstSeconds(source.y, source.sr, 12);
  const sr = source.sr;
  const nFft = 1024;
  const hopLength = 256;
  const d = librosa.stft(y, { nFft, hopLength });
  const full = librosa.magnitude(d);
  const width = Math.max(2, Math.round(sr / hopLength));
  const filtered = minimumMatrix(full, nnFilterMedian(full, {
    width,
    neighbors: Math.min(12, Math.max(2, Math.floor(full.cols / 4)))
  }));
  const foregroundResidual = clampMatrix(subtractMatrix(full, filtered), 0);
  const maskI = librosa.softmask(filtered, scaleMatrix(foregroundResidual, 2), { power: 2 });
  const maskV = librosa.softmask(foregroundResidual, scaleMatrix(filtered, 10), { power: 2 });
  const background = multiplyMatrix(maskI, full);
  const foreground = multiplyMatrix(maskV, full);

  return {
    metrics: [
      { label: "Spectrum", value: shape(full) },
      { label: "Filter width", value: `${width} frames` },
      { label: "Foreground energy", value: `${percentEnergy(foreground, full)}%` },
      { label: "Background energy", value: `${percentEnergy(background, full)}%` }
    ],
    plots: [
      heatmapPlot("Input spectrum", "amplitude dB", relativeDb(librosa.amplitudeToDb(full, { topDb: null }))),
      heatmapPlot("Nearest-neighbor filter", "median aggregate", relativeDb(librosa.amplitudeToDb(filtered, { topDb: null }))),
      heatmapPlot("Foreground mask", "sporadic component", maskV),
      heatmapPlot("Foreground spectrum", "masked magnitude", relativeDb(librosa.amplitudeToDb(foreground, { topDb: null })))
    ]
  };
}

async function runMusicSync(librosa, source) {
  const sr = source.sr;
  const slow = firstSeconds(source.y, sr, 7);
  const fast = librosa.timeStretch(slow, { rate: 1.32, nFft: 1024, hopLength: 256 });
  const hopLength = 512;
  const slowChroma = librosa.chromaStft(slow, {
    sr,
    nFft: 2048,
    hopLength,
    tuning: 0,
    norm: 2
  });
  const fastChroma = librosa.chromaStft(fast, {
    sr,
    nFft: 2048,
    hopLength,
    tuning: 0,
    norm: 2
  });
  const alignment = librosa.dtw(slowChroma, fastChroma, { backtrack: true });

  return {
    metrics: [
      { label: "Slow chroma", value: shape(slowChroma) },
      { label: "Fast chroma", value: shape(fastChroma) },
      { label: "DTW cost", value: shape(alignment.cost) },
      { label: "Path pairs", value: `${alignment.path.rows}` }
    ],
    plots: [
      heatmapPlot("Slow chroma", "source timing", slowChroma),
      heatmapPlot("Fast chroma", "time-stretched", fastChroma),
      dtwPlot("Accumulated cost", "warping path", alignment.cost, alignment.path),
      alignmentPlot("Time-domain alignment", "path correspondences", slow, fast, sr, hopLength, alignment.path)
    ]
  };
}

async function runEnhancedChroma(librosa, source) {
  const y = firstSeconds(source.y, source.sr, 8);
  const sr = source.sr;
  const hopLength = 512;
  const orig = librosa.chromaCqt(y, { sr, hopLength, nOctaves: 5 });
  const over = librosa.chromaCqt(y, {
    sr,
    hopLength,
    binsPerOctave: 36,
    nOctaves: 5
  });
  const harmonic = librosa.harmonicEffect(y, {
    margin: 8,
    nFft: 2048,
    hopLength
  });
  const harm = librosa.chromaCqt(harmonic, {
    sr,
    hopLength,
    binsPerOctave: 36,
    nOctaves: 5
  });
  const filtered = minimumMatrix(harm, nnFilterMedian(harm, {
    width: 2,
    neighbors: Math.min(8, Math.max(2, Math.floor(harm.cols / 4)))
  }));
  const smooth = librosa.medianFilter2d(filtered, { size: [1, 9], mode: "edge" });

  return {
    metrics: [
      { label: "Original chroma", value: shape(orig) },
      { label: "3x oversampled", value: shape(over) },
      { label: "Harmonic audio", value: `${formatSeconds(harmonic.length / sr)} s` },
      { label: "Median width", value: "9 frames" }
    ],
    plots: [
      heatmapPlot("Original chroma", "chroma CQT", orig),
      heatmapPlot("3x oversampled", "36 bins/octave", over),
      heatmapPlot("Harmonic chroma", "margin 8", harm),
      heatmapPlot("Smoothed chroma", "non-local + median", smooth)
    ]
  };
}

async function runSegmentation(librosa, source) {
  const y = firstSeconds(source.y, source.sr, 12);
  const sr = source.sr;
  const hopLength = 512;
  const chroma = librosa.chromaStft(y, {
    sr,
    nFft: 2048,
    hopLength,
    norm: 2
  });
  const envelope = librosa.onsetStrength(y, {
    sr,
    nFft: 2048,
    hopLength
  });
  const beatResult = librosa.beatTrack(envelope, {
    sr,
    hopLength,
    startBpm: 112,
    trim: false
  });
  const beatFrames = robustBeatFrames(librosa, beatResult.beats, chroma.cols);
  const sync = syncMedian(chroma, beatFrames);
  const recurrence = librosa.recurrenceMatrix(sync, {
    k: Math.min(4, Math.max(1, sync.cols - 1)),
    width: Math.min(2, Math.max(1, sync.cols - 1)),
    mode: "affinity",
    sym: true,
    self: true
  });
  const enhanced = librosa.pathEnhance(recurrence, {
    n: 5,
    nFilters: 5,
    clip: true
  });
  const boundaries = Array.from(librosa.agglomerative(sync, {
    k: Math.min(5, Math.max(1, sync.cols))
  })).filter((value) => Number.isFinite(value));
  const frameBoundaries = uniqueSorted([
    0,
    ...boundaries.map((value) => beatFrames[Math.max(0, Math.min(beatFrames.length - 1, Math.round(value)))] ?? 0),
    chroma.cols
  ]);

  return {
    metrics: [
      { label: "Tempo", value: `${formatNumber(beatResult.tempo)} BPM` },
      { label: "Beat frames", value: `${beatFrames.length}` },
      { label: "Sync chroma", value: shape(sync) },
      { label: "Segments", value: `${Math.max(1, frameBoundaries.length - 1)}` }
    ],
    plots: [
      heatmapPlot("Beat-synchronous chroma", "median per beat span", sync, { wide: true }),
      heatmapPlot("Recurrence similarity", "affinity graph", recurrence),
      heatmapPlot("Path-enhanced graph", "diagonal continuity", enhanced),
      segmentPlot("Segment boundaries", "agglomerative over beat features", y.length / sr, frameBoundaries, sr, hopLength, { wide: true, compact: true })
    ]
  };
}

function heatmapPlot(title, meta, matrix, options = {}) {
  return {
    type: "heatmap",
    title,
    meta,
    matrix,
    options,
    ...options
  };
}

function linePlot(title, meta, series, options = {}) {
  return {
    type: "line",
    title,
    meta,
    series,
    options,
    ...options
  };
}

function dtwPlot(title, meta, cost, path) {
  return {
    type: "dtw",
    title,
    meta,
    cost,
    path
  };
}

function alignmentPlot(title, meta, slow, fast, sr, hopLength, path) {
  return {
    type: "alignment",
    title,
    meta,
    slow,
    fast,
    sr,
    hopLength,
    path
  };
}

function segmentPlot(title, meta, duration, frames, sr, hopLength, options = {}) {
  return {
    type: "segments",
    title,
    meta,
    duration,
    frames,
    sr,
    hopLength,
    ...options
  };
}

function normalizeAudio(input) {
  let peak = 0;
  for (const value of input) peak = Math.max(peak, Math.abs(value));
  if (!peak) return new Float64Array(input);
  const scale = 0.92 / peak;
  const out = new Float64Array(input.length);
  for (let i = 0; i < input.length; i++) out[i] = input[i] * scale;
  return out;
}

function ensureSampleRate(librosa, y, origSr, targetSr) {
  if (origSr === targetSr) return y;
  return librosa.resample(y, {
    origSr,
    targetSr,
    resType: "linear"
  });
}

function firstSeconds(y, sr, seconds) {
  return y.slice(0, Math.min(y.length, Math.max(1, Math.floor(sr * seconds))));
}

function relativeDb(matrix, floor = -80) {
  let max = -Infinity;
  for (const value of matrix.data) {
    if (Number.isFinite(value) && value > max) max = value;
  }
  const data = new Float64Array(matrix.data.length);
  for (let i = 0; i < matrix.data.length; i++) {
    data[i] = Math.max(floor, matrix.data[i] - max);
  }
  return { rows: matrix.rows, cols: matrix.cols, data };
}

function mapMatrix(matrix, fn) {
  const data = new Float64Array(matrix.data.length);
  for (let i = 0; i < matrix.data.length; i++) data[i] = fn(matrix.data[i], i);
  return { rows: matrix.rows, cols: matrix.cols, data };
}

function scaleMatrix(matrix, scalar) {
  return mapMatrix(matrix, (value) => value * scalar);
}

function clampMatrix(matrix, min) {
  return mapMatrix(matrix, (value) => Math.max(min, value));
}

function subtractMatrix(a, b) {
  return combineMatrices(a, b, (x, y) => x - y);
}

function multiplyMatrix(a, b) {
  return combineMatrices(a, b, (x, y) => x * y);
}

function minimumMatrix(a, b) {
  return combineMatrices(a, b, Math.min);
}

function matrixDifference(a, b) {
  return combineMatrices(a, b, (x, y) => x - y);
}

function combineMatrices(a, b, fn) {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error(`Matrix shape mismatch: ${shape(a)} vs ${shape(b)}`);
  }
  const data = new Float64Array(a.data.length);
  for (let i = 0; i < data.length; i++) data[i] = fn(a.data[i], b.data[i]);
  return { rows: a.rows, cols: a.cols, data };
}

function nnFilterMedian(matrix, options = {}) {
  const rows = matrix.rows;
  const cols = matrix.cols;
  const width = options.width ?? 1;
  const neighbors = Math.min(options.neighbors ?? 8, Math.max(1, cols - 1));
  const norms = new Float64Array(cols);
  for (let col = 0; col < cols; col++) {
    let sum = 0;
    for (let row = 0; row < rows; row++) {
      const value = matrix.data[row * cols + col];
      sum += value * value;
    }
    norms[col] = Math.sqrt(sum) || 1;
  }

  const out = new Float64Array(rows * cols);
  const scratch = new Float64Array(Math.max(1, neighbors));
  const scores = new Array(cols);

  for (let col = 0; col < cols; col++) {
    scores.length = 0;
    for (let other = 0; other < cols; other++) {
      if (Math.abs(other - col) <= width && cols > width * 2 + 1) continue;
      let dot = 0;
      for (let row = 0; row < rows; row++) {
        dot += matrix.data[row * cols + col] * matrix.data[row * cols + other];
      }
      scores.push({ col: other, score: dot / (norms[col] * norms[other]) });
    }
    scores.sort((a, b) => b.score - a.score);
    const chosen = scores.slice(0, neighbors);
    if (!chosen.length) chosen.push({ col, score: 1 });

    for (let row = 0; row < rows; row++) {
      for (let k = 0; k < chosen.length; k++) {
        scratch[k] = matrix.data[row * cols + chosen[k].col];
      }
      out[row * cols + col] = medianInPlace(scratch.subarray(0, chosen.length));
    }
  }

  return { rows, cols, data: out };
}

function syncMedian(matrix, boundaries) {
  const frames = uniqueSorted(Array.from(boundaries, (value) => Math.round(value)))
    .filter((value) => value >= 0 && value <= matrix.cols);
  if (frames[0] !== 0) frames.unshift(0);
  if (frames[frames.length - 1] !== matrix.cols) frames.push(matrix.cols);
  const cols = Math.max(1, frames.length - 1);
  const out = new Float64Array(matrix.rows * cols);
  const scratch = new Float64Array(matrix.cols);

  for (let segment = 0; segment < cols; segment++) {
    const start = frames[segment];
    const end = Math.max(start + 1, frames[segment + 1]);
    for (let row = 0; row < matrix.rows; row++) {
      let count = 0;
      for (let col = start; col < Math.min(end, matrix.cols); col++) {
        scratch[count++] = matrix.data[row * matrix.cols + col];
      }
      out[row * cols + segment] = count ? medianInPlace(scratch.subarray(0, count)) : 0;
    }
  }
  return { rows: matrix.rows, cols, data: out };
}

function robustBeatFrames(librosa, beats, maxFrame) {
  const cleaned = uniqueSorted(Array.from(beats, (value) => Math.round(value)))
    .filter((value) => value >= 0 && value <= maxFrame);
  let frames = cleaned;
  if (frames.length < 4) {
    const step = Math.max(3, Math.floor(maxFrame / 8));
    frames = [];
    for (let frame = 0; frame <= maxFrame; frame += step) frames.push(frame);
  }
  return Array.from(librosa.fixFrames(new Float64Array(frames), {
    xMin: 0,
    xMax: maxFrame
  }));
}

function medianInPlace(values) {
  const sorted = Array.from(values).sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function percentEnergy(part, full) {
  let a = 0;
  let b = 0;
  for (let i = 0; i < full.data.length; i++) {
    a += Math.abs(part.data[i]);
    b += Math.abs(full.data[i]);
  }
  return formatNumber((a / Math.max(1e-12, b)) * 100);
}

function shape(matrix) {
  return `${matrix.rows} x ${matrix.cols}`;
}

function formatSeconds(value) {
  return `${formatNumber(value)}s`;
}

function formatNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return "--";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits
  }).format(value);
}

function uniqueSorted(values) {
  return Array.from(new Set(values.filter(Number.isFinite))).sort((a, b) => a - b);
}
