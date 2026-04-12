import createModule from "../wasm/librosa_wasm.mjs";
import type { CreateLibrosaOptions, Librosa } from "./types.js";

export type * from "./types.js";

const optionalOptionsArity: Record<string, number> = {
  framesToSamples: 1,
  samplesToFrames: 1,
  framesToTime: 1,
  timeToFrames: 1,
  timeToSamples: 1,
  samplesToTime: 1,
  noteToMidi: 1,
  midiToNote: 1,
  hzToNote: 1,
  hzToMel: 1,
  melToHz: 1,
  hzToOcts: 1,
  octsToHz: 1,
  a4ToTuning: 1,
  tuningToA4: 1,
  fftFrequencies: 0,
  cqtFrequencies: 0,
  melFrequencies: 0,
  tempoFrequencies: 0,
  fourierTempoFrequencies: 0,
  weighting: 1,
  resample: 1,
  autocorrelate: 1,
  lpc: 1,
  zeroCrossings: 1,
  clicks: 1,
  clicksFrames: 1,
  tone: 1,
  chirp: 0,
  muCompress: 1,
  muExpand: 1,
  duration: 1,
  stft: 1,
  istft: 1,
  magphase: 1,
  powerToDb: 1,
  powerToDbScalar: 1,
  dbToPower: 1,
  amplitudeToDb: 1,
  dbToAmplitude: 1,
  perceptualWeighting: 2,
  phaseVocoder: 1,
  griffinlim: 1,
  pcen: 1,
  getWindow: 1,
  windowSumsquare: 1,
  cqt: 1,
  vqt: 1,
  pseudoCqt: 1,
  icqt: 1,
  griffinlimCqt: 1,
  melspectrogram: 1,
  mfcc: 1,
  chromaStft: 1,
  chromaCqt: 1,
  chromaCens: 1,
  chromaVqt: 1,
  spectralCentroid: 1,
  spectralBandwidth: 1,
  spectralRolloff: 1,
  spectralFlatness: 1,
  spectralContrast: 1,
  rms: 1,
  zeroCrossingRate: 1,
  polyFeatures: 1,
  tonnetz: 1,
  delta: 1,
  stackMemory: 1,
  fourierTempogram: 1,
  tempogramRatio: 1,
  melToStft: 1,
  melToAudio: 1,
  mfccToMel: 1,
  melFilter: 0,
  chromaFilter: 0,
  windowBandwidth: 1,
  cqToChroma: 0,
  waveletLengths: 1,
  wavelet: 1,
  diagonalFilter: 0,
  mrFrequencies: 0,
  pitchTuning: 1,
  estimateTuning: 1,
  piptrack: 1,
  yin: 1,
  pyin: 1,
  tempogram: 1,
  tempo: 1,
  tempoFrames: 1,
  beatTrack: 1,
  beatTrackTimes: 1,
  maximumFilter1d: 1,
  matchEvents: 2,
  onsetStrength: 1,
  onsetStrengthMulti: 1,
  onsetDetect: 1,
  onsetDetectTimes: 1,
  timeStretch: 1,
  pitchShift: 1,
  trim: 1,
  split: 1,
  preemphasis: 1,
  deemphasis: 1,
  remix: 2,
  harmonicEffect: 1,
  percussiveEffect: 1,
  medianFilter2d: 1,
  hpss: 1,
  decomposeNmf: 1,
  transitionUniform: 0,
  transitionLoop: 1,
  transitionCycle: 1,
  transitionLocal: 0,
  viterbi: 2,
  viterbiDiscriminative: 2,
  viterbiBinary: 2,
  dtw: 2,
  rqa: 1,
  recurrenceToLag: 1,
  lagToRecurrence: 1,
  crossSimilarity: 2,
  recurrenceMatrix: 1,
  pathEnhance: 1,
  agglomerative: 1,
  subsegment: 2,
  interpHarmonics: 3,
  f0Harmonics: 4,
  salience: 3,
  frame: 1,
  padCenter: 1,
  fixLength: 1,
  fixFrames: 1,
  normalize: 1,
  localmax: 1,
  localmin: 1,
  peakPick: 1,
  softmask: 2,
  phasor: 1,
  fillOffDiagonal: 1,
  stack: 1,
  validInt: 1,
  pythagoreanIntervals: 0,
  plimitIntervals: 1,
  intervalFrequencies: 1,
  melaToSvara: 1,
  keyToNotes: 1,
  fifthsToNote: 2,
  intervalToFjs: 1,
  midiToSvaraH: 2,
  midiToSvaraC: 3,
  hzToFjs: 1
};

export async function createLibrosa(options: CreateLibrosaOptions = {}): Promise<Librosa> {
  const moduleOptions: Record<string, unknown> = {
    ...(options.moduleOptions ?? {})
  };

  if (options.locateFile) {
    moduleOptions.locateFile = options.locateFile;
  }

  return wrapLibrosa((await createModule(moduleOptions)) as Librosa);
}

export default createLibrosa;

function wrapLibrosa(module: Librosa): Librosa {
  return new Proxy(module, {
    get(target, property, receiver) {
      const value = Reflect.get(target, property, receiver);
      if (typeof property !== "string" || typeof value !== "function") {
        return value;
      }

      const requiredArgs = optionalOptionsArity[property];
      if (requiredArgs === undefined) {
        return value.bind(target);
      }

      return (...args: unknown[]) => {
        if (args.length === requiredArgs) {
          return value.call(target, ...args, {});
        }
        return value.call(target, ...args);
      };
    }
  }) as Librosa;
}
