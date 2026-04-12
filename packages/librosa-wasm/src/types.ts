export type VectorInput = ArrayLike<number>;
export type StringVectorInput = ArrayLike<string>;

export interface Matrix {
  rows: number;
  cols: number;
  data: Float64Array;
}

export interface ComplexMatrix {
  rows: number;
  cols: number;
  data: Float64Array;
}

export interface BoolMatrix {
  rows: number;
  cols: number;
  data: Uint8Array;
}

export interface BuildInfo {
  fftBackend: "pffft";
  wasmSimd: boolean;
  pffftSimdSize: number;
  pffftSimdArch: string;
}

export interface PairMatrixResult<T = Matrix> {
  harmonic: T;
  percussive: T;
}

export interface MagphaseResult {
  magnitude: Matrix;
  phase: ComplexMatrix;
}

export interface PiptrackResult {
  pitches: Matrix;
  magnitudes: Matrix;
}

export interface PyinResult {
  f0: Float64Array;
  voicedFlag: Uint8Array;
  voicedProb: Float64Array;
}

export interface BeatTrackResult {
  tempo: number;
  beats: Float64Array;
}

export interface TrimResult {
  audio: Float64Array;
  index: Float64Array;
}

export interface NmfResult {
  components: Matrix;
  activations: Matrix;
}

export interface DtwBacktrackResult {
  cost: Matrix;
  path: Matrix;
}

export interface RqaBacktrackResult {
  score: Matrix;
  path: Matrix;
}

export interface WaveletLengthsResult {
  lengths: Float64Array;
  cutoff: number;
}

export interface WaveletResult {
  filters: ComplexMatrix;
  lengths: Float64Array;
}

export interface MrFrequenciesResult {
  frequencies: Float64Array;
  sampleRates: Float64Array;
}

export interface CreateLibrosaOptions {
  locateFile?: (path: string, prefix: string) => string;
  moduleOptions?: Record<string, unknown>;
}

export type NumericArrayResult = number | Float64Array;
export type StringArrayResult = string | string[];

export interface Librosa {
  buildInfo(): BuildInfo;

  framesToSamples(frames: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;
  samplesToFrames(samples: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;
  framesToTime(frames: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;
  timeToFrames(times: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;
  timeToSamples(times: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;
  samplesToTime(samples: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;
  midiToHz(midi: number | VectorInput): NumericArrayResult;
  hzToMidi(frequency: number | VectorInput): NumericArrayResult;
  noteToMidi(note: string | StringVectorInput, options?: Record<string, unknown>): NumericArrayResult;
  noteToHz(note: string | StringVectorInput): NumericArrayResult;
  midiToNote(midi: number | VectorInput, options?: Record<string, unknown>): StringArrayResult;
  hzToNote(frequency: number | VectorInput, options?: Record<string, unknown>): StringArrayResult;
  hzToMel(frequency: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;
  melToHz(mel: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;
  hzToOcts(frequency: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;
  octsToHz(octs: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;
  a4ToTuning(a4: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;
  tuningToA4(tuning: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;
  fftFrequencies(options?: Record<string, unknown>): Float64Array;
  cqtFrequencies(options?: Record<string, unknown>): Float64Array;
  melFrequencies(options?: Record<string, unknown>): Float64Array;
  tempoFrequencies(options?: Record<string, unknown>): Float64Array;
  fourierTempoFrequencies(options?: Record<string, unknown>): Float64Array;
  weighting(frequency: number | VectorInput, options?: Record<string, unknown>): NumericArrayResult;

  toMono(y: VectorInput | Matrix): Float64Array;
  resample(y: VectorInput, options?: Record<string, unknown>): Float64Array;
  autocorrelate(y: VectorInput | Matrix, options?: Record<string, unknown>): Float64Array | Matrix;
  lpc(y: VectorInput | Matrix, options?: Record<string, unknown>): Float64Array | Matrix;
  zeroCrossings(y: VectorInput | Matrix, options?: Record<string, unknown>): Uint8Array | BoolMatrix;
  clicks(times: VectorInput, options?: Record<string, unknown>): Float64Array;
  clicksFrames(frames: VectorInput, options?: Record<string, unknown>): Float64Array;
  tone(frequency: number, options?: Record<string, unknown>): Float64Array;
  chirp(options?: Record<string, unknown>): Float64Array;
  muCompress(x: VectorInput, options?: Record<string, unknown>): Float64Array;
  muExpand(x: VectorInput, options?: Record<string, unknown>): Float64Array;
  duration(input: VectorInput | Matrix, options?: Record<string, unknown>): number;

  stft(y: VectorInput, options?: Record<string, unknown>): ComplexMatrix;
  istft(d: ComplexMatrix, options?: Record<string, unknown>): Float64Array;
  magphase(d: ComplexMatrix, options?: Record<string, unknown>): MagphaseResult;
  magnitude(d: ComplexMatrix): Matrix;
  phase(d: ComplexMatrix): Matrix;
  powerToDb(s: Matrix, options?: Record<string, unknown>): Matrix;
  powerToDbScalar(s: number, options?: Record<string, unknown>): number;
  dbToPower(s: number | Matrix, options?: Record<string, unknown>): number | Matrix;
  amplitudeToDb(s: number | Matrix, options?: Record<string, unknown>): number | Matrix;
  dbToAmplitude(s: number | Matrix, options?: Record<string, unknown>): number | Matrix;
  perceptualWeighting(s: Matrix, frequencies: VectorInput, options?: Record<string, unknown>): Matrix;
  phaseVocoder(d: ComplexMatrix, options?: Record<string, unknown>): ComplexMatrix;
  griffinlim(s: Matrix, options?: Record<string, unknown>): Float64Array;
  pcen(s: Matrix, options?: Record<string, unknown>): Matrix;
  getWindow(window: string, options?: Record<string, unknown>): Float64Array;
  windowSumsquare(window: VectorInput, options?: Record<string, unknown>): Float64Array;

  cqt(y: VectorInput, options?: Record<string, unknown>): ComplexMatrix;
  vqt(y: VectorInput, options?: Record<string, unknown>): ComplexMatrix;
  pseudoCqt(y: VectorInput, options?: Record<string, unknown>): Matrix;
  icqt(c: ComplexMatrix, options?: Record<string, unknown>): Float64Array;
  griffinlimCqt(c: Matrix, options?: Record<string, unknown>): Float64Array;

  melspectrogram(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  mfcc(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  chromaStft(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  chromaCqt(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  chromaCens(y: VectorInput, options?: Record<string, unknown>): Matrix;
  chromaVqt(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  spectralCentroid(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  spectralBandwidth(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  spectralRolloff(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  spectralFlatness(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  spectralContrast(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  rms(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  zeroCrossingRate(y: VectorInput, options?: Record<string, unknown>): Matrix;
  polyFeatures(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  tonnetz(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  delta(data: Matrix, options?: Record<string, unknown>): Matrix;
  stackMemory(data: Matrix, options?: Record<string, unknown>): Matrix;
  fourierTempogram(input: VectorInput, options?: Record<string, unknown>): ComplexMatrix;
  tempogramRatio(tg: Matrix, options?: Record<string, unknown>): Matrix;
  melToStft(m: Matrix, options?: Record<string, unknown>): Matrix;
  melToAudio(m: Matrix, options?: Record<string, unknown>): Float64Array;
  mfccToMel(m: Matrix, options?: Record<string, unknown>): Matrix;

  melFilter(options?: Record<string, unknown>): Matrix;
  chromaFilter(options?: Record<string, unknown>): Matrix;
  windowBandwidth(window: string, options?: Record<string, unknown>): number;
  cqToChroma(options?: Record<string, unknown>): Matrix;
  relativeBandwidth(freqs: VectorInput): Float64Array;
  waveletLengths(freqs: VectorInput, options?: Record<string, unknown>): WaveletLengthsResult;
  wavelet(freqs: VectorInput, options?: Record<string, unknown>): WaveletResult;
  diagonalFilter(options?: Record<string, unknown>): Matrix;
  mrFrequencies(options?: Record<string, unknown>): MrFrequenciesResult;

  pitchTuning(frequencies: VectorInput, options?: Record<string, unknown>): number;
  estimateTuning(input: VectorInput | Matrix, options?: Record<string, unknown>): number;
  piptrack(input: VectorInput | Matrix, options?: Record<string, unknown>): PiptrackResult;
  yin(y: VectorInput, options?: Record<string, unknown>): Float64Array;
  pyin(y: VectorInput, options?: Record<string, unknown>): PyinResult;

  tempogram(input: VectorInput, options?: Record<string, unknown>): Matrix;
  tempo(input: VectorInput, options?: Record<string, unknown>): number;
  tempoFrames(envelope: VectorInput, options?: Record<string, unknown>): Float64Array;
  beatTrack(input: VectorInput, options?: Record<string, unknown>): BeatTrackResult;
  beatTrackTimes(y: VectorInput, options?: Record<string, unknown>): BeatTrackResult;
  maximumFilter1d(s: Matrix, options?: Record<string, unknown>): Matrix;
  matchEvents(eventsFrom: VectorInput, eventsTo: VectorInput, options?: Record<string, unknown>): Float64Array;
  onsetStrength(input: VectorInput | Matrix, options?: Record<string, unknown>): Float64Array;
  onsetStrengthMulti(input: VectorInput | Matrix, options?: Record<string, unknown>): Matrix;
  onsetDetect(input: VectorInput, options?: Record<string, unknown>): Float64Array;
  onsetDetectTimes(y: VectorInput, options?: Record<string, unknown>): Float64Array;
  onsetBacktrack(events: VectorInput, energy: VectorInput): Float64Array;

  timeStretch(y: VectorInput, options?: Record<string, unknown>): Float64Array;
  pitchShift(y: VectorInput, options?: Record<string, unknown>): Float64Array;
  trim(y: VectorInput, options?: Record<string, unknown>): TrimResult;
  split(y: VectorInput, options?: Record<string, unknown>): Matrix;
  preemphasis(y: VectorInput, options?: Record<string, unknown>): Float64Array;
  deemphasis(y: VectorInput, options?: Record<string, unknown>): Float64Array;
  remix(y: VectorInput, intervals: Matrix | ArrayLike<ArrayLike<number>>, options?: Record<string, unknown>): Float64Array;
  harmonicEffect(y: VectorInput, options?: Record<string, unknown>): Float64Array;
  percussiveEffect(y: VectorInput, options?: Record<string, unknown>): Float64Array;

  medianFilter2d(s: Matrix, options?: Record<string, unknown>): Matrix;
  hpss(s: Matrix | ComplexMatrix, options?: Record<string, unknown>): PairMatrixResult<Matrix | ComplexMatrix>;
  decomposeNmf(s: Matrix, options?: Record<string, unknown>): NmfResult;

  transitionUniform(options?: Record<string, unknown>): Matrix;
  transitionLoop(prob: number | VectorInput, options?: Record<string, unknown>): Matrix;
  transitionCycle(prob: number | VectorInput, options?: Record<string, unknown>): Matrix;
  transitionLocal(options?: Record<string, unknown>): Matrix;
  viterbi(prob: Matrix, transition: Matrix, options?: Record<string, unknown>): Int32Array | { states: Int32Array; logp: number };
  viterbiDiscriminative(prob: Matrix, transition: Matrix, options?: Record<string, unknown>): Int32Array;
  viterbiBinary(prob: Matrix, transition: Matrix, options?: Record<string, unknown>): Matrix | { states: Matrix; logp: Float64Array };
  cdistEuclidean(x: Matrix, y: Matrix): Matrix;
  cdistCosine(x: Matrix, y: Matrix): Matrix;
  dtw(x: Matrix, y: Matrix, options?: Record<string, unknown>): Matrix | DtwBacktrackResult;
  rqa(sim: Matrix, options?: Record<string, unknown>): Matrix | RqaBacktrackResult;
  recurrenceToLag(rec: Matrix, options?: Record<string, unknown>): Matrix;
  lagToRecurrence(lag: Matrix, options?: Record<string, unknown>): Matrix;
  crossSimilarity(data: Matrix, dataRef: Matrix, options?: Record<string, unknown>): Matrix;
  recurrenceMatrix(data: Matrix, options?: Record<string, unknown>): Matrix;
  pathEnhance(r: Matrix, options?: Record<string, unknown>): Matrix;
  agglomerative(data: Matrix, options?: Record<string, unknown>): Float64Array;
  subsegment(data: Matrix, frames: VectorInput, options?: Record<string, unknown>): Float64Array;

  interpHarmonics(x: VectorInput | Matrix, freqs: VectorInput, harmonics: VectorInput, options?: Record<string, unknown>): Matrix;
  f0Harmonics(x: Matrix, f0: VectorInput, freqs: VectorInput, harmonics: VectorInput, options?: Record<string, unknown>): Matrix;
  salience(s: Matrix, freqs: VectorInput, harmonics: VectorInput, options?: Record<string, unknown>): Matrix;

  frame(x: VectorInput, options?: Record<string, unknown>): Matrix;
  padCenter(data: VectorInput | Matrix, options?: Record<string, unknown>): Float64Array | Matrix;
  fixLength(data: VectorInput | Matrix, options?: Record<string, unknown>): Float64Array | Matrix;
  fixFrames(frames: VectorInput, options?: Record<string, unknown>): Float64Array;
  normalize(data: VectorInput | Matrix, options?: Record<string, unknown>): Float64Array | Matrix;
  localmax(data: VectorInput | Matrix, options?: Record<string, unknown>): Uint8Array | BoolMatrix;
  localmin(data: VectorInput | Matrix, options?: Record<string, unknown>): Uint8Array | BoolMatrix;
  peakPick(x: VectorInput, options?: Record<string, unknown>): Float64Array;
  softmask(x: Matrix, xRef: Matrix, options?: Record<string, unknown>): Matrix;
  abs2(x: Float64Array | ComplexMatrix): Float64Array | Matrix;
  phasor(angles: VectorInput, options?: Record<string, unknown>): Float64Array;
  fillOffDiagonal(x: Matrix, options?: Record<string, unknown>): Matrix;
  stack(arrays: ArrayLike<VectorInput>, options?: Record<string, unknown>): Matrix;
  validAudio(y: VectorInput | Matrix): boolean;
  isPositiveInt(x: number): boolean;
  validInt(x: number, options?: Record<string, unknown>): number;

  pythagoreanIntervals(options?: Record<string, unknown>): Float64Array;
  plimitIntervals(primes: VectorInput, options?: Record<string, unknown>): Float64Array;
  intervalFrequencies(intervals: string | VectorInput, options?: Record<string, unknown>): Float64Array;
  thaatToDegrees(thaat: string): Int32Array;
  melaToDegrees(mela: number | string): Int32Array;
  melaToSvara(mela: number | string, options?: Record<string, unknown>): string[];
  listMela(): Record<string, number>;
  listThaat(): string[];
  keyToDegrees(key: string): Int32Array;
  keyToNotes(key: string, options?: Record<string, unknown>): string[];
  fifthsToNote(unison: string, fifths: number, options?: Record<string, unknown>): string;
  intervalToFjs(interval: number, options?: Record<string, unknown>): string;
  midiToSvaraH(midi: number, sa: number, options?: Record<string, unknown>): string;
  midiToSvaraC(midi: number, sa: number, mela: number | string, options?: Record<string, unknown>): string;
  hzToFjs(frequency: number, options?: Record<string, unknown>): string;
}
