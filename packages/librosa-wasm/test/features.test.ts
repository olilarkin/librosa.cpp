import test from "node:test";
import assert from "node:assert/strict";
import { assertFiniteComplexMatrix, assertFiniteMatrix, librosa, matrix, sine, vector } from "./helpers.js";

test("features: mel spectrogram, MFCC, and chroma STFT", async () => {
  const l = await librosa();
  const y = sine(4096, 440);

  const mel = l.melspectrogram(y, { sr: 22050, nFft: 1024, hopLength: 256, nMels: 32 });
  assert.equal(mel.rows, 32);
  assertFiniteMatrix(mel);

  const mfcc = l.mfcc(y, { sr: 22050, nFft: 1024, hopLength: 256, nMfcc: 13, nMels: 32 });
  assert.equal(mfcc.rows, 13);
  assert.equal(mfcc.cols, mel.cols);
  assertFiniteMatrix(mfcc);

  const fromMel = l.mfcc(l.powerToDb(mel, { topDb: null }), { nMfcc: 10 });
  assert.equal(fromMel.rows, 10);

  const chroma = l.chromaStft(y, { sr: 22050, nFft: 1024, hopLength: 256 });
  assert.equal(chroma.rows, 12);
  assertFiniteMatrix(chroma);
});

test("features: spectral descriptors, RMS, ZCR, polynomial features, and deltas", async () => {
  const l = await librosa();
  const y = sine(4096, 440);

  assertFiniteMatrix(l.spectralCentroid(y, { sr: 22050, nFft: 1024, hopLength: 256 }));
  assertFiniteMatrix(l.spectralBandwidth(y, { sr: 22050, nFft: 1024, hopLength: 256 }));
  assertFiniteMatrix(l.spectralRolloff(y, { sr: 22050, nFft: 1024, hopLength: 256 }));
  assertFiniteMatrix(l.spectralFlatness(y, { nFft: 1024, hopLength: 256 }));
  assertFiniteMatrix(l.spectralContrast(y, { sr: 22050, nFft: 1024, hopLength: 256, nBands: 4 }));
  assertFiniteMatrix(l.rms(y, { frameLength: 1024, hopLength: 256 }));
  assertFiniteMatrix(l.zeroCrossingRate(y, { frameLength: 1024, hopLength: 256 }));
  assertFiniteMatrix(l.polyFeatures(y, { sr: 22050, nFft: 1024, hopLength: 256, order: 2 }));

  const data = matrix(2, 5, [
    1, 2, 3, 4, 5,
    2, 4, 6, 8, 10
  ]);
  const delta = l.delta(data, { width: 3, order: 1, mode: "nearest" });
  assert.equal(delta.rows, data.rows);
  assert.equal(delta.cols, data.cols);

  const stacked = l.stackMemory(data, { nSteps: 3, delay: 1 });
  assert.equal(stacked.rows, 6);
  assert.equal(stacked.cols, data.cols);
});

test("features: CQT-family chroma projections", async () => {
  const l = await librosa();
  const pseudo = matrix(24, 3, new Array(72).fill(0).map((_, i) => (i % 24) + 1));

  const chroma = l.chromaCqt(pseudo, { nChroma: 12, binsPerOctave: 12 });
  assert.equal(chroma.rows, 12);
  assertFiniteMatrix(chroma);

  const vqtChroma = l.chromaVqt(pseudo, { binsPerOctave: 12 });
  assert.equal(vqtChroma.rows, 12);
  assertFiniteMatrix(vqtChroma);
});

test("features: Fourier tempogram, tempogram ratio, inverse helpers, and tonnetz", async () => {
  const l = await librosa();
  const envelope = vector(new Array(64).fill(0).map((_, i) => (i % 8 === 0 ? 1 : 0)));

  const ft = l.fourierTempogram(envelope, { winLength: 32, hopLength: 512, center: false });
  assert.equal(ft.rows, 17);
  assertFiniteComplexMatrix(ft);

  const tg = matrix(8, 3, new Array(24).fill(0).map((_, i) => (i % 8 === 2 ? 1 : 0)));
  const ratio = l.tempogramRatio(tg, { factors: vector([0.5, 1, 2]), maxTempo: null });
  assert.equal(ratio.rows, 3);
  assert.equal(ratio.cols, 3);

  const mel = matrix(8, 3, new Array(24).fill(1));
  const stft = l.melToStft(mel, { sr: 22050, nFft: 64 });
  assert.equal(stft.rows, 33);
  assert.equal(stft.cols, 3);

  const approxMel = l.mfccToMel(matrix(4, 3, new Array(12).fill(0)), { nMels: 8 });
  assert.equal(approxMel.rows, 8);

  const tonnetz = l.tonnetz(matrix(12, 4, new Array(48).fill(1 / 12)));
  assert.equal(tonnetz.rows, 6);
});
