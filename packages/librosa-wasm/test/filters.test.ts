import test from "node:test";
import assert from "node:assert/strict";
import { assertClose, assertFiniteComplexMatrix, assertFiniteMatrix, librosa, vector } from "./helpers.js";

test("filters: mel, chroma, and CQ-to-chroma shapes", async () => {
  const l = await librosa();

  const mel = l.melFilter({ sr: 22050, nFft: 1024, nMels: 40 });
  assert.equal(mel.rows, 40);
  assert.equal(mel.cols, 513);
  assertFiniteMatrix(mel);

  const chroma = l.chromaFilter({ sr: 22050, nFft: 1024, nChroma: 12 });
  assert.equal(chroma.rows, 12);
  assert.equal(chroma.cols, 513);
  assertFiniteMatrix(chroma);

  const cq = l.cqToChroma({ nInput: 84, binsPerOctave: 12, nChroma: 12 });
  assert.equal(cq.rows, 12);
  assert.equal(cq.cols, 84);
});

test("filters: window bandwidth, wavelet lengths, wavelet filters, and diagonal filter", async () => {
  const l = await librosa();

  assert.ok(l.windowBandwidth("hann") > 0);

  const freqs = vector([110, 220, 440, 880]);
  const relative = l.relativeBandwidth(freqs);
  assert.equal(relative.length, freqs.length);

  const lengths = l.waveletLengths(freqs, { sr: 22050 });
  assert.equal(lengths.lengths.length, freqs.length);
  assert.ok(lengths.cutoff > 0);

  const wavelet = l.wavelet(freqs, { sr: 22050, padFft: true });
  assert.equal(wavelet.filters.rows, freqs.length);
  assertFiniteComplexMatrix(wavelet.filters);
  assert.equal(wavelet.lengths.length, freqs.length);

  const diagonal = l.diagonalFilter({ window: "hann", n: 5, slope: 1 });
  assert.equal(diagonal.rows, 5);
  assert.equal(diagonal.cols, 5);
  assertClose(diagonal.data.reduce((sum, value) => sum + value, 0), 1, 1e-9);
});

test("filters: multi-rate frequency helper", async () => {
  const l = await librosa();

  const mr = l.mrFrequencies();
  assert.equal(mr.frequencies.length, mr.sampleRates.length);
  assert.ok(mr.frequencies.length > 0);
});
