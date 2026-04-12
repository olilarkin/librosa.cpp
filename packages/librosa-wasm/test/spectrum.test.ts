import test from "node:test";
import assert from "node:assert/strict";
import {
  assertArrayClose,
  assertClose,
  assertFiniteComplexMatrix,
  assertFiniteMatrix,
  complexMatrix,
  librosa,
  matrix,
  sine,
  vector
} from "./helpers.js";

test("spectrum: STFT shape, complex output, and ISTFT reconstruction", async () => {
  const l = await librosa();
  const y = sine(4096, 440);
  const d = l.stft(y, { nFft: 1024, hopLength: 256 });

  assert.equal(d.rows, 513);
  assert.ok(d.cols > 0);
  assertFiniteComplexMatrix(d);
  assert.ok(d.data.some((value, index) => index % 2 === 1 && Math.abs(value) > 1e-10));

  const reconstructed = l.istft(d, { hopLength: 256, nFft: 1024, length: y.length });
  assert.equal(reconstructed.length, y.length);
  for (let i = 1024; i < y.length - 1024; i += 257) {
    assertClose(reconstructed[i], y[i], 1e-4);
  }
});

test("spectrum: magnitude, phase, magphase, and dB conversions", async () => {
  const l = await librosa();
  const d = complexMatrix(2, 2, [
    3, 4, 0, 1,
    1, 0, 1, 1
  ]);

  const magnitude = l.magnitude(d);
  assertArrayClose(magnitude.data, [5, 1, 1, Math.sqrt(2)]);

  const phase = l.phase(complexMatrix(2, 2, [
    1, 0, 0, 1,
    -1, 0, 0, -1
  ]));
  assertClose(phase.data[0], 0);
  assertClose(phase.data[1], Math.PI / 2);
  assertClose(Math.abs(phase.data[2]), Math.PI);
  assertClose(phase.data[3], -Math.PI / 2);

  const split = l.magphase(d);
  assertArrayClose(split.magnitude.data, magnitude.data);
  assert.equal(split.phase.data.length, d.data.length);

  const db = l.powerToDb(matrix(2, 2, [1, 10, 100, 1000]), { topDb: null });
  assertArrayClose(db.data, [0, 10, 20, 30], 1e-9);
  assertClose(l.dbToPower(20) as number, 100, 1e-9);
  assertClose(l.amplitudeToDb(10) as number, 20, 1e-9);
  assertClose(l.dbToAmplitude(20) as number, 10, 1e-9);
});

test("spectrum: phase vocoder, Griffin-Lim, PCEN, windows, and weighting", async () => {
  const l = await librosa();
  const y = sine(4096);
  const d = l.stft(y, { nFft: 1024, hopLength: 256 });
  const mag = l.magnitude(d);

  const faster = l.phaseVocoder(d, { rate: 2, hopLength: 256, nFft: 1024 });
  assert.equal(faster.rows, d.rows);
  assert.ok(faster.cols < d.cols);

  const reconstructed = l.griffinlim(mag, {
    nIter: 2,
    hopLength: 256,
    nFft: 1024,
    length: 4096,
    randomState: 0
  });
  assert.equal(reconstructed.length, 4096);

  const pcen = l.pcen(matrix(2, 3, [1, 2, 3, 4, 5, 6]), { sr: 22050, hopLength: 512 });
  assertFiniteMatrix(pcen);

  const window = l.getWindow("hann", { length: 8 });
  assert.equal(window.length, 8);
  assert.ok(window.every((value) => value >= 0 && value <= 1));

  const sumsquare = l.windowSumsquare(window, { nFrames: 3, hopLength: 2, nFft: 8 });
  assert.equal(sumsquare.length, 12);

  const weighted = l.perceptualWeighting(matrix(1, 2, [1, 2]), vector([1000, 2000]), { kind: "A" });
  assertFiniteMatrix(weighted);
});
