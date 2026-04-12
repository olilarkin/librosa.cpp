import test from "node:test";
import assert from "node:assert/strict";
import { assertArrayClose, assertClose, librosa, matrix, sine, vector } from "./helpers.js";

test("audio: mono conversion, duration, and resampling", async () => {
  const l = await librosa();

  assertArrayClose(l.toMono(matrix(2, 3, [1, 2, 3, 3, 2, 1])), [2, 2, 2]);
  assertArrayClose(l.toMono(vector([1, 2, 3])), [1, 2, 3]);
  assertClose(l.duration(vector([0, 0, 0, 0]), { sr: 2 }), 2);
  assertClose(l.duration(matrix(3, 4, new Array(12).fill(0)), {
    sr: 8,
    hopLength: 2,
    nFft: 4,
    center: true
  }), 0.75);

  const y = sine(256, 220, 8000);
  const resampled = l.resample(y, { origSr: 8000, targetSr: 4000, resType: "linear" });
  assert.ok(Math.abs(resampled.length - 128) <= 1);
});

test("audio: autocorrelation, LPC, and zero crossings", async () => {
  const l = await librosa();

  assertArrayClose(l.autocorrelate(vector([1, 2, 3]), { maxSize: 3 }) as Float64Array, [14, 8, 3]);
  const coeffs = l.lpc(sine(512), { order: 2 }) as Float64Array;
  assert.equal(coeffs.length, 3);
  assertClose(coeffs[0], 1, 1e-12);

  assert.deepEqual(Array.from(l.zeroCrossings(vector([-1, 1, -1]), { pad: false }) as Uint8Array), [0, 1, 1]);
  assert.deepEqual(Array.from(l.zeroCrossings(vector([-1, 1, -1]), { pad: true }) as Uint8Array), [1, 1, 1]);
});

test("audio: synthesis and mu-law roundtrip", async () => {
  const l = await librosa();

  const tone = l.tone(440, { sr: 8000, duration: 0.25 });
  assert.equal(tone.length, 2000);
  assertClose(tone[0], 0, 1e-12);

  const chirp = l.chirp({ fmin: 100, fmax: 800, sr: 8000, length: 256, linear: true });
  assert.equal(chirp.length, 256);

  const clicks = l.clicks(vector([0, 0.1]), { sr: 1000, clickDuration: 0.01, length: 200 });
  assert.equal(clicks.length, 200);
  assert.ok(clicks.some((value) => Math.abs(value) > 0));

  const x = vector([-1, -0.5, 0, 0.5, 1]);
  const compressed = l.muCompress(x, { quantize: false });
  const expanded = l.muExpand(compressed, { quantize: false });
  assertArrayClose(expanded, x, 1e-10);
});
