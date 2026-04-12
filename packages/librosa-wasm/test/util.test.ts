import test from "node:test";
import assert from "node:assert/strict";
import { assertArrayClose, assertClose, assertFiniteMatrix, librosa, matrix, vector } from "./helpers.js";

test("util: frame, pad_center, fix_length, and fix_frames", async () => {
  const l = await librosa();

  const frames = l.frame(vector([0, 1, 2, 3, 4]), { frameLength: 3, hopLength: 1 });
  assert.equal(frames.rows, 3);
  assert.equal(frames.cols, 3);
  assertArrayClose(frames.data, [0, 1, 2, 1, 2, 3, 2, 3, 4]);

  assertArrayClose(l.padCenter(vector([1, 2, 3]), { size: 7 }) as Float64Array, [0, 0, 1, 2, 3, 0, 0]);
  assertArrayClose(l.fixLength(vector([1, 2, 3]), { size: 5 }) as Float64Array, [1, 2, 3, 0, 0]);
  assertArrayClose(l.fixLength(vector([1, 2, 3]), { size: 2 }) as Float64Array, [1, 2]);
  assertArrayClose(l.fixFrames(vector([3, 1, 1]), { xMin: 0, xMax: 4, pad: true }), [0, 1, 3, 4]);
});

test("util: normalize, local extrema, peak pick, and softmask", async () => {
  const l = await librosa();

  assertArrayClose(l.normalize(vector([3, 4]), { norm: 2 }) as Float64Array, [0.6, 0.8], 1e-12);
  assert.deepEqual(Array.from(l.localmax(vector([0, 1, 0, 2, 1])) as Uint8Array), [0, 1, 0, 1, 0]);
  assert.deepEqual(Array.from(l.localmin(vector([1, 0, 1, 0, 2])) as Uint8Array), [0, 1, 0, 1, 0]);
  assert.deepEqual(Array.from(l.peakPick(vector([0, 1, 0, 0, 2, 0]), {
    preMax: 1,
    postMax: 1,
    preAvg: 1,
    postAvg: 1,
    delta: 0.1,
    wait: 0
  })), [1, 4]);

  const mask = l.softmask(matrix(1, 2, [3, 4]), matrix(1, 2, [1, 4]), { power: 1 });
  assertFiniteMatrix(mask);
  assertClose(mask.data[0], 0.75);
  assertClose(mask.data[1], 0.5);
});

test("util: complex helpers and stacking", async () => {
  const l = await librosa();

  assertArrayClose(l.abs2(vector([3, 4, 1, 1])) as Float64Array, [25, 2]);

  const phasor = l.phasor(vector([0, Math.PI / 2]));
  assertArrayClose(phasor, [1, 0, 0, 1], 1e-12);

  const filled = l.fillOffDiagonal(matrix(3, 3, [
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  ]), { fillValue: 0 });
  assert.equal(filled.rows, 3);
  assert.equal(filled.cols, 3);

  const stacked = l.stack([vector([1, 2]), vector([3, 4])], { axis: 0 });
  assertArrayClose(stacked.data, [1, 2, 3, 4]);
});

test("util: validation helpers", async () => {
  const l = await librosa();

  assert.equal(l.validAudio(vector([0, 0.5, -0.5])), true);
  assert.equal(l.isPositiveInt(2), true);
  assert.equal(l.isPositiveInt(0), false);
  assert.equal(l.validInt(2.9), 2);
  assert.equal(l.validInt(2.9, { useFloor: false }), 2);
});
