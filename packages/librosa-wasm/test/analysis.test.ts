import test from "node:test";
import assert from "node:assert/strict";
import {
  assertArrayClose,
  assertClose,
  assertFiniteMatrix,
  complexMatrix,
  librosa,
  matrix,
  vector
} from "./helpers.js";

test("decompose: median filter, HPSS, and NMF", async () => {
  const l = await librosa();
  const s = matrix(3, 3, [
    1, 2, 3,
    4, 100, 6,
    7, 8, 9
  ]);

  const filtered = l.medianFilter2d(s, { size: [3, 3], mode: "edge" });
  assert.equal(filtered.rows, s.rows);
  assert.equal(filtered.cols, s.cols);

  const hpss = l.hpss(matrix(3, 4, [
    1, 1, 1, 1,
    0, 10, 0, 10,
    2, 2, 2, 2
  ]), { kernelSize: 3 });
  assert.equal(hpss.harmonic.rows, 3);
  assert.equal(hpss.percussive.cols, 4);

  const complex = l.hpss(complexMatrix(1, 2, [1, 1, 2, 0]), { complex: true, kernelSize: 3 });
  assert.equal(complex.harmonic.data.length, 4);

  const nmf = l.decomposeNmf(matrix(3, 4, [
    1, 2, 3, 4,
    2, 4, 6, 8,
    1, 1, 1, 1
  ]), { nComponents: 2, maxIter: 10 });
  assert.equal(nmf.components.rows, 3);
  assert.equal(nmf.components.cols, 2);
  assert.equal(nmf.activations.rows, 2);
  assert.equal(nmf.activations.cols, 4);
});

test("sequence: transitions, Viterbi, distances, DTW, and RQA", async () => {
  const l = await librosa();

  const uniform = l.transitionUniform({ nStates: 3 });
  assertArrayClose(uniform.data, new Array(9).fill(1 / 3));

  const loop = l.transitionLoop(0.8, { nStates: 2 });
  assertClose(loop.data[0], 0.8);
  assertClose(loop.data[1], 0.2);

  const cycle = l.transitionCycle(vector([0.1, 0.2]), { nStates: 2 });
  assert.equal(cycle.rows, 2);
  assert.equal(cycle.cols, 2);

  const local = l.transitionLocal({ nStates: 5, width: 3 });
  assert.equal(local.rows, 5);

  const prob = matrix(2, 3, [0.9, 0.1, 0.1, 0.1, 0.9, 0.9]);
  const trans = matrix(2, 2, [0.8, 0.2, 0.2, 0.8]);
  assert.deepEqual(Array.from(l.viterbi(prob, trans) as Int32Array), [0, 1, 1]);

  const withLogp = l.viterbi(prob, trans, { returnLogp: true }) as { states: Int32Array; logp: number };
  assert.deepEqual(Array.from(withLogp.states), [0, 1, 1]);
  assert.ok(Number.isFinite(withLogp.logp));

  const discriminative = l.viterbiDiscriminative(prob, trans);
  assert.equal(discriminative.length, 3);

  const binary = l.viterbiBinary(matrix(2, 3, [0.9, 0.1, 0.9, 0.1, 0.9, 0.1]), trans) as ReturnType<typeof l.viterbiBinary>;
  assert.equal((binary as { rows: number }).rows, 2);

  const x = matrix(2, 3, [0, 1, 2, 0, 0, 0]);
  const y = matrix(2, 2, [0, 2, 0, 0]);
  assertFiniteMatrix(l.cdistEuclidean(x, y));
  assertFiniteMatrix(l.cdistCosine(x, y));

  const dtw = l.dtw(x, y, { backtrack: true }) as { cost: typeof x; path: typeof x };
  assertFiniteMatrix(dtw.cost);
  assert.equal(dtw.path.cols, 2);

  const rqa = l.rqa(matrix(3, 3, [
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
  ]), { backtrack: true }) as { score: typeof x; path: typeof x };
  assertFiniteMatrix(rqa.score);
  assert.equal(rqa.path.cols, 2);
});

test("segment: recurrence/lag conversion, similarity, path enhancement, and clustering", async () => {
  const l = await librosa();
  const rec = matrix(3, 3, [
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
  ]);

  const lag = l.recurrenceToLag(rec, { pad: true });
  const roundtrip = l.lagToRecurrence(lag);
  assert.equal(roundtrip.rows, 3);
  assert.equal(roundtrip.cols, 3);

  const data = matrix(2, 5, [
    0, 1, 2, 10, 11,
    0, 0, 0, 0, 0
  ]);
  const cross = l.crossSimilarity(data, data, { k: 2 });
  assert.equal(cross.rows, 5);
  assert.equal(cross.cols, 5);

  const recurrence = l.recurrenceMatrix(data, { k: 2, self: true });
  assert.equal(recurrence.rows, 5);
  assert.equal(recurrence.cols, 5);

  const enhanced = l.pathEnhance(recurrence, { n: 3, nFilters: 3 });
  assert.equal(enhanced.rows, recurrence.rows);
  assert.equal(enhanced.cols, recurrence.cols);

  assert.deepEqual(Array.from(l.agglomerative(data, { k: 2 }))[0], 0);
  assert.deepEqual(Array.from(l.subsegment(data, vector([0, 5]), { nSegments: 2 }))[0], 0);
});

test("harmonic helpers: interpolation, f0 harmonics, and salience", async () => {
  const l = await librosa();
  const spectrum = matrix(3, 2, [
    1, 2,
    3, 4,
    5, 6
  ]);
  const freqs = vector([100, 200, 400]);
  const harmonics = vector([1, 2]);

  const interp = l.interpHarmonics(spectrum, freqs, harmonics);
  assert.equal(interp.rows, 6);
  assert.equal(interp.cols, 2);

  const f0 = l.f0Harmonics(spectrum, vector([100, 200]), freqs, harmonics);
  assert.equal(f0.rows, 2);
  assert.equal(f0.cols, 2);

  const salience = l.salience(spectrum, freqs, harmonics, { filterPeaks: false });
  assert.equal(salience.rows, spectrum.rows);
  assert.equal(salience.cols, spectrum.cols);
});
