import test from "node:test";
import assert from "node:assert/strict";
import {
  assertArrayClose,
  assertClose,
  assertFiniteMatrix,
  impulseTrain,
  librosa,
  matrix,
  sine,
  vector
} from "./helpers.js";

test("pitch: tuning, piptrack, YIN, and pYIN", async () => {
  const l = await librosa();
  const y = sine(4096, 440);

  assertClose(l.pitchTuning(vector([440, 880, 1760])), 0, 1e-9);

  const pip = l.piptrack(y, { sr: 22050, nFft: 1024, hopLength: 256, fmin: 100, fmax: 1000 });
  assert.equal(pip.pitches.rows, 513);
  assert.equal(pip.pitches.cols, pip.magnitudes.cols);
  assertFiniteMatrix(pip.pitches);

  const yin = l.yin(y, { fmin: 100, fmax: 1000, sr: 22050, frameLength: 1024, hopLength: 256 });
  assert.ok(yin.length > 0);
  assert.ok(yin.some((value) => value > 300 && value < 600));

  const pyin = l.pyin(y, {
    fmin: 100,
    fmax: 1000,
    sr: 22050,
    frameLength: 1024,
    hopLength: 256,
    nThresholds: 10
  });
  assert.equal(pyin.f0.length, pyin.voicedFlag.length);
  assert.equal(pyin.f0.length, pyin.voicedProb.length);
});

test("beat and onset: envelopes, tempo, beat tracking, and event helpers", async () => {
  const l = await librosa();
  const envelope = impulseTrain(64, 8);

  const tg = l.tempogram(envelope, { winLength: 16, center: false });
  assert.equal(tg.rows, 16);
  assert.equal(tg.cols, envelope.length - 16 + 1);

  assert.ok(l.tempo(envelope, { sr: 22050, hopLength: 512, maxTempo: 300 }) > 0);
  assert.equal(l.tempoFrames(envelope, { maxTempo: 300 }).length, envelope.length);

  const beats = l.beatTrack(envelope, { bpm: 120, trim: false });
  assert.ok(beats.tempo > 0);
  assert.ok(beats.beats.length > 0);

  assert.deepEqual(Array.from(l.matchEvents(vector([1, 5]), vector([0, 4, 8]))), [0, 1]);
  assert.deepEqual(Array.from(l.onsetBacktrack(vector([3, 5]), vector([2, 1, 0, 3, 0, 4]))), [2, 4]);

  const s = matrix(2, 4, [0, 1, 0, 2, 3, 1, 4, 0]);
  const maxed = l.maximumFilter1d(s, { size: 3, axis: -1 });
  assert.equal(maxed.rows, s.rows);
  assert.equal(maxed.cols, s.cols);
});

test("onset: strength and detection from synthetic transients", async () => {
  const l = await librosa();
  const y = new Float64Array(4096);
  y[512] = 1;
  y[1536] = 1;
  y[2560] = 1;

  const strength = l.onsetStrength(y, { sr: 22050, nFft: 1024, hopLength: 256 });
  assert.ok(strength.length > 0);

  const multi = l.onsetStrengthMulti(y, { sr: 22050, nFft: 1024, hopLength: 256, channels: [0, 32, 64] });
  assert.equal(multi.rows, 2);

  const detected = l.onsetDetect(strength, {
    envelope: true,
    preMax: 1,
    postMax: 1,
    preAvg: 1,
    postAvg: 1,
    wait: 1,
    delta: 0.01
  });
  assert.ok(detected.length > 0);
});

test("effects: stretch, pitch shift, trim/split, filters, remix, and HPSS effects", async () => {
  const l = await librosa();
  const y = sine(4096, 440);

  const stretched = l.timeStretch(y, { rate: 2, nFft: 1024, hopLength: 256 });
  assert.ok(stretched.length < y.length);

  const shifted = l.pitchShift(y, {
    sr: 22050,
    nSteps: 1,
    resType: "linear",
    nFft: 1024,
    hopLength: 256
  });
  assert.equal(shifted.length, y.length);

  const withSilence = new Float64Array(4096);
  withSilence.set(y.slice(0, 1024), 1024);
  const trimmed = l.trim(withSilence, { topDb: 40, frameLength: 512, hopLength: 128 });
  assert.ok(trimmed.audio.length < withSilence.length);
  assert.equal(trimmed.index.length, 2);

  const intervals = l.split(withSilence, { topDb: 40, frameLength: 512, hopLength: 128 });
  assert.equal(intervals.cols, 2);

  const pre = l.preemphasis(vector([1, 2, 3]), { coef: 0.5 });
  assertArrayClose(pre, [1, 1.5, 2]);
  const de = l.deemphasis(pre, { coef: 0.5 });
  assertArrayClose(de, [1, 2, 3]);

  assertArrayClose(l.remix(vector([1, 2, 3, 4]), matrix(2, 2, [2, 4, 0, 2]), { alignZeros: false }), [3, 4, 1, 2]);

  assert.equal(l.harmonicEffect(y, { nFft: 1024, hopLength: 256 }).length, y.length);
  assert.equal(l.percussiveEffect(y, { nFft: 1024, hopLength: 256 }).length, y.length);
});
