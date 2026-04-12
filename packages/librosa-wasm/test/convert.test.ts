import test from "node:test";
import assert from "node:assert/strict";
import { assertArrayClose, assertClose, librosa, vector } from "./helpers.js";

test("convert: frame, sample, and time conversions match native formulas", async () => {
  const l = await librosa();

  assert.equal(l.framesToSamples(3, { hopLength: 512 }), 1536);
  assert.equal(l.framesToSamples(3, { hopLength: 512, nFft: 2048 }), 2560);
  assert.equal(l.samplesToFrames(2560, { hopLength: 512, nFft: 2048 }), 3);

  assertClose(l.framesToTime(2, { sr: 1000, hopLength: 100 }) as number, 0.2);
  assert.equal(l.timeToFrames(0.2, { sr: 1000, hopLength: 100 }), 2);
  assert.equal(l.timeToSamples(0.5, { sr: 8000 }), 4000);
  assertClose(l.samplesToTime(4000, { sr: 8000 }) as number, 0.5);

  assertArrayClose(l.framesToSamples(vector([0, 1, 2]), { hopLength: 64 }) as Float64Array, [0, 64, 128]);
});

test("convert: note, midi, hz, mel, octave, and tuning conversions", async () => {
  const l = await librosa();

  assertClose(l.midiToHz(69) as number, 440);
  assertClose(l.hzToMidi(440) as number, 69);
  assert.equal(l.noteToMidi("A4"), 69);
  assertClose(l.noteToHz("A4") as number, 440);
  assert.equal(l.midiToNote(69, { unicode: false }), "A4");
  assert.equal(l.hzToNote(440, { unicode: false }), "A4");

  assertClose(l.hzToMel(1000) as number, 15, 1e-12);
  assertClose(l.melToHz(15) as number, 1000, 1e-9);
  assertClose(l.octsToHz(l.hzToOcts(440) as number) as number, 440, 1e-9);
  assertClose(l.tuningToA4(l.a4ToTuning(442) as number) as number, 442, 1e-9);

  assertArrayClose(l.midiToHz(vector([69, 81])) as Float64Array, [440, 880], 1e-9);
});

test("convert: frequency helper arrays and weighting", async () => {
  const l = await librosa();

  assertArrayClose(l.fftFrequencies({ sr: 8, nFft: 8 }), [0, 1, 2, 3, 4]);
  assert.equal(l.melFrequencies({ nMels: 4, fmin: 0, fmax: 8000 }).length, 4);
  assert.equal(l.cqtFrequencies({ nBins: 12, fmin: 55 }).length, 12);
  assert.equal(l.tempoFrequencies({ nBins: 16, hopLength: 512, sr: 22050 }).length, 16);
  assert.ok(Number.isFinite(l.weighting(1000, { kind: "A" }) as number));
});

test("convert: svara and FJS helpers", async () => {
  const l = await librosa();

  assert.equal(l.midiToSvaraH(60, 60, { unicode: false }), "S");
  assert.equal(l.midiToSvaraC(67, 60, 1, { unicode: false }).length > 0, true);
  assert.equal(l.hzToFjs(660, { fmin: 440, unison: "A", unicode: false }), "E");
});
