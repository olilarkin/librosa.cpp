import test from "node:test";
import assert from "node:assert/strict";
import { assertArrayClose, assertClose, librosa, vector } from "./helpers.js";

test("intervals: pythagorean, p-limit, and interval frequencies", async () => {
  const l = await librosa();

  const pyth = l.pythagoreanIntervals({ binsPerOctave: 12 });
  assert.equal(pyth.length, 12);
  assertClose(pyth[0], 1);

  const plimit = l.plimitIntervals(vector([3, 5]), { binsPerOctave: 12 });
  assert.equal(plimit.length, 12);

  const equal = l.intervalFrequencies("equal", { nBins: 4, fmin: 55, binsPerOctave: 12 });
  assert.equal(equal.length, 4);
  assertClose(equal[0], 55);

  const explicit = l.intervalFrequencies(vector([1, 1.5]), { nBins: 4, fmin: 100 });
  assertArrayClose(explicit, [100, 150, 200, 300]);
});

test("notation: thaat, mela, key spelling, fifths, and FJS", async () => {
  const l = await librosa();

  assert.deepEqual(Array.from(l.thaatToDegrees("bilaval")), [0, 2, 4, 5, 7, 9, 11]);
  assert.equal(l.listThaat().includes("bilaval"), true);
  assert.deepEqual(Array.from(l.melaToDegrees(1)), [0, 1, 2, 5, 7, 8, 9]);
  assert.equal(l.melaToSvara("kanakangi", { unicode: false }).length, 12);
  assert.equal(l.listMela().kanakangi, 1);

  assert.deepEqual(Array.from(l.keyToDegrees("C:maj")), [0, 2, 4, 5, 7, 9, 11]);
  assert.deepEqual(l.keyToNotes("F:maj", { unicode: false }).slice(0, 6), ["C", "Db", "D", "Eb", "E", "F"]);
  assert.equal(l.fifthsToNote("C", 1, { unicode: false }), "G");
  assert.equal(l.intervalToFjs(3 / 2, { unison: "C", unicode: false }), "G");
  assert.equal(l.intervalToFjs(5 / 4, { unison: "C", unicode: false }), "E^5");
});
