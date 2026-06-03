"use strict";
// Smoke test for the CommonJS build: load the package exactly as a `require()`
// consumer (Electron main, Ableton Extension Host, plain Node) would, then run
// a real computation to prove the .cjs glue can locate and instantiate the wasm.
const assert = require("node:assert");
const { test } = require("node:test");

const entry = require("../dist/src/index.cjs");

test("CJS entry exposes createLibrosa", () => {
  assert.strictEqual(typeof entry, "function");
  assert.strictEqual(typeof entry.createLibrosa, "function");
  assert.strictEqual(entry.default, entry);
});

test("CJS build instantiates the wasm module and computes", async () => {
  const librosa = await entry();
  const y = librosa.tone(440, { sr: 22050, duration: 0.1 });
  assert.ok(y.length > 0, "tone() should return samples");
  const mfcc = librosa.mfcc(y, { sr: 22050, nMfcc: 13 });
  assert.strictEqual(mfcc.rows, 13);
});
