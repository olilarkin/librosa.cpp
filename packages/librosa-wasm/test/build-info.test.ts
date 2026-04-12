import test from "node:test";
import assert from "node:assert/strict";
import { librosa } from "./helpers.js";

test("build: wasm module reports PFFFT and SIMD build metadata", async () => {
  const l = await librosa();
  const info = l.buildInfo();

  assert.equal(info.fftBackend, "pffft");
  assert.equal(info.wasmSimd, true);
  assert.ok(info.pffftSimdSize >= 1);
  assert.ok(info.pffftSimdArch.length > 0);
});
