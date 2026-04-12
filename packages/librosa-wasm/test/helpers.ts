import assert from "node:assert/strict";
import { createLibrosa, type ComplexMatrix, type Librosa, type Matrix } from "../src/index.js";

let instance: Promise<Librosa> | undefined;

export function librosa(): Promise<Librosa> {
  instance ??= createLibrosa();
  return instance;
}

export function vector(values: ArrayLike<number>): Float64Array {
  return Float64Array.from(Array.from(values));
}

export function matrix(rows: number, cols: number, values: ArrayLike<number>): Matrix {
  const data = Float64Array.from(Array.from(values));
  assert.equal(data.length, rows * cols);
  return { rows, cols, data };
}

export function complexMatrix(rows: number, cols: number, values: ArrayLike<number>): ComplexMatrix {
  const data = Float64Array.from(Array.from(values));
  assert.equal(data.length, rows * cols * 2);
  return { rows, cols, data };
}

export function sine(length: number, frequency = 440, sr = 22050): Float64Array {
  const y = new Float64Array(length);
  for (let i = 0; i < y.length; i += 1) {
    y[i] = Math.sin((2 * Math.PI * frequency * i) / sr);
  }
  return y;
}

export function impulseTrain(length: number, period: number): Float64Array {
  const y = new Float64Array(length);
  for (let i = 0; i < y.length; i += period) {
    y[i] = 1;
  }
  return y;
}

export function assertClose(actual: number, expected: number, tolerance = 1e-9): void {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `expected ${actual} to be within ${tolerance} of ${expected}`
  );
}

export function assertArrayClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance = 1e-9
): void {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < actual.length; i += 1) {
    assertClose(Number(actual[i]), Number(expected[i]), tolerance);
  }
}

export function assertFiniteMatrix(value: Matrix): void {
  assert.ok(value.rows > 0);
  assert.ok(value.cols > 0);
  assert.equal(value.data.length, value.rows * value.cols);
  for (const sample of value.data) {
    assert.ok(Number.isFinite(sample));
  }
}

export function assertFiniteComplexMatrix(value: ComplexMatrix): void {
  assert.ok(value.rows > 0);
  assert.ok(value.cols > 0);
  assert.equal(value.data.length, value.rows * value.cols * 2);
  for (const sample of value.data) {
    assert.ok(Number.isFinite(sample));
  }
}
