# @olilarkin/librosa-wasm

`librosa.cpp` compiled to WebAssembly with WASM SIMD and PFFFT.

```ts
import { createLibrosa } from "@olilarkin/librosa-wasm";

const librosa = await createLibrosa();
const y = librosa.tone(440, { sr: 22050, duration: 1.0 });
const mfcc = librosa.mfcc(y, { sr: 22050, nMfcc: 13 });
```

Matrices are row-major objects:

```ts
type Matrix = { rows: number; cols: number; data: Float64Array };
type ComplexMatrix = { rows: number; cols: number; data: Float64Array }; // interleaved real, imag
```

This build uses the `pffft` FFT backend. PFFFT is SIMD-friendly and works well
for the FFT sizes used by the default librosa workflows, but it only supports
FFT lengths factorable by small primes. Prefer power-of-two FFT sizes such as
`512`, `1024`, `2048`, and `4096`.

## Local build

```bash
npm install
npm run build
npm test
```

The build expects Emscripten, CMake, Ninja, and the repository submodules to be
available.
