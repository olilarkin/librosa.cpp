# Librosa WASM Gallery Showcase

This is a dependency-free browser showcase for the local
`@olilarkin/librosa-wasm` package. It provides JavaScript versions of selected
librosa gallery examples, runs the audio analysis in a Web Worker, and renders
the analysis outputs with Canvas on the UI thread.

Run it from this directory:

```bash
npm run prepare:audio
npm start
```

Then open:

```text
http://localhost:5175/examples/web/
```

The app imports the package by name through an import map:

```js
import { createLibrosa } from "@olilarkin/librosa-wasm";
```

The server root is the repository root so the package's generated
`dist/wasm/librosa_wasm.wasm` can be fetched by the Emscripten module.

`npm run prepare:audio` copies a small selected set of clips from
`librosa/data` into `examples/web/audio/`. Set `LIBROSA_DATA_DIR` to reuse a
local checkout:

```bash
LIBROSA_DATA_DIR=/Users/oli/Dev/librosa_data npm run prepare:audio
```

The copied audio directory is ignored by git, so the large data repo does not
need to be committed or added as a submodule. If the files are absent, the demo
falls back to generated audio.

The GitHub Pages workflow builds the WASM package, runs `npm run prepare:audio`,
and publishes the demo at the Pages root. It copies only the web demo, the
package `dist/` output, the selected audio clips, and the logo image into the
deployment artifact.
