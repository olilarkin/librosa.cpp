#!/usr/bin/env node
// Post-build step: emit a CommonJS build alongside the ESM one.
//
// Emscripten is configured with -sEXPORT_ES6=1 -sMODULARIZE=1, so the only
// glue it produces is an ES module (dist/wasm/librosa_wasm.mjs). CommonJS
// consumers (Electron main, Ableton Extension Host, plain `require()` in Node)
// cannot load that. Rather than ask emscripten to emit a second flavour (which
// would need a second, slow wasm link), we transform the existing ESM glue into
// a `.cjs` with a handful of pure-text substitutions, then write a CJS wrapper
// that mirrors src/index.ts. This runs after `build:wasm` + `build:ts`, needs no
// Emscripten toolchain, and is therefore safe to run in CI on the npm publish job.

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const pkgRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

/** Apply a substitution, failing loudly if the expected text is missing. */
function substitute(source, find, replace, label) {
  if (typeof find === "string") {
    if (!source.includes(find)) {
      throw new Error(
        `make-cjs: expected to find ${label} but it was not present — ` +
          `the emscripten glue likely changed; update scripts/make-cjs.mjs.`
      );
    }
    return source.split(find).join(replace);
  }
  if (!find.test(source)) {
    throw new Error(
      `make-cjs: expected to match ${label} but nothing matched — ` +
        `the emscripten glue likely changed; update scripts/make-cjs.mjs.`
    );
  }
  return source.replace(find, replace);
}

// --- 1. Convert the emscripten ESM glue to CommonJS -------------------------
//
// The transform must tolerate different emscripten releases, which emit the
// Node `require` shim two different ways:
//   * 3.1.x: a top-level static `import{createRequire}from"module";` followed by
//            `var require=createRequire(import.meta.url);`
//   * 4.0.x: an inline `const{createRequire}=await import("module");` +
//            `var require=createRequire(import.meta.url)` inside the Node branch
// Both forms are removed (a .cjs already has a real `require`), then every
// `import.meta.url` and the `export default` are rewritten. A post-conversion
// check fails loudly if any ES-module construct survives, so a future emscripten
// upgrade can't silently produce a broken `.cjs`.
{
  const mjsPath = resolve(pkgRoot, "dist/wasm/librosa_wasm.mjs");
  let glue = readFileSync(mjsPath, "utf8");

  // A file: URL for the .cjs itself, so `new URL("librosa_wasm.wasm", …)` still
  // resolves the sibling wasm binary at runtime. Named without an "import"
  // substring so the leftover-ESM check below stays unambiguous.
  const banner =
    `const __cjsScriptUrl = require("url").pathToFileURL(__filename).href;\n`;

  // Drop the createRequire shim in either form, plus the `require` assignment it
  // feeds. These are best-effort removals; the assertions below verify the net
  // result regardless of which (if any) matched.
  glue = glue
    .replace(/import\s*\{\s*createRequire\s*\}\s*from\s*["']module["']\s*;?/g, "")
    .replace(/const\s*\{\s*createRequire\s*\}\s*=\s*await\s+import\(\s*["']module["']\s*\)\s*;?/g, "")
    .replace(/\b(?:var|const|let)\s+require\s*=\s*createRequire\(\s*import\.meta\.url\s*\)\s*;?/g, "");

  // Every remaining `import.meta.url` -> our CJS stand-in.
  glue = substitute(glue, /import\.meta\.url/g, "__cjsScriptUrl", "import.meta.url");

  // `export default <name>;` -> `module.exports = <name>;`
  glue = substitute(glue, /export\s+default\s+([A-Za-z0-9_$]+)\s*;?/, "module.exports = $1;", "the default export");

  // Fail loudly if any ESM construct survived the conversion.
  const leftovers = [
    [/\bcreateRequire\b/, "a stray createRequire reference"],
    [/import\.meta/, "a stray import.meta"],
    [/\bexport\s+(?:default|\{|\*|const|function|class)/, "a stray ESM export"],
    [/(?:^|[;\n}])\s*import\s*[{*"'A-Za-z]/, "a stray static ESM import"]
  ];
  for (const [re, label] of leftovers) {
    if (re.test(glue)) {
      throw new Error(
        `make-cjs: ${label} survived ESM->CJS conversion — ` +
          `the emscripten glue changed; update scripts/make-cjs.mjs.`
      );
    }
  }

  writeFileSync(resolve(pkgRoot, "dist/wasm/librosa_wasm.cjs"), banner + glue);
}

// --- 2. Emit a CommonJS wrapper mirroring the compiled ESM entry ------------
//
// The wrapper logic (arity table + Proxy) is identical; only the module syntax
// differs. We transform the already-compiled dist/src/index.js so the table
// stays single-sourced from src/index.ts.
{
  const esmPath = resolve(pkgRoot, "dist/src/index.js");
  let wrapper = readFileSync(esmPath, "utf8");

  wrapper = substitute(
    wrapper,
    `import createModule from "../wasm/librosa_wasm.mjs";`,
    `"use strict";\nconst createModule = require("../wasm/librosa_wasm.cjs");`,
    "the wasm module import"
  );

  wrapper = substitute(
    wrapper,
    `export async function createLibrosa`,
    `async function createLibrosa`,
    "the createLibrosa export"
  );

  wrapper = substitute(
    wrapper,
    `export default createLibrosa;`,
    `module.exports = createLibrosa;\nmodule.exports.createLibrosa = createLibrosa;\nmodule.exports.default = createLibrosa;`,
    "the default export"
  );

  // The sourcemap belongs to the ESM build; drop the now-wrong reference.
  wrapper = wrapper.replace(/\n?\/\/# sourceMappingURL=index\.js\.map\s*$/, "\n");

  writeFileSync(resolve(pkgRoot, "dist/src/index.cjs"), wrapper);
}

// --- 3. Emit type declarations matching the CJS `module.exports` -------------
//
// The ESM `index.d.ts` describes a default + named export. The CJS runtime sets
// `module.exports = createLibrosa` (a callable), so the .d.cts must use
// `export =` with a merged namespace; otherwise `import x = require(pkg)` under
// node16/nodenext would type the require as a non-callable namespace object.
// The re-exported type names are derived from types.d.ts so this stays in sync.
{
  const typesDts = readFileSync(resolve(pkgRoot, "dist/src/types.d.ts"), "utf8");
  const typeNames = [
    ...typesDts.matchAll(/^export\s+(?:declare\s+)?(?:type|interface|class|enum)\s+([A-Za-z0-9_]+)/gm)
  ].map((m) => m[1]);
  if (typeNames.length === 0) {
    throw new Error("make-cjs: found no exported type names in types.d.ts");
  }

  const indent = (names) => names.map((n) => `  ${n}`).join(",\n");
  const dcts =
    `import type {\n${indent(typeNames)}\n} from "./types.js";\n` +
    `declare function createLibrosa(options?: CreateLibrosaOptions): Promise<Librosa>;\n` +
    `declare namespace createLibrosa {\n` +
    `  export {\n` +
    `    createLibrosa,\n` +
    `    createLibrosa as default,\n` +
    `${typeNames.map((n) => `    ${n}`).join(",\n")}\n` +
    `  };\n` +
    `}\n` +
    `export = createLibrosa;\n`;

  writeFileSync(resolve(pkgRoot, "dist/src/index.d.cts"), dcts);
}

console.log("make-cjs: wrote dist/wasm/librosa_wasm.cjs, dist/src/index.cjs, dist/src/index.d.cts");
