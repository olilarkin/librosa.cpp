import { copyFile, mkdir, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const webDir = resolve(scriptDir, "..");
const repoRoot = resolve(webDir, "../..");
const cacheDir = resolve(repoRoot, ".cache/librosa-data");
const outputDir = resolve(webDir, "audio");
const remote = "https://github.com/librosa/data.git";

const sources = [
  {
    input: "audio/admiralbob77_-_Choice_-_Drum-bass.ogg",
    output: "choice-drum-bass.ogg"
  },
  {
    input: "audio/sorohanro_-_solo-trumpet-06.ogg",
    output: "solo-trumpet.ogg"
  },
  {
    input: "audio/442789__lena-orsa__happy-music-pistachio-ice-cream-ragtime.ogg",
    output: "pistachio-ragtime.ogg"
  },
  {
    input: "audio/5703-47212-0000.ogg",
    output: "libri-speech.ogg"
  }
];

const usesProvidedDataDir = Boolean(process.env.LIBROSA_DATA_DIR);
const dataDir = resolve(process.env.LIBROSA_DATA_DIR ?? cacheDir);

if (!existsSync(join(dataDir, ".git"))) {
  await cloneDataRepo(dataDir);
}

if (!usesProvidedDataDir) {
  checkoutAudioFiles(dataDir);
}
await mkdir(outputDir, { recursive: true });

for (const source of sources) {
  const from = join(dataDir, source.input);
  const to = join(outputDir, source.output);
  if (!existsSync(from)) {
    throw new Error(`Missing source audio: ${from}`);
  }
  await copyFile(from, to);
  console.log(`${source.input} -> ${to}`);
}

async function cloneDataRepo(target) {
  await rm(target, { recursive: true, force: true });
  await mkdir(dirname(target), { recursive: true });
  execFileSync("git", ["clone", "--depth", "1", "--filter=blob:none", "--sparse", remote, target], {
    stdio: "inherit"
  });
}

function checkoutAudioFiles(target) {
  execFileSync("git", ["-C", target, "sparse-checkout", "set", "--no-cone", ...sources.map((source) => `/${source.input}`)], {
    stdio: "inherit"
  });
}
