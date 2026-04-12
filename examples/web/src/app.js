const GALLERY_ROOT = "https://librosa.org/librosa_gallery/auto_examples/";
const TARGET_SR = 22050;

const examples = [
  {
    id: "presets",
    number: "01",
    title: "Presets",
    category: "Default parameter sweeps",
    link: `${GALLERY_ROOT}plot_presets.html`,
    runner: runPresets,
    snippet: `const y441 = librosa.resample(y, { origSr: sr, targetSr: 44100 });
const M = librosa.melspectrogram(y441, {
  sr: 44100, nFft: 4096, hopLength: 1024, nMels: 80
});
const MHighres = librosa.melspectrogram(y441, {
  sr: 44100, nFft: 4096, hopLength: 512, nMels: 80
});
const y11025 = librosa.resample(y441, { origSr: 44100, targetSr: 11025 });
const MLow = librosa.melspectrogram(y11025, {
  sr: 11025, nFft: 4096, hopLength: 1024, nMels: 80
});`
  },
  {
    id: "superflux",
    number: "02",
    title: "Superflux onsets",
    category: "Onset strength and peak picking",
    link: `${GALLERY_ROOT}plot_superflux.html`,
    runner: runSuperflux,
    snippet: `const hopLength = Math.round(sr / 200);
const S = librosa.melspectrogram(y, {
  sr, nFft: 1024, hopLength, nMels: 96, fmin: 27.5, fmax: 8000
});
const logMel = relativeDb(librosa.powerToDb(S, { topDb: null }));
const spectralFlux = librosa.onsetStrength(y, { sr, nFft: 1024, hopLength });
const superflux = librosa.onsetStrength(logMel, {
  sr, nFft: 1024, hopLength, lag: 2, maxSize: 3
});
const events = librosa.onsetDetect(superflux, {
  envelope: true, sr, hopLength
});
const eventTimes = Array.from(events, (frame) => frame * hopLength / sr);`
  },
  {
    id: "hpss",
    number: "03",
    title: "Harmonic-percussive source separation",
    category: "Median filtering and margins",
    link: `${GALLERY_ROOT}plot_hprss.html`,
    runner: runHpss,
    snippet: `const D = librosa.stft(y, { nFft: 2048, hopLength: 512 });
const magnitude = librosa.magnitude(D);
const { harmonic, percussive } = librosa.hpss(magnitude, {
  kernelSize: 31,
  margin: 1
});
const wideMargin = librosa.hpss(magnitude, {
  kernelSize: 31,
  margin: 8
});`
  },
  {
    id: "vocal",
    number: "04",
    title: "Vocal separation",
    category: "REPET-SIM style soft masks",
    link: `${GALLERY_ROOT}plot_vocal_separation.html`,
    runner: runVocalSeparation,
    snippet: `const D = librosa.stft(y, { nFft: 1024, hopLength: 256 });
const SFull = librosa.magnitude(D);
const SFilter = minimumMatrix(SFull, nnFilterMedian(SFull, {
  width: Math.round(sr / 256),
  neighbors: 10
}));
const residual = subtractMatrix(SFull, SFilter);
const maskI = librosa.softmask(SFilter, scaleMatrix(residual, 2), { power: 2 });
const maskV = librosa.softmask(residual, scaleMatrix(SFilter, 10), { power: 2 });
const background = multiplyMatrix(maskI, SFull);
const foreground = multiplyMatrix(maskV, SFull);`
  },
  {
    id: "sync",
    number: "05",
    title: "Music synchronization with DTW",
    category: "Chroma alignment",
    link: `${GALLERY_ROOT}plot_music_sync.html`,
    runner: runMusicSync,
    snippet: `const slow = firstSeconds(y, sr, 7);
const fast = librosa.timeStretch(slow, { rate: 1.32, nFft: 1024, hopLength: 256 });
const slowChroma = librosa.chromaStft(slow, {
  sr, nFft: 2048, hopLength: 512, tuning: 0, norm: 2
});
const fastChroma = librosa.chromaStft(fast, {
  sr, nFft: 2048, hopLength: 512, tuning: 0, norm: 2
});
const { cost, path } = librosa.dtw(slowChroma, fastChroma, { backtrack: true });`
  },
  {
    id: "chroma",
    number: "06",
    title: "Enhanced chroma",
    category: "Oversampling, harmonic isolation, smoothing",
    link: `${GALLERY_ROOT}plot_chroma.html`,
    runner: runEnhancedChroma,
    snippet: `const chromaOrig = librosa.chromaCqt(y, { sr, hopLength: 512 });
const chromaOver = librosa.chromaCqt(y, {
  sr, hopLength: 512, binsPerOctave: 36, nOctaves: 5
});
const yHarm = librosa.harmonicEffect(y, { margin: 8, nFft: 2048, hopLength: 512 });
const chromaHarm = librosa.chromaCqt(yHarm, {
  sr, hopLength: 512, binsPerOctave: 36, nOctaves: 5
});
const nonLocal = minimumMatrix(chromaHarm, nnFilterMedian(chromaHarm));
const smooth = librosa.medianFilter2d(nonLocal, { size: [1, 9], mode: "edge" });`
  },
  {
    id: "segmentation",
    number: "07",
    title: "Laplacian segmentation",
    category: "Recurrence graph and segment boundaries",
    link: `${GALLERY_ROOT}plot_segmentation.html`,
    runner: runSegmentation,
    snippet: `const C = librosa.chromaStft(y, { sr, nFft: 2048, hopLength: 512, norm: 2 });
const envelope = librosa.onsetStrength(y, { sr, nFft: 2048, hopLength: 512 });
const beats = librosa.beatTrack(envelope, {
  sr, hopLength: 512, startBpm: 112, trim: false
}).beats;
const beatFrames = librosa.fixFrames(beats, { xMin: 0, xMax: C.cols });
const Csync = syncMedian(C, beatFrames);
const R = librosa.recurrenceMatrix(Csync, {
  width: 2, mode: "affinity", sym: true, self: true
});
const enhanced = librosa.pathEnhance(R, { n: 5, nFilters: 5 });
const boundaries = librosa.agglomerative(Csync, { k: 5 });`
  }
];

const demoSources = [
  {
    id: "tutorial",
    title: "Choice drum+bass",
    detail: "Admiral Bob",
    url: "./audio/choice-drum-bass.ogg",
    create: () => createDemoSource(TARGET_SR, 12, "tutorial")
  },
  {
    id: "drums",
    title: "Solo trumpet",
    detail: "Mihai Sorohan",
    url: "./audio/solo-trumpet.ogg",
    create: () => createDemoSource(TARGET_SR, 12, "drums")
  },
  {
    id: "harmony",
    title: "Pistachio ragtime",
    detail: "The Piano Lady",
    url: "./audio/pistachio-ragtime.ogg",
    create: () => createDemoSource(TARGET_SR, 12, "harmony")
  },
  {
    id: "sparse",
    title: "Speech excerpt",
    detail: "LibriVox",
    url: "./audio/libri-speech.ogg",
    create: () => createDemoSource(TARGET_SR, 10, "sparse")
  }
];

const els = {
  codeSnippet: document.querySelector("#codeSnippet"),
  codePanel: document.querySelector(".codePanel"),
  demoSourceList: document.querySelector("#demoSourceList"),
  exampleCategory: document.querySelector("#exampleCategory"),
  exampleList: document.querySelector("#exampleList"),
  exampleTitle: document.querySelector("#exampleTitle"),
  generatedView: document.querySelector("#generatedView"),
  metrics: document.querySelector("#metrics"),
  plots: document.querySelector("#plots"),
  runTime: document.querySelector("#runTime"),
  sidebarToggle: document.querySelector("#sidebarToggle"),
  sourceName: document.querySelector("#sourceName"),
  sourcePlayButton: document.querySelector("#sourcePlayButton"),
  sourceStats: document.querySelector("#sourceStats"),
  sourceWaveform: document.querySelector("#sourceWaveform"),
  viewToggleButtons: document.querySelectorAll(".viewToggleButton")
};

const state = {
  activeId: "superflux",
  activeSourceId: "tutorial",
  worker: null,
  workerReady: false,
  nextRequestId: 1,
  pendingRequests: new Map(),
  currentPlots: [],
  audioContext: null,
  playbackStartedAt: 0,
  playbackSource: null,
  playbackFrame: 0,
  playing: false,
  running: false,
  source: demoSources[0].create(),
  viewMode: "generated"
};

renderExampleList();
renderDemoSourceList();
bindUi();
renderActiveExample();
renderSource();
renderViewMode();
boot();

async function boot() {
  try {
    setRuntimeStatus("Loading worker");
    const info = await initAnalysisWorker();
    setRuntimeStatus(`${info.fftBackend.toUpperCase()} ${info.wasmSimd ? "SIMD" : "scalar"}`, "ready");
    await loadDemoSource(activeDemoSource());
    renderSource();
    await runActiveExample();
  } catch (error) {
    setRuntimeStatus("WASM load failed", "error");
    showError(error);
  }
}

function bindUi() {
  els.sourcePlayButton.addEventListener("click", () => {
    toggleSourcePlayback();
  });

  els.sidebarToggle.addEventListener("click", () => {
    const collapsed = document.body.classList.toggle("sidebarCollapsed");
    els.sidebarToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
    els.sidebarToggle.setAttribute("aria-label", collapsed ? "Expand sidebar" : "Collapse sidebar");
    if (state.currentPlots.length) requestAnimationFrame(() => renderPlots(state.currentPlots));
  });

  for (const button of els.viewToggleButtons) {
    button.addEventListener("click", () => {
      state.viewMode = button.dataset.viewMode;
      renderViewMode();
    });
  }

  window.addEventListener("resize", () => {
    renderSource();
    if (state.currentPlots.length) renderPlots(state.currentPlots);
  });
}

function renderExampleList() {
  els.exampleList.replaceChildren(...examples.map((example) => {
    const button = document.createElement("button");
    button.className = "exampleButton";
    button.type = "button";
    button.dataset.exampleId = example.id;
    button.innerHTML = `
      <span class="exampleNumber">${example.number}</span>
      <span class="exampleText">
        <strong>${example.title}</strong>
        <span>${example.category}</span>
      </span>
    `;
    button.addEventListener("click", () => {
      if (state.running) return;
      state.activeId = example.id;
      renderActiveExample();
      runActiveExample();
    });
    return button;
  }));
}

function renderDemoSourceList() {
  els.demoSourceList.replaceChildren(...demoSources.map((source) => {
    const button = document.createElement("button");
    button.className = "demoSourceButton";
    button.type = "button";
    button.dataset.sourceId = source.id;
    button.innerHTML = `
      <strong>${source.title}</strong>
      <span>${source.detail}</span>
    `;
    button.addEventListener("click", async () => {
      if (state.running) return;
      stopSourcePlayback();
      state.activeSourceId = source.id;
      renderDemoSourceSelection();
      await loadDemoSource(source);
      renderSource();
      runActiveExample();
    });
    return button;
  }));
  renderDemoSourceSelection();
}

async function loadDemoSource(source) {
  if (!source.url || !state.workerReady) {
    state.source = source.create();
    return;
  }
  try {
    state.source = await loadAudioUrl(source.url, source.title);
  } catch (error) {
    console.warn(`Falling back to generated audio for ${source.title}`, error);
    state.source = source.create();
  }
}

function renderDemoSourceSelection() {
  for (const button of els.demoSourceList.querySelectorAll(".demoSourceButton")) {
    button.classList.toggle("active", button.dataset.sourceId === state.activeSourceId);
  }
}

function renderActiveExample() {
  const example = activeExample();
  for (const button of els.exampleList.querySelectorAll(".exampleButton")) {
    button.classList.toggle("active", button.dataset.exampleId === example.id);
  }
  els.exampleCategory.textContent = example.category;
  els.exampleTitle.textContent = example.title;
  els.codeSnippet.textContent = example.snippet;
}

function renderViewMode() {
  const showingCode = state.viewMode === "code";
  els.generatedView.hidden = showingCode;
  els.codePanel.hidden = !showingCode;
  for (const button of els.viewToggleButtons) {
    const active = button.dataset.viewMode === state.viewMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }
  if (!showingCode && state.currentPlots.length) {
    requestAnimationFrame(() => redrawCurrentPlots());
  }
}

function renderSource() {
  const { name, samples, sr, y } = state.source;
  els.sourceName.textContent = name;
  els.sourceStats.textContent = `${formatSeconds(samples / sr)}, ${formatInteger(samples)} samples, ${formatInteger(sr)} Hz`;
  updatePlaybackButton();
  drawWaveform(els.sourceWaveform, y, {
    color: "#f39a2f",
    duration: sourceDuration(),
    fill: "#1c1224",
    playheadTime: currentPlaybackTime()
  });
}

async function toggleSourcePlayback() {
  if (state.playing) {
    stopSourcePlayback();
    return;
  }
  await playSource();
}

async function playSource() {
  stopSourcePlayback();
  const { y, sr } = state.source;
  const context = await audioContext();
  const buffer = context.createBuffer(1, y.length, sr);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < y.length; i++) channel[i] = y[i];

  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.addEventListener("ended", () => {
    if (state.playbackSource === source) {
      state.playbackSource = null;
      state.playing = false;
      state.playbackStartedAt = 0;
      updatePlaybackButton();
      stopPlaybackAnimation();
      renderSource();
      if (state.currentPlots.length) redrawCurrentPlots();
    }
  });
  state.playbackSource = source;
  const startAt = context.currentTime + 0.02;
  state.playbackStartedAt = startAt;
  state.playing = true;
  updatePlaybackButton();
  source.start(startAt);
  startPlaybackAnimation();
}

async function audioContext() {
  if (!state.audioContext) {
    state.audioContext = new AudioContext();
  }
  if (state.audioContext.state === "suspended") {
    await state.audioContext.resume();
  }
  return state.audioContext;
}

function stopSourcePlayback() {
  if (state.playbackSource) {
    const source = state.playbackSource;
    state.playbackSource = null;
    source.onended = null;
    try {
      source.stop();
    } catch {
      // The source may already have reached its natural end.
    }
  }
  state.playing = false;
  state.playbackStartedAt = 0;
  updatePlaybackButton();
  stopPlaybackAnimation();
  renderSource();
  if (state.currentPlots.length) redrawCurrentPlots();
}

function updatePlaybackButton() {
  els.sourcePlayButton.textContent = state.playing ? "Stop" : "Play";
  els.sourcePlayButton.classList.toggle("playing", state.playing);
  els.sourcePlayButton.setAttribute("aria-label", state.playing ? "Stop source audio" : "Play source audio");
}

function startPlaybackAnimation() {
  stopPlaybackAnimation();
  const tick = () => {
    renderSource();
    if (state.currentPlots.length) redrawCurrentPlots();
    if (state.playing) {
      state.playbackFrame = requestAnimationFrame(tick);
    }
  };
  state.playbackFrame = requestAnimationFrame(tick);
}

function stopPlaybackAnimation() {
  if (state.playbackFrame) {
    cancelAnimationFrame(state.playbackFrame);
    state.playbackFrame = 0;
  }
}

function currentPlaybackTime() {
  if (!state.playing || !state.audioContext) return null;
  return Math.min(sourceDuration(), Math.max(0, state.audioContext.currentTime - state.playbackStartedAt));
}

function sourceDuration() {
  return state.source.samples / state.source.sr;
}

function redrawCurrentPlots() {
  const canvases = els.plots.querySelectorAll("canvas[data-plot-index]");
  for (const canvas of canvases) {
    const plot = state.currentPlots[Number(canvas.dataset.plotIndex)];
    if (plot) drawPlot(canvas, plot);
  }
}

async function runActiveExample() {
  if (!state.workerReady || state.running) return;
  const example = activeExample();
  state.running = true;
  els.plots.dataset.redraw = example.id;
  clearOutputs();
  showLoading(`Analyzing ${example.title}`);

  const started = performance.now();
  try {
    const requestSource = transferableSource(state.source);
    const output = await requestWorker("run", {
      exampleId: example.id,
      source: requestSource
    }, [requestSource.y.buffer]);
    renderMetrics(output.metrics ?? []);
    renderPlots(output.plots ?? []);
    els.runTime.textContent = `${Math.round(performance.now() - started)} ms`;
  } catch (error) {
    showError(error);
  } finally {
    state.running = false;
  }
}

async function loadAudioUrl(url, name) {
  setRuntimeStatus("Decoding audio");
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Audio not found: ${url}`);
  }
  const context = new AudioContext();
  const buffer = await context.decodeAudioData(await response.arrayBuffer());
  await context.close();
  let audio = downmix(buffer);
  let sr = buffer.sampleRate;

  showLoading("Preparing audio");
  const prepared = await requestWorker("prepareSource", {
    y: audio,
    sr,
    targetSr: TARGET_SR,
    maxSeconds: 20
  }, [audio.buffer]);
  audio = prepared.y;
  sr = prepared.sr;
  setRuntimeStatus("Audio ready", "ready");
  return {
    name,
    samples: audio.length,
    sr,
    y: audio
  };
}

function clearOutputs() {
  els.metrics.replaceChildren();
  els.plots.replaceChildren();
  els.runTime.textContent = "--";
}

function showLoading(message) {
  els.plots.replaceChildren();
  const box = document.createElement("div");
  box.className = "loadingState";
  box.setAttribute("role", "status");
  box.setAttribute("aria-live", "polite");
  box.textContent = message;
  els.plots.append(box);
}

function showError(error) {
  const message = error instanceof Error ? error.message : String(error);
  els.plots.replaceChildren();
  const box = document.createElement("div");
  box.className = "emptyState";
  box.textContent = message;
  els.plots.append(box);
}

function renderMetrics(metrics) {
  els.metrics.replaceChildren(...metrics.map(({ label, value }) => {
    const item = document.createElement("div");
    item.className = "metric";
    item.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    return item;
  }));
}

function renderPlots(plots) {
  state.currentPlots = plots;
  els.plots.replaceChildren(...plots.map((plot, index) => {
    const item = document.createElement("article");
    item.className = `plot ${plot.wide ? "wide" : ""} ${plot.tall ? "tall" : ""} ${plot.compact ? "compact" : ""}`;
    const header = document.createElement("div");
    header.className = "plotHead";
    header.innerHTML = `<h3>${plot.title}</h3><span class="plotMeta">${plot.meta ?? ""}</span>`;
    const canvas = document.createElement("canvas");
    canvas.dataset.plotIndex = `${index}`;
    item.append(header, canvas);
    requestAnimationFrame(() => drawPlot(canvas, plot));
    return item;
  }));
}

function drawPlot(canvas, plot) {
  const playhead = {
    duration: plot.duration ?? sourceDuration(),
    playheadTime: currentPlaybackTime()
  };
  if (typeof plot.draw === "function") {
    plot.draw(canvas);
    if (plot.playhead !== false) drawCanvasPlayhead(canvas, playhead);
    return;
  }
  switch (plot.type) {
    case "heatmap":
      drawMatrix(canvas, plot.matrix, { ...playhead, ...(plot.options ?? {}) });
      break;
    case "line":
      drawLineSeries(canvas, plot.series, { ...playhead, ...(plot.options ?? {}) });
      break;
    case "dtw":
      drawMatrix(canvas, plot.cost, { palette: "gray" });
      drawPath(canvas, plot.path, plot.cost.rows, plot.cost.cols);
      drawCanvasPlayhead(canvas, playhead);
      break;
    case "alignment":
      drawAlignment(canvas, plot.slow, plot.fast, plot.sr, plot.hopLength, plot.path);
      drawCanvasPlayhead(canvas, playhead);
      break;
    case "segments":
      drawSegments(canvas, plot.duration, plot.frames, plot.sr, plot.hopLength);
      drawCanvasPlayhead(canvas, { duration: plot.duration, playheadTime: currentPlaybackTime() });
      break;
    default:
      throw new Error(`Unknown plot type: ${plot.type}`);
  }
}

function initAnalysisWorker() {
  state.worker = new Worker(new URL("./analysis-worker.js", import.meta.url), { type: "module" });
  state.worker.addEventListener("message", handleWorkerMessage);
  state.worker.addEventListener("error", (event) => {
    rejectAllWorkerRequests(new Error(event.message || "Analysis worker failed"));
  });
  return requestWorker("init").then((info) => {
    state.workerReady = true;
    return info;
  });
}

function requestWorker(type, payload = {}, transfer = []) {
  if (!state.worker) throw new Error("Analysis worker is not available");
  const id = state.nextRequestId++;
  const promise = new Promise((resolve, reject) => {
    state.pendingRequests.set(id, { resolve, reject });
  });
  state.worker.postMessage({ id, type, payload }, transfer);
  return promise;
}

function handleWorkerMessage(event) {
  const { id, ok, payload, error } = event.data ?? {};
  const pending = state.pendingRequests.get(id);
  if (!pending) return;
  state.pendingRequests.delete(id);
  if (ok) pending.resolve(payload);
  else pending.reject(new Error(error || "Analysis worker request failed"));
}

function rejectAllWorkerRequests(error) {
  for (const pending of state.pendingRequests.values()) pending.reject(error);
  state.pendingRequests.clear();
}

function transferableSource(source) {
  const y = source.y.slice();
  return {
    name: source.name,
    samples: source.samples,
    sr: source.sr,
    y
  };
}

function setRuntimeStatus(text, type = "") {
  void text;
  void type;
}

function activeExample() {
  return examples.find((example) => example.id === state.activeId) ?? examples[0];
}

function activeDemoSource() {
  return demoSources.find((source) => source.id === state.activeSourceId) ?? demoSources[0];
}

async function runPresets(librosa, source) {
  const clip = firstSeconds(source.y, source.sr, 5);
  const duration = clip.length / source.sr;
  const y441 = ensureSampleRate(librosa, clip, source.sr, 44100);
  const y11025 = ensureSampleRate(librosa, y441, 44100, 11025);

  const base = librosa.melspectrogram(y441, {
    sr: 44100,
    nFft: 4096,
    hopLength: 1024,
    nMels: 80
  });
  const highres = librosa.melspectrogram(y441, {
    sr: 44100,
    nFft: 4096,
    hopLength: 512,
    nMels: 80
  });
  const low = librosa.melspectrogram(y11025, {
    sr: 11025,
    nFft: 4096,
    hopLength: 1024,
    nMels: 80
  });

  return {
    metrics: [
      { label: "44100 / 1024", value: shape(base) },
      { label: "44100 / 512", value: shape(highres) },
      { label: "11025 / 1024", value: shape(low) },
      { label: "n_fft", value: "4096" }
    ],
    plots: [
      heatmapPlot("44100 / 1024 / 4096", "mel dB", relativeDb(librosa.powerToDb(base, { topDb: null })), { duration }),
      heatmapPlot("44100 / 512 / 4096", "mel dB", relativeDb(librosa.powerToDb(highres, { topDb: null })), { duration }),
      heatmapPlot("11025 / 1024 / 4096", "mel dB", relativeDb(librosa.powerToDb(low, { topDb: null })), { wide: true, duration })
    ]
  };
}

async function runSuperflux(librosa, source) {
  const y = firstSeconds(source.y, source.sr, 7);
  const sr = source.sr;
  const duration = y.length / sr;
  const nFft = 1024;
  const hopLength = Math.max(64, Math.round(sr / 200));
  const fmax = Math.min(8000, sr / 2 - 100);
  const mel = librosa.melspectrogram(y, {
    sr,
    nFft,
    hopLength,
    fmin: 27.5,
    fmax,
    nMels: 96
  });
  const melDb = relativeDb(librosa.powerToDb(mel, { topDb: null }));
  const spectralFlux = librosa.onsetStrength(y, { sr, nFft, hopLength });
  const defaultEvents = librosa.onsetDetect(spectralFlux, {
    envelope: true,
    sr,
    hopLength,
    preMax: 3,
    postMax: 3,
    preAvg: 3,
    postAvg: 5,
    delta: 0.05,
    wait: 4
  });
  const superflux = librosa.onsetStrength(melDb, {
    sr,
    nFft,
    hopLength,
    lag: 2,
    maxSize: 3
  });
  const superEvents = librosa.onsetDetect(superflux, {
    envelope: true,
    sr,
    hopLength,
    preMax: 3,
    postMax: 3,
    preAvg: 3,
    postAvg: 5,
    delta: 0.05,
    wait: 4
  });

  return {
    metrics: [
      { label: "Mel spectrogram", value: shape(mel) },
      { label: "Default onsets", value: `${defaultEvents.length}` },
      { label: "Superflux onsets", value: `${superEvents.length}` },
      { label: "Hop", value: `${hopLength}` }
    ],
    plots: [
      heatmapPlot("Mel spectrogram", "power to dB", melDb, { wide: true, duration }),
      linePlot("Onset envelopes", "spectral flux vs Superflux", [
        { label: "Spectral flux", values: spectralFlux, color: "#f39a2f" },
        { label: "Superflux", values: superflux, color: "#d13d8c" }
      ], {
        wide: true,
        duration,
        markers: [
          ...Array.from(defaultEvents, (frame) => ({ x: (frame * hopLength) / sr, maxX: y.length / sr, color: "rgba(243, 154, 47, 0.45)" })),
          ...Array.from(superEvents, (frame) => ({ x: (frame * hopLength) / sr, maxX: y.length / sr, color: "rgba(209, 61, 140, 0.55)" }))
        ]
      })
    ]
  };
}

async function runHpss(librosa, source) {
  const y = firstSeconds(source.y, source.sr, 10);
  const duration = y.length / source.sr;
  const nFft = 2048;
  const hopLength = 512;
  const d = librosa.stft(y, { nFft, hopLength });
  const mag = librosa.magnitude(d);
  const base = librosa.hpss(mag, { kernelSize: 31, margin: 1 });
  const margin8 = librosa.hpss(mag, { kernelSize: 31, margin: 8 });

  return {
    metrics: [
      { label: "STFT", value: shape(mag) },
      { label: "Margin 1 harmonic", value: `${percentEnergy(base.harmonic, mag)}%` },
      { label: "Margin 1 percussive", value: `${percentEnergy(base.percussive, mag)}%` },
      { label: "Margin 8 harmonic", value: `${percentEnergy(margin8.harmonic, mag)}%` }
    ],
    plots: [
      heatmapPlot("Full spectrogram", "amplitude dB", relativeDb(librosa.amplitudeToDb(mag, { topDb: null })), { duration }),
      heatmapPlot("Harmonic spectrogram", "margin 1", relativeDb(librosa.amplitudeToDb(base.harmonic, { topDb: null })), { duration }),
      heatmapPlot("Percussive spectrogram", "margin 1", relativeDb(librosa.amplitudeToDb(base.percussive, { topDb: null })), { duration }),
      heatmapPlot("Harmonic vs percussive", "margin 8", matrixDifference(
        relativeDb(librosa.amplitudeToDb(margin8.harmonic, { topDb: null })),
        relativeDb(librosa.amplitudeToDb(margin8.percussive, { topDb: null }))
      ), { duration })
    ]
  };
}

async function runVocalSeparation(librosa, source) {
  const y = firstSeconds(source.y, source.sr, 12);
  const sr = source.sr;
  const duration = y.length / sr;
  const nFft = 1024;
  const hopLength = 256;
  const d = librosa.stft(y, { nFft, hopLength });
  const full = librosa.magnitude(d);
  const width = Math.max(2, Math.round(sr / hopLength));
  const filtered = minimumMatrix(full, nnFilterMedian(full, {
    width,
    neighbors: Math.min(12, Math.max(2, Math.floor(full.cols / 4)))
  }));
  const foregroundResidual = clampMatrix(subtractMatrix(full, filtered), 0);
  const maskI = librosa.softmask(filtered, scaleMatrix(foregroundResidual, 2), { power: 2 });
  const maskV = librosa.softmask(foregroundResidual, scaleMatrix(filtered, 10), { power: 2 });
  const background = multiplyMatrix(maskI, full);
  const foreground = multiplyMatrix(maskV, full);

  return {
    metrics: [
      { label: "Spectrum", value: shape(full) },
      { label: "Filter width", value: `${width} frames` },
      { label: "Foreground energy", value: `${percentEnergy(foreground, full)}%` },
      { label: "Background energy", value: `${percentEnergy(background, full)}%` }
    ],
    plots: [
      heatmapPlot("Input spectrum", "amplitude dB", relativeDb(librosa.amplitudeToDb(full, { topDb: null })), { duration }),
      heatmapPlot("Nearest-neighbor filter", "median aggregate", relativeDb(librosa.amplitudeToDb(filtered, { topDb: null })), { duration }),
      heatmapPlot("Foreground mask", "sporadic component", maskV, { duration }),
      heatmapPlot("Foreground spectrum", "masked magnitude", relativeDb(librosa.amplitudeToDb(foreground, { topDb: null })), { duration })
    ]
  };
}

async function runMusicSync(librosa, source) {
  const sr = source.sr;
  const slow = firstSeconds(source.y, sr, 7);
  const fast = librosa.timeStretch(slow, { rate: 1.32, nFft: 1024, hopLength: 256 });
  const slowDuration = slow.length / sr;
  const fastDuration = fast.length / sr;
  const hopLength = 512;
  const slowChroma = librosa.chromaStft(slow, {
    sr,
    nFft: 2048,
    hopLength,
    tuning: 0,
    norm: 2
  });
  const fastChroma = librosa.chromaStft(fast, {
    sr,
    nFft: 2048,
    hopLength,
    tuning: 0,
    norm: 2
  });
  const alignment = librosa.dtw(slowChroma, fastChroma, { backtrack: true });

  return {
    metrics: [
      { label: "Slow chroma", value: shape(slowChroma) },
      { label: "Fast chroma", value: shape(fastChroma) },
      { label: "DTW cost", value: shape(alignment.cost) },
      { label: "Path pairs", value: `${alignment.path.rows}` }
    ],
    plots: [
      heatmapPlot("Slow chroma", "source timing", slowChroma, { duration: slowDuration }),
      heatmapPlot("Fast chroma", "time-stretched", fastChroma, { duration: fastDuration, playhead: false }),
      dtwPlot("Accumulated cost", "warping path", alignment.cost, alignment.path, { playhead: false }),
      alignmentPlot("Time-domain alignment", "path correspondences", slow, fast, sr, hopLength, alignment.path, { duration: slowDuration })
    ]
  };
}

async function runEnhancedChroma(librosa, source) {
  const y = firstSeconds(source.y, source.sr, 8);
  const sr = source.sr;
  const duration = y.length / sr;
  const hopLength = 512;
  const orig = librosa.chromaCqt(y, { sr, hopLength, nOctaves: 5 });
  const over = librosa.chromaCqt(y, {
    sr,
    hopLength,
    binsPerOctave: 36,
    nOctaves: 5
  });
  const harmonic = librosa.harmonicEffect(y, {
    margin: 8,
    nFft: 2048,
    hopLength
  });
  const harm = librosa.chromaCqt(harmonic, {
    sr,
    hopLength,
    binsPerOctave: 36,
    nOctaves: 5
  });
  const filtered = minimumMatrix(harm, nnFilterMedian(harm, {
    width: 2,
    neighbors: Math.min(8, Math.max(2, Math.floor(harm.cols / 4)))
  }));
  const smooth = librosa.medianFilter2d(filtered, { size: [1, 9], mode: "edge" });

  return {
    metrics: [
      { label: "Original chroma", value: shape(orig) },
      { label: "3x oversampled", value: shape(over) },
      { label: "Harmonic audio", value: `${formatSeconds(harmonic.length / sr)} s` },
      { label: "Median width", value: "9 frames" }
    ],
    plots: [
      heatmapPlot("Original chroma", "chroma CQT", orig, { duration }),
      heatmapPlot("3x oversampled", "36 bins/octave", over, { duration }),
      heatmapPlot("Harmonic chroma", "margin 8", harm, { duration }),
      heatmapPlot("Smoothed chroma", "non-local + median", smooth, { duration })
    ]
  };
}

async function runSegmentation(librosa, source) {
  const y = firstSeconds(source.y, source.sr, 12);
  const sr = source.sr;
  const duration = y.length / sr;
  const hopLength = 512;
  const chroma = librosa.chromaStft(y, {
    sr,
    nFft: 2048,
    hopLength,
    norm: 2
  });
  const envelope = librosa.onsetStrength(y, {
    sr,
    nFft: 2048,
    hopLength
  });
  const beatResult = librosa.beatTrack(envelope, {
    sr,
    hopLength,
    startBpm: 112,
    trim: false
  });
  const beatFrames = robustBeatFrames(librosa, beatResult.beats, chroma.cols);
  const sync = syncMedian(chroma, beatFrames);
  const recurrence = librosa.recurrenceMatrix(sync, {
    k: Math.min(4, Math.max(1, sync.cols - 1)),
    width: Math.min(2, Math.max(1, sync.cols - 1)),
    mode: "affinity",
    sym: true,
    self: true
  });
  const enhanced = librosa.pathEnhance(recurrence, {
    n: 5,
    nFilters: 5,
    clip: true
  });
  const boundaries = Array.from(librosa.agglomerative(sync, {
    k: Math.min(5, Math.max(1, sync.cols))
  })).filter((value) => Number.isFinite(value));
  const frameBoundaries = uniqueSorted([
    0,
    ...boundaries.map((value) => beatFrames[Math.max(0, Math.min(beatFrames.length - 1, Math.round(value)))] ?? 0),
    chroma.cols
  ]);

  return {
    metrics: [
      { label: "Tempo", value: `${formatNumber(beatResult.tempo)} BPM` },
      { label: "Beat frames", value: `${beatFrames.length}` },
      { label: "Sync chroma", value: shape(sync) },
      { label: "Segments", value: `${Math.max(1, frameBoundaries.length - 1)}` }
    ],
    plots: [
      heatmapPlot("Beat-synchronous chroma", "median per beat span", sync, { wide: true, duration }),
      heatmapPlot("Recurrence similarity", "affinity graph", recurrence, { playhead: false }),
      heatmapPlot("Path-enhanced graph", "diagonal continuity", enhanced, { playhead: false }),
      segmentPlot("Segment boundaries", "agglomerative over beat features", duration, frameBoundaries, sr, hopLength, { wide: true, compact: true })
    ]
  };
}

function heatmapPlot(title, meta, matrix, options = {}) {
  return {
    title,
    meta,
    ...options,
    draw(canvas) {
      drawMatrix(canvas, matrix, options);
    }
  };
}

function linePlot(title, meta, series, options = {}) {
  return {
    title,
    meta,
    ...options,
    draw(canvas) {
      drawLineSeries(canvas, series, options);
    }
  };
}

function dtwPlot(title, meta, cost, path, options = {}) {
  return {
    title,
    meta,
    ...options,
    draw(canvas) {
      drawMatrix(canvas, cost, { palette: "gray" });
      drawPath(canvas, path, cost.rows, cost.cols);
    }
  };
}

function alignmentPlot(title, meta, slow, fast, sr, hopLength, path, options = {}) {
  return {
    title,
    meta,
    ...options,
    draw(canvas) {
      drawAlignment(canvas, slow, fast, sr, hopLength, path);
    }
  };
}

function segmentPlot(title, meta, duration, frames, sr, hopLength, options = {}) {
  return {
    title,
    meta,
    ...options,
    draw(canvas) {
      drawSegments(canvas, duration, frames, sr, hopLength);
    }
  };
}

function drawMatrix(canvas, matrix, options = {}) {
  const { width, height, dpr } = fitCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(width, height);
  const [min, max] = matrixExtent(matrix, options);
  const span = max - min || 1;
  const palette = options.palette ?? "magma";

  for (let py = 0; py < height; py++) {
    const row = Math.min(matrix.rows - 1, Math.max(0, Math.floor((1 - py / Math.max(1, height - 1)) * matrix.rows)));
    for (let px = 0; px < width; px++) {
      const col = Math.min(matrix.cols - 1, Math.max(0, Math.floor((px / Math.max(1, width - 1)) * matrix.cols)));
      const value = matrix.data[row * matrix.cols + col];
      const t = clamp01((value - min) / span);
      const [r, g, b] = colorRamp(t, palette);
      const offset = (py * width + px) * 4;
      image.data[offset] = r;
      image.data[offset + 1] = g;
      image.data[offset + 2] = b;
      image.data[offset + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  drawAxes(ctx, width, height, dpr);
  drawPlayhead(ctx, width, height, dpr, options);
}

function drawWaveform(canvas, y, options = {}) {
  const { width, height, dpr } = fitCanvas(canvas);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = options.fill ?? "#160f1f";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = options.color ?? "#f39a2f";
  ctx.lineWidth = Math.max(1, dpr);
  ctx.beginPath();
  const mid = height * 0.5;
  const amp = height * 0.42;
  const step = Math.max(1, Math.floor(y.length / width));
  for (let x = 0; x < width; x++) {
    let min = 1;
    let max = -1;
    const start = x * step;
    const end = Math.min(y.length, start + step);
    for (let i = start; i < end; i++) {
      const v = y[i] ?? 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x, mid - max * amp);
    ctx.lineTo(x, mid - min * amp);
  }
  ctx.stroke();
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.lineWidth = dpr;
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(width, mid);
  ctx.stroke();
  drawPlayhead(ctx, width, height, dpr, options);
}

function drawLineSeries(canvas, series, options = {}) {
  const { width, height, dpr } = fitCanvas(canvas);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#160f1f";
  ctx.fillRect(0, 0, width, height);

  const maxLen = Math.max(...series.map((item) => item.values.length), 1);
  const maxValue = Math.max(...series.flatMap((item) => Array.from(item.values, Math.abs)), 1e-9);

  for (const marker of options.markers ?? []) {
    const x = clamp01(marker.x / marker.maxX) * width;
    ctx.strokeStyle = marker.color;
    ctx.lineWidth = Math.max(1, dpr);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  series.forEach((item, index) => {
    const yBase = (index + 0.5) * (height / series.length);
    const yAmp = height / series.length * 0.36;
    ctx.strokeStyle = item.color;
    ctx.lineWidth = Math.max(1.5, 1.5 * dpr);
    ctx.beginPath();
    item.values.forEach((value, i) => {
      const x = maxLen <= 1 ? 0 : (i / (maxLen - 1)) * width;
      const y = yBase - (value / maxValue) * yAmp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.82)";
    ctx.font = `${12 * dpr}px ui-sans-serif, system-ui`;
    ctx.fillText(item.label, 10 * dpr, yBase - yAmp - 4 * dpr);
  });

  drawAxes(ctx, width, height, dpr);
  drawPlayhead(ctx, width, height, dpr, options);
}

function drawPath(canvas, path, rows, cols) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.width;
  const height = canvas.height;
  ctx.strokeStyle = "#f39a2f";
  ctx.lineWidth = Math.max(2, 2 * dpr);
  ctx.beginPath();
  for (let i = 0; i < path.rows; i++) {
    const r = path.data[i * 2];
    const c = path.data[i * 2 + 1];
    const x = (c / Math.max(1, cols - 1)) * width;
    const y = height - (r / Math.max(1, rows - 1)) * height;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawAlignment(canvas, slow, fast, sr, hopLength, path) {
  const { width, height, dpr } = fitCanvas(canvas);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#160f1f";
  ctx.fillRect(0, 0, width, height);
  const top = { y: height * 0.3, h: height * 0.18 };
  const bottom = { y: height * 0.72, h: height * 0.18 };
  drawWaveformTrack(ctx, slow, top.y, top.h, width, "#d13d8c");
  drawWaveformTrack(ctx, fast, bottom.y, bottom.h, width, "#f39a2f");

  const count = Math.min(36, path.rows);
  ctx.strokeStyle = "rgba(140, 74, 224, 0.38)";
  ctx.lineWidth = dpr;
  for (let i = 0; i < count; i++) {
    const p = Math.floor((i / Math.max(1, count - 1)) * Math.max(0, path.rows - 1));
    const slowFrame = path.data[p * 2];
    const fastFrame = path.data[p * 2 + 1];
    const x1 = ((slowFrame * hopLength) / Math.max(1, slow.length)) * width;
    const x2 = ((fastFrame * hopLength) / Math.max(1, fast.length)) * width;
    ctx.beginPath();
    ctx.moveTo(x1, top.y + top.h);
    ctx.lineTo(x2, bottom.y - bottom.h);
    ctx.stroke();
  }

  ctx.fillStyle = "rgba(255,255,255,0.82)";
  ctx.font = `${12 * dpr}px ui-sans-serif, system-ui`;
  ctx.fillText(`slow ${formatSeconds(slow.length / sr)} s`, 10 * dpr, 20 * dpr);
  ctx.fillText(`fast ${formatSeconds(fast.length / sr)} s`, 10 * dpr, height - 12 * dpr);
}

function drawWaveformTrack(ctx, data, center, radius, width, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1, window.devicePixelRatio || 1);
  ctx.beginPath();
  const step = Math.max(1, Math.floor(data.length / width));
  for (let x = 0; x < width; x++) {
    let min = 1;
    let max = -1;
    const start = x * step;
    const end = Math.min(data.length, start + step);
    for (let i = start; i < end; i++) {
      const value = data[i] ?? 0;
      if (value < min) min = value;
      if (value > max) max = value;
    }
    ctx.moveTo(x, center - max * radius);
    ctx.lineTo(x, center - min * radius);
  }
  ctx.stroke();
}

function drawSegments(canvas, duration, frameBoundaries, sr, hopLength) {
  const { width, height, dpr } = fitCanvas(canvas);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#160f1f";
  ctx.fillRect(0, 0, width, height);
  const colors = ["#7b00a8", "#d13d8c", "#f39a2f", "#8c4ae0", "#c94b68", "#ad6d00"];
  const times = frameBoundaries.map((frame) => (frame * hopLength) / sr);
  const uniqueTimes = uniqueSorted([0, ...times, duration]).filter((time) => time >= 0 && time <= duration);
  for (let i = 0; i < uniqueTimes.length - 1; i++) {
    const x0 = (uniqueTimes[i] / duration) * width;
    const x1 = (uniqueTimes[i + 1] / duration) * width;
    ctx.fillStyle = colors[i % colors.length];
    ctx.globalAlpha = 0.72;
    ctx.fillRect(x0, height * 0.18, Math.max(2 * dpr, x1 - x0), height * 0.58);
    ctx.globalAlpha = 1;
  }
  ctx.strokeStyle = "rgba(255,255,255,0.88)";
  ctx.lineWidth = Math.max(1, dpr);
  for (const time of uniqueTimes) {
    const x = (time / duration) * width;
    ctx.beginPath();
    ctx.moveTo(x, height * 0.12);
    ctx.lineTo(x, height * 0.84);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(255,255,255,0.84)";
  ctx.font = `${12 * dpr}px ui-sans-serif, system-ui`;
  ctx.fillText(`${formatSeconds(duration)} s`, 10 * dpr, height - 12 * dpr);
}

function drawCanvasPlayhead(canvas, options = {}) {
  const ctx = canvas.getContext("2d");
  drawPlayhead(ctx, canvas.width, canvas.height, window.devicePixelRatio || 1, options);
}

function drawPlayhead(ctx, width, height, dpr, options = {}) {
  const time = options.playheadTime;
  const duration = options.duration;
  if (!Number.isFinite(time) || !Number.isFinite(duration) || duration <= 0 || time < 0 || time > duration) {
    return;
  }
  const x = clamp01(time / duration) * width;
  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.94)";
  ctx.lineWidth = Math.max(2, 2 * dpr);
  ctx.shadowBlur = 8 * dpr;
  ctx.shadowColor = "rgba(209, 61, 140, 0.75)";
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, height);
  ctx.stroke();

  ctx.fillStyle = "#f39a2f";
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.arc(x, Math.max(6 * dpr, 7 * dpr), 4.5 * dpr, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawAxes(ctx, width, height, dpr) {
  ctx.strokeStyle = "rgba(255,255,255,0.2)";
  ctx.lineWidth = dpr;
  ctx.strokeRect(0.5 * dpr, 0.5 * dpr, width - dpr, height - dpr);
}

function fitCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, dpr };
}

function colorRamp(t, palette) {
  const stops = palettes[palette] ?? palettes.magma;
  const scaled = t * (stops.length - 1);
  const left = Math.floor(scaled);
  const right = Math.min(stops.length - 1, left + 1);
  const local = scaled - left;
  return [
    Math.round(lerp(stops[left][0], stops[right][0], local)),
    Math.round(lerp(stops[left][1], stops[right][1], local)),
    Math.round(lerp(stops[left][2], stops[right][2], local))
  ];
}

const palettes = {
  magma: [
    [22, 15, 31],
    [50, 20, 74],
    [89, 15, 127],
    [123, 0, 168],
    [209, 61, 140],
    [243, 154, 47],
    [255, 225, 150]
  ],
  gray: [
    [16, 24, 28],
    [54, 64, 69],
    [112, 126, 130],
    [194, 204, 204],
    [250, 250, 244]
  ]
};

function createDemoSource(sr = TARGET_SR, seconds = 12, variant = "tutorial") {
  const samples = Math.floor(sr * seconds);
  const y = new Float64Array(samples);
  const bpm = variant === "drums" ? 126 : variant === "sparse" ? 92 : 112;
  const beat = 60 / bpm;
  const sections = [
    { until: 3, chord: [261.63, 329.63, 392.0], bass: 130.81 },
    { until: 6, chord: [293.66, 369.99, 440.0], bass: 146.83 },
    { until: 9, chord: [220.0, 261.63, 329.63], bass: 110.0 },
    { until: 12.1, chord: [246.94, 311.13, 392.0], bass: 123.47 }
  ];
  const melody = [523.25, 587.33, 659.25, 783.99, 739.99, 659.25, 587.33, 523.25];

  for (let i = 0; i < samples; i++) {
    const t = i / sr;
    const section = sections.find((item) => t < item.until) ?? sections[sections.length - 1];
    const beatPhase = (t % beat) / beat;
    const beatIndex = Math.floor(t / beat);
    const halfBeatPhase = (t % (beat / 2)) / (beat / 2);
    const barPhase = (t % (beat * 4)) / (beat * 4);
    const chordEnv = 0.55 + 0.45 * Math.exp(-beatPhase * 4);
    const vibrato = 1 + 0.012 * Math.sin(2 * Math.PI * 5.8 * t);
    let value = 0;

    if (variant !== "drums" && variant !== "sparse") {
      for (let c = 0; c < section.chord.length; c++) {
        const level = variant === "harmony" ? 0.18 : 0.12;
        value += level * chordEnv * Math.sin(2 * Math.PI * section.chord[c] * t);
      }
      value += (variant === "harmony" ? 0.22 : 0.16) * Math.sin(2 * Math.PI * section.bass * t);
    }

    if (variant === "tutorial") {
      const note = melody[beatIndex % melody.length] * vibrato;
      const noteEnv = Math.exp(-beatPhase * 2.4);
      value += 0.2 * noteEnv * Math.sin(2 * Math.PI * note * t);
    }

    if (variant === "harmony") {
      const upper = section.chord[(beatIndex + 1) % section.chord.length] * 2;
      value += 0.08 * Math.sin(2 * Math.PI * upper * t + Math.sin(2 * Math.PI * 0.22 * t));
    }

    if (variant !== "harmony") {
      const kick = Math.exp(-beatPhase * 34) * Math.sin(2 * Math.PI * 82 * t);
      const hat = Math.exp(-halfBeatPhase * 60) * noiseAt(i);
      const snare = beatIndex % 2 === 1 ? Math.exp(-beatPhase * 28) * noiseAt(i * 3 + 17) : 0;
      const clap = variant === "drums" && barPhase > 0.48 && barPhase < 0.52
        ? Math.exp(-Math.abs(barPhase - 0.5) * 120) * noiseAt(i * 5 + 29)
        : 0;
      const sparseClick = variant === "sparse" && beatIndex % 3 === 0
        ? Math.exp(-beatPhase * 90) * Math.sin(2 * Math.PI * 1800 * t)
        : 0;
      value += 0.34 * kick + 0.08 * hat + 0.16 * snare + 0.12 * clap + 0.5 * sparseClick;
    }

    y[i] = value;
  }

  const names = {
    tutorial: "Tutorial mix",
    drums: "Percussion grid",
    harmony: "Harmonic steps",
    sparse: "Sparse clicks"
  };

  return {
    name: names[variant] ?? names.tutorial,
    samples,
    sr,
    y: normalizeAudio(y)
  };
}

function noiseAt(x) {
  const value = Math.sin(x * 12.9898) * 43758.5453;
  return 2 * (value - Math.floor(value)) - 1;
}

function downmix(buffer) {
  const out = new Float64Array(buffer.length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < data.length; i++) {
      out[i] += data[i] / buffer.numberOfChannels;
    }
  }
  return out;
}

function normalizeAudio(input) {
  let peak = 0;
  for (const value of input) peak = Math.max(peak, Math.abs(value));
  if (!peak) return new Float64Array(input);
  const scale = 0.92 / peak;
  const out = new Float64Array(input.length);
  for (let i = 0; i < input.length; i++) out[i] = input[i] * scale;
  return out;
}

function ensureSampleRate(librosa, y, origSr, targetSr) {
  if (origSr === targetSr) return y;
  return librosa.resample(y, {
    origSr,
    targetSr,
    resType: "linear"
  });
}

function firstSeconds(y, sr, seconds) {
  return y.slice(0, Math.min(y.length, Math.max(1, Math.floor(sr * seconds))));
}

function matrixExtent(matrix, options = {}) {
  if (Number.isFinite(options.min) && Number.isFinite(options.max)) return [options.min, options.max];
  let min = Infinity;
  let max = -Infinity;
  for (const value of matrix.data) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  return [min, max];
}

function relativeDb(matrix, floor = -80) {
  let max = -Infinity;
  for (const value of matrix.data) {
    if (Number.isFinite(value) && value > max) max = value;
  }
  const data = new Float64Array(matrix.data.length);
  for (let i = 0; i < matrix.data.length; i++) {
    data[i] = Math.max(floor, matrix.data[i] - max);
  }
  return { rows: matrix.rows, cols: matrix.cols, data };
}

function cloneMatrix(matrix) {
  return {
    rows: matrix.rows,
    cols: matrix.cols,
    data: new Float64Array(matrix.data)
  };
}

function mapMatrix(matrix, fn) {
  const data = new Float64Array(matrix.data.length);
  for (let i = 0; i < matrix.data.length; i++) data[i] = fn(matrix.data[i], i);
  return { rows: matrix.rows, cols: matrix.cols, data };
}

function scaleMatrix(matrix, scalar) {
  return mapMatrix(matrix, (value) => value * scalar);
}

function clampMatrix(matrix, min) {
  return mapMatrix(matrix, (value) => Math.max(min, value));
}

function subtractMatrix(a, b) {
  return combineMatrices(a, b, (x, y) => x - y);
}

function multiplyMatrix(a, b) {
  return combineMatrices(a, b, (x, y) => x * y);
}

function minimumMatrix(a, b) {
  return combineMatrices(a, b, Math.min);
}

function matrixDifference(a, b) {
  return combineMatrices(a, b, (x, y) => x - y);
}

function combineMatrices(a, b, fn) {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error(`Matrix shape mismatch: ${shape(a)} vs ${shape(b)}`);
  }
  const data = new Float64Array(a.data.length);
  for (let i = 0; i < data.length; i++) data[i] = fn(a.data[i], b.data[i]);
  return { rows: a.rows, cols: a.cols, data };
}

function nnFilterMedian(matrix, options = {}) {
  const rows = matrix.rows;
  const cols = matrix.cols;
  const width = options.width ?? 1;
  const neighbors = Math.min(options.neighbors ?? 8, Math.max(1, cols - 1));
  const norms = new Float64Array(cols);
  for (let col = 0; col < cols; col++) {
    let sum = 0;
    for (let row = 0; row < rows; row++) {
      const value = matrix.data[row * cols + col];
      sum += value * value;
    }
    norms[col] = Math.sqrt(sum) || 1;
  }

  const out = new Float64Array(rows * cols);
  const scratch = new Float64Array(rows * Math.max(1, neighbors));
  const scores = new Array(cols);

  for (let col = 0; col < cols; col++) {
    scores.length = 0;
    for (let other = 0; other < cols; other++) {
      if (Math.abs(other - col) <= width && cols > width * 2 + 1) continue;
      let dot = 0;
      for (let row = 0; row < rows; row++) {
        dot += matrix.data[row * cols + col] * matrix.data[row * cols + other];
      }
      scores.push({ col: other, score: dot / (norms[col] * norms[other]) });
    }
    scores.sort((a, b) => b.score - a.score);
    const chosen = scores.slice(0, neighbors);
    if (!chosen.length) chosen.push({ col, score: 1 });

    for (let row = 0; row < rows; row++) {
      for (let k = 0; k < chosen.length; k++) {
        scratch[k] = matrix.data[row * cols + chosen[k].col];
      }
      out[row * cols + col] = medianInPlace(scratch.subarray(0, chosen.length));
    }
  }

  return { rows, cols, data: out };
}

function syncMedian(matrix, boundaries) {
  const frames = uniqueSorted(Array.from(boundaries, (value) => Math.round(value)))
    .filter((value) => value >= 0 && value <= matrix.cols);
  if (frames[0] !== 0) frames.unshift(0);
  if (frames[frames.length - 1] !== matrix.cols) frames.push(matrix.cols);
  const cols = Math.max(1, frames.length - 1);
  const out = new Float64Array(matrix.rows * cols);
  const scratch = new Float64Array(matrix.cols);

  for (let segment = 0; segment < cols; segment++) {
    const start = frames[segment];
    const end = Math.max(start + 1, frames[segment + 1]);
    for (let row = 0; row < matrix.rows; row++) {
      let count = 0;
      for (let col = start; col < Math.min(end, matrix.cols); col++) {
        scratch[count++] = matrix.data[row * matrix.cols + col];
      }
      out[row * cols + segment] = count ? medianInPlace(scratch.subarray(0, count)) : 0;
    }
  }
  return { rows: matrix.rows, cols, data: out };
}

function robustBeatFrames(librosa, beats, maxFrame) {
  const cleaned = uniqueSorted(Array.from(beats, (value) => Math.round(value)))
    .filter((value) => value >= 0 && value <= maxFrame);
  let frames = cleaned;
  if (frames.length < 4) {
    const step = Math.max(3, Math.floor(maxFrame / 8));
    frames = [];
    for (let frame = 0; frame <= maxFrame; frame += step) frames.push(frame);
  }
  return Array.from(librosa.fixFrames(new Float64Array(frames), {
    xMin: 0,
    xMax: maxFrame
  }));
}

function medianInPlace(values) {
  const sorted = Array.from(values).sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function percentEnergy(part, full) {
  let a = 0;
  let b = 0;
  for (let i = 0; i < full.data.length; i++) {
    a += Math.abs(part.data[i]);
    b += Math.abs(full.data[i]);
  }
  return formatNumber((a / Math.max(1e-12, b)) * 100);
}

function shape(matrix) {
  return `${matrix.rows} x ${matrix.cols}`;
}

function formatSeconds(value) {
  return `${formatNumber(value)}s`;
}

function formatNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return "--";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits
  }).format(value);
}

function formatInteger(value) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0
  }).format(value);
}

function uniqueSorted(values) {
  return Array.from(new Set(values.filter(Number.isFinite))).sort((a, b) => a - b);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}
