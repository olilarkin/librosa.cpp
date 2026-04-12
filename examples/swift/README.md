# Librosa Swift Example

This is a minimal SwiftPM executable that consumes the local `Librosa` package
from this repository.

Run it with a generated 440 Hz tone:

```bash
cd examples/swift
swift run
```

Or analyze an audio file with AudioToolbox-backed loading:

```bash
swift run LibrosaSwiftExample /path/to/audio.wav
```

The example prints basic analysis output:

- MFCC matrix dimensions
- Mel-spectrogram dimensions
- Onset envelope frame count
- Estimated tempo
- Mean spectral centroid and rolloff
- A short preview of the first MFCC frame
