# librosa CLI Reference

A command-line tool for audio analysis, powered by the librosa C++ library.

## Building

```bash
cmake -S . -B build -DLIBROSA_BUILD_CLI=ON
cmake --build build --target librosa_cli
```

The binary is produced at `build/librosa`.

## Usage

```
librosa [OPTIONS] <file> <SUBCOMMAND> [SUBCOMMAND OPTIONS]
```

The audio file path is a required positional argument, followed by exactly one subcommand.

## Global Options

### Audio Loading

| Flag | Default | Description |
|------|---------|-------------|
| `--sr <rate>` | 22050 | Target sample rate. Use `0` for native rate. |
| `--mono` / `--no-mono` | `--mono` | Force mono mixdown (averages channels). |
| `--offset <seconds>` | 0 | Start reading after this time. |
| `--duration <seconds>` | -1 (all) | Only load this much audio. |

### STFT Parameters

Shared by subcommands that compute spectrograms internally (mfcc, melspectrogram, chroma, spectral-*, stft, etc.).

| Flag | Default | Description |
|------|---------|-------------|
| `--n-fft <int>` | 2048 | FFT window size. |
| `--hop-length <int>` | 512 | Samples between successive frames. |
| `--win-length <int>` | 0 (=n_fft) | Analysis window length. 0 uses n_fft. |
| `--window <type>` | hann | Window function: `hann`, `hamming`, `blackman`, `bartlett`. |

### Output Format

| Flag | Default | Description |
|------|---------|-------------|
| `--format <fmt>` | text | Output format: `text`, `json`, `csv`. |
| `--precision <int>` | 6 | Decimal places for floating-point values. |
| `--no-time` | off | Omit the time column from per-frame output. |

## Output Formats

### Text (default)

Tab-separated, one line per frame. Time column first (unless `--no-time`). Designed for piping to `awk`, `cut`, or plotting tools.

```
0.023220	440.00	0.98
0.046440	441.20	0.97
```

### JSON

Structured output with metadata. NaN pitch values become JSON `null`.

```json
{
  "file": "input.wav",
  "command": "mfcc",
  "shape": [43, 20],
  "times": [0.023220, 0.046440, ...],
  "data": [[-12.34, 4.56, ...], ...]
}
```

### CSV

Standard CSV with a header row. Compatible with pandas, Excel, R, etc.

```
time,c0,c1,c2
0.023220,-12.340,4.560,-1.230
```

## Subcommands

### info

Show audio file metadata.

```bash
librosa file.wav info
```

Output: key-value pairs for filename, native sample rate, duration, sample count, and channel count.

### tempo

Estimate global tempo in BPM.

```bash
librosa file.wav tempo
librosa file.wav tempo --start-bpm 140
```

| Option | Default | Description |
|--------|---------|-------------|
| `--start-bpm` | 120 | Initial tempo guess for the estimator. |

Output: scalar BPM value.

### beat

Detect beat positions.

```bash
librosa file.wav beat
librosa file.wav beat --tightness 200
```

| Option | Default | Description |
|--------|---------|-------------|
| `--start-bpm` | 120 | Initial tempo guess. |
| `--tightness` | 100 | How closely beats adhere to the estimated tempo. |

Output: 1D vector of beat times (seconds).

### onset

Detect onset (note attack) positions.

```bash
librosa file.wav onset
librosa file.wav onset --backtrack
```

| Option | Default | Description |
|--------|---------|-------------|
| `--backtrack` | off | Backtrack onsets to the nearest preceding energy minimum. |

Output: 1D vector of onset times (seconds).

### pitch

Estimate fundamental frequency per frame using pYIN or YIN.

```bash
librosa file.wav pitch
librosa file.wav pitch --method yin --fmin 80 --fmax 1000
```

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | pyin | Algorithm: `pyin` (probabilistic) or `yin`. |
| `--fmin` | 65 | Minimum frequency in Hz (C2). |
| `--fmax` | 2093 | Maximum frequency in Hz (C7). |

Output: per-frame with columns `time`, `f0`, `voiced_probability`. Unvoiced frames show `nan` for f0 (or `null` in JSON).

### mfcc

Compute Mel-frequency cepstral coefficients.

```bash
librosa file.wav mfcc
librosa file.wav mfcc --n-mfcc 13 --n-mels 40
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-mfcc` | 20 | Number of MFCCs to return. |
| `--n-mels` | 128 | Number of mel bands in the filterbank. |

Output: matrix (n_frames x n_mfcc). Column labels: `c0`, `c1`, ...

### melspectrogram

Compute mel spectrogram in decibels.

```bash
librosa file.wav melspectrogram
librosa file.wav melspectrogram --n-mels 64
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-mels` | 128 | Number of mel bands. |

Output: matrix (n_frames x n_mels) in dB. Column labels: `m0`, `m1`, ...

### chroma

Compute a chromagram (pitch class energy distribution).

```bash
librosa file.wav chroma
librosa file.wav chroma --variant cqt
```

| Option | Default | Description |
|--------|---------|-------------|
| `--variant` | stft | Algorithm: `stft`, `cqt`, or `cens`. |

Output: matrix (n_frames x 12). Column labels: `C`, `C#`, `D`, ..., `B`.

### spectral-centroid

Compute the spectral centroid (center of mass of the spectrum).

```bash
librosa file.wav spectral-centroid
```

Output: per-frame, single column `centroid` (Hz).

### spectral-bandwidth

Compute the spectral bandwidth (weighted spread around the centroid).

```bash
librosa file.wav spectral-bandwidth
```

Output: per-frame, single column `bandwidth` (Hz).

### spectral-rolloff

Compute the spectral rolloff frequency.

```bash
librosa file.wav spectral-rolloff
librosa file.wav spectral-rolloff --roll-percent 0.95
```

| Option | Default | Description |
|--------|---------|-------------|
| `--roll-percent` | 0.85 | Fraction of total energy below rolloff. |

Output: per-frame, single column `rolloff` (Hz).

### spectral-flatness

Compute spectral flatness (tonality measure: 0 = tonal, 1 = noisy).

```bash
librosa file.wav spectral-flatness
```

Output: per-frame, single column `flatness`.

### spectral-contrast

Compute spectral contrast across frequency bands.

```bash
librosa file.wav spectral-contrast
librosa file.wav spectral-contrast --n-bands 4
```

| Option | Default | Description |
|--------|---------|-------------|
| `--n-bands` | 6 | Number of frequency bands. |

Output: matrix (n_frames x n_bands+1). Column labels: `band0`, `band1`, ..., `valley`.

### rms

Compute root-mean-square energy per frame.

```bash
librosa file.wav rms
```

Output: per-frame, single column `rms`.

### zcr

Compute zero crossing rate per frame.

```bash
librosa file.wav zcr
```

Output: per-frame, single column `zcr`.

### tonnetz

Compute tonal centroid features (Harte et al., 2006).

```bash
librosa file.wav tonnetz
```

Output: matrix (n_frames x 6). Column labels: `fifth_x`, `fifth_y`, `minor_x`, `minor_y`, `major_x`, `major_y`.

### tuning

Estimate tuning offset from A440.

```bash
librosa file.wav tuning
```

Output: scalar value in fractions of a semitone bin (multiply by 100 for cents).

### stft

Compute magnitude spectrogram in decibels.

```bash
librosa file.wav stft
```

Output: matrix (n_frames x n_fft/2+1) in dB. Column labels: `f0`, `f1`, ...

This produces large output. Consider piping to a file or using `--format csv` for downstream processing.

### trim

Find non-silent region boundaries.

```bash
librosa file.wav trim
librosa file.wav trim --top-db 30
```

| Option | Default | Description |
|--------|---------|-------------|
| `--top-db` | 60 | Threshold in dB below reference to consider as silence. |

Output: key-value pairs for `start_time`, `end_time`, `start_sample`, `end_sample`.

### hpss

Harmonic/percussive source separation. Writes a WAV file.

```bash
librosa file.wav hpss -o harmonic.wav
librosa file.wav hpss -o percussive.wav --component percussive
```

| Option | Default | Description |
|--------|---------|-------------|
| `-o`, `--output` | (required) | Output WAV file path. |
| `--component` | harmonic | Which component to extract: `harmonic` or `percussive`. |

Output: writes a 16-bit PCM WAV file. Progress message to stderr.

## Examples

```bash
# Quick file overview
librosa song.wav info

# Get tempo estimate
librosa song.wav tempo

# Onset times as JSON for downstream processing
librosa song.wav --format json onset | python3 -m json.tool

# Extract 13 MFCCs as CSV, no time column
librosa song.wav --no-time --format csv mfcc --n-mfcc 13 > features.csv

# Compare pitch methods
librosa song.wav pitch --method pyin > pyin.tsv
librosa song.wav pitch --method yin  > yin.tsv

# Load at native sample rate, analyze first 10 seconds
librosa --sr 0 --duration 10 song.wav melspectrogram --format json > mel.json

# Separate harmonic and percussive components
librosa song.wav hpss -o harmonic.wav --component harmonic
librosa song.wav hpss -o percussive.wav --component percussive

# Chroma from constant-Q transform
librosa song.wav chroma --variant cqt

# Custom STFT parameters for all spectral features
librosa --n-fft 4096 --hop-length 1024 song.wav spectral-centroid
```

## Architecture

```
cli/
├── main.cpp              Entry point, CLI11 setup, global flags
├── common.hpp/cpp        CommonOptions, load_audio(), make_time_axis(), parse_window()
├── formatter.hpp/cpp     OutputFormatter (text/json/csv)
└── commands/
    ├── info.cpp           info
    ├── tempo.cpp          tempo
    ├── beat.cpp           beat
    ├── onset.cpp          onset
    ├── pitch.cpp          pitch (pyin/yin)
    ├── mfcc.cpp           mfcc
    ├── melspectrogram.cpp melspectrogram
    ├── chroma.cpp         chroma (stft/cqt/cens)
    ├── spectral.cpp       spectral-centroid, -bandwidth, -rolloff, -flatness, -contrast
    ├── rms.cpp            rms
    ├── zcr.cpp            zcr
    ├── tonnetz.cpp        tonnetz
    ├── tuning.cpp         tuning
    ├── stft.cpp           stft
    ├── trim.cpp           trim
    └── hpss.cpp           hpss
```

Each command file exports a single `register_<name>(CLI::App&, CommonOptions&)` function that:
1. Adds a subcommand with its specific flags.
2. Sets a callback that loads audio, calls the librosa C++ API, and feeds `OutputFormatter`.

Subcommand-local options use `std::make_shared<T>` so CLI11 can write parsed values into them after the lambda capture. Global options live in `CommonOptions` and are captured by reference.
