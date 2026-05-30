# Cross-Validation Tests

This directory contains tests that validate the C++ librosa implementation against the Python reference implementation.

## Overview

The cross-validation system works in two steps:

1. **Generate Reference Data** (Python): Run `generate_references.py` to create JSON files containing expected outputs from Python librosa.

2. **Run Cross-Validation Tests** (C++): The `test_crossval.cpp` tests load the reference data and compare C++ outputs against them.

## Usage

### Step 1: Generate Reference Data

```bash
# From the repo root
pip install -r requirements-dev.txt

# Generate reference data (writes to tests/crossval/reference_data/)
python tests/crossval/generate_references.py
```

This creates a `reference_data/` directory containing JSON files with test vectors.

### Step 2: Build and Run Tests

```bash
# From the repo root
cmake -S . -B build -DLIBROSA_BUILD_CROSSVAL_TESTS=ON
cmake --build build --target crossval_tests
./build/crossval_tests
```

## Reference Data Format

Each JSON file contains:
- `shape`: Array dimensions
- `dtype`: Data type string
- `data`: Flattened array data as a list
- `metadata`: (optional) Input parameters used to generate the data

Example:
```json
{
  "shape": [6],
  "dtype": "float64",
  "data": [150.48, 283.23, 549.64, 999.98, 2839.85, 4511.65],
  "metadata": {
    "input_hz": [100, 200, 440, 1000, 4000, 8000]
  }
}
```

## Adding New Tests

1. Add a new function to `generate_references.py` that saves reference data
2. Add corresponding test cases to `test_crossval.cpp`
3. Re-run `generate_references.py` to update reference data

## Tolerance Levels

Different modules may require different tolerance levels due to:
- Floating-point precision differences
- Algorithm implementation variations
- FFT library differences

Current tolerances:
- `DEFAULT_TOLERANCE = 1e-5`: For exact mathematical operations
- `1e-6`: For optional SOXR resampling parity when libsoxr is enabled
- `LOOSE_TOLERANCE = 1e-3`: For filter banks, spectral features

## Modules Covered

- [x] Convert (hz_to_mel, mel_to_hz, hz_to_midi, amplitude_to_db, power_to_db)
- [x] Audio (SOXR resampling modes when built with `LIBROSA_USE_SOXR=ON`)
- [x] Filters (mel filterbank, chroma filterbank)
- [x] Spectrum (STFT magnitude/phase)
- [x] Features (melspectrogram, MFCC, chroma, spectral features, RMS, ZCR)
- [x] Onset (onset_strength, onset_detect)
- [x] Beat (tempo, beat_track)
- [x] Decompose (HPSS)
- [x] Sequence (Viterbi, DTW, transition matrices)

## Troubleshooting

**"Reference data not found"**: Make sure you've run `generate_references.py` first.

**Large discrepancies**: Check if the C++ and Python are using the same:
- Sample rate
- FFT size
- Hop length
- Window type
- Normalization settings
