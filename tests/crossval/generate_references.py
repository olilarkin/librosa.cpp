#!/usr/bin/env python3
"""
Generate reference outputs from Python librosa for cross-validation testing.

This script generates test vectors that can be loaded by C++ tests to verify
the C++ implementation matches the Python reference.

Output format: JSON files with arrays stored as nested lists.
"""

import json
import numpy as np
import librosa
import os
from pathlib import Path

# Output directory for reference data
OUTPUT_DIR = Path(__file__).parent / "reference_data"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_array(name: str, arr: np.ndarray, metadata: dict = None):
    """Save a numpy array to JSON format."""
    data = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "data": arr.flatten().tolist(),
    }
    if metadata:
        data["metadata"] = metadata

    filepath = OUTPUT_DIR / f"{name}.json"
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filepath}")


def generate_convert_references():
    """Generate references for convert module."""
    print("\n=== Convert Module ===")

    # hz_to_mel
    hz_values = np.array([100, 200, 440, 1000, 4000, 8000], dtype=np.float64)
    mel_htk = librosa.hz_to_mel(hz_values, htk=True)
    mel_slaney = librosa.hz_to_mel(hz_values, htk=False)
    save_array("convert_hz_to_mel_htk", mel_htk, {"input_hz": hz_values.tolist()})
    save_array("convert_hz_to_mel_slaney", mel_slaney, {"input_hz": hz_values.tolist()})

    # mel_to_hz
    mel_values = np.array([100, 500, 1000, 1500, 2000, 2500], dtype=np.float64)
    hz_htk = librosa.mel_to_hz(mel_values, htk=True)
    hz_slaney = librosa.mel_to_hz(mel_values, htk=False)
    save_array("convert_mel_to_hz_htk", hz_htk, {"input_mel": mel_values.tolist()})
    save_array("convert_mel_to_hz_slaney", hz_slaney, {"input_mel": mel_values.tolist()})

    # hz_to_midi
    hz_midi = np.array([261.63, 440.0, 880.0], dtype=np.float64)
    midi = librosa.hz_to_midi(hz_midi)
    save_array("convert_hz_to_midi", midi, {"input_hz": hz_midi.tolist()})

    # midi_to_hz
    midi_values = np.array([60.0, 69.0, 81.0], dtype=np.float64)
    hz_from_midi = librosa.midi_to_hz(midi_values)
    save_array("convert_midi_to_hz", hz_from_midi, {"input_midi": midi_values.tolist()})

    # amplitude_to_db / power_to_db
    amp = np.array([0.001, 0.01, 0.1, 1.0, 10.0], dtype=np.float64)
    db_amp = librosa.amplitude_to_db(amp, ref=1.0)
    db_power = librosa.power_to_db(amp, ref=1.0)
    save_array("convert_amplitude_to_db", db_amp, {"input": amp.tolist()})
    save_array("convert_power_to_db", db_power, {"input": amp.tolist()})


def generate_spectrum_references():
    """Generate references for spectrum module."""
    print("\n=== Spectrum Module ===")

    # Create a simple test signal
    sr = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz sine wave
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    save_array("spectrum_test_signal", y, {"sr": sr, "duration": duration, "freq": 440})

    # STFT
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Save magnitude and phase separately
    save_array("spectrum_stft_magnitude", np.abs(D),
               {"n_fft": n_fft, "hop_length": hop_length})
    save_array("spectrum_stft_phase", np.angle(D),
               {"n_fft": n_fft, "hop_length": hop_length})


def generate_resample_references():
    """Generate references for SOXR resampling."""
    print("\n=== Resample Module ===")

    sr = 22050
    target_sr = 8000
    duration = 0.25
    t = np.arange(int(sr * duration), dtype=np.float64) / sr
    y = (
        0.5 * np.sin(2 * np.pi * 220.0 * t)
        + 0.25 * np.sin(2 * np.pi * 997.0 * t)
        + 0.1 * np.sin(2 * np.pi * 3200.0 * t)
    )
    y[::997] += 0.05
    y = y.astype(np.float64)

    save_array("resample_test_signal", y, {"sr": sr, "target_sr": target_sr})

    for res_type in ["soxr_vhq", "soxr_hq", "soxr_mq", "soxr_lq", "soxr_qq"]:
        y_hat = librosa.resample(
            y, orig_sr=sr, target_sr=target_sr, res_type=res_type
        )
        save_array(f"resample_{res_type}", y_hat,
                   {"sr": sr, "target_sr": target_sr, "res_type": res_type})


def generate_filters_references():
    """Generate references for filters module."""
    print("\n=== Filters Module ===")

    sr = 22050
    n_fft = 2048
    n_mels = 128

    # Mel filter bank
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    save_array("filters_mel_filterbank", mel_fb,
               {"sr": sr, "n_fft": n_fft, "n_mels": n_mels})

    # Mel filter bank with HTK
    mel_fb_htk = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, htk=True)
    save_array("filters_mel_filterbank_htk", mel_fb_htk,
               {"sr": sr, "n_fft": n_fft, "n_mels": n_mels, "htk": True})

    # Chroma filter bank
    n_chroma = 12
    chroma_fb = librosa.filters.chroma(sr=sr, n_fft=n_fft, n_chroma=n_chroma)
    save_array("filters_chroma_filterbank", chroma_fb,
               {"sr": sr, "n_fft": n_fft, "n_chroma": n_chroma})


def generate_feature_references():
    """Generate references for feature extraction."""
    print("\n=== Feature Module ===")

    # Load or create test audio
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Mix of frequencies
    y = (0.5 * np.sin(2 * np.pi * 440 * t) +
         0.3 * np.sin(2 * np.pi * 880 * t) +
         0.2 * np.sin(2 * np.pi * 1320 * t)).astype(np.float32)

    save_array("feature_test_signal", y, {"sr": sr, "duration": duration})

    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    save_array("feature_melspectrogram", mel_spec, {"sr": sr})

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    save_array("feature_mfcc", mfcc, {"sr": sr, "n_mfcc": 13})

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    save_array("feature_chroma_stft", chroma, {"sr": sr})

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    save_array("feature_spectral_centroid", centroid, {"sr": sr})

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    save_array("feature_spectral_bandwidth", bandwidth, {"sr": sr})

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    save_array("feature_spectral_rolloff", rolloff, {"sr": sr})

    # RMS
    rms = librosa.feature.rms(y=y)
    save_array("feature_rms", rms, {"sr": sr})

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    save_array("feature_zero_crossing_rate", zcr, {"sr": sr})


def generate_onset_references():
    """Generate references for onset detection."""
    print("\n=== Onset Module ===")

    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Signal with clear transients
    y = np.zeros_like(t)
    # Add clicks at regular intervals
    click_times = [0.0, 0.5, 1.0, 1.5]
    for ct in click_times:
        idx = int(ct * sr)
        if idx < len(y):
            y[idx:idx+100] = np.exp(-np.linspace(0, 5, 100))
    y = y.astype(np.float32)

    save_array("onset_test_signal", y, {"sr": sr, "click_times": click_times})

    # Onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    save_array("onset_strength", onset_env, {"sr": sr})

    # Onset detect
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
    save_array("onset_detect_frames", onsets.astype(np.float64), {"sr": sr})


def generate_beat_references():
    """Generate references for beat tracking."""
    print("\n=== Beat Module ===")

    sr = 22050
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Signal with regular beats (~120 BPM)
    bpm = 120
    beat_interval = 60.0 / bpm
    y = np.zeros_like(t)

    beat_time = 0
    while beat_time < duration:
        idx = int(beat_time * sr)
        if idx < len(y) - 200:
            y[idx:idx+200] = np.exp(-np.linspace(0, 10, 200)) * np.sin(2 * np.pi * 100 * np.linspace(0, 0.01, 200))
        beat_time += beat_interval
    y = y.astype(np.float32)

    save_array("beat_test_signal", y, {"sr": sr, "bpm": bpm})

    # Tempo estimation
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    save_array("beat_tempo", tempo, {"sr": sr})

    # Beat tracking
    tempo_val, beats = librosa.beat.beat_track(y=y, sr=sr)
    # librosa.beat.beat_track returns tempo as a (1,) ndarray in 0.11; use
    # .item() to get a Python scalar (NumPy 2.x rejects float(1d_ndarray)).
    save_array("beat_track_frames", beats.astype(np.float64),
               {"sr": sr, "tempo": float(np.asarray(tempo_val).reshape(-1)[0])})


def generate_decompose_references():
    """Generate references for decompose module."""
    print("\n=== Decompose Module ===")

    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Mix of harmonic (sustained) and percussive (transient) content
    harmonic = np.sin(2 * np.pi * 440 * t) * np.exp(-t)
    percussive = np.zeros_like(t)
    for i in range(4):
        idx = int(i * 0.25 * sr)
        if idx < len(percussive) - 100:
            percussive[idx:idx+100] = np.exp(-np.linspace(0, 20, 100))

    y = (harmonic + 0.5 * percussive).astype(np.float32)
    save_array("decompose_test_signal", y, {"sr": sr})

    # HPSS
    D = librosa.stft(y)
    H, P = librosa.decompose.hpss(np.abs(D))
    save_array("decompose_hpss_harmonic", H, {"sr": sr})
    save_array("decompose_hpss_percussive", P, {"sr": sr})


def generate_sequence_references():
    """Generate references for sequence module."""
    print("\n=== Sequence Module ===")

    # Viterbi - Wikipedia example
    # States: Healthy (0), Fever (1)
    # Observations: Normal, Cold, Dizzy
    prob = np.array([
        [0.1, 0.4, 0.5],  # Healthy: P(normal), P(cold), P(dizzy)
        [0.6, 0.3, 0.1],  # Fever: P(normal), P(cold), P(dizzy)
    ])

    # Observation sequence: Normal, Cold, Dizzy
    obs_seq = [0, 1, 2]

    # Observation probability matrix
    obs_prob = prob[:, obs_seq]  # Shape: (2, 3)

    transition = np.array([
        [0.7, 0.3],  # From Healthy
        [0.4, 0.6],  # From Fever
    ])

    # Initial probabilities
    init = np.array([0.6, 0.4])

    # Combine initial prob with first observation
    obs_prob_with_init = obs_prob.copy()
    obs_prob_with_init[:, 0] *= init

    path = librosa.sequence.viterbi(obs_prob_with_init, transition)
    save_array("sequence_viterbi_path", path.astype(np.float64),
               {"obs_prob_shape": list(obs_prob.shape)})

    # DTW
    # Note: librosa expects X.shape=(K, N) where K is feature dimension
    X = np.array([[1, 2, 3, 4, 5]], dtype=np.float64)  # Shape: (1, 5)
    Y = np.array([[1, 2, 2, 3, 4, 5]], dtype=np.float64)  # Shape: (1, 6)

    D, wp = librosa.sequence.dtw(X, Y, backtrack=True)
    save_array("sequence_dtw_cost", D, {"X_shape": list(X.shape), "Y_shape": list(Y.shape)})
    save_array("sequence_dtw_path", wp.astype(np.float64), {})

    # Transition matrices
    n_states = 4
    trans_uniform = librosa.sequence.transition_uniform(n_states)
    trans_loop = librosa.sequence.transition_loop(n_states, prob=0.9)
    save_array("sequence_transition_uniform", trans_uniform, {"n_states": n_states})
    save_array("sequence_transition_loop", trans_loop, {"n_states": n_states, "prob": 0.9})


def generate_new_feature_references():
    """Generate references for newly ported features."""
    print("\n=== New Feature References ===")

    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Signal with harmonic + transient content
    harmonic = np.sin(2 * np.pi * 440 * t) * np.exp(-t * 0.5)
    percussive = np.zeros_like(t)
    for i in range(4):
        idx = int(i * 0.25 * sr)
        if idx < len(percussive) - 100:
            percussive[idx:idx+100] = np.exp(-np.linspace(0, 20, 100))
    y_hp = (harmonic + 0.5 * percussive).astype(np.float64)

    # Effects: harmonic / percussive
    y_harm = librosa.effects.harmonic(y_hp)
    y_perc = librosa.effects.percussive(y_hp)
    save_array("effects_test_signal", y_hp, {"sr": sr})
    save_array("effects_harmonic", y_harm, {"sr": sr})
    save_array("effects_percussive", y_perc, {"sr": sr})

    # Feature: stack_memory
    data = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    stacked = librosa.feature.stack_memory(data, n_steps=3, delay=1)
    save_array("feature_stack_memory_input", data, {"n_steps": 3, "delay": 1})
    save_array("feature_stack_memory", stacked, {"n_steps": 3, "delay": 1})

    # Harmonic: interp_harmonics
    y_short = (0.5 * np.sin(2 * np.pi * 440 * t[:int(sr*0.5)]) +
               0.3 * np.sin(2 * np.pi * 880 * t[:int(sr*0.5)])).astype(np.float64)
    D = librosa.stft(y_short)
    S = np.abs(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    harmonics = np.array([1.0, 2.0, 3.0])
    S_harm = librosa.interp_harmonics(S, freqs=freqs, harmonics=harmonics)
    # interp_harmonics returns (n_harmonics, n_freq, n_frames); reshape to (n_harmonics*n_freq, n_frames)
    S_harm_flat = S_harm.reshape(-1, S_harm.shape[-1])
    save_array("harmonic_interp_harmonics", S_harm_flat,
               {"harmonics": harmonics.tolist(), "sr": sr})
    save_array("harmonic_interp_harmonics_input_S", S, {"sr": sr})
    save_array("harmonic_interp_harmonics_freqs", freqs, {"sr": sr})

    # Feature: mel_to_stft
    mel_spec = librosa.feature.melspectrogram(y=y_short, sr=sr)
    mel_inv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=sr)
    save_array("feature_mel_to_stft_input", mel_spec, {"sr": sr})
    save_array("feature_mel_to_stft", mel_inv, {"sr": sr})

    # Feature: mfcc_to_mel
    mfcc_data = librosa.feature.mfcc(y=y_short, sr=sr, n_mfcc=20)
    mel_from_mfcc = librosa.feature.inverse.mfcc_to_mel(mfcc_data, n_mels=128)
    save_array("feature_mfcc_to_mel_input", mfcc_data, {"sr": sr, "n_mfcc": 20})
    save_array("feature_mfcc_to_mel", mel_from_mfcc, {"sr": sr, "n_mels": 128})

    # Onset envelope for tempogram tests
    onset_env = librosa.onset.onset_strength(y=y_hp, sr=sr)
    save_array("feature_onset_envelope", onset_env, {"sr": sr})

    # Feature: fourier_tempogram
    ftgram = librosa.feature.fourier_tempogram(onset_envelope=onset_env, sr=sr)
    save_array("feature_fourier_tempogram_mag", np.abs(ftgram),
               {"sr": sr, "shape": list(ftgram.shape)})

    # Beat: PLP
    plp_result = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    save_array("beat_plp", plp_result, {"sr": sr})


def generate_additional_feature_references():
    """Generate references for additional spectral features."""
    print("\n=== Additional Feature References ===")

    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Reuse same signal as feature_test_signal
    y = (0.5 * np.sin(2 * np.pi * 440 * t) +
         0.3 * np.sin(2 * np.pi * 880 * t) +
         0.2 * np.sin(2 * np.pi * 1320 * t)).astype(np.float32)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    save_array("feature_spectral_contrast", contrast, {"sr": sr})

    # Spectral flatness
    flatness = librosa.feature.spectral_flatness(y=y)
    save_array("feature_spectral_flatness", flatness, {"sr": sr})

    # Poly features
    poly = librosa.feature.poly_features(y=y, sr=sr, order=1)
    save_array("feature_poly_features", poly, {"sr": sr, "order": 1})

    # Tonnetz
    ton = librosa.feature.tonnetz(y=y, sr=sr)
    save_array("feature_tonnetz", ton, {"sr": sr})

    # Delta (on MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    save_array("feature_delta_input", mfcc, {"sr": sr, "n_mfcc": 13})
    d = librosa.feature.delta(mfcc, width=9, order=1)
    save_array("feature_delta", d, {"width": 9, "order": 1})


def generate_chroma_cqt_references():
    """Generate references for chroma_cqt and chroma_cens."""
    print("\n=== Chroma CQT/CENS References ===")

    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # C4 + E4 (major third)
    y = (0.5 * np.sin(2 * np.pi * 261.63 * t) +
         0.5 * np.sin(2 * np.pi * 329.63 * t)).astype(np.float32)

    save_array("chroma_cqt_test_signal", y, {"sr": sr})

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    save_array("feature_chroma_cqt", chroma, {"sr": sr})

    cens = librosa.feature.chroma_cens(y=y, sr=sr)
    save_array("feature_chroma_cens", cens, {"sr": sr})


def generate_tempogram_references():
    """Generate references for autocorrelation tempogram."""
    print("\n=== Tempogram References ===")

    sr = 22050
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 120 BPM clicks
    bpm = 120
    beat_interval = 60.0 / bpm
    y = np.zeros_like(t)
    beat_time = 0
    while beat_time < duration:
        idx = int(beat_time * sr)
        if idx < len(y) - 200:
            y[idx:idx+200] = np.exp(-np.linspace(0, 10, 200)) * \
                np.sin(2 * np.pi * 100 * np.linspace(0, 0.01, 200))
        beat_time += beat_interval
    y = y.astype(np.float32)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    save_array("beat_tempogram_onset_env", onset_env, {"sr": sr})

    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
    save_array("beat_tempogram", tg, {"sr": sr})


def generate_pitch_references():
    """Generate references for pitch estimation (YIN, pYIN)."""
    print("\n=== Pitch References ===")

    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    save_array("pitch_test_signal", y, {"sr": sr, "freq": 440})

    # YIN
    f0_yin = librosa.yin(y, fmin=65, fmax=2093, sr=sr)
    save_array("pitch_yin_f0", f0_yin, {"sr": sr, "fmin": 65, "fmax": 2093})

    # pYIN
    f0_pyin, voiced, voiced_prob = librosa.pyin(y, fmin=65, fmax=2093, sr=sr)
    # Replace NaN with -1 for JSON serialization
    f0_pyin_clean = np.where(np.isnan(f0_pyin), -1.0, f0_pyin)
    save_array("pitch_pyin_f0", f0_pyin_clean, {"sr": sr, "fmin": 65, "fmax": 2093})
    save_array("pitch_pyin_voiced", voiced.astype(np.float64), {"sr": sr})
    save_array("pitch_pyin_voiced_prob", voiced_prob, {"sr": sr})


def generate_cqt_references():
    """Generate references for CQT."""
    print("\n=== CQT References ===")

    sr = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    save_array("cqt_test_signal", y, {"sr": sr, "freq": 440})

    C = librosa.cqt(y, sr=sr)
    save_array("cqt_magnitude", np.abs(C), {"sr": sr})


def generate_pcen_references():
    """Generate references for PCEN."""
    print("\n=== PCEN References ===")

    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t) +
         0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

    S = librosa.feature.melspectrogram(y=y, sr=sr)
    save_array("pcen_input", S, {"sr": sr})

    P = librosa.pcen(S * (2 ** 31), sr=sr, hop_length=512)
    save_array("pcen_output", P, {"sr": sr})


def generate_effects_references():
    """Generate references for trim and split effects."""
    print("\n=== Effects (trim/split) References ===")

    sr = 22050

    # Trim: silence + tone + silence
    silence1 = np.zeros(int(0.2 * sr))
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.6, int(0.6 * sr), endpoint=False))
    silence2 = np.zeros(int(0.2 * sr))
    y_trim = np.concatenate([silence1, tone, silence2]).astype(np.float32)
    save_array("effects_trim_signal", y_trim, {"sr": sr})

    y_trimmed, indices = librosa.effects.trim(y_trim, top_db=20)
    save_array("effects_trim_result", y_trimmed, {"sr": sr})
    save_array("effects_trim_indices", np.array(indices, dtype=np.float64), {"sr": sr})

    # Split: two non-silent segments separated by silence
    seg1 = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, int(0.3 * sr), endpoint=False))
    gap = np.zeros(int(0.4 * sr))
    seg2 = 0.5 * np.sin(2 * np.pi * 880 * np.linspace(0, 0.3, int(0.3 * sr), endpoint=False))
    y_split = np.concatenate([seg1, gap, seg2]).astype(np.float32)
    save_array("effects_split_signal", y_split, {"sr": sr})

    intervals = librosa.effects.split(y_split, top_db=20)
    save_array("effects_split_intervals", intervals.astype(np.float64), {"sr": sr})


def generate_transition_references():
    """Generate references for transition_cycle and transition_local."""
    print("\n=== Transition References ===")

    trans_cycle = librosa.sequence.transition_cycle(4, prob=0.8)
    save_array("sequence_transition_cycle", trans_cycle, {"n_states": 4, "prob": 0.8})

    trans_local = librosa.sequence.transition_local(4, width=3)
    save_array("sequence_transition_local", trans_local, {"n_states": 4, "width": 3})


def generate_onset_multi_references():
    """Generate references for onset_strength_multi."""
    print("\n=== Onset Multi References ===")

    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Reuse same signal as onset_test_signal
    y = np.zeros_like(t)
    click_times = [0.0, 0.5, 1.0, 1.5]
    for ct in click_times:
        idx = int(ct * sr)
        if idx < len(y):
            y[idx:idx+100] = np.exp(-np.linspace(0, 5, 100))
    y = y.astype(np.float32)

    channels = [0, 32, 64, 96, 128]
    osm = librosa.onset.onset_strength_multi(y=y, sr=sr, channels=channels)
    save_array("onset_strength_multi", osm, {"sr": sr, "channels": channels})


def generate_nmf_references():
    """Generate references for NMF decomposition."""
    print("\n=== NMF References ===")

    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t) +
         0.3 * np.sin(2 * np.pi * 880 * t) +
         0.2 * np.sin(2 * np.pi * 1320 * t)).astype(np.float32)

    S = np.abs(librosa.stft(y))
    # Use power spectrogram, take first 64 bins x 10 frames to keep small
    S_small = (S[:64, :10] ** 2)
    save_array("decompose_nmf_input", S_small, {"sr": sr})

    W, H = librosa.decompose.decompose(S_small, n_components=4, sort=True)
    save_array("decompose_nmf_components", W, {"n_components": 4})
    save_array("decompose_nmf_activations", H, {"n_components": 4})


def generate_hybrid_cqt_references():
    """Generate references for hybrid CQT."""
    print("\n=== Hybrid CQT References ===")

    sr = 22050
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    save_array("hybrid_cqt_test_signal", y, {"sr": sr, "freq": 440})

    C = librosa.hybrid_cqt(y, sr=sr)
    save_array("hybrid_cqt_magnitude", C, {"sr": sr})


def generate_match_intervals_references():
    """Generate references for match_intervals."""
    print("\n=== Match Intervals References ===")

    ints_from = np.array([[3, 5], [1, 4], [4, 5]], dtype=np.float64)
    ints_to = np.array([[0, 2], [1, 3], [4, 5], [6, 7]], dtype=np.float64)

    save_array("match_intervals_from", ints_from, {})
    save_array("match_intervals_to", ints_to, {})

    result_strict = librosa.util.match_intervals(ints_from, ints_to, strict=True)
    save_array("match_intervals_result", result_strict.astype(np.float64), {})

    # Non-strict: reverse matching where [6,7] has no overlap with ints_from
    result_nonstrict = librosa.util.match_intervals(ints_to, ints_from, strict=False)
    save_array("match_intervals_nonstrict_result", result_nonstrict.astype(np.float64), {})


def generate_semitone_fb_references():
    """Generate references for semitone_filterbank."""
    print("\n=== Semitone Filterbank References ===")

    import scipy.signal

    filterbank, sample_rates = librosa.filters.semitone_filterbank(flayout="sos")
    center_freqs, _ = librosa.filters.mr_frequencies(tuning=0.0)

    save_array("semitone_fb_sample_rates", sample_rates, {})
    save_array("semitone_fb_center_freqs", center_freqs, {})

    # Evaluate frequency response of each filter at its center frequency
    # This allows testing filter behavior regardless of Butterworth vs elliptic
    responses = np.zeros(len(filterbank))
    for i, (sos, sr, fc) in enumerate(zip(filterbank, sample_rates, center_freqs)):
        w, h = scipy.signal.sosfreqz(sos, worN=[fc / (sr / 2) * np.pi])
        responses[i] = np.abs(h[0])

    save_array("semitone_fb_filter_response", responses, {})


def generate_iirt_references():
    """Generate references for iirt (IIR time-frequency representation)."""
    print("\n=== IIRT References ===")

    import scipy.signal

    sr = 22050
    duration = 1.0
    t = np.arange(int(sr * duration)) / sr
    y = np.sin(2 * np.pi * 440 * t).astype(np.float64)

    save_array("iirt_test_signal", y, {"sr": sr, "duration": duration, "freq": 440})

    result = librosa.iirt(y=y, sr=sr, win_length=2048, hop_length=512, center=True)
    save_array("iirt_output", result, {"sr": sr, "win_length": 2048, "hop_length": 512})


def generate_reassigned_spectrogram_references():
    """Generate references for reassigned_spectrogram."""
    print("\n=== Reassigned Spectrogram References ===")

    sr = 4000
    duration = 0.5
    t = np.arange(int(sr * duration)) / sr
    y = np.sin(2 * np.pi * (200 + 1400 * t / duration) * t).astype(np.float64)

    save_array("reassigned_test_signal", y, {"sr": sr})

    freqs, times, mags = librosa.reassigned_spectrogram(
        y=y, sr=sr, n_fft=256, center=False, fill_nan=True
    )
    save_array("reassigned_freqs", freqs, {"sr": sr, "n_fft": 256})
    save_array("reassigned_times", times, {"sr": sr, "n_fft": 256})
    save_array("reassigned_mags", mags, {"sr": sr, "n_fft": 256})


def generate_fmt_references():
    """Generate references for fmt (Fast Mellin Transform)."""
    print("\n=== FMT References ===")

    n = 1024
    x = np.linspace(0, 1, num=n, endpoint=False)
    y = np.sin(2 * np.pi * 3.0 * x).astype(np.float64)

    save_array("fmt_test_signal", y, {"n": n, "freq": 3.0})

    result = librosa.fmt(y, n_fmt=512)
    # Save real and imaginary parts separately
    save_array("fmt_output_real", result.real.astype(np.float64), {"n_fmt": 512})
    save_array("fmt_output_imag", result.imag.astype(np.float64), {"n_fmt": 512})


def main():
    """Generate all reference data."""
    print("Generating cross-validation reference data...")
    print(f"Output directory: {OUTPUT_DIR}")

    generate_convert_references()
    generate_spectrum_references()
    generate_resample_references()
    generate_filters_references()
    generate_feature_references()
    generate_onset_references()
    generate_beat_references()
    generate_decompose_references()
    generate_sequence_references()
    generate_new_feature_references()
    generate_additional_feature_references()
    generate_chroma_cqt_references()
    generate_tempogram_references()
    generate_pitch_references()
    generate_cqt_references()
    generate_pcen_references()
    generate_effects_references()
    generate_transition_references()
    generate_onset_multi_references()
    generate_nmf_references()
    generate_hybrid_cqt_references()
    generate_match_intervals_references()
    generate_semitone_fb_references()
    generate_iirt_references()
    generate_reassigned_spectrogram_references()
    generate_fmt_references()

    print("\n=== Done ===")
    print(f"Reference files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
