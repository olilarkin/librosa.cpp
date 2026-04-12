#ifndef LIBROSA_C_H
#define LIBROSA_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LibrosaVector {
    int64_t count;
    double *data;
} LibrosaVector;

typedef struct LibrosaMatrix {
    int64_t rows;
    int64_t columns;
    double *data;
} LibrosaMatrix;

typedef struct LibrosaIndexVector {
    int64_t count;
    int64_t *data;
} LibrosaIndexVector;

typedef struct LibrosaAudioData {
    int64_t channels;
    int64_t samples;
    double sample_rate;
    double *data;
} LibrosaAudioData;

typedef struct LibrosaAudioFileInfo {
    int64_t samples;
    double sample_rate;
    int32_t channels;
    double duration;
} LibrosaAudioFileInfo;

typedef struct LibrosaBeatTrackResult {
    double tempo;
    LibrosaIndexVector beats;
} LibrosaBeatTrackResult;

typedef struct LibrosaTrimResult {
    LibrosaVector audio;
    int64_t start;
    int64_t end;
} LibrosaTrimResult;

typedef struct LibrosaDTWResult {
    LibrosaMatrix cost;
    LibrosaIndexVector path;
} LibrosaDTWResult;

#define LIBROSA_STATUS_OK 0
#define LIBROSA_STATUS_ERROR 1

const char *librosa_last_error_message(void);
void librosa_clear_error(void);

void librosa_vector_free(LibrosaVector *vector);
void librosa_matrix_free(LibrosaMatrix *matrix);
void librosa_index_vector_free(LibrosaIndexVector *vector);
void librosa_audio_data_free(LibrosaAudioData *audio);
void librosa_beat_track_result_free(LibrosaBeatTrackResult *result);
void librosa_trim_result_free(LibrosaTrimResult *result);
void librosa_dtw_result_free(LibrosaDTWResult *result);

int librosa_audio_info(const char *path, LibrosaAudioFileInfo *out_info);
int librosa_load(const char *path,
                 int has_target_sample_rate,
                 double target_sample_rate,
                 int mono,
                 double offset,
                 int has_duration,
                 double duration,
                 LibrosaAudioData *out_audio);

int librosa_midi_to_hz(double midi, double *out_hz);
int librosa_hz_to_midi(double hz, double *out_midi);
int librosa_hz_to_mel(double hz, int htk, double *out_mel);
int librosa_mel_to_hz(double mel, int htk, double *out_hz);
int librosa_note_to_midi(const char *note, int round_midi, double *out_midi);
int librosa_note_to_hz(const char *note, double *out_hz);
int librosa_fft_frequencies(double sample_rate, int n_fft, LibrosaVector *out);
int librosa_mel_frequencies(int n_mels, double fmin, double fmax, int htk, LibrosaVector *out);

int librosa_tone(double frequency,
                 double sample_rate,
                 int has_length,
                 int64_t length,
                 int has_duration,
                 double duration,
                 int has_phi,
                 double phi,
                 LibrosaVector *out);
int librosa_chirp(double fmin,
                  double fmax,
                  double sample_rate,
                  int has_length,
                  int64_t length,
                  int has_duration,
                  double duration,
                  int linear,
                  int has_phi,
                  double phi,
                  LibrosaVector *out);
int librosa_resample(const double *y,
                     int64_t count,
                     double original_sample_rate,
                     double target_sample_rate,
                     const char *res_type,
                     int fix,
                     int scale,
                     LibrosaVector *out);

int librosa_stft_magnitude(const double *y,
                           int64_t count,
                           int n_fft,
                           int hop_length,
                           LibrosaMatrix *out);
int librosa_amplitude_to_db(const double *s,
                            int64_t rows,
                            int64_t columns,
                            double ref,
                            double amin,
                            int has_top_db,
                            double top_db,
                            LibrosaMatrix *out);
int librosa_power_to_db(const double *s,
                        int64_t rows,
                        int64_t columns,
                        double ref,
                        double amin,
                        int has_top_db,
                        double top_db,
                        LibrosaMatrix *out);
int librosa_hpss(const double *s,
                 int64_t rows,
                 int64_t columns,
                 int kernel_size,
                 double power,
                 int mask,
                 double margin,
                 LibrosaMatrix *out_harmonic,
                 LibrosaMatrix *out_percussive);
int librosa_cqt_magnitude(const double *y,
                          int64_t count,
                          double sample_rate,
                          int hop_length,
                          int has_fmin,
                          double fmin,
                          int n_bins,
                          int bins_per_octave,
                          LibrosaMatrix *out);
int librosa_chroma_cqt(const double *y,
                       int64_t count,
                       double sample_rate,
                       int hop_length,
                       int has_fmin,
                       double fmin,
                       int n_chroma,
                       int n_octaves,
                       int bins_per_octave,
                       LibrosaMatrix *out);
int librosa_harmonic_effect(const double *y,
                            int64_t count,
                            int kernel_size,
                            double power,
                            int mask,
                            double margin,
                            int n_fft,
                            int hop_length,
                            LibrosaVector *out);
int librosa_nn_filter(const double *s,
                      int64_t rows,
                      int64_t columns,
                      const char *metric,
                      int aggregate_median,
                      int k,
                      int width,
                      LibrosaMatrix *out);
int librosa_softmask(const double *x,
                     const double *x_ref,
                     int64_t rows,
                     int64_t columns,
                     double power,
                     int split_zeros,
                     LibrosaMatrix *out);
int librosa_median_filter(const double *s,
                          int64_t rows,
                          int64_t columns,
                          int size_rows,
                          int size_columns,
                          LibrosaMatrix *out);
int librosa_sync(const double *s,
                 int64_t rows,
                 int64_t columns,
                 const int64_t *indices,
                 int64_t index_count,
                 int aggregate_median,
                 int pad,
                 int axis,
                 LibrosaMatrix *out);
int librosa_recurrence_matrix(const double *s,
                              int64_t rows,
                              int64_t columns,
                              int k,
                              int width,
                              const char *metric,
                              int sym,
                              const char *mode,
                              double bandwidth,
                              int self,
                              LibrosaMatrix *out);
int librosa_timelag_median_filter(const double *s,
                                  int64_t rows,
                                  int64_t columns,
                                  int size_rows,
                                  int size_columns,
                                  int pad,
                                  LibrosaMatrix *out);
int librosa_laplacian_components(const double *s,
                                 int64_t rows,
                                 int64_t columns,
                                 int components,
                                 int median_filter_rows,
                                 LibrosaMatrix *out);

int librosa_melspectrogram(const double *y,
                           int64_t count,
                           double sample_rate,
                           int n_fft,
                           int hop_length,
                           int n_mels,
                           double fmin,
                           int has_fmax,
                           double fmax,
                           int htk,
                           int norm_slaney,
                           LibrosaMatrix *out);
int librosa_mfcc(const double *y,
                 int64_t count,
                 double sample_rate,
                 int n_mfcc,
                 int n_fft,
                 int hop_length,
                 int n_mels,
                 double fmin,
                 int has_fmax,
                 double fmax,
                 int htk,
                 LibrosaMatrix *out);
int librosa_chroma_stft(const double *y,
                        int64_t count,
                        double sample_rate,
                        int n_fft,
                        int hop_length,
                        int n_chroma,
                        LibrosaMatrix *out);
int librosa_chroma_stft_options(const double *y,
                                int64_t count,
                                double sample_rate,
                                int n_fft,
                                int hop_length,
                                int n_chroma,
                                int has_tuning,
                                double tuning,
                                double norm,
                                LibrosaMatrix *out);
int librosa_dtw(const double *x,
                int64_t x_rows,
                int64_t x_columns,
                const double *y,
                int64_t y_rows,
                int64_t y_columns,
                const char *metric,
                int subseq,
                LibrosaDTWResult *out);
int librosa_spectral_centroid(const double *y,
                              int64_t count,
                              double sample_rate,
                              int n_fft,
                              int hop_length,
                              LibrosaMatrix *out);
int librosa_spectral_bandwidth(const double *y,
                               int64_t count,
                               double sample_rate,
                               int n_fft,
                               int hop_length,
                               double p,
                               int norm,
                               LibrosaMatrix *out);
int librosa_spectral_rolloff(const double *y,
                             int64_t count,
                             double sample_rate,
                             int n_fft,
                             int hop_length,
                             double roll_percent,
                             LibrosaMatrix *out);
int librosa_spectral_flatness(const double *y,
                              int64_t count,
                              int n_fft,
                              int hop_length,
                              LibrosaMatrix *out);
int librosa_spectral_contrast(const double *y,
                              int64_t count,
                              double sample_rate,
                              int n_fft,
                              int hop_length,
                              double fmin,
                              int n_bands,
                              double quantile,
                              int linear,
                              LibrosaMatrix *out);
int librosa_rms(const double *y,
                int64_t count,
                int frame_length,
                int hop_length,
                int center,
                LibrosaMatrix *out);
int librosa_zero_crossing_rate(const double *y,
                               int64_t count,
                               int frame_length,
                               int hop_length,
                               int center,
                               double threshold,
                               LibrosaMatrix *out);

int librosa_onset_strength(const double *y,
                           int64_t count,
                           double sample_rate,
                           int n_fft,
                           int hop_length,
                           int lag,
                           int max_size,
                           int detrend,
                           int center,
                           LibrosaVector *out);
int librosa_onset_strength_spectrogram(const double *s,
                                       int64_t rows,
                                       int64_t columns,
                                       double sample_rate,
                                       int n_fft,
                                       int hop_length,
                                       int lag,
                                       int max_size,
                                       int detrend,
                                       int center,
                                       LibrosaVector *out);
int librosa_onset_detect(const double *y,
                         int64_t count,
                         double sample_rate,
                         int hop_length,
                         int backtrack,
                         int normalize,
                         LibrosaIndexVector *out);
int librosa_onset_detect_envelope(const double *onset_envelope,
                                  int64_t count,
                                  double sample_rate,
                                  int hop_length,
                                  int backtrack,
                                  int normalize,
                                  LibrosaIndexVector *out);
int librosa_tempo(const double *onset_envelope,
                  int64_t count,
                  double sample_rate,
                  int hop_length,
                  double start_bpm,
                  double std_bpm,
                  double ac_size,
                  int has_max_tempo,
                  double max_tempo,
                  double *out_tempo);
int librosa_tempo_audio(const double *y,
                        int64_t count,
                        double sample_rate,
                        int hop_length,
                        double start_bpm,
                        double std_bpm,
                        double ac_size,
                        int has_max_tempo,
                        double max_tempo,
                        double *out_tempo);
int librosa_beat_track(const double *onset_envelope,
                       int64_t count,
                       double sample_rate,
                       int hop_length,
                       double start_bpm,
                       double tightness,
                       int trim,
                       int has_bpm,
                       double bpm,
                       LibrosaBeatTrackResult *out);
int librosa_beat_track_audio(const double *y,
                             int64_t count,
                             double sample_rate,
                             int hop_length,
                             double start_bpm,
                             double tightness,
                             int trim,
                             int has_bpm,
                             double bpm,
                             LibrosaBeatTrackResult *out);

int librosa_time_stretch(const double *y,
                         int64_t count,
                         double rate,
                         int n_fft,
                         int hop_length,
                         LibrosaVector *out);
int librosa_pitch_shift(const double *y,
                        int64_t count,
                        double sample_rate,
                        double steps,
                        int bins_per_octave,
                        const char *res_type,
                        int n_fft,
                        int hop_length,
                        LibrosaVector *out);
int librosa_trim(const double *y,
                 int64_t count,
                 double top_db,
                 int frame_length,
                 int hop_length,
                 LibrosaTrimResult *out);

#ifdef __cplusplus
}
#endif

#endif
