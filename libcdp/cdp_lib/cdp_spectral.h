/*
 * CDP Spectral Processing - Direct buffer implementations
 *
 * These functions implement spectral processing operations
 * (FFT analysis/synthesis, time stretch, etc.) working directly
 * on memory buffers without file I/O.
 */

#ifndef CDP_SPECTRAL_H
#define CDP_SPECTRAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Spectral frame structure - amplitude and frequency pairs
 */
typedef struct cdp_spectral_frame {
    float *data;           /* Amp/freq pairs: [a0,f0,a1,f1,...] */
    int num_bins;          /* Number of frequency bins (N/2 + 1) */
    int fft_size;          /* Original FFT size N */
    float sample_rate;
} cdp_spectral_frame;

/*
 * Spectral data - sequence of frames
 */
typedef struct cdp_spectral_data {
    cdp_spectral_frame *frames;
    int num_frames;
    int num_bins;
    int fft_size;
    int overlap;
    float sample_rate;
    float frame_time;      /* Duration of one frame in seconds */
} cdp_spectral_data;

/*
 * Perform FFT analysis on audio buffer.
 *
 * Args:
 *   audio: Input audio samples
 *   num_samples: Number of samples
 *   channels: Number of channels (will convert to mono if > 1)
 *   sample_rate: Sample rate in Hz
 *   fft_size: FFT window size (power of 2)
 *   overlap: Overlap factor (1-4)
 *
 * Returns: Spectral data, or NULL on error.
 */
cdp_spectral_data* cdp_spectral_analyze(const float *audio, size_t num_samples,
                                         int channels, int sample_rate,
                                         int fft_size, int overlap);

/*
 * Perform FFT synthesis from spectral data.
 *
 * Args:
 *   spectral: Input spectral data
 *   out_samples: Output - number of samples produced
 *
 * Returns: Audio samples, or NULL on error.
 */
float* cdp_spectral_synthesize(const cdp_spectral_data *spectral,
                                size_t *out_samples);

/*
 * Time-stretch spectral data.
 *
 * Args:
 *   input: Input spectral data
 *   factor: Stretch factor (2.0 = twice as long)
 *
 * Returns: New stretched spectral data, or NULL on error.
 */
cdp_spectral_data* cdp_spectral_time_stretch(const cdp_spectral_data *input,
                                              double factor);

/*
 * Blur (average) spectral data over time.
 *
 * Args:
 *   input: Input spectral data
 *   num_windows: Number of windows to average over
 *
 * Returns: New blurred spectral data, or NULL on error.
 */
cdp_spectral_data* cdp_spectral_blur(const cdp_spectral_data *input,
                                      int num_windows);

/*
 * Free spectral data.
 */
void cdp_spectral_data_free(cdp_spectral_data *data);

/*
 * Shift all frequencies by a fixed Hz offset.
 *
 * Args:
 *   input: Input spectral data
 *   shift_hz: Frequency offset in Hz (positive = up, negative = down)
 *
 * Returns: New spectral data with shifted frequencies, or NULL on error.
 */
cdp_spectral_data* cdp_spectral_freq_shift(const cdp_spectral_data *input,
                                            double shift_hz);

/*
 * Stretch frequencies differentially (higher frequencies stretched more).
 *
 * Creates inharmonic effects by applying progressively more transposition
 * to higher frequencies.
 *
 * Args:
 *   input: Input spectral data
 *   max_stretch: Maximum transposition ratio for highest frequencies
 *   freq_divide: Frequency below which no stretching occurs
 *   exponent: Stretching curve (1.0 = linear, >1 = more effect on highs)
 *
 * Returns: New spectral data with stretched frequencies, or NULL on error.
 */
cdp_spectral_data* cdp_spectral_freq_stretch(const cdp_spectral_data *input,
                                              double max_stretch,
                                              double freq_divide,
                                              double exponent);

/*
 * Apply lowpass filter in spectral domain.
 *
 * Args:
 *   input: Input spectral data
 *   cutoff_freq: Cutoff frequency in Hz
 *   attenuation_db: Attenuation of stopped frequencies in dB (negative)
 *
 * Returns: New filtered spectral data, or NULL on error.
 */
cdp_spectral_data* cdp_spectral_filter_lowpass(const cdp_spectral_data *input,
                                                double cutoff_freq,
                                                double attenuation_db);

/*
 * Apply highpass filter in spectral domain.
 *
 * Args:
 *   input: Input spectral data
 *   cutoff_freq: Cutoff frequency in Hz
 *   attenuation_db: Attenuation of stopped frequencies in dB (negative)
 *
 * Returns: New filtered spectral data, or NULL on error.
 */
cdp_spectral_data* cdp_spectral_filter_highpass(const cdp_spectral_data *input,
                                                 double cutoff_freq,
                                                 double attenuation_db);

#ifdef __cplusplus
}
#endif

#endif /* CDP_SPECTRAL_H */
