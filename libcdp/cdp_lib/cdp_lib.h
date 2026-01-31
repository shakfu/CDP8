/*
 * CDP Library Interface
 *
 * This provides a clean C API for calling CDP processing functions
 * on memory buffers, bypassing file I/O.
 */

#ifndef CDP_LIB_H
#define CDP_LIB_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define CDP_LIB_OK              0
#define CDP_LIB_ERR_MEMORY     -1
#define CDP_LIB_ERR_PARAM      -2
#define CDP_LIB_ERR_PROCESS    -3
#define CDP_LIB_ERR_INIT       -4

/*
 * Audio buffer structure for CDP library
 */
typedef struct cdp_lib_buffer {
    float *data;           /* Sample data (interleaved if stereo) */
    size_t length;         /* Number of samples */
    int channels;          /* Number of channels */
    int sample_rate;       /* Sample rate in Hz */
} cdp_lib_buffer;

/*
 * CDP library context
 */
typedef struct cdp_lib_ctx cdp_lib_ctx;

/*
 * Initialize CDP library.
 * Must be called before any processing functions.
 *
 * Returns: New context, or NULL on failure.
 */
cdp_lib_ctx* cdp_lib_init(void);

/*
 * Cleanup CDP library and free all resources.
 */
void cdp_lib_cleanup(cdp_lib_ctx* ctx);

/*
 * Get last error message.
 */
const char* cdp_lib_get_error(cdp_lib_ctx* ctx);

/*
 * Create a new buffer.
 */
cdp_lib_buffer* cdp_lib_buffer_create(size_t length, int channels, int sample_rate);

/*
 * Create a buffer from existing data (takes ownership).
 */
cdp_lib_buffer* cdp_lib_buffer_from_data(float *data, size_t length,
                                          int channels, int sample_rate);

/*
 * Free a buffer.
 */
void cdp_lib_buffer_free(cdp_lib_buffer* buf);

/* =========================================================================
 * Processing Functions
 * ========================================================================= */

/*
 * Time-stretch audio without changing pitch.
 *
 * Uses CDP's phase vocoder for high-quality stretching.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (will be converted to mono if stereo)
 *   factor: Stretch factor (2.0 = twice as long, 0.5 = half as long)
 *   fft_size: FFT window size (power of 2, 256-8192). Default 1024.
 *   overlap: Overlap factor (1-4). Default 3.
 *
 * Returns: New buffer with stretched audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_time_stretch(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double factor,
                                      int fft_size,
                                      int overlap);

/*
 * Spectral blur - average spectrum over time.
 *
 * Creates a smeared, washed-out effect.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   blur_time: Time to blur over in seconds
 *   fft_size: FFT window size
 *
 * Returns: New buffer with blurred audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_spectral_blur(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double blur_time,
                                       int fft_size);

/*
 * Modify loudness (gain in dB).
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   gain_db: Gain change in dB (positive = louder)
 *
 * Returns: New buffer with modified loudness, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_loudness(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double gain_db);

/*
 * Change playback speed (affects both pitch and duration).
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   speed: Speed factor (2.0 = double speed, 0.5 = half speed)
 *
 * Returns: New buffer with modified speed, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_speed(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input,
                               double speed);

/*
 * Pitch shift without changing duration.
 *
 * Combines speed change with time-stretch compensation.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   semitones: Pitch shift in semitones (positive = up, negative = down)
 *   fft_size: FFT window size for time stretch. Default 1024.
 *
 * Returns: New buffer with pitch-shifted audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_pitch_shift(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input,
                                     double semitones,
                                     int fft_size);

/*
 * Shift all frequencies by a fixed Hz offset.
 *
 * Unlike pitch shift, this adds a constant Hz value to all frequencies,
 * creating inharmonic effects.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   shift_hz: Frequency offset in Hz (positive = up, negative = down)
 *   fft_size: FFT window size. Default 1024.
 *
 * Returns: New buffer with shifted frequencies, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_spectral_shift(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        double shift_hz,
                                        int fft_size);

/*
 * Stretch frequencies differentially (higher frequencies stretched more).
 *
 * Creates inharmonic effects by applying progressively more transposition
 * to higher frequencies.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   max_stretch: Maximum transposition ratio for highest frequencies
 *   freq_divide: Frequency below which no stretching occurs
 *   exponent: Stretching curve (1.0 = linear, >1 = more effect on highs)
 *   fft_size: FFT window size. Default 1024.
 *
 * Returns: New buffer with stretched spectrum, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_spectral_stretch(cdp_lib_ctx* ctx,
                                          const cdp_lib_buffer* input,
                                          double max_stretch,
                                          double freq_divide,
                                          double exponent,
                                          int fft_size);

/*
 * Apply lowpass filter.
 *
 * Attenuates frequencies above the cutoff.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   cutoff_freq: Cutoff frequency in Hz
 *   attenuation_db: Attenuation in dB (negative, e.g. -60)
 *   fft_size: FFT window size. Default 1024.
 *
 * Returns: New buffer with filtered audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_filter_lowpass(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        double cutoff_freq,
                                        double attenuation_db,
                                        int fft_size);

/*
 * Apply highpass filter.
 *
 * Attenuates frequencies below the cutoff.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   cutoff_freq: Cutoff frequency in Hz
 *   attenuation_db: Attenuation in dB (negative, e.g. -60)
 *   fft_size: FFT window size. Default 1024.
 *
 * Returns: New buffer with filtered audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_filter_highpass(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double cutoff_freq,
                                         double attenuation_db,
                                         int fft_size);

#ifdef __cplusplus
}
#endif

#endif /* CDP_LIB_H */
