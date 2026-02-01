/*
 * CDP Transform Functions - Time/Pitch/Spectral Transformations
 *
 * Audio transformation operations including time stretch, pitch shift, and spectral manipulation.
 */

#ifndef CDP_TRANSFORM_H
#define CDP_TRANSFORM_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Apply gain (loudness change) to audio.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   gain_db: Gain in decibels
 *
 * Returns: Processed audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_loudness(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double gain_db);

/*
 * Change playback speed (affects pitch and duration).
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   speed: Speed factor (2.0 = double speed, 0.5 = half speed)
 *
 * Returns: Resampled audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_speed(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input,
                               double speed);

/*
 * Time stretch (change duration without affecting pitch).
 *
 * Uses phase vocoder for high-quality time stretching.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   factor: Stretch factor (2.0 = twice as long)
 *   fft_size: FFT size (0 for default 1024)
 *   overlap: Overlap factor (0 for default 3)
 *
 * Returns: Time-stretched audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_time_stretch(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double factor,
                                      int fft_size,
                                      int overlap);

/*
 * Spectral blur (average spectral content over time).
 *
 * Creates smeared, sustained textures from transient sounds.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   blur_time: Blur window in seconds
 *   fft_size: FFT size (0 for default 1024)
 *
 * Returns: Blurred audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_spectral_blur(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double blur_time,
                                       int fft_size);

/*
 * Pitch shift (change pitch without affecting duration).
 *
 * Uses speed change followed by time stretch compensation.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   semitones: Pitch shift in semitones (positive = up, negative = down)
 *   fft_size: FFT size (0 for default 1024)
 *
 * Returns: Pitch-shifted audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_pitch_shift(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input,
                                     double semitones,
                                     int fft_size);

/*
 * Spectral frequency shift (add Hz offset to all frequencies).
 *
 * Creates inharmonic bell-like or metallic sounds.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   shift_hz: Frequency offset in Hz
 *   fft_size: FFT size (0 for default 1024)
 *
 * Returns: Frequency-shifted audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_spectral_shift(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        double shift_hz,
                                        int fft_size);

/*
 * Spectral frequency stretch (differential stretching).
 *
 * Stretches higher frequencies more than lower ones for inharmonic effects.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   max_stretch: Maximum stretch ratio for highest frequencies
 *   freq_divide: Frequency below which no stretching occurs
 *   exponent: Stretching curve (1.0 = linear)
 *   fft_size: FFT size (0 for default 1024)
 *
 * Returns: Frequency-stretched audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_spectral_stretch(cdp_lib_ctx* ctx,
                                          const cdp_lib_buffer* input,
                                          double max_stretch,
                                          double freq_divide,
                                          double exponent,
                                          int fft_size);

#ifdef __cplusplus
}
#endif

#endif /* CDP_TRANSFORM_H */
