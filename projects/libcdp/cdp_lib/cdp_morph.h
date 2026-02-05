/*
 * CDP Morph Functions - Spectral Morphing and Cross-synthesis
 *
 * Spectral interpolation and cross-synthesis between two sounds.
 */

#ifndef CDP_MORPH_H
#define CDP_MORPH_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Morph - interpolates between two sounds in the spectral domain.
 *
 * Args:
 *   ctx: Library context
 *   input1: First input audio buffer
 *   input2: Second input audio buffer
 *   morph_start: Position (0-1) where morphing begins
 *   morph_end: Position (0-1) where morphing completes
 *   exponent: Curve exponent for interpolation (1 = linear)
 *   fft_size: FFT size (0 for default 1024)
 *
 * Returns: Morphed audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_morph(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input1,
                               const cdp_lib_buffer* input2,
                               double morph_start,
                               double morph_end,
                               double exponent,
                               int fft_size);

/*
 * Morph glide - creates a smooth glide between two sounds.
 *
 * Uses representative spectral frames and interpolates over specified duration.
 *
 * Args:
 *   ctx: Library context
 *   input1: First input audio buffer
 *   input2: Second input audio buffer
 *   duration: Output duration in seconds
 *   fft_size: FFT size (0 for default 1024)
 *
 * Returns: Glided audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_morph_glide(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input1,
                                     const cdp_lib_buffer* input2,
                                     double duration,
                                     int fft_size);

/*
 * Cross-synthesis - combines amplitude from one sound with frequency from another.
 *
 * Args:
 *   ctx: Library context
 *   input1: First input audio buffer
 *   input2: Second input audio buffer
 *   mode: 0 = amp from input1 + freq from input2,
 *         1 = amp from input2 + freq from input1
 *   mix: Dry/wet mix (0 = original, 1 = full cross-synth)
 *   fft_size: FFT size (0 for default 1024)
 *
 * Returns: Cross-synthesized audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_cross_synth(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input1,
                                     const cdp_lib_buffer* input2,
                                     int mode,
                                     double mix,
                                     int fft_size);

#ifdef __cplusplus
}
#endif

#endif /* CDP_MORPH_H */
