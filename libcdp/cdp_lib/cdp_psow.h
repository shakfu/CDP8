/*
 * CDP Pitch-Synchronous Operations (PSOW)
 *
 * PSOW algorithms operate on pitch-synchronous grains (FOFs - Formant Wave Functions).
 * These functions automatically detect pitch and extract/manipulate grains at
 * pitch-synchronous boundaries for natural-sounding transformations.
 *
 * Based on CDP's psow algorithms.
 */

#ifndef CDP_PSOW_H
#define CDP_PSOW_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Time-stretch audio while preserving pitch using PSOLA.
 *
 * Uses pitch-synchronous overlap-add to stretch or compress time
 * without affecting pitch. Grains are extracted at pitch-period boundaries
 * and repositioned with overlap-add.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (mono only)
 *   stretch_factor: Time stretch ratio (0.5 = half duration, 2.0 = double duration)
 *                   Range: 0.25 to 4.0
 *   grain_count: Number of consecutive grains to keep together (1-8)
 *                Higher values preserve more local coherence
 *
 * Returns: New buffer with time-stretched audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_psow_stretch(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double stretch_factor,
                                      int grain_count);

/*
 * Extract pitch-synchronous grains from a position in the audio.
 *
 * Grabs one or more pitch periods (grains/FOFs) from the specified time
 * position and optionally extends them to create a sustained sound.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (mono only)
 *   time: Time in seconds at which to grab grains
 *   duration: Output duration in seconds (0 = grab single grain/chunk)
 *   grain_count: Number of consecutive grains to grab (1-16)
 *   density: Overlap density for output (0.5 to 4.0)
 *            1.0 = grains follow without overlap
 *            2.0 = grains overlap by 2x (can transpose up octave)
 *            0.5 = grains separated by gaps
 *
 * Returns: New buffer with grabbed/extended grains, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_psow_grab(cdp_lib_ctx* ctx,
                                   const cdp_lib_buffer* input,
                                   double time,
                                   double duration,
                                   int grain_count,
                                   double density);

/*
 * Duplicate pitch-synchronous grains for time-stretching.
 *
 * Time-stretches by repeating each group of grains a specified number of times.
 * Creates a more rhythmic/stuttered stretch than psow_stretch.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (mono only)
 *   repeat_count: Number of times to repeat each grain group (1-8)
 *   grain_count: Number of grains per group (1-8)
 *
 * Returns: New buffer with duplicated grains, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_psow_dupl(cdp_lib_ctx* ctx,
                                   const cdp_lib_buffer* input,
                                   int repeat_count,
                                   int grain_count);

/*
 * Interpolate between two pitch-synchronous grains.
 *
 * Creates a morphing sound by interpolating between two single-grain sounds.
 * Input sounds should be single grains extracted using psow_grab with duration=0.
 *
 * Args:
 *   ctx: Library context
 *   grain1: First grain (single pitch period)
 *   grain2: Second grain (single pitch period)
 *   start_dur: Duration to sustain initial grain (seconds)
 *   interp_dur: Duration of interpolation (seconds)
 *   end_dur: Duration to sustain final grain (seconds)
 *
 * Returns: New buffer with interpolated result, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_psow_interp(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* grain1,
                                     const cdp_lib_buffer* grain2,
                                     double start_dur,
                                     double interp_dur,
                                     double end_dur);

#ifdef __cplusplus
}
#endif

#endif /* CDP_PSOW_H */
