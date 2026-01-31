/*
 * CDP Distortion Processing - Direct buffer implementations
 *
 * These functions implement distortion operations working directly
 * on memory buffers without file I/O.
 *
 * Note: Implementations are in cdp_lib.c to access the context structure.
 */

#ifndef CDP_DISTORT_H
#define CDP_DISTORT_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Apply clipping distortion.
 *
 * Clips the signal at a threshold, creating harmonic distortion.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (will be converted to mono if stereo)
 *   clip_level: Clipping threshold (0.0 to 1.0)
 *   depth: Depth/shape of distortion pattern (0.0 to 1.0)
 *          0 = hard clipping, 1 = soft clipping
 *
 * Returns: New buffer with distortion applied, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_distort_overload(cdp_lib_ctx* ctx,
                                          const cdp_lib_buffer* input,
                                          double clip_level,
                                          double depth);

/*
 * Reverse wavecycles for phase distortion.
 *
 * Detects zero crossings and reverses groups of wavecycles.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (will be converted to mono if stereo)
 *   cycle_count: Number of cycles in each reversed group
 *
 * Returns: New buffer with reversed cycles, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_distort_reverse(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         int cycle_count);

/*
 * Apply fractal distortion.
 *
 * Adds harmonic complexity by recursively overlaying wavecycle patterns.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (will be converted to mono if stereo)
 *   scaling: Fractal scaling factor (affects harmonic content)
 *   loudness: Output loudness (0.0 to 1.0)
 *
 * Returns: New buffer with fractal distortion, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_distort_fractal(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double scaling,
                                         double loudness);

/*
 * Shuffle segments of audio.
 *
 * Divides audio into chunks and rearranges them pseudo-randomly.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (will be converted to mono if stereo)
 *   chunk_count: Number of chunks to divide the audio into
 *   seed: Random seed for reproducible results (0 = use time-based seed)
 *
 * Returns: New buffer with shuffled segments, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_distort_shuffle(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         int chunk_count,
                                         unsigned int seed);

#ifdef __cplusplus
}
#endif

#endif /* CDP_DISTORT_H */
