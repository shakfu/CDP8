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

/*
 * Cut sound into waveset segments with decaying envelope.
 *
 * Divides audio into segments based on zero-crossing waveset boundaries
 * and applies a decaying envelope to each segment. Creates a "cut-up"
 * distortion effect.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   cycle_count: Number of wavesets (half-cycles) per segment (1-100)
 *   cycle_step: Number of wavesets to step between segment starts (1-100)
 *               If equal to cycle_count, segments are contiguous.
 *               If less, segments overlap. If greater, there are gaps.
 *   exponent: Envelope decay exponent (0.1 to 10.0)
 *             1.0 = linear decay, >1 = faster initial decay, <1 = slower decay
 *   min_level: Minimum output level to keep (0.0 to 1.0, 0 = keep all)
 *              Segments below this level are removed
 *
 * Returns: New buffer with cut segments, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_distort_cut(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input,
                                     int cycle_count,
                                     int cycle_step,
                                     double exponent,
                                     double min_level);

/*
 * Interpolate between waveset-groups at marked time positions.
 *
 * Finds waveset groups at specified time markers and morphs between them,
 * creating smooth transitions with optional phase flipping and randomization.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (will be converted to mono if stereo)
 *   markers: Array of time positions in seconds where waveset groups are located
 *   marker_count: Number of markers in the array (minimum 2)
 *   unit_ms: Approximate size of waveset group to find at each marker (1.0 to 100.0 ms)
 *   stretch: Time stretch factor for output (0.5 to 2.0, 1.0 = no stretch)
 *   random: Randomize waveset durations (0.0 to 1.0, 0 = no randomization)
 *   flip_phase: If non-zero, flip phase of alternate interpolated wavesets
 *   seed: Random seed (0 = time-based)
 *
 * Returns: New buffer with interpolated wavesets, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_distort_mark(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      const double* markers,
                                      int marker_count,
                                      double unit_ms,
                                      double stretch,
                                      double random,
                                      int flip_phase,
                                      unsigned int seed);

/*
 * Time-stretch by repeating wavecycles.
 *
 * Detects wavecycles and repeats groups of them to create time-stretching
 * or rhythmic stuttering effects.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (will be converted to mono if stereo)
 *   multiplier: Number of times to repeat each wavecycle group (1-100)
 *   cycle_count: Number of wavecycles in each repeated group (1-100)
 *   skip_cycles: Number of cycles to skip at start of file (0 or more)
 *   splice_ms: Splice length in milliseconds for smooth transitions (1.0 to 50.0)
 *   mode: 0 = time-stretch (output longer), 1 = maintain time (skip ahead after repeats)
 *
 * Returns: New buffer with repeated wavecycles, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_distort_repeat(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        int multiplier,
                                        int cycle_count,
                                        int skip_cycles,
                                        double splice_ms,
                                        int mode);

/*
 * Shift or swap half-wavecycle groups for phase distortion.
 *
 * Detects half-wavecycles (zero crossings) and either shifts alternate
 * groups forward in time, or swaps adjacent groups.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (will be converted to mono if stereo)
 *   group_size: Number of half-wavecycles per group (1-50)
 *               1 = single half-wavesets, 2 = waveset + half, etc.
 *   shift: For mode 0: number of groups to shift forward (1-50, with wrap-around)
 *          Ignored in mode 1.
 *   mode: 0 = shift alternate groups forward, 1 = swap adjacent groups
 *
 * Returns: New buffer with shifted/swapped wavecycles, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_distort_shift(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       int group_size,
                                       int shift,
                                       int mode);

/*
 * Progressive warp distortion with modular sample folding.
 *
 * Applies a progressive warp transformation that folds sample values
 * through a modular arithmetic pattern, creating unique distortion effects.
 * The warp increment is applied either per-sample (mode 0) or per-waveset
 * group (mode 1).
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (mode 1 requires mono)
 *   warp: Progressive sample multiplier (0.0001 to 0.1)
 *         Controls the rate of warping progression
 *   mode: 0 = warp increment per sample (works with stereo)
 *         1 = warp increment per waveset group (mono only)
 *   waveset_count: For mode 1: after how many wavesets does warp increment (1-256)
 *                  Ignored in mode 0.
 *
 * Returns: New buffer with warp distortion applied, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_distort_warp(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double warp,
                                      int mode,
                                      int waveset_count);

#ifdef __cplusplus
}
#endif

#endif /* CDP_DISTORT_H */
