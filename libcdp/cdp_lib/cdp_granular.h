/*
 * CDP Granular Processing - Direct buffer implementations
 *
 * Implements granular synthesis operations working directly
 * on memory buffers without file I/O.
 *
 * Note: Implementations are in cdp_lib.c to access the context structure.
 */

#ifndef CDP_GRANULAR_H
#define CDP_GRANULAR_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Apply granular brassage.
 *
 * Breaks sound into grains and reassembles with optional modifications.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   velocity: Playback speed through source (1.0 = normal, 2.0 = twice as fast)
 *   density: Grain density multiplier (1.0 = normal, 2.0 = twice as many grains)
 *   grainsize_ms: Grain size in milliseconds
 *   scatter: Time scatter of grains (0.0 to 1.0, 0 = no scatter)
 *   pitch_shift: Pitch shift per grain in semitones (-12 to +12)
 *   amp_variation: Random amplitude variation (0.0 to 1.0)
 *
 * Returns: New buffer with brassage applied, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_brassage(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double velocity,
                                  double density,
                                  double grainsize_ms,
                                  double scatter,
                                  double pitch_shift,
                                  double amp_variation);

/*
 * Freeze a segment of audio by repeated iteration.
 *
 * Creates a sustained texture by repeating a frozen segment with variations.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   start_time: Start of segment to freeze (seconds)
 *   end_time: End of segment to freeze (seconds)
 *   duration: Output duration in seconds
 *   delay: Average delay between iterations in seconds
 *   randomize: Delay time randomization (0.0 to 1.0)
 *   pitch_scatter: Max random pitch shift in semitones (0 to 12)
 *   amp_cut: Max random amplitude reduction (0.0 to 1.0)
 *   gain: Gain adjustment for frozen segment
 *
 * Returns: New buffer with frozen audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_freeze(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                double start_time,
                                double end_time,
                                double duration,
                                double delay,
                                double randomize,
                                double pitch_scatter,
                                double amp_cut,
                                double gain);

/*
 * Generate grain cloud from source audio (CDP: grain)
 *
 * Extracts grains based on amplitude threshold and generates a cloud
 * by placing them at random or regular intervals.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   gate: Amplitude threshold for grain detection (0.0 to 1.0)
 *   grainsize_ms: Target grain size in milliseconds
 *   density: Grain density (grains per second)
 *   duration: Output duration in seconds (0 = same as input)
 *   scatter: Position scatter amount (0.0 to 1.0)
 *   seed: Random seed (0 = use time)
 *
 * Returns: New buffer with grain cloud, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_grain_cloud(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input,
                                     double gate,
                                     double grainsize_ms,
                                     double density,
                                     double duration,
                                     double scatter,
                                     unsigned int seed);

/*
 * Extend audio duration using grain repetition (CDP: grainex extend)
 *
 * Finds grains in source and extends duration by repeating grains
 * with variations.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   grainsize_ms: Window size to detect grains (milliseconds)
 *   trough: Acceptable trough height relative to peaks (0.0 to 1.0)
 *   extension: How much duration to add (seconds)
 *   start_time: Start of grain material in source (seconds)
 *   end_time: End of grain material in source (seconds)
 *
 * Returns: New buffer with extended audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_grain_extend(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double grainsize_ms,
                                      double trough,
                                      double extension,
                                      double start_time,
                                      double end_time);

/*
 * Generate simple texture (CDP: texture SIMPLE_TEX)
 *
 * Creates texture by layering source at multiple transpositions
 * and time offsets.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   duration: Output duration in seconds
 *   density: Events per second
 *   pitch_range: Pitch range in semitones (symmetric around 0)
 *   amp_range: Amplitude variation (0.0 to 1.0)
 *   spatial_range: Stereo spread (0.0 to 1.0, 0 = mono center)
 *   seed: Random seed (0 = use time)
 *
 * Returns: New stereo buffer with texture, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_texture_simple(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        double duration,
                                        double density,
                                        double pitch_range,
                                        double amp_range,
                                        double spatial_range,
                                        unsigned int seed);

/*
 * Generate multi-layer texture (CDP: texture GROUPS)
 *
 * Creates complex texture with grouped events and decorations.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   duration: Output duration in seconds
 *   density: Events per second
 *   group_size: Average notes per group (1-16)
 *   group_spread: Time spread within group (seconds)
 *   pitch_range: Pitch range in semitones
 *   pitch_center: Center pitch offset in semitones
 *   amp_decay: Amplitude decay through group (0.0 to 1.0)
 *   seed: Random seed (0 = use time)
 *
 * Returns: New buffer with multi-layer texture, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_texture_multi(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double duration,
                                       double density,
                                       int group_size,
                                       double group_spread,
                                       double pitch_range,
                                       double pitch_center,
                                       double amp_decay,
                                       unsigned int seed);

#ifdef __cplusplus
}
#endif

#endif /* CDP_GRANULAR_H */
