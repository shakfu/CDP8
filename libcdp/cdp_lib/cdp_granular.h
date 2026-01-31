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

#ifdef __cplusplus
}
#endif

#endif /* CDP_GRANULAR_H */
