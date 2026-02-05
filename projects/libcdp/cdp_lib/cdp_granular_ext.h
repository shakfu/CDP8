/*
 * CDP Granular Extended - Additional grain manipulation algorithms
 *
 * Implements: grain_reorder, grain_rerhythm, grain_reverse, grain_timewarp,
 *             grain_repitch, grain_position
 */

#ifndef CDP_GRANULAR_EXT_H
#define CDP_GRANULAR_EXT_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Grain Reorder - Rearranges grains according to a permutation pattern.
 *
 * Detects grains using amplitude gate, then reorders them based on the
 * provided order array.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   order: Array of grain indices specifying new order (NULL = shuffle)
 *   order_count: Number of elements in order array (0 = shuffle all)
 *   gate: Amplitude threshold for grain detection (0-1). Default 0.1.
 *   grainsize_ms: Typical grain size in ms for detection. Default 50.
 *   seed: Random seed for shuffling (when order is NULL)
 *
 * Returns: Reordered audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_grain_reorder(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       const int* order,
                                       size_t order_count,
                                       double gate,
                                       double grainsize_ms,
                                       unsigned int seed);

/*
 * Grain Rerhythm - Changes the timing between grains.
 *
 * Detects grains, then repositions them according to new timing values.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   times: Array of new grain start times in seconds (NULL = use ratios)
 *   time_count: Number of time values
 *   ratios: Array of inter-grain duration ratios (NULL = use times)
 *   ratio_count: Number of ratio values
 *   gate: Amplitude threshold for grain detection (0-1)
 *   grainsize_ms: Typical grain size in ms
 *   seed: Random seed (for when neither times nor ratios provided)
 *
 * Returns: Rerhythmed audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_grain_rerhythm(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        const double* times,
                                        size_t time_count,
                                        const double* ratios,
                                        size_t ratio_count,
                                        double gate,
                                        double grainsize_ms,
                                        unsigned int seed);

/*
 * Grain Reverse - Reverses the order of grains.
 *
 * Detects grains and outputs them in reverse order. Individual grains
 * are NOT reversed internally (use grain_repitch with negative ratio for that).
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   gate: Amplitude threshold for grain detection (0-1)
 *   grainsize_ms: Typical grain size in ms
 *
 * Returns: Reversed-grain audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_grain_reverse(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double gate,
                                       double grainsize_ms);

/*
 * Grain Timewarp - Non-linear time stretch of grain sequence.
 *
 * Stretches or compresses the timing between grains according to a
 * time-varying stretch factor.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   stretch: Uniform stretch factor (>1 = slower, <1 = faster)
 *   stretch_curve: Array of (time, stretch) pairs for varying stretch (or NULL)
 *   curve_points: Number of points in stretch_curve (pairs)
 *   gate: Amplitude threshold for grain detection (0-1)
 *   grainsize_ms: Typical grain size in ms
 *
 * Returns: Time-warped audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_grain_timewarp(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        double stretch,
                                        const double* stretch_curve,
                                        size_t curve_points,
                                        double gate,
                                        double grainsize_ms);

/*
 * Grain Repitch - Variable pitch shifting per grain.
 *
 * Time-stretches individual grains to change their pitch without
 * affecting overall timing.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   pitch_semitones: Uniform pitch shift in semitones
 *   pitch_curve: Array of (time, semitones) pairs for varying pitch (or NULL)
 *   curve_points: Number of points in pitch_curve
 *   gate: Amplitude threshold for grain detection (0-1)
 *   grainsize_ms: Typical grain size in ms
 *
 * Returns: Repitched audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_grain_repitch(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double pitch_semitones,
                                       const double* pitch_curve,
                                       size_t curve_points,
                                       double gate,
                                       double grainsize_ms);

/*
 * Grain Position - Position grains at specific times.
 *
 * Places grains at specified positions in the output, allowing
 * rhythmic restructuring.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   positions: Array of output positions in seconds
 *   position_count: Number of positions
 *   duration: Total output duration (0 = auto)
 *   gate: Amplitude threshold for grain detection (0-1)
 *   grainsize_ms: Typical grain size in ms
 *
 * Returns: Repositioned audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_grain_position(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        const double* positions,
                                        size_t position_count,
                                        double duration,
                                        double gate,
                                        double grainsize_ms);

/*
 * Grain Omit - Selectively remove grains.
 *
 * Keeps only specified grains from the input.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   keep: Number of grains to keep out of each group
 *   out_of: Group size (e.g., keep 1 out_of 3)
 *   gate: Amplitude threshold for grain detection (0-1)
 *   grainsize_ms: Typical grain size in ms
 *   seed: Random seed for selection variation
 *
 * Returns: Filtered audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_grain_omit(cdp_lib_ctx* ctx,
                                    const cdp_lib_buffer* input,
                                    int keep,
                                    int out_of,
                                    double gate,
                                    double grainsize_ms,
                                    unsigned int seed);

/*
 * Grain Duplicate - Repeat grains.
 *
 * Duplicates each grain a specified number of times.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   repeats: Number of times to repeat each grain
 *   gate: Amplitude threshold for grain detection (0-1)
 *   grainsize_ms: Typical grain size in ms
 *   seed: Random seed for variation
 *
 * Returns: Audio buffer with duplicated grains, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_grain_duplicate(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         int repeats,
                                         double gate,
                                         double grainsize_ms,
                                         unsigned int seed);

#ifdef __cplusplus
}
#endif

#endif /* CDP_GRANULAR_EXT_H */
