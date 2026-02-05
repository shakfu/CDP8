/*
 * CDP Flutter - Spatial Tremolo Effect
 *
 * Flutter distributes a loudness tremulation (tremolo) around stereo or
 * multichannel output. The tremolo cycles between channel groups, creating
 * a spatial movement effect.
 *
 * Based on CDP's flutter algorithm.
 */

#ifndef CDP_FLUTTER_H
#define CDP_FLUTTER_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Apply flutter (spatial tremolo) effect.
 *
 * Creates a tremolo that alternates between left and right channels,
 * producing a spatial movement effect.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (mono converts to stereo, stereo required)
 *   frequency: Tremolo frequency in Hz (0.1 to 50.0)
 *   depth: Tremolo depth (0.0 to 16.0). 1.0 = full depth to silence,
 *          >1.0 = narrower peaks, sharper transitions
 *   gain: Overall output gain (0.0 to 1.0)
 *   randomize: If non-zero, randomize channel order after each cycle
 *
 * Returns: New stereo buffer with flutter effect, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_flutter(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double frequency,
                                 double depth,
                                 double gain,
                                 int randomize);

/*
 * Apply flutter with custom channel sets (multichannel).
 *
 * For multichannel audio, allows specifying which channels are
 * grouped together for the tremolo effect.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (must be multichannel)
 *   frequency: Tremolo frequency in Hz (0.1 to 50.0)
 *   depth: Tremolo depth (0.0 to 16.0)
 *   gain: Overall output gain (0.0 to 1.0)
 *   channel_sets: Array of channel set definitions
 *                 Each set is terminated by -1, entire array ends with -2
 *                 Example: {0, -1, 1, -1, -2} = two sets: {ch0}, {ch1}
 *                 Example: {0, 1, -1, 2, 3, -1, -2} = two sets: {ch0,ch1}, {ch2,ch3}
 *   randomize: If non-zero, randomize channel set order after each cycle
 *
 * Returns: New buffer with flutter effect, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_flutter_multi(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double frequency,
                                       double depth,
                                       double gain,
                                       const int* channel_sets,
                                       int randomize);

#ifdef __cplusplus
}
#endif

#endif /* CDP_FLUTTER_H */
