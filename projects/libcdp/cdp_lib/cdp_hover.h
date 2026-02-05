/*
 * CDP Hover - Zigzag Reading / Pitch Hovering Effect
 *
 * Implements the CDP hover algorithm that reads through an audio file
 * with zigzag motion at a specified frequency.
 */

#ifndef CDP_HOVER_H
#define CDP_HOVER_H

#include "cdp_lib.h"

/*
 * Apply hover effect - zigzag reading at specified frequency.
 *
 * Parameters:
 *   ctx        - Processing context
 *   input      - Input buffer (mono only)
 *   frequency  - Rate of zigzag oscillation in Hz (determines read width)
 *                At 44100 Hz sample rate and 1 Hz frequency,
 *                reads 22050 samples forward then 22050 back
 *   location   - Position in source file (0.0 to 1.0 normalized)
 *   frq_rand   - Random variation of frequency (0.0 to 1.0)
 *   loc_rand   - Random variation of location (0.0 to 1.0)
 *   splice_ms  - Splice length at zig/zag boundaries in milliseconds
 *   duration   - Output duration in seconds (0 = same as input)
 *
 * Returns:
 *   New buffer with hover effect applied, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_hover(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input,
                               double frequency,
                               double location,
                               double frq_rand,
                               double loc_rand,
                               double splice_ms,
                               double duration);

#endif /* CDP_HOVER_H */
