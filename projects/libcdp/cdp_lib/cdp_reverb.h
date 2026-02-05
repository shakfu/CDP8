/*
 * CDP Reverb Processing - Direct buffer implementations
 *
 * Implements a Feedback Delay Network (FDN) reverb working directly
 * on memory buffers without file I/O.
 *
 * Note: Implementations are in cdp_lib.c to access the context structure.
 */

#ifndef CDP_REVERB_H
#define CDP_REVERB_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Apply reverb using a Feedback Delay Network.
 *
 * Uses 8 comb filters and 4 allpass filters for dense, natural reverb.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   mix: Dry/wet balance (0.0 = dry, 1.0 = wet). Default 0.5.
 *   decay_time: Reverb decay time in seconds (RT60). Default 2.0.
 *   damping: High frequency damping (0.0 to 1.0). Default 0.5.
 *            Higher values = darker reverb tail.
 *   lpfreq: Lowpass filter cutoff in Hz. Default 8000.
 *   predelay: Pre-delay time in milliseconds. Default 0.
 *
 * Returns: New buffer with reverb applied, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_reverb(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                double mix,
                                double decay_time,
                                double damping,
                                double lpfreq,
                                double predelay);

#ifdef __cplusplus
}
#endif

#endif /* CDP_REVERB_H */
