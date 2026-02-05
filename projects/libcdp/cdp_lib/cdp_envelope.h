/*
 * CDP Envelope Processing - Direct buffer implementations
 *
 * These functions implement envelope operations working directly
 * on memory buffers without file I/O.
 *
 * Note: Implementations are in cdp_lib.c to access the context structure.
 */

#ifndef CDP_ENVELOPE_H
#define CDP_ENVELOPE_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fade curve types */
#define CDP_FADE_LINEAR      0
#define CDP_FADE_EXPONENTIAL 1

/*
 * Apply dovetail fades (fade-in and fade-out).
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   fade_in_dur: Fade-in duration in seconds
 *   fade_out_dur: Fade-out duration in seconds
 *   fade_in_type: CDP_FADE_LINEAR or CDP_FADE_EXPONENTIAL
 *   fade_out_type: CDP_FADE_LINEAR or CDP_FADE_EXPONENTIAL
 *
 * Returns: New buffer with fades applied, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_dovetail(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double fade_in_dur,
                                  double fade_out_dur,
                                  int fade_in_type,
                                  int fade_out_type);

/*
 * Apply tremolo (amplitude modulation).
 *
 * Modulates the amplitude with a sine wave LFO.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   freq: Tremolo frequency in Hz (0 to 500)
 *   depth: Modulation depth (0.0 = none, 1.0 = full)
 *   gain: Output gain multiplier (default 1.0)
 *
 * Returns: New buffer with tremolo applied, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_tremolo(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double freq,
                                 double depth,
                                 double gain);

/*
 * Modify the attack portion of a sound.
 *
 * Applies a gain envelope to the attack region.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   attack_gain: Gain multiplier for attack (e.g., 2.0 = double)
 *   attack_time: Duration of attack region in seconds
 *
 * Returns: New buffer with modified attack, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_attack(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                double attack_gain,
                                double attack_time);

#ifdef __cplusplus
}
#endif

#endif /* CDP_ENVELOPE_H */
