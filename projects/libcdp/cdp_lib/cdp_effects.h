/*
 * CDP Effects Functions - Bitcrush, Ring Mod, Delay, Chorus, Flanger
 *
 * Time-domain audio effects.
 */

#ifndef CDP_EFFECTS_H
#define CDP_EFFECTS_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Bitcrush - reduces bit depth and sample rate for lo-fi effect.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   bit_depth: Target bit depth (1-16)
 *   downsample: Downsampling factor (1 = no downsampling)
 *
 * Returns: Processed audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_bitcrush(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  int bit_depth,
                                  int downsample);

/*
 * Ring modulation - multiplies audio with a sine wave carrier.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   freq: Carrier frequency in Hz
 *   mix: Dry/wet mix (0=dry, 1=wet)
 *
 * Returns: Processed audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_ring_mod(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double freq,
                                  double mix);

/*
 * Delay - simple delay with feedback.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   delay_ms: Delay time in milliseconds
 *   feedback: Feedback amount (0 to <1)
 *   mix: Dry/wet mix (0=dry, 1=wet)
 *
 * Returns: Processed audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_delay(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input,
                               double delay_ms,
                               double feedback,
                               double mix);

/*
 * Chorus - creates a shimmering chorus effect using modulated delay.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   rate: LFO rate in Hz
 *   depth_ms: Modulation depth in milliseconds
 *   mix: Dry/wet mix (0=dry, 1=wet)
 *
 * Returns: Processed audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_chorus(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                double rate,
                                double depth_ms,
                                double mix);

/*
 * Flanger - creates a sweeping comb filter effect.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   rate: LFO rate in Hz
 *   depth_ms: Modulation depth in milliseconds
 *   feedback: Feedback amount (-0.95 to 0.95)
 *   mix: Dry/wet mix (0=dry, 1=wet)
 *
 * Returns: Processed audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_flanger(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double rate,
                                 double depth_ms,
                                 double feedback,
                                 double mix);

#ifdef __cplusplus
}
#endif

#endif /* CDP_EFFECTS_H */
