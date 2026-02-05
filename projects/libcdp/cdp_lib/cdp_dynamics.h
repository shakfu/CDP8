/*
 * CDP Dynamics Functions - Gate, Compressor, Limiter, Envelope
 *
 * Dynamic range processing and envelope manipulation.
 */

#ifndef CDP_DYNAMICS_H
#define CDP_DYNAMICS_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Noise gate - silences audio below threshold.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   threshold_db: Gate threshold in dB
 *   attack_ms: Attack time in milliseconds
 *   release_ms: Release time in milliseconds
 *   hold_ms: Hold time in milliseconds
 *
 * Returns: Gated audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_gate(cdp_lib_ctx* ctx,
                              const cdp_lib_buffer* input,
                              double threshold_db,
                              double attack_ms,
                              double release_ms,
                              double hold_ms);

/*
 * Compressor - reduces dynamic range above threshold.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   threshold_db: Compression threshold in dB
 *   ratio: Compression ratio (e.g., 4 = 4:1)
 *   attack_ms: Attack time in milliseconds
 *   release_ms: Release time in milliseconds
 *   makeup_gain_db: Makeup gain in dB
 *
 * Returns: Compressed audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_compressor(cdp_lib_ctx* ctx,
                                    const cdp_lib_buffer* input,
                                    double threshold_db,
                                    double ratio,
                                    double attack_ms,
                                    double release_ms,
                                    double makeup_gain_db);

/*
 * Limiter - prevents audio from exceeding threshold.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   threshold_db: Limiting threshold in dB
 *   attack_ms: Attack time in milliseconds (0 for hard limiting)
 *   release_ms: Release time in milliseconds
 *
 * Returns: Limited audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_limiter(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double threshold_db,
                                 double attack_ms,
                                 double release_ms);

/*
 * Envelope follower - extracts amplitude envelope from audio.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   attack_ms: Attack time in milliseconds
 *   release_ms: Release time in milliseconds
 *   mode: 0 = peak, 1 = RMS
 *
 * Returns: Envelope as mono audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_envelope_follow(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double attack_ms,
                                         double release_ms,
                                         int mode);

/*
 * Envelope apply - modulates audio amplitude with an envelope.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   envelope: Envelope buffer (from envelope_follow or generated)
 *   depth: Modulation depth (0 to 1)
 *
 * Returns: Modulated audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_envelope_apply(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        const cdp_lib_buffer* envelope,
                                        double depth);

#ifdef __cplusplus
}
#endif

#endif /* CDP_DYNAMICS_H */
