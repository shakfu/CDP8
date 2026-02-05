/*
 * CDP FOF Extraction and Synthesis (FOFEX)
 *
 * FOFs (Formant Wave Functions) are pitch-synchronous grains extracted
 * from voiced sounds. These can be used to construct new sounds with
 * different pitches while preserving the formant characteristics.
 *
 * Based on CDP's fofex algorithms.
 */

#ifndef CDP_FOFEX_H
#define CDP_FOFEX_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Extract a single FOF (pitch-synchronous grain) from a position.
 *
 * Finds the pitch period at the specified time and extracts
 * one or more complete pitch cycles as a windowed grain.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (mono only)
 *   time: Time in seconds at which to extract FOF
 *   fof_count: Number of pitch periods to include (1-8)
 *   window: If non-zero, apply raised cosine window to edges
 *
 * Returns: New buffer containing the extracted FOF, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_fofex_extract(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double time,
                                       int fof_count,
                                       int window);

/*
 * Extract all FOFs from an audio file.
 *
 * Analyzes the entire file and extracts all pitch-synchronous FOFs,
 * returning them concatenated in a single buffer. Each FOF is
 * zero-padded to uniform length.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (mono only)
 *   fof_count: Number of pitch periods per FOF group (1-4)
 *   min_level_db: Minimum level in dB below peak (0 = keep all, -40 = reject quiet)
 *   window: If non-zero, apply raised cosine window to edges
 *   fof_info: Output - receives number of FOFs extracted and unit length
 *             fof_info[0] = number of FOFs, fof_info[1] = samples per FOF
 *
 * Returns: New buffer containing all FOFs (uniform length, concatenated),
 *          or NULL on error.
 */
cdp_lib_buffer* cdp_lib_fofex_extract_all(cdp_lib_ctx* ctx,
                                           const cdp_lib_buffer* input,
                                           int fof_count,
                                           double min_level_db,
                                           int window,
                                           int* fof_info);

/*
 * Synthesize audio using extracted FOFs.
 *
 * Creates new audio by repeating FOFs at specified pitch, applying
 * an amplitude envelope.
 *
 * Args:
 *   ctx: Library context
 *   fof: FOF buffer (single FOF or bank of FOFs)
 *   duration: Output duration in seconds
 *   frequency: Target frequency in Hz (pitch of output)
 *   amplitude: Output amplitude (0.0 to 1.0)
 *   fof_index: Which FOF to use if fof is a bank (0-based, -1 = average all)
 *   fof_unit_len: If > 0, fof is a bank with this many samples per FOF
 *
 * Returns: New buffer with synthesized audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_fofex_synth(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* fof,
                                     double duration,
                                     double frequency,
                                     double amplitude,
                                     int fof_index,
                                     int fof_unit_len);

/*
 * Resynthesize audio with modified pitch using FOFs.
 *
 * Extracts FOFs from input and resynthesizes at a different pitch
 * while preserving formant characteristics.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (mono only)
 *   pitch_shift: Pitch shift in semitones (-24 to +24)
 *   preserve_formants: If non-zero, formants are preserved
 *                      If zero, formants shift with pitch
 *
 * Returns: New buffer with pitch-shifted audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_fofex_repitch(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double pitch_shift,
                                       int preserve_formants);

#ifdef __cplusplus
}
#endif

#endif /* CDP_FOFEX_H */
