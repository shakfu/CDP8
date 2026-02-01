/*
 * CDP Analysis Functions - Low-level API (CDP-style)
 *
 * These functions work on pre-analyzed spectral data from cdp_spectral_analyze().
 * They match CDP's original architecture where analysis is performed on spectral files.
 *
 * For high-level functions that work directly on audio buffers, see cdp_lib.h:
 * - cdp_lib_pitch()
 * - cdp_lib_formants()
 * - cdp_lib_get_partials()
 */

#ifndef CDP_ANALYSIS_H
#define CDP_ANALYSIS_H

#include "cdp_lib.h"
#include "cdp_spectral.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * LOW-LEVEL API: Work on spectral data (CDP-style)
 *
 * These functions match CDP's architecture where analysis is performed
 * on pre-analyzed spectral data from cdp_spectral_analyze().
 * ========================================================================= */

/*
 * Extract pitch from spectral data using harmonic matching (CDP style).
 *
 * Finds spectral peaks and matches them to harmonic series to determine
 * fundamental frequency. This is the algorithm used in CDP's pitch tools.
 *
 * Args:
 *   ctx: Library context
 *   spectral: Pre-analyzed spectral data from cdp_spectral_analyze()
 *   min_freq: Minimum expected frequency in Hz (default 50)
 *   max_freq: Maximum expected frequency in Hz (default 2000)
 *   num_peaks: Number of peaks to consider for matching (default 8)
 *
 * Returns: Pitch data, or NULL on error.
 */
cdp_pitch_data* cdp_lib_pitch_from_spectrum(cdp_lib_ctx* ctx,
                                             const cdp_spectral_data* spectral,
                                             double min_freq,
                                             double max_freq,
                                             int num_peaks);

/*
 * Extract formant envelope from spectral data (CDP style).
 *
 * Extracts spectral envelope by finding peaks in each spectral frame.
 * This matches CDP's formant extraction which works on spectral files.
 *
 * Args:
 *   ctx: Library context
 *   spectral: Pre-analyzed spectral data from cdp_spectral_analyze()
 *   bands_per_octave: Number of formant bands per octave (default 6, max 12)
 *
 * Returns: Formant data, or NULL on error.
 */
cdp_formant_data* cdp_lib_formants_from_spectrum(cdp_lib_ctx* ctx,
                                                  const cdp_spectral_data* spectral,
                                                  int bands_per_octave);

/*
 * Extract harmonic partials from spectral data given fundamental (CDP style).
 *
 * Given a pitch contour, extracts the amplitudes of harmonics at each frame.
 * This matches CDP's get_partials which requires known pitch.
 *
 * Args:
 *   ctx: Library context
 *   spectral: Pre-analyzed spectral data from cdp_spectral_analyze()
 *   pitch: Pitch contour (fundamental frequency per frame)
 *   max_harmonics: Maximum number of harmonics to extract (default 32)
 *   amp_threshold: Minimum amplitude threshold in dB (default -60)
 *
 * Returns: Partial data (one track per harmonic), or NULL on error.
 */
cdp_partial_data* cdp_lib_partials_from_spectrum(cdp_lib_ctx* ctx,
                                                  const cdp_spectral_data* spectral,
                                                  const cdp_pitch_data* pitch,
                                                  int max_harmonics,
                                                  double amp_threshold);

#ifdef __cplusplus
}
#endif

#endif /* CDP_ANALYSIS_H */
