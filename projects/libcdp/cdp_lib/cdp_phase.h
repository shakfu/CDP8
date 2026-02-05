/*
 * CDP Phase - Phase Manipulation Effects
 *
 * Provides phase inversion and stereo enhancement via phase shifting.
 */

#ifndef CDP_PHASE_H
#define CDP_PHASE_H

#include "cdp_lib.h"

/*
 * Apply phase inversion - inverts the phase of all samples.
 *
 * Multiplies all samples by -1, effectively inverting the waveform.
 * Works with both mono and stereo audio.
 *
 * Parameters:
 *   ctx   - Processing context
 *   input - Input buffer (mono or stereo)
 *
 * Returns:
 *   New buffer with inverted phase, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_phase_invert(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input);

/*
 * Enhance stereo separation using phase subtraction.
 *
 * Enhances the stereo image by subtracting a portion of each channel
 * from the other, emphasizing differences between L and R.
 *
 * For each sample pair:
 *   newLeft = L - (transfer * R)
 *   newRight = R - (transfer * L)
 *
 * The output is automatically normalized to preserve the original
 * maximum level.
 *
 * Parameters:
 *   ctx      - Processing context
 *   input    - Input buffer (must be stereo)
 *   transfer - Amount of signal used in phase-cancellation (0.0 to 1.0).
 *              0 = no change, 1 = full cancellation.
 *              Default is 1.0 for maximum stereo enhancement.
 *
 * Returns:
 *   New stereo buffer with enhanced separation, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_phase_stereo(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double transfer);

#endif /* CDP_PHASE_H */
