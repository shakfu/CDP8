/*
 * CDP Morph Native Wrapper - Wraps original dev/morph algorithms
 *
 * These functions wrap the original CDP morph algorithms (specglide, specbridge,
 * specmorph) from dev/morph/morph.c using the multi-input shim layer.
 *
 * This provides access to the exact algorithms Trevor Wishart designed,
 * as opposed to the reimplemented versions in cdp_morph.c.
 */

#ifndef CDP_MORPH_NATIVE_H
#define CDP_MORPH_NATIVE_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Morph Glide Native - Uses original CDP specglide algorithm.
 *
 * Creates a smooth glide between two spectral frames - one from each input.
 * The original algorithm reads single windows from each file and interpolates
 * amplitudes linearly while frequencies follow an exponential curve.
 *
 * Args:
 *   ctx: Library context
 *   input1: First input audio buffer (single spectral window used)
 *   input2: Second input audio buffer (single spectral window used)
 *   duration: Output duration in seconds (number of output windows)
 *   fft_size: FFT size (power of 2, typically 1024-4096)
 *
 * Returns: Glided audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_morph_glide_native(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input1,
                                        const cdp_lib_buffer* input2,
                                        double duration,
                                        int fft_size);

/*
 * Morph Bridge Native - Uses original CDP specbridge algorithm.
 *
 * Creates a bridge (crossfade) between two spectral files. Supports multiple
 * normalization modes and interpolation timing control.
 *
 * Args:
 *   ctx: Library context
 *   input1: First input audio buffer
 *   input2: Second input audio buffer
 *   mode: Bridge mode (0-5):
 *         0 = BRG_NO_NORMALISE - no amplitude normalization
 *         1 = BRG_NORM_TO_MIN - normalize to minimum level
 *         2 = BRG_NORM_TO_FILE1 - normalize to file 1 level
 *         3 = BRG_NORM_TO_FILE2 - normalize to file 2 level
 *         4 = BRG_NORM_FROM_1_TO_2 - progressive normalization 1->2
 *         5 = BRG_NORM_FROM_2_TO_1 - progressive normalization 2->1
 *   offset: Time offset (seconds) to start file2 relative to file1
 *   interp_start: Normalized position (0-1) where interpolation starts
 *   interp_end: Normalized position (0-1) where interpolation ends
 *   fft_size: FFT size (power of 2)
 *
 * Returns: Bridged audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_morph_bridge_native(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input1,
                                         const cdp_lib_buffer* input2,
                                         int mode,
                                         double offset,
                                         double interp_start,
                                         double interp_end,
                                         int fft_size);

/*
 * Morph Native - Uses original CDP specmorph algorithm.
 *
 * Full spectral morphing with separate control over amplitude and frequency
 * interpolation timing. Supports multiple interpolation curves.
 *
 * Args:
 *   ctx: Library context
 *   input1: First input audio buffer
 *   input2: Second input audio buffer
 *   mode: Interpolation mode:
 *         0 = MPH_LINE - linear interpolation
 *         1 = MPH_COSIN - cosine interpolation (smoother)
 *   amp_start: Normalized time (0-1) where amplitude morph starts
 *   amp_end: Normalized time (0-1) where amplitude morph ends
 *   freq_start: Normalized time (0-1) where frequency morph starts
 *   freq_end: Normalized time (0-1) where frequency morph ends
 *   amp_exp: Exponent for amplitude curve (1=linear, >1=slow start, <1=fast start)
 *   freq_exp: Exponent for frequency curve
 *   stagger: Time offset (seconds) between file2 and output
 *   fft_size: FFT size (power of 2)
 *
 * Returns: Morphed audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_morph_morph_native(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input1,
                                        const cdp_lib_buffer* input2,
                                        int mode,
                                        double amp_start,
                                        double amp_end,
                                        double freq_start,
                                        double freq_end,
                                        double amp_exp,
                                        double freq_exp,
                                        double stagger,
                                        int fft_size);

#ifdef __cplusplus
}
#endif

#endif /* CDP_MORPH_NATIVE_H */
