/*
 * CDP Wrapper - High-level interface to CDP processing functions.
 *
 * This provides a clean C API for calling CDP algorithms on memory buffers,
 * bypassing the command-line interface and file I/O.
 */

#ifndef CDP_WRAPPER_H
#define CDP_WRAPPER_H

#include "cdp.h"  /* For cdp_buffer */

#ifdef __cplusplus
extern "C" {
#endif

/*
 * CDP wrapper context - holds state for CDP processing.
 */
typedef struct cdp_wrapper_ctx cdp_wrapper_ctx;

/*
 * Initialize CDP wrapper.
 * Must be called before any CDP processing.
 *
 * Returns: New context, or NULL on failure.
 */
cdp_wrapper_ctx* cdp_wrapper_init(void);

/*
 * Clean up CDP wrapper.
 */
void cdp_wrapper_cleanup(cdp_wrapper_ctx* ctx);

/*
 * Time-stretch audio using CDP's phase vocoder.
 *
 * Args:
 *   ctx: Wrapper context
 *   input: Input buffer (mono audio)
 *   factor: Stretch factor (2.0 = twice as long)
 *   fft_size: FFT analysis window size (power of 2, 256-32768)
 *   overlap: Overlap factor (1-4)
 *
 * Returns: New buffer with stretched audio, or NULL on error.
 *          Caller must free with cdp_buffer_free().
 */
cdp_buffer* cdp_wrapper_time_stretch(cdp_wrapper_ctx* ctx,
                                     const cdp_buffer* input,
                                     double factor,
                                     int fft_size,
                                     int overlap);

/*
 * Apply spectral blur using CDP's blur algorithm.
 *
 * Args:
 *   ctx: Wrapper context
 *   input: Input buffer (mono audio)
 *   blur_windows: Number of analysis windows to blur over
 *   fft_size: FFT analysis window size
 *
 * Returns: New buffer with blurred audio, or NULL on error.
 */
cdp_buffer* cdp_wrapper_spectral_blur(cdp_wrapper_ctx* ctx,
                                      const cdp_buffer* input,
                                      int blur_windows,
                                      int fft_size);

/*
 * Apply loudness change using CDP's modify loudness.
 *
 * Args:
 *   ctx: Wrapper context
 *   input: Input buffer
 *   gain_db: Gain change in dB
 *
 * Returns: New buffer with modified loudness, or NULL on error.
 */
cdp_buffer* cdp_wrapper_loudness(cdp_wrapper_ctx* ctx,
                                 const cdp_buffer* input,
                                 double gain_db);

/*
 * Get last error message.
 */
const char* cdp_wrapper_get_error(cdp_wrapper_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* CDP_WRAPPER_H */
