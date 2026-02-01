/*
 * CDP Filter Functions - Lowpass, Highpass, Bandpass, Notch, Parametric EQ
 *
 * Spectral-domain filtering operations.
 */

#ifndef CDP_FILTERS_H
#define CDP_FILTERS_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Lowpass filter - passes frequencies below cutoff.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   cutoff_freq: Cutoff frequency in Hz
 *   attenuation_db: Attenuation in dB for rejected frequencies
 *   fft_size: FFT size (0 for default 1024)
 *
 * Returns: Filtered audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_filter_lowpass(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        double cutoff_freq,
                                        double attenuation_db,
                                        int fft_size);

/*
 * Highpass filter - passes frequencies above cutoff.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   cutoff_freq: Cutoff frequency in Hz
 *   attenuation_db: Attenuation in dB for rejected frequencies
 *   fft_size: FFT size (0 for default 1024)
 *
 * Returns: Filtered audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_filter_highpass(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double cutoff_freq,
                                         double attenuation_db,
                                         int fft_size);

/*
 * Bandpass filter - passes frequencies between low and high cutoffs.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   low_freq: Lower cutoff frequency in Hz
 *   high_freq: Upper cutoff frequency in Hz
 *   attenuation_db: Attenuation in dB for rejected frequencies
 *   fft_size: FFT size (0 for default 1024)
 *
 * Returns: Filtered audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_filter_bandpass(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double low_freq,
                                         double high_freq,
                                         double attenuation_db,
                                         int fft_size);

/*
 * Notch filter - attenuates frequencies around a center frequency.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   center_freq: Center frequency of the notch in Hz
 *   width_hz: Width of the notch in Hz
 *   attenuation_db: Attenuation in dB for notched frequencies
 *   fft_size: FFT size (0 for default 1024)
 *
 * Returns: Filtered audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_filter_notch(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double center_freq,
                                      double width_hz,
                                      double attenuation_db,
                                      int fft_size);

/*
 * Parametric EQ - bell curve boost/cut at a center frequency.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   center_freq: Center frequency in Hz
 *   gain_db: Gain in dB (positive for boost, negative for cut)
 *   q: Q factor (bandwidth = center_freq / Q)
 *   fft_size: FFT size (0 for default 1024)
 *
 * Returns: EQ'd audio buffer, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_eq_parametric(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double center_freq,
                                       double gain_db,
                                       double q,
                                       int fft_size);

#ifdef __cplusplus
}
#endif

#endif /* CDP_FILTERS_H */
