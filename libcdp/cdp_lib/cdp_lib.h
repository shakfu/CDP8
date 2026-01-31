/*
 * CDP Library Interface
 *
 * This provides a clean C API for calling CDP processing functions
 * on memory buffers, bypassing file I/O.
 */

#ifndef CDP_LIB_H
#define CDP_LIB_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
#define CDP_LIB_OK              0
#define CDP_LIB_ERR_MEMORY     -1
#define CDP_LIB_ERR_PARAM      -2
#define CDP_LIB_ERR_PROCESS    -3
#define CDP_LIB_ERR_INIT       -4

/*
 * Audio buffer structure for CDP library
 */
typedef struct cdp_lib_buffer {
    float *data;           /* Sample data (interleaved if stereo) */
    size_t length;         /* Number of samples */
    int channels;          /* Number of channels */
    int sample_rate;       /* Sample rate in Hz */
} cdp_lib_buffer;

/*
 * CDP library context
 */
typedef struct cdp_lib_ctx cdp_lib_ctx;

/*
 * Initialize CDP library.
 * Must be called before any processing functions.
 *
 * Returns: New context, or NULL on failure.
 */
cdp_lib_ctx* cdp_lib_init(void);

/*
 * Cleanup CDP library and free all resources.
 */
void cdp_lib_cleanup(cdp_lib_ctx* ctx);

/*
 * Get last error message.
 */
const char* cdp_lib_get_error(cdp_lib_ctx* ctx);

/*
 * Create a new buffer.
 */
cdp_lib_buffer* cdp_lib_buffer_create(size_t length, int channels, int sample_rate);

/*
 * Create a buffer from existing data (takes ownership).
 */
cdp_lib_buffer* cdp_lib_buffer_from_data(float *data, size_t length,
                                          int channels, int sample_rate);

/*
 * Free a buffer.
 */
void cdp_lib_buffer_free(cdp_lib_buffer* buf);

/* =========================================================================
 * Processing Functions
 * ========================================================================= */

/*
 * Time-stretch audio without changing pitch.
 *
 * Uses CDP's phase vocoder for high-quality stretching.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (will be converted to mono if stereo)
 *   factor: Stretch factor (2.0 = twice as long, 0.5 = half as long)
 *   fft_size: FFT window size (power of 2, 256-8192). Default 1024.
 *   overlap: Overlap factor (1-4). Default 3.
 *
 * Returns: New buffer with stretched audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_time_stretch(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double factor,
                                      int fft_size,
                                      int overlap);

/*
 * Spectral blur - average spectrum over time.
 *
 * Creates a smeared, washed-out effect.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   blur_time: Time to blur over in seconds
 *   fft_size: FFT window size
 *
 * Returns: New buffer with blurred audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_spectral_blur(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double blur_time,
                                       int fft_size);

/*
 * Modify loudness (gain in dB).
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   gain_db: Gain change in dB (positive = louder)
 *
 * Returns: New buffer with modified loudness, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_loudness(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double gain_db);

/*
 * Change playback speed (affects both pitch and duration).
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   speed: Speed factor (2.0 = double speed, 0.5 = half speed)
 *
 * Returns: New buffer with modified speed, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_speed(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input,
                               double speed);

/*
 * Pitch shift without changing duration.
 *
 * Combines speed change with time-stretch compensation.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   semitones: Pitch shift in semitones (positive = up, negative = down)
 *   fft_size: FFT window size for time stretch. Default 1024.
 *
 * Returns: New buffer with pitch-shifted audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_pitch_shift(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input,
                                     double semitones,
                                     int fft_size);

/*
 * Shift all frequencies by a fixed Hz offset.
 *
 * Unlike pitch shift, this adds a constant Hz value to all frequencies,
 * creating inharmonic effects.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   shift_hz: Frequency offset in Hz (positive = up, negative = down)
 *   fft_size: FFT window size. Default 1024.
 *
 * Returns: New buffer with shifted frequencies, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_spectral_shift(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        double shift_hz,
                                        int fft_size);

/*
 * Stretch frequencies differentially (higher frequencies stretched more).
 *
 * Creates inharmonic effects by applying progressively more transposition
 * to higher frequencies.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   max_stretch: Maximum transposition ratio for highest frequencies
 *   freq_divide: Frequency below which no stretching occurs
 *   exponent: Stretching curve (1.0 = linear, >1 = more effect on highs)
 *   fft_size: FFT window size. Default 1024.
 *
 * Returns: New buffer with stretched spectrum, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_spectral_stretch(cdp_lib_ctx* ctx,
                                          const cdp_lib_buffer* input,
                                          double max_stretch,
                                          double freq_divide,
                                          double exponent,
                                          int fft_size);

/*
 * Apply lowpass filter.
 *
 * Attenuates frequencies above the cutoff.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   cutoff_freq: Cutoff frequency in Hz
 *   attenuation_db: Attenuation in dB (negative, e.g. -60)
 *   fft_size: FFT window size. Default 1024.
 *
 * Returns: New buffer with filtered audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_filter_lowpass(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        double cutoff_freq,
                                        double attenuation_db,
                                        int fft_size);

/*
 * Apply highpass filter.
 *
 * Attenuates frequencies below the cutoff.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   cutoff_freq: Cutoff frequency in Hz
 *   attenuation_db: Attenuation in dB (negative, e.g. -60)
 *   fft_size: FFT window size. Default 1024.
 *
 * Returns: New buffer with filtered audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_filter_highpass(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double cutoff_freq,
                                         double attenuation_db,
                                         int fft_size);

/*
 * Apply bandpass filter.
 *
 * Passes frequencies between low and high cutoffs.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   low_freq: Low cutoff frequency in Hz
 *   high_freq: High cutoff frequency in Hz
 *   attenuation_db: Attenuation in dB (negative, e.g. -60)
 *   fft_size: FFT window size. Default 1024.
 *
 * Returns: New buffer with filtered audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_filter_bandpass(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double low_freq,
                                         double high_freq,
                                         double attenuation_db,
                                         int fft_size);

/*
 * Apply notch (band-reject) filter.
 *
 * Attenuates a narrow frequency band.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   center_freq: Center frequency to notch out in Hz
 *   width_hz: Width of the notch in Hz
 *   attenuation_db: Attenuation in dB (negative, e.g. -60)
 *   fft_size: FFT window size. Default 1024.
 *
 * Returns: New buffer with filtered audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_filter_notch(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double center_freq,
                                      double width_hz,
                                      double attenuation_db,
                                      int fft_size);

/*
 * Noise gate - silence audio below threshold.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   threshold_db: Threshold in dB (e.g. -40)
 *   attack_ms: Attack time in milliseconds
 *   release_ms: Release time in milliseconds
 *   hold_ms: Hold time before release in milliseconds
 *
 * Returns: New buffer with gated audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_gate(cdp_lib_ctx* ctx,
                              const cdp_lib_buffer* input,
                              double threshold_db,
                              double attack_ms,
                              double release_ms,
                              double hold_ms);

/*
 * Bitcrush - reduce bit depth and/or sample rate.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   bit_depth: Target bit depth (1-16, 16 = no reduction)
 *   downsample: Downsample factor (1 = none, 2 = half rate, etc.)
 *
 * Returns: New buffer with crushed audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_bitcrush(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  int bit_depth,
                                  int downsample);

/*
 * Ring modulation - multiply signal by a carrier frequency.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   freq: Carrier frequency in Hz
 *   mix: Dry/wet mix (0.0 = dry, 1.0 = wet)
 *
 * Returns: New buffer with ring-modulated audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_ring_mod(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double freq,
                                  double mix);

/*
 * Delay effect with feedback.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   delay_ms: Delay time in milliseconds
 *   feedback: Feedback amount (0.0 to <1.0)
 *   mix: Dry/wet mix (0.0 = dry, 1.0 = wet)
 *
 * Returns: New buffer with delayed audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_delay(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input,
                               double delay_ms,
                               double feedback,
                               double mix);

/*
 * Chorus effect - modulated delay for thickening.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   rate: LFO rate in Hz (typically 0.5-5)
 *   depth_ms: Modulation depth in milliseconds (typically 1-20)
 *   mix: Dry/wet mix (0.0 = dry, 1.0 = wet)
 *
 * Returns: New buffer with chorus effect, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_chorus(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                double rate,
                                double depth_ms,
                                double mix);

/*
 * Flanger effect - short modulated delay with feedback.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   rate: LFO rate in Hz (typically 0.1-2)
 *   depth_ms: Modulation depth in milliseconds (typically 1-10)
 *   feedback: Feedback amount (-0.95 to 0.95)
 *   mix: Dry/wet mix (0.0 = dry, 1.0 = wet)
 *
 * Returns: New buffer with flanger effect, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_flanger(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double rate,
                                 double depth_ms,
                                 double feedback,
                                 double mix);

/*
 * Parametric EQ - boost or cut at a frequency with adjustable Q.
 *
 * Applies a bell-shaped gain curve centered at the specified frequency.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   center_freq: Center frequency in Hz
 *   gain_db: Gain in dB (positive = boost, negative = cut)
 *   q: Q factor (0.1 to 10, higher = narrower bandwidth)
 *   fft_size: FFT window size. Default 1024.
 *
 * Returns: New buffer with EQ applied, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_eq_parametric(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double center_freq,
                                       double gain_db,
                                       double q,
                                       int fft_size);

/*
 * Extract amplitude envelope from audio.
 *
 * Uses peak or RMS detection with attack/release smoothing.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   attack_ms: Attack time in milliseconds (how fast envelope rises)
 *   release_ms: Release time in milliseconds (how fast envelope falls)
 *   mode: 0 = peak detection, 1 = RMS detection
 *
 * Returns: New buffer containing envelope values, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_envelope_follow(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double attack_ms,
                                         double release_ms,
                                         int mode);

/*
 * Apply an envelope to audio (amplitude modulation).
 *
 * Multiplies the input audio by the envelope values.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   envelope: Envelope buffer (from envelope_follow or generated)
 *   depth: Modulation depth (0.0 = no effect, 1.0 = full modulation)
 *
 * Returns: New buffer with envelope applied, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_envelope_apply(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        const cdp_lib_buffer* envelope,
                                        double depth);

/*
 * Dynamic range compressor.
 *
 * Reduces the volume of audio above the threshold.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   threshold_db: Level above which compression starts (e.g., -20)
 *   ratio: Compression ratio (e.g., 4.0 means 4:1)
 *   attack_ms: Attack time in milliseconds
 *   release_ms: Release time in milliseconds
 *   makeup_gain_db: Makeup gain in dB to compensate for level reduction
 *
 * Returns: New buffer with compression applied, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_compressor(cdp_lib_ctx* ctx,
                                    const cdp_lib_buffer* input,
                                    double threshold_db,
                                    double ratio,
                                    double attack_ms,
                                    double release_ms,
                                    double makeup_gain_db);

/*
 * Limiter - prevent audio from exceeding a threshold.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   threshold_db: Maximum level in dB (e.g., -0.1)
 *   attack_ms: Attack time in milliseconds (0 for hard limiting)
 *   release_ms: Release time in milliseconds
 *
 * Returns: New buffer with limiting applied, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_limiter(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double threshold_db,
                                 double attack_ms,
                                 double release_ms);

#ifdef __cplusplus
}
#endif

#endif /* CDP_LIB_H */
