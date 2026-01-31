# Cython declarations for CDP library

from libc.stddef cimport size_t

cdef extern from "cdp_lib.h":
    # Opaque context type
    ctypedef struct cdp_lib_ctx:
        pass

    # Buffer structure
    ctypedef struct cdp_lib_buffer:
        float *data
        size_t length
        int channels
        int sample_rate

    # Initialization
    cdp_lib_ctx* cdp_lib_init()
    void cdp_lib_cleanup(cdp_lib_ctx* ctx)
    const char* cdp_lib_get_error(cdp_lib_ctx* ctx)

    # Buffer management
    cdp_lib_buffer* cdp_lib_buffer_create(size_t length, int channels, int sample_rate)
    cdp_lib_buffer* cdp_lib_buffer_from_data(float *data, size_t length,
                                              int channels, int sample_rate)
    void cdp_lib_buffer_free(cdp_lib_buffer* buf)

    # Core processing functions
    cdp_lib_buffer* cdp_lib_time_stretch(cdp_lib_ctx* ctx,
                                          const cdp_lib_buffer* input,
                                          double factor,
                                          int fft_size,
                                          int overlap)

    cdp_lib_buffer* cdp_lib_spectral_blur(cdp_lib_ctx* ctx,
                                           const cdp_lib_buffer* input,
                                           double blur_time,
                                           int fft_size)

    cdp_lib_buffer* cdp_lib_loudness(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double gain_db)

    cdp_lib_buffer* cdp_lib_speed(cdp_lib_ctx* ctx,
                                   const cdp_lib_buffer* input,
                                   double speed)

    # Spectral operations
    cdp_lib_buffer* cdp_lib_pitch_shift(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double semitones,
                                         int fft_size)

    cdp_lib_buffer* cdp_lib_spectral_shift(cdp_lib_ctx* ctx,
                                            const cdp_lib_buffer* input,
                                            double shift_hz,
                                            int fft_size)

    cdp_lib_buffer* cdp_lib_spectral_stretch(cdp_lib_ctx* ctx,
                                              const cdp_lib_buffer* input,
                                              double max_stretch,
                                              double freq_divide,
                                              double exponent,
                                              int fft_size)

    cdp_lib_buffer* cdp_lib_filter_lowpass(cdp_lib_ctx* ctx,
                                            const cdp_lib_buffer* input,
                                            double cutoff_freq,
                                            double attenuation_db,
                                            int fft_size)

    cdp_lib_buffer* cdp_lib_filter_highpass(cdp_lib_ctx* ctx,
                                             const cdp_lib_buffer* input,
                                             double cutoff_freq,
                                             double attenuation_db,
                                             int fft_size)

    cdp_lib_buffer* cdp_lib_filter_bandpass(cdp_lib_ctx* ctx,
                                             const cdp_lib_buffer* input,
                                             double low_freq,
                                             double high_freq,
                                             double attenuation_db,
                                             int fft_size)

    cdp_lib_buffer* cdp_lib_filter_notch(cdp_lib_ctx* ctx,
                                          const cdp_lib_buffer* input,
                                          double center_freq,
                                          double width_hz,
                                          double attenuation_db,
                                          int fft_size)

    cdp_lib_buffer* cdp_lib_gate(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double threshold_db,
                                  double attack_ms,
                                  double release_ms,
                                  double hold_ms)

    cdp_lib_buffer* cdp_lib_bitcrush(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      int bit_depth,
                                      int downsample)

    cdp_lib_buffer* cdp_lib_ring_mod(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double freq,
                                      double mix)

    cdp_lib_buffer* cdp_lib_delay(cdp_lib_ctx* ctx,
                                   const cdp_lib_buffer* input,
                                   double delay_ms,
                                   double feedback,
                                   double mix)

    cdp_lib_buffer* cdp_lib_chorus(cdp_lib_ctx* ctx,
                                    const cdp_lib_buffer* input,
                                    double rate,
                                    double depth_ms,
                                    double mix)

    cdp_lib_buffer* cdp_lib_flanger(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input,
                                     double rate,
                                     double depth_ms,
                                     double feedback,
                                     double mix)

cdef extern from "cdp_envelope.h":
    int CDP_FADE_LINEAR
    int CDP_FADE_EXPONENTIAL

    cdp_lib_buffer* cdp_lib_dovetail(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double fade_in_dur,
                                      double fade_out_dur,
                                      int fade_in_type,
                                      int fade_out_type)

    cdp_lib_buffer* cdp_lib_tremolo(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input,
                                     double freq,
                                     double depth,
                                     double gain)

    cdp_lib_buffer* cdp_lib_attack(cdp_lib_ctx* ctx,
                                    const cdp_lib_buffer* input,
                                    double attack_gain,
                                    double attack_time)

cdef extern from "cdp_distort.h":
    cdp_lib_buffer* cdp_lib_distort_overload(cdp_lib_ctx* ctx,
                                              const cdp_lib_buffer* input,
                                              double clip_level,
                                              double depth)

    cdp_lib_buffer* cdp_lib_distort_reverse(cdp_lib_ctx* ctx,
                                             const cdp_lib_buffer* input,
                                             int cycle_count)

    cdp_lib_buffer* cdp_lib_distort_fractal(cdp_lib_ctx* ctx,
                                             const cdp_lib_buffer* input,
                                             double scaling,
                                             double loudness)

    cdp_lib_buffer* cdp_lib_distort_shuffle(cdp_lib_ctx* ctx,
                                             const cdp_lib_buffer* input,
                                             int chunk_count,
                                             unsigned int seed)

cdef extern from "cdp_reverb.h":
    cdp_lib_buffer* cdp_lib_reverb(cdp_lib_ctx* ctx,
                                    const cdp_lib_buffer* input,
                                    double mix,
                                    double decay_time,
                                    double damping,
                                    double lpfreq,
                                    double predelay)

cdef extern from "cdp_granular.h":
    cdp_lib_buffer* cdp_lib_brassage(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double velocity,
                                      double density,
                                      double grainsize_ms,
                                      double scatter,
                                      double pitch_shift,
                                      double amp_variation)

    cdp_lib_buffer* cdp_lib_freeze(cdp_lib_ctx* ctx,
                                    const cdp_lib_buffer* input,
                                    double start_time,
                                    double end_time,
                                    double duration,
                                    double delay,
                                    double randomize,
                                    double pitch_scatter,
                                    double amp_cut,
                                    double gain)
