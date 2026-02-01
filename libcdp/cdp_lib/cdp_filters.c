/*
 * CDP Filter Functions - Implementation
 *
 * Implements lowpass, highpass, bandpass, notch, and parametric EQ filters.
 */

#include "cdp_filters.h"
#include "cdp_lib_internal.h"
#include "cdp_spectral.h"

cdp_lib_buffer* cdp_lib_filter_lowpass(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        double cutoff_freq,
                                        double attenuation_db,
                                        int fft_size) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (cutoff_freq <= 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "cutoff_freq must be positive");
        return NULL;
    }

    if (fft_size == 0) fft_size = 1024;

    /* 1. Analyze input */
    cdp_spectral_data *spectral = cdp_spectral_analyze(
        input->data, input->length,
        input->channels, input->sample_rate,
        fft_size, 3);

    if (spectral == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral analysis failed");
        return NULL;
    }

    /* 2. Apply filter */
    cdp_spectral_data *filtered = cdp_spectral_filter_lowpass(
        spectral, cutoff_freq, attenuation_db);
    cdp_spectral_data_free(spectral);

    /* 3. Synthesize and create output buffer */
    return cdp_lib_spectral_to_buffer(ctx, filtered, input->sample_rate);
}

cdp_lib_buffer* cdp_lib_filter_highpass(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double cutoff_freq,
                                         double attenuation_db,
                                         int fft_size) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (cutoff_freq <= 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "cutoff_freq must be positive");
        return NULL;
    }

    if (fft_size == 0) fft_size = 1024;

    /* 1. Analyze input */
    cdp_spectral_data *spectral = cdp_spectral_analyze(
        input->data, input->length,
        input->channels, input->sample_rate,
        fft_size, 3);

    if (spectral == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral analysis failed");
        return NULL;
    }

    /* 2. Apply filter */
    cdp_spectral_data *filtered = cdp_spectral_filter_highpass(
        spectral, cutoff_freq, attenuation_db);
    cdp_spectral_data_free(spectral);

    /* 3. Synthesize and create output buffer */
    return cdp_lib_spectral_to_buffer(ctx, filtered, input->sample_rate);
}

cdp_lib_buffer* cdp_lib_filter_bandpass(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double low_freq,
                                         double high_freq,
                                         double attenuation_db,
                                         int fft_size) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (low_freq <= 0 || high_freq <= low_freq) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Invalid frequency range: low_freq must be positive and less than high_freq");
        return NULL;
    }

    if (fft_size == 0) fft_size = 1024;

    /* 1. Analyze input */
    cdp_spectral_data *spectral = cdp_spectral_analyze(
        input->data, input->length,
        input->channels, input->sample_rate,
        fft_size, 3);

    if (spectral == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral analysis failed");
        return NULL;
    }

    /* 2. Apply bandpass (attenuate below low_freq and above high_freq) */
    double atten = pow(10.0, attenuation_db / 20.0);
    int num_bins = spectral->num_bins;
    float freq_per_bin = (float)input->sample_rate / fft_size;

    for (int f = 0; f < spectral->num_frames; f++) {
        float *amp = spectral->frames[f].data;

        for (int b = 0; b < num_bins; b++) {
            float bin_freq = b * freq_per_bin;
            if (bin_freq < low_freq || bin_freq > high_freq) {
                amp[b] *= (float)atten;
            }
        }
    }

    /* 3. Synthesize and create output buffer */
    return cdp_lib_spectral_to_buffer(ctx, spectral, input->sample_rate);
}

cdp_lib_buffer* cdp_lib_filter_notch(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double center_freq,
                                      double width_hz,
                                      double attenuation_db,
                                      int fft_size) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (center_freq <= 0 || width_hz <= 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "center_freq and width_hz must be positive");
        return NULL;
    }

    if (fft_size == 0) fft_size = 1024;

    /* 1. Analyze input */
    cdp_spectral_data *spectral = cdp_spectral_analyze(
        input->data, input->length,
        input->channels, input->sample_rate,
        fft_size, 3);

    if (spectral == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral analysis failed");
        return NULL;
    }

    /* 2. Apply notch filter */
    double atten = pow(10.0, attenuation_db / 20.0);
    int num_bins = spectral->num_bins;
    double notch_low = center_freq - width_hz / 2.0;
    double notch_high = center_freq + width_hz / 2.0;
    float freq_per_bin = (float)input->sample_rate / fft_size;

    for (int f = 0; f < spectral->num_frames; f++) {
        float *amp = spectral->frames[f].data;

        for (int b = 0; b < num_bins; b++) {
            float bin_freq = b * freq_per_bin;
            if (bin_freq >= notch_low && bin_freq <= notch_high) {
                amp[b] *= (float)atten;
            }
        }
    }

    /* 3. Synthesize and create output buffer */
    return cdp_lib_spectral_to_buffer(ctx, spectral, input->sample_rate);
}

cdp_lib_buffer* cdp_lib_eq_parametric(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double center_freq,
                                       double gain_db,
                                       double q,
                                       int fft_size) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (center_freq <= 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "center_freq must be positive");
        return NULL;
    }

    if (q <= 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Q must be positive");
        return NULL;
    }

    if (fft_size == 0) fft_size = 1024;

    /* 1. Analyze input */
    cdp_spectral_data *spectral = cdp_spectral_analyze(
        input->data, input->length,
        input->channels, input->sample_rate,
        fft_size, 3);

    if (spectral == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral analysis failed");
        return NULL;
    }

    /* 2. Apply parametric EQ (bell curve) */
    int num_bins = spectral->num_bins;
    float freq_per_bin = (float)input->sample_rate / fft_size;

    /* Calculate bandwidth from Q: BW = center_freq / Q */
    double bandwidth = center_freq / q;

    for (int f = 0; f < spectral->num_frames; f++) {
        float *amp = spectral->frames[f].data;

        for (int b = 0; b < num_bins; b++) {
            float bin_freq = b * freq_per_bin;

            /* Bell curve gain calculation */
            /* Using a Gaussian-like curve centered at center_freq */
            double freq_ratio = (bin_freq - center_freq) / (bandwidth / 2.0);
            double bell = exp(-0.5 * freq_ratio * freq_ratio);

            /* Interpolate between unity gain and target gain based on bell shape */
            double gain_linear = pow(10.0, gain_db / 20.0);
            double effective_gain = 1.0 + (gain_linear - 1.0) * bell;

            amp[b] *= (float)effective_gain;
        }
    }

    /* 3. Synthesize and create output buffer */
    return cdp_lib_spectral_to_buffer(ctx, spectral, input->sample_rate);
}
