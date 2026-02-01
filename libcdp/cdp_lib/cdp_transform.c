/*
 * CDP Transform Functions - Implementation
 *
 * Implements time/pitch/spectral transformation operations.
 */

#include "cdp_transform.h"
#include "cdp_lib_internal.h"
#include "cdp_spectral.h"

cdp_lib_buffer* cdp_lib_loudness(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double gain_db) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    /* Simple implementation - apply gain directly */
    double gain = pow(10.0, gain_db / 20.0);

    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    for (size_t i = 0; i < input->length; i++) {
        output->data[i] = (float)(input->data[i] * gain);
    }

    return output;
}

cdp_lib_buffer* cdp_lib_speed(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input,
                               double speed) {
    if (ctx == NULL || input == NULL || speed <= 0) {
        return NULL;
    }

    /*
     * Speed change via resampling.
     * For now, use simple linear interpolation.
     * A proper implementation would use CDP's modify speed function.
     */

    size_t input_frames = input->length / input->channels;
    size_t output_frames = (size_t)(input_frames / speed);
    size_t output_length = output_frames * input->channels;

    cdp_lib_buffer* output = cdp_lib_buffer_create(
        output_length, input->channels, input->sample_rate);
    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Linear interpolation resampling */
    for (size_t out_frame = 0; out_frame < output_frames; out_frame++) {
        double in_pos = out_frame * speed;
        size_t in_frame = (size_t)in_pos;
        double frac = in_pos - in_frame;

        if (in_frame + 1 >= input_frames) {
            in_frame = input_frames - 2;
            frac = 1.0;
        }

        for (int ch = 0; ch < input->channels; ch++) {
            float s0 = input->data[in_frame * input->channels + ch];
            float s1 = input->data[(in_frame + 1) * input->channels + ch];
            output->data[out_frame * input->channels + ch] =
                (float)(s0 + (s1 - s0) * frac);
        }
    }

    return output;
}

cdp_lib_buffer* cdp_lib_time_stretch(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double factor,
                                      int fft_size,
                                      int overlap) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (factor <= 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Stretch factor must be positive");
        return NULL;
    }

    if (fft_size == 0) fft_size = 1024;
    if (overlap == 0) overlap = 3;

    /* 1. Analyze input to spectral data */
    cdp_spectral_data *spectral = cdp_spectral_analyze(
        input->data, input->length,
        input->channels, input->sample_rate,
        fft_size, overlap);

    if (spectral == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral analysis failed");
        return NULL;
    }

    /* 2. Time-stretch spectral data */
    cdp_spectral_data *stretched = cdp_spectral_time_stretch(spectral, factor);
    cdp_spectral_data_free(spectral);

    /* 3. Synthesize and create output buffer */
    return cdp_lib_spectral_to_buffer(ctx, stretched, input->sample_rate);
}

cdp_lib_buffer* cdp_lib_spectral_blur(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double blur_time,
                                       int fft_size) {
    if (ctx == NULL || input == NULL) {
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

    /* Calculate number of windows to blur over */
    int blur_windows = (int)(blur_time / spectral->frame_time);
    if (blur_windows < 1) blur_windows = 1;

    /* 2. Blur spectral data */
    cdp_spectral_data *blurred = cdp_spectral_blur(spectral, blur_windows);
    cdp_spectral_data_free(spectral);

    /* 3. Synthesize and create output buffer */
    return cdp_lib_spectral_to_buffer(ctx, blurred, input->sample_rate);
}

cdp_lib_buffer* cdp_lib_pitch_shift(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input,
                                     double semitones,
                                     int fft_size) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (semitones == 0) {
        /* No change - just copy */
        cdp_lib_buffer *output = cdp_lib_buffer_create(
            input->length, input->channels, input->sample_rate);
        if (output == NULL) {
            snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                     "Failed to allocate output buffer");
            return NULL;
        }
        memcpy(output->data, input->data, input->length * sizeof(float));
        return output;
    }

    if (fft_size == 0) fft_size = 1024;

    /* Calculate speed factor from semitones: speed = 2^(semitones/12) */
    double speed_factor = pow(2.0, semitones / 12.0);

    /* 1. Change speed (affects pitch and duration) */
    cdp_lib_buffer *sped = cdp_lib_speed(ctx, input, speed_factor);
    if (sped == NULL) {
        return NULL;  /* Error already set */
    }

    /* 2. Time-stretch to restore original duration
     * If we sped up (positive semitones), audio is shorter, need to stretch
     * stretch_factor = speed_factor to compensate
     */
    cdp_lib_buffer *result = cdp_lib_time_stretch(ctx, sped, speed_factor, fft_size, 3);
    cdp_lib_buffer_free(sped);

    return result;
}

cdp_lib_buffer* cdp_lib_spectral_shift(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        double shift_hz,
                                        int fft_size) {
    if (ctx == NULL || input == NULL) {
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

    /* 2. Shift frequencies */
    cdp_spectral_data *shifted = cdp_spectral_freq_shift(spectral, shift_hz);
    cdp_spectral_data_free(spectral);

    /* 3. Synthesize and create output buffer */
    return cdp_lib_spectral_to_buffer(ctx, shifted, input->sample_rate);
}

cdp_lib_buffer* cdp_lib_spectral_stretch(cdp_lib_ctx* ctx,
                                          const cdp_lib_buffer* input,
                                          double max_stretch,
                                          double freq_divide,
                                          double exponent,
                                          int fft_size) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (max_stretch <= 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "max_stretch must be positive");
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

    /* 2. Stretch frequencies */
    cdp_spectral_data *stretched = cdp_spectral_freq_stretch(
        spectral, max_stretch, freq_divide, exponent);
    cdp_spectral_data_free(spectral);

    /* 3. Synthesize and create output buffer */
    return cdp_lib_spectral_to_buffer(ctx, stretched, input->sample_rate);
}
