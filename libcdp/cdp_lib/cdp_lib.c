/*
 * CDP Library Interface - Implementation
 *
 * This implements the library interface by:
 * 1. Setting up CDP's global state
 * 2. Initializing the dz (datalist) structure
 * 3. Configuring I/O redirection
 * 4. Calling CDP processing functions
 * 5. Extracting results
 */

#include "cdp_lib.h"
#include "cdp_spectral.h"
#include "cdp_envelope.h"
#include "cdp_distort.h"
#include "cdp_reverb.h"
#include "cdp_granular.h"
#include "cdp_analysis.h"

#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* Library context */
struct cdp_lib_ctx {
    int initialized;
    char error_msg[512];
};

/* =========================================================================
 * Public API Implementation
 * ========================================================================= */

cdp_lib_ctx* cdp_lib_init(void) {
    cdp_lib_ctx* ctx = (cdp_lib_ctx*)calloc(1, sizeof(cdp_lib_ctx));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->initialized = 1;
    return ctx;
}

void cdp_lib_cleanup(cdp_lib_ctx* ctx) {
    if (ctx == NULL) return;
    free(ctx);
}

const char* cdp_lib_get_error(cdp_lib_ctx* ctx) {
    if (ctx == NULL) return "Context is NULL";
    return ctx->error_msg;
}

cdp_lib_buffer* cdp_lib_buffer_create(size_t length, int channels, int sample_rate) {
    cdp_lib_buffer* buf = (cdp_lib_buffer*)calloc(1, sizeof(cdp_lib_buffer));
    if (buf == NULL) return NULL;

    buf->data = (float*)calloc(length, sizeof(float));
    if (buf->data == NULL) {
        free(buf);
        return NULL;
    }

    buf->length = length;
    buf->channels = channels;
    buf->sample_rate = sample_rate;

    return buf;
}

cdp_lib_buffer* cdp_lib_buffer_from_data(float *data, size_t length,
                                          int channels, int sample_rate) {
    cdp_lib_buffer* buf = (cdp_lib_buffer*)calloc(1, sizeof(cdp_lib_buffer));
    if (buf == NULL) return NULL;

    buf->data = data;
    buf->length = length;
    buf->channels = channels;
    buf->sample_rate = sample_rate;

    return buf;
}

void cdp_lib_buffer_free(cdp_lib_buffer* buf) {
    if (buf == NULL) return;
    if (buf->data) free(buf->data);
    free(buf);
}

/* =========================================================================
 * Processing Functions
 * ========================================================================= */

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

/*
 * Time stretch using phase vocoder
 */
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

    if (stretched == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Time stretch failed");
        return NULL;
    }

    /* 3. Synthesize back to audio */
    size_t out_samples;
    float *audio = cdp_spectral_synthesize(stretched, &out_samples);
    cdp_spectral_data_free(stretched);

    if (audio == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral synthesis failed");
        return NULL;
    }

    /* 4. Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(
        audio, out_samples, 1, input->sample_rate);

    if (output == NULL) {
        free(audio);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    return output;
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

    if (blurred == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral blur failed");
        return NULL;
    }

    /* 3. Synthesize back to audio */
    size_t out_samples;
    float *audio = cdp_spectral_synthesize(blurred, &out_samples);
    cdp_spectral_data_free(blurred);

    if (audio == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral synthesis failed");
        return NULL;
    }

    /* 4. Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(
        audio, out_samples, 1, input->sample_rate);

    if (output == NULL) {
        free(audio);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    return output;
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

    if (shifted == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral frequency shift failed");
        return NULL;
    }

    /* 3. Synthesize back to audio */
    size_t out_samples;
    float *audio = cdp_spectral_synthesize(shifted, &out_samples);
    cdp_spectral_data_free(shifted);

    if (audio == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral synthesis failed");
        return NULL;
    }

    /* 4. Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(
        audio, out_samples, 1, input->sample_rate);

    if (output == NULL) {
        free(audio);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    return output;
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

    if (stretched == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral frequency stretch failed");
        return NULL;
    }

    /* 3. Synthesize back to audio */
    size_t out_samples;
    float *audio = cdp_spectral_synthesize(stretched, &out_samples);
    cdp_spectral_data_free(stretched);

    if (audio == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral synthesis failed");
        return NULL;
    }

    /* 4. Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(
        audio, out_samples, 1, input->sample_rate);

    if (output == NULL) {
        free(audio);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    return output;
}

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

    if (filtered == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Lowpass filter failed");
        return NULL;
    }

    /* 3. Synthesize back to audio */
    size_t out_samples;
    float *audio = cdp_spectral_synthesize(filtered, &out_samples);
    cdp_spectral_data_free(filtered);

    if (audio == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral synthesis failed");
        return NULL;
    }

    /* 4. Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(
        audio, out_samples, 1, input->sample_rate);

    if (output == NULL) {
        free(audio);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    return output;
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

    if (filtered == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Highpass filter failed");
        return NULL;
    }

    /* 3. Synthesize back to audio */
    size_t out_samples;
    float *audio = cdp_spectral_synthesize(filtered, &out_samples);
    cdp_spectral_data_free(filtered);

    if (audio == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral synthesis failed");
        return NULL;
    }

    /* 4. Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(
        audio, out_samples, 1, input->sample_rate);

    if (output == NULL) {
        free(audio);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    return output;
}

/* =========================================================================
 * Envelope Operations (from cdp_envelope.h)
 * ========================================================================= */

cdp_lib_buffer* cdp_lib_dovetail(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double fade_in_dur,
                                  double fade_out_dur,
                                  int fade_in_type,
                                  int fade_out_type) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (fade_in_dur < 0 || fade_out_dur < 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Fade durations must be non-negative");
        return NULL;
    }

    /* Create output buffer (copy of input) */
    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }
    memcpy(output->data, input->data, input->length * sizeof(float));

    size_t frames = input->length / input->channels;
    int channels = input->channels;

    /* Calculate fade lengths in samples */
    size_t fade_in_samples = (size_t)(fade_in_dur * input->sample_rate);
    size_t fade_out_samples = (size_t)(fade_out_dur * input->sample_rate);

    /* Clamp to available length */
    if (fade_in_samples > frames) fade_in_samples = frames;
    if (fade_out_samples > frames) fade_out_samples = frames;

    /* Ensure fades don't overlap */
    if (fade_in_samples + fade_out_samples > frames) {
        size_t total = fade_in_samples + fade_out_samples;
        fade_in_samples = (size_t)((double)fade_in_samples / total * frames);
        fade_out_samples = frames - fade_in_samples;
    }

    /* Apply fade-in */
    for (size_t i = 0; i < fade_in_samples; i++) {
        double t = (double)i / (double)fade_in_samples;
        double gain;

        if (fade_in_type == CDP_FADE_EXPONENTIAL) {
            /* Equal power curve */
            gain = sin(t * M_PI / 2.0);
        } else {
            /* Linear */
            gain = t;
        }

        for (int ch = 0; ch < channels; ch++) {
            output->data[i * channels + ch] *= (float)gain;
        }
    }

    /* Apply fade-out */
    size_t fade_out_start = frames - fade_out_samples;
    for (size_t i = 0; i < fade_out_samples; i++) {
        double t = (double)i / (double)fade_out_samples;
        double gain;

        if (fade_out_type == CDP_FADE_EXPONENTIAL) {
            /* Equal power curve */
            gain = cos(t * M_PI / 2.0);
        } else {
            /* Linear */
            gain = 1.0 - t;
        }

        size_t frame = fade_out_start + i;
        for (int ch = 0; ch < channels; ch++) {
            output->data[frame * channels + ch] *= (float)gain;
        }
    }

    return output;
}

cdp_lib_buffer* cdp_lib_tremolo(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double freq,
                                 double depth,
                                 double gain) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (freq < 0 || freq > 500) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Tremolo frequency must be between 0 and 500 Hz");
        return NULL;
    }

    if (depth < 0 || depth > 1) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Tremolo depth must be between 0 and 1");
        return NULL;
    }

    /* Create output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    size_t frames = input->length / input->channels;
    int channels = input->channels;
    double sample_rate = input->sample_rate;

    /* Angular frequency */
    double omega = 2.0 * M_PI * freq / sample_rate;

    /*
     * Tremolo formula:
     * output = input * gain * (1 - depth * (1 - sin(omega * t)) / 2)
     *
     * This modulates between (1-depth) and 1.0
     */
    for (size_t i = 0; i < frames; i++) {
        /* LFO oscillates between 0 and 1 */
        double lfo = (1.0 + sin(omega * i)) / 2.0;

        /* Modulation: when depth=1, goes from 0 to 1
         * when depth=0, constant at 1 */
        double mod = 1.0 - depth * (1.0 - lfo);

        double sample_gain = gain * mod;

        for (int ch = 0; ch < channels; ch++) {
            output->data[i * channels + ch] =
                (float)(input->data[i * channels + ch] * sample_gain);
        }
    }

    return output;
}

cdp_lib_buffer* cdp_lib_attack(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                double attack_gain,
                                double attack_time) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (attack_time < 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Attack time must be non-negative");
        return NULL;
    }

    if (attack_gain < 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Attack gain must be non-negative");
        return NULL;
    }

    /* Create output buffer (copy of input) */
    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }
    memcpy(output->data, input->data, input->length * sizeof(float));

    size_t frames = input->length / input->channels;
    int channels = input->channels;

    /* Calculate attack length in frames */
    size_t attack_frames = (size_t)(attack_time * input->sample_rate);
    if (attack_frames > frames) attack_frames = frames;

    /*
     * Apply smooth attack envelope:
     * - Starts at attack_gain
     * - Smoothly transitions to 1.0 over attack_time
     * - Uses cosine interpolation for smooth transition
     */
    for (size_t i = 0; i < attack_frames; i++) {
        double t = (double)i / (double)attack_frames;

        /* Cosine interpolation from attack_gain to 1.0 */
        double g = attack_gain + (1.0 - attack_gain) * (1.0 - cos(t * M_PI)) / 2.0;

        for (int ch = 0; ch < channels; ch++) {
            output->data[i * channels + ch] *= (float)g;
        }
    }

    return output;
}

/* =========================================================================
 * Distortion Operations (from cdp_distort.h)
 * ========================================================================= */

/*
 * Helper: Convert buffer to mono if needed
 */
static cdp_lib_buffer* to_mono_if_needed(cdp_lib_ctx* ctx, const cdp_lib_buffer* input) {
    if (input->channels == 1) {
        /* Already mono - make a copy */
        cdp_lib_buffer* output = cdp_lib_buffer_create(
            input->length, 1, input->sample_rate);
        if (output == NULL) {
            snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                     "Failed to allocate buffer");
            return NULL;
        }
        memcpy(output->data, input->data, input->length * sizeof(float));
        return output;
    }

    /* Convert to mono by averaging channels */
    size_t frames = input->length / input->channels;
    cdp_lib_buffer* output = cdp_lib_buffer_create(frames, 1, input->sample_rate);
    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate buffer");
        return NULL;
    }

    for (size_t i = 0; i < frames; i++) {
        float sum = 0;
        for (int ch = 0; ch < input->channels; ch++) {
            sum += input->data[i * input->channels + ch];
        }
        output->data[i] = sum / input->channels;
    }

    return output;
}

cdp_lib_buffer* cdp_lib_distort_overload(cdp_lib_ctx* ctx,
                                          const cdp_lib_buffer* input,
                                          double clip_level,
                                          double depth) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (clip_level <= 0 || clip_level > 1) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "clip_level must be between 0 and 1");
        return NULL;
    }

    if (depth < 0 || depth > 1) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "depth must be between 0 and 1");
        return NULL;
    }

    /* Convert to mono if needed */
    cdp_lib_buffer* mono = to_mono_if_needed(ctx, input);
    if (mono == NULL) return NULL;

    /*
     * Soft clipping formula (depth controls softness):
     * - depth = 0: hard clipping
     * - depth = 1: soft clipping (tanh-like)
     *
     * y = clip_level * tanh(x / clip_level) when depth = 1
     * y = clip(x, -clip_level, clip_level) when depth = 0
     */
    for (size_t i = 0; i < mono->length; i++) {
        float x = mono->data[i];

        if (depth > 0) {
            /* Soft clipping using tanh approximation */
            float normalized = x / (float)clip_level;

            /* Mix between hard and soft clipping based on depth */
            float soft = (float)clip_level * tanhf(normalized);
            float hard;
            if (x > clip_level) hard = (float)clip_level;
            else if (x < -clip_level) hard = -(float)clip_level;
            else hard = x;

            mono->data[i] = (float)(depth * soft + (1.0 - depth) * hard);
        } else {
            /* Pure hard clipping */
            if (x > clip_level) mono->data[i] = (float)clip_level;
            else if (x < -clip_level) mono->data[i] = -(float)clip_level;
        }
    }

    return mono;
}

cdp_lib_buffer* cdp_lib_distort_reverse(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         int cycle_count) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (cycle_count < 1) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "cycle_count must be at least 1");
        return NULL;
    }

    /* Convert to mono if needed */
    cdp_lib_buffer* mono = to_mono_if_needed(ctx, input);
    if (mono == NULL) return NULL;

    /*
     * Find zero crossings and reverse groups of cycles.
     */
    size_t num_samples = mono->length;

    /* Find all zero crossings */
    size_t *crossings = (size_t*)malloc(num_samples * sizeof(size_t) / 10 + 100);
    if (crossings == NULL) {
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate crossings array");
        return NULL;
    }

    size_t num_crossings = 0;
    crossings[num_crossings++] = 0;  /* Start */

    for (size_t i = 1; i < num_samples; i++) {
        /* Detect sign change (zero crossing) */
        if ((mono->data[i-1] >= 0 && mono->data[i] < 0) ||
            (mono->data[i-1] < 0 && mono->data[i] >= 0)) {
            crossings[num_crossings++] = i;
        }
    }
    crossings[num_crossings++] = num_samples;  /* End */

    /* Process groups of cycles */
    int cycles_processed = 0;
    size_t group_start = 0;

    for (size_t c = 1; c < num_crossings; c++) {
        cycles_processed++;

        if (cycles_processed >= cycle_count || c == num_crossings - 1) {
            /* Reverse this group */
            size_t group_end = crossings[c];

            /* Reverse samples in this range */
            size_t left = group_start;
            size_t right = group_end - 1;
            while (left < right) {
                float temp = mono->data[left];
                mono->data[left] = mono->data[right];
                mono->data[right] = temp;
                left++;
                right--;
            }

            group_start = group_end;
            cycles_processed = 0;
        }
    }

    free(crossings);
    return mono;
}

cdp_lib_buffer* cdp_lib_distort_fractal(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double scaling,
                                         double loudness) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (scaling <= 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "scaling must be positive");
        return NULL;
    }

    if (loudness < 0 || loudness > 1) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "loudness must be between 0 and 1");
        return NULL;
    }

    /* Convert to mono if needed */
    cdp_lib_buffer* mono = to_mono_if_needed(ctx, input);
    if (mono == NULL) return NULL;

    size_t num_samples = mono->length;

    /*
     * Fractal distortion: overlay scaled copies of the waveform
     * onto itself at different scales (like a fractal pattern).
     *
     * output[i] = input[i] + k1*input[i*s1] + k2*input[i*s2] + ...
     *
     * where s1, s2, ... are different scaling factors
     */

    /* Allocate output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(num_samples, 1, mono->sample_rate);
    if (output == NULL) {
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Number of fractal iterations */
    int num_iterations = 4;
    double iteration_scale = scaling;
    double iteration_gain = 0.5;

    for (size_t i = 0; i < num_samples; i++) {
        double sum = mono->data[i];
        double gain = iteration_gain;
        double scale = iteration_scale;

        for (int iter = 0; iter < num_iterations; iter++) {
            size_t src_idx = (size_t)((double)i * scale);
            if (src_idx < num_samples) {
                sum += gain * mono->data[src_idx];
            }
            gain *= 0.5;
            scale *= scaling;
        }

        output->data[i] = (float)(sum * loudness);
    }

    cdp_lib_buffer_free(mono);

    /* Normalize to prevent clipping */
    float peak = 0;
    for (size_t i = 0; i < num_samples; i++) {
        float abs_val = fabsf(output->data[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 1.0f / peak;
        for (size_t i = 0; i < num_samples; i++) {
            output->data[i] *= norm;
        }
    }

    return output;
}

cdp_lib_buffer* cdp_lib_distort_shuffle(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         int chunk_count,
                                         unsigned int seed) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (chunk_count < 2) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "chunk_count must be at least 2");
        return NULL;
    }

    /* Convert to mono if needed */
    cdp_lib_buffer* mono = to_mono_if_needed(ctx, input);
    if (mono == NULL) return NULL;

    size_t num_samples = mono->length;
    size_t chunk_size = num_samples / chunk_count;

    if (chunk_size < 1) {
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Audio too short for requested chunk count");
        return NULL;
    }

    /* Create output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(num_samples, 1, mono->sample_rate);
    if (output == NULL) {
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Create chunk order array */
    int* order = (int*)malloc(chunk_count * sizeof(int));
    if (order == NULL) {
        cdp_lib_buffer_free(mono);
        cdp_lib_buffer_free(output);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate order array");
        return NULL;
    }

    /* Initialize order array */
    for (int i = 0; i < chunk_count; i++) {
        order[i] = i;
    }

    /* Shuffle using Fisher-Yates algorithm */
    if (seed == 0) {
        seed = (unsigned int)time(NULL);
    }
    srand(seed);

    for (int i = chunk_count - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = order[i];
        order[i] = order[j];
        order[j] = temp;
    }

    /* Copy chunks in shuffled order */
    for (int i = 0; i < chunk_count; i++) {
        size_t src_start = (size_t)order[i] * chunk_size;
        size_t dst_start = (size_t)i * chunk_size;
        size_t copy_size = chunk_size;

        /* Handle last chunk (may be larger) */
        if (i == chunk_count - 1) {
            copy_size = num_samples - dst_start;
            if (order[i] == chunk_count - 1) {
                /* Source is also the last chunk */
                size_t src_remaining = num_samples - src_start;
                if (src_remaining < copy_size) copy_size = src_remaining;
            }
        }

        memcpy(output->data + dst_start, mono->data + src_start,
               copy_size * sizeof(float));
    }

    free(order);
    cdp_lib_buffer_free(mono);

    return output;
}

/* =========================================================================
 * Reverb (from cdp_reverb.h)
 * ========================================================================= */

/* Comb filter delay line */
typedef struct {
    float *buffer;
    size_t size;
    size_t read_pos;
    float feedback;
    float damp1;
    float damp2;
    float filterstore;
} comb_filter;

/* Allpass filter delay line */
typedef struct {
    float *buffer;
    size_t size;
    size_t index;
    float feedback;
} allpass_filter;

static void comb_init(comb_filter *c, size_t size, float feedback, float damp) {
    c->buffer = (float*)calloc(size, sizeof(float));
    c->size = size;
    c->read_pos = 0;
    c->feedback = feedback;
    c->damp1 = damp;
    c->damp2 = 1.0f - damp;
    c->filterstore = 0;
}

static void comb_free(comb_filter *c) {
    if (c->buffer) free(c->buffer);
}

static float comb_process(comb_filter *c, float input) {
    float output = c->buffer[c->read_pos];

    /* Low-pass filter the feedback */
    c->filterstore = output * c->damp2 + c->filterstore * c->damp1;

    c->buffer[c->read_pos] = input + c->filterstore * c->feedback;

    c->read_pos++;
    if (c->read_pos >= c->size) c->read_pos = 0;

    return output;
}

static void allpass_init(allpass_filter *a, size_t size, float feedback) {
    a->buffer = (float*)calloc(size, sizeof(float));
    a->size = size;
    a->index = 0;
    a->feedback = feedback;
}

static void allpass_free(allpass_filter *a) {
    if (a->buffer) free(a->buffer);
}

static float allpass_process(allpass_filter *a, float input) {
    float bufout = a->buffer[a->index];

    float output = -input + bufout;
    a->buffer[a->index] = input + bufout * a->feedback;

    a->index++;
    if (a->index >= a->size) a->index = 0;

    return output;
}

/* Freeverb-style tuning constants (in samples at 44100 Hz) */
#define COMB_TUNING_L1 1116
#define COMB_TUNING_L2 1188
#define COMB_TUNING_L3 1277
#define COMB_TUNING_L4 1356
#define COMB_TUNING_L5 1422
#define COMB_TUNING_L6 1491
#define COMB_TUNING_L7 1557
#define COMB_TUNING_L8 1617
#define ALLPASS_TUNING_L1 556
#define ALLPASS_TUNING_L2 441
#define ALLPASS_TUNING_L3 341
#define ALLPASS_TUNING_L4 225

cdp_lib_buffer* cdp_lib_reverb(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                double mix,
                                double decay_time,
                                double damping,
                                double lpfreq,
                                double predelay) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    /* Validate parameters */
    if (mix < 0 || mix > 1) mix = 0.5;
    if (decay_time <= 0) decay_time = 2.0;
    if (damping < 0) damping = 0;
    if (damping > 1) damping = 1;
    if (lpfreq <= 0) lpfreq = 8000;
    if (predelay < 0) predelay = 0;

    int sample_rate = input->sample_rate;
    double scale = (double)sample_rate / 44100.0;

    /* Convert to mono if stereo */
    cdp_lib_buffer* mono = to_mono_if_needed(ctx, input);
    if (mono == NULL) return NULL;

    size_t input_samples = mono->length;

    /* Calculate output length (input + reverb tail) */
    size_t tail_samples = (size_t)(decay_time * sample_rate);
    size_t output_samples = input_samples + tail_samples;

    /* Calculate pre-delay in samples */
    size_t predelay_samples = (size_t)(predelay * sample_rate / 1000.0);

    /* Allocate output buffer (stereo) */
    cdp_lib_buffer* output = cdp_lib_buffer_create(output_samples * 2, 2, sample_rate);
    if (output == NULL) {
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Calculate feedback coefficient from RT60 */
    /* RT60 = -3 * delay_time / log10(feedback^2) */
    /* feedback = 10^(-3 * delay_time / RT60) */
    double avg_delay = (COMB_TUNING_L1 + COMB_TUNING_L8) / 2.0 * scale / sample_rate;
    double room_feedback = pow(10.0, -3.0 * avg_delay / decay_time);
    if (room_feedback > 0.99) room_feedback = 0.99;

    /* Initialize comb filters for left channel */
    comb_filter combs_l[8];
    comb_init(&combs_l[0], (size_t)(COMB_TUNING_L1 * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_l[1], (size_t)(COMB_TUNING_L2 * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_l[2], (size_t)(COMB_TUNING_L3 * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_l[3], (size_t)(COMB_TUNING_L4 * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_l[4], (size_t)(COMB_TUNING_L5 * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_l[5], (size_t)(COMB_TUNING_L6 * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_l[6], (size_t)(COMB_TUNING_L7 * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_l[7], (size_t)(COMB_TUNING_L8 * scale), (float)room_feedback, (float)damping);

    /* Initialize comb filters for right channel (slightly offset for stereo) */
    comb_filter combs_r[8];
    comb_init(&combs_r[0], (size_t)((COMB_TUNING_L1 + 23) * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_r[1], (size_t)((COMB_TUNING_L2 + 23) * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_r[2], (size_t)((COMB_TUNING_L3 + 23) * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_r[3], (size_t)((COMB_TUNING_L4 + 23) * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_r[4], (size_t)((COMB_TUNING_L5 + 23) * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_r[5], (size_t)((COMB_TUNING_L6 + 23) * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_r[6], (size_t)((COMB_TUNING_L7 + 23) * scale), (float)room_feedback, (float)damping);
    comb_init(&combs_r[7], (size_t)((COMB_TUNING_L8 + 23) * scale), (float)room_feedback, (float)damping);

    /* Initialize allpass filters */
    allpass_filter allpasses_l[4], allpasses_r[4];
    allpass_init(&allpasses_l[0], (size_t)(ALLPASS_TUNING_L1 * scale), 0.5f);
    allpass_init(&allpasses_l[1], (size_t)(ALLPASS_TUNING_L2 * scale), 0.5f);
    allpass_init(&allpasses_l[2], (size_t)(ALLPASS_TUNING_L3 * scale), 0.5f);
    allpass_init(&allpasses_l[3], (size_t)(ALLPASS_TUNING_L4 * scale), 0.5f);

    allpass_init(&allpasses_r[0], (size_t)((ALLPASS_TUNING_L1 + 23) * scale), 0.5f);
    allpass_init(&allpasses_r[1], (size_t)((ALLPASS_TUNING_L2 + 23) * scale), 0.5f);
    allpass_init(&allpasses_r[2], (size_t)((ALLPASS_TUNING_L3 + 23) * scale), 0.5f);
    allpass_init(&allpasses_r[3], (size_t)((ALLPASS_TUNING_L4 + 23) * scale), 0.5f);

    /* Pre-delay buffer */
    float *predelay_buf = NULL;
    if (predelay_samples > 0) {
        predelay_buf = (float*)calloc(predelay_samples, sizeof(float));
    }
    size_t predelay_idx = 0;

    /* Simple one-pole lowpass filter state */
    float lp_state_l = 0, lp_state_r = 0;
    double lp_coeff = 1.0 - exp(-2.0 * M_PI * lpfreq / sample_rate);

    /* Process audio */
    for (size_t i = 0; i < output_samples; i++) {
        /* Get input sample (or 0 if past end of input) */
        float in_sample;
        if (i < input_samples) {
            in_sample = mono->data[i];
        } else {
            in_sample = 0;
        }

        /* Apply pre-delay */
        float delayed_in = in_sample;
        if (predelay_buf != NULL) {
            float old = predelay_buf[predelay_idx];
            predelay_buf[predelay_idx] = in_sample;
            predelay_idx++;
            if (predelay_idx >= predelay_samples) predelay_idx = 0;
            delayed_in = old;
        }

        /* Process through comb filters (in parallel) */
        float wet_l = 0, wet_r = 0;
        for (int c = 0; c < 8; c++) {
            wet_l += comb_process(&combs_l[c], delayed_in);
            wet_r += comb_process(&combs_r[c], delayed_in);
        }

        /* Process through allpass filters (in series) */
        for (int a = 0; a < 4; a++) {
            wet_l = allpass_process(&allpasses_l[a], wet_l);
            wet_r = allpass_process(&allpasses_r[a], wet_r);
        }

        /* Apply lowpass filter */
        lp_state_l += (float)(lp_coeff * (wet_l - lp_state_l));
        lp_state_r += (float)(lp_coeff * (wet_r - lp_state_r));
        wet_l = lp_state_l;
        wet_r = lp_state_r;

        /* Scale down wet signal */
        wet_l *= 0.015f;
        wet_r *= 0.015f;

        /* Mix dry and wet */
        float dry = (i < input_samples) ? mono->data[i] : 0.0f;
        float out_l = (float)(dry * (1.0 - mix) + wet_l * mix);
        float out_r = (float)(dry * (1.0 - mix) + wet_r * mix);

        output->data[i * 2] = out_l;
        output->data[i * 2 + 1] = out_r;
    }

    /* Cleanup */
    for (int c = 0; c < 8; c++) {
        comb_free(&combs_l[c]);
        comb_free(&combs_r[c]);
    }
    for (int a = 0; a < 4; a++) {
        allpass_free(&allpasses_l[a]);
        allpass_free(&allpasses_r[a]);
    }
    if (predelay_buf) free(predelay_buf);
    cdp_lib_buffer_free(mono);

    return output;
}

/* =========================================================================
 * Granular Processing (from cdp_granular.h)
 * ========================================================================= */

/*
 * Apply Hanning window to a grain
 */
static void apply_grain_window(float *grain, size_t size) {
    for (size_t i = 0; i < size; i++) {
        float window = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (size - 1)));
        grain[i] *= window;
    }
}

cdp_lib_buffer* cdp_lib_brassage(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double velocity,
                                  double density,
                                  double grainsize_ms,
                                  double scatter,
                                  double pitch_shift,
                                  double amp_variation) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    /* Validate parameters */
    if (velocity <= 0) velocity = 1.0;
    if (density <= 0) density = 1.0;
    if (grainsize_ms <= 0) grainsize_ms = 50.0;
    if (scatter < 0) scatter = 0;
    if (scatter > 1) scatter = 1;
    if (amp_variation < 0) amp_variation = 0;
    if (amp_variation > 1) amp_variation = 1;

    int sample_rate = input->sample_rate;

    /* Convert to mono if needed */
    cdp_lib_buffer* mono = to_mono_if_needed(ctx, input);
    if (mono == NULL) return NULL;

    size_t input_samples = mono->length;
    double input_duration = (double)input_samples / sample_rate;

    /* Calculate output duration based on velocity */
    double output_duration = input_duration / velocity;
    size_t output_samples = (size_t)(output_duration * sample_rate);

    /* Grain parameters */
    size_t grain_size = (size_t)(grainsize_ms * sample_rate / 1000.0);
    if (grain_size < 10) grain_size = 10;
    if (grain_size > input_samples) grain_size = input_samples;

    /* Calculate grain spacing based on density */
    double grain_spacing = grainsize_ms / density / 2.0 / 1000.0;  /* 50% overlap at density=1 */
    size_t hop_size = (size_t)(grain_spacing * sample_rate);
    if (hop_size < 1) hop_size = 1;

    /* Pitch shift as speed ratio */
    double pitch_ratio = pow(2.0, pitch_shift / 12.0);

    /* Allocate output buffer (initialize to zero) */
    cdp_lib_buffer* output = cdp_lib_buffer_create(output_samples, 1, sample_rate);
    if (output == NULL) {
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Allocate grain buffer */
    size_t max_grain = (size_t)(grain_size * 2);  /* Allow for pitch down */
    float *grain = (float*)malloc(max_grain * sizeof(float));
    if (grain == NULL) {
        cdp_lib_buffer_free(mono);
        cdp_lib_buffer_free(output);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate grain buffer");
        return NULL;
    }

    /* Initialize random seed */
    srand((unsigned int)time(NULL));

    /* Process grains */
    double src_pos = 0;  /* Position in source */
    size_t dst_pos = 0;  /* Position in destination */

    while (dst_pos < output_samples && src_pos < input_samples - grain_size) {
        /* Calculate actual grain size with pitch shift */
        size_t actual_grain = (size_t)(grain_size / pitch_ratio);
        if (actual_grain > max_grain) actual_grain = max_grain;
        if (actual_grain < 2) actual_grain = 2;

        /* Apply scatter to source position */
        double scattered_pos = src_pos;
        if (scatter > 0) {
            double scatter_range = grain_size * scatter * 2;
            scattered_pos += (((double)rand() / RAND_MAX) - 0.5) * scatter_range;
            if (scattered_pos < 0) scattered_pos = 0;
            if (scattered_pos > input_samples - grain_size)
                scattered_pos = input_samples - grain_size;
        }

        /* Extract grain with pitch shift (resampling) */
        for (size_t i = 0; i < actual_grain; i++) {
            double src_idx = scattered_pos + i * pitch_ratio;
            size_t idx0 = (size_t)src_idx;
            double frac = src_idx - idx0;

            if (idx0 + 1 >= input_samples) {
                grain[i] = 0;
            } else {
                /* Linear interpolation */
                grain[i] = (float)(mono->data[idx0] * (1.0 - frac) +
                                    mono->data[idx0 + 1] * frac);
            }
        }

        /* Apply window to grain */
        apply_grain_window(grain, actual_grain);

        /* Apply amplitude variation */
        double amp = 1.0;
        if (amp_variation > 0) {
            amp = 1.0 - amp_variation * ((double)rand() / RAND_MAX);
        }

        /* Overlap-add grain to output */
        for (size_t i = 0; i < actual_grain && dst_pos + i < output_samples; i++) {
            output->data[dst_pos + i] += (float)(grain[i] * amp);
        }

        /* Advance positions */
        src_pos += hop_size * velocity;
        dst_pos += hop_size;
    }

    free(grain);
    cdp_lib_buffer_free(mono);

    /* Normalize if needed */
    float peak = 0;
    for (size_t i = 0; i < output_samples; i++) {
        float abs_val = fabsf(output->data[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 1.0f / peak;
        for (size_t i = 0; i < output_samples; i++) {
            output->data[i] *= norm;
        }
    }

    return output;
}

cdp_lib_buffer* cdp_lib_freeze(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                double start_time,
                                double end_time,
                                double duration,
                                double delay,
                                double randomize,
                                double pitch_scatter,
                                double amp_cut,
                                double gain) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    int sample_rate = input->sample_rate;

    /* Validate parameters */
    if (start_time < 0) start_time = 0;
    if (end_time <= start_time) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "end_time must be greater than start_time");
        return NULL;
    }
    if (duration <= 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "duration must be positive");
        return NULL;
    }
    if (delay <= 0) delay = 0.05;
    if (randomize < 0) randomize = 0;
    if (randomize > 1) randomize = 1;
    if (pitch_scatter < 0) pitch_scatter = 0;
    if (pitch_scatter > 12) pitch_scatter = 12;
    if (amp_cut < 0) amp_cut = 0;
    if (amp_cut > 1) amp_cut = 1;
    if (gain <= 0) gain = 1.0;

    /* Convert to mono if needed */
    cdp_lib_buffer* mono = to_mono_if_needed(ctx, input);
    if (mono == NULL) return NULL;

    size_t input_samples = mono->length;
    double input_duration = (double)input_samples / sample_rate;

    /* Clamp times to input duration */
    if (start_time >= input_duration) start_time = input_duration * 0.8;
    if (end_time > input_duration) end_time = input_duration;

    /* Calculate segment boundaries */
    size_t seg_start = (size_t)(start_time * sample_rate);
    size_t seg_end = (size_t)(end_time * sample_rate);
    if (seg_end > input_samples) seg_end = input_samples;
    size_t seg_len = seg_end - seg_start;

    if (seg_len < 10) {
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Segment too short");
        return NULL;
    }

    /* Calculate output size */
    size_t output_samples = (size_t)(duration * sample_rate);

    /* Allocate output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(output_samples, 1, sample_rate);
    if (output == NULL) {
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Allocate resampled segment buffer (for pitch shifting) */
    size_t max_seg = seg_len * 2;  /* Allow for pitch down */
    float *seg_buf = (float*)malloc(max_seg * sizeof(float));
    if (seg_buf == NULL) {
        cdp_lib_buffer_free(mono);
        cdp_lib_buffer_free(output);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate segment buffer");
        return NULL;
    }

    /* Initialize random seed */
    srand((unsigned int)time(NULL));

    /* Process iterations */
    size_t write_pos = 0;
    size_t base_delay_samples = (size_t)(delay * sample_rate);

    while (write_pos < output_samples) {
        /* Calculate pitch ratio for this iteration */
        double pitch_ratio = 1.0;
        if (pitch_scatter > 0) {
            double pitch_semitones = (((double)rand() / RAND_MAX) - 0.5) * 2 * pitch_scatter;
            pitch_ratio = pow(2.0, pitch_semitones / 12.0);
        }

        /* Resample segment with pitch shift */
        size_t resampled_len = (size_t)(seg_len / pitch_ratio);
        if (resampled_len > max_seg) resampled_len = max_seg;
        if (resampled_len < 1) resampled_len = 1;

        for (size_t i = 0; i < resampled_len; i++) {
            double src_idx = seg_start + i * pitch_ratio;
            size_t idx0 = (size_t)src_idx;
            double frac = src_idx - idx0;

            if (idx0 + 1 >= input_samples) {
                seg_buf[i] = 0;
            } else {
                seg_buf[i] = (float)(mono->data[idx0] * (1.0 - frac) +
                                     mono->data[idx0 + 1] * frac);
            }
        }

        /* Calculate amplitude for this iteration */
        double iter_amp = gain;
        if (amp_cut > 0) {
            iter_amp *= 1.0 - amp_cut * ((double)rand() / RAND_MAX);
        }

        /* Apply fade envelope */
        size_t fade_len = resampled_len / 10;
        if (fade_len < 10) fade_len = 10;

        /* Crossfade and add to output */
        for (size_t i = 0; i < resampled_len && write_pos + i < output_samples; i++) {
            /* Fade envelope */
            double fade = 1.0;
            if (i < fade_len) {
                fade = (double)i / fade_len;
            } else if (i > resampled_len - fade_len) {
                fade = (double)(resampled_len - i) / fade_len;
            }

            output->data[write_pos + i] += (float)(seg_buf[i] * iter_amp * fade);
        }

        /* Calculate next delay */
        size_t iter_delay = base_delay_samples;
        if (randomize > 0) {
            double rand_factor = 1.0 + (((double)rand() / RAND_MAX) - 0.5) * 2 * randomize;
            iter_delay = (size_t)(iter_delay * rand_factor);
        }
        if (iter_delay < 1) iter_delay = 1;

        write_pos += iter_delay;
    }

    free(seg_buf);
    cdp_lib_buffer_free(mono);

    /* Normalize if needed */
    float peak = 0;
    for (size_t i = 0; i < output_samples; i++) {
        float abs_val = fabsf(output->data[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 1.0f / peak;
        for (size_t i = 0; i < output_samples; i++) {
            output->data[i] *= norm;
        }
    }

    return output;
}

/* =========================================================================
 * Additional Granular Operations (CDP: grain, texture)
 * ========================================================================= */

/* Helper: find grains based on amplitude threshold */
static int find_grains(const float* data, size_t length, int sample_rate,
                       double gate, double grainsize_ms,
                       size_t** grain_starts, size_t** grain_lengths, size_t* grain_count) {

    size_t min_grain = (size_t)(grainsize_ms * sample_rate / 1000.0 * 0.5);
    size_t max_grain = (size_t)(grainsize_ms * sample_rate / 1000.0 * 2.0);
    if (min_grain < 10) min_grain = 10;

    /* First pass: count grains */
    size_t count = 0;
    int in_grain = 0;
    size_t grain_start = 0;

    for (size_t i = 0; i < length; i++) {
        float abs_val = fabsf(data[i]);
        if (!in_grain && abs_val > gate) {
            in_grain = 1;
            grain_start = i;
        } else if (in_grain && abs_val < gate * 0.5) {
            size_t glen = i - grain_start;
            if (glen >= min_grain && glen <= max_grain) {
                count++;
            }
            in_grain = 0;
        }
    }

    if (count == 0) {
        /* No grains found - use regular intervals */
        count = length / (size_t)(grainsize_ms * sample_rate / 1000.0);
        if (count < 1) count = 1;

        *grain_starts = (size_t*)malloc(count * sizeof(size_t));
        *grain_lengths = (size_t*)malloc(count * sizeof(size_t));
        if (*grain_starts == NULL || *grain_lengths == NULL) {
            free(*grain_starts);
            free(*grain_lengths);
            return -1;
        }

        size_t grain_size = (size_t)(grainsize_ms * sample_rate / 1000.0);
        for (size_t i = 0; i < count; i++) {
            (*grain_starts)[i] = i * grain_size;
            (*grain_lengths)[i] = grain_size;
            if ((*grain_starts)[i] + grain_size > length) {
                (*grain_lengths)[i] = length - (*grain_starts)[i];
            }
        }
        *grain_count = count;
        return 0;
    }

    /* Allocate arrays */
    *grain_starts = (size_t*)malloc(count * sizeof(size_t));
    *grain_lengths = (size_t*)malloc(count * sizeof(size_t));
    if (*grain_starts == NULL || *grain_lengths == NULL) {
        free(*grain_starts);
        free(*grain_lengths);
        return -1;
    }

    /* Second pass: record grain positions */
    size_t idx = 0;
    in_grain = 0;

    for (size_t i = 0; i < length && idx < count; i++) {
        float abs_val = fabsf(data[i]);
        if (!in_grain && abs_val > gate) {
            in_grain = 1;
            grain_start = i;
        } else if (in_grain && abs_val < gate * 0.5) {
            size_t glen = i - grain_start;
            if (glen >= min_grain && glen <= max_grain) {
                (*grain_starts)[idx] = grain_start;
                (*grain_lengths)[idx] = glen;
                idx++;
            }
            in_grain = 0;
        }
    }

    *grain_count = idx;
    return 0;
}

cdp_lib_buffer* cdp_lib_grain_cloud(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input,
                                     double gate,
                                     double grainsize_ms,
                                     double density,
                                     double duration,
                                     double scatter,
                                     unsigned int seed) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    /* Validate parameters */
    if (gate <= 0) gate = 0.1;
    if (gate > 1) gate = 1.0;
    if (grainsize_ms <= 0) grainsize_ms = 50.0;
    if (density <= 0) density = 10.0;
    if (duration <= 0) duration = (double)input->length / input->sample_rate;
    if (scatter < 0) scatter = 0;
    if (scatter > 1) scatter = 1;

    int sample_rate = input->sample_rate;

    /* Convert to mono if needed */
    cdp_lib_buffer* mono = to_mono_if_needed(ctx, input);
    if (mono == NULL) return NULL;

    /* Initialize random seed */
    if (seed == 0) seed = (unsigned int)time(NULL);
    srand(seed);

    /* Find grains in source */
    size_t* grain_starts = NULL;
    size_t* grain_lengths = NULL;
    size_t grain_count = 0;

    if (find_grains(mono->data, mono->length, sample_rate, gate, grainsize_ms,
                    &grain_starts, &grain_lengths, &grain_count) < 0) {
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate grain arrays");
        return NULL;
    }

    /* Calculate output size */
    size_t output_samples = (size_t)(duration * sample_rate);

    /* Allocate output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(output_samples, 1, sample_rate);
    if (output == NULL) {
        free(grain_starts);
        free(grain_lengths);
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Calculate grain interval based on density */
    double grain_interval = 1.0 / density;
    size_t hop_samples = (size_t)(grain_interval * sample_rate);
    if (hop_samples < 1) hop_samples = 1;

    /* Generate grain cloud */
    size_t write_pos = 0;

    while (write_pos < output_samples) {
        /* Select a random grain */
        size_t grain_idx = (size_t)(rand() % grain_count);
        size_t src_start = grain_starts[grain_idx];
        size_t grain_len = grain_lengths[grain_idx];

        /* Apply scatter to write position */
        size_t actual_pos = write_pos;
        if (scatter > 0) {
            double scatter_range = hop_samples * scatter;
            int scatter_offset = (int)(((double)rand() / RAND_MAX - 0.5) * 2 * scatter_range);
            if ((int)write_pos + scatter_offset >= 0) {
                actual_pos = write_pos + scatter_offset;
            }
        }

        /* Apply window and add grain to output */
        for (size_t i = 0; i < grain_len && actual_pos + i < output_samples; i++) {
            /* Hann window */
            double window = 0.5 * (1.0 - cos(2.0 * M_PI * i / grain_len));

            if (src_start + i < mono->length) {
                output->data[actual_pos + i] += (float)(mono->data[src_start + i] * window);
            }
        }

        write_pos += hop_samples;
    }

    free(grain_starts);
    free(grain_lengths);
    cdp_lib_buffer_free(mono);

    /* Normalize if needed */
    float peak = 0;
    for (size_t i = 0; i < output_samples; i++) {
        float abs_val = fabsf(output->data[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 0.95f / peak;
        for (size_t i = 0; i < output_samples; i++) {
            output->data[i] *= norm;
        }
    }

    return output;
}

cdp_lib_buffer* cdp_lib_grain_extend(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double grainsize_ms,
                                      double trough,
                                      double extension,
                                      double start_time,
                                      double end_time) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    /* Validate parameters */
    if (grainsize_ms <= 0) grainsize_ms = 15.0;
    if (trough <= 0) trough = 0.3;
    if (trough > 1) trough = 1.0;
    if (extension <= 0) extension = 1.0;

    int sample_rate = input->sample_rate;
    double input_duration = (double)input->length / sample_rate / input->channels;

    if (start_time < 0) start_time = 0;
    if (end_time <= start_time) end_time = input_duration;
    if (end_time > input_duration) end_time = input_duration;

    /* Convert to mono if needed */
    cdp_lib_buffer* mono = to_mono_if_needed(ctx, input);
    if (mono == NULL) return NULL;

    size_t input_samples = mono->length;

    /* Calculate segment boundaries */
    size_t seg_start = (size_t)(start_time * sample_rate);
    size_t seg_end = (size_t)(end_time * sample_rate);
    if (seg_end > input_samples) seg_end = input_samples;
    size_t seg_len = seg_end - seg_start;

    /* Find grains in segment using envelope tracking */
    size_t window_samples = (size_t)(grainsize_ms * sample_rate / 1000.0);
    if (window_samples < 10) window_samples = 10;

    /* Calculate envelope */
    size_t env_len = seg_len / window_samples + 1;
    float* envelope = (float*)malloc(env_len * sizeof(float));
    if (envelope == NULL) {
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate envelope buffer");
        return NULL;
    }

    for (size_t i = 0; i < env_len; i++) {
        size_t start = seg_start + i * window_samples;
        size_t end = start + window_samples;
        if (end > seg_end) end = seg_end;

        float max_val = 0;
        for (size_t j = start; j < end; j++) {
            float abs_val = fabsf(mono->data[j]);
            if (abs_val > max_val) max_val = abs_val;
        }
        envelope[i] = max_val;
    }

    /* Find grain boundaries (peaks in envelope) */
    size_t max_grains = env_len;
    size_t* grain_indices = (size_t*)malloc(max_grains * sizeof(size_t));
    if (grain_indices == NULL) {
        free(envelope);
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate grain indices");
        return NULL;
    }

    size_t grain_count = 0;
    for (size_t i = 1; i < env_len - 1 && grain_count < max_grains; i++) {
        /* Check if this is a peak */
        if (envelope[i] > envelope[i-1] && envelope[i] > envelope[i+1]) {
            /* Check trough depths on either side */
            float left_trough = envelope[i-1] / (envelope[i] + 0.0001f);
            float right_trough = envelope[i+1] / (envelope[i] + 0.0001f);

            if (left_trough < trough || right_trough < trough) {
                grain_indices[grain_count++] = i;
            }
        }
    }

    free(envelope);

    /* If no grains found, use regular spacing */
    if (grain_count == 0) {
        grain_count = seg_len / window_samples;
        if (grain_count < 1) grain_count = 1;
        for (size_t i = 0; i < grain_count; i++) {
            grain_indices[i] = i;
        }
    }

    /* Calculate output size */
    size_t output_samples = input_samples + (size_t)(extension * sample_rate);

    /* Allocate output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(output_samples, 1, sample_rate);
    if (output == NULL) {
        free(grain_indices);
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Copy audio before segment */
    for (size_t i = 0; i < seg_start && i < output_samples; i++) {
        output->data[i] = mono->data[i];
    }

    /* Generate extended grains */
    srand((unsigned int)time(NULL));

    size_t write_pos = seg_start;
    size_t extension_samples = (size_t)(extension * sample_rate);
    size_t target_end = seg_start + seg_len + extension_samples;

    /* Create permutation array for grain order variation */
    size_t* perm = (size_t*)malloc(grain_count * sizeof(size_t));
    if (perm == NULL) {
        free(grain_indices);
        cdp_lib_buffer_free(output);
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate permutation array");
        return NULL;
    }

    for (size_t i = 0; i < grain_count; i++) {
        perm[i] = i;
    }

    /* Splice length for crossfades (15ms as per CDP) */
    size_t splice_len = (size_t)(15.0 * sample_rate / 1000.0);
    if (splice_len < 10) splice_len = 10;

    size_t perm_idx = 0;
    while (write_pos < target_end && write_pos < output_samples) {
        /* Shuffle permutation when we've used all grains */
        if (perm_idx >= grain_count) {
            for (size_t i = grain_count - 1; i > 0; i--) {
                size_t j = rand() % (i + 1);
                size_t tmp = perm[i];
                perm[i] = perm[j];
                perm[j] = tmp;
            }
            perm_idx = 0;
        }

        /* Get grain boundaries */
        size_t grain_idx = grain_indices[perm[perm_idx++]];
        size_t grain_start = seg_start + grain_idx * window_samples;
        size_t grain_end;

        /* Find next grain or use segment end */
        if (perm_idx < grain_count) {
            size_t next_idx = grain_indices[perm_idx < grain_count ? perm[perm_idx] : 0];
            grain_end = seg_start + next_idx * window_samples;
        } else {
            grain_end = grain_start + window_samples * 2;
        }

        if (grain_end > seg_end) grain_end = seg_end;
        size_t grain_len = grain_end - grain_start;
        if (grain_len < window_samples) grain_len = window_samples;

        /* Copy grain with crossfade */
        for (size_t i = 0; i < grain_len && write_pos + i < output_samples; i++) {
            /* Crossfade envelope */
            double fade = 1.0;
            if (i < splice_len) {
                fade = (double)i / splice_len;
            } else if (i > grain_len - splice_len) {
                fade = (double)(grain_len - i) / splice_len;
            }

            size_t src_idx = grain_start + i;
            if (src_idx < input_samples) {
                output->data[write_pos + i] += (float)(mono->data[src_idx] * fade);
            }
        }

        write_pos += grain_len - splice_len;  /* Overlap splices */
    }

    /* Copy audio after segment */
    size_t src_pos = seg_end;
    while (write_pos < output_samples && src_pos < input_samples) {
        output->data[write_pos++] = mono->data[src_pos++];
    }

    free(perm);
    free(grain_indices);
    cdp_lib_buffer_free(mono);

    /* Normalize if needed */
    float peak = 0;
    for (size_t i = 0; i < output_samples; i++) {
        float abs_val = fabsf(output->data[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 0.95f / peak;
        for (size_t i = 0; i < output_samples; i++) {
            output->data[i] *= norm;
        }
    }

    return output;
}

cdp_lib_buffer* cdp_lib_texture_simple(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        double duration,
                                        double density,
                                        double pitch_range,
                                        double amp_range,
                                        double spatial_range,
                                        unsigned int seed) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    /* Validate parameters */
    if (duration <= 0) duration = 5.0;
    if (density <= 0) density = 5.0;
    if (pitch_range < 0) pitch_range = 0;
    if (pitch_range > 24) pitch_range = 24;
    if (amp_range < 0) amp_range = 0;
    if (amp_range > 1) amp_range = 1;
    if (spatial_range < 0) spatial_range = 0;
    if (spatial_range > 1) spatial_range = 1;

    int sample_rate = input->sample_rate;

    /* Convert to mono source if needed */
    cdp_lib_buffer* mono = to_mono_if_needed(ctx, input);
    if (mono == NULL) return NULL;

    size_t input_samples = mono->length;

    /* Initialize random seed */
    if (seed == 0) seed = (unsigned int)time(NULL);
    srand(seed);

    /* Calculate output size (stereo) */
    size_t output_samples = (size_t)(duration * sample_rate);

    /* Allocate stereo output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(output_samples * 2, 2, sample_rate);
    if (output == NULL) {
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Calculate number of events */
    size_t event_count = (size_t)(duration * density);
    if (event_count < 1) event_count = 1;

    /* Splice length for note attacks/releases (15ms) */
    size_t splice_samples = (size_t)(15.0 * sample_rate / 1000.0);
    if (splice_samples < 10) splice_samples = 10;

    /* Generate texture events */
    for (size_t ev = 0; ev < event_count; ev++) {
        /* Random event time */
        double event_time = ((double)rand() / RAND_MAX) * duration;
        size_t event_pos = (size_t)(event_time * sample_rate);

        /* Random pitch (transposition) */
        double pitch_semitones = (((double)rand() / RAND_MAX) - 0.5) * 2 * pitch_range;
        double pitch_ratio = pow(2.0, pitch_semitones / 12.0);

        /* Random amplitude */
        double amp = 1.0 - amp_range * ((double)rand() / RAND_MAX);

        /* Random stereo position */
        double pan = 0.5;
        if (spatial_range > 0) {
            pan = 0.5 + (((double)rand() / RAND_MAX) - 0.5) * spatial_range;
        }
        double left_gain = cos(pan * M_PI / 2);
        double right_gain = sin(pan * M_PI / 2);

        /* Calculate note duration (resampled source length) */
        size_t note_len = (size_t)(input_samples / pitch_ratio);

        /* Add note to output */
        for (size_t i = 0; i < note_len && event_pos + i < output_samples; i++) {
            /* Source position with pitch shift */
            double src_idx = i * pitch_ratio;
            size_t idx0 = (size_t)src_idx;
            double frac = src_idx - idx0;

            if (idx0 + 1 >= input_samples) break;

            /* Interpolated sample */
            float sample = (float)(mono->data[idx0] * (1.0 - frac) +
                                   mono->data[idx0 + 1] * frac);

            /* Apply envelope (fade in/out) */
            double env = 1.0;
            if (i < splice_samples) {
                env = (double)i / splice_samples;
            } else if (i > note_len - splice_samples) {
                env = (double)(note_len - i) / splice_samples;
            }

            /* Apply amp and panning */
            float final_sample = (float)(sample * amp * env);
            size_t out_idx = (event_pos + i) * 2;

            if (out_idx + 1 < output->length) {
                output->data[out_idx] += (float)(final_sample * left_gain);
                output->data[out_idx + 1] += (float)(final_sample * right_gain);
            }
        }
    }

    cdp_lib_buffer_free(mono);

    /* Normalize if needed */
    float peak = 0;
    for (size_t i = 0; i < output->length; i++) {
        float abs_val = fabsf(output->data[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 0.95f / peak;
        for (size_t i = 0; i < output->length; i++) {
            output->data[i] *= norm;
        }
    }

    return output;
}

cdp_lib_buffer* cdp_lib_texture_multi(cdp_lib_ctx* ctx,
                                       const cdp_lib_buffer* input,
                                       double duration,
                                       double density,
                                       int group_size,
                                       double group_spread,
                                       double pitch_range,
                                       double pitch_center,
                                       double amp_decay,
                                       unsigned int seed) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    /* Validate parameters */
    if (duration <= 0) duration = 5.0;
    if (density <= 0) density = 2.0;
    if (group_size < 1) group_size = 1;
    if (group_size > 16) group_size = 16;
    if (group_spread < 0) group_spread = 0.1;
    if (pitch_range < 0) pitch_range = 0;
    if (pitch_range > 24) pitch_range = 24;
    if (amp_decay < 0) amp_decay = 0;
    if (amp_decay > 1) amp_decay = 1;

    int sample_rate = input->sample_rate;

    /* Convert to mono source if needed */
    cdp_lib_buffer* mono = to_mono_if_needed(ctx, input);
    if (mono == NULL) return NULL;

    size_t input_samples = mono->length;

    /* Initialize random seed */
    if (seed == 0) seed = (unsigned int)time(NULL);
    srand(seed);

    /* Calculate output size (stereo) */
    size_t output_samples = (size_t)(duration * sample_rate);

    /* Allocate stereo output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(output_samples * 2, 2, sample_rate);
    if (output == NULL) {
        cdp_lib_buffer_free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Calculate number of groups */
    size_t group_count = (size_t)(duration * density);
    if (group_count < 1) group_count = 1;

    /* Splice length (15ms) */
    size_t splice_samples = (size_t)(15.0 * sample_rate / 1000.0);
    if (splice_samples < 10) splice_samples = 10;

    /* Generate texture groups */
    for (size_t g = 0; g < group_count; g++) {
        /* Random group time */
        double group_time = ((double)rand() / RAND_MAX) * duration;

        /* Random group pitch center */
        double group_pitch = pitch_center + (((double)rand() / RAND_MAX) - 0.5) * pitch_range;

        /* Random stereo position for group */
        double group_pan = 0.5 + (((double)rand() / RAND_MAX) - 0.5) * 0.8;

        /* Actual notes in this group (vary slightly) */
        int notes_in_group = group_size + (rand() % 3) - 1;
        if (notes_in_group < 1) notes_in_group = 1;

        /* Generate notes in group */
        for (int n = 0; n < notes_in_group; n++) {
            /* Note time within group */
            double note_offset = ((double)n / notes_in_group) * group_spread;
            note_offset += (((double)rand() / RAND_MAX) - 0.5) * group_spread * 0.3;
            double note_time = group_time + note_offset;

            if (note_time < 0) note_time = 0;
            if (note_time >= duration) continue;

            size_t note_pos = (size_t)(note_time * sample_rate);

            /* Note pitch (spread around group center) */
            double note_pitch = group_pitch + (((double)rand() / RAND_MAX) - 0.5) * 4;
            double pitch_ratio = pow(2.0, note_pitch / 12.0);

            /* Note amplitude (decay through group) */
            double amp = 1.0 - amp_decay * ((double)n / notes_in_group);
            amp *= 0.5 + 0.5 * ((double)rand() / RAND_MAX);  /* Add variation */

            /* Note pan (spread around group pan) */
            double note_pan = group_pan + (((double)rand() / RAND_MAX) - 0.5) * 0.3;
            if (note_pan < 0) note_pan = 0;
            if (note_pan > 1) note_pan = 1;
            double left_gain = cos(note_pan * M_PI / 2);
            double right_gain = sin(note_pan * M_PI / 2);

            /* Calculate note duration */
            size_t note_len = (size_t)(input_samples / pitch_ratio);

            /* Add note to output */
            for (size_t i = 0; i < note_len && note_pos + i < output_samples; i++) {
                /* Source position with pitch shift */
                double src_idx = i * pitch_ratio;
                size_t idx0 = (size_t)src_idx;
                double frac = src_idx - idx0;

                if (idx0 + 1 >= input_samples) break;

                /* Interpolated sample */
                float sample = (float)(mono->data[idx0] * (1.0 - frac) +
                                       mono->data[idx0 + 1] * frac);

                /* Apply envelope */
                double env = 1.0;
                if (i < splice_samples) {
                    env = (double)i / splice_samples;
                } else if (i > note_len - splice_samples) {
                    env = (double)(note_len - i) / splice_samples;
                }

                /* Apply amp and panning */
                float final_sample = (float)(sample * amp * env);
                size_t out_idx = (note_pos + i) * 2;

                if (out_idx + 1 < output->length) {
                    output->data[out_idx] += (float)(final_sample * left_gain);
                    output->data[out_idx + 1] += (float)(final_sample * right_gain);
                }
            }
        }
    }

    cdp_lib_buffer_free(mono);

    /* Normalize if needed */
    float peak = 0;
    for (size_t i = 0; i < output->length; i++) {
        float abs_val = fabsf(output->data[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 0.95f / peak;
        for (size_t i = 0; i < output->length; i++) {
            output->data[i] *= norm;
        }
    }

    return output;
}

/* =========================================================================
 * Additional Filters
 * ========================================================================= */

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

    /* 3. Synthesize back to audio */
    size_t out_samples;
    float *audio = cdp_spectral_synthesize(spectral, &out_samples);
    cdp_spectral_data_free(spectral);

    if (audio == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral synthesis failed");
        return NULL;
    }

    /* 4. Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(
        audio, out_samples, 1, input->sample_rate);

    if (output == NULL) {
        free(audio);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    return output;
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

    /* 3. Synthesize back to audio */
    size_t out_samples;
    float *audio = cdp_spectral_synthesize(spectral, &out_samples);
    cdp_spectral_data_free(spectral);

    if (audio == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral synthesis failed");
        return NULL;
    }

    /* 4. Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(
        audio, out_samples, 1, input->sample_rate);

    if (output == NULL) {
        free(audio);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    return output;
}

/* =========================================================================
 * Gate
 * ========================================================================= */

cdp_lib_buffer* cdp_lib_gate(cdp_lib_ctx* ctx,
                              const cdp_lib_buffer* input,
                              double threshold_db,
                              double attack_ms,
                              double release_ms,
                              double hold_ms) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    /* Convert threshold from dB to linear */
    double threshold = pow(10.0, threshold_db / 20.0);

    /* Convert times to samples */
    int sample_rate = input->sample_rate;
    size_t attack_samples = (size_t)(attack_ms * sample_rate / 1000.0);
    size_t release_samples = (size_t)(release_ms * sample_rate / 1000.0);
    size_t hold_samples = (size_t)(hold_ms * sample_rate / 1000.0);

    if (attack_samples < 1) attack_samples = 1;
    if (release_samples < 1) release_samples = 1;

    /* Create output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    size_t frames = input->length / input->channels;
    int channels = input->channels;

    /* Gate state */
    double envelope = 0;
    double gain = 0;
    size_t hold_counter = 0;
    int gate_open = 0;

    /* Attack/release coefficients */
    double attack_coeff = 1.0 / attack_samples;
    double release_coeff = 1.0 / release_samples;

    for (size_t i = 0; i < frames; i++) {
        /* Calculate peak level for this frame */
        float peak = 0;
        for (int ch = 0; ch < channels; ch++) {
            float abs_val = fabsf(input->data[i * channels + ch]);
            if (abs_val > peak) peak = abs_val;
        }

        /* Update envelope (simple peak follower) */
        if (peak > envelope) {
            envelope = peak;
        } else {
            envelope *= 0.9999;  /* Slow decay */
        }

        /* Gate logic */
        if (envelope > threshold) {
            /* Above threshold - open gate */
            gate_open = 1;
            hold_counter = hold_samples;
        } else if (hold_counter > 0) {
            /* In hold phase */
            hold_counter--;
        } else {
            /* Below threshold and hold expired - close gate */
            gate_open = 0;
        }

        /* Update gain with attack/release */
        double target_gain = gate_open ? 1.0 : 0.0;
        if (gain < target_gain) {
            gain += attack_coeff;
            if (gain > target_gain) gain = target_gain;
        } else if (gain > target_gain) {
            gain -= release_coeff;
            if (gain < target_gain) gain = target_gain;
        }

        /* Apply gain */
        for (int ch = 0; ch < channels; ch++) {
            output->data[i * channels + ch] =
                (float)(input->data[i * channels + ch] * gain);
        }
    }

    return output;
}

/* =========================================================================
 * Bitcrush
 * ========================================================================= */

cdp_lib_buffer* cdp_lib_bitcrush(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  int bit_depth,
                                  int downsample) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (bit_depth < 1) bit_depth = 1;
    if (bit_depth > 16) bit_depth = 16;
    if (downsample < 1) downsample = 1;

    /* Create output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    size_t frames = input->length / input->channels;
    int channels = input->channels;

    /* Calculate quantization levels */
    double levels = pow(2.0, bit_depth);
    double quantize_factor = levels / 2.0;

    /* Sample-and-hold state for downsampling */
    float *hold_values = (float*)calloc(channels, sizeof(float));
    if (hold_values == NULL) {
        cdp_lib_buffer_free(output);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate hold buffer");
        return NULL;
    }

    for (size_t i = 0; i < frames; i++) {
        /* Only sample at downsample intervals */
        if (i % downsample == 0) {
            for (int ch = 0; ch < channels; ch++) {
                float sample = input->data[i * channels + ch];

                /* Quantize to bit_depth */
                /* Scale to 0..levels, round, scale back */
                double scaled = (sample + 1.0) * quantize_factor;
                scaled = floor(scaled);
                if (scaled < 0) scaled = 0;
                if (scaled >= levels) scaled = levels - 1;
                float quantized = (float)((scaled / quantize_factor) - 1.0);

                hold_values[ch] = quantized;
            }
        }

        /* Output held values */
        for (int ch = 0; ch < channels; ch++) {
            output->data[i * channels + ch] = hold_values[ch];
        }
    }

    free(hold_values);
    return output;
}

/* =========================================================================
 * Ring Modulation
 * ========================================================================= */

cdp_lib_buffer* cdp_lib_ring_mod(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double freq,
                                  double mix) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (freq <= 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Carrier frequency must be positive");
        return NULL;
    }

    if (mix < 0) mix = 0;
    if (mix > 1) mix = 1;

    /* Create output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    size_t frames = input->length / input->channels;
    int channels = input->channels;
    double sample_rate = input->sample_rate;

    /* Angular frequency */
    double omega = 2.0 * M_PI * freq / sample_rate;

    for (size_t i = 0; i < frames; i++) {
        /* Carrier signal */
        float carrier = (float)sin(omega * i);

        for (int ch = 0; ch < channels; ch++) {
            float dry = input->data[i * channels + ch];
            float wet = dry * carrier;

            output->data[i * channels + ch] =
                (float)(dry * (1.0 - mix) + wet * mix);
        }
    }

    return output;
}

/* =========================================================================
 * Delay
 * ========================================================================= */

cdp_lib_buffer* cdp_lib_delay(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input,
                               double delay_ms,
                               double feedback,
                               double mix) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (delay_ms <= 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Delay time must be positive");
        return NULL;
    }

    if (feedback < 0) feedback = 0;
    if (feedback >= 1) feedback = 0.99;
    if (mix < 0) mix = 0;
    if (mix > 1) mix = 1;

    int sample_rate = input->sample_rate;
    int channels = input->channels;
    size_t delay_samples = (size_t)(delay_ms * sample_rate / 1000.0);

    /* Create output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Allocate delay buffers for each channel */
    float **delay_buf = (float**)malloc(channels * sizeof(float*));
    if (delay_buf == NULL) {
        cdp_lib_buffer_free(output);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate delay buffer");
        return NULL;
    }

    for (int ch = 0; ch < channels; ch++) {
        delay_buf[ch] = (float*)calloc(delay_samples, sizeof(float));
        if (delay_buf[ch] == NULL) {
            for (int j = 0; j < ch; j++) free(delay_buf[j]);
            free(delay_buf);
            cdp_lib_buffer_free(output);
            snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                     "Failed to allocate delay buffer");
            return NULL;
        }
    }

    size_t write_pos = 0;
    size_t frames = input->length / channels;

    for (size_t i = 0; i < frames; i++) {
        for (int ch = 0; ch < channels; ch++) {
            float dry = input->data[i * channels + ch];

            /* Read from delay buffer */
            float delayed = delay_buf[ch][write_pos];

            /* Write to delay buffer with feedback */
            delay_buf[ch][write_pos] = dry + delayed * (float)feedback;

            /* Mix output */
            output->data[i * channels + ch] =
                (float)(dry * (1.0 - mix) + delayed * mix);
        }

        /* Advance write position */
        write_pos++;
        if (write_pos >= delay_samples) write_pos = 0;
    }

    /* Cleanup */
    for (int ch = 0; ch < channels; ch++) {
        free(delay_buf[ch]);
    }
    free(delay_buf);

    return output;
}

/* =========================================================================
 * Chorus
 * ========================================================================= */

cdp_lib_buffer* cdp_lib_chorus(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                double rate,
                                double depth_ms,
                                double mix) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (rate <= 0) rate = 1.0;
    if (depth_ms <= 0) depth_ms = 5.0;
    if (mix < 0) mix = 0;
    if (mix > 1) mix = 1;

    int sample_rate = input->sample_rate;
    int channels = input->channels;

    /* Base delay around 20-30ms for chorus effect */
    double base_delay_ms = 25.0;
    size_t max_delay_samples = (size_t)((base_delay_ms + depth_ms) * sample_rate / 1000.0) + 10;

    /* Create output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Allocate delay buffers */
    float **delay_buf = (float**)malloc(channels * sizeof(float*));
    if (delay_buf == NULL) {
        cdp_lib_buffer_free(output);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate delay buffer");
        return NULL;
    }

    for (int ch = 0; ch < channels; ch++) {
        delay_buf[ch] = (float*)calloc(max_delay_samples, sizeof(float));
        if (delay_buf[ch] == NULL) {
            for (int j = 0; j < ch; j++) free(delay_buf[j]);
            free(delay_buf);
            cdp_lib_buffer_free(output);
            snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                     "Failed to allocate delay buffer");
            return NULL;
        }
    }

    double omega = 2.0 * M_PI * rate / sample_rate;
    size_t write_pos = 0;
    size_t frames = input->length / channels;

    /* Convert depth to samples */
    double depth_samples = depth_ms * sample_rate / 1000.0;
    double base_delay_samples = base_delay_ms * sample_rate / 1000.0;

    for (size_t i = 0; i < frames; i++) {
        /* LFO value (0 to 1) */
        double lfo = (1.0 + sin(omega * i)) / 2.0;

        /* Modulated delay time in samples */
        double delay_time = base_delay_samples + lfo * depth_samples;

        /* Calculate read position with linear interpolation */
        double read_pos_f = (double)write_pos - delay_time;
        if (read_pos_f < 0) read_pos_f += max_delay_samples;

        size_t read_pos0 = (size_t)read_pos_f % max_delay_samples;
        size_t read_pos1 = (read_pos0 + 1) % max_delay_samples;
        double frac = read_pos_f - floor(read_pos_f);

        for (int ch = 0; ch < channels; ch++) {
            float dry = input->data[i * channels + ch];

            /* Write to delay buffer */
            delay_buf[ch][write_pos] = dry;

            /* Read with interpolation */
            float delayed = (float)(delay_buf[ch][read_pos0] * (1.0 - frac) +
                                    delay_buf[ch][read_pos1] * frac);

            /* Mix output */
            output->data[i * channels + ch] =
                (float)(dry * (1.0 - mix) + delayed * mix);
        }

        /* Advance write position */
        write_pos++;
        if (write_pos >= max_delay_samples) write_pos = 0;
    }

    /* Cleanup */
    for (int ch = 0; ch < channels; ch++) {
        free(delay_buf[ch]);
    }
    free(delay_buf);

    return output;
}

/* =========================================================================
 * Flanger
 * ========================================================================= */

cdp_lib_buffer* cdp_lib_flanger(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double rate,
                                 double depth_ms,
                                 double feedback,
                                 double mix) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (rate <= 0) rate = 0.5;
    if (depth_ms <= 0) depth_ms = 3.0;
    if (feedback < -0.95) feedback = -0.95;
    if (feedback > 0.95) feedback = 0.95;
    if (mix < 0) mix = 0;
    if (mix > 1) mix = 1;

    int sample_rate = input->sample_rate;
    int channels = input->channels;

    /* Base delay around 1-5ms for flanger effect */
    double base_delay_ms = 2.0;
    size_t max_delay_samples = (size_t)((base_delay_ms + depth_ms) * sample_rate / 1000.0) + 10;

    /* Create output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Allocate delay buffers */
    float **delay_buf = (float**)malloc(channels * sizeof(float*));
    if (delay_buf == NULL) {
        cdp_lib_buffer_free(output);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate delay buffer");
        return NULL;
    }

    for (int ch = 0; ch < channels; ch++) {
        delay_buf[ch] = (float*)calloc(max_delay_samples, sizeof(float));
        if (delay_buf[ch] == NULL) {
            for (int j = 0; j < ch; j++) free(delay_buf[j]);
            free(delay_buf);
            cdp_lib_buffer_free(output);
            snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                     "Failed to allocate delay buffer");
            return NULL;
        }
    }

    double omega = 2.0 * M_PI * rate / sample_rate;
    size_t write_pos = 0;
    size_t frames = input->length / channels;

    /* Convert depth to samples */
    double depth_samples = depth_ms * sample_rate / 1000.0;
    double base_delay_samples = base_delay_ms * sample_rate / 1000.0;

    for (size_t i = 0; i < frames; i++) {
        /* LFO value (0 to 1) */
        double lfo = (1.0 + sin(omega * i)) / 2.0;

        /* Modulated delay time in samples */
        double delay_time = base_delay_samples + lfo * depth_samples;

        /* Calculate read position with linear interpolation */
        double read_pos_f = (double)write_pos - delay_time;
        if (read_pos_f < 0) read_pos_f += max_delay_samples;

        size_t read_pos0 = (size_t)read_pos_f % max_delay_samples;
        size_t read_pos1 = (read_pos0 + 1) % max_delay_samples;
        double frac = read_pos_f - floor(read_pos_f);

        for (int ch = 0; ch < channels; ch++) {
            float dry = input->data[i * channels + ch];

            /* Read with interpolation */
            float delayed = (float)(delay_buf[ch][read_pos0] * (1.0 - frac) +
                                    delay_buf[ch][read_pos1] * frac);

            /* Write to delay buffer with feedback */
            delay_buf[ch][write_pos] = dry + delayed * (float)feedback;

            /* Mix output */
            output->data[i * channels + ch] =
                (float)(dry * (1.0 - mix) + delayed * mix);
        }

        /* Advance write position */
        write_pos++;
        if (write_pos >= max_delay_samples) write_pos = 0;
    }

    /* Cleanup */
    for (int ch = 0; ch < channels; ch++) {
        free(delay_buf[ch]);
    }
    free(delay_buf);

    return output;
}

/* =========================================================================
 * Parametric EQ
 * ========================================================================= */

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

    /* 3. Synthesize back to audio */
    size_t out_samples;
    float *audio = cdp_spectral_synthesize(spectral, &out_samples);
    cdp_spectral_data_free(spectral);

    if (audio == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral synthesis failed");
        return NULL;
    }

    /* 4. Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(
        audio, out_samples, 1, input->sample_rate);

    if (output == NULL) {
        free(audio);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    return output;
}

/* =========================================================================
 * Envelope Follower
 * ========================================================================= */

cdp_lib_buffer* cdp_lib_envelope_follow(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double attack_ms,
                                         double release_ms,
                                         int mode) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (attack_ms < 0 || release_ms < 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Attack and release times must be non-negative");
        return NULL;
    }

    /* Convert to mono if stereo */
    size_t num_frames = input->length / input->channels;
    float *mono = (float *)malloc(num_frames * sizeof(float));
    if (mono == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate mono buffer");
        return NULL;
    }

    if (input->channels == 1) {
        memcpy(mono, input->data, num_frames * sizeof(float));
    } else {
        for (size_t i = 0; i < num_frames; i++) {
            float sum = 0;
            for (int ch = 0; ch < input->channels; ch++) {
                sum += input->data[i * input->channels + ch];
            }
            mono[i] = sum / input->channels;
        }
    }

    /* Calculate coefficients */
    double attack_coef = attack_ms > 0 ?
        exp(-1.0 / (attack_ms * 0.001 * input->sample_rate)) : 0.0;
    double release_coef = release_ms > 0 ?
        exp(-1.0 / (release_ms * 0.001 * input->sample_rate)) : 0.0;

    /* Allocate envelope buffer */
    float *envelope = (float *)malloc(num_frames * sizeof(float));
    if (envelope == NULL) {
        free(mono);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate envelope buffer");
        return NULL;
    }

    /* RMS window size (for RMS mode) */
    int rms_window = (int)(input->sample_rate * 0.01);  /* 10ms window */
    if (rms_window < 1) rms_window = 1;

    double env_value = 0.0;

    for (size_t i = 0; i < num_frames; i++) {
        double input_level;

        if (mode == 1) {
            /* RMS mode */
            double sum_sq = 0.0;
            int count = 0;
            for (int j = -(rms_window / 2); j <= rms_window / 2; j++) {
                int idx = (int)i + j;
                if (idx >= 0 && idx < (int)num_frames) {
                    sum_sq += mono[idx] * mono[idx];
                    count++;
                }
            }
            input_level = sqrt(sum_sq / count);
        } else {
            /* Peak mode */
            input_level = fabs(mono[i]);
        }

        /* Attack/release envelope follower */
        if (input_level > env_value) {
            /* Attack */
            env_value = attack_coef * env_value + (1.0 - attack_coef) * input_level;
        } else {
            /* Release */
            env_value = release_coef * env_value + (1.0 - release_coef) * input_level;
        }

        envelope[i] = (float)env_value;
    }

    free(mono);

    /* Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(
        envelope, num_frames, 1, input->sample_rate);

    if (output == NULL) {
        free(envelope);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    return output;
}

/* =========================================================================
 * Envelope Apply
 * ========================================================================= */

cdp_lib_buffer* cdp_lib_envelope_apply(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        const cdp_lib_buffer* envelope,
                                        double depth) {
    if (ctx == NULL || input == NULL || envelope == NULL) {
        return NULL;
    }

    if (depth < 0 || depth > 1) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Depth must be 0.0 to 1.0");
        return NULL;
    }

    /* Allocate output */
    cdp_lib_buffer *output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);

    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    size_t num_frames = input->length / input->channels;
    size_t env_frames = envelope->length;

    for (size_t i = 0; i < num_frames; i++) {
        /* Get envelope value (with interpolation if lengths differ) */
        float env_val;
        if (env_frames == num_frames) {
            env_val = envelope->data[i];
        } else {
            /* Linear interpolation */
            double pos = (double)i * (env_frames - 1) / (num_frames - 1);
            size_t idx = (size_t)pos;
            double frac = pos - idx;
            if (idx >= env_frames - 1) {
                env_val = envelope->data[env_frames - 1];
            } else {
                env_val = (float)((1.0 - frac) * envelope->data[idx] +
                                  frac * envelope->data[idx + 1]);
            }
        }

        /* Apply envelope with depth control */
        float mod = (float)(1.0 - depth + depth * env_val);

        for (int ch = 0; ch < input->channels; ch++) {
            output->data[i * input->channels + ch] =
                input->data[i * input->channels + ch] * mod;
        }
    }

    return output;
}

/* =========================================================================
 * Compressor
 * ========================================================================= */

cdp_lib_buffer* cdp_lib_compressor(cdp_lib_ctx* ctx,
                                    const cdp_lib_buffer* input,
                                    double threshold_db,
                                    double ratio,
                                    double attack_ms,
                                    double release_ms,
                                    double makeup_gain_db) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (ratio < 1.0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Ratio must be >= 1.0");
        return NULL;
    }

    if (attack_ms < 0 || release_ms < 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Attack and release times must be non-negative");
        return NULL;
    }

    /* Allocate output */
    cdp_lib_buffer *output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);

    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Convert threshold to linear */
    double threshold_lin = pow(10.0, threshold_db / 20.0);
    double makeup_lin = pow(10.0, makeup_gain_db / 20.0);

    /* Calculate coefficients */
    double attack_coef = attack_ms > 0 ?
        exp(-1.0 / (attack_ms * 0.001 * input->sample_rate)) : 0.0;
    double release_coef = release_ms > 0 ?
        exp(-1.0 / (release_ms * 0.001 * input->sample_rate)) : 0.0;

    size_t num_frames = input->length / input->channels;
    int channels = input->channels;

    double env = 0.0;  /* Envelope for gain smoothing */

    for (size_t i = 0; i < num_frames; i++) {
        /* Find peak across all channels for this frame */
        float peak = 0.0f;
        for (int ch = 0; ch < channels; ch++) {
            float sample = fabsf(input->data[i * channels + ch]);
            if (sample > peak) peak = sample;
        }

        /* Calculate gain reduction */
        double gain_reduction = 1.0;
        if (peak > threshold_lin) {
            /* How many dB above threshold */
            double over_db = 20.0 * log10(peak / threshold_lin);
            /* Apply ratio: reduce the overage */
            double reduced_db = over_db / ratio;
            /* Calculate gain to achieve this reduction */
            gain_reduction = pow(10.0, (reduced_db - over_db) / 20.0);
        }

        /* Smooth the gain with attack/release */
        if (gain_reduction < env) {
            /* Attack (gain is decreasing = more compression) */
            env = attack_coef * env + (1.0 - attack_coef) * gain_reduction;
        } else {
            /* Release (gain is increasing = less compression) */
            env = release_coef * env + (1.0 - release_coef) * gain_reduction;
        }

        /* Apply gain with makeup */
        float final_gain = (float)(env * makeup_lin);

        for (int ch = 0; ch < channels; ch++) {
            output->data[i * channels + ch] =
                input->data[i * channels + ch] * final_gain;
        }
    }

    return output;
}

/* =========================================================================
 * Limiter
 * ========================================================================= */

cdp_lib_buffer* cdp_lib_limiter(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double threshold_db,
                                 double attack_ms,
                                 double release_ms) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (attack_ms < 0 || release_ms < 0) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Attack and release times must be non-negative");
        return NULL;
    }

    /* Allocate output */
    cdp_lib_buffer *output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);

    if (output == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate output buffer");
        return NULL;
    }

    /* Convert threshold to linear */
    double threshold_lin = pow(10.0, threshold_db / 20.0);

    /* Calculate coefficients */
    double attack_coef = attack_ms > 0 ?
        exp(-1.0 / (attack_ms * 0.001 * input->sample_rate)) : 0.0;
    double release_coef = release_ms > 0 ?
        exp(-1.0 / (release_ms * 0.001 * input->sample_rate)) : 0.0;

    size_t num_frames = input->length / input->channels;
    int channels = input->channels;

    double gain = 1.0;  /* Current gain */

    for (size_t i = 0; i < num_frames; i++) {
        /* Find peak across all channels for this frame */
        float peak = 0.0f;
        for (int ch = 0; ch < channels; ch++) {
            float sample = fabsf(input->data[i * channels + ch]);
            if (sample > peak) peak = sample;
        }

        /* Calculate required gain to stay under threshold */
        double target_gain = 1.0;
        if (peak * gain > threshold_lin) {
            target_gain = threshold_lin / peak;
        }

        /* Smooth the gain with attack/release */
        if (target_gain < gain) {
            /* Attack (need to reduce gain quickly) */
            if (attack_ms == 0) {
                gain = target_gain;  /* Hard limiting */
            } else {
                gain = attack_coef * gain + (1.0 - attack_coef) * target_gain;
            }
        } else {
            /* Release (can increase gain) */
            gain = release_coef * gain + (1.0 - release_coef) * target_gain;
        }

        /* Ensure we never exceed threshold (hard clip as safety) */
        for (int ch = 0; ch < channels; ch++) {
            float sample = input->data[i * channels + ch] * (float)gain;
            /* Hard clip at threshold as final safety */
            if (sample > threshold_lin) sample = (float)threshold_lin;
            if (sample < -threshold_lin) sample = (float)-threshold_lin;
            output->data[i * channels + ch] = sample;
        }
    }

    return output;
}

/* =========================================================================
 * Morphing and Cross-synthesis (CDP: morph, combine)
 * ========================================================================= */

cdp_lib_buffer* cdp_lib_morph(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input1,
                               const cdp_lib_buffer* input2,
                               double morph_start,
                               double morph_end,
                               double exponent,
                               int fft_size) {
    if (ctx == NULL || input1 == NULL || input2 == NULL) {
        return NULL;
    }

    /* Validate parameters */
    if (morph_start < 0) morph_start = 0;
    if (morph_start > 1) morph_start = 1;
    if (morph_end < morph_start) morph_end = morph_start;
    if (morph_end > 1) morph_end = 1;
    if (exponent <= 0) exponent = 1.0;
    if (fft_size == 0) fft_size = 1024;

    /* Ensure sample rates match */
    if (input1->sample_rate != input2->sample_rate) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Sample rates must match for morphing");
        return NULL;
    }

    int sample_rate = input1->sample_rate;

    /* Convert to mono if needed */
    cdp_lib_buffer* mono1 = to_mono_if_needed(ctx, input1);
    if (mono1 == NULL) return NULL;

    cdp_lib_buffer* mono2 = to_mono_if_needed(ctx, input2);
    if (mono2 == NULL) {
        cdp_lib_buffer_free(mono1);
        return NULL;
    }

    /* 1. Analyze both inputs */
    cdp_spectral_data *spec1 = cdp_spectral_analyze(
        mono1->data, mono1->length, 1, sample_rate,
        fft_size, 4);

    if (spec1 == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral analysis of input1 failed");
        cdp_lib_buffer_free(mono1);
        cdp_lib_buffer_free(mono2);
        return NULL;
    }

    cdp_spectral_data *spec2 = cdp_spectral_analyze(
        mono2->data, mono2->length, 1, sample_rate,
        fft_size, 4);

    if (spec2 == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral analysis of input2 failed");
        cdp_spectral_data_free(spec1);
        cdp_lib_buffer_free(mono1);
        cdp_lib_buffer_free(mono2);
        return NULL;
    }

    /* Calculate output frames (use max) */
    int output_frames = spec1->num_frames > spec2->num_frames ?
                        spec1->num_frames : spec2->num_frames;

    int num_bins = spec1->num_bins;

    /* Allocate output spectral data */
    cdp_spectral_data *spec_out = (cdp_spectral_data *)calloc(1, sizeof(cdp_spectral_data));
    if (spec_out == NULL) {
        cdp_spectral_data_free(spec1);
        cdp_spectral_data_free(spec2);
        cdp_lib_buffer_free(mono1);
        cdp_lib_buffer_free(mono2);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate spectral data");
        return NULL;
    }

    spec_out->frames = (cdp_spectral_frame *)calloc(output_frames, sizeof(cdp_spectral_frame));
    if (spec_out->frames == NULL) {
        free(spec_out);
        cdp_spectral_data_free(spec1);
        cdp_spectral_data_free(spec2);
        cdp_lib_buffer_free(mono1);
        cdp_lib_buffer_free(mono2);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate spectral frames");
        return NULL;
    }

    spec_out->num_frames = output_frames;
    spec_out->num_bins = num_bins;
    spec_out->fft_size = fft_size;
    spec_out->overlap = spec1->overlap;
    spec_out->sample_rate = (float)sample_rate;
    spec_out->frame_time = spec1->frame_time;

    /* Process each frame */
    for (int f = 0; f < output_frames; f++) {
        /* Allocate output frame */
        spec_out->frames[f].data = (float *)malloc(num_bins * 2 * sizeof(float));
        spec_out->frames[f].num_bins = num_bins;
        spec_out->frames[f].fft_size = fft_size;
        spec_out->frames[f].sample_rate = (float)sample_rate;

        if (spec_out->frames[f].data == NULL) {
            cdp_spectral_data_free(spec_out);
            cdp_spectral_data_free(spec1);
            cdp_spectral_data_free(spec2);
            cdp_lib_buffer_free(mono1);
            cdp_lib_buffer_free(mono2);
            snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                     "Failed to allocate frame data");
            return NULL;
        }

        /* Calculate morph position (0 to 1) */
        double pos = (output_frames > 1) ? (double)f / (output_frames - 1) : 0.5;

        /* Calculate interpolation factor based on morph timing */
        double interp = 0.0;
        if (pos < morph_start) {
            interp = 0.0;  /* Before morph: use input1 */
        } else if (pos > morph_end) {
            interp = 1.0;  /* After morph: use input2 */
        } else if (morph_end > morph_start) {
            /* During morph: interpolate */
            interp = (pos - morph_start) / (morph_end - morph_start);
            /* Apply exponent curve */
            interp = pow(interp, exponent);
        }

        /* Get source frame indices (clamp to available frames) */
        int f1 = f < spec1->num_frames ? f : spec1->num_frames - 1;
        int f2 = f < spec2->num_frames ? f : spec2->num_frames - 1;

        float *amp1 = spec1->frames[f1].data;
        float *freq1 = spec1->frames[f1].data + num_bins;
        float *amp2 = spec2->frames[f2].data;
        float *freq2 = spec2->frames[f2].data + num_bins;

        float *out_amp = spec_out->frames[f].data;
        float *out_freq = spec_out->frames[f].data + num_bins;

        /* Interpolate each bin */
        for (int b = 0; b < num_bins; b++) {
            /* Amplitude: linear interpolation */
            out_amp[b] = amp1[b] + (float)interp * (amp2[b] - amp1[b]);

            /* Frequency: exponential interpolation for natural pitch transition */
            if (freq1[b] <= 0 || freq2[b] <= 0) {
                /* Handle zero frequencies */
                out_freq[b] = freq1[b] + (float)interp * (freq2[b] - freq1[b]);
            } else {
                /* Exponential interpolation: f_out = f1 * 2^(log2(f2/f1) * interp) */
                double log_ratio = log2(freq2[b] / freq1[b]);
                out_freq[b] = (float)(freq1[b] * pow(2.0, log_ratio * interp));
            }
        }
    }

    /* Synthesize output */
    size_t synth_len;
    float *synth_data = cdp_spectral_synthesize(spec_out, &synth_len);

    cdp_spectral_data_free(spec1);
    cdp_spectral_data_free(spec2);
    cdp_spectral_data_free(spec_out);
    cdp_lib_buffer_free(mono1);
    cdp_lib_buffer_free(mono2);

    if (synth_data == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral synthesis failed");
        return NULL;
    }

    /* Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(synth_data, synth_len, 1, sample_rate);
    if (output == NULL) {
        free(synth_data);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    /* Normalize */
    float peak = 0;
    for (size_t i = 0; i < output->length; i++) {
        float abs_val = fabsf(output->data[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 0.95f / peak;
        for (size_t i = 0; i < output->length; i++) {
            output->data[i] *= norm;
        }
    }

    return output;
}

cdp_lib_buffer* cdp_lib_morph_glide(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input1,
                                     const cdp_lib_buffer* input2,
                                     double duration,
                                     int fft_size) {
    if (ctx == NULL || input1 == NULL || input2 == NULL) {
        return NULL;
    }

    /* Validate parameters */
    if (duration <= 0) duration = 1.0;
    if (fft_size == 0) fft_size = 1024;

    /* Ensure sample rates match */
    if (input1->sample_rate != input2->sample_rate) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Sample rates must match for glide");
        return NULL;
    }

    int sample_rate = input1->sample_rate;

    /* Convert to mono if needed */
    cdp_lib_buffer* mono1 = to_mono_if_needed(ctx, input1);
    if (mono1 == NULL) return NULL;

    cdp_lib_buffer* mono2 = to_mono_if_needed(ctx, input2);
    if (mono2 == NULL) {
        cdp_lib_buffer_free(mono1);
        return NULL;
    }

    /* Analyze both inputs */
    cdp_spectral_data *spec1 = cdp_spectral_analyze(
        mono1->data, mono1->length, 1, sample_rate,
        fft_size, 4);

    if (spec1 == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral analysis of input1 failed");
        cdp_lib_buffer_free(mono1);
        cdp_lib_buffer_free(mono2);
        return NULL;
    }

    cdp_spectral_data *spec2 = cdp_spectral_analyze(
        mono2->data, mono2->length, 1, sample_rate,
        fft_size, 4);

    if (spec2 == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral analysis of input2 failed");
        cdp_spectral_data_free(spec1);
        cdp_lib_buffer_free(mono1);
        cdp_lib_buffer_free(mono2);
        return NULL;
    }

    /* Calculate output frames based on duration */
    int hop_size = fft_size / 4;
    int output_frames = (int)(duration * sample_rate / hop_size);
    if (output_frames < 2) output_frames = 2;

    int num_bins = spec1->num_bins;

    /* Allocate output spectral data */
    cdp_spectral_data *spec_out = (cdp_spectral_data *)calloc(1, sizeof(cdp_spectral_data));
    if (spec_out == NULL) {
        cdp_spectral_data_free(spec1);
        cdp_spectral_data_free(spec2);
        cdp_lib_buffer_free(mono1);
        cdp_lib_buffer_free(mono2);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate spectral data");
        return NULL;
    }

    spec_out->frames = (cdp_spectral_frame *)calloc(output_frames, sizeof(cdp_spectral_frame));
    if (spec_out->frames == NULL) {
        free(spec_out);
        cdp_spectral_data_free(spec1);
        cdp_spectral_data_free(spec2);
        cdp_lib_buffer_free(mono1);
        cdp_lib_buffer_free(mono2);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate spectral frames");
        return NULL;
    }

    spec_out->num_frames = output_frames;
    spec_out->num_bins = num_bins;
    spec_out->fft_size = fft_size;
    spec_out->overlap = spec1->overlap;
    spec_out->sample_rate = (float)sample_rate;
    spec_out->frame_time = (float)hop_size / sample_rate;

    /* Get representative frames (middle) from each input */
    int frame1 = spec1->num_frames / 2;
    int frame2 = spec2->num_frames / 2;

    float *src_amp1 = spec1->frames[frame1].data;
    float *src_freq1 = spec1->frames[frame1].data + num_bins;
    float *src_amp2 = spec2->frames[frame2].data;
    float *src_freq2 = spec2->frames[frame2].data + num_bins;

    /* Process each output frame with linear glide */
    for (int f = 0; f < output_frames; f++) {
        /* Allocate output frame */
        spec_out->frames[f].data = (float *)malloc(num_bins * 2 * sizeof(float));
        spec_out->frames[f].num_bins = num_bins;
        spec_out->frames[f].fft_size = fft_size;
        spec_out->frames[f].sample_rate = (float)sample_rate;

        if (spec_out->frames[f].data == NULL) {
            cdp_spectral_data_free(spec_out);
            cdp_spectral_data_free(spec1);
            cdp_spectral_data_free(spec2);
            cdp_lib_buffer_free(mono1);
            cdp_lib_buffer_free(mono2);
            snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                     "Failed to allocate frame data");
            return NULL;
        }

        /* Linear interpolation factor */
        double interp = (output_frames > 1) ? (double)f / (output_frames - 1) : 0.5;

        float *out_amp = spec_out->frames[f].data;
        float *out_freq = spec_out->frames[f].data + num_bins;

        /* Interpolate each bin */
        for (int b = 0; b < num_bins; b++) {
            /* Amplitude: linear interpolation */
            out_amp[b] = src_amp1[b] + (float)interp * (src_amp2[b] - src_amp1[b]);

            /* Frequency: exponential interpolation */
            if (src_freq1[b] <= 0 || src_freq2[b] <= 0) {
                out_freq[b] = src_freq1[b] + (float)interp * (src_freq2[b] - src_freq1[b]);
            } else {
                double log_ratio = log2(src_freq2[b] / src_freq1[b]);
                out_freq[b] = (float)(src_freq1[b] * pow(2.0, log_ratio * interp));
            }
        }
    }

    /* Synthesize output */
    size_t synth_len;
    float *synth_data = cdp_spectral_synthesize(spec_out, &synth_len);

    cdp_spectral_data_free(spec1);
    cdp_spectral_data_free(spec2);
    cdp_spectral_data_free(spec_out);
    cdp_lib_buffer_free(mono1);
    cdp_lib_buffer_free(mono2);

    if (synth_data == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral synthesis failed");
        return NULL;
    }

    /* Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(synth_data, synth_len, 1, sample_rate);
    if (output == NULL) {
        free(synth_data);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    /* Normalize */
    float peak = 0;
    for (size_t i = 0; i < output->length; i++) {
        float abs_val = fabsf(output->data[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 0.95f / peak;
        for (size_t i = 0; i < output->length; i++) {
            output->data[i] *= norm;
        }
    }

    return output;
}

cdp_lib_buffer* cdp_lib_cross_synth(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input1,
                                     const cdp_lib_buffer* input2,
                                     int mode,
                                     double mix,
                                     int fft_size) {
    if (ctx == NULL || input1 == NULL || input2 == NULL) {
        return NULL;
    }

    /* Validate parameters */
    if (mode < 0) mode = 0;
    if (mode > 1) mode = 1;
    if (mix < 0) mix = 0;
    if (mix > 1) mix = 1;
    if (fft_size == 0) fft_size = 1024;

    /* Ensure sample rates match */
    if (input1->sample_rate != input2->sample_rate) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Sample rates must match for cross-synthesis");
        return NULL;
    }

    int sample_rate = input1->sample_rate;

    /* Convert to mono if needed */
    cdp_lib_buffer* mono1 = to_mono_if_needed(ctx, input1);
    if (mono1 == NULL) return NULL;

    cdp_lib_buffer* mono2 = to_mono_if_needed(ctx, input2);
    if (mono2 == NULL) {
        cdp_lib_buffer_free(mono1);
        return NULL;
    }

    /* Determine amp source and freq source based on mode */
    cdp_lib_buffer* amp_source = (mode == 0) ? mono1 : mono2;
    cdp_lib_buffer* freq_source = (mode == 0) ? mono2 : mono1;

    /* Analyze both inputs */
    cdp_spectral_data *spec_amp = cdp_spectral_analyze(
        amp_source->data, amp_source->length, 1, sample_rate,
        fft_size, 4);

    if (spec_amp == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral analysis of amp source failed");
        cdp_lib_buffer_free(mono1);
        cdp_lib_buffer_free(mono2);
        return NULL;
    }

    cdp_spectral_data *spec_freq = cdp_spectral_analyze(
        freq_source->data, freq_source->length, 1, sample_rate,
        fft_size, 4);

    if (spec_freq == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral analysis of freq source failed");
        cdp_spectral_data_free(spec_amp);
        cdp_lib_buffer_free(mono1);
        cdp_lib_buffer_free(mono2);
        return NULL;
    }

    /* Use minimum frame count */
    int output_frames = spec_amp->num_frames < spec_freq->num_frames ?
                        spec_amp->num_frames : spec_freq->num_frames;

    int num_bins = spec_amp->num_bins;

    /* Allocate output spectral data */
    cdp_spectral_data *spec_out = (cdp_spectral_data *)calloc(1, sizeof(cdp_spectral_data));
    if (spec_out == NULL) {
        cdp_spectral_data_free(spec_amp);
        cdp_spectral_data_free(spec_freq);
        cdp_lib_buffer_free(mono1);
        cdp_lib_buffer_free(mono2);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate spectral data");
        return NULL;
    }

    spec_out->frames = (cdp_spectral_frame *)calloc(output_frames, sizeof(cdp_spectral_frame));
    if (spec_out->frames == NULL) {
        free(spec_out);
        cdp_spectral_data_free(spec_amp);
        cdp_spectral_data_free(spec_freq);
        cdp_lib_buffer_free(mono1);
        cdp_lib_buffer_free(mono2);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to allocate spectral frames");
        return NULL;
    }

    spec_out->num_frames = output_frames;
    spec_out->num_bins = num_bins;
    spec_out->fft_size = fft_size;
    spec_out->overlap = spec_amp->overlap;
    spec_out->sample_rate = (float)sample_rate;
    spec_out->frame_time = spec_amp->frame_time;

    /* Process each frame: take amplitude from one, frequency from other */
    for (int f = 0; f < output_frames; f++) {
        /* Allocate output frame */
        spec_out->frames[f].data = (float *)malloc(num_bins * 2 * sizeof(float));
        spec_out->frames[f].num_bins = num_bins;
        spec_out->frames[f].fft_size = fft_size;
        spec_out->frames[f].sample_rate = (float)sample_rate;

        if (spec_out->frames[f].data == NULL) {
            cdp_spectral_data_free(spec_out);
            cdp_spectral_data_free(spec_amp);
            cdp_spectral_data_free(spec_freq);
            cdp_lib_buffer_free(mono1);
            cdp_lib_buffer_free(mono2);
            snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                     "Failed to allocate frame data");
            return NULL;
        }

        float *amp_src = spec_amp->frames[f].data;
        float *freq_amp_src = spec_amp->frames[f].data + num_bins;
        float *amp_freq_src = spec_freq->frames[f].data;
        float *freq_src = spec_freq->frames[f].data + num_bins;

        float *out_amp = spec_out->frames[f].data;
        float *out_freq = spec_out->frames[f].data + num_bins;

        for (int b = 0; b < num_bins; b++) {
            /* Cross: amp from amp_source, freq from freq_source */
            float amp_cross = amp_src[b];
            float freq_cross = freq_src[b];

            /* Original: amp from freq_source, freq from amp_source */
            float amp_orig = amp_freq_src[b];
            float freq_orig = freq_amp_src[b];

            /* Mix between original and cross-synthesized */
            out_amp[b] = amp_orig + (float)mix * (amp_cross - amp_orig);
            out_freq[b] = freq_orig + (float)mix * (freq_cross - freq_orig);
        }
    }

    /* Synthesize output */
    size_t synth_len;
    float *synth_data = cdp_spectral_synthesize(spec_out, &synth_len);

    cdp_spectral_data_free(spec_amp);
    cdp_spectral_data_free(spec_freq);
    cdp_spectral_data_free(spec_out);
    cdp_lib_buffer_free(mono1);
    cdp_lib_buffer_free(mono2);

    if (synth_data == NULL) {
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Spectral synthesis failed");
        return NULL;
    }

    /* Create output buffer */
    cdp_lib_buffer *output = cdp_lib_buffer_from_data(synth_data, synth_len, 1, sample_rate);
    if (output == NULL) {
        free(synth_data);
        snprintf(ctx->error_msg, sizeof(ctx->error_msg),
                 "Failed to create output buffer");
        return NULL;
    }

    /* Normalize */
    float peak = 0;
    for (size_t i = 0; i < output->length; i++) {
        float abs_val = fabsf(output->data[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 0.95f / peak;
        for (size_t i = 0; i < output->length; i++) {
            output->data[i] *= norm;
        }
    }

    return output;
}
/* =========================================================================
 * Analysis Functions (CDP: pitch, formants, get_partials)
 * This file is appended to cdp_lib.c
 * ========================================================================= */

/* Helper: Apply Hamming window */
static void analysis_apply_hamming(float *frame, int size) {
    for (int i = 0; i < size; i++) {
        double w = 0.54 - 0.46 * cos(2.0 * M_PI * i / (size - 1));
        frame[i] *= (float)w;
    }
}

/* Helper: Compute autocorrelation */
static void analysis_autocorrelation(const float *frame, int size, float *r, int max_lag) {
    for (int lag = 0; lag < max_lag; lag++) {
        double sum = 0;
        for (int i = 0; i < size - lag; i++) {
            sum += frame[i] * frame[i + lag];
        }
        r[lag] = (float)sum;
    }
}

/* Memory Management */
void cdp_pitch_data_free(cdp_pitch_data* data) {
    if (data == NULL) return;
    if (data->pitch) free(data->pitch);
    if (data->confidence) free(data->confidence);
    free(data);
}

void cdp_formant_data_free(cdp_formant_data* data) {
    if (data == NULL) return;
    if (data->f1) free(data->f1);
    if (data->f2) free(data->f2);
    if (data->f3) free(data->f3);
    if (data->f4) free(data->f4);
    if (data->b1) free(data->b1);
    if (data->b2) free(data->b2);
    if (data->b3) free(data->b3);
    if (data->b4) free(data->b4);
    free(data);
}

void cdp_partial_data_free(cdp_partial_data* data) {
    if (data == NULL) return;
    if (data->tracks) {
        for (int i = 0; i < data->num_tracks; i++) {
            if (data->tracks[i].freq) free(data->tracks[i].freq);
            if (data->tracks[i].amp) free(data->tracks[i].amp);
        }
        free(data->tracks);
    }
    free(data);
}

/* YIN Pitch Detection helpers */
static void yin_difference(const float *frame, int size, float *d, int max_lag) {
    d[0] = 0;
    for (int tau = 1; tau < max_lag; tau++) {
        double sum = 0;
        for (int i = 0; i < size - max_lag; i++) {
            double diff = frame[i] - frame[i + tau];
            sum += diff * diff;
        }
        d[tau] = (float)sum;
    }
}

static void yin_cumulative_mean(float *d, int max_lag) {
    d[0] = 1.0f;
    double running_sum = 0;
    for (int tau = 1; tau < max_lag; tau++) {
        running_sum += d[tau];
        if (running_sum > 0) {
            d[tau] = d[tau] * tau / (float)running_sum;
        } else {
            d[tau] = 1.0f;
        }
    }
}

static int yin_find_minimum(const float *d, int max_lag, float threshold) {
    int tau = 2;
    while (tau < max_lag - 1) {
        if (d[tau] < threshold) {
            while (tau + 1 < max_lag - 1 && d[tau + 1] < d[tau]) tau++;
            return tau;
        }
        tau++;
    }
    return -1;
}

static float yin_parabolic_interp(const float *d, int tau, int max_lag) {
    if (tau <= 0 || tau >= max_lag - 1) return (float)tau;
    float s0 = d[tau - 1], s1 = d[tau], s2 = d[tau + 1];
    float denom = 2.0f * (2.0f * s1 - s0 - s2);
    if (fabsf(denom) < 1e-10f) return (float)tau;
    return (float)tau + (s0 - s2) / denom;
}

/* HIGH-LEVEL: YIN Pitch Detection */
cdp_pitch_data* cdp_lib_pitch(cdp_lib_ctx* ctx, const cdp_lib_buffer* input,
                               double min_freq, double max_freq, int frame_size, int hop_size) {
    if (ctx == NULL || input == NULL) return NULL;
    if (min_freq <= 0) min_freq = 50.0;
    if (max_freq <= 0) max_freq = 2000.0;
    if (frame_size <= 0) frame_size = 2048;
    if (hop_size <= 0) hop_size = 512;

    cdp_lib_buffer *mono = to_mono_if_needed(ctx, input);
    if (mono == NULL) return NULL;

    int num_frames = (int)((mono->length - frame_size) / hop_size) + 1;
    if (num_frames <= 0) { cdp_lib_buffer_free(mono); return NULL; }

    cdp_pitch_data *result = (cdp_pitch_data *)calloc(1, sizeof(cdp_pitch_data));
    if (!result) { cdp_lib_buffer_free(mono); return NULL; }

    result->pitch = (float *)calloc(num_frames, sizeof(float));
    result->confidence = (float *)calloc(num_frames, sizeof(float));
    if (!result->pitch || !result->confidence) {
        cdp_lib_buffer_free(mono); cdp_pitch_data_free(result); return NULL;
    }

    result->num_frames = num_frames;
    result->frame_time = (float)hop_size / input->sample_rate;
    result->sample_rate = (float)input->sample_rate;

    int min_lag = (int)(input->sample_rate / max_freq);
    int max_lag = (int)(input->sample_rate / min_freq);
    if (max_lag > frame_size / 2) max_lag = frame_size / 2;
    if (min_lag < 2) min_lag = 2;

    float *frame = (float *)malloc(frame_size * sizeof(float));
    float *d = (float *)malloc(max_lag * sizeof(float));
    if (!frame || !d) {
        cdp_lib_buffer_free(mono); free(frame); free(d);
        cdp_pitch_data_free(result); return NULL;
    }

    for (int f = 0; f < num_frames; f++) {
        memcpy(frame, mono->data + f * hop_size, frame_size * sizeof(float));
        yin_difference(frame, frame_size, d, max_lag);
        yin_cumulative_mean(d, max_lag);
        int tau = yin_find_minimum(d, max_lag, 0.15f);
        if (tau >= min_lag && tau <= max_lag) {
            float refined = yin_parabolic_interp(d, tau, max_lag);
            result->pitch[f] = (float)input->sample_rate / refined;
            result->confidence[f] = fmaxf(0, fminf(1, 1.0f - d[tau]));
        }
    }

    cdp_lib_buffer_free(mono); free(frame); free(d);
    return result;
}

/* LPC helpers */
static int lpc_levinson_durbin(const float *r, int order, float *a) {
    float e = r[0];
    if (e <= 0) return -1;
    float *a_prev = (float *)calloc(order, sizeof(float));
    if (!a_prev) return -1;

    for (int i = 0; i < order; i++) {
        float lambda = r[i + 1];
        for (int j = 0; j < i; j++) lambda -= a_prev[j] * r[i - j];
        lambda /= e;
        if (fabsf(lambda) >= 1.0f) { free(a_prev); return -1; }
        a[i] = lambda;
        for (int j = 0; j < i; j++) a[j] = a_prev[j] - lambda * a_prev[i - 1 - j];
        memcpy(a_prev, a, (i + 1) * sizeof(float));
        e *= (1.0f - lambda * lambda);
        if (e <= 0) { free(a_prev); return -1; }
    }
    free(a_prev);
    return 0;
}

static int lpc_find_formants(const float *a, int order, float sr, float *f, float *bw, int max) {
    float *spec = (float *)malloc(512 * sizeof(float));
    if (!spec) return 0;

    for (int i = 0; i < 512; i++) {
        float w = 2.0f * (float)M_PI * i / 512 * sr / 2 / sr;
        float re = 1.0f, im = 0.0f;
        for (int k = 0; k < order; k++) {
            re -= a[k] * cosf((k + 1) * w);
            im += a[k] * sinf((k + 1) * w);
        }
        float mag_sq = re * re + im * im;
        spec[i] = (mag_sq > 1e-10f) ? 1.0f / mag_sq : 0;
    }

    float thresh = 0;
    for (int i = 0; i < 512; i++) if (spec[i] > thresh) thresh = spec[i];
    thresh *= 0.1f;

    int n = 0;
    for (int i = 2; i < 510 && n < max; i++) {
        if (spec[i] > spec[i-1] && spec[i] > spec[i-2] &&
            spec[i] > spec[i+1] && spec[i] > spec[i+2] && spec[i] > thresh) {
            float freq = (float)i / 512 * sr / 2;
            if (freq > 100 && freq < sr/2 - 100) {
                f[n] = freq; bw[n] = sr / 512 * 2; n++;
            }
        }
    }
    free(spec);
    return n;
}

/* HIGH-LEVEL: LPC Formant Analysis */
cdp_formant_data* cdp_lib_formants(cdp_lib_ctx* ctx, const cdp_lib_buffer* input,
                                    int lpc_order, int frame_size, int hop_size) {
    if (ctx == NULL || input == NULL) return NULL;
    if (lpc_order <= 0) lpc_order = 12;
    if (frame_size <= 0) frame_size = 1024;
    if (hop_size <= 0) hop_size = 256;

    cdp_lib_buffer *mono = to_mono_if_needed(ctx, input);
    if (!mono) return NULL;

    int num_frames = (int)((mono->length - frame_size) / hop_size) + 1;
    if (num_frames <= 0) { cdp_lib_buffer_free(mono); return NULL; }

    cdp_formant_data *result = (cdp_formant_data *)calloc(1, sizeof(cdp_formant_data));
    if (!result) { cdp_lib_buffer_free(mono); return NULL; }

    result->f1 = (float *)calloc(num_frames, sizeof(float));
    result->f2 = (float *)calloc(num_frames, sizeof(float));
    result->f3 = (float *)calloc(num_frames, sizeof(float));
    result->f4 = (float *)calloc(num_frames, sizeof(float));
    result->b1 = (float *)calloc(num_frames, sizeof(float));
    result->b2 = (float *)calloc(num_frames, sizeof(float));
    result->b3 = (float *)calloc(num_frames, sizeof(float));
    result->b4 = (float *)calloc(num_frames, sizeof(float));
    result->num_frames = num_frames;
    result->frame_time = (float)hop_size / input->sample_rate;
    result->sample_rate = (float)input->sample_rate;

    float *frame = (float *)malloc(frame_size * sizeof(float));
    float *r = (float *)malloc((lpc_order + 1) * sizeof(float));
    float *a = (float *)calloc(lpc_order, sizeof(float));
    float formants[4], bandwidths[4];

    for (int f = 0; f < num_frames; f++) {
        size_t start = f * hop_size;
        frame[0] = mono->data[start];
        for (int i = 1; i < frame_size && start + i < mono->length; i++)
            frame[i] = mono->data[start + i] - 0.97f * mono->data[start + i - 1];
        analysis_apply_hamming(frame, frame_size);
        analysis_autocorrelation(frame, frame_size, r, lpc_order + 1);
        memset(a, 0, lpc_order * sizeof(float));
        if (lpc_levinson_durbin(r, lpc_order, a) == 0) {
            int nf = lpc_find_formants(a, lpc_order, input->sample_rate, formants, bandwidths, 4);
            result->f1[f] = nf > 0 ? formants[0] : 0;
            result->f2[f] = nf > 1 ? formants[1] : 0;
            result->f3[f] = nf > 2 ? formants[2] : 0;
            result->f4[f] = nf > 3 ? formants[3] : 0;
            result->b1[f] = nf > 0 ? bandwidths[0] : 0;
            result->b2[f] = nf > 1 ? bandwidths[1] : 0;
            result->b3[f] = nf > 2 ? bandwidths[2] : 0;
            result->b4[f] = nf > 3 ? bandwidths[3] : 0;
        }
    }

    cdp_lib_buffer_free(mono); free(frame); free(r); free(a);
    return result;
}

/* Partial tracking types */
typedef struct { float freq; float amp; int bin; int track_id; } analysis_peak;
typedef struct { int id; int start_frame; int last_frame; float last_freq; int active; } analysis_track;

static int analysis_find_peaks(const float *amp, const float *freq, int num_bins,
                                float min_amp, analysis_peak *peaks, int max_peaks) {
    int n = 0;
    for (int b = 1; b < num_bins - 1 && n < max_peaks; b++) {
        if (amp[b] > amp[b-1] && amp[b] > amp[b+1] && amp[b] > min_amp) {
            peaks[n].freq = freq[b]; peaks[n].amp = amp[b];
            peaks[n].bin = b; peaks[n].track_id = -1; n++;
        }
    }
    return n;
}

/* HIGH-LEVEL: Partial Tracking */
cdp_partial_data* cdp_lib_get_partials(cdp_lib_ctx* ctx, const cdp_lib_buffer* input,
                                        double min_amp_db, int max_partials,
                                        double freq_tolerance, int fft_size, int hop_size) {
    if (ctx == NULL || input == NULL) return NULL;
    if (min_amp_db >= 0) min_amp_db = -60.0;
    if (max_partials <= 0) max_partials = 100;
    if (freq_tolerance <= 0) freq_tolerance = 50.0;
    if (fft_size <= 0) fft_size = 2048;
    if (hop_size <= 0) hop_size = 512;

    float min_amp = (float)pow(10.0, min_amp_db / 20.0);
    int overlap = fft_size / hop_size;

    cdp_spectral_data *spectral = cdp_spectral_analyze(input->data, input->length,
                                                        input->channels, input->sample_rate,
                                                        fft_size, overlap);
    if (!spectral) { snprintf(ctx->error_msg, sizeof(ctx->error_msg), "Spectral analysis failed"); return NULL; }

    int num_frames = spectral->num_frames;
    int num_bins = spectral->num_bins;

    analysis_peak *peaks = (analysis_peak *)malloc(max_partials * sizeof(analysis_peak));
    int max_tracks = max_partials * 10;
    analysis_track *tracks = (analysis_track *)calloc(max_tracks, sizeof(analysis_track));
    int num_tracks = 0, next_track_id = 0;

    typedef struct { int id; int frame; float freq; float amp; } point_t;
    int max_points = num_frames * max_partials;
    point_t *points = (point_t *)malloc(max_points * sizeof(point_t));
    int num_points = 0;

    if (!peaks || !tracks || !points) {
        cdp_spectral_data_free(spectral); free(peaks); free(tracks); free(points);
        return NULL;
    }

    for (int f = 0; f < num_frames; f++) {
        float *amp = spectral->frames[f].data;
        float *freq = spectral->frames[f].data + num_bins;
        int np = analysis_find_peaks(amp, freq, num_bins, min_amp, peaks, max_partials);

        for (int p = 0; p < np; p++) {
            float best_dist = (float)freq_tolerance;
            int best_track = -1;
            for (int t = 0; t < num_tracks; t++) {
                if (tracks[t].active && tracks[t].last_frame == f - 1) {
                    float dist = fabsf(peaks[p].freq - tracks[t].last_freq);
                    if (dist < best_dist) { best_dist = dist; best_track = t; }
                }
            }
            if (best_track >= 0) {
                peaks[p].track_id = tracks[best_track].id;
                tracks[best_track].last_frame = f;
                tracks[best_track].last_freq = peaks[p].freq;
            } else if (num_tracks < max_tracks) {
                peaks[p].track_id = next_track_id++;
                tracks[num_tracks].id = peaks[p].track_id;
                tracks[num_tracks].start_frame = f;
                tracks[num_tracks].last_frame = f;
                tracks[num_tracks].last_freq = peaks[p].freq;
                tracks[num_tracks].active = 1;
                num_tracks++;
            }
            if (peaks[p].track_id >= 0 && num_points < max_points) {
                points[num_points].id = peaks[p].track_id;
                points[num_points].frame = f;
                points[num_points].freq = peaks[p].freq;
                points[num_points].amp = peaks[p].amp;
                num_points++;
            }
        }
        for (int t = 0; t < num_tracks; t++)
            if (tracks[t].active && tracks[t].last_frame < f) tracks[t].active = 0;
    }

    cdp_spectral_data_free(spectral); free(peaks);

    /* Build result from tracks with >= 3 frames */
    int *ts = (int *)calloc(next_track_id, sizeof(int));
    int *te = (int *)calloc(next_track_id, sizeof(int));
    int *tc = (int *)calloc(next_track_id, sizeof(int));
    for (int i = 0; i < next_track_id; i++) { ts[i] = num_frames; te[i] = -1; }
    for (int i = 0; i < num_points; i++) {
        int tid = points[i].id;
        if (tid >= 0 && tid < next_track_id) {
            if (points[i].frame < ts[tid]) ts[tid] = points[i].frame;
            if (points[i].frame > te[tid]) te[tid] = points[i].frame;
            tc[tid]++;
        }
    }

    int valid = 0;
    for (int i = 0; i < next_track_id; i++) if (tc[i] >= 3) valid++;
    if (valid > max_partials) valid = max_partials;

    cdp_partial_data *result = (cdp_partial_data *)calloc(1, sizeof(cdp_partial_data));
    result->tracks = (cdp_partial_track *)calloc(valid, sizeof(cdp_partial_track));
    result->num_tracks = valid;
    result->total_frames = num_frames;
    result->frame_time = (float)hop_size / input->sample_rate;
    result->sample_rate = (float)input->sample_rate;
    result->fft_size = fft_size;

    int ri = 0;
    for (int tid = 0; tid < next_track_id && ri < valid; tid++) {
        if (tc[tid] < 3) continue;
        int len = te[tid] - ts[tid] + 1;
        result->tracks[ri].freq = (float *)calloc(len, sizeof(float));
        result->tracks[ri].amp = (float *)calloc(len, sizeof(float));
        result->tracks[ri].start_frame = ts[tid];
        result->tracks[ri].end_frame = te[tid];
        result->tracks[ri].num_frames = len;
        for (int i = 0; i < num_points; i++) {
            if (points[i].id == tid) {
                int l = points[i].frame - ts[tid];
                if (l >= 0 && l < len) {
                    result->tracks[ri].freq[l] = points[i].freq;
                    result->tracks[ri].amp[l] = points[i].amp;
                }
            }
        }
        ri++;
    }

    free(tracks); free(points); free(ts); free(te); free(tc);
    return result;
}

/* LOW-LEVEL: CDP-style pitch from spectrum */
static int find_loudest_peaks_sorted(const float *amp, const float *freq, int num_bins,
                                      float min_freq, float *pf, float *pa, int n) {
    int cnt = 0;
    for (int b = 1; b < num_bins - 1; b++) {
        if (amp[b] > amp[b-1] && amp[b] > amp[b+1] && freq[b] >= min_freq) {
            int pos = cnt;
            while (pos > 0 && amp[b] > pa[pos - 1]) {
                if (pos < n) { pf[pos] = pf[pos-1]; pa[pos] = pa[pos-1]; }
                pos--;
            }
            if (pos < n) { pf[pos] = freq[b]; pa[pos] = amp[b]; if (cnt < n) cnt++; }
        }
    }
    return cnt;
}

static float find_fundamental_harmonic(const float *pf, int np, float minf, float maxf) {
    if (np < 2) return 0;
    float lo = pf[0] < pf[1] ? pf[0] : pf[1];
    float hi = pf[0] > pf[1] ? pf[0] : pf[1];
    for (int n = 1; n <= 8; n++) {
        for (int m = n + 1; m <= 16; m++) {
            float exp = lo * (float)m / (float)n;
            if (fabsf(exp - hi) / hi < 0.05f) {
                float f0 = lo / (float)n;
                if (f0 >= minf && f0 <= maxf) return f0;
            }
        }
    }
    float low = pf[0];
    for (int i = 1; i < np; i++) if (pf[i] < low) low = pf[i];
    return (low >= minf && low <= maxf) ? low : 0;
}

cdp_pitch_data* cdp_lib_pitch_from_spectrum(cdp_lib_ctx* ctx, const cdp_spectral_data* spectral,
                                             double min_freq, double max_freq, int num_peaks) {
    if (!ctx || !spectral) return NULL;
    if (min_freq <= 0) min_freq = 50.0;
    if (max_freq <= 0) max_freq = 2000.0;
    if (num_peaks <= 0) num_peaks = 8;
    if (num_peaks > 16) num_peaks = 16;

    cdp_pitch_data *result = (cdp_pitch_data *)calloc(1, sizeof(cdp_pitch_data));
    result->pitch = (float *)calloc(spectral->num_frames, sizeof(float));
    result->confidence = (float *)calloc(spectral->num_frames, sizeof(float));
    result->num_frames = spectral->num_frames;
    result->frame_time = spectral->frame_time;
    result->sample_rate = spectral->sample_rate;

    float *pf = (float *)malloc(num_peaks * sizeof(float));
    float *pa = (float *)malloc(num_peaks * sizeof(float));

    for (int f = 0; f < spectral->num_frames; f++) {
        float *amp = spectral->frames[f].data;
        float *freq = spectral->frames[f].data + spectral->num_bins;
        int np = find_loudest_peaks_sorted(amp, freq, spectral->num_bins, (float)min_freq, pf, pa, num_peaks);
        float f0 = find_fundamental_harmonic(pf, np, (float)min_freq, (float)max_freq);
        result->pitch[f] = f0;
        result->confidence[f] = (f0 > 0 && np > 0) ? (pa[0] > 0.01f ? 1.0f : pa[0] / 0.01f) : 0;
    }

    free(pf); free(pa);
    return result;
}

/* LOW-LEVEL: CDP-style formants from spectrum */
cdp_formant_data* cdp_lib_formants_from_spectrum(cdp_lib_ctx* ctx, const cdp_spectral_data* spectral,
                                                  int bands_per_octave) {
    if (!ctx || !spectral) return NULL;
    if (bands_per_octave <= 0) bands_per_octave = 6;
    if (bands_per_octave > 12) bands_per_octave = 12;

    cdp_formant_data *result = (cdp_formant_data *)calloc(1, sizeof(cdp_formant_data));
    result->f1 = (float *)calloc(spectral->num_frames, sizeof(float));
    result->f2 = (float *)calloc(spectral->num_frames, sizeof(float));
    result->f3 = (float *)calloc(spectral->num_frames, sizeof(float));
    result->f4 = (float *)calloc(spectral->num_frames, sizeof(float));
    result->b1 = (float *)calloc(spectral->num_frames, sizeof(float));
    result->b2 = (float *)calloc(spectral->num_frames, sizeof(float));
    result->b3 = (float *)calloc(spectral->num_frames, sizeof(float));
    result->b4 = (float *)calloc(spectral->num_frames, sizeof(float));
    result->num_frames = spectral->num_frames;
    result->frame_time = spectral->frame_time;
    result->sample_rate = spectral->sample_rate;

    float fpb = spectral->sample_rate / (2.0f * (spectral->num_bins - 1));

    for (int f = 0; f < spectral->num_frames; f++) {
        float *amp = spectral->frames[f].data;
        float form[4] = {0}, bw[4] = {0};
        int nf = 0;
        for (int b = 2; b < spectral->num_bins - 2 && nf < 4; b++) {
            if (amp[b] > amp[b-1] && amp[b] > amp[b-2] && amp[b] > amp[b+1] && amp[b] > amp[b+2]) {
                float freq = b * fpb;
                if (freq > 100 && freq < spectral->sample_rate/2 - 100) {
                    form[nf] = freq;
                    int w = 1;
                    while (b-w > 0 && b+w < spectral->num_bins && amp[b-w] > amp[b]*0.5f && amp[b+w] > amp[b]*0.5f) w++;
                    bw[nf] = 2 * w * fpb;
                    nf++;
                }
            }
        }
        result->f1[f] = form[0]; result->f2[f] = form[1]; result->f3[f] = form[2]; result->f4[f] = form[3];
        result->b1[f] = bw[0]; result->b2[f] = bw[1]; result->b3[f] = bw[2]; result->b4[f] = bw[3];
    }
    return result;
}

/* LOW-LEVEL: CDP-style partials from spectrum given pitch */
cdp_partial_data* cdp_lib_partials_from_spectrum(cdp_lib_ctx* ctx, const cdp_spectral_data* spectral,
                                                  const cdp_pitch_data* pitch, int max_harmonics,
                                                  double amp_threshold) {
    if (!ctx || !spectral || !pitch) return NULL;
    if (max_harmonics <= 0) max_harmonics = 32;
    if (amp_threshold >= 0) amp_threshold = -60.0;

    float min_amp = (float)pow(10.0, amp_threshold / 20.0);
    int num_frames = spectral->num_frames < pitch->num_frames ? spectral->num_frames : pitch->num_frames;
    float fpb = spectral->sample_rate / (2.0f * (spectral->num_bins - 1));

    cdp_partial_data *result = (cdp_partial_data *)calloc(1, sizeof(cdp_partial_data));
    result->tracks = (cdp_partial_track *)calloc(max_harmonics, sizeof(cdp_partial_track));
    result->num_tracks = max_harmonics;
    result->total_frames = num_frames;
    result->frame_time = spectral->frame_time;
    result->sample_rate = spectral->sample_rate;
    result->fft_size = spectral->fft_size;

    for (int h = 0; h < max_harmonics; h++) {
        result->tracks[h].freq = (float *)calloc(num_frames, sizeof(float));
        result->tracks[h].amp = (float *)calloc(num_frames, sizeof(float));
        result->tracks[h].start_frame = 0;
        result->tracks[h].end_frame = num_frames - 1;
        result->tracks[h].num_frames = num_frames;
    }

    for (int f = 0; f < num_frames; f++) {
        float f0 = pitch->pitch[f];
        if (f0 <= 0) continue;
        float *amp = spectral->frames[f].data;
        float *freq = spectral->frames[f].data + spectral->num_bins;

        for (int h = 0; h < max_harmonics; h++) {
            float tf = f0 * (h + 1);
            if (tf >= spectral->sample_rate / 2) break;
            int tb = (int)(tf / fpb + 0.5f);
            if (tb < 1 || tb >= spectral->num_bins - 1) continue;

            int bb = tb; float ba = amp[tb];
            for (int d = -3; d <= 3; d++) {
                int b = tb + d;
                if (b >= 1 && b < spectral->num_bins - 1 && amp[b] > ba) { ba = amp[b]; bb = b; }
            }
            if (ba >= min_amp) {
                result->tracks[h].freq[f] = freq[bb];
                result->tracks[h].amp[f] = ba;
            }
        }
    }
    return result;
}
