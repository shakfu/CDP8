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
