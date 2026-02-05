/*
 * CDP Dynamics Functions - Implementation
 *
 * Implements gate, compressor, limiter, and envelope processing.
 */

#include "cdp_dynamics.h"
#include "cdp_lib_internal.h"

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
