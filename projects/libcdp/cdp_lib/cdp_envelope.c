/*
 * CDP Envelope Processing - Implementation
 *
 * Implements dovetail fades, tremolo, and attack modification.
 */

#include "cdp_envelope.h"
#include "cdp_lib_internal.h"

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
        cdp_lib_set_error(ctx, "Fade durations must be non-negative");
        return NULL;
    }

    /* Create output buffer (copy of input) */
    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        cdp_lib_set_error(ctx, "Failed to allocate output buffer");
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
        cdp_lib_set_error(ctx, "Tremolo frequency must be between 0 and 500 Hz");
        return NULL;
    }

    if (depth < 0 || depth > 1) {
        cdp_lib_set_error(ctx, "Tremolo depth must be between 0 and 1");
        return NULL;
    }

    /* Create output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        cdp_lib_set_error(ctx, "Failed to allocate output buffer");
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
        cdp_lib_set_error(ctx, "Attack time must be non-negative");
        return NULL;
    }

    if (attack_gain < 0) {
        cdp_lib_set_error(ctx, "Attack gain must be non-negative");
        return NULL;
    }

    /* Create output buffer (copy of input) */
    cdp_lib_buffer* output = cdp_lib_buffer_create(
        input->length, input->channels, input->sample_rate);
    if (output == NULL) {
        cdp_lib_set_error(ctx, "Failed to allocate output buffer");
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
