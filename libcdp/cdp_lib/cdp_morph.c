/*
 * CDP Morph Functions - Implementation
 *
 * Implements spectral morphing and cross-synthesis.
 */

#include "cdp_morph.h"
#include "cdp_lib_internal.h"
#include "cdp_spectral.h"

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
    cdp_lib_buffer* mono1 = cdp_lib_to_mono(ctx, input1);
    if (mono1 == NULL) return NULL;

    cdp_lib_buffer* mono2 = cdp_lib_to_mono(ctx, input2);
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

    cdp_lib_normalize_if_clipping(output, 0.95f);
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
    cdp_lib_buffer* mono1 = cdp_lib_to_mono(ctx, input1);
    if (mono1 == NULL) return NULL;

    cdp_lib_buffer* mono2 = cdp_lib_to_mono(ctx, input2);
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

    cdp_lib_normalize_if_clipping(output, 0.95f);
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
    cdp_lib_buffer* mono1 = cdp_lib_to_mono(ctx, input1);
    if (mono1 == NULL) return NULL;

    cdp_lib_buffer* mono2 = cdp_lib_to_mono(ctx, input2);
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

    cdp_lib_normalize_if_clipping(output, 0.95f);
    return output;
}
