/*
 * CDP Distortion Processing - Implementation
 *
 * Implements overload, reverse cycles, fractal, and shuffle operations.
 */

#include "cdp_distort.h"
#include "cdp_lib_internal.h"
#include <time.h>

cdp_lib_buffer* cdp_lib_distort_overload(cdp_lib_ctx* ctx,
                                          const cdp_lib_buffer* input,
                                          double clip_level,
                                          double depth) {
    if (ctx == NULL || input == NULL) {
        return NULL;
    }

    if (clip_level <= 0 || clip_level > 1) {
        cdp_lib_set_error(ctx, "clip_level must be between 0 and 1");
        return NULL;
    }

    if (depth < 0 || depth > 1) {
        cdp_lib_set_error(ctx, "depth must be between 0 and 1");
        return NULL;
    }

    /* Convert to mono if needed */
    cdp_lib_buffer* mono = cdp_lib_to_mono(ctx, input);
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
        cdp_lib_set_error(ctx, "cycle_count must be at least 1");
        return NULL;
    }

    /* Convert to mono if needed */
    cdp_lib_buffer* mono = cdp_lib_to_mono(ctx, input);
    if (mono == NULL) return NULL;

    /*
     * Find zero crossings and reverse groups of cycles.
     */
    size_t num_samples = mono->length;

    /* Find all zero crossings */
    size_t *crossings = (size_t*)malloc(num_samples * sizeof(size_t) / 10 + 100);
    if (crossings == NULL) {
        cdp_lib_buffer_free(mono);
        cdp_lib_set_error(ctx, "Failed to allocate crossings array");
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
        cdp_lib_set_error(ctx, "scaling must be positive");
        return NULL;
    }

    if (loudness < 0 || loudness > 1) {
        cdp_lib_set_error(ctx, "loudness must be between 0 and 1");
        return NULL;
    }

    /* Convert to mono if needed */
    cdp_lib_buffer* mono = cdp_lib_to_mono(ctx, input);
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
        cdp_lib_set_error(ctx, "Failed to allocate output buffer");
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
        cdp_lib_set_error(ctx, "chunk_count must be at least 2");
        return NULL;
    }

    /* Convert to mono if needed */
    cdp_lib_buffer* mono = cdp_lib_to_mono(ctx, input);
    if (mono == NULL) return NULL;

    size_t num_samples = mono->length;
    size_t chunk_size = num_samples / chunk_count;

    if (chunk_size < 1) {
        cdp_lib_buffer_free(mono);
        cdp_lib_set_error(ctx, "Audio too short for requested chunk count");
        return NULL;
    }

    /* Create output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(num_samples, 1, mono->sample_rate);
    if (output == NULL) {
        cdp_lib_buffer_free(mono);
        cdp_lib_set_error(ctx, "Failed to allocate output buffer");
        return NULL;
    }

    /* Create chunk order array */
    int* order = (int*)malloc(chunk_count * sizeof(int));
    if (order == NULL) {
        cdp_lib_buffer_free(mono);
        cdp_lib_buffer_free(output);
        cdp_lib_set_error(ctx, "Failed to allocate order array");
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
