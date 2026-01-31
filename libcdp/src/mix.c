/*
 * CDP Library - Mixing Operations
 * Copyright (c) 2024 Composers Desktop Project Ltd
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "cdp.h"
#include <stdlib.h>
#include <string.h>

/* Internal helper to set error on context */
extern void cdp_set_error(cdp_context* ctx, cdp_error err, const char* msg);

/*
 * Mix two buffers together with optional gains.
 */
cdp_buffer* cdp_mix2(cdp_context* ctx, const cdp_buffer* a, const cdp_buffer* b,
                     double gain_a, double gain_b)
{
    if (!a || !b) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer");
        return NULL;
    }

    if (a->info.channels != b->info.channels) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                               "Buffers must have same channel count");
        return NULL;
    }

    if (a->info.sample_rate != b->info.sample_rate) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                               "Buffers must have same sample rate");
        return NULL;
    }

    /* Output length is maximum of both inputs */
    size_t out_frames = a->info.frame_count > b->info.frame_count
                        ? a->info.frame_count : b->info.frame_count;

    cdp_buffer* out = cdp_buffer_create(out_frames, a->info.channels, a->info.sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY, "Failed to allocate output buffer");
        return NULL;
    }

    size_t min_samples = a->sample_count < b->sample_count
                         ? a->sample_count : b->sample_count;

    /* Mix overlapping region */
    for (size_t i = 0; i < min_samples; i++) {
        out->samples[i] = (float)(a->samples[i] * gain_a + b->samples[i] * gain_b);
    }

    /* Copy remainder from longer buffer */
    if (a->sample_count > b->sample_count) {
        for (size_t i = min_samples; i < a->sample_count; i++) {
            out->samples[i] = (float)(a->samples[i] * gain_a);
        }
    } else if (b->sample_count > a->sample_count) {
        for (size_t i = min_samples; i < b->sample_count; i++) {
            out->samples[i] = (float)(b->samples[i] * gain_b);
        }
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}

/*
 * Mix multiple buffers together with optional per-buffer gains.
 */
cdp_buffer* cdp_mix(cdp_context* ctx, cdp_buffer** buffers, const double* gains,
                    int count)
{
    if (!buffers || count < 1) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "Invalid arguments");
        return NULL;
    }

    if (count == 1) {
        /* Single buffer - just copy with gain */
        double g = gains ? gains[0] : 1.0;
        cdp_buffer* out = cdp_buffer_copy(buffers[0]->samples, buffers[0]->sample_count,
                                          buffers[0]->info.channels,
                                          buffers[0]->info.sample_rate);
        if (!out) {
            if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY, "Failed to allocate buffer");
            return NULL;
        }
        if (g != 1.0) {
            for (size_t i = 0; i < out->sample_count; i++) {
                out->samples[i] *= (float)g;
            }
        }
        return out;
    }

    /* Verify all buffers are compatible */
    int channels = buffers[0]->info.channels;
    int sample_rate = buffers[0]->info.sample_rate;
    size_t max_frames = buffers[0]->info.frame_count;

    for (int i = 1; i < count; i++) {
        if (!buffers[i]) {
            if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer in array");
            return NULL;
        }
        if (buffers[i]->info.channels != channels) {
            if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                                   "All buffers must have same channel count");
            return NULL;
        }
        if (buffers[i]->info.sample_rate != sample_rate) {
            if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                                   "All buffers must have same sample rate");
            return NULL;
        }
        if (buffers[i]->info.frame_count > max_frames) {
            max_frames = buffers[i]->info.frame_count;
        }
    }

    cdp_buffer* out = cdp_buffer_create(max_frames, channels, sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY, "Failed to allocate output buffer");
        return NULL;
    }

    /* Mix all buffers */
    for (int b = 0; b < count; b++) {
        double g = gains ? gains[b] : 1.0;
        size_t samples = buffers[b]->sample_count;

        for (size_t i = 0; i < samples; i++) {
            out->samples[i] += (float)(buffers[b]->samples[i] * g);
        }
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}
