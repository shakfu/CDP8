/*
 * CDP Library - Utility Operations
 * Copyright (c) 2024 Composers Desktop Project Ltd
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "cdp.h"
#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Internal helper to set error on context */
extern void cdp_set_error(cdp_context* ctx, cdp_error err, const char* msg);

/*
 * Reverse audio buffer.
 */
cdp_buffer* cdp_reverse(cdp_context* ctx, const cdp_buffer* buf)
{
    if (!buf) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer");
        return NULL;
    }

    size_t frame_count = buf->info.frame_count;
    int channels = buf->info.channels;

    cdp_buffer* out = cdp_buffer_create(frame_count, channels, buf->info.sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY, "Failed to allocate buffer");
        return NULL;
    }

    /* Reverse frame by frame (preserving channel order within each frame) */
    for (size_t i = 0; i < frame_count; i++) {
        size_t src_frame = frame_count - 1 - i;
        for (int ch = 0; ch < channels; ch++) {
            out->samples[i * channels + ch] = buf->samples[src_frame * channels + ch];
        }
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}

/*
 * Apply fade in to buffer (in-place).
 *
 * fade_type: 0 = linear, 1 = exponential (equal power)
 */
cdp_error cdp_fade_in(cdp_context* ctx, cdp_buffer* buf, double duration, int fade_type)
{
    if (!buf) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer");
        return CDP_ERROR_INVALID_ARG;
    }

    if (duration <= 0.0) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "Duration must be positive");
        return CDP_ERROR_INVALID_ARG;
    }

    size_t fade_frames = (size_t)(duration * buf->info.sample_rate);
    if (fade_frames > buf->info.frame_count) {
        fade_frames = buf->info.frame_count;
    }

    int channels = buf->info.channels;

    for (size_t i = 0; i < fade_frames; i++) {
        double t = (double)i / (double)fade_frames;  /* 0 to 1 */
        double gain;

        if (fade_type == 1) {
            /* Exponential (equal power): sin curve */
            gain = sin(t * M_PI / 2.0);
        } else {
            /* Linear */
            gain = t;
        }

        for (int ch = 0; ch < channels; ch++) {
            buf->samples[i * channels + ch] *= (float)gain;
        }
    }

    if (ctx) cdp_clear_error(ctx);
    return CDP_OK;
}

/*
 * Apply fade out to buffer (in-place).
 *
 * fade_type: 0 = linear, 1 = exponential (equal power)
 */
cdp_error cdp_fade_out(cdp_context* ctx, cdp_buffer* buf, double duration, int fade_type)
{
    if (!buf) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer");
        return CDP_ERROR_INVALID_ARG;
    }

    if (duration <= 0.0) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "Duration must be positive");
        return CDP_ERROR_INVALID_ARG;
    }

    size_t fade_frames = (size_t)(duration * buf->info.sample_rate);
    if (fade_frames > buf->info.frame_count) {
        fade_frames = buf->info.frame_count;
    }

    int channels = buf->info.channels;
    size_t start_frame = buf->info.frame_count - fade_frames;

    for (size_t i = 0; i < fade_frames; i++) {
        double t = (double)i / (double)fade_frames;  /* 0 to 1 */
        double gain;

        if (fade_type == 1) {
            /* Exponential (equal power): cos curve */
            gain = cos(t * M_PI / 2.0);
        } else {
            /* Linear */
            gain = 1.0 - t;
        }

        size_t frame = start_frame + i;
        for (int ch = 0; ch < channels; ch++) {
            buf->samples[frame * channels + ch] *= (float)gain;
        }
    }

    if (ctx) cdp_clear_error(ctx);
    return CDP_OK;
}

/*
 * Concatenate multiple buffers into one.
 * All buffers must have same channel count and sample rate.
 */
cdp_buffer* cdp_concat(cdp_context* ctx, cdp_buffer** buffers, int count)
{
    if (!buffers || count < 1) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "Invalid arguments");
        return NULL;
    }

    /* Verify all buffers are compatible and calculate total length */
    int channels = buffers[0]->info.channels;
    int sample_rate = buffers[0]->info.sample_rate;
    size_t total_frames = 0;

    for (int i = 0; i < count; i++) {
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
        total_frames += buffers[i]->info.frame_count;
    }

    cdp_buffer* out = cdp_buffer_create(total_frames, channels, sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY, "Failed to allocate buffer");
        return NULL;
    }

    /* Copy each buffer in sequence */
    size_t offset = 0;
    for (int i = 0; i < count; i++) {
        size_t samples = buffers[i]->sample_count;
        memcpy(out->samples + offset, buffers[i]->samples, samples * sizeof(float));
        offset += samples;
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}
