/*
 * CDP Library - Channel Operations
 * Copyright (c) 2024 Composers Desktop Project Ltd
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "cdp.h"
#include <stdlib.h>
#include <string.h>

/* Internal helper to set error on context */
extern void cdp_set_error(cdp_context* ctx, cdp_error err, const char* msg);

/*
 * Convert multi-channel buffer to mono by averaging all channels.
 */
cdp_buffer* cdp_to_mono(cdp_context* ctx, const cdp_buffer* buf)
{
    if (!buf) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer");
        return NULL;
    }

    if (buf->info.channels == 1) {
        /* Already mono - just copy */
        return cdp_buffer_copy(buf->samples, buf->sample_count,
                               1, buf->info.sample_rate);
    }

    size_t frame_count = buf->info.frame_count;
    int channels = buf->info.channels;

    cdp_buffer* out = cdp_buffer_create(frame_count, 1, buf->info.sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY, "Failed to allocate mono buffer");
        return NULL;
    }

    /* Average all channels */
    float scale = 1.0f / (float)channels;
    for (size_t i = 0; i < frame_count; i++) {
        float sum = 0.0f;
        for (int ch = 0; ch < channels; ch++) {
            sum += buf->samples[i * channels + ch];
        }
        out->samples[i] = sum * scale;
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}

/*
 * Convert mono buffer to stereo by duplicating the channel.
 */
cdp_buffer* cdp_to_stereo(cdp_context* ctx, const cdp_buffer* buf)
{
    if (!buf) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer");
        return NULL;
    }

    if (buf->info.channels != 1) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                               "to_stereo requires mono input");
        return NULL;
    }

    size_t frame_count = buf->info.frame_count;

    cdp_buffer* out = cdp_buffer_create(frame_count, 2, buf->info.sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY, "Failed to allocate stereo buffer");
        return NULL;
    }

    /* Duplicate mono to both channels */
    for (size_t i = 0; i < frame_count; i++) {
        float sample = buf->samples[i];
        out->samples[i * 2] = sample;      /* Left */
        out->samples[i * 2 + 1] = sample;  /* Right */
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}

/*
 * Extract a single channel from a multi-channel buffer.
 */
cdp_buffer* cdp_extract_channel(cdp_context* ctx, const cdp_buffer* buf, int channel)
{
    if (!buf) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer");
        return NULL;
    }

    if (channel < 0 || channel >= buf->info.channels) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                               "Channel index out of range");
        return NULL;
    }

    size_t frame_count = buf->info.frame_count;
    int channels = buf->info.channels;

    cdp_buffer* out = cdp_buffer_create(frame_count, 1, buf->info.sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY, "Failed to allocate channel buffer");
        return NULL;
    }

    /* Extract the specified channel */
    for (size_t i = 0; i < frame_count; i++) {
        out->samples[i] = buf->samples[i * channels + channel];
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}

/*
 * Merge two mono buffers into a stereo buffer.
 */
cdp_buffer* cdp_merge_channels(cdp_context* ctx, const cdp_buffer* left,
                                const cdp_buffer* right)
{
    if (!left || !right) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer");
        return NULL;
    }

    if (left->info.channels != 1 || right->info.channels != 1) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                               "merge_channels requires mono inputs");
        return NULL;
    }

    if (left->info.frame_count != right->info.frame_count) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                               "Buffers must have same length");
        return NULL;
    }

    if (left->info.sample_rate != right->info.sample_rate) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                               "Buffers must have same sample rate");
        return NULL;
    }

    size_t frame_count = left->info.frame_count;

    cdp_buffer* out = cdp_buffer_create(frame_count, 2, left->info.sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY, "Failed to allocate stereo buffer");
        return NULL;
    }

    /* Interleave left and right */
    for (size_t i = 0; i < frame_count; i++) {
        out->samples[i * 2] = left->samples[i];
        out->samples[i * 2 + 1] = right->samples[i];
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}

/*
 * Split a multi-channel buffer into separate mono buffers.
 * Returns an array of cdp_buffer pointers (one per channel).
 * Caller must free each buffer and the array itself.
 */
cdp_buffer** cdp_split_channels(cdp_context* ctx, const cdp_buffer* buf,
                                 int* out_num_channels)
{
    if (!buf) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer");
        return NULL;
    }

    int channels = buf->info.channels;
    size_t frame_count = buf->info.frame_count;

    /* Allocate array of buffer pointers */
    cdp_buffer** out = (cdp_buffer**)calloc(channels, sizeof(cdp_buffer*));
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY, "Failed to allocate buffer array");
        return NULL;
    }

    /* Create a mono buffer for each channel */
    for (int ch = 0; ch < channels; ch++) {
        out[ch] = cdp_buffer_create(frame_count, 1, buf->info.sample_rate);
        if (!out[ch]) {
            /* Cleanup on failure */
            for (int j = 0; j < ch; j++) {
                cdp_buffer_destroy(out[j]);
            }
            free(out);
            if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY, "Failed to allocate channel buffer");
            return NULL;
        }

        /* Extract channel data */
        for (size_t i = 0; i < frame_count; i++) {
            out[ch]->samples[i] = buf->samples[i * channels + ch];
        }
    }

    if (out_num_channels) {
        *out_num_channels = channels;
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}

/*
 * Interleave multiple mono buffers into a single multi-channel buffer.
 */
cdp_buffer* cdp_interleave(cdp_context* ctx, cdp_buffer** buffers, int num_channels)
{
    if (!buffers || num_channels < 1) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "Invalid arguments");
        return NULL;
    }

    /* Verify all buffers are mono and have same length/sample rate */
    size_t frame_count = buffers[0]->info.frame_count;
    int sample_rate = buffers[0]->info.sample_rate;

    for (int ch = 0; ch < num_channels; ch++) {
        if (!buffers[ch]) {
            if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer in array");
            return NULL;
        }
        if (buffers[ch]->info.channels != 1) {
            if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                                   "interleave requires mono buffers");
            return NULL;
        }
        if (buffers[ch]->info.frame_count != frame_count) {
            if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                                   "Buffers must have same length");
            return NULL;
        }
        if (buffers[ch]->info.sample_rate != sample_rate) {
            if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                                   "Buffers must have same sample rate");
            return NULL;
        }
    }

    cdp_buffer* out = cdp_buffer_create(frame_count, num_channels, sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY,
                               "Failed to allocate interleaved buffer");
        return NULL;
    }

    /* Interleave samples */
    for (size_t i = 0; i < frame_count; i++) {
        for (int ch = 0; ch < num_channels; ch++) {
            out->samples[i * num_channels + ch] = buffers[ch]->samples[i];
        }
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}
