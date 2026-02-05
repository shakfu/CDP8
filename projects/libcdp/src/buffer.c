/*
 * CDP Library - Buffer Management Implementation
 * Copyright (c) 2024 Composers Desktop Project Ltd
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "cdp.h"
#include <stdlib.h>
#include <string.h>

cdp_buffer* cdp_buffer_create(size_t frame_count, int channels, int sample_rate)
{
    if (channels <= 0 || sample_rate <= 0) {
        return NULL;
    }

    cdp_buffer* buf = (cdp_buffer*)calloc(1, sizeof(cdp_buffer));
    if (!buf) {
        return NULL;
    }

    size_t sample_count = frame_count * (size_t)channels;

    if (sample_count > 0) {
        buf->samples = (float*)calloc(sample_count, sizeof(float));
        if (!buf->samples) {
            free(buf);
            return NULL;
        }
    }

    buf->sample_count = sample_count;
    buf->capacity = sample_count;
    buf->owns_memory = 1;

    buf->info.sample_rate = sample_rate;
    buf->info.channels = channels;
    buf->info.frame_count = frame_count;
    buf->info.sample_count = sample_count;

    return buf;
}

cdp_buffer* cdp_buffer_wrap(float* samples, size_t sample_count,
                            int channels, int sample_rate)
{
    if (!samples || channels <= 0 || sample_rate <= 0) {
        return NULL;
    }

    cdp_buffer* buf = (cdp_buffer*)calloc(1, sizeof(cdp_buffer));
    if (!buf) {
        return NULL;
    }

    buf->samples = samples;
    buf->sample_count = sample_count;
    buf->capacity = sample_count;
    buf->owns_memory = 0;  /* Caller owns the memory */

    buf->info.sample_rate = sample_rate;
    buf->info.channels = channels;
    buf->info.frame_count = sample_count / (size_t)channels;
    buf->info.sample_count = sample_count;

    return buf;
}

cdp_buffer* cdp_buffer_copy(const float* samples, size_t sample_count,
                            int channels, int sample_rate)
{
    if (!samples || channels <= 0 || sample_rate <= 0) {
        return NULL;
    }

    size_t frame_count = sample_count / (size_t)channels;
    cdp_buffer* buf = cdp_buffer_create(frame_count, channels, sample_rate);
    if (!buf) {
        return NULL;
    }

    memcpy(buf->samples, samples, sample_count * sizeof(float));
    return buf;
}

void cdp_buffer_destroy(cdp_buffer* buf)
{
    if (!buf) {
        return;
    }

    if (buf->owns_memory && buf->samples) {
        free(buf->samples);
    }

    free(buf);
}

cdp_error cdp_buffer_resize(cdp_buffer* buf, size_t new_frame_count)
{
    if (!buf) {
        return CDP_ERROR_INVALID_ARG;
    }

    if (!buf->owns_memory) {
        /* Cannot resize a wrapped buffer */
        return CDP_ERROR_STATE;
    }

    size_t new_sample_count = new_frame_count * (size_t)buf->info.channels;

    if (new_sample_count > buf->capacity) {
        /* Need to allocate more space */
        float* new_samples = (float*)realloc(buf->samples,
                                             new_sample_count * sizeof(float));
        if (!new_samples) {
            return CDP_ERROR_MEMORY;
        }

        /* Zero new samples */
        if (new_sample_count > buf->sample_count) {
            memset(new_samples + buf->sample_count, 0,
                   (new_sample_count - buf->sample_count) * sizeof(float));
        }

        buf->samples = new_samples;
        buf->capacity = new_sample_count;
    }

    buf->sample_count = new_sample_count;
    buf->info.frame_count = new_frame_count;
    buf->info.sample_count = new_sample_count;

    return CDP_OK;
}

void cdp_buffer_clear(cdp_buffer* buf)
{
    if (buf && buf->samples && buf->sample_count > 0) {
        memset(buf->samples, 0, buf->sample_count * sizeof(float));
    }
}
