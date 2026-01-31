/*
 * CDP Library - Spatial/Panning Operations
 * Copyright (c) 2024 Composers Desktop Project Ltd
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * Based on original CDP pan.c by Trevor Wishart
 */

#include "cdp.h"
#include <stdlib.h>
#include <math.h>

/* Internal helper to set error on context */
extern void cdp_set_error(cdp_context* ctx, cdp_error err, const char* msg);

#define ROOT2 1.4142135623730951

/*
 * Calculate left/right gains for a given pan position.
 * Uses CDP's geometric model for natural-sounding panning.
 *
 * Position range:
 *   -1.0 = full left
 *    0.0 = center
 *   +1.0 = full right
 *   Values beyond -1/+1 simulate sound beyond speakers (with distance falloff)
 */
static void pancalc(double position, double* leftgain, double* rightgain)
{
    double relpos = fabs(position);
    double temp, reldist;

    if (relpos <= 1.0) {
        /* Between the speakers - linear pan with distance compensation */
        temp = 1.0 + (relpos * relpos);
        reldist = ROOT2 / sqrt(temp);
        temp = (position + 1.0) / 2.0;  /* 0 to 1 range */
        *rightgain = temp * reldist;
        *leftgain = (1.0 - temp) * reldist;
    } else {
        /* Outside the speakers - inverse square falloff */
        temp = (relpos * relpos) + 1.0;
        reldist = sqrt(temp) / ROOT2;
        double invsquare = 1.0 / (reldist * reldist);

        if (position < 0.0) {
            /* Signal to left */
            *leftgain = invsquare;
            *rightgain = 0.0;
        } else {
            /* Signal to right */
            *rightgain = invsquare;
            *leftgain = 0.0;
        }
    }
}

/*
 * Pan a mono buffer to stereo with a static pan position.
 */
cdp_buffer* cdp_pan(cdp_context* ctx, const cdp_buffer* buf, double position)
{
    if (!buf) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer");
        return NULL;
    }

    if (buf->info.channels != 1) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                               "pan requires mono input");
        return NULL;
    }

    size_t frame_count = buf->info.frame_count;

    cdp_buffer* out = cdp_buffer_create(frame_count, 2, buf->info.sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY,
                               "Failed to allocate stereo buffer");
        return NULL;
    }

    double leftgain, rightgain;
    pancalc(position, &leftgain, &rightgain);

    for (size_t i = 0; i < frame_count; i++) {
        float sample = buf->samples[i];
        out->samples[i * 2] = (float)(sample * leftgain);
        out->samples[i * 2 + 1] = (float)(sample * rightgain);
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}

/*
 * Pan a mono buffer to stereo with time-varying position from breakpoints.
 */
cdp_buffer* cdp_pan_envelope(cdp_context* ctx, const cdp_buffer* buf,
                              const cdp_breakpoint* points, size_t point_count)
{
    if (!buf || !points || point_count < 1) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "Invalid arguments");
        return NULL;
    }

    if (buf->info.channels != 1) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                               "pan requires mono input");
        return NULL;
    }

    size_t frame_count = buf->info.frame_count;
    int sample_rate = buf->info.sample_rate;

    cdp_buffer* out = cdp_buffer_create(frame_count, 2, sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY,
                               "Failed to allocate stereo buffer");
        return NULL;
    }

    /* Process each sample */
    size_t bp_idx = 0;
    for (size_t i = 0; i < frame_count; i++) {
        double time = (double)i / sample_rate;

        /* Find surrounding breakpoints */
        while (bp_idx + 1 < point_count && points[bp_idx + 1].time <= time) {
            bp_idx++;
        }

        /* Interpolate position */
        double position;
        if (bp_idx + 1 >= point_count) {
            /* Past last breakpoint */
            position = points[point_count - 1].value;
        } else if (time <= points[0].time) {
            /* Before first breakpoint */
            position = points[0].value;
        } else {
            /* Linear interpolation between breakpoints */
            double t0 = points[bp_idx].time;
            double t1 = points[bp_idx + 1].time;
            double v0 = points[bp_idx].value;
            double v1 = points[bp_idx + 1].value;
            double frac = (time - t0) / (t1 - t0);
            position = v0 + frac * (v1 - v0);
        }

        double leftgain, rightgain;
        pancalc(position, &leftgain, &rightgain);

        float sample = buf->samples[i];
        out->samples[i * 2] = (float)(sample * leftgain);
        out->samples[i * 2 + 1] = (float)(sample * rightgain);
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}

/*
 * Mirror (swap) left and right channels of a stereo buffer.
 */
cdp_buffer* cdp_mirror(cdp_context* ctx, const cdp_buffer* buf)
{
    if (!buf) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer");
        return NULL;
    }

    if (buf->info.channels != 2) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                               "mirror requires stereo input");
        return NULL;
    }

    size_t frame_count = buf->info.frame_count;

    cdp_buffer* out = cdp_buffer_create(frame_count, 2, buf->info.sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY,
                               "Failed to allocate buffer");
        return NULL;
    }

    /* Swap L and R */
    for (size_t i = 0; i < frame_count; i++) {
        out->samples[i * 2] = buf->samples[i * 2 + 1];      /* New L = Old R */
        out->samples[i * 2 + 1] = buf->samples[i * 2];      /* New R = Old L */
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}

/*
 * Narrow/widen stereo image.
 *
 * width: 0.0 = mono, 1.0 = unchanged, >1.0 = wider (can cause phase issues)
 *
 * Algorithm: Cross-feed channels inversely proportional to width.
 *   narrow = (1.0 - width) / 2.0
 *   new_L = L * (1 - narrow) + R * narrow
 *   new_R = R * (1 - narrow) + L * narrow
 */
cdp_buffer* cdp_narrow(cdp_context* ctx, const cdp_buffer* buf, double width)
{
    if (!buf) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "NULL buffer");
        return NULL;
    }

    if (buf->info.channels != 2) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                               "narrow requires stereo input");
        return NULL;
    }

    if (width < 0.0) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                               "width must be >= 0");
        return NULL;
    }

    size_t frame_count = buf->info.frame_count;

    cdp_buffer* out = cdp_buffer_create(frame_count, 2, buf->info.sample_rate);
    if (!out) {
        if (ctx) cdp_set_error(ctx, CDP_ERROR_MEMORY,
                               "Failed to allocate buffer");
        return NULL;
    }

    /* Calculate cross-feed amount */
    double narrow = (1.0 - width) / 2.0;
    double keep = 1.0 - narrow;

    for (size_t i = 0; i < frame_count; i++) {
        float left = buf->samples[i * 2];
        float right = buf->samples[i * 2 + 1];

        out->samples[i * 2] = (float)(left * keep + right * narrow);
        out->samples[i * 2 + 1] = (float)(right * keep + left * narrow);
    }

    if (ctx) cdp_clear_error(ctx);
    return out;
}
