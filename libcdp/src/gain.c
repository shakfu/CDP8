/*
 * CDP Library - Gain/Amplitude Operations Implementation
 * Copyright (c) 2024 Composers Desktop Project Ltd
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "cdp.h"
#include <math.h>
#include <float.h>

/* Internal error setter (defined in context.c) */
extern void cdp_set_error(cdp_context* ctx, cdp_error err, const char* fmt, ...);

/* Clamp a value to [-1.0, 1.0] */
static inline float clamp_sample(float s)
{
    if (s > 1.0f) return 1.0f;
    if (s < -1.0f) return -1.0f;
    return s;
}

/*============================================================================
 * Utility Functions
 *============================================================================*/

double cdp_gain_to_db(double gain)
{
    if (gain <= 0.0) {
        return -INFINITY;
    }
    return 20.0 * log10(gain);
}

double cdp_db_to_gain(double db)
{
    return pow(10.0, db / 20.0);
}

/*============================================================================
 * Peak Finding
 *============================================================================*/

double cdp_find_peak(cdp_context* ctx, const cdp_buffer* buf, cdp_peak* peak)
{
    if (!buf || !buf->samples) {
        if (ctx) {
            cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "Invalid buffer");
        }
        return -1.0;
    }

    float max_level = 0.0f;
    size_t max_pos = 0;

    const float* samples = buf->samples;
    const size_t count = buf->sample_count;

    for (size_t i = 0; i < count; i++) {
        float level = fabsf(samples[i]);
        if (level > max_level) {
            max_level = level;
            max_pos = i / (size_t)buf->info.channels;  /* Frame position */
        }
    }

    if (peak) {
        peak->level = max_level;
        peak->position = max_pos;
    }

    return (double)max_level;
}

/*============================================================================
 * Gain Operations
 *============================================================================*/

cdp_error cdp_gain(cdp_context* ctx, cdp_buffer* buf, double gain, cdp_flags flags)
{
    if (!buf || !buf->samples) {
        if (ctx) {
            cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "Invalid buffer");
        }
        return CDP_ERROR_INVALID_ARG;
    }

    if (gain == 1.0) {
        /* Unity gain - nothing to do */
        return CDP_OK;
    }

    float* samples = buf->samples;
    const size_t count = buf->sample_count;
    const int do_clip = (flags & CDP_FLAG_CLIP) != 0;
    const float gain_f = (float)gain;

    if (do_clip) {
        for (size_t i = 0; i < count; i++) {
            samples[i] = clamp_sample(samples[i] * gain_f);
        }
    } else {
        for (size_t i = 0; i < count; i++) {
            samples[i] *= gain_f;
        }
    }

    if (ctx) {
        cdp_clear_error(ctx);
    }

    return CDP_OK;
}

cdp_error cdp_gain_db(cdp_context* ctx, cdp_buffer* buf, double gain_db,
                      cdp_flags flags)
{
    double gain = cdp_db_to_gain(gain_db);
    return cdp_gain(ctx, buf, gain, flags);
}

cdp_error cdp_gain_envelope(cdp_context* ctx, cdp_buffer* buf,
                            const cdp_breakpoint* points, size_t point_count,
                            cdp_flags flags)
{
    if (!buf || !buf->samples) {
        if (ctx) {
            cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "Invalid buffer");
        }
        return CDP_ERROR_INVALID_ARG;
    }

    if (!points || point_count < 2) {
        if (ctx) {
            cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                          "Envelope requires at least 2 breakpoints");
        }
        return CDP_ERROR_INVALID_ARG;
    }

    float* samples = buf->samples;
    const int channels = buf->info.channels;
    const int sample_rate = buf->info.sample_rate;
    const size_t frame_count = buf->info.frame_count;
    const int do_clip = (flags & CDP_FLAG_CLIP) != 0;

    /* Time increment per frame */
    const double time_incr = 1.0 / (double)sample_rate;

    size_t bp_idx = 0;  /* Current breakpoint index */
    double time = 0.0;

    for (size_t frame = 0; frame < frame_count; frame++) {
        /* Find the bracketing breakpoints for current time */
        while (bp_idx < point_count - 1 && points[bp_idx + 1].time <= time) {
            bp_idx++;
        }

        /* Interpolate gain value */
        double gain;
        if (bp_idx >= point_count - 1) {
            /* Past last breakpoint - use final value */
            gain = points[point_count - 1].value;
        } else if (time <= points[0].time) {
            /* Before first breakpoint - use first value */
            gain = points[0].value;
        } else {
            /* Linear interpolation between breakpoints */
            double t0 = points[bp_idx].time;
            double t1 = points[bp_idx + 1].time;
            double v0 = points[bp_idx].value;
            double v1 = points[bp_idx + 1].value;

            double t = (time - t0) / (t1 - t0);
            gain = v0 + t * (v1 - v0);
        }

        /* Apply gain to all channels in this frame */
        float gain_f = (float)gain;
        size_t base = frame * (size_t)channels;

        if (do_clip) {
            for (int ch = 0; ch < channels; ch++) {
                samples[base + ch] = clamp_sample(samples[base + ch] * gain_f);
            }
        } else {
            for (int ch = 0; ch < channels; ch++) {
                samples[base + ch] *= gain_f;
            }
        }

        time += time_incr;
    }

    if (ctx) {
        cdp_clear_error(ctx);
    }

    return CDP_OK;
}

/*============================================================================
 * Normalization
 *============================================================================*/

cdp_error cdp_normalize(cdp_context* ctx, cdp_buffer* buf, double target_level)
{
    if (!buf || !buf->samples) {
        if (ctx) {
            cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "Invalid buffer");
        }
        return CDP_ERROR_INVALID_ARG;
    }

    if (target_level <= 0.0) {
        if (ctx) {
            cdp_set_error(ctx, CDP_ERROR_INVALID_ARG,
                          "Target level must be positive");
        }
        return CDP_ERROR_INVALID_ARG;
    }

    /* Find current peak */
    double current_peak = cdp_find_peak(ctx, buf, NULL);
    if (current_peak < 0.0) {
        return CDP_ERROR_INVALID_ARG;  /* Error already set */
    }

    if (current_peak < FLT_EPSILON) {
        /* Silent file - cannot normalize */
        if (ctx) {
            cdp_set_error(ctx, CDP_ERROR_GOAL_FAILED,
                          "Cannot normalize silent audio");
        }
        return CDP_ERROR_GOAL_FAILED;
    }

    /* Calculate required gain */
    double gain = target_level / current_peak;

    /* Apply gain (no clipping needed if target_level <= 1.0) */
    cdp_flags flags = (target_level > 1.0) ? CDP_FLAG_CLIP : CDP_FLAG_NONE;
    return cdp_gain(ctx, buf, gain, flags);
}

cdp_error cdp_normalize_db(cdp_context* ctx, cdp_buffer* buf, double target_db)
{
    double target_level = cdp_db_to_gain(target_db);
    return cdp_normalize(ctx, buf, target_level);
}

/*============================================================================
 * Phase Inversion
 *============================================================================*/

cdp_error cdp_phase_invert(cdp_context* ctx, cdp_buffer* buf)
{
    if (!buf || !buf->samples) {
        if (ctx) {
            cdp_set_error(ctx, CDP_ERROR_INVALID_ARG, "Invalid buffer");
        }
        return CDP_ERROR_INVALID_ARG;
    }

    float* samples = buf->samples;
    const size_t count = buf->sample_count;

    for (size_t i = 0; i < count; i++) {
        samples[i] = -samples[i];
    }

    if (ctx) {
        cdp_clear_error(ctx);
    }

    return CDP_OK;
}
