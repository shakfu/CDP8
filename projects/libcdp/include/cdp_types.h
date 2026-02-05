/*
 * CDP Library - Type Definitions
 * Copyright (c) 2024 Composers Desktop Project Ltd
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef CDP_TYPES_H
#define CDP_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque context handle.
 * All CDP operations require a context.
 */
typedef struct cdp_context cdp_context;

/**
 * Audio format information.
 */
typedef struct cdp_audio_info {
    int sample_rate;        /* Samples per second (e.g., 44100, 48000) */
    int channels;           /* Number of channels (1=mono, 2=stereo, etc.) */
    size_t frame_count;     /* Number of sample frames */
    size_t sample_count;    /* Total samples (frame_count * channels) */
} cdp_audio_info;

/**
 * Audio buffer.
 * Samples are interleaved floats in range [-1.0, 1.0].
 * For stereo: [L0, R0, L1, R1, L2, R2, ...]
 */
typedef struct cdp_buffer {
    float* samples;         /* Sample data (interleaved) */
    size_t sample_count;    /* Number of samples in buffer */
    size_t capacity;        /* Allocated capacity (samples) */
    cdp_audio_info info;    /* Format information */
    int owns_memory;        /* Non-zero if buffer owns the sample memory */
} cdp_buffer;

/**
 * Breakpoint for time-varying parameters.
 */
typedef struct cdp_breakpoint {
    double time;            /* Time in seconds */
    double value;           /* Parameter value at this time */
} cdp_breakpoint;

/**
 * Peak information for a channel.
 */
typedef struct cdp_peak {
    float level;            /* Peak level (absolute value, 0.0 to 1.0+) */
    size_t position;        /* Sample frame position of peak */
} cdp_peak;

/**
 * Processing options/flags.
 */
typedef enum cdp_flags {
    CDP_FLAG_NONE           = 0,
    CDP_FLAG_CLIP           = (1 << 0),  /* Clip samples to [-1.0, 1.0] */
    CDP_FLAG_NORMALIZE      = (1 << 1),  /* Normalize after processing */
    CDP_FLAG_PRESERVE_PEAK  = (1 << 2),  /* Track peak information */
} cdp_flags;

#ifdef __cplusplus
}
#endif

#endif /* CDP_TYPES_H */
