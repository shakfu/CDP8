/*
 * CDP Library - Main Public API
 * Copyright (c) 2024 Composers Desktop Project Ltd
 * SPDX-License-Identifier: LGPL-2.1-or-later
 *
 * A library for audio processing based on the Composers Desktop Project.
 */

#ifndef CDP_H
#define CDP_H

#include "cdp_error.h"
#include "cdp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Library version */
#define CDP_VERSION_MAJOR 0
#define CDP_VERSION_MINOR 1
#define CDP_VERSION_PATCH 0

/**
 * Get library version string.
 */
const char* cdp_version(void);

/*============================================================================
 * Context Management
 *============================================================================*/

/**
 * Create a new CDP context.
 *
 * A context holds all state for a processing operation and must be
 * created before calling any other CDP functions.
 *
 * @return New context, or NULL on allocation failure.
 */
cdp_context* cdp_context_create(void);

/**
 * Destroy a CDP context and free all associated resources.
 *
 * @param ctx Context to destroy (may be NULL).
 */
void cdp_context_destroy(cdp_context* ctx);

/**
 * Get the last error code from a context.
 *
 * @param ctx Context to query.
 * @return Last error code, or CDP_OK if no error.
 */
cdp_error cdp_get_error(const cdp_context* ctx);

/**
 * Get the last error message from a context.
 *
 * @param ctx Context to query.
 * @return Error message string (valid until next CDP call on this context).
 */
const char* cdp_get_error_message(const cdp_context* ctx);

/**
 * Clear any error state in a context.
 *
 * @param ctx Context to clear.
 */
void cdp_clear_error(cdp_context* ctx);

/*============================================================================
 * Buffer Management
 *============================================================================*/

/**
 * Create a new audio buffer.
 *
 * @param frame_count Number of sample frames.
 * @param channels Number of channels.
 * @param sample_rate Sample rate in Hz.
 * @return New buffer, or NULL on allocation failure.
 */
cdp_buffer* cdp_buffer_create(size_t frame_count, int channels, int sample_rate);

/**
 * Create a buffer wrapping existing sample data (no copy).
 *
 * The caller retains ownership of the sample data and must ensure
 * it remains valid for the lifetime of the buffer.
 *
 * @param samples Pointer to sample data.
 * @param sample_count Total number of samples.
 * @param channels Number of channels.
 * @param sample_rate Sample rate in Hz.
 * @return New buffer, or NULL on failure.
 */
cdp_buffer* cdp_buffer_wrap(float* samples, size_t sample_count,
                            int channels, int sample_rate);

/**
 * Create a buffer by copying existing sample data.
 *
 * @param samples Pointer to sample data.
 * @param sample_count Total number of samples.
 * @param channels Number of channels.
 * @param sample_rate Sample rate in Hz.
 * @return New buffer, or NULL on allocation failure.
 */
cdp_buffer* cdp_buffer_copy(const float* samples, size_t sample_count,
                            int channels, int sample_rate);

/**
 * Destroy a buffer and free its resources.
 *
 * @param buf Buffer to destroy (may be NULL).
 */
void cdp_buffer_destroy(cdp_buffer* buf);

/**
 * Resize a buffer, preserving existing samples.
 *
 * @param buf Buffer to resize.
 * @param new_frame_count New frame count.
 * @return CDP_OK on success, error code on failure.
 */
cdp_error cdp_buffer_resize(cdp_buffer* buf, size_t new_frame_count);

/**
 * Clear a buffer (set all samples to zero).
 *
 * @param buf Buffer to clear.
 */
void cdp_buffer_clear(cdp_buffer* buf);

/*============================================================================
 * File I/O
 *============================================================================*/

/**
 * Read an audio file into a buffer.
 *
 * Supports WAV, AIFF, and other common formats.
 *
 * @param ctx Context for error reporting.
 * @param path Path to audio file.
 * @return New buffer containing audio data, or NULL on error.
 */
cdp_buffer* cdp_read_file(cdp_context* ctx, const char* path);

/**
 * Write a buffer to an audio file.
 *
 * Format is inferred from file extension.
 *
 * @param ctx Context for error reporting.
 * @param path Path to output file.
 * @param buf Buffer to write.
 * @return CDP_OK on success, error code on failure.
 */
cdp_error cdp_write_file(cdp_context* ctx, const char* path,
                         const cdp_buffer* buf);

/*============================================================================
 * Gain / Amplitude Operations
 *============================================================================*/

/**
 * Apply constant gain to a buffer (in-place).
 *
 * @param ctx Context for error reporting.
 * @param buf Buffer to process.
 * @param gain Linear gain factor (1.0 = unity, 2.0 = +6dB, 0.5 = -6dB).
 * @param flags Processing flags (CDP_FLAG_CLIP to clip output).
 * @return CDP_OK on success, error code on failure.
 */
cdp_error cdp_gain(cdp_context* ctx, cdp_buffer* buf, double gain, cdp_flags flags);

/**
 * Apply gain in decibels to a buffer (in-place).
 *
 * @param ctx Context for error reporting.
 * @param buf Buffer to process.
 * @param gain_db Gain in decibels (0 = unity, +6 = double, -6 = half).
 * @param flags Processing flags.
 * @return CDP_OK on success, error code on failure.
 */
cdp_error cdp_gain_db(cdp_context* ctx, cdp_buffer* buf, double gain_db,
                      cdp_flags flags);

/**
 * Apply time-varying gain envelope to a buffer (in-place).
 *
 * @param ctx Context for error reporting.
 * @param buf Buffer to process.
 * @param points Array of breakpoints (time, gain pairs).
 * @param point_count Number of breakpoints.
 * @param flags Processing flags.
 * @return CDP_OK on success, error code on failure.
 */
cdp_error cdp_gain_envelope(cdp_context* ctx, cdp_buffer* buf,
                            const cdp_breakpoint* points, size_t point_count,
                            cdp_flags flags);

/**
 * Find peak level in a buffer.
 *
 * @param ctx Context for error reporting.
 * @param buf Buffer to analyze.
 * @param peak Output: peak information (may be NULL).
 * @return Peak level (0.0 to 1.0+), or negative on error.
 */
double cdp_find_peak(cdp_context* ctx, const cdp_buffer* buf, cdp_peak* peak);

/**
 * Normalize a buffer to a target peak level (in-place).
 *
 * @param ctx Context for error reporting.
 * @param buf Buffer to process.
 * @param target_level Target peak level (0.0 to 1.0, typically 1.0 or 0.99).
 * @return CDP_OK on success, error code on failure.
 */
cdp_error cdp_normalize(cdp_context* ctx, cdp_buffer* buf, double target_level);

/**
 * Normalize a buffer to a target level in dB (in-place).
 *
 * @param ctx Context for error reporting.
 * @param buf Buffer to process.
 * @param target_db Target peak in dB (0 = 0dBFS, -3 = -3dBFS).
 * @return CDP_OK on success, error code on failure.
 */
cdp_error cdp_normalize_db(cdp_context* ctx, cdp_buffer* buf, double target_db);

/**
 * Invert phase of a buffer (in-place).
 *
 * @param ctx Context for error reporting.
 * @param buf Buffer to process.
 * @return CDP_OK on success, error code on failure.
 */
cdp_error cdp_phase_invert(cdp_context* ctx, cdp_buffer* buf);

/*============================================================================
 * Utility Functions
 *============================================================================*/

/**
 * Convert linear gain to decibels.
 *
 * @param gain Linear gain factor.
 * @return Gain in dB (returns -INFINITY for gain <= 0).
 */
double cdp_gain_to_db(double gain);

/**
 * Convert decibels to linear gain.
 *
 * @param db Gain in decibels.
 * @return Linear gain factor.
 */
double cdp_db_to_gain(double db);

#ifdef __cplusplus
}
#endif

#endif /* CDP_H */
