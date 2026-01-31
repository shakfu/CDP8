/*
 * CDP Library - Error Handling
 * Copyright (c) 2024 Composers Desktop Project Ltd
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef CDP_ERROR_H
#define CDP_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Error codes returned by CDP functions.
 *
 * Success codes are >= 0, error codes are < 0.
 */
typedef enum cdp_error {
    /* Success codes */
    CDP_OK              =  0,   /* Operation completed successfully */
    CDP_CONTINUE        =  1,   /* Continue processing (internal use) */

    /* Error codes (negative) */
    CDP_ERROR_GOAL_FAILED    = -1,   /* Operation succeeded but goal not achieved */
    CDP_ERROR_INVALID_ARG    = -2,   /* Invalid argument provided */
    CDP_ERROR_DATA           = -3,   /* Data unsuitable or incorrect */
    CDP_ERROR_MEMORY         = -4,   /* Memory allocation failed */
    CDP_ERROR_IO             = -5,   /* I/O operation failed */
    CDP_ERROR_FORMAT         = -6,   /* Unsupported or invalid format */
    CDP_ERROR_STATE          = -7,   /* Invalid state for operation */
    CDP_ERROR_NOT_FOUND      = -8,   /* Resource not found */
    CDP_ERROR_INTERNAL       = -9,   /* Internal/programming error */
} cdp_error;

/**
 * Get human-readable description of an error code.
 */
const char* cdp_error_string(cdp_error err);

#ifdef __cplusplus
}
#endif

#endif /* CDP_ERROR_H */
