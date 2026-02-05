/*
 * CDP Library - Error Handling Implementation
 * Copyright (c) 2024 Composers Desktop Project Ltd
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "cdp_error.h"

const char* cdp_error_string(cdp_error err)
{
    switch (err) {
    case CDP_OK:
        return "Success";
    case CDP_CONTINUE:
        return "Continue processing";
    case CDP_ERROR_GOAL_FAILED:
        return "Goal not achieved";
    case CDP_ERROR_INVALID_ARG:
        return "Invalid argument";
    case CDP_ERROR_DATA:
        return "Invalid or unsuitable data";
    case CDP_ERROR_MEMORY:
        return "Memory allocation failed";
    case CDP_ERROR_IO:
        return "I/O error";
    case CDP_ERROR_FORMAT:
        return "Unsupported format";
    case CDP_ERROR_STATE:
        return "Invalid state for operation";
    case CDP_ERROR_NOT_FOUND:
        return "Resource not found";
    case CDP_ERROR_INTERNAL:
        return "Internal error";
    default:
        return "Unknown error";
    }
}
