/*
 * CDP Library - Context Management Implementation
 * Copyright (c) 2024 Composers Desktop Project Ltd
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "cdp.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

/* Maximum error message length */
#define CDP_ERROR_MSG_SIZE 1024

/* Internal context structure */
struct cdp_context {
    cdp_error last_error;
    char error_message[CDP_ERROR_MSG_SIZE];
};

/* Version string */
static const char version_string[] = "0.1.0";

const char* cdp_version(void)
{
    return version_string;
}

cdp_context* cdp_context_create(void)
{
    cdp_context* ctx = (cdp_context*)calloc(1, sizeof(cdp_context));
    if (ctx) {
        ctx->last_error = CDP_OK;
        ctx->error_message[0] = '\0';
    }
    return ctx;
}

void cdp_context_destroy(cdp_context* ctx)
{
    free(ctx);
}

cdp_error cdp_get_error(const cdp_context* ctx)
{
    if (!ctx) {
        return CDP_ERROR_INVALID_ARG;
    }
    return ctx->last_error;
}

const char* cdp_get_error_message(const cdp_context* ctx)
{
    if (!ctx) {
        return "Invalid context";
    }
    if (ctx->error_message[0] == '\0') {
        return cdp_error_string(ctx->last_error);
    }
    return ctx->error_message;
}

void cdp_clear_error(cdp_context* ctx)
{
    if (ctx) {
        ctx->last_error = CDP_OK;
        ctx->error_message[0] = '\0';
    }
}

/*
 * Internal function to set error state.
 * Not exposed in public API.
 */
void cdp_set_error(cdp_context* ctx, cdp_error err, const char* fmt, ...)
{
    if (!ctx) {
        return;
    }

    ctx->last_error = err;

    if (fmt) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(ctx->error_message, CDP_ERROR_MSG_SIZE, fmt, args);
        va_end(args);
    } else {
        ctx->error_message[0] = '\0';
    }
}
