/*
 * CDP Library - Internal Header
 *
 * Private definitions shared across cdp_lib module files.
 * NOT for public use - include cdp_lib.h instead.
 */

#ifndef CDP_LIB_INTERNAL_H
#define CDP_LIB_INTERNAL_H

#include "cdp_lib.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Library context structure (private)
 */
struct cdp_lib_ctx {
    int initialized;
    char error_msg[512];
};

/*
 * Convert stereo to mono if needed.
 * Returns a new buffer (caller must free) or NULL on error.
 * If input is already mono, returns a copy.
 */
cdp_lib_buffer* cdp_lib_to_mono(cdp_lib_ctx* ctx, const cdp_lib_buffer* input);

/*
 * Set error message in context.
 */
static inline void cdp_lib_set_error(cdp_lib_ctx* ctx, const char* msg) {
    if (ctx && msg) {
        strncpy(ctx->error_msg, msg, sizeof(ctx->error_msg) - 1);
        ctx->error_msg[sizeof(ctx->error_msg) - 1] = '\0';
    }
}

#ifdef __cplusplus
}
#endif

#endif /* CDP_LIB_INTERNAL_H */
