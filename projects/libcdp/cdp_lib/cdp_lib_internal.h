/*
 * CDP Library - Internal Header
 *
 * Private definitions shared across cdp_lib module files.
 * NOT for public use - include cdp_lib.h instead.
 */

#ifndef CDP_LIB_INTERNAL_H
#define CDP_LIB_INTERNAL_H

#include "cdp_lib.h"
#include "cdp_spectral.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

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
    uint64_t prng_state;  /* xorshift64 PRNG state */
};

/*
 * Seed the context's PRNG. If seed is 0, uses current time.
 */
static inline void cdp_lib_seed(cdp_lib_ctx* ctx, uint64_t seed) {
    if (seed == 0) {
        seed = (uint64_t)time(NULL) ^ ((uint64_t)clock() << 32);
    }
    /* Ensure non-zero state (xorshift64 requires this) */
    ctx->prng_state = seed ? seed : 1;
}

/*
 * Generate random uint64 using xorshift64.
 */
static inline uint64_t cdp_lib_random_u64(cdp_lib_ctx* ctx) {
    uint64_t x = ctx->prng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    ctx->prng_state = x;
    return x;
}

/*
 * Generate random double in [0.0, 1.0).
 */
static inline double cdp_lib_random(cdp_lib_ctx* ctx) {
    return (cdp_lib_random_u64(ctx) >> 11) * (1.0 / 9007199254740992.0);
}

/*
 * Generate random double in [min, max).
 */
static inline double cdp_lib_random_range(cdp_lib_ctx* ctx, double min, double max) {
    return min + cdp_lib_random(ctx) * (max - min);
}

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

/*
 * Normalize buffer if peak exceeds threshold.
 * Scales all samples so peak equals target_peak.
 * Returns the original peak value.
 */
static inline float cdp_lib_normalize_if_clipping(cdp_lib_buffer* buf, float target_peak) {
    if (buf == NULL || buf->data == NULL || buf->length == 0) {
        return 0.0f;
    }

    float peak = 0.0f;
    for (size_t i = 0; i < buf->length; i++) {
        float abs_val = fabsf(buf->data[i]);
        if (abs_val > peak) peak = abs_val;
    }

    if (peak > 1.0f) {
        float scale = target_peak / peak;
        for (size_t i = 0; i < buf->length; i++) {
            buf->data[i] *= scale;
        }
    }

    return peak;
}

/*
 * Find peak amplitude in buffer.
 */
static inline float cdp_lib_find_peak(const cdp_lib_buffer* buf) {
    if (buf == NULL || buf->data == NULL || buf->length == 0) {
        return 0.0f;
    }

    float peak = 0.0f;
    for (size_t i = 0; i < buf->length; i++) {
        float abs_val = fabsf(buf->data[i]);
        if (abs_val > peak) peak = abs_val;
    }
    return peak;
}

/*
 * Synthesize spectral data to output buffer.
 * Frees the spectral data regardless of success/failure.
 * Returns new buffer on success, NULL on error (with error message set).
 */
static inline cdp_lib_buffer* cdp_lib_spectral_to_buffer(
        cdp_lib_ctx* ctx,
        cdp_spectral_data* spectral,
        int sample_rate) {

    if (spectral == NULL) {
        cdp_lib_set_error(ctx, "Spectral processing failed");
        return NULL;
    }

    size_t out_samples;
    float *audio = cdp_spectral_synthesize(spectral, &out_samples);
    cdp_spectral_data_free(spectral);

    if (audio == NULL) {
        cdp_lib_set_error(ctx, "Spectral synthesis failed");
        return NULL;
    }

    cdp_lib_buffer *output = cdp_lib_buffer_from_data(
        audio, out_samples, 1, sample_rate);

    if (output == NULL) {
        free(audio);
        cdp_lib_set_error(ctx, "Failed to create output buffer");
        return NULL;
    }

    return output;
}

#ifdef __cplusplus
}
#endif

#endif /* CDP_LIB_INTERNAL_H */
