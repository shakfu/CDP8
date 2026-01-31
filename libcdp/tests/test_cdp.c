/*
 * CDP Library - Unit Tests
 * Copyright (c) 2024 Composers Desktop Project Ltd
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "cdp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Simple test framework */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        return 0; \
    } \
} while(0)

#define ASSERT_EQ(a, b, msg) ASSERT((a) == (b), msg)
#define ASSERT_NE(a, b, msg) ASSERT((a) != (b), msg)
#define ASSERT_NULL(p, msg) ASSERT((p) == NULL, msg)
#define ASSERT_NOT_NULL(p, msg) ASSERT((p) != NULL, msg)

#define ASSERT_FLOAT_EQ(a, b, eps, msg) do { \
    if (fabs((a) - (b)) > (eps)) { \
        printf("  FAIL: %s (expected %f, got %f)\n", msg, (double)(b), (double)(a)); \
        return 0; \
    } \
} while(0)

#define RUN_TEST(test) do { \
    printf("Running %s...\n", #test); \
    tests_run++; \
    if (test()) { \
        printf("  PASS\n"); \
        tests_passed++; \
    } else { \
        tests_failed++; \
    } \
} while(0)

/*============================================================================
 * Context Tests
 *============================================================================*/

static int test_context_create_destroy(void)
{
    cdp_context* ctx = cdp_context_create();
    ASSERT_NOT_NULL(ctx, "Context should be created");

    ASSERT_EQ(cdp_get_error(ctx), CDP_OK, "Initial error should be CDP_OK");

    cdp_context_destroy(ctx);
    return 1;
}

static int test_context_error_handling(void)
{
    cdp_context* ctx = cdp_context_create();
    ASSERT_NOT_NULL(ctx, "Context should be created");

    /* Initially no error */
    ASSERT_EQ(cdp_get_error(ctx), CDP_OK, "Initial error should be CDP_OK");

    /* Clear should work on clean context */
    cdp_clear_error(ctx);
    ASSERT_EQ(cdp_get_error(ctx), CDP_OK, "Error should still be CDP_OK");

    cdp_context_destroy(ctx);
    return 1;
}

/*============================================================================
 * Buffer Tests
 *============================================================================*/

static int test_buffer_create(void)
{
    cdp_buffer* buf = cdp_buffer_create(1000, 2, 44100);
    ASSERT_NOT_NULL(buf, "Buffer should be created");

    ASSERT_EQ(buf->info.frame_count, 1000, "Frame count should be 1000");
    ASSERT_EQ(buf->info.channels, 2, "Channels should be 2");
    ASSERT_EQ(buf->info.sample_rate, 44100, "Sample rate should be 44100");
    ASSERT_EQ(buf->info.sample_count, 2000, "Sample count should be 2000");
    ASSERT_EQ(buf->sample_count, 2000, "Buffer sample count should be 2000");
    ASSERT_EQ(buf->owns_memory, 1, "Buffer should own memory");

    /* Samples should be zeroed */
    for (size_t i = 0; i < buf->sample_count; i++) {
        ASSERT_FLOAT_EQ(buf->samples[i], 0.0f, 1e-10f, "Samples should be zero");
    }

    cdp_buffer_destroy(buf);
    return 1;
}

static int test_buffer_wrap(void)
{
    float samples[100];
    for (int i = 0; i < 100; i++) {
        samples[i] = (float)i / 100.0f;
    }

    cdp_buffer* buf = cdp_buffer_wrap(samples, 100, 1, 44100);
    ASSERT_NOT_NULL(buf, "Wrapped buffer should be created");

    ASSERT_EQ(buf->samples, samples, "Wrapped buffer should point to original data");
    ASSERT_EQ(buf->owns_memory, 0, "Wrapped buffer should not own memory");

    /* Verify data is accessible */
    ASSERT_FLOAT_EQ(buf->samples[50], 0.5f, 1e-6f, "Sample 50 should be 0.5");

    cdp_buffer_destroy(buf);

    /* Original data should still be valid */
    ASSERT_FLOAT_EQ(samples[50], 0.5f, 1e-6f, "Original data should be unchanged");

    return 1;
}

static int test_buffer_copy(void)
{
    float samples[100];
    for (int i = 0; i < 100; i++) {
        samples[i] = (float)i / 100.0f;
    }

    cdp_buffer* buf = cdp_buffer_copy(samples, 100, 1, 44100);
    ASSERT_NOT_NULL(buf, "Copied buffer should be created");

    ASSERT_NE(buf->samples, samples, "Copied buffer should have new memory");
    ASSERT_EQ(buf->owns_memory, 1, "Copied buffer should own memory");

    /* Verify data was copied */
    ASSERT_FLOAT_EQ(buf->samples[50], 0.5f, 1e-6f, "Sample 50 should be 0.5");

    /* Modify original - copy should be unaffected */
    samples[50] = 0.0f;
    ASSERT_FLOAT_EQ(buf->samples[50], 0.5f, 1e-6f, "Copied data should be independent");

    cdp_buffer_destroy(buf);
    return 1;
}

static int test_buffer_resize(void)
{
    cdp_buffer* buf = cdp_buffer_create(100, 2, 44100);
    ASSERT_NOT_NULL(buf, "Buffer should be created");

    /* Set some data */
    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = 0.5f;
    }

    /* Resize larger */
    cdp_error err = cdp_buffer_resize(buf, 200);
    ASSERT_EQ(err, CDP_OK, "Resize should succeed");
    ASSERT_EQ(buf->info.frame_count, 200, "Frame count should be 200");
    ASSERT_EQ(buf->sample_count, 400, "Sample count should be 400");

    /* Original data should be preserved */
    ASSERT_FLOAT_EQ(buf->samples[0], 0.5f, 1e-6f, "Original data should be preserved");

    /* New samples should be zero */
    ASSERT_FLOAT_EQ(buf->samples[300], 0.0f, 1e-10f, "New samples should be zero");

    cdp_buffer_destroy(buf);
    return 1;
}

static int test_buffer_clear(void)
{
    cdp_buffer* buf = cdp_buffer_create(100, 1, 44100);
    ASSERT_NOT_NULL(buf, "Buffer should be created");

    /* Set data */
    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = 1.0f;
    }

    cdp_buffer_clear(buf);

    /* All samples should be zero */
    for (size_t i = 0; i < buf->sample_count; i++) {
        ASSERT_FLOAT_EQ(buf->samples[i], 0.0f, 1e-10f, "Samples should be zero after clear");
    }

    cdp_buffer_destroy(buf);
    return 1;
}

/*============================================================================
 * Gain Tests
 *============================================================================*/

static int test_gain_unity(void)
{
    cdp_context* ctx = cdp_context_create();
    cdp_buffer* buf = cdp_buffer_create(100, 1, 44100);

    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = 0.5f;
    }

    cdp_error err = cdp_gain(ctx, buf, 1.0, CDP_FLAG_NONE);
    ASSERT_EQ(err, CDP_OK, "Unity gain should succeed");

    /* Samples should be unchanged */
    for (size_t i = 0; i < buf->sample_count; i++) {
        ASSERT_FLOAT_EQ(buf->samples[i], 0.5f, 1e-6f, "Samples should be unchanged");
    }

    cdp_buffer_destroy(buf);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_gain_double(void)
{
    cdp_context* ctx = cdp_context_create();
    cdp_buffer* buf = cdp_buffer_create(100, 1, 44100);

    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = 0.25f;
    }

    cdp_error err = cdp_gain(ctx, buf, 2.0, CDP_FLAG_NONE);
    ASSERT_EQ(err, CDP_OK, "Gain should succeed");

    for (size_t i = 0; i < buf->sample_count; i++) {
        ASSERT_FLOAT_EQ(buf->samples[i], 0.5f, 1e-6f, "Samples should be doubled");
    }

    cdp_buffer_destroy(buf);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_gain_clipping(void)
{
    cdp_context* ctx = cdp_context_create();
    cdp_buffer* buf = cdp_buffer_create(100, 1, 44100);

    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = 0.6f;
    }

    /* Without clipping flag - should exceed 1.0 */
    cdp_error err = cdp_gain(ctx, buf, 2.0, CDP_FLAG_NONE);
    ASSERT_EQ(err, CDP_OK, "Gain should succeed");
    ASSERT_FLOAT_EQ(buf->samples[0], 1.2f, 1e-6f, "Should exceed 1.0 without clipping");

    /* Reset and try with clipping */
    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = 0.6f;
    }

    err = cdp_gain(ctx, buf, 2.0, CDP_FLAG_CLIP);
    ASSERT_EQ(err, CDP_OK, "Gain with clip should succeed");
    ASSERT_FLOAT_EQ(buf->samples[0], 1.0f, 1e-6f, "Should be clipped to 1.0");

    cdp_buffer_destroy(buf);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_gain_db(void)
{
    cdp_context* ctx = cdp_context_create();
    cdp_buffer* buf = cdp_buffer_create(100, 1, 44100);

    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = 0.5f;
    }

    /* +6dB should approximately double */
    cdp_error err = cdp_gain_db(ctx, buf, 6.0, CDP_FLAG_NONE);
    ASSERT_EQ(err, CDP_OK, "Gain dB should succeed");

    /* 6dB = 10^(6/20) = ~1.995 */
    ASSERT_FLOAT_EQ(buf->samples[0], 0.5f * 1.9953f, 0.01f, "6dB should ~double");

    cdp_buffer_destroy(buf);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_gain_envelope(void)
{
    cdp_context* ctx = cdp_context_create();

    /* 1 second of audio at 1000 Hz for easy calculation */
    cdp_buffer* buf = cdp_buffer_create(1000, 1, 1000);

    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = 1.0f;
    }

    /* Linear ramp from 0 to 1 over 1 second */
    cdp_breakpoint points[] = {
        {0.0, 0.0},
        {1.0, 1.0}
    };

    cdp_error err = cdp_gain_envelope(ctx, buf, points, 2, CDP_FLAG_NONE);
    ASSERT_EQ(err, CDP_OK, "Envelope should succeed");

    /* Check some points along the envelope */
    ASSERT_FLOAT_EQ(buf->samples[0], 0.0f, 0.01f, "Start should be ~0");
    ASSERT_FLOAT_EQ(buf->samples[500], 0.5f, 0.01f, "Middle should be ~0.5");
    ASSERT_FLOAT_EQ(buf->samples[999], 0.999f, 0.01f, "End should be ~1");

    cdp_buffer_destroy(buf);
    cdp_context_destroy(ctx);
    return 1;
}

/*============================================================================
 * Peak Finding Tests
 *============================================================================*/

static int test_find_peak(void)
{
    cdp_context* ctx = cdp_context_create();
    cdp_buffer* buf = cdp_buffer_create(100, 1, 44100);

    /* Set a known peak */
    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = 0.5f;
    }
    buf->samples[42] = 0.8f;
    buf->samples[73] = -0.9f;  /* Negative peak is higher in absolute value */

    cdp_peak peak;
    double level = cdp_find_peak(ctx, buf, &peak);

    ASSERT_FLOAT_EQ(level, 0.9f, 1e-6f, "Peak level should be 0.9");
    ASSERT_FLOAT_EQ(peak.level, 0.9f, 1e-6f, "Peak struct level should be 0.9");
    ASSERT_EQ(peak.position, 73, "Peak position should be 73");

    cdp_buffer_destroy(buf);
    cdp_context_destroy(ctx);
    return 1;
}

/*============================================================================
 * Normalization Tests
 *============================================================================*/

static int test_normalize(void)
{
    cdp_context* ctx = cdp_context_create();
    cdp_buffer* buf = cdp_buffer_create(100, 1, 44100);

    /* Set data with peak of 0.5 */
    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = 0.25f;
    }
    buf->samples[50] = 0.5f;

    cdp_error err = cdp_normalize(ctx, buf, 1.0);
    ASSERT_EQ(err, CDP_OK, "Normalize should succeed");

    /* Peak should now be 1.0, all samples doubled */
    ASSERT_FLOAT_EQ(buf->samples[50], 1.0f, 1e-6f, "Peak should be normalized to 1.0");
    ASSERT_FLOAT_EQ(buf->samples[0], 0.5f, 1e-6f, "Other samples should be scaled");

    cdp_buffer_destroy(buf);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_normalize_silent(void)
{
    cdp_context* ctx = cdp_context_create();
    cdp_buffer* buf = cdp_buffer_create(100, 1, 44100);

    /* All zeros */
    cdp_buffer_clear(buf);

    cdp_error err = cdp_normalize(ctx, buf, 1.0);
    ASSERT_EQ(err, CDP_ERROR_GOAL_FAILED, "Normalizing silence should fail");

    cdp_buffer_destroy(buf);
    cdp_context_destroy(ctx);
    return 1;
}

/*============================================================================
 * Phase Inversion Tests
 *============================================================================*/

static int test_phase_invert(void)
{
    cdp_context* ctx = cdp_context_create();
    cdp_buffer* buf = cdp_buffer_create(100, 1, 44100);

    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = 0.5f;
    }
    buf->samples[50] = -0.3f;

    cdp_error err = cdp_phase_invert(ctx, buf);
    ASSERT_EQ(err, CDP_OK, "Phase invert should succeed");

    ASSERT_FLOAT_EQ(buf->samples[0], -0.5f, 1e-6f, "Positive should become negative");
    ASSERT_FLOAT_EQ(buf->samples[50], 0.3f, 1e-6f, "Negative should become positive");

    cdp_buffer_destroy(buf);
    cdp_context_destroy(ctx);
    return 1;
}

/*============================================================================
 * Utility Function Tests
 *============================================================================*/

static int test_db_conversion(void)
{
    /* Unity gain = 0 dB */
    ASSERT_FLOAT_EQ(cdp_gain_to_db(1.0), 0.0, 1e-6, "Unity gain should be 0 dB");
    ASSERT_FLOAT_EQ(cdp_db_to_gain(0.0), 1.0, 1e-6, "0 dB should be unity gain");

    /* Double/half gain = +/-6 dB (approximately) */
    ASSERT_FLOAT_EQ(cdp_gain_to_db(2.0), 6.0206, 0.001, "Double should be ~6 dB");
    ASSERT_FLOAT_EQ(cdp_gain_to_db(0.5), -6.0206, 0.001, "Half should be ~-6 dB");

    /* Round trip */
    double gain = 1.5;
    double db = cdp_gain_to_db(gain);
    double back = cdp_db_to_gain(db);
    ASSERT_FLOAT_EQ(back, gain, 1e-10, "Round trip should preserve value");

    return 1;
}

/*============================================================================
 * Main
 *============================================================================*/

int main(void)
{
    printf("CDP Library Tests\n");
    printf("=================\n\n");

    printf("Version: %s\n\n", cdp_version());

    /* Context tests */
    RUN_TEST(test_context_create_destroy);
    RUN_TEST(test_context_error_handling);

    /* Buffer tests */
    RUN_TEST(test_buffer_create);
    RUN_TEST(test_buffer_wrap);
    RUN_TEST(test_buffer_copy);
    RUN_TEST(test_buffer_resize);
    RUN_TEST(test_buffer_clear);

    /* Gain tests */
    RUN_TEST(test_gain_unity);
    RUN_TEST(test_gain_double);
    RUN_TEST(test_gain_clipping);
    RUN_TEST(test_gain_db);
    RUN_TEST(test_gain_envelope);

    /* Peak finding tests */
    RUN_TEST(test_find_peak);

    /* Normalization tests */
    RUN_TEST(test_normalize);
    RUN_TEST(test_normalize_silent);

    /* Phase inversion tests */
    RUN_TEST(test_phase_invert);

    /* Utility tests */
    RUN_TEST(test_db_conversion);

    /* Summary */
    printf("\n=================\n");
    printf("Tests run: %d\n", tests_run);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);

    return tests_failed == 0 ? 0 : 1;
}
