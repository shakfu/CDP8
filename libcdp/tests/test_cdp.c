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
 * File I/O Tests
 *============================================================================*/

static const char* TEST_WAV_FILE = "/tmp/cdp_test.wav";
static const char* TEST_WAV_FILE_16 = "/tmp/cdp_test_16.wav";
static const char* TEST_WAV_FILE_24 = "/tmp/cdp_test_24.wav";

static int test_write_read_wav_float(void)
{
    cdp_context* ctx = cdp_context_create();

    /* Create a buffer with known data */
    cdp_buffer* buf = cdp_buffer_create(100, 2, 44100);
    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = (float)i / (float)buf->sample_count - 0.5f;
    }

    /* Write to file */
    cdp_error err = cdp_write_file(ctx, TEST_WAV_FILE, buf);
    ASSERT_EQ(err, CDP_OK, "Write should succeed");

    /* Read back */
    cdp_buffer* buf2 = cdp_read_file(ctx, TEST_WAV_FILE);
    ASSERT_NOT_NULL(buf2, "Read should succeed");

    /* Verify properties */
    ASSERT_EQ(buf2->info.channels, 2, "Channels should match");
    ASSERT_EQ(buf2->info.sample_rate, 44100, "Sample rate should match");
    ASSERT_EQ(buf2->sample_count, buf->sample_count, "Sample count should match");

    /* Verify data (float should be exact) */
    for (size_t i = 0; i < buf->sample_count; i++) {
        ASSERT_FLOAT_EQ(buf2->samples[i], buf->samples[i], 1e-6f,
                        "Samples should match");
    }

    cdp_buffer_destroy(buf);
    cdp_buffer_destroy(buf2);
    cdp_context_destroy(ctx);

    remove(TEST_WAV_FILE);
    return 1;
}

static int test_write_read_wav_pcm16(void)
{
    cdp_context* ctx = cdp_context_create();

    /* Create a buffer with known data */
    cdp_buffer* buf = cdp_buffer_create(100, 1, 48000);
    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = (float)i / (float)buf->sample_count - 0.5f;
    }

    /* Write to file as 16-bit PCM */
    cdp_error err = cdp_write_file_pcm16(ctx, TEST_WAV_FILE_16, buf);
    ASSERT_EQ(err, CDP_OK, "Write PCM16 should succeed");

    /* Read back */
    cdp_buffer* buf2 = cdp_read_file(ctx, TEST_WAV_FILE_16);
    ASSERT_NOT_NULL(buf2, "Read should succeed");

    /* Verify properties */
    ASSERT_EQ(buf2->info.channels, 1, "Channels should match");
    ASSERT_EQ(buf2->info.sample_rate, 48000, "Sample rate should match");

    /* Verify data (16-bit has limited precision: ~1/32768) */
    for (size_t i = 0; i < buf->sample_count; i++) {
        ASSERT_FLOAT_EQ(buf2->samples[i], buf->samples[i], 2.0f/32768.0f,
                        "Samples should match within 16-bit precision");
    }

    cdp_buffer_destroy(buf);
    cdp_buffer_destroy(buf2);
    cdp_context_destroy(ctx);

    remove(TEST_WAV_FILE_16);
    return 1;
}

static int test_write_read_wav_pcm24(void)
{
    cdp_context* ctx = cdp_context_create();

    /* Create a buffer with known data */
    cdp_buffer* buf = cdp_buffer_create(100, 2, 96000);
    for (size_t i = 0; i < buf->sample_count; i++) {
        buf->samples[i] = (float)i / (float)buf->sample_count - 0.5f;
    }

    /* Write to file as 24-bit PCM */
    cdp_error err = cdp_write_file_pcm24(ctx, TEST_WAV_FILE_24, buf);
    ASSERT_EQ(err, CDP_OK, "Write PCM24 should succeed");

    /* Read back */
    cdp_buffer* buf2 = cdp_read_file(ctx, TEST_WAV_FILE_24);
    ASSERT_NOT_NULL(buf2, "Read should succeed");

    /* Verify properties */
    ASSERT_EQ(buf2->info.channels, 2, "Channels should match");
    ASSERT_EQ(buf2->info.sample_rate, 96000, "Sample rate should match");

    /* Verify data (24-bit has good precision: ~1/8388608) */
    for (size_t i = 0; i < buf->sample_count; i++) {
        ASSERT_FLOAT_EQ(buf2->samples[i], buf->samples[i], 2.0f/8388608.0f,
                        "Samples should match within 24-bit precision");
    }

    cdp_buffer_destroy(buf);
    cdp_buffer_destroy(buf2);
    cdp_context_destroy(ctx);

    remove(TEST_WAV_FILE_24);
    return 1;
}

static int test_read_nonexistent_file(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* buf = cdp_read_file(ctx, "/nonexistent/path/to/file.wav");
    ASSERT_NULL(buf, "Reading nonexistent file should fail");
    ASSERT_EQ(cdp_get_error(ctx), CDP_ERROR_IO, "Error should be IO error");

    cdp_context_destroy(ctx);
    return 1;
}

/*============================================================================
 * Spatial/Pan Tests
 *============================================================================*/

static int test_pan_center(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* mono = cdp_buffer_create(100, 1, 44100);
    for (size_t i = 0; i < 100; i++) {
        mono->samples[i] = 0.8f;
    }

    /* Center pan should give equal L and R */
    cdp_buffer* stereo = cdp_pan(ctx, mono, 0.0);
    ASSERT_NOT_NULL(stereo, "Pan should succeed");
    ASSERT_EQ(stereo->info.channels, 2, "Output should be stereo");
    ASSERT_EQ(stereo->info.frame_count, 100, "Frame count should match");

    /* At center, L and R should be equal */
    for (size_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(stereo->samples[i * 2], stereo->samples[i * 2 + 1], 1e-6f,
                        "Center pan: L and R should be equal");
    }

    cdp_buffer_destroy(mono);
    cdp_buffer_destroy(stereo);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_pan_left(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* mono = cdp_buffer_create(100, 1, 44100);
    for (size_t i = 0; i < 100; i++) {
        mono->samples[i] = 1.0f;
    }

    /* Full left pan */
    cdp_buffer* stereo = cdp_pan(ctx, mono, -1.0);
    ASSERT_NOT_NULL(stereo, "Pan should succeed");

    /* Left should be louder than right */
    ASSERT(stereo->samples[0] > stereo->samples[1], "Full left: L > R");
    ASSERT_FLOAT_EQ(stereo->samples[1], 0.0f, 0.01f, "Full left: R should be ~0");

    cdp_buffer_destroy(mono);
    cdp_buffer_destroy(stereo);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_pan_right(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* mono = cdp_buffer_create(100, 1, 44100);
    for (size_t i = 0; i < 100; i++) {
        mono->samples[i] = 1.0f;
    }

    /* Full right pan */
    cdp_buffer* stereo = cdp_pan(ctx, mono, 1.0);
    ASSERT_NOT_NULL(stereo, "Pan should succeed");

    /* Right should be louder than left */
    ASSERT(stereo->samples[1] > stereo->samples[0], "Full right: R > L");
    ASSERT_FLOAT_EQ(stereo->samples[0], 0.0f, 0.01f, "Full right: L should be ~0");

    cdp_buffer_destroy(mono);
    cdp_buffer_destroy(stereo);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_pan_requires_mono(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* stereo = cdp_buffer_create(100, 2, 44100);
    cdp_buffer* result = cdp_pan(ctx, stereo, 0.0);

    ASSERT_NULL(result, "Pan on stereo should fail");
    ASSERT_EQ(cdp_get_error(ctx), CDP_ERROR_INVALID_ARG, "Should be invalid arg");

    cdp_buffer_destroy(stereo);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_mirror(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* stereo = cdp_buffer_create(100, 2, 44100);
    for (size_t i = 0; i < 100; i++) {
        stereo->samples[i * 2] = 0.3f;      /* Left */
        stereo->samples[i * 2 + 1] = 0.9f;  /* Right */
    }

    cdp_buffer* mirrored = cdp_mirror(ctx, stereo);
    ASSERT_NOT_NULL(mirrored, "Mirror should succeed");
    ASSERT_EQ(mirrored->info.channels, 2, "Output should be stereo");

    /* L and R should be swapped */
    for (size_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(mirrored->samples[i * 2], 0.9f, 1e-6f, "L should be old R");
        ASSERT_FLOAT_EQ(mirrored->samples[i * 2 + 1], 0.3f, 1e-6f, "R should be old L");
    }

    cdp_buffer_destroy(stereo);
    cdp_buffer_destroy(mirrored);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_narrow_to_mono(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* stereo = cdp_buffer_create(100, 2, 44100);
    for (size_t i = 0; i < 100; i++) {
        stereo->samples[i * 2] = 0.2f;      /* Left */
        stereo->samples[i * 2 + 1] = 0.8f;  /* Right */
    }

    /* Width 0 = mono (L and R become average) */
    cdp_buffer* narrowed = cdp_narrow(ctx, stereo, 0.0);
    ASSERT_NOT_NULL(narrowed, "Narrow should succeed");

    /* Both channels should be the average: (0.2 + 0.8) / 2 = 0.5 */
    for (size_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(narrowed->samples[i * 2], 0.5f, 1e-6f, "L should be average");
        ASSERT_FLOAT_EQ(narrowed->samples[i * 2 + 1], 0.5f, 1e-6f, "R should be average");
    }

    cdp_buffer_destroy(stereo);
    cdp_buffer_destroy(narrowed);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_narrow_unchanged(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* stereo = cdp_buffer_create(100, 2, 44100);
    for (size_t i = 0; i < 100; i++) {
        stereo->samples[i * 2] = 0.2f;
        stereo->samples[i * 2 + 1] = 0.8f;
    }

    /* Width 1.0 = unchanged */
    cdp_buffer* result = cdp_narrow(ctx, stereo, 1.0);
    ASSERT_NOT_NULL(result, "Narrow should succeed");

    for (size_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(result->samples[i * 2], 0.2f, 1e-6f, "L should be unchanged");
        ASSERT_FLOAT_EQ(result->samples[i * 2 + 1], 0.8f, 1e-6f, "R should be unchanged");
    }

    cdp_buffer_destroy(stereo);
    cdp_buffer_destroy(result);
    cdp_context_destroy(ctx);
    return 1;
}

/*============================================================================
 * Mixing Tests
 *============================================================================*/

static int test_mix2_equal_length(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* a = cdp_buffer_create(100, 1, 44100);
    cdp_buffer* b = cdp_buffer_create(100, 1, 44100);

    for (size_t i = 0; i < 100; i++) {
        a->samples[i] = 0.3f;
        b->samples[i] = 0.5f;
    }

    cdp_buffer* mix = cdp_mix2(ctx, a, b, 1.0, 1.0);
    ASSERT_NOT_NULL(mix, "Mix should succeed");
    ASSERT_EQ(mix->sample_count, 100, "Output length should match");

    for (size_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(mix->samples[i], 0.8f, 1e-6f, "Mix should sum samples");
    }

    cdp_buffer_destroy(a);
    cdp_buffer_destroy(b);
    cdp_buffer_destroy(mix);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_mix2_with_gains(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* a = cdp_buffer_create(100, 1, 44100);
    cdp_buffer* b = cdp_buffer_create(100, 1, 44100);

    for (size_t i = 0; i < 100; i++) {
        a->samples[i] = 1.0f;
        b->samples[i] = 1.0f;
    }

    /* Mix with 0.5 gain each -> output should be 1.0 */
    cdp_buffer* mix = cdp_mix2(ctx, a, b, 0.5, 0.5);
    ASSERT_NOT_NULL(mix, "Mix should succeed");

    for (size_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(mix->samples[i], 1.0f, 1e-6f, "Mix with gains");
    }

    cdp_buffer_destroy(a);
    cdp_buffer_destroy(b);
    cdp_buffer_destroy(mix);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_mix2_different_lengths(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* a = cdp_buffer_create(100, 1, 44100);
    cdp_buffer* b = cdp_buffer_create(50, 1, 44100);

    for (size_t i = 0; i < 100; i++) {
        a->samples[i] = 0.4f;
    }
    for (size_t i = 0; i < 50; i++) {
        b->samples[i] = 0.3f;
    }

    cdp_buffer* mix = cdp_mix2(ctx, a, b, 1.0, 1.0);
    ASSERT_NOT_NULL(mix, "Mix should succeed");
    ASSERT_EQ(mix->sample_count, 100, "Output should be length of longer input");

    /* First 50 samples: 0.4 + 0.3 = 0.7 */
    ASSERT_FLOAT_EQ(mix->samples[25], 0.7f, 1e-6f, "Overlapping region");

    /* Last 50 samples: just 0.4 from buffer a */
    ASSERT_FLOAT_EQ(mix->samples[75], 0.4f, 1e-6f, "Non-overlapping region");

    cdp_buffer_destroy(a);
    cdp_buffer_destroy(b);
    cdp_buffer_destroy(mix);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_mix_multiple(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* bufs[3];
    bufs[0] = cdp_buffer_create(100, 1, 44100);
    bufs[1] = cdp_buffer_create(100, 1, 44100);
    bufs[2] = cdp_buffer_create(100, 1, 44100);

    for (size_t i = 0; i < 100; i++) {
        bufs[0]->samples[i] = 0.2f;
        bufs[1]->samples[i] = 0.3f;
        bufs[2]->samples[i] = 0.1f;
    }

    cdp_buffer* mix = cdp_mix(ctx, bufs, NULL, 3);
    ASSERT_NOT_NULL(mix, "Mix should succeed");

    /* Sum: 0.2 + 0.3 + 0.1 = 0.6 */
    for (size_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(mix->samples[i], 0.6f, 1e-6f, "Mix should sum all buffers");
    }

    cdp_buffer_destroy(bufs[0]);
    cdp_buffer_destroy(bufs[1]);
    cdp_buffer_destroy(bufs[2]);
    cdp_buffer_destroy(mix);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_mix_multiple_with_gains(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* bufs[2];
    bufs[0] = cdp_buffer_create(100, 1, 44100);
    bufs[1] = cdp_buffer_create(100, 1, 44100);

    for (size_t i = 0; i < 100; i++) {
        bufs[0]->samples[i] = 1.0f;
        bufs[1]->samples[i] = 1.0f;
    }

    double gains[2] = {0.25, 0.75};
    cdp_buffer* mix = cdp_mix(ctx, bufs, gains, 2);
    ASSERT_NOT_NULL(mix, "Mix should succeed");

    /* 1.0*0.25 + 1.0*0.75 = 1.0 */
    for (size_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(mix->samples[i], 1.0f, 1e-6f, "Mix with gains");
    }

    cdp_buffer_destroy(bufs[0]);
    cdp_buffer_destroy(bufs[1]);
    cdp_buffer_destroy(mix);
    cdp_context_destroy(ctx);
    return 1;
}

/*============================================================================
 * Channel Operation Tests
 *============================================================================*/

static int test_to_mono_from_stereo(void)
{
    cdp_context* ctx = cdp_context_create();

    /* Create stereo buffer: L=0.4, R=0.8 -> mono should be 0.6 */
    cdp_buffer* stereo = cdp_buffer_create(100, 2, 44100);
    for (size_t i = 0; i < stereo->info.frame_count; i++) {
        stereo->samples[i * 2] = 0.4f;      /* Left */
        stereo->samples[i * 2 + 1] = 0.8f;  /* Right */
    }

    cdp_buffer* mono = cdp_to_mono(ctx, stereo);
    ASSERT_NOT_NULL(mono, "to_mono should succeed");
    ASSERT_EQ(mono->info.channels, 1, "Output should be mono");
    ASSERT_EQ(mono->info.frame_count, 100, "Frame count should be preserved");
    ASSERT_EQ(mono->info.sample_rate, 44100, "Sample rate should be preserved");

    /* Check averaging: (0.4 + 0.8) / 2 = 0.6 */
    for (size_t i = 0; i < mono->sample_count; i++) {
        ASSERT_FLOAT_EQ(mono->samples[i], 0.6f, 1e-6f, "Mono should be average of channels");
    }

    cdp_buffer_destroy(stereo);
    cdp_buffer_destroy(mono);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_to_mono_already_mono(void)
{
    cdp_context* ctx = cdp_context_create();

    /* Already mono - should just copy */
    cdp_buffer* mono_in = cdp_buffer_create(100, 1, 44100);
    for (size_t i = 0; i < mono_in->sample_count; i++) {
        mono_in->samples[i] = 0.5f;
    }

    cdp_buffer* mono_out = cdp_to_mono(ctx, mono_in);
    ASSERT_NOT_NULL(mono_out, "to_mono on mono should succeed");
    ASSERT_EQ(mono_out->info.channels, 1, "Output should be mono");

    for (size_t i = 0; i < mono_out->sample_count; i++) {
        ASSERT_FLOAT_EQ(mono_out->samples[i], 0.5f, 1e-6f, "Data should be copied");
    }

    cdp_buffer_destroy(mono_in);
    cdp_buffer_destroy(mono_out);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_to_stereo(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* mono = cdp_buffer_create(100, 1, 44100);
    for (size_t i = 0; i < mono->sample_count; i++) {
        mono->samples[i] = 0.7f;
    }

    cdp_buffer* stereo = cdp_to_stereo(ctx, mono);
    ASSERT_NOT_NULL(stereo, "to_stereo should succeed");
    ASSERT_EQ(stereo->info.channels, 2, "Output should be stereo");
    ASSERT_EQ(stereo->info.frame_count, 100, "Frame count should be preserved");

    /* Both channels should have same data */
    for (size_t i = 0; i < stereo->info.frame_count; i++) {
        ASSERT_FLOAT_EQ(stereo->samples[i * 2], 0.7f, 1e-6f, "Left should match");
        ASSERT_FLOAT_EQ(stereo->samples[i * 2 + 1], 0.7f, 1e-6f, "Right should match");
    }

    cdp_buffer_destroy(mono);
    cdp_buffer_destroy(stereo);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_to_stereo_requires_mono(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* stereo = cdp_buffer_create(100, 2, 44100);
    cdp_buffer* result = cdp_to_stereo(ctx, stereo);

    ASSERT_NULL(result, "to_stereo on stereo should fail");
    ASSERT_EQ(cdp_get_error(ctx), CDP_ERROR_INVALID_ARG, "Should be invalid arg error");

    cdp_buffer_destroy(stereo);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_extract_channel(void)
{
    cdp_context* ctx = cdp_context_create();

    /* Create stereo buffer with different data per channel */
    cdp_buffer* stereo = cdp_buffer_create(100, 2, 44100);
    for (size_t i = 0; i < stereo->info.frame_count; i++) {
        stereo->samples[i * 2] = 0.3f;      /* Left */
        stereo->samples[i * 2 + 1] = 0.9f;  /* Right */
    }

    /* Extract left channel */
    cdp_buffer* left = cdp_extract_channel(ctx, stereo, 0);
    ASSERT_NOT_NULL(left, "Extract left should succeed");
    ASSERT_EQ(left->info.channels, 1, "Left should be mono");
    for (size_t i = 0; i < left->sample_count; i++) {
        ASSERT_FLOAT_EQ(left->samples[i], 0.3f, 1e-6f, "Left data should match");
    }

    /* Extract right channel */
    cdp_buffer* right = cdp_extract_channel(ctx, stereo, 1);
    ASSERT_NOT_NULL(right, "Extract right should succeed");
    ASSERT_EQ(right->info.channels, 1, "Right should be mono");
    for (size_t i = 0; i < right->sample_count; i++) {
        ASSERT_FLOAT_EQ(right->samples[i], 0.9f, 1e-6f, "Right data should match");
    }

    cdp_buffer_destroy(stereo);
    cdp_buffer_destroy(left);
    cdp_buffer_destroy(right);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_extract_channel_out_of_range(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* stereo = cdp_buffer_create(100, 2, 44100);

    cdp_buffer* result = cdp_extract_channel(ctx, stereo, 2);
    ASSERT_NULL(result, "Extract channel 2 from stereo should fail");
    ASSERT_EQ(cdp_get_error(ctx), CDP_ERROR_INVALID_ARG, "Should be invalid arg error");

    result = cdp_extract_channel(ctx, stereo, -1);
    ASSERT_NULL(result, "Extract channel -1 should fail");

    cdp_buffer_destroy(stereo);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_merge_channels(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* left = cdp_buffer_create(100, 1, 44100);
    cdp_buffer* right = cdp_buffer_create(100, 1, 44100);

    for (size_t i = 0; i < 100; i++) {
        left->samples[i] = 0.2f;
        right->samples[i] = 0.8f;
    }

    cdp_buffer* stereo = cdp_merge_channels(ctx, left, right);
    ASSERT_NOT_NULL(stereo, "Merge should succeed");
    ASSERT_EQ(stereo->info.channels, 2, "Output should be stereo");
    ASSERT_EQ(stereo->info.frame_count, 100, "Frame count should match");

    for (size_t i = 0; i < stereo->info.frame_count; i++) {
        ASSERT_FLOAT_EQ(stereo->samples[i * 2], 0.2f, 1e-6f, "Left should match");
        ASSERT_FLOAT_EQ(stereo->samples[i * 2 + 1], 0.8f, 1e-6f, "Right should match");
    }

    cdp_buffer_destroy(left);
    cdp_buffer_destroy(right);
    cdp_buffer_destroy(stereo);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_split_channels(void)
{
    cdp_context* ctx = cdp_context_create();

    /* Create stereo buffer */
    cdp_buffer* stereo = cdp_buffer_create(100, 2, 44100);
    for (size_t i = 0; i < stereo->info.frame_count; i++) {
        stereo->samples[i * 2] = 0.1f;
        stereo->samples[i * 2 + 1] = 0.9f;
    }

    int num_channels = 0;
    cdp_buffer** channels = cdp_split_channels(ctx, stereo, &num_channels);

    ASSERT_NOT_NULL(channels, "Split should succeed");
    ASSERT_EQ(num_channels, 2, "Should have 2 channels");
    ASSERT_NOT_NULL(channels[0], "Left buffer should exist");
    ASSERT_NOT_NULL(channels[1], "Right buffer should exist");

    /* Check left channel */
    ASSERT_EQ(channels[0]->info.channels, 1, "Left should be mono");
    for (size_t i = 0; i < channels[0]->sample_count; i++) {
        ASSERT_FLOAT_EQ(channels[0]->samples[i], 0.1f, 1e-6f, "Left data should match");
    }

    /* Check right channel */
    ASSERT_EQ(channels[1]->info.channels, 1, "Right should be mono");
    for (size_t i = 0; i < channels[1]->sample_count; i++) {
        ASSERT_FLOAT_EQ(channels[1]->samples[i], 0.9f, 1e-6f, "Right data should match");
    }

    /* Cleanup */
    cdp_buffer_destroy(channels[0]);
    cdp_buffer_destroy(channels[1]);
    free(channels);
    cdp_buffer_destroy(stereo);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_interleave(void)
{
    cdp_context* ctx = cdp_context_create();

    cdp_buffer* ch0 = cdp_buffer_create(100, 1, 44100);
    cdp_buffer* ch1 = cdp_buffer_create(100, 1, 44100);
    cdp_buffer* ch2 = cdp_buffer_create(100, 1, 44100);

    for (size_t i = 0; i < 100; i++) {
        ch0->samples[i] = 0.1f;
        ch1->samples[i] = 0.5f;
        ch2->samples[i] = 0.9f;
    }

    cdp_buffer* channels[3] = {ch0, ch1, ch2};
    cdp_buffer* interleaved = cdp_interleave(ctx, channels, 3);

    ASSERT_NOT_NULL(interleaved, "Interleave should succeed");
    ASSERT_EQ(interleaved->info.channels, 3, "Should have 3 channels");
    ASSERT_EQ(interleaved->info.frame_count, 100, "Frame count should match");

    for (size_t i = 0; i < 100; i++) {
        ASSERT_FLOAT_EQ(interleaved->samples[i * 3 + 0], 0.1f, 1e-6f, "Ch0 should match");
        ASSERT_FLOAT_EQ(interleaved->samples[i * 3 + 1], 0.5f, 1e-6f, "Ch1 should match");
        ASSERT_FLOAT_EQ(interleaved->samples[i * 3 + 2], 0.9f, 1e-6f, "Ch2 should match");
    }

    cdp_buffer_destroy(ch0);
    cdp_buffer_destroy(ch1);
    cdp_buffer_destroy(ch2);
    cdp_buffer_destroy(interleaved);
    cdp_context_destroy(ctx);
    return 1;
}

static int test_split_interleave_roundtrip(void)
{
    cdp_context* ctx = cdp_context_create();

    /* Create a 4-channel buffer with unique data */
    cdp_buffer* original = cdp_buffer_create(50, 4, 48000);
    for (size_t i = 0; i < original->info.frame_count; i++) {
        for (int ch = 0; ch < 4; ch++) {
            original->samples[i * 4 + ch] = (float)(ch + 1) * 0.2f;
        }
    }

    /* Split into channels */
    int num_channels = 0;
    cdp_buffer** channels = cdp_split_channels(ctx, original, &num_channels);
    ASSERT_NOT_NULL(channels, "Split should succeed");
    ASSERT_EQ(num_channels, 4, "Should have 4 channels");

    /* Interleave back */
    cdp_buffer* reconstructed = cdp_interleave(ctx, channels, num_channels);
    ASSERT_NOT_NULL(reconstructed, "Interleave should succeed");

    /* Verify round-trip */
    ASSERT_EQ(reconstructed->info.channels, 4, "Should have 4 channels");
    ASSERT_EQ(reconstructed->info.frame_count, 50, "Frame count should match");
    ASSERT_EQ(reconstructed->info.sample_rate, 48000, "Sample rate should match");

    for (size_t i = 0; i < original->sample_count; i++) {
        ASSERT_FLOAT_EQ(reconstructed->samples[i], original->samples[i], 1e-6f,
                        "Roundtrip should preserve data");
    }

    /* Cleanup */
    for (int ch = 0; ch < num_channels; ch++) {
        cdp_buffer_destroy(channels[ch]);
    }
    free(channels);
    cdp_buffer_destroy(original);
    cdp_buffer_destroy(reconstructed);
    cdp_context_destroy(ctx);
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

    /* File I/O tests */
    RUN_TEST(test_write_read_wav_float);
    RUN_TEST(test_write_read_wav_pcm16);
    RUN_TEST(test_write_read_wav_pcm24);
    RUN_TEST(test_read_nonexistent_file);

    /* Spatial/pan tests */
    RUN_TEST(test_pan_center);
    RUN_TEST(test_pan_left);
    RUN_TEST(test_pan_right);
    RUN_TEST(test_pan_requires_mono);
    RUN_TEST(test_mirror);
    RUN_TEST(test_narrow_to_mono);
    RUN_TEST(test_narrow_unchanged);

    /* Mixing tests */
    RUN_TEST(test_mix2_equal_length);
    RUN_TEST(test_mix2_with_gains);
    RUN_TEST(test_mix2_different_lengths);
    RUN_TEST(test_mix_multiple);
    RUN_TEST(test_mix_multiple_with_gains);

    /* Channel operation tests */
    RUN_TEST(test_to_mono_from_stereo);
    RUN_TEST(test_to_mono_already_mono);
    RUN_TEST(test_to_stereo);
    RUN_TEST(test_to_stereo_requires_mono);
    RUN_TEST(test_extract_channel);
    RUN_TEST(test_extract_channel_out_of_range);
    RUN_TEST(test_merge_channels);
    RUN_TEST(test_split_channels);
    RUN_TEST(test_interleave);
    RUN_TEST(test_split_interleave_roundtrip);

    /* Summary */
    printf("\n=================\n");
    printf("Tests run: %d\n", tests_run);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);

    return tests_failed == 0 ? 0 : 1;
}
