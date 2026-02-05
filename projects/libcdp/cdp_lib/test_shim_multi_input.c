/*
 * Test program for CDP shim layer multi-input support
 * Tests the multi-slot input functionality needed for morph operations
 */

#include "cdp_shim.h"
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
#define ASSERT_GE(a, b, msg) ASSERT((a) >= (b), msg)
#define ASSERT_LT(a, b, msg) ASSERT((a) < (b), msg)

#define ASSERT_FLOAT_EQ(a, b, eps, msg) do { \
    if (fabs((a) - (b)) > (eps)) { \
        printf("  FAIL: %s (expected %f, got %f)\n", msg, (double)(b), (double)(a)); \
        return 0; \
    } \
} while(0)

#define RUN_TEST(test) do { \
    printf("Running %s...\n", #test); \
    tests_run++; \
    cdp_shim_cleanup(); \
    if (test()) { \
        printf("  PASS\n"); \
        tests_passed++; \
    } else { \
        tests_failed++; \
    } \
    cdp_shim_cleanup(); \
} while(0)

/*============================================================================
 * Multi-Input Slot Tests
 *============================================================================*/

static int test_init_cleanup(void)
{
    /* Test that init/cleanup cycle works */
    ASSERT_EQ(cdp_shim_init(), 0, "Init should succeed");
    cdp_shim_cleanup();

    /* Should be able to init again after cleanup */
    ASSERT_EQ(cdp_shim_init(), 0, "Re-init should succeed");

    return 1;
}

static int test_set_single_input_slot(void)
{
    float data[100];
    for (int i = 0; i < 100; i++) {
        data[i] = (float)i / 100.0f;
    }

    /* Set input at slot 0 */
    int fd = cdp_shim_set_input_slot(0, data, 100, 1, 44100);
    ASSERT_GE(fd, SHIM_INPUT_FD_BASE, "FD should be >= base");
    ASSERT_LT(fd, SHIM_INPUT_FD_BASE + SHIM_MAX_INPUT_SLOTS, "FD should be < base + max");

    /* Get FD for slot 0 */
    int fd2 = cdp_shim_get_input_fd(0);
    ASSERT_EQ(fd, fd2, "Get FD should return same FD");

    return 1;
}

static int test_set_multiple_input_slots(void)
{
    float data1[100], data2[200], data3[50];
    for (int i = 0; i < 100; i++) data1[i] = 0.1f;
    for (int i = 0; i < 200; i++) data2[i] = 0.2f;
    for (int i = 0; i < 50; i++) data3[i] = 0.3f;

    /* Register multiple input slots */
    int fd0 = cdp_shim_set_input_slot(0, data1, 100, 1, 44100);
    int fd1 = cdp_shim_set_input_slot(1, data2, 200, 1, 44100);
    int fd2 = cdp_shim_set_input_slot(2, data3, 50, 2, 48000);

    ASSERT_GE(fd0, SHIM_INPUT_FD_BASE, "Slot 0 FD valid");
    ASSERT_GE(fd1, SHIM_INPUT_FD_BASE, "Slot 1 FD valid");
    ASSERT_GE(fd2, SHIM_INPUT_FD_BASE, "Slot 2 FD valid");

    /* All FDs should be different */
    ASSERT_NE(fd0, fd1, "FDs should be different");
    ASSERT_NE(fd1, fd2, "FDs should be different");

    return 1;
}

static int test_read_from_slots(void)
{
    float data1[100], data2[100];
    for (int i = 0; i < 100; i++) {
        data1[i] = 1.0f;
        data2[i] = 2.0f;
    }

    int fd1 = cdp_shim_set_input_slot(0, data1, 100, 1, 44100);
    int fd2 = cdp_shim_set_input_slot(1, data2, 100, 1, 44100);

    /* Read from slot 0 */
    float buf1[50];
    int read1 = shim_fgetfbufEx(buf1, 50, fd1, 0);
    ASSERT_EQ(read1, 50, "Should read 50 samples from slot 0");
    ASSERT_FLOAT_EQ(buf1[0], 1.0f, 1e-6f, "Slot 0 data should be 1.0");

    /* Read from slot 1 */
    float buf2[50];
    int read2 = shim_fgetfbufEx(buf2, 50, fd2, 0);
    ASSERT_EQ(read2, 50, "Should read 50 samples from slot 1");
    ASSERT_FLOAT_EQ(buf2[0], 2.0f, 1e-6f, "Slot 1 data should be 2.0");

    return 1;
}

static int test_independent_read_positions(void)
{
    float data1[100], data2[100];
    for (int i = 0; i < 100; i++) {
        data1[i] = (float)i;
        data2[i] = (float)(i + 1000);
    }

    int fd1 = cdp_shim_set_input_slot(0, data1, 100, 1, 44100);
    int fd2 = cdp_shim_set_input_slot(1, data2, 100, 1, 44100);

    /* Read 30 from slot 0 */
    float buf[50];
    shim_fgetfbufEx(buf, 30, fd1, 0);

    /* Slot 0 position should be 30, slot 1 should be 0 */
    /* Read from slot 1 - should start at position 0 */
    shim_fgetfbufEx(buf, 10, fd2, 0);
    ASSERT_FLOAT_EQ(buf[0], 1000.0f, 1e-6f, "Slot 1 should start at position 0");

    /* Continue reading from slot 0 - should start at position 30 */
    shim_fgetfbufEx(buf, 10, fd1, 0);
    ASSERT_FLOAT_EQ(buf[0], 30.0f, 1e-6f, "Slot 0 should continue at position 30");

    return 1;
}

static int test_seek_per_slot(void)
{
    float data1[100], data2[100];
    for (int i = 0; i < 100; i++) {
        data1[i] = (float)i;
        data2[i] = (float)(i + 1000);
    }

    int fd1 = cdp_shim_set_input_slot(0, data1, 100, 1, 44100);
    int fd2 = cdp_shim_set_input_slot(1, data2, 100, 1, 44100);

    /* Seek slot 0 to position 50 */
    int pos = shim_sndseekEx(fd1, 50, 0);  /* SEEK_SET */
    ASSERT_EQ(pos, 50, "Seek should return position 50");

    /* Read from slot 0 - should be at position 50 */
    float buf[10];
    shim_fgetfbufEx(buf, 1, fd1, 0);
    ASSERT_FLOAT_EQ(buf[0], 50.0f, 1e-6f, "Slot 0 should be at position 50");

    /* Slot 1 should still be at position 0 */
    shim_fgetfbufEx(buf, 1, fd2, 0);
    ASSERT_FLOAT_EQ(buf[0], 1000.0f, 1e-6f, "Slot 1 should still be at position 0");

    return 1;
}

static int test_size_per_slot(void)
{
    float data1[100], data2[200];
    for (int i = 0; i < 100; i++) data1[i] = 0.0f;
    for (int i = 0; i < 200; i++) data2[i] = 0.0f;

    int fd1 = cdp_shim_set_input_slot(0, data1, 100, 1, 44100);
    int fd2 = cdp_shim_set_input_slot(1, data2, 200, 1, 44100);

    ASSERT_EQ(shim_sndsizeEx(fd1), 100, "Slot 0 size should be 100");
    ASSERT_EQ(shim_sndsizeEx(fd2), 200, "Slot 1 size should be 200");

    return 1;
}

static int test_legacy_single_input(void)
{
    /* Test that the legacy cdp_shim_set_input still works */
    float data[100];
    for (int i = 0; i < 100; i++) {
        data[i] = 0.5f;
    }

    int ret = cdp_shim_set_input(data, 100, 1, 44100);
    ASSERT_EQ(ret, 0, "Legacy set_input should return 0");

    /* Legacy SHIM_INPUT_FD should work */
    float buf[10];
    int read = shim_fgetfbufEx(buf, 10, SHIM_INPUT_FD, 0);
    ASSERT_EQ(read, 10, "Should read from legacy FD");
    ASSERT_FLOAT_EQ(buf[0], 0.5f, 1e-6f, "Data should match");

    return 1;
}

static int test_membuf_lookup(void)
{
    float data[100];
    for (int i = 0; i < 100; i++) data[i] = 0.0f;

    int fd = cdp_shim_set_input_slot(3, data, 100, 2, 48000);

    cdp_membuf* buf = cdp_shim_get_membuf(fd);
    ASSERT(buf != NULL, "Should get membuf for valid FD");
    ASSERT_EQ(buf->length, 100, "Membuf length should be 100");
    ASSERT_EQ(buf->channels, 2, "Membuf channels should be 2");
    ASSERT_EQ(buf->sample_rate, 48000, "Membuf sample_rate should be 48000");

    /* Invalid FD */
    cdp_membuf* invalid = cdp_shim_get_membuf(99999);
    ASSERT(invalid == NULL, "Should return NULL for invalid FD");

    return 1;
}

/*============================================================================
 * Temp Buffer Tests
 *============================================================================*/

static int test_create_temp(void)
{
    int fd = cdp_shim_create_temp(1, 44100);
    ASSERT_GE(fd, SHIM_TEMP_FD_BASE, "Temp FD should be >= base");

    cdp_membuf* buf = cdp_shim_get_membuf(fd);
    ASSERT(buf != NULL, "Should get membuf for temp");
    ASSERT_EQ(buf->channels, 1, "Temp channels should be 1");
    ASSERT_EQ(buf->sample_rate, 44100, "Temp sample_rate should be 44100");

    cdp_shim_free_temp(fd);
    return 1;
}

static int test_write_to_temp(void)
{
    int fd = cdp_shim_create_temp(1, 44100);

    float data[100];
    for (int i = 0; i < 100; i++) {
        data[i] = (float)i;
    }

    /* Write to temp buffer */
    int written = shim_fputfbufEx(data, 100, fd);
    ASSERT_EQ(written, 100, "Should write 100 samples");

    /* Seek back to start */
    shim_sndseekEx(fd, 0, 0);

    /* Read back */
    float buf[100];
    int read = shim_fgetfbufEx(buf, 100, fd, 0);
    ASSERT_EQ(read, 100, "Should read 100 samples");
    ASSERT_FLOAT_EQ(buf[50], 50.0f, 1e-6f, "Data should match");

    cdp_shim_free_temp(fd);
    return 1;
}

static int test_temp_auto_grow(void)
{
    int fd = cdp_shim_create_temp(1, 44100);

    /* Write more than initial capacity */
    float data[100000];
    for (int i = 0; i < 100000; i++) {
        data[i] = (float)i;
    }

    int written = shim_fputfbufEx(data, 100000, fd);
    ASSERT_EQ(written, 100000, "Should write all samples");

    cdp_membuf* buf = cdp_shim_get_membuf(fd);
    ASSERT_GE((int)buf->capacity, 100000, "Capacity should have grown");

    cdp_shim_free_temp(fd);
    return 1;
}

/*============================================================================
 * Output Buffer Tests
 *============================================================================*/

static int test_output_buffer(void)
{
    /* Set output buffer */
    cdp_shim_set_output(NULL, 0, 1, 44100);

    /* Write to output */
    float data[100];
    for (int i = 0; i < 100; i++) {
        data[i] = 0.75f;
    }

    int written = shim_fputfbufEx(data, 100, SHIM_OUTPUT_FD);
    ASSERT_EQ(written, 100, "Should write 100 samples");

    /* Get output */
    float *out_data;
    int channels, sample_rate;
    size_t length = cdp_shim_get_output(&out_data, &channels, &sample_rate);

    ASSERT_EQ((int)length, 100, "Output length should be 100");
    ASSERT_EQ(channels, 1, "Output channels should be 1");
    ASSERT_EQ(sample_rate, 44100, "Output sample_rate should be 44100");
    ASSERT_FLOAT_EQ(out_data[50], 0.75f, 1e-6f, "Output data should match");

    return 1;
}

static int test_output_auto_grow(void)
{
    cdp_shim_set_output(NULL, 0, 1, 44100);

    /* Write large amount */
    float data[50000];
    for (int i = 0; i < 50000; i++) {
        data[i] = (float)i;
    }

    int written = shim_fputfbufEx(data, 50000, SHIM_OUTPUT_FD);
    ASSERT_EQ(written, 50000, "Should write all samples");

    /* Write more */
    written = shim_fputfbufEx(data, 50000, SHIM_OUTPUT_FD);
    ASSERT_EQ(written, 50000, "Should write more samples");

    float *out_data;
    int channels, sample_rate;
    size_t length = cdp_shim_get_output(&out_data, &channels, &sample_rate);
    ASSERT_EQ((int)length, 100000, "Total output should be 100000");

    return 1;
}

/*============================================================================
 * Reset Tests
 *============================================================================*/

static int test_reset_slot(void)
{
    float data[100];
    for (int i = 0; i < 100; i++) data[i] = (float)i;

    int fd = cdp_shim_set_input_slot(0, data, 100, 1, 44100);

    /* Read some */
    float buf[50];
    shim_fgetfbufEx(buf, 50, fd, 0);

    /* Reset slot */
    cdp_shim_reset_slot(0);

    /* Should be back at position 0 */
    shim_fgetfbufEx(buf, 1, fd, 0);
    ASSERT_FLOAT_EQ(buf[0], 0.0f, 1e-6f, "Position should be reset to 0");

    return 1;
}

static int test_reset_all(void)
{
    float data1[100], data2[100];
    for (int i = 0; i < 100; i++) {
        data1[i] = (float)i;
        data2[i] = (float)(i + 1000);
    }

    int fd1 = cdp_shim_set_input_slot(0, data1, 100, 1, 44100);
    int fd2 = cdp_shim_set_input_slot(1, data2, 100, 1, 44100);

    /* Read some from both */
    float buf[50];
    shim_fgetfbufEx(buf, 50, fd1, 0);
    shim_fgetfbufEx(buf, 30, fd2, 0);

    /* Reset all */
    cdp_shim_reset_all();

    /* Both should be back at position 0 */
    shim_fgetfbufEx(buf, 1, fd1, 0);
    ASSERT_FLOAT_EQ(buf[0], 0.0f, 1e-6f, "Slot 0 should be reset");

    shim_fgetfbufEx(buf, 1, fd2, 0);
    ASSERT_FLOAT_EQ(buf[0], 1000.0f, 1e-6f, "Slot 1 should be reset");

    return 1;
}

/*============================================================================
 * Edge Cases
 *============================================================================*/

static int test_invalid_slot(void)
{
    float data[100];

    /* Negative slot */
    int fd = cdp_shim_set_input_slot(-1, data, 100, 1, 44100);
    ASSERT_EQ(fd, -1, "Negative slot should fail");

    /* Slot >= max */
    fd = cdp_shim_set_input_slot(SHIM_MAX_INPUT_SLOTS, data, 100, 1, 44100);
    ASSERT_EQ(fd, -1, "Slot >= max should fail");

    return 1;
}

static int test_unregistered_slot(void)
{
    /* Get FD for unregistered slot */
    int fd = cdp_shim_get_input_fd(5);
    ASSERT_EQ(fd, -1, "Unregistered slot should return -1");

    return 1;
}

static int test_read_eof(void)
{
    float data[10];
    for (int i = 0; i < 10; i++) data[i] = 0.0f;

    int fd = cdp_shim_set_input_slot(0, data, 10, 1, 44100);

    /* Read all */
    float buf[20];
    int read = shim_fgetfbufEx(buf, 10, fd, 0);
    ASSERT_EQ(read, 10, "Should read all 10");

    /* Read again - should get EOF (0) */
    read = shim_fgetfbufEx(buf, 10, fd, 0);
    ASSERT_EQ(read, 0, "Should return 0 at EOF");

    return 1;
}

/*============================================================================
 * Main
 *============================================================================*/

int main(void)
{
    printf("CDP Shim Multi-Input Tests\n");
    printf("==========================\n\n");

    /* Basic tests */
    RUN_TEST(test_init_cleanup);
    RUN_TEST(test_set_single_input_slot);
    RUN_TEST(test_set_multiple_input_slots);

    /* Read tests */
    RUN_TEST(test_read_from_slots);
    RUN_TEST(test_independent_read_positions);
    RUN_TEST(test_seek_per_slot);
    RUN_TEST(test_size_per_slot);
    RUN_TEST(test_membuf_lookup);

    /* Legacy compatibility */
    RUN_TEST(test_legacy_single_input);

    /* Temp buffer tests */
    RUN_TEST(test_create_temp);
    RUN_TEST(test_write_to_temp);
    RUN_TEST(test_temp_auto_grow);

    /* Output buffer tests */
    RUN_TEST(test_output_buffer);
    RUN_TEST(test_output_auto_grow);

    /* Reset tests */
    RUN_TEST(test_reset_slot);
    RUN_TEST(test_reset_all);

    /* Edge cases */
    RUN_TEST(test_invalid_slot);
    RUN_TEST(test_unregistered_slot);
    RUN_TEST(test_read_eof);

    /* Summary */
    printf("\n==========================\n");
    printf("Tests run: %d\n", tests_run);
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);

    return tests_failed == 0 ? 0 : 1;
}
