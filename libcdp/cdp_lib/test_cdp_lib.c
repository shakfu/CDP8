/*
 * Test program for CDP library integration
 */

#include "cdp_lib.h"
#include "cdp_envelope.h"
#include "cdp_distort.h"
#include "cdp_reverb.h"
#include "cdp_granular.h"
#include <stdio.h>
#include <math.h>

#define SAMPLE_RATE 44100
#define DURATION 0.5
#define FREQ 440.0

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    printf("CDP Library Integration Test\n");
    printf("============================\n\n");

    /* Initialize library */
    printf("Initializing CDP library... ");
    cdp_lib_ctx* ctx = cdp_lib_init();
    if (ctx == NULL) {
        printf("FAILED\n");
        return 1;
    }
    printf("OK\n");

    /* Create test signal - sine wave */
    size_t num_samples = (size_t)(SAMPLE_RATE * DURATION);
    cdp_lib_buffer* input = cdp_lib_buffer_create(num_samples, 1, SAMPLE_RATE);
    if (input == NULL) {
        printf("Failed to create input buffer\n");
        cdp_lib_cleanup(ctx);
        return 1;
    }

    printf("Generating %zu samples of %.0f Hz sine wave... ", num_samples, FREQ);
    for (size_t i = 0; i < num_samples; i++) {
        input->data[i] = (float)(0.5 * sin(2.0 * M_PI * FREQ * i / SAMPLE_RATE));
    }
    printf("OK\n");

    /* Test 1: Loudness change */
    printf("\nTest 1: Loudness (+6 dB)... ");
    cdp_lib_buffer* louder = cdp_lib_loudness(ctx, input, 6.0);
    if (louder == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        /* Check that amplitude roughly doubled */
        float ratio = louder->data[num_samples/4] / input->data[num_samples/4];
        if (ratio > 1.9 && ratio < 2.1) {
            printf("OK (ratio: %.2f)\n", ratio);
        } else {
            printf("UNEXPECTED (ratio: %.2f, expected ~2.0)\n", ratio);
        }
        cdp_lib_buffer_free(louder);
    }

    /* Test 2: Speed change */
    printf("Test 2: Speed (2x)... ");
    cdp_lib_buffer* faster = cdp_lib_speed(ctx, input, 2.0);
    if (faster == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        /* Check that length is roughly halved */
        float length_ratio = (float)faster->length / (float)input->length;
        if (length_ratio > 0.45 && length_ratio < 0.55) {
            printf("OK (length ratio: %.2f)\n", length_ratio);
        } else {
            printf("UNEXPECTED (length ratio: %.2f, expected ~0.5)\n", length_ratio);
        }
        cdp_lib_buffer_free(faster);
    }

    /* Test 3: Time stretch */
    printf("Test 3: Time stretch (2x)... ");
    cdp_lib_buffer* stretched = cdp_lib_time_stretch(ctx, input, 2.0, 1024, 3);
    if (stretched == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        float length_ratio = (float)stretched->length / (float)input->length;
        if (length_ratio > 1.8 && length_ratio < 2.2) {
            printf("OK (ratio: %.2f)\n", length_ratio);
        } else {
            printf("UNEXPECTED (ratio: %.2f, expected ~2.0)\n", length_ratio);
        }
        cdp_lib_buffer_free(stretched);
    }

    /* Test 4: Spectral blur */
    printf("Test 4: Spectral blur (50ms)... ");
    cdp_lib_buffer* blurred = cdp_lib_spectral_blur(ctx, input, 0.05, 1024);
    if (blurred == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", blurred->length);
        cdp_lib_buffer_free(blurred);
    }

    /* Test 5: Pitch shift */
    printf("Test 5: Pitch shift (+5 semitones)... ");
    cdp_lib_buffer* pitched = cdp_lib_pitch_shift(ctx, input, 5.0, 1024);
    if (pitched == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", pitched->length);
        cdp_lib_buffer_free(pitched);
    }

    /* Test 6: Lowpass filter */
    printf("Test 6: Lowpass filter (1000 Hz)... ");
    cdp_lib_buffer* lowpassed = cdp_lib_filter_lowpass(ctx, input, 1000.0, -60.0, 1024);
    if (lowpassed == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", lowpassed->length);
        cdp_lib_buffer_free(lowpassed);
    }

    /* Test 7: Dovetail fades */
    printf("Test 7: Dovetail fades... ");
    cdp_lib_buffer* dovetailed = cdp_lib_dovetail(ctx, input, 0.05, 0.05, 1, 1);
    if (dovetailed == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        /* Check that first sample is nearly zero */
        if (fabsf(dovetailed->data[0]) < 0.1f) {
            printf("OK (fade-in applied)\n");
        } else {
            printf("UNEXPECTED (first sample: %.3f)\n", dovetailed->data[0]);
        }
        cdp_lib_buffer_free(dovetailed);
    }

    /* Test 8: Tremolo */
    printf("Test 8: Tremolo (5 Hz, depth 0.5)... ");
    cdp_lib_buffer* tremoloed = cdp_lib_tremolo(ctx, input, 5.0, 0.5, 1.0);
    if (tremoloed == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", tremoloed->length);
        cdp_lib_buffer_free(tremoloed);
    }

    /* Test 9: Distort overload */
    printf("Test 9: Distort overload (clip 0.3)... ");
    cdp_lib_buffer* distorted = cdp_lib_distort_overload(ctx, input, 0.3, 0.5);
    if (distorted == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", distorted->length);
        cdp_lib_buffer_free(distorted);
    }

    /* Test 10: Reverb */
    printf("Test 10: Reverb (1s decay)... ");
    cdp_lib_buffer* reverbed = cdp_lib_reverb(ctx, input, 0.5, 1.0, 0.5, 8000, 0);
    if (reverbed == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        /* Reverb adds a tail, so output should be longer */
        if (reverbed->length > input->length) {
            printf("OK (length: %zu, original: %zu)\n", reverbed->length, input->length);
        } else {
            printf("UNEXPECTED (length: %zu, expected > %zu)\n",
                   reverbed->length, input->length);
        }
        cdp_lib_buffer_free(reverbed);
    }

    /* Test 11: Brassage */
    printf("Test 11: Brassage... ");
    cdp_lib_buffer* brassaged = cdp_lib_brassage(ctx, input, 1.0, 1.0, 50.0, 0, 0, 0, 12345);
    if (brassaged == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", brassaged->length);
        cdp_lib_buffer_free(brassaged);
    }

    /* Test 12: Freeze */
    printf("Test 12: Freeze (0.1-0.2s, 1s output)... ");
    cdp_lib_buffer* frozen = cdp_lib_freeze(ctx, input, 0.1, 0.2, 1.0, 0.05, 0.2, 0, 0.1, 1.0, 12345);
    if (frozen == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        /* Output should be ~1 second (44100 samples) */
        size_t expected = (size_t)(1.0 * SAMPLE_RATE);
        if (frozen->length > expected * 0.9 && frozen->length < expected * 1.1) {
            printf("OK (length: %zu, expected ~%zu)\n", frozen->length, expected);
        } else {
            printf("UNEXPECTED (length: %zu, expected ~%zu)\n", frozen->length, expected);
        }
        cdp_lib_buffer_free(frozen);
    }

    /* Cleanup */
    cdp_lib_buffer_free(input);
    cdp_lib_cleanup(ctx);

    printf("\nDone.\n");
    return 0;
}
