/*
 * Test program for CDP library integration
 */

#include "cdp_lib.h"
#include "cdp_envelope.h"
#include "cdp_distort.h"
#include "cdp_reverb.h"
#include "cdp_granular.h"
#include "cdp_experimental.h"
#include "cdp_filters.h"
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

    /* Test 13: Spectral focus */
    printf("Test 13: Spectral focus (440 Hz, 100 Hz BW, +6dB)... ");
    cdp_lib_buffer* focused = cdp_lib_spectral_focus(ctx, input, 440.0, 100.0, 6.0, 1024);
    if (focused == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", focused->length);
        cdp_lib_buffer_free(focused);
    }

    /* Test 14: Spectral hilite */
    printf("Test 14: Spectral hilite (-20dB threshold, +6dB boost)... ");
    cdp_lib_buffer* hilited = cdp_lib_spectral_hilite(ctx, input, -20.0, 6.0, 1024);
    if (hilited == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", hilited->length);
        cdp_lib_buffer_free(hilited);
    }

    /* Test 15: Spectral fold */
    printf("Test 15: Spectral fold (2000 Hz)... ");
    cdp_lib_buffer* folded = cdp_lib_spectral_fold(ctx, input, 2000.0, 1024);
    if (folded == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", folded->length);
        cdp_lib_buffer_free(folded);
    }

    /* Test 16: Spectral clean */
    printf("Test 16: Spectral clean (-40dB threshold)... ");
    cdp_lib_buffer* cleaned = cdp_lib_spectral_clean(ctx, input, -40.0, 1024);
    if (cleaned == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", cleaned->length);
        cdp_lib_buffer_free(cleaned);
    }

    /* Test 17: Strange attractor */
    printf("Test 17: Strange (Lorenz) modulation... ");
    cdp_lib_buffer* strange = cdp_lib_strange(ctx, input, 0.5, 2.0, 12345);
    if (strange == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", strange->length);
        cdp_lib_buffer_free(strange);
    }

    /* Test 18: Brownian modulation */
    printf("Test 18: Brownian modulation (pitch)... ");
    cdp_lib_buffer* brownian = cdp_lib_brownian(ctx, input, 0.1, 0.9, 0, 12345);
    if (brownian == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", brownian->length);
        cdp_lib_buffer_free(brownian);
    }

    /* Test 19: Crystal texture */
    printf("Test 19: Crystal texture... ");
    cdp_lib_buffer* crystal = cdp_lib_crystal(ctx, input, 50.0, 0.5, 2.0, 12345);
    if (crystal == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        /* Crystal adds decay tail, so output should be longer */
        if (crystal->length > input->length) {
            printf("OK (length: %zu, original: %zu)\n", crystal->length, input->length);
        } else {
            printf("UNEXPECTED (length: %zu, expected > %zu)\n",
                   crystal->length, input->length);
        }
        cdp_lib_buffer_free(crystal);
    }

    /* Test 20: Fractal processing */
    printf("Test 20: Fractal (depth 3, 0.5 ratio)... ");
    cdp_lib_buffer* fractal = cdp_lib_fractal(ctx, input, 3, 0.5, 0.7, 12345);
    if (fractal == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", fractal->length);
        cdp_lib_buffer_free(fractal);
    }

    /* Test 21: Quirk */
    printf("Test 21: Quirk (probability 0.3, both modes)... ");
    cdp_lib_buffer* quirk = cdp_lib_quirk(ctx, input, 0.3, 0.5, 2, 12345);
    if (quirk == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", quirk->length);
        cdp_lib_buffer_free(quirk);
    }

    /* Test 22: Chirikov map */
    printf("Test 22: Chirikov (K=2.0)... ");
    cdp_lib_buffer* chirikov = cdp_lib_chirikov(ctx, input, 2.0, 0.5, 2.0, 12345);
    if (chirikov == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", chirikov->length);
        cdp_lib_buffer_free(chirikov);
    }

    /* Test 23: Cantor set gating */
    printf("Test 23: Cantor (depth 4)... ");
    cdp_lib_buffer* cantor = cdp_lib_cantor(ctx, input, 4, 0.5, 5.0, 12345);
    if (cantor == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", cantor->length);
        cdp_lib_buffer_free(cantor);
    }

    /* Test 24: Cascade */
    printf("Test 24: Cascade (6 echoes)... ");
    cdp_lib_buffer* cascade = cdp_lib_cascade(ctx, input, 6, 100.0, 0.95, 0.7, 0.8, 12345);
    if (cascade == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        /* Cascade adds echoes, so output should be longer */
        if (cascade->length > input->length) {
            printf("OK (length: %zu, original: %zu)\n", cascade->length, input->length);
        } else {
            printf("UNEXPECTED (length: %zu, expected > %zu)\n",
                   cascade->length, input->length);
        }
        cdp_lib_buffer_free(cascade);
    }

    /* Test 25: Fracture */
    printf("Test 25: Fracture (50ms fragments)... ");
    cdp_lib_buffer* fracture = cdp_lib_fracture(ctx, input, 50.0, 0.5, 0.3, 12345);
    if (fracture == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", fracture->length);
        cdp_lib_buffer_free(fracture);
    }

    /* Test 26: Tesselate */
    printf("Test 26: Tesselate (mirror pattern)... ");
    cdp_lib_buffer* tesselate = cdp_lib_tesselate(ctx, input, 50.0, 1, 0.25, 0.3, 12345);
    if (tesselate == NULL) {
        printf("FAILED: %s\n", cdp_lib_get_error(ctx));
    } else {
        printf("OK (length: %zu)\n", tesselate->length);
        cdp_lib_buffer_free(tesselate);
    }

    /* Cleanup */
    cdp_lib_buffer_free(input);
    cdp_lib_cleanup(ctx);

    printf("\nDone.\n");
    return 0;
}
