/*
 * CDP Constrict - Silence Constriction Effect Implementation
 *
 * Shortens the duration of zero-level (silent) sections in sound.
 */

#include "cdp_constrict.h"
#include "cdp_lib_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MIN_CONSTRICTION 0.0
#define MAX_CONSTRICTION 200.0

/*
 * Apply constrict effect - shorten or remove silent sections.
 */
cdp_lib_buffer* cdp_lib_constrict(cdp_lib_ctx* ctx,
                                   const cdp_lib_buffer* input,
                                   double constriction) {
    if (ctx == NULL || input == NULL) {
        if (ctx) cdp_lib_set_error(ctx, "NULL input");
        return NULL;
    }

    if (constriction < MIN_CONSTRICTION || constriction > MAX_CONSTRICTION) {
        cdp_lib_set_error(ctx, "constriction must be between 0.0 and 200.0");
        return NULL;
    }

    int channels = input->channels;
    int sample_rate = input->sample_rate;
    size_t input_length = input->length;

    if (input_length == 0) {
        cdp_lib_set_error(ctx, "Input buffer is empty");
        return NULL;
    }

    /* Calculate decimation factor and overlap mode */
    int overlap_mode = 0;
    double decimation = 1.0;

    if (constriction > 100.0) {
        overlap_mode = 1;
        decimation = (constriction - 100.0) / 100.0;
        if (decimation > 1.0) decimation = 1.0;
        if (decimation < 0.0) decimation = 0.0;
    } else {
        decimation = 1.0 - (constriction / 100.0);
        if (decimation > 1.0) decimation = 1.0;
        if (decimation < 0.0) decimation = 0.0;
    }

    /* First pass: calculate output size and max sample for gain compensation */
    size_t output_size = 0;
    double max_sample = 0.0;
    int in_zero_section = 0;
    size_t zero_start = 0;

    /* Count non-zero samples and calculate reduced silence */
    for (size_t i = 0; i < input_length; i++) {
        if (input->data[i] == 0.0f) {
            if (!in_zero_section) {
                in_zero_section = 1;
                zero_start = i;
            }
        } else {
            if (in_zero_section) {
                /* End of zero section */
                size_t zero_count = i - zero_start;

                /* Align to channel boundaries */
                if (channels > 1) {
                    size_t start_remnant = zero_start % channels;
                    size_t end_aligned = i;
                    while ((end_aligned % channels) != start_remnant && end_aligned < input_length) {
                        end_aligned++;
                    }
                    zero_count = end_aligned - zero_start;
                }

                if (!overlap_mode) {
                    /* Calculate reduced zero count */
                    size_t frames = zero_count / channels;
                    frames = (size_t)round((double)frames * decimation);
                    size_t reduced_zeros = frames * channels;
                    output_size += reduced_zeros;
                }
                /* In overlap mode, zeros are removed entirely */

                in_zero_section = 0;
            }
            output_size++;
            double abs_val = fabs(input->data[i]);
            if (abs_val > max_sample) max_sample = abs_val;
        }
    }

    /* Handle trailing zeros */
    if (in_zero_section) {
        size_t zero_count = input_length - zero_start;
        if (channels > 1) {
            zero_count = (zero_count / channels) * channels;
        }
        if (!overlap_mode) {
            size_t frames = zero_count / channels;
            frames = (size_t)round((double)frames * decimation);
            output_size += frames * channels;
        }
    }

    /* If overlap mode and max_sample would cause clipping, we need gain compensation */
    double gain = 1.0;
    if (overlap_mode) {
        /* In overlap mode, we need to do a more careful analysis */
        /* For now, estimate based on max overlap potential */
        /* A more accurate approach would require simulating the overlap */
        if (max_sample > 0.5) {
            gain = 0.5 / max_sample;
        }

        /* Re-estimate output size for overlap mode */
        /* In overlap mode, output is smaller due to overlapping sections */
        output_size = 0;
        in_zero_section = 0;
        size_t virtual_pos = 0;

        for (size_t i = 0; i < input_length; i++) {
            if (input->data[i] == 0.0f) {
                if (!in_zero_section) {
                    in_zero_section = 1;
                    zero_start = i;
                }
            } else {
                if (in_zero_section) {
                    size_t zero_count = i - zero_start;
                    if (channels > 1) {
                        zero_count = (zero_count / channels) * channels;
                    }
                    /* Calculate backtrack amount */
                    size_t gap = (size_t)round((double)zero_count * decimation);
                    gap = (gap / channels) * channels;
                    if (virtual_pos >= gap) {
                        virtual_pos -= gap;
                    } else {
                        virtual_pos = 0;
                    }
                    in_zero_section = 0;
                }
                virtual_pos++;
            }
        }
        output_size = virtual_pos > 0 ? virtual_pos : 1;
    }

    /* Ensure minimum output size */
    if (output_size == 0) {
        output_size = channels;  /* At least one frame of silence */
    }

    /* Allocate output buffer */
    cdp_lib_buffer* output = cdp_lib_buffer_create(output_size, channels, sample_rate);
    if (output == NULL) {
        cdp_lib_set_error(ctx, "Failed to allocate output buffer");
        return NULL;
    }
    memset(output->data, 0, output_size * sizeof(float));

    /* Second pass: generate output */
    size_t out_pos = 0;
    in_zero_section = 0;
    zero_start = 0;

    if (overlap_mode) {
        /* Overlap mode: backtrack output position at silence boundaries */
        for (size_t i = 0; i < input_length; i++) {
            if (input->data[i] == 0.0f) {
                if (!in_zero_section) {
                    in_zero_section = 1;
                    zero_start = i;
                }
            } else {
                if (in_zero_section) {
                    /* End of zero section - calculate backtrack */
                    size_t zero_count = i - zero_start;
                    if (channels > 1) {
                        zero_count = (zero_count / channels) * channels;
                    }
                    size_t gap = (size_t)round((double)zero_count * decimation);
                    gap = (gap / channels) * channels;
                    if (out_pos >= gap) {
                        out_pos -= gap;
                    } else {
                        out_pos = 0;
                    }
                    in_zero_section = 0;
                }

                /* Add sample with gain and overlap */
                if (out_pos < output_size) {
                    output->data[out_pos] += (float)(input->data[i] * gain);
                    /* Clamp to prevent overflow */
                    if (output->data[out_pos] > 1.0f) output->data[out_pos] = 1.0f;
                    if (output->data[out_pos] < -1.0f) output->data[out_pos] = -1.0f;
                }
                out_pos++;
            }
        }
    } else {
        /* Non-overlap mode: reduce silence duration */
        for (size_t i = 0; i < input_length; i++) {
            if (input->data[i] == 0.0f) {
                if (!in_zero_section) {
                    in_zero_section = 1;
                    zero_start = i;
                }
            } else {
                if (in_zero_section) {
                    /* End of zero section - write reduced zeros */
                    size_t zero_count = i - zero_start;

                    /* Align to channel boundaries */
                    if (channels > 1) {
                        size_t start_remnant = zero_start % channels;
                        size_t end_aligned = i;
                        while ((end_aligned % channels) != start_remnant && end_aligned < input_length) {
                            end_aligned++;
                        }
                        zero_count = end_aligned - zero_start;
                    }

                    /* Calculate reduced zero count */
                    size_t frames = zero_count / channels;
                    frames = (size_t)round((double)frames * decimation);
                    size_t reduced_zeros = frames * channels;

                    /* Write zeros */
                    for (size_t j = 0; j < reduced_zeros && out_pos < output_size; j++) {
                        output->data[out_pos++] = 0.0f;
                    }
                    in_zero_section = 0;
                }

                /* Copy non-zero sample */
                if (out_pos < output_size) {
                    output->data[out_pos++] = input->data[i];
                }
            }
        }

        /* Handle trailing zeros */
        if (in_zero_section) {
            size_t zero_count = input_length - zero_start;
            if (channels > 1) {
                zero_count = (zero_count / channels) * channels;
            }
            size_t frames = zero_count / channels;
            frames = (size_t)round((double)frames * decimation);
            size_t reduced_zeros = frames * channels;

            for (size_t j = 0; j < reduced_zeros && out_pos < output_size; j++) {
                output->data[out_pos++] = 0.0f;
            }
        }
    }

    /* Trim output to actual used size */
    if (out_pos < output_size && out_pos > 0) {
        /* Create a new properly sized buffer */
        cdp_lib_buffer* trimmed = cdp_lib_buffer_create(out_pos, channels, sample_rate);
        if (trimmed != NULL) {
            memcpy(trimmed->data, output->data, out_pos * sizeof(float));
            cdp_lib_buffer_free(output);
            output = trimmed;
        }
    }

    return output;
}
