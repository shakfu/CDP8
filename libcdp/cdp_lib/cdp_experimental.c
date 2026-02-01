/*
 * CDP Experimental Processing - Implementation
 *
 * Implements experimental audio transformations including
 * chaotic modulation, random walks, and crystalline textures.
 */

#include "cdp_experimental.h"
#include "cdp_spectral.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Simple linear congruential generator for reproducibility */
static unsigned int lcg_next(unsigned int *state) {
    *state = (*state * 1103515245 + 12345) & 0x7fffffff;
    return *state;
}

/* Random float in range [0, 1) */
static float lcg_float(unsigned int *state) {
    return (float)lcg_next(state) / (float)0x7fffffff;
}

/* Random float in range [-1, 1) */
static float lcg_float_bipolar(unsigned int *state) {
    return lcg_float(state) * 2.0f - 1.0f;
}

/*
 * Lorenz attractor state
 */
typedef struct {
    double x, y, z;
} lorenz_state;

/*
 * Step the Lorenz attractor
 */
static void lorenz_step(lorenz_state *s, double dt) {
    /* Lorenz parameters (classic chaotic regime) */
    const double sigma = 10.0;
    const double rho = 28.0;
    const double beta = 8.0 / 3.0;

    double dx = sigma * (s->y - s->x);
    double dy = s->x * (rho - s->z) - s->y;
    double dz = s->x * s->y - beta * s->z;

    s->x += dx * dt;
    s->y += dy * dt;
    s->z += dz * dt;
}

/*
 * Initialize Lorenz attractor from seed and settle it
 */
static void lorenz_init(lorenz_state *s, unsigned int seed) {
    /* Initialize from seed */
    unsigned int rng = seed;
    s->x = lcg_float_bipolar(&rng) * 10.0;
    s->y = lcg_float_bipolar(&rng) * 10.0;
    s->z = lcg_float(&rng) * 30.0 + 10.0;

    /* Settle the attractor for 1000 steps */
    for (int i = 0; i < 1000; i++) {
        lorenz_step(s, 0.01);
    }
}

cdp_lib_buffer* cdp_lib_strange(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double chaos_amount,
                                 double rate,
                                 unsigned int seed) {
    if (ctx == NULL || input == NULL || input->data == NULL) return NULL;
    if (chaos_amount < 0) chaos_amount = 0;
    if (chaos_amount > 1) chaos_amount = 1;
    if (rate <= 0) rate = 1.0;

    int fft_size = 1024;
    int overlap = 3;

    /* Analyze input */
    cdp_spectral_data *spectral = cdp_spectral_analyze(
        input->data, input->length, input->channels,
        input->sample_rate, fft_size, overlap);

    if (spectral == NULL) return NULL;

    /* Initialize Lorenz attractor */
    lorenz_state lorenz;
    lorenz_init(&lorenz, seed);

    int num_bins = spectral->num_bins;
    double dt = rate * spectral->frame_time;

    /* Process each frame with chaotic modulation */
    for (int frame = 0; frame < spectral->num_frames; frame++) {
        /* Step the attractor */
        lorenz_step(&lorenz, dt);

        /* Normalize attractor outputs to [-1, 1] range */
        float mod_pitch = (float)(lorenz.y / 30.0);  /* y typically in [-30, 30] */
        float mod_amp = (float)(lorenz.x / 20.0);    /* x typically in [-20, 20] */

        /* Clamp to [-1, 1] */
        if (mod_pitch > 1.0f) mod_pitch = 1.0f;
        if (mod_pitch < -1.0f) mod_pitch = -1.0f;
        if (mod_amp > 1.0f) mod_amp = 1.0f;
        if (mod_amp < -1.0f) mod_amp = -1.0f;

        /* Apply scaled modulation */
        float pitch_mod = 1.0f + mod_pitch * (float)chaos_amount * 0.5f;
        float amp_mod = 1.0f + mod_amp * (float)chaos_amount * 0.3f;

        float *amp = spectral->frames[frame].data;
        float *freq = spectral->frames[frame].data + num_bins;

        for (int bin = 0; bin < num_bins; bin++) {
            freq[bin] *= pitch_mod;
            amp[bin] *= amp_mod;

            /* Keep frequencies positive */
            if (freq[bin] < 0) freq[bin] = 0;
        }
    }

    /* Synthesize output */
    size_t out_samples;
    float *output_data = cdp_spectral_synthesize(spectral, &out_samples);
    cdp_spectral_data_free(spectral);

    if (output_data == NULL) return NULL;

    return cdp_lib_buffer_from_data(output_data, out_samples,
                                    1, input->sample_rate);
}

cdp_lib_buffer* cdp_lib_brownian(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double step_size,
                                  double smoothing,
                                  int target,
                                  unsigned int seed) {
    if (ctx == NULL || input == NULL || input->data == NULL) return NULL;
    if (step_size < 0) step_size = 0.1;
    if (smoothing < 0) smoothing = 0;
    if (smoothing > 1) smoothing = 1;
    if (target < 0 || target > 2) target = 0;

    int fft_size = 1024;
    int overlap = 3;

    /* Analyze input */
    cdp_spectral_data *spectral = cdp_spectral_analyze(
        input->data, input->length, input->channels,
        input->sample_rate, fft_size, overlap);

    if (spectral == NULL) return NULL;

    unsigned int rng = seed;
    int num_bins = spectral->num_bins;

    /* Random walk state */
    double walk = 0.0;
    double smoothed = 0.0;

    /* Soft bounds for random walk */
    double max_walk = step_size * 100.0;

    for (int frame = 0; frame < spectral->num_frames; frame++) {
        /* Take a random step */
        double step = lcg_float_bipolar(&rng) * step_size;
        walk += step;

        /* Soft bounds - apply restoring force near limits */
        if (walk > max_walk * 0.8) {
            walk -= (walk - max_walk * 0.8) * 0.1;
        }
        if (walk < -max_walk * 0.8) {
            walk -= (walk + max_walk * 0.8) * 0.1;
        }

        /* Exponential smoothing */
        smoothed = smoothing * smoothed + (1.0 - smoothing) * walk;

        float *amp = spectral->frames[frame].data;
        float *freq = spectral->frames[frame].data + num_bins;

        switch (target) {
            case 0: {
                /* Pitch modulation (semitones) */
                float pitch_ratio = (float)pow(2.0, smoothed / 12.0);
                for (int bin = 0; bin < num_bins; bin++) {
                    freq[bin] *= pitch_ratio;
                    if (freq[bin] < 0) freq[bin] = 0;
                }
                break;
            }
            case 1: {
                /* Amplitude modulation (dB) */
                float gain = (float)pow(10.0, smoothed / 20.0);
                for (int bin = 0; bin < num_bins; bin++) {
                    amp[bin] *= gain;
                }
                break;
            }
            case 2: {
                /* Filter cutoff modulation - attenuate above/below threshold */
                float cutoff = (float)(1000.0 * pow(2.0, smoothed / 12.0));
                float freq_per_bin = spectral->sample_rate / spectral->fft_size;
                for (int bin = 0; bin < num_bins; bin++) {
                    float bin_freq = bin * freq_per_bin;
                    if (bin_freq > cutoff) {
                        float atten = cutoff / bin_freq;
                        if (atten < 0.01f) atten = 0.01f;
                        amp[bin] *= atten;
                    }
                }
                break;
            }
        }
    }

    /* Synthesize output */
    size_t out_samples;
    float *output_data = cdp_spectral_synthesize(spectral, &out_samples);
    cdp_spectral_data_free(spectral);

    if (output_data == NULL) return NULL;

    return cdp_lib_buffer_from_data(output_data, out_samples,
                                    1, input->sample_rate);
}

/*
 * Apply Hann window to a grain
 */
static void apply_hann_window(float *data, int size) {
    for (int i = 0; i < size; i++) {
        float window = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (size - 1)));
        data[i] *= window;
    }
}

cdp_lib_buffer* cdp_lib_crystal(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double density,
                                 double decay,
                                 double pitch_scatter,
                                 unsigned int seed) {
    if (ctx == NULL || input == NULL || input->data == NULL) return NULL;
    if (density <= 0) density = 50.0;
    if (decay <= 0) decay = 0.5;
    if (pitch_scatter < 0) pitch_scatter = 0;

    int sample_rate = input->sample_rate;
    size_t input_len = input->length;
    int channels = input->channels;

    /* Convert to mono for processing */
    float *mono;
    size_t mono_len;
    if (channels > 1) {
        mono_len = input_len / channels;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        for (size_t i = 0; i < mono_len; i++) {
            float sum = 0;
            for (int c = 0; c < channels; c++) {
                sum += input->data[i * channels + c];
            }
            mono[i] = sum / channels;
        }
    } else {
        mono_len = input_len;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        memcpy(mono, input->data, mono_len * sizeof(float));
    }

    /* Grain size: 20ms */
    int grain_size = (int)(0.020 * sample_rate);
    if (grain_size < 64) grain_size = 64;

    /* Number of echo layers */
    int num_echoes = 6;

    /* Output length: input + decay tail */
    size_t output_len = mono_len + (size_t)(decay * sample_rate);
    float *output = (float *)calloc(output_len, sizeof(float));
    if (output == NULL) {
        free(mono);
        return NULL;
    }

    unsigned int rng = seed;

    /* Time between grains */
    double grain_interval = 1.0 / density;
    int grain_hop = (int)(grain_interval * sample_rate);
    if (grain_hop < 1) grain_hop = 1;

    /* Process each grain position */
    for (size_t pos = 0; pos < mono_len; pos += grain_hop) {
        /* Extract grain */
        int actual_size = grain_size;
        if (pos + grain_size > mono_len) {
            actual_size = (int)(mono_len - pos);
        }
        if (actual_size < 64) continue;

        float *grain = (float *)malloc(actual_size * sizeof(float));
        if (grain == NULL) continue;

        memcpy(grain, mono + pos, actual_size * sizeof(float));
        apply_hann_window(grain, actual_size);

        /* Create echo layers */
        for (int echo = 0; echo <= num_echoes; echo++) {
            /* Calculate decay for this echo */
            float echo_amp = (float)pow(0.5, (double)echo / 2.0);

            /* Timing jitter for shimmer effect */
            int jitter = (int)(lcg_float_bipolar(&rng) * grain_size * 0.5);

            /* Pitch scatter */
            float pitch_ratio = 1.0f;
            if (pitch_scatter > 0 && echo > 0) {
                float scatter = lcg_float_bipolar(&rng) * (float)pitch_scatter;
                pitch_ratio = (float)pow(2.0, scatter / 12.0);
            }

            /* Echo delay increases geometrically */
            int echo_delay = (int)(decay * sample_rate * (1.0 - pow(0.7, echo)));

            /* Calculate output position */
            size_t out_pos = pos + echo_delay + jitter;
            if (out_pos >= output_len) continue;

            /* Resample grain if pitch scatter applied */
            if (fabsf(pitch_ratio - 1.0f) > 0.001f) {
                int new_size = (int)(actual_size / pitch_ratio);
                if (new_size < 32) new_size = 32;
                if (new_size > actual_size * 4) new_size = actual_size * 4;

                for (int i = 0; i < new_size && out_pos + i < output_len; i++) {
                    float src_pos = i * pitch_ratio;
                    int src_idx = (int)src_pos;
                    float frac = src_pos - src_idx;

                    if (src_idx + 1 < actual_size) {
                        float sample = grain[src_idx] * (1.0f - frac) +
                                       grain[src_idx + 1] * frac;
                        output[out_pos + i] += sample * echo_amp;
                    }
                }
            } else {
                /* No pitch change, direct copy */
                for (int i = 0; i < actual_size && out_pos + i < output_len; i++) {
                    output[out_pos + i] += grain[i] * echo_amp;
                }
            }
        }

        free(grain);
    }

    free(mono);

    /* Normalize output to prevent clipping */
    float peak = 0.0f;
    for (size_t i = 0; i < output_len; i++) {
        float abs_val = fabsf(output[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 0.95f / peak;
        for (size_t i = 0; i < output_len; i++) {
            output[i] *= norm;
        }
    }

    return cdp_lib_buffer_from_data(output, output_len, 1, sample_rate);
}

cdp_lib_buffer* cdp_lib_fractal(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 int depth,
                                 double pitch_ratio,
                                 double decay,
                                 unsigned int seed) {
    if (ctx == NULL || input == NULL || input->data == NULL) return NULL;
    if (depth < 1) depth = 1;
    if (depth > 6) depth = 6;
    if (pitch_ratio <= 0) pitch_ratio = 0.5;
    if (decay < 0) decay = 0;
    if (decay > 1) decay = 1;

    int sample_rate = input->sample_rate;
    size_t input_len = input->length;
    int channels = input->channels;

    /* Convert to mono */
    float *mono;
    size_t mono_len;
    if (channels > 1) {
        mono_len = input_len / channels;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        for (size_t i = 0; i < mono_len; i++) {
            float sum = 0;
            for (int c = 0; c < channels; c++) {
                sum += input->data[i * channels + c];
            }
            mono[i] = sum / channels;
        }
    } else {
        mono_len = input_len;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        memcpy(mono, input->data, mono_len * sizeof(float));
    }

    /* Output starts as copy of input */
    float *output = (float *)calloc(mono_len, sizeof(float));
    if (output == NULL) {
        free(mono);
        return NULL;
    }
    memcpy(output, mono, mono_len * sizeof(float));

    unsigned int rng = seed;

    /* Add fractal layers */
    float layer_amp = 1.0f;
    double current_ratio = 1.0;

    for (int d = 1; d <= depth; d++) {
        layer_amp *= (float)decay;
        current_ratio *= pitch_ratio;

        /* Calculate resampled length */
        size_t layer_len = (size_t)(mono_len / current_ratio);
        if (layer_len < 64) break;

        /* Add timing jitter */
        int jitter = (int)(lcg_float_bipolar(&rng) * mono_len * 0.05);

        /* Resample and add layer */
        for (size_t i = 0; i < mono_len; i++) {
            double src_pos = i * current_ratio;
            size_t src_idx = (size_t)src_pos;
            float frac = (float)(src_pos - src_idx);

            if (src_idx + 1 < mono_len) {
                float sample = mono[src_idx] * (1.0f - frac) +
                               mono[src_idx + 1] * frac;

                size_t out_idx = (size_t)((int)i + jitter);
                if (out_idx < mono_len) {
                    output[out_idx] += sample * layer_amp;
                }
            }
        }
    }

    free(mono);

    /* Normalize to prevent clipping */
    float peak = 0.0f;
    for (size_t i = 0; i < mono_len; i++) {
        float abs_val = fabsf(output[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 0.95f / peak;
        for (size_t i = 0; i < mono_len; i++) {
            output[i] *= norm;
        }
    }

    return cdp_lib_buffer_from_data(output, mono_len, 1, sample_rate);
}

cdp_lib_buffer* cdp_lib_quirk(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input,
                               double probability,
                               double intensity,
                               int mode,
                               unsigned int seed) {
    if (ctx == NULL || input == NULL || input->data == NULL) return NULL;
    if (probability < 0) probability = 0;
    if (probability > 1) probability = 1;
    if (intensity < 0) intensity = 0;
    if (intensity > 1) intensity = 1;
    if (mode < 0 || mode > 2) mode = 2;

    int sample_rate = input->sample_rate;
    size_t input_len = input->length;
    int channels = input->channels;

    /* Convert to mono */
    float *mono;
    size_t mono_len;
    if (channels > 1) {
        mono_len = input_len / channels;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        for (size_t i = 0; i < mono_len; i++) {
            float sum = 0;
            for (int c = 0; c < channels; c++) {
                sum += input->data[i * channels + c];
            }
            mono[i] = sum / channels;
        }
    } else {
        mono_len = input_len;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        memcpy(mono, input->data, mono_len * sizeof(float));
    }

    /* Allocate output (may be longer due to timing quirks) */
    size_t max_output = mono_len + (size_t)(mono_len * intensity * 0.5);
    float *output = (float *)calloc(max_output, sizeof(float));
    if (output == NULL) {
        free(mono);
        return NULL;
    }

    unsigned int rng = seed;

    /* Process in chunks */
    int chunk_size = sample_rate / 20;  /* 50ms chunks */
    size_t out_pos = 0;

    for (size_t pos = 0; pos < mono_len && out_pos < max_output; pos += chunk_size) {
        int actual_chunk = chunk_size;
        if (pos + actual_chunk > mono_len) {
            actual_chunk = (int)(mono_len - pos);
        }

        /* Check if quirk triggers */
        float roll = lcg_float(&rng);
        int do_quirk = (roll < probability);

        if (do_quirk) {
            float quirk_type = lcg_float(&rng);

            if ((mode == 0 || mode == 2) && quirk_type < 0.5f) {
                /* Pitch quirk: pitch shift the chunk */
                float pitch_shift = 1.0f + lcg_float_bipolar(&rng) * (float)intensity;
                int new_len = (int)(actual_chunk / pitch_shift);
                if (new_len < 8) new_len = 8;

                for (int i = 0; i < new_len && out_pos + i < max_output; i++) {
                    float src_pos_f = i * pitch_shift;
                    int src_idx = (int)src_pos_f;
                    float frac = src_pos_f - src_idx;

                    if (pos + src_idx + 1 < mono_len) {
                        output[out_pos + i] = mono[pos + src_idx] * (1.0f - frac) +
                                              mono[pos + src_idx + 1] * frac;
                    }
                }
                out_pos += new_len;
            } else if (mode == 1 || mode == 2) {
                /* Timing quirk: insert gap or repeat */
                if (lcg_float(&rng) < 0.5f) {
                    /* Insert silence gap */
                    int gap_len = (int)(actual_chunk * intensity * lcg_float(&rng));
                    out_pos += gap_len;
                } else {
                    /* Repeat chunk */
                    int repeats = 1 + (int)(lcg_float(&rng) * intensity * 2);
                    for (int r = 0; r < repeats; r++) {
                        for (int i = 0; i < actual_chunk && out_pos + i < max_output; i++) {
                            output[out_pos + i] = mono[pos + i];
                        }
                        out_pos += actual_chunk;
                    }
                }
            }
        } else {
            /* No quirk - copy normally */
            for (int i = 0; i < actual_chunk && out_pos + i < max_output; i++) {
                output[out_pos + i] = mono[pos + i];
            }
            out_pos += actual_chunk;
        }
    }

    free(mono);

    /* Trim output to actual length */
    size_t output_len = out_pos;
    if (output_len == 0) output_len = 1;

    float *trimmed = (float *)realloc(output, output_len * sizeof(float));
    if (trimmed == NULL) {
        free(output);
        return NULL;
    }

    return cdp_lib_buffer_from_data(trimmed, output_len, 1, sample_rate);
}

/*
 * Chirikov standard map state
 */
typedef struct {
    double theta;
    double p;
} chirikov_state;

static void chirikov_step(chirikov_state *s, double k) {
    s->p = fmod(s->p + k * sin(s->theta), 2.0 * M_PI);
    s->theta = fmod(s->theta + s->p, 2.0 * M_PI);

    /* Keep in range */
    while (s->theta < 0) s->theta += 2.0 * M_PI;
    while (s->theta >= 2.0 * M_PI) s->theta -= 2.0 * M_PI;
    while (s->p < -M_PI) s->p += 2.0 * M_PI;
    while (s->p >= M_PI) s->p -= 2.0 * M_PI;
}

cdp_lib_buffer* cdp_lib_chirikov(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double k_param,
                                  double mod_depth,
                                  double rate,
                                  unsigned int seed) {
    if (ctx == NULL || input == NULL || input->data == NULL) return NULL;
    if (k_param < 0.5) k_param = 0.5;
    if (k_param > 10.0) k_param = 10.0;
    if (mod_depth < 0) mod_depth = 0;
    if (mod_depth > 1) mod_depth = 1;
    if (rate <= 0) rate = 1.0;

    int fft_size = 1024;
    int overlap = 3;

    /* Analyze input */
    cdp_spectral_data *spectral = cdp_spectral_analyze(
        input->data, input->length, input->channels,
        input->sample_rate, fft_size, overlap);

    if (spectral == NULL) return NULL;

    /* Initialize Chirikov map from seed */
    chirikov_state chir;
    unsigned int rng = seed;
    chir.theta = lcg_float(&rng) * 2.0 * M_PI;
    chir.p = lcg_float_bipolar(&rng) * M_PI;

    /* Settle the map */
    for (int i = 0; i < 100; i++) {
        chirikov_step(&chir, k_param);
    }

    int num_bins = spectral->num_bins;
    double dt = rate * spectral->frame_time;

    /* Process each frame with Chirikov modulation */
    for (int frame = 0; frame < spectral->num_frames; frame++) {
        /* Step the map */
        chirikov_step(&chir, k_param * dt);

        /* Normalize outputs to [-1, 1] */
        float mod_pitch = (float)(chir.theta / M_PI - 1.0);  /* theta in [0, 2pi] -> [-1, 1] */
        float mod_amp = (float)(chir.p / M_PI);               /* p in [-pi, pi] -> [-1, 1] */

        /* Apply scaled modulation */
        float pitch_mod = 1.0f + mod_pitch * (float)mod_depth * 0.3f;
        float amp_mod = 1.0f + mod_amp * (float)mod_depth * 0.2f;

        float *amp = spectral->frames[frame].data;
        float *freq = spectral->frames[frame].data + num_bins;

        for (int bin = 0; bin < num_bins; bin++) {
            freq[bin] *= pitch_mod;
            amp[bin] *= amp_mod;

            if (freq[bin] < 0) freq[bin] = 0;
            if (amp[bin] < 0) amp[bin] = 0;
        }
    }

    /* Synthesize output */
    size_t out_samples;
    float *output_data = cdp_spectral_synthesize(spectral, &out_samples);
    cdp_spectral_data_free(spectral);

    if (output_data == NULL) return NULL;

    return cdp_lib_buffer_from_data(output_data, out_samples, 1, input->sample_rate);
}

cdp_lib_buffer* cdp_lib_cantor(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                int depth,
                                double duty_cycle,
                                double smooth_ms,
                                unsigned int seed) {
    if (ctx == NULL || input == NULL || input->data == NULL) return NULL;
    if (depth < 1) depth = 1;
    if (depth > 8) depth = 8;
    if (duty_cycle < 0) duty_cycle = 0;
    if (duty_cycle > 1) duty_cycle = 1;
    if (smooth_ms < 0) smooth_ms = 0;

    int sample_rate = input->sample_rate;
    size_t input_len = input->length;
    int channels = input->channels;

    /* Convert to mono */
    float *mono;
    size_t mono_len;
    if (channels > 1) {
        mono_len = input_len / channels;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        for (size_t i = 0; i < mono_len; i++) {
            float sum = 0;
            for (int c = 0; c < channels; c++) {
                sum += input->data[i * channels + c];
            }
            mono[i] = sum / channels;
        }
    } else {
        mono_len = input_len;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        memcpy(mono, input->data, mono_len * sizeof(float));
    }

    /* Generate Cantor set mask */
    float *mask = (float *)malloc(mono_len * sizeof(float));
    if (mask == NULL) {
        free(mono);
        return NULL;
    }

    /* Initialize mask to 1 (all audio passes) */
    for (size_t i = 0; i < mono_len; i++) {
        mask[i] = 1.0f;
    }

    unsigned int rng = seed;

    /* Apply Cantor set recursively */
    for (int d = 0; d < depth; d++) {
        /* Number of segments at this depth */
        int num_segments = 1 << d;  /* 2^d */
        size_t seg_len = mono_len / num_segments;
        if (seg_len < 3) break;

        /* Size of middle third to remove */
        size_t third_len = seg_len / 3;
        size_t middle_start = third_len;
        size_t middle_end = 2 * third_len;

        /* Add variation based on duty_cycle */
        double remove_ratio = 1.0 - duty_cycle;

        for (int seg = 0; seg < num_segments; seg++) {
            size_t seg_start = seg * seg_len;

            /* Add random variation */
            float variation = lcg_float(&rng) * 0.2f - 0.1f;
            float seg_remove = (float)(remove_ratio + variation);
            if (seg_remove < 0) seg_remove = 0;
            if (seg_remove > 1) seg_remove = 1;

            /* Apply removal to middle third */
            for (size_t i = middle_start; i < middle_end && seg_start + i < mono_len; i++) {
                mask[seg_start + i] *= (1.0f - seg_remove);
            }
        }
    }

    /* Apply smoothing */
    int smooth_samples = (int)(smooth_ms * sample_rate / 1000.0);
    if (smooth_samples > 1) {
        float *smooth_mask = (float *)malloc(mono_len * sizeof(float));
        if (smooth_mask != NULL) {
            for (size_t i = 0; i < mono_len; i++) {
                float sum = 0;
                int count = 0;
                for (int j = -smooth_samples; j <= smooth_samples; j++) {
                    size_t idx = (size_t)((int)i + j);
                    if (idx < mono_len) {
                        sum += mask[idx];
                        count++;
                    }
                }
                smooth_mask[i] = sum / count;
            }
            memcpy(mask, smooth_mask, mono_len * sizeof(float));
            free(smooth_mask);
        }
    }

    /* Apply mask */
    float *output = (float *)malloc(mono_len * sizeof(float));
    if (output == NULL) {
        free(mono);
        free(mask);
        return NULL;
    }

    for (size_t i = 0; i < mono_len; i++) {
        output[i] = mono[i] * mask[i];
    }

    free(mono);
    free(mask);

    return cdp_lib_buffer_from_data(output, mono_len, 1, sample_rate);
}

cdp_lib_buffer* cdp_lib_cascade(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 int num_echoes,
                                 double delay_ms,
                                 double pitch_decay,
                                 double amp_decay,
                                 double filter_decay,
                                 unsigned int seed) {
    if (ctx == NULL || input == NULL || input->data == NULL) return NULL;
    if (num_echoes < 1) num_echoes = 1;
    if (num_echoes > 12) num_echoes = 12;
    if (delay_ms < 10) delay_ms = 10;
    if (pitch_decay <= 0) pitch_decay = 0.95;
    if (amp_decay < 0) amp_decay = 0;
    if (amp_decay > 1) amp_decay = 1;
    if (filter_decay < 0) filter_decay = 0;
    if (filter_decay > 1) filter_decay = 1;

    int sample_rate = input->sample_rate;
    size_t input_len = input->length;
    int channels = input->channels;

    /* Convert to mono */
    float *mono;
    size_t mono_len;
    if (channels > 1) {
        mono_len = input_len / channels;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        for (size_t i = 0; i < mono_len; i++) {
            float sum = 0;
            for (int c = 0; c < channels; c++) {
                sum += input->data[i * channels + c];
            }
            mono[i] = sum / channels;
        }
    } else {
        mono_len = input_len;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        memcpy(mono, input->data, mono_len * sizeof(float));
    }

    int delay_samples = (int)(delay_ms * sample_rate / 1000.0);
    size_t output_len = mono_len + (size_t)(delay_samples * num_echoes);
    float *output = (float *)calloc(output_len, sizeof(float));
    if (output == NULL) {
        free(mono);
        return NULL;
    }

    /* Copy original */
    memcpy(output, mono, mono_len * sizeof(float));

    unsigned int rng = seed;

    /* Add cascading echoes */
    double current_pitch = 1.0;
    float current_amp = 1.0f;
    double current_filter = 1.0;

    for (int echo = 1; echo <= num_echoes; echo++) {
        current_pitch *= pitch_decay;
        current_amp *= (float)amp_decay;
        current_filter *= filter_decay;

        int echo_delay = delay_samples * echo;
        /* Add jitter */
        echo_delay += (int)(lcg_float_bipolar(&rng) * delay_samples * 0.1);

        /* Calculate resampled length for pitch */
        size_t echo_len = (size_t)(mono_len / current_pitch);
        if (echo_len < 64) continue;

        /* Simple one-pole lowpass for filter decay */
        float filter_coef = (float)(1.0 - current_filter * 0.5);
        float prev_sample = 0;

        for (size_t i = 0; i < echo_len; i++) {
            double src_pos = i * current_pitch;
            size_t src_idx = (size_t)src_pos;
            float frac = (float)(src_pos - src_idx);

            if (src_idx + 1 < mono_len) {
                float sample = mono[src_idx] * (1.0f - frac) +
                               mono[src_idx + 1] * frac;

                /* Apply lowpass */
                sample = filter_coef * sample + (1.0f - filter_coef) * prev_sample;
                prev_sample = sample;

                size_t out_idx = echo_delay + i;
                if (out_idx < output_len) {
                    output[out_idx] += sample * current_amp;
                }
            }
        }
    }

    free(mono);

    /* Normalize */
    float peak = 0.0f;
    for (size_t i = 0; i < output_len; i++) {
        float abs_val = fabsf(output[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 0.95f / peak;
        for (size_t i = 0; i < output_len; i++) {
            output[i] *= norm;
        }
    }

    return cdp_lib_buffer_from_data(output, output_len, 1, sample_rate);
}

cdp_lib_buffer* cdp_lib_fracture(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double fragment_ms,
                                  double gap_ratio,
                                  double scatter,
                                  unsigned int seed) {
    if (ctx == NULL || input == NULL || input->data == NULL) return NULL;
    if (fragment_ms < 10) fragment_ms = 10;
    if (gap_ratio < 0) gap_ratio = 0;
    if (gap_ratio > 2) gap_ratio = 2;
    if (scatter < 0) scatter = 0;
    if (scatter > 1) scatter = 1;

    int sample_rate = input->sample_rate;
    size_t input_len = input->length;
    int channels = input->channels;

    /* Convert to mono */
    float *mono;
    size_t mono_len;
    if (channels > 1) {
        mono_len = input_len / channels;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        for (size_t i = 0; i < mono_len; i++) {
            float sum = 0;
            for (int c = 0; c < channels; c++) {
                sum += input->data[i * channels + c];
            }
            mono[i] = sum / channels;
        }
    } else {
        mono_len = input_len;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        memcpy(mono, input->data, mono_len * sizeof(float));
    }

    int fragment_samples = (int)(fragment_ms * sample_rate / 1000.0);
    int num_fragments = (int)(mono_len / fragment_samples) + 1;

    /* Create fragment order array */
    int *order = (int *)malloc(num_fragments * sizeof(int));
    if (order == NULL) {
        free(mono);
        return NULL;
    }
    for (int i = 0; i < num_fragments; i++) {
        order[i] = i;
    }

    /* Shuffle based on scatter */
    unsigned int rng = seed;
    int num_swaps = (int)(num_fragments * scatter);
    for (int s = 0; s < num_swaps; s++) {
        int i = (int)(lcg_float(&rng) * num_fragments);
        int j = (int)(lcg_float(&rng) * num_fragments);
        if (i < num_fragments && j < num_fragments) {
            int temp = order[i];
            order[i] = order[j];
            order[j] = temp;
        }
    }

    /* Calculate output length with gaps */
    size_t output_len = (size_t)(mono_len * (1.0 + gap_ratio));
    float *output = (float *)calloc(output_len, sizeof(float));
    if (output == NULL) {
        free(mono);
        free(order);
        return NULL;
    }

    /* Place fragments with gaps */
    size_t out_pos = 0;
    for (int f = 0; f < num_fragments && out_pos < output_len; f++) {
        int src_frag = order[f];
        size_t src_start = src_frag * fragment_samples;
        int frag_len = fragment_samples;
        if (src_start + frag_len > mono_len) {
            frag_len = (int)(mono_len - src_start);
        }
        if (frag_len <= 0) continue;

        /* Apply short fade in/out */
        int fade_len = frag_len / 10;
        if (fade_len < 8) fade_len = 8;
        if (fade_len > frag_len / 2) fade_len = frag_len / 2;

        for (int i = 0; i < frag_len && out_pos + i < output_len; i++) {
            float sample = mono[src_start + i];

            /* Fade envelope */
            float env = 1.0f;
            if (i < fade_len) {
                env = (float)i / fade_len;
            } else if (i >= frag_len - fade_len) {
                env = (float)(frag_len - i) / fade_len;
            }

            output[out_pos + i] = sample * env;
        }
        out_pos += frag_len;

        /* Add random gap */
        int gap = (int)(fragment_samples * gap_ratio * lcg_float(&rng));
        out_pos += gap;
    }

    free(mono);
    free(order);

    /* Trim to actual length */
    if (out_pos < output_len) {
        float *trimmed = (float *)realloc(output, out_pos * sizeof(float));
        if (trimmed != NULL) {
            output = trimmed;
            output_len = out_pos;
        }
    }

    return cdp_lib_buffer_from_data(output, output_len, 1, sample_rate);
}

cdp_lib_buffer* cdp_lib_tesselate(cdp_lib_ctx* ctx,
                                   const cdp_lib_buffer* input,
                                   double tile_ms,
                                   int pattern,
                                   double overlap,
                                   double transform,
                                   unsigned int seed) {
    if (ctx == NULL || input == NULL || input->data == NULL) return NULL;
    if (tile_ms < 20) tile_ms = 20;
    if (pattern < 0 || pattern > 3) pattern = 0;
    if (overlap < 0) overlap = 0;
    if (overlap > 0.5) overlap = 0.5;
    if (transform < 0) transform = 0;
    if (transform > 1) transform = 1;

    int sample_rate = input->sample_rate;
    size_t input_len = input->length;
    int channels = input->channels;

    /* Convert to mono */
    float *mono;
    size_t mono_len;
    if (channels > 1) {
        mono_len = input_len / channels;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        for (size_t i = 0; i < mono_len; i++) {
            float sum = 0;
            for (int c = 0; c < channels; c++) {
                sum += input->data[i * channels + c];
            }
            mono[i] = sum / channels;
        }
    } else {
        mono_len = input_len;
        mono = (float *)malloc(mono_len * sizeof(float));
        if (mono == NULL) return NULL;
        memcpy(mono, input->data, mono_len * sizeof(float));
    }

    int tile_samples = (int)(tile_ms * sample_rate / 1000.0);
    int hop_samples = (int)(tile_samples * (1.0 - overlap));
    if (hop_samples < 1) hop_samples = 1;
    int num_tiles = (int)(mono_len / hop_samples);

    /* Output same length as input */
    float *output = (float *)calloc(mono_len, sizeof(float));
    if (output == NULL) {
        free(mono);
        return NULL;
    }

    unsigned int rng = seed;

    /* Generate tile indices based on pattern */
    for (int t = 0; t < num_tiles; t++) {
        int src_tile;
        int do_reverse = 0;
        float amp_scale = 1.0f;

        switch (pattern) {
            case 0:  /* Repeat */
                src_tile = t % num_tiles;
                break;
            case 1:  /* Mirror */
                if ((t / num_tiles) % 2 == 1) {
                    src_tile = num_tiles - 1 - (t % num_tiles);
                    do_reverse = 1;
                } else {
                    src_tile = t % num_tiles;
                }
                break;
            case 2:  /* Rotate */
                src_tile = (t + t / num_tiles) % num_tiles;
                break;
            case 3:  /* Random */
            default:
                src_tile = (int)(lcg_float(&rng) * num_tiles);
                do_reverse = lcg_float(&rng) < 0.3f;
                amp_scale = 0.8f + lcg_float(&rng) * 0.4f;
                break;
        }

        if (src_tile < 0) src_tile = 0;
        if (src_tile >= num_tiles) src_tile = num_tiles - 1;

        size_t src_start = src_tile * hop_samples;
        size_t dst_start = t * hop_samples;

        /* Apply transform variations */
        if (transform > 0 && pattern != 3) {
            if (lcg_float(&rng) < transform) {
                do_reverse = !do_reverse;
            }
            amp_scale = 1.0f + lcg_float_bipolar(&rng) * (float)transform * 0.3f;
        }

        /* Copy tile with Hann window */
        for (int i = 0; i < tile_samples; i++) {
            size_t src_idx = src_start + (do_reverse ? (tile_samples - 1 - i) : i);
            size_t dst_idx = dst_start + i;

            if (src_idx >= mono_len || dst_idx >= mono_len) continue;

            /* Hann window for overlap-add */
            float window = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (tile_samples - 1)));
            output[dst_idx] += mono[src_idx] * window * amp_scale;
        }
    }

    free(mono);

    /* Normalize */
    float peak = 0.0f;
    for (size_t i = 0; i < mono_len; i++) {
        float abs_val = fabsf(output[i]);
        if (abs_val > peak) peak = abs_val;
    }
    if (peak > 1.0f) {
        float norm = 0.95f / peak;
        for (size_t i = 0; i < mono_len; i++) {
            output[i] *= norm;
        }
    }

    return cdp_lib_buffer_from_data(output, mono_len, 1, sample_rate);
}
