/*
 * CDP Synthesis - Implementation
 *
 * Implements synthesis operations for generating audio waveforms.
 */

#include "cdp_synth.h"
#include "cdp_lib_internal.h"

/*
 * Synth Wave - generate basic waveforms.
 */
cdp_lib_buffer* cdp_lib_synth_wave(cdp_lib_ctx* ctx,
                                    int waveform,
                                    double frequency,
                                    double amplitude,
                                    double duration,
                                    int sample_rate,
                                    int channels)
{
    if (!ctx) {
        return NULL;
    }

    /* Validate parameters */
    if (waveform < 0 || waveform > 4) {
        cdp_lib_set_error(ctx, "synth_wave: invalid waveform type (0-4)");
        return NULL;
    }
    if (frequency < 20.0) frequency = 20.0;
    if (frequency > 20000.0) frequency = 20000.0;
    if (amplitude < 0.0) amplitude = 0.0;
    if (amplitude > 1.0) amplitude = 1.0;
    if (duration < 0.001) duration = 0.001;
    if (duration > 3600.0) duration = 3600.0;
    if (sample_rate < 8000) sample_rate = 8000;
    if (sample_rate > 192000) sample_rate = 192000;
    if (channels < 1) channels = 1;
    if (channels > 2) channels = 2;

    /* Calculate number of frames */
    size_t num_frames = (size_t)(duration * sample_rate);
    if (num_frames < 1) num_frames = 1;

    /* Allocate output buffer */
    float* out_data = (float*)calloc(num_frames * channels, sizeof(float));
    if (!out_data) {
        cdp_lib_set_error(ctx, "synth_wave: memory allocation failed");
        return NULL;
    }

    /* Calculate phase increment per sample */
    double phase_incr = 2.0 * M_PI * frequency / sample_rate;
    double phase = 0.0;

    /* Generate waveform */
    for (size_t i = 0; i < num_frames; i++) {
        float sample = 0.0f;

        switch (waveform) {
        case CDP_WAVE_SINE:
            sample = (float)(sin(phase) * amplitude);
            break;

        case CDP_WAVE_SQUARE:
            sample = (phase < M_PI) ? (float)amplitude : (float)(-amplitude);
            break;

        case CDP_WAVE_SAW:
            /* Sawtooth: rises from -1 to +1 over the period */
            sample = (float)(((phase / M_PI) - 1.0) * amplitude);
            break;

        case CDP_WAVE_RAMP:
            /* Ramp (reverse sawtooth): falls from +1 to -1 over the period */
            sample = (float)((1.0 - (phase / M_PI)) * amplitude);
            break;

        case CDP_WAVE_TRIANGLE:
            /* Triangle: rises to peak at PI/2, falls to trough at 3*PI/2 */
            if (phase < M_PI) {
                /* First half: -1 to +1 */
                sample = (float)(((2.0 * phase / M_PI) - 1.0) * amplitude);
            } else {
                /* Second half: +1 to -1 */
                sample = (float)((3.0 - (2.0 * phase / M_PI)) * amplitude);
            }
            break;
        }

        /* Write to all channels */
        for (int ch = 0; ch < channels; ch++) {
            out_data[i * channels + ch] = sample;
        }

        /* Advance phase */
        phase += phase_incr;
        while (phase >= 2.0 * M_PI) {
            phase -= 2.0 * M_PI;
        }
    }

    /* Apply short fade in/out to avoid clicks (2ms) */
    size_t fade_samples = (size_t)(0.002 * sample_rate);
    if (fade_samples > num_frames / 4) fade_samples = num_frames / 4;
    if (fade_samples < 1) fade_samples = 1;

    for (size_t i = 0; i < fade_samples; i++) {
        double env = (double)i / (double)fade_samples;
        for (int ch = 0; ch < channels; ch++) {
            out_data[i * channels + ch] *= (float)env;
            out_data[(num_frames - 1 - i) * channels + ch] *= (float)env;
        }
    }

    cdp_lib_buffer* output = cdp_lib_buffer_from_data(out_data, num_frames * channels, channels, sample_rate);
    if (!output) {
        free(out_data);
        cdp_lib_set_error(ctx, "synth_wave: failed to create output buffer");
        return NULL;
    }

    return output;
}

/*
 * Synth Noise - generate white or pink noise.
 */
cdp_lib_buffer* cdp_lib_synth_noise(cdp_lib_ctx* ctx,
                                     int pink,
                                     double amplitude,
                                     double duration,
                                     int sample_rate,
                                     int channels,
                                     unsigned int seed)
{
    if (!ctx) {
        return NULL;
    }

    /* Validate parameters */
    if (amplitude < 0.0) amplitude = 0.0;
    if (amplitude > 1.0) amplitude = 1.0;
    if (duration < 0.001) duration = 0.001;
    if (duration > 3600.0) duration = 3600.0;
    if (sample_rate < 8000) sample_rate = 8000;
    if (sample_rate > 192000) sample_rate = 192000;
    if (channels < 1) channels = 1;
    if (channels > 2) channels = 2;

    /* Initialize random */
    cdp_lib_seed(ctx, seed);

    /* Calculate number of frames */
    size_t num_frames = (size_t)(duration * sample_rate);
    if (num_frames < 1) num_frames = 1;

    /* Allocate output buffer */
    float* out_data = (float*)calloc(num_frames * channels, sizeof(float));
    if (!out_data) {
        cdp_lib_set_error(ctx, "synth_noise: memory allocation failed");
        return NULL;
    }

    /* Pink noise filter state (Voss-McCartney algorithm approximation) */
    double b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;

    /* Generate noise */
    for (size_t i = 0; i < num_frames; i++) {
        float sample;

        if (pink) {
            /* Pink noise using Paul Kellet's refined method */
            double white = cdp_lib_random(ctx) * 2.0 - 1.0;

            b0 = 0.99886 * b0 + white * 0.0555179;
            b1 = 0.99332 * b1 + white * 0.0750759;
            b2 = 0.96900 * b2 + white * 0.1538520;
            b3 = 0.86650 * b3 + white * 0.3104856;
            b4 = 0.55000 * b4 + white * 0.5329522;
            b5 = -0.7616 * b5 - white * 0.0168980;

            double pink_val = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
            b6 = white * 0.115926;

            /* Normalize pink noise (it tends to be louder) */
            sample = (float)(pink_val * 0.11 * amplitude);
        } else {
            /* White noise */
            sample = (float)((cdp_lib_random(ctx) * 2.0 - 1.0) * amplitude);
        }

        /* Write to all channels (different noise per channel for stereo) */
        out_data[i * channels] = sample;
        if (channels > 1) {
            if (pink) {
                /* Generate independent pink noise for right channel */
                double white = cdp_lib_random(ctx) * 2.0 - 1.0;
                b0 = 0.99886 * b0 + white * 0.0555179;
                b1 = 0.99332 * b1 + white * 0.0750759;
                b2 = 0.96900 * b2 + white * 0.1538520;
                b3 = 0.86650 * b3 + white * 0.3104856;
                b4 = 0.55000 * b4 + white * 0.5329522;
                b5 = -0.7616 * b5 - white * 0.0168980;
                double pink_val = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
                b6 = white * 0.115926;
                out_data[i * channels + 1] = (float)(pink_val * 0.11 * amplitude);
            } else {
                out_data[i * channels + 1] = (float)((cdp_lib_random(ctx) * 2.0 - 1.0) * amplitude);
            }
        }
    }

    /* Apply short fade in/out to avoid clicks (2ms) */
    size_t fade_samples = (size_t)(0.002 * sample_rate);
    if (fade_samples > num_frames / 4) fade_samples = num_frames / 4;
    if (fade_samples < 1) fade_samples = 1;

    for (size_t i = 0; i < fade_samples; i++) {
        double env = (double)i / (double)fade_samples;
        for (int ch = 0; ch < channels; ch++) {
            out_data[i * channels + ch] *= (float)env;
            out_data[(num_frames - 1 - i) * channels + ch] *= (float)env;
        }
    }

    cdp_lib_buffer* output = cdp_lib_buffer_from_data(out_data, num_frames * channels, channels, sample_rate);
    if (!output) {
        free(out_data);
        cdp_lib_set_error(ctx, "synth_noise: failed to create output buffer");
        return NULL;
    }

    return output;
}

/*
 * Synth Click - generate click track.
 */
cdp_lib_buffer* cdp_lib_synth_click(cdp_lib_ctx* ctx,
                                     double tempo,
                                     int beats_per_bar,
                                     double duration,
                                     double click_freq,
                                     double click_dur_ms,
                                     int sample_rate)
{
    if (!ctx) {
        return NULL;
    }

    /* Validate parameters */
    if (tempo < 20.0) tempo = 20.0;
    if (tempo > 400.0) tempo = 400.0;
    if (beats_per_bar < 0) beats_per_bar = 0;
    if (beats_per_bar > 16) beats_per_bar = 16;
    if (duration < 0.1) duration = 0.1;
    if (duration > 3600.0) duration = 3600.0;
    if (click_freq < 200.0) click_freq = 200.0;
    if (click_freq > 8000.0) click_freq = 8000.0;
    if (click_dur_ms < 1.0) click_dur_ms = 1.0;
    if (click_dur_ms > 100.0) click_dur_ms = 100.0;
    if (sample_rate < 8000) sample_rate = 8000;
    if (sample_rate > 192000) sample_rate = 192000;

    /* Calculate number of frames (mono output) */
    size_t num_frames = (size_t)(duration * sample_rate);
    if (num_frames < 1) num_frames = 1;

    /* Allocate output buffer */
    float* out_data = (float*)calloc(num_frames, sizeof(float));
    if (!out_data) {
        cdp_lib_set_error(ctx, "synth_click: memory allocation failed");
        return NULL;
    }

    /* Calculate timing */
    double beat_duration = 60.0 / tempo;  /* seconds per beat */
    size_t samples_per_beat = (size_t)(beat_duration * sample_rate);
    size_t click_samples = (size_t)(click_dur_ms * 0.001 * sample_rate);
    if (click_samples > samples_per_beat / 2) click_samples = samples_per_beat / 2;

    /* Phase increment for click tone */
    double phase_incr = 2.0 * M_PI * click_freq / sample_rate;

    /* Generate clicks */
    size_t beat_count = 0;
    size_t next_beat_sample = 0;

    while (next_beat_sample < num_frames) {
        /* Determine click amplitude (accent on beat 1 of bar) */
        double click_amp = 0.8;
        if (beats_per_bar > 0 && (beat_count % beats_per_bar) == 0) {
            click_amp = 1.0;  /* Accent */
        }

        /* Generate click at this beat position */
        double phase = 0.0;
        for (size_t j = 0; j < click_samples && (next_beat_sample + j) < num_frames; j++) {
            /* Sine tone with exponential decay envelope */
            double env = exp(-5.0 * (double)j / (double)click_samples);
            float sample = (float)(sin(phase) * click_amp * env);
            out_data[next_beat_sample + j] += sample;
            phase += phase_incr;
        }

        /* Move to next beat */
        beat_count++;
        next_beat_sample = (size_t)(beat_count * beat_duration * sample_rate);
    }

    /* Normalize if needed */
    float max_val = 0;
    for (size_t i = 0; i < num_frames; i++) {
        float abs_val = fabsf(out_data[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    if (max_val > 0.95f) {
        float scale = 0.95f / max_val;
        for (size_t i = 0; i < num_frames; i++) {
            out_data[i] *= scale;
        }
    }

    cdp_lib_buffer* output = cdp_lib_buffer_from_data(out_data, num_frames, 1, sample_rate);
    if (!output) {
        free(out_data);
        cdp_lib_set_error(ctx, "synth_click: failed to create output buffer");
        return NULL;
    }

    return output;
}

/*
 * Helper: Convert MIDI note to frequency.
 * MIDI 69 = A4 = 440 Hz
 */
static double midi_to_freq(double midi_note)
{
    return 440.0 * pow(2.0, (midi_note - 69.0) / 12.0);
}

/*
 * Synth Chord - generate chord from MIDI notes.
 */
cdp_lib_buffer* cdp_lib_synth_chord(cdp_lib_ctx* ctx,
                                     const double* midi_notes,
                                     int num_notes,
                                     double amplitude,
                                     double duration,
                                     double detune_cents,
                                     int sample_rate,
                                     int channels)
{
    if (!ctx || !midi_notes) {
        if (ctx) cdp_lib_set_error(ctx, "synth_chord: invalid parameters");
        return NULL;
    }

    /* Validate parameters */
    if (num_notes < 1) num_notes = 1;
    if (num_notes > 16) num_notes = 16;
    if (amplitude < 0.0) amplitude = 0.0;
    if (amplitude > 1.0) amplitude = 1.0;
    if (duration < 0.001) duration = 0.001;
    if (duration > 3600.0) duration = 3600.0;
    if (detune_cents < 0.0) detune_cents = 0.0;
    if (detune_cents > 50.0) detune_cents = 50.0;
    if (sample_rate < 8000) sample_rate = 8000;
    if (sample_rate > 192000) sample_rate = 192000;
    if (channels < 1) channels = 1;
    if (channels > 2) channels = 2;

    /* Calculate number of frames */
    size_t num_frames = (size_t)(duration * sample_rate);
    if (num_frames < 1) num_frames = 1;

    /* Allocate output buffer */
    float* out_data = (float*)calloc(num_frames * channels, sizeof(float));
    if (!out_data) {
        cdp_lib_set_error(ctx, "synth_chord: memory allocation failed");
        return NULL;
    }

    /* Allocate phase accumulators for each note */
    double* phases = (double*)calloc(num_notes, sizeof(double));
    double* freqs = (double*)malloc(num_notes * sizeof(double));
    if (!phases || !freqs) {
        free(out_data);
        if (phases) free(phases);
        if (freqs) free(freqs);
        cdp_lib_set_error(ctx, "synth_chord: memory allocation failed");
        return NULL;
    }

    /* Convert MIDI notes to frequencies with optional detuning */
    for (int n = 0; n < num_notes; n++) {
        double midi = midi_notes[n];
        /* Clamp MIDI note to valid range */
        if (midi < 0.0) midi = 0.0;
        if (midi > 127.0) midi = 127.0;

        /* Apply detuning: alternate slightly sharp/flat for richness */
        double detune_semitones = 0.0;
        if (detune_cents > 0.0 && num_notes > 1) {
            /* Spread detuning: even notes sharp, odd notes flat */
            double detune_factor = (n % 2 == 0) ? 1.0 : -1.0;
            /* Scale detuning by note position for variation */
            double position_scale = 0.5 + 0.5 * ((double)n / (double)(num_notes - 1));
            detune_semitones = detune_factor * (detune_cents / 100.0) * position_scale;
        }

        freqs[n] = midi_to_freq(midi + detune_semitones);
        phases[n] = 0.0;
    }

    /* Per-note amplitude (mix equally) */
    double note_amp = amplitude / sqrt((double)num_notes);  /* Equal power mixing */

    /* Generate samples */
    for (size_t i = 0; i < num_frames; i++) {
        double sample = 0.0;

        /* Sum all notes */
        for (int n = 0; n < num_notes; n++) {
            sample += sin(phases[n]) * note_amp;

            /* Advance phase */
            phases[n] += 2.0 * M_PI * freqs[n] / sample_rate;
            while (phases[n] >= 2.0 * M_PI) {
                phases[n] -= 2.0 * M_PI;
            }
        }

        /* Write to all channels */
        for (int ch = 0; ch < channels; ch++) {
            out_data[i * channels + ch] = (float)sample;
        }
    }

    free(phases);
    free(freqs);

    /* Apply fade in/out to avoid clicks (5ms) */
    size_t fade_samples = (size_t)(0.005 * sample_rate);
    if (fade_samples > num_frames / 4) fade_samples = num_frames / 4;
    if (fade_samples < 1) fade_samples = 1;

    for (size_t i = 0; i < fade_samples; i++) {
        double env = (double)i / (double)fade_samples;
        for (int ch = 0; ch < channels; ch++) {
            out_data[i * channels + ch] *= (float)env;
            out_data[(num_frames - 1 - i) * channels + ch] *= (float)env;
        }
    }

    /* Normalize if needed */
    float max_val = 0;
    for (size_t i = 0; i < num_frames * channels; i++) {
        float abs_val = fabsf(out_data[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    if (max_val > 0.95f) {
        float scale = 0.95f / max_val;
        for (size_t i = 0; i < num_frames * channels; i++) {
            out_data[i] *= scale;
        }
    }

    cdp_lib_buffer* output = cdp_lib_buffer_from_data(out_data, num_frames * channels, channels, sample_rate);
    if (!output) {
        free(out_data);
        cdp_lib_set_error(ctx, "synth_chord: failed to create output buffer");
        return NULL;
    }

    return output;
}
