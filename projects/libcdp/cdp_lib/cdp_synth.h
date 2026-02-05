/*
 * CDP Synthesis - Waveform generation functions
 *
 * Implements synthesis operations for generating audio waveforms
 * directly in memory without file I/O.
 */

#ifndef CDP_SYNTH_H
#define CDP_SYNTH_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Waveform types for synth_wave */
#define CDP_WAVE_SINE     0
#define CDP_WAVE_SQUARE   1
#define CDP_WAVE_SAW      2
#define CDP_WAVE_RAMP     3
#define CDP_WAVE_TRIANGLE 4

/*
 * Synth Wave - generate basic waveforms.
 *
 * Synthesizes standard waveforms (sine, square, saw, ramp, triangle)
 * at specified frequency and duration.
 *
 * Args:
 *   ctx: Library context
 *   waveform: Waveform type (CDP_WAVE_SINE, CDP_WAVE_SQUARE, etc.)
 *   frequency: Frequency in Hz (20 to 20000)
 *   amplitude: Peak amplitude (0.0 to 1.0)
 *   duration: Duration in seconds (0.001 to 3600)
 *   sample_rate: Sample rate in Hz (default 44100)
 *   channels: Number of output channels (1 or 2)
 *
 * Returns: New buffer with synthesized waveform, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_synth_wave(cdp_lib_ctx* ctx,
                                    int waveform,
                                    double frequency,
                                    double amplitude,
                                    double duration,
                                    int sample_rate,
                                    int channels);

/*
 * Synth Noise - generate noise.
 *
 * Synthesizes white or pink noise at specified duration.
 *
 * Args:
 *   ctx: Library context
 *   pink: If non-zero, generate pink noise; otherwise white noise
 *   amplitude: Peak amplitude (0.0 to 1.0)
 *   duration: Duration in seconds (0.001 to 3600)
 *   sample_rate: Sample rate in Hz (default 44100)
 *   channels: Number of output channels (1 or 2)
 *   seed: Random seed (0 = use time)
 *
 * Returns: New buffer with synthesized noise, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_synth_noise(cdp_lib_ctx* ctx,
                                     int pink,
                                     double amplitude,
                                     double duration,
                                     int sample_rate,
                                     int channels,
                                     unsigned int seed);

/*
 * Synth Click - generate click track.
 *
 * Synthesizes a click track at specified tempo and duration.
 *
 * Args:
 *   ctx: Library context
 *   tempo: Tempo in BPM (20 to 400)
 *   beats_per_bar: Beats per bar for accent pattern (1 to 16, 0 = no accent)
 *   duration: Duration in seconds (0.1 to 3600)
 *   click_freq: Click frequency in Hz (200 to 8000, default 1000)
 *   click_dur_ms: Click duration in milliseconds (1 to 100, default 10)
 *   sample_rate: Sample rate in Hz (default 44100)
 *
 * Returns: New mono buffer with click track, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_synth_click(cdp_lib_ctx* ctx,
                                     double tempo,
                                     int beats_per_bar,
                                     double duration,
                                     double click_freq,
                                     double click_dur_ms,
                                     int sample_rate);

/*
 * Synth Chord - generate chord from MIDI notes.
 *
 * Synthesizes a chord by mixing multiple sine waves at specified MIDI pitches.
 * Includes optional detuning for a richer, more natural sound.
 *
 * Args:
 *   ctx: Library context
 *   midi_notes: Array of MIDI note numbers (0-127, where 60 = middle C)
 *   num_notes: Number of notes in the chord (1 to 16)
 *   amplitude: Peak amplitude (0.0 to 1.0)
 *   duration: Duration in seconds (0.001 to 3600)
 *   detune_cents: Detuning amount in cents (0-50, default 0)
 *                 Adds slight pitch variations between notes for richer sound
 *   sample_rate: Sample rate in Hz (default 44100)
 *   channels: Number of output channels (1 or 2)
 *
 * Returns: New buffer with synthesized chord, or NULL on error.
 *
 * Note: MIDI note to frequency: freq = 440 * 2^((note - 69) / 12)
 */
cdp_lib_buffer* cdp_lib_synth_chord(cdp_lib_ctx* ctx,
                                     const double* midi_notes,
                                     int num_notes,
                                     double amplitude,
                                     double duration,
                                     double detune_cents,
                                     int sample_rate,
                                     int channels);

#ifdef __cplusplus
}
#endif

#endif /* CDP_SYNTH_H */
