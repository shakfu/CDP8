/*
 * CDP Playback/Time Manipulation - Direct buffer implementations
 *
 * Implements time-based manipulation operations working directly
 * on memory buffers without file I/O.
 *
 * Algorithms: zigzag, iterate, stutter, bounce, drunk, loop
 */

#ifndef CDP_PLAYBACK_H
#define CDP_PLAYBACK_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Zigzag playback - alternates between forward and backward playback.
 *
 * Plays audio forward then backward through specified time segments,
 * creating a zigzag pattern through the sound.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   times: Array of time points in seconds (defines segment boundaries)
 *   num_times: Number of time points (must be >= 2)
 *   splice_ms: Crossfade splice length in milliseconds (default 15)
 *
 * Returns: New buffer with zigzag playback, or NULL on error.
 *
 * Example: times = [0, 1, 2, 3] plays 0->1 forward, 2->1 backward, 2->3 forward
 */
cdp_lib_buffer* cdp_lib_zigzag(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                const double* times,
                                int num_times,
                                double splice_ms);

/*
 * Iterate - repeat audio with variations.
 *
 * Creates multiple iterations/copies of the sound with optional pitch shift
 * and amplitude decay. Each iteration can be delayed and modified.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   repeats: Number of repetitions (1-100)
 *   delay: Delay between iterations in seconds
 *   delay_rand: Random variation in delay (0.0 to 1.0)
 *   pitch_shift: Max pitch shift per iteration in semitones (+/- range)
 *   gain_decay: Amplitude decay per iteration (0.5 = halve each time, 1.0 = no decay)
 *   seed: Random seed (0 = use time)
 *
 * Returns: New buffer with iterated audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_iterate(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 int repeats,
                                 double delay,
                                 double delay_rand,
                                 double pitch_shift,
                                 double gain_decay,
                                 unsigned int seed);

/*
 * Stutter - segment-based stuttering effect.
 *
 * Cuts audio into segments and randomly repeats/reorders them with
 * optional silence insertions and transposition.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   segment_ms: Average segment size in milliseconds
 *   duration: Output duration in seconds
 *   silence_prob: Probability of silence insert between segments (0.0 to 1.0)
 *   silence_min_ms: Minimum silence duration in milliseconds
 *   silence_max_ms: Maximum silence duration in milliseconds
 *   transpose_range: Max transposition in semitones (+/- range, 0 = none)
 *   seed: Random seed (0 = use time)
 *
 * Returns: New buffer with stutter effect, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_stutter(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double segment_ms,
                                 double duration,
                                 double silence_prob,
                                 double silence_min_ms,
                                 double silence_max_ms,
                                 double transpose_range,
                                 unsigned int seed);

/*
 * Bounce - bouncing ball effect.
 *
 * Creates accelerating repetitions of audio, like a bouncing ball
 * with decreasing time between bounces and decreasing amplitude.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   bounces: Number of bounces (1-100)
 *   initial_delay: Initial delay between bounces in seconds
 *   shrink: How much to shrink delay each bounce (0.5 = halve, 0.9 = 10% shorter)
 *   end_level: Final amplitude level (0.0 to 1.0)
 *   level_curve: Level decay curve (1.0 = linear, <1 = fast decay, >1 = slow decay)
 *   cut_bounces: If true, cut each bounce to fit; if false, allow overlap
 *
 * Returns: New buffer with bounce effect, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_bounce(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                int bounces,
                                double initial_delay,
                                double shrink,
                                double end_level,
                                double level_curve,
                                int cut_bounces);

/*
 * Drunk - "drunken walk" random navigation through audio.
 *
 * Randomly navigates through the audio, taking random steps forward
 * or backward within a configurable range (ambitus) around a center
 * point (locus). Creates unpredictable, wandering playback.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   duration: Output duration in seconds
 *   step_ms: Average step size in milliseconds
 *   step_rand: Random variation in step (0.0 to 1.0)
 *   locus: Center position in seconds (navigation centers around this)
 *   ambitus: Maximum range of movement in seconds (distance from locus)
 *   overlap: Overlap between segments (0.0 to 0.9)
 *   splice_ms: Crossfade splice length in milliseconds
 *   seed: Random seed (0 = use time)
 *
 * Returns: New buffer with drunk walk playback, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_drunk(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input,
                               double duration,
                               double step_ms,
                               double step_rand,
                               double locus,
                               double ambitus,
                               double overlap,
                               double splice_ms,
                               unsigned int seed);

/*
 * Loop - loop a section of audio with variations.
 *
 * Repeats a section of audio multiple times, optionally stepping
 * through the file and adding random variation to loop start points.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   start: Loop start position in seconds
 *   length_ms: Loop length in milliseconds
 *   step_ms: Step between loop iterations in milliseconds (0 = no stepping)
 *   search_ms: Random search field for loop start in milliseconds (0 = no variation)
 *   repeats: Number of loop repetitions (1-1000)
 *   splice_ms: Crossfade splice length in milliseconds
 *   seed: Random seed (0 = use time)
 *
 * Returns: New buffer with looped audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_loop(cdp_lib_ctx* ctx,
                              const cdp_lib_buffer* input,
                              double start,
                              double length_ms,
                              double step_ms,
                              double search_ms,
                              int repeats,
                              double splice_ms,
                              unsigned int seed);

/*
 * Retime - time-domain time stretching/compression.
 *
 * Changes the duration of audio using overlap-add (TDOLA) without changing pitch.
 * Uses grain-based processing with crossfade blending for smooth results.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   ratio: Time ratio (0.5 = half speed/twice duration, 2.0 = double speed/half duration)
 *   grain_ms: Grain size in milliseconds (10-200, default 50)
 *   overlap: Overlap factor between grains (0.1 to 0.9, default 0.5)
 *
 * Returns: New buffer with retimed audio, or NULL on error.
 *
 * Note: For high-quality time stretching with better transient preservation,
 * use the spectral time_stretch function instead.
 */
cdp_lib_buffer* cdp_lib_retime(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                double ratio,
                                double grain_ms,
                                double overlap);

/*
 * Scramble - reorder wavesets (zero-crossing segments) in audio.
 *
 * Detects wavesets based on zero crossings and reorders them according to
 * the specified mode. Creates glitchy, granular, or sorted textures.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (mono recommended, stereo uses left channel for detection)
 *   mode: Reordering mode:
 *         0 = shuffle (random order)
 *         1 = reverse (play wavesets in reverse order)
 *         2 = by_size_up (smallest to largest - rising pitch effect)
 *         3 = by_size_down (largest to smallest - falling pitch effect)
 *         4 = by_level_up (quietest to loudest)
 *         5 = by_level_down (loudest to quietest)
 *   group_size: Number of half-cycles per waveset group (1-64, default 2)
 *   seed: Random seed for shuffle mode (0 = use time)
 *
 * Returns: New buffer with scrambled wavesets, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_scramble(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  int mode,
                                  int group_size,
                                  unsigned int seed);

/*
 * Splinter - create fragmenting/splintering effect by shrinking and repeating.
 *
 * Takes a segment of audio and creates a "splintering" effect by progressively
 * shrinking and repeating it. Creates percussive, granular fragmentation effects.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   start: Start position of segment to splinter (seconds)
 *   duration_ms: Duration of segment to splinter (milliseconds, 5-5000)
 *   repeats: Number of repetitions (2-500)
 *   min_shrink: Minimum shrinkage factor at end (0.01 to 1.0, default 0.1)
 *               0.1 = final segment is 10% of original length
 *   shrink_curve: Shrinkage curve (0.1 to 10.0, default 1.0)
 *                 1.0 = linear, >1 = faster shrink at start, <1 = faster at end
 *   accel: Acceleration of repetition rate (0.5 to 4.0, default 1.5)
 *          >1 = repetitions get faster, <1 = slower, 1.0 = constant rate
 *   seed: Random seed for timing variation (0 = use time, >0 for reproducible)
 *
 * Returns: New buffer with splintered audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_splinter(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double start,
                                  double duration_ms,
                                  int repeats,
                                  double min_shrink,
                                  double shrink_curve,
                                  double accel,
                                  unsigned int seed);

/*
 * Spin - rotate audio around the stereo field.
 *
 * Creates a spinning/rotating spatial effect by continuously panning audio
 * around the stereo field. Can include doppler pitch shift for realism.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer (mono or stereo)
 *   rate: Spin rate in Hz (cycles per second). Range: -20 to +20.
 *         Positive = clockwise, negative = counterclockwise.
 *   doppler: Doppler pitch shift in semitones (0-12, default 0).
 *            0 = no doppler, higher values = more pitch variation.
 *   depth: Depth of panning (0.0 to 1.0, default 1.0).
 *          1.0 = full rotation, 0.5 = partial rotation.
 *
 * Returns: New stereo buffer with spinning audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_spin(cdp_lib_ctx* ctx,
                              const cdp_lib_buffer* input,
                              double rate,
                              double doppler,
                              double depth);

/*
 * Rotor - dual-rotation modulation effect.
 *
 * Applies two independent rotating modulators (pitch and amplitude) with
 * different cycle lengths, creating evolving interference patterns.
 * Inspired by CDP's rotor synthesis concept of rotating "armatures".
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   pitch_rate: Pitch modulation rate in Hz (0.01 to 20). Cycles per second.
 *   pitch_depth: Pitch modulation depth in semitones (0 to 12).
 *   amp_rate: Amplitude modulation rate in Hz (0.01 to 20). Cycles per second.
 *   amp_depth: Amplitude modulation depth (0.0 to 1.0). 1.0 = full tremolo.
 *   phase_offset: Initial phase offset between modulators (0.0 to 1.0).
 *                 Creates different interference patterns.
 *
 * Returns: New buffer with rotor modulation, or NULL on error.
 *
 * Note: When pitch_rate and amp_rate have different values, the combined
 * effect creates evolving patterns that cycle through various combinations
 * of pitch and amplitude modulation over time.
 */
cdp_lib_buffer* cdp_lib_rotor(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input,
                               double pitch_rate,
                               double pitch_depth,
                               double amp_rate,
                               double amp_depth,
                               double phase_offset);

#ifdef __cplusplus
}
#endif

#endif /* CDP_PLAYBACK_H */
