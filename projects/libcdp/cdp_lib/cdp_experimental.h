/*
 * CDP Experimental Processing - Novel audio transformations
 *
 * These functions implement experimental audio processing operations
 * including chaotic modulation, random walks, and crystalline textures.
 */

#ifndef CDP_EXPERIMENTAL_H
#define CDP_EXPERIMENTAL_H

#include "cdp_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Strange attractor (Lorenz) modulation.
 *
 * Uses the Lorenz attractor to chaotically modulate pitch and amplitude.
 * Creates complex, evolving timbral changes that are deterministic but
 * appear chaotic.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   chaos_amount: Amount of chaotic modulation (0.0 to 1.0)
 *   rate: Speed of chaotic evolution (typically 0.1 to 10.0)
 *   seed: Random seed for initial attractor state
 *
 * Returns: New buffer with chaotic modulation, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_strange(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double chaos_amount,
                                 double rate,
                                 unsigned int seed);

/*
 * Brownian (random walk) modulation.
 *
 * Applies random walk modulation to pitch, amplitude, or filter cutoff.
 * Creates organic, drifting parameter changes.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   step_size: Maximum step size per sample (in target units)
 *   smoothing: Smoothing factor (0.0 to 1.0, higher = smoother)
 *   target: Modulation target: 0=pitch (semitones), 1=amp (dB), 2=filter (Hz)
 *   seed: Random seed for reproducibility
 *
 * Returns: New buffer with random walk modulation, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_brownian(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double step_size,
                                  double smoothing,
                                  int target,
                                  unsigned int seed);

/*
 * Crystal textures - granular with decaying echoes.
 *
 * Extracts small grains and creates shimmering, crystalline textures
 * through multiple decaying echo layers with pitch scatter.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   density: Grain density (grains per second, typically 20-200)
 *   decay: Echo decay time in seconds
 *   pitch_scatter: Random pitch variation in semitones
 *   seed: Random seed for reproducibility
 *
 * Returns: New buffer with crystalline texture, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_crystal(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 double density,
                                 double decay,
                                 double pitch_scatter,
                                 unsigned int seed);

/*
 * Fractal processing - self-similar recursive layering.
 *
 * Creates fractal textures by recursively layering pitch-shifted copies
 * of the input at decreasing amplitudes, creating self-similar structures.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   depth: Recursion depth (1-6, higher = more layers)
 *   pitch_ratio: Pitch ratio between layers (typically 0.5 for octave down)
 *   decay: Amplitude decay per layer (0.0 to 1.0)
 *   seed: Random seed for timing variations
 *
 * Returns: New buffer with fractal processing, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_fractal(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 int depth,
                                 double pitch_ratio,
                                 double decay,
                                 unsigned int seed);

/*
 * Quirk - unpredictable glitchy transformations.
 *
 * Applies random, unexpected pitch and timing shifts with probability-based
 * triggers. Creates glitchy, surprising audio artifacts.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   probability: Probability of quirk occurring (0.0 to 1.0)
 *   intensity: Intensity of quirks (0.0 to 1.0)
 *   mode: 0=pitch quirks, 1=timing quirks, 2=both
 *   seed: Random seed for reproducibility
 *
 * Returns: New buffer with quirk effects, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_quirk(cdp_lib_ctx* ctx,
                               const cdp_lib_buffer* input,
                               double probability,
                               double intensity,
                               int mode,
                               unsigned int seed);

/*
 * Chirikov map modulation - chaotic standard map.
 *
 * Uses the Chirikov standard map (a classic chaotic system) to modulate
 * pitch and amplitude. Different from Lorenz - produces more periodic
 * chaos with islands of stability.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   k_param: Chirikov K parameter (0.5 to 10.0, chaos increases with K)
 *   mod_depth: Modulation depth (0.0 to 1.0)
 *   rate: Rate of map iteration
 *   seed: Random seed for initial conditions
 *
 * Returns: New buffer with Chirikov modulation, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_chirikov(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double k_param,
                                  double mod_depth,
                                  double rate,
                                  unsigned int seed);

/*
 * Cantor set gating - fractal silence pattern.
 *
 * Applies a Cantor set pattern as a gating function, recursively removing
 * middle thirds of audio segments to create fractally-structured silences.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   depth: Recursion depth (1-8, higher = finer fractal detail)
 *   duty_cycle: Proportion of audio kept vs silenced (0.0 to 1.0)
 *   smooth_ms: Crossfade time in ms to smooth transitions
 *   seed: Random seed for variation
 *
 * Returns: New buffer with Cantor gating, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_cantor(cdp_lib_ctx* ctx,
                                const cdp_lib_buffer* input,
                                int depth,
                                double duty_cycle,
                                double smooth_ms,
                                unsigned int seed);

/*
 * Cascade - cascading echoes with progressive transformation.
 *
 * Creates cascading delays where each echo is progressively transformed
 * (pitch shifted down, filtered darker, amplitude reduced).
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   num_echoes: Number of cascade stages (1-12)
 *   delay_ms: Base delay time in milliseconds
 *   pitch_decay: Pitch ratio per stage (e.g., 0.95 = 5% down each stage)
 *   amp_decay: Amplitude decay per stage (0.0 to 1.0)
 *   filter_decay: Filter cutoff decay per stage (0.0 to 1.0)
 *   seed: Random seed for timing jitter
 *
 * Returns: New buffer with cascading echoes, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_cascade(cdp_lib_ctx* ctx,
                                 const cdp_lib_buffer* input,
                                 int num_echoes,
                                 double delay_ms,
                                 double pitch_decay,
                                 double amp_decay,
                                 double filter_decay,
                                 unsigned int seed);

/*
 * Fracture - break audio into fragments with gaps.
 *
 * Breaks the audio into fragments with random gaps and optional
 * reordering. Creates broken, fractured textures.
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   fragment_ms: Average fragment size in milliseconds
 *   gap_ratio: Ratio of gaps to fragments (0.0 to 2.0)
 *   scatter: Amount of fragment reordering (0.0 = none, 1.0 = full shuffle)
 *   seed: Random seed for reproducibility
 *
 * Returns: New buffer with fractured audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_fracture(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double fragment_ms,
                                  double gap_ratio,
                                  double scatter,
                                  unsigned int seed);

/*
 * Tesselate - tile audio segments in patterns.
 *
 * Divides audio into tiles and arranges them in patterns with optional
 * transformations per tile (pitch, amplitude, reverse).
 *
 * Args:
 *   ctx: Library context
 *   input: Input audio buffer
 *   tile_ms: Tile size in milliseconds
 *   pattern: Pattern mode: 0=repeat, 1=mirror, 2=rotate, 3=random
 *   overlap: Tile overlap ratio (0.0 to 0.5)
 *   transform: Transform intensity (0.0 to 1.0)
 *   seed: Random seed for random pattern mode
 *
 * Returns: New buffer with tessellated audio, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_tesselate(cdp_lib_ctx* ctx,
                                   const cdp_lib_buffer* input,
                                   double tile_ms,
                                   int pattern,
                                   double overlap,
                                   double transform,
                                   unsigned int seed);

#ifdef __cplusplus
}
#endif

#endif /* CDP_EXPERIMENTAL_H */
