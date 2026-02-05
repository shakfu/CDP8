# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (Demos)

**Synthesis Demos (01-07)** - Generate test sounds and demonstrate API:
- `01_basic_operations.py` - Buffers, gain, fades, panning, mixing
- `02_effects_and_processing.py` - Delay, reverb, modulation, distortion, filters
- `03_spectral_processing.py` - Blur, focus, time stretch, pitch shift, freeze
- `04_granular_synthesis.py` - Brassage, wrappage, grain manipulation
- `05_pitch_synchronous.py` - PSOW stretch/grab, FOF repitch, hover
- `06_creative_techniques.py` - Effect chains and sound design recipes
- `07_morphing.py` - Spectral morph, glide, cross-synthesis

**FX Processing Demos (fx01-fx07)** - CLI tools for processing real audio:
- `fx01_time_and_pitch.py` - Time stretch, pitch shift, spectral shift
- `fx02_spectral_effects.py` - Blur, focus, fold, freeze effects
- `fx03_granular.py` - Brassage, wrappage, grain operations
- `fx04_reverb_delay_mod.py` - Reverb, delay, tremolo, chorus, flanger, ring mod
- `fx05_distortion_dynamics.py` - Waveset distortion, bitcrush, filters, compression
- `fx06_psow_fof.py` - PSOW stretch/grab, FOF repitch, hover
- `fx07_creative_chains.py` - Complex effect chains (ambient, industrial, shimmer, etc.)

All FX demos accept `input.wav -o output_dir/` arguments.

**Makefile targets:**
- `make demos` - Run all demos, output to `build/`
- `make demos-clean` - Remove generated WAV files

### Added (CDP Algorithm Ports)

**Analysis:**
- `pitch` - pitch tracking using YIN algorithm (CDP: pitch)
- `formants` - formant analysis using LPC (CDP: formants)
- `get_partials` - partial/harmonic tracking (CDP: get_partials)

**Spectral Processing:**
- `spectral_focus` - super-Gaussian frequency enhancement (CDP: focus)
- `spectral_hilite` - boost spectral peaks above threshold (CDP: hilite)
- `spectral_fold` - fold spectrum at frequency for metallic effects (CDP: specfold)
- `spectral_clean` - spectral noise gate (CDP: speclean)

**Experimental/Chaos:**
- `strange` - Lorenz attractor chaotic modulation (CDP: strange)
- `brownian` - random walk modulation of pitch/amp/filter (CDP: brownian)
- `crystal` - crystalline textures with decaying echoes (CDP: crystal)
- `fractal` - recursive wavecycle overlay with pitch ratio and decay (CDP: fractal)
- `quirk` - probabilistic reverse/dropout transformations (CDP: quirk)
- `chirikov` - standard map chaotic modulation (CDP: chirikov)
- `cantor` - Cantor set fractal gating pattern (CDP: cantor)
- `cascade` - cascading echoes with pitch/amp/filter decay (CDP: cascade)
- `fracture` - fragment and scatter audio with gaps (CDP: fracture)
- `tesselate` - tile-based pattern transformations (CDP: tesselate)

**Playback/Time Manipulation:**
- `zigzag` - alternating forward/backward playback through time points (CDP: zigzag)
- `iterate` - repeated playback with pitch/gain variations (CDP: iterate)
- `stutter` - segment-based stuttering with silence inserts (CDP: stutter)
- `bounce` - bouncing ball effect with accelerating repeats (CDP: bounce)
- `drunk` - "drunk walk" random navigation through audio (CDP: drunk)
- `loop` - looping with crossfades and variations (CDP: loop)
- `retime` - time-domain time stretching/compression using TDOLA (CDP: retime)
- `scramble` - waveset reordering (shuffle, reverse, by size/level) (CDP: scramble)
- `splinter` - fragmenting effect with shrinking repeats (CDP: splinter)
- `hover` - zigzag reading at specified frequency for hovering pitch effect (CDP: hover)
- `constrict` - shorten or remove silent sections (CDP: constrict)
- `phase_invert` - invert phase of audio signal (CDP: phase mode 1)
- `phase_stereo` - enhance stereo separation via phase subtraction (CDP: phase mode 2)
- `wrappage` - granular texture with stereo spatial distribution (CDP: wrappage)

**Spatial Effects:**
- `spin` - rotate audio around stereo field with optional doppler (CDP: spin)
- `rotor` - dual-rotation modulation creating interference patterns (CDP: rotor)
- `flutter` - spatial tremolo with loudness modulation alternating L/R (CDP: flutter)

**Extended Granular:**
- `grain_reorder` - reorder detected grains (shuffle, reverse, rotate) (CDP: grain)
- `grain_rerhythm` - change timing/rhythm of grains (CDP: grain)
- `grain_reverse` - reverse individual grains in place (CDP: grain)
- `grain_timewarp` - time-stretch/compress grain spacing (CDP: grain)
- `grain_repitch` - pitch-shift grains with interpolation (CDP: grain)
- `grain_position` - reposition grains in stereo field (CDP: grain)
- `grain_omit` - probabilistically omit grains (CDP: grain)
- `grain_duplicate` - duplicate grains with variations (CDP: grain)

**Granular/Texture:**
- `grain_cloud` - grain cloud generation from amplitude-detected grains (CDP: grain)
- `grain_extend` - extend duration using grain repetition (CDP: grainex extend)
- `texture_simple` - simple texture layering (CDP: texture SIMPLE_TEX)
- `texture_multi` - multi-layer grouped texture (CDP: texture GROUPS)

**Morphing/Cross-synthesis:**
- `morph` - spectral interpolation between two sounds (CDP: SPECMORPH)
- `morph_glide` - simple spectral glide between two sounds (CDP: SPECGLIDE)
- `cross_synth` - combine amp from one sound with freq from another (CDP: combine)
- `morph_glide_native` - native CDP specglide wrapper (original algorithm)
- `morph_bridge_native` - native CDP specbridge wrapper (original algorithm)
- `morph_native` - native CDP specmorph wrapper (original algorithm)

**Synthesis:**
- `synth_wave` - waveform synthesis (sine, square, saw, ramp, triangle) (CDP: wave)
- `synth_noise` - noise generation (white, pink) (CDP: synth noise)
- `synth_click` - click/metronome track generation (CDP: click)
- `synth_chord` - chord synthesis from MIDI pitch list (CDP: multi_syn)

**Pitch-Synchronous Operations (PSOW):**
- `psow_stretch` - time-stretch while preserving pitch using PSOLA (CDP: psow stretch)
- `psow_grab` - extract pitch-synchronous grains from a position (CDP: psow grab)
- `psow_dupl` - duplicate grains for time-stretching (CDP: psow dupl)
- `psow_interp` - interpolate between two grains (CDP: psow interp)

**FOF Extraction and Synthesis (FOFEX):**
- `fofex_extract` - extract single FOF at specified time (CDP: fofex)
- `fofex_extract_all` - extract all FOFs to uniform-length bank (CDP: fofex)
- `fofex_synth` - synthesize audio from FOFs at target pitch (CDP: fofex)
- `fofex_repitch` - repitch audio with optional formant preservation (CDP: fofex)

**Distortion:**
- `distort_cut` - waveset segmentation with decaying envelope (CDP: distcut)
- `distort_mark` - interpolate between waveset groups at time markers (CDP: distmark)
- `distort_repeat` - time-stretch by repeating wavecycles (CDP: distrep)
- `distort_shift` - shift/swap half-wavecycle groups (CDP: distshift)
- `distort_warp` - progressive warp distortion with modular sample folding (CDP: distwarp)

**Filtering:**
- `filter_bandpass` - spectral bandpass filter
- `filter_notch` - spectral notch (band-reject) filter

### Added (Non-CDP Additions)

Standard DSP functions not derived from CDP algorithms:

**EQ:**
- `eq_parametric` - parametric equalizer with center frequency, gain, and Q factor

**Dynamics:**
- `gate` - noise gate with attack/release/hold envelope
- `compressor` - dynamic range compression with threshold, ratio, attack/release
- `limiter` - peak limiting with attack/release
- `envelope_follow` - extract amplitude envelope (peak or RMS mode)
- `envelope_apply` - apply envelope to sound with depth control

**Effects:**
- `bitcrush` - bit depth and sample rate reduction
- `ring_mod` - ring modulation with carrier frequency
- `delay` - feedback delay with mix control
- `chorus` - modulated delay (LFO-based)
- `flanger` - short modulated delay with feedback

### Added (Constants)

- Waveform types: `WAVE_SINE`, `WAVE_SQUARE`, `WAVE_SAW`, `WAVE_RAMP`, `WAVE_TRIANGLE`
- Scramble modes: `SCRAMBLE_SHUFFLE`, `SCRAMBLE_REVERSE`, `SCRAMBLE_SIZE_UP`, `SCRAMBLE_SIZE_DOWN`, `SCRAMBLE_LEVEL_UP`, `SCRAMBLE_LEVEL_DOWN`

### Fixed
- Phase vocoder frequency calculation now uses correct hop size
- Spectral filters now use bin center frequency for accurate filtering

## [0.1.0] - 2025-01-31

### Added
- Native CDP library integration (no subprocess overhead)
- **Spectral processing:** `time_stretch`, `spectral_blur`, `modify_speed`, `pitch_shift`, `spectral_shift`, `spectral_stretch`, `filter_lowpass`, `filter_highpass`
- **Envelope operations:** `dovetail`, `tremolo`, `attack`
- **Distortion:** `distort_overload`, `distort_reverse`, `distort_fractal`, `distort_shuffle`
- **Reverb:** `reverb` - FDN reverb (8 comb + 4 allpass filters)
- **Granular:** `brassage`, `freeze`
- **Core operations:** `gain`, `gain_db`, `normalize`, `normalize_db`, `phase_invert`
- **Spatial:** `pan`, `pan_envelope`, `mirror`, `narrow`
- **Mixing:** `mix`, `mix2`
- **Buffer utilities:** `reverse`, `fade_in`, `fade_out`, `concat`
- **Channel operations:** `to_mono`, `to_stereo`, `extract_channel`, `merge_channels`, `split_channels`, `interleave`
- **File I/O:** `read_file`, `write_file` (WAV: float, PCM16, PCM24)
- Cython bindings with zero-copy buffer protocol support
- Comprehensive test suite
