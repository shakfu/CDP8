# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (CDP Algorithm Ports)
- **Filtering:** `filter_bandpass`, `filter_notch` - spectral bandpass and notch filters
- **Granular:** `grain_cloud` - grain cloud generation from amplitude-detected grains (CDP: grain)
- **Granular:** `grain_extend` - extend duration using grain repetition (CDP: grainex extend)
- **Texture:** `texture_simple` - simple texture layering (CDP: texture SIMPLE_TEX)
- **Texture:** `texture_multi` - multi-layer grouped texture (CDP: texture GROUPS)
- **Morphing:** `morph` - spectral interpolation between two sounds (CDP: SPECMORPH)
- **Morphing:** `morph_glide` - simple spectral glide between two sounds (CDP: SPECGLIDE)
- **Cross-synthesis:** `cross_synth` - combine amp from one sound with freq from another (CDP: combine)

### Added (Non-CDP Additions)
Standard DSP functions not derived from CDP algorithms:
- **EQ:** `eq_parametric` - parametric equalizer with center frequency, gain, and Q factor
- **Dynamics:** `gate` - noise gate with attack/release/hold envelope
- **Dynamics:** `compressor` - dynamic range compression with threshold, ratio, attack/release
- **Dynamics:** `limiter` - peak limiting with attack/release
- **Envelope:** `envelope_follow` - extract amplitude envelope (peak or RMS mode)
- **Envelope:** `envelope_apply` - apply envelope to sound with depth control
- **Distortion:** `bitcrush` - bit depth and sample rate reduction
- **Modulation:** `ring_mod` - ring modulation with carrier frequency
- **Delay effects:** `delay`, `chorus`, `flanger` - feedback delay, modulated delays

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
