# CDP Native Library Integration TODO

This document tracks the remaining CDP algorithms to be integrated as native library functions (no subprocess overhead).

## Current Status

### Completed Native Implementations (cdp_lib)

All core audio processing functions are now native - no subprocess overhead.

**Spectral Processing:**
- [x] `time_stretch` - Phase vocoder time stretch
- [x] `spectral_blur` - Spectral averaging/smearing
- [x] `modify_speed` - Playback speed change (affects pitch)
- [x] `pitch_shift` - Pitch shift without changing duration
- [x] `spectral_shift` - Shift all frequencies by Hz offset
- [x] `spectral_stretch` - Differential frequency stretching
- [x] `filter_lowpass` - Spectral lowpass filter
- [x] `filter_highpass` - Spectral highpass filter
- [x] `filter_bandpass` - Spectral bandpass filter
- [x] `filter_notch` - Spectral notch (band-reject) filter

**Envelope & Dynamics:**
- [x] `dovetail` - Fade in/out envelopes (linear/exponential)
- [x] `tremolo` - LFO amplitude modulation
- [x] `attack` - Attack transient reshaping
- [x] `gate` - Noise gate with attack/release/hold

**Distortion:**
- [x] `distort_overload` - Soft/hard clipping
- [x] `distort_reverse` - Reverse wavecycles (zero-crossing based)
- [x] `distort_fractal` - Recursive wavecycle overlay
- [x] `distort_shuffle` - Segment rearrangement
- [x] `bitcrush` - Bit depth and sample rate reduction
- [x] `ring_mod` - Ring modulation (carrier multiply)

**Reverb & Spatial:**
- [x] `reverb` - FDN reverb (8 comb + 4 allpass filters)
- [x] `delay` - Feedback delay with mix control
- [x] `chorus` - Modulated delay (LFO-based)
- [x] `flanger` - Short modulated delay with feedback

**Granular:**
- [x] `brassage` - Granular resynthesis with pitch/time params
- [x] `freeze` - Segment repetition with crossfade

**Core (from original libcdp):**
- [x] `gain`, `gain_db` - Amplitude adjustment
- [x] `normalize`, `normalize_db` - Peak normalization
- [x] `phase_invert` - Phase inversion
- [x] `pan`, `pan_envelope` - Stereo panning
- [x] `mirror`, `narrow` - Stereo field manipulation
- [x] `mix`, `mix2` - Audio mixing
- [x] `reverse`, `fade_in`, `fade_out`, `concat` - Buffer utilities
- [x] `to_mono`, `to_stereo`, `extract_channel`, `merge_channels`, `split_channels`, `interleave` - Channel operations
- [x] `read_file`, `write_file` - WAV I/O (float, PCM16, PCM24)

---

## Future Priorities

### Priority 1: Additional Filtering & EQ

| Algorithm | Description | Notes |
|-----------|-------------|-------|
| Parametric EQ | Boost/cut with Q | Multiple bands |

### Priority 2: Additional Envelope & Dynamics

| Algorithm | Description | Notes |
|-----------|-------------|-------|
| Envelope follow | Extract amplitude envelope | RMS or peak tracking |
| Envelope apply | Apply envelope to sound | Amplitude modulation |
| Compressor | Dynamic range compression | Not in CDP |
| Limiter | Hard/soft limiting | Peak control |

### Priority 3: Additional Granular & Texture

| Algorithm | Description | Notes |
|-----------|-------------|-------|
| Grain | Basic granular synthesis | Grain cloud generation |
| Texture | Multi-layer textures | Complex parameter control |
| Extend | Extend duration | Various algorithms |

### Priority 4: Analysis & Resynthesis

| Algorithm | Description | Notes |
|-----------|-------------|-------|
| Pitch tracking | Extract pitch contour | Autocorrelation/cepstrum |
| Formant analysis | Extract formant frequencies | LPC or spectral peaks |
| Partial tracking | Extract sinusoidal partials | McAulay-Quatieri |
| Morph | Interpolate between sounds | Spectral interpolation |
| Cross-synthesis | Combine spectral features | Amp from A, freq from B |

### Priority 5: Synthesis

| Algorithm | Description | Notes |
|-----------|-------------|-------|
| Oscillators | Basic waveform generation | Sine, saw, square, etc. |
| FM synthesis | Frequency modulation | Carrier + modulators |
| Additive | Sum of sinusoids | Partial specification |
| Noise | Noise generation | White, pink, brown |

---

## Implementation Notes

### Architecture
Each native function should:
1. Accept `cdp_lib_buffer*` for input/output
2. Return newly allocated buffer (caller frees)
3. Set error message in context on failure
4. Be thread-safe (no global state)

### FFT Infrastructure
We have `fft_()` from `mxfft.c` available. For spectral operations:
- Use `cdp_spectral_analyze()` for STFT
- Use `cdp_spectral_synthesize()` for inverse STFT
- Extend `cdp_spectral_data` structure as needed

### Testing
Each new algorithm needs:
1. C unit test in `test_cdp_lib.c`
2. Python test in `tests/test_pycdp.py`

### Cython Bindings
For each C function:
1. Add declaration to `cdp_lib.pxd`
2. Add wrapper function to `_core.pyx`
3. Export in `__init__.py`

---

## CDP Executable Reference (220 total)

### Spectral Processing
`blur`, `focus`, `hilite`, `spec`, `specanal`, `specav`, `specenv`, `specfnu`,
`specfold`, `specgrids`, `speclean`, `specnu`, `specross`, `specsphinx`,
`spectrum`, `spectstr`, `spectune`, `spectwin`, `speculate`, `specvu`

### Time/Pitch Modification
`modify`, `stretch`, `stretcha`, `pvoc`, `repitch`, `pitch`, `pmodify`,
`retime`, `ts`, `strans`

### Filtering
`filter`, `filtrage`, `synfilt`, `notchinvert`

### Distortion
`distort`, `distortt`, `distcut`, `distmark`, `distmore`, `distrep`, `distshift`

### Envelope
`envel`, `envcut`, `envnu`, `envspeak`, `tremolo`, `tremenv`

### Granular/Texture
`grain`, `grainex`, `texture`, `newtex`, `texmchan`, `brassage`

### Spatial
`reverb`, `rmverb`, `mchanrev`, `panorama`, `abfpan`, `abfpan2`, `mchanpan`

### Synthesis
`synth`, `newsynth`, `multisynth`, `multiosc`, `phasor`, `impulse`, `waveform`

### Morphing/Combining
`morph`, `newmorph`, `combine`, `submix`, `multimix`, `newmix`, `nmix`

### Analysis
`pitch`, `pitchinfo`, `formants`, `get_partials`, `features`, `onset`,
`peak`, `peakfind`, `rmsinfo`, `sndinfo`, `specinfo`

### Utilities
`housekeep`, `sfedit`, `sfprops`, `channelx`, `tostereo`, `mton`

### Experimental/Advanced
`strange`, `fractal`, `frfractal`, `quirk`, `crystal`, `brownian`,
`chirikov`, `cantor`, `cascade`, `fracture`, `tesselate`
