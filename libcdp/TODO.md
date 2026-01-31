# CDP Native Library Integration TODO

This document tracks CDP algorithms to be integrated as native library functions (no subprocess overhead).

**Important:** Future development should focus on porting actual CDP algorithms. Non-CDP additions are welcome but should be clearly marked as such.

## Current Status

### Completed: CDP Algorithm Ports

Native implementations of actual CDP algorithms (replacing subprocess calls).

**Spectral Processing (CDP: `blur`, `stretch`, `pvoc`, `spec*`):**
- [x] `time_stretch` - Phase vocoder time stretch
- [x] `spectral_blur` - Spectral averaging/smearing
- [x] `modify_speed` - Playback speed change (affects pitch)
- [x] `pitch_shift` - Pitch shift without changing duration
- [x] `spectral_shift` - Shift all frequencies by Hz offset
- [x] `spectral_stretch` - Differential frequency stretching

**Filtering (CDP: `filter`, `filtrage`, `synfilt`):**
- [x] `filter_lowpass` - Spectral lowpass filter
- [x] `filter_highpass` - Spectral highpass filter
- [x] `filter_bandpass` - Spectral bandpass filter
- [x] `filter_notch` - Spectral notch (band-reject) filter

**Envelope (CDP: `envel`, `tremolo`, `envcut`):**
- [x] `dovetail` - Fade in/out envelopes (linear/exponential)
- [x] `tremolo` - LFO amplitude modulation
- [x] `attack` - Attack transient reshaping

**Distortion (CDP: `distort`, `distortt`, `distcut`, etc.):**
- [x] `distort_overload` - Soft/hard clipping
- [x] `distort_reverse` - Reverse wavecycles (zero-crossing based)
- [x] `distort_fractal` - Recursive wavecycle overlay
- [x] `distort_shuffle` - Segment rearrangement

**Reverb & Spatial (CDP: `reverb`, `rmverb`, `panorama`):**
- [x] `reverb` - FDN reverb (8 comb + 4 allpass filters)

**Granular (CDP: `brassage`, `grain`, `texture`):**
- [x] `brassage` - Granular resynthesis with pitch/time params
- [x] `freeze` - Segment repetition with crossfade
- [x] `grain_cloud` - Grain cloud generation from amplitude-detected grains
- [x] `grain_extend` - Extend duration using grain repetition
- [x] `texture_simple` - Simple texture layering with pitch/amp/spatial variation
- [x] `texture_multi` - Multi-layer grouped texture generation

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

### Completed: Non-CDP Additions

Useful audio processing functions not derived from CDP algorithms.

**Dynamics (standard DSP, not in CDP):**
- [x] `gate` - Noise gate with attack/release/hold
- [x] `compressor` - Dynamic range compression
- [x] `limiter` - Hard/soft limiting
- [x] `envelope_follow` - Extract amplitude envelope (RMS/peak tracking)
- [x] `envelope_apply` - Apply envelope to sound

**EQ (standard DSP):**
- [x] `eq_parametric` - Parametric EQ with Q factor

**Effects (standard DSP):**
- [x] `bitcrush` - Bit depth and sample rate reduction
- [x] `ring_mod` - Ring modulation (carrier multiply)
- [x] `delay` - Feedback delay with mix control
- [x] `chorus` - Modulated delay (LFO-based)
- [x] `flanger` - Short modulated delay with feedback

---

## Future Priorities: CDP Algorithm Ports

Focus on actual CDP algorithms. Reference the CDP executable list at the bottom.

### Priority 1: Morphing/Combining (CDP: `morph`, `combine`)

| CDP Tool | Algorithm | Description |
|----------|-----------|-------------|
| `morph` | Spectral morph | Interpolate between sounds |
| `newmorph` | New morph | Updated morph algorithm |
| `combine` | Spectral combine | Cross-synthesis (amp/freq mixing) |

### Priority 2: Analysis (CDP: `pitch`, `formants`, `get_partials`)

| CDP Tool | Algorithm | Description |
|----------|-----------|-------------|
| `pitch` | Pitch tracking | Extract pitch contour |
| `formants` | Formant analysis | Extract formant frequencies |
| `get_partials` | Partial tracking | Extract sinusoidal partials |

### Priority 3: Additional Spectral (CDP: `focus`, `hilite`, `specfold`)

| CDP Tool | Algorithm | Description |
|----------|-----------|-------------|
| `focus` | Spectral focus | Enhance specific frequencies |
| `hilite` | Spectral highlight | Boost spectral peaks |
| `specfold` | Spectral fold | Fold spectrum at frequency |
| `speclean` | Spectral clean | Remove spectral noise |

### Priority 4: Experimental (CDP: `strange`, `fractal`, `crystal`)

| CDP Tool | Algorithm | Description |
|----------|-----------|-------------|
| `strange` | Strange attractor | Chaotic modulation |
| `fractal` | Fractal | Fractal-based processing |
| `crystal` | Crystal | Crystalline textures |
| `brownian` | Brownian motion | Random walk modulation |

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

### Porting CDP Algorithms
When porting a CDP algorithm:
1. Study the original CDP source code
2. Understand the algorithm's parameters and behavior
3. Implement following the architecture above
4. Test against known CDP outputs where possible

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
