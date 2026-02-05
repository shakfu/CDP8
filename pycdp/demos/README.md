# pycdp Demos

Example scripts demonstrating pycdp usage, from basic to advanced.

## Running the Demos

Each demo is a standalone Python script that creates test sounds programmatically:

```bash
cd pycdp/demos
python 01_basic_operations.py
python 02_effects_and_processing.py
# etc.
```

## Demo Overview

### 01_basic_operations.py
Fundamental operations:
- Creating buffers from arrays
- Gain and normalization
- Phase inversion
- Fades, reverse, concatenation
- Stereo panning and mixing
- File I/O (read_file, write_file)

### 02_effects_and_processing.py
Audio effects:
- Delay and reverb
- Modulation (tremolo, chorus, flanger, ring mod)
- Spatial effects (spin, rotor, flutter)
- Distortion (waveset repeat, reverse, fractal)
- Dynamics (compressor, gate, limiter)
- Filters (lowpass, highpass, bandpass, notch)

### 03_spectral_processing.py
CDP's spectral processing:
- Spectral blur and focus
- Time stretching (preserves pitch)
- Pitch shifting (preserves duration)
- Spectral frequency shift (inharmonic)
- Spectral stretch (inharmonic partials)
- Freeze effects

### 04_granular_synthesis.py
Granular synthesis:
- Brassage (classic granular time-stretch)
- Grain operations (reorder, reverse, timewarp, repitch)
- Wrappage (spatial granular texture)
- Grain cloud and extend

### 05_pitch_synchronous.py
Pitch-synchronous (PSOW) processing:
- PSOW stretching and duplication
- PSOW grain extraction (grab)
- PSOW interpolation
- FOF extraction and synthesis
- FOF repitching
- Hover effect (zigzag reading)
- Comparison of pitch-shift methods

### 06_creative_techniques.py
Creative sound design recipes:
- Ambient pad from percussion
- Rhythmic texture from sustained sound
- Vocal-like texture from simple tone
- Glitch/stutter effects
- Ethereal space creation
- Real-world usage tips

### 07_morphing.py
Spectral morphing between sounds:
- Basic morph (interpolate between sounds)
- Morph glide (smooth spectral transition)
- Cross-synthesis (combine characteristics)
- Native morph variants with advanced control
- Vowel and instrument transformations
- Tips for effective morphing

## Working with Real Audio Files

To use these techniques with your own audio:

```python
import pycdp

# Read your file
buf = pycdp.read_file("my_sound.wav")

# Process (example: time stretch + reverb)
stretched = pycdp.time_stretch(buf, factor=2.0)
result = pycdp.reverb(stretched, decay_time=2.0, mix=0.4)

# Normalize to safe level
result = pycdp.normalize(result, target=0.95)

# Write output
pycdp.write_file("output.wav", result)
```

## Quick Reference

### Time Stretching
```python
# Phase vocoder (any sound)
pycdp.time_stretch(buf, factor=2.0)

# Granular (any sound)
pycdp.brassage(buf, velocity=0.5)  # velocity < 1 = stretch

# PSOW (pitched sounds)
pycdp.psow_stretch(buf, stretch_factor=2.0)
```

### Pitch Shifting
```python
# Phase vocoder (any sound, semitones)
pycdp.pitch_shift(buf, semitones=7.0)

# FOF (formant-aware)
pycdp.fofex_repitch(buf, pitch_shift=7.0, preserve_formants=True)

# Granular
pycdp.brassage(buf, velocity=1.0, pitch_shift=7.0)

# Spectral shift (Hz, inharmonic)
pycdp.spectral_shift(buf, shift_hz=100.0)
```

### Creating Textures
```python
# Granular cloud
pycdp.wrappage(buf, grain_size=50.0, density=3.0, velocity=0.3, spread=1.0)

# Spectral smear
pycdp.spectral_blur(buf, blur_time=0.1)

# Frozen moment
pycdp.freeze(buf, start_time=0.5, end_time=0.6, duration=5.0)
```

### Spatial Processing
```python
# Mono to stereo
pycdp.pan(buf, position=0.0)          # Center
pycdp.pan_envelope(buf, [(0, -1), (1, 1)])  # Sweep L to R

# Rotation
pycdp.spin(buf, rate=0.5)             # 0.5 Hz rotation
pycdp.flutter(buf, frequency=4.0)     # Stereo tremolo

# Width control
pycdp.narrow(stereo_buf, width=0.5)   # Reduce width
```

### Dynamics
```python
# Compression
pycdp.compressor(buf, threshold_db=-20.0, ratio=4.0)

# Limiting
pycdp.limiter(buf, threshold_db=-1.0)

# Gating
pycdp.gate(buf, threshold_db=-40.0)
```

### Filters
```python
pycdp.filter_lowpass(buf, cutoff_freq=1000.0)
pycdp.filter_highpass(buf, cutoff_freq=500.0)
pycdp.filter_bandpass(buf, low_freq=500.0, high_freq=2000.0)
pycdp.filter_notch(buf, center_freq=1000.0, width_hz=100.0)
```

### Spectral Morphing
```python
# Basic morph (interpolate between sounds)
pycdp.morph(buf1, buf2, morph_start=0.0, morph_end=1.0)

# Morph glide (smooth transition with specified duration)
pycdp.morph_glide(buf1, buf2, duration=2.0)

# Cross-synthesis (combine amp from one, freq from other)
pycdp.cross_synth(buf1, buf2, mode=0, mix=1.0)

# Native morph with separate amp/freq control
pycdp.morph_native(buf1, buf2, mode=0,
                   amp_start=0.0, amp_end=0.5,
                   freq_start=0.5, freq_end=1.0)

# Native bridge with normalization
pycdp.morph_bridge_native(buf1, buf2, mode=1, offset=0.1)
```
