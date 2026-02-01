# pycdp

Python bindings for the CDP (Composers Desktop Project) audio processing library.

## Overview

pycdp provides Python access to CDP's extensive audio processing algorithms. It uses Cython with memoryviews and the buffer protocol for zero-copy interoperability with numpy, `array.array`, and other buffer-compatible objects - **no numpy dependency required**.

## Installation

```bash
# Build and install in development mode
make build

# Or with uv directly
uv sync
```

## Quick Start

```python
import pycdp

# Load audio file
buf = pycdp.read_file("input.wav")

# Apply processing
stretched = pycdp.time_stretch(buf, stretch_factor=2.0)
shifted = pycdp.pitch_shift(buf, semitones=5)

# Save result
pycdp.write_file("output.wav", stretched)
```

## Usage

### High-level API

Works with any float32 buffer (numpy arrays, `array.array('f')`, memoryview, etc.):

```python
import array
import pycdp

# Create sample data
samples = array.array('f', [0.5, 0.3, -0.2, 0.8, -0.4])

# Apply gain (linear or decibels)
result = pycdp.gain(samples, gain_factor=2.0)
result = pycdp.gain_db(samples, db=6.0)  # +6dB = ~2x

# Normalize to target peak level
result = pycdp.normalize(samples, target=1.0)
result = pycdp.normalize_db(samples, target_db=-3.0)  # -3dBFS

# Phase invert
result = pycdp.phase_invert(samples)

# Find peak level
level, position = pycdp.peak(samples)
```

### With numpy

```python
import numpy as np
import pycdp

samples = np.random.randn(44100).astype(np.float32) * 0.5
result = pycdp.normalize(samples, target=0.9)

# Result supports buffer protocol - zero-copy to numpy
output = np.asarray(result)
```

### File I/O

```python
import pycdp

# Read audio file (returns Buffer)
buf = pycdp.read_file("input.wav")

# Write audio file
pycdp.write_file("output.wav", buf)
```

### Low-level API

For more control, use explicit Context and Buffer objects:

```python
import pycdp

# Create context and buffer
ctx = pycdp.Context()
buf = pycdp.Buffer.create(1000, channels=2, sample_rate=44100)

# Fill buffer
for i in range(len(buf)):
    buf[i] = 0.5

# Process in-place
pycdp.apply_gain(ctx, buf, 2.0, clip=True)
pycdp.apply_normalize(ctx, buf, target_level=0.9)

# Get peak info
level, pos = pycdp.get_peak(ctx, buf)

# Access via buffer protocol
mv = memoryview(buf)
```

## API Reference

### File I/O

| Function | Description |
|----------|-------------|
| `read_file(path)` | Read audio file, returns Buffer |
| `write_file(path, buffer)` | Write buffer to audio file |

### Gain and Normalization

| Function | Description |
|----------|-------------|
| `gain(samples, gain_factor, ...)` | Apply linear gain |
| `gain_db(samples, db, ...)` | Apply gain in decibels |
| `normalize(samples, target, ...)` | Normalize to target peak (0-1) |
| `normalize_db(samples, target_db, ...)` | Normalize to target dB |
| `phase_invert(samples, ...)` | Invert phase |
| `peak(samples, ...)` | Find peak level and position |

### Spatial and Panning

| Function | Description |
|----------|-------------|
| `pan(samples, position, ...)` | Pan mono to stereo (-1 to 1) |
| `pan_envelope(samples, envelope, ...)` | Pan with time-varying envelope |
| `mirror(samples, ...)` | Mirror/swap stereo channels |
| `narrow(samples, width, ...)` | Adjust stereo width (0=mono, 1=full) |

### Mixing

| Function | Description |
|----------|-------------|
| `mix(buffers, ...)` | Mix multiple buffers together |
| `mix2(buf1, buf2, ...)` | Mix two buffers |

### Buffer Utilities

| Function | Description |
|----------|-------------|
| `reverse(samples, ...)` | Reverse audio |
| `fade_in(samples, duration, ...)` | Apply fade in |
| `fade_out(samples, duration, ...)` | Apply fade out |
| `concat(buffers, ...)` | Concatenate buffers |

### Channel Operations

| Function | Description |
|----------|-------------|
| `to_mono(samples, ...)` | Convert to mono |
| `to_stereo(samples, ...)` | Convert mono to stereo |
| `extract_channel(samples, channel, ...)` | Extract single channel |
| `merge_channels(left, right, ...)` | Merge two mono buffers to stereo |
| `split_channels(samples, ...)` | Split stereo to two mono buffers |
| `interleave(channels, ...)` | Interleave multiple mono buffers |

### Time and Pitch

| Function | Description |
|----------|-------------|
| `time_stretch(samples, stretch_factor, ...)` | Time stretch without pitch change |
| `modify_speed(samples, speed, ...)` | Change speed (affects pitch) |
| `pitch_shift(samples, semitones, ...)` | Shift pitch without time change |

### Spectral Processing

| Function | Description |
|----------|-------------|
| `spectral_blur(samples, blur_amount, ...)` | Blur/smear spectrum over time |
| `spectral_shift(samples, shift, ...)` | Shift spectrum up/down |
| `spectral_stretch(samples, stretch, ...)` | Stretch/compress spectrum |
| `spectral_focus(samples, freq, bandwidth, ...)` | Focus on frequency region |
| `spectral_hilite(samples, freq, gain, ...)` | Highlight frequency region |
| `spectral_fold(samples, freq, ...)` | Fold spectrum around frequency |
| `spectral_clean(samples, threshold, ...)` | Remove spectral noise |

### Filters

| Function | Description |
|----------|-------------|
| `filter_lowpass(samples, cutoff, ...)` | Low-pass filter |
| `filter_highpass(samples, cutoff, ...)` | High-pass filter |
| `filter_bandpass(samples, low, high, ...)` | Band-pass filter |
| `filter_notch(samples, freq, width, ...)` | Notch/band-reject filter |

### Dynamics and EQ

| Function | Description |
|----------|-------------|
| `gate(samples, threshold, ...)` | Noise gate |
| `compressor(samples, threshold, ratio, ...)` | Dynamic range compressor |
| `limiter(samples, threshold, ...)` | Peak limiter |
| `eq_parametric(samples, freq, gain, q, ...)` | Parametric EQ band |
| `envelope_follow(samples, ...)` | Extract amplitude envelope |
| `envelope_apply(samples, envelope, ...)` | Apply envelope to audio |

### Effects

| Function | Description |
|----------|-------------|
| `bitcrush(samples, bits, ...)` | Bit depth reduction |
| `ring_mod(samples, freq, ...)` | Ring modulation |
| `delay(samples, time, feedback, ...)` | Delay effect |
| `chorus(samples, depth, rate, ...)` | Chorus effect |
| `flanger(samples, depth, rate, ...)` | Flanger effect |
| `reverb(samples, size, damping, ...)` | Reverb effect |

### Envelope Shaping

| Function | Description |
|----------|-------------|
| `dovetail(samples, fade_time, ...)` | Apply dovetail fades |
| `tremolo(samples, rate, depth, ...)` | Tremolo effect |
| `attack(samples, attack_time, ...)` | Modify attack transient |

### Distortion

| Function | Description |
|----------|-------------|
| `distort_overload(samples, gain, ...)` | Overload/saturation distortion |
| `distort_reverse(samples, ...)` | Reverse distortion effect |
| `distort_fractal(samples, ...)` | Fractal distortion |
| `distort_shuffle(samples, ...)` | Shuffle distortion |

### Granular Processing

| Function | Description |
|----------|-------------|
| `brassage(samples, ...)` | Granular brassage |
| `freeze(samples, position, ...)` | Granular freeze at position |
| `grain_cloud(samples, density, ...)` | Granular cloud synthesis |
| `grain_extend(samples, extension, ...)` | Granular time extension |
| `texture_simple(samples, ...)` | Simple texture synthesis |
| `texture_multi(samples, ...)` | Multi-layer texture synthesis |

### Morphing and Cross-synthesis

| Function | Description |
|----------|-------------|
| `morph(buf1, buf2, amount, ...)` | Spectral morph between sounds |
| `morph_glide(buf1, buf2, ...)` | Gliding morph over time |
| `cross_synth(carrier, modulator, ...)` | Cross-synthesis (vocoder-like) |

### Analysis

| Function | Description |
|----------|-------------|
| `pitch(samples, ...)` | Extract pitch data |
| `formants(samples, ...)` | Extract formant data |
| `get_partials(samples, ...)` | Extract partial/harmonic data |

### Experimental/Chaos

| Function | Description |
|----------|-------------|
| `strange(samples, ...)` | Strange attractor transformation |
| `brownian(samples, ...)` | Brownian motion transformation |
| `crystal(samples, ...)` | Crystal growth patterns |
| `fractal(samples, ...)` | Fractal transformation |
| `quirk(samples, ...)` | Quirky transformation |
| `chirikov(samples, ...)` | Chirikov map transformation |
| `cantor(samples, ...)` | Cantor set transformation |
| `cascade(samples, ...)` | Cascade transformation |
| `fracture(samples, ...)` | Fracture transformation |
| `tesselate(samples, ...)` | Tesselation transformation |

### Utility Functions

| Function | Description |
|----------|-------------|
| `gain_to_db(gain)` | Convert linear gain to decibels |
| `db_to_gain(db)` | Convert decibels to linear gain |
| `version()` | Get library version string |

### Low-level Functions

These work with explicit Context and Buffer objects:

| Function | Description |
|----------|-------------|
| `apply_gain(ctx, buf, gain, clip)` | Apply gain in-place |
| `apply_gain_db(ctx, buf, db, clip)` | Apply dB gain in-place |
| `apply_normalize(ctx, buf, target)` | Normalize in-place |
| `apply_normalize_db(ctx, buf, target_db)` | Normalize to dB in-place |
| `apply_phase_invert(ctx, buf)` | Invert phase in-place |
| `get_peak(ctx, buf)` | Get peak level and position |

### Classes

- `Context` - Processing context (holds error state)
- `Buffer` - Audio buffer with buffer protocol support
  - `Buffer.create(frames, channels, sample_rate)` - Create new buffer
  - Supports indexing, len(), and memoryview

### Constants

- `FLAG_NONE` - No processing flags
- `FLAG_CLIP` - Clip output to [-1.0, 1.0]

### Exceptions

- `CDPError` - Raised on processing errors

## Architecture

```
pycdp/                      # Python package
  src/pycdp/
    __init__.py             # Public exports
    _core.pyx               # Cython bindings
    cdp_lib.pxd             # Cython declarations
  CMakeLists.txt            # Builds extension

libcdp/                     # C library
  include/
    cdp.h                   # Main public API
    cdp_error.h             # Error codes
    cdp_types.h             # Type definitions
  cdp_lib/
    cdp_lib.h               # Library internal header
    cdp_*.h                 # Category headers (filters, effects, etc.)
  src/
    context.c               # Context management
    buffer.c                # Buffer management
    gain.c                  # Gain/amplitude operations
    io.c                    # File I/O
    channel.c               # Channel operations
    mix.c                   # Mixing operations
    spatial.c               # Spatial/panning
    utils.c                 # Utilities
    error.c                 # Error handling
```

## Development

```bash
# Build
make build

# Run tests
make test

# Lint and format
make lint
make format

# Type check
make typecheck

# Full QA
make qa

# Build wheel
make wheel

# See all targets
make help
```

## Adding New Operations

To add more CDP operations:

1. Add C implementation to `libcdp/cdp_lib/<operation>.c`
2. Add function declarations to appropriate header in `libcdp/cdp_lib/`
3. Export from `libcdp/cdp_lib/cdp_lib.h`
4. Update `libcdp/CMakeLists.txt` to include new source file
5. Add Cython declarations to `pycdp/src/pycdp/cdp_lib.pxd`
6. Add Cython bindings to `pycdp/src/pycdp/_core.pyx`
7. Export from `pycdp/src/pycdp/__init__.py`
8. Add tests to both `libcdp/tests/` and `pycdp/tests/`

## License

LGPL-2.1-or-later (same as CDP)
