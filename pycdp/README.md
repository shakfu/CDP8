# pycdp

Python bindings for the CDP (Composers Desktop Project) audio processing library.

## Overview

pycdp provides Python access to CDP's audio processing algorithms. It uses Cython with memoryviews and the buffer protocol for zero-copy interoperability with numpy, `array.array`, and other buffer-compatible objects - **no numpy dependency required**.

## Installation

```bash
# Build and install in development mode
make build

# Or with uv directly
uv sync
```

## Usage

### High-level API

Works with any float32 buffer (numpy arrays, `array.array('f')`, memoryview, etc.):

```python
import array
import pycdp

# Create sample data
samples = array.array('f', [0.5, 0.3, -0.2, 0.8, -0.4])

# Apply gain (linear)
result = pycdp.gain(samples, gain_factor=2.0)

# Apply gain (decibels)
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

### High-level Functions

| Function | Description |
|----------|-------------|
| `gain(samples, gain_factor, sample_rate, clip)` | Apply linear gain |
| `gain_db(samples, db, sample_rate, clip)` | Apply gain in decibels |
| `normalize(samples, target, sample_rate)` | Normalize to target peak (0-1) |
| `normalize_db(samples, target_db, sample_rate)` | Normalize to target dB |
| `phase_invert(samples, sample_rate)` | Invert phase |
| `peak(samples, sample_rate)` | Find peak level and position |

### Utility Functions

| Function | Description |
|----------|-------------|
| `gain_to_db(gain)` | Convert linear gain to decibels |
| `db_to_gain(db)` | Convert decibels to linear gain |
| `version()` | Get library version string |

### Classes

- `Context` - Processing context (holds error state)
- `Buffer` - Audio buffer with buffer protocol support

### Constants

- `FLAG_NONE` - No processing flags
- `FLAG_CLIP` - Clip output to [-1.0, 1.0]

### Exceptions

- `CDPError` - Raised on processing errors (e.g., normalizing silence)

## Architecture

```
pycdp/                      # Python package
  src/pycdp/
    __init__.py             # Public exports
    _core.pyx               # Cython bindings
  CMakeLists.txt            # Builds libcdp + Cython extension

libcdp/                     # C library
  include/
    cdp.h                   # Public API
    cdp_error.h             # Error codes
    cdp_types.h             # Type definitions
  src/
    context.c               # Context management
    buffer.c                # Buffer management
    gain.c                  # Gain/amplitude operations
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

To add more CDP operations (e.g., filters, time-stretch):

1. Add C implementation to `libcdp/src/<operation>.c`
2. Add function declarations to `libcdp/include/cdp.h`
3. Update `pycdp/CMakeLists.txt` to include new source file
4. Add Cython bindings to `pycdp/src/pycdp/_core.pyx`
5. Export from `pycdp/src/pycdp/__init__.py`
6. Add tests to both `libcdp/tests/` and `pycdp/tests/`

## License

LGPL-2.1-or-later (same as CDP)
