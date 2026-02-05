# CDP8 Library Conversion Design

## Overview

This document describes the architecture for converting CDP8's 220 executables into a C library (`libcdp`) with Python/Cython bindings.

## Goals

1. **Programmatic access**: Call CDP functions from code without subprocess overhead
2. **In-memory processing**: Process audio buffers directly, not just files
3. **Thread safety**: Multiple concurrent operations without interference
4. **Incremental conversion**: One program at a time, maintaining backwards compatibility
5. **Python integration**: Clean Cython wrapper for Python ecosystem

---

## Architecture

### Layer Structure

```
+--------------------------------------------------+
|              Python/Cython Bindings              |
|                   (cycdp)                        |
+--------------------------------------------------+
|                  Public C API                    |
|        (libcdp.h - stable, versioned)            |
+--------------------------------------------------+
|               Internal Adapters                  |
|     (bridge old code to new API patterns)        |
+--------------------------------------------------+
|              Existing CDP Code                   |
|    (cdp2k, sfsys, pvxio2, processing code)       |
+--------------------------------------------------+
```

### Directory Structure

```
CDP8/
  libcdp/
    include/
      cdp.h              # Public API header
      cdp_types.h        # Public type definitions
      cdp_error.h        # Error codes and handling
    src/
      context.c          # Context management
      error.c            # Error handling
      buffer.c           # Buffer management
      io.c               # File and memory I/O abstraction
      adapters/
        gain.c           # Adapter for gain processing
        channels.c       # Adapter for channel operations
        ...              # One per converted program
    python/
      cycdp/
        __init__.py
        _cdp.pyx         # Cython definitions
        _cdp.pxd         # Cython declarations
        gain.py          # High-level Python wrapper
        ...
      setup.py
    tests/
      test_gain.c        # C unit tests
      test_gain.py       # Python unit tests
```

---

## Core Data Structures

### Context Structure

```c
// cdp_types.h

typedef struct cdp_context cdp_context;

// Opaque handle - internal structure hidden from users
cdp_context* cdp_context_create(void);
void cdp_context_destroy(cdp_context* ctx);

// Internal structure (in context.c)
struct cdp_context {
    // Error state (thread-local would be ideal, but this works per-context)
    int error_code;
    char error_message[2400];

    // Processing state (wraps dataptr dz)
    struct datalist* dz;

    // Mode flags (moved from globals)
    int interactive_mode;    // was: sloom
    int batch_mode;          // was: sloombatch

    // I/O abstraction
    cdp_io_source* input;
    cdp_io_sink* output;

    // Buffer management
    cdp_buffer_pool* buffers;
};
```

### I/O Abstraction

```c
// cdp_types.h

typedef enum {
    CDP_IO_FILE,
    CDP_IO_MEMORY
} cdp_io_type;

typedef struct {
    cdp_io_type type;
    union {
        struct {
            const char* path;
            int fd;  // internal file descriptor
        } file;
        struct {
            float* samples;
            size_t sample_count;
            size_t position;
            int channels;
            int sample_rate;
        } memory;
    } data;
} cdp_io_source;

typedef struct {
    cdp_io_type type;
    union {
        struct {
            const char* path;
            int fd;
        } file;
        struct {
            float* samples;
            size_t capacity;
            size_t sample_count;
            int channels;
            int sample_rate;
        } memory;
    } data;
} cdp_io_sink;
```

### Audio Format

```c
// cdp_types.h

typedef struct {
    int sample_rate;
    int channels;
    size_t sample_count;  // total samples (frames * channels)
    size_t frame_count;   // sample_count / channels
} cdp_audio_info;

typedef struct {
    float* samples;          // interleaved float samples
    size_t sample_count;
    cdp_audio_info info;
    int owns_memory;         // 1 if we should free samples
} cdp_buffer;
```

---

## Error Handling

```c
// cdp_error.h

typedef enum {
    CDP_OK              =  0,
    CDP_FINISHED        =  0,
    CDP_CONTINUE        =  1,
    CDP_GOAL_FAILED     = -1,
    CDP_USER_ERROR      = -2,
    CDP_DATA_ERROR      = -3,
    CDP_MEMORY_ERROR    = -4,
    CDP_SYSTEM_ERROR    = -5,
    CDP_PROGRAM_ERROR   = -6,
    CDP_INVALID_ARG     = -10,
    CDP_NOT_INITIALIZED = -11,
} cdp_error;

// Get last error from context
cdp_error cdp_get_error(cdp_context* ctx);
const char* cdp_get_error_message(cdp_context* ctx);

// Set error (internal use)
void cdp_set_error(cdp_context* ctx, cdp_error code, const char* fmt, ...);
```

---

## Public API Design

### Pattern: Every Function

```c
cdp_error cdp_<operation>(
    cdp_context* ctx,           // Always first
    <inputs>,                   // Input files/buffers
    <outputs>,                  // Output files/buffers (or in-place)
    <parameters>                // Operation-specific params
);
```

### Gain/Loudness API (First Implementation)

```c
// cdp.h

// --- Context Management ---
cdp_context* cdp_context_create(void);
void cdp_context_destroy(cdp_context* ctx);

// --- I/O Setup ---
// File-based
cdp_error cdp_set_input_file(cdp_context* ctx, const char* path);
cdp_error cdp_set_output_file(cdp_context* ctx, const char* path);

// Memory-based
cdp_error cdp_set_input_buffer(cdp_context* ctx,
                               const float* samples,
                               size_t sample_count,
                               int channels,
                               int sample_rate);
cdp_error cdp_set_output_buffer(cdp_context* ctx,
                                float* samples,
                                size_t capacity);
// Or let library allocate
cdp_error cdp_set_output_buffer_auto(cdp_context* ctx);
cdp_buffer* cdp_get_output_buffer(cdp_context* ctx);

// --- Gain Operations ---

// Simple gain (multiply all samples)
cdp_error cdp_gain(cdp_context* ctx, double gain_factor);

// Gain in decibels
cdp_error cdp_gain_db(cdp_context* ctx, double gain_db);

// Normalize to target level (0.0-1.0, where 1.0 = 0dBFS)
cdp_error cdp_normalize(cdp_context* ctx, double target_level);

// Normalize to specific dB level
cdp_error cdp_normalize_db(cdp_context* ctx, double target_db);

// Phase invert
cdp_error cdp_phase_invert(cdp_context* ctx);

// Time-varying gain envelope
typedef struct {
    double time;   // seconds
    double gain;   // linear multiplier
} cdp_gain_point;

cdp_error cdp_gain_envelope(cdp_context* ctx,
                            const cdp_gain_point* points,
                            size_t point_count);

// --- Utilities ---

// Find peak level (returns in *peak_level, 0.0-1.0+)
cdp_error cdp_find_peak(cdp_context* ctx, double* peak_level);

// Get audio info from current input
cdp_error cdp_get_audio_info(cdp_context* ctx, cdp_audio_info* info);
```

---

## Cython Bindings

### Low-level Cython (_cdp.pxd)

```cython
# _cdp.pxd
cdef extern from "cdp.h":
    ctypedef struct cdp_context:
        pass

    ctypedef struct cdp_audio_info:
        int sample_rate
        int channels
        size_t sample_count
        size_t frame_count

    ctypedef struct cdp_buffer:
        float* samples
        size_t sample_count
        cdp_audio_info info
        int owns_memory

    ctypedef int cdp_error

    cdp_context* cdp_context_create()
    void cdp_context_destroy(cdp_context* ctx)

    cdp_error cdp_get_error(cdp_context* ctx)
    const char* cdp_get_error_message(cdp_context* ctx)

    cdp_error cdp_set_input_file(cdp_context* ctx, const char* path)
    cdp_error cdp_set_output_file(cdp_context* ctx, const char* path)
    cdp_error cdp_set_input_buffer(cdp_context* ctx, const float* samples,
                                   size_t sample_count, int channels, int sample_rate)
    cdp_error cdp_set_output_buffer_auto(cdp_context* ctx)
    cdp_buffer* cdp_get_output_buffer(cdp_context* ctx)

    cdp_error cdp_gain(cdp_context* ctx, double gain_factor)
    cdp_error cdp_gain_db(cdp_context* ctx, double gain_db)
    cdp_error cdp_normalize(cdp_context* ctx, double target_level)
    cdp_error cdp_phase_invert(cdp_context* ctx)
    cdp_error cdp_find_peak(cdp_context* ctx, double* peak_level)
    cdp_error cdp_get_audio_info(cdp_context* ctx, cdp_audio_info* info)
```

### High-level Python (gain.py)

```python
# cycdp/gain.py
import numpy as np
from . import _cdp

class CDPError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)

class AudioProcessor:
    """Context manager for CDP audio processing."""

    def __init__(self):
        self._ctx = _cdp.Context()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._ctx = None

    def _check_error(self, result):
        if result < 0:
            raise CDPError(result, self._ctx.get_error_message())

    # File-based processing
    def gain_file(self, input_path: str, output_path: str,
                  gain: float, db: bool = False) -> None:
        """Apply gain to audio file."""
        self._check_error(self._ctx.set_input_file(input_path))
        self._check_error(self._ctx.set_output_file(output_path))
        if db:
            self._check_error(self._ctx.gain_db(gain))
        else:
            self._check_error(self._ctx.gain(gain))

    def normalize_file(self, input_path: str, output_path: str,
                       target: float = 1.0, db: bool = False) -> None:
        """Normalize audio file to target level."""
        self._check_error(self._ctx.set_input_file(input_path))
        self._check_error(self._ctx.set_output_file(output_path))
        if db:
            self._check_error(self._ctx.normalize_db(target))
        else:
            self._check_error(self._ctx.normalize(target))

    # In-memory processing
    def gain(self, samples: np.ndarray, sample_rate: int,
             gain: float, db: bool = False) -> np.ndarray:
        """Apply gain to audio buffer, return new buffer."""
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        channels = 1 if samples.ndim == 1 else samples.shape[1]
        flat = samples.ravel()

        self._check_error(self._ctx.set_input_buffer(
            flat, len(flat), channels, sample_rate))
        self._check_error(self._ctx.set_output_buffer_auto())

        if db:
            self._check_error(self._ctx.gain_db(gain))
        else:
            self._check_error(self._ctx.gain(gain))

        output = self._ctx.get_output_buffer()
        result = output.to_numpy()
        if channels > 1:
            result = result.reshape(-1, channels)
        return result

    def normalize(self, samples: np.ndarray, sample_rate: int,
                  target: float = 1.0) -> np.ndarray:
        """Normalize audio buffer to target level."""
        # Similar pattern to gain()
        ...

    def find_peak(self, samples: np.ndarray, sample_rate: int) -> float:
        """Find peak level in audio buffer."""
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        channels = 1 if samples.ndim == 1 else samples.shape[1]
        flat = samples.ravel()

        self._check_error(self._ctx.set_input_buffer(
            flat, len(flat), channels, sample_rate))
        return self._ctx.find_peak()


# Convenience functions
def gain(input_path: str, output_path: str, gain: float, db: bool = False):
    """Apply gain to audio file."""
    with AudioProcessor() as proc:
        proc.gain_file(input_path, output_path, gain, db)

def normalize(input_path: str, output_path: str, target: float = 1.0):
    """Normalize audio file."""
    with AudioProcessor() as proc:
        proc.normalize_file(input_path, output_path, target)
```

---

## Implementation Strategy

### Phase 1: Foundation (gain.c as proof of concept)

1. **Create libcdp directory structure**
2. **Implement context management**
   - `cdp_context_create()` / `cdp_context_destroy()`
   - Wrap global state
3. **Implement I/O abstraction**
   - File-based I/O using existing sfsys
   - Memory-based I/O with new code
4. **Adapt gain.c**
   - Create `libcdp/src/adapters/gain.c`
   - Wrap `gain()`, `do_normalise()`, `find_max()`, etc.
   - Replace global `errstr` with context error
5. **Create Cython bindings for gain**
6. **Write tests** (C and Python)

### Phase 2: Expand Core Operations

Programs to convert (in order of complexity/usefulness):
1. **houskeep/channels.c** - channel operations
2. **sfedit/** - basic editing (cut, splice, etc.)
3. **filter/** - filtering operations
4. **stretch/** - time stretching
5. **pitch/** - pitch shifting
6. **morph/** - morphing
7. ... continue with remaining programs

### Phase 3: Spectral Processing

Convert phase vocoder operations:
- **pvoc/** - analysis/synthesis
- **spec/** - spectral processing
- **blur/** - spectral blur
- etc.

### Phase 4: Advanced Operations

- Complex multi-file operations
- Granular synthesis (texture/, grain/)
- Formant processing

---

## Build System Integration

### CMakeLists.txt additions

```cmake
# libcdp/CMakeLists.txt

# Build libcdp as shared library
add_library(cdp SHARED
    src/context.c
    src/error.c
    src/buffer.c
    src/io.c
    src/adapters/gain.c
)

target_include_directories(cdp
    PUBLIC include
    PRIVATE ${CMAKE_SOURCE_DIR}/dev/include
    PRIVATE ${CMAKE_SOURCE_DIR}/dev/newinclude
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)

# Link existing CDP libraries
target_link_libraries(cdp
    cdp2k
    sfsys
    pvxio2
    ${EXTRA_LIBRARIES}
)

# Install headers and library
install(TARGETS cdp DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/cdp)

# Python extension (optional, if Cython found)
find_package(Python COMPONENTS Interpreter Development)
if(Python_FOUND)
    # Cython build handled by setup.py, but we can trigger it
    add_custom_target(cycdp
        COMMAND ${Python_EXECUTABLE} setup.py build_ext --inplace
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python
        DEPENDS cdp
    )
endif()
```

---

## Testing Strategy

### C Tests (using simple assertion framework)

```c
// tests/test_gain.c
#include <assert.h>
#include <math.h>
#include "cdp.h"

void test_gain_basic() {
    cdp_context* ctx = cdp_context_create();
    assert(ctx != NULL);

    // Create test buffer
    float samples[100];
    for (int i = 0; i < 100; i++) {
        samples[i] = 0.5f;
    }

    assert(cdp_set_input_buffer(ctx, samples, 100, 1, 44100) == CDP_OK);
    assert(cdp_set_output_buffer_auto(ctx) == CDP_OK);
    assert(cdp_gain(ctx, 2.0) == CDP_OK);

    cdp_buffer* out = cdp_get_output_buffer(ctx);
    assert(out != NULL);
    assert(out->sample_count == 100);

    for (int i = 0; i < 100; i++) {
        assert(fabs(out->samples[i] - 1.0f) < 0.0001f);
    }

    cdp_context_destroy(ctx);
    printf("test_gain_basic: PASSED\n");
}

int main() {
    test_gain_basic();
    // ... more tests
    return 0;
}
```

### Python Tests (pytest)

```python
# tests/test_gain.py
import numpy as np
import pytest
from cycdp import AudioProcessor, gain, normalize

def test_gain_doubles_amplitude():
    with AudioProcessor() as proc:
        samples = np.full(100, 0.5, dtype=np.float32)
        result = proc.gain(samples, 44100, gain=2.0)
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

def test_normalize_to_peak():
    with AudioProcessor() as proc:
        samples = np.array([0.5, -0.3, 0.2], dtype=np.float32)
        result = proc.normalize(samples, 44100, target=1.0)
        assert abs(result.max()) == pytest.approx(1.0, rel=1e-5)

def test_file_round_trip(tmp_path):
    # Create test file, process, verify
    ...
```

---

## Migration Notes

### Handling Globals

| Global | Location | Strategy |
|--------|----------|----------|
| `errstr[2400]` | tkglobals.h | Move to `ctx->error_message` |
| `sloom` | main.c | Move to `ctx->interactive_mode` |
| `sloombatch` | main.c | Move to `ctx->batch_mode` |
| `anal_infiles` | main.c | Move to `ctx->dz` or remove |
| `is_converted_to_stereo` | main.c | Move to `ctx->dz` |

### Wrapper Pattern

For each existing function like `gain()`:

```c
// Original (in dev/modify/gain.c)
static void gain(int samples, double multiplier, double maxnoclip,
                 int *numclipped, int do_clip, double *peak, dataptr dz);

// Adapter (in libcdp/src/adapters/gain.c)
cdp_error cdp_gain(cdp_context* ctx, double gain_factor) {
    if (!ctx) return CDP_INVALID_ARG;
    if (!ctx->input) {
        cdp_set_error(ctx, CDP_INVALID_ARG, "No input set");
        return CDP_INVALID_ARG;
    }

    // Setup dz from context
    dataptr dz = ctx->dz;
    setup_dz_from_io(ctx);

    // Call original
    int numclipped = 0;
    double peak = 0.0;
    gain(dz->ssampsread, gain_factor, F_MAXSAMP,
         &numclipped, 1, &peak, dz);

    // Handle results
    if (numclipped > 0) {
        cdp_set_error(ctx, CDP_OK,
                      "Warning: %d samples clipped", numclipped);
    }

    finalize_output(ctx);
    return CDP_OK;
}
```

---

## Version History

| Version | Changes |
|---------|---------|
| 0.1.0 | Initial design - gain operations only |
| 0.2.0 | (planned) Add channel operations |
| 0.3.0 | (planned) Add basic editing |
| ... | ... |
| 1.0.0 | All 220 programs converted |

