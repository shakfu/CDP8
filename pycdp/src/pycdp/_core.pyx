# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
pycdp Cython extension module - CDP audio processing bindings.

Uses memoryviews and buffer protocol for zero-copy interop with
numpy, array.array, and other buffer-compatible objects.
"""

from libc.stddef cimport size_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cpython.buffer cimport PyBUF_FORMAT, PyBUF_WRITABLE
from cpython.mem cimport PyMem_Malloc, PyMem_Free

# =============================================================================
# C declarations from cdp.h
# =============================================================================
cdef extern from "cdp.h":
    # Error codes
    ctypedef enum cdp_error:
        CDP_OK
        CDP_CONTINUE
        CDP_ERROR_GOAL_FAILED
        CDP_ERROR_INVALID_ARG
        CDP_ERROR_DATA
        CDP_ERROR_MEMORY
        CDP_ERROR_IO
        CDP_ERROR_FORMAT
        CDP_ERROR_STATE
        CDP_ERROR_NOT_FOUND
        CDP_ERROR_INTERNAL

    # Flags
    ctypedef enum cdp_flags:
        CDP_FLAG_NONE
        CDP_FLAG_CLIP
        CDP_FLAG_NORMALIZE
        CDP_FLAG_PRESERVE_PEAK

    # Audio info
    ctypedef struct cdp_audio_info:
        int sample_rate
        int channels
        size_t frame_count
        size_t sample_count

    # Buffer
    ctypedef struct cdp_buffer:
        float* samples
        size_t sample_count
        size_t capacity
        cdp_audio_info info
        int owns_memory

    # Breakpoint
    ctypedef struct cdp_breakpoint:
        double time
        double value

    # Peak info
    ctypedef struct cdp_peak:
        float level
        size_t position

    # Context (opaque)
    ctypedef struct cdp_context:
        pass

    # Version
    const char* cdp_version()

    # Context management
    cdp_context* cdp_context_create()
    void cdp_context_destroy(cdp_context* ctx)
    cdp_error cdp_get_error(const cdp_context* ctx)
    const char* cdp_get_error_message(const cdp_context* ctx)
    void cdp_clear_error(cdp_context* ctx)

    # Buffer management
    cdp_buffer* cdp_buffer_create(size_t frame_count, int channels, int sample_rate)
    cdp_buffer* cdp_buffer_wrap(float* samples, size_t sample_count,
                                int channels, int sample_rate)
    cdp_buffer* cdp_buffer_copy(const float* samples, size_t sample_count,
                                int channels, int sample_rate)
    void cdp_buffer_destroy(cdp_buffer* buf)
    cdp_error cdp_buffer_resize(cdp_buffer* buf, size_t new_frame_count)
    void cdp_buffer_clear(cdp_buffer* buf)

    # Gain operations
    cdp_error cdp_gain(cdp_context* ctx, cdp_buffer* buf, double gain, cdp_flags flags)
    cdp_error cdp_gain_db(cdp_context* ctx, cdp_buffer* buf, double gain_db, cdp_flags flags)
    cdp_error cdp_gain_envelope(cdp_context* ctx, cdp_buffer* buf,
                                const cdp_breakpoint* points, size_t point_count,
                                cdp_flags flags)
    double cdp_find_peak(cdp_context* ctx, const cdp_buffer* buf, cdp_peak* peak)
    cdp_error cdp_normalize(cdp_context* ctx, cdp_buffer* buf, double target_level)
    cdp_error cdp_normalize_db(cdp_context* ctx, cdp_buffer* buf, double target_db)
    cdp_error cdp_phase_invert(cdp_context* ctx, cdp_buffer* buf)

    # Utilities
    double cdp_gain_to_db(double gain)
    double cdp_db_to_gain(double db)

    # Error string
    const char* cdp_error_string(cdp_error err)


# =============================================================================
# Python exports
# =============================================================================

# Export constants
FLAG_NONE = CDP_FLAG_NONE
FLAG_CLIP = CDP_FLAG_CLIP


class CDPError(Exception):
    """Exception raised for CDP library errors."""
    def __init__(self, code, message):
        self.code = code
        super().__init__(f"CDP error {code}: {message}")


cpdef str version():
    """Get CDP library version string."""
    return cdp_version().decode('utf-8')


cpdef double gain_to_db(double gain):
    """Convert linear gain to decibels."""
    return cdp_gain_to_db(gain)


cpdef double db_to_gain(double db):
    """Convert decibels to linear gain."""
    return cdp_db_to_gain(db)


# =============================================================================
# Context class
# =============================================================================
cdef class Context:
    """CDP processing context."""
    cdef cdp_context* _ctx

    def __cinit__(self):
        self._ctx = cdp_context_create()
        if self._ctx is NULL:
            raise MemoryError("Failed to create CDP context")

    def __dealloc__(self):
        if self._ctx is not NULL:
            cdp_context_destroy(self._ctx)
            self._ctx = NULL

    cpdef int get_error(self):
        """Get last error code."""
        return cdp_get_error(self._ctx)

    cpdef str get_error_message(self):
        """Get last error message."""
        return cdp_get_error_message(self._ctx).decode('utf-8')

    cpdef void clear_error(self):
        """Clear error state."""
        cdp_clear_error(self._ctx)

    cdef void _check_error(self, cdp_error err) except *:
        """Raise exception if error occurred."""
        if err < 0:
            msg = cdp_get_error_message(self._ctx).decode('utf-8')
            raise CDPError(err, msg)

    cdef cdp_context* ptr(self):
        """Get raw pointer (for internal use)."""
        return self._ctx


# =============================================================================
# Buffer class with buffer protocol support
# =============================================================================
cdef class Buffer:
    """CDP audio buffer with Python buffer protocol support.

    Can be created from any object supporting the buffer protocol
    (numpy arrays, array.array, memoryview, bytes, etc.)
    """
    cdef cdp_buffer* _buf
    cdef bint _owns_buffer
    cdef Py_ssize_t _shape[1]
    cdef Py_ssize_t _strides[1]

    def __cinit__(self):
        self._buf = NULL
        self._owns_buffer = False

    def __dealloc__(self):
        if self._owns_buffer and self._buf is not NULL:
            cdp_buffer_destroy(self._buf)
            self._buf = NULL

    @staticmethod
    def create(size_t frame_count, int channels=1, int sample_rate=44100):
        """Create a new buffer with allocated memory."""
        cdef Buffer buf = Buffer()
        buf._buf = cdp_buffer_create(frame_count, channels, sample_rate)
        if buf._buf is NULL:
            raise MemoryError("Failed to create buffer")
        buf._owns_buffer = True
        return buf

    @staticmethod
    def from_memoryview(float[::1] samples not None,
                        int channels=1, int sample_rate=44100):
        """Create a buffer by copying from a memoryview."""
        cdef Buffer buf = Buffer()
        cdef size_t sample_count = samples.shape[0]

        buf._buf = cdp_buffer_copy(&samples[0], sample_count,
                                   channels, sample_rate)
        if buf._buf is NULL:
            raise MemoryError("Failed to copy buffer")

        buf._owns_buffer = True
        return buf

    def to_list(self):
        """Convert buffer to Python list."""
        if self._buf is NULL:
            raise ValueError("Buffer is not initialized")

        cdef list result = []
        cdef size_t i
        for i in range(self._buf.sample_count):
            result.append(self._buf.samples[i])
        return result

    def to_bytes(self):
        """Convert buffer to bytes (raw float32 data)."""
        if self._buf is NULL:
            raise ValueError("Buffer is not initialized")

        cdef size_t nbytes = self._buf.sample_count * sizeof(float)
        return (<char*>self._buf.samples)[:nbytes]

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        """Support buffer protocol for zero-copy access."""
        if self._buf is NULL:
            raise ValueError("Buffer is not initialized")

        self._shape[0] = <Py_ssize_t>self._buf.sample_count
        self._strides[0] = <Py_ssize_t>sizeof(float)

        buffer.buf = <void*>self._buf.samples
        buffer.format = 'f'  # float32
        buffer.internal = NULL
        buffer.itemsize = sizeof(float)
        buffer.len = self._buf.sample_count * sizeof(float)
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0  # Always writable since we own the memory
        buffer.shape = self._shape
        buffer.strides = self._strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    @property
    def sample_count(self):
        if self._buf is NULL:
            return 0
        return self._buf.sample_count

    @property
    def frame_count(self):
        if self._buf is NULL:
            return 0
        return self._buf.info.frame_count

    @property
    def channels(self):
        if self._buf is NULL:
            return 0
        return self._buf.info.channels

    @property
    def sample_rate(self):
        if self._buf is NULL:
            return 0
        return self._buf.info.sample_rate

    def __len__(self):
        if self._buf is NULL:
            return 0
        return self._buf.sample_count

    def __getitem__(self, size_t index):
        if self._buf is NULL:
            raise ValueError("Buffer is not initialized")
        if index >= self._buf.sample_count:
            raise IndexError("Index out of range")
        return self._buf.samples[index]

    def __setitem__(self, size_t index, float value):
        if self._buf is NULL:
            raise ValueError("Buffer is not initialized")
        if index >= self._buf.sample_count:
            raise IndexError("Index out of range")
        self._buf.samples[index] = value

    def clear(self):
        """Set all samples to zero."""
        if self._buf is not NULL:
            cdp_buffer_clear(self._buf)

    cdef cdp_buffer* ptr(self):
        """Get raw pointer (for internal use)."""
        return self._buf


# =============================================================================
# Processing functions (low-level, work with Buffer objects)
# =============================================================================

def apply_gain(Context ctx, Buffer buf, double gain, bint clip=False):
    """Apply gain to buffer (in-place)."""
    cdef cdp_flags flags = CDP_FLAG_CLIP if clip else CDP_FLAG_NONE
    cdef cdp_error err = cdp_gain(ctx.ptr(), buf.ptr(), gain, flags)
    ctx._check_error(err)


def apply_gain_db(Context ctx, Buffer buf, double gain_db, bint clip=False):
    """Apply gain in dB to buffer (in-place)."""
    cdef cdp_flags flags = CDP_FLAG_CLIP if clip else CDP_FLAG_NONE
    cdef cdp_error err = cdp_gain_db(ctx.ptr(), buf.ptr(), gain_db, flags)
    ctx._check_error(err)


def apply_normalize(Context ctx, Buffer buf, double target_level=1.0):
    """Normalize buffer to target level (in-place)."""
    cdef cdp_error err = cdp_normalize(ctx.ptr(), buf.ptr(), target_level)
    ctx._check_error(err)


def apply_normalize_db(Context ctx, Buffer buf, double target_db=0.0):
    """Normalize buffer to target dB level (in-place)."""
    cdef cdp_error err = cdp_normalize_db(ctx.ptr(), buf.ptr(), target_db)
    ctx._check_error(err)


def apply_phase_invert(Context ctx, Buffer buf):
    """Invert phase of buffer (in-place)."""
    cdef cdp_error err = cdp_phase_invert(ctx.ptr(), buf.ptr())
    ctx._check_error(err)


def get_peak(Context ctx, Buffer buf):
    """Find peak level in buffer. Returns (level, frame_position)."""
    cdef cdp_peak peak
    cdef double level = cdp_find_peak(ctx.ptr(), buf.ptr(), &peak)
    if level < 0:
        ctx._check_error(<cdp_error>-1)
    return (peak.level, peak.position)


# =============================================================================
# High-level functions using memoryviews (work with any buffer protocol object)
# =============================================================================

def gain(float[::1] samples not None, double gain_factor=1.0,
         int sample_rate=44100, bint clip=False):
    """Apply gain to audio samples.

    Args:
        samples: Input samples (any float32 buffer: numpy array, array.array('f'), etc.)
        gain_factor: Linear gain (1.0 = unity, 2.0 = +6dB).
        sample_rate: Sample rate in Hz.
        clip: If True, clip output to [-1.0, 1.0].

    Returns:
        Buffer with processed samples (supports buffer protocol).
    """
    cdef Context ctx = Context()
    cdef Buffer buf = Buffer.from_memoryview(samples, 1, sample_rate)

    apply_gain(ctx, buf, gain_factor, clip)
    return buf


def gain_db(float[::1] samples not None, double db=0.0,
            int sample_rate=44100, bint clip=False):
    """Apply gain in decibels to audio samples.

    Args:
        samples: Input samples (any float32 buffer).
        db: Gain in dB (0 = unity, +6 = double, -6 = half).
        sample_rate: Sample rate in Hz.
        clip: If True, clip output to [-1.0, 1.0].

    Returns:
        Buffer with processed samples.
    """
    cdef Context ctx = Context()
    cdef Buffer buf = Buffer.from_memoryview(samples, 1, sample_rate)

    apply_gain_db(ctx, buf, db, clip)
    return buf


def normalize(float[::1] samples not None, double target=1.0,
              int sample_rate=44100):
    """Normalize audio to target peak level.

    Args:
        samples: Input samples (any float32 buffer).
        target: Target peak level (0.0 to 1.0).
        sample_rate: Sample rate in Hz.

    Returns:
        Buffer with normalized samples.

    Raises:
        CDPError: If audio is silent.
    """
    cdef Context ctx = Context()
    cdef Buffer buf = Buffer.from_memoryview(samples, 1, sample_rate)

    apply_normalize(ctx, buf, target)
    return buf


def normalize_db(float[::1] samples not None, double target_db=0.0,
                 int sample_rate=44100):
    """Normalize audio to target peak level in dB.

    Args:
        samples: Input samples (any float32 buffer).
        target_db: Target peak in dB (0 = 0dBFS, -3 = -3dBFS).
        sample_rate: Sample rate in Hz.

    Returns:
        Buffer with normalized samples.
    """
    cdef Context ctx = Context()
    cdef Buffer buf = Buffer.from_memoryview(samples, 1, sample_rate)

    apply_normalize_db(ctx, buf, target_db)
    return buf


def phase_invert(float[::1] samples not None, int sample_rate=44100):
    """Invert phase of audio samples.

    Args:
        samples: Input samples (any float32 buffer).
        sample_rate: Sample rate in Hz.

    Returns:
        Buffer with phase-inverted samples.
    """
    cdef Context ctx = Context()
    cdef Buffer buf = Buffer.from_memoryview(samples, 1, sample_rate)

    apply_phase_invert(ctx, buf)
    return buf


def peak(float[::1] samples not None, int sample_rate=44100):
    """Find peak level in audio samples.

    Args:
        samples: Input samples (any float32 buffer).
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple of (peak_level, sample_position).
    """
    cdef Context ctx = Context()
    cdef Buffer buf = Buffer.from_memoryview(samples, 1, sample_rate)

    return get_peak(ctx, buf)
