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

    # File I/O
    cdp_buffer* cdp_read_file(cdp_context* ctx, const char* path)
    cdp_error cdp_write_file(cdp_context* ctx, const char* path, const cdp_buffer* buf)
    cdp_error cdp_write_file_pcm16(cdp_context* ctx, const char* path, const cdp_buffer* buf)
    cdp_error cdp_write_file_pcm24(cdp_context* ctx, const char* path, const cdp_buffer* buf)

    # Spatial/panning operations
    cdp_buffer* cdp_pan(cdp_context* ctx, const cdp_buffer* buf, double position)
    cdp_buffer* cdp_pan_envelope(cdp_context* ctx, const cdp_buffer* buf,
                                  const cdp_breakpoint* points, size_t point_count)
    cdp_buffer* cdp_mirror(cdp_context* ctx, const cdp_buffer* buf)
    cdp_buffer* cdp_narrow(cdp_context* ctx, const cdp_buffer* buf, double width)

    # Mixing operations
    cdp_buffer* cdp_mix2(cdp_context* ctx, const cdp_buffer* a, const cdp_buffer* b,
                         double gain_a, double gain_b)
    cdp_buffer* cdp_mix(cdp_context* ctx, cdp_buffer** buffers, const double* gains,
                        int count)

    # Channel operations
    cdp_buffer* cdp_to_mono(cdp_context* ctx, const cdp_buffer* buf)
    cdp_buffer* cdp_to_stereo(cdp_context* ctx, const cdp_buffer* buf)
    cdp_buffer* cdp_extract_channel(cdp_context* ctx, const cdp_buffer* buf, int channel)
    cdp_buffer* cdp_merge_channels(cdp_context* ctx, const cdp_buffer* left,
                                    const cdp_buffer* right)
    cdp_buffer** cdp_split_channels(cdp_context* ctx, const cdp_buffer* buf,
                                     int* out_num_channels)
    cdp_buffer* cdp_interleave(cdp_context* ctx, cdp_buffer** buffers, int num_channels)

    # Buffer utilities
    cdp_buffer* cdp_reverse(cdp_context* ctx, const cdp_buffer* buf)
    cdp_error cdp_fade_in(cdp_context* ctx, cdp_buffer* buf, double duration, int fade_type)
    cdp_error cdp_fade_out(cdp_context* ctx, cdp_buffer* buf, double duration, int fade_type)
    cdp_buffer* cdp_concat(cdp_context* ctx, cdp_buffer** buffers, int count)

    # Conversion utilities
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


# =============================================================================
# File I/O functions
# =============================================================================

def read_file(str path not None):
    """Read an audio file into a Buffer.

    Currently supports WAV files (16/24/32-bit PCM and 32-bit float).

    Args:
        path: Path to audio file.

    Returns:
        Buffer containing audio data.

    Raises:
        CDPError: If file cannot be read.
    """
    cdef Context ctx = Context()
    cdef bytes path_bytes = path.encode('utf-8')
    cdef cdp_buffer* c_buf = cdp_read_file(ctx.ptr(), path_bytes)

    if c_buf is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    # Wrap the C buffer in a Python Buffer object
    cdef Buffer buf = Buffer()
    buf._buf = c_buf
    buf._owns_buffer = True
    return buf


def write_file(str path not None, Buffer buf not None, str format="float"):
    """Write a Buffer to an audio file.

    Args:
        path: Path to output file.
        buf: Buffer to write.
        format: Output format - "float" (32-bit float), "pcm16", or "pcm24".

    Raises:
        CDPError: If file cannot be written.
        ValueError: If format is invalid.
    """
    cdef Context ctx = Context()
    cdef bytes path_bytes = path.encode('utf-8')
    cdef cdp_error err

    if format == "float":
        err = cdp_write_file(ctx.ptr(), path_bytes, buf.ptr())
    elif format == "pcm16":
        err = cdp_write_file_pcm16(ctx.ptr(), path_bytes, buf.ptr())
    elif format == "pcm24":
        err = cdp_write_file_pcm24(ctx.ptr(), path_bytes, buf.ptr())
    else:
        raise ValueError(f"Invalid format: {format}. Use 'float', 'pcm16', or 'pcm24'.")

    ctx._check_error(err)


# =============================================================================
# Spatial/panning operations
# =============================================================================

def pan(Buffer buf not None, double position=0.0):
    """Pan a mono buffer to stereo with a static pan position.

    Uses CDP's geometric panning model for natural sound positioning.

    Args:
        buf: Input buffer (must be mono).
        position: Pan position: -1.0 = left, 0.0 = center, +1.0 = right.
                  Values beyond -1/+1 simulate sound beyond speakers.

    Returns:
        New stereo Buffer.

    Raises:
        CDPError: If input is not mono.
    """
    cdef Context ctx = Context()
    cdef cdp_buffer* c_result = cdp_pan(ctx.ptr(), buf.ptr(), position)

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


def pan_envelope(Buffer buf not None, list points not None):
    """Pan a mono buffer to stereo with time-varying position.

    Uses CDP's geometric panning model with breakpoint-based automation.

    Args:
        buf: Input buffer (must be mono).
        points: List of (time, position) tuples defining the pan envelope.
                Times are in seconds, positions are -1.0 (left) to +1.0 (right).
                Example: [(0.0, -1.0), (1.0, 0.0), (2.0, 1.0)] pans left to right.

    Returns:
        New stereo Buffer.

    Raises:
        CDPError: If input is not mono.
        ValueError: If points list is empty.
    """
    if len(points) == 0:
        raise ValueError("Points list cannot be empty")

    cdef Context ctx = Context()
    cdef size_t point_count = len(points)
    cdef cdp_breakpoint* c_points = <cdp_breakpoint*>malloc(point_count * sizeof(cdp_breakpoint))

    if c_points is NULL:
        raise MemoryError("Failed to allocate breakpoints")

    cdef size_t i
    for i in range(point_count):
        c_points[i].time = points[i][0]
        c_points[i].value = points[i][1]

    cdef cdp_buffer* c_result = cdp_pan_envelope(ctx.ptr(), buf.ptr(), c_points, point_count)
    free(c_points)

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


def mirror(Buffer buf not None):
    """Mirror (swap) left and right channels of a stereo buffer.

    Args:
        buf: Input buffer (must be stereo).

    Returns:
        New Buffer with swapped channels.

    Raises:
        CDPError: If input is not stereo.
    """
    cdef Context ctx = Context()
    cdef cdp_buffer* c_result = cdp_mirror(ctx.ptr(), buf.ptr())

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


def narrow(Buffer buf not None, double width=1.0):
    """Narrow or widen stereo image.

    Args:
        buf: Input buffer (must be stereo).
        width: Stereo width: 0.0 = mono, 1.0 = unchanged, >1.0 = wider.

    Returns:
        New Buffer with adjusted stereo width.

    Raises:
        CDPError: If input is not stereo or width is negative.
    """
    cdef Context ctx = Context()
    cdef cdp_buffer* c_result = cdp_narrow(ctx.ptr(), buf.ptr(), width)

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


# =============================================================================
# Mixing operations
# =============================================================================

def mix2(Buffer a not None, Buffer b not None, double gain_a=1.0, double gain_b=1.0):
    """Mix two buffers together with optional gains.

    Args:
        a: First buffer.
        b: Second buffer (must have same channels and sample rate).
        gain_a: Gain for first buffer (default 1.0).
        gain_b: Gain for second buffer (default 1.0).

    Returns:
        New Buffer containing the mix. Length is max of both inputs.

    Raises:
        CDPError: If buffers are incompatible.
    """
    cdef Context ctx = Context()
    cdef cdp_buffer* c_result = cdp_mix2(ctx.ptr(), a.ptr(), b.ptr(), gain_a, gain_b)

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


def mix(list buffers not None, list gains=None):
    """Mix multiple buffers together with optional gains.

    Args:
        buffers: List of Buffers to mix (all must have same channels/rate).
        gains: Optional list of gains (one per buffer). Default is unity gain.

    Returns:
        New Buffer containing the mix. Length is max of all inputs.

    Raises:
        CDPError: If buffers are incompatible.
        ValueError: If buffers list is empty or gains length doesn't match.
    """
    if len(buffers) == 0:
        raise ValueError("Buffer list cannot be empty")

    if gains is not None and len(gains) != len(buffers):
        raise ValueError("Gains list must have same length as buffers list")

    cdef Context ctx = Context()
    cdef int count = len(buffers)
    cdef int i
    cdef Buffer buf
    cdef cdp_buffer** c_buffers = <cdp_buffer**>malloc(count * sizeof(cdp_buffer*))
    cdef double* c_gains = NULL

    if c_buffers is NULL:
        raise MemoryError("Failed to allocate buffer array")

    if gains is not None:
        c_gains = <double*>malloc(count * sizeof(double))
        if c_gains is NULL:
            free(c_buffers)
            raise MemoryError("Failed to allocate gains array")
        for i in range(count):
            c_gains[i] = gains[i]

    for i in range(count):
        buf = buffers[i]
        c_buffers[i] = buf.ptr()

    cdef cdp_buffer* c_result = cdp_mix(ctx.ptr(), c_buffers, c_gains, count)

    free(c_buffers)
    if c_gains is not NULL:
        free(c_gains)

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


# =============================================================================
# Buffer utilities
# =============================================================================

def reverse(Buffer buf not None):
    """Reverse audio buffer.

    Args:
        buf: Input buffer.

    Returns:
        New Buffer with reversed audio.
    """
    cdef Context ctx = Context()
    cdef cdp_buffer* c_result = cdp_reverse(ctx.ptr(), buf.ptr())

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


def fade_in(Buffer buf not None, double duration, str curve="linear"):
    """Apply fade in to buffer (in-place).

    Args:
        buf: Buffer to process (modified in-place).
        duration: Fade duration in seconds.
        curve: "linear" or "exponential" (equal power).

    Returns:
        The same buffer (for chaining).

    Raises:
        ValueError: If curve type is invalid.
    """
    cdef int fade_type
    if curve == "linear":
        fade_type = 0
    elif curve == "exponential":
        fade_type = 1
    else:
        raise ValueError(f"Invalid curve: {curve}. Use 'linear' or 'exponential'.")

    cdef Context ctx = Context()
    cdef cdp_error err = cdp_fade_in(ctx.ptr(), buf.ptr(), duration, fade_type)
    ctx._check_error(err)
    return buf


def fade_out(Buffer buf not None, double duration, str curve="linear"):
    """Apply fade out to buffer (in-place).

    Args:
        buf: Buffer to process (modified in-place).
        duration: Fade duration in seconds.
        curve: "linear" or "exponential" (equal power).

    Returns:
        The same buffer (for chaining).

    Raises:
        ValueError: If curve type is invalid.
    """
    cdef int fade_type
    if curve == "linear":
        fade_type = 0
    elif curve == "exponential":
        fade_type = 1
    else:
        raise ValueError(f"Invalid curve: {curve}. Use 'linear' or 'exponential'.")

    cdef Context ctx = Context()
    cdef cdp_error err = cdp_fade_out(ctx.ptr(), buf.ptr(), duration, fade_type)
    ctx._check_error(err)
    return buf


def concat(list buffers not None):
    """Concatenate multiple buffers into one.

    Args:
        buffers: List of Buffers (all must have same channels/rate).

    Returns:
        New concatenated Buffer.

    Raises:
        CDPError: If buffers are incompatible.
        ValueError: If list is empty.
    """
    if len(buffers) == 0:
        raise ValueError("Buffer list cannot be empty")

    cdef Context ctx = Context()
    cdef int count = len(buffers)
    cdef cdp_buffer** c_buffers = <cdp_buffer**>malloc(count * sizeof(cdp_buffer*))

    if c_buffers is NULL:
        raise MemoryError("Failed to allocate buffer array")

    cdef int i
    cdef Buffer b
    for i in range(count):
        b = buffers[i]
        c_buffers[i] = b.ptr()

    cdef cdp_buffer* c_result = cdp_concat(ctx.ptr(), c_buffers, count)
    free(c_buffers)

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


# =============================================================================
# Channel operations
# =============================================================================

def to_mono(Buffer buf not None):
    """Convert multi-channel buffer to mono by averaging all channels.

    Args:
        buf: Input buffer (any channel count).

    Returns:
        New mono Buffer.

    Raises:
        CDPError: On error.
    """
    cdef Context ctx = Context()
    cdef cdp_buffer* c_result = cdp_to_mono(ctx.ptr(), buf.ptr())

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


def to_stereo(Buffer buf not None):
    """Convert mono buffer to stereo by duplicating the channel.

    Args:
        buf: Input buffer (must be mono).

    Returns:
        New stereo Buffer.

    Raises:
        CDPError: If input is not mono.
    """
    cdef Context ctx = Context()
    cdef cdp_buffer* c_result = cdp_to_stereo(ctx.ptr(), buf.ptr())

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


def extract_channel(Buffer buf not None, int channel):
    """Extract a single channel from a multi-channel buffer.

    Args:
        buf: Input buffer.
        channel: Channel index (0-based, 0=left, 1=right for stereo).

    Returns:
        New mono Buffer containing the extracted channel.

    Raises:
        CDPError: If channel index is out of range.
    """
    cdef Context ctx = Context()
    cdef cdp_buffer* c_result = cdp_extract_channel(ctx.ptr(), buf.ptr(), channel)

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


def merge_channels(Buffer left not None, Buffer right not None):
    """Merge two mono buffers into a stereo buffer.

    Args:
        left: Left channel buffer (must be mono).
        right: Right channel buffer (must be mono, same length and sample rate).

    Returns:
        New stereo Buffer.

    Raises:
        CDPError: If inputs are not compatible mono buffers.
    """
    cdef Context ctx = Context()
    cdef cdp_buffer* c_result = cdp_merge_channels(ctx.ptr(), left.ptr(), right.ptr())

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


def split_channels(Buffer buf not None):
    """Split a multi-channel buffer into separate mono buffers.

    Args:
        buf: Input buffer.

    Returns:
        List of mono Buffers (one per channel).

    Raises:
        CDPError: On error.
    """
    cdef Context ctx = Context()
    cdef int num_channels = 0
    cdef cdp_buffer** c_buffers = cdp_split_channels(ctx.ptr(), buf.ptr(), &num_channels)

    if c_buffers is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    # Wrap each buffer in a Python Buffer object
    cdef list result = []
    cdef Buffer pybuf
    cdef int i

    for i in range(num_channels):
        pybuf = Buffer()
        pybuf._buf = c_buffers[i]
        pybuf._owns_buffer = True
        result.append(pybuf)

    # Free the array (but not the buffers - they're now owned by Python)
    free(c_buffers)

    return result


def interleave(list buffers not None):
    """Interleave multiple mono buffers into a single multi-channel buffer.

    Args:
        buffers: List of mono Buffers (all must have same length and sample rate).

    Returns:
        New interleaved multi-channel Buffer.

    Raises:
        CDPError: If buffers are not compatible.
        ValueError: If list is empty.
    """
    if len(buffers) == 0:
        raise ValueError("Buffer list cannot be empty")

    cdef Context ctx = Context()
    cdef int num_channels = len(buffers)
    cdef cdp_buffer** c_buffers = <cdp_buffer**>malloc(num_channels * sizeof(cdp_buffer*))

    if c_buffers is NULL:
        raise MemoryError("Failed to allocate buffer array")

    cdef int i
    cdef Buffer b
    for i in range(num_channels):
        b = buffers[i]
        c_buffers[i] = b.ptr()

    cdef cdp_buffer* c_result = cdp_interleave(ctx.ptr(), c_buffers, num_channels)
    free(c_buffers)

    if c_result is NULL:
        msg = ctx.get_error_message()
        raise CDPError(ctx.get_error(), msg)

    cdef Buffer result = Buffer()
    result._buf = c_result
    result._owns_buffer = True
    return result


# =============================================================================
# CDP Library - Native spectral processing (no subprocess)
# =============================================================================

cdef extern from "cdp_lib.h":
    ctypedef struct cdp_lib_ctx:
        pass

    ctypedef struct cdp_lib_buffer:
        float *data
        size_t length
        int channels
        int sample_rate

    cdp_lib_ctx* cdp_lib_init()
    void cdp_lib_cleanup(cdp_lib_ctx* ctx)
    const char* cdp_lib_get_error(cdp_lib_ctx* ctx)

    cdp_lib_buffer* cdp_lib_buffer_create(size_t length, int channels, int sample_rate)
    cdp_lib_buffer* cdp_lib_buffer_from_data(float *data, size_t length,
                                              int channels, int sample_rate)
    void cdp_lib_buffer_free(cdp_lib_buffer* buf)

    cdp_lib_buffer* cdp_lib_time_stretch(cdp_lib_ctx* ctx,
                                          const cdp_lib_buffer* input,
                                          double factor,
                                          int fft_size,
                                          int overlap)

    cdp_lib_buffer* cdp_lib_spectral_blur(cdp_lib_ctx* ctx,
                                           const cdp_lib_buffer* input,
                                           double blur_time,
                                           int fft_size)

    cdp_lib_buffer* cdp_lib_loudness(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double gain_db)

    cdp_lib_buffer* cdp_lib_speed(cdp_lib_ctx* ctx,
                                   const cdp_lib_buffer* input,
                                   double speed)

    # Spectral operations
    cdp_lib_buffer* cdp_lib_pitch_shift(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double semitones,
                                         int fft_size)

    cdp_lib_buffer* cdp_lib_spectral_shift(cdp_lib_ctx* ctx,
                                            const cdp_lib_buffer* input,
                                            double shift_hz,
                                            int fft_size)

    cdp_lib_buffer* cdp_lib_spectral_stretch(cdp_lib_ctx* ctx,
                                              const cdp_lib_buffer* input,
                                              double max_stretch,
                                              double freq_divide,
                                              double exponent,
                                              int fft_size)

    cdp_lib_buffer* cdp_lib_filter_lowpass(cdp_lib_ctx* ctx,
                                            const cdp_lib_buffer* input,
                                            double cutoff_freq,
                                            double attenuation_db,
                                            int fft_size)

    cdp_lib_buffer* cdp_lib_filter_highpass(cdp_lib_ctx* ctx,
                                             const cdp_lib_buffer* input,
                                             double cutoff_freq,
                                             double attenuation_db,
                                             int fft_size)

    cdp_lib_buffer* cdp_lib_filter_bandpass(cdp_lib_ctx* ctx,
                                             const cdp_lib_buffer* input,
                                             double low_freq,
                                             double high_freq,
                                             double attenuation_db,
                                             int fft_size)

    cdp_lib_buffer* cdp_lib_filter_notch(cdp_lib_ctx* ctx,
                                          const cdp_lib_buffer* input,
                                          double center_freq,
                                          double width_hz,
                                          double attenuation_db,
                                          int fft_size)

    cdp_lib_buffer* cdp_lib_gate(cdp_lib_ctx* ctx,
                                  const cdp_lib_buffer* input,
                                  double threshold_db,
                                  double attack_ms,
                                  double release_ms,
                                  double hold_ms)

    cdp_lib_buffer* cdp_lib_bitcrush(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      int bit_depth,
                                      int downsample)

    cdp_lib_buffer* cdp_lib_ring_mod(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double freq,
                                      double mix)

    cdp_lib_buffer* cdp_lib_delay(cdp_lib_ctx* ctx,
                                   const cdp_lib_buffer* input,
                                   double delay_ms,
                                   double feedback,
                                   double mix)

    cdp_lib_buffer* cdp_lib_chorus(cdp_lib_ctx* ctx,
                                    const cdp_lib_buffer* input,
                                    double rate,
                                    double depth_ms,
                                    double mix)

    cdp_lib_buffer* cdp_lib_flanger(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input,
                                     double rate,
                                     double depth_ms,
                                     double feedback,
                                     double mix)

    cdp_lib_buffer* cdp_lib_eq_parametric(cdp_lib_ctx* ctx,
                                           const cdp_lib_buffer* input,
                                           double center_freq,
                                           double gain_db,
                                           double q,
                                           int fft_size)

    cdp_lib_buffer* cdp_lib_envelope_follow(cdp_lib_ctx* ctx,
                                             const cdp_lib_buffer* input,
                                             double attack_ms,
                                             double release_ms,
                                             int mode)

    cdp_lib_buffer* cdp_lib_envelope_apply(cdp_lib_ctx* ctx,
                                            const cdp_lib_buffer* input,
                                            const cdp_lib_buffer* envelope,
                                            double depth)

    cdp_lib_buffer* cdp_lib_compressor(cdp_lib_ctx* ctx,
                                        const cdp_lib_buffer* input,
                                        double threshold_db,
                                        double ratio,
                                        double attack_ms,
                                        double release_ms,
                                        double makeup_gain_db)

    cdp_lib_buffer* cdp_lib_limiter(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input,
                                     double threshold_db,
                                     double attack_ms,
                                     double release_ms)

    cdp_lib_buffer* cdp_lib_morph(cdp_lib_ctx* ctx,
                                   const cdp_lib_buffer* input1,
                                   const cdp_lib_buffer* input2,
                                   double morph_start,
                                   double morph_end,
                                   double exponent,
                                   int fft_size)

    cdp_lib_buffer* cdp_lib_morph_glide(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input1,
                                         const cdp_lib_buffer* input2,
                                         double duration,
                                         int fft_size)

    cdp_lib_buffer* cdp_lib_cross_synth(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input1,
                                         const cdp_lib_buffer* input2,
                                         int mode,
                                         double mix,
                                         int fft_size)

cdef extern from "cdp_envelope.h":
    int CDP_FADE_LINEAR
    int CDP_FADE_EXPONENTIAL

    cdp_lib_buffer* cdp_lib_dovetail(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double fade_in_dur,
                                      double fade_out_dur,
                                      int fade_in_type,
                                      int fade_out_type)

    cdp_lib_buffer* cdp_lib_tremolo(cdp_lib_ctx* ctx,
                                     const cdp_lib_buffer* input,
                                     double freq,
                                     double depth,
                                     double gain)

    cdp_lib_buffer* cdp_lib_attack(cdp_lib_ctx* ctx,
                                    const cdp_lib_buffer* input,
                                    double attack_gain,
                                    double attack_time)

cdef extern from "cdp_distort.h":
    cdp_lib_buffer* cdp_lib_distort_overload(cdp_lib_ctx* ctx,
                                              const cdp_lib_buffer* input,
                                              double clip_level,
                                              double depth)

    cdp_lib_buffer* cdp_lib_distort_reverse(cdp_lib_ctx* ctx,
                                             const cdp_lib_buffer* input,
                                             int cycle_count)

    cdp_lib_buffer* cdp_lib_distort_fractal(cdp_lib_ctx* ctx,
                                             const cdp_lib_buffer* input,
                                             double scaling,
                                             double loudness)

    cdp_lib_buffer* cdp_lib_distort_shuffle(cdp_lib_ctx* ctx,
                                             const cdp_lib_buffer* input,
                                             int chunk_count,
                                             unsigned int seed)

cdef extern from "cdp_reverb.h":
    cdp_lib_buffer* cdp_lib_reverb(cdp_lib_ctx* ctx,
                                    const cdp_lib_buffer* input,
                                    double mix,
                                    double decay_time,
                                    double damping,
                                    double lpfreq,
                                    double predelay)

cdef extern from "cdp_granular.h":
    cdp_lib_buffer* cdp_lib_brassage(cdp_lib_ctx* ctx,
                                      const cdp_lib_buffer* input,
                                      double velocity,
                                      double density,
                                      double grainsize_ms,
                                      double scatter,
                                      double pitch_shift,
                                      double amp_variation)

    cdp_lib_buffer* cdp_lib_freeze(cdp_lib_ctx* ctx,
                                    const cdp_lib_buffer* input,
                                    double start_time,
                                    double end_time,
                                    double duration,
                                    double delay,
                                    double randomize,
                                    double pitch_scatter,
                                    double amp_cut,
                                    double gain)

    cdp_lib_buffer* cdp_lib_grain_cloud(cdp_lib_ctx* ctx,
                                         const cdp_lib_buffer* input,
                                         double gate,
                                         double grainsize_ms,
                                         double density,
                                         double duration,
                                         double scatter,
                                         unsigned int seed)

    cdp_lib_buffer* cdp_lib_grain_extend(cdp_lib_ctx* ctx,
                                          const cdp_lib_buffer* input,
                                          double grainsize_ms,
                                          double trough,
                                          double extension,
                                          double start_time,
                                          double end_time)

    cdp_lib_buffer* cdp_lib_texture_simple(cdp_lib_ctx* ctx,
                                            const cdp_lib_buffer* input,
                                            double duration,
                                            double density,
                                            double pitch_range,
                                            double amp_range,
                                            double spatial_range,
                                            unsigned int seed)

    cdp_lib_buffer* cdp_lib_texture_multi(cdp_lib_ctx* ctx,
                                           const cdp_lib_buffer* input,
                                           double duration,
                                           double density,
                                           int group_size,
                                           double group_spread,
                                           double pitch_range,
                                           double pitch_center,
                                           double amp_decay,
                                           unsigned int seed)


# Global CDP library context (lazily initialized)
cdef cdp_lib_ctx* _cdp_lib_ctx = NULL


cdef cdp_lib_ctx* _get_cdp_lib_ctx() except NULL:
    """Get or create the global CDP library context."""
    global _cdp_lib_ctx
    if _cdp_lib_ctx is NULL:
        _cdp_lib_ctx = cdp_lib_init()
        if _cdp_lib_ctx is NULL:
            raise MemoryError("Failed to initialize CDP library")
    return _cdp_lib_ctx


cdef cdp_lib_buffer* _buffer_to_cdp_lib(Buffer buf) except NULL:
    """Convert a pycdp Buffer to a cdp_lib_buffer."""
    cdef cdp_lib_buffer* lib_buf = cdp_lib_buffer_create(
        buf.sample_count, buf.channels, buf.sample_rate)
    if lib_buf is NULL:
        raise MemoryError("Failed to create CDP library buffer")

    # Copy data
    cdef size_t i
    for i in range(buf.sample_count):
        lib_buf.data[i] = buf._buf.samples[i]

    return lib_buf


cdef Buffer _cdp_lib_to_buffer(cdp_lib_buffer* lib_buf):
    """Convert a cdp_lib_buffer to a pycdp Buffer."""
    cdef Buffer result = Buffer.create(
        lib_buf.length // lib_buf.channels,
        lib_buf.channels,
        lib_buf.sample_rate)

    # Copy data
    cdef size_t i
    for i in range(lib_buf.length):
        result._buf.samples[i] = lib_buf.data[i]

    return result


def time_stretch(Buffer buf not None, double factor, int fft_size=1024, int overlap=3):
    """Time-stretch audio without changing pitch (native implementation).

    Uses phase vocoder for high-quality time stretching.
    This is a native implementation - no subprocess overhead.

    Args:
        buf: Input Buffer.
        factor: Stretch factor (2.0 = twice as long, 0.5 = half as long).
        fft_size: FFT window size (power of 2, 256-8192). Default 1024.
        overlap: Overlap factor (1-4). Default 3.

    Returns:
        New Buffer with time-stretched audio.

    Raises:
        CDPError: If processing fails.
    """
    if factor <= 0:
        raise ValueError("Stretch factor must be positive")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_time_stretch(
        ctx, input_buf, factor, fft_size, overlap)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Time stretch failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def spectral_blur(Buffer buf not None, double blur_time, int fft_size=1024):
    """Apply spectral blur (native implementation).

    Averages the spectrum over time, creating a smeared effect.
    This is a native implementation - no subprocess overhead.

    Args:
        buf: Input Buffer.
        blur_time: Time to blur over in seconds.
        fft_size: FFT window size. Default 1024.

    Returns:
        New Buffer with blurred audio.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_spectral_blur(
        ctx, input_buf, blur_time, fft_size)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Spectral blur failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def modify_speed(Buffer buf not None, double speed_factor):
    """Change playback speed (native implementation).

    Changes both pitch and duration.
    This is a native implementation - no subprocess overhead.

    Args:
        buf: Input Buffer.
        speed_factor: Speed multiplier (2.0 = double speed/octave up).

    Returns:
        New Buffer with modified speed.

    Raises:
        CDPError: If processing fails.
    """
    if speed_factor <= 0:
        raise ValueError("Speed factor must be positive")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_speed(ctx, input_buf, speed_factor)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Speed change failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


# =============================================================================
# Native spectral operations
# =============================================================================

def pitch_shift(Buffer buf not None, double semitones, int fft_size=1024):
    """Pitch shift without changing duration (native implementation).

    Args:
        buf: Input Buffer.
        semitones: Pitch shift in semitones (positive = up, negative = down).
        fft_size: FFT window size. Default 1024.

    Returns:
        New Buffer with pitch-shifted audio.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_pitch_shift(ctx, input_buf, semitones, fft_size)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Pitch shift failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def spectral_shift(Buffer buf not None, double shift_hz, int fft_size=1024):
    """Shift all frequencies by a fixed Hz offset (native implementation).

    Unlike pitch shift, this adds a constant Hz value, creating inharmonic effects.

    Args:
        buf: Input Buffer.
        shift_hz: Frequency offset in Hz (positive = up, negative = down).
        fft_size: FFT window size. Default 1024.

    Returns:
        New Buffer with shifted frequencies.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_spectral_shift(ctx, input_buf, shift_hz, fft_size)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Spectral shift failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def spectral_stretch(Buffer buf not None, double max_stretch,
                            double freq_divide=1000.0, double exponent=1.0, int fft_size=1024):
    """Stretch frequencies differentially (native implementation).

    Higher frequencies get stretched more, creating inharmonic effects.

    Args:
        buf: Input Buffer.
        max_stretch: Maximum transposition ratio for highest frequencies.
        freq_divide: Frequency below which no stretching occurs. Default 1000 Hz.
        exponent: Stretching curve (1.0 = linear). Default 1.0.
        fft_size: FFT window size. Default 1024.

    Returns:
        New Buffer with stretched spectrum.

    Raises:
        CDPError: If processing fails.
    """
    if max_stretch <= 0:
        raise ValueError("max_stretch must be positive")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_spectral_stretch(
        ctx, input_buf, max_stretch, freq_divide, exponent, fft_size)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Spectral stretch failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def filter_lowpass(Buffer buf not None, double cutoff_freq,
                          double attenuation_db=-60, int fft_size=1024):
    """Apply lowpass filter (native implementation).

    Args:
        buf: Input Buffer.
        cutoff_freq: Cutoff frequency in Hz.
        attenuation_db: Attenuation in dB (negative). Default -60.
        fft_size: FFT window size. Default 1024.

    Returns:
        New Buffer with filtered audio.

    Raises:
        CDPError: If processing fails.
    """
    if cutoff_freq <= 0:
        raise ValueError("cutoff_freq must be positive")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_filter_lowpass(
        ctx, input_buf, cutoff_freq, attenuation_db, fft_size)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Lowpass filter failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def filter_highpass(Buffer buf not None, double cutoff_freq,
                           double attenuation_db=-60, int fft_size=1024):
    """Apply highpass filter (native implementation).

    Args:
        buf: Input Buffer.
        cutoff_freq: Cutoff frequency in Hz.
        attenuation_db: Attenuation in dB (negative). Default -60.
        fft_size: FFT window size. Default 1024.

    Returns:
        New Buffer with filtered audio.

    Raises:
        CDPError: If processing fails.
    """
    if cutoff_freq <= 0:
        raise ValueError("cutoff_freq must be positive")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_filter_highpass(
        ctx, input_buf, cutoff_freq, attenuation_db, fft_size)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Highpass filter failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def filter_bandpass(Buffer buf not None, double low_freq, double high_freq,
                    double attenuation_db=-60, int fft_size=1024):
    """Apply bandpass filter (native implementation).

    Args:
        buf: Input Buffer.
        low_freq: Low cutoff frequency in Hz.
        high_freq: High cutoff frequency in Hz.
        attenuation_db: Attenuation in dB (negative). Default -60.
        fft_size: FFT window size. Default 1024.

    Returns:
        New Buffer with filtered audio.

    Raises:
        CDPError: If processing fails.
    """
    if low_freq <= 0:
        raise ValueError("low_freq must be positive")
    if high_freq <= low_freq:
        raise ValueError("high_freq must be greater than low_freq")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_filter_bandpass(
        ctx, input_buf, low_freq, high_freq, attenuation_db, fft_size)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Bandpass filter failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def filter_notch(Buffer buf not None, double center_freq, double width_hz,
                 double attenuation_db=-60, int fft_size=1024):
    """Apply notch (band-reject) filter (native implementation).

    Args:
        buf: Input Buffer.
        center_freq: Center frequency to notch out in Hz.
        width_hz: Width of the notch in Hz.
        attenuation_db: Attenuation in dB (negative). Default -60.
        fft_size: FFT window size. Default 1024.

    Returns:
        New Buffer with filtered audio.

    Raises:
        CDPError: If processing fails.
    """
    if center_freq <= 0:
        raise ValueError("center_freq must be positive")
    if width_hz <= 0:
        raise ValueError("width_hz must be positive")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_filter_notch(
        ctx, input_buf, center_freq, width_hz, attenuation_db, fft_size)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Notch filter failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def gate(Buffer buf not None, double threshold_db, double attack_ms=1.0,
         double release_ms=50.0, double hold_ms=10.0):
    """Apply noise gate (native implementation).

    Silences audio below threshold with attack/release envelope.

    Args:
        buf: Input Buffer.
        threshold_db: Threshold in dB (e.g., -40).
        attack_ms: Attack time in milliseconds. Default 1.0.
        release_ms: Release time in milliseconds. Default 50.0.
        hold_ms: Hold time before release in milliseconds. Default 10.0.

    Returns:
        New Buffer with gated audio.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_gate(
        ctx, input_buf, threshold_db, attack_ms, release_ms, hold_ms)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Gate failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def bitcrush(Buffer buf not None, int bit_depth=8, int downsample=1):
    """Apply bitcrusher effect (native implementation).

    Reduces bit depth and/or sample rate for lo-fi effects.

    Args:
        buf: Input Buffer.
        bit_depth: Target bit depth (1-16, 16 = no reduction). Default 8.
        downsample: Downsample factor (1 = none, 2 = half rate). Default 1.

    Returns:
        New Buffer with crushed audio.

    Raises:
        CDPError: If processing fails.
    """
    if bit_depth < 1 or bit_depth > 16:
        raise ValueError("bit_depth must be 1-16")
    if downsample < 1:
        raise ValueError("downsample must be >= 1")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_bitcrush(
        ctx, input_buf, bit_depth, downsample)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Bitcrush failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def ring_mod(Buffer buf not None, double freq, double mix=1.0):
    """Apply ring modulation (native implementation).

    Multiplies signal by a carrier frequency.

    Args:
        buf: Input Buffer.
        freq: Carrier frequency in Hz.
        mix: Dry/wet mix (0.0 = dry, 1.0 = wet). Default 1.0.

    Returns:
        New Buffer with ring-modulated audio.

    Raises:
        CDPError: If processing fails.
    """
    if freq <= 0:
        raise ValueError("freq must be positive")
    if mix < 0 or mix > 1:
        raise ValueError("mix must be 0.0-1.0")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_ring_mod(ctx, input_buf, freq, mix)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Ring modulation failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def delay(Buffer buf not None, double delay_ms, double feedback=0.3, double mix=0.5):
    """Apply delay effect with feedback (native implementation).

    Args:
        buf: Input Buffer.
        delay_ms: Delay time in milliseconds.
        feedback: Feedback amount (0.0 to <1.0). Default 0.3.
        mix: Dry/wet mix (0.0 = dry, 1.0 = wet). Default 0.5.

    Returns:
        New Buffer with delayed audio.

    Raises:
        CDPError: If processing fails.
    """
    if delay_ms <= 0:
        raise ValueError("delay_ms must be positive")
    if feedback < 0 or feedback >= 1:
        raise ValueError("feedback must be 0.0 to <1.0")
    if mix < 0 or mix > 1:
        raise ValueError("mix must be 0.0-1.0")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_delay(ctx, input_buf, delay_ms, feedback, mix)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Delay failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def chorus(Buffer buf not None, double rate=1.5, double depth_ms=5.0, double mix=0.5):
    """Apply chorus effect (native implementation).

    Modulated delay for thickening sounds.

    Args:
        buf: Input Buffer.
        rate: LFO rate in Hz (typically 0.5-5). Default 1.5.
        depth_ms: Modulation depth in milliseconds (typically 1-20). Default 5.0.
        mix: Dry/wet mix (0.0 = dry, 1.0 = wet). Default 0.5.

    Returns:
        New Buffer with chorus effect.

    Raises:
        CDPError: If processing fails.
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    if depth_ms <= 0:
        raise ValueError("depth_ms must be positive")
    if mix < 0 or mix > 1:
        raise ValueError("mix must be 0.0-1.0")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_chorus(ctx, input_buf, rate, depth_ms, mix)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Chorus failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def flanger(Buffer buf not None, double rate=0.5, double depth_ms=3.0,
            double feedback=0.5, double mix=0.5):
    """Apply flanger effect (native implementation).

    Short modulated delay with feedback for sweeping comb filter.

    Args:
        buf: Input Buffer.
        rate: LFO rate in Hz (typically 0.1-2). Default 0.5.
        depth_ms: Modulation depth in milliseconds (typically 1-10). Default 3.0.
        feedback: Feedback amount (-0.95 to 0.95). Default 0.5.
        mix: Dry/wet mix (0.0 = dry, 1.0 = wet). Default 0.5.

    Returns:
        New Buffer with flanger effect.

    Raises:
        CDPError: If processing fails.
    """
    if rate <= 0:
        raise ValueError("rate must be positive")
    if depth_ms <= 0:
        raise ValueError("depth_ms must be positive")
    if feedback < -0.95 or feedback > 0.95:
        raise ValueError("feedback must be -0.95 to 0.95")
    if mix < 0 or mix > 1:
        raise ValueError("mix must be 0.0-1.0")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_flanger(
        ctx, input_buf, rate, depth_ms, feedback, mix)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Flanger failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


# =============================================================================
# EQ and Dynamics
# =============================================================================

def eq_parametric(Buffer buf not None, double center_freq, double gain_db,
                  double q=1.0, int fft_size=1024):
    """Apply parametric EQ (native implementation).

    Boost or cut at a specific frequency with adjustable bandwidth (Q).

    Args:
        buf: Input Buffer.
        center_freq: Center frequency in Hz.
        gain_db: Gain in dB (positive = boost, negative = cut).
        q: Q factor (0.1-10, higher = narrower bandwidth). Default 1.0.
        fft_size: FFT window size. Default 1024.

    Returns:
        New Buffer with EQ applied.

    Raises:
        CDPError: If processing fails.
    """
    if center_freq <= 0:
        raise ValueError("center_freq must be positive")
    if q <= 0:
        raise ValueError("q must be positive")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_eq_parametric(
        ctx, input_buf, center_freq, gain_db, q, fft_size)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Parametric EQ failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def envelope_follow(Buffer buf not None, double attack_ms=10.0,
                    double release_ms=100.0, str mode="peak"):
    """Extract amplitude envelope from audio (native implementation).

    Uses peak or RMS detection with attack/release smoothing.

    Args:
        buf: Input Buffer.
        attack_ms: Attack time in milliseconds. Default 10.0.
        release_ms: Release time in milliseconds. Default 100.0.
        mode: "peak" or "rms". Default "peak".

    Returns:
        New Buffer containing envelope values (mono).

    Raises:
        CDPError: If processing fails.
    """
    if attack_ms < 0:
        raise ValueError("attack_ms must be non-negative")
    if release_ms < 0:
        raise ValueError("release_ms must be non-negative")

    cdef int mode_int = 0 if mode == "peak" else 1

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_envelope_follow(
        ctx, input_buf, attack_ms, release_ms, mode_int)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Envelope follow failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def envelope_apply(Buffer buf not None, Buffer envelope not None, double depth=1.0):
    """Apply an envelope to audio (native implementation).

    Multiplies audio by envelope values (amplitude modulation).

    Args:
        buf: Input Buffer to process.
        envelope: Envelope Buffer (from envelope_follow or generated).
        depth: Modulation depth (0.0 = no effect, 1.0 = full). Default 1.0.

    Returns:
        New Buffer with envelope applied.

    Raises:
        CDPError: If processing fails.
    """
    if depth < 0 or depth > 1:
        raise ValueError("depth must be 0.0 to 1.0")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)
    cdef cdp_lib_buffer* env_buf = _buffer_to_cdp_lib(envelope)

    cdef cdp_lib_buffer* output_buf = cdp_lib_envelope_apply(
        ctx, input_buf, env_buf, depth)

    cdp_lib_buffer_free(input_buf)
    cdp_lib_buffer_free(env_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Envelope apply failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def compressor(Buffer buf not None, double threshold_db=-20.0, double ratio=4.0,
               double attack_ms=10.0, double release_ms=100.0, double makeup_gain_db=0.0):
    """Apply dynamic range compression (native implementation).

    Reduces the volume of audio above the threshold.

    Args:
        buf: Input Buffer.
        threshold_db: Level above which compression starts. Default -20.0.
        ratio: Compression ratio (e.g., 4.0 = 4:1). Default 4.0.
        attack_ms: Attack time in milliseconds. Default 10.0.
        release_ms: Release time in milliseconds. Default 100.0.
        makeup_gain_db: Makeup gain in dB. Default 0.0.

    Returns:
        New Buffer with compression applied.

    Raises:
        CDPError: If processing fails.
    """
    if ratio < 1.0:
        raise ValueError("ratio must be >= 1.0")
    if attack_ms < 0:
        raise ValueError("attack_ms must be non-negative")
    if release_ms < 0:
        raise ValueError("release_ms must be non-negative")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_compressor(
        ctx, input_buf, threshold_db, ratio, attack_ms, release_ms, makeup_gain_db)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Compressor failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def limiter(Buffer buf not None, double threshold_db=-0.1,
            double attack_ms=0.0, double release_ms=50.0):
    """Apply limiting to prevent clipping (native implementation).

    Prevents audio from exceeding the threshold.

    Args:
        buf: Input Buffer.
        threshold_db: Maximum level in dB. Default -0.1.
        attack_ms: Attack time (0 = hard limiting). Default 0.0.
        release_ms: Release time in milliseconds. Default 50.0.

    Returns:
        New Buffer with limiting applied.

    Raises:
        CDPError: If processing fails.
    """
    if attack_ms < 0:
        raise ValueError("attack_ms must be non-negative")
    if release_ms < 0:
        raise ValueError("release_ms must be non-negative")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_limiter(
        ctx, input_buf, threshold_db, attack_ms, release_ms)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Limiter failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


# =============================================================================
# Native envelope operations
# =============================================================================

def dovetail(Buffer buf not None, double fade_in_dur, double fade_out_dur,
                    str fade_type="exponential"):
    """Apply fade-in and fade-out envelopes (native implementation).

    Args:
        buf: Input Buffer.
        fade_in_dur: Fade-in duration in seconds.
        fade_out_dur: Fade-out duration in seconds.
        fade_type: "linear" or "exponential" (default).

    Returns:
        New Buffer with fades applied.

    Raises:
        CDPError: If processing fails.
        ValueError: If fade_type is invalid.
    """
    cdef int fade_in_type, fade_out_type

    if fade_type == "linear":
        fade_in_type = CDP_FADE_LINEAR
        fade_out_type = CDP_FADE_LINEAR
    elif fade_type == "exponential":
        fade_in_type = CDP_FADE_EXPONENTIAL
        fade_out_type = CDP_FADE_EXPONENTIAL
    else:
        raise ValueError(f"Invalid fade_type: {fade_type}. Use 'linear' or 'exponential'.")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_dovetail(
        ctx, input_buf, fade_in_dur, fade_out_dur, fade_in_type, fade_out_type)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Dovetail failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def tremolo(Buffer buf not None, double freq, double depth, double gain=1.0):
    """Apply tremolo / amplitude modulation (native implementation).

    Args:
        buf: Input Buffer.
        freq: Tremolo frequency in Hz (0 to 500).
        depth: Modulation depth (0.0 = none, 1.0 = full).
        gain: Output gain multiplier. Default 1.0.

    Returns:
        New Buffer with tremolo applied.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_tremolo(ctx, input_buf, freq, depth, gain)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Tremolo failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def attack(Buffer buf not None, double attack_gain, double attack_time):
    """Modify the attack portion of a sound (native implementation).

    Args:
        buf: Input Buffer.
        attack_gain: Gain multiplier for attack (e.g., 2.0 = double).
        attack_time: Duration of attack region in seconds.

    Returns:
        New Buffer with modified attack.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_attack(ctx, input_buf, attack_gain, attack_time)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Attack modification failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


# =============================================================================
# Native distortion operations
# =============================================================================

def distort_overload(Buffer buf not None, double clip_level, double depth=0.5):
    """Apply clipping distortion (native implementation).

    Args:
        buf: Input Buffer (will be converted to mono if stereo).
        clip_level: Clipping threshold (0.0 to 1.0).
        depth: Distortion depth (0 = hard clip, 1 = soft clip). Default 0.5.

    Returns:
        New mono Buffer with distortion applied.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_distort_overload(ctx, input_buf, clip_level, depth)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Distort overload failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def distort_reverse(Buffer buf not None, int cycle_count):
    """Reverse groups of wavecycles (native implementation).

    Args:
        buf: Input Buffer (will be converted to mono if stereo).
        cycle_count: Number of cycles in each reversed group.

    Returns:
        New mono Buffer with reversed cycles.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_distort_reverse(ctx, input_buf, cycle_count)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Distort reverse failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def distort_fractal(Buffer buf not None, double scaling, double loudness=1.0):
    """Apply fractal distortion (native implementation).

    Adds harmonic complexity by recursively overlaying wavecycle patterns.

    Args:
        buf: Input Buffer (will be converted to mono if stereo).
        scaling: Fractal scaling factor (affects harmonic content).
        loudness: Output loudness (0.0 to 1.0). Default 1.0.

    Returns:
        New mono Buffer with fractal distortion.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_distort_fractal(ctx, input_buf, scaling, loudness)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Distort fractal failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def distort_shuffle(Buffer buf not None, int chunk_count, unsigned int seed=0):
    """Shuffle segments of audio (native implementation).

    Args:
        buf: Input Buffer (will be converted to mono if stereo).
        chunk_count: Number of chunks to divide the audio into.
        seed: Random seed (0 = time-based). Default 0.

    Returns:
        New mono Buffer with shuffled segments.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_distort_shuffle(ctx, input_buf, chunk_count, seed)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Distort shuffle failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


# =============================================================================
# Native reverb
# =============================================================================

def reverb(Buffer buf not None, double mix=0.5, double decay_time=2.0,
                  double damping=0.5, double lpfreq=8000.0, double predelay=0.0):
    """Apply reverb (native implementation).

    Uses a Feedback Delay Network (8 combs + 4 allpass) for dense reverb.

    Args:
        buf: Input Buffer.
        mix: Dry/wet balance (0.0 = dry, 1.0 = wet). Default 0.5.
        decay_time: Reverb decay time in seconds (RT60). Default 2.0.
        damping: High frequency damping (0.0 to 1.0). Default 0.5.
        lpfreq: Lowpass filter cutoff in Hz. Default 8000.
        predelay: Pre-delay time in milliseconds. Default 0.

    Returns:
        New stereo Buffer with reverb applied (includes tail).

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_reverb(
        ctx, input_buf, mix, decay_time, damping, lpfreq, predelay)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Reverb failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


# =============================================================================
# Native granular operations
# =============================================================================

def brassage(Buffer buf not None, double velocity=1.0, double density=1.0,
                    double grainsize_ms=50.0, double scatter=0.0,
                    double pitch_shift=0.0, double amp_variation=0.0):
    """Apply granular brassage (native implementation).

    Breaks sound into grains and reassembles with optional modifications.

    Args:
        buf: Input Buffer.
        velocity: Playback speed through source (1.0 = normal). Default 1.0.
        density: Grain density multiplier. Default 1.0.
        grainsize_ms: Grain size in milliseconds. Default 50.
        scatter: Time scatter of grains (0.0 to 1.0). Default 0.
        pitch_shift: Pitch shift per grain in semitones. Default 0.
        amp_variation: Random amplitude variation (0.0 to 1.0). Default 0.

    Returns:
        New Buffer with brassage applied.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_brassage(
        ctx, input_buf, velocity, density, grainsize_ms,
        scatter, pitch_shift, amp_variation)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Brassage failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def freeze(Buffer buf not None, double start_time, double end_time,
                  double duration, double delay=0.05, double randomize=0.2,
                  double pitch_scatter=0.0, double amp_cut=0.1, double gain=1.0):
    """Freeze a segment of audio by repeated iteration (native implementation).

    Creates a sustained texture by repeating a frozen segment with variations.

    Args:
        buf: Input Buffer.
        start_time: Start of segment to freeze (seconds).
        end_time: End of segment to freeze (seconds).
        duration: Output duration in seconds.
        delay: Average delay between iterations. Default 0.05.
        randomize: Delay time randomization (0.0 to 1.0). Default 0.2.
        pitch_scatter: Max random pitch shift in semitones (0 to 12). Default 0.
        amp_cut: Max random amplitude reduction (0.0 to 1.0). Default 0.1.
        gain: Gain adjustment for frozen segment. Default 1.0.

    Returns:
        New Buffer with frozen audio.

    Raises:
        CDPError: If processing fails.
    """
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")
    if duration <= 0:
        raise ValueError("duration must be positive")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_freeze(
        ctx, input_buf, start_time, end_time, duration,
        delay, randomize, pitch_scatter, amp_cut, gain)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Freeze failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def grain_cloud(Buffer buf not None, double gate=0.1, double grainsize_ms=50.0,
                double density=10.0, double duration=0.0, double scatter=0.3,
                unsigned int seed=0):
    """Generate grain cloud from source audio (CDP: grain).

    Extracts grains based on amplitude threshold and generates a cloud
    by placing them at random or regular intervals.

    Args:
        buf: Input Buffer.
        gate: Amplitude threshold for grain detection (0.0 to 1.0). Default 0.1.
        grainsize_ms: Target grain size in milliseconds. Default 50.0.
        density: Grain density (grains per second). Default 10.0.
        duration: Output duration in seconds (0 = same as input). Default 0.
        scatter: Position scatter amount (0.0 to 1.0). Default 0.3.
        seed: Random seed (0 = use time). Default 0.

    Returns:
        New Buffer with grain cloud.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_grain_cloud(
        ctx, input_buf, gate, grainsize_ms, density, duration, scatter, seed)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Grain cloud failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def grain_extend(Buffer buf not None, double grainsize_ms=15.0, double trough=0.3,
                 double extension=1.0, double start_time=0.0, double end_time=0.0):
    """Extend audio duration using grain repetition (CDP: grainex extend).

    Finds grains in source and extends duration by repeating grains
    with variations.

    Args:
        buf: Input Buffer.
        grainsize_ms: Window size to detect grains (milliseconds). Default 15.0.
        trough: Acceptable trough height relative to peaks (0.0 to 1.0). Default 0.3.
        extension: How much duration to add (seconds). Default 1.0.
        start_time: Start of grain material in source (seconds). Default 0.0.
        end_time: End of grain material in source (seconds, 0 = end). Default 0.0.

    Returns:
        New Buffer with extended audio.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_grain_extend(
        ctx, input_buf, grainsize_ms, trough, extension, start_time, end_time)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Grain extend failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def texture_simple(Buffer buf not None, double duration=5.0, double density=5.0,
                   double pitch_range=6.0, double amp_range=0.3,
                   double spatial_range=0.8, unsigned int seed=0):
    """Generate simple texture (CDP: texture SIMPLE_TEX).

    Creates texture by layering source at multiple transpositions
    and time offsets.

    Args:
        buf: Input Buffer.
        duration: Output duration in seconds. Default 5.0.
        density: Events per second. Default 5.0.
        pitch_range: Pitch range in semitones (symmetric around 0). Default 6.0.
        amp_range: Amplitude variation (0.0 to 1.0). Default 0.3.
        spatial_range: Stereo spread (0.0 to 1.0, 0 = mono center). Default 0.8.
        seed: Random seed (0 = use time). Default 0.

    Returns:
        New stereo Buffer with texture.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_texture_simple(
        ctx, input_buf, duration, density, pitch_range, amp_range, spatial_range, seed)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Texture simple failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def texture_multi(Buffer buf not None, double duration=5.0, double density=2.0,
                  int group_size=4, double group_spread=0.2,
                  double pitch_range=8.0, double pitch_center=0.0,
                  double amp_decay=0.3, unsigned int seed=0):
    """Generate multi-layer texture (CDP: texture GROUPS).

    Creates complex texture with grouped events and decorations.

    Args:
        buf: Input Buffer.
        duration: Output duration in seconds. Default 5.0.
        density: Groups per second. Default 2.0.
        group_size: Average notes per group (1-16). Default 4.
        group_spread: Time spread within group (seconds). Default 0.2.
        pitch_range: Pitch range in semitones. Default 8.0.
        pitch_center: Center pitch offset in semitones. Default 0.0.
        amp_decay: Amplitude decay through group (0.0 to 1.0). Default 0.3.
        seed: Random seed (0 = use time). Default 0.

    Returns:
        New stereo Buffer with multi-layer texture.

    Raises:
        CDPError: If processing fails.
    """
    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input_buf = _buffer_to_cdp_lib(buf)

    cdef cdp_lib_buffer* output_buf = cdp_lib_texture_multi(
        ctx, input_buf, duration, density, group_size, group_spread,
        pitch_range, pitch_center, amp_decay, seed)

    cdp_lib_buffer_free(input_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Texture multi failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def morph(Buffer buf1 not None, Buffer buf2 not None,
          double morph_start=0.0, double morph_end=1.0,
          double exponent=1.0, int fft_size=1024):
    """Spectral morph between two sounds (CDP: SPECMORPH).

    Interpolates amplitude and frequency between two sounds over time.
    Amplitude interpolation is linear; frequency interpolation is exponential.

    Args:
        buf1: First input Buffer (source).
        buf2: Second input Buffer (target).
        morph_start: Time when morphing begins (0.0 to 1.0 of duration). Default 0.0.
        morph_end: Time when morphing ends (0.0 to 1.0 of duration). Default 1.0.
        exponent: Interpolation curve (1.0 = linear, <1 = fast start, >1 = slow start). Default 1.0.
        fft_size: FFT window size. Default 1024.

    Returns:
        New Buffer with morphed audio.

    Raises:
        CDPError: If processing fails.
    """
    if buf1.sample_rate != buf2.sample_rate:
        raise ValueError("Sample rates must match for morphing")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input1_buf = _buffer_to_cdp_lib(buf1)
    cdef cdp_lib_buffer* input2_buf = _buffer_to_cdp_lib(buf2)

    cdef cdp_lib_buffer* output_buf = cdp_lib_morph(
        ctx, input1_buf, input2_buf, morph_start, morph_end, exponent, fft_size)

    cdp_lib_buffer_free(input1_buf)
    cdp_lib_buffer_free(input2_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Morph failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def morph_glide(Buffer buf1 not None, Buffer buf2 not None,
                double duration=1.0, int fft_size=1024):
    """Simple spectral glide between two sounds (CDP: SPECGLIDE).

    Creates a linear glide from one spectrum to another over the specified duration.

    Args:
        buf1: First input Buffer.
        buf2: Second input Buffer.
        duration: Output duration in seconds. Default 1.0.
        fft_size: FFT window size. Default 1024.

    Returns:
        New Buffer with glided audio.

    Raises:
        CDPError: If processing fails.
    """
    if buf1.sample_rate != buf2.sample_rate:
        raise ValueError("Sample rates must match for glide")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input1_buf = _buffer_to_cdp_lib(buf1)
    cdef cdp_lib_buffer* input2_buf = _buffer_to_cdp_lib(buf2)

    cdef cdp_lib_buffer* output_buf = cdp_lib_morph_glide(
        ctx, input1_buf, input2_buf, duration, fft_size)

    cdp_lib_buffer_free(input1_buf)
    cdp_lib_buffer_free(input2_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Morph glide failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result


def cross_synth(Buffer buf1 not None, Buffer buf2 not None,
                int mode=0, double mix=1.0, int fft_size=1024):
    """Cross-synthesis: combine amplitude from one sound with frequencies from another.

    Args:
        buf1: First input Buffer (amplitude source by default).
        buf2: Second input Buffer (frequency source by default).
        mode: 0 = amp from buf1, freq from buf2 (default)
              1 = amp from buf2, freq from buf1
        mix: Mix between original and cross-synthesized (0.0 = original, 1.0 = full cross). Default 1.0.
        fft_size: FFT window size. Default 1024.

    Returns:
        New Buffer with cross-synthesized audio.

    Raises:
        CDPError: If processing fails.
    """
    if buf1.sample_rate != buf2.sample_rate:
        raise ValueError("Sample rates must match for cross-synthesis")

    cdef cdp_lib_ctx* ctx = _get_cdp_lib_ctx()
    cdef cdp_lib_buffer* input1_buf = _buffer_to_cdp_lib(buf1)
    cdef cdp_lib_buffer* input2_buf = _buffer_to_cdp_lib(buf2)

    cdef cdp_lib_buffer* output_buf = cdp_lib_cross_synth(
        ctx, input1_buf, input2_buf, mode, mix, fft_size)

    cdp_lib_buffer_free(input1_buf)
    cdp_lib_buffer_free(input2_buf)

    if output_buf is NULL:
        error_msg = cdp_lib_get_error(ctx)
        raise CDPError(-1, error_msg.decode('utf-8') if error_msg else "Cross-synthesis failed")

    cdef Buffer result = _cdp_lib_to_buffer(output_buf)
    cdp_lib_buffer_free(output_buf)

    return result
