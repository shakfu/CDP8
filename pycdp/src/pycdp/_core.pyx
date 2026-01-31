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
