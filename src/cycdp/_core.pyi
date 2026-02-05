"""Type stubs for cycdp._core Cython extension module."""

from typing import Sequence, overload

# Type alias for buffer-like objects (numpy arrays, array.array('f'), memoryview, etc.)
# In practice, any object supporting the buffer protocol with float32 data
from typing import Protocol, runtime_checkable

@runtime_checkable
class BufferLike(Protocol):
    """Protocol for objects supporting the buffer protocol with float32 data."""
    def __buffer__(self, flags: int) -> memoryview: ...

# =============================================================================
# Constants
# =============================================================================

FLAG_NONE: int
FLAG_CLIP: int

# Waveform types for synth_wave
WAVE_SINE: int
WAVE_SQUARE: int
WAVE_SAW: int
WAVE_RAMP: int
WAVE_TRIANGLE: int

# Scramble modes
SCRAMBLE_SHUFFLE: int
SCRAMBLE_REVERSE: int
SCRAMBLE_SIZE_UP: int
SCRAMBLE_SIZE_DOWN: int
SCRAMBLE_LEVEL_UP: int
SCRAMBLE_LEVEL_DOWN: int

# =============================================================================
# Exception
# =============================================================================

class CDPError(Exception):
    """Exception raised for CDP library errors."""

    code: int
    def __init__(self, code: int, message: str) -> None: ...

# =============================================================================
# Utility functions
# =============================================================================

def version() -> str:
    """Get CDP library version string."""
    ...

def gain_to_db(gain: float) -> float:
    """Convert linear gain to decibels."""
    ...

def db_to_gain(db: float) -> float:
    """Convert decibels to linear gain."""
    ...

# =============================================================================
# Classes
# =============================================================================

class Context:
    """CDP processing context."""
    def __init__(self) -> None: ...
    def get_error(self) -> int:
        """Get last error code."""
        ...
    def get_error_message(self) -> str:
        """Get last error message."""
        ...
    def clear_error(self) -> None:
        """Clear error state."""
        ...

class Buffer:
    """CDP audio buffer with Python buffer protocol support."""

    @staticmethod
    def create(frame_count: int, channels: int = 1, sample_rate: int = 44100) -> Buffer:
        """Create a new buffer with allocated memory."""
        ...

    @staticmethod
    def from_memoryview(
        samples: BufferLike, channels: int = 1, sample_rate: int = 44100
    ) -> Buffer:
        """Create a buffer by copying from a memoryview."""
        ...

    def to_list(self) -> list[float]:
        """Convert buffer to Python list."""
        ...

    def to_bytes(self) -> bytes:
        """Convert buffer to bytes (raw float32 data)."""
        ...

    @property
    def sample_count(self) -> int: ...
    @property
    def frame_count(self) -> int: ...
    @property
    def channels(self) -> int: ...
    @property
    def sample_rate(self) -> int: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> float: ...
    def __setitem__(self, index: int, value: float) -> None: ...
    def clear(self) -> None:
        """Set all samples to zero."""
        ...

# =============================================================================
# Low-level functions (work with Buffer objects)
# =============================================================================

def apply_gain(ctx: Context, buf: Buffer, gain: float, clip: bool = False) -> None:
    """Apply gain to buffer (in-place)."""
    ...

def apply_gain_db(
    ctx: Context, buf: Buffer, gain_db: float, clip: bool = False
) -> None:
    """Apply gain in dB to buffer (in-place)."""
    ...

def apply_normalize(ctx: Context, buf: Buffer, target_level: float = 1.0) -> None:
    """Normalize buffer to target level (in-place)."""
    ...

def apply_normalize_db(ctx: Context, buf: Buffer, target_db: float = 0.0) -> None:
    """Normalize buffer to target dB level (in-place)."""
    ...

def apply_phase_invert(ctx: Context, buf: Buffer) -> None:
    """Invert phase of buffer (in-place)."""
    ...

def get_peak(ctx: Context, buf: Buffer) -> tuple[float, int]:
    """Find peak level in buffer. Returns (level, frame_position)."""
    ...

# =============================================================================
# High-level functions (accept any float32 buffer via memoryview)
# =============================================================================

def gain(
    samples: BufferLike,
    gain_factor: float = 1.0,
    sample_rate: int = 44100,
    clip: bool = False,
) -> Buffer:
    """Apply gain to audio samples."""
    ...

def gain_db(
    samples: BufferLike, db: float = 0.0, sample_rate: int = 44100, clip: bool = False
) -> Buffer:
    """Apply gain in decibels to audio samples."""
    ...

def normalize(
    samples: BufferLike, target: float = 1.0, sample_rate: int = 44100
) -> Buffer:
    """Normalize audio to target peak level."""
    ...

def normalize_db(
    samples: BufferLike, target_db: float = 0.0, sample_rate: int = 44100
) -> Buffer:
    """Normalize audio to target peak level in dB."""
    ...

@overload
def phase_invert(samples: BufferLike, sample_rate: int = 44100) -> Buffer:
    """Invert phase of audio samples (high-level)."""
    ...

@overload
def phase_invert(buf: Buffer) -> Buffer:
    """Invert phase of buffer (CDP lib version)."""
    ...

def peak(samples: BufferLike, sample_rate: int = 44100) -> tuple[float, int]:
    """Find peak level in audio samples. Returns (peak_level, sample_position)."""
    ...

# =============================================================================
# File I/O
# =============================================================================

def read_file(path: str) -> Buffer:
    """Read an audio file into a Buffer."""
    ...

def write_file(path: str, buf: Buffer, format: str = "float") -> None:
    """Write a Buffer to an audio file."""
    ...

# =============================================================================
# Spatial/panning
# =============================================================================

def pan(buf: Buffer, position: float = 0.0) -> Buffer:
    """Pan a mono buffer to stereo with a static pan position."""
    ...

def pan_envelope(buf: Buffer, points: list[tuple[float, float]]) -> Buffer:
    """Pan a mono buffer to stereo with time-varying position."""
    ...

def mirror(buf: Buffer) -> Buffer:
    """Mirror (swap) left and right channels of a stereo buffer."""
    ...

def narrow(buf: Buffer, width: float = 1.0) -> Buffer:
    """Narrow or widen stereo image."""
    ...

# =============================================================================
# Mixing
# =============================================================================

def mix2(a: Buffer, b: Buffer, gain_a: float = 1.0, gain_b: float = 1.0) -> Buffer:
    """Mix two buffers together with optional gains."""
    ...

def mix(buffers: list[Buffer], gains: list[float] | None = None) -> Buffer:
    """Mix multiple buffers together with optional gains."""
    ...

# =============================================================================
# Buffer utilities
# =============================================================================

def reverse(buf: Buffer) -> Buffer:
    """Reverse audio buffer."""
    ...

def fade_in(buf: Buffer, duration: float, curve: str = "linear") -> Buffer:
    """Apply fade in to buffer (in-place)."""
    ...

def fade_out(buf: Buffer, duration: float, curve: str = "linear") -> Buffer:
    """Apply fade out to buffer (in-place)."""
    ...

def concat(buffers: list[Buffer]) -> Buffer:
    """Concatenate multiple buffers into one."""
    ...

# =============================================================================
# Channel operations
# =============================================================================

def to_mono(buf: Buffer) -> Buffer:
    """Convert multi-channel buffer to mono by averaging all channels."""
    ...

def to_stereo(buf: Buffer) -> Buffer:
    """Convert mono buffer to stereo by duplicating the channel."""
    ...

def extract_channel(buf: Buffer, channel: int) -> Buffer:
    """Extract a single channel from a multi-channel buffer."""
    ...

def merge_channels(left: Buffer, right: Buffer) -> Buffer:
    """Merge two mono buffers into a stereo buffer."""
    ...

def split_channels(buf: Buffer) -> list[Buffer]:
    """Split a multi-channel buffer into separate mono buffers."""
    ...

def interleave(buffers: list[Buffer]) -> Buffer:
    """Interleave multiple mono buffers into a multi-channel buffer."""
    ...

# =============================================================================
# Spectral processing
# =============================================================================

def time_stretch(
    buf: Buffer, factor: float, fft_size: int = 1024, overlap: int = 3
) -> Buffer:
    """Time-stretch audio without changing pitch."""
    ...

def spectral_blur(buf: Buffer, blur_time: float, fft_size: int = 1024) -> Buffer:
    """Blur/smear the spectrum over time."""
    ...

def modify_speed(buf: Buffer, speed_factor: float) -> Buffer:
    """Change playback speed (affects both time and pitch)."""
    ...

def pitch_shift(buf: Buffer, semitones: float, fft_size: int = 1024) -> Buffer:
    """Shift pitch without changing duration."""
    ...

def spectral_shift(buf: Buffer, shift_hz: float, fft_size: int = 1024) -> Buffer:
    """Shift spectrum up or down by a fixed frequency."""
    ...

def spectral_stretch(
    buf: Buffer, max_stretch: float, fft_size: int = 1024, mode: int = 0
) -> Buffer:
    """Stretch or compress the spectrum."""
    ...

def filter_lowpass(
    buf: Buffer, cutoff_freq: float, fft_size: int = 1024, rolloff: float = 1.0
) -> Buffer:
    """Apply low-pass filter."""
    ...

def filter_highpass(
    buf: Buffer, cutoff_freq: float, fft_size: int = 1024, rolloff: float = 1.0
) -> Buffer:
    """Apply high-pass filter."""
    ...

def filter_bandpass(
    buf: Buffer,
    low_freq: float,
    high_freq: float,
    fft_size: int = 1024,
    rolloff: float = 1.0,
) -> Buffer:
    """Apply band-pass filter."""
    ...

def filter_notch(
    buf: Buffer,
    center_freq: float,
    width_hz: float,
    fft_size: int = 1024,
    depth: float = 1.0,
) -> Buffer:
    """Apply notch (band-reject) filter."""
    ...

# =============================================================================
# Dynamics and effects
# =============================================================================

def gate(
    buf: Buffer,
    threshold_db: float,
    attack_ms: float = 1.0,
    hold_ms: float = 50.0,
    release_ms: float = 50.0,
) -> Buffer:
    """Apply noise gate."""
    ...

def bitcrush(buf: Buffer, bit_depth: int = 8, downsample: int = 1) -> Buffer:
    """Apply bit depth reduction and downsampling."""
    ...

def ring_mod(buf: Buffer, freq: float, mix: float = 1.0) -> Buffer:
    """Apply ring modulation."""
    ...

def delay(
    buf: Buffer, delay_ms: float, feedback: float = 0.3, mix: float = 0.5
) -> Buffer:
    """Apply delay effect."""
    ...

def chorus(
    buf: Buffer, rate: float = 1.5, depth_ms: float = 5.0, mix: float = 0.5
) -> Buffer:
    """Apply chorus effect."""
    ...

def flanger(
    buf: Buffer,
    rate: float = 0.5,
    depth_ms: float = 3.0,
    feedback: float = 0.7,
    mix: float = 0.5,
) -> Buffer:
    """Apply flanger effect."""
    ...

# =============================================================================
# EQ and dynamics
# =============================================================================

def eq_parametric(
    buf: Buffer, center_freq: float, gain_db: float, q: float = 1.0
) -> Buffer:
    """Apply parametric EQ band."""
    ...

def envelope_follow(
    buf: Buffer, attack_ms: float = 10.0, release_ms: float = 100.0, mode: str = "peak"
) -> Buffer:
    """Extract amplitude envelope from audio."""
    ...

def envelope_apply(buf: Buffer, envelope: Buffer, depth: float = 1.0) -> Buffer:
    """Apply envelope to audio."""
    ...

def compressor(
    buf: Buffer,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    makeup_db: float = 0.0,
) -> Buffer:
    """Apply dynamic range compression."""
    ...

def limiter(
    buf: Buffer,
    threshold_db: float = -0.1,
    attack_ms: float = 0.1,
    release_ms: float = 50.0,
) -> Buffer:
    """Apply peak limiting."""
    ...

# =============================================================================
# Envelope operations
# =============================================================================

def dovetail(
    buf: Buffer, fade_in_dur: float, fade_out_dur: float, fade_type: int = 0
) -> Buffer:
    """Apply dovetail fades."""
    ...

def tremolo(buf: Buffer, freq: float, depth: float, gain: float = 1.0) -> Buffer:
    """Apply tremolo effect."""
    ...

def attack(buf: Buffer, attack_gain: float, attack_time: float) -> Buffer:
    """Modify attack transient."""
    ...

# =============================================================================
# Distortion operations
# =============================================================================

def distort_overload(buf: Buffer, clip_level: float, depth: float = 0.5) -> Buffer:
    """Apply overload/saturation distortion."""
    ...

def distort_reverse(buf: Buffer, cycle_count: int) -> Buffer:
    """Apply reverse distortion effect."""
    ...

def distort_fractal(buf: Buffer, scaling: float, loudness: float = 1.0) -> Buffer:
    """Apply fractal distortion."""
    ...

def distort_shuffle(buf: Buffer, chunk_count: int, seed: int = 0) -> Buffer:
    """Apply shuffle distortion."""
    ...

def distort_cut(
    buf: Buffer, cycle_count: int = 4, cycle_step: int = 4, decay: float = 0.9
) -> Buffer:
    """Waveset cut with decaying envelope."""
    ...

def distort_mark(
    buf: Buffer, markers: Sequence[float], unit_ms: float = 10.0, interp_ms: float = 5.0
) -> Buffer:
    """Interpolate wavesets at time markers."""
    ...

def distort_repeat(
    buf: Buffer, multiplier: int = 2, cycle_count: int = 1, skip: int = 0
) -> Buffer:
    """Time-stretch by repeating wavecycles."""
    ...

def distort_shift(
    buf: Buffer, group_size: int = 1, shift: int = 1, mode: int = 0
) -> Buffer:
    """Shift/swap half-wavecycle groups."""
    ...

def distort_warp(
    buf: Buffer, warp: float = 0.001, mode: int = 0, waveset_count: int = 1
) -> Buffer:
    """Progressive warp distortion with sample folding."""
    ...

# =============================================================================
# Reverb
# =============================================================================

def reverb(
    buf: Buffer,
    mix: float = 0.5,
    decay_time: float = 2.0,
    damping: float = 0.5,
    room_size: float = 0.5,
) -> Buffer:
    """Apply reverb effect."""
    ...

# =============================================================================
# Granular operations
# =============================================================================

def brassage(
    buf: Buffer,
    velocity: float = 1.0,
    density: float = 1.0,
    grainsize_ms: float = 50.0,
    pitch_shift: float = 0.0,
    scatter: float = 0.0,
) -> Buffer:
    """Apply granular brassage."""
    ...

def freeze(
    buf: Buffer,
    start_time: float,
    end_time: float,
    grainsize_ms: float = 50.0,
    density: float = 1.0,
    duration: float = 5.0,
) -> Buffer:
    """Granular freeze at position."""
    ...

def grain_cloud(
    buf: Buffer,
    gate: float = 0.1,
    grainsize_ms: float = 50.0,
    density: float = 1.0,
    duration: float = 5.0,
) -> Buffer:
    """Grain cloud generation from amplitude-detected grains."""
    ...

def grain_extend(
    buf: Buffer, grainsize_ms: float = 15.0, trough: float = 0.3, repeats: int = 2
) -> Buffer:
    """Extend duration using grain repetition."""
    ...

def texture_simple(
    buf: Buffer,
    duration: float = 5.0,
    density: float = 5.0,
    grainsize_ms: float = 50.0,
    scatter: float = 0.5,
) -> Buffer:
    """Simple texture synthesis."""
    ...

def texture_multi(
    buf: Buffer,
    duration: float = 5.0,
    density: float = 2.0,
    grainsize_ms: float = 50.0,
    num_layers: int = 4,
    pitch_spread: float = 0.0,
) -> Buffer:
    """Multi-layer grouped texture synthesis."""
    ...

# =============================================================================
# Extended granular operations
# =============================================================================

def grain_reorder(
    buf: Buffer,
    order: list[int] | None = None,
    gate: float = 0.1,
    grainsize_ms: float = 50.0,
) -> Buffer:
    """Reorder detected grains."""
    ...

def grain_rerhythm(
    buf: Buffer,
    times: list[float] | None = None,
    ratios: list[float] | None = None,
    gate: float = 0.1,
    grainsize_ms: float = 50.0,
) -> Buffer:
    """Change timing/rhythm of grains."""
    ...

def grain_reverse(buf: Buffer, gate: float = 0.1, grainsize_ms: float = 50.0) -> Buffer:
    """Reverse individual grains in place."""
    ...

def grain_timewarp(
    buf: Buffer,
    stretch: float = 1.0,
    stretch_curve: list[tuple[float, float]] | None = None,
    gate: float = 0.1,
    grainsize_ms: float = 50.0,
) -> Buffer:
    """Time-stretch/compress grain spacing."""
    ...

def grain_repitch(
    buf: Buffer,
    pitch_semitones: float = 0.0,
    pitch_curve: list[tuple[float, float]] | None = None,
    gate: float = 0.1,
    grainsize_ms: float = 50.0,
) -> Buffer:
    """Pitch-shift grains with interpolation."""
    ...

def grain_position(
    buf: Buffer,
    positions: list[float] | None = None,
    duration: float = 0.0,
    gate: float = 0.1,
    grainsize_ms: float = 50.0,
) -> Buffer:
    """Reposition grains in stereo field."""
    ...

def grain_omit(
    buf: Buffer,
    keep: int = 1,
    out_of: int = 2,
    gate: float = 0.1,
    grainsize_ms: float = 50.0,
) -> Buffer:
    """Probabilistically omit grains."""
    ...

def grain_duplicate(
    buf: Buffer, repeats: int = 2, gate: float = 0.1, grainsize_ms: float = 50.0
) -> Buffer:
    """Duplicate grains with variations."""
    ...

# =============================================================================
# Spectral operations
# =============================================================================

def spectral_focus(
    buf: Buffer,
    center_freq: float = 1000.0,
    bandwidth: float = 500.0,
    gain: float = 2.0,
    fft_size: int = 1024,
) -> Buffer:
    """Focus on frequency region."""
    ...

def spectral_hilite(
    buf: Buffer, threshold_db: float = -20.0, gain: float = 2.0, fft_size: int = 1024
) -> Buffer:
    """Highlight frequency region."""
    ...

def spectral_fold(
    buf: Buffer, fold_freq: float = 2000.0, fft_size: int = 1024
) -> Buffer:
    """Fold spectrum around frequency."""
    ...

def spectral_clean(
    buf: Buffer, threshold_db: float = -40.0, fft_size: int = 1024
) -> Buffer:
    """Remove spectral noise."""
    ...

# =============================================================================
# Experimental/Chaos
# =============================================================================

def strange(
    buf: Buffer, chaos_amount: float = 0.5, rate: float = 2.0, mode: int = 0
) -> Buffer:
    """Strange attractor transformation."""
    ...

def brownian(
    buf: Buffer, step_size: float = 0.1, smoothing: float = 0.9, mode: int = 0
) -> Buffer:
    """Brownian motion transformation."""
    ...

def crystal(
    buf: Buffer, density: float = 50.0, decay: float = 0.5, pitch_variance: float = 0.1
) -> Buffer:
    """Crystal growth patterns."""
    ...

def fractal(
    buf: Buffer, depth: int = 3, pitch_ratio: float = 0.5, amp_decay: float = 0.7
) -> Buffer:
    """Fractal transformation."""
    ...

def quirk(
    buf: Buffer, probability: float = 0.3, intensity: float = 0.5, seed: int = 0
) -> Buffer:
    """Quirky transformation."""
    ...

def chirikov(
    buf: Buffer, k_param: float = 2.0, mod_depth: float = 0.5, rate: float = 10.0
) -> Buffer:
    """Chirikov map transformation."""
    ...

def cantor(
    buf: Buffer, depth: int = 4, duty_cycle: float = 0.5, fade_ms: float = 5.0
) -> Buffer:
    """Cantor set transformation."""
    ...

def cascade(
    buf: Buffer,
    num_echoes: int = 6,
    delay_ms: float = 100.0,
    decay: float = 0.7,
    pitch_shift: float = -2.0,
) -> Buffer:
    """Cascade transformation."""
    ...

def fracture(
    buf: Buffer, fragment_ms: float = 50.0, gap_ratio: float = 0.5, scatter: float = 0.3
) -> Buffer:
    """Fracture transformation."""
    ...

def tesselate(
    buf: Buffer, tile_ms: float = 50.0, pattern: int = 1, overlap: float = 0.5
) -> Buffer:
    """Tesselation transformation."""
    ...

# =============================================================================
# Morphing/Cross-synthesis
# =============================================================================

def morph(
    buf1: Buffer, buf2: Buffer, amount: float = 0.5, fft_size: int = 1024
) -> Buffer:
    """Spectral morph between sounds."""
    ...

def morph_glide(buf1: Buffer, buf2: Buffer, fft_size: int = 1024) -> Buffer:
    """Gliding morph over time."""
    ...

def cross_synth(
    buf1: Buffer, buf2: Buffer, fft_size: int = 1024, mode: int = 0
) -> Buffer:
    """Cross-synthesis (vocoder-like)."""
    ...

def morph_glide_native(buf1: Buffer, buf2: Buffer, fft_size: int = 1024) -> Buffer:
    """Native CDP specglide wrapper."""
    ...

def morph_bridge_native(
    buf1: Buffer, buf2: Buffer, fft_size: int = 1024, interp_count: int = 8
) -> Buffer:
    """Native CDP specbridge wrapper."""
    ...

def morph_native(
    buf1: Buffer, buf2: Buffer, fft_size: int = 1024, interp_count: int = 8
) -> Buffer:
    """Native CDP specmorph wrapper."""
    ...

# =============================================================================
# Analysis
# =============================================================================

def pitch(
    buf: Buffer,
    min_freq: float = 50.0,
    max_freq: float = 2000.0,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> list[tuple[float, float]]:
    """Extract pitch data."""
    ...

def formants(
    buf: Buffer, lpc_order: int = 12, frame_size: int = 1024, hop_size: int = 256
) -> list[list[tuple[float, float]]]:
    """Extract formant data."""
    ...

def get_partials(
    buf: Buffer,
    min_amp_db: float = -60.0,
    max_partials: int = 100,
    fft_size: int = 4096,
) -> list[tuple[float, float, float]]:
    """Extract partial/harmonic data."""
    ...

# =============================================================================
# Playback/Time manipulation
# =============================================================================

def zigzag(buf: Buffer, times: Sequence[float], splice_ms: float = 15.0) -> Buffer:
    """Alternating forward/backward playback through time points."""
    ...

def iterate(
    buf: Buffer,
    repeats: int = 4,
    delay: float = 0.5,
    pitch_shift: float = 0.0,
    gain_decay: float = 0.9,
) -> Buffer:
    """Repeat audio with pitch shift and gain decay variations."""
    ...

def stutter(
    buf: Buffer,
    segment_ms: float = 100.0,
    duration: float = 5.0,
    silence_ratio: float = 0.5,
) -> Buffer:
    """Segment-based stuttering with silence inserts."""
    ...

def bounce(
    buf: Buffer,
    bounces: int = 8,
    initial_delay: float = 0.5,
    decay: float = 0.7,
    pitch_drop: float = 0.0,
) -> Buffer:
    """Bouncing ball effect with accelerating repeats."""
    ...

def drunk(
    buf: Buffer,
    duration: float = 5.0,
    step_ms: float = 100.0,
    max_step: float = 0.1,
    splice_ms: float = 10.0,
) -> Buffer:
    """Random 'drunk walk' navigation through audio."""
    ...

def loop(
    buf: Buffer,
    start: float = 0.0,
    length_ms: float = 500.0,
    repeats: int = 4,
    crossfade_ms: float = 10.0,
) -> Buffer:
    """Loop a section with crossfades and variations."""
    ...

def retime(
    buf: Buffer, ratio: float = 1.0, grain_ms: float = 50.0, overlap: float = 0.5
) -> Buffer:
    """Time-domain time stretch/compress (TDOLA)."""
    ...

def scramble(buf: Buffer, mode: int = 0, group_size: int = 2, seed: int = 0) -> Buffer:
    """Reorder wavesets (shuffle, reverse, by size/level)."""
    ...

def splinter(
    buf: Buffer,
    start: float = 0.0,
    duration_ms: float = 50.0,
    repeats: int = 20,
    shrink: float = 0.9,
) -> Buffer:
    """Fragmenting effect with shrinking repeats."""
    ...

# =============================================================================
# Spatial effects
# =============================================================================

def spin(
    buf: Buffer, rate: float = 1.0, doppler: float = 0.0, depth: float = 1.0
) -> Buffer:
    """Rotate audio around stereo field with optional doppler."""
    ...

def rotor(
    buf: Buffer,
    pitch_rate: float = 1.0,
    pitch_depth: float = 2.0,
    amp_rate: float = 1.5,
    amp_depth: float = 0.5,
) -> Buffer:
    """Dual-rotation modulation (pitch + amplitude interference)."""
    ...

def flutter(
    buf: Buffer, frequency: float = 4.0, depth: float = 1.0, phase: float = 0.0
) -> Buffer:
    """Spatial tremolo (loudness modulation alternating L/R)."""
    ...

# =============================================================================
# Synthesis
# =============================================================================

def synth_wave(
    waveform: int = ...,
    frequency: float = 440.0,
    amplitude: float = 0.8,
    duration: float = 1.0,
    sample_rate: int = 44100,
) -> Buffer:
    """Generate waveforms (sine, square, saw, ramp, triangle)."""
    ...

def synth_noise(
    pink: int = 0,
    amplitude: float = 0.8,
    duration: float = 1.0,
    sample_rate: int = 44100,
) -> Buffer:
    """Generate white or pink noise."""
    ...

def synth_click(
    tempo: float = 120.0,
    beats_per_bar: int = 4,
    duration: float = 10.0,
    sample_rate: int = 44100,
) -> Buffer:
    """Generate click/metronome track."""
    ...

def synth_chord(
    midi_notes: Sequence[int],
    amplitude: float = 0.8,
    duration: float = 1.0,
    sample_rate: int = 44100,
) -> Buffer:
    """Synthesize chord from MIDI note list."""
    ...

# =============================================================================
# Pitch-synchronous operations (PSOW)
# =============================================================================

def psow_stretch(
    buf: Buffer, stretch_factor: float = 1.0, grain_count: int = 1
) -> Buffer:
    """Time-stretch while preserving pitch (PSOLA)."""
    ...

def psow_grab(
    buf: Buffer, time: float = 0.0, duration: float = 0.0, grain_count: int = 1
) -> Buffer:
    """Extract pitch-synchronous grains from position."""
    ...

def psow_dupl(buf: Buffer, repeat_count: int = 2, grain_count: int = 1) -> Buffer:
    """Duplicate grains for time-stretching."""
    ...

def psow_interp(grain1: Buffer, grain2: Buffer, interp_count: int = 8) -> Buffer:
    """Interpolate between two grains."""
    ...

# =============================================================================
# FOF extraction and synthesis (FOFEX)
# =============================================================================

def fofex_extract(
    buf: Buffer, time: float, fof_count: int = 1, window: bool = True
) -> Buffer:
    """Extract single FOF (pitch-synchronous grain) at time."""
    ...

def fofex_extract_all(
    buf: Buffer, fof_count: int = 1, min_level_db: float = 0.0, window: bool = True
) -> Buffer:
    """Extract all FOFs to uniform-length bank."""
    ...

def fofex_synth(
    fof_bank: Buffer, duration: float, frequency: float, amplitude: float = 1.0
) -> Buffer:
    """Synthesize audio from FOFs at target pitch."""
    ...

def fofex_repitch(
    buf: Buffer, pitch_shift: float, preserve_formants: bool = True
) -> Buffer:
    """Repitch audio with optional formant preservation."""
    ...

# =============================================================================
# Hover/Constrict/Phase
# =============================================================================

def hover(
    buf: Buffer,
    frequency: float = 440.0,
    location: float = 0.5,
    duration: float = 5.0,
    grainsize_ms: float = 50.0,
) -> Buffer:
    """Zigzag reading at specified frequency for hovering pitch effect."""
    ...

def constrict(buf: Buffer, constriction: float = 50.0) -> Buffer:
    """Shorten or remove silent sections."""
    ...

def phase_stereo(buf: Buffer, transfer: float = 1.0) -> Buffer:
    """Enhance stereo separation via phase subtraction."""
    ...

# =============================================================================
# Granular texture
# =============================================================================

def wrappage(
    buf: Buffer,
    grain_size: float = 50.0,
    density: float = 1.0,
    velocity: float = 1.0,
    pitch: float = 0.0,
    spread: float = 1.0,
    jitter: float = 0.1,
    splice_ms: float = 5.0,
    duration: float = 0.0,
) -> Buffer:
    """Granular texture with stereo spatial distribution."""
    ...
