"""
pycdp - Python bindings for CDP audio processing library.

Example usage:
    >>> import numpy as np
    >>> import pycdp
    >>> samples = np.array([0.5, 0.3, -0.2], dtype=np.float32)
    >>> pycdp.gain(samples, gain_factor=2.0)
    array([1. , 0.6, -0.4], dtype=float32)
    >>> pycdp.normalize(samples, target=1.0)
    array([1. , 0.6, -0.4], dtype=float32)
"""

from pycdp._core import (
    # Version and utilities
    version,
    gain_to_db,
    db_to_gain,
    # Constants
    FLAG_NONE,
    FLAG_CLIP,
    # Exception
    CDPError,
    # Classes
    Context,
    Buffer,
    # Low-level functions (work with Buffer objects)
    apply_gain,
    apply_gain_db,
    apply_normalize,
    apply_normalize_db,
    apply_phase_invert,
    get_peak,
    # High-level functions (accept any float32 buffer via memoryview)
    gain,
    gain_db,
    normalize,
    normalize_db,
    phase_invert,
    peak,
    # File I/O
    read_file,
    write_file,
    # Spatial/panning
    pan,
    pan_envelope,
    mirror,
    narrow,
    # Mixing
    mix,
    mix2,
    # Buffer utilities
    reverse,
    fade_in,
    fade_out,
    concat,
    # Channel operations
    to_mono,
    to_stereo,
    extract_channel,
    merge_channels,
    split_channels,
    interleave,
    # Spectral processing
    time_stretch,
    spectral_blur,
    modify_speed,
    pitch_shift,
    spectral_shift,
    spectral_stretch,
    filter_lowpass,
    filter_highpass,
    filter_bandpass,
    filter_notch,
    # Dynamics/effects
    gate,
    bitcrush,
    ring_mod,
    delay,
    chorus,
    flanger,
    # EQ and dynamics
    eq_parametric,
    envelope_follow,
    envelope_apply,
    compressor,
    limiter,
    # Envelope operations
    dovetail,
    tremolo,
    attack,
    # Distortion operations
    distort_overload,
    distort_reverse,
    distort_fractal,
    distort_shuffle,
    # Reverb
    reverb,
    # Granular operations
    brassage,
    freeze,
    grain_cloud,
    grain_extend,
    texture_simple,
    texture_multi,
    # Morphing/Cross-synthesis
    morph,
    morph_glide,
    cross_synth,
    # Analysis
    pitch,
    formants,
    get_partials,
)

__all__ = [
    # Version
    "version",
    # Utilities
    "gain_to_db",
    "db_to_gain",
    # Constants
    "FLAG_NONE",
    "FLAG_CLIP",
    # Exception
    "CDPError",
    # Classes
    "Context",
    "Buffer",
    # Low-level functions
    "apply_gain",
    "apply_gain_db",
    "apply_normalize",
    "apply_normalize_db",
    "apply_phase_invert",
    "get_peak",
    # High-level functions
    "gain",
    "gain_db",
    "normalize",
    "normalize_db",
    "phase_invert",
    "peak",
    # File I/O
    "read_file",
    "write_file",
    # Spatial/panning
    "pan",
    "pan_envelope",
    "mirror",
    "narrow",
    # Mixing
    "mix",
    "mix2",
    # Buffer utilities
    "reverse",
    "fade_in",
    "fade_out",
    "concat",
    # Channel operations
    "to_mono",
    "to_stereo",
    "extract_channel",
    "merge_channels",
    "split_channels",
    "interleave",
    # Spectral processing
    "time_stretch",
    "spectral_blur",
    "modify_speed",
    "pitch_shift",
    "spectral_shift",
    "spectral_stretch",
    "filter_lowpass",
    "filter_highpass",
    "filter_bandpass",
    "filter_notch",
    # Dynamics/effects
    "gate",
    "bitcrush",
    "ring_mod",
    "delay",
    "chorus",
    "flanger",
    # EQ and dynamics
    "eq_parametric",
    "envelope_follow",
    "envelope_apply",
    "compressor",
    "limiter",
    # Envelope operations
    "dovetail",
    "tremolo",
    "attack",
    # Distortion operations
    "distort_overload",
    "distort_reverse",
    "distort_fractal",
    "distort_shuffle",
    # Reverb
    "reverb",
    # Granular operations
    "brassage",
    "freeze",
    "grain_cloud",
    "grain_extend",
    "texture_simple",
    "texture_multi",
    # Morphing/Cross-synthesis
    "morph",
    "morph_glide",
    "cross_synth",
    # Analysis
    "pitch",
    "formants",
    "get_partials",
]

__version__ = version()
