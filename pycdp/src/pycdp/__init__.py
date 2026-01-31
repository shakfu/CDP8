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
]

__version__ = version()
