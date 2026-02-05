#!/usr/bin/env python3
"""
Demo 4: Granular Synthesis

This demo shows CDP's granular processing capabilities:
- Brassage (classic granular time-stretch)
- Grain manipulation (reorder, reverse, repitch)
- Wrappage (granular texture with spatial distribution)
"""

import cycdp
import math
import array
import os

# Create output directory (use PYCDP_DEMO_OUTPUT env var or default to demos/build)
OUTPUT_DIR = os.environ.get("PYCDP_DEMO_OUTPUT", "build")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_vocal_like_sound(duration=1.0, sample_rate=44100):
    """Create a sound that simulates vocal-like qualities."""
    samples = array.array('f')
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Fundamental with formant-like harmonics
        f0 = 150.0  # Fundamental frequency
        val = 0.4 * math.sin(2 * math.pi * f0 * t)
        val += 0.3 * math.sin(2 * math.pi * f0 * 2 * t)
        val += 0.2 * math.sin(2 * math.pi * f0 * 3 * t)
        val += 0.15 * math.sin(2 * math.pi * f0 * 4 * t)
        val += 0.1 * math.sin(2 * math.pi * f0 * 5 * t)
        # Add slight vibrato
        val *= (1.0 + 0.02 * math.sin(2 * math.pi * 5 * t))
        # Envelope
        env = min(t / 0.1, 1.0) * max(0, 1.0 - (t - 0.7) / 0.3)
        samples.append(val * env * 0.7)
    return cycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

print("=== Granular Synthesis Demo ===\n")

buf = create_vocal_like_sound(duration=0.5)
print(f"Created test sound: {buf.frame_count} frames")

# =============================================================================
# Brassage - Classic Granular Time-Stretching
# =============================================================================

print("\n=== Brassage (Granular Time-Stretch) ===")

# Time stretch 2x with granular processing
stretched = cycdp.brassage(buf,
    velocity=0.5,         # Half speed = 2x stretch
    density=1.0,          # Normal grain density
    grainsize_ms=50.0,    # 50ms grains
    pitch_shift=0.0       # No pitch change
)
print(f"Brassage stretch 2x: {buf.frame_count} -> {stretched.frame_count} frames")

# Time compress with pitch preservation
compressed = cycdp.brassage(buf,
    velocity=2.0,         # Double speed
    density=1.0,
    grainsize_ms=30.0
)
print(f"Brassage compress 0.5x: {compressed.frame_count} frames")

# Pitch shift without time change
pitched = cycdp.brassage(buf,
    velocity=1.0,
    density=1.0,
    grainsize_ms=40.0,
    pitch_shift=7.0       # Up a fifth (7 semitones)
)
print(f"Brassage pitch +7 semitones: {pitched.frame_count} frames")

# Dense granular cloud
cloud = cycdp.brassage(buf,
    velocity=0.25,        # 4x stretch
    density=4.0,          # High overlap
    grainsize_ms=80.0,    # Larger grains
    scatter=0.5           # Random grain placement
)
print(f"Granular cloud (4x stretch, high density): {cloud.frame_count} frames")

# With amplitude variation
varied = cycdp.brassage(buf,
    velocity=0.5,
    density=2.0,
    grainsize_ms=40.0,
    scatter=0.3,
    amp_variation=0.4     # Random amplitude per grain
)
print(f"Brassage with amplitude variation")

# =============================================================================
# Grain Operations
# =============================================================================

print("\n=== Grain Operations ===")

# Shuffle grains (random reorder)
shuffled = cycdp.grain_reorder(buf)  # order=None means shuffle
print(f"Shuffled grains (random order)")

# Custom grain order (first 4 grains in reverse)
custom_order = cycdp.grain_reorder(buf, order=[3, 2, 1, 0])
print(f"Custom grain order: [3, 2, 1, 0]")

# Reverse grain order (grains in reverse sequence)
reversed_grains = cycdp.grain_reverse(buf)
print(f"Reversed grain order")

# Time-warp grain spacing
warped = cycdp.grain_timewarp(buf, stretch=1.5)
print(f"Time-warped grains: 1.5x spacing")

# Variable time warp with curve
warped_curve = cycdp.grain_timewarp(buf, stretch_curve=[
    (0.0, 1.0),   # Start at normal speed
    (0.5, 2.0),   # Slow down in middle
    (1.0, 0.5)    # Speed up at end
])
print(f"Time-warped with curve: normal -> slow -> fast")

# Repitch grains
repitched = cycdp.grain_repitch(buf, pitch_semitones=5.0)
print(f"Repitched grains: +5 semitones")

# Variable pitch with curve
repitched_curve = cycdp.grain_repitch(buf, pitch_curve=[
    (0.0, 0.0),    # Start at original pitch
    (0.5, 7.0),    # Up a fifth in middle
    (1.0, -5.0)    # Down at end
])
print(f"Repitched with curve: 0 -> +7 -> -5 semitones")

# =============================================================================
# Wrappage - Granular Texture with Spatial Distribution
# =============================================================================

print("\n=== Wrappage (Spatial Granular Texture) ===")

# Basic wrappage - creates stereo texture from mono
texture = cycdp.wrappage(buf,
    grain_size=50.0,
    density=1.5,
    velocity=1.0,
    spread=1.0            # Full stereo spread
)
print(f"Basic wrappage texture: {texture.channels} channels")

# Time-stretched texture
stretched_texture = cycdp.wrappage(buf,
    grain_size=40.0,
    density=2.0,
    velocity=0.5,         # Half speed = 2x stretch
    spread=0.8
)
print(f"Stretched texture (2x): {stretched_texture.frame_count} frames")

# Frozen texture (hold a moment)
frozen = cycdp.wrappage(buf,
    grain_size=60.0,
    density=3.0,
    velocity=0.0,         # Freeze
    spread=1.0,
    jitter=0.3,           # Add variation
    duration=2.0          # Required when velocity=0
)
print(f"Frozen texture (2 seconds): {frozen.frame_count} frames")

# Pitched texture
pitched_texture = cycdp.wrappage(buf,
    grain_size=35.0,
    density=2.0,
    velocity=1.0,
    pitch=-12.0,          # Octave down
    spread=0.6
)
print(f"Pitched texture (octave down): {pitched_texture.frame_count} frames")

# Dense cloud texture
dense_cloud = cycdp.wrappage(buf,
    grain_size=20.0,      # Small grains
    density=5.0,          # Very dense
    velocity=0.2,         # Slow
    pitch=0.0,
    spread=1.0,
    jitter=0.5            # High variation
)
print(f"Dense cloud texture: {dense_cloud.frame_count} frames")

# Sparse, rhythmic texture
sparse = cycdp.wrappage(buf,
    grain_size=30.0,
    density=0.5,          # Gaps between grains
    velocity=1.0,
    spread=0.7
)
print(f"Sparse texture: {sparse.frame_count} frames")

# =============================================================================
# Creative Granular Chains
# =============================================================================

print("\n=== Creative Granular Chain ===")

# Create an evolving texture
result = buf

# First pass: stretch and add space
result = cycdp.wrappage(result,
    grain_size=60.0,
    density=2.0,
    velocity=0.5,
    spread=0.7
)
print(f"Pass 1 (wrappage stretch): {result.frame_count} frames, {result.channels} channels")

# Convert stereo result to mono for second granular pass
# (wrappage outputs stereo, but some granular functions need mono)
result_mono = cycdp.to_mono(result)
print(f"Converted to mono: {result_mono.frame_count} frames")

# Second pass: add rhythmic variation
result_final = cycdp.brassage(result_mono,
    velocity=1.0,
    density=1.5,
    grainsize_ms=40.0,
    scatter=0.3
)
print(f"Pass 2 (brassage scatter): {result_final.frame_count} frames")

# =============================================================================
# Grain Cloud Textures
# =============================================================================

print("\n=== Grain Cloud Textures ===")

# Create from short source using grain_cloud
grain_cloud = cycdp.grain_cloud(buf, gate=0.1, grainsize_ms=30.0)
print(f"Grain cloud: {grain_cloud.frame_count} frames")

# Extend grains into sustained texture
extended = cycdp.grain_extend(buf, grainsize_ms=20.0, trough=0.3)
print(f"Extended grains: {extended.frame_count} frames")

# =============================================================================
# Write Output Files
# =============================================================================

print("\n=== Writing Output Files ===")

outputs = {
    "04_source.wav": cycdp.normalize(buf),
    "04_brassage_stretch.wav": cycdp.normalize(stretched),
    "04_brassage_pitch.wav": cycdp.normalize(pitched),
    "04_granular_cloud.wav": cycdp.normalize(cloud),
    "04_grain_shuffle.wav": cycdp.normalize(shuffled),
    "04_grain_reverse.wav": cycdp.normalize(reversed_grains),
    "04_grain_repitch.wav": cycdp.normalize(repitched),
    "04_wrappage_basic.wav": cycdp.normalize(texture),
    "04_wrappage_frozen.wav": cycdp.normalize(frozen),
    "04_wrappage_dense.wav": cycdp.normalize(dense_cloud),
    "04_grain_extend.wav": cycdp.normalize(extended),
}

for filename, buffer in outputs.items():
    path = os.path.join(OUTPUT_DIR, filename)
    cycdp.write_file(path, buffer)
    print(f"  Wrote: {filename}")

print("\n=== Demo Complete ===")
