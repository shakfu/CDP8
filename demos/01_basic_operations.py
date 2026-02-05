#!/usr/bin/env python3
"""
Demo 1: Basic Operations with cycdp

This demo shows fundamental operations:
- Creating buffers from scratch
- Reading/writing audio files
- Basic gain and normalization
- Simple effects
"""

import cycdp
import math
import array
import os

# Create output directory (use PYCDP_DEMO_OUTPUT env var or default to demos/build)
OUTPUT_DIR = os.environ.get("PYCDP_DEMO_OUTPUT", "build")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"cycdp version: {cycdp.version()}")
print()

# =============================================================================
# Creating Buffers Programmatically
# =============================================================================

print("=== Creating Buffers ===")

# Create a simple sine wave (440 Hz, 1 second, mono)
sample_rate = 44100
duration = 1.0
frequency = 440.0

samples = array.array('f')
for i in range(int(sample_rate * duration)):
    t = i / sample_rate
    samples.append(0.5 * math.sin(2 * math.pi * frequency * t))

# Create a Buffer from the samples
buf = cycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)
print(f"Created buffer: {buf.frame_count} frames, {buf.channels} channel(s), {buf.sample_rate} Hz")

# =============================================================================
# Basic Gain Operations
# =============================================================================

print("\n=== Gain Operations ===")

# Apply gain (double the amplitude)
louder = cycdp.gain(buf, 2.0)
print(f"Applied 2x gain: peak = {cycdp.peak(louder)[0]:.3f}")

# Apply gain in decibels
boosted = cycdp.gain_db(buf, 6.0)  # +6 dB (approximately 2x)
print(f"Applied +6 dB: peak = {cycdp.peak(boosted)[0]:.3f}")

# Normalize to unity (peak = 1.0)
normalized = cycdp.normalize(buf)
print(f"Normalized: peak = {cycdp.peak(normalized)[0]:.3f}")

# Normalize to specific level in dB
normalized_db = cycdp.normalize_db(buf, -3.0)  # -3 dB below unity
print(f"Normalized to -3 dB: peak = {cycdp.peak(normalized_db)[0]:.3f}")

# =============================================================================
# Phase Operations
# =============================================================================

print("\n=== Phase Operations ===")

# Invert phase
inverted = cycdp.phase_invert(buf)
print(f"Phase inverted: first sample {buf[0]:.4f} -> {inverted[0]:.4f}")

# Double invert should return original
double_inverted = cycdp.phase_invert(inverted)
print(f"Double inverted matches original: {abs(buf[0] - double_inverted[0]) < 1e-6}")

# =============================================================================
# Buffer Utilities
# =============================================================================

print("\n=== Buffer Utilities ===")

# Reverse
reversed_buf = cycdp.reverse(buf)
print(f"Reversed buffer: {reversed_buf.frame_count} frames")

# Fade in (note: duration is in seconds, and modifies in-place)
# Make a copy first since fade_in modifies in-place
buf_copy = cycdp.Buffer.from_memoryview(
    array.array('f', [buf[i] for i in range(buf.frame_count)]),
    channels=1, sample_rate=sample_rate
)
cycdp.fade_in(buf_copy, duration=0.1)  # 100ms = 0.1s
print(f"Applied 100ms fade in")

# Fade out
buf_copy2 = cycdp.Buffer.from_memoryview(
    array.array('f', [buf[i] for i in range(buf.frame_count)]),
    channels=1, sample_rate=sample_rate
)
cycdp.fade_out(buf_copy2, duration=0.1)
print(f"Applied 100ms fade out")

# Concatenate buffers
concatenated = cycdp.concat([buf, reversed_buf])
print(f"Concatenated: {buf.frame_count} + {reversed_buf.frame_count} = {concatenated.frame_count} frames")

# =============================================================================
# Stereo Operations
# =============================================================================

print("\n=== Stereo Operations ===")

# Pan mono to stereo
panned_center = cycdp.pan(buf, 0.0)   # Center
panned_left = cycdp.pan(buf, -1.0)    # Hard left
panned_right = cycdp.pan(buf, 1.0)    # Hard right
print(f"Panned mono to stereo: {panned_center.channels} channels")

# Pan with envelope (automated panning)
pan_envelope = [(0.0, -1.0), (0.5, 1.0), (1.0, -1.0)]  # L -> R -> L
auto_panned = cycdp.pan_envelope(buf, pan_envelope)
print(f"Applied pan envelope: L -> R -> L")

# Mirror stereo (swap L/R)
mirrored = cycdp.mirror(panned_center)
print(f"Mirrored stereo channels")

# Narrow stereo width
narrowed = cycdp.narrow(panned_center, 0.5)  # 50% width
print(f"Narrowed stereo width to 50%")

# =============================================================================
# Mixing
# =============================================================================

print("\n=== Mixing ===")

# Create a second tone (different frequency)
samples2 = array.array('f')
for i in range(int(sample_rate * duration)):
    t = i / sample_rate
    samples2.append(0.3 * math.sin(2 * math.pi * 880 * t))  # 880 Hz

buf2 = cycdp.Buffer.from_memoryview(samples2, channels=1, sample_rate=sample_rate)

# Mix two buffers
mixed = cycdp.mix2(buf, buf2, gain_a=0.7, gain_b=0.3)
print(f"Mixed two buffers with gains 0.7 and 0.3")

# Mix multiple buffers
multi_mixed = cycdp.mix([buf, buf2, reversed_buf], gains=[0.5, 0.3, 0.2])
print(f"Mixed three buffers")

# =============================================================================
# Reading/Writing Files (if you have audio files)
# =============================================================================

print("\n=== File I/O ===")
print("To read a file:  buf = cycdp.read_file('input.wav')")
print("To write a file: cycdp.write_file('output.wav', buf)")

# =============================================================================
# Write Output Files
# =============================================================================

print("\n=== Writing Output Files ===")

outputs = {
    "01_sine_440hz.wav": cycdp.normalize(buf),
    "01_normalized.wav": normalized,
    "01_reversed.wav": cycdp.normalize(reversed_buf),
    "01_faded.wav": cycdp.normalize(buf_copy),
    "01_panned_center.wav": cycdp.normalize(panned_center),
    "01_auto_panned.wav": cycdp.normalize(auto_panned),
    "01_mixed.wav": cycdp.normalize(mixed),
}

for filename, buffer in outputs.items():
    path = os.path.join(OUTPUT_DIR, filename)
    cycdp.write_file(path, buffer)
    print(f"  Wrote: {filename}")

print("\n=== Demo Complete ===")
