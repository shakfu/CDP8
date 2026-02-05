#!/usr/bin/env python3
"""
Demo 2: Effects and Audio Processing

This demo shows various audio effects:
- Delay and reverb
- Modulation effects (tremolo, chorus, flanger)
- Distortion
- Dynamics processing
- Filters
"""

import cycdp
import math
import array
import os

# Create output directory (use PYCDP_DEMO_OUTPUT env var or default to demos/build)
OUTPUT_DIR = os.environ.get("PYCDP_DEMO_OUTPUT", "build")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_test_sound(duration=0.5, sample_rate=44100):
    """Create a test sound with harmonics."""
    samples = array.array('f')
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Fundamental + harmonics for richer sound
        val = 0.4 * math.sin(2 * math.pi * 220 * t)       # Fundamental
        val += 0.2 * math.sin(2 * math.pi * 440 * t)      # 2nd harmonic
        val += 0.1 * math.sin(2 * math.pi * 660 * t)      # 3rd harmonic
        # Apply simple envelope
        env = min(t / 0.01, 1.0) * max(0, 1.0 - (t - 0.3) / 0.2)
        samples.append(val * env)
    return cycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

print("=== Effects and Processing Demo ===\n")

# Create test sound
buf = create_test_sound(duration=0.5)
print(f"Created test sound: {buf.frame_count} frames")

# =============================================================================
# Delay Effects
# =============================================================================

print("\n=== Delay Effects ===")

# Simple delay
delayed = cycdp.delay(buf, delay_ms=250.0, feedback=0.4, mix=0.5)
print(f"Applied delay: 250ms, 40% feedback, 50% mix")
print(f"  Output length: {delayed.frame_count} frames")

# =============================================================================
# Reverb
# =============================================================================

print("\n=== Reverb ===")

# Reverb effect
reverbed = cycdp.reverb(buf, decay_time=2.0, damping=0.5, mix=0.4)
print(f"Applied reverb: decay=2.0s, damping=0.5")
print(f"  Output length: {reverbed.frame_count} frames")

# =============================================================================
# Modulation Effects
# =============================================================================

print("\n=== Modulation Effects ===")

# Tremolo (amplitude modulation)
tremolo = cycdp.tremolo(buf, freq=6.0, depth=0.7)
print(f"Applied tremolo: 6 Hz, 70% depth")

# Chorus
chorus = cycdp.chorus(buf, rate=1.5, depth_ms=20.0, mix=0.5)
print(f"Applied chorus")

# Flanger
flanged = cycdp.flanger(buf, rate=0.5, depth_ms=5.0, feedback=0.5)
print(f"Applied flanger")

# Ring modulation
ring = cycdp.ring_mod(buf, freq=150.0)
print(f"Applied ring modulation: 150 Hz carrier")

# =============================================================================
# Spatial Effects
# =============================================================================

print("\n=== Spatial Effects ===")

# Spin - rotate around stereo field
spun = cycdp.spin(buf, rate=0.5)  # 0.5 Hz rotation
print(f"Applied spin: 0.5 Hz rotation")
print(f"  Output: {spun.channels} channels")

# Rotor - dual rotation modulation
rotor = cycdp.rotor(buf, pitch_rate=3.0, pitch_depth=2.0, amp_rate=2.0, amp_depth=0.5)
print(f"Applied rotor: pitch_rate=3Hz, amp_rate=2Hz")

# Flutter - spatial tremolo
flutter = cycdp.flutter(buf, frequency=4.0, depth=0.6)
print(f"Applied flutter: 4 Hz, 60% depth")
print(f"  Output: {flutter.channels} channels")

# =============================================================================
# Distortion Effects
# =============================================================================

print("\n=== Distortion Effects ===")

# Waveset repeat
repeated = cycdp.distort_repeat(buf, multiplier=2)
print(f"Applied waveset repeat: 2x")

# Waveset reverse
wrev = cycdp.distort_reverse(buf, cycle_count=4)
print(f"Applied waveset reverse: 4 cycles")

# Fractal distortion
fractal = cycdp.distort_fractal(buf, scaling=0.5)
print(f"Applied fractal distortion: scaling=0.5")

# Bitcrush
crushed = cycdp.bitcrush(buf, bit_depth=8)
print(f"Applied bitcrush: 8 bits")

# =============================================================================
# Dynamics Processing
# =============================================================================

print("\n=== Dynamics Processing ===")

# Create a sound with varying dynamics for compression demo
dynamic_samples = array.array('f')
for i in range(44100):
    t = i / 44100
    # Quiet section then loud section
    if t < 0.5:
        amp = 0.2
    else:
        amp = 0.8
    dynamic_samples.append(amp * math.sin(2 * math.pi * 440 * t))

dynamic_buf = cycdp.Buffer.from_memoryview(dynamic_samples, channels=1, sample_rate=44100)

# Compression
compressed = cycdp.compressor(dynamic_buf, threshold_db=-12.0, ratio=4.0)
print(f"Applied compression: -12dB threshold, 4:1 ratio")

# Gate
gated = cycdp.gate(dynamic_buf, threshold_db=-20.0)
print(f"Applied gate: -20dB threshold")

# Limiter
limited = cycdp.limiter(dynamic_buf, threshold_db=-3.0)
print(f"Applied limiter: -3dB threshold")

# =============================================================================
# Filter Effects
# =============================================================================

print("\n=== Filter Effects ===")

# Low-pass filter
lowpass = cycdp.filter_lowpass(buf, cutoff_freq=1000.0)
print(f"Applied low-pass filter: 1000 Hz cutoff")

# High-pass filter
highpass = cycdp.filter_highpass(buf, cutoff_freq=500.0)
print(f"Applied high-pass filter: 500 Hz cutoff")

# Band-pass filter (low_freq and high_freq)
bandpass = cycdp.filter_bandpass(buf, low_freq=600.0, high_freq=1000.0)
print(f"Applied band-pass filter: 600-1000 Hz")

# Notch filter
notch = cycdp.filter_notch(buf, center_freq=1000.0, width_hz=100.0)
print(f"Applied notch filter: 1000 Hz center, 100 Hz width")

# =============================================================================
# Write Output Files
# =============================================================================

print("\n=== Writing Output Files ===")

outputs = {
    "02_source.wav": cycdp.normalize(buf),
    "02_delay.wav": cycdp.normalize(delayed),
    "02_reverb.wav": cycdp.normalize(reverbed),
    "02_tremolo.wav": cycdp.normalize(tremolo),
    "02_chorus.wav": cycdp.normalize(chorus),
    "02_flanger.wav": cycdp.normalize(flanged),
    "02_ring_mod.wav": cycdp.normalize(ring),
    "02_spin.wav": cycdp.normalize(spun),
    "02_flutter.wav": cycdp.normalize(flutter),
    "02_distort_repeat.wav": cycdp.normalize(repeated),
    "02_bitcrush.wav": cycdp.normalize(crushed),
    "02_compressed.wav": cycdp.normalize(compressed),
    "02_lowpass.wav": cycdp.normalize(lowpass),
    "02_highpass.wav": cycdp.normalize(highpass),
}

for filename, buffer in outputs.items():
    path = os.path.join(OUTPUT_DIR, filename)
    cycdp.write_file(path, buffer)
    print(f"  Wrote: {filename}")

print("\n=== Demo Complete ===")
