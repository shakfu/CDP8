#!/usr/bin/env python3
"""
Demo 3: Spectral Processing

This demo shows CDP's powerful spectral processing capabilities:
- Spectral blur and focus
- Time stretching and pitch shifting
- Spectral transformations
- Freeze effects
"""

import cycdp
import math
import array
import os

# Create output directory (use PYCDP_DEMO_OUTPUT env var or default to demos/build)
OUTPUT_DIR = os.environ.get("PYCDP_DEMO_OUTPUT", "build")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_complex_sound(duration=1.0, sample_rate=44100):
    """Create a sound with multiple frequency components."""
    samples = array.array('f')
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Multiple sine waves at different frequencies
        val = 0.3 * math.sin(2 * math.pi * 200 * t)   # Low
        val += 0.25 * math.sin(2 * math.pi * 400 * t)  # Mid-low
        val += 0.2 * math.sin(2 * math.pi * 800 * t)   # Mid
        val += 0.15 * math.sin(2 * math.pi * 1600 * t) # Mid-high
        val += 0.1 * math.sin(2 * math.pi * 3200 * t)  # High
        # Envelope
        env = min(t / 0.05, 1.0) * max(0, 1.0 - t / duration)
        samples.append(val * env)
    return cycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

print("=== Spectral Processing Demo ===\n")

buf = create_complex_sound(duration=0.5)
print(f"Created complex test sound: {buf.frame_count} frames")

# =============================================================================
# Spectral Blur and Focus
# =============================================================================

print("\n=== Spectral Blur & Focus ===")

# Blur - smear frequencies over time
blurred = cycdp.spectral_blur(buf, blur_time=0.1)  # blur_time in seconds
print(f"Applied spectral blur: blur_time=0.1s")
print(f"  Creates a softer, more diffuse sound")

# Focus - enhance frequencies around a center point
focused = cycdp.spectral_focus(buf, center_freq=800.0, bandwidth=200.0, gain_db=6.0)
print(f"Applied spectral focus: center=800Hz, bandwidth=200Hz, +6dB")
print(f"  Emphasizes frequencies around 800 Hz")

# =============================================================================
# Time Stretching (preserves pitch)
# =============================================================================

print("\n=== Time Stretching ===")

# Stretch to double length (without pitch change)
stretched = cycdp.time_stretch(buf, factor=2.0)
print(f"Applied time stretch: 2x longer")
print(f"  Input: {buf.frame_count} frames")
print(f"  Output: {stretched.frame_count} frames")

# Compress to half length
compressed = cycdp.time_stretch(buf, factor=0.5)
print(f"Applied time compress: 0.5x (half length)")
print(f"  Output: {compressed.frame_count} frames")

# =============================================================================
# Pitch Shifting (preserves duration)
# =============================================================================

print("\n=== Pitch Shifting ===")

# Shift up one octave
shifted_up = cycdp.pitch_shift(buf, semitones=12.0)
print(f"Applied pitch shift: +12 semitones (octave up)")

# Shift down a fifth
shifted_down = cycdp.pitch_shift(buf, semitones=-7.0)
print(f"Applied pitch shift: -7 semitones (fifth down)")

# Micro-tuning
detuned = cycdp.pitch_shift(buf, semitones=0.5)
print(f"Applied pitch shift: +0.5 semitones (quarter tone)")

# =============================================================================
# Spectral Frequency Shift (inharmonic effects)
# =============================================================================

print("\n=== Spectral Frequency Shift ===")

# Shift all frequencies by constant Hz (creates inharmonic effects)
shifted_hz = cycdp.spectral_shift(buf, shift_hz=100.0)
print(f"Applied spectral shift: +100 Hz (inharmonic)")
print(f"  Unlike pitch shift, this adds constant Hz to all frequencies")

shifted_hz_down = cycdp.spectral_shift(buf, shift_hz=-50.0)
print(f"Applied spectral shift: -50 Hz")

# =============================================================================
# Spectral Stretching (inharmonic partials)
# =============================================================================

print("\n=== Spectral Stretch (Inharmonic) ===")

# Stretch higher frequencies more than lower ones
spec_stretched = cycdp.spectral_stretch(buf, max_stretch=2.0, freq_divide=500.0)
print(f"Applied spectral stretch: max_stretch=2.0, freq_divide=500Hz")
print(f"  Frequencies above 500 Hz get progressively stretched")
print(f"  Creates inharmonic, bell-like textures")

# =============================================================================
# Spectral Cleaning and Effects
# =============================================================================

print("\n=== Spectral Cleaning & Effects ===")

# Spectral gate - remove quiet frequency bins
cleaned = cycdp.spectral_clean(buf, threshold_db=-40.0)
print(f"Applied spectral clean: threshold=-40dB")
print(f"  Removes frequency bins below threshold")

# Spectral highlight - boost peaks
hilited = cycdp.spectral_hilite(buf, threshold_db=-20.0, boost_db=6.0)
print(f"Applied spectral hilite: boost peaks by 6dB")

# Spectral fold - mirror frequencies above fold point
folded = cycdp.spectral_fold(buf, fold_freq=2000.0)
print(f"Applied spectral fold: fold_freq=2000Hz")
print(f"  Creates metallic, inharmonic textures")

# =============================================================================
# Freeze Effect
# =============================================================================

print("\n=== Freeze Effect ===")

# Freeze a segment and sustain it
frozen = cycdp.freeze(buf, start_time=0.1, end_time=0.2, duration=2.0)
print(f"Applied freeze: segment 0.1-0.2s -> 2.0s output")
print(f"  Output: {frozen.frame_count} frames")

# Freeze with pitch scatter for texture
frozen_scatter = cycdp.freeze(buf, start_time=0.1, end_time=0.2,
                               duration=2.0, pitch_scatter=2.0, randomize=0.5)
print(f"Applied freeze with scatter: pitch_scatter=2 semitones")

# =============================================================================
# Spectral Effects Chain
# =============================================================================

print("\n=== Spectral Effect Chain ===")

# Combine multiple spectral operations
result = buf
result = cycdp.time_stretch(result, factor=1.5)   # Slightly longer
result = cycdp.spectral_blur(result, blur_time=0.05)  # Soften
result = cycdp.pitch_shift(result, semitones=-5.0)     # Down a fourth

print(f"Chain: time_stretch 1.5x -> blur 0.05s -> pitch_shift -5 semitones")
print(f"  Input: {buf.frame_count} frames")
print(f"  Output: {result.frame_count} frames")

# =============================================================================
# FFT Size Considerations
# =============================================================================

print("\n=== FFT Size Considerations ===")

print("Spectral processing uses Phase Vocoder analysis:")
print("- Higher FFT size = better frequency resolution, smeared transients")
print("- Lower FFT size = better time resolution, less frequency detail")
print("- Most functions allow specifying fft_size parameter")

# Example with explicit FFT size
blurred_hires = cycdp.spectral_blur(buf, blur_time=0.1, fft_size=4096)
print(f"\nHigh-resolution blur (FFT=4096): cleaner frequencies")

blurred_lores = cycdp.spectral_blur(buf, blur_time=0.1, fft_size=512)
print(f"Low-resolution blur (FFT=512): better transient preservation")

# =============================================================================
# Write Output Files
# =============================================================================

print("\n=== Writing Output Files ===")

outputs = {
    "03_source.wav": cycdp.normalize(buf),
    "03_blur.wav": cycdp.normalize(blurred),
    "03_focus.wav": cycdp.normalize(focused),
    "03_time_stretch_2x.wav": cycdp.normalize(stretched),
    "03_time_compress_half.wav": cycdp.normalize(compressed),
    "03_pitch_up_octave.wav": cycdp.normalize(shifted_up),
    "03_pitch_down_fifth.wav": cycdp.normalize(shifted_down),
    "03_spectral_shift_100hz.wav": cycdp.normalize(shifted_hz),
    "03_spectral_stretch.wav": cycdp.normalize(spec_stretched),
    "03_spectral_fold.wav": cycdp.normalize(folded),
    "03_freeze.wav": cycdp.normalize(frozen),
    "03_effect_chain.wav": cycdp.normalize(result),
}

for filename, buffer in outputs.items():
    path = os.path.join(OUTPUT_DIR, filename)
    cycdp.write_file(path, buffer)
    print(f"  Wrote: {filename}")

print("\n=== Demo Complete ===")
