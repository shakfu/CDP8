#!/usr/bin/env python3
"""
Demo 5: Pitch-Synchronous Processing

This demo shows CDP's pitch-synchronous (PSOW) processing:
- PSOW stretching and duplication
- PSOW grain extraction and interpolation
- FOF synthesis and manipulation
- Hover (zigzag reading) effects
"""

import pycdp
import math
import array
import os

# Create output directory (use PYCDP_DEMO_OUTPUT env var or default to demos/build)
OUTPUT_DIR = os.environ.get("PYCDP_DEMO_OUTPUT", "build")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_pitched_sound(frequency=220.0, duration=0.5, sample_rate=44100):
    """Create a clearly pitched sound for PSOW processing."""
    samples = array.array('f')
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Sawtooth-ish wave (clear pitch, rich harmonics)
        val = 0.0
        for h in range(1, 8):  # First 7 harmonics
            val += (0.5 / h) * math.sin(2 * math.pi * frequency * h * t)
        # Amplitude envelope
        env = min(t / 0.02, 1.0) * max(0, 1.0 - (t - 0.3) / 0.2)
        samples.append(val * env * 0.5)
    return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

print("=== Pitch-Synchronous Processing Demo ===\n")

buf = create_pitched_sound(frequency=220.0, duration=0.5)
print(f"Created pitched sound (A3 = 220 Hz): {buf.frame_count} frames")

# =============================================================================
# PSOW Stretching - Time-stretch preserving pitch characteristics
# =============================================================================

print("\n=== PSOW Stretching ===")

# Stretch 2x while preserving pitch characteristics
stretched = pycdp.psow_stretch(buf, stretch_factor=2.0)
print(f"PSOW stretch 2x: {buf.frame_count} -> {stretched.frame_count} frames")
print(f"  Preserves individual pitch cycles")

# Compress
compressed = pycdp.psow_stretch(buf, stretch_factor=0.5)
print(f"PSOW compress 0.5x: {compressed.frame_count} frames")

# Extreme stretch for drone-like effect
drone = pycdp.psow_stretch(buf, stretch_factor=4.0)
print(f"PSOW extreme stretch 4x: {drone.frame_count} frames")

# Stretch with grain grouping (keeps multiple cycles together)
smooth_stretch = pycdp.psow_stretch(buf, stretch_factor=2.0, grain_count=4)
print(f"PSOW stretch 2x with grain_count=4: smoother, preserves local coherence")

# =============================================================================
# PSOW Duplication - Rhythmic time-stretching
# =============================================================================

print("\n=== PSOW Duplication ===")

# Duplicate each grain (creates stuttering stretch)
doubled = pycdp.psow_dupl(buf, repeat_count=2)
print(f"PSOW duplicate 2x: {doubled.frame_count} frames")
print(f"  Each pitch cycle repeated twice")

# More repetitions
stuttered = pycdp.psow_dupl(buf, repeat_count=4)
print(f"PSOW duplicate 4x: {stuttered.frame_count} frames")

# Duplicate groups of grains
grouped = pycdp.psow_dupl(buf, repeat_count=2, grain_count=3)
print(f"PSOW duplicate: 3 grains repeated 2x each")

# =============================================================================
# PSOW Grab - Extract pitch-synchronous grains
# =============================================================================

print("\n=== PSOW Grab ===")

# Grab a single grain (pitch period) at a specific time
grain = pycdp.psow_grab(buf, time=0.1)
print(f"Grabbed single grain at 0.1s: {grain.frame_count} samples")

# Grab and extend into sustained sound
sustained = pycdp.psow_grab(buf, time=0.15, duration=1.0, density=1.0)
print(f"Grabbed and sustained for 1.0s: {sustained.frame_count} frames")

# Grab with higher density (pitch shift up)
high_density = pycdp.psow_grab(buf, time=0.15, duration=0.5, density=2.0)
print(f"Grabbed with density 2.0 (octave up): {high_density.frame_count} frames")

# Grab multiple grains
multi_grain = pycdp.psow_grab(buf, time=0.1, grain_count=4, duration=0.5)
print(f"Grabbed 4 consecutive grains, sustained 0.5s")

# =============================================================================
# PSOW Interpolation - Morph between grains
# =============================================================================

print("\n=== PSOW Interpolation ===")

# Create a second pitched sound for interpolation
buf2 = create_pitched_sound(frequency=330.0, duration=0.5)  # E4
print(f"Created second sound (E4 = 330 Hz)")

# Extract single grains from each sound
grain1 = pycdp.psow_grab(buf, time=0.1)
grain2 = pycdp.psow_grab(buf2, time=0.1)
print(f"Extracted grains: {grain1.frame_count} and {grain2.frame_count} samples")

# Interpolate between grains
interp = pycdp.psow_interp(grain1, grain2, start_dur=0.2, interp_dur=0.5, end_dur=0.2)
print(f"Interpolated grains: {interp.frame_count} frames")
print(f"  Morphs from first grain to second over 0.5s")

# =============================================================================
# FOF Extraction and Synthesis
# =============================================================================

print("\n=== FOF (Formant) Processing ===")

# Extract FOF (formant) data at a specific time
fof = pycdp.fofex_extract(buf, time=0.1)
print(f"Extracted FOF at 0.1s: {fof.frame_count} samples")

# Extract multiple FOFs
multi_fof = pycdp.fofex_extract(buf, time=0.1, fof_count=4)
print(f"Extracted 4 FOFs: {multi_fof.frame_count} samples")

# Extract all FOFs from the sound
all_fofs = pycdp.fofex_extract_all(buf)
print(f"Extracted all FOFs")

# Synthesize from FOF data
synth = pycdp.fofex_synth(fof, duration=0.5, frequency=440.0)
print(f"Synthesized from FOF at 440 Hz: {synth.frame_count} frames")

# Synthesize at different pitch
synth_low = pycdp.fofex_synth(fof, duration=0.5, frequency=110.0)
print(f"Synthesized from FOF at 110 Hz: {synth_low.frame_count} frames")

# Repitch using FOF (preserves formants)
repitched = pycdp.fofex_repitch(buf, pitch_shift=5.0, preserve_formants=True)
print(f"FOF repitch +5 semitones (formants preserved)")

# Repitch without formant preservation for comparison
repitched_no_formant = pycdp.fofex_repitch(buf, pitch_shift=5.0, preserve_formants=False)
print(f"FOF repitch +5 semitones (formants shifted)")

# =============================================================================
# Hover - Zigzag Reading Effect
# =============================================================================

print("\n=== Hover (Zigzag Reading) ===")

# Hover creates a pitch-hovering effect by reading back and forth
hovered = pycdp.hover(buf,
    frequency=10.0,    # Oscillation rate in Hz
    location=0.5,      # Center position (0-1)
    splice_ms=2.0      # Splice length
)
print(f"Hover effect: 10 Hz oscillation at center")
print(f"  Output: {hovered.frame_count} frames")

# Faster hover for tremolo-like effect
fast_hover = pycdp.hover(buf,
    frequency=50.0,
    location=0.3,
    frq_rand=0.2       # Add randomness
)
print(f"Fast hover: 50 Hz with randomness")

# Slow hover for pitch exploration
slow_hover = pycdp.hover(buf,
    frequency=2.0,
    location=0.5,
    loc_rand=0.3,
    duration=2.0       # Extend output
)
print(f"Slow hover: 2 Hz exploring the sound")
print(f"  Output: {slow_hover.frame_count} frames")

# =============================================================================
# Combined Pitch-Synchronous Processing
# =============================================================================

print("\n=== Combined PSOW Chain ===")

result = buf

# Step 1: Stretch
result = pycdp.psow_stretch(result, stretch_factor=1.5)
print(f"Step 1 - PSOW stretch 1.5x: {result.frame_count} frames")

# Step 2: Apply hover for subtle pitch variation
result = pycdp.hover(result, frequency=5.0, frq_rand=0.2, loc_rand=0.1)
print(f"Step 2 - Add hover for pitch variation")

# Step 3: Pitch shift using FOF
result = pycdp.fofex_repitch(result, pitch_shift=-5.0, preserve_formants=True)
print(f"Step 3 - FOF repitch down 5 semitones")

print(f"Final result: {result.frame_count} frames")

# =============================================================================
# Comparison: Different Pitch Methods
# =============================================================================

print("\n=== Comparison: Different Pitch Shift Methods ===")

# Standard pitch shift (phase vocoder)
pv_shifted = pycdp.pitch_shift(buf, semitones=7.0)
print(f"Pitch shift (phase vocoder) +7: shifts all frequencies equally")

# FOF repitch (formant preservation)
fof_shifted = pycdp.fofex_repitch(buf, pitch_shift=7.0, preserve_formants=True)
print(f"FOF repitch +7 (preserve): maintains formant positions")

# FOF repitch (no preservation)
fof_shifted_np = pycdp.fofex_repitch(buf, pitch_shift=7.0, preserve_formants=False)
print(f"FOF repitch +7 (no preserve): shifts formants with pitch")

# Brassage pitch (granular)
gran_shifted = pycdp.brassage(buf, velocity=1.0, pitch_shift=7.0, grainsize_ms=40.0)
print(f"Brassage pitch +7 (granular): grain-based shifting")

print("\nEach method has different characteristics:")
print("- Phase vocoder: Works on any sound, may sound metallic")
print("- FOF (preserve): Excellent for voice, maintains timbre")
print("- FOF (no preserve): Creates more alien transformations")
print("- Granular: Can add texture, good for textures")

# =============================================================================
# Write Output Files
# =============================================================================

print("\n=== Writing Output Files ===")

outputs = {
    "05_source.wav": pycdp.normalize(buf),
    "05_psow_stretch_2x.wav": pycdp.normalize(stretched),
    "05_psow_stretch_4x.wav": pycdp.normalize(drone),
    "05_psow_dupl.wav": pycdp.normalize(doubled),
    "05_psow_grab_sustained.wav": pycdp.normalize(sustained),
    "05_psow_interp.wav": pycdp.normalize(interp),
    "05_fof_synth_440.wav": pycdp.normalize(synth),
    "05_fof_synth_110.wav": pycdp.normalize(synth_low),
    "05_fof_repitch.wav": pycdp.normalize(repitched),
    "05_hover.wav": pycdp.normalize(hovered),
    "05_hover_slow.wav": pycdp.normalize(slow_hover),
    "05_pitch_pv.wav": pycdp.normalize(pv_shifted),
    "05_pitch_fof.wav": pycdp.normalize(fof_shifted),
    "05_pitch_granular.wav": pycdp.normalize(gran_shifted),
}

for filename, buffer in outputs.items():
    path = os.path.join(OUTPUT_DIR, filename)
    pycdp.write_file(path, buffer)
    print(f"  Wrote: {filename}")

print("\n=== Demo Complete ===")
