#!/usr/bin/env python3
"""
Demo 6: Creative Techniques and Effect Chains

This demo shows creative sound design techniques:
- Building effect chains
- Creating textures and atmospheres
- Sound transformation recipes
- Tips for real-world usage
"""

import cycdp
import math
import array
import os

# Create output directory (use PYCDP_DEMO_OUTPUT env var or default to demos/build)
OUTPUT_DIR = os.environ.get("PYCDP_DEMO_OUTPUT", "build")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_percussive_sound(sample_rate=44100):
    """Create a percussive impact sound."""
    samples = array.array('f')
    duration = 0.3
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Noise burst with pitch
        noise = (hash(i) % 1000) / 1000.0 - 0.5
        tone = math.sin(2 * math.pi * 100 * t * (1.0 - t * 2))
        # Fast decay envelope
        env = math.exp(-t * 20)
        samples.append((noise * 0.3 + tone * 0.7) * env * 0.8)
    return cycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

def create_pad_sound(sample_rate=44100):
    """Create a sustained pad-like sound."""
    samples = array.array('f')
    duration = 1.0
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Multiple detuned oscillators
        val = 0.0
        for detune in [-0.02, -0.01, 0, 0.01, 0.02]:
            freq = 200 * (1.0 + detune)
            val += 0.2 * math.sin(2 * math.pi * freq * t)
            val += 0.1 * math.sin(2 * math.pi * freq * 2 * t)
        # Slow envelope
        env = min(t / 0.2, 1.0) * min(1.0, (duration - t) / 0.3)
        samples.append(val * env)
    return cycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

print("=== Creative Techniques Demo ===\n")

# =============================================================================
# Recipe 1: Ambient Pad from Percussive Sound
# =============================================================================

print("=== Recipe 1: Ambient Pad from Percussion ===")

perc = create_percussive_sound()
print(f"Starting with percussive sound: {perc.frame_count} frames")

# Step 1: Extreme time stretch
step1 = cycdp.time_stretch(perc, factor=8.0)
print(f"  1. Time stretch 8x: {step1.frame_count} frames")

# Step 2: Blur to smooth out artifacts
step2 = cycdp.spectral_blur(step1, blur_time=0.1)
print(f"  2. Spectral blur 0.1s")

# Step 3: Add reverb for space
step3 = cycdp.reverb(step2, decay_time=3.0, damping=0.4, mix=0.5)
print(f"  3. Large reverb")

# Step 4: Fade in/out for smooth loop
cycdp.fade_in(step3, duration=0.5, curve="exponential")
cycdp.fade_out(step3, duration=1.0, curve="exponential")
ambient_pad = step3
print(f"  4. Exponential fades")
print(f"  Result: {ambient_pad.frame_count} frames of ambient texture")

# =============================================================================
# Recipe 2: Rhythmic Texture from Sustained Sound
# =============================================================================

print("\n=== Recipe 2: Rhythmic Texture from Pad ===")

pad = create_pad_sound()
print(f"Starting with pad sound: {pad.frame_count} frames")

# Step 1: Granular processing with scatter
step1 = cycdp.brassage(pad,
    velocity=0.5,
    density=0.8,       # Gaps between grains
    grainsize_ms=30.0,
    scatter=0.6
)
print(f"  1. Brassage with scatter: {step1.frame_count} frames")

# Step 2: Add rhythmic tremolo
step2 = cycdp.tremolo(step1, freq=4.0, depth=0.8)
print(f"  2. Rhythmic tremolo at 4 Hz")

# Step 3: Stereo spread with flutter
rhythmic = cycdp.flutter(step2, frequency=2.0, depth=0.5)
print(f"  3. Stereo flutter")
print(f"  Result: {rhythmic.frame_count} frames, {rhythmic.channels} channels")

# =============================================================================
# Recipe 3: Vocal-like Texture from Simple Tone
# =============================================================================

print("\n=== Recipe 3: Vocal Texture from Tone ===")

# Create simple tone
tone_samples = array.array('f')
for i in range(22050):  # 0.5 seconds
    t = i / 44100
    tone_samples.append(0.5 * math.sin(2 * math.pi * 150 * t))
tone = cycdp.Buffer.from_memoryview(tone_samples, channels=1, sample_rate=44100)
print(f"Starting with simple 150 Hz tone")

# Step 1: Add formant-like filtering - use bandpass to create formant bands
step1 = cycdp.filter_bandpass(tone, low_freq=400.0, high_freq=600.0)
step1b = cycdp.filter_bandpass(tone, low_freq=1400.0, high_freq=1600.0)
step1c = cycdp.filter_bandpass(tone, low_freq=2400.0, high_freq=2600.0)
step1 = cycdp.mix([step1, step1b, step1c], gains=[0.5, 0.3, 0.2])
print(f"  1. Formant filtering (500, 1500, 2500 Hz regions)")

# Step 2: Add pitch variation using hover
step2 = cycdp.hover(step1, frequency=5.0, frq_rand=0.3, loc_rand=0.2)
print(f"  2. Hover for pitch variation")

# Step 3: Time stretch for evolution
vocal_texture = cycdp.time_stretch(step2, factor=2.0)
print(f"  3. Time stretch 2x")
print(f"  Result: {vocal_texture.frame_count} frames")

# =============================================================================
# Recipe 4: Glitch/Stutter Effect
# =============================================================================

print("\n=== Recipe 4: Glitch/Stutter Effect ===")

source = create_percussive_sound()
print(f"Starting with percussive sound")

# Step 1: Grain shuffle for randomness
step1 = cycdp.grain_reorder(source)  # Random shuffle
print(f"  1. Shuffle grains")

# Step 2: Splinter for stuttering
step2 = cycdp.splinter(step1, start=0.0, duration_ms=50.0, repeats=8, min_shrink=0.3)
print(f"  2. Splinter effect")

# Step 3: Distortion for edge
glitched = cycdp.distort_repeat(step2, multiplier=2)
print(f"  3. Waveset distortion")
print(f"  Result: {glitched.frame_count} frames")

# =============================================================================
# Recipe 5: Ethereal Space
# =============================================================================

print("\n=== Recipe 5: Ethereal Space ===")

pad = create_pad_sound()
print(f"Starting with pad sound")

# Step 1: Pitch shift up
step1 = cycdp.pitch_shift(pad, semitones=12.0)
print(f"  1. Shift up octave")

# Step 2: Wrappage for spatial texture
step2 = cycdp.wrappage(step1,
    grain_size=80.0,
    density=3.0,
    velocity=0.3,
    spread=1.0,
    jitter=0.4
)
print(f"  2. Wrappage texture: {step2.frame_count} frames")

# Step 3: Large reverb
step3 = cycdp.reverb(step2, decay_time=4.0, damping=0.2, mix=0.6)
print(f"  3. Large hall reverb")

# Step 4: Spectral blur for softness
ethereal = cycdp.spectral_blur(step3, blur_time=0.08)
print(f"  4. Spectral blur")
print(f"  Result: {ethereal.frame_count} frames, {ethereal.channels} channels")

# =============================================================================
# Tips for Real-World Usage
# =============================================================================

print("\n=== Tips for Real-World Usage ===")

tips = """
1. FILE I/O:
   buf = cycdp.read_file("input.wav")
   cycdp.write_file("output.wav", result)

2. NORMALIZE BEFORE OUTPUT:
   result = cycdp.normalize(result, target=0.9)
   # Or in dB:
   result = cycdp.normalize_db(result, target_db=-1.0)

3. CHECK PEAK LEVELS:
   peak_val, peak_pos = cycdp.peak(buf)
   print(f"Peak level: {peak_val} at sample {peak_pos}")

4. MONO/STEREO CONVERSION:
   # Mono to stereo (centered):
   stereo = cycdp.pan(mono_buf, 0.0)

   # Stereo to mono:
   mono = cycdp.to_mono(stereo_buf)

   # Reduce stereo width:
   narrowed = cycdp.narrow(stereo_buf, 0.5)

5. PROCESSING STEREO FILES:
   Many CDP effects require mono input. To process stereo:
   - Use to_mono() first, or
   - Split, process, recombine:
     left, right = cycdp.split_channels(stereo_buf)
     left_proc = cycdp.some_effect(left)
     right_proc = cycdp.some_effect(right)
     result = cycdp.merge_channels(left_proc, right_proc)

6. AVOID CLIPPING:
   # Always normalize or reduce gain before writing
   peak_val, _ = cycdp.peak(result)
   if peak_val > 1.0:
       result = cycdp.normalize(result, target=0.95)

7. EFFECT ORDER MATTERS:
   # Generally: time-based -> spectral -> dynamics -> spatial
   # Example: time_stretch -> spectral_blur -> compressor -> reverb

8. GRANULAR PARAMETER TIPS:
   - Small grains (10-30ms): rhythmic, choppy
   - Medium grains (40-80ms): smooth time-stretch
   - Large grains (100ms+): ambient, sustaining
   - Density <1: gaps, rhythmic
   - Density >1: overlap, smooth/dense

9. SPECTRAL PARAMETER TIPS:
   - FFT 512-1024: better transients
   - FFT 2048-4096: better frequency resolution
   - Higher blur = softer, dreamier
   - Use pitch_shift for musical intervals (semitones)
   - Use spectral_shift for inharmonic effects (Hz)
"""

print(tips)

# =============================================================================
# Write Output Files
# =============================================================================

print("=== Writing Output Files ===")

outputs = {
    "06_ambient_pad.wav": cycdp.normalize(ambient_pad),
    "06_rhythmic_texture.wav": cycdp.normalize(rhythmic),
    "06_vocal_texture.wav": cycdp.normalize(vocal_texture),
    "06_glitch.wav": cycdp.normalize(glitched),
    "06_ethereal.wav": cycdp.normalize(ethereal),
}

for filename, buffer in outputs.items():
    path = os.path.join(OUTPUT_DIR, filename)
    cycdp.write_file(path, buffer)
    print(f"  Wrote: {filename}")

print("\n=== Demo Complete ===")
