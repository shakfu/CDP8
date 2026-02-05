#!/usr/bin/env python3
"""
Demo 7: Spectral Morphing

This demo shows CDP's spectral morphing capabilities:
- Basic morph (interpolate between two sounds)
- Morph glide (smooth spectral transition)
- Cross-synthesis (combine characteristics of two sounds)
- Native morph variants with advanced control
"""

import pycdp
import math
import array
import os

# Create output directory (use PYCDP_DEMO_OUTPUT env var or default to demos/build)
OUTPUT_DIR = os.environ.get("PYCDP_DEMO_OUTPUT", "build")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_brass_like_sound(frequency=200.0, duration=0.5, sample_rate=44100):
    """Create a brass-like sound (rich harmonics, bright attack)."""
    samples = array.array('f')
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Sawtooth-like harmonics (bright, brassy)
        val = 0.0
        for h in range(1, 10):
            harmonic_amp = 1.0 / h
            val += harmonic_amp * math.sin(2 * math.pi * frequency * h * t)
        # Attack envelope (fast attack, slow decay)
        attack = min(t / 0.02, 1.0)
        sustain = max(0, 1.0 - (t - 0.3) / 0.2)
        env = attack * sustain
        samples.append(val * env * 0.3)
    return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

def create_flute_like_sound(frequency=400.0, duration=0.5, sample_rate=44100):
    """Create a flute-like sound (mostly fundamental, soft)."""
    samples = array.array('f')
    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Mostly fundamental with slight 2nd harmonic (flute-like)
        val = 0.8 * math.sin(2 * math.pi * frequency * t)
        val += 0.15 * math.sin(2 * math.pi * frequency * 2 * t)
        val += 0.05 * math.sin(2 * math.pi * frequency * 3 * t)
        # Add subtle breath noise
        noise = ((hash(i) % 1000) / 1000.0 - 0.5) * 0.02
        # Soft envelope
        attack = min(t / 0.08, 1.0)
        sustain = max(0, 1.0 - (t - 0.35) / 0.15)
        env = attack * sustain
        samples.append((val + noise) * env * 0.5)
    return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

def create_vowel_sound(vowel='a', frequency=150.0, duration=0.5, sample_rate=44100):
    """Create a vowel-like sound with formants."""
    samples = array.array('f')

    # Formant frequencies for different vowels (approximate)
    formants = {
        'a': [(800, 1.0), (1200, 0.5), (2500, 0.3)],   # "ah"
        'i': [(300, 1.0), (2300, 0.6), (3000, 0.3)],   # "ee"
        'o': [(500, 1.0), (1000, 0.5), (2300, 0.2)],   # "oh"
        'u': [(350, 1.0), (700, 0.4), (2500, 0.1)],    # "oo"
    }

    formant_list = formants.get(vowel, formants['a'])

    for i in range(int(sample_rate * duration)):
        t = i / sample_rate
        # Glottal pulse (fundamental with harmonics)
        val = 0.0
        for h in range(1, 15):
            harmonic_amp = 1.0 / (h * h)  # Roll-off
            harmonic_freq = frequency * h
            # Apply formant filtering
            formant_weight = 0.0
            for formant_freq, formant_amp in formant_list:
                # Resonance around formant frequency
                diff = abs(harmonic_freq - formant_freq)
                bandwidth = 100.0
                resonance = math.exp(-(diff * diff) / (2 * bandwidth * bandwidth))
                formant_weight = max(formant_weight, resonance * formant_amp)
            val += harmonic_amp * formant_weight * math.sin(2 * math.pi * harmonic_freq * t)
        # Envelope
        attack = min(t / 0.05, 1.0)
        sustain = max(0, 1.0 - (t - 0.35) / 0.15)
        env = attack * sustain
        samples.append(val * env * 0.6)
    return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

print("=== Spectral Morphing Demo ===\n")

# Create test sounds
brass = create_brass_like_sound(frequency=200.0)
flute = create_flute_like_sound(frequency=400.0)
vowel_a = create_vowel_sound(vowel='a', frequency=150.0)
vowel_i = create_vowel_sound(vowel='i', frequency=150.0)

print(f"Created test sounds:")
print(f"  Brass-like: {brass.frame_count} frames at 200 Hz")
print(f"  Flute-like: {flute.frame_count} frames at 400 Hz")
print(f"  Vowel 'a':  {vowel_a.frame_count} frames at 150 Hz")
print(f"  Vowel 'i':  {vowel_i.frame_count} frames at 150 Hz")

# =============================================================================
# Basic Morph - Interpolate between two sounds
# =============================================================================

print("\n=== Basic Morph ===")

# Simple morph: brass to flute
morphed = pycdp.morph(brass, flute, morph_start=0.0, morph_end=1.0)
print(f"Morph brass->flute: {morphed.frame_count} frames")
print(f"  Starts as brass, ends as flute")

# Partial morph: only morph in middle section
partial = pycdp.morph(brass, flute, morph_start=0.3, morph_end=0.7)
print(f"Partial morph (30%-70%): stays brass, morphs, stays flute")

# Fast morph with exponent
fast_morph = pycdp.morph(brass, flute, morph_start=0.0, morph_end=1.0, exponent=0.5)
print(f"Fast-start morph (exponent=0.5): quick transition then slow")

# Slow morph with exponent
slow_morph = pycdp.morph(brass, flute, morph_start=0.0, morph_end=1.0, exponent=2.0)
print(f"Slow-start morph (exponent=2.0): slow transition then quick")

# =============================================================================
# Morph Glide - Simple spectral transition
# =============================================================================

print("\n=== Morph Glide ===")

# Short glide
short_glide = pycdp.morph_glide(brass, flute, duration=0.5)
print(f"Short glide (0.5s): {short_glide.frame_count} frames")

# Long glide
long_glide = pycdp.morph_glide(brass, flute, duration=2.0)
print(f"Long glide (2.0s): {long_glide.frame_count} frames")

# Vowel glide: "ah" to "ee"
vowel_glide = pycdp.morph_glide(vowel_a, vowel_i, duration=1.0)
print(f"Vowel glide 'a'->'i': {vowel_glide.frame_count} frames")

# =============================================================================
# Cross-Synthesis - Combine characteristics
# =============================================================================

print("\n=== Cross-Synthesis ===")

# Amplitude from brass, frequencies from flute
cross1 = pycdp.cross_synth(brass, flute, mode=0, mix=1.0)
print(f"Cross-synth (brass amp + flute freq): {cross1.frame_count} frames")
print(f"  Creates a bright sound with flute pitch contour")

# Amplitude from flute, frequencies from brass
cross2 = pycdp.cross_synth(brass, flute, mode=1, mix=1.0)
print(f"Cross-synth (flute amp + brass freq): {cross2.frame_count} frames")
print(f"  Creates a soft sound with brass harmonics")

# Partial cross-synthesis
partial_cross = pycdp.cross_synth(brass, flute, mode=0, mix=0.5)
print(f"Partial cross-synth (50% mix): blend of original and cross")

# =============================================================================
# Native Morph Functions (Original CDP algorithms)
# =============================================================================

print("\n=== Native Morph Functions ===")
print("(These use the original CDP algorithms with more control)")

# Native glide with explicit parameters
native_glide = pycdp.morph_glide_native(brass, flute, duration=1.0, fft_size=1024)
print(f"Native glide: {native_glide.frame_count} frames")
print(f"  Uses original CDP SPECGLIDE algorithm")

# Native bridge - crossfade with normalization control
print("\n--- Native Bridge (SPECBRIDGE) ---")

bridge_no_norm = pycdp.morph_bridge_native(brass, flute, mode=0)
print(f"Bridge mode 0 (no normalization): {bridge_no_norm.frame_count} frames")

bridge_norm_min = pycdp.morph_bridge_native(brass, flute, mode=1)
print(f"Bridge mode 1 (normalize to minimum)")

bridge_norm_file1 = pycdp.morph_bridge_native(brass, flute, mode=2)
print(f"Bridge mode 2 (normalize to file 1 level)")

# Bridge with offset (start file2 later)
bridge_offset = pycdp.morph_bridge_native(brass, flute, mode=0, offset=0.1)
print(f"Bridge with 0.1s offset: file2 starts later")

# Bridge with custom interpolation range
bridge_custom = pycdp.morph_bridge_native(brass, flute, mode=0,
                                          interp_start=0.2, interp_end=0.8)
print(f"Bridge with custom interp (0.2-0.8): morph in middle only")

# Native morph with full control
print("\n--- Native Morph (SPECMORPH) ---")

# Linear morph
linear_morph = pycdp.morph_native(brass, flute, mode=0)
print(f"Native morph linear (mode=0): {linear_morph.frame_count} frames")

# Cosine morph (smoother)
cosine_morph = pycdp.morph_native(brass, flute, mode=1)
print(f"Native morph cosine (mode=1): smoother transitions")

# Separate amplitude and frequency morphing
# Morph amplitude quickly, frequency slowly
amp_fast = pycdp.morph_native(brass, flute, mode=0,
                              amp_start=0.0, amp_end=0.3,
                              freq_start=0.0, freq_end=1.0)
print(f"Amp morphs quickly (0-30%), freq morphs over whole duration")

# Morph frequency first, then amplitude
freq_first = pycdp.morph_native(brass, flute, mode=0,
                                amp_start=0.5, amp_end=1.0,
                                freq_start=0.0, freq_end=0.5)
print(f"Freq morphs first (0-50%), then amp morphs (50-100%)")

# With exponents for curved interpolation
curved_morph = pycdp.morph_native(brass, flute, mode=0,
                                  amp_exp=2.0, freq_exp=0.5)
print(f"Curved morph: amp slow-start (exp=2), freq fast-start (exp=0.5)")

# =============================================================================
# Creative Morphing Techniques
# =============================================================================

print("\n=== Creative Morphing Techniques ===")

# Vowel morphing chain: a -> i -> o
print("\n--- Vowel Transformation ---")
vowel_o = create_vowel_sound(vowel='o', frequency=150.0)

morph_a_i = pycdp.morph(vowel_a, vowel_i)
morph_i_o = pycdp.morph(vowel_i, vowel_o)
vowel_chain = pycdp.concat([morph_a_i, morph_i_o])
print(f"Vowel chain 'a'->'i'->'o': {vowel_chain.frame_count} frames")

# Hybrid sound: cross-synth then morph
print("\n--- Hybrid Sound Creation ---")
hybrid = pycdp.cross_synth(brass, flute, mode=0, mix=0.7)
hybrid_morphed = pycdp.morph(hybrid, vowel_a, morph_start=0.2, morph_end=0.8)
print(f"Hybrid (brass+flute) morphed to vowel: {hybrid_morphed.frame_count} frames")

# Self-morphing with pitch shift
print("\n--- Self-Morphing ---")
brass_shifted = pycdp.pitch_shift(brass, semitones=7.0)  # Up a fifth
self_morph = pycdp.morph(brass, brass_shifted)
print(f"Self-morph (original to pitched): creates evolving timbre")

# =============================================================================
# Tips for Morphing
# =============================================================================

print("\n=== Tips for Effective Morphing ===")

tips = """
1. MATCHING DURATIONS:
   - Morph works best when inputs have similar durations
   - Use time_stretch to match lengths if needed:
     buf2_stretched = pycdp.time_stretch(buf2, factor=len(buf1)/len(buf2))

2. MATCHING PITCHES:
   - Morphing between different pitches can sound unnatural
   - Use pitch_shift to match fundamental frequencies first

3. FFT SIZE TRADE-OFFS:
   - Larger FFT (2048, 4096): Better frequency resolution, smeared transients
   - Smaller FFT (512, 1024): Better transient preservation, less frequency detail
   - For tonal sounds: use larger FFT
   - For percussive sounds: use smaller FFT

4. MORPHING PERCUSSIVE SOUNDS:
   - Cross-synthesis often works better than morph for percussion
   - Amplitude envelope drives the rhythm

5. CREATIVE APPLICATIONS:
   - Voice morphing: use morph_native with separate amp/freq control
   - Instrument hybrids: use cross_synth for "talking instruments"
   - Evolving textures: chain multiple morphs together

6. NORMALIZATION MODES (bridge_native):
   - Mode 0: No normalization (preserve original levels)
   - Mode 1: Normalize to quieter file (smooth transition)
   - Mode 2/3: Match specific file level
   - Mode 4/5: Progressive normalization (smooth level change)

7. COMPARISON OF MORPH FUNCTIONS:
   - morph(): Simple, good default
   - morph_glide(): Fixed duration output, smooth glide
   - cross_synth(): Combine characteristics without time morphing
   - morph_native(): Full control over timing and curves
   - morph_bridge_native(): Crossfade with offset and normalization
"""

print(tips)

# =============================================================================
# Write Output Files
# =============================================================================

print("=== Writing Output Files ===")

outputs = {
    "07_brass.wav": pycdp.normalize(brass),
    "07_flute.wav": pycdp.normalize(flute),
    "07_vowel_a.wav": pycdp.normalize(vowel_a),
    "07_vowel_i.wav": pycdp.normalize(vowel_i),
    "07_morph_brass_flute.wav": pycdp.normalize(morphed),
    "07_morph_glide_long.wav": pycdp.normalize(long_glide),
    "07_vowel_glide.wav": pycdp.normalize(vowel_glide),
    "07_cross_synth.wav": pycdp.normalize(cross1),
    "07_native_glide.wav": pycdp.normalize(native_glide),
    "07_native_bridge.wav": pycdp.normalize(bridge_no_norm),
    "07_native_morph_linear.wav": pycdp.normalize(linear_morph),
    "07_native_morph_cosine.wav": pycdp.normalize(cosine_morph),
    "07_vowel_chain.wav": pycdp.normalize(vowel_chain),
    "07_hybrid_morphed.wav": pycdp.normalize(hybrid_morphed),
}

for filename, buffer in outputs.items():
    path = os.path.join(OUTPUT_DIR, filename)
    pycdp.write_file(path, buffer)
    print(f"  Wrote: {filename}")

print("\n=== Demo Complete ===")
