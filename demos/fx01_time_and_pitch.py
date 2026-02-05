#!/usr/bin/env python3
"""
FX Demo 1: Time and Pitch Processing

Applies time stretching and pitch shifting to an input audio file.

Usage:
    python fx01_time_and_pitch.py input.wav -o output_dir/
    python fx01_time_and_pitch.py input.wav  # outputs to ./build/
"""

import argparse
import cycdp
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Apply time and pitch processing effects to an audio file."
    )
    parser.add_argument("input", help="Input WAV file path")
    parser.add_argument(
        "-o", "--output",
        default="build",
        help="Output directory (default: build/)"
    )
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=== FX Demo 1: Time and Pitch Processing ===\n")

    # Load source
    buf = cycdp.read_file(args.input)
    print(f"Loaded: {args.input}")
    print(f"  {buf.frame_count / buf.sample_rate:.2f}s, {buf.channels}ch, {buf.sample_rate}Hz")

    outputs = {}

    # --- Time Stretching (Phase Vocoder) ---
    print("\n--- Time Stretching (Phase Vocoder) ---")

    stretched_2x = cycdp.time_stretch(buf, factor=2.0)
    print(f"2x stretch: {stretched_2x.frame_count / buf.sample_rate:.2f}s")
    outputs["fx01_stretch_2x.wav"] = stretched_2x

    stretched_half = cycdp.time_stretch(buf, factor=0.5)
    print(f"0.5x compress: {stretched_half.frame_count / buf.sample_rate:.2f}s")
    outputs["fx01_compress_half.wav"] = stretched_half

    # Extreme stretch for ambient texture
    stretched_4x = cycdp.time_stretch(buf, factor=4.0)
    print(f"4x stretch (ambient): {stretched_4x.frame_count / buf.sample_rate:.2f}s")
    outputs["fx01_stretch_4x_ambient.wav"] = stretched_4x

    # --- Pitch Shifting (Phase Vocoder) ---
    print("\n--- Pitch Shifting (Phase Vocoder) ---")

    pitch_up_5th = cycdp.pitch_shift(buf, semitones=7.0)
    print(f"Up a 5th (+7 semitones)")
    outputs["fx01_pitch_up_fifth.wav"] = pitch_up_5th

    pitch_down_octave = cycdp.pitch_shift(buf, semitones=-12.0)
    print(f"Down an octave (-12 semitones)")
    outputs["fx01_pitch_down_octave.wav"] = pitch_down_octave

    pitch_up_octave = cycdp.pitch_shift(buf, semitones=12.0)
    print(f"Up an octave (+12 semitones)")
    outputs["fx01_pitch_up_octave.wav"] = pitch_up_octave

    # Subtle detuning
    detuned = cycdp.pitch_shift(buf, semitones=-0.5)
    print(f"Detuned (-0.5 semitones)")
    outputs["fx01_detuned.wav"] = detuned

    # --- Spectral Shift (Inharmonic) ---
    print("\n--- Spectral Shift (Inharmonic) ---")

    shifted_up_100 = cycdp.spectral_shift(buf, shift_hz=100.0)
    print(f"Shift +100 Hz (metallic)")
    outputs["fx01_spectral_shift_up.wav"] = shifted_up_100

    shifted_down_50 = cycdp.spectral_shift(buf, shift_hz=-50.0)
    print(f"Shift -50 Hz (deeper)")
    outputs["fx01_spectral_shift_down.wav"] = shifted_down_50

    # --- Combined: Stretch + Pitch ---
    print("\n--- Combined Effects ---")

    # Slow and low
    slow_low = cycdp.time_stretch(buf, factor=2.0)
    slow_low = cycdp.pitch_shift(slow_low, semitones=-5.0)
    print(f"Slow + low (2x stretch, -5 semitones)")
    outputs["fx01_slow_and_low.wav"] = slow_low

    # Fast and high
    fast_high = cycdp.time_stretch(buf, factor=0.5)
    fast_high = cycdp.pitch_shift(fast_high, semitones=5.0)
    print(f"Fast + high (0.5x, +5 semitones)")
    outputs["fx01_fast_and_high.wav"] = fast_high

    # --- Write outputs ---
    print(f"\n=== Writing to {args.output}/ ===")
    for filename, buffer in outputs.items():
        path = os.path.join(args.output, filename)
        cycdp.write_file(path, cycdp.normalize(buffer, target=0.95))
        print(f"  {filename}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
