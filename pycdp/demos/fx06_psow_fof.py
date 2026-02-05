#!/usr/bin/env python3
"""
FX Demo 6: Pitch-Synchronous and FOF Processing

Applies PSOW and FOF processing to an input audio file.
Best for pitched/tonal material.

Usage:
    python fx06_psow_fof.py input.wav -o output_dir/
    python fx06_psow_fof.py input.wav  # outputs to ./build/
"""

import argparse
import pycdp
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Apply pitch-synchronous and FOF effects to an audio file."
    )
    parser.add_argument("input", help="Input WAV file path")
    parser.add_argument(
        "-o", "--output",
        default="build",
        help="Output directory (default: build/)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("=== FX Demo 6: Pitch-Synchronous and FOF Processing ===\n")

    buf = pycdp.read_file(args.input)
    print(f"Loaded: {args.input}")
    print(f"  {buf.frame_count / buf.sample_rate:.2f}s, {buf.channels}ch, {buf.sample_rate}Hz")

    duration = buf.frame_count / buf.sample_rate
    outputs = {}

    # --- PSOW Stretch ---
    print("\n--- PSOW Stretch ---")

    psow_stretch_2x = pycdp.psow_stretch(buf, stretch_factor=2.0)
    print(f"PSOW stretch 2x")
    outputs["fx06_psow_stretch_2x.wav"] = psow_stretch_2x

    psow_stretch_3x = pycdp.psow_stretch(buf, stretch_factor=3.0)
    print(f"PSOW stretch 3x")
    outputs["fx06_psow_stretch_3x.wav"] = psow_stretch_3x

    psow_compress = pycdp.psow_stretch(buf, stretch_factor=0.5)
    print(f"PSOW compress 0.5x")
    outputs["fx06_psow_compress.wav"] = psow_compress

    # With grain grouping for smoother result
    psow_smooth = pycdp.psow_stretch(buf, stretch_factor=2.0, grain_count=4)
    print(f"PSOW stretch 2x (smooth, grain_count=4)")
    outputs["fx06_psow_stretch_smooth.wav"] = psow_smooth

    # --- PSOW Duplication ---
    print("\n--- PSOW Duplication ---")

    psow_dupl_2 = pycdp.psow_dupl(buf, repeat_count=2)
    print(f"PSOW duplicate 2x (stutter stretch)")
    outputs["fx06_psow_dupl_2x.wav"] = psow_dupl_2

    psow_dupl_4 = pycdp.psow_dupl(buf, repeat_count=4)
    print(f"PSOW duplicate 4x")
    outputs["fx06_psow_dupl_4x.wav"] = psow_dupl_4

    # --- PSOW Grab (extract and sustain) ---
    print("\n--- PSOW Grab ---")

    # Grab from different parts of the sound
    grab_early = pycdp.psow_grab(buf, time=min(1.0, duration * 0.1), duration=4.0, density=1.0)
    print(f"Grab early, sustain 4s")
    outputs["fx06_psow_grab_early.wav"] = grab_early

    grab_mid = pycdp.psow_grab(buf, time=duration / 2, duration=4.0, density=1.0)
    print(f"Grab middle, sustain 4s")
    outputs["fx06_psow_grab_mid.wav"] = grab_mid

    # Grab with pitch change via density
    grab_high = pycdp.psow_grab(buf, time=duration * 0.25, duration=4.0, density=2.0)
    print(f"Grab with density 2.0 (octave up)")
    outputs["fx06_psow_grab_octave_up.wav"] = grab_high

    grab_low = pycdp.psow_grab(buf, time=duration * 0.25, duration=4.0, density=0.5)
    print(f"Grab with density 0.5 (octave down)")
    outputs["fx06_psow_grab_octave_down.wav"] = grab_low

    # --- FOF Repitch ---
    print("\n--- FOF Repitch ---")

    fof_up_preserve = pycdp.fofex_repitch(buf, pitch_shift=7.0, preserve_formants=True)
    print(f"FOF repitch +7 (formants preserved)")
    outputs["fx06_fof_up_preserve.wav"] = fof_up_preserve

    fof_up_shift = pycdp.fofex_repitch(buf, pitch_shift=7.0, preserve_formants=False)
    print(f"FOF repitch +7 (formants shifted)")
    outputs["fx06_fof_up_shift.wav"] = fof_up_shift

    fof_down_preserve = pycdp.fofex_repitch(buf, pitch_shift=-7.0, preserve_formants=True)
    print(f"FOF repitch -7 (formants preserved)")
    outputs["fx06_fof_down_preserve.wav"] = fof_down_preserve

    fof_down_shift = pycdp.fofex_repitch(buf, pitch_shift=-7.0, preserve_formants=False)
    print(f"FOF repitch -7 (formants shifted)")
    outputs["fx06_fof_down_shift.wav"] = fof_down_shift

    # Extreme pitch shifts
    fof_extreme_up = pycdp.fofex_repitch(buf, pitch_shift=12.0, preserve_formants=True)
    print(f"FOF repitch +12 (octave up, formants preserved)")
    outputs["fx06_fof_octave_up.wav"] = fof_extreme_up

    fof_extreme_down = pycdp.fofex_repitch(buf, pitch_shift=-12.0, preserve_formants=True)
    print(f"FOF repitch -12 (octave down, formants preserved)")
    outputs["fx06_fof_octave_down.wav"] = fof_extreme_down

    # --- Hover ---
    print("\n--- Hover (Zigzag Reading) ---")

    hover_slow = pycdp.hover(buf, frequency=3.0, location=0.5, frq_rand=0.2, loc_rand=0.1)
    print(f"Slow hover (3 Hz)")
    outputs["fx06_hover_slow.wav"] = hover_slow

    hover_fast = pycdp.hover(buf, frequency=20.0, location=0.5, frq_rand=0.3)
    print(f"Fast hover (20 Hz, tremolo-like)")
    outputs["fx06_hover_fast.wav"] = hover_fast

    hover_explore = pycdp.hover(buf, frequency=1.0, location=0.3, loc_rand=0.5,
                                duration=min(20.0, duration * 2))
    print(f"Exploring hover (slow, random location)")
    outputs["fx06_hover_explore.wav"] = hover_explore

    # --- Comparison: Different pitch methods ---
    print("\n--- Pitch Method Comparison ---")

    # Phase vocoder pitch
    pv_pitch = pycdp.pitch_shift(buf, semitones=5.0)
    print(f"Phase vocoder +5")
    outputs["fx06_compare_pv.wav"] = pv_pitch

    # FOF pitch (preserve formants)
    fof_pitch = pycdp.fofex_repitch(buf, pitch_shift=5.0, preserve_formants=True)
    print(f"FOF +5 (preserve formants)")
    outputs["fx06_compare_fof_preserve.wav"] = fof_pitch

    # Granular pitch
    gran_pitch = pycdp.brassage(buf, velocity=1.0, pitch_shift=5.0, grainsize_ms=40.0)
    print(f"Granular +5")
    outputs["fx06_compare_granular.wav"] = gran_pitch

    # --- Write outputs ---
    print(f"\n=== Writing to {args.output}/ ===")
    for filename, buffer in outputs.items():
        path = os.path.join(args.output, filename)
        pycdp.write_file(path, pycdp.normalize(buffer, target=0.95))
        print(f"  {filename}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
