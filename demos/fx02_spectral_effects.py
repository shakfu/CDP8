#!/usr/bin/env python3
"""
FX Demo 2: Spectral Effects

Applies spectral processing - blur, freeze, fold, etc.

Usage:
    python fx02_spectral_effects.py input.wav -o output_dir/
    python fx02_spectral_effects.py input.wav  # outputs to ./build/
"""

import argparse
import cycdp
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Apply spectral effects to an audio file."
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

    print("=== FX Demo 2: Spectral Effects ===\n")

    buf = cycdp.read_file(args.input)
    print(f"Loaded: {args.input}")
    print(f"  {buf.frame_count / buf.sample_rate:.2f}s, {buf.channels}ch, {buf.sample_rate}Hz")

    outputs = {}

    # --- Spectral Blur ---
    print("\n--- Spectral Blur ---")

    blur_light = cycdp.spectral_blur(buf, blur_time=0.05)
    print(f"Light blur (50ms)")
    outputs["fx02_blur_light.wav"] = blur_light

    blur_heavy = cycdp.spectral_blur(buf, blur_time=0.2)
    print(f"Heavy blur (200ms) - dreamy")
    outputs["fx02_blur_heavy.wav"] = blur_heavy

    blur_extreme = cycdp.spectral_blur(buf, blur_time=0.5)
    print(f"Extreme blur (500ms) - wash")
    outputs["fx02_blur_extreme.wav"] = blur_extreme

    # --- Spectral Focus ---
    print("\n--- Spectral Focus ---")

    focus_low = cycdp.spectral_focus(buf, center_freq=200.0, bandwidth=100.0, gain_db=6.0)
    print(f"Focus on lows (200 Hz)")
    outputs["fx02_focus_low.wav"] = focus_low

    focus_mid = cycdp.spectral_focus(buf, center_freq=1000.0, bandwidth=300.0, gain_db=6.0)
    print(f"Focus on mids (1000 Hz)")
    outputs["fx02_focus_mid.wav"] = focus_mid

    focus_high = cycdp.spectral_focus(buf, center_freq=4000.0, bandwidth=1000.0, gain_db=6.0)
    print(f"Focus on highs (4000 Hz)")
    outputs["fx02_focus_high.wav"] = focus_high

    # --- Spectral Fold ---
    print("\n--- Spectral Fold ---")

    fold_low = cycdp.spectral_fold(buf, fold_freq=1000.0)
    print(f"Fold at 1000 Hz (metallic)")
    outputs["fx02_fold_1k.wav"] = fold_low

    fold_high = cycdp.spectral_fold(buf, fold_freq=3000.0)
    print(f"Fold at 3000 Hz (brighter)")
    outputs["fx02_fold_3k.wav"] = fold_high

    # --- Spectral Stretch (Inharmonic) ---
    print("\n--- Spectral Stretch ---")

    spec_stretch = cycdp.spectral_stretch(buf, max_stretch=1.5, freq_divide=500.0)
    print(f"Spectral stretch (bell-like)")
    outputs["fx02_spectral_stretch.wav"] = spec_stretch

    spec_stretch_extreme = cycdp.spectral_stretch(buf, max_stretch=3.0, freq_divide=300.0)
    print(f"Extreme spectral stretch")
    outputs["fx02_spectral_stretch_extreme.wav"] = spec_stretch_extreme

    # --- Spectral Clean ---
    print("\n--- Spectral Clean ---")

    cleaned = cycdp.spectral_clean(buf, threshold_db=-50.0)
    print(f"Cleaned (remove quiet bins)")
    outputs["fx02_cleaned.wav"] = cleaned

    # --- Freeze ---
    print("\n--- Freeze Effects ---")

    duration = buf.frame_count / buf.sample_rate

    # Freeze early part
    freeze_early = cycdp.freeze(buf, start_time=min(0.5, duration * 0.1),
                                end_time=min(1.0, duration * 0.2), duration=8.0)
    print(f"Freeze early section -> 8s")
    outputs["fx02_freeze_early.wav"] = freeze_early

    # Freeze middle
    mid_time = duration / 2
    freeze_mid = cycdp.freeze(buf, start_time=mid_time, end_time=mid_time + 0.5, duration=8.0)
    print(f"Freeze middle section -> 8s")
    outputs["fx02_freeze_mid.wav"] = freeze_mid

    # Freeze with scatter
    freeze_scatter = cycdp.freeze(buf, start_time=duration * 0.25, end_time=duration * 0.35,
                                  duration=10.0, pitch_scatter=3.0, randomize=0.5)
    print(f"Freeze with pitch scatter")
    outputs["fx02_freeze_scatter.wav"] = freeze_scatter

    # --- Combined ---
    print("\n--- Combined Spectral ---")

    # Blur + stretch + pitch down = dark ambient
    dark_ambient = cycdp.time_stretch(buf, factor=3.0)
    dark_ambient = cycdp.spectral_blur(dark_ambient, blur_time=0.15)
    dark_ambient = cycdp.pitch_shift(dark_ambient, semitones=-7.0)
    print(f"Dark ambient (stretch + blur + pitch down)")
    outputs["fx02_dark_ambient.wav"] = dark_ambient

    # --- Write outputs ---
    print(f"\n=== Writing to {args.output}/ ===")
    for filename, buffer in outputs.items():
        path = os.path.join(args.output, filename)
        cycdp.write_file(path, cycdp.normalize(buffer, target=0.95))
        print(f"  {filename}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
