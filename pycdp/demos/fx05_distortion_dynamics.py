#!/usr/bin/env python3
"""
FX Demo 5: Distortion and Dynamics

Applies distortion, filtering, and dynamics processing to an input audio file.

Usage:
    python fx05_distortion_dynamics.py input.wav -o output_dir/
    python fx05_distortion_dynamics.py input.wav  # outputs to ./build/
"""

import argparse
import pycdp
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Apply distortion and dynamics effects to an audio file."
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

    print("=== FX Demo 5: Distortion and Dynamics ===\n")

    buf = pycdp.read_file(args.input)
    print(f"Loaded: {args.input}")
    print(f"  {buf.frame_count / buf.sample_rate:.2f}s, {buf.channels}ch, {buf.sample_rate}Hz")

    outputs = {}

    # --- Waveset Distortion ---
    print("\n--- Waveset Distortion ---")

    repeat_2x = pycdp.distort_repeat(buf, multiplier=2)
    print(f"Waveset repeat 2x")
    outputs["fx05_waveset_repeat_2x.wav"] = repeat_2x

    repeat_4x = pycdp.distort_repeat(buf, multiplier=4)
    print(f"Waveset repeat 4x (harsh)")
    outputs["fx05_waveset_repeat_4x.wav"] = repeat_4x

    wave_reverse = pycdp.distort_reverse(buf, cycle_count=2)
    print(f"Waveset reverse (every 2 cycles)")
    outputs["fx05_waveset_reverse.wav"] = wave_reverse

    wave_reverse_4 = pycdp.distort_reverse(buf, cycle_count=4)
    print(f"Waveset reverse (every 4 cycles)")
    outputs["fx05_waveset_reverse_4.wav"] = wave_reverse_4

    # --- Fractal Distortion ---
    print("\n--- Fractal Distortion ---")

    fractal_light = pycdp.distort_fractal(buf, scaling=0.7)
    print(f"Fractal light (0.7 scaling)")
    outputs["fx05_fractal_light.wav"] = fractal_light

    fractal_heavy = pycdp.distort_fractal(buf, scaling=0.3)
    print(f"Fractal heavy (0.3 scaling)")
    outputs["fx05_fractal_heavy.wav"] = fractal_heavy

    # --- Bitcrush ---
    print("\n--- Bitcrush ---")

    crush_12 = pycdp.bitcrush(buf, bit_depth=12)
    print(f"12-bit (subtle)")
    outputs["fx05_bitcrush_12.wav"] = crush_12

    crush_8 = pycdp.bitcrush(buf, bit_depth=8)
    print(f"8-bit (retro)")
    outputs["fx05_bitcrush_8.wav"] = crush_8

    crush_4 = pycdp.bitcrush(buf, bit_depth=4)
    print(f"4-bit (extreme)")
    outputs["fx05_bitcrush_4.wav"] = crush_4

    # --- Filters ---
    print("\n--- Filters ---")

    lp_dark = pycdp.filter_lowpass(buf, cutoff_freq=500.0)
    print(f"Lowpass 500 Hz (dark)")
    outputs["fx05_lowpass_500.wav"] = lp_dark

    lp_warm = pycdp.filter_lowpass(buf, cutoff_freq=2000.0)
    print(f"Lowpass 2000 Hz (warm)")
    outputs["fx05_lowpass_2k.wav"] = lp_warm

    hp_thin = pycdp.filter_highpass(buf, cutoff_freq=1000.0)
    print(f"Highpass 1000 Hz (thin)")
    outputs["fx05_highpass_1k.wav"] = hp_thin

    hp_telephone = pycdp.filter_highpass(buf, cutoff_freq=300.0)
    hp_telephone = pycdp.filter_lowpass(hp_telephone, cutoff_freq=3000.0)
    print(f"Telephone (300-3000 Hz bandpass)")
    outputs["fx05_telephone.wav"] = hp_telephone

    bp_nasal = pycdp.filter_bandpass(buf, low_freq=800.0, high_freq=1500.0)
    print(f"Bandpass 800-1500 Hz (nasal)")
    outputs["fx05_bandpass_nasal.wav"] = bp_nasal

    notch = pycdp.filter_notch(buf, center_freq=1000.0, width_hz=200.0)
    print(f"Notch at 1000 Hz")
    outputs["fx05_notch_1k.wav"] = notch

    # --- Dynamics ---
    print("\n--- Dynamics Processing ---")

    compressed = pycdp.compressor(buf, threshold_db=-12.0, ratio=4.0)
    print(f"Compression 4:1 at -12dB")
    outputs["fx05_compressed.wav"] = compressed

    compressed_heavy = pycdp.compressor(buf, threshold_db=-20.0, ratio=8.0)
    print(f"Heavy compression 8:1 at -20dB")
    outputs["fx05_compressed_heavy.wav"] = compressed_heavy

    limited = pycdp.limiter(buf, threshold_db=-3.0)
    print(f"Limited at -3dB")
    outputs["fx05_limited.wav"] = limited

    gated = pycdp.gate(buf, threshold_db=-30.0)
    print(f"Gated at -30dB")
    outputs["fx05_gated.wav"] = gated

    # --- Combined ---
    print("\n--- Combined Effects ---")

    # Lo-fi: bitcrush + lowpass + compression
    lofi = pycdp.bitcrush(buf, bit_depth=10)
    lofi = pycdp.filter_lowpass(lofi, cutoff_freq=4000.0)
    lofi = pycdp.compressor(lofi, threshold_db=-15.0, ratio=3.0)
    print(f"Lo-fi chain")
    outputs["fx05_lofi.wav"] = lofi

    # Industrial: waveset + fractal + compression
    industrial = pycdp.distort_repeat(buf, multiplier=2)
    industrial = pycdp.distort_fractal(industrial, scaling=0.5)
    industrial = pycdp.compressor(industrial, threshold_db=-10.0, ratio=6.0)
    print(f"Industrial chain")
    outputs["fx05_industrial.wav"] = industrial

    # --- Write outputs ---
    print(f"\n=== Writing to {args.output}/ ===")
    for filename, buffer in outputs.items():
        path = os.path.join(args.output, filename)
        pycdp.write_file(path, pycdp.normalize(buffer, target=0.95))
        print(f"  {filename}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
