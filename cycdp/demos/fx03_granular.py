#!/usr/bin/env python3
"""
FX Demo 3: Granular Processing

Applies granular synthesis techniques to an input audio file.

Usage:
    python fx03_granular.py input.wav -o output_dir/
    python fx03_granular.py input.wav  # outputs to ./build/
"""

import argparse
import cycdp
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Apply granular processing effects to an audio file."
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

    print("=== FX Demo 3: Granular Processing ===\n")

    buf = cycdp.read_file(args.input)
    print(f"Loaded: {args.input}")
    print(f"  {buf.frame_count / buf.sample_rate:.2f}s, {buf.channels}ch, {buf.sample_rate}Hz")

    outputs = {}

    # --- Brassage Time Stretch ---
    print("\n--- Brassage (Granular Time Stretch) ---")

    brass_stretch = cycdp.brassage(buf, velocity=0.5, density=1.0, grainsize_ms=50.0)
    print(f"Granular 2x stretch")
    outputs["fx03_brassage_stretch.wav"] = brass_stretch

    brass_compress = cycdp.brassage(buf, velocity=2.0, density=1.0, grainsize_ms=30.0)
    print(f"Granular 0.5x compress")
    outputs["fx03_brassage_compress.wav"] = brass_compress

    # --- Brassage with Scatter ---
    print("\n--- Brassage with Scatter ---")

    brass_scatter = cycdp.brassage(buf, velocity=0.5, density=1.5, grainsize_ms=40.0, scatter=0.5)
    print(f"Scattered grains (random placement)")
    outputs["fx03_brassage_scatter.wav"] = brass_scatter

    brass_scatter_dense = cycdp.brassage(buf, velocity=0.25, density=3.0, grainsize_ms=60.0, scatter=0.7)
    print(f"Dense scattered cloud")
    outputs["fx03_brassage_dense_scatter.wav"] = brass_scatter_dense

    # --- Brassage Pitch ---
    print("\n--- Brassage Pitch Shift ---")

    brass_pitch_up = cycdp.brassage(buf, velocity=1.0, density=1.0, grainsize_ms=40.0, pitch_shift=7.0)
    print(f"Granular pitch +7 semitones")
    outputs["fx03_brassage_pitch_up.wav"] = brass_pitch_up

    brass_pitch_down = cycdp.brassage(buf, velocity=1.0, density=1.0, grainsize_ms=40.0, pitch_shift=-12.0)
    print(f"Granular pitch -12 semitones")
    outputs["fx03_brassage_pitch_down.wav"] = brass_pitch_down

    # --- Grain Manipulation ---
    print("\n--- Grain Manipulation ---")

    grain_shuffle = cycdp.grain_reorder(buf)
    print(f"Shuffled grains")
    outputs["fx03_grain_shuffle.wav"] = grain_shuffle

    grain_reverse = cycdp.grain_reverse(buf)
    print(f"Reversed grain order")
    outputs["fx03_grain_reverse.wav"] = grain_reverse

    grain_repitch = cycdp.grain_repitch(buf, pitch_semitones=5.0)
    print(f"Repitched grains (+5 semitones)")
    outputs["fx03_grain_repitch.wav"] = grain_repitch

    # Variable pitch curve
    grain_pitch_curve = cycdp.grain_repitch(buf, pitch_curve=[
        (0.0, 0.0),
        (0.25, 7.0),
        (0.5, 0.0),
        (0.75, -7.0),
        (1.0, 0.0)
    ])
    print(f"Pitch curve (wave pattern)")
    outputs["fx03_grain_pitch_wave.wav"] = grain_pitch_curve

    # --- Wrappage (Stereo Granular Texture) ---
    print("\n--- Wrappage (Stereo Granular) ---")

    wrap_basic = cycdp.wrappage(buf, grain_size=50.0, density=2.0, velocity=1.0, spread=1.0)
    print(f"Basic wrappage (stereo texture)")
    outputs["fx03_wrappage_basic.wav"] = wrap_basic

    wrap_slow = cycdp.wrappage(buf, grain_size=60.0, density=2.0, velocity=0.3, spread=0.8)
    print(f"Slow wrappage (stretched)")
    outputs["fx03_wrappage_slow.wav"] = wrap_slow

    duration = buf.frame_count / buf.sample_rate
    wrap_frozen = cycdp.wrappage(buf, grain_size=80.0, density=4.0, velocity=0.0,
                                 spread=1.0, jitter=0.4, duration=min(16.0, duration * 2))
    print(f"Frozen wrappage (sustained texture)")
    outputs["fx03_wrappage_frozen.wav"] = wrap_frozen

    wrap_pitched = cycdp.wrappage(buf, grain_size=40.0, density=2.0, velocity=0.5,
                                  pitch=-12.0, spread=0.7)
    print(f"Pitched wrappage (octave down)")
    outputs["fx03_wrappage_pitched.wav"] = wrap_pitched

    wrap_sparse = cycdp.wrappage(buf, grain_size=30.0, density=0.4, velocity=1.0, spread=1.0)
    print(f"Sparse wrappage (rhythmic gaps)")
    outputs["fx03_wrappage_sparse.wav"] = wrap_sparse

    # --- Grain Cloud ---
    print("\n--- Grain Cloud ---")

    cloud = cycdp.grain_cloud(buf, gate=0.1, grainsize_ms=25.0)
    print(f"Grain cloud")
    outputs["fx03_grain_cloud.wav"] = cloud

    extended = cycdp.grain_extend(buf, grainsize_ms=30.0, trough=0.3)
    print(f"Extended grains")
    outputs["fx03_grain_extend.wav"] = extended

    # --- Write outputs ---
    print(f"\n=== Writing to {args.output}/ ===")
    for filename, buffer in outputs.items():
        path = os.path.join(args.output, filename)
        cycdp.write_file(path, cycdp.normalize(buffer, target=0.95))
        print(f"  {filename}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
