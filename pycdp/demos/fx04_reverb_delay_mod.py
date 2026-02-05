#!/usr/bin/env python3
"""
FX Demo 4: Reverb, Delay, and Modulation

Applies space and modulation effects to an input audio file.

Usage:
    python fx04_reverb_delay_mod.py input.wav -o output_dir/
    python fx04_reverb_delay_mod.py input.wav  # outputs to ./build/
"""

import argparse
import pycdp
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Apply reverb, delay, and modulation effects to an audio file."
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

    print("=== FX Demo 4: Reverb, Delay, and Modulation ===\n")

    buf = pycdp.read_file(args.input)
    print(f"Loaded: {args.input}")
    print(f"  {buf.frame_count / buf.sample_rate:.2f}s, {buf.channels}ch, {buf.sample_rate}Hz")

    outputs = {}

    # --- Reverb ---
    print("\n--- Reverb ---")

    reverb_small = pycdp.reverb(buf, decay_time=0.8, damping=0.6, mix=0.3)
    print(f"Small room (0.8s decay)")
    outputs["fx04_reverb_small.wav"] = reverb_small

    reverb_medium = pycdp.reverb(buf, decay_time=2.0, damping=0.4, mix=0.4)
    print(f"Medium hall (2s decay)")
    outputs["fx04_reverb_medium.wav"] = reverb_medium

    reverb_large = pycdp.reverb(buf, decay_time=4.0, damping=0.3, mix=0.5)
    print(f"Large hall (4s decay)")
    outputs["fx04_reverb_large.wav"] = reverb_large

    reverb_huge = pycdp.reverb(buf, decay_time=8.0, damping=0.2, mix=0.6)
    print(f"Cathedral (8s decay)")
    outputs["fx04_reverb_cathedral.wav"] = reverb_huge

    reverb_dark = pycdp.reverb(buf, decay_time=3.0, damping=0.8, mix=0.5)
    print(f"Dark reverb (high damping)")
    outputs["fx04_reverb_dark.wav"] = reverb_dark

    reverb_bright = pycdp.reverb(buf, decay_time=3.0, damping=0.1, mix=0.5)
    print(f"Bright reverb (low damping)")
    outputs["fx04_reverb_bright.wav"] = reverb_bright

    # --- Delay ---
    print("\n--- Delay ---")

    delay_short = pycdp.delay(buf, delay_ms=125.0, feedback=0.3, mix=0.4)
    print(f"Short delay (125ms, 1/8 note at 120bpm)")
    outputs["fx04_delay_short.wav"] = delay_short

    delay_quarter = pycdp.delay(buf, delay_ms=500.0, feedback=0.4, mix=0.4)
    print(f"Quarter note delay (500ms)")
    outputs["fx04_delay_quarter.wav"] = delay_quarter

    delay_long = pycdp.delay(buf, delay_ms=750.0, feedback=0.5, mix=0.35)
    print(f"Dotted quarter delay (750ms)")
    outputs["fx04_delay_dotted.wav"] = delay_long

    delay_heavy = pycdp.delay(buf, delay_ms=400.0, feedback=0.7, mix=0.5)
    print(f"Heavy feedback delay (70%)")
    outputs["fx04_delay_heavy_fb.wav"] = delay_heavy

    delay_slapback = pycdp.delay(buf, delay_ms=80.0, feedback=0.1, mix=0.3)
    print(f"Slapback delay (80ms)")
    outputs["fx04_delay_slapback.wav"] = delay_slapback

    # --- Tremolo ---
    print("\n--- Tremolo ---")

    trem_slow = pycdp.tremolo(buf, freq=2.0, depth=0.6)
    print(f"Slow tremolo (2 Hz)")
    outputs["fx04_tremolo_slow.wav"] = trem_slow

    trem_medium = pycdp.tremolo(buf, freq=6.0, depth=0.7)
    print(f"Medium tremolo (6 Hz)")
    outputs["fx04_tremolo_medium.wav"] = trem_medium

    trem_fast = pycdp.tremolo(buf, freq=12.0, depth=0.5)
    print(f"Fast tremolo (12 Hz)")
    outputs["fx04_tremolo_fast.wav"] = trem_fast

    # --- Chorus ---
    print("\n--- Chorus ---")

    chorus_light = pycdp.chorus(buf, rate=1.0, depth_ms=15.0, mix=0.4)
    print(f"Light chorus")
    outputs["fx04_chorus_light.wav"] = chorus_light

    chorus_heavy = pycdp.chorus(buf, rate=0.5, depth_ms=30.0, mix=0.5)
    print(f"Heavy chorus (wide)")
    outputs["fx04_chorus_heavy.wav"] = chorus_heavy

    # --- Flanger ---
    print("\n--- Flanger ---")

    flanger_slow = pycdp.flanger(buf, rate=0.2, depth_ms=5.0, feedback=0.5)
    print(f"Slow flanger")
    outputs["fx04_flanger_slow.wav"] = flanger_slow

    flanger_fast = pycdp.flanger(buf, rate=1.0, depth_ms=3.0, feedback=0.6)
    print(f"Fast flanger")
    outputs["fx04_flanger_fast.wav"] = flanger_fast

    flanger_jet = pycdp.flanger(buf, rate=0.1, depth_ms=8.0, feedback=0.8)
    print(f"Jet flanger (high feedback)")
    outputs["fx04_flanger_jet.wav"] = flanger_jet

    # --- Ring Modulation ---
    print("\n--- Ring Modulation ---")

    ring_low = pycdp.ring_mod(buf, freq=50.0)
    print(f"Ring mod 50 Hz (rumble)")
    outputs["fx04_ring_50hz.wav"] = ring_low

    ring_mid = pycdp.ring_mod(buf, freq=200.0)
    print(f"Ring mod 200 Hz (robotic)")
    outputs["fx04_ring_200hz.wav"] = ring_mid

    ring_high = pycdp.ring_mod(buf, freq=800.0)
    print(f"Ring mod 800 Hz (metallic)")
    outputs["fx04_ring_800hz.wav"] = ring_high

    # --- Spatial ---
    print("\n--- Spatial Effects ---")

    spin = pycdp.spin(buf, rate=0.25)
    print(f"Slow spin (0.25 Hz rotation)")
    outputs["fx04_spin_slow.wav"] = spin

    spin_fast = pycdp.spin(buf, rate=2.0)
    print(f"Fast spin (2 Hz rotation)")
    outputs["fx04_spin_fast.wav"] = spin_fast

    flutter = pycdp.flutter(buf, frequency=3.0, depth=0.7)
    print(f"Flutter (stereo tremolo)")
    outputs["fx04_flutter.wav"] = flutter

    # --- Combined ---
    print("\n--- Combined Effects ---")

    # Delay into reverb
    delay_verb = pycdp.delay(buf, delay_ms=375.0, feedback=0.4, mix=0.3)
    delay_verb = pycdp.reverb(delay_verb, decay_time=2.5, damping=0.4, mix=0.4)
    print(f"Delay into reverb")
    outputs["fx04_delay_into_reverb.wav"] = delay_verb

    # Chorus + reverb (lush)
    lush = pycdp.chorus(buf, rate=0.8, depth_ms=20.0, mix=0.4)
    lush = pycdp.reverb(lush, decay_time=3.0, damping=0.3, mix=0.45)
    print(f"Lush (chorus + reverb)")
    outputs["fx04_lush.wav"] = lush

    # --- Write outputs ---
    print(f"\n=== Writing to {args.output}/ ===")
    for filename, buffer in outputs.items():
        path = os.path.join(args.output, filename)
        pycdp.write_file(path, pycdp.normalize(buffer, target=0.95))
        print(f"  {filename}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
