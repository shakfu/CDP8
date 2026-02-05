#!/usr/bin/env python3
"""
FX Demo 7: Creative Effect Chains

Complex effect combinations and sound design recipes.

Usage:
    python fx07_creative_chains.py input.wav -o output_dir/
    python fx07_creative_chains.py input.wav  # outputs to ./build/
"""

import argparse
import pycdp
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Apply creative effect chains to an audio file."
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

    print("=== FX Demo 7: Creative Effect Chains ===\n")

    buf = pycdp.read_file(args.input)
    print(f"Loaded: {args.input}")
    print(f"  {buf.frame_count / buf.sample_rate:.2f}s, {buf.channels}ch, {buf.sample_rate}Hz")

    duration = buf.frame_count / buf.sample_rate
    outputs = {}

    # --- Ambient Textures ---
    print("\n--- Ambient Textures ---")

    # Pad from source: stretch + blur + reverb
    ambient_pad = pycdp.time_stretch(buf, factor=4.0)
    ambient_pad = pycdp.spectral_blur(ambient_pad, blur_time=0.15)
    ambient_pad = pycdp.pitch_shift(ambient_pad, semitones=-5.0)
    ambient_pad = pycdp.reverb(ambient_pad, decay_time=4.0, damping=0.3, mix=0.5)
    print(f"Ambient pad (stretch + blur + pitch + reverb)")
    outputs["fx07_ambient_pad.wav"] = ambient_pad

    # Frozen texture
    freeze_start = min(duration * 0.25, duration - 2.0)
    freeze_end = min(freeze_start + 2.0, duration)
    frozen_texture = pycdp.freeze(buf, start_time=freeze_start, end_time=freeze_end,
                                  duration=20.0, pitch_scatter=2.0, randomize=0.4)
    frozen_texture = pycdp.reverb(frozen_texture, decay_time=3.0, mix=0.4)
    print(f"Frozen texture with reverb")
    outputs["fx07_frozen_texture.wav"] = frozen_texture

    # Granular cloud
    cloud = pycdp.wrappage(buf, grain_size=70.0, density=4.0, velocity=0.2,
                           spread=1.0, jitter=0.4)
    cloud = pycdp.spectral_blur(cloud, blur_time=0.1)
    cloud = pycdp.reverb(cloud, decay_time=5.0, damping=0.2, mix=0.5)
    print(f"Granular cloud (wrappage + blur + reverb)")
    outputs["fx07_granular_cloud.wav"] = cloud

    # --- Dark/Heavy Textures ---
    print("\n--- Dark/Heavy Textures ---")

    # Dark drone
    dark_drone = pycdp.time_stretch(buf, factor=6.0)
    dark_drone = pycdp.pitch_shift(dark_drone, semitones=-12.0)
    dark_drone = pycdp.spectral_blur(dark_drone, blur_time=0.2)
    dark_drone = pycdp.filter_lowpass(dark_drone, cutoff_freq=800.0)
    dark_drone = pycdp.reverb(dark_drone, decay_time=6.0, damping=0.6, mix=0.5)
    print(f"Dark drone")
    outputs["fx07_dark_drone.wav"] = dark_drone

    # Industrial texture
    industrial = pycdp.distort_repeat(buf, multiplier=3)
    industrial = pycdp.brassage(industrial, velocity=0.7, density=1.5, grainsize_ms=30.0, scatter=0.4)
    industrial = pycdp.filter_bandpass(industrial, low_freq=200.0, high_freq=4000.0)
    industrial = pycdp.compressor(industrial, threshold_db=-15.0, ratio=6.0)
    print(f"Industrial texture")
    outputs["fx07_industrial.wav"] = industrial

    # --- Rhythmic Textures ---
    print("\n--- Rhythmic Textures ---")

    # Stuttered
    stuttered = pycdp.psow_dupl(buf, repeat_count=3)
    stuttered = pycdp.tremolo(stuttered, freq=4.0, depth=0.6)
    stuttered = pycdp.delay(stuttered, delay_ms=187.5, feedback=0.4, mix=0.35)
    print(f"Stuttered rhythm")
    outputs["fx07_stuttered.wav"] = stuttered

    # Granular pulse
    pulse = pycdp.wrappage(buf, grain_size=25.0, density=0.6, velocity=1.0, spread=0.8)
    pulse = pycdp.compressor(pulse, threshold_db=-12.0, ratio=4.0)
    pulse = pycdp.delay(pulse, delay_ms=125.0, feedback=0.3, mix=0.3)
    print(f"Granular pulse")
    outputs["fx07_granular_pulse.wav"] = pulse

    # --- Ethereal/Bright ---
    print("\n--- Ethereal/Bright ---")

    # Shimmer
    shimmer = pycdp.pitch_shift(buf, semitones=12.0)
    shimmer = pycdp.spectral_blur(shimmer, blur_time=0.08)
    shimmer_low = pycdp.pitch_shift(buf, semitones=7.0)
    shimmer = pycdp.mix2(shimmer, shimmer_low, gain_a=0.6, gain_b=0.4)
    shimmer = pycdp.reverb(shimmer, decay_time=4.0, damping=0.2, mix=0.6)
    print(f"Shimmer (octave + fifth layer)")
    outputs["fx07_shimmer.wav"] = shimmer

    # Crystalline
    crystal = pycdp.spectral_stretch(buf, max_stretch=2.0, freq_divide=400.0)
    crystal = pycdp.spectral_blur(crystal, blur_time=0.05)
    crystal = pycdp.filter_highpass(crystal, cutoff_freq=500.0)
    crystal = pycdp.reverb(crystal, decay_time=3.0, damping=0.1, mix=0.5)
    print(f"Crystalline (spectral stretch + highpass)")
    outputs["fx07_crystalline.wav"] = crystal

    # --- Metallic/Inharmonic ---
    print("\n--- Metallic/Inharmonic ---")

    # Bell-like
    bell = pycdp.spectral_stretch(buf, max_stretch=3.0, freq_divide=300.0)
    bell = pycdp.reverb(bell, decay_time=4.0, damping=0.3, mix=0.4)
    print(f"Bell-like (extreme spectral stretch)")
    outputs["fx07_bell.wav"] = bell

    # Metallic fold
    metallic = pycdp.spectral_fold(buf, fold_freq=1500.0)
    metallic = pycdp.ring_mod(metallic, freq=100.0)
    metallic = pycdp.reverb(metallic, decay_time=2.0, mix=0.3)
    print(f"Metallic (fold + ring mod)")
    outputs["fx07_metallic.wav"] = metallic

    # --- Vocal/Formant ---
    print("\n--- Vocal/Formant Effects ---")

    # Formant shift up (chipmunk-ish but natural)
    formant_up = pycdp.fofex_repitch(buf, pitch_shift=5.0, preserve_formants=True)
    formant_up = pycdp.chorus(formant_up, rate=1.0, depth_ms=15.0, mix=0.3)
    print(f"Formant-preserved pitch up")
    outputs["fx07_formant_up.wav"] = formant_up

    # Formant shift down (deeper but natural)
    formant_down = pycdp.fofex_repitch(buf, pitch_shift=-5.0, preserve_formants=True)
    formant_down = pycdp.reverb(formant_down, decay_time=1.5, mix=0.3)
    print(f"Formant-preserved pitch down")
    outputs["fx07_formant_down.wav"] = formant_down

    # --- Evolving/Moving ---
    print("\n--- Evolving/Moving ---")

    # Spinning pad
    spinning = pycdp.time_stretch(buf, factor=2.0)
    spinning = pycdp.spectral_blur(spinning, blur_time=0.1)
    spinning = pycdp.spin(spinning, rate=0.15)
    spinning = pycdp.reverb(spinning, decay_time=3.0, mix=0.4)
    print(f"Spinning pad")
    outputs["fx07_spinning.wav"] = spinning

    # Hover exploration
    hover_exp = pycdp.hover(buf, frequency=0.5, location=0.5, loc_rand=0.6,
                            duration=min(30.0, duration * 3))
    hover_exp = pycdp.spectral_blur(hover_exp, blur_time=0.1)
    hover_exp = pycdp.reverb(hover_exp, decay_time=2.5, mix=0.4)
    print(f"Hover exploration")
    outputs["fx07_hover_exploration.wav"] = hover_exp

    # --- Lo-Fi/Degraded ---
    print("\n--- Lo-Fi/Degraded ---")

    # Tape simulation
    tape = pycdp.filter_lowpass(buf, cutoff_freq=8000.0)
    tape = pycdp.filter_highpass(tape, cutoff_freq=80.0)
    tape = pycdp.chorus(tape, rate=0.3, depth_ms=8.0, mix=0.2)  # Wow/flutter
    tape = pycdp.compressor(tape, threshold_db=-18.0, ratio=3.0)
    print(f"Tape simulation")
    outputs["fx07_tape.wav"] = tape

    # Degraded transmission
    degraded = pycdp.bitcrush(buf, bit_depth=10)
    degraded = pycdp.filter_bandpass(degraded, low_freq=400.0, high_freq=3500.0)
    degraded = pycdp.tremolo(degraded, freq=0.5, depth=0.3)  # Signal fade
    degraded = pycdp.reverb(degraded, decay_time=1.0, damping=0.7, mix=0.2)
    print(f"Degraded transmission")
    outputs["fx07_degraded.wav"] = degraded

    # --- Write outputs ---
    print(f"\n=== Writing to {args.output}/ ===")
    for filename, buffer in outputs.items():
        path = os.path.join(args.output, filename)
        pycdp.write_file(path, pycdp.normalize(buffer, target=0.95))
        print(f"  {filename}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
