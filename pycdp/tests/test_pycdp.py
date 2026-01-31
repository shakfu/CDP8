"""
Tests for pycdp - Python bindings for CDP audio processing library.

Uses array.array for testing (no numpy dependency required).
"""

import array
import pytest

import pycdp


class TestVersion:
    def test_version_string(self):
        v = pycdp.version()
        assert isinstance(v, str)
        assert len(v) > 0
        # Should be semver-like
        parts = v.split('.')
        assert len(parts) >= 2


class TestUtilities:
    def test_gain_to_db_unity(self):
        assert pycdp.gain_to_db(1.0) == pytest.approx(0.0)

    def test_gain_to_db_double(self):
        # +6dB = double amplitude
        assert pycdp.gain_to_db(2.0) == pytest.approx(6.0206, rel=0.01)

    def test_db_to_gain_zero(self):
        assert pycdp.db_to_gain(0.0) == pytest.approx(1.0)

    def test_db_to_gain_six(self):
        assert pycdp.db_to_gain(6.0206) == pytest.approx(2.0, rel=0.01)

    def test_round_trip(self):
        for gain_val in [0.1, 0.5, 1.0, 2.0, 10.0]:
            db = pycdp.gain_to_db(gain_val)
            back = pycdp.db_to_gain(db)
            assert back == pytest.approx(gain_val, rel=1e-10)


class TestContext:
    def test_create_destroy(self):
        ctx = pycdp.Context()
        assert ctx is not None
        del ctx  # Should not raise

    def test_initial_no_error(self):
        ctx = pycdp.Context()
        assert ctx.get_error() == 0  # CDP_OK


class TestBuffer:
    def test_create(self):
        buf = pycdp.Buffer.create(1000, channels=2, sample_rate=44100)
        assert buf.frame_count == 1000
        assert buf.channels == 2
        assert buf.sample_rate == 44100
        assert buf.sample_count == 2000
        assert len(buf) == 2000

    def test_from_array(self):
        samples = array.array('f', [0.5] * 100)
        buf = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)
        assert buf.sample_count == 100
        assert buf[0] == pytest.approx(0.5)

    def test_indexing(self):
        buf = pycdp.Buffer.create(100, channels=1, sample_rate=44100)
        buf[0] = 0.5
        buf[50] = -0.3
        assert buf[0] == pytest.approx(0.5)
        assert buf[50] == pytest.approx(-0.3)

    def test_to_list(self):
        samples = array.array('f', [0.1, 0.2, 0.3])
        buf = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)
        lst = buf.to_list()
        assert lst == pytest.approx([0.1, 0.2, 0.3], rel=1e-6)

    def test_buffer_protocol(self):
        """Buffer should support the buffer protocol."""
        buf = pycdp.Buffer.create(100, channels=1, sample_rate=44100)
        buf[0] = 0.5
        # Create memoryview from buffer
        mv = memoryview(buf)
        assert mv.format == 'f'
        assert len(mv) == 100


class TestGain:
    def test_unity_gain(self):
        samples = array.array('f', [0.5] * 100)
        result = pycdp.gain(samples, gain_factor=1.0)
        assert result[0] == pytest.approx(0.5, rel=1e-6)

    def test_double_gain(self):
        samples = array.array('f', [0.25] * 100)
        result = pycdp.gain(samples, gain_factor=2.0)
        assert result[0] == pytest.approx(0.5, rel=1e-6)

    def test_half_gain(self):
        samples = array.array('f', [0.8] * 100)
        result = pycdp.gain(samples, gain_factor=0.5)
        assert result[0] == pytest.approx(0.4, rel=1e-6)

    def test_clipping(self):
        samples = array.array('f', [0.6] * 100)
        # Without clipping - should exceed 1.0
        result_no_clip = pycdp.gain(samples, gain_factor=2.0, clip=False)
        assert result_no_clip[0] == pytest.approx(1.2, rel=1e-6)

        # With clipping - should be clamped
        result_clip = pycdp.gain(samples, gain_factor=2.0, clip=True)
        assert result_clip[0] == pytest.approx(1.0, rel=1e-6)


class TestGainDb:
    def test_zero_db(self):
        samples = array.array('f', [0.5] * 100)
        result = pycdp.gain_db(samples, db=0.0)
        assert result[0] == pytest.approx(0.5, rel=1e-5)

    def test_plus_six_db(self):
        samples = array.array('f', [0.5] * 100)
        result = pycdp.gain_db(samples, db=6.0)
        # 6dB ~= 2x
        assert result[0] == pytest.approx(1.0, rel=0.02)

    def test_minus_six_db(self):
        samples = array.array('f', [0.5] * 100)
        result = pycdp.gain_db(samples, db=-6.0)
        # -6dB ~= 0.5x
        assert result[0] == pytest.approx(0.25, rel=0.02)


class TestNormalize:
    def test_normalize_to_unity(self):
        samples = array.array('f', [0.25, -0.5, 0.3])
        result = pycdp.normalize(samples, target=1.0)
        # Peak should be 1.0 (at index 1, which was -0.5)
        assert abs(result[1]) == pytest.approx(1.0, rel=1e-6)

    def test_normalize_to_target(self):
        samples = array.array('f', [0.25, -0.5, 0.3])
        result = pycdp.normalize(samples, target=0.8)
        assert abs(result[1]) == pytest.approx(0.8, rel=1e-6)

    def test_normalize_silent_raises(self):
        samples = array.array('f', [0.0] * 100)
        with pytest.raises(pycdp.CDPError):
            pycdp.normalize(samples)

    def test_preserves_relative_levels(self):
        samples = array.array('f', [0.2, -0.4, 0.1])
        result = pycdp.normalize(samples, target=1.0)
        # Ratios should be preserved
        assert result[0] / abs(result[1]) == pytest.approx(0.5, rel=1e-6)
        assert result[2] / abs(result[1]) == pytest.approx(0.25, rel=1e-6)


class TestPhaseInvert:
    def test_invert_positive(self):
        samples = array.array('f', [0.5] * 100)
        result = pycdp.phase_invert(samples)
        assert result[0] == pytest.approx(-0.5, rel=1e-6)

    def test_invert_negative(self):
        samples = array.array('f', [-0.3] * 100)
        result = pycdp.phase_invert(samples)
        assert result[0] == pytest.approx(0.3, rel=1e-6)

    def test_double_invert_identity(self):
        samples = array.array('f', [0.1, -0.2, 0.3, -0.4, 0.5])
        result1 = pycdp.phase_invert(samples)
        # Need to convert Buffer to array for second call
        result2 = pycdp.phase_invert(memoryview(result1))
        for i in range(len(samples)):
            assert result2[i] == pytest.approx(samples[i], rel=1e-6)


class TestPeak:
    def test_find_peak_positive(self):
        samples = array.array('f', [0.5, 0.8, 0.3, 0.1])
        level, pos = pycdp.peak(samples)
        assert level == pytest.approx(0.8, rel=1e-6)
        assert pos == 1

    def test_find_peak_negative(self):
        samples = array.array('f', [0.5, -0.9, 0.3, 0.1])
        level, pos = pycdp.peak(samples)
        assert level == pytest.approx(0.9, rel=1e-6)
        assert pos == 1


class TestLowLevelAPI:
    """Test the low-level API with explicit Context and Buffer."""

    def test_apply_gain(self):
        ctx = pycdp.Context()
        buf = pycdp.Buffer.create(100, channels=1, sample_rate=44100)

        # Fill with values
        for i in range(100):
            buf[i] = 0.5

        pycdp.apply_gain(ctx, buf, 2.0, clip=False)

        for i in range(100):
            assert buf[i] == pytest.approx(1.0, rel=1e-6)

    def test_apply_normalize(self):
        ctx = pycdp.Context()
        samples = array.array('f', [0.25, -0.5, 0.3])
        buf = pycdp.Buffer.from_memoryview(samples)

        pycdp.apply_normalize(ctx, buf, target_level=1.0)

        # Peak should be 1.0
        level, _ = pycdp.get_peak(ctx, buf)
        assert level == pytest.approx(1.0, rel=1e-6)


class TestBufferInterop:
    """Test interoperability with various buffer types."""

    def test_with_array_array(self):
        samples = array.array('f', [0.5] * 100)
        result = pycdp.gain(samples, gain_factor=2.0)
        assert result[0] == pytest.approx(1.0, rel=1e-6)

    def test_with_memoryview(self):
        samples = array.array('f', [0.5] * 100)
        mv = memoryview(samples)
        result = pycdp.gain(mv, gain_factor=2.0)
        assert result[0] == pytest.approx(1.0, rel=1e-6)

    def test_result_supports_memoryview(self):
        samples = array.array('f', [0.5] * 100)
        result = pycdp.gain(samples, gain_factor=2.0)
        # Result should support buffer protocol
        mv = memoryview(result)
        assert mv[0] == pytest.approx(1.0, rel=1e-6)


class TestBufferUtilities:
    """Test buffer utility operations."""

    def test_reverse(self):
        """Reverse should reverse sample order."""
        samples = array.array('f', [1.0, 2.0, 3.0, 4.0, 5.0])
        buf = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        rev = pycdp.reverse(buf)
        assert rev.sample_count == 5
        assert rev[0] == pytest.approx(5.0, rel=1e-6)
        assert rev[4] == pytest.approx(1.0, rel=1e-6)
        assert rev[2] == pytest.approx(3.0, rel=1e-6)

    def test_reverse_stereo(self):
        """Reverse should preserve channel order within frames."""
        buf = pycdp.Buffer.create(3, channels=2, sample_rate=44100)
        # Frame 0: L=1, R=2
        buf[0] = 1.0
        buf[1] = 2.0
        # Frame 1: L=3, R=4
        buf[2] = 3.0
        buf[3] = 4.0
        # Frame 2: L=5, R=6
        buf[4] = 5.0
        buf[5] = 6.0

        rev = pycdp.reverse(buf)
        # Frame 0 should now be old Frame 2
        assert rev[0] == pytest.approx(5.0, rel=1e-6)
        assert rev[1] == pytest.approx(6.0, rel=1e-6)
        # Frame 2 should now be old Frame 0
        assert rev[4] == pytest.approx(1.0, rel=1e-6)
        assert rev[5] == pytest.approx(2.0, rel=1e-6)

    def test_fade_in_linear(self):
        """Linear fade in should ramp from 0 to 1."""
        # 1 second at 100 Hz
        buf = pycdp.Buffer.create(100, channels=1, sample_rate=100)
        for i in range(100):
            buf[i] = 1.0

        # Fade in over 0.5 seconds
        pycdp.fade_in(buf, duration=0.5, curve="linear")

        assert buf[0] == pytest.approx(0.0, abs=0.02)
        assert buf[25] == pytest.approx(0.5, abs=0.02)
        assert buf[75] == pytest.approx(1.0, rel=1e-6)

    def test_fade_out_linear(self):
        """Linear fade out should ramp from 1 to 0."""
        buf = pycdp.Buffer.create(100, channels=1, sample_rate=100)
        for i in range(100):
            buf[i] = 1.0

        pycdp.fade_out(buf, duration=0.5, curve="linear")

        assert buf[25] == pytest.approx(1.0, rel=1e-6)
        assert buf[75] == pytest.approx(0.5, abs=0.02)
        assert buf[99] == pytest.approx(0.0, abs=0.02)

    def test_fade_exponential(self):
        """Exponential fade should use equal-power curve."""
        buf = pycdp.Buffer.create(100, channels=1, sample_rate=100)
        for i in range(100):
            buf[i] = 1.0

        pycdp.fade_in(buf, duration=0.5, curve="exponential")

        # Exponential fade is smoother - mid-point should be higher than linear
        assert buf[25] > 0.5  # Equal power curve is above linear at midpoint

    def test_fade_invalid_curve_raises(self):
        """Invalid curve should raise ValueError."""
        buf = pycdp.Buffer.create(100, channels=1, sample_rate=44100)
        with pytest.raises(ValueError):
            pycdp.fade_in(buf, duration=0.5, curve="invalid")

    def test_concat(self):
        """Concat should join buffers end-to-end."""
        a_samples = array.array('f', [1.0] * 50)
        b_samples = array.array('f', [2.0] * 30)
        c_samples = array.array('f', [3.0] * 20)
        a = pycdp.Buffer.from_memoryview(a_samples, channels=1, sample_rate=44100)
        b = pycdp.Buffer.from_memoryview(b_samples, channels=1, sample_rate=44100)
        c = pycdp.Buffer.from_memoryview(c_samples, channels=1, sample_rate=44100)

        result = pycdp.concat([a, b, c])
        assert result.sample_count == 100

        assert result[25] == pytest.approx(1.0, rel=1e-6)
        assert result[60] == pytest.approx(2.0, rel=1e-6)
        assert result[90] == pytest.approx(3.0, rel=1e-6)

    def test_concat_empty_raises(self):
        """concat should fail with empty list."""
        with pytest.raises(ValueError):
            pycdp.concat([])


class TestSpatial:
    """Test spatial/panning operations."""

    def test_pan_center(self):
        """Center pan should give equal L and R."""
        samples = array.array('f', [0.8] * 100)
        mono = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        stereo = pycdp.pan(mono, position=0.0)
        assert stereo.channels == 2
        assert stereo.frame_count == 100

        # At center, L and R should be equal
        for i in range(100):
            assert stereo[i * 2] == pytest.approx(stereo[i * 2 + 1], rel=1e-6)

    def test_pan_left(self):
        """Full left pan should have L > R."""
        samples = array.array('f', [1.0] * 100)
        mono = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        stereo = pycdp.pan(mono, position=-1.0)

        # Left should be louder, right should be ~0
        assert stereo[0] > stereo[1]
        assert stereo[1] == pytest.approx(0.0, abs=0.01)

    def test_pan_right(self):
        """Full right pan should have R > L."""
        samples = array.array('f', [1.0] * 100)
        mono = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        stereo = pycdp.pan(mono, position=1.0)

        # Right should be louder, left should be ~0
        assert stereo[1] > stereo[0]
        assert stereo[0] == pytest.approx(0.0, abs=0.01)

    def test_pan_requires_mono(self):
        """pan should fail on stereo input."""
        stereo = pycdp.Buffer.create(100, channels=2, sample_rate=44100)
        with pytest.raises(pycdp.CDPError):
            pycdp.pan(stereo, position=0.0)

    def test_pan_envelope_left_to_right(self):
        """Pan envelope should move sound from left to right."""
        # 1 second of audio at 1000 Hz sample rate for easy calculation
        samples = array.array('f', [1.0] * 1000)
        mono = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=1000)

        # Pan from left (-1) to right (+1) over 1 second
        points = [(0.0, -1.0), (1.0, 1.0)]
        stereo = pycdp.pan_envelope(mono, points)

        assert stereo.channels == 2
        assert stereo.frame_count == 1000

        # At start (t=0): should be full left
        assert stereo[0] > stereo[1]  # L > R
        assert stereo[1] == pytest.approx(0.0, abs=0.02)

        # At middle (t=0.5): should be roughly center
        mid = 500
        # L and R should be approximately equal at center
        assert abs(stereo[mid * 2] - stereo[mid * 2 + 1]) < 0.1

        # At end (t=1.0): should be full right
        end = 999
        assert stereo[end * 2 + 1] > stereo[end * 2]  # R > L
        assert stereo[end * 2] == pytest.approx(0.0, abs=0.02)

    def test_pan_envelope_static(self):
        """Pan envelope with single point should behave like static pan."""
        samples = array.array('f', [0.8] * 100)
        mono = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        # Single point at center
        points = [(0.0, 0.0)]
        stereo = pycdp.pan_envelope(mono, points)

        # Should be same as static center pan
        for i in range(100):
            assert stereo[i * 2] == pytest.approx(stereo[i * 2 + 1], rel=1e-6)

    def test_pan_envelope_empty_raises(self):
        """pan_envelope should fail with empty points list."""
        samples = array.array('f', [1.0] * 100)
        mono = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        with pytest.raises(ValueError):
            pycdp.pan_envelope(mono, [])

    def test_mirror(self):
        """Mirror should swap L and R."""
        stereo = pycdp.Buffer.create(100, channels=2, sample_rate=44100)
        for i in range(100):
            stereo[i * 2] = 0.3      # Left
            stereo[i * 2 + 1] = 0.9  # Right

        mirrored = pycdp.mirror(stereo)
        assert mirrored.channels == 2

        for i in range(100):
            assert mirrored[i * 2] == pytest.approx(0.9, rel=1e-6)      # New L = old R
            assert mirrored[i * 2 + 1] == pytest.approx(0.3, rel=1e-6)  # New R = old L

    def test_mirror_requires_stereo(self):
        """mirror should fail on mono input."""
        mono = pycdp.Buffer.create(100, channels=1, sample_rate=44100)
        with pytest.raises(pycdp.CDPError):
            pycdp.mirror(mono)

    def test_narrow_to_mono(self):
        """Width 0 should produce mono (L=R)."""
        stereo = pycdp.Buffer.create(100, channels=2, sample_rate=44100)
        for i in range(100):
            stereo[i * 2] = 0.2
            stereo[i * 2 + 1] = 0.8

        narrowed = pycdp.narrow(stereo, width=0.0)

        # Both channels should be average: (0.2 + 0.8) / 2 = 0.5
        for i in range(100):
            assert narrowed[i * 2] == pytest.approx(0.5, rel=1e-6)
            assert narrowed[i * 2 + 1] == pytest.approx(0.5, rel=1e-6)

    def test_narrow_unchanged(self):
        """Width 1.0 should leave stereo unchanged."""
        stereo = pycdp.Buffer.create(100, channels=2, sample_rate=44100)
        for i in range(100):
            stereo[i * 2] = 0.2
            stereo[i * 2 + 1] = 0.8

        result = pycdp.narrow(stereo, width=1.0)

        for i in range(100):
            assert result[i * 2] == pytest.approx(0.2, rel=1e-6)
            assert result[i * 2 + 1] == pytest.approx(0.8, rel=1e-6)

    def test_narrow_requires_stereo(self):
        """narrow should fail on mono input."""
        mono = pycdp.Buffer.create(100, channels=1, sample_rate=44100)
        with pytest.raises(pycdp.CDPError):
            pycdp.narrow(mono, width=0.5)


class TestMixing:
    """Test mixing operations."""

    def test_mix2_equal_length(self):
        """Mix two buffers of equal length."""
        a_samples = array.array('f', [0.3] * 100)
        b_samples = array.array('f', [0.5] * 100)
        a = pycdp.Buffer.from_memoryview(a_samples, channels=1, sample_rate=44100)
        b = pycdp.Buffer.from_memoryview(b_samples, channels=1, sample_rate=44100)

        result = pycdp.mix2(a, b)
        assert result.sample_count == 100

        for i in range(100):
            assert result[i] == pytest.approx(0.8, rel=1e-6)

    def test_mix2_with_gains(self):
        """Mix two buffers with gains."""
        a_samples = array.array('f', [1.0] * 100)
        b_samples = array.array('f', [1.0] * 100)
        a = pycdp.Buffer.from_memoryview(a_samples, channels=1, sample_rate=44100)
        b = pycdp.Buffer.from_memoryview(b_samples, channels=1, sample_rate=44100)

        result = pycdp.mix2(a, b, gain_a=0.5, gain_b=0.5)

        for i in range(100):
            assert result[i] == pytest.approx(1.0, rel=1e-6)

    def test_mix2_different_lengths(self):
        """Mix buffers of different lengths."""
        a = pycdp.Buffer.create(100, channels=1, sample_rate=44100)
        b = pycdp.Buffer.create(50, channels=1, sample_rate=44100)

        for i in range(100):
            a[i] = 0.4
        for i in range(50):
            b[i] = 0.3

        result = pycdp.mix2(a, b)
        assert result.sample_count == 100

        # First 50: 0.4 + 0.3 = 0.7
        assert result[25] == pytest.approx(0.7, rel=1e-6)
        # Last 50: just 0.4
        assert result[75] == pytest.approx(0.4, rel=1e-6)

    def test_mix_multiple(self):
        """Mix multiple buffers."""
        bufs = []
        for val in [0.2, 0.3, 0.1]:
            samples = array.array('f', [val] * 100)
            bufs.append(pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100))

        result = pycdp.mix(bufs)

        # Sum: 0.2 + 0.3 + 0.1 = 0.6
        for i in range(100):
            assert result[i] == pytest.approx(0.6, rel=1e-6)

    def test_mix_with_gains(self):
        """Mix multiple buffers with gains."""
        a_samples = array.array('f', [1.0] * 100)
        b_samples = array.array('f', [1.0] * 100)
        a = pycdp.Buffer.from_memoryview(a_samples, channels=1, sample_rate=44100)
        b = pycdp.Buffer.from_memoryview(b_samples, channels=1, sample_rate=44100)

        result = pycdp.mix([a, b], gains=[0.25, 0.75])

        # 1.0*0.25 + 1.0*0.75 = 1.0
        for i in range(100):
            assert result[i] == pytest.approx(1.0, rel=1e-6)

    def test_mix_empty_raises(self):
        """mix should fail with empty list."""
        with pytest.raises(ValueError):
            pycdp.mix([])

    def test_mix_gains_length_mismatch_raises(self):
        """mix should fail if gains length doesn't match."""
        samples = array.array('f', [0.5] * 100)
        buf = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        with pytest.raises(ValueError):
            pycdp.mix([buf, buf], gains=[1.0])  # Only 1 gain for 2 buffers


class TestChannelOperations:
    """Test channel operations."""

    def test_to_mono_from_stereo(self):
        """Convert stereo to mono by averaging."""
        # Create stereo buffer: L=0.4, R=0.8 -> mono should be 0.6
        stereo = pycdp.Buffer.create(100, channels=2, sample_rate=44100)
        for i in range(100):
            stereo[i * 2] = 0.4      # Left
            stereo[i * 2 + 1] = 0.8  # Right

        mono = pycdp.to_mono(stereo)
        assert mono.channels == 1
        assert mono.frame_count == 100
        assert mono.sample_rate == 44100

        for i in range(mono.sample_count):
            assert mono[i] == pytest.approx(0.6, rel=1e-6)

    def test_to_mono_already_mono(self):
        """Converting mono to mono should just copy."""
        samples = array.array('f', [0.5] * 100)
        mono_in = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        mono_out = pycdp.to_mono(mono_in)
        assert mono_out.channels == 1
        assert mono_out.sample_count == 100

        for i in range(mono_out.sample_count):
            assert mono_out[i] == pytest.approx(0.5, rel=1e-6)

    def test_to_stereo(self):
        """Convert mono to stereo by duplicating."""
        samples = array.array('f', [0.7] * 100)
        mono = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        stereo = pycdp.to_stereo(mono)
        assert stereo.channels == 2
        assert stereo.frame_count == 100

        for i in range(100):
            assert stereo[i * 2] == pytest.approx(0.7, rel=1e-6)      # Left
            assert stereo[i * 2 + 1] == pytest.approx(0.7, rel=1e-6)  # Right

    def test_to_stereo_requires_mono(self):
        """to_stereo should fail on stereo input."""
        stereo = pycdp.Buffer.create(100, channels=2, sample_rate=44100)
        with pytest.raises(pycdp.CDPError):
            pycdp.to_stereo(stereo)

    def test_extract_channel(self):
        """Extract left and right channels from stereo."""
        stereo = pycdp.Buffer.create(100, channels=2, sample_rate=44100)
        for i in range(100):
            stereo[i * 2] = 0.3      # Left
            stereo[i * 2 + 1] = 0.9  # Right

        left = pycdp.extract_channel(stereo, 0)
        assert left.channels == 1
        for i in range(left.sample_count):
            assert left[i] == pytest.approx(0.3, rel=1e-6)

        right = pycdp.extract_channel(stereo, 1)
        assert right.channels == 1
        for i in range(right.sample_count):
            assert right[i] == pytest.approx(0.9, rel=1e-6)

    def test_extract_channel_out_of_range(self):
        """extract_channel should fail with invalid channel index."""
        stereo = pycdp.Buffer.create(100, channels=2, sample_rate=44100)
        with pytest.raises(pycdp.CDPError):
            pycdp.extract_channel(stereo, 2)

    def test_merge_channels(self):
        """Merge two mono buffers into stereo."""
        left_samples = array.array('f', [0.2] * 100)
        right_samples = array.array('f', [0.8] * 100)
        left = pycdp.Buffer.from_memoryview(left_samples, channels=1, sample_rate=44100)
        right = pycdp.Buffer.from_memoryview(right_samples, channels=1, sample_rate=44100)

        stereo = pycdp.merge_channels(left, right)
        assert stereo.channels == 2
        assert stereo.frame_count == 100

        for i in range(100):
            assert stereo[i * 2] == pytest.approx(0.2, rel=1e-6)
            assert stereo[i * 2 + 1] == pytest.approx(0.8, rel=1e-6)

    def test_split_channels(self):
        """Split stereo into separate mono buffers."""
        stereo = pycdp.Buffer.create(100, channels=2, sample_rate=44100)
        for i in range(100):
            stereo[i * 2] = 0.1
            stereo[i * 2 + 1] = 0.9

        channels = pycdp.split_channels(stereo)
        assert len(channels) == 2
        assert channels[0].channels == 1
        assert channels[1].channels == 1

        for i in range(100):
            assert channels[0][i] == pytest.approx(0.1, rel=1e-6)
            assert channels[1][i] == pytest.approx(0.9, rel=1e-6)

    def test_interleave(self):
        """Interleave multiple mono buffers."""
        ch0_samples = array.array('f', [0.1] * 100)
        ch1_samples = array.array('f', [0.5] * 100)
        ch2_samples = array.array('f', [0.9] * 100)
        ch0 = pycdp.Buffer.from_memoryview(ch0_samples, channels=1, sample_rate=44100)
        ch1 = pycdp.Buffer.from_memoryview(ch1_samples, channels=1, sample_rate=44100)
        ch2 = pycdp.Buffer.from_memoryview(ch2_samples, channels=1, sample_rate=44100)

        interleaved = pycdp.interleave([ch0, ch1, ch2])
        assert interleaved.channels == 3
        assert interleaved.frame_count == 100

        for i in range(100):
            assert interleaved[i * 3 + 0] == pytest.approx(0.1, rel=1e-6)
            assert interleaved[i * 3 + 1] == pytest.approx(0.5, rel=1e-6)
            assert interleaved[i * 3 + 2] == pytest.approx(0.9, rel=1e-6)

    def test_interleave_empty_raises(self):
        """interleave should fail with empty list."""
        with pytest.raises(ValueError):
            pycdp.interleave([])

    def test_split_interleave_roundtrip(self):
        """Split and interleave should be inverses."""
        original = pycdp.Buffer.create(50, channels=4, sample_rate=48000)
        for i in range(50):
            for ch in range(4):
                original[i * 4 + ch] = (ch + 1) * 0.2

        channels = pycdp.split_channels(original)
        reconstructed = pycdp.interleave(channels)

        assert reconstructed.channels == 4
        assert reconstructed.frame_count == 50

        for i in range(original.sample_count):
            assert reconstructed[i] == pytest.approx(original[i], rel=1e-6)


class TestFileIO:
    """Test file I/O functionality."""

    def test_write_and_read_float(self, tmp_path):
        """Test writing and reading a float WAV file."""
        # Create test data
        samples = array.array('f', [0.5, -0.5, 0.25, -0.25, 0.0])
        buf = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        # Write to file
        path = str(tmp_path / "test_float.wav")
        pycdp.write_file(path, buf, format="float")

        # Read back
        result = pycdp.read_file(path)
        assert result.channels == 1
        assert result.sample_rate == 44100
        assert result.sample_count == 5

        # Verify data
        for i in range(5):
            assert result[i] == pytest.approx(samples[i], rel=1e-6)

    def test_write_and_read_pcm16(self, tmp_path):
        """Test writing and reading a PCM16 WAV file."""
        samples = array.array('f', [0.5, -0.5, 0.25, -0.25, 0.0])
        buf = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        path = str(tmp_path / "test_pcm16.wav")
        pycdp.write_file(path, buf, format="pcm16")

        result = pycdp.read_file(path)
        assert result.channels == 1
        assert result.sample_rate == 44100
        assert result.sample_count == 5

        # PCM16 has limited precision: 1/32768 ~ 3e-5
        tolerance = 2.0 / 32768.0
        for i in range(5):
            assert result[i] == pytest.approx(samples[i], abs=tolerance)

    def test_write_and_read_pcm24(self, tmp_path):
        """Test writing and reading a PCM24 WAV file."""
        samples = array.array('f', [0.5, -0.5, 0.25, -0.25, 0.0])
        buf = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        path = str(tmp_path / "test_pcm24.wav")
        pycdp.write_file(path, buf, format="pcm24")

        result = pycdp.read_file(path)
        assert result.channels == 1
        assert result.sample_rate == 44100
        assert result.sample_count == 5

        # PCM24 has high precision: 1/8388608 ~ 1.2e-7
        tolerance = 2.0 / 8388608.0
        for i in range(5):
            assert result[i] == pytest.approx(samples[i], abs=tolerance)

    def test_stereo_file(self, tmp_path):
        """Test writing and reading a stereo WAV file."""
        # Interleaved stereo: L R L R L R
        samples = array.array('f', [0.5, -0.5, 0.25, -0.25, 0.1, -0.1])
        buf = pycdp.Buffer.from_memoryview(samples, channels=2, sample_rate=48000)

        path = str(tmp_path / "test_stereo.wav")
        pycdp.write_file(path, buf)

        result = pycdp.read_file(path)
        assert result.channels == 2
        assert result.sample_rate == 48000
        assert result.frame_count == 3

        for i in range(6):
            assert result[i] == pytest.approx(samples[i], rel=1e-6)

    def test_read_nonexistent_file_raises(self):
        """Test that reading a nonexistent file raises CDPError."""
        with pytest.raises(pycdp.CDPError):
            pycdp.read_file("/nonexistent/path/file.wav")

    def test_write_invalid_format_raises(self, tmp_path):
        """Test that writing with invalid format raises ValueError."""
        samples = array.array('f', [0.5])
        buf = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        path = str(tmp_path / "test.wav")
        with pytest.raises(ValueError, match="Invalid format"):
            pycdp.write_file(path, buf, format="invalid")


class TestSpectral:
    """Test spectral processing functions."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_time_stretch_double(self, sine_wave):
        """Time stretch should approximately double the length."""
        original_frames = sine_wave.frame_count
        stretched = pycdp.time_stretch(sine_wave, factor=2.0)
        # Should be roughly double length (allow some tolerance for FFT windowing)
        assert stretched.frame_count > original_frames * 1.5
        assert stretched.frame_count < original_frames * 2.5

    def test_time_stretch_half(self, sine_wave):
        """Time stretch with 0.5 should halve the length."""
        original_frames = sine_wave.frame_count
        compressed = pycdp.time_stretch(sine_wave, factor=0.5)
        assert compressed.frame_count > original_frames * 0.3
        assert compressed.frame_count < original_frames * 0.7

    def test_spectral_blur(self, sine_wave):
        """Spectral blur should run without error."""
        blurred = pycdp.spectral_blur(sine_wave, blur_time=0.05)
        # Output should have similar length
        assert blurred.frame_count > sine_wave.frame_count * 0.8
        assert blurred.frame_count < sine_wave.frame_count * 1.2

    def test_speed(self, sine_wave):
        """Speed change should change duration."""
        original_frames = sine_wave.frame_count
        faster = pycdp.modify_speed(sine_wave, speed_factor=2.0)
        # Double speed = half duration
        assert faster.frame_count < original_frames * 0.6

    def test_pitch_shift(self, sine_wave):
        """Pitch shift should maintain similar duration."""
        original_frames = sine_wave.frame_count
        shifted = pycdp.pitch_shift(sine_wave, semitones=5)
        # Duration should be roughly the same (within tolerance for spectral processing)
        assert shifted.frame_count > original_frames * 0.7
        assert shifted.frame_count < original_frames * 1.5

    def test_spectral_shift(self, sine_wave):
        """Spectral shift should run without error."""
        shifted = pycdp.spectral_shift(sine_wave, shift_hz=100)
        assert shifted.frame_count > 0

    def test_spectral_stretch(self, sine_wave):
        """Spectral stretch should run without error."""
        stretched = pycdp.spectral_stretch(sine_wave, max_stretch=1.5)
        assert stretched.frame_count > 0

    def test_filter_lowpass(self, sine_wave):
        """Lowpass filter should run without error."""
        filtered = pycdp.filter_lowpass(sine_wave, cutoff_freq=1000)
        assert filtered.frame_count > 0

    def test_filter_highpass(self, sine_wave):
        """Highpass filter should run without error."""
        filtered = pycdp.filter_highpass(sine_wave, cutoff_freq=200)
        assert filtered.frame_count > 0


class TestEnvelope:
    """Test envelope operations."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_dovetail(self, sine_wave):
        """Dovetail should apply fades."""
        result = pycdp.dovetail(sine_wave, fade_in_dur=0.05, fade_out_dur=0.05)
        assert result.frame_count == sine_wave.frame_count
        # Start should be near zero (faded in)
        assert abs(result[0]) < 0.1
        # End should be near zero (faded out)
        assert abs(result[result.sample_count - 1]) < 0.1

    def test_tremolo(self, sine_wave):
        """Tremolo should run without error."""
        result = pycdp.tremolo(sine_wave, freq=5.0, depth=0.5)
        assert result.frame_count == sine_wave.frame_count

    def test_attack(self, sine_wave):
        """Attack should run without error."""
        result = pycdp.attack(sine_wave, attack_gain=2.0, attack_time=0.1)
        assert result.frame_count == sine_wave.frame_count


class TestDistortion:
    """Test distortion operations."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_distort_overload(self, sine_wave):
        """Overload distortion should run without error."""
        result = pycdp.distort_overload(sine_wave, clip_level=0.3)
        assert result.frame_count > 0

    def test_distort_reverse(self, sine_wave):
        """Reverse cycles should run without error."""
        result = pycdp.distort_reverse(sine_wave, cycle_count=5)
        assert result.frame_count > 0

    def test_distort_fractal(self, sine_wave):
        """Fractal distortion should run without error."""
        result = pycdp.distort_fractal(sine_wave, scaling=1.5)
        assert result.frame_count > 0

    def test_distort_shuffle(self, sine_wave):
        """Shuffle should run without error."""
        result = pycdp.distort_shuffle(sine_wave, chunk_count=10, seed=42)
        assert result.frame_count > 0


class TestReverb:
    """Test reverb."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_reverb(self, sine_wave):
        """Reverb should run without error and add a tail."""
        result = pycdp.reverb(sine_wave, mix=0.5, decay_time=1.0)
        # Should be stereo
        assert result.channels == 2
        # Should have reverb tail
        assert result.frame_count > sine_wave.frame_count


class TestGranular:
    """Test granular operations."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_brassage(self, sine_wave):
        """Brassage should run without error."""
        result = pycdp.brassage(sine_wave, velocity=1.0, density=1.0)
        assert result.frame_count > 0

    def test_brassage_slower(self, sine_wave):
        """Brassage with slow velocity should produce longer output."""
        original_frames = sine_wave.frame_count
        result = pycdp.brassage(sine_wave, velocity=0.5)
        # Slower velocity = longer output
        assert result.frame_count > original_frames

    def test_freeze(self, sine_wave):
        """Freeze should produce specified duration."""
        result = pycdp.freeze(
            sine_wave,
            start_time=0.1,
            end_time=0.2,
            duration=1.0
        )
        # Should be roughly 1 second (44100 samples at 44.1kHz)
        expected_samples = 44100
        assert result.frame_count > expected_samples * 0.9
        assert result.frame_count < expected_samples * 1.1


class TestFilters:
    """Test bandpass and notch filters."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_filter_bandpass(self, sine_wave):
        """Bandpass filter should pass frequencies in range."""
        # 440Hz should pass through 200-800Hz bandpass
        result = pycdp.filter_bandpass(sine_wave, low_freq=200, high_freq=800)
        assert result.frame_count > 0
        # Signal should remain strong since 440Hz is in passband
        # Note: some amplitude loss from STFT/ISTFT process is expected
        peak = max(abs(result[i]) for i in range(result.sample_count))
        assert peak > 0.15  # Significant signal remains

    def test_filter_bandpass_reject(self, sine_wave):
        """Bandpass filter should attenuate out-of-band frequencies."""
        # 440Hz should be attenuated by 1000-2000Hz bandpass
        result = pycdp.filter_bandpass(sine_wave, low_freq=1000, high_freq=2000)
        assert result.frame_count > 0
        # Signal should be reduced
        peak = max(abs(result[i]) for i in range(result.sample_count))
        assert peak < 0.1  # Signal mostly removed

    def test_filter_notch(self, sine_wave):
        """Notch filter should remove narrow frequency band."""
        # Notch at 440Hz should attenuate 440Hz sine
        result = pycdp.filter_notch(sine_wave, center_freq=440, width_hz=100)
        assert result.frame_count > 0
        peak = max(abs(result[i]) for i in range(result.sample_count))
        assert peak < 0.2  # Signal attenuated

    def test_filter_notch_pass(self, sine_wave):
        """Notch filter should pass frequencies outside notch."""
        # Notch at 1000Hz should pass 440Hz sine
        result = pycdp.filter_notch(sine_wave, center_freq=1000, width_hz=100)
        assert result.frame_count > 0
        # Note: some amplitude loss from STFT/ISTFT process is expected
        peak = max(abs(result[i]) for i in range(result.sample_count))
        assert peak > 0.15  # Signal mostly preserved


class TestGate:
    """Test noise gate."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_gate(self, sine_wave):
        """Gate should run without error."""
        result = pycdp.gate(sine_wave, threshold_db=-20.0)
        assert result.frame_count == sine_wave.frame_count

    def test_gate_silence_quiet(self):
        """Gate should silence very quiet audio."""
        # Create very quiet signal
        samples = array.array('f', [0.001 * i / 1000 for i in range(44100)])
        buf = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)
        result = pycdp.gate(buf, threshold_db=-20.0)
        # Most should be gated to zero (threshold is 0.1 amplitude)
        quiet_count = sum(1 for i in range(result.sample_count) if abs(result[i]) < 0.0001)
        assert quiet_count > result.sample_count * 0.5


class TestBitcrush:
    """Test bitcrusher effect."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_bitcrush(self, sine_wave):
        """Bitcrush should run without error."""
        result = pycdp.bitcrush(sine_wave, bit_depth=8)
        assert result.frame_count == sine_wave.frame_count

    def test_bitcrush_downsample(self, sine_wave):
        """Bitcrush with downsample should create staircase effect."""
        result = pycdp.bitcrush(sine_wave, bit_depth=16, downsample=4)
        assert result.frame_count == sine_wave.frame_count
        # With downsample=4, every 4 samples should be the same
        # Check a few sample runs
        same_count = 0
        for i in range(0, result.sample_count - 4, 4):
            if result[i] == result[i + 1] == result[i + 2] == result[i + 3]:
                same_count += 1
        assert same_count > 0  # Should have some repeated samples

    def test_bitcrush_invalid_depth(self, sine_wave):
        """Bitcrush should reject invalid bit depths."""
        with pytest.raises(ValueError):
            pycdp.bitcrush(sine_wave, bit_depth=0)
        with pytest.raises(ValueError):
            pycdp.bitcrush(sine_wave, bit_depth=17)


class TestRingMod:
    """Test ring modulation."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_ring_mod(self, sine_wave):
        """Ring modulation should run without error."""
        result = pycdp.ring_mod(sine_wave, freq=100)
        assert result.frame_count == sine_wave.frame_count

    def test_ring_mod_dry_wet(self, sine_wave):
        """Ring mod with mix=0 should return dry signal."""
        result = pycdp.ring_mod(sine_wave, freq=100, mix=0.0)
        # Should be same as input
        for i in range(result.sample_count):
            assert result[i] == pytest.approx(sine_wave[i], abs=1e-6)


class TestDelay:
    """Test delay effect."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_delay(self, sine_wave):
        """Delay should run without error."""
        result = pycdp.delay(sine_wave, delay_ms=100)
        assert result.frame_count == sine_wave.frame_count

    def test_delay_with_feedback(self, sine_wave):
        """Delay with feedback should run without error."""
        result = pycdp.delay(sine_wave, delay_ms=50, feedback=0.5, mix=0.5)
        assert result.frame_count == sine_wave.frame_count


class TestChorus:
    """Test chorus effect."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_chorus(self, sine_wave):
        """Chorus should run without error."""
        result = pycdp.chorus(sine_wave, rate=1.5, depth_ms=5.0)
        assert result.frame_count == sine_wave.frame_count

    def test_chorus_dry(self, sine_wave):
        """Chorus with mix=0 should return dry signal."""
        result = pycdp.chorus(sine_wave, rate=1.5, depth_ms=5.0, mix=0.0)
        # Should be same as input
        for i in range(result.sample_count):
            assert result[i] == pytest.approx(sine_wave[i], abs=1e-6)


class TestFlanger:
    """Test flanger effect."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_flanger(self, sine_wave):
        """Flanger should run without error."""
        result = pycdp.flanger(sine_wave, rate=0.5, depth_ms=3.0)
        assert result.frame_count == sine_wave.frame_count

    def test_flanger_with_feedback(self, sine_wave):
        """Flanger with feedback should run without error."""
        result = pycdp.flanger(sine_wave, rate=0.3, depth_ms=5.0, feedback=0.7, mix=0.5)
        assert result.frame_count == sine_wave.frame_count

    def test_flanger_dry(self, sine_wave):
        """Flanger with mix=0 should return dry signal."""
        result = pycdp.flanger(sine_wave, rate=0.5, depth_ms=3.0, mix=0.0)
        # Should be same as input
        for i in range(result.sample_count):
            assert result[i] == pytest.approx(sine_wave[i], abs=1e-6)


class TestParametricEQ:
    """Test parametric EQ."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_eq_parametric_boost(self, sine_wave):
        """EQ boost should run without error."""
        result = pycdp.eq_parametric(sine_wave, center_freq=440, gain_db=6.0, q=1.0)
        assert result.frame_count > 0

    def test_eq_parametric_cut(self, sine_wave):
        """EQ cut should reduce signal."""
        result = pycdp.eq_parametric(sine_wave, center_freq=440, gain_db=-12.0, q=1.0)
        assert result.frame_count > 0
        # Signal should be reduced
        peak = max(abs(result[i]) for i in range(result.sample_count))
        assert peak < 0.4  # Less than original 0.5

    def test_eq_parametric_narrow_q(self, sine_wave):
        """Narrow Q should affect less of the spectrum."""
        result = pycdp.eq_parametric(sine_wave, center_freq=440, gain_db=6.0, q=5.0)
        assert result.frame_count > 0


class TestEnvelopeFollow:
    """Test envelope follower."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_envelope_follow_peak(self, sine_wave):
        """Envelope follow (peak) should produce mono envelope."""
        result = pycdp.envelope_follow(sine_wave, attack_ms=1.0, release_ms=50.0, mode="peak")
        assert result.channels == 1
        assert result.frame_count == sine_wave.frame_count
        # Envelope should be non-negative
        for i in range(result.sample_count):
            assert result[i] >= 0

    def test_envelope_follow_rms(self, sine_wave):
        """Envelope follow (RMS) should produce smooth envelope."""
        result = pycdp.envelope_follow(sine_wave, attack_ms=1.0, release_ms=50.0, mode="rms")
        assert result.channels == 1
        assert result.frame_count == sine_wave.frame_count


class TestEnvelopeApply:
    """Test envelope apply."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_envelope_apply(self, sine_wave):
        """Envelope apply should modulate amplitude."""
        # Create a simple ramp envelope
        samples = array.array('f', [i / 22050 for i in range(22050)])
        envelope = pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=44100)

        result = pycdp.envelope_apply(sine_wave, envelope, depth=1.0)
        assert result.frame_count == sine_wave.frame_count

    def test_envelope_apply_zero_depth(self, sine_wave):
        """Envelope apply with depth=0 should not change signal."""
        envelope = pycdp.envelope_follow(sine_wave, attack_ms=1.0, release_ms=50.0)
        result = pycdp.envelope_apply(sine_wave, envelope, depth=0.0)
        # Should be same as input
        for i in range(min(100, result.sample_count)):
            assert result[i] == pytest.approx(sine_wave[i], abs=1e-6)


class TestCompressor:
    """Test compressor."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_compressor(self, sine_wave):
        """Compressor should run without error."""
        result = pycdp.compressor(sine_wave, threshold_db=-10.0, ratio=4.0)
        assert result.frame_count == sine_wave.frame_count

    def test_compressor_reduces_peaks(self, sine_wave):
        """Compressor should reduce peaks above threshold."""
        # Original peak is 0.5 = -6dB
        # Threshold at -10dB should compress
        result = pycdp.compressor(sine_wave, threshold_db=-10.0, ratio=4.0,
                                  attack_ms=0.1, release_ms=50.0)
        result_peak = max(abs(result[i]) for i in range(result.sample_count))
        original_peak = max(abs(sine_wave[i]) for i in range(sine_wave.sample_count))
        # Compressed peak should be lower than original
        assert result_peak < original_peak

    def test_compressor_with_makeup(self, sine_wave):
        """Compressor makeup gain should boost output."""
        result = pycdp.compressor(sine_wave, threshold_db=-10.0, ratio=4.0,
                                  makeup_gain_db=6.0)
        assert result.frame_count == sine_wave.frame_count


class TestLimiter:
    """Test limiter."""

    @pytest.fixture
    def sine_wave(self):
        """Create a simple sine wave buffer for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_limiter(self, sine_wave):
        """Limiter should run without error."""
        result = pycdp.limiter(sine_wave, threshold_db=-6.0)
        assert result.frame_count == sine_wave.frame_count

    def test_limiter_caps_peaks(self, sine_wave):
        """Limiter should cap peaks at threshold."""
        # Original peak is 0.5 = -6dB
        # Limit at -12dB = 0.25
        result = pycdp.limiter(sine_wave, threshold_db=-12.0, attack_ms=0.0, release_ms=50.0)
        result_peak = max(abs(result[i]) for i in range(result.sample_count))
        # Peak should be at or below threshold (0.25)
        assert result_peak <= 0.26  # Small tolerance

    def test_limiter_hard(self, sine_wave):
        """Hard limiter (attack=0) should strictly limit."""
        result = pycdp.limiter(sine_wave, threshold_db=-6.0, attack_ms=0.0, release_ms=50.0)
        threshold_lin = 10 ** (-6.0 / 20.0)  # ~0.5
        # All samples should be at or below threshold
        for i in range(result.sample_count):
            assert abs(result[i]) <= threshold_lin + 0.001


class TestGrainCloud:
    """Test grain cloud generation (CDP: grain)."""

    @pytest.fixture
    def short_sound(self):
        """Create a short sound with amplitude variations for grain detection."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate) *
            (0.5 + 0.5 * math.sin(2 * math.pi * 10 * i / sample_rate))  # AM modulation
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_grain_cloud_runs(self, short_sound):
        """Grain cloud should run without error."""
        result = pycdp.grain_cloud(short_sound, duration=1.0, density=10.0, seed=42)
        assert result.frame_count > 0

    def test_grain_cloud_duration(self, short_sound):
        """Grain cloud should respect duration parameter."""
        duration = 2.0
        result = pycdp.grain_cloud(short_sound, duration=duration, density=10.0, seed=42)
        expected_samples = int(duration * short_sound.sample_rate)
        assert abs(result.frame_count - expected_samples) < 100

    def test_grain_cloud_reproducible(self, short_sound):
        """Same seed should produce same result."""
        result1 = pycdp.grain_cloud(short_sound, duration=0.5, density=10.0, seed=12345)
        result2 = pycdp.grain_cloud(short_sound, duration=0.5, density=10.0, seed=12345)
        # First few samples should be identical
        for i in range(min(100, result1.sample_count, result2.sample_count)):
            assert result1[i] == pytest.approx(result2[i], abs=1e-6)


class TestGrainExtend:
    """Test grain extend (CDP: grainex extend)."""

    @pytest.fixture
    def short_sound(self):
        """Create a short sound for testing."""
        import math
        sample_rate = 44100
        duration = 0.5
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_grain_extend_runs(self, short_sound):
        """Grain extend should run without error."""
        result = pycdp.grain_extend(short_sound, extension=1.0)
        assert result.frame_count > 0

    def test_grain_extend_increases_length(self, short_sound):
        """Grain extend should increase duration."""
        extension = 1.0
        result = pycdp.grain_extend(short_sound, extension=extension)
        # Output should be longer than input by approximately extension amount
        input_duration = short_sound.frame_count / short_sound.sample_rate
        output_duration = result.frame_count / result.sample_rate
        assert output_duration >= input_duration


class TestTextureSimple:
    """Test simple texture generation (CDP: texture SIMPLE_TEX)."""

    @pytest.fixture
    def short_sound(self):
        """Create a short sound for texture source."""
        import math
        sample_rate = 44100
        duration = 0.2
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_texture_simple_runs(self, short_sound):
        """Texture simple should run without error."""
        result = pycdp.texture_simple(short_sound, duration=1.0, density=5.0, seed=42)
        assert result.frame_count > 0

    def test_texture_simple_stereo_output(self, short_sound):
        """Texture simple should produce stereo output."""
        result = pycdp.texture_simple(short_sound, duration=1.0, density=5.0, seed=42)
        assert result.channels == 2

    def test_texture_simple_duration(self, short_sound):
        """Texture simple should respect duration."""
        duration = 2.0
        result = pycdp.texture_simple(short_sound, duration=duration, density=5.0, seed=42)
        expected_samples = int(duration * short_sound.sample_rate)
        assert abs(result.frame_count - expected_samples) < 100


class TestTextureMulti:
    """Test multi-layer texture generation (CDP: texture GROUPS)."""

    @pytest.fixture
    def short_sound(self):
        """Create a short sound for texture source."""
        import math
        sample_rate = 44100
        duration = 0.2
        samples = array.array('f', [
            0.5 * math.sin(2 * math.pi * 440 * i / sample_rate)
            for i in range(int(sample_rate * duration))
        ])
        return pycdp.Buffer.from_memoryview(samples, channels=1, sample_rate=sample_rate)

    def test_texture_multi_runs(self, short_sound):
        """Texture multi should run without error."""
        result = pycdp.texture_multi(short_sound, duration=1.0, density=2.0, seed=42)
        assert result.frame_count > 0

    def test_texture_multi_stereo_output(self, short_sound):
        """Texture multi should produce stereo output."""
        result = pycdp.texture_multi(short_sound, duration=1.0, density=2.0, seed=42)
        assert result.channels == 2

    def test_texture_multi_groups(self, short_sound):
        """Texture multi should support different group sizes."""
        result1 = pycdp.texture_multi(short_sound, duration=1.0, density=2.0, group_size=2, seed=42)
        result2 = pycdp.texture_multi(short_sound, duration=1.0, density=2.0, group_size=8, seed=42)
        # Both should produce output
        assert result1.frame_count > 0
        assert result2.frame_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
