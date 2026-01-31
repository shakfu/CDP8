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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
