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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
