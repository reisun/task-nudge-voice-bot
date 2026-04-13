"""Tests for audio resampler (24kHz <-> 48kHz)."""

import numpy as np
import pytest

from src.audio.resampler import resample_24k_to_48k, resample_48k_to_24k


class TestResample24kTo48k:
    def test_output_length_doubles(self):
        samples = np.array([100, 200, 300, 400], dtype=np.int16)
        result = resample_24k_to_48k(samples.tobytes())
        out = np.frombuffer(result, dtype=np.int16)
        assert len(out) == 8

    def test_original_samples_preserved(self):
        samples = np.array([1000, 2000, 3000], dtype=np.int16)
        result = resample_24k_to_48k(samples.tobytes())
        out = np.frombuffer(result, dtype=np.int16)
        # 偶数インデックスに元のサンプルが保持される
        assert out[0] == 1000
        assert out[2] == 2000
        assert out[4] == 3000

    def test_interpolated_values(self):
        samples = np.array([0, 1000], dtype=np.int16)
        result = resample_24k_to_48k(samples.tobytes())
        out = np.frombuffer(result, dtype=np.int16)
        # out[0]=0, out[1]=500(補間), out[2]=1000, out[3]=1000(末尾複製)
        assert out[1] == 500

    def test_empty_input(self):
        assert resample_24k_to_48k(b"") == b""

    def test_typical_chunk_size(self):
        """100ms @ 24kHz = 2400 samples -> 4800 samples @ 48kHz."""
        samples = np.zeros(2400, dtype=np.int16)
        result = resample_24k_to_48k(samples.tobytes())
        out = np.frombuffer(result, dtype=np.int16)
        assert len(out) == 4800


class TestResample48kTo24k:
    def test_output_length_halves(self):
        samples = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=np.int16)
        result = resample_48k_to_24k(samples.tobytes())
        out = np.frombuffer(result, dtype=np.int16)
        assert len(out) == 4

    def test_averaging(self):
        samples = np.array([100, 200, 300, 400], dtype=np.int16)
        result = resample_48k_to_24k(samples.tobytes())
        out = np.frombuffer(result, dtype=np.int16)
        # (100+200)/2=150, (300+400)/2=350
        assert out[0] == 150
        assert out[1] == 350

    def test_empty_input(self):
        assert resample_48k_to_24k(b"") == b""

    def test_odd_length_truncates(self):
        """奇数長は最後のサンプルを切り捨てる."""
        samples = np.array([100, 200, 300], dtype=np.int16)
        result = resample_48k_to_24k(samples.tobytes())
        out = np.frombuffer(result, dtype=np.int16)
        assert len(out) == 1
        assert out[0] == 150

    def test_typical_frame_size(self):
        """960 samples @ 48kHz (20ms) -> 480 samples @ 24kHz."""
        samples = np.zeros(960, dtype=np.int16)
        result = resample_48k_to_24k(samples.tobytes())
        out = np.frombuffer(result, dtype=np.int16)
        assert len(out) == 480


class TestRoundTrip:
    def test_length_preserved(self):
        """24k -> 48k -> 24k でサンプル数が一致する."""
        rng = np.random.default_rng(42)
        original = (rng.random(2400) * 20000 - 10000).astype(np.int16)
        upsampled = resample_24k_to_48k(original.tobytes())
        roundtrip = resample_48k_to_24k(upsampled)
        restored = np.frombuffer(roundtrip, dtype=np.int16)
        assert len(restored) == len(original)

    def test_low_freq_signal_preserved(self):
        """低周波信号は補間・デシメーションで十分保持される."""
        # 1kHz正弦波 @ 24kHz (ナイキスト以下)
        t = np.arange(2400) / 24000.0
        original = (np.sin(2 * np.pi * 1000 * t) * 10000).astype(np.int16)
        upsampled = resample_24k_to_48k(original.tobytes())
        roundtrip = resample_48k_to_24k(upsampled)
        restored = np.frombuffer(roundtrip, dtype=np.int16)
        mean_diff = np.mean(np.abs(original.astype(np.int32) - restored.astype(np.int32)))
        assert mean_diff < 500
