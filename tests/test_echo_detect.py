"""Tests for cross-correlation echo detection."""

import threading
from unittest.mock import patch

import numpy as np
import pytest

from src.audio.echo_detect import EchoDetector


def _make_detector(**kwargs):
    defaults = {"sample_rate": 24000, "buffer_sec": 1.0, "threshold": 0.6}
    defaults.update(kwargs)
    with patch.dict("os.environ", {"ECHO_DETECT_ENABLED": "true"}):
        return EchoDetector(**defaults)


def _sin_wave(freq: float, duration_sec: float, rate: int = 24000) -> bytes:
    """テスト用正弦波を PCM16 bytes で生成."""
    t = np.arange(int(rate * duration_sec)) / rate
    samples = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    return samples.tobytes()


def _noise(duration_sec: float, rate: int = 24000, seed: int = 42) -> bytes:
    """テスト用ランダムノイズを PCM16 bytes で生成."""
    rng = np.random.default_rng(seed)
    n = int(rate * duration_sec)
    samples = (rng.random(n) * 20000 - 10000).astype(np.int16)
    return samples.tobytes()


class TestFeedPlayback:
    def test_fills_buffer(self):
        det = _make_detector(buffer_sec=0.5)  # 12000 samples
        data = _sin_wave(440, 0.3)  # 7200 samples
        det.feed_playback(data)
        assert det._write_pos == 7200

    def test_wraps_around(self):
        det = _make_detector(buffer_sec=0.5)  # 12000 samples
        det.feed_playback(_sin_wave(440, 0.4))  # 9600 samples
        det.feed_playback(_sin_wave(440, 0.4))  # 9600 more → wraps
        assert det._write_pos == (9600 + 9600) % 12000

    def test_large_input_fills_entire_buffer(self):
        det = _make_detector(buffer_sec=0.5)  # 12000 samples
        det.feed_playback(_sin_wave(440, 1.0))  # 24000 > buffer
        assert det._write_pos == 0  # reset


class TestProcess:
    def test_empty_buffer_passes_through(self):
        """再生バッファが空ならそのまま通す."""
        det = _make_detector()
        mic_data = _noise(0.1)
        result = det.process(mic_data)
        assert result == mic_data

    def test_echo_detected_and_suppressed(self):
        """再生音のコピー（遅延付き）はエコーとして検出・減衰される."""
        det = _make_detector(threshold=0.5)
        # AI の音声を再生
        playback = _sin_wave(1000, 0.5)
        det.feed_playback(playback)

        # 同じ音がマイクに回り込む (100ms分を切り出し)
        echo_chunk = _sin_wave(1000, 0.1)
        result = det.process(echo_chunk)

        # 減衰されていること (元より小さい)
        orig_rms = np.sqrt(np.mean(
            np.frombuffer(echo_chunk, dtype=np.int16).astype(np.float32) ** 2
        ))
        result_rms = np.sqrt(np.mean(
            np.frombuffer(result, dtype=np.int16).astype(np.float32) ** 2
        ))
        assert result_rms < orig_rms * 0.5

    def test_user_voice_passes_through(self):
        """再生音と異なるユーザーの声はそのまま通す."""
        det = _make_detector(threshold=0.5)
        # AI は 1000Hz を再生
        det.feed_playback(_sin_wave(1000, 0.5))

        # ユーザーはランダムノイズ (再生音と無相関)
        user_voice = _noise(0.1, seed=99)
        result = det.process(user_voice)
        assert result == user_voice

    def test_silence_passes_through(self):
        """無音入力はそのまま通す."""
        det = _make_detector()
        det.feed_playback(_sin_wave(440, 0.5))
        silence = np.zeros(2400, dtype=np.int16).tobytes()
        result = det.process(silence)
        assert result == silence

    def test_disabled_passes_through(self):
        """ECHO_DETECT_ENABLED=false なら全てそのまま通す."""
        with patch.dict("os.environ", {"ECHO_DETECT_ENABLED": "false"}):
            det = EchoDetector()
        playback = _sin_wave(1000, 0.5)
        det.feed_playback(playback)
        echo = _sin_wave(1000, 0.1)
        result = det.process(echo)
        assert result == echo


class TestThreadSafety:
    def test_concurrent_access_no_crash(self):
        """feed_playback と process の並行アクセスでクラッシュしない."""
        det = _make_detector()
        errors = []

        def writer():
            try:
                for _ in range(50):
                    det.feed_playback(_sin_wave(440, 0.05))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    det.process(_noise(0.1))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0
