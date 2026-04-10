"""Audio preprocessing: noise gate, AGC, dynamic range compression.

All processing is numpy-based with zero native dependencies.
Each stage can be independently enabled/disabled via environment variables.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """リアルタイム音声前処理パイプライン."""

    def __init__(self) -> None:
        # ノイズゲート: 閾値以下の音を無音化
        self.noise_gate_enabled = os.environ.get("AUDIO_NOISE_GATE", "true").lower() == "true"
        self.noise_gate_threshold = float(os.environ.get("AUDIO_NOISE_GATE_THRESHOLD", "0.015"))

        # AGC: 目標RMSに正規化
        self.agc_enabled = os.environ.get("AUDIO_AGC", "true").lower() == "true"
        self.agc_target_rms = float(os.environ.get("AUDIO_AGC_TARGET_RMS", "0.1"))
        self.agc_max_gain = float(os.environ.get("AUDIO_AGC_MAX_GAIN", "5.0"))

        # ダイナミックレンジ圧縮: 大きい音を抑制
        self.compressor_enabled = os.environ.get("AUDIO_COMPRESSOR", "true").lower() == "true"
        self.compressor_threshold = float(os.environ.get("AUDIO_COMPRESSOR_THRESHOLD", "0.3"))
        self.compressor_ratio = float(os.environ.get("AUDIO_COMPRESSOR_RATIO", "4.0"))

        logger.info(
            "Audio preprocessor: noise_gate=%s (threshold=%.3f), "
            "agc=%s (target_rms=%.2f), compressor=%s (threshold=%.2f, ratio=%.1f)",
            self.noise_gate_enabled, self.noise_gate_threshold,
            self.agc_enabled, self.agc_target_rms,
            self.compressor_enabled, self.compressor_threshold, self.compressor_ratio,
        )

    def process(self, pcm_data: bytes) -> bytes:
        """PCM16音声チャンクを前処理して返す."""
        audio = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        if self.noise_gate_enabled:
            audio = self._noise_gate(audio)

        if self.agc_enabled:
            audio = self._agc(audio)

        if self.compressor_enabled:
            audio = self._compress(audio)

        # クリッピング防止
        audio = np.clip(audio, -1.0, 1.0)
        return (audio * 32767).astype(np.int16).tobytes()

    def _noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """閾値以下の音を無音化."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < self.noise_gate_threshold:
            return np.zeros_like(audio)
        return audio

    def _agc(self, audio: np.ndarray) -> np.ndarray:
        """自動音量調整: 目標RMSに近づける."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-6:
            return audio  # 無音はスキップ
        gain = self.agc_target_rms / rms
        gain = min(gain, self.agc_max_gain)  # 過大増幅を防止
        return audio * gain

    def _compress(self, audio: np.ndarray) -> np.ndarray:
        """ダイナミックレンジ圧縮: 閾値を超えた部分を圧縮."""
        threshold = self.compressor_threshold
        ratio = self.compressor_ratio

        abs_audio = np.abs(audio)
        mask = abs_audio > threshold
        if not np.any(mask):
            return audio

        # 閾値超え部分のみ圧縮
        over = abs_audio[mask] - threshold
        compressed = threshold + over / ratio
        result = audio.copy()
        result[mask] = np.sign(audio[mask]) * compressed
        return result
