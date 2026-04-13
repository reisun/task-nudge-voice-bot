"""Cross-correlation based echo detection.

Compares mic input against a ring buffer of recently played audio.
High correlation → echo (AI voice looping back) → suppress.
Low correlation → user voice → pass through.
Preserves barge-in capability.
"""

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)


class EchoDetector:
    """再生音との相互相関でエコーを検出し抑制."""

    def __init__(
        self,
        sample_rate: int = 24000,
        buffer_sec: float | None = None,
        threshold: float | None = None,
        suppression_db: float | None = None,
    ) -> None:
        self._enabled = os.environ.get("ECHO_DETECT_ENABLED", "true").lower() == "true"
        buf_sec = buffer_sec if buffer_sec is not None else float(os.environ.get("ECHO_DETECT_BUFFER_SEC", "1.0"))
        self._threshold = threshold if threshold is not None else float(os.environ.get("ECHO_DETECT_THRESHOLD", "0.6"))
        sup_db = suppression_db if suppression_db is not None else float(os.environ.get("ECHO_DETECT_SUPPRESSION_DB", "-30"))

        self._buffer_size = int(sample_rate * buf_sec)
        self._buffer = np.zeros(self._buffer_size, dtype=np.float32)
        self._write_pos = 0
        self._suppression_gain = 10 ** (sup_db / 20)  # dB → linear
        self._lock = threading.Lock()

        logger.info(
            "EchoDetector: enabled=%s, buffer=%.1fs (%d samples), "
            "threshold=%.2f, suppression=%ddB",
            self._enabled, buf_sec, self._buffer_size,
            self._threshold, int(sup_db),
        )

    def feed_playback(self, pcm_data: bytes) -> None:
        """スピーカー出力をリングバッファに記録."""
        if not self._enabled:
            return
        samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        n = len(samples)
        with self._lock:
            if n >= self._buffer_size:
                self._buffer[:] = samples[-self._buffer_size:]
                self._write_pos = 0
            elif self._write_pos + n <= self._buffer_size:
                self._buffer[self._write_pos:self._write_pos + n] = samples
                self._write_pos += n
            else:
                first = self._buffer_size - self._write_pos
                self._buffer[self._write_pos:] = samples[:first]
                self._buffer[:n - first] = samples[first:]
                self._write_pos = n - first

    def process(self, pcm_data: bytes) -> bytes:
        """マイク入力をエコー判定し、エコーなら減衰して返す."""
        if not self._enabled:
            return pcm_data

        mic = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        mic_norm = np.linalg.norm(mic)
        if mic_norm < 1e-6:
            return pcm_data  # 無音はそのまま

        with self._lock:
            ref = self._buffer.copy()
            wp = self._write_pos

        # リングバッファを連続化 (write_pos を先頭に並べ替え)
        ref = np.roll(ref, -wp)
        ref_energy = np.sum(ref ** 2)
        if ref_energy < 1e-6:
            return pcm_data  # 再生バッファが空（AIがまだ喋っていない）

        # 正規化相互相関のピークを計算
        peak_corr = self._compute_peak_correlation(ref, mic, mic_norm)

        if peak_corr > self._threshold:
            logger.debug("Echo detected (corr=%.3f)", peak_corr)
            suppressed = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32)
            suppressed *= self._suppression_gain
            return suppressed.astype(np.int16).tobytes()

        return pcm_data

    def _compute_peak_correlation(
        self, ref: np.ndarray, mic: np.ndarray, mic_norm: float,
    ) -> float:
        """正規化相互相関のピーク値を計算."""
        n_mic = len(mic)
        n_ref = len(ref)
        if n_ref < n_mic:
            return 0.0

        # 相互相関 (valid mode: refの中でmicをスライド)
        corr = np.correlate(ref, mic, mode="valid")

        # 各ラグ位置のウィンドウnormを効率計算 (累積二乗和)
        cumsum_sq = np.concatenate(([0.0], np.cumsum(ref ** 2)))
        window_energy = cumsum_sq[n_mic:] - cumsum_sq[:n_ref - n_mic + 1]
        window_norm = np.sqrt(np.maximum(window_energy, 1e-12))

        # 正規化
        ncc = corr / (window_norm * mic_norm)
        return float(np.max(np.abs(ncc)))
