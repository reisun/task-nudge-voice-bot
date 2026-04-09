"""Acoustic Echo Cancellation using SpeexDSP."""

import collections
import logging
import threading

from speexdsp import EchoCanceller

logger = logging.getLogger(__name__)

# SpeexDSP の frame_size はサンプル数（バイト数ではない）
# 24kHz, 16bit mono → 1サンプル = 2バイト
DEFAULT_FRAME_SAMPLES = 480  # 20ms at 24kHz
DEFAULT_FILTER_LENGTH = 4800  # 200ms（反響の長さに合わせて調整）
DEFAULT_RATE = 24000


class EchoCanceller24k:
    """24kHz対応のエコーキャンセラー.

    スピーカーに送る音声を参照信号として保持し、
    マイク入力からエコー成分を除去する。
    """

    def __init__(self, frame_samples: int = DEFAULT_FRAME_SAMPLES,
                 filter_length: int = DEFAULT_FILTER_LENGTH,
                 rate: int = DEFAULT_RATE) -> None:
        self.frame_samples = frame_samples
        self.frame_bytes = frame_samples * 2  # 16bit = 2 bytes/sample
        self._ec = EchoCanceller.create(frame_samples, filter_length, rate)
        # スピーカー参照信号のバッファ（スレッドセーフ）
        self._ref_buffer = collections.deque(maxlen=50)
        self._lock = threading.Lock()
        self._silence = b'\x00' * self.frame_bytes
        logger.info("Echo canceller initialized (frame=%d samples, filter=%d, rate=%d)",
                     frame_samples, filter_length, rate)

    def feed_speaker(self, data: bytes) -> None:
        """スピーカーに送る音声を参照バッファに追加."""
        # フレームサイズ単位に分割して格納
        with self._lock:
            offset = 0
            while offset + self.frame_bytes <= len(data):
                self._ref_buffer.append(data[offset:offset + self.frame_bytes])
                offset += self.frame_bytes

    def process(self, mic_data: bytes) -> bytes:
        """マイク入力からエコーを除去して返す."""
        result = bytearray()
        offset = 0
        while offset + self.frame_bytes <= len(mic_data):
            mic_frame = mic_data[offset:offset + self.frame_bytes]
            with self._lock:
                if self._ref_buffer:
                    ref_frame = self._ref_buffer.popleft()
                else:
                    ref_frame = self._silence
            clean = self._ec.process(mic_frame, ref_frame)
            result.extend(clean)
            offset += self.frame_bytes
        return bytes(result)
