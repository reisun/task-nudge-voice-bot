"""Echo suppression via mute-while-playing.

SpeexDSP AEC はビルド依存が重いため、AI再生中にマイク送信を
抑制するシンプルな方式で代替する。
AI再生終了後に短いクールダウンを設けることで残響もカバーする。
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

DEFAULT_COOLDOWN_SEC = 0.3  # 再生終了後のミュート継続時間


class EchoSuppressor:
    """AI再生中のマイク送信を抑制するエコー対策."""

    def __init__(self, cooldown: float = DEFAULT_COOLDOWN_SEC) -> None:
        self._cooldown = cooldown
        self._playing = False
        self._last_play_time: float = 0
        self._lock = threading.Lock()

    def on_play_start(self) -> None:
        """AI音声の再生が始まった時に呼ぶ."""
        with self._lock:
            self._playing = True

    def on_play_data(self) -> None:
        """AI音声データが来るたびに呼ぶ."""
        with self._lock:
            self._playing = True
            self._last_play_time = time.monotonic()

    def on_play_end(self) -> None:
        """AI音声の再生が終わった時に呼ぶ."""
        with self._lock:
            self._playing = False
            self._last_play_time = time.monotonic()

    def should_send_mic(self) -> bool:
        """マイク音声をAPIに送信してよいかを返す."""
        with self._lock:
            if self._playing:
                return False
            # クールダウン期間中も抑制
            if self._last_play_time > 0:
                elapsed = time.monotonic() - self._last_play_time
                if elapsed < self._cooldown:
                    return False
            return True
