"""Activation modes: wake word detection and push-to-talk."""

import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)


class WakeWordActivation:
    """OpenWakeWordを使ったウェイクワード検出."""

    def __init__(self, sensitivity: float = 0.5) -> None:
        self.sensitivity = sensitivity
        self._model = None

    def setup(self) -> None:
        from openwakeword.model import Model
        # inference_framework="onnx" を指定して tflite 依存を回避
        self._model = Model(inference_framework="onnx")
        logger.info("Wake word model loaded")

    def detect(self, audio_chunk: bytes, rate: int = 24000) -> bool:
        """PCM16音声チャンクからウェイクワードを検出."""
        if not self._model:
            return False
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
        # OpenWakeWord expects 16kHz; resample if needed
        if rate != 16000:
            factor = rate // 16000
            audio_np = audio_np[::factor]
        prediction = self._model.predict(audio_np)
        for model_name, score in prediction.items():
            if score > self.sensitivity:
                logger.info("Wake word detected: %s (score=%.2f)", model_name, score)
                self._model.reset()
                return True
        return False


class PushToTalkActivation:
    """キーボードホットキーによるPush-to-talk."""

    def __init__(self, hotkey: str = "ctrl+space") -> None:
        self.hotkey = hotkey
        self._active = False
        self._lock = threading.Lock()

    def setup(self) -> None:
        import keyboard
        keyboard.add_hotkey(self.hotkey, self._toggle)
        logger.info("Push-to-talk hotkey registered: %s", self.hotkey)

    def _toggle(self) -> None:
        with self._lock:
            self._active = not self._active
            state = "ON" if self._active else "OFF"
            logger.info("Push-to-talk: %s", state)

    @property
    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def deactivate(self) -> None:
        with self._lock:
            self._active = False

    def detect(self, audio_chunk: bytes, rate: int = 24000) -> bool:
        """ホットキーが押されている間は常にTrue."""
        return self.is_active


def create_activation(mode: str | None = None):
    """環境変数に基づいてアクティベーションモードを生成."""
    mode = mode or os.environ.get("ACTIVATION_MODE", "wakeword")
    if mode == "pushtotalk":
        hotkey = os.environ.get("PTT_HOTKEY", "ctrl+space")
        return PushToTalkActivation(hotkey=hotkey)
    else:
        sensitivity = float(os.environ.get("WAKEWORD_SENSITIVITY", "0.5"))
        return WakeWordActivation(sensitivity=sensitivity)
