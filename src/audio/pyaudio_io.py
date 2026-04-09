"""PyAudio-based audio I/O implementation (PC / Raspberry Pi共通)."""

import pyaudio

from .base import AudioInput, AudioOutput

# OpenAI Realtime API requirements
RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 2400  # 100ms at 24kHz


class PyAudioInput(AudioInput):
    """PyAudioを使ったマイク入力."""

    def __init__(self, rate: int = RATE, channels: int = CHANNELS,
                 chunk: int = CHUNK) -> None:
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None

    def open(self) -> None:
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=FORMAT,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

    def read(self, frames: int = 0) -> bytes:
        if not self._stream:
            raise RuntimeError("Stream not open")
        return self._stream.read(frames or self.chunk, exception_on_overflow=False)

    def close(self) -> None:
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None


class PyAudioOutput(AudioOutput):
    """PyAudioを使ったスピーカー出力."""

    def __init__(self, rate: int = RATE, channels: int = CHANNELS) -> None:
        self.rate = rate
        self.channels = channels
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None

    def open(self) -> None:
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=FORMAT,
            channels=self.channels,
            rate=self.rate,
            output=True,
        )

    def write(self, data: bytes) -> None:
        if not self._stream:
            raise RuntimeError("Stream not open")
        self._stream.write(data)

    def close(self) -> None:
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None
