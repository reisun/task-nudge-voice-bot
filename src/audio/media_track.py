"""Custom aiortc MediaStreamTrack that feeds preprocessed mic audio to WebRTC."""

import asyncio
import fractions
import logging
import threading

import av
import numpy as np
from aiortc import MediaStreamTrack

from src.audio.base import AudioInput
from src.audio.preprocessing import AudioPreprocessor
from src.audio.resampler import resample_24k_to_48k

logger = logging.getLogger(__name__)

WEBRTC_SAMPLE_RATE = 48000
WEBRTC_FRAME_SAMPLES = 960  # 20ms @ 48kHz
FRAMES_PER_READ = 5  # 100ms mic read → 5 x 20ms WebRTC frames


class MicAudioTrack(MediaStreamTrack):
    """PyAudio mic input → preprocessed → resampled → WebRTC audio frames."""

    kind = "audio"

    def __init__(
        self,
        mic: AudioInput,
        preprocessor: AudioPreprocessor,
    ) -> None:
        super().__init__()
        self._mic = mic
        self._preprocessor = preprocessor
        self._queue: asyncio.Queue[av.AudioFrame] = asyncio.Queue(maxsize=20)
        self._pts = 0  # reader thread のみが更新
        self._fallback_pts = 0  # recv タイムアウト時のみ使用 (event loop thread)
        self._time_base = fractions.Fraction(1, WEBRTC_SAMPLE_RATE)
        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start_reading(self) -> None:
        """Start the background mic reader thread."""
        self._loop = asyncio.get_event_loop()
        self._stop_event.clear()
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="mic-reader",
        )
        self._reader_thread.start()

    def _read_loop(self) -> None:
        """Background thread: read mic → preprocess → resample → enqueue frames."""
        silence_48k = np.zeros(WEBRTC_FRAME_SAMPLES, dtype=np.int16)
        while not self._stop_event.is_set():
            try:
                pcm_data = self._mic.read()
            except Exception:
                if self._stop_event.is_set():
                    break
                logger.warning("Mic read error", exc_info=True)
                continue

            processed = self._preprocessor.process(pcm_data)
            resampled = resample_24k_to_48k(processed)
            samples_48k = np.frombuffer(resampled, dtype=np.int16)

            for i in range(FRAMES_PER_READ):
                start = i * WEBRTC_FRAME_SAMPLES
                end = start + WEBRTC_FRAME_SAMPLES
                chunk = samples_48k[start:end] if end <= len(samples_48k) else silence_48k

                frame = av.AudioFrame.from_ndarray(
                    chunk.reshape(1, -1), format="s16", layout="mono",
                )
                frame.sample_rate = WEBRTC_SAMPLE_RATE
                frame.pts = self._pts
                frame.time_base = self._time_base
                self._pts += WEBRTC_FRAME_SAMPLES

                try:
                    self._loop.call_soon_threadsafe(self._enqueue_frame, frame)
                except RuntimeError:
                    pass  # イベントループ終了時

    def _enqueue_frame(self, frame: av.AudioFrame) -> None:
        """イベントループ上で安全にフレームをキューに投入."""
        if self._queue.full():
            try:
                self._queue.get_nowait()  # 古いフレームを破棄
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(frame)

    async def recv(self) -> av.AudioFrame:
        """Called by aiortc to pull the next audio frame."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            # タイムアウト時は無音フレームを返してWebRTCの継続性を維持
            silence = np.zeros(WEBRTC_FRAME_SAMPLES, dtype=np.int16)
            frame = av.AudioFrame.from_ndarray(
                silence.reshape(1, -1), format="s16", layout="mono",
            )
            frame.sample_rate = WEBRTC_SAMPLE_RATE
            frame.pts = self._fallback_pts
            frame.time_base = self._time_base
            self._fallback_pts += WEBRTC_FRAME_SAMPLES
            return frame

    def stop(self) -> None:
        """Stop the mic reader and clean up."""
        self._stop_event.set()
        super().stop()
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
