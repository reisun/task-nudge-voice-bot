"""OpenAI Realtime API session management via WebRTC."""

import asyncio
import json
import logging
import os
import time

import aiohttp
import numpy as np
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

from src.audio.media_track import MicAudioTrack
from src.audio.resampler import resample_48k_to_24k

logger = logging.getLogger(__name__)

OPENAI_REALTIME_BASE = "https://api.openai.com/v1/realtime"
OPENAI_MODEL = "gpt-4o-realtime-preview"


class RealtimeSession:
    """OpenAI Realtime APIとの音声会話セッション (WebRTC)."""

    def __init__(
        self,
        system_prompt: str,
        tools: list[dict] | None = None,
        on_audio: callable = None,
        on_function_call: callable = None,
        on_response_done: callable = None,
        mic_track: MicAudioTrack | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.on_audio = on_audio
        self.on_function_call = on_function_call
        self.on_response_done = on_response_done
        self._mic_track = mic_track
        self._api_key = os.environ["OPENAI_API_KEY"]
        self._voice = os.environ.get("OPENAI_VOICE", "alloy")
        self._idle_timeout = int(os.environ.get("SESSION_IDLE_TIMEOUT", "10"))
        self._last_activity = time.monotonic()
        self.timed_out = False
        self._pc: RTCPeerConnection | None = None
        self._dc = None  # RTCDataChannel
        self._dc_ready = asyncio.Event()
        self._session_configured = asyncio.Event()
        self._closed = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    async def _create_ephemeral_token(self) -> str:
        """エフェメラルトークンを取得."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OPENAI_REALTIME_BASE}/sessions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": OPENAI_MODEL, "voice": self._voice},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["client_secret"]["value"]

    async def connect(self) -> None:
        """WebRTC接続を確立しセッションを設定."""
        token = await self._create_ephemeral_token()

        ice_config = RTCConfiguration(iceServers=[
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        ])
        self._pc = RTCPeerConnection(configuration=ice_config)

        # マイク音声トラック追加
        if self._mic_track:
            self._pc.addTrack(self._mic_track)
            self._mic_track.start_reading()

        # Data Channel作成 (イベント送受信用)
        self._dc = self._pc.createDataChannel("oai-events", ordered=True)
        self._dc.on("open", self._on_dc_open)
        self._dc.on("message", self._on_dc_message)

        # リモート音声トラック受信ハンドラ
        @self._pc.on("track")
        def on_track(track):
            logger.info("Remote track received: kind=%s", track.kind)
            if track.kind == "audio":
                task = asyncio.ensure_future(self._receive_remote_audio(track))
                self._tasks.append(task)

        # SDP offer作成・送信
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OPENAI_REALTIME_BASE}?model={OPENAI_MODEL}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/sdp",
                },
                data=self._pc.localDescription.sdp,
            ) as resp:
                resp.raise_for_status()
                answer_sdp = await resp.text()

        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
        await self._pc.setRemoteDescription(answer)

        # Data Channelが開きsession.updatedを受信するまで待機
        try:
            await asyncio.wait_for(self._session_configured.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.error("Session configuration timeout")
            await self.close()
            raise ConnectionError("WebRTC session configuration failed")

        logger.info("Realtime session connected via WebRTC (voice=%s)", self._voice)

    def _on_dc_open(self) -> None:
        """Data Channel開通時にセッション設定を送信."""
        session_config = {
            "instructions": self.system_prompt,
            "voice": self._voice,
            "turn_detection": {
                "type": "server_vad",
                "threshold": float(os.environ.get("VAD_THRESHOLD", "0.5")),
                "prefix_padding_ms": int(os.environ.get("VAD_PREFIX_PADDING_MS", "300")),
                "silence_duration_ms": int(os.environ.get("VAD_SILENCE_DURATION_MS", "500")),
            },
        }
        if self.tools:
            session_config["tools"] = self.tools
            session_config["tool_choice"] = "auto"

        self._dc.send(json.dumps({
            "type": "session.update",
            "session": session_config,
        }))
        self._dc_ready.set()
        logger.debug("Data channel open, session.update sent")

    def _on_dc_message(self, message: str) -> None:
        """Data Channelからのイベントを処理."""
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from data channel: %s", message[:200])
            return

        event_type = event.get("type", "")

        if event_type == "input_audio_buffer.speech_started":
            self._touch()
            logger.info("Speech started (user speaking)")

        elif event_type == "response.audio.delta":
            # WebRTCでは通常発生しないが、念のため記録
            logger.info("Unexpected response.audio.delta via data channel")

        elif event_type == "response.function_call_arguments.done":
            self._touch()
            logger.info("Function call: %s", event.get("name"))
            if self.on_function_call:
                asyncio.ensure_future(self._handle_function_call(event))

        elif event_type == "response.done":
            self._touch()
            logger.info("Response done")
            if self.on_response_done:
                self.on_response_done()

        elif event_type == "error":
            logger.error("Realtime API error: %s", event)

        elif event_type == "session.created":
            logger.debug("Session created: %s", event.get("session", {}).get("id", ""))

        elif event_type == "session.updated":
            self._session_configured.set()
            logger.debug("Session updated")

    async def _handle_function_call(self, event: dict) -> None:
        """Function callを処理して結果を返送."""
        try:
            result = await self.on_function_call(
                event["name"], json.loads(event["arguments"]), event["call_id"],
            )
            if result is not None and self._dc:
                self._dc.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": event["call_id"],
                        "output": json.dumps(result),
                    },
                }))
                self._dc.send(json.dumps({"type": "response.create"}))
        except Exception:
            logger.exception("Function call handling error")

    async def _receive_remote_audio(self, track) -> None:
        """リモート音声トラックからOpenAIの音声を受信・再生."""
        logger.info("Starting remote audio receiver")
        frame_count = 0
        try:
            while not self._closed.is_set():
                try:
                    frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                frame_count += 1
                if frame_count == 1:
                    logger.info(
                        "First audio frame: format=%s, layout=%s, "
                        "sample_rate=%d, samples=%d, dtype=%s",
                        frame.format.name, frame.layout.name,
                        frame.sample_rate, frame.samples,
                        frame.to_ndarray().dtype,
                    )

                self._touch()

                if self.on_audio:
                    raw = frame.to_ndarray()
                    if raw.ndim > 1:
                        raw = raw[0]
                    if raw.dtype in (np.float32, np.float64):
                        raw = np.clip(raw * 32767, -32768, 32767)
                    pcm16_48k = raw.astype(np.int16).tobytes()
                    pcm16_24k = resample_48k_to_24k(pcm16_48k)
                    self.on_audio(pcm16_24k)
        except Exception:
            if not self._closed.is_set():
                logger.warning("Remote audio track ended", exc_info=True)
        finally:
            logger.info("Remote audio receiver stopped (frames=%d)", frame_count)

    async def send_audio(self, pcm_data: bytes) -> None:
        """互換性のためのno-op. WebRTCではMicAudioTrackが自動送信."""
        pass

    async def trigger_response(self, nudge_message: str | None = None) -> None:
        """AIに応答を開始させる（定時通知や会話開始時）."""
        if not self._dc:
            return
        if nudge_message:
            self._dc.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": nudge_message}],
                },
            }))
        self._dc.send(json.dumps({"type": "response.create"}))

    def _touch(self) -> None:
        """アクティビティタイムスタンプを更新."""
        self._last_activity = time.monotonic()

    async def _idle_watchdog(self) -> None:
        """無音タイムアウトを監視し、超過したらセッションを終了."""
        while not self._closed.is_set():
            await asyncio.sleep(5)
            idle = time.monotonic() - self._last_activity
            if idle >= self._idle_timeout:
                logger.info("Session idle for %ds, closing", int(idle))
                self.timed_out = True
                await self.close()
                return

    async def listen(self) -> None:
        """Data Channelイベントとリモート音声の受信を待機."""
        if not self._pc:
            return
        watchdog = asyncio.create_task(self._idle_watchdog())
        self._tasks.append(watchdog)
        try:
            await self._closed.wait()
        finally:
            watchdog.cancel()

    async def close(self) -> None:
        """セッションを閉じる."""
        if self._closed.is_set():
            return
        self._closed.set()

        if self._mic_track:
            self._mic_track.stop()

        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

        if self._pc:
            try:
                await self._pc.close()
            except Exception:
                logger.debug("PeerConnection close error", exc_info=True)
            self._pc = None
        self._dc = None
        logger.info("Realtime session closed")
