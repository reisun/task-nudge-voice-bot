"""OpenAI Realtime API session management."""

import asyncio
import base64
import json
import logging
import os
import time

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class RealtimeSession:
    """OpenAI Realtime APIとの音声会話セッション."""

    def __init__(self, system_prompt: str, tools: list[dict] | None = None,
                 on_audio: callable = None,
                 on_function_call: callable = None,
                 on_response_done: callable = None) -> None:
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.on_audio = on_audio
        self.on_function_call = on_function_call
        self.on_response_done = on_response_done
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._connection = None
        self._voice = os.environ.get("OPENAI_VOICE", "alloy")
        self._idle_timeout = int(os.environ.get("SESSION_IDLE_TIMEOUT", "10"))
        self._last_activity = time.monotonic()
        self.timed_out = False

    async def connect(self) -> None:
        """Realtime APIに接続しセッションを設定."""
        self._connection = await self._client.beta.realtime.connect(
            model="gpt-4o-realtime-preview",
        ).__aenter__()

        session_config = {
            "instructions": self.system_prompt,
            "voice": self._voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
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

        await self._connection.session.update(session=session_config)
        logger.info("Realtime session connected (voice=%s)", self._voice)

    async def send_audio(self, pcm_data: bytes) -> None:
        """PCM16音声データをAPIに送信."""
        if not self._connection:
            return
        try:
            encoded = base64.b64encode(pcm_data).decode()
            await self._connection.input_audio_buffer.append(audio=encoded)
        except Exception:
            pass  # セッション終了中の送信エラーは無視

    async def trigger_response(self, nudge_message: str | None = None) -> None:
        """AIに応答を開始させる（定時通知や会話開始時）."""
        if not self._connection:
            return
        if nudge_message:
            await self._connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": nudge_message}],
                }
            )
        await self._connection.response.create()

    def _touch(self) -> None:
        """アクティビティタイムスタンプを更新."""
        self._last_activity = time.monotonic()

    async def _idle_watchdog(self) -> None:
        """無音タイムアウトを監視し、超過したらセッションを終了."""
        while self._connection:
            await asyncio.sleep(5)
            idle = time.monotonic() - self._last_activity
            if idle >= self._idle_timeout:
                logger.info("Session idle for %ds, closing", int(idle))
                self.timed_out = True
                await self.close()
                return

    async def listen(self) -> None:
        """サーバーからのイベントを受信し処理する."""
        if not self._connection:
            return
        # アイドル監視を並行起動
        watchdog = asyncio.create_task(self._idle_watchdog())
        try:
            async for event in self._connection:
                event_type = event.type

                if event_type == "response.audio.delta":
                    self._touch()
                    if self.on_audio:
                        audio_bytes = base64.b64decode(event.delta)
                        self.on_audio(audio_bytes)

                elif event_type == "input_audio_buffer.speech_started":
                    self._touch()

                elif event_type == "response.function_call_arguments.done":
                    self._touch()
                    if self.on_function_call:
                        result = await self.on_function_call(
                            event.name, json.loads(event.arguments), event.call_id,
                        )
                        if result is not None and self._connection:
                            await self._connection.conversation.item.create(
                                item={
                                    "type": "function_call_output",
                                    "call_id": event.call_id,
                                    "output": json.dumps(result),
                                }
                            )
                            await self._connection.response.create()

                elif event_type == "response.done":
                    self._touch()
                    if self.on_response_done:
                        self.on_response_done()

                elif event_type == "error":
                    logger.error("Realtime API error: %s", event)
        except Exception:
            logger.warning("Realtime connection lost", exc_info=True)
        finally:
            watchdog.cancel()

    async def close(self) -> None:
        """セッションを閉じる."""
        if self._connection:
            try:
                await self._connection.close()
            except Exception:
                logger.debug("Connection close error (already closed)", exc_info=True)
            self._connection = None
            logger.info("Realtime session closed")
