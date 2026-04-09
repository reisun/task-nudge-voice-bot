"""task-nudge-voice-bot — 音声タスクアシスタント."""

import asyncio
import logging
import os
import threading
from functools import partial

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger(__name__)

from src.audio.pyaudio_io import PyAudioInput, PyAudioOutput
from src.audio.activation import create_activation
from src.conversation.context import build_system_prompt
from src.conversation.realtime import RealtimeSession
from src.conversation.tools import (
    get_tool_definitions, handle_function_call, refresh_task_list,
)
from src.ticktick.client import TickTickClient
from src.ticktick.habits import get_habits
from src.scheduler.nudge import create_scheduler


class VoiceBot:
    """メインアプリケーション."""

    CACHE_INTERVAL_SEC = 300  # 5分ごとにタスクを再取得

    def __init__(self) -> None:
        self.ticktick = TickTickClient()
        self.activation = create_activation()
        self.mic = PyAudioInput()
        self.speaker = PyAudioOutput()
        self._session: RealtimeSession | None = None
        self._listening = False
        self._session_lock = asyncio.Lock()
        # タスクキャッシュ
        self._cached_categorized: dict = {}
        self._cached_habits: list[dict] = []

    def _fetch_context(self) -> tuple[dict, list[dict]]:
        """TickTickからタスクと習慣を取得してキャッシュ更新."""
        try:
            self._cached_categorized = self.ticktick.get_categorized_tasks()
            self._cached_habits = get_habits()
            refresh_task_list(self._cached_categorized)
            logger.info("Task cache updated")
        except Exception:
            logger.exception("Failed to refresh task cache")
        return self._cached_categorized, self._cached_habits

    async def _periodic_cache(self) -> None:
        """バックグラウンドで定期的にタスクキャッシュを更新."""
        loop = asyncio.get_event_loop()
        while True:
            await asyncio.sleep(self.CACHE_INTERVAL_SEC)
            await loop.run_in_executor(None, self._fetch_context)

    async def start_conversation(self, nudge: bool = False) -> None:
        """音声会話セッションを開始."""
        async with self._session_lock:
            if self._session:
                logger.info("Session already active, skipping")
                return

            logger.info("Starting conversation (nudge=%s)", nudge)
            # キャッシュ済みのデータを使う（即座に開始）
            categorized = self._cached_categorized
            habits = self._cached_habits
            system_prompt = build_system_prompt(categorized, habits, nudge=nudge)

            session = RealtimeSession(
                system_prompt=system_prompt,
                tools=get_tool_definitions(),
                on_audio=self._play_audio,
                on_function_call=partial(
                    handle_function_call, ticktick=self.ticktick,
                ),
            )
            await session.connect()
            self._session = session

        # 定時通知モードならAIに先に喋らせる
        if nudge:
            await self._session.trigger_response(
                "定時通知です。今のタスク状況を教えてください。"
            )

        # 音声送信とイベント受信を並行実行
        try:
            await asyncio.gather(
                self._stream_audio(),
                self._session.listen(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            await self._stop_session()

    async def _stream_audio(self) -> None:
        """マイクからの音声をRealtimeセッションに送信."""
        self._listening = True
        loop = asyncio.get_event_loop()
        try:
            while self._listening and self._session:
                audio = await loop.run_in_executor(None, self.mic.read)
                if self._session:
                    await self._session.send_audio(audio)
        except Exception:
            logger.exception("Audio streaming error")

    def _play_audio(self, data: bytes) -> None:
        """Realtime APIからの音声をスピーカーで再生."""
        try:
            self.speaker.write(data)
        except Exception:
            logger.warning("Audio playback error", exc_info=True)

    async def _stop_session(self) -> None:
        """セッションを終了."""
        self._listening = False
        async with self._session_lock:
            if self._session:
                await self._session.close()
                self._session = None
                logger.info("Conversation ended")

    def _on_nudge(self) -> None:
        """スケジューラーから呼ばれる定時通知コールバック."""
        logger.info("Scheduled nudge triggered")
        asyncio.run_coroutine_threadsafe(
            self.start_conversation(nudge=True),
            self._loop,
        )

    async def run(self) -> None:
        """メインループ: ウェイクワード/PTT検出 → 会話開始."""
        self._loop = asyncio.get_event_loop()

        # 起動時にタスクを先読み
        logger.info("Fetching initial task data...")
        await self._loop.run_in_executor(None, self._fetch_context)

        # アクティベーションとスケジューラーのセットアップ
        self.activation.setup()
        scheduler = create_scheduler(self._on_nudge)
        scheduler.start()

        # バックグラウンドでキャッシュ更新を開始
        cache_task = asyncio.create_task(self._periodic_cache())

        logger.info("Voice bot started. Waiting for activation...")

        self.mic.open()
        self.speaker.open()

        try:
            while True:
                # マイクから音声を読んでウェイクワード/PTT検出
                audio = await asyncio.get_event_loop().run_in_executor(
                    None, self.mic.read,
                )
                if self.activation.detect(audio):
                    logger.info("Activation detected!")
                    await self.start_conversation(nudge=False)
                    logger.info("Conversation finished. Waiting for activation...")
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            cache_task.cancel()
            scheduler.shutdown(wait=False)
            self.mic.close()
            self.speaker.close()


def main():
    bot = VoiceBot()
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
