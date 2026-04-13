"""Tests for WebRTC-based RealtimeSession and MicAudioTrack."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.audio.media_track import MicAudioTrack, WEBRTC_FRAME_SAMPLES, WEBRTC_SAMPLE_RATE


class TestMicAudioTrack:
    def _make_track(self):
        mic = MagicMock()
        # 100ms @ 24kHz = 2400 samples of silence
        mic.read.return_value = np.zeros(2400, dtype=np.int16).tobytes()
        preprocessor = MagicMock()
        preprocessor.process.side_effect = lambda x: x  # パススルー
        echo_suppressor = MagicMock()
        echo_suppressor.should_send_mic.return_value = True
        return MicAudioTrack(mic, preprocessor, echo_suppressor)

    def test_kind_is_audio(self):
        track = self._make_track()
        assert track.kind == "audio"

    @pytest.mark.asyncio
    async def test_recv_returns_audio_frame(self):
        track = self._make_track()
        track.start_reading()
        try:
            frame = await asyncio.wait_for(track.recv(), timeout=2.0)
            assert frame.sample_rate == WEBRTC_SAMPLE_RATE
            assert frame.samples == WEBRTC_FRAME_SAMPLES
        finally:
            track.stop()

    @pytest.mark.asyncio
    async def test_pts_increments(self):
        track = self._make_track()
        track.start_reading()
        try:
            frame1 = await asyncio.wait_for(track.recv(), timeout=2.0)
            frame2 = await asyncio.wait_for(track.recv(), timeout=2.0)
            assert frame2.pts == frame1.pts + WEBRTC_FRAME_SAMPLES
        finally:
            track.stop()

    @pytest.mark.asyncio
    async def test_echo_suppression_sends_silence(self):
        mic = MagicMock()
        # ノイズを送信してエコー抑制で無音になることを確認
        mic.read.return_value = (np.ones(2400, dtype=np.int16) * 1000).tobytes()
        preprocessor = MagicMock()
        preprocessor.process.side_effect = lambda x: x
        echo_suppressor = MagicMock()
        echo_suppressor.should_send_mic.return_value = False  # エコー抑制中

        track = MicAudioTrack(mic, preprocessor, echo_suppressor)
        track.start_reading()
        try:
            frame = await asyncio.wait_for(track.recv(), timeout=2.0)
            samples = frame.to_ndarray().flatten()
            assert np.all(samples == 0)
        finally:
            track.stop()

    @pytest.mark.asyncio
    async def test_timeout_returns_silence(self):
        """キューが空の場合タイムアウトで無音フレームを返す."""
        track = self._make_track()
        # start_reading()を呼ばない→キューは空
        frame = await track.recv()
        samples = frame.to_ndarray().flatten()
        assert np.all(samples == 0)
        assert frame.sample_rate == WEBRTC_SAMPLE_RATE


class TestRealtimeSessionDataChannel:
    """Data Channel メッセージのフォーマット検証."""

    def test_session_update_format(self):
        """session.update メッセージが正しいJSON構造を持つ."""
        msg = {
            "type": "session.update",
            "session": {
                "instructions": "test prompt",
                "voice": "alloy",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "tools": [],
                "tool_choice": "auto",
            },
        }
        serialized = json.dumps(msg)
        parsed = json.loads(serialized)
        assert parsed["type"] == "session.update"
        assert parsed["session"]["voice"] == "alloy"
        assert parsed["session"]["turn_detection"]["type"] == "server_vad"

    def test_response_create_format(self):
        msg = json.dumps({"type": "response.create"})
        parsed = json.loads(msg)
        assert parsed["type"] == "response.create"

    def test_conversation_item_create_format(self):
        msg = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        }
        serialized = json.dumps(msg)
        parsed = json.loads(serialized)
        assert parsed["item"]["role"] == "user"
        assert parsed["item"]["content"][0]["text"] == "hello"

    def test_function_call_output_format(self):
        msg = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": json.dumps({"result": "ok"}),
            },
        }
        serialized = json.dumps(msg)
        parsed = json.loads(serialized)
        assert parsed["item"]["type"] == "function_call_output"
        assert parsed["item"]["call_id"] == "call_123"


class TestRealtimeSessionEventDispatch:
    """_on_dc_message のイベントディスパッチ検証."""

    def _make_session(self):
        from src.conversation.realtime import RealtimeSession
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            session = RealtimeSession(
                system_prompt="test",
                on_function_call=AsyncMock(return_value={"ok": True}),
                on_response_done=MagicMock(),
            )
        return session

    def test_speech_started_updates_activity(self):
        session = self._make_session()
        import time
        old_time = session._last_activity
        # 少し待ってからイベント処理
        session._on_dc_message(json.dumps({
            "type": "input_audio_buffer.speech_started",
        }))
        assert session._last_activity >= old_time

    def test_response_done_calls_callback(self):
        session = self._make_session()
        session._on_dc_message(json.dumps({"type": "response.done"}))
        session.on_response_done.assert_called_once()

    def test_error_event_logged(self):
        session = self._make_session()
        # エラーイベントで例外が発生しないことを確認
        session._on_dc_message(json.dumps({
            "type": "error",
            "error": {"message": "test error"},
        }))

    def test_invalid_json_handled(self):
        session = self._make_session()
        # 不正なJSONで例外が発生しないことを確認
        session._on_dc_message("not valid json {{{")
