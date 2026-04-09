"""activation.py のPush-to-talkロジックのテスト.

WakeWordActivationはopenwakewordに依存するため、
PushToTalkActivationのロジックのみテストする。
"""

from src.audio.activation import PushToTalkActivation, create_activation


class TestPushToTalkActivation:

    def test_initial_state_inactive(self):
        ptt = PushToTalkActivation()
        assert ptt.is_active is False

    def test_toggle_on(self):
        ptt = PushToTalkActivation()
        ptt._toggle()
        assert ptt.is_active is True

    def test_toggle_off(self):
        ptt = PushToTalkActivation()
        ptt._toggle()
        ptt._toggle()
        assert ptt.is_active is False

    def test_detect_when_active(self):
        ptt = PushToTalkActivation()
        ptt._toggle()
        assert ptt.detect(b"\x00" * 100) is True

    def test_detect_when_inactive(self):
        ptt = PushToTalkActivation()
        assert ptt.detect(b"\x00" * 100) is False

    def test_deactivate(self):
        ptt = PushToTalkActivation()
        ptt._toggle()
        ptt.deactivate()
        assert ptt.is_active is False


class TestCreateActivation:

    def test_pushtotalk_mode(self):
        act = create_activation("pushtotalk")
        assert isinstance(act, PushToTalkActivation)

    def test_wakeword_mode_returns_wakeword(self):
        # WakeWordActivationのインスタンス生成は可能（setupしなければ外部依存なし）
        from src.audio.activation import WakeWordActivation
        act = create_activation("wakeword")
        assert isinstance(act, WakeWordActivation)
