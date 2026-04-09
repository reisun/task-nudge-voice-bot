"""Audio I/O abstraction layer."""

import abc


class AudioInput(abc.ABC):
    """マイク入力の抽象クラス."""

    @abc.abstractmethod
    def open(self) -> None:
        """ストリームを開く."""

    @abc.abstractmethod
    def read(self, frames: int) -> bytes:
        """PCMデータを読み取る."""

    @abc.abstractmethod
    def close(self) -> None:
        """ストリームを閉じる."""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()


class AudioOutput(abc.ABC):
    """スピーカー出力の抽象クラス."""

    @abc.abstractmethod
    def open(self) -> None:
        """ストリームを開く."""

    @abc.abstractmethod
    def write(self, data: bytes) -> None:
        """PCMデータを再生する."""

    @abc.abstractmethod
    def close(self) -> None:
        """ストリームを閉じる."""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
