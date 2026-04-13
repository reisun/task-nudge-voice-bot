"""24kHz / 48kHz sample-rate conversion for WebRTC (Opus) <-> PyAudio (PCM16)."""

import numpy as np


def resample_24k_to_48k(pcm16_bytes: bytes) -> bytes:
    """24kHz PCM16 mono -> 48kHz PCM16 mono (linear interpolation)."""
    samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
    n = len(samples)
    if n == 0:
        return b""
    # 線形補間で2倍にアップサンプリング
    out = np.empty(n * 2, dtype=np.float32)
    out[0::2] = samples
    out[1::2] = np.append(
        (samples[:-1] + samples[1:]) / 2,
        samples[-1],
    )
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


def resample_48k_to_24k(pcm16_bytes: bytes) -> bytes:
    """48kHz PCM16 mono -> 24kHz PCM16 mono (LPF + decimation)."""
    samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32)
    n = len(samples)
    if n == 0:
        return b""
    # 2タップ移動平均 LPF でエイリアシング軽減してからデシメーション
    if n % 2 != 0:
        samples = samples[:n - 1]
    averaged = (samples[0::2] + samples[1::2]) / 2
    return np.clip(averaged, -32768, 32767).astype(np.int16).tobytes()
