# task-nudge-voice-bot

TickTickのタスク情報をもとに、音声で会話しながらタスク管理をサポートするBot。

## 機能

- TickTickからタスク情報を定期取得
- OpenAI Realtime APIによるリアルタイム音声会話
- 定時にBotが先に喋りかけてNudge
- ウェイクワード / Push-to-talk で会話開始
- 音声コマンドでタスク完了操作

## セットアップ (uv — 推奨)

```powershell
# uv をインストール (Windows PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# 環境変数を設定
cp .env.example .env
# .env を編集

# 起動 (Python 3.11 を自動取得)
uv run task-nudge-voice
```

### Linux / Raspberry Pi

```bash
# uv をインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# PortAudio が必要
sudo apt install portaudio19-dev

# 起動
uv run task-nudge-voice
```

### Docker

```bash
docker compose up
```

## 設定

| 環境変数 | 説明 | デフォルト |
|---|---|---|
| ACTIVATION_MODE | `wakeword` or `pushtotalk` | `wakeword` |
| OPENAI_VOICE | 音声 (alloy, echo, shimmer等) | `alloy` |
| NUDGE_START_HOUR | 定時通知の開始時刻 (JST) | `7` |
| NUDGE_END_HOUR | 定時通知の終了時刻 (JST) | `23` |
