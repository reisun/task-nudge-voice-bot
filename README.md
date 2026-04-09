# task-nudge-voice-bot

TickTickのタスク情報をもとに、音声で会話しながらタスク管理をサポートするBot。

## 機能

- TickTickからタスク情報を定期取得
- OpenAI Realtime APIによるリアルタイム音声会話
- 定時にBotが先に喋りかけてNudge
- ウェイクワード / Push-to-talk で会話開始
- 音声コマンドでタスク完了操作

## セットアップ

```bash
# 依存インストール (portaudioが必要)
sudo apt install portaudio19-dev python3-dev
pip install -r requirements.txt

# 環境変数を設定
cp .env.example .env
# .env を編集

# 起動
python -m src.main
```

## 設定

| 環境変数 | 説明 | デフォルト |
|---|---|---|
| ACTIVATION_MODE | `wakeword` or `pushtotalk` | `wakeword` |
| OPENAI_VOICE | 音声 (alloy, echo, shimmer等) | `alloy` |
| NUDGE_START_HOUR | 定時通知の開始時刻 (JST) | `7` |
| NUDGE_END_HOUR | 定時通知の終了時刻 (JST) | `23` |
