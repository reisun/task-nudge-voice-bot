"""タスク情報からsystem promptを生成する."""

from datetime import datetime
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")

_LABEL_MAP = {
    "overdue": "【期限切れ】",
    "today": "【今日】",
    "week": "【今週】",
    "no_date": "【期限未設定】",
}

_CATEGORY_ORDER = ["overdue", "today", "week", "no_date"]


def format_tasks(categorized: dict[str, list[dict]]) -> str:
    """カテゴリ別タスクリストを文字列化."""
    lines = []
    idx = 1
    for cat_key in _CATEGORY_ORDER:
        tasks = categorized.get(cat_key, [])
        if not tasks:
            continue
        lines.append(f"\n{_LABEL_MAP[cat_key]}")
        for t in tasks:
            due = t.get("dueDate", "")
            due_suffix = f" (期限: {due[:10]})" if due else ""
            lines.append(f"{idx}. {t.get('title', '(no title)')}{due_suffix}")
            idx += 1
    return "\n".join(lines)


def format_habits(habits: list[dict]) -> str:
    """習慣リストを文字列化."""
    lines = ["\n\n【習慣】"]
    for h in habits:
        checked = h.get("checked_today", False)
        mark = "done" if checked else "not yet"
        lines.append(f"- {h.get('name', '(no name)')} [{mark}]")
    return "\n".join(lines)


def build_system_prompt(categorized: dict[str, list[dict]],
                        habits: list[dict] | None = None,
                        nudge: bool = False) -> str:
    """Realtime APIに渡すsystem promptを構築."""
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")

    tasks_text = format_tasks(categorized)
    if not tasks_text.strip():
        tasks_text = "タスクはありません。"

    habits_text = ""
    if habits:
        habits_text = format_habits(habits)

    nudge_instruction = ""
    if nudge:
        nudge_instruction = (
            "\n\n## 定時通知モード\n"
            "今は定時通知の時間です。まずあなたから話しかけてください。\n"
            "タスクの状況を簡潔に伝え、優先順位や進め方について軽く提案してください。\n"
            "押し付けがましくならず、やる気が出るような声かけを心がけてください。"
        )

    return f"""\
あなたは音声タスクアシスタントです。
ユーザーのタスク管理をサポートし、自然な会話で手助けします。
日本語で、カジュアルで簡潔な口調で応答してください。
音声会話なので、1-3文程度の短い応答を心がけてください。

## 現在時刻
{now} (日本時間)

## タスク一覧
{tasks_text}
{habits_text}

## できること
- タスクの状況説明や優先順位の相談
- タスクの完了（complete_task関数を使用）
- 励ましやモチベーション支援
{nudge_instruction}"""
