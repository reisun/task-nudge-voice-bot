"""タスク情報からsystem promptを生成する."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _load_prompt(name: str, default: str = "") -> str:
    """prompts/ ディレクトリからテキストファイルを読み込む."""
    path = _PROMPTS_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return default

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

    persona = _load_prompt("persona.txt", "あなたは音声タスクアシスタントです。")
    capabilities = _load_prompt("capabilities.txt", "- タスクの状況説明や優先順位の相談")

    nudge_instruction = ""
    if nudge:
        nudge_text = _load_prompt("nudge.txt", "タスクの状況を簡潔に伝えてください。")
        nudge_instruction = f"\n\n## 定時通知モード\n{nudge_text}"

    return f"""\
{persona}

## 現在時刻
{now} (日本時間)

## タスク一覧
{tasks_text}
{habits_text}

## できること
{capabilities}
{nudge_instruction}"""
