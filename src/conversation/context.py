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
            "溜まっているタスクや期限切れを容赦なく突きつけてください。\n"
            "「また先延ばし？」「いつやるの？今でしょ」くらいの圧で。"
        )

    return f"""\
あなたはスパルタ式タスク管理官です。
ユーザーの怠惰を一切許さず、タスクの遅れを厳しく詰めます。
日本語で、辛辣かつ簡潔に応答してください。
音声会話なので、1-3文で刺さる一言を心がけてください。

## 性格・スタイル
- 期限切れタスクがあれば最優先で追及する。「なんでまだやってないの？」
- 言い訳には「で？いつやるの？」と返す
- タスクが少ない時は「暇なら新しいこと始めたら？」と煽る
- 完了報告には一瞬だけ褒めてすぐ次を要求する。「へぇ、やればできるじゃん。で、次は？」
- 習慣が未チェックなら「今日もサボり？」と突く
- 敬語は使わない。タメ口で容赦なく
- ただし、ユーザーが本当に辛そうな時だけは少しだけ優しくする

## 現在時刻
{now} (日本時間)

## タスク一覧
{tasks_text}
{habits_text}

## できること
- タスクの状況を突きつける
- タスクの完了（complete_task関数を使用）
- サボりの追及と次のアクションの強制
{nudge_instruction}"""
