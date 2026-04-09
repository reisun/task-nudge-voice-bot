"""Realtime API用のfunction calling定義とハンドラー."""

import logging

from src.ticktick.client import TickTickClient

logger = logging.getLogger(__name__)

# タスクのフラットリスト（カテゴリ順）
_all_tasks: list[dict] = []

CATEGORY_ORDER = ["overdue", "today", "week", "no_date"]


def refresh_task_list(categorized: dict[str, list[dict]]) -> None:
    """カテゴリ別タスクからフラットリストを更新."""
    global _all_tasks
    _all_tasks = [t for cat in CATEGORY_ORDER for t in categorized.get(cat, [])]


def get_tool_definitions() -> list[dict]:
    """Realtime APIに渡すtool定義."""
    return [
        {
            "type": "function",
            "name": "complete_task",
            "description": (
                "タスクを完了にする。番号（1始まり）またはタスク名の一部で指定。"
                "例: '1' で1番目のタスクを完了、'買い物' で名前に '買い物' を含むタスクを完了。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "hint": {
                        "type": "string",
                        "description": "タスク番号またはタスク名の一部",
                    }
                },
                "required": ["hint"],
            },
        },
        {
            "type": "function",
            "name": "list_tasks",
            "description": "現在のタスク一覧を取得する。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    ]


def find_task(hint: str) -> dict | None:
    """ヒント文字列からタスクを検索."""
    if not hint:
        return _all_tasks[0] if len(_all_tasks) == 1 else None
    if hint.isdigit():
        idx = int(hint) - 1
        if 0 <= idx < len(_all_tasks):
            return _all_tasks[idx]
    for task in _all_tasks:
        if hint.lower() in task.get("title", "").lower():
            return task
    return None


async def handle_function_call(name: str, arguments: dict, call_id: str,
                               ticktick: TickTickClient) -> dict:
    """function callを処理して結果を返す."""
    if name == "complete_task":
        hint = arguments.get("hint", "")
        task = find_task(hint)
        if not task:
            return {"success": False, "message": f"タスクが見つかりません: {hint}"}
        try:
            ticktick.complete_task(task["_project_id"], task["id"])
            title = task.get("title", "(no title)")
            logger.info("Completed task: %s", title)
            # リストから除去
            _all_tasks.remove(task)
            return {"success": True, "message": f"タスク「{title}」を完了しました"}
        except Exception as e:
            logger.exception("Failed to complete task")
            return {"success": False, "message": f"完了に失敗しました: {e}"}

    elif name == "list_tasks":
        if not _all_tasks:
            return {"tasks": [], "message": "タスクはありません"}
        task_list = []
        for i, t in enumerate(_all_tasks, 1):
            due = t.get("dueDate", "")
            task_list.append({
                "number": i,
                "title": t.get("title", "(no title)"),
                "due": due[:10] if due else "期限なし",
            })
        return {"tasks": task_list}

    return {"error": f"Unknown function: {name}"}
