"""context.py のフォーマット・prompt生成ロジックのテスト."""

from src.conversation.context import format_tasks, format_habits, build_system_prompt


class TestFormatTasks:

    def test_basic(self):
        categorized = {
            "overdue": [{"title": "期限切れ", "dueDate": "2026-04-08"}],
            "today": [{"title": "今日やる", "dueDate": "2026-04-09"}],
            "week": [],
            "no_date": [],
        }
        result = format_tasks(categorized)
        assert "【期限切れ】" in result
        assert "1. 期限切れ" in result
        assert "【今日】" in result
        assert "2. 今日やる" in result

    def test_empty(self):
        categorized = {"overdue": [], "today": [], "week": [], "no_date": []}
        assert format_tasks(categorized) == ""

    def test_no_date_task(self):
        categorized = {"overdue": [], "today": [], "week": [],
                       "no_date": [{"title": "期限なし"}]}
        result = format_tasks(categorized)
        assert "【期限未設定】" in result
        assert "(期限:" not in result

    def test_sequential_numbering(self):
        categorized = {
            "overdue": [{"title": "A", "dueDate": "2026-04-07"}],
            "today": [{"title": "B", "dueDate": "2026-04-09"},
                      {"title": "C", "dueDate": "2026-04-09"}],
            "week": [], "no_date": [],
        }
        result = format_tasks(categorized)
        assert "1. A" in result
        assert "2. B" in result
        assert "3. C" in result


class TestFormatHabits:

    def test_checked(self):
        result = format_habits([{"name": "運動", "checked_today": True}])
        assert "[done]" in result
        assert "運動" in result

    def test_unchecked(self):
        result = format_habits([{"name": "読書", "checked_today": False}])
        assert "[not yet]" in result

    def test_missing_checked(self):
        result = format_habits([{"name": "瞑想"}])
        assert "[not yet]" in result


class TestBuildSystemPrompt:

    def test_contains_task_info(self):
        categorized = {
            "overdue": [], "today": [{"title": "テスト書く", "dueDate": "2026-04-09"}],
            "week": [], "no_date": [],
        }
        prompt = build_system_prompt(categorized)
        assert "テスト書く" in prompt
        assert "音声タスクアシスタント" in prompt

    def test_nudge_mode(self):
        categorized = {"overdue": [], "today": [], "week": [], "no_date": []}
        prompt = build_system_prompt(categorized, nudge=True)
        assert "定時通知モード" in prompt

    def test_no_nudge_mode(self):
        categorized = {"overdue": [], "today": [], "week": [], "no_date": []}
        prompt = build_system_prompt(categorized, nudge=False)
        assert "定時通知モード" not in prompt

    def test_with_habits(self):
        categorized = {"overdue": [], "today": [], "week": [], "no_date": []}
        habits = [{"name": "運動", "checked_today": True}]
        prompt = build_system_prompt(categorized, habits=habits)
        assert "運動" in prompt
        assert "【習慣】" in prompt
