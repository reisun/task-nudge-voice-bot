"""tools.py のタスク検索・function callロジックのテスト."""

import pytest

from src.conversation.tools import (
    find_task, refresh_task_list, get_tool_definitions, _all_tasks,
)
import src.conversation.tools as tools_module


class TestFindTask:

    def setup_method(self):
        self.tasks = [
            {"title": "買い物", "id": "1", "_project_id": "p1"},
            {"title": "レポート提出", "id": "2", "_project_id": "p1"},
            {"title": "ジョギング", "id": "3", "_project_id": "p2"},
        ]
        tools_module._all_tasks = list(self.tasks)

    def teardown_method(self):
        tools_module._all_tasks = []

    def test_find_by_number(self):
        assert find_task("1")["title"] == "買い物"

    def test_find_by_number_last(self):
        assert find_task("3")["title"] == "ジョギング"

    def test_number_out_of_range(self):
        assert find_task("0") is None
        assert find_task("4") is None

    def test_find_by_name(self):
        assert find_task("レポート")["title"] == "レポート提出"

    def test_case_insensitive(self):
        tools_module._all_tasks = [{"title": "Test Task", "id": "1", "_project_id": "p1"}]
        assert find_task("test")["title"] == "Test Task"

    def test_no_match(self):
        assert find_task("存在しない") is None

    def test_empty_hint_single(self):
        tools_module._all_tasks = [self.tasks[0]]
        assert find_task("")["title"] == "買い物"

    def test_empty_hint_multiple(self):
        assert find_task("") is None


class TestRefreshTaskList:

    def test_flattens_categories(self):
        categorized = {
            "overdue": [{"title": "A"}],
            "today": [{"title": "B"}, {"title": "C"}],
            "week": [{"title": "D"}],
            "no_date": [],
        }
        refresh_task_list(categorized)
        assert [t["title"] for t in tools_module._all_tasks] == ["A", "B", "C", "D"]

    def test_empty(self):
        refresh_task_list({"overdue": [], "today": [], "week": [], "no_date": []})
        assert tools_module._all_tasks == []


class TestGetToolDefinitions:

    def test_returns_tools(self):
        tools = get_tool_definitions()
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert "complete_task" in names
        assert "list_tasks" in names
