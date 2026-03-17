"""Todo tools for structured task management."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


def _session_key_to_path(workspace: Path, session_key: str) -> Path:
    """Convert session key to a safe filesystem path."""
    safe = session_key.replace(":", "_").replace("/", "_").replace("\\", "_")
    return workspace / "todos" / f"{safe}.json"


def _load_todos(path: Path) -> list[dict]:
    """Load todos from JSON file, returns empty list if file doesn't exist."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_todos(path: Path, todos: list[dict]) -> None:
    """Save todos to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(todos, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_status(status: str) -> str:
    """Return emoji icon for status."""
    return {
        "pending": "⏳",
        "in_progress": "🔄",
        "completed": "✅",
        "deleted": "🗑️",
    }.get(status, status)


class AddTodoTool(Tool):
    """Add a new task to the todo list."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._session_key = ""

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the session context."""
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "add_todo"

    @property
    def description(self) -> str:
        return (
            "Add a task to the todo list. Use before starting multi-step work "
            "to track progress. Each task should represent a meaningful unit of work."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Short task title in imperative form (e.g. 'Refactor auth module')",
                    "maxLength": 100,
                },
                "description": {
                    "type": "string",
                    "description": "Optional: detailed description of what needs to be done",
                },
                "blocked_by": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: list of task IDs that must be completed first",
                },
            },
            "required": ["subject"],
        }

    async def execute(
        self,
        subject: str,
        description: str = "",
        blocked_by: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        if not self._session_key:
            return "Error: no session context"

        path = _session_key_to_path(self._workspace, self._session_key)
        todos = _load_todos(path)

        task_id = "t-" + uuid.uuid4().hex[:6]
        now = datetime.now(timezone.utc).isoformat()

        task = {
            "id": task_id,
            "subject": subject,
            "description": description,
            "status": "pending",
            "blocked_by": blocked_by or [],
            "created_at": now,
            "updated_at": now,
        }
        todos.append(task)
        _save_todos(path, todos)

        return f"Task added: [{task_id}] {subject}"


class UpdateTodoTool(Tool):
    """Update the status of a task."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._session_key = ""

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the session context."""
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "update_todo"

    @property
    def description(self) -> str:
        return (
            "Update a task's status. "
            "Set to 'in_progress' when starting, 'completed' when done, 'deleted' if no longer needed."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task ID to update (e.g. 't-a1b2c3')",
                },
                "status": {
                    "type": "string",
                    "enum": ["in_progress", "completed", "deleted"],
                    "description": "New status for the task",
                },
            },
            "required": ["task_id", "status"],
        }

    async def execute(self, task_id: str, status: str, **kwargs: Any) -> str:
        if not self._session_key:
            return "Error: no session context"

        path = _session_key_to_path(self._workspace, self._session_key)
        todos = _load_todos(path)

        for task in todos:
            if task["id"] == task_id:
                task["status"] = status
                task["updated_at"] = datetime.now(timezone.utc).isoformat()
                _save_todos(path, todos)
                icon = _format_status(status)
                return f"Task {task_id} updated to {icon} {status}"

        return f"Error: task '{task_id}' not found"


class ListTodosTool(Tool):
    """List all tasks in the current session."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._session_key = ""

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the session context."""
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "list_todos"

    @property
    def description(self) -> str:
        return "List all tasks and their statuses. Use at the start of a session to check for unfinished work."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "filter": {
                    "type": "string",
                    "enum": ["all", "active", "completed"],
                    "description": "Filter tasks: 'all' (default), 'active' (pending/in_progress), 'completed'",
                }
            },
        }

    async def execute(self, filter: str = "all", **kwargs: Any) -> str:
        if not self._session_key:
            return "Error: no session context"

        path = _session_key_to_path(self._workspace, self._session_key)
        todos = _load_todos(path)

        if filter == "active":
            todos = [t for t in todos if t["status"] in ("pending", "in_progress")]
        elif filter == "completed":
            todos = [t for t in todos if t["status"] == "completed"]
        else:  # "all" - exclude deleted
            todos = [t for t in todos if t["status"] != "deleted"]

        if not todos:
            return "No tasks found."

        lines = []
        for t in todos:
            icon = _format_status(t["status"])
            blocked = (
                f" [blocked by: {', '.join(t['blocked_by'])}]"
                if t.get("blocked_by")
                else ""
            )
            lines.append(f"{icon} [{t['id']}] {t['subject']}{blocked}")

        return "\n".join(lines)


class GetTodoTool(Tool):
    """Get details of a specific task."""

    def __init__(self, workspace: Path):
        self._workspace = workspace
        self._session_key = ""

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the session context."""
        self._session_key = f"{channel}:{chat_id}"

    @property
    def name(self) -> str:
        return "get_todo"

    @property
    def description(self) -> str:
        return "Get full details of a specific task including description and dependencies."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "The task ID to retrieve",
                }
            },
            "required": ["task_id"],
        }

    async def execute(self, task_id: str, **kwargs: Any) -> str:
        if not self._session_key:
            return "Error: no session context"

        path = _session_key_to_path(self._workspace, self._session_key)
        todos = _load_todos(path)

        for task in todos:
            if task["id"] == task_id:
                icon = _format_status(task["status"])
                lines = [
                    f"ID: {task['id']}",
                    f"Status: {icon} {task['status']}",
                    f"Subject: {task['subject']}",
                ]
                if task.get("description"):
                    lines.append(f"Description: {task['description']}")
                if task.get("blocked_by"):
                    lines.append(f"Blocked by: {', '.join(task['blocked_by'])}")
                lines.append(f"Created: {task['created_at']}")
                lines.append(f"Updated: {task['updated_at']}")
                return "\n".join(lines)

        return f"Error: task '{task_id}' not found"
