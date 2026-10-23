"""Productivity tools for Jarvis - calendar, reminders, notes."""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path


from .base import Tool, async_read_file, async_write_file

logger = logging.getLogger(__name__)


class NotesTool(Tool):
    """Tool for managing notes."""

    name = "notes"
    description = "Create, read, update, search, and manage notes"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "read", "list", "search", "update", "delete"],
                "description": "Action to perform",
            },
            "title": {
                "type": "string",
                "description": "Note title (required for create, read, update, delete actions)",
            },
            "content": {
                "type": "string",
                "description": "Note content (for create and update actions)",
            },
            "query": {
                "type": "string",
                "description": "Search query (for search action) - searches titles and content",
            },
        },
        "required": ["action"],
    }
    requires_permission = False

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "notes"
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_note_path(self, title: str) -> Path:
        """Get path for a note."""
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        return self.storage_path / f"{safe_title}.json"

    async def execute(
        self,
        action: str,
        title: str = "",
        content: str = "",
        query: str = "",
        **kwargs
    ) -> str:
        """Execute notes action."""
        if action == "create":
            if not title or not title.strip():
                return "Error: Please provide a title for the note. Usage: notes action=create title=\"Your Title\" content=\"Your content\""
            note = {
                "title": title,
                "content": content or "",
                "created": datetime.now().isoformat(),
                "modified": datetime.now().isoformat(),
            }
            note_path = self._get_note_path(title)
            await async_write_file(note_path, json.dumps(note, indent=2))
            return f"Created note: {title}"

        elif action == "read":
            if not title or not title.strip():
                return "Error: Please provide a title to read. Usage: notes action=read title=\"Your Title\""
            note_path = self._get_note_path(title)
            if not note_path.exists():
                return f"Note not found: {title}. Use action=list to see available notes."
            note = json.loads(await async_read_file(note_path))
            return f"# {note['title']}\n\n{note['content']}\n\nCreated: {note['created']}"

        elif action == "list":
            notes = []
            for note_file in self.storage_path.glob("*.json"):
                try:
                    note = json.loads(await async_read_file(note_file))
                    notes.append(f"- {note['title']} ({note['created'][:10]})")
                except Exception:
                    continue
            if not notes:
                return "No notes found. Create one with: notes action=create title=\"Your Title\" content=\"Your content\""
            return f"Notes ({len(notes)}):\n" + "\n".join(notes)

        elif action == "search":
            if not query or not query.strip():
                return "Error: Please provide a search query. Usage: notes action=search query=\"your search term\""
            query_lower = query.lower()
            results = []
            for note_file in self.storage_path.glob("*.json"):
                try:
                    note = json.loads(await async_read_file(note_file))
                    title = note.get("title", "")
                    content = note.get("content", "")
                    if query_lower in title.lower() or query_lower in content.lower():
                        snippet = content[:100] + "..." if len(content) > 100 else content
                        results.append(f"- **{title}**: {snippet}")
                except Exception:
                    continue
            if not results:
                return f"No notes found matching '{query}'. Try a different search term or use action=list to see all notes."
            return f"Found {len(results)} note(s) matching '{query}':\n\n" + "\n\n".join(results)

        elif action == "update":
            if not title or not title.strip():
                return "Error: Please provide a title to update. Usage: notes action=update title=\"Your Title\" content=\"New content\""
            note_path = self._get_note_path(title)
            if not note_path.exists():
                return f"Note not found: {title}. Use action=create to create it first."
            
            note = json.loads(await async_read_file(note_path))
            
            if content:
                note["content"] = content
            note["modified"] = datetime.now().isoformat()
            
            await async_write_file(note_path, json.dumps(note, indent=2))
            return f"Updated note: {title}"

        elif action == "delete":
            if not title or not title.strip():
                return "Error: Please provide a title to delete. Usage: notes action=delete title=\"Your Title\""
            note_path = self._get_note_path(title)
            if not note_path.exists():
                return f"Note not found: {title}"
            note_path.unlink()
            return f"Deleted note: {title}"

        return f"Unknown action: {action}. Available actions: create, read, list, search, update, delete"


class RemindersTool(Tool):
    """Tool for managing reminders."""

    name = "reminders"
    description = "Create and manage reminders"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "delete", "clear"],
                "description": "Action to perform",
            },
            "text": {
                "type": "string",
                "description": "Reminder text",
            },
            "time": {
                "type": "string",
                "description": "When to remind (e.g., 'in 5 minutes', 'tomorrow at 9am')",
            },
        },
        "required": ["action"],
    }
    requires_permission = False

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.reminders_file = self.storage_path / "reminders.json"

    async def _load_reminders(self) -> list[dict]:
        """Load reminders from file."""
        if not self.reminders_file.exists():
            return []
        try:
            data = await async_read_file(self.reminders_file)
            return json.loads(data)
        except Exception:
            return []

    async def _save_reminders(self, reminders: list[dict]) -> None:
        """Save reminders to file."""
        await async_write_file(self.reminders_file, json.dumps(reminders, indent=2))

    def _parse_time(self, time_str: str) -> datetime | None:
        """Parse a time string into a datetime."""
        time_str = time_str.lower().strip()

        now = datetime.now()

        if "in" in time_str:
            parts = time_str.replace("in", "").strip().split()
            if len(parts) >= 2:
                try:
                    amount = int(parts[0])
                    unit = parts[1]

                    if "minute" in unit:
                        return now + timedelta(minutes=amount)
                    elif "hour" in unit:
                        return now + timedelta(hours=amount)
                    elif "day" in unit:
                        return now + timedelta(days=amount)
                except ValueError:
                    logger.debug(f"Failed to parse time amount: {parts[0]}")
            else:
                logger.debug(f"Malformed time string: {time_str}")

        logger.debug(f"Could not parse time '{time_str}', defaulting to 1 hour")
        return now + timedelta(hours=1)

    async def execute(
        self,
        action: str,
        text: str | None = None,
        time: str | None = None,
        **kwargs
    ) -> str:
        """Execute reminders action."""
        reminders = await self._load_reminders()

        if action == "create":
            if not text:
                return "Error: reminder text required"

            remind_time = self._parse_time(time) if time else None

            reminder = {
                "text": text,
                "time": remind_time.isoformat() if remind_time else None,
                "created": datetime.now().isoformat(),
            }
            reminders.append(reminder)
            await self._save_reminders(reminders)

            time_str = f" at {remind_time.strftime('%Y-%m-%d %H:%M')}" if remind_time else ""
            return f"Created reminder: {text}{time_str}"

        elif action == "list":
            if not reminders:
                return "No reminders"
            output = ["Reminders:"]
            for i, r in enumerate(reminders, 1):
                time_str = f" ({r['time']})" if r.get("time") else ""
                output.append(f"{i}. {r['text']}{time_str}")
            return "\n".join(output)

        elif action == "delete":
            if not reminders:
                return "No reminders to delete"
            reminders.pop()  # Remove last
            await self._save_reminders(reminders)
            return "Deleted last reminder"

        elif action == "clear":
            await self._save_reminders([])
            return "Cleared all reminders"

        return f"Unknown action: {action}"


class ProductivityTools:
    """Collection of productivity tools."""

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path
        self.tools = [
            NotesTool(storage_path),
            RemindersTool(storage_path),
        ]

    def get_tools(self) -> list[Tool]:
        """Get all productivity tools."""
        return self.tools
