"""Free Time Tools - User interaction with Jarvis's free exploration.

Tools for:
- Viewing sandbox results
- Approving/denying permission requests
- Exploring the sandbox manually
- Managing saved explorations
"""
import logging

from .base import Tool

logger = logging.getLogger(__name__)


class ViewSandboxResultsTool(Tool):
    """View saved sandbox exploration results."""

    name = "view_sandbox_results"
    description = "View results from Jarvis's free time sandbox explorations"
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to show (default 5)",
            },
        },
    }
    requires_permission = False

    async def execute(self, limit: int = 5, **kwargs) -> str:
        """View saved sandbox results."""
        from core.free_time import get_free_time_engine

        engine = get_free_time_engine()
        results = engine.get_saved_results()[:limit]

        if not results:
            return "No saved exploration results yet. Jarvis hasn't saved any sandbox sessions."

        output = ["📋 **Saved Sandbox Results**\n"]

        for i, result in enumerate(results, 1):
            output.append(f"**{i}. {result.get('name', 'Unknown')}**")
            output.append(f"   📅 {result.get('timestamp', 'Unknown date')}")
            output.append(f"   📁 Path: {result.get('path', 'N/A')}")
            output.append(f"   🔢 Commands: {result.get('commands_run', 0)}")
            output.append(f"   📄 Files: {len(result.get('files_created', []))}")
            if result.get('description'):
                output.append(f"   📝 {result['description']}")
            output.append("")

        return "\n".join(output)


class ViewPermissionRequestsTool(Tool):
    """View pending permission requests from sandbox."""

    name = "view_permission_requests"
    description = "View pending permission requests from Jarvis's sandbox (software installations, etc.)"
    parameters = {
        "type": "object",
        "properties": {},
    }
    requires_permission = False

    async def execute(self, **kwargs) -> str:
        """View pending permission requests."""
        from core.free_time import get_free_time_engine

        engine = get_free_time_engine()
        pending = engine.get_pending_permissions()

        if not pending:
            return "✅ No pending permission requests. Jarvis is satisfied with current sandbox tools."

        output = ["🔑 **Pending Permission Requests**\n"]
        output.append("Jarvis wants to install additional tools for exploration:\n")

        for i, req in enumerate(pending, 1):
            output.append(f"**{i}. [{req.request_type.upper()}] {req.name}**")
            output.append(f"   📝 Reason: {req.reason}")
            output.append(f"   🆔 ID: {req.id}")
            output.append(f"   ⏰ Requested: {req.timestamp.strftime('%Y-%m-%d %H:%M')}")
            output.append("")

        output.append("\n💡 Use `approve_permission <id>` or `deny_permission <id>` to respond.")

        return "\n".join(output)


class ApprovePermissionTool(Tool):
    """Approve a permission request from sandbox."""

    name = "approve_permission"
    description = "Approve a permission request from Jarvis's sandbox"
    parameters = {
        "type": "object",
        "properties": {
            "request_id": {
                "type": "string",
                "description": "The ID of the permission request to approve",
            },
            "note": {
                "type": "string",
                "description": "Optional note for Jarvis",
            },
        },
        "required": ["request_id"],
    }
    requires_permission = False

    async def execute(self, request_id: str, note: str = "", **kwargs) -> str:
        """Approve a permission request."""
        from core.free_time import get_free_time_engine

        engine = get_free_time_engine()

        if engine.approve_permission(request_id, note):
            return f"✅ Permission request `{request_id}` approved.\n\nJarvis can now use this tool in the sandbox."
        else:
            return f"❌ Permission request `{request_id}` not found or already resolved."


class DenyPermissionTool(Tool):
    """Deny a permission request from sandbox."""

    name = "deny_permission"
    description = "Deny a permission request from Jarvis's sandbox"
    parameters = {
        "type": "object",
        "properties": {
            "request_id": {
                "type": "string",
                "description": "The ID of the permission request to deny",
            },
            "note": {
                "type": "string",
                "description": "Optional note explaining why",
            },
        },
        "required": ["request_id"],
    }
    requires_permission = False

    async def execute(self, request_id: str, note: str = "", **kwargs) -> str:
        """Deny a permission request."""
        from core.free_time import get_free_time_engine

        engine = get_free_time_engine()

        if engine.deny_permission(request_id, note):
            return f"🚫 Permission request `{request_id}` denied."
        else:
            return f"❌ Permission request `{request_id}` not found or already resolved."


class SandboxExploreTool(Tool):
    """Manually trigger a sandbox exploration."""

    name = "sandbox_explore"
    description = "Tell Jarvis to go explore in its sandbox (free time activity)"
    parameters = {
        "type": "object",
        "properties": {
            "activity": {
                "type": "string",
                "description": "Optional: specific activity to explore (e.g., 'create a maze solver')",
            },
            "save_results": {
                "type": "boolean",
                "description": "Whether to save results for later (default true)",
            },
        },
    }
    requires_permission = False

    async def execute(self, activity: str = "", save_results: bool = True, **kwargs) -> str:
        """Trigger a sandbox exploration."""
        from core.free_time import get_free_time_engine

        engine = get_free_time_engine()

        # Run exploration
        result = await engine.explore(duration_hint=60)

        output = ["🎮 **Sandbox Exploration Complete**\n"]
        output.append(f"⏱️ Duration: {result.duration_seconds:.1f}s")
        output.append(f"🔢 Commands run: {result.commands_run}")
        output.append(f"📄 Files created: {len(result.files_created)}")
        output.append(f"\n📝 Summary: {result.summary}")

        if result.insights:
            output.append("\n💡 **Insights:**")
            for insight in result.insights[:5]:
                output.append(f"  • {insight}")

        if result.error:
            output.append(f"\n⚠️ Error: {result.error}")

        # Save if requested
        if save_results and result.commands_run > 0:
            import random
            name = f"exploration_{random.randint(1000, 9999)}"
            saved_path = engine.save_exploration(name, result.summary)
            if saved_path:
                output.append(f"\n💾 Results saved to: {saved_path}")

        return "\n".join(output)


class FreeTimeStatsTool(Tool):
    """Get free time statistics."""

    name = "free_time_stats"
    description = "View statistics about Jarvis's free time explorations"
    parameters = {
        "type": "object",
        "properties": {},
    }
    requires_permission = False

    async def execute(self, **kwargs) -> str:
        """Get free time statistics."""
        from core.free_time import get_free_time_engine

        engine = get_free_time_engine()
        stats = engine.get_stats()

        output = ["⏰ **Free Time Statistics**\n"]
        output.append(f"🎮 Sessions: {stats['session_count']}")
        output.append(f"⏱️ Total time: {stats['total_time_hours']} hours ({stats['total_time_seconds']:.0f}s)")
        output.append(f"📁 Saved results: {stats['saved_results']}")
        output.append(f"🔑 Pending permissions: {stats['pending_permissions']}")
        output.append(f"📊 State: {stats['state']}")

        if stats['last_session']:
            output.append(f"🕒 Last session: {stats['last_session']}")

        return "\n".join(output)


class ReadSandboxFileTool(Tool):
    """Read a file from the sandbox."""

    name = "read_sandbox_file"
    description = "Read a file from a saved sandbox exploration"
    parameters = {
        "type": "object",
        "properties": {
            "result_name": {
                "type": "string",
                "description": "Name of the saved result (from view_sandbox_results)",
            },
            "filename": {
                "type": "string",
                "description": "Name of the file to read",
            },
        },
        "required": ["result_name", "filename"],
    }
    requires_permission = False

    async def execute(self, result_name: str, filename: str, **kwargs) -> str:
        """Read a file from saved sandbox results."""
        from core.free_time import SAVED_RESULTS

        # Find the result directory
        for result_dir in SAVED_RESULTS.iterdir():
            if result_dir.is_dir() and result_name in result_dir.name:
                # Safe filename
                safe_name = filename.replace("..", "").replace("/", "_")
                for subdir in ["code", "output", "data"]:
                    file_path = result_dir / subdir / safe_name
                    if file_path.exists():
                        try:
                            content = file_path.read_text(encoding='utf-8')
                            return f"📄 **{filename}**\n\n```\n{content[:5000]}\n```"
                        except Exception as e:
                            return f"❌ Error reading file: {e}"

        return f"❌ File not found: {filename} in {result_name}"


def get_free_time_tools():
    """Get all free time tools."""
    return [
        ViewSandboxResultsTool(),
        ViewPermissionRequestsTool(),
        ApprovePermissionTool(),
        DenyPermissionTool(),
        SandboxExploreTool(),
        FreeTimeStatsTool(),
        ReadSandboxFileTool(),
    ]
