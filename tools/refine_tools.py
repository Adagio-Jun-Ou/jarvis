"""Self-refinement tools for Jarvis.

These tools give Jarvis the ability to analyze and improve its own code
with permission-based safety controls.

Inspired by OpenCode's tool-based approach:
1. Scan codebase for issues
2. Propose improvements with explanations
3. Apply changes with permission
4. Track and rollback changes
"""
import logging
from pathlib import Path

from core.helpers import safe_error_message
from .base import Tool

logger = logging.getLogger(__name__)


class RefineScanTool(Tool):
    """Scan codebase for issues and improvement opportunities."""

    name = "refine_scan"
    description = "Scan codebase for bugs, security issues, and improvement opportunities"
    parameters = {
        "type": "object",
        "properties": {
            "include_ai": {
                "type": "boolean",
                "description": "Include AI-powered deep analysis (slower but more thorough)",
            },
            "file_pattern": {
                "type": "string",
                "description": "Optional file pattern to limit scan (e.g., 'core/*.py')",
            },
        },
        "required": [],
    }
    requires_permission = False

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or Path.cwd()

    async def execute(self, include_ai: bool = False, file_pattern: str | None = None, **kwargs) -> str:
        """Execute the scan."""
        try:
            from core.auto_refine import CodeAnalyzer

            analyzer = CodeAnalyzer(self.base_path)

            if file_pattern:
                from pathlib import PurePath
                pattern_parts = file_pattern.replace('\\', '/').split('/')
                filtered_issues = []
                for issue in analyzer.scan_directory().issues_found:
                    issue_path = PurePath(issue.file_path)
                    if any(p in str(issue_path) for p in pattern_parts):
                        filtered_issues.append(issue)

                if not filtered_issues:
                    return "No issues found matching the pattern."

                result = f"Found {len(filtered_issues)} issues in {file_pattern}:\n\n"
            else:
                scan_result = analyzer.scan_directory()
                filtered_issues = scan_result.issues_found
                result = f"Scanned {scan_result.files_scanned} files in {scan_result.duration_seconds:.1f}s\n\n"
                result += f"Found {len(filtered_issues)} issues:\n\n"

            if not filtered_issues:
                return result + "No issues found."

            by_severity = {}
            for issue in filtered_issues[:20]:
                sev = issue.severity.value
                if sev not in by_severity:
                    by_severity[sev] = []
                by_severity[sev].append(issue)

            severity_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢", "suggestion": "💡"}

            for sev in ["critical", "high", "medium", "low", "suggestion"]:
                if sev in by_severity:
                    icon = severity_icons.get(sev, "❓")
                    result += f"### {icon} {sev.upper()} ({len(by_severity[sev])})\n\n"
                    for issue in by_severity[sev][:5]:
                        result += f"- **{issue.file_path}:{issue.line_number}** - {issue.title}\n"
                        if issue.suggested_fix:
                            result += f"  - Suggestion: {issue.suggested_fix[:100]}\n"
                    result += "\n"

            result += "\n*Use `refine_propose` to generate fixes for critical/high issues.*"
            return result

        except Exception as e:
            logger.error(f"Refine scan failed: {e}")
            return f"Scan failed: {safe_error_message(e, 'Refinement operation failed')}"


class RefineProposeTool(Tool):
    """Generate proposed fixes for issues."""

    name = "refine_propose"
    description = "Generate proposed fixes for code issues (requires permission to apply)"
    parameters = {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "description": "Focus area: 'critical', 'high', 'security', 'performance', or 'all'",
            },
            "max_fixes": {
                "type": "integer",
                "description": "Maximum number of fixes to propose (default 5)",
            },
        },
        "required": [],
    }
    requires_permission = False

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or Path.cwd()
        self._manager = None

    async def _get_manager(self):
        if self._manager is None:
            from core.auto_refine import AutoRefineManager
            from core.llm import OllamaClient
            try:
                client = OllamaClient()
                self._manager = AutoRefineManager(client, self.base_path)
            except Exception as e:
                logger.error(f"Failed to create refine manager: {e}")
                return None
        return self._manager

    async def execute(self, focus: str = "high", max_fixes: int = 5, **kwargs) -> str:
        """Generate proposed fixes."""
        try:
            manager = await self._get_manager()
            if not manager:
                return "Refine manager not available."

            scan_result = await manager.scan_codebase(include_ai=False)

            from core.auto_refine import IssueSeverity
            severity_filter = {
                "critical": [IssueSeverity.CRITICAL],
                "high": [IssueSeverity.CRITICAL, IssueSeverity.HIGH],
                "security": [IssueSeverity.CRITICAL],
                "performance": [IssueSeverity.MEDIUM],
                "all": list(IssueSeverity),
            }.get(focus.lower(), [IssueSeverity.CRITICAL, IssueSeverity.HIGH])

            filtered = [i for i in scan_result.issues_found if i.severity in severity_filter]
            filtered = filtered[:max_fixes]

            if not filtered:
                return f"No issues found for focus '{focus}'."

            proposed = await manager.propose_fixes(filtered, max_fixes=max_fixes)

            if not proposed:
                return "Could not generate fixes for the found issues."

            result = f"Generated {len(proposed)} proposed fixes:\n\n"
            for change in proposed:
                result += f"### {change.id}\n"
                result += f"- **File:** {change.file_path}\n"
                result += f"- **Risk:** {change.risk_level}\n"
                result += f"- **Status:** {change.status.value}\n"
                result += f"- **Explanation:** {change.explanation[:200]}...\n\n"

            result += "\n*Use `refine_apply` to apply these changes (requires permission).*"
            return result

        except Exception as e:
            logger.error(f"Refine propose failed: {e}")
            return f"Failed to propose fixes: {safe_error_message(e, 'Refinement operation failed')}"


class RefineApplyTool(Tool):
    """Apply proposed changes with permission."""

    name = "refine_apply"
    description = "Apply proposed code changes (requires permission)"
    parameters = {
        "type": "object",
        "properties": {
            "change_id": {
                "type": "string",
                "description": "ID of the proposed change to apply, or 'all' for all pending",
            },
        },
        "required": ["change_id"],
    }
    requires_permission = True

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or Path.cwd()
        self._manager = None

    async def _get_manager(self):
        if self._manager is None:
            from core.auto_refine import AutoRefineManager
            from core.llm import OllamaClient
            try:
                client = OllamaClient()
                self._manager = AutoRefineManager(client, self.base_path)
            except Exception:
                return None
        return self._manager

    async def execute(self, change_id: str, **kwargs) -> str:
        """Apply a proposed change."""
        try:
            manager = await self._get_manager()
            if not manager:
                return "Refine manager not available."

            if change_id.lower() == "all":
                results = await manager.apply_approved_changes(auto_permission=False)
                output = "Applied changes:\n\n"
                for cid, success, msg in results:
                    status = "✅" if success else "❌"
                    output += f"{status} {cid}: {msg}\n"
                return output

            success, msg = manager.apply_change(change_id)
            if success:
                return f"✅ Successfully applied change {change_id}\n\n{msg}\n\n*Use `refine_rollback {change_id}` to undo if needed.*"
            else:
                return f"❌ Failed to apply change: {msg}"

        except Exception as e:
            logger.error(f"Refine apply failed: {e}")
            return f"Failed to apply change: {safe_error_message(e, 'Refinement operation failed')}"


class RefineStatusTool(Tool):
    """Check refinement system status."""

    name = "refine_status"
    description = "Check the status of the refinement system and pending changes"
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    requires_permission = False

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or Path.cwd()

    async def execute(self, **kwargs) -> str:
        """Get refinement status."""
        try:
            from core.auto_refine import AutoRefineManager
            from core.llm import OllamaClient

            client = OllamaClient()
            manager = AutoRefineManager(client, self.base_path)
            stats = manager.get_statistics()

            result = "## Refinement Status\n\n"
            result += f"- **Total Changes:** {stats.get('total', 0)}\n"
            result += f"- **Pending:** {stats.get('pending', 0)}\n"
            result += f"- **Applied:** {stats.get('applied', 0)}\n"
            result += f"- **Rejected:** {stats.get('rejected', 0)}\n"
            result += f"- **Failed:** {stats.get('failed', 0)}\n"
            result += f"- **Rolled Back:** {stats.get('rolled_back', 0)}\n"

            if 'last_scan' in stats:
                scan = stats['last_scan']
                result += "\n### Last Scan\n"
                result += f"- **Files Scanned:** {scan.get('files_scanned', 'N/A')}\n"
                result += f"- **Issues Found:** {scan.get('issues_found', 'N/A')}\n"
                result += f"- **Duration:** {scan.get('duration', 'N/A')}s\n"

            pending = manager.get_pending_changes()
            if pending:
                result += f"\n### Pending Changes ({len(pending)})\n\n"
                for change in pending[:5]:
                    result += f"- `{change.id}`: {change.file_path} ({change.risk_level} risk)\n"

            return result

        except Exception as e:
            return f"Unable to get refinement status: {safe_error_message(e, 'Refinement operation failed')}"


class RefineRollbackTool(Tool):
    """Rollback an applied change."""

    name = "refine_rollback"
    description = "Rollback a previously applied code change"
    parameters = {
        "type": "object",
        "properties": {
            "change_id": {
                "type": "string",
                "description": "ID of the change to rollback",
            },
        },
        "required": ["change_id"],
    }
    requires_permission = True

    def __init__(self, base_path: Path | None = None):
        self.base_path = base_path or Path.cwd()
        self._manager = None

    async def _get_manager(self):
        if self._manager is None:
            from core.auto_refine import AutoRefineManager
            from core.llm import OllamaClient
            try:
                client = OllamaClient()
                self._manager = AutoRefineManager(client, self.base_path)
            except Exception:
                return None
        return self._manager

    async def execute(self, change_id: str, **kwargs) -> str:
        """Rollback a change."""
        try:
            manager = await self._get_manager()
            if not manager:
                return "Refine manager not available."

            success, msg = manager.rollback_change(change_id)
            if success:
                return f"✅ Successfully rolled back change {change_id}\n\n{msg}"
            else:
                return f"❌ Failed to rollback: {msg}"

        except Exception as e:
            return f"Rollback failed: {safe_error_message(e, 'Refinement operation failed')}"


class SelfReflectTool(Tool):
    """Analyze Jarvis's own code and behavior."""

    name = "self_reflect"
    description = "Trigger self-reflection on code, behavior, or performance"
    parameters = {
        "type": "object",
        "properties": {
            "aspect": {
                "type": "string",
                "description": "What to reflect on: 'code', 'behavior', 'performance', or 'all'",
            },
        },
        "required": [],
    }
    requires_permission = False

    async def execute(self, aspect: str = "all", **kwargs) -> str:
        """Perform self-reflection."""
        try:
            from core.meta import SelfReflectionEngine

            engine = SelfReflectionEngine()
            reflection = await engine.reflect(aspect)

            if reflection:
                result = f"## Self-Reflection: {aspect}\n\n"
                result += reflection.get('summary', 'No summary available')
                result += "\n\n### Observations\n"
                for obs in reflection.get('observations', []):
                    result += f"- {obs}\n"
                result += "\n### Suggestions\n"
                for sug in reflection.get('suggestions', []):
                    result += f"- {sug}\n"
                return result
            else:
                return "Self-reflection completed but no insights generated."

        except Exception as e:
            return f"Self-reflection failed: {safe_error_message(e, 'Refinement operation failed')}"


def get_refine_tools(base_path: Path | None = None) -> list[Tool]:
    """Get all refinement-related tools."""
    return [
        RefineScanTool(base_path),
        RefineProposeTool(base_path),
        RefineApplyTool(base_path),
        RefineStatusTool(base_path),
        RefineRollbackTool(base_path),
        SelfReflectTool(),
    ]
