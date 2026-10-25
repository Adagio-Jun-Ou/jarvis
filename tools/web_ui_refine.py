"""Web UI Self-Refinement Tools for Jarvis.

Allows Jarvis to analyze, understand, and propose improvements to its own web interface.
All changes require explicit user permission.

Key capabilities:
- View web UI structure and code
- Capture screenshots for visual analysis
- Propose UI improvements with full permission workflow
- Analyze UI components and their interactions
"""
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .base import Tool

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent.parent / "web"
WEB_SRC = WEB_DIR / "src"
WEB_COMPONENTS = WEB_SRC / "components"
MOBILE_DIR = WEB_SRC / "mobile"


@dataclass
class UIViewAnalysis:
    """Result of analyzing a UI component"""
    file_path: str
    component_name: str
    line_count: int
    has_css: bool
    imports: list[str]
    props: list[str]
    state_vars: list[str]
    handlers: list[str]
    issues: list[str]
    suggestions: list[str]


class WebUIViewTool(Tool):
    """View and understand the web UI codebase."""

    name = "web_ui_view"
    description = """View and analyze the Jarvis web UI code.
    
    Returns the structure and content of UI components.
    Use this to understand the current UI before proposing changes.
    
    Views can be:
    - 'structure': List all UI files including mobile PWA
    - 'component': Read a specific component file (e.g., 'MobileApp', 'MobileSettings')
    - 'styles': Read CSS/styling files including mobile CSS  
    - 'full': Read multiple related files at once including mobile
    
    Mobile components are in the 'mobile/' directory:
    - MobileApp, MobileLogin, BiometricAuth, MobileTools, MobileDashboard, MobileSettings
    - Mobile.css for all mobile styling
    """
    parameters = {
        "type": "object",
        "properties": {
            "view_type": {
                "type": "string",
                "enum": ["structure", "component", "styles", "full"],
                "description": "Type of view to get",
                "default": "structure",
            },
            "component": {
                "type": "string",
                "description": "Component name (e.g., 'ChatTab', 'DashboardTab')",
            },
            "include_css": {
                "type": "boolean",
                "description": "Include related CSS files",
                "default": True,
            },
        },
    }
    requires_permission = False  # Reading is safe

    async def execute(
        self,
        view_type: str = "structure",
        component: Optional[str] = None,
        include_css: bool = True,
        **kwargs  # Accept extra params for flexibility
    ) -> str:
        """View and analyze the Jarvis web UI code.
        
        Args:
            view_type: Type of view to get (structure, component, styles, full)
            component: Component name for 'component' view type
            include_css: Whether to include CSS files
            **kwargs: Additional parameters (ignored for compatibility)
        
        Returns:
            UI structure or component content
        """
        # Handle common parameter variations
        if not component:
            component = kwargs.get('name', kwargs.get('file', None))
        
        if view_type == "structure":
            return self._get_structure()
        elif view_type == "component":
            if not component:
                return "Error: component name required for component view"
            return self._get_component(component, include_css)
        elif view_type == "styles":
            return self._get_styles()
        elif view_type == "full":
            return self._get_full_view()
        return f"Unknown view type: {view_type}"

    def _get_structure(self) -> str:
        lines = ["# Jarvis Web UI Structure\n"]
        lines.append(f"Web directory: {WEB_DIR}\n\n")

        if not WEB_SRC.exists():
            return "Web UI source directory not found"

        lines.append("## Main Files\n")
        for f in WEB_SRC.glob("*.tsx"):
            size = f.stat().st_size
            lines.append(f"- `{f.name}` ({size:,} bytes)")
        for f in WEB_SRC.glob("*.css"):
            size = f.stat().st_size
            lines.append(f"- `{f.name}` ({size:,} bytes)")

        lines.append("\n## Components\n")
        if WEB_COMPONENTS.exists():
            for f in sorted(WEB_COMPONENTS.glob("*.tsx")):
                size = f.stat().st_size
                lines.append(f"- `{f.name}` ({size:,} bytes)")
            for f in sorted(WEB_COMPONENTS.glob("*.css")):
                size = f.stat().st_size
                lines.append(f"- `{f.name}` ({size:,} bytes)")

        lines.append("\n## Mobile PWA\n")
        if MOBILE_DIR.exists():
            for f in sorted(MOBILE_DIR.glob("*.tsx")):
                size = f.stat().st_size
                lines.append(f"- `mobile/{f.name}` ({size:,} bytes)")
            for f in sorted(MOBILE_DIR.glob("*.css")):
                size = f.stat().st_size
                lines.append(f"- `mobile/{f.name}` ({size:,} bytes)")

        return "\n".join(lines)

    def _get_component(self, name: str, include_css: bool) -> str:
        name = name.replace(".tsx", "").replace(".css", "")
        
        tsx_file = WEB_COMPONENTS / f"{name}.tsx"
        if not tsx_file.exists():
            tsx_file = WEB_SRC / f"{name}.tsx"
        if not tsx_file.exists():
            tsx_file = MOBILE_DIR / f"{name}.tsx"
        
        if not tsx_file.exists():
            return f"Component '{name}' not found. Available: {self._list_components()}"

        lines = [f"# {name}.tsx\n"]
        try:
            content = tsx_file.read_text(encoding='utf-8')
            lines.append(f"```tsx\n{content}\n```\n")
        except Exception as e:
            lines.append(f"Error reading file: {e}\n")

        if include_css:
            css_file = WEB_COMPONENTS / f"{name}.css"
            if not css_file.exists():
                css_file = WEB_SRC / f"{name}.css"
            if not css_file.exists():
                css_file = MOBILE_DIR / f"{name}.css"
            
            if css_file.exists():
                lines.append(f"\n## {name}.css\n")
                try:
                    css_content = css_file.read_text(encoding='utf-8')
                    lines.append(f"```css\n{css_content}\n```\n")
                except Exception as e:
                    lines.append(f"Error reading CSS: {e}\n")

        return "\n".join(lines)

    def _get_styles(self) -> str:
        lines = ["# Web UI Styles\n"]

        for css_file in [WEB_SRC / "App.css", WEB_SRC / "index.css"]:
            if css_file.exists():
                lines.append(f"\n## {css_file.name}\n")
                try:
                    content = css_file.read_text(encoding='utf-8')
                    lines.append(f"```css\n{content}\n```\n")
                except Exception as e:
                    lines.append(f"Error: {e}\n")

        mobile_css = MOBILE_DIR / "Mobile.css"
        if mobile_css.exists():
            lines.append("\n## Mobile.css (Mobile PWA Styles)\n")
            try:
                content = mobile_css.read_text(encoding='utf-8')
                lines.append(f"```css\n{content}\n```\n")
            except Exception as e:
                lines.append(f"Error: {e}\n")

        return "\n".join(lines)

    def _get_full_view(self) -> str:
        lines = ["# Full Web UI View\n\n"]
        lines.append(self._get_structure())
        lines.append("\n\n---\n\n")
        lines.append("## App.tsx (Main Entry)\n")
        
        app_file = WEB_SRC / "App.tsx"
        if app_file.exists():
            try:
                content = app_file.read_text(encoding='utf-8')
                lines.append(f"```tsx\n{content}\n```\n")
            except Exception as e:
                lines.append(f"Error: {e}\n")

        lines.append("\n\n---\n\n")
        lines.append("## Mobile App.tsx (Mobile PWA Entry)\n")
        
        mobile_app = MOBILE_DIR / "MobileApp.tsx"
        if mobile_app.exists():
            try:
                content = mobile_app.read_text(encoding='utf-8')
                lines.append(f"```tsx\n{content}\n```\n")
            except Exception as e:
                lines.append(f"Error: {e}\n")

        return "\n".join(lines)

    def _list_components(self) -> str:
        components = []
        if WEB_COMPONENTS.exists():
            components.extend(f.stem for f in WEB_COMPONENTS.glob("*.tsx"))
        if MOBILE_DIR.exists():
            components.extend("mobile/" + f.stem for f in MOBILE_DIR.glob("*.tsx"))
        if components:
            return ", ".join(sorted(components))
        return "none"


class WebUIAnalyzeTool(Tool):
    """Analyze UI component for improvements."""

    name = "web_ui_analyze"
    description = """Analyze a UI component and identify potential improvements.
    
    Returns analysis including:
    - Code structure and complexity
    - Accessibility issues
    - Performance concerns
    - Style consistency
    - Suggested improvements
    """
    parameters = {
        "type": "object",
        "properties": {
            "component": {
                "type": "string",
                "description": "Component to analyze",
            },
            "focus": {
                "type": "string",
                "enum": ["all", "accessibility", "performance", "structure", "styles"],
                "description": "Focus area for analysis",
                "default": "all",
            },
        },
    }
    requires_permission = False

    async def execute(
        self,
        component: str = "",
        focus: str = "all",
        **kwargs
    ) -> str:
        component = component.replace(".tsx", "").replace(".css", "")
        
        tsx_file = WEB_COMPONENTS / f"{component}.tsx"
        if not tsx_file.exists():
            tsx_file = WEB_SRC / f"{component}.tsx"
        
        if not tsx_file.exists():
            return f"Component '{component}' not found"

        try:
            content = tsx_file.read_text(encoding='utf-8')
        except Exception as e:
            return f"Error reading component: {e}"

        lines = [f"# Analysis: {component}\n\n"]

        issues = []
        suggestions = []

        line_count = len(content.split('\n'))
        lines.append(f"**Lines:** {line_count}\n")

        if "useState" in content:
            state_count = content.count("useState")
            lines.append(f"**State hooks:** {state_count}\n")
            if state_count > 5:
                issues.append(f"Many state hooks ({state_count}) - consider useReducer")

        if "useEffect" in content:
            effect_count = content.count("useEffect")
            lines.append(f"**Effect hooks:** {effect_count}\n")
            if effect_count > 3:
                issues.append(f"Many effect hooks ({effect_count}) - check for unnecessary effects")

        imports = [l for l in content.split('\n') if l.strip().startswith('import ')]
        lines.append(f"**Imports:** {len(imports)}\n")

        if 'aria-' not in content.lower() and 'role=' not in content.lower():
            issues.append("No ARIA attributes found - consider accessibility")
        
        if 'keyboard' not in content.lower() and 'onKeyDown' not in content and 'onKeyPress' not in content:
            issues.append("No keyboard handlers - consider keyboard accessibility")

        large_inline_styles = content.count('style={{')
        if large_inline_styles > 5:
            issues.append(f"Many inline styles ({large_inline_styles}) - consider CSS classes")

        if line_count > 500:
            issues.append(f"Large component ({line_count} lines) - consider splitting")
            suggestions.append("Extract sub-components for better maintainability")

        if "TODO" in content or "FIXME" in content or "HACK" in content:
            issues.append("Contains TODO/FIXME/HACK comments")

        if issues:
            lines.append("\n## Issues Found\n")
            for issue in issues:
                lines.append(f"- {issue}")
        else:
            lines.append("\n**No major issues found**\n")

        if suggestions:
            lines.append("\n## Suggestions\n")
            for sug in suggestions:
                lines.append(f"- {sug}")

        css_file = WEB_COMPONENTS / f"{component}.css"
        if not css_file.exists():
            css_file = WEB_SRC / f"{component}.css"
        
        if css_file.exists():
            css_content = css_file.read_text(encoding='utf-8')
            css_lines = len(css_content.split('\n'))
            lines.append("\n## CSS\n")
            lines.append(f"**CSS lines:** {css_lines}\n")
            
            if '!important' in css_content:
                issues.append("CSS uses !important - consider specificity")
            if css_content.count('position: fixed') > 2:
                issues.append("Multiple fixed positions - may cause overlap issues")

        return "\n".join(lines)


class WebUIProposeTool(Tool):
    """Propose a change to the web UI with permission workflow."""

    name = "web_ui_propose"
    description = """Propose an improvement to the web UI.
    
    This creates a formal proposal that requires user approval.
    The change will be reviewed and shown to the user before being applied.
    
    Steps:
    1. Describe what you want to change and why
    2. Provide the new/modified code
    3. User reviews and approves/rejects
    4. If approved, change is applied
    """
    parameters = {
        "type": "object",
        "properties": {
            "target_file": {
                "type": "string",
                "description": "File to modify (e.g., 'ChatTab.tsx', 'App.css')",
            },
            "description": {
                "type": "string",
                "description": "Brief description of the change",
            },
            "rationale": {
                "type": "string",
                "description": "Why this change is needed",
            },
            "old_code": {
                "type": "string",
                "description": "The exact code to replace (empty for new files)",
            },
            "new_code": {
                "type": "string",
                "description": "The new code to insert",
            },
            "change_type": {
                "type": "string",
                "enum": ["bug_fix", "feature", "improvement", "style", "accessibility", "refactor"],
                "description": "Type of change",
                "default": "improvement",
            },
        },
        "required": ["target_file", "description", "rationale", "new_code"],
    }
    requires_permission = True

    async def execute(
        self,
        target_file: str = "",
        description: str = "",
        rationale: str = "",
        new_code: str = "",
        old_code: str = "",
        change_type: str = "improvement",
        **kwargs
    ) -> str:
        file_path = self._resolve_path(target_file)
        if not file_path:
            return f"Error: Cannot modify '{target_file}'. Web UI files must be in web/src/ or web/src/components/"

        from core.self_edit import get_self_edit_engine, EditType

        engine = get_self_edit_engine()

        type_map = {
            "bug_fix": EditType.BUG_FIX,
            "feature": EditType.FEATURE,
            "improvement": EditType.EVOLUTION,
            "style": EditType.REFACTOR,
            "accessibility": EditType.EVOLUTION,
            "refactor": EditType.REFACTOR,
        }

        edit_type = type_map.get(change_type, EditType.EVOLUTION)

        if old_code:
            try:
                current = file_path.read_text(encoding='utf-8')
                if old_code not in current:
                    return f"Error: The 'old_code' was not found exactly in {target_file}.\n\nPlease use web_ui_view to see the current content and provide the exact code to replace."
            except Exception as e:
                return f"Error reading file: {e}"

        edit = engine.propose_edit(
            file_path=str(file_path),
            edit_type=edit_type,
            description=description,
            rationale=rationale,
            old_code=old_code,
            new_code=new_code,
        )

        return f"""# Web UI Change Proposed

**File:** {target_file}
**Type:** {change_type}
**Description:** {description}
**Rationale:** {rationale}

**Edit ID:** {edit.id}
**Status:** {edit.status.value}

---

This change requires your approval. Use:
- `approve_web_ui_edit(edit_id="{edit.id}")` to approve
- `reject_web_ui_edit(edit_id="{edit.id}")` to reject

Or use the /api/refine endpoints to review pending changes.
"""


    def _resolve_path(self, target: str) -> Optional[Path]:
        target = target.replace('\\', '/').strip('/')
        
        if target.startswith('web/'):
            target = target[4:]
        
        if target.startswith('mobile/'):
            target = target[7:]
            candidate = MOBILE_DIR / target
            if candidate.exists() and self._is_safe_path(candidate):
                return candidate
            if target.endswith('.tsx') or target.endswith('.css'):
                candidate = MOBILE_DIR / target
                if self._is_safe_path(candidate):
                    return candidate

        candidate = WEB_DIR / target
        if candidate.exists() and self._is_safe_path(candidate):
            return candidate

        candidate = WEB_COMPONENTS / target
        if candidate.exists() and self._is_safe_path(candidate):
            return candidate

        if target.endswith('.tsx') or target.endswith('.css'):
            candidate = WEB_COMPONENTS / target
            parent = candidate.parent
            if parent.exists() or parent == WEB_COMPONENTS:
                if self._is_safe_path(candidate):
                    return candidate

        return None

    def _is_safe_path(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
            web_resolved = WEB_DIR.resolve()
            return str(resolved).startswith(str(web_resolved))
        except Exception:
            return False


class WebUIApproveTool(Tool):
    """Approve a pending web UI change."""

    name = "approve_web_ui_edit"
    description = "Approve a pending web UI edit by ID."
    parameters = {
        "type": "object",
        "properties": {
            "edit_id": {
                "type": "string",
                "description": "The edit ID to approve",
            },
        },
        "required": ["edit_id"],
    }
    requires_permission = True

    async def execute(self, edit_id: str, **kwargs) -> str:
        from core.self_edit import get_self_edit_engine

        engine = get_self_edit_engine()
        
        edit = engine.get_edit(edit_id)
        if not edit:
            return f"Edit '{edit_id}' not found"

        success = engine.approve_edit(edit_id)
        if success:
            applied = engine.apply_edit(edit_id)
            if applied:
                return f"✓ Edit {edit_id} approved and applied to {edit.file_path}"
            else:
                return f"✗ Edit {edit_id} approved but failed to apply. Check logs."
        return f"✗ Failed to approve edit {edit_id}"


class WebUIRejectTool(Tool):
    """Reject a pending web UI change."""

    name = "reject_web_ui_edit"
    description = "Reject a pending web UI edit by ID."
    parameters = {
        "type": "object",
        "properties": {
            "edit_id": {
                "type": "string",
                "description": "The edit ID to reject",
            },
            "reason": {
                "type": "string",
                "description": "Reason for rejection",
            },
        },
        "required": ["edit_id"],
    }
    requires_permission = True

    async def execute(self, edit_id: str, reason: str = "", **kwargs) -> str:
        from core.self_edit import get_self_edit_engine

        engine = get_self_edit_engine()
        
        edit = engine.get_edit(edit_id)
        if not edit:
            return f"Edit '{edit_id}' not found"

        success = engine.reject_edit(edit_id, reason)
        if success:
            return f"✗ Edit {edit_id} rejected. {reason}"
        return f"Failed to reject edit {edit_id}"


class WebUIHistoryTool(Tool):
    """View history of web UI changes."""

    name = "web_ui_history"
    description = "View the history of web UI modifications."
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of edits to show",
                "default": 10,
            },
            "status": {
                "type": "string",
                "enum": ["all", "pending", "applied", "rejected"],
                "description": "Filter by status",
                "default": "all",
            },
        },
    }
    requires_permission = False

    async def execute(self, limit: int = 10, status: str = "all", **kwargs) -> str:
        from core.self_edit import get_self_edit_engine, EditStatus

        engine = get_self_edit_engine()
        edits = engine.get_all_edits()

        web_edits = [e for e in edits if 'web/' in e.file_path or 'web\\' in e.file_path]

        if status == "pending":
            web_edits = [e for e in web_edits if e.status in (EditStatus.PROPOSED, EditStatus.PENDING_REVIEW)]
        elif status == "applied":
            web_edits = [e for e in web_edits if e.status == EditStatus.APPLIED]
        elif status == "rejected":
            web_edits = [e for e in web_edits if e.status == EditStatus.REJECTED]

        web_edits = web_edits[:limit]

        if not web_edits:
            return "No web UI edits found."

        lines = ["# Web UI Edit History\n"]
        for edit in web_edits:
            lines.append(f"\n## {edit.id}")
            lines.append(f"- **File:** {edit.file_path}")
            lines.append(f"- **Type:** {edit.edit_type.value}")
            lines.append(f"- **Status:** {edit.status.value}")
            lines.append(f"- **Description:** {edit.description}")
            if edit.applied_at:
                lines.append(f"- **Applied:** {edit.applied_at}")

        return "\n".join(lines)


class WebUIScreenshotTool(Tool):
    """Capture a screenshot of Jarvis's own web UI for visual analysis."""

    name = "web_ui_screenshot"
    description = """Capture a screenshot of Jarvis's own web interface.
    
    This allows Jarvis to see its UI for visual analysis and improvement.
    Requires browser automation (Playwright or Selenium).
    """
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to capture (default: http://localhost:5173)",
                "default": "http://localhost:5173",
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture full scrollable page",
                "default": False,
            },
            "selector": {
                "type": "string",
                "description": "CSS selector to capture specific element",
            },
        },
    }
    requires_permission = True

    _last_screenshot: str | None = None

    async def execute(
        self,
        url: str = "http://localhost:5173",
        full_page: bool = False,
        selector: Optional[str] = None,
        **kwargs
    ) -> str:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return "Playwright not installed. Install with: pip install playwright && playwright install"

        try:
            import time
            from pathlib import Path

            screenshot_dir = Path.home() / ".jarvis" / "ui_screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = screenshot_dir / f"jarvis_ui_{int(time.time())}.png"

            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True)
                page = await browser.new_page()
                
                try:
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                except Exception as nav_e:
                    await browser.close()
                    return f"Navigation error: {nav_e}. Is the frontend running at {url}?"

                await page.wait_for_timeout(1000)

                if selector:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            await element.screenshot(path=str(screenshot_path))
                        else:
                            await browser.close()
                            return f"Element not found: {selector}"
                    except Exception as e:
                        await browser.close()
                        return f"Element screenshot error: {e}"
                else:
                    await page.screenshot(path=str(screenshot_path), full_page=full_page)

                title = await page.title()
                await browser.close()

            WebUIScreenshotTool._last_screenshot = str(screenshot_path)

            return f"""Screenshot captured!
URL: {url}
Title: {title}
Saved to: {screenshot_path}

To analyze this image, use the analyze-image tool with the path above.
The screenshot shows the current state of Jarvis's web interface.
"""

        except Exception as e:
            return f"Screenshot error: {e}"

    @classmethod
    def get_last_screenshot(cls) -> Optional[str]:
        return cls._last_screenshot


TOOLS = [
    WebUIViewTool,
    WebUIAnalyzeTool,
    WebUIProposeTool,
    WebUIApproveTool,
    WebUIRejectTool,
    WebUIHistoryTool,
    WebUIScreenshotTool,
]


class MobileUIViewTool(Tool):
    """View and analyze the mobile PWA UI."""

    name = "mobile_ui_view"
    description = """View and analyze the Jarvis mobile PWA code.
    
    The mobile PWA is a separate interface optimized for phones with:
    - FaceID/WebAuthn authentication
    - Image upload with native vision analysis
    - Chat, Tools, Dashboard, and Settings tabs
    - HTTPS access via Tailscale Funnel
    
    View types:
    - 'structure': List all mobile files
    - 'component': Read specific mobile component (MobileApp, MobileSettings, etc.)
    - 'styles': Read Mobile.css styling
    - 'endpoints': Show mobile API endpoints
    - 'url': Show the public mobile URL
    """
    parameters = {
        "type": "object",
        "properties": {
            "view_type": {
                "type": "string",
                "enum": ["structure", "component", "styles", "endpoints", "url"],
                "description": "Type of view to get",
                "default": "structure",
            },
            "component": {
                "type": "string",
                "description": "Component name for component view (MobileApp, MobileLogin, BiometricAuth, MobileTools, MobileDashboard, MobileSettings)",
            },
        },
    }
    requires_permission = False

    async def execute(
        self,
        view_type: str = "structure",
        component: Optional[str] = None,
        **kwargs
    ) -> str:
        if view_type == "structure":
            return self._get_structure()
        elif view_type == "component":
            if not component:
                return "Error: component name required. Available: MobileApp, MobileLogin, BiometricAuth, MobileTools, MobileDashboard, MobileSettings"
            return self._get_component(component)
        elif view_type == "styles":
            return self._get_styles()
        elif view_type == "endpoints":
            return self._get_endpoints()
        elif view_type == "url":
            return self._get_url()
        return f"Unknown view type: {view_type}"

    def _get_structure(self) -> str:
        lines = ["# Jarvis Mobile PWA Structure\n"]
        lines.append(f"Mobile directory: {MOBILE_DIR}\n\n")
        
        if not MOBILE_DIR.exists():
            return "Mobile PWA directory not found"
        
        lines.append("## Components\n")
        for f in sorted(MOBILE_DIR.glob("*.tsx")):
            size = f.stat().st_size
            lines.append(f"- `{f.name}` ({size:,} bytes)")
        
        for f in sorted(MOBILE_DIR.glob("*.css")):
            size = f.stat().st_size
            lines.append(f"- `{f.name}` ({size:,} bytes)")
        
        lines.append("\n## Features\n")
        lines.append("- FaceID/WebAuthn authentication (BiometricAuth.tsx)")
        lines.append("- Password fallback authentication")
        lines.append("- Image upload from gallery/camera")
        lines.append("- Native vision analysis for images")
        lines.append("- Chat with thinking blocks")
        lines.append("- Tools tab for tool calls")
        lines.append("- Dashboard with subsystem status")
        lines.append("- Settings with model selection")
        
        return "\n".join(lines)

    def _get_component(self, name: str) -> str:
        name = name.replace(".tsx", "").replace(".css", "")
        
        tsx_file = MOBILE_DIR / f"{name}.tsx"
        if not tsx_file.exists():
            available = ", ".join(f.stem for f in MOBILE_DIR.glob("*.tsx"))
            return f"Component '{name}' not found. Available: {available}"
        
        lines = [f"# {name}.tsx\n"]
        try:
            content = tsx_file.read_text(encoding='utf-8')
            lines.append(f"```tsx\n{content}\n```\n")
        except Exception as e:
            lines.append(f"Error reading file: {e}\n")
        
        css_file = MOBILE_DIR / f"{name}.css"
        if css_file.exists():
            lines.append(f"\n## {name}.css\n")
            try:
                css_content = css_file.read_text(encoding='utf-8')
                lines.append(f"```css\n{css_content}\n```\n")
            except Exception as e:
                lines.append(f"Error reading CSS: {e}\n")
        
        return "\n".join(lines)

    def _get_styles(self) -> str:
        lines = ["# Mobile PWA Styles\n"]
        
        mobile_css = MOBILE_DIR / "Mobile.css"
        if mobile_css.exists():
            try:
                content = mobile_css.read_text(encoding='utf-8')
                lines.append(f"```css\n{content}\n```\n")
            except Exception as e:
                lines.append(f"Error: {e}\n")
        else:
            lines.append("Mobile.css not found")
        
        return "\n".join(lines)

    def _get_endpoints(self) -> str:
        return """# Mobile API Endpoints

## Authentication
- `POST /api/mobile/auth/status` - Check device registration status
- `POST /api/mobile/auth/password` - Password authentication
- `POST /api/mobile/register/challenge` - WebAuthn registration challenge
- `POST /api/mobile/register/verify` - Verify WebAuthn registration
- `POST /api/mobile/auth/challenge` - WebAuthn authentication challenge  
- `POST /api/mobile/auth/verify` - Verify WebAuthn authentication
- `POST /api/mobile/logout` - Logout and invalidate token

## Chat
- `POST /api/mobile/chat` - Send message with optional images
  - Body: `{token, message, images?: string[]}`
  - Images are base64 data URLs
  - Returns: `{success, response}`
- `POST /api/mobile/analyze-image` - Analyze image with vision
  - Body: `{token, image}`
  - Returns: `{success, description}`

## WebSocket (limited support through Tailscale Funnel)
- `WS /api/mobile/ws` - WebSocket chat (may not work through Funnel)

## Frontend
- `GET /mobile` - Mobile PWA entry point
- `GET /manifest.json` - PWA manifest
- `GET /sw-mobile.js` - Service worker
"""

    def _get_url(self) -> str:
        import socket
        try:
            hostname = socket.gethostname()
        except (OSError, socket.error):
            hostname = "your-machine"
        
        return f"""# Mobile PWA URLs

## Local Development
- http://localhost:8080/mobile

## Tailscale (within tailnet)
- https://{hostname}.your-tailnet.ts.net/mobile

## Tailscale Funnel (public - requires FaceID real domain)
- https://{hostname}.your-tailnet.ts.net/mobile

Note: WebAuthn/FaceID requires HTTPS with a real domain.
The Tailscale Funnel provides valid Let's Encrypt SSL certificates.

## To access from your phone:
1. Open Safari on iOS
2. Navigate to the Tailscale Funnel URL above
3. Authenticate with FaceID or password
"""


TOOLS = [
    WebUIViewTool,
    WebUIAnalyzeTool,
    WebUIProposeTool,
    WebUIApproveTool,
    WebUIRejectTool,
    WebUIHistoryTool,
    WebUIScreenshotTool,
    MobileUIViewTool,
]