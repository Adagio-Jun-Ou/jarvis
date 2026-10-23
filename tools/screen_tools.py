"""Screen Vision and Interaction Tools for Jarvis.

Provides tools for:
- Screen capture and analysis
- UI element detection and interaction
- Development context awareness
- Mouse/keyboard automation
"""
import asyncio
import logging
from typing import Optional

from core.helpers import safe_error_message
from tools.base import Tool

logger = logging.getLogger(__name__)

HAS_SCREEN_VISION = False
try:
    from core.screen_vision import (
        ScreenVision,
        ScreenContext,
        UIElementType,
        UIElement,
        get_screen_vision,
    )
    HAS_SCREEN_VISION = True
except ImportError:
    pass


class SeeScreenTool(Tool):
    """Capture and analyze the current screen."""

    name = "see_screen"
    description = "Capture a screenshot and analyze what's on screen. Detects UI elements, text, and context (IDE, terminal, browser, etc.)."
    parameters = {
        "type": "object",
        "properties": {
            "monitor": {
                "type": "integer",
                "description": "Monitor number to capture (1 = primary)",
                "default": 1,
            },
            "region": {
                "type": "array",
                "description": "Optional region to capture [left, top, right, bottom]",
                "items": {"type": "integer"},
            },
        },
    }
    requires_permission = True

    async def execute(self, monitor: int = 1, region: Optional[list[int]] = None) -> str:
        if not HAS_SCREEN_VISION:
            return "Screen vision not available. Install mss, pytesseract, and pynput."

        sv = get_screen_vision()
        region_tuple = tuple(region) if region else None

        loop = asyncio.get_event_loop()
        try:
            analysis = await loop.run_in_executor(
                None, lambda: sv.see(monitor=monitor, region=region_tuple)
            )

            result = analysis.to_dict()
            lines = [
                "**Screen Analysis**",
                f"Context: {result['context']} (confidence: {result['context_confidence']:.0%})",
                f"Found {result['text_count']} text blocks, {result['ui_element_count']} UI elements",
            ]

            if analysis.development_context:
                dev = analysis.development_context
                if dev.get("languages"):
                    lines.append(f"Languages: {', '.join(dev['languages'])}")
                if dev.get("frameworks"):
                    lines.append(f"Frameworks: {', '.join(dev['frameworks'])}")
                if dev.get("errors"):
                    lines.append(f"Errors detected: {len(dev['errors'])}")

            key_elements = [
                elem for elem in analysis.ui_elements
                if elem.confidence > 0.6 and elem.text
            ][:5]

            if key_elements:
                lines.append("\n**Key UI Elements:**")
                for elem in key_elements:
                    lines.append(f"  - {elem.element_type.name}: '{elem.text[:50]}' ({elem.confidence:.0%})")

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return f"Failed to capture screen: {safe_error_message(e, 'Screen capture failed')}"


class FindElementTool(Tool):
    """Find a UI element by text content."""

    name = "find_element"
    description = "Find a UI element on screen by its text content. Returns element location and type."
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to search for in UI elements",
            },
            "element_type": {
                "type": "string",
                "description": "Optional type filter: BUTTON, INPUT_FIELD, TEXT_AREA, CHECKBOX, TAB, LINK, CODE_EDITOR, TERMINAL_PROMPT",
            },
        },
        "required": ["text"],
    }

    async def execute(self, text: str, element_type: Optional[str] = None) -> str:
        if not HAS_SCREEN_VISION:
            return "Screen vision not available."

        sv = get_screen_vision()

        etype = None
        if element_type:
            try:
                etype = UIElementType[element_type.upper()]
            except KeyError:
                pass

        element = sv.find_element(text, etype)

        if element:
            return "\n".join([
                f"Found element: {element.element_type.name}",
                f"Text: '{element.text}'" if element.text else "No text",
                f"Position: ({element.bbox.center[0]}, {element.bbox.center[1]})",
                f"Size: {element.bbox.width}x{element.bbox.height}",
                f"Confidence: {element.confidence:.0%}",
            ])

        return f"Element with text '{text}' not found on screen."


class ClickElementTool(Tool):
    """Click at a specific position or on a found element."""

    name = "click_element"
    description = "Click at specific coordinates or find and click a UI element by text. Requires screen vision permission."
    parameters = {
        "type": "object",
        "properties": {
            "x": {
                "type": "integer",
                "description": "X coordinate to click",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate to click",
            },
            "text": {
                "type": "string",
                "description": "Text of element to find and click (alternative to x/y)",
            },
            "button": {
                "type": "string",
                "description": "Mouse button: left, right, middle",
                "default": "left",
            },
            "clicks": {
                "type": "integer",
                "description": "Number of clicks (1=single, 2=double)",
                "default": 1,
            },
        },
    }
    requires_permission = True

    async def execute(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        text: Optional[str] = None,
        button: str = "left",
        clicks: int = 1,
    ) -> str:
        if not HAS_SCREEN_VISION:
            return "Screen vision not available."

        sv = get_screen_vision()

        if not sv.automation._enabled:
            return "Automation is disabled. Enable with enable_automation tool first."

        if x is not None and y is not None:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, lambda: sv.automation.click(x, y, button, clicks)
            )
            if success:
                return f"Clicked at ({x}, {y}) with {button} button"
            return f"Failed to click at ({x}, {y})"

        if text:
            element = sv.find_element(text)
            if element:
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(
                    None, lambda: sv.click_element(element)
                )

                sv.learner.record_click(
                    element, success, sv._last_context or ScreenContext.UNKNOWN
                )
                sv.learner.save_state()

                if success:
                    return f"Clicked on '{text}' at {element.bbox.center}"
                return f"Failed to click on '{text}'"

            return f"Element '{text}' not found on screen. Capture screen first with see_screen."

        return "Provide either x/y coordinates or text to find element."


class TypeTextTool(Tool):
    """Type text using keyboard automation."""

    name = "type_text"
    description = "Type text at current cursor position. Enable automation first."
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to type",
            },
            "interval": {
                "type": "number",
                "description": "Interval between keystrokes in seconds",
                "default": 0.02,
            },
            "find_element": {
                "type": "string",
                "description": "Optional: find and click this element first",
            },
        },
        "required": ["text"],
    }
    requires_permission = True

    async def execute(
        self,
        text: str,
        interval: float = 0.02,
        find_element: Optional[str] = None,
    ) -> str:
        if not HAS_SCREEN_VISION:
            return "Screen vision not available."

        sv = get_screen_vision()

        if not sv.automation._enabled:
            return "Automation is disabled. Enable with enable_automation tool first."

        loop = asyncio.get_event_loop()

        if find_element:
            element = sv.find_element(find_element)
            if element:
                await loop.run_in_executor(None, lambda: sv.click_element(element))
                await asyncio.sleep(0.1)
            else:
                return f"Element '{find_element}' not found"

        success = await loop.run_in_executor(
            None, lambda: sv.automation.type_text(text, interval)
        )

        if success:
            return f"Typed {len(text)} characters"
        return "Failed to type text"


class PressKeyTool(Tool):
    """Press a key with optional modifiers."""

    name = "press_key"
    description = "Press a key with optional modifiers (Ctrl, Alt, Shift, etc.). Enable automation first."
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Key to press (e.g., 'enter', 'tab', 'escape', 'a', '1')",
            },
            "modifiers": {
                "type": "array",
                "description": "Modifier keys (e.g., ['ctrl', 'shift'])",
                "items": {"type": "string"},
            },
        },
        "required": ["key"],
    }
    requires_permission = True

    async def execute(self, key: str, modifiers: Optional[list[str]] = None) -> str:
        if not HAS_SCREEN_VISION:
            return "Screen vision not available."

        sv = get_screen_vision()

        if not sv.automation._enabled:
            return "Automation is disabled. Enable with enable_automation tool first."

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, lambda: sv.automation.press_key(key, modifiers)
        )

        if success:
            mod_str = " + ".join(modifiers) + " + " if modifiers else ""
            return f"Pressed {mod_str}{key}"
        return f"Failed to press {key}"


class HotkeyTool(Tool):
    """Execute a keyboard hotkey combination."""

    name = "hotkey"
    description = "Execute a keyboard hotkey (e.g., Ctrl+C, Alt+Tab). Enable automation first."
    parameters = {
        "type": "object",
        "properties": {
            "keys": {
                "type": "array",
                "description": "Keys to press together (e.g., ['ctrl', 'c'])",
                "items": {"type": "string"},
            },
        },
        "required": ["keys"],
    }
    requires_permission = True

    async def execute(self, keys: list[str]) -> str:
        if not HAS_SCREEN_VISION:
            return "Screen vision not available."

        sv = get_screen_vision()

        if not sv.automation._enabled:
            return "Automation is disabled. Enable with enable_automation tool first."

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, lambda: sv.automation.hotkey(*keys)
        )

        if success:
            return f"Pressed hotkey: {' + '.join(keys)}"
        return "Failed to press hotkey"


class GetDevContextTool(Tool):
    """Get the current development context."""

    name = "get_development_context"
    description = "Get what the user is currently working on: languages, frameworks, files, tasks, errors."
    parameters = {
        "type": "object",
        "properties": {},
    }

    async def execute(self) -> str:
        if not HAS_SCREEN_VISION:
            return "Screen vision not available."

        sv = get_screen_vision()
        ctx = sv.get_development_context()

        if not ctx:
            return "No development context detected. Try see_screen first, or no coding context on screen."

        lines = ["**Development Context**"]

        if ctx.get("languages"):
            lines.append(f"Languages: {', '.join(ctx['languages'])}")

        if ctx.get("frameworks"):
            lines.append(f"Frameworks: {', '.join(ctx['frameworks'])}")

        if ctx.get("files"):
            lines.append(f"Files: {', '.join(ctx['files'][:5])}")

        if ctx.get("tasks"):
            lines.append(f"Tasks: {', '.join(ctx['tasks'])}")

        if ctx.get("errors"):
            lines.append("Errors:")
            for err in ctx["errors"][:3]:
                lines.append(f"  - {err[:100]}")

        return "\n".join(lines)


class WhatAmIDoingTool(Tool):
    """Describe what the user is currently doing."""

    name = "what_am_i_doing"
    description = "Describe what the user is currently doing based on screen analysis. Requires prior see_screen call."
    parameters = {
        "type": "object",
        "properties": {},
    }

    async def execute(self) -> str:
        if not HAS_SCREEN_VISION:
            return "Screen vision not available."

        sv = get_screen_vision()
        return sv.what_am_i_doing()


class EnableAutomationTool(Tool):
    """Enable or disable screen automation."""

    name = "enable_automation"
    description = "Enable or disable mouse/keyboard automation for screen interaction."
    parameters = {
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "True to enable, False to disable",
            },
        },
        "required": ["enabled"],
    }
    requires_permission = True

    async def execute(self, enabled: bool) -> str:
        if not HAS_SCREEN_VISION:
            return "Screen vision not available."

        sv = get_screen_vision()

        if enabled:
            sv.automation.enable()
            return "Automation ENABLED. I can now control mouse and keyboard."
        else:
            sv.automation.disable()
            return "Automation DISABLED. I can no longer control mouse and keyboard."


class GetLearningStatsTool(Tool):
    """Get learning statistics from screen vision."""

    name = "get_screen_learning_stats"
    description = "Get statistics about what screen vision has learned from interactions."
    parameters = {
        "type": "object",
        "properties": {},
    }

    async def execute(self) -> str:
        if not HAS_SCREEN_VISION:
            return "Screen vision not available."

        sv = get_screen_vision()
        stats = sv.learner.get_learning_stats()

        lines = ["**Screen Vision Learning Stats**"]
        lines.append(f"Element templates: {stats['element_templates']}")
        lines.append(f"Text patterns: {stats['text_patterns']}")
        lines.append(f"App layouts learned: {stats['app_layouts']}")
        lines.append(f"Click successes: {stats['click_successes']}")
        lines.append(f"Click failures: {stats['click_failures']}")

        if stats['click_successes'] + stats['click_failures'] > 0:
            lines.append(f"Click success rate: {stats['click_success_rate']:.0%}")

        return "\n".join(lines)


class ScrollTool(Tool):
    """Scroll the screen."""

    name = "scroll"
    description = "Scroll up or down. Enable automation first."
    parameters = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "description": "Direction to scroll: 'up' or 'down'",
                "enum": ["up", "down"],
            },
            "amount": {
                "type": "integer",
                "description": "Number of scroll clicks",
                "default": 3,
            },
        },
        "required": ["direction"],
    }
    requires_permission = True

    async def execute(self, direction: str, amount: int = 3) -> str:
        if not HAS_SCREEN_VISION:
            return "Screen vision not available."

        sv = get_screen_vision()

        if not sv.automation._enabled:
            return "Automation is disabled. Enable with enable_automation tool first."

        clicks = amount if direction == "down" else -amount

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, lambda: sv.automation.scroll(clicks)
        )

        if success:
            return f"Scrolled {direction} {amount} clicks"
        return "Failed to scroll"


SCREEN_TOOLS = [
    SeeScreenTool,
    FindElementTool,
    ClickElementTool,
    TypeTextTool,
    PressKeyTool,
    HotkeyTool,
    GetDevContextTool,
    WhatAmIDoingTool,
    EnableAutomationTool,
    GetLearningStatsTool,
    ScrollTool,
]


def register_screen_tools(registry) -> int:
    """Register all screen tools with a tool registry or agent."""
    if not HAS_SCREEN_VISION:
        logger.warning("Screen vision not available, skipping tool registration")
        return 0

    tools = []
    for tool_class in SCREEN_TOOLS:
        try:
            tools.append(tool_class())
        except Exception as e:
            logger.error(f"Failed to create {tool_class.name}: {e}")

    if not tools:
        return 0

    # Agent uses register_tools(list), ToolRegistry uses register(tool)
    if hasattr(registry, 'register_tools'):
        registry.register_tools(tools)
    else:
        for tool in tools:
            registry.register(tool)

    logger.info(f"Registered {len(tools)} screen vision tools")
    return len(tools)


__all__ = ["SCREEN_TOOLS", "register_screen_tools"]
