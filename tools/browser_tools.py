"""Browser automation and web interaction tools for Jarvis.

Provides tools for:
- Browser automation (Playwright/Selenium)
- Web scraping with JavaScript rendering
- Screenshot capture
- Form filling and interaction
- Page navigation and content extraction

Inspired by OpenCode's web tools and browser automation capabilities.
"""
import asyncio
import logging
import time
from typing import Optional
from dataclasses import dataclass

from .base import Tool
from core.helpers import safe_error_message

logger = logging.getLogger(__name__)

HAS_PLAYWRIGHT = False
HAS_SELENIUM = False

try:
    from playwright.async_api import async_playwright, Browser, Page
    HAS_PLAYWRIGHT = True
except ImportError:
    pass

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    HAS_SELENIUM = True
except ImportError:
    pass


@dataclass
class BrowserSession:
    """Active browser session."""
    id: str
    browser: any
    page: any
    url: str
    title: str
    created_at: float
    last_used: float = 0.0
    pw: any = None  # Playwright instance — must be closed when browser closes


MAX_SESSIONS = 10
_sessions_lock = asyncio.Lock()
_shared_sessions: dict = {}


async def _get_session(sessions: dict, session_id: Optional[str]) -> Optional[BrowserSession]:
    """Thread-safe session lookup with last_used update."""
    async with _sessions_lock:
        if not sessions:
            return None
        sid = session_id or list(sessions.keys())[-1]
        session = sessions.get(sid)
        if session:
            session.last_used = time.time()
        return session


class BrowserLaunchTool(Tool):
    """Launch a browser instance."""

    name = "browser_launch"
    description = "Launch a new browser instance for automation. Returns session ID for subsequent operations."
    parameters = {
        "type": "object",
        "properties": {
            "headless": {
                "type": "boolean",
                "description": "Run in headless mode (default True)",
                "default": True,
            },
            "url": {
                "type": "string",
                "description": "Optional URL to navigate to on launch",
            },
            "browser_type": {
                "type": "string",
                "description": "Browser type: chromium, firefox, webkit (default chromium)",
                "default": "chromium",
            },
        },
    }
    requires_permission = True

    async def execute(
        self,
        headless: bool = True,
        url: Optional[str] = None,
        browser_type: str = "chromium",
    ) -> str:
        if HAS_PLAYWRIGHT:
            return await self._launch_playwright(headless, url, browser_type)
        elif HAS_SELENIUM:
            return await self._launch_selenium(headless, url)
        else:
            return "Browser automation not available. Install with: pip install playwright && playwright install"

    async def _launch_playwright(self, headless: bool, url: Optional[str], browser_type: str) -> str:
        try:
            from playwright.async_api import async_playwright
            import time

            pw = await async_playwright().start()

            if browser_type == "firefox":
                browser = await pw.firefox.launch(headless=headless)
            elif browser_type == "webkit":
                browser = await pw.webkit.launch(headless=headless)
            else:
                browser = await pw.chromium.launch(headless=headless)

            page = await browser.new_page()

            if url:
                await page.goto(url, wait_until="networkidle")

            session_id = f"browser_{int(time.time() * 1000)}"
            now = time.time()
            new_session = BrowserSession(
                id=session_id,
                browser=browser,
                pw=pw,
                page=page,
                url=url or "about:blank",
                title=await page.title() if url else "New Page",
                created_at=now,
                last_used=now,
            )
            evicted_browser = None
            evicted_pw = None
            async with _sessions_lock:
                if len(_shared_sessions) >= MAX_SESSIONS:
                    oldest_id = min(_shared_sessions, key=lambda k: _shared_sessions[k].last_used)
                    oldest = _shared_sessions.pop(oldest_id)
                    evicted_browser = oldest.browser
                    evicted_pw = oldest.pw
                _shared_sessions[session_id] = new_session

            if evicted_browser is not None:
                try:
                    await evicted_browser.close()
                except Exception as e:
                    logger.debug(f"Failed to close evicted Playwright browser: {e}")
            if evicted_pw is not None:
                try:
                    await evicted_pw.stop()
                except Exception:
                    pass

            return "\n".join([
                "Browser launched successfully",
                f"Session ID: {session_id}",
                f"Browser: {browser_type}",
                f"Headless: {headless}",
                f"URL: {url or 'about:blank'}",
            ])
        except Exception as e:
            return f"Failed to launch browser: {safe_error_message(e)}"

    async def _launch_selenium(self, headless: bool, url: Optional[str]) -> str:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            import time

            options = Options()
            if headless:
                options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")

            driver = webdriver.Chrome(options=options)

            if url:
                driver.get(url)

            session_id = f"browser_{int(time.time() * 1000)}"
            now = time.time()
            new_session = BrowserSession(
                id=session_id,
                browser=driver,
                page=driver,
                url=url or "about:blank",
                title=driver.title,
                created_at=now,
                last_used=now,
            )
            evicted_browser = None
            async with _sessions_lock:
                if len(_shared_sessions) >= MAX_SESSIONS:
                    oldest_id = min(_shared_sessions, key=lambda k: _shared_sessions[k].last_used)
                    oldest = _shared_sessions.pop(oldest_id)
                    evicted_browser = oldest.browser
                _shared_sessions[session_id] = new_session

            if evicted_browser is not None:
                try:
                    evicted_browser.quit()
                except Exception as e:
                    logger.debug(f"Failed to close evicted Selenium browser: {e}")

            return "\n".join([
                "Browser launched successfully (Selenium)",
                f"Session ID: {session_id}",
                f"Headless: {headless}",
                f"URL: {url or 'about:blank'}",
            ])
        except Exception as e:
            return f"Failed to launch Selenium browser: {safe_error_message(e)}"


class BrowserNavigateTool(Tool):
    """Navigate to a URL in the browser."""

    name = "browser_navigate"
    description = "Navigate the browser to a specific URL."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to navigate to",
            },
            "session_id": {
                "type": "string",
                "description": "Browser session ID (uses last session if not provided)",
            },
            "wait_until": {
                "type": "string",
                "description": "Wait condition: load, domcontentloaded, networkidle",
                "default": "networkidle",
            },
        },
        "required": ["url"],
    }
    requires_permission = True

    def __init__(self, sessions: dict = None):
        self._sessions = sessions or _shared_sessions

    async def execute(
        self,
        url: str,
        session_id: Optional[str] = None,
        wait_until: str = "networkidle",
    ) -> str:
        if not self._sessions:
            return "No browser session. Launch a browser first with browser_launch."

        session = await _get_session(self._sessions, session_id)

        if not session:
            return f"Session {session_id} not found."

        try:
            if HAS_PLAYWRIGHT:
                await session.page.goto(url, wait_until=wait_until)
                title = await session.page.title()
            else:
                session.page.get(url)
                title = session.page.title

            session.url = url
            session.title = title

            return "\n".join([
                f"Navigated to: {url}",
                f"Title: {title}",
            ])
        except Exception as e:
            return f"Navigation error: {safe_error_message(e)}"


class BrowserScreenshotTool(Tool):
    """Take a screenshot of the current page."""

    name = "browser_screenshot"
    description = "Take a screenshot of the browser page."
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Browser session ID",
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture full page (default False)",
                "default": False,
            },
            "selector": {
                "type": "string",
                "description": "CSS selector to capture specific element",
            },
        },
    }
    requires_permission = False

    def __init__(self, sessions: dict = None):
        self._sessions = sessions or _shared_sessions

    async def execute(
        self,
        session_id: Optional[str] = None,
        full_page: bool = False,
        selector: Optional[str] = None,
    ) -> str:
        if not self._sessions:
            return "No browser session. Launch a browser first with browser_launch."

        session = await _get_session(self._sessions, session_id)

        if not session:
            return f"Session {session_id} not found."

        try:
            from pathlib import Path
            import time

            screenshot_dir = Path.home() / ".jarvis" / "screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = screenshot_dir / f"screenshot_{int(time.time())}.png"

            if HAS_PLAYWRIGHT:
                if selector:
                    element = await session.page.query_selector(selector)
                    if element:
                        await element.screenshot(path=str(screenshot_path))
                    else:
                        return f"Element not found: {selector}"
                else:
                    await session.page.screenshot(path=str(screenshot_path), full_page=full_page)
            else:
                session.page.save_screenshot(str(screenshot_path))

            return "\n".join([
                f"Screenshot saved to: {screenshot_path}",
                f"Full page: {full_page}",
                f"Selector: {selector or 'none'}",
            ])
        except Exception as e:
            return f"Screenshot error: {safe_error_message(e)}"


class BrowserClickTool(Tool):
    """Click an element on the page."""

    name = "browser_click"
    description = "Click an element on the browser page by CSS selector or text."
    parameters = {
        "type": "object",
        "properties": {
            "selector": {
                "type": "string",
                "description": "CSS selector for the element to click",
            },
            "text": {
                "type": "string",
                "description": "Alternative: click element containing this text",
            },
            "session_id": {
                "type": "string",
                "description": "Browser session ID",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds (default 5000)",
                "default": 5000,
            },
        },
    }
    requires_permission = True

    def __init__(self, sessions: dict = None):
        self._sessions = sessions or _shared_sessions

    async def execute(
        self,
        selector: Optional[str] = None,
        text: Optional[str] = None,
        session_id: Optional[str] = None,
        timeout: int = 5000,
    ) -> str:
        if not self._sessions:
            return "No browser session. Launch a browser first with browser_launch."

        if not selector and not text:
            return "Provide either selector or text parameter."

        session = await _get_session(self._sessions, session_id)

        if not session:
            return f"Session {session_id} not found."

        try:
            if HAS_PLAYWRIGHT:
                if text:
                    element = session.page.get_by_text(text).first
                    await element.click(timeout=timeout)
                else:
                    await session.page.click(selector, timeout=timeout)
            else:
                if text:
                    safe_text = text.replace("'", "\\'")
                    element = session.page.find_element(By.XPATH, f"//*[contains(text(), '{safe_text}')]")
                else:
                    element = session.page.find_element(By.CSS_SELECTOR, selector)
                element.click()

            return f"Clicked: {text or selector}"
        except Exception as e:
            return f"Click error: {safe_error_message(e)}"


class BrowserTypeTool(Tool):
    """Type text into an input field."""

    name = "browser_type"
    description = "Type text into an input field on the browser page."
    parameters = {
        "type": "object",
        "properties": {
            "selector": {
                "type": "string",
                "description": "CSS selector for the input field",
            },
            "text": {
                "type": "string",
                "description": "Text to type",
            },
            "session_id": {
                "type": "string",
                "description": "Browser session ID",
            },
            "clear_first": {
                "type": "boolean",
                "description": "Clear field before typing (default True)",
                "default": True,
            },
            "press_enter": {
                "type": "boolean",
                "description": "Press Enter after typing (default False)",
                "default": False,
            },
        },
        "required": ["selector", "text"],
    }
    requires_permission = True

    def __init__(self, sessions: dict = None):
        self._sessions = sessions or _shared_sessions

    async def execute(
        self,
        selector: str,
        text: str,
        session_id: Optional[str] = None,
        clear_first: bool = True,
        press_enter: bool = False,
    ) -> str:
        if not self._sessions:
            return "No browser session. Launch a browser first with browser_launch."

        session = await _get_session(self._sessions, session_id)

        if not session:
            return f"Session {session_id} not found."

        try:
            if HAS_PLAYWRIGHT:
                if clear_first:
                    await session.page.fill(selector, text)
                else:
                    await session.page.type(selector, text)

                if press_enter:
                    await session.page.press(selector, "Enter")
            else:
                element = session.page.find_element(By.CSS_SELECTOR, selector)
                if clear_first:
                    element.clear()
                element.send_keys(text)

                if press_enter:
                    element.send_keys(Keys.RETURN)

            return f"Typed into {selector}: {text[:50]}{'...' if len(text) > 50 else ''}"
        except Exception as e:
            return f"Type error: {safe_error_message(e)}"


class BrowserGetContentTool(Tool):
    """Get page content."""

    name = "browser_get_content"
    description = "Get the text content of the current page or a specific element."
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Browser session ID",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector for specific element (optional)",
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum content length (default 10000)",
                "default": 10000,
            },
        },
    }
    requires_permission = False

    def __init__(self, sessions: dict = None):
        self._sessions = sessions or _shared_sessions

    async def execute(
        self,
        session_id: Optional[str] = None,
        selector: Optional[str] = None,
        max_length: int = 10000,
    ) -> str:
        if not self._sessions:
            return "No browser session. Launch a browser first with browser_launch."

        session = await _get_session(self._sessions, session_id)

        if not session:
            return f"Session {session_id} not found."

        try:
            if HAS_PLAYWRIGHT:
                if selector:
                    element = await session.page.query_selector(selector)
                    if element:
                        content = await element.inner_text()
                    else:
                        return f"Element not found: {selector}"
                else:
                    content = await session.page.content()
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(content, "html.parser")
                    for element in soup(["script", "style", "nav", "header", "footer"]):
                        element.decompose()
                    content = soup.get_text(separator="\n", strip=True)
            else:
                if selector:
                    element = session.page.find_element(By.CSS_SELECTOR, selector)
                    content = element.text
                else:
                    content = session.page.page_source
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(content, "html.parser")
                    for element in soup(["script", "style", "nav", "header", "footer"]):
                        element.decompose()
                    content = soup.get_text(separator="\n", strip=True)

            if len(content) > max_length:
                content = content[:max_length] + "\n... (truncated)"

            return "\n".join([
                f"URL: {session.url}",
                f"Title: {session.title}",
                f"Selector: {selector or 'full page'}",
                "",
                content,
            ])
        except Exception as e:
            return f"Get content error: {safe_error_message(e)}"


class BrowserWaitTool(Tool):
    """Wait for an element or condition."""

    name = "browser_wait"
    description = "Wait for an element to appear or a condition to be met."
    parameters = {
        "type": "object",
        "properties": {
            "selector": {
                "type": "string",
                "description": "CSS selector to wait for",
            },
            "session_id": {
                "type": "string",
                "description": "Browser session ID",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in milliseconds (default 10000)",
                "default": 10000,
            },
            "state": {
                "type": "string",
                "description": "State to wait for: visible, hidden, attached, detached",
                "default": "visible",
            },
        },
        "required": ["selector"],
    }
    requires_permission = False

    def __init__(self, sessions: dict = None):
        self._sessions = sessions or _shared_sessions

    async def execute(
        self,
        selector: str,
        session_id: Optional[str] = None,
        timeout: int = 10000,
        state: str = "visible",
    ) -> str:
        if not self._sessions:
            return "No browser session. Launch a browser first with browser_launch."

        session = await _get_session(self._sessions, session_id)

        if not session:
            return f"Session {session_id} not found."

        try:
            if HAS_PLAYWRIGHT:
                await session.page.wait_for_selector(selector, timeout=timeout, state=state)
            else:
                wait = WebDriverWait(session.page, timeout / 1000)
                if state == "visible":
                    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, selector)))
                else:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))

            return f"Element {selector} is {state}"
        except Exception as e:
            return f"Wait error: {safe_error_message(e)}"


class BrowserCloseTool(Tool):
    """Close a browser session."""

    name = "browser_close"
    description = "Close a browser session."
    parameters = {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Browser session ID (closes all if not provided)",
            },
        },
    }
    requires_permission = True

    def __init__(self, sessions: dict = None):
        self._sessions = sessions or _shared_sessions

    async def execute(self, session_id: Optional[str] = None) -> str:
        if not self._sessions:
            return "No browser sessions to close."

        if session_id:
            async with _sessions_lock:
                session = self._sessions.pop(session_id, None)
            if not session:
                return f"Session {session_id} not found."

            try:
                if HAS_PLAYWRIGHT:
                    await session.browser.close()
                    if session.pw:
                        await session.pw.stop()
                else:
                    session.browser.quit()
            except Exception as e:
                logger.warning(f"Failed to close browser session {session_id}: {e}")

            return f"Closed session: {session_id}"
        else:
            async with _sessions_lock:
                sessions_to_close = list(self._sessions.items())
                self._sessions.clear()
            closed = []
            for sid, session in sessions_to_close:
                try:
                    if HAS_PLAYWRIGHT:
                        await session.browser.close()
                        if session.pw:
                            await session.pw.stop()
                    else:
                        session.browser.quit()
                    closed.append(sid)
                except Exception as e:
                    logger.warning(f"Failed to close browser session {sid}: {e}")
            return f"Closed {len(closed)} browser session(s)"


class BrowserEvaluateTool(Tool):
    """Execute JavaScript in the browser."""

    name = "browser_evaluate"
    description = "Execute JavaScript code in the browser page."
    parameters = {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": "JavaScript code to execute",
            },
            "session_id": {
                "type": "string",
                "description": "Browser session ID",
            },
        },
        "required": ["script"],
    }
    requires_permission = True

    def __init__(self, sessions: dict = None):
        self._sessions = sessions or _shared_sessions

    async def execute(
        self,
        script: str,
        session_id: Optional[str] = None,
    ) -> str:
        if not self._sessions:
            return "No browser session. Launch a browser first with browser_launch."

        session = await _get_session(self._sessions, session_id)

        if not session:
            return f"Session {session_id} not found."

        try:
            if HAS_PLAYWRIGHT:
                result = await session.page.evaluate(script)
            else:
                result = session.page.execute_script(script)

            return f"Result: {result}"
        except Exception as e:
            return f"JavaScript error: {safe_error_message(e)}"


class BrowserListTool(Tool):
    """List active browser sessions."""

    name = "browser_list"
    description = "List all active browser sessions."
    parameters = {
        "type": "object",
        "properties": {},
    }
    requires_permission = False

    def __init__(self, sessions: dict = None):
        self._sessions = sessions or _shared_sessions

    async def execute(self) -> str:
        async with _sessions_lock:
            if not self._sessions:
                return "No active browser sessions."

            lines = [f"Active browser sessions: {len(self._sessions)}"]
            for sid, session in self._sessions.items():
                lines.append(f"  - {sid}: {session.url} ({session.title})")

            return "\n".join(lines)


BROWSER_TOOLS = [
    BrowserLaunchTool,
    BrowserNavigateTool,
    BrowserScreenshotTool,
    BrowserClickTool,
    BrowserTypeTool,
    BrowserGetContentTool,
    BrowserWaitTool,
    BrowserCloseTool,
    BrowserEvaluateTool,
    BrowserListTool,
]


def register_browser_tools(registry, sessions: dict = None) -> int:
    """Register all browser tools with a tool registry or agent."""
    if not HAS_PLAYWRIGHT and not HAS_SELENIUM:
        logger.warning("Browser automation not available. Install playwright or selenium.")
        return 0

    sessions = sessions or _shared_sessions

    tools = []
    for tool_class in BROWSER_TOOLS:
        try:
            if tool_class in (BrowserNavigateTool, BrowserScreenshotTool, BrowserClickTool,
                            BrowserTypeTool, BrowserGetContentTool, BrowserWaitTool,
                            BrowserCloseTool, BrowserEvaluateTool, BrowserListTool):
                tools.append(tool_class(sessions=sessions))
            else:
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

    logger.info(f"Registered {len(tools)} browser automation tools")
    return len(tools)


__all__ = ["BROWSER_TOOLS", "register_browser_tools", "BrowserSession"]
