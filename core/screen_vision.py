"""Screen Vision and Interaction System for Jarvis.

Enables Jarvis to:
1. See what's on the user's screen
2. Extract text via OCR
3. Detect UI elements (buttons, inputs, windows)
4. Understand development context (IDE, terminal, browser)
5. Interact via mouse/keyboard automation
6. Help with development tasks alongside the user

Architecture:
- Screen capture: mss (fast, cross-platform)
- OCR: pytesseract/easyocr fallback
- UI detection: Computer vision + heuristics
- Automation: pynput (cross-platform, low-level)
- Context analysis: Template matching + text analysis
"""
import base64
import hashlib
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

HAS_SCREEN_DEPS = False
HAS_OCR = False
HAS_AUTOMATION = False

try:
    import mss
    import mss.tools
    from PIL import Image
    HAS_SCREEN_DEPS = True
except ImportError:
    pass

try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    try:
        import easyocr
        HAS_OCR = True
    except ImportError:
        pass

try:
    import pynput
    from pynput import mouse, keyboard
    HAS_AUTOMATION = True
except ImportError:
    pass


class ScreenContext(Enum):
    """Detected screen context types."""
    UNKNOWN = 0
    IDE_CODE = 1
    TERMINAL = 2
    BROWSER_WEB = 3
    BROWSER_DOCS = 4
    BROWSER_GITHUB = 5
    FILE_MANAGER = 6
    CHAT_MESSAGING = 7
    DESIGN_TOOL = 8
    SPREADSHEET = 9
    EMAIL = 10
    VIDEO_CALL = 11
    DESKTOP = 12


class UIElementType(Enum):
    """Types of detectable UI elements."""
    UNKNOWN = 0
    BUTTON = 1
    INPUT_FIELD = 2
    TEXT_AREA = 3
    CHECKBOX = 4
    RADIO_BUTTON = 5
    DROPDOWN = 6
    MENU_ITEM = 7
    TAB = 8
    ICON = 9
    LINK = 10
    WINDOW = 11
    DIALOG = 12
    CODE_EDITOR = 13
    TERMINAL_PROMPT = 14


@dataclass
class BoundingBox:
    """Bounding box for UI elements."""
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class UIElement:
    """Detected UI element."""
    element_type: UIElementType
    bbox: BoundingBox
    text: Optional[str] = None
    confidence: float = 0.5
    attributes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.element_type.name,
            "bbox": self.bbox.to_dict(),
            "text": self.text,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }


@dataclass
class TextBlock:
    """Detected text block from OCR."""
    text: str
    bbox: BoundingBox
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
        }


@dataclass
class ScreenCapture:
    """Screen capture result."""
    image: np.ndarray
    width: int
    height: int
    monitor: int
    timestamp: float
    image_hash: str

    def to_base64(self, format: str = "PNG") -> str:
        if not HAS_SCREEN_DEPS:
            return ""
        img = Image.fromarray(self.image)
        buf = io.BytesIO()
        img.save(buf, format=format)
        return base64.b64encode(buf.getvalue()).decode('utf-8')


@dataclass
class ScreenAnalysis:
    """Complete screen analysis result."""
    capture: ScreenCapture
    context: ScreenContext
    context_confidence: float
    text_blocks: list[TextBlock]
    ui_elements: list[UIElement]
    windows: list[dict]
    summary: str
    development_context: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "context": self.context.name,
            "context_confidence": self.context_confidence,
            "text_count": len(self.text_blocks),
            "ui_element_count": len(self.ui_elements),
            "window_count": len(self.windows),
            "summary": self.summary,
            "development_context": self.development_context,
            "timestamp": self.capture.timestamp,
        }


class ScreenCaptureEngine:
    """Handles screen capture functionality."""

    def __init__(self):
        self._sct = None
        self._last_capture: Optional[ScreenCapture] = None
        self._capture_count = 0

    def _get_sct(self):
        if self._sct is None and HAS_SCREEN_DEPS:
            self._sct = mss.mss()
        return self._sct

    def get_monitors(self) -> list[dict]:
        if not HAS_SCREEN_DEPS:
            return [{"id": 0, "name": "default", "width": 1920, "height": 1080}]

        sct = self._get_sct()
        monitors = []
        for i, mon in enumerate(sct.monitors):
            monitors.append({
                "id": i,
                "name": f"Monitor {i}" if i > 0 else "All",
                "width": mon.get("width", 1920),
                "height": mon.get("height", 1080),
            })
        return monitors

    def capture(self, monitor: int = 1, region: Optional[tuple[int, int, int, int]] = None) -> ScreenCapture:
        if not HAS_SCREEN_DEPS:
            return self._dummy_capture()

        sct = self._get_sct()

        if region:
            monitor_def = {
                "left": region[0],
                "top": region[1],
                "width": region[2] - region[0],
                "height": region[3] - region[1],
            }
        else:
            monitor_def = monitor if isinstance(monitor, dict) else sct.monitors[monitor]

        screenshot = sct.grab(monitor_def)
        img = np.array(screenshot)

        if img.shape[2] == 4:
            img = img[:, :, :3]

        height, width = img.shape[:2]
        img_hash = hashlib.sha256(img.tobytes()).hexdigest()[:16]

        self._capture_count += 1
        self._last_capture = ScreenCapture(
            image=img,
            width=width,
            height=height,
            monitor=monitor if not region else -1,
            timestamp=time.time(),
            image_hash=img_hash,
        )

        return self._last_capture

    def capture_window(self, window_title: str) -> Optional[ScreenCapture]:
        if not HAS_SCREEN_DEPS:
            return self._dummy_capture()

        try:
            import pygetwindow as gw
            windows = gw.getWindowsWithTitle(window_title)
            if windows:
                win = windows[0]
                if win.left >= 0 and win.top >= 0:
                    region = (win.left, win.top, win.right, win.bottom)
                    return self.capture(region=region)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Window capture error: {e}")

        return None

    def _dummy_capture(self) -> ScreenCapture:
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        return ScreenCapture(
            image=img,
            width=1920,
            height=1080,
            monitor=0,
            timestamp=time.time(),
            image_hash="dummy",
        )

    def get_stats(self) -> dict:
        return {
            "captures_taken": self._capture_count,
            "has_screen_deps": HAS_SCREEN_DEPS,
            "last_capture_time": self._last_capture.timestamp if self._last_capture else None,
        }


class OCREngine:
    """Handles text extraction from images."""

    def __init__(self):
        self._tesseract_available = False
        self._easyocr_reader = None
        self._check_ocr()

    def _check_ocr(self):
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
        except Exception as e:
            logger.debug(f"Tesseract not available: {e}")

    def _get_easyocr(self):
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                logger.debug(f"EasyOCR init failed: {e}")
        return self._easyocr_reader

    def extract_text(self, image: np.ndarray) -> list[TextBlock]:
        if not HAS_OCR:
            return self._dummy_text()

        blocks = []

        if self._tesseract_available:
            blocks = self._tesseract_ocr(image)

        if not blocks:
            reader = self._get_easyocr()
            if reader:
                blocks = self._easyocr_ocr(image, reader)

        return blocks

    def _tesseract_ocr(self, image: np.ndarray) -> list[TextBlock]:
        try:
            import pytesseract
            from PIL import Image

            pil_img = Image.fromarray(image)
            data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

            blocks = []
            n_boxes = len(data.get('text', []))

            for i in range(n_boxes):
                text = data['text'][i].strip()
                if not text:
                    continue

                conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0

                blocks.append(TextBlock(
                    text=text,
                    bbox=BoundingBox(
                        x=data['left'][i],
                        y=data['top'][i],
                        width=data['width'][i],
                        height=data['height'][i],
                    ),
                    confidence=conf / 100.0 if conf > 1 else conf,
                ))

            return self._merge_nearby_blocks(blocks)
        except Exception as e:
            logger.debug(f"Tesseract OCR error: {e}")
            return []

    def _easyocr_ocr(self, image: np.ndarray, reader) -> list[TextBlock]:
        try:
            results = reader.readtext(image)

            blocks = []
            for detection in results:
                bbox_points, text, conf = detection

                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]

                blocks.append(TextBlock(
                    text=text,
                    bbox=BoundingBox(
                        x=int(min(xs)),
                        y=int(min(ys)),
                        width=int(max(xs) - min(xs)),
                        height=int(max(ys) - min(ys)),
                    ),
                    confidence=conf,
                ))

            return blocks
        except Exception as e:
            logger.debug(f"EasyOCR error: {e}")
            return []

    def _merge_nearby_blocks(self, blocks: list[TextBlock], threshold: int = 20) -> list[TextBlock]:
        if len(blocks) < 2:
            return blocks

        merged = []
        used = set()

        for i, block in enumerate(blocks):
            if i in used:
                continue

            group = [block]
            used.add(i)

            for j, other in enumerate(blocks):
                if j in used:
                    continue

                if abs(block.bbox.y - other.bbox.y) < threshold:
                    if abs(block.bbox.x + block.bbox.width - other.bbox.x) < threshold:
                        group.append(other)
                        used.add(j)

            if len(group) > 1:
                all_text = " ".join(b.text for b in group)
                min_x = min(b.bbox.x for b in group)
                min_y = min(b.bbox.y for b in group)
                max_right = max(b.bbox.x + b.bbox.width for b in group)
                max_bottom = max(b.bbox.y + b.bbox.height for b in group)

                merged.append(TextBlock(
                    text=all_text,
                    bbox=BoundingBox(x=min_x, y=min_y, width=max_right - min_x, height=max_bottom - min_y),
                    confidence=max(b.confidence for b in group),
                ))
            else:
                merged.append(block)

        return merged

    def _dummy_text(self) -> list[TextBlock]:
        return [TextBlock(
            text="[OCR not available - install pytesseract or easyocr]",
            bbox=BoundingBox(x=0, y=0, width=400, height=30),
            confidence=0.0,
        )]


class UIElementDetector:
    """Detects UI elements in screenshots."""

    def __init__(self):
        self._templates = {}
        self._button_colors = [
            (0.1, 0.4, 0.6, 0.8),
            (0.2, 0.2, 0.8, 1.0),
            (0.3, 0.1, 0.4, 0.7),
        ]

    def detect(self, image: np.ndarray, text_blocks: list[TextBlock]) -> list[UIElement]:
        elements = []

        elements.extend(self._detect_buttons(image, text_blocks))
        elements.extend(self._detect_input_fields(image, text_blocks))
        elements.extend(self._detect_code_regions(image, text_blocks))
        elements.extend(self._detect_terminal_prompts(image, text_blocks))

        return elements

    def _detect_buttons(self, image: np.ndarray, text_blocks: list[TextBlock]) -> list[UIElement]:
        buttons = []
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        h, w = gray.shape
        regions = []

        for y in range(0, h - 30, 20):
            for x in range(0, w - 60, 30):
                region = gray[y:y+30, x:x+60]
                if region.size == 0:
                    continue

                std = np.std(region)
                mean = np.mean(region)

                edges = np.abs(np.diff(region, axis=0)).sum() + np.abs(np.diff(region, axis=1)).sum()

                if std < 30 and edges > 50:
                    regions.append((x, y, 60, 30, mean, edges))

        for x, y, bw, bh, mean, edges in regions[:50]:
            matching_text = None
            for tb in text_blocks:
                if tb.bbox.x >= x and tb.bbox.y >= y:
                    if tb.bbox.x < x + bw and tb.bbox.y < y + bh:
                        matching_text = tb.text
                        break

            buttons.append(UIElement(
                element_type=UIElementType.BUTTON,
                bbox=BoundingBox(x=x, y=y, width=bw, height=bh),
                text=matching_text,
                confidence=0.6 if matching_text else 0.4,
                attributes={"mean_intensity": float(mean)},
            ))

        return buttons[:20]

    def _detect_input_fields(self, image: np.ndarray, text_blocks: list[TextBlock]) -> list[UIElement]:
        inputs = []
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image

        h, w = gray.shape

        for y in range(0, h - 25, 30):
            for x in range(0, w - 150, 50):
                region = gray[y:y+25, x:x+150]
                if region.size == 0:
                    continue

                std = np.std(region)

                edge_h = np.abs(np.diff(region, axis=0)).sum()
                np.abs(np.diff(region, axis=1)).sum()

                if std < 20 and edge_h > 100:
                    has_text = any(
                        tb.bbox.x >= x and tb.bbox.x < x + 150 and
                        tb.bbox.y >= y and tb.bbox.y < y + 25
                        for tb in text_blocks
                    )

                    inputs.append(UIElement(
                        element_type=UIElementType.INPUT_FIELD,
                        bbox=BoundingBox(x=x, y=y, width=150, height=25),
                        confidence=0.5 if not has_text else 0.3,
                    ))

        return inputs[:10]

    def _detect_code_regions(self, image: np.ndarray, text_blocks: list[TextBlock]) -> list[UIElement]:
        code_regions = []

        code_indicators = ['def ', 'class ', 'import ', 'function', 'const ', 'let ', 'async ']

        for tb in text_blocks:
            if any(ind in tb.text for ind in code_indicators):
                x, y = tb.bbox.x, tb.bbox.y
                w = max(400, tb.bbox.width)
                h = max(200, tb.bbox.height * 10)

                code_regions.append(UIElement(
                    element_type=UIElementType.CODE_EDITOR,
                    bbox=BoundingBox(x=x, y=y, width=w, height=min(h, 800)),
                    text=tb.text[:100],
                    confidence=0.8,
                ))

        return code_regions[:3]

    def _detect_terminal_prompts(self, image: np.ndarray, text_blocks: list[TextBlock]) -> list[UIElement]:
        prompts = []

        prompt_patterns = ['$', '#', '>', '~/', './', 'python', 'node', 'npm']

        for tb in text_blocks:
            text = tb.text.strip()
            for pattern in prompt_patterns:
                if text.startswith(pattern) or pattern in text[:10]:
                    prompts.append(UIElement(
                        element_type=UIElementType.TERMINAL_PROMPT,
                        bbox=tb.bbox,
                        text=tb.text,
                        confidence=0.7,
                    ))
                    break

        return prompts[:5]


class ContextAnalyzer:
    """Analyzes screen content to determine context."""

    def __init__(self):
        self._ide_indicators = [
            'vscode', 'visual studio', 'pycharm', 'intellij', 'sublime',
            'atom', 'vim', 'neovim', 'emacs', 'code::blocks',
            'def ', 'class ', 'import ', 'function ', 'const ', 'async ',
            '.py', '.js', '.ts', '.go', '.rs', '.java', '.cpp',
        ]
        self._terminal_indicators = [
            '$', '#', '~/', './', 'bash', 'zsh', 'sh ', 'cmd ', 'powershell',
            'python ', 'node ', 'npm ', 'pip ', 'git ', 'docker ',
        ]
        self._browser_indicators = [
            'http://', 'https://', 'www.', '.com', '.org', '.io',
            'chrome', 'firefox', 'safari', 'edge', 'browser',
        ]
        self._github_indicators = [
            'github.com', 'repository', 'pull request', 'issue', 'commit',
            'branch', 'merge', 'fork', 'star', 'clone',
        ]
        self._docs_indicators = [
            'documentation', 'readme', 'getting started', 'api reference',
            'tutorial', 'guide', 'example', 'learn',
        ]

    def analyze(self, image: np.ndarray, text_blocks: list[TextBlock]) -> tuple[ScreenContext, float]:
        all_text = " ".join(tb.text.lower() for tb in text_blocks if tb.text)

        scores = {
            ScreenContext.IDE_CODE: self._score_indicators(all_text, self._ide_indicators),
            ScreenContext.TERMINAL: self._score_indicators(all_text, self._terminal_indicators),
            ScreenContext.BROWSER_WEB: self._score_indicators(all_text, self._browser_indicators),
            ScreenContext.BROWSER_GITHUB: self._score_indicators(all_text, self._github_indicators),
            ScreenContext.BROWSER_DOCS: self._score_indicators(all_text, self._docs_indicators),
        }

        best_context = max(scores, key=scores.get)
        best_score = scores[best_context]

        if best_score < 0.1:
            return ScreenContext.UNKNOWN, 0.0

        return best_context, min(1.0, best_score)

    def _score_indicators(self, text: str, indicators: list[str]) -> float:
        matches = sum(1 for ind in indicators if ind.lower() in text)
        return matches / max(1, len(indicators)) * 10


class DevelopmentContextAnalyzer:
    """Analyzes development-specific context."""

    def analyze(self, text_blocks: list[TextBlock], ui_elements: list[UIElement]) -> Optional[dict]:
        all_text = " ".join(tb.text for tb in text_blocks if tb.text)

        context = {
            "languages": self._detect_languages(all_text),
            "frameworks": self._detect_frameworks(all_text),
            "files": self._detect_files(all_text),
            "tasks": self._detect_tasks(all_text, ui_elements),
            "errors": self._detect_errors(all_text),
        }

        return context if any(context.values()) else None

    def _detect_languages(self, text: str) -> list[str]:
        languages = []
        lang_indicators = {
            'python': ['def ', 'import ', 'from ', 'self.', '__init__'],
            'javascript': ['function ', 'const ', 'let ', 'var ', '=>', 'async '],
            'typescript': [': string', ': number', ': boolean', 'interface ', 'type '],
            'go': ['func ', 'package ', 'import "fmt"', 'go.mod'],
            'rust': ['fn ', 'let mut', 'impl ', '::', 'cargo'],
            'java': ['public class', 'private ', 'void ', 'System.out'],
        }

        for lang, indicators in lang_indicators.items():
            if sum(1 for ind in indicators if ind in text) >= 2:
                languages.append(lang)

        return languages

    def _detect_frameworks(self, text: str) -> list[str]:
        frameworks = []
        fw_indicators = {
            'react': ['useState', 'useEffect', 'jsx', 'component'],
            'vue': ['vue', 'v-if', 'v-for', 'computed'],
            'django': ['django', 'models.Model', 'views.py'],
            'flask': ['flask', '@app.route'],
            'express': ['express', 'app.get', 'app.post'],
            'fastapi': ['fastapi', '@app.get', 'pydantic'],
        }

        for fw, indicators in fw_indicators.items():
            if any(ind.lower() in text.lower() for ind in indicators):
                frameworks.append(fw)

        return frameworks

    def _detect_files(self, text: str) -> list[str]:
        import re
        file_patterns = [
            r'[\w/]+\.(py|js|ts|go|rs|java|cpp|c|h|html|css|json|yaml|yml|md)',
        ]

        files = []
        for pattern in file_patterns:
            matches = re.findall(pattern, text)
            files.extend(matches[:10])

        return list(set(files))[:10]

    def _detect_tasks(self, text: str, ui_elements: list[UIElement]) -> list[str]:
        tasks = []

        task_patterns = [
            r'(fix|bug|issue|error)',
            r'(feature|add|implement)',
            r'(refactor|clean|optimize)',
            r'(test|spec)',
            r'(document|docs|readme)',
        ]

        text_lower = text.lower()
        for pattern in task_patterns:
            import re
            if re.search(pattern, text_lower):
                tasks.append(pattern.split('|')[0].strip('()'))

        for elem in ui_elements:
            if elem.element_type == UIElementType.CODE_EDITOR:
                tasks.append("coding")
            if elem.element_type == UIElementType.TERMINAL_PROMPT:
                tasks.append("terminal_work")

        return list(set(tasks))

    def _detect_errors(self, text: str) -> list[str]:
        import re
        error_patterns = [
            r'(Error|Exception|Traceback)[^\n]*',
            r'FAILED[^\n]*',
            r'(warning|warn)[^\n]*',
        ]

        errors = []
        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            errors.extend(matches[:5])

        return errors


class AutomationController:
    """Controls mouse and keyboard for automation."""

    def __init__(self):
        self._mouse_controller = None
        self._keyboard_controller = None
        self._actions_log: list[dict] = []
        self._enabled = False

    def _init_controllers(self):
        if not HAS_AUTOMATION:
            return False

        if self._mouse_controller is None:
            from pynput import mouse, keyboard
            self._mouse_controller = mouse.Controller()
            self._keyboard_controller = keyboard.Controller()

        return True

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def move_mouse(self, x: int, y: int, smooth: bool = True) -> bool:
        if not self._enabled or not self._init_controllers():
            return False

        try:
            if smooth:
                current = self._mouse_controller.position
                steps = 20
                for i in range(steps):
                    t = i / steps
                    nx = int(current[0] + (x - current[0]) * t)
                    ny = int(current[1] + (y - current[1]) * t)
                    self._mouse_controller.position = (nx, ny)
                    time.sleep(0.01)

            self._mouse_controller.position = (x, y)
            self._log_action("mouse_move", {"x": x, "y": y})
            return True
        except Exception as e:
            logger.error(f"Mouse move error: {e}")
            return False

    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", clicks: int = 1) -> bool:
        if not self._enabled or not self._init_controllers():
            return False

        try:
            from pynput import mouse

            if x is not None and y is not None:
                self._mouse_controller.position = (x, y)

            btn = mouse.Button.left if button == "left" else mouse.Button.right

            for _ in range(clicks):
                self._mouse_controller.press(btn)
                time.sleep(0.05)
                self._mouse_controller.release(btn)

            self._log_action("click", {"x": x, "y": y, "button": button, "clicks": clicks})
            return True
        except Exception as e:
            logger.error(f"Click error: {e}")
            return False

    def type_text(self, text: str, interval: float = 0.02) -> bool:
        if not self._enabled or not self._init_controllers():
            return False

        try:
            for char in text:
                self._keyboard_controller.type(char)
                time.sleep(interval)

            self._log_action("type", {"text_length": len(text)})
            return True
        except Exception as e:
            logger.error(f"Type error: {e}")
            return False

    def press_key(self, key: str, modifiers: Optional[list[str]] = None) -> bool:
        if not self._enabled or not self._init_controllers():
            return False

        try:
            from pynput import keyboard

            if modifiers:
                for mod in modifiers:
                    mod_key = getattr(keyboard.Key, mod, None) or getattr(keyboard.KeyCode, mod, None)
                    if mod_key:
                        self._keyboard_controller.press(mod_key)

            key_obj = getattr(keyboard.Key, key, None) or keyboard.KeyCode.from_char(key)
            if key_obj:
                self._keyboard_controller.press(key_obj)
                time.sleep(0.05)
                self._keyboard_controller.release(key_obj)

            if modifiers:
                for mod in reversed(modifiers):
                    mod_key = getattr(keyboard.Key, mod, None) or getattr(keyboard.KeyCode, mod, None)
                    if mod_key:
                        self._keyboard_controller.release(mod_key)

            self._log_action("key_press", {"key": key, "modifiers": modifiers})
            return True
        except Exception as e:
            logger.error(f"Key press error: {e}")
            return False

    def hotkey(self, *keys: str) -> bool:
        if not self._enabled or not self._init_controllers():
            return False

        try:
            from pynput import keyboard

            key_objs = []
            for key in keys:
                key_obj = getattr(keyboard.Key, key, None) or keyboard.KeyCode.from_char(key)
                if key_obj:
                    key_objs.append(key_obj)
                    self._keyboard_controller.press(key_obj)

            time.sleep(0.05)

            for key_obj in reversed(key_objs):
                self._keyboard_controller.release(key_obj)

            self._log_action("hotkey", {"keys": list(keys)})
            return True
        except Exception as e:
            logger.error(f"Hotkey error: {e}")
            return False

    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        if not self._enabled or not self._init_controllers():
            return False

        try:
            if x is not None and y is not None:
                self._mouse_controller.position = (x, y)

            self._mouse_controller.scroll(0, clicks)
            self._log_action("scroll", {"clicks": clicks})
            return True
        except Exception as e:
            logger.error(f"Scroll error: {e}")
            return False

    def _log_action(self, action: str, details: dict):
        self._actions_log.append({
            "action": action,
            "details": details,
            "timestamp": time.time(),
        })

        if len(self._actions_log) > 100:
            self._actions_log = self._actions_log[-100:]

    def get_actions_log(self) -> list[dict]:
        return self._actions_log.copy()

    def get_stats(self) -> dict:
        return {
            "enabled": self._enabled,
            "has_automation": HAS_AUTOMATION,
            "actions_logged": len(self._actions_log),
        }


class ScreenVision:
    """Main screen vision and interaction system."""

    def __init__(self, enable_automation: bool = False, storage_path: Optional[str] = None):
        self.capture_engine = ScreenCaptureEngine()
        self.ocr_engine = OCREngine()
        self.ui_detector = UIElementDetector()
        self.context_analyzer = ContextAnalyzer()
        self.dev_context_analyzer = DevelopmentContextAnalyzer()
        self.automation = AutomationController()
        self.learner = ScreenVisionLearner(storage_path=storage_path)

        if enable_automation:
            self.automation.enable()

        self._analysis_history: list[ScreenAnalysis] = []
        self._last_context: Optional[ScreenContext] = None

    def see(self, monitor: int = 1, region: Optional[tuple] = None) -> ScreenAnalysis:
        capture = self.capture_engine.capture(monitor=monitor, region=region)

        text_blocks = self.ocr_engine.extract_text(capture.image)

        ui_elements = self.ui_detector.detect(capture.image, text_blocks)

        for elem in ui_elements:
            elem.confidence = self.learner.boost_confidence(elem, self._last_context or ScreenContext.UNKNOWN)
            if elem.text:
                self.learner.record_text_pattern(elem.text, elem.element_type)

        context, confidence = self.context_analyzer.analyze(capture.image, text_blocks)
        self._last_context = context

        windows = self._detect_windows(capture.image, text_blocks)

        for win in windows:
            if win.get("title"):
                self.learner.record_app_layout(win["title"], ui_elements)

        dev_context = None
        if context in (ScreenContext.IDE_CODE, ScreenContext.TERMINAL, ScreenContext.BROWSER_GITHUB):
            dev_context = self.dev_context_analyzer.analyze(text_blocks, ui_elements)

        summary = self._generate_summary(context, text_blocks, ui_elements, dev_context)

        analysis = ScreenAnalysis(
            capture=capture,
            context=context,
            context_confidence=confidence,
            text_blocks=text_blocks,
            ui_elements=ui_elements,
            windows=windows,
            summary=summary,
            development_context=dev_context,
        )

        self._analysis_history.append(analysis)
        if len(self._analysis_history) > 50:
            self._analysis_history = self._analysis_history[-50:]

        return analysis

    def see_window(self, window_title: str) -> Optional[ScreenAnalysis]:
        capture = self.capture_engine.capture_window(window_title)
        if capture is None:
            return None

        text_blocks = self.ocr_engine.extract_text(capture.image)
        ui_elements = self.ui_detector.detect(capture.image, text_blocks)
        context, confidence = self.context_analyzer.analyze(capture.image, text_blocks)
        dev_context = self.dev_context_analyzer.analyze(text_blocks, ui_elements)

        return ScreenAnalysis(
            capture=capture,
            context=context,
            context_confidence=confidence,
            text_blocks=text_blocks,
            ui_elements=ui_elements,
            windows=[{"title": window_title}],
            summary=self._generate_summary(context, text_blocks, ui_elements, dev_context),
            development_context=dev_context,
        )

    def find_element(self, text: str, element_type: Optional[UIElementType] = None) -> Optional[UIElement]:
        if not self._analysis_history:
            return None

        latest = self._analysis_history[-1]

        for elem in latest.ui_elements:
            if element_type and elem.element_type != element_type:
                continue

            if elem.text and text.lower() in elem.text.lower():
                return elem

        for tb in latest.text_blocks:
            if text.lower() in tb.text.lower():
                return UIElement(
                    element_type=UIElementType.UNKNOWN,
                    bbox=tb.bbox,
                    text=tb.text,
                    confidence=tb.confidence,
                )

        return None

    def find_all_elements(self, text: str, element_type: Optional[UIElementType] = None) -> list[UIElement]:
        if not self._analysis_history:
            return []

        latest = self._analysis_history[-1]
        results = []

        for elem in latest.ui_elements:
            if element_type and elem.element_type != element_type:
                continue

            if elem.text and text.lower() in elem.text.lower():
                results.append(elem)

        return results

    def click_element(self, element: UIElement) -> bool:
        x, y = element.bbox.center
        return self.automation.click(x, y)

    def type_in_element(self, element: UIElement, text: str) -> bool:
        if not self.click_element(element):
            return False
        time.sleep(0.1)
        return self.automation.type_text(text)

    def get_development_context(self) -> Optional[dict]:
        if not self._analysis_history:
            return None
        return self._analysis_history[-1].development_context

    def what_am_i_doing(self) -> str:
        if not self._analysis_history:
            return "I haven't seen your screen yet."

        latest = self._analysis_history[-1]

        if latest.context == ScreenContext.IDE_CODE:
            dev_ctx = latest.development_context
            if dev_ctx:
                langs = ", ".join(dev_ctx.get("languages", []))
                tasks = ", ".join(dev_ctx.get("tasks", []))
                errors = dev_ctx.get("errors", [])

                msg = f"You're coding in {langs or 'unknown language'}. "
                if tasks:
                    msg += f"Tasks: {tasks}. "
                if errors:
                    msg += f"I see {len(errors)} error(s)."
                return msg
            return "You're working in an IDE."

        elif latest.context == ScreenContext.TERMINAL:
            return "You're working in a terminal."

        elif latest.context == ScreenContext.BROWSER_GITHUB:
            return "You're browsing GitHub."

        elif latest.context == ScreenContext.BROWSER_DOCS:
            return "You're reading documentation."

        elif latest.context == ScreenContext.BROWSER_WEB:
            return "You're browsing the web."

        return f"You're on the {latest.context.name.lower().replace('_', ' ')} screen."

    def _detect_windows(self, image: np.ndarray, text_blocks: list[TextBlock]) -> list[dict]:
        windows = []

        try:
            import pygetwindow as gw
            for win in gw.getAllWindows():
                if win.title and win.visible:
                    windows.append({
                        "title": win.title,
                        "x": win.left,
                        "y": win.top,
                        "width": win.width,
                        "height": win.height,
                        "active": win.isActive,
                    })
        except ImportError:
            pass

        return windows[:10]

    def _generate_summary(
        self,
        context: ScreenContext,
        text_blocks: list[TextBlock],
        ui_elements: list[UIElement],
        dev_context: Optional[dict],
    ) -> str:
        parts = [f"Context: {context.name}"]

        if text_blocks:
            key_texts = [tb.text[:50] for tb in text_blocks[:5] if tb.text.strip()]
            if key_texts:
                parts.append(f"Text visible: {len(text_blocks)} blocks")

        if ui_elements:
            button_count = sum(1 for e in ui_elements if e.element_type == UIElementType.BUTTON)
            if button_count:
                parts.append(f"{button_count} buttons detected")

        if dev_context:
            if dev_context.get("languages"):
                parts.append(f"Languages: {', '.join(dev_context['languages'])}")
            if dev_context.get("errors"):
                parts.append(f"Errors detected: {len(dev_context['errors'])}")

        return ". ".join(parts)

    def get_stats(self) -> dict:
        return {
            "capture": self.capture_engine.get_stats(),
            "automation": self.automation.get_stats(),
            "analyses_performed": len(self._analysis_history),
            "has_screen_deps": HAS_SCREEN_DEPS,
            "has_ocr": HAS_OCR,
            "has_automation": HAS_AUTOMATION,
        }


_screen_vision: Optional[ScreenVision] = None
_lock = threading.Lock()


class ScreenVisionLearner:
    """Learns from screen interactions to improve over time.
    
    Tracks:
    - Successful element clicks (positive reinforcement)
    - Context patterns (what contexts lead to what actions)
    - UI element templates (buttons, inputs with consistent patterns)
    - Text patterns (common UI text and their locations)
    - Application-specific layouts
    """

    MAX_ELEMENT_TEMPLATES = 500
    MAX_TEXT_PATTERNS = 2000
    MAX_APP_LAYOUTS = 100
    MAX_CONFIDENCE_BOOST = 1000

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._element_templates: dict[str, dict] = {}
        self._click_successes: list[dict] = []
        self._click_failures: list[dict] = []
        self._context_patterns: dict[str, dict] = {}
        self._text_patterns: dict[str, int] = {}
        self._app_layouts: dict[str, dict] = {}
        self._confidence_boost: dict[str, float] = {}
        self._lock = threading.Lock()
        self._load_state()

    def _load_state(self):
        if not self.storage_path:
            return

        try:
            from pathlib import Path
            from core.helpers import read_json_with_integrity

            state_file = Path(self.storage_path) / "screen_vision_learned.json"
            if state_file.exists():
                data = read_json_with_integrity(state_file)
                self._element_templates = data.get("element_templates", {})
                self._text_patterns = data.get("text_patterns", {})
                self._app_layouts = data.get("app_layouts", {})
                self._confidence_boost = data.get("confidence_boost", {})
                logger.info(f"Loaded {len(self._text_patterns)} learned text patterns")
        except Exception as e:
            logger.debug(f"Could not load learned state: {e}")

    def save_state(self):
        if not self.storage_path:
            return

        try:
            from pathlib import Path
            from core.helpers import atomic_write_json

            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            state_file = Path(self.storage_path) / "screen_vision_learned.json"

            atomic_write_json(state_file, {
                    "element_templates": self._element_templates,
                    "text_patterns": dict(list(self._text_patterns.items())[-1000:]),
                    "app_layouts": self._app_layouts,
                    "confidence_boost": self._confidence_boost,
                })
        except Exception as e:
            logger.debug(f"Could not save learned state: {e}")

    def record_click(self, element: UIElement, success: bool, context: ScreenContext):
        with self._lock:
            record = {
                "element_type": element.element_type.name,
                "text": element.text,
                "bbox": element.bbox.to_dict() if element.bbox else None,
                "context": context.name,
                "success": success,
                "timestamp": time.time(),
            }

            if success:
                self._click_successes.append(record)
                if len(self._click_successes) > 500:
                    self._click_successes = self._click_successes[-500:]

                if element.text:
                    key = f"{element.element_type.name}:{element.text[:30]}"
                    self._confidence_boost[key] = min(1.0, self._confidence_boost.get(key, 0.5) + 0.1)
                    if len(self._confidence_boost) > self.MAX_CONFIDENCE_BOOST:
                        sorted_keys = sorted(self._confidence_boost, key=self._confidence_boost.get)[:len(self._confidence_boost) - self.MAX_CONFIDENCE_BOOST]
                        for k in sorted_keys:
                            del self._confidence_boost[k]

                self._update_template(element, context)
            else:
                self._click_failures.append(record)
                if len(self._click_failures) > 200:
                    self._click_failures = self._click_failures[-200:]

    def _update_template(self, element: UIElement, context: ScreenContext):
        key = f"{context.name}:{element.element_type.name}"

        if key not in self._element_templates:
            if len(self._element_templates) >= self.MAX_ELEMENT_TEMPLATES:
                least_used = sorted(self._element_templates.items(), key=lambda x: x[1].get("count", 0))[:1]
                if least_used:
                    del self._element_templates[least_used[0][0]]
            self._element_templates[key] = {
                "count": 0,
                "avg_width": 0,
                "avg_height": 0,
                "texts": {},
            }

        template = self._element_templates[key]
        template["count"] += 1

        if element.bbox:
            n = template["count"]
            template["avg_width"] = (template["avg_width"] * (n - 1) + element.bbox.width) / n
            template["avg_height"] = (template["avg_height"] * (n - 1) + element.bbox.height) / n

        if element.text:
            text_key = element.text[:50].lower()
            template["texts"][text_key] = template["texts"].get(text_key, 0) + 1

    def record_text_pattern(self, text: str, element_type: UIElementType):
        with self._lock:
            key = f"{element_type.name}:{text[:30].lower()}"
            self._text_patterns[key] = self._text_patterns.get(key, 0) + 1
            if len(self._text_patterns) > self.MAX_TEXT_PATTERNS:
                least_used = sorted(self._text_patterns.items(), key=lambda x: x[1])[:len(self._text_patterns) - self.MAX_TEXT_PATTERNS]
                for k, _ in least_used:
                    del self._text_patterns[k]

    def record_app_layout(self, app_name: str, elements: list[UIElement]):
        with self._lock:
            if app_name not in self._app_layouts:
                if len(self._app_layouts) >= self.MAX_APP_LAYOUTS:
                    least_seen = sorted(self._app_layouts.items(), key=lambda x: x[1].get("seen_count", 0))[:1]
                    if least_seen:
                        del self._app_layouts[least_seen[0][0]]
                self._app_layouts[app_name] = {
                    "seen_count": 0,
                    "element_types": {},
                    "common_texts": {},
                }

            layout = self._app_layouts[app_name]
            layout["seen_count"] += 1

            for elem in elements:
                et = elem.element_type.name
                layout["element_types"][et] = layout["element_types"].get(et, 0) + 1

                if elem.text:
                    text_key = elem.text[:30].lower()
                    layout["common_texts"][text_key] = layout["common_texts"].get(text_key, 0) + 1

    def boost_confidence(self, element: UIElement, context: ScreenContext) -> float:
        boost = 0.0

        if element.text:
            key = f"{element.element_type.name}:{element.text[:30]}"
            boost += self._confidence_boost.get(key, 0.0) * 0.2

        template_key = f"{context.name}:{element.element_type.name}"
        template = self._element_templates.get(template_key, {})

        if template.get("count", 0) > 5:
            if element.bbox:
                expected_w = template.get("avg_width", element.bbox.width)
                expected_h = template.get("avg_height", element.bbox.height)

                w_diff = abs(element.bbox.width - expected_w) / max(expected_w, 1)
                h_diff = abs(element.bbox.height - expected_h) / max(expected_h, 1)

                if w_diff < 0.3 and h_diff < 0.3:
                    boost += 0.1

        if element.text:
            text_key = element.text[:50].lower()
            pattern_key = f"{element.element_type.name}:{text_key}"
            if self._text_patterns.get(pattern_key, 0) > 3:
                boost += 0.15

        return min(1.0, element.confidence + boost)

    def suggest_elements(self, context: ScreenContext, text_hint: Optional[str] = None) -> list[dict]:
        suggestions = []

        template_key = f"{context.name}:"
        for key, template in self._element_templates.items():
            if key.startswith(template_key):
                element_type = key.split(":")[1]

                if text_hint:
                    matching_texts = [
                        text for text, count in template.get("texts", {}).items()
                        if text_hint.lower() in text and count > 2
                    ]
                    if matching_texts:
                        suggestions.append({
                            "element_type": element_type,
                            "expected_width": template.get("avg_width"),
                            "expected_height": template.get("avg_height"),
                            "matching_texts": matching_texts[:3],
                            "confidence": 0.7 + len(matching_texts) * 0.05,
                        })

        return sorted(suggestions, key=lambda x: x["confidence"], reverse=True)[:5]

    def get_learning_stats(self) -> dict:
        with self._lock:
            success_rate = 0.0
            total_clicks = len(self._click_successes) + len(self._click_failures)
            if total_clicks > 0:
                success_rate = len(self._click_successes) / total_clicks

            return {
                "element_templates": len(self._element_templates),
                "click_successes": len(self._click_successes),
                "click_failures": len(self._click_failures),
                "click_success_rate": success_rate,
                "text_patterns": len(self._text_patterns),
                "app_layouts": len(self._app_layouts),
                "confidence_boosts": len(self._confidence_boost),
            }


def get_screen_vision(enable_automation: bool = False) -> ScreenVision:
    global _screen_vision
    if _screen_vision is None:
        with _lock:
            if _screen_vision is None:
                _screen_vision = ScreenVision(enable_automation=enable_automation)
    return _screen_vision


__all__ = [
    "ScreenContext",
    "UIElementType",
    "BoundingBox",
    "UIElement",
    "TextBlock",
    "ScreenCapture",
    "ScreenAnalysis",
    "ScreenVision",
    "AutomationController",
    "get_screen_vision",
]
