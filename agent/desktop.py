"""Desktop control — the machine Jarvis lives on.

Ported from Jarvis's screen tools as a lean primitive layer: capture the
screen (mss), read it (Tesseract OCR when installed), find text on it,
and act (pynput mouse/keyboard). Every dependency is optional and lazy;
a missing one is a clear tool error naming the package, never a crash.

Two capability families, because they are two blast radii:

* ``desktop.read`` — screenshot / OCR / find. Power 0.
* ``desktop.act``  — click / type / key / hotkey / scroll. Power 2, and
  policy-ASK by default: a supervised chat proceeds and says so, an
  unattended board run parks on approval. Acting on the person's own
  desktop is exactly the class of thing a human should be able to gate.

``set_backend`` swaps the OS layer for a fake — tests never move a real
mouse.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

log = logging.getLogger("agent.desktop")

_backend: Any = None


class _RealBackend:
    """mss + pynput + pytesseract, all imported lazily."""

    def screenshot(self, path: str, monitor: int = 1) -> dict:
        try:
            import mss
            import mss.tools
        except ImportError as exc:
            raise RuntimeError("install the desktop deps: pip install mss pynput") from exc
        with mss.mss() as sct:
            mon = sct.monitors[monitor] if monitor < len(sct.monitors) else sct.monitors[1]
            shot = sct.grab(mon)
            mss.tools.to_png(shot.rgb, shot.size, output=path)
            return {"width": shot.width, "height": shot.height}

    def ocr(self, path: str) -> list[dict]:
        """Words with boxes: [{text, x, y, w, h, conf}] — empty when OCR
        is unavailable (the screenshot is still useful to a vision model)."""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            return []
        try:
            data = pytesseract.image_to_data(
                Image.open(path), output_type=pytesseract.Output.DICT
            )
        except Exception as exc:  # noqa: BLE001 — tesseract binary missing etc.
            log.debug("ocr unavailable: %s", exc)
            return []
        out = []
        for i, text in enumerate(data.get("text", [])):
            t = (text or "").strip()
            if not t:
                continue
            try:
                conf = float(data["conf"][i])
            except (ValueError, TypeError, KeyError):
                conf = -1.0
            out.append({
                "text": t, "x": int(data["left"][i]), "y": int(data["top"][i]),
                "w": int(data["width"][i]), "h": int(data["height"][i]), "conf": conf,
            })
        return out

    def _mouse(self):
        from pynput.mouse import Button, Controller

        return Controller(), Button

    def _keys(self):
        from pynput.keyboard import Controller, Key

        return Controller(), Key

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        mouse, button_cls = self._mouse()
        mouse.position = (x, y)
        time.sleep(0.05)
        mouse.click(getattr(button_cls, button, button_cls.left), clicks)

    def type_text(self, text: str, interval: float = 0.02) -> None:
        kb, _ = self._keys()
        for ch in text:
            kb.type(ch)
            if interval:
                time.sleep(interval)

    def press(self, key: str, modifiers: list[str] | None = None) -> None:
        kb, key_cls = self._keys()
        mods = [getattr(key_cls, m.lower()) for m in (modifiers or []) if hasattr(key_cls, m.lower())]
        k = getattr(key_cls, key.lower(), key)
        for m in mods:
            kb.press(m)
        kb.press(k)
        kb.release(k)
        for m in reversed(mods):
            kb.release(m)

    def hotkey(self, keys: list[str]) -> None:
        self.press(keys[-1], keys[:-1])

    def scroll(self, clicks: int) -> None:
        mouse, _ = self._mouse()
        mouse.scroll(0, clicks)


def set_backend(backend: Any) -> None:
    global _backend
    _backend = backend


def _be() -> Any:
    global _backend
    if _backend is None:
        _backend = _RealBackend()
    return _backend


# ---- the tools ---------------------------------------------------------------

_MAX_TEXT = 4000


def screenshot(user_email: str, monitor: int = 1, *, ocr: bool = True) -> dict:
    """Capture the screen into the user's outputs/ and, when OCR is
    available, return the text seen with positions."""
    from agent.workspace_tools import output_path

    name = f"screen-{time.strftime('%Y%m%d-%H%M%S')}.png"
    path = output_path(user_email, name)
    try:
        size = _be().screenshot(path, monitor)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    words = _be().ocr(path) if ocr else []
    text = " ".join(w["text"] for w in words)[:_MAX_TEXT]
    return {
        "ok": True, "path": f"outputs/{name}", **size,
        "text": text, "words": len(words),
        "note": "" if words else "no OCR text (install tesseract + pytesseract to read the screen)",
    }


def find_text(user_email: str, text: str, monitor: int = 1) -> dict:
    """Locate text on screen; returns the center to click."""
    from agent.workspace_tools import output_path

    path = output_path(user_email, "screen-find.png")
    try:
        _be().screenshot(path, monitor)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    needle = (text or "").strip().lower()
    for w in _be().ocr(path):
        if needle and needle in w["text"].lower():
            return {"ok": True, "found": True, "text": w["text"],
                    "x": w["x"] + w["w"] // 2, "y": w["y"] + w["h"] // 2,
                    "confidence": w["conf"]}
    return {"ok": True, "found": False, "text": text}


def click(x: int, y: int, button: str = "left", clicks: int = 1) -> dict:
    try:
        _be().click(int(x), int(y), button, max(1, min(int(clicks), 3)))
        return {"ok": True, "clicked": [int(x), int(y)], "button": button}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def type_text(text: str, interval: float = 0.02) -> dict:
    text = str(text or "")
    if len(text) > 2000:
        return {"ok": False, "error": "type at most 2000 characters per call"}
    try:
        _be().type_text(text, float(interval))
        return {"ok": True, "typed": len(text)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def press_key(key: str, modifiers: list[str] | None = None) -> dict:
    try:
        _be().press(str(key), list(modifiers or []))
        return {"ok": True, "pressed": ([*(modifiers or []), key])}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def hotkey(keys: list[str]) -> dict:
    keys = [str(k) for k in (keys or []) if str(k).strip()]
    if not keys:
        return {"ok": False, "error": "keys is required, e.g. ['ctrl', 's']"}
    try:
        _be().hotkey(keys)
        return {"ok": True, "pressed": keys}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def scroll(direction: str = "down", amount: int = 3) -> dict:
    clicks = -abs(int(amount)) if direction == "down" else abs(int(amount))
    try:
        _be().scroll(clicks)
        return {"ok": True, "direction": direction, "amount": abs(clicks)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def available() -> dict:
    """What the desktop layer can do on this machine."""
    caps = {}
    for mod in ("mss", "pynput", "pytesseract"):
        try:
            __import__(mod)
            caps[mod] = True
        except ImportError:
            caps[mod] = False
    caps["display"] = bool(os.environ.get("DISPLAY")) or os.name == "nt"
    return caps
