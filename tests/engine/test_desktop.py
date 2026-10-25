"""Desktop control — Jarvis on the machine it lives on.

* reads land in the user's outputs/ and carry OCR text with positions;
* acts go through the backend seam (a test never moves a real mouse);
* the family is governed: reads power 0, acts power 2 and policy-ASK;
* a missing dependency is a clear error, not a crash.
"""

from __future__ import annotations

import pytest

from agent import desktop


class FakeOS:
    def __init__(self):
        self.calls = []

    def screenshot(self, path, monitor=1):
        open(path, "wb").write(b"\x89PNG")
        return {"width": 1920, "height": 1080}

    def ocr(self, path):
        return [{"text": "Save", "x": 100, "y": 20, "w": 40, "h": 16, "conf": 92.0},
                {"text": "Cancel", "x": 200, "y": 20, "w": 60, "h": 16, "conf": 90.0}]

    def click(self, x, y, button="left", clicks=1):
        self.calls.append(("click", x, y, button, clicks))

    def type_text(self, text, interval=0.02):
        self.calls.append(("type", text))

    def press(self, key, modifiers=None):
        self.calls.append(("press", key, list(modifiers or [])))

    def hotkey(self, keys):
        self.calls.append(("hotkey", list(keys)))

    def scroll(self, clicks):
        self.calls.append(("scroll", clicks))


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.setenv("ALEX_WORKSPACE_ROOT", str(tmp_path))
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    fake = FakeOS()
    desktop.set_backend(fake)
    yield fake
    desktop.set_backend(None)
    config._settings = None
    testing._fakes = None


def test_screenshot_lands_in_outputs_with_ocr_text(env):
    out = desktop.screenshot("dana@alex.dev")
    assert out["ok"] and out["path"].startswith("outputs/screen-")
    assert "Save" in out["text"] and out["words"] == 2
    assert out["width"] == 1920


def test_find_text_returns_a_clickable_center(env):
    hit = desktop.find_text("dana@alex.dev", "cancel")
    assert hit["found"] and (hit["x"], hit["y"]) == (230, 28)
    assert desktop.find_text("dana@alex.dev", "nope")["found"] is False


def test_acts_go_through_the_backend(env):
    fake = env
    assert desktop.click(10, 20)["ok"]
    assert desktop.type_text("hello")["ok"]
    assert desktop.press_key("s", ["ctrl"])["ok"]
    assert desktop.hotkey(["ctrl", "shift", "p"])["ok"]
    assert desktop.scroll("down", 2)["ok"]
    assert fake.calls == [
        ("click", 10, 20, "left", 1), ("type", "hello"),
        ("press", "s", ["ctrl"]), ("hotkey", ["ctrl", "shift", "p"]),
        ("scroll", -2),
    ]
    assert desktop.type_text("x" * 2001)["ok"] is False
    assert desktop.hotkey([])["ok"] is False


def test_a_missing_dependency_is_a_clear_error(env, monkeypatch):
    class Broken:
        def screenshot(self, path, monitor=1):
            raise RuntimeError("install the desktop deps: pip install mss pynput")

    desktop.set_backend(Broken())
    out = desktop.screenshot("dana@alex.dev")
    assert out["ok"] is False and "mss" in out["error"]


def test_the_family_is_governed(env):
    from agent.execution_governor import CAPABILITY_POWER
    from agent.guardrails import TOOL_CAPABILITY
    from agent.tool_policy import DEFAULT_ASK_CAPABILITIES, decision
    from agent.tools.registry import TOOL_REGISTRY

    for name in ("desktop_screenshot", "desktop_find_text"):
        assert TOOL_REGISTRY[name].capability == "desktop.read"
        assert TOOL_REGISTRY[name].untrusted_output, "screen text is outside-authored"
    for name in ("desktop_click", "desktop_type", "desktop_key", "desktop_hotkey", "desktop_scroll"):
        assert TOOL_CAPABILITY[name] == "desktop.act"
    assert CAPABILITY_POWER["desktop.read"] == 0 and CAPABILITY_POWER["desktop.act"] == 2
    assert "desktop.act" in DEFAULT_ASK_CAPABILITIES
    assert decision("desktop_click", "desktop.act") == "ask"
    assert decision("desktop_screenshot", "desktop.read") == "allow"


def test_the_tools_run_through_the_vm_with_identity_injected(env):
    from agent.omnigent import MockExecutor
    from agent.omnigent.executor import ToolCallRequest
    from agent.vm import make_vm

    vm = make_vm(session_id="chat-desk", user_email="dana@alex.dev",
                 executor=MockExecutor())
    res = vm._invoke_tool(ToolCallRequest(name="desktop_screenshot", args={}))
    assert res["ok"] and "Save" in res["text"]
    res = vm._invoke_tool(ToolCallRequest(name="desktop_click", args={"x": 1, "y": 2}))
    assert res["ok"], "supervised chat: the act proceeds (and is announced)"
    vm.close()
