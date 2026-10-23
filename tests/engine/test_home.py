"""Home automation: real calls or an honest refusal, never a placeholder.

* unconfigured -> a clear error saying how to configure;
* Home Assistant calls carry the token and map actions to services;
* the command map runs argv (no shell) and reports the exit code;
* the family is governed (read 0 / act 2, act policy-ASK).
"""

from __future__ import annotations

import json

import pytest

from agent import home


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    for k in ("HOME_ASSISTANT_URL", "HOME_ASSISTANT_TOKEN", "ALEX_HOME_COMMANDS"):
        monkeypatch.delenv(k, raising=False)
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


def test_unconfigured_is_an_honest_refusal(env):
    out = home.control("light.desk", "on")
    assert out["ok"] is False and "HOME_ASSISTANT_URL" in out["error"]
    assert home.status()["ok"] is False
    assert home.configured() == {"home_assistant": False, "commands": []}


def test_home_assistant_calls_map_actions_to_services(env, monkeypatch):
    monkeypatch.setenv("HOME_ASSISTANT_URL", "http://ha.local:8123/")
    monkeypatch.setenv("HOME_ASSISTANT_TOKEN", "tok")
    seen = []

    def fake(method, path, body=None):
        seen.append((method, path, body))
        if path.startswith("states/"):
            return {"state": "on", "attributes": {"friendly_name": "Desk lamp"}}
        if path == "states":
            return [{"entity_id": "light.desk", "state": "on", "attributes": {}}]
        return {}

    monkeypatch.setattr(home, "_ha_request", fake)
    assert home.control("light.desk", "on")["ok"]
    assert home.control("light.desk", "brightness", "40")["ok"]
    assert home.status("light.desk")["name"] == "Desk lamp"
    assert home.status()["count"] == 1
    assert seen[0] == ("POST", "services/light/turn_on", {"entity_id": "light.desk"})
    assert seen[1][2]["brightness_pct"] == 40


def test_the_command_map_runs_argv_without_a_shell(env, monkeypatch, tmp_path):
    cmds = tmp_path / "home.json"
    cmds.write_text(json.dumps({
        "fan": {"on": ["python", "-c", "print('fan on')"],
                "off": ["python", "-c", "import sys; sys.exit(3)"]},
    }), encoding="utf-8")
    monkeypatch.setenv("ALEX_HOME_COMMANDS", str(cmds))
    out = home.control("fan", "on")
    assert out["ok"] and "fan on" in out["output"] and out["platform"] == "command"
    assert home.control("fan", "off")["exit"] == 3
    assert home.control("fan", "dance")["ok"] is False
    assert home.configured()["commands"] == ["fan"]


def test_the_family_is_governed(env):
    from agent.execution_governor import CAPABILITY_POWER
    from agent.tool_policy import decision
    from agent.tools.registry import TOOL_REGISTRY

    assert TOOL_REGISTRY["home_status"].capability == "home.read"
    assert TOOL_REGISTRY["home_control"].capability == "home.act"
    assert CAPABILITY_POWER["home.act"] == 2
    assert decision("home_control", "home.act") == "ask"
