"""The environment agent-run code is allowed to see.

Both execution paths used to hand the child `{**os.environ, ...}`. Every
other control in the sandbox — the workspace jail, the runtime cap, the
binary allowlist, the constitution's rules about what the agent may
touch — assumes that code which misbehaves cannot simply take the
process's credentials and act somewhere the sandbox does not reach. That
assumption was false: `print(os.environ)` was enough.

These tests are written against secrets that look like the real ones
this system holds, so a regression fails loudly rather than subtly.
"""

from __future__ import annotations

import os

import pytest

from agent import python_tool

# Names shaped like the credentials this process genuinely carries.
SECRETS = {
    "DATABRICKS_TOKEN": "dapi-real-token",
    "DATABRICKS_CLIENT_SECRET": "sp-secret",
    "ALEX_ADMIN_EMAILS": "ceo@corp.example",
    "OPENAI_API_KEY": "sk-live-key",
    "ANTHROPIC_API_KEY": "sk-ant-live",
    "AWS_SECRET_ACCESS_KEY": "aws-secret",
    "PGPASSWORD": "warehouse-password",
    "SESSION_SIGNING_KEY": "cookie-signing-key",
}


@pytest.fixture
def secrets(monkeypatch):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    return SECRETS


def test_child_env_withholds_every_secret(secrets):
    env = python_tool.child_env()
    for name in secrets:
        assert name not in env, f"{name} was handed to agent-run code"


def test_child_env_leaks_no_secret_value_under_any_name(secrets):
    """Not just the names — a renamed copy would leak just as well."""
    values = set(secrets.values())
    assert not (values & set(python_tool.child_env().values()))


def test_child_env_keeps_what_a_program_needs_to_run():
    env = python_tool.child_env()
    assert "PATH" in env, "a child with no PATH cannot find its interpreter"
    if os.name == "nt":
        assert "SYSTEMROOT" in env, "Windows programs need SYSTEMROOT"


def test_child_env_applies_caller_supplied_values(secrets):
    env = python_tool.child_env({"HOME": "/work/dana", "PYTHONPATH": "/work"})
    assert env["HOME"] == "/work/dana"
    assert env["PYTHONPATH"] == "/work"
    assert "DATABRICKS_TOKEN" not in env


def test_child_env_is_an_allowlist_not_a_denylist(monkeypatch):
    """A secret nobody thought to name must still be withheld — that is
    the whole reason this is an allowlist."""
    monkeypatch.setenv("SOME_FUTURE_VENDOR_CREDENTIAL", "unguessable")
    assert "SOME_FUTURE_VENDOR_CREDENTIAL" not in python_tool.child_env()


def test_run_python_cannot_read_the_process_secrets(secrets, tmp_path):
    """End to end through the real subprocess, not the helper alone."""
    out = python_tool.run_python(
        "import os\n"
        "print('|'.join(f'{k}={v}' for k, v in sorted(os.environ.items())))",
        cwd=str(tmp_path),
    )
    for name, value in secrets.items():
        assert name not in out, f"run_python exposed {name}"
        assert value not in out, f"run_python exposed the value of {name}"


def test_run_python_still_works(tmp_path):
    """A sandbox that breaks ordinary code would just get switched off."""
    out = python_tool.run_python(
        "import json, sys, os\n"
        "print(json.dumps({'n': sum(range(10)), 'py': sys.version_info[0]}))",
        cwd=str(tmp_path),
    )
    assert '"n": 45' in out
    assert '"py": 3' in out


def test_workspace_bash_cannot_read_the_process_secrets(secrets, monkeypatch, tmp_path):
    """The shell path allowlists binaries, but an allowlisted binary is
    still agent-directed and used to inherit everything."""
    from agent import workspace_tools

    monkeypatch.setattr(workspace_tools, "_user_root", lambda _e: str(tmp_path))
    captured = {}

    def fake_run(argv, **kwargs):
        captured["env"] = kwargs.get("env") or {}

        class R:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return R()

    monkeypatch.setattr(workspace_tools.subprocess, "run", fake_run)
    workspace_tools.bash("dana@alex.dev", ["python", "--version"])

    env = captured["env"]
    assert env, "no environment was captured"
    for name, value in secrets.items():
        assert name not in env
        assert value not in env.values()
    assert env.get("HOME") == str(tmp_path), "the workspace HOME was lost"


def test_pip_env_keeps_secrets_out_but_index_config_in(secrets, monkeypatch):
    """Installing runs the package's build code, so pip is an execution
    path too — but an install that cannot reach the configured index is
    useless, so its own network settings pass through."""
    monkeypatch.setenv("PIP_INDEX_URL", "https://pypi.internal.corp/simple")
    monkeypatch.setenv("HTTPS_PROXY", "http://egress.corp:3128")

    env = python_tool.pip_env()

    assert env["PIP_INDEX_URL"] == "https://pypi.internal.corp/simple"
    assert env["HTTPS_PROXY"] == "http://egress.corp:3128"
    for name, value in secrets.items():
        assert name not in env
        assert value not in env.values()


def test_pip_install_does_not_inherit_the_process_environment(secrets, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["env"] = kwargs.get("env")

        class R:
            returncode = 0
            stdout = "Successfully installed rich"
            stderr = ""

        return R()

    monkeypatch.setattr(python_tool.subprocess, "run", fake_run)
    python_tool.pip_install(["rich"])

    env = captured["env"]
    assert env is not None, "pip_install passed no env, so it inherited everything"
    for name in secrets:
        assert name not in env


# ---- resource bounds -----------------------------------------------------
# A timeout alone does not bound what agent-run code can do to the host.
# A process that allocates without limit takes the machine into swap long
# before its sixty seconds are up, and it takes down every other user's
# workspace with it, not just its own run.


def test_memory_hog_is_killed_and_the_run_survives(tmp_path, monkeypatch):
    monkeypatch.setattr(python_tool, "MAX_MEMORY_MB", 192)
    monkeypatch.setattr(python_tool, "MAX_RUNTIME_SECONDS", 30)

    out = python_tool.run_python(
        # Allocate in visible steps so the watchdog has something to see,
        # and hold references so nothing is collected away.
        "blocks = []\n"
        "for i in range(200):\n"
        "    blocks.append(bytearray(16 * 1024 * 1024))\n"
        "print('ALLOCATED ALL', len(blocks))\n",
        cwd=str(tmp_path),
    )

    assert "ALLOCATED ALL" not in out, "the process allocated 3GB unimpeded"


def test_a_modest_allocation_is_not_killed(tmp_path, monkeypatch):
    """A limit that kills ordinary work would just get raised until it
    stopped protecting anything."""
    monkeypatch.setattr(python_tool, "MAX_MEMORY_MB", 512)

    out = python_tool.run_python(
        "buf = bytearray(8 * 1024 * 1024)\nprint('OK', len(buf))",
        cwd=str(tmp_path),
    )

    assert "OK 8388608" in out


def test_the_timeout_still_applies(tmp_path, monkeypatch):
    monkeypatch.setattr(python_tool, "MAX_RUNTIME_SECONDS", 2)

    out = python_tool.run_python("import time\ntime.sleep(30)", cwd=str(tmp_path))

    assert "timed out" in out


def test_the_watchdog_does_not_outlive_its_process(tmp_path):
    """One polling thread per run that never retires is a slow leak."""
    import threading

    before = threading.active_count()
    for _ in range(3):
        python_tool.run_python("print('quick')", cwd=str(tmp_path))

    assert threading.active_count() <= before, "watchdog threads leaked"


def test_limits_are_configurable_from_the_environment(monkeypatch):
    """Production sets these from the cluster policy, so they must not be
    baked in."""
    monkeypatch.setenv("ALEX_RUN_MEM_MB", "4096")
    monkeypatch.setenv("ALEX_RUN_MAX_PROCS", "8")
    import importlib

    reloaded = importlib.reload(python_tool)
    try:
        assert reloaded.MAX_MEMORY_MB == 4096
        assert reloaded.MAX_PROCESSES == 8
    finally:
        monkeypatch.undo()
        importlib.reload(python_tool)
