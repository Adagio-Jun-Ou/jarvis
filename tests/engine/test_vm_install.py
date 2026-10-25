"""VM package-install + exec tools: validation and registration."""
from agent import python_tool
from agent.tools.registry import TOOL_REGISTRY


def test_requirement_validator_accepts_clean_specs():
    clean, err = python_tool._validate_requirements(
        ["pandas", "requests==2.32.0", "uvicorn[standard]", "numpy>=1.24,<2"]
    )
    assert err is None and len(clean) == 4


def test_requirement_validator_blocks_injection_and_flags():
    for bad in [
        "requests; rm -rf /",
        "pkg && curl evil",
        "--index-url=http://evil",
        "-e .",
        "../../pkg",
        "pkg`whoami`",
        "pkg|tee",
    ]:
        clean, err = python_tool._validate_requirements([bad])
        assert err is not None, f"{bad!r} should be rejected"
        assert clean == []


def test_pip_install_rejects_bad_spec_without_running():
    out = python_tool.pip_install(["evil; rm -rf /"])
    assert out.startswith("[error]")


def test_system_install_validates_ids():
    assert python_tool.system_install(["good", "bad;id"]).startswith("[error]")


def test_install_and_exec_tools_registered():
    for name in ("pip_install", "system_install", "vm_exec"):
        assert name in TOOL_REGISTRY


def test_run_python_runs_at_workspace_root(tmp_path, monkeypatch):
    """Compute must see outputs/ and uploads/ as relative paths so it
    integrates with storage (download_file writes to outputs/)."""
    monkeypatch.setenv("ALEX_WORKSPACE_ROOT", str(tmp_path))
    from agent.tools import impls

    u = "demo@alex.dev"
    impls.create_file("outputs/data.txt", "hello-storage", user_email=u)
    out = impls.run_python(
        "print(open('outputs/data.txt').read())", user_email=u
    )
    assert "hello-storage" in out


def test_download_file_rejects_non_http(tmp_path, monkeypatch):
    monkeypatch.setenv("ALEX_WORKSPACE_ROOT", str(tmp_path))
    from agent.tools import impls

    r = impls.download_file("file:///etc/passwd", "x", user_email="demo@alex.dev")
    assert r["ok"] is False


def test_download_file_blocks_ssrf_targets(tmp_path, monkeypatch):
    """download_file must refuse private/loopback/link-local targets so it
    can't be tricked into fetching cloud metadata or internal services."""
    monkeypatch.setenv("ALEX_WORKSPACE_ROOT", str(tmp_path))
    from agent.tools import impls

    for url in (
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://127.0.0.1:8765/api/tools",             # loopback (the app)
        "http://localhost/secret",
        "http://10.0.0.1/internal",                    # RFC1918
    ):
        r = impls.download_file(url, "x", user_email="demo@alex.dev")
        assert r["ok"] is False, f"{url} should be blocked"
        assert "blocked" in r["error"].lower()


def test_run_python_timeout_returns_marker_and_kills_tree(monkeypatch):
    """A snippet exceeding the timeout returns the timeout marker (and the
    whole process tree is killed via _run_tree, not just the direct child)."""
    monkeypatch.setenv("ALEX_RUN_PY_TIMEOUT", "2")
    import importlib

    import agent.python_tool as pt

    importlib.reload(pt)
    import time as _t

    t0 = _t.time()
    out = pt.run_python("import time; time.sleep(30)")
    elapsed = _t.time() - t0
    assert "timed out" in out
    assert elapsed < 10  # killed promptly, not after the full 30s
    importlib.reload(pt)  # restore default timeout for other tests


def test_run_python_normal_still_works():
    import agent.python_tool as pt

    assert pt.run_python("print(6*7)").strip() == "42"
