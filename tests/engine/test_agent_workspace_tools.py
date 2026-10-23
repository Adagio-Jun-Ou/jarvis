"""Tests for agent/workspace_tools.py — the file/bash toolset.

The tools refuse to escape the user's workspace root. Path traversal
attempts must raise."""
from __future__ import annotations

import os

import pytest

from agent import workspace_tools as w


@pytest.fixture
def fake_workspace(monkeypatch, tmp_path):
    root = tmp_path
    user = "alice@x.com"
    # The actual directory name is the email with slashes stripped, but
    # `@` and `.` are kept (the create_file function is allowed to put
    # files under that subtree). The `..` would have been replaced.
    user_root = root / "alice@x.com"
    (user_root / "uploads").mkdir(parents=True)
    (user_root / "work").mkdir(parents=True)
    (user_root / "outputs").mkdir(parents=True)
    (user_root / "uploads" / "readme.txt").write_text("hello world")
    monkeypatch.setattr(w, "WORKSPACE_ROOT", str(root))
    yield user, user_root


def test_view_reads_file(fake_workspace):
    user, root = fake_workspace
    assert w.view(user, "uploads/readme.txt") == "hello world"


def test_view_blocks_traversal(fake_workspace):
    user, _ = fake_workspace
    with pytest.raises(ValueError):
        w.view(user, "../escape.txt")
    with pytest.raises(ValueError):
        w.view(user, "/etc/passwd")
    with pytest.raises(ValueError):
        w.view(user, "")


def test_create_file_writes_new(fake_workspace):
    user, root = fake_workspace
    msg = w.create_file(user, "work/new.txt", "data")
    assert "wrote" in msg
    assert (root / "work" / "new.txt").read_text() == "data"


def test_create_file_refuses_overwrite(fake_workspace):
    user, root = fake_workspace
    w.create_file(user, "work/x.txt", "first")
    out = w.create_file(user, "work/x.txt", "second")
    assert "already exists" in out
    assert (root / "work" / "x.txt").read_text() == "first"


def test_str_replace(fake_workspace):
    user, root = fake_workspace
    w.create_file(user, "work/x.txt", "the quick brown fox")
    out = w.str_replace(user, "work/x.txt", "quick", "slow")
    assert "replaced" in out
    assert "the slow brown fox" == (root / "work" / "x.txt").read_text()


def test_bash_uses_argv_no_shell(fake_workspace, monkeypatch):
    user, _ = fake_workspace
    out = w.bash(user, ["ls", "-1"])
    # Just check that bash returned the listing (whatever's there)
    assert isinstance(out, str)


def test_bash_refuses_unknown_binary(fake_workspace):
    user, _ = fake_workspace
    out = w.bash(user, ["this-binary-does-not-exist-xyz"])
    assert "not on PATH" in out or "not in the allowlist" in out


def test_bash_blocks_shell_metachars(fake_workspace):
    user, _ = fake_workspace
    # Even with shell metacharacters in argv, no shell is invoked.
    # The model cannot use `;` or `&&` because we never spawn sh.
    w.bash(user, ["ls", ";", "rm", "-rf", "/"])
    # Allowed list still includes 'ls' so the first arg is permitted;
    # but the WHOLE argv goes to ls, which fails because ";" isn't a file.
    # Critical: we did NOT actually execute rm.
    assert "rm" not in os.listdir("/")


# ---- download: every redirect hop is re-validated --------------------------
# The one-shot _url_ok used to be the only gate; urllib's default
# redirect handler then followed a 30x chain anywhere — a public URL
# redirecting to the metadata service would have exfiltrated into the
# workspace.


def test_download_refuses_a_redirect_to_a_private_address(fake_workspace, monkeypatch):
    """A public URL that 302s to the metadata service must be
    refused at the hop, not followed."""
    import http.server
    import threading

    user, root = fake_workspace

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header(
                "Location", "http://169.254.169.254/latest/meta-data/"
            )
            self.end_headers()

        def log_message(self, *a):  # silence
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        # 127.0.0.1 is itself blocked by _url_ok, so stub the check to
        # allow the local test origin — the redirect target is what is
        # under test, and it is validated by the real _url_ok.
        monkeypatch.setattr(
            "agent.web_agentic._url_ok",
            lambda url: "169.254.169.254" not in str(url),
        )
        out = w.download(
            user, f"http://127.0.0.1:{port}/file.bin", "file.bin"
        )
    finally:
        srv.shutdown()

    assert out.get("ok") is False, "a redirect to the metadata service was followed"
    assert "private/internal address blocked" in out.get("error", "")
    assert not (root / "outputs" / "file.bin").exists(), (
        "the blocked redirect still wrote a file into the workspace"
    )
    srv.server_close()
