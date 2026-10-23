"""
Workspace tools: the agent's "computer". The model uses these to read,
view, and modify the user's workspace directory (uploads/, work/, outputs/).

The cluster that runs these tools is a per-user single-user cluster under
`governance/sandbox_cluster_policy.json`. Egress is denied at the network
layer. The agent's tools are wrapped by `guardrails.guarded(...)` so
every call is dual-ACL-checked and constitutionally intercepted.

The `bash` tool runs commands via `subprocess.run(argv, shell=False)`,
NOT `shell=True`. Every command goes through a strict allowlist of safe
binaries. The model is NEVER in control of the shell interpreter.

Filesystem access uses `os.path.realpath` + containment checks against
the user's workspace root. There is no path traversal.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

log = logging.getLogger("agent.workspace_tools")

WORKSPACE_ROOT = os.environ.get(
    "ALEX_WORKSPACE_ROOT",
    "/Volumes/media/alex/workspaces",
)

# Allowlist of safe binary names. Anything else is refused.
SAFE_BINS = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "awk",
        "sed",
        "jq",
        "yq",
        "curl",
        "git",
        "make",
        "tar",
        "gzip",
        "gunzip",
        "python",
        "python3",
        "find",
        "sort",
        "uniq",
        "tr",
        "cut",
        "diff",
    }
)


def _workspace_root() -> str:
    """Resolve the workspace root at CALL time, not import time, so a
    test fixture or a dev server that sets ``ALEX_WORKSPACE_ROOT`` after
    this module is imported still takes effect."""
    return os.environ.get("ALEX_WORKSPACE_ROOT", WORKSPACE_ROOT)


def _user_root(user_email: str) -> str:
    safe = user_email.replace("/", "").replace("..", "_")
    if not safe:
        raise ValueError("invalid user_email")
    return os.path.join(_workspace_root(), safe)


def _resolve_under(user_root: str, rel: str) -> str:
    """Resolve a path under `user_root` and ensure it doesn't escape."""
    if not rel:
        raise ValueError("empty path")
    candidate = os.path.realpath(os.path.join(user_root, rel))
    real_root = os.path.realpath(user_root)
    if not (candidate == real_root or candidate.startswith(real_root + os.sep)):
        raise ValueError(f"path {rel!r} escapes the workspace")
    return candidate


def view(user_email: str, path: str, *, max_bytes: int = 200_000) -> str:
    """Read a file from the user's workspace. Bounded."""
    p = _resolve_under(_user_root(user_email), path)
    if not os.path.isfile(p):
        return f"[error] not a file: {path}"
    with open(p, "rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        data = data[:max_bytes] + f"\n[truncated to {max_bytes} bytes]".encode()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return f"[binary, {len(data)} bytes]"


def create_file(user_email: str, path: str, content: str) -> str:
    """Create a new file in the user's workspace. Overwrites only if the
    file already exists (i.e., we don't silently clobber a model
    mistake)."""
    p = _resolve_under(_user_root(user_email), path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p):
        return f"[error] file already exists: {path}"
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    _journal_change(user_email, "create", path, snapshot=None)
    return f"wrote {len(content)} bytes to {path}" + _diagnostics(path, content)


# ---------------------------------------------------------------------------
# Diagnostics feedback — the LSP-loop idea at honest scope.
#
# The strong coding harnesses feed compiler/linter diagnostics back into
# the loop right after an edit, so the model fixes its own syntax error
# on the next pass instead of shipping it. A full LSP is not available
# in this sandbox; what IS available, dependency-free, is a real parse
# of the formats the agent actually writes: Python (compile), JSON, and
# YAML when the library is present. A failed parse rides back on the
# tool result itself — the one place the model is guaranteed to look.
# ---------------------------------------------------------------------------


def _diagnostics(path: str, content: str) -> str:
    try:
        lower = path.lower()
        if lower.endswith(".py"):
            try:
                compile(content, path, "exec")
            except SyntaxError as exc:
                return (
                    f"\n[diagnostic] Python syntax error in {path} line "
                    f"{exc.lineno}: {exc.msg}. Fix it before moving on."
                )
        elif lower.endswith(".json"):
            import json

            try:
                json.loads(content)
            except ValueError as exc:
                return (
                    f"\n[diagnostic] invalid JSON in {path}: {exc}. "
                    "Fix it before moving on."
                )
        elif lower.endswith((".yml", ".yaml")):
            try:
                import yaml

                yaml.safe_load(content)
            except ImportError:
                return ""
            except Exception as exc:  # noqa: BLE001
                return (
                    f"\n[diagnostic] invalid YAML in {path}: {exc}. "
                    "Fix it before moving on."
                )
    except Exception:  # noqa: BLE001
        log.debug("diagnostics check failed", exc_info=True)
    return ""


# ---------------------------------------------------------------------------
# Undo — file changes are snapshotted, and the last one can be reverted.
#
# The recovery loop every strong harness ships (OpenCode's /undo): a
# wrong edit is one tool call from gone rather than an archaeology dig.
# Before each mutation the prior content is copied into .snapshots/ and
# an append-only journal records the operation; undo pops the journal.
# Bounded: only the newest SNAPSHOT_KEEP entries retain their snapshot
# files, so a long session cannot grow the folder without limit.
# ---------------------------------------------------------------------------

SNAPSHOT_DIR = ".snapshots"
SNAPSHOT_KEEP = int(os.environ.get("ALEX_SNAPSHOT_KEEP", "20"))


def _journal_change(
    user_email: str, op: str, rel_path: str, *, snapshot: str | None
) -> None:
    """Record one mutation. Best-effort: a journal failure must never
    fail the write it describes — but it is logged, because an undo
    that silently stopped covering changes is worse than none."""
    import json
    import time as _time

    try:
        root = _user_root(user_email)
        snap_dir = os.path.join(root, SNAPSHOT_DIR)
        os.makedirs(snap_dir, exist_ok=True)
        entry: dict = {"ts": _time.time(), "op": op, "path": rel_path}
        if snapshot is not None:
            name = f"{int(_time.time() * 1000)}__{rel_path.replace(os.sep, '_').replace('/', '_')}"
            snap_path = os.path.join(snap_dir, name)
            with open(snap_path, "w", encoding="utf-8") as f:
                f.write(snapshot)
            entry["snapshot"] = name
        with open(os.path.join(snap_dir, "journal.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        _prune_snapshots(snap_dir)
    except Exception:  # noqa: BLE001
        log.warning("undo journal write failed for %s", rel_path,
                    exc_info=True)


def _read_journal(snap_dir: str) -> list[dict]:
    import json

    journal_path = os.path.join(snap_dir, "journal.jsonl")
    if not os.path.isfile(journal_path):
        return []
    out: list[dict] = []
    with open(journal_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _write_journal(snap_dir: str, entries: list[dict]) -> None:
    import json

    with open(os.path.join(snap_dir, "journal.jsonl"), "w",
              encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _prune_snapshots(snap_dir: str) -> None:
    entries = _read_journal(snap_dir)
    if len(entries) <= SNAPSHOT_KEEP:
        return
    stale, keep = entries[:-SNAPSHOT_KEEP], entries[-SNAPSHOT_KEEP:]
    for entry in stale:
        name = entry.get("snapshot")
        if name:
            try:
                os.remove(os.path.join(snap_dir, os.path.basename(name)))
            except OSError:
                pass
    _write_journal(snap_dir, keep)


def undo_file_change(user_email: str) -> str:
    """Revert the most recent file change in the user's workspace.

    A replaced file gets its previous content back; a created file is
    removed. Repeatable: each call pops one change off the journal.
    """
    root = _user_root(user_email)
    snap_dir = os.path.join(root, SNAPSHOT_DIR)
    entries = _read_journal(snap_dir)
    if not entries:
        return "[error] nothing to undo"
    entry = entries.pop()
    rel = str(entry.get("path") or "")
    try:
        target = _resolve_under(root, rel)
    except ValueError:
        _write_journal(snap_dir, entries)
        return f"[error] journal entry for {rel!r} is not under the workspace"
    op = entry.get("op")
    if op == "create":
        try:
            if os.path.isfile(target):
                os.remove(target)
        except OSError as exc:
            return f"[error] could not remove {rel}: {exc}"
        _write_journal(snap_dir, entries)
        return f"undid create: removed {rel}"
    name = entry.get("snapshot")
    if not name:
        _write_journal(snap_dir, entries)
        return f"[error] no snapshot recorded for {rel}; nothing restored"
    snap_path = os.path.join(snap_dir, os.path.basename(str(name)))
    if not os.path.isfile(snap_path):
        _write_journal(snap_dir, entries)
        return f"[error] snapshot for {rel} has been pruned; nothing restored"
    with open(snap_path, encoding="utf-8") as f:
        previous = f.read()
    with open(target, "w", encoding="utf-8") as f:
        f.write(previous)
    _write_journal(snap_dir, entries)
    return f"undid change: restored {rel} ({len(previous)} bytes)"


def work_dir(user_email: str) -> str:
    """The user's workspace ROOT (created), used as the cwd for compute.

    Running at the root means relative paths the model naturally uses —
    ``outputs/report.xlsx``, ``uploads/data.csv``, ``work/tmp.txt`` —
    all resolve correctly and match the path convention of the file
    tools (view/create_file/download_file).  This is the seam between
    functional compute and storage."""
    d = _user_root(user_email)
    os.makedirs(d, exist_ok=True)
    for sub in ("uploads", "work", "outputs"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    return d


def download(user_email: str, url: str, filename: str = "", *, max_bytes: int = 100_000_000) -> dict:
    """Download a file (software installer, dataset, asset) from an
    http/https URL into the user's ``outputs/`` storage so it can be used
    for the task.  Bounded in size; the filename is basenamed so it can't
    escape the workspace."""
    import urllib.error
    import urllib.parse
    import urllib.request

    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.netloc:
        return {"ok": False, "error": "only http/https URLs are allowed"}
    # SSRF guard: refuse private/loopback/link-local (cloud metadata)
    # targets, reusing the vetted web-egress check.  This stops the agent
    # from being tricked into fetching http://169.254.169.254/ or internal
    # services and saving the response into the workspace.
    try:
        from agent.web_agentic import _url_ok

        if not _url_ok(url):
            return {"ok": False, "error": "URL blocked (private/internal address)"}
    except Exception:  # noqa: BLE001
        pass

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        """Every hop re-validated: urllib's default handler follows a
        30x chain silently, so a public URL that redirects to the
        metadata service (or any internal host) would pass the one
        check above and then exfiltrate into the workspace. Refuse
        redirects here and let the caller decide — the model can name
        the new URL explicitly, and it goes through this same gate."""

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            try:
                from agent.web_agentic import _url_ok

                if not _url_ok(newurl):
                    raise urllib.error.HTTPError(
                        newurl, code,
                        "redirect to a private/internal address blocked",
                        headers, fp,
                    )
            except urllib.error.HTTPError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise urllib.error.HTTPError(
                    newurl, code, f"redirect validation failed: {exc}",
                    headers, fp,
                ) from exc
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    name = os.path.basename(filename or u.path) or "download.bin"
    name = "".join(c for c in name if c.isalnum() or c in " ._-").strip() or "download.bin"
    dest = _resolve_under(_user_root(user_email), os.path.join("outputs", name))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Alex-VM/1.0"})
        opener = urllib.request.build_opener(_NoRedirect)
        with opener.open(req, timeout=60) as r:
            total = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        f.close()
                        os.unlink(dest)
                        return {"ok": False, "error": f"file exceeds {max_bytes} bytes"}
                    f.write(chunk)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"download failed: {exc}"}
    rel = f"outputs/{name}"
    return {
        "ok": True,
        "path": rel,
        "bytes": total,
        "download": f"/api/files/download?path={rel}",
        "message": f"Downloaded {name} ({total} bytes) into your workspace.",
    }


def output_path(user_email: str, filename: str) -> str:
    """Absolute path under the user's ``outputs/`` folder for a generated
    artifact, with the parent directory created.  Used by binary
    generators (Office documents) that write with their own libraries.
    The filename is basenamed so it can't escape ``outputs/``."""
    safe_name = os.path.basename(filename).strip() or "document"
    rel = os.path.join("outputs", safe_name)
    p = _resolve_under(_user_root(user_email), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def str_replace(user_email: str, path: str, old: str, new: str) -> str:
    """Replace the FIRST occurrence of `old` with `new` in `path`."""
    p = _resolve_under(_user_root(user_email), path)
    if not os.path.isfile(p):
        return f"[error] not a file: {path}"
    with open(p, encoding="utf-8") as f:
        text = f.read()
    if old not in text:
        return f"[error] substring not found in {path}"
    _journal_change(user_email, "replace", path, snapshot=text)
    text = text.replace(old, new, 1)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return f"replaced 1 occurrence in {path}" + _diagnostics(path, text)


def list_presentable(user_email: str) -> list[dict]:
    """Return files in `outputs/` that the user can be shown."""
    out_dir = os.path.join(_user_root(user_email), "outputs")
    if not os.path.isdir(out_dir):
        return []
    out: list[dict] = []
    for root, _, files in os.walk(out_dir):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, out_dir)
            st = os.stat(full)
            out.append(
                {"path": f"outputs/{rel}", "size": st.st_size, "modified": st.st_mtime}
            )
    return out


def bash(user_email: str, argv: list[str], *, timeout_seconds: int = 30) -> str:
    """Run a single command as an argv list. The first element must be a
    safe binary. NO shell, NO `shell=True`."""
    if not argv:
        return "[error] empty argv"
    bin_name = os.path.basename(argv[0])
    if bin_name not in SAFE_BINS:
        return f"[error] binary {bin_name!r} is not in the allowlist"
    if not shutil.which(argv[0]):
        return f"[error] binary {bin_name!r} not on PATH"
    cwd = _user_root(user_email)
    from agent.python_tool import child_env

    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            # Allowlisted, not inherited: an allowlisted binary is still
            # agent-directed, and `env` would otherwise hand it every
            # secret this process holds. See python_tool.child_env.
            env=child_env({"HOME": cwd}),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return f"[error] timed out after {timeout_seconds}s"
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        out = f"[exit {proc.returncode}]\n" + out
    if len(out) > 200_000:
        out = out[:200_000] + "\n[truncated to 200k bytes]"
    return out


# ---------------------------------------------------------------------------
# The factory the agent uses to assemble per-user tools
# ---------------------------------------------------------------------------


def make_workspace_tools(user_email: str) -> list:
    """Return a list of `@tool`-decorated functions bound to this user.
    The agent wraps each with `guardrails.guarded(...)` so the dual-ACL
    check runs on every call."""

    def _view(path: str) -> str:
        """Read a file from your workspace. Path is relative to your workspace root."""
        return view(user_email, path)

    def _create(path: str, content: str) -> str:
        """Create a new file in your workspace. Fails if it already exists."""
        return create_file(user_email, path, content)

    def _str_replace(path: str, old: str, new: str) -> str:
        """Replace the first occurrence of a substring in a file."""
        return str_replace(user_email, path, old, new)

    def _bash(argv: list[str]) -> str:
        """Run a single command. Provide the command as a list of args (e.g. ['ls','-la']). The shell is not used."""
        return bash(user_email, argv)

    def _present() -> list[dict]:
        """List files in outputs/ that you can show to the user."""
        return list_presentable(user_email)

    _view.name = "view"  # type: ignore[attr-defined]
    _create.name = "create_file"  # type: ignore[attr-defined]
    _str_replace.name = "str_replace"  # type: ignore[attr-defined]
    _bash.name = "bash"  # type: ignore[attr-defined]
    _present.name = "present_files"  # type: ignore[attr-defined]
    return [_view, _create, _str_replace, _bash, _present]
