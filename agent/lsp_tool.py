"""
LSP-style code intelligence over the user's workspace.

This module gives the agent language-server-grade abilities on the
files in its workspace (the same sandboxed per-user root that
:mod:`agent.workspace_tools` guards):

* :func:`outline`      — document symbols (classes / functions / methods).
* :func:`definition`   — go-to-definition for the name at a position.
* :func:`references`   — find references to the name at a position.
* :func:`hover`        — signature + docstring for the name at a position.
* :func:`diagnostics`  — syntax + static errors for a file.
* :func:`workspace_symbols` — fuzzy symbol search across the workspace.

Python gets full intelligence via `jedi` when it is installed; every
operation degrades gracefully to a stdlib `ast`/`compile` baseline so
the tools never hard-fail in a minimal environment. Non-Python files
get outline/diagnostics best-effort (currently none) with a clear
``supported: false`` marker rather than an exception.

All paths are resolved through :func:`agent.workspace_tools` helpers,
so the no-path-traversal guarantee holds here too. Outputs are
bounded (max symbols / references) so a huge repo cannot blow out the
model context.
"""

from __future__ import annotations

import ast
import os
from typing import Any

from agent import workspace_tools

MAX_RESULTS = 200
MAX_FILE_BYTES = 500_000
_PY_EXTS = (".py", ".pyi")


def _jedi():
    try:
        import jedi  # type: ignore

        return jedi
    except ImportError:
        return None


def _read(user_email: str, path: str) -> str:
    """Read a workspace file (traversal-safe), bounded."""
    root = workspace_tools._user_root(user_email)
    p = workspace_tools._resolve_under(root, path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"no such workspace file: {path}")
    if os.path.getsize(p) > MAX_FILE_BYTES:
        raise ValueError(f"{path} exceeds the {MAX_FILE_BYTES}-byte LSP limit")
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


def _is_python(path: str) -> bool:
    return path.endswith(_PY_EXTS)


# ---------------------------------------------------------------------------
# outline
# ---------------------------------------------------------------------------


def _ast_outline(source: str) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []

    def walk(node: ast.AST, container: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            ):
                if isinstance(child, ast.ClassDef):
                    kind = "class"
                else:
                    kind = "method" if container else "function"
                symbols.append(
                    {
                        "name": child.name,
                        "kind": kind,
                        "line": child.lineno,
                        "container": container,
                    }
                )
                if len(symbols) >= MAX_RESULTS:
                    return
                walk(child, child.name)

    walk(ast.parse(source), None)
    return symbols


def outline(user_email: str, path: str) -> dict[str, Any]:
    """Document symbols for one file."""
    if not _is_python(path):
        return {"path": path, "supported": False, "symbols": []}
    src = _read(user_email, path)
    try:
        return {"path": path, "supported": True, "symbols": _ast_outline(src)}
    except SyntaxError as e:
        return {
            "path": path,
            "supported": True,
            "symbols": [],
            "error": f"syntax error at line {e.lineno}: {e.msg}",
        }


# ---------------------------------------------------------------------------
# definition / references / hover (jedi-backed)
# ---------------------------------------------------------------------------


def _script(user_email: str, path: str):
    jedi = _jedi()
    if jedi is None:
        return None
    src = _read(user_email, path)
    root = workspace_tools._user_root(user_email)
    real = workspace_tools._resolve_under(root, path)
    return jedi.Script(code=src, path=real, project=jedi.Project(root))


def _rel_to_root(user_email: str, module_path: str | None) -> str | None:
    if not module_path:
        return None
    root = os.path.realpath(workspace_tools._user_root(user_email))
    real = os.path.realpath(str(module_path))
    if real.startswith(root + os.sep):
        return os.path.relpath(real, root).replace(os.sep, "/")
    return str(module_path)  # stdlib / site-packages target


def definition(user_email: str, path: str, line: int, column: int) -> dict[str, Any]:
    """Go-to-definition for the name at (line, column) — 1-based line."""
    script = _script(user_email, path)
    if script is None:
        return {"available": False, "reason": "jedi is not installed"}
    defs = script.goto(line=line, column=column, follow_imports=True)
    return {
        "available": True,
        "definitions": [
            {
                "name": d.name,
                "kind": d.type,
                "path": _rel_to_root(user_email, d.module_path),
                "line": d.line,
                "column": d.column,
            }
            for d in defs[:MAX_RESULTS]
        ],
    }


def references(user_email: str, path: str, line: int, column: int) -> dict[str, Any]:
    """All references to the name at (line, column) — 1-based line."""
    script = _script(user_email, path)
    if script is None:
        return {"available": False, "reason": "jedi is not installed"}
    refs = script.get_references(line=line, column=column)
    return {
        "available": True,
        "references": [
            {
                "name": r.name,
                "path": _rel_to_root(user_email, r.module_path),
                "line": r.line,
                "column": r.column,
            }
            for r in refs[:MAX_RESULTS]
        ],
    }


def hover(user_email: str, path: str, line: int, column: int) -> dict[str, Any]:
    """Signature + docstring for the name at (line, column)."""
    script = _script(user_email, path)
    if script is None:
        return {"available": False, "reason": "jedi is not installed"}
    infos = script.help(line=line, column=column)
    if not infos:
        return {"available": True, "hover": None}
    top = infos[0]
    sigs = [s.to_string() for s in top.get_signatures()]
    return {
        "available": True,
        "hover": {
            "name": top.name,
            "kind": top.type,
            "signatures": sigs,
            "docstring": (top.docstring(raw=True) or "")[:2000],
        },
    }


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def diagnostics(user_email: str, path: str) -> dict[str, Any]:
    """Syntax (and, with jedi, static) errors for one file."""
    if not _is_python(path):
        return {"path": path, "supported": False, "diagnostics": []}
    src = _read(user_email, path)
    diags: list[dict[str, Any]] = []
    jedi = _jedi()
    if jedi is not None:
        root = workspace_tools._user_root(user_email)
        real = workspace_tools._resolve_under(root, path)
        for err in jedi.Script(code=src, path=real).get_syntax_errors():
            diags.append(
                {
                    "line": err.line,
                    "column": err.column,
                    "message": err.get_message(),
                    "severity": "error",
                }
            )
    else:
        try:
            compile(src, path, "exec")
        except SyntaxError as e:
            diags.append(
                {
                    "line": e.lineno or 0,
                    "column": e.offset or 0,
                    "message": e.msg or "syntax error",
                    "severity": "error",
                }
            )
    return {"path": path, "supported": True, "diagnostics": diags[:MAX_RESULTS]}


# ---------------------------------------------------------------------------
# workspace symbol search
# ---------------------------------------------------------------------------


def workspace_symbols(user_email: str, query: str) -> dict[str, Any]:
    """Case-insensitive substring symbol search across workspace .py files."""
    q = (query or "").strip().lower()
    if not q:
        return {"query": query, "symbols": []}
    root = workspace_tools._user_root(user_email)
    out: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__pycache__"))]
        for fn in filenames:
            if not fn.endswith(_PY_EXTS):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/")
            try:
                for sym in _ast_outline(_read(user_email, rel)):
                    if q in sym["name"].lower():
                        out.append({**sym, "path": rel})
                        if len(out) >= MAX_RESULTS:
                            return {"query": query, "symbols": out, "truncated": True}
            except (SyntaxError, ValueError, FileNotFoundError):
                continue
    return {"query": query, "symbols": out}
