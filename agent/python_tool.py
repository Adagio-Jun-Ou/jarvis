"""
Governed code execution.

The agent has TWO execution paths:

  * `run_python` — a sandboxed Python invocation. In production this
    routes to the agent's own compute VM via a `command_execution`
    Lakeflow job (a separate cluster under the sandbox cluster policy).
    The job is the trust boundary; the agent sends a payload and
    awaits the result.

  * `bash` and friends — the workspace tools in `workspace_tools.py`.
    These run on a per-user single-user cluster with the egress policy
    enforced at the network layer.

This module is the *Python* tool. It deliberately does NOT shell out
via `subprocess.run(cmd, shell=True)`; everything is `shlex.split` +
list-form argv + a strict allowlist of binary invocations.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading

log = logging.getLogger("agent.python_tool")

# Hard cap on runtime. The sandbox cluster has a tighter cap (vm_policy
# in control_plane.sql) but we also enforce here as defense in depth.
MAX_RUNTIME_SECONDS = int(os.environ.get("ALEX_RUN_PY_TIMEOUT", "60"))
MAX_OUTPUT_BYTES = 1_000_000

# A timeout alone does not bound what code can do to the host: a process
# that allocates without limit can push the machine into swap long before
# its sixty seconds are up, and it takes the whole workspace down with
# it, not just its own run. These are the other two dimensions.
MAX_MEMORY_MB = int(os.environ.get("ALEX_RUN_MEM_MB", "2048"))
MAX_PROCESSES = int(os.environ.get("ALEX_RUN_MAX_PROCS", "96"))
# How often the Windows watchdog samples. Fine enough to catch a runaway
# before it hurts, coarse enough not to be its own load.
_WATCH_INTERVAL_S = 0.25

# Win32 constants for the job object below, from winnt.h. Named here
# rather than inline so the flag arithmetic reads as intent.
_JOB_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_LIMIT_JOB_MEMORY = 0x00000200
_JOB_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_EXTENDED_LIMIT_INFO = 9  # JobObjectExtendedLimitInformation
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001

# The only Python binary we'll ever invoke. Set by the bundle to the
# cluster's Python path; default to whatever's on PATH.
PYTHON_BIN = os.environ.get("ALEX_PYTHON_BIN", shutil.which("python") or "python3")


# The environment a child process is allowed to see. Everything else is
# withheld, including anything the server holds for its own use.
#
# Passing the parent environment through — the previous behaviour — meant
# agent-written code could read every secret the process was started
# with: Databricks tokens, warehouse credentials, model API keys, SSO
# client secrets. No sandbox escape was needed, just `print(os.environ)`.
# The timeout and the workspace jail both assume the code cannot simply
# take the keys and act elsewhere; this is what makes that assumption
# true. It is an allowlist rather than a denylist because a denylist has
# to be right about every secret anyone ever adds.
_ENV_ALLOW = frozenset({
    # Finding and running the interpreter at all.
    "PATH", "PATHEXT", "COMSPEC", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR",
    "PYTHONPATH", "PYTHONHOME", "PYTHONIOENCODING", "PYTHONUTF8",
    "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE",
    # Somewhere to write scratch files.
    "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
    # Locale and platform facts programs reasonably expect.
    "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
})


# Installing a package runs that package's build code, so pip is an
# execution path too. It gets the base allowlist plus its own network
# and index configuration, without which installs fail in precisely the
# environments that have a private index or an egress proxy. That is a
# deliberate trade: a malicious package can read the index credential
# the operator configured for fetching it, and nothing else.
_PIP_ENV_ALLOW = frozenset({
    "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST",
    "PIP_CONFIG_FILE", "PIP_CACHE_DIR", "PIP_NO_CACHE_DIR",
    "PIP_DEFAULT_TIMEOUT", "PIP_RETRIES", "VIRTUAL_ENV",
    # Matching is case-insensitive, so the lowercase spellings of these
    # that POSIX tools actually use are covered by the same entry.
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE",
})


def child_env(
    extra: dict | None = None, *, allow: frozenset[str] = _ENV_ALLOW
) -> dict[str, str]:
    """The environment handed to agent-run code: allowlisted, plus what
    the caller explicitly adds. ``extra`` comes from our own code, never
    from the model — no tool exposes it as a parameter."""
    base = {k: v for k, v in os.environ.items() if k.upper() in allow}
    base.update({str(k): str(v) for k, v in (extra or {}).items()})
    return base


def pip_env(extra: dict | None = None) -> dict[str, str]:
    """:func:`child_env` widened by pip's own network/index settings."""
    return child_env(extra, allow=_ENV_ALLOW | _PIP_ENV_ALLOW)


def _rlimits(memory_mb: int, cpu_seconds: int):
    """A ``preexec_fn`` applying kernel resource limits, or None.

    POSIX only — `resource` does not exist on Windows, which is why the
    watchdog below exists. Applied in the child between fork and exec, so
    the limits are the kernel's to enforce and the child cannot lift
    them: `setrlimit` on a hard limit is one-way for an unprivileged
    process.
    """
    try:
        import resource
    except ImportError:
        return None

    def apply() -> None:  # pragma: no cover - POSIX only
        for name, limit in (
            ("RLIMIT_AS", memory_mb * 1024 * 1024),
            ("RLIMIT_CPU", cpu_seconds),
            ("RLIMIT_NPROC", MAX_PROCESSES),
            ("RLIMIT_FSIZE", MAX_OUTPUT_BYTES * 64),
            ("RLIMIT_CORE", 0),  # no core dumps: they can be huge
        ):
            what = getattr(resource, name, None)
            if what is None:
                continue
            try:
                resource.setrlimit(what, (limit, limit))
            except (ValueError, OSError):
                # A limit lower than one already in force, or one this
                # kernel refuses. Skip it rather than fail the run.
                pass

    return apply


def _watch_memory(proc: subprocess.Popen, memory_mb: int, stop) -> None:
    """Kill the process tree if it exceeds ``memory_mb`` resident.

    The fallback for platforms with no `setrlimit` (Windows). It measures
    the *tree*, not the single process, because spawning a child is the
    obvious way around a per-process cap. Sampling means a process can
    briefly exceed the limit before being killed, so this is a backstop
    against a runaway rather than a precise quota — which is what the
    limit is for.
    """
    try:
        import psutil
    except ImportError:  # no watchdog available; the timeout still applies
        log.debug("psutil missing: no memory limit on this platform")
        return
    limit = memory_mb * 1024 * 1024
    try:
        root = psutil.Process(proc.pid)
    except Exception:  # noqa: BLE001
        return
    while not stop.wait(_WATCH_INTERVAL_S):
        try:
            procs = [root, *root.children(recursive=True)]
            used = 0
            for p in procs:
                try:
                    used += p.memory_info().rss
                except Exception:  # noqa: BLE001
                    continue  # exited between listing and measuring
            if used > limit:
                log.warning(
                    "killing run: %.0f MB exceeds the %d MB limit",
                    used / 1024 / 1024, memory_mb,
                )
                _kill_tree(proc)
                return
        except Exception:  # noqa: BLE001
            return  # the process is gone; nothing left to police


def _windows_job(memory_mb: int):
    """A Windows job object capping memory and process count, or None.

    The kernel's answer to `setrlimit`, and the reason the watchdog above
    is only a fallback: a job memory limit makes the *allocation itself*
    fail, so a runaway is stopped at the moment it asks for the memory. A
    sampling watchdog cannot promise that — a tight loop allocating
    16 MB blocks passes two gigabytes between two samples, which is
    exactly what it did before this existed.

    ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` means the whole tree dies with
    the handle even if we crash, and ``JOB_OBJECT_LIMIT_JOB_MEMORY``
    applies to the tree as a whole, so spawning children does not
    multiply the budget.
    """
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _Basic),
            ("IoInfo", _IO),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        log.debug("CreateJobObject failed; no memory limit for this run")
        return None

    info = _Extended()
    info.BasicLimitInformation.LimitFlags = (
        _JOB_LIMIT_JOB_MEMORY
        | _JOB_LIMIT_PROCESS_MEMORY
        | _JOB_LIMIT_ACTIVE_PROCESS
        | _JOB_LIMIT_KILL_ON_JOB_CLOSE
    )
    info.BasicLimitInformation.ActiveProcessLimit = MAX_PROCESSES
    info.ProcessMemoryLimit = memory_mb * 1024 * 1024
    info.JobMemoryLimit = memory_mb * 1024 * 1024
    if not kernel32.SetInformationJobObject(
        job, _JOB_EXTENDED_LIMIT_INFO, ctypes.byref(info), ctypes.sizeof(info)
    ):
        log.debug("SetInformationJobObject failed; no memory limit for this run")
        kernel32.CloseHandle(job)
        return None
    return job


def _assign_to_job(job, pid: int) -> bool:
    """Put a process under a job object's limits."""
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(
        _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid
    )
    if not handle:
        return False
    try:
        return bool(kernel32.AssignProcessToJobObject(job, handle))
    finally:
        kernel32.CloseHandle(handle)


def _run_tree(
    argv: list[str],
    *,
    cwd: str | None,
    env: dict | None,
    timeout: float,
    memory_mb: int = 0,
) -> tuple[int, str]:
    """Run a subprocess under a time, memory and process-count bound.

    On timeout or a memory overrun, the WHOLE process tree is killed so
    grandchildren (a nested Popen, a fork bomb) cannot orphan and keep
    consuming the VM's resources. Returns (returncode, combined_output);
    ``returncode`` is 124 on timeout, the conventional timeout code.
    """
    memory_mb = memory_mb or MAX_MEMORY_MB
    popen_kw: dict = {}
    job = None
    kernel_enforced = False
    if os.name == "nt":
        popen_kw["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
        job = _windows_job(memory_mb)
    else:
        popen_kw["start_new_session"] = True  # own process group
        limits = _rlimits(memory_mb, int(timeout) + 5)
        if limits is not None:
            popen_kw["preexec_fn"] = limits
            kernel_enforced = True

    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=child_env(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        **popen_kw,
    )
    if job is not None:
        # Assigned immediately after spawn rather than via CREATE_SUSPENDED,
        # which subprocess does not expose a thread handle to resume. The
        # window is the few microseconds before a fresh interpreter has
        # even loaded, so there is nothing running in it to exceed a limit.
        kernel_enforced = _assign_to_job(job, proc.pid)
        if not kernel_enforced:
            log.debug("could not assign pid %s to the job object", proc.pid)

    # Where the kernel enforces the limit, do not also pay for polling.
    watcher = stop = None
    if not kernel_enforced:
        stop = threading.Event()
        watcher = threading.Thread(
            target=_watch_memory, args=(proc, memory_mb, stop), daemon=True
        )
        watcher.start()

    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out or ""
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, _ = proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            out = ""
        return 124, (out or "") + f"\n[error] timed out after {timeout:.0f}s"
    finally:
        # Retire the watchdog on every path including the timeout: a
        # polling thread per run that outlives its process is its own
        # slow leak. Closing the job handle kills anything still in it,
        # which is what collects grandchildren we never saw.
        if stop is not None:
            stop.set()
        if watcher is not None:
            watcher.join(timeout=2)
        if job is not None:
            import ctypes

            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=10,
            )
        else:
            import signal as _sig

            os.killpg(os.getpgid(proc.pid), _sig.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def run_python(
    code: str, *, cwd: str | None = None, env: dict | None = None
) -> str:
    """Run a small Python snippet. Returns the combined stdout/stderr,
    truncated. The snippet is NOT shell-expanded; we pass it via
    `python -c "<code>"` after a sanitization pass.

    The function deliberately does NOT use `shell=True`. The code is
    passed as a single argument to `python -c`, which is a documented
    Python feature. There is no command-injection vector.
    """
    if not code or not code.strip():
        return ""
    # Strip a leading shebang if any.
    code = code.lstrip()
    if code.startswith("#!"):
        nl = code.find("\n")
        if nl >= 0:
            code = code[nl + 1 :]
    argv = [PYTHON_BIN, "-I", "-c", code]
    try:
        rc, out = _run_tree(
            argv, cwd=cwd, env=env, timeout=MAX_RUNTIME_SECONDS
        )
    except FileNotFoundError:
        return f"[error] python interpreter not found: {PYTHON_BIN}"
    if len(out) > MAX_OUTPUT_BYTES:
        out = out[:MAX_OUTPUT_BYTES] + f"\n[truncated to {MAX_OUTPUT_BYTES} bytes]"
    if rc != 0 and rc != 124:
        out = f"[exit {rc}]\n" + out
    return out


def run_shell(
    argv: list[str],
    *,
    cwd: str | None = None,
    env: dict | None = None,
    timeout_seconds: int = MAX_RUNTIME_SECONDS,
) -> str:
    """Run a single shell command as an argv list. No `shell=True`.
    The first element of `argv` must be a binary name in `ALLOWLIST_BINS`."""
    if not argv:
        return "[error] empty argv"
    bin_name = os.path.basename(argv[0])
    if bin_name not in ALLOWLIST_BINS:
        return f"[error] binary {bin_name!r} is not in the allowlist"
    if not shutil.which(argv[0]):
        return f"[error] binary {bin_name!r} not on PATH"
    rc, out = _run_tree(argv, cwd=cwd, env=env, timeout=timeout_seconds)
    if len(out) > MAX_OUTPUT_BYTES:
        out = out[:MAX_OUTPUT_BYTES] + f"\n[truncated to {MAX_OUTPUT_BYTES} bytes]"
    if rc != 0 and rc != 124:
        out = f"[exit {rc}]\n" + out
    return out


# Allowlist of safe binaries for the shell-style invocation. The agent's
# bash tool is the workspace_tools tool, NOT this one; this is only a
# fallback for short, non-interactive commands the model needs to run.
ALLOWLIST_BINS = frozenset(
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
        # Package managers + build/runtime tools the agent's VM may use.
        "pip",
        "pip3",
        "uv",
        "node",
        "npm",
        "npx",
        "winget",
        "apt-get",
        "brew",
        "dir",
        "where",
        "echo",
    }
)


# A pip/uv requirement token: package name with optional extras and a
# version specifier. NO flags, URLs, paths, or shell metacharacters — so
# the model cannot smuggle `--index-url`, `-e .`, `; rm -rf`, etc.
_REQ_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9,._-]+\])?"
    r"([=<>!~]=?[A-Za-z0-9.*+!-]+(,[=<>!~]=?[A-Za-z0-9.*+!-]+)*)?$"
)


def _validate_requirements(packages: list[str]) -> tuple[list[str], str | None]:
    """Validate a list of package requirement strings. Returns
    (clean, error). Rejects anything that isn't a plain name[+extras][+version]."""
    if not packages:
        return [], "no packages given"
    clean: list[str] = []
    for raw in packages:
        tok = str(raw).strip()
        if not tok or not _REQ_RE.match(tok):
            return [], f"invalid or unsafe package spec: {raw!r}"
        clean.append(tok)
    if len(clean) > 25:
        return [], "too many packages in one call (max 25)"
    return clean, None


# Package managers the agent may drive, with their non-interactive
# install invocation.  The command name must be on PATH to be used.
_SYS_MANAGERS: dict[str, list[str]] = {
    "winget": [
        "winget", "install", "-e", "--silent",
        "--accept-package-agreements", "--accept-source-agreements", "--id",
    ],
    "apt-get": ["apt-get", "install", "-y"],
    "brew": ["brew", "install"],
}
_SYS_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")

MAX_INSTALL_SECONDS = int(os.environ.get("ALEX_INSTALL_TIMEOUT", "300"))

# ---------------------------------------------------------------------------
# The approved library — the terminal's package universe.
#
# The agent gets a FULL Python data/ML stack, but a curated one: every
# name here is widely used, actively maintained, and something an
# enterprise analyst's task plausibly needs. Everything else is refused
# with a message that says how an operator extends the set — because
# "pip install anything" is a supply-chain door (typosquats, abandoned
# packages) and, on Databricks Apps, a COST door: installs burn the
# app container's 2 vCPUs. The genuinely heavy frameworks (torch,
# transformers) are approved but belong on the sandbox cluster, not the
# app container — the tool description and the runbook both say so.
# ---------------------------------------------------------------------------

APPROVED_PACKAGES: frozenset[str] = frozenset({
    # core numerics / frames
    "numpy", "pandas", "polars", "pyarrow", "duckdb", "scipy",
    # ML / stats
    "scikit-learn", "statsmodels", "xgboost", "lightgbm", "joblib",
    # deep learning / NLP (heavy — prefer the sandbox cluster)
    "torch", "transformers", "sentence-transformers", "onnxruntime",
    "nltk", "spacy",
    # plotting / imaging
    "matplotlib", "seaborn", "plotly", "pillow",
    # documents / files
    "openpyxl", "xlsxwriter", "python-docx", "python-pptx", "pypdf",
    "pyyaml", "jsonschema", "tabulate", "beautifulsoup4", "lxml",
    # data access
    "requests", "httpx", "sqlalchemy", "psycopg", "psycopg2-binary",
    "databricks-sdk", "databricks-sql-connector", "mlflow",
    # utilities
    "tqdm", "networkx", "sympy", "regex", "rich", "python-dateutil",
    "pytz",
})

_BASE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _package_base_name(spec: str) -> str:
    """The bare package name of a requirement spec: extras and version
    pins stripped, normalized the way PyPI compares names."""
    m = _BASE_NAME_RE.match(spec.strip())
    return (m.group(0) if m else "").lower().replace("_", "-")


def _approved_names() -> frozenset[str]:
    """The approved set plus the operator's extensions.

    ``ALEX_EXTRA_PACKAGES`` (comma/space separated) is deliberately an
    OPERATOR control — an environment variable set at deploy time —
    never something the model or a request can write."""
    extra = {
        _package_base_name(p)
        for p in re.split(r"[,\s]+", os.environ.get("ALEX_EXTRA_PACKAGES", ""))
        if p.strip()
    }
    return APPROVED_PACKAGES | frozenset(x for x in extra if x)


def pip_install(
    packages: list[str], *, upgrade: bool = False, cwd: str | None = None
) -> str:
    """Install Python packages into the VM's interpreter with
    ``python -m pip install`` so a subsequent ``run_python`` can import
    them.  Package specs are validated (name[extras][version] only) and
    must come from the approved library set."""
    clean, err = _validate_requirements(packages)
    if err:
        return f"[error] {err}"
    approved = _approved_names()
    refused = [p for p in clean if _package_base_name(p) not in approved]
    if refused:
        return (
            f"[error] not in the approved library: {', '.join(refused)}. "
            "The approved set covers the Python data/ML stack (numpy, "
            "pandas, polars, scikit-learn, torch, matplotlib, and kin). "
            "If a task genuinely needs another package, ask the user to "
            "have an operator add it to ALEX_EXTRA_PACKAGES — do not "
            "retry with lookalike names."
        )
    argv = [PYTHON_BIN, "-m", "pip", "install", "--disable-pip-version-check"]
    if upgrade:
        argv.append("--upgrade")
    argv += clean
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=pip_env(),
            capture_output=True,
            text=True,
            timeout=MAX_INSTALL_SECONDS,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return f"[error] pip install timed out after {MAX_INSTALL_SECONDS}s"
    out = (proc.stdout or "") + (proc.stderr or "")
    # pip is chatty; keep the tail (where success/errors land).
    if len(out) > 6000:
        out = "…\n" + out[-6000:]
    status = "installed" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
    return f"[pip {status}] {' '.join(clean)}\n{out}"


def system_install(packages: list[str], manager: str | None = None) -> str:
    """Install OS-level packages via the platform package manager
    (winget on Windows, apt-get on Debian/Ubuntu, brew on macOS).  Package
    IDs are validated; installs are non-interactive."""
    if not packages:
        return "[error] no packages given"
    ids: list[str] = []
    for p in packages:
        t = str(p).strip()
        if not t or not _SYS_ID_RE.match(t):
            return f"[error] invalid package id: {p!r}"
        ids.append(t)
    # Pick the manager: explicit, else first available on PATH.
    chosen = manager if manager in _SYS_MANAGERS else None
    if chosen and not shutil.which(chosen):
        return f"[error] requested manager {chosen!r} is not available on this VM"
    if not chosen:
        for m in _SYS_MANAGERS:
            if shutil.which(m):
                chosen = m
                break
    if not chosen:
        return "[error] no supported package manager found (winget/apt-get/brew)"
    base = _SYS_MANAGERS[chosen]
    results: list[str] = []
    for pid in ids:
        try:
            proc = subprocess.run(
                base + [pid],
                capture_output=True,
                text=True,
                timeout=MAX_INSTALL_SECONDS,
                shell=False,
                # The allowlist exists precisely so a package's install
                # hooks cannot read the server's secrets; this was the
                # one path that inherited os.environ verbatim —
                # DATABRICKS_TOKEN, N8N_API_KEY and friends one
                # `printenv` away from a postinstall script.
                env=child_env(),
            )
        except subprocess.TimeoutExpired:
            results.append(f"{pid}: timed out")
            continue
        tail = ((proc.stdout or "") + (proc.stderr or ""))[-1500:]
        ok = "installed" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
        results.append(f"{pid}: {ok}\n{tail}")
    return f"[{chosen}]\n" + "\n\n".join(results)


def vm_exec(argv: list[str], *, cwd: str | None = None, timeout_seconds: int | None = None) -> str:
    """Run one allowlisted command in the VM (the agent's CLI surface).
    ``argv`` is a list — the first element is the binary (must be in
    :data:`ALLOWLIST_BINS`); no shell, so no metacharacter injection.
    Package managers (pip/winget/apt-get/brew/npm) are allowlisted."""
    return run_shell(
        argv, cwd=cwd, timeout_seconds=timeout_seconds or MAX_INSTALL_SECONDS
    )


# Sanity check at import time in production: refuse to load if the
# binary is not on PATH.
def _smoke_check() -> None:
    if not shutil.which(PYTHON_BIN):
        log.warning(
            "python_tool: %s is not on PATH; run_python will return an error",
            PYTHON_BIN,
        )


_smoke_check()
