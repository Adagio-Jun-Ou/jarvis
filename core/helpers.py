"""Utility helpers for common patterns across the codebase."""
import asyncio
import logging
import os
import re as _re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 100_000
LOG_INJECTION_PATTERN = _re.compile(r'[\r\n\t]')

_SAFE_ERROR_MESSAGES = {
    "PermissionError": "Permission denied",
    "FileNotFoundError": "Resource not found",
    "ValueError": "Invalid input",
    "TypeError": "Invalid type",
    "KeyError": "Resource not found",
    "ConnectionError": "Connection failed",
    "TimeoutError": "Operation timed out",
    "OSError": "System error",
    "RuntimeError": "Internal error",
}


def safe_error_message(exc: BaseException, default: str = "Internal error") -> str:
    """Convert exception to safe error message for external responses.
    
    Prevents information disclosure by mapping exception types to generic
    messages while logging the full traceback internally.
    """
    exc_type = type(exc).__name__
    safe_msg = _SAFE_ERROR_MESSAGES.get(exc_type, default)
    logger.debug(f"Sanitized error: {exc_type}: {exc!r} -> {safe_msg}")
    return safe_msg

# --- Protected path checks -------------------------------------------------
#
# Several subsystems (native files, terminal, docker volume mounts, execution
# context) maintain their own list of system directories that must never be
# read, written or mounted. Those policies differ on purpose, but they all need
# the same *matching* behaviour, which is what these helpers provide:
#
#   - path semantics instead of string prefixes, so "/etcetera" is not treated
#     as living inside "/etc" and Windows separators/case are handled;
#   - platform awareness, since a POSIX-only list matches nothing on Windows
#     and silently leaves the real system directories unprotected.

_WINDOWS_PROTECTED_ENV_VARS = (
    "SystemRoot", "windir", "ProgramFiles", "ProgramFiles(x86)", "ProgramData",
)


def platform_protected_roots() -> list[Path]:
    """System directories that are protected on the current platform.

    Returns the Windows system locations (read from the environment so the
    result is correct regardless of install drive). On POSIX this is empty:
    callers already carry explicit "/etc", "/root", ... lists.
    """
    if os.name != "nt":
        return []
    roots = []
    for var in _WINDOWS_PROTECTED_ENV_VARS:
        value = os.environ.get(var)
        if value:
            roots.append(Path(value))
    return roots


def path_is_within(candidate: Any, root: Any) -> bool:
    """True when `candidate` is `root` itself or sits underneath it."""
    try:
        c = Path(os.path.normcase(str(candidate)))
        r = Path(os.path.normcase(str(root)))
        return c == r or c.is_relative_to(r)
    except Exception:
        return False


def is_protected_path(candidate: Any, roots: Any = ()) -> bool:
    """True if `candidate` falls inside any of `roots` or a platform system dir.

    Fails closed: an uninterpretable path is reported as protected.
    """
    try:
        resolved = Path(candidate)
    except Exception:
        return True
    for root in list(roots) + platform_protected_roots():
        if path_is_within(resolved, root):
            return True
    return False


_tracked_tasks: set[asyncio.Task] = set()


def create_tracked_task(coro, *, name: str = "unnamed") -> asyncio.Task:
    """Create an asyncio.Task that is tracked for error logging and cleanup.

    Unlike bare asyncio.create_task(), this ensures:
    - Exceptions are logged instead of silently swallowed
    - Tasks are tracked in a weak set so they can be awaited on shutdown
    - A done callback removes the task from tracking and logs errors

    Args:
        coro: Coroutine to schedule
        name: Human-readable name for logging

    Returns:
        The created asyncio.Task
    """
    task = asyncio.create_task(coro)
    _tracked_tasks.add(task)

    def _on_done(t: asyncio.Task) -> None:
        _tracked_tasks.discard(t)
        try:
            exc = t.exception()
            if exc is not None and not isinstance(exc, asyncio.CancelledError):
                logger.error(f"Tracked task '{name}' failed: {exc}", exc_info=exc)
        except asyncio.CancelledError:
            pass

    task.add_done_callback(_on_done)
    return task


async def cancel_tracked_tasks() -> None:
    """Cancel all tracked tasks and wait for them to finish."""
    if not _tracked_tasks:
        return
    for task in list(_tracked_tasks):
        if not task.done():
            task.cancel()
    await asyncio.gather(*list(_tracked_tasks), return_exceptions=True)
    _tracked_tasks.clear()


def safe_get(data: dict, *keys: str, default: Any = "") -> Any:
    """Safely get nested dictionary values.
    
    Args:
        data: Dictionary to traverse
        *keys: Keys to traverse in order
        default: Default value if any key is missing
    
    Returns:
        Value at path or default
    
    Example:
        safe_get(response, "message", "content") 
        # Equivalent to: response.get("message", {}).get("content", "")
    """
    result = data
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
            if result is None:
                return default
        else:
            return default
    return result if result is not None else default


def get_response_content(response: dict | Any, default: str = "") -> str:
    """Extract content from LLM response safely.

    Refused responses (the api_manager's ``_refused`` flag — rate limit /
    queue full / timeout / LLM error) carry the error string in the same
    shape as a genuine answer. A refusal is not an answer, so it extracts
    to *default* here — the single chokepoint every subsystem's LLM flow
    passes through. Without this, modules processed "Rate limit exceeded"
    as if the model had said it.

    Args:
        response: LLM response dict or any value
        default: Default string if content not found

    Returns:
        Content string or default
    """
    if not isinstance(response, dict):
        return default
    if response.get("_refused"):
        return default
    message = response.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        return content if isinstance(content, str) else default
    return default


def get_response_thinking(response: dict | Any, default: str = "") -> str:
    """Extract thinking from LLM response safely.
    
    Args:
        response: LLM response dict or any value
        default: Default string if thinking not found
    
    Returns:
        Thinking string or default
    """
    if not isinstance(response, dict):
        return default
    if response.get("_refused"):
        return default
    thinking = response.get("thinking")
    if thinking and isinstance(thinking, str):
        return thinking
    message = response.get("message")
    if isinstance(message, dict):
        thinking = message.get("thinking")
        return thinking if isinstance(thinking, str) else default
    return default


def get_tool_call_info(tool_call: dict) -> tuple[str | None, dict | None]:
    """Extract name and arguments from tool call safely.
    
    Args:
        tool_call: Tool call dict
    
    Returns:
        Tuple of (name, arguments) or (None, None)
    """
    if not isinstance(tool_call, dict):
        return None, None
    func = tool_call.get("function")
    if not isinstance(func, dict):
        return None, None
    name = func.get("name")
    args = func.get("arguments")
    return (
        name if isinstance(name, str) else None,
        args if isinstance(args, dict) else None
    )


import hashlib as _hashlib
import hmac as _hmac_mod
import secrets as _secrets
import os as _os
import socket as _socket

def _derive_hmac_key() -> bytes:
    key_path = _os.path.expanduser("~/.jarvis/.hmac_key")
    try:
        if _os.path.exists(key_path):
            with open(key_path, "rb") as f:
                return _hashlib.sha256(f.read() + b"jarvis_v1").digest()
    except Exception:
        pass
    key = _secrets.token_bytes(32)
    _os.makedirs(_os.path.dirname(key_path), exist_ok=True)
    with open(key_path, "wb") as f:
        f.write(key)
    _os.chmod(key_path, 0o600)
    return _hashlib.sha256(key + b"jarvis_v1").digest()

_PERSISTENCE_HMAC_KEY = _derive_hmac_key()

def compute_hmac(data_bytes: bytes) -> str:
    """Compute HMAC-SHA256 for persistence integrity verification."""
    return _hmac_mod.new(_PERSISTENCE_HMAC_KEY, data_bytes, _hashlib.sha256).hexdigest()


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Atomically write JSON data to a file using write-then-rename pattern.
    
    Prevents corruption if the process crashes mid-write by writing to a
    temporary file first, then atomically renaming it to the target path.
    
    Security: Sets file mode to 0600 (owner-only read/write) to prevent
    other users from reading sensitive data.
    Includes HMAC integrity check.
    """
    import json
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    try:
        content = json.dumps(data, indent=indent, default=str)
        content_bytes = content.encode('utf-8')
        hmac_value = compute_hmac(content_bytes)
        wrapped = {"_hmac": hmac_value, "_data": data}
        wrapped_bytes = json.dumps(wrapped, indent=indent, default=str).encode('utf-8')
        fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, wrapped_bytes)
        finally:
            os.close(fd)
        tmp_path.replace(path)
        os.chmod(str(path), 0o600)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def read_json_with_integrity(path: Path) -> Any:
    """Read JSON file with HMAC integrity verification.
    
    Returns the data if integrity check passes, or raises ValueError if tampered.
    Falls back to reading plain JSON if no HMAC wrapper is found.
    """
    import json
    path = Path(path)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding='utf-8')
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "_hmac" in parsed and "_data" in parsed:
            data_bytes = json.dumps(parsed["_data"], indent=2, default=str).encode('utf-8')
            expected_hmac = compute_hmac(data_bytes)
            if not _hmac_compare(expected_hmac, parsed["_hmac"]):
                logger.warning(f"Integrity check failed for {path}: HMAC mismatch (key rotation or tampering)")
                raise ValueError(f"Data integrity check failed for {path}")
            return parsed["_data"]
        return parsed
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from {path}: {e}")
        return None


def _hmac_compare(a: str, b: str) -> bool:
    """Constant-time HMAC comparison to prevent timing attacks."""
    import hmac as _hmac
    return _hmac.compare_digest(a.encode(), b.encode())


def clamp_input(text: str, max_length: int = MAX_INPUT_LENGTH) -> str:
    """Clamp input string to max length to prevent resource exhaustion."""
    if len(text) > max_length:
        return text[:max_length]
    return text


def sanitize_log_message(msg: str) -> str:
    """Sanitize log messages to prevent log injection.

    Strips CR/LF/TAB characters that could be used for log injection attacks.
    """
    return LOG_INJECTION_PATTERN.sub(' ', msg)[:1000]


def validate_request_fields(
    data: dict,
    required: list[str] | None = None,
    optional: list[str] | None = None,
    max_string_length: int = MAX_INPUT_LENGTH,
    max_dict_keys: int = 50,
) -> tuple[dict, str | None]:
    """Validate request body fields.

    Returns (cleaned_data, error_message).
    If error_message is not None, validation failed.
    """
    if not isinstance(data, dict):
        return {}, "Request body must be a JSON object"

    if len(data) > max_dict_keys:
        return {}, f"Too many fields (max {max_dict_keys})"

    required = required or []
    optional = optional or []

    cleaned = {}
    for field in required:
        if field not in data:
            return {}, f"Missing required field: {field}"
        val = data[field]
        if isinstance(val, str) and len(val) > max_string_length:
            return {}, f"Field '{field}' exceeds max length ({max_string_length})"
        cleaned[field] = val

    for field in optional:
        if field in data:
            val = data[field]
            if isinstance(val, str) and len(val) > max_string_length:
                return {}, f"Field '{field}' exceeds max length ({max_string_length})"
            cleaned[field] = val

    return cleaned, None