"""Structured logging with correlation IDs for request tracing.

Provides JSON-formatted logs with:
- Unique request ID per conversation turn
- User/session identification
- Timing information
- Error context
"""
import contextvars
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TextIO

logger = logging.getLogger(__name__)

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar('request_id', default=None)
session_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar('session_id', default=None)
user_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar('user_id', default=None)


@dataclass
class LogContext:
    """Context for structured logging."""
    request_id: str = ""
    session_id: str = ""
    user_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """JSON-formatted log output."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        request_id = request_id_var.get()
        if request_id:
            log_entry["request_id"] = request_id
        
        session_id = session_id_var.get()
        if session_id:
            log_entry["session_id"] = session_id
        
        user_id = user_id_var.get()
        if user_id:
            log_entry["user_id"] = user_id
        
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }
        
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'pathname', 'process', 'processName', 'relativeCreated',
                'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                'message', 'taskName',
            }:
                extra_fields[key] = value
        
        if extra_fields and self.include_extra:
            log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)


class StructuredLogger:
    """Logger with structured context support.
    
    Usage:
        log = get_structured_logger(__name__)
        
        with request_context(user_id="user123", session_id="session456"):
            log.info("Processing request", extra={"action": "chat", "tokens": 150})
    """
    
    def __init__(self, name: str, logger: logging.Logger | None = None):
        self._name = name
        self._logger = logger or logging.getLogger(name)
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        extra = kwargs.pop('extra', {})
        exc_info = kwargs.pop('exc_info', None)
        
        if kwargs:
            extra['context'] = kwargs
        
        self._logger.log(level, message, extra=extra, exc_info=exc_info)
    
    def debug(self, message: str, **kwargs) -> None:
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        self._log(logging.ERROR, message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        kwargs['exc_info'] = True
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        self._log(logging.CRITICAL, message, **kwargs)


class RequestContext:
    """Context manager for request tracing.
    
    Usage:
        with RequestContext() as ctx:
            logger.info("Processing")  # Automatically includes request_id
    """
    
    def __init__(
        self,
        request_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ):
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.session_id = session_id
        self.user_id = user_id
        self._tokens: list[contextvars.Token] = []
        self.MAX_TOKENS = 50
    
    def __enter__(self) -> "RequestContext":
        self._tokens.append(request_id_var.set(self.request_id))
        if self.session_id:
            self._tokens.append(session_id_var.set(self.session_id))
        if self.user_id:
            self._tokens.append(user_id_var.set(self.user_id))
        self._tokens = self._tokens[-self.MAX_TOKENS:]
        return self
    
    def __exit__(self, *args) -> None:
        # Each Token already carries the ContextVar it came from. The previous
        # implementation looked the var up by `token.var.old_value` — `.old_value`
        # belongs to the Token, not the ContextVar — so every lookup raised
        # AttributeError, was swallowed, and no variable was ever reset. Request
        # IDs then leaked from one request into the next, which makes correlation
        # IDs worse than having none.
        for token in reversed(self._tokens):
            try:
                token.var.reset(token)
            except (ValueError, RuntimeError) as e:
                # reset() rejects a token created in a different Context.
                logger.debug(f"Token reset failed for {token.var.name}: {e}")
        self._tokens = []


@dataclass
class TimingContext:
    """Context for timing operations."""
    operation: str
    logger: StructuredLogger | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    
    def __enter__(self) -> "TimingContext":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        if self.logger:
            self.logger.debug(
                f"{self.operation} completed",
                extra={
                    "operation": self.operation,
                    "duration_ms": round(duration_ms, 2),
                    **self.extra,
                }
            )


def setup_structured_logging(
    level: int = logging.INFO,
    output: TextIO | None = None,
    include_extra: bool = True,
) -> None:
    """Configure structured JSON logging.
    
    Args:
        level: Log level
        output: Output stream (default: stdout)
        include_extra: Include extra fields in output
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    handler = logging.StreamHandler(output or sys.stdout)
    handler.setFormatter(StructuredFormatter(include_extra=include_extra))
    root_logger.addHandler(handler)


def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)


def request_context(
    request_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> RequestContext:
    """Create a request context for tracing."""
    return RequestContext(request_id, session_id, user_id)


def get_request_id() -> str | None:
    """Get the current request ID."""
    return request_id_var.get()


def get_session_id() -> str | None:
    """Get the current session ID."""
    return session_id_var.get()


def get_user_id() -> str | None:
    """Get the current user ID."""
    return user_id_var.get()


def set_request_id(request_id: str) -> contextvars.Token:
    """Set the request ID in context."""
    return request_id_var.set(request_id)


def set_session_id(session_id: str) -> contextvars.Token:
    """Set the session ID in context."""
    return session_id_var.set(session_id)


def timing(operation: str, logger: StructuredLogger | None = None, **extra) -> TimingContext:
    """Create a timing context."""
    return TimingContext(operation, logger, extra)