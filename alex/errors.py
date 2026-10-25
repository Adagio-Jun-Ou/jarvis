"""
Errors that cross module boundaries.

Keep this small. Per-module error types live in the modules that raise them
(e.g. `constitution.ConstitutionalBlock`). The names here are the ones that
the SDK, app, and agent endpoint all have to recognize.
"""

from __future__ import annotations


class AlexError(Exception):
    """Base for every error surfaced across module boundaries."""

    code: str = "alex_error"
    # Canonical HTTP status for this error class. The app's global handler
    # maps an AlexError to this instead of blanket-500'ing everything, so a
    # Denied surfaces as 403 and a NotFound as 404 even when it bubbles up.
    http_status: int = 500

    def __init__(self, message: str = "", *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        """The JSON body the API returns for this error."""
        out: dict = {"error": self.code}
        if self.message:
            out["message"] = self.message
        if self.details:
            out["details"] = self.details
        return out


class Denied(AlexError):
    """A guardrail/ACL refused an action."""

    code = "denied"
    http_status = 403


class NotFound(AlexError):
    code = "not_found"
    http_status = 404


class BadRequest(AlexError):
    code = "bad_request"
    http_status = 400


class UpstreamError(AlexError):
    """An external service (model serving, vector search, etc.) failed."""

    code = "upstream_error"
    http_status = 502


class ConstitutionalBlock(AlexError):
    code = "constitutional_block"
    http_status = 403
