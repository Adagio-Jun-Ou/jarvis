"""React-based web interface for Jarvis.

Features:
- WebSocket real-time chat
- Conversation persistence (save/load past conversations)
- Multiple conversation management (new, switch, delete)
- Integration with memory subsystems
- Smart mode detection (chat vs agentic)
"""

import json
import logging
import os
import asyncio
import time
import uuid
import re
import hashlib
import hmac
import secrets
import threading
from collections import deque, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import WebSocket, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.responses import Response as FastAPIResponse
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from core.mode_detector import analyze_message, ExecutionMode
from core.helpers import get_response_content, safe_error_message, validate_request_fields

if TYPE_CHECKING:
    from core.agent import JarvisAgent

logger = logging.getLogger(__name__)

WEB_DIST = Path(__file__).parent.parent / "web" / "dist"
CONVERSATIONS_DIR = Path.home() / ".jarvis" / "conversations"


def _audit_log(request: Request, action: str, detail: str = ""):
    """Log a security-relevant action for audit trail."""
    client_ip = request.client.host if request.client else "unknown"
    msg = f"[AUDIT] {action}: ip={client_ip}"
    if detail:
        msg += f" {detail[:200]}"
    logger.warning(msg)


class ConversationManager:
    """Manages conversation persistence and switching."""
    
    def __init__(self, storage_dir: Path | None = None):
        self.storage_dir = storage_dir or CONVERSATIONS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.current_conversation_id: str | None = None
        self._lock = asyncio.Lock()
    
    async def _write_file(self, path: Path, content: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, path.write_text, content)
    
    async def _read_file(self, path: Path) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, path.read_text)
    
    async def _unlink_file(self, path: Path) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, path.unlink)
    
    def _generate_id(self) -> str:
        return f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _validate_conv_id(self, conv_id: str) -> bool:
        """Validate conversation ID to prevent path traversal."""
        if not conv_id or '..' in conv_id or '/' in conv_id or '\\' in conv_id:
            return False
        if not conv_id.startswith('conv_') or len(conv_id) > 64:
            return False
        return True

    async def create_conversation(self, title: str = "New Conversation") -> dict:
        """Create a new conversation."""
        async with self._lock:
            conv_id = self._generate_id()
            now = datetime.now().isoformat()
            
            conversation = {
                "id": conv_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
                "messages": [],
                "message_count": 0,
            }
            
            file_path = self.storage_dir / f"{conv_id}.json"
            await self._write_file(file_path, json.dumps(conversation, indent=2))
            
            self.current_conversation_id = conv_id
            return conversation
    
    async def save_conversation(
        self,
        conv_id: str,
        messages: list[dict],
        title: str | None = None,
    ) -> bool:
        """Save/update a conversation."""
        if not self._validate_conv_id(conv_id):
            logger.warning(f"Invalid conversation ID rejected: {conv_id!r}")
            return False
        async with self._lock:
            file_path = self.storage_dir / f"{conv_id}.json"
            
            if file_path.exists():
                try:
                    conversation = json.loads(await self._read_file(file_path))
                except Exception:
                    conversation = {}
            else:
                conversation = {
                    "id": conv_id,
                    "created_at": datetime.now().isoformat(),
                }
            
            conversation["messages"] = messages
            conversation["message_count"] = len(messages)
            conversation["updated_at"] = datetime.now().isoformat()
            
            if title:
                conversation["title"] = title
            elif not conversation.get("title"):
                if messages:
                    first_user_msg = next(
                        (m for m in messages if m.get("role") == "user"),
                        None
                    )
                    if first_user_msg:
                        content = first_user_msg.get("content", "")
                        conversation["title"] = content[:50] + ("..." if len(content) > 50 else "")
                    else:
                        conversation["title"] = "Conversation"
                else:
                    conversation["title"] = "New Conversation"
            
            await self._write_file(file_path, json.dumps(conversation, indent=2))
            return True
    
    async def load_conversation(self, conv_id: str) -> dict | None:
        """Load a conversation by ID."""
        if not self._validate_conv_id(conv_id):
            logger.warning(f"Invalid conversation ID rejected: {conv_id!r}")
            return None
        file_path = self.storage_dir / f"{conv_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            return json.loads(await self._read_file(file_path))
        except Exception as e:
            logger.error(f"Failed to load conversation {conv_id}: {e}")
            return None
    
    async def list_conversations(self, limit: int = 50) -> list[dict]:
        """List all conversations, sorted by update time (newest first)."""
        conversations = []
        
        for file_path in self.storage_dir.glob("conv_*.json"):
            try:
                data = json.loads(await self._read_file(file_path))
                conversations.append({
                    "id": data.get("id", file_path.stem),
                    "title": data.get("title", "Untitled"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "message_count": data.get("message_count", 0),
                })
            except Exception:
                continue
        
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return conversations[:limit]
    
    async def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation."""
        if not self._validate_conv_id(conv_id):
            logger.warning(f"Invalid conversation ID rejected: {conv_id!r}")
            return False
        async with self._lock:
            file_path = self.storage_dir / f"{conv_id}.json"
            
            if file_path.exists():
                await self._unlink_file(file_path)
                if self.current_conversation_id == conv_id:
                    self.current_conversation_id = None
                return True
            return False
    
    async def search_conversations(self, query: str, limit: int = 20) -> list[dict]:
        """Search conversations by content."""
        results = []
        query_lower = query.lower()
        
        for file_path in self.storage_dir.glob("conv_*.json"):
            try:
                data = json.loads(await self._read_file(file_path))
                
                if query_lower in data.get("title", "").lower():
                    results.append({
                        "id": data.get("id"),
                        "title": data.get("title"),
                        "updated_at": data.get("updated_at"),
                        "message_count": data.get("message_count", 0),
                        "match_type": "title",
                    })
                    continue
                
                for msg in data.get("messages", []):
                    if query_lower in msg.get("content", "").lower():
                        results.append({
                            "id": data.get("id"),
                            "title": data.get("title"),
                            "updated_at": data.get("updated_at"),
                            "message_count": data.get("message_count", 0),
                            "match_type": "content",
                            "snippet": msg.get("content", "")[:100],
                        })
                        break
            except Exception:
                continue
        
        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return results[:limit]


class WebInterface:
    """React-based web interface for Jarvis."""

    def __init__(self, agent: "JarvisAgent", port: int = 8080, require_auth: bool = False, force_https: bool = True, rate_limit_per_minute: int = 2000):
        self.agent = agent
        self.port = port
        self.force_https = force_https
        self.connections: set[WebSocket] = set()
        self.conversation_manager = ConversationManager()
        self._current_messages: list[dict] = []
        self._ws_rate_limits: dict[str, deque] = {}  # client_id -> deque of timestamps
        self._ws_rate_limit_lock = threading.Lock()  # Lock for rate limit operations
        self._ws_rate_limit_maxlen = 3100  # Max timestamps per client (slightly above max_per_minute)
        self._ws_sessions: dict[str, str] = {}  # websocket -> session_id
        self._api_activity_connections: set[WebSocket] = set()  # API activity subscribers
        self._broadcast_queue: asyncio.Queue | None = None  # Lazy init in _setup_routes
        self._mobile_connections: set[WebSocket] = set()  # Mobile PWA connections
        self._mobile_http_clients: int = 0  # Active mobile HTTP clients
        self._mobile_http_last_activity: dict[str, float] = {}  # device_id -> timestamp
        self._ui_state: dict = {
            "active_tab": "chat",
            "tool_calls": [],
            "thinking": False,
            "last_response": None,
            "system_status": {},
        }
        self._ui_update_handlers: list = []  # Callbacks for UI updates
        
        self._background_tasks: set = set()
        # Cross-group route handlers (see _setup_routes split): route
        # bodies in one group call endpoint functions defined in another;
        # they were scope siblings before the split and are registered
        # per group at setup time.
        self._route_handlers: dict = {}
        self._terminal_rate_limits: dict[str, deque] = {}
        self._approval_secret = secrets.token_hex(32)

        # Status endpoint cache
        self._status_cache: dict = {}
        self._status_cache_time: float = 0
        self._status_cache_lock = threading.Lock()

        # Initialize security middleware
        self._consecutive_agent_errors = 0
        from core.auth_middleware import SecurityMiddleware, SecurityConfig
        # rate_limit_per_minute is now actually enforced by the middleware as
        # a per-IP sliding window (loopback-exempt). The web dashboard polls
        # aggressively (flow visualization at 200ms), so the default stays
        # generous; operators can lower it via config.yaml's
        # security.rate_limit_requests_per_minute, which run.py plumbs through.
        self.security = SecurityMiddleware(SecurityConfig(
            require_auth=require_auth,
            rate_limit_requests_per_minute=rate_limit_per_minute,
        ))
        self._require_auth = require_auth

        import uvicorn
        from fastapi import FastAPI

        self.app = FastAPI(title="Jarvis", docs_url=None, redoc_url=None, openapi_url=None)
        self._setup_routes()
        self._setup_middleware()
        self._setup_oracle_routes()
        self._setup_streams_routes()
        self._setup_mobile_routes_safe()
        
        self._setup_ui_broadcaster()

        ssl_dir = Path(__file__).parent.parent / "ssl"
        ssl_keyfile = ssl_dir / "key.pem"
        ssl_certfile = ssl_dir / "cert.pem"

        config_kwargs = {
            "host": "0.0.0.0",
            "port": self.port,
            "log_level": "warning",
        }

        # Use HTTPS if force_https is True and certificates exist
        if self.force_https and ssl_keyfile.exists() and ssl_certfile.exists():
            config_kwargs["ssl_keyfile"] = str(ssl_keyfile)
            config_kwargs["ssl_certfile"] = str(ssl_certfile)
            logger.info(f"HTTPS enabled on port {self.port}")

        config = uvicorn.Config(self.app, **config_kwargs)
        self.server = uvicorn.Server(config)
    
    def _setup_ui_broadcaster(self):
        """Setup UI state broadcaster for live updates to web and mobile."""
        self._ui_state = {
            "active_tab": "chat",
            "tool_calls": [],
            "thinking": False,
            "last_response": None,
            "last_user_input": None,
            "system_status": {},
            "connected_clients": 0,
        }
    
    async def _broadcast_ui_update(self, event_type: str, data: dict):
        """Broadcast UI update to all connected clients (web + mobile)."""
        message = {"type": event_type, "data": data, "timestamp": time.time()}
        message_json = json.dumps(message)
        
        disconnected = set()
        
        for ws in self.connections:
            try:
                await ws.send_text(message_json)
            except Exception:
                disconnected.add(ws)
        
        for ws in self._mobile_connections:
            try:
                await ws.send_text(message_json)
            except Exception:
                disconnected.add(ws)
        
        self.connections -= disconnected
        self._mobile_connections -= disconnected
    
    async def _update_ui_state(self, key: str, value):
        """Update UI state and broadcast to all clients."""
        self._ui_state[key] = value
        await self._broadcast_ui_update("state_update", {key: value})
    
    async def _emit_thinking_start(self):
        """Emit thinking start event."""
        self._ui_state["thinking"] = True
        await self._broadcast_ui_update("thinking", {"status": "start"})
    
    async def _emit_thinking_end(self):
        """Emit thinking end event."""
        self._ui_state["thinking"] = False
        await self._broadcast_ui_update("thinking", {"status": "end"})
    
    async def _emit_tool_call(self, tool_name: str, tool_id: str, status: str, result: str = None):
        """Emit tool call event."""
        tool_event = {
            "id": tool_id,
            "name": tool_name,
            "status": status,
            "result": result,
            "timestamp": time.time(),
        }
        
        if status == "start":
            self._ui_state["tool_calls"].append(tool_event)
            if len(self._ui_state["tool_calls"]) > 100:
                self._ui_state["tool_calls"] = self._ui_state["tool_calls"][-100:]
        elif status in ["complete", "error"]:
            for tc in self._ui_state["tool_calls"]:
                if tc["id"] == tool_id:
                    tc["status"] = status
                    tc["result"] = result
                    break
        
        await self._broadcast_ui_update("tool_call", tool_event)
    
    async def _emit_token(self, token: str):
        """Emit streaming token."""
        await self._broadcast_ui_update("token", {"content": token})
    
    async def _emit_response_complete(self, response: str, thinking: str = None):
        """Emit complete response."""
        self._ui_state["last_response"] = response
        self._ui_state["thinking"] = False
        await self._broadcast_ui_update("response_complete", {
            "content": response,
            "thinking": thinking,
        })
    
    async def _emit_message(self, role: str, content: str):
        """Emit user or assistant message."""
        await self._broadcast_ui_update("message", {
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
    
    def _setup_middleware(self):
        """Setup security middleware for the FastAPI app."""
        from starlette.requests import Request as StarletteRequest
        from starlette.responses import JSONResponse
        
        security_headers = self.security.get_security_headers()
        
        _allowed_cors_origins = set()
        cors_env = os.environ.get("JARVIS_CORS_ORIGINS", "")
        if cors_env:
            for origin in cors_env.split(","):
                origin = origin.strip()
                if origin:
                    _allowed_cors_origins.add(origin)
        # Published on self so csrf_middleware can see it. It previously read
        # getattr(self, '_allowed_cors_origins', set()) while this was only a
        # local, so the default always won and JARVIS_CORS_ORIGINS was silently
        # ignored by the CSRF gate.
        self._allowed_cors_origins = _allowed_cors_origins

        def _is_same_origin(origin: str, host: str) -> bool:
            """True when an Origin header refers to the serving host itself."""
            if not origin or not host:
                return False
            try:
                from urllib.parse import urlparse
                return urlparse(origin).netloc.lower() == host.lower()
            except Exception:
                return False

        @self.app.middleware("http")
        async def cors_middleware(request: StarletteRequest, call_next):
            # Cross-origin access is deny-by-default. This used to fail OPEN:
            # an unconfigured allow-list (the shipped default) meant "allow
            # every origin", so any site could read authenticated GET responses
            # — and GETs are not covered by csrf_middleware.
            if request.method == "OPTIONS":
                origin = request.headers.get("origin", "")
                host = request.headers.get("host", "")
                if origin and not (
                    origin in _allowed_cors_origins or _is_same_origin(origin, host)
                ):
                    return JSONResponse(
                        {"error": "CORS origin not allowed"},
                        status_code=403,
                        headers=security_headers,
                    )
                response = JSONResponse({}, status_code=200)
                if origin:
                    response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key, Authorization"
                response.headers["Access-Control-Max-Age"] = "86400"
                for header, value in security_headers.items():
                    response.headers[header] = value
                return response

            response = await call_next(request)
            origin = request.headers.get("origin", "")
            host = request.headers.get("host", "")
            # Only echo an allow-origin for an explicitly permitted origin. Same
            # origin needs no header at all, so an empty allow-list means the
            # UI keeps working while third-party sites are refused.
            if origin and (origin in _allowed_cors_origins or _is_same_origin(origin, host)):
                response.headers["Access-Control-Allow-Origin"] = origin
            return response
        
        @self.app.middleware("http")
        async def security_middleware(request: StarletteRequest, call_next):
            # Add security headers to all responses
            response = await call_next(request)
            for header, value in security_headers.items():
                response.headers[header] = value
            return response
        
        @self.app.middleware("http")
        async def csrf_middleware(request: StarletteRequest, call_next):
            if request.method in ("POST", "PUT", "PATCH", "DELETE"):
                origin = request.headers.get("origin", "")
                host = request.headers.get("host", "")
                referer = request.headers.get("referer", "")
                
                allowed_netlocs = set()
                if host:
                    allowed_netlocs.add(host.lower())
                # Allow-list entries are full origins ("https://host[:port]");
                # normalize to netloc so they actually match (they never did
                # before: netloc was compared against the full entry).
                for allowed in getattr(self, '_allowed_cors_origins', set()):
                    try:
                        from urllib.parse import urlparse
                        netloc = urlparse(allowed).netloc.lower()
                        allowed_netlocs.add(netloc if netloc else str(allowed).lower())
                    except Exception:
                        continue

                if origin:
                    try:
                        from urllib.parse import urlparse
                        origin_host = urlparse(origin).netloc.lower()
                        # Exact netloc equality only. The previous test was
                        #   A or (B and C)
                        # (`and` binds tighter than `or`) where B/C were a
                        # suffix+prefix pair, so an Origin merely *starting with*
                        # the host name and *ending with* ":<port>" matched:
                        # allowed "localhost:8080" accepted the attacker origin
                        # "localhost.evil.com:8080", bypassing CSRF entirely for
                        # POST/PUT/PATCH/DELETE.
                        origin_matches = origin_host in allowed_netlocs
                        if not origin_matches and origin_host != host.lower():
                            return JSONResponse(
                                {"error": "CSRF check failed", "message": "Origin mismatch"},
                                status_code=403,
                                headers=security_headers,
                            )
                    except Exception as e:
                        logger.warning(f"CSRF origin check error: {e}")
                        return JSONResponse(
                            {"error": "CSRF check failed", "message": "Invalid Origin header"},
                            status_code=403,
                            headers=security_headers,
                        )
                elif referer and host:
                    try:
                        from urllib.parse import urlparse
                        referer_host = urlparse(referer).netloc
                        if referer_host and referer_host != host:
                            return JSONResponse(
                                {"error": "CSRF check failed", "message": "Referer mismatch"},
                                status_code=403,
                                headers=security_headers,
                            )
                    except Exception as e:
                        logger.warning(f"CSRF referer check error: {e}")
                        return JSONResponse(
                            {"error": "CSRF check failed", "message": "Invalid Referer header"},
                            status_code=403,
                            headers=security_headers,
                        )
                elif not origin and not referer:
                    if request.headers.get("upgrade", "").lower() != "websocket":
                        logger.warning(f"CSRF: Missing Origin and Referer on {request.method} {request.url.path}")
                        return JSONResponse(
                            {"error": "CSRF check failed", "message": "Missing Origin and Referer headers"},
                            status_code=403,
                            headers=security_headers,
                        )
            return await call_next(request)
        
        @self.app.middleware("http")
        async def content_type_middleware(request: StarletteRequest, call_next):
            if request.method in ("POST", "PUT", "PATCH"):
                ct = request.headers.get("content-type", "")
                if ct and not any(ct.startswith(t) for t in (
                    "application/json", "application/x-www-form-urlencoded",
                    "multipart/form-data", "text/plain",
                )):
                    return JSONResponse(
                        {"error": "Unsupported Media Type", "message": "Content-Type must be application/json or form data"},
                        status_code=415,
                        headers=security_headers,
                    )
            return await call_next(request)
        
        @self.app.middleware("http")
        async def request_size_middleware(request: StarletteRequest, call_next):
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > 10 * 1024 * 1024:
                        return JSONResponse(
                            {"error": "Payload Too Large", "message": "Request body exceeds 10MB limit"},
                            status_code=413,
                            headers=security_headers,
                        )
                except ValueError:
                    return JSONResponse(
                        {"error": "Bad Request", "message": "Invalid Content-Length header"},
                        status_code=400,
                        headers=security_headers,
                    )
            return await call_next(request)
        
        @self.app.middleware("http")
        async def auth_middleware(request: StarletteRequest, call_next):
            # Skip auth for health check and static files
            if request.url.path in ["/health", "/", "/favicon.ico"]:
                return await call_next(request)
            
            # Skip auth for static assets
            if request.url.path.startswith("/assets/"):
                return await call_next(request)
            
            # Check authentication if required
            if self._require_auth:
                api_key = (
                    request.headers.get("X-API-Key") or
                    request.headers.get("Authorization", "").replace("Bearer ", "")
                )
                
                is_valid, result = self.security.api_key_validator.validate(api_key)
                if not is_valid:
                    return JSONResponse(
                        {"error": "Authentication required", "message": result},
                        status_code=401,
                        headers=security_headers,
                    )
            
            return await call_next(request)
        
        @self.app.middleware("http")
        async def rate_limit_middleware(request: StarletteRequest, call_next):
            try:
                from core.rate_limiter import get_rate_limiter, RateLimit
                limiter = get_rate_limiter()
                client_id = request.client.host if request.client else "unknown"
                endpoint = request.url.path
                allowed, retry_after, info = limiter.check_rate_limit(client_id, endpoint)
                if not allowed:
                    return JSONResponse(
                        {"error": "Rate limited", "message": "Too many requests", **info},
                        status_code=429,
                        headers={**security_headers, "Retry-After": str(int(retry_after) + 1)},
                    )
            except Exception as e:
                logger.warning(f"Rate limiter error — allowing request (degraded): {e}")
                response = await call_next(request)
                return response

            # try/finally: release() was a bare statement after call_next, so any
            # exception escaping a route skipped it and leaked the concurrency
            # slot acquired by check_rate_limit(). Slots only ever accumulated,
            # so a client hitting enough 500s was locked out permanently.
            try:
                return await call_next(request)
            finally:
                limiter.release(client_id)

        # Registered last, so it is the OUTERMOST middleware and every log line
        # emitted while handling a request — including from the middleware above
        # — carries the correlation id. Honours an inbound X-Request-ID so a
        # trace can span a proxy, and echoes it back so a user-reported failure
        # can be located in the logs.
        @self.app.middleware("http")
        async def request_id_middleware(request: StarletteRequest, call_next):
            try:
                from core.structured_logging import RequestContext
            except ImportError:
                return await call_next(request)

            incoming = (request.headers.get("X-Request-ID") or "").strip()
            # Bound and sanitise: this value reaches log lines and a response header.
            if not incoming or len(incoming) > 64 or not incoming.replace("-", "").isalnum():
                incoming = None

            with RequestContext(request_id=incoming) as ctx:
                response = await call_next(request)
                response.headers["X-Request-ID"] = ctx.request_id
                return response

        @self.app.exception_handler(401)
        async def unauthorized_handler(request, exc):
            return JSONResponse(
                {"error": "Unauthorized", "message": "Valid API key required"},
                status_code=401,
                headers=security_headers
            )
        
        @self.app.exception_handler(403)
        async def forbidden_handler(request, exc):
            return JSONResponse(
                {"error": "Forbidden", "message": "Access denied"},
                status_code=403,
                headers=security_headers
            )
        
        @self.app.exception_handler(500)
        async def internal_error_handler(request, exc):
            import traceback
            logger.error(f"Internal server error: {exc}", exc_info=True)
            return JSONResponse(
                {"error": "Internal Server Error", "message": "An unexpected error occurred"},
                status_code=500,
                headers=security_headers
            )
        
        @self.app.exception_handler(429)
        async def rate_limit_handler(request, exc):
            return JSONResponse(
                {"error": "Rate Limited", "message": "Too many requests"},
                status_code=429,
                headers=security_headers
            )

    def _check_ws_rate_limit(self, client_id: str, max_per_minute: int = 3000) -> bool:
        """Check WebSocket rate limit (per-client, per-minute).
        
        Returns True if allowed, False if rate limited.
        """
        now = time.time()
        
        with self._ws_rate_limit_lock:
            if client_id not in self._ws_rate_limits:
                self._ws_rate_limits[client_id] = deque(maxlen=self._ws_rate_limit_maxlen)
            
            timestamps = self._ws_rate_limits[client_id]
            
            # Remove timestamps older than 1 minute
            while timestamps and now - timestamps[0] > 60:
                timestamps.popleft()
            
            if len(timestamps) >= max_per_minute:
                return False
            
            timestamps.append(now)
            return True

    def _cleanup_rate_limits(self):
        """Clean up old rate limit entries to prevent memory leak."""
        now = time.time()
        window_start = now - 120  # Keep 2 minutes of data

        with self._ws_rate_limit_lock:
            for client_id in list(self._ws_rate_limits.keys()):
                timestamps = self._ws_rate_limits[client_id]
                while timestamps and timestamps[0] < window_start:
                    timestamps.popleft()
                if not timestamps:
                    del self._ws_rate_limits[client_id]

    def _ws_origin_allowed(self, websocket: WebSocket) -> bool:
        """Cross-site WebSocket hijacking guard.

        Browsers ALWAYS send Origin on a WebSocket handshake and do NOT
        enforce same-origin on WS connections, so without this check any
        website open in the user's browser could connect to /ws and chat
        as the user (the HTTP CSRF middleware only covers POST/PUT/PATCH/
        DELETE, and a WS handshake is a GET). Non-browser clients
        (mobile app, scripts) send no Origin header and are allowed.

        Policy — identical semantics to the HTTP CSRF origin check, so
        there is one policy, not two:
          - Origin absent  -> allowed (not a browser)
          - Origin present -> netloc must equal the Host header or be in
            the JARVIS_CORS_ORIGINS allow-list (exact equality; the
            suffix/prefix weakness that admitted localhost.evil.com is
            not repeated here).
        """
        origin = websocket.headers.get("origin", "")
        if not origin:
            return True

        host = websocket.headers.get("host", "")
        try:
            from urllib.parse import urlparse
            origin_netloc = urlparse(origin).netloc.lower()
        except Exception:
            return False

        if not origin_netloc:
            return False
        if host and origin_netloc == host.lower():
            return True
        # Allow-list entries are full origins ("https://host[:port]") —
        # compare netloc-to-netloc, not raw string to netloc.
        from urllib.parse import urlparse as _urlparse
        for allowed in getattr(self, "_allowed_cors_origins", set()):
            try:
                if _urlparse(allowed).netloc.lower() == origin_netloc:
                    return True
                # An entry with no scheme parses to an empty netloc; fall
                # back to comparing the entry itself (bare "host:port").
                if not _urlparse(allowed).netloc and allowed.lower() == origin_netloc:
                    return True
            except Exception:
                continue
        return False

    async def _ws_keepalive_loop(self, websocket: WebSocket, ws_mgr) -> None:
        """Background task that sends periodic pings and detects stale connections."""
        try:
            while True:
                await asyncio.sleep(ws_mgr.PING_INTERVAL)
                try:
                    ws_mgr.record_ping_sent(websocket)
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
                stale = ws_mgr.get_stale_connections()
                for ws_id, info in stale:
                    if info.ws is websocket:
                        try:
                            await websocket.close(code=4000, reason="Keepalive timeout")
                        except Exception:
                            pass
                        return
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Keepalive loop ended")

    def _setup_routes(self):
        from fastapi.responses import StreamingResponse

        self._broadcast_queue = asyncio.Queue(maxsize=100)

        def _on_llm_call(call_record: dict):
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    loop.call_soon_threadsafe(
                        lambda: self._broadcast_queue.put_nowait(call_record)
                    )
            except Exception as e:
                logger.debug(f"Failed to broadcast LLM call: {e}")

        self.agent.llm.add_call_callback(_on_llm_call)

        async def _broadcast_task():
            cleanup_counter = 0
            while True:
                try:
                    call_record = await self._broadcast_queue.get()
                    dead = set()
                    for ws in list(self._api_activity_connections):
                        try:
                            await ws.send_json({
                                "type": "api_call",
                                "call": call_record,
                            })
                        except Exception as e:
                            logger.debug(f"WebSocket send failed: {e}")
                            dead.add(ws)
                    for ws in dead:
                        try:
                            await ws.close()
                        except Exception as e:
                            logger.debug(f"Failed to close WebSocket: {e}")
                        self._api_activity_connections.discard(ws)
                    
                    cleanup_counter += 1
                    if cleanup_counter % 100 == 0:
                        self._cleanup_rate_limits()
                        
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Broadcast task error: {e}")
                    await asyncio.sleep(0.5)

        task = asyncio.create_task(_broadcast_task())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        assets_dir = WEB_DIST / "assets"
        if assets_dir.exists():
            self.app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # Serve PWA icons and manifest
        icons_dir = WEB_DIST / "icons"
        if icons_dir.exists():
            self.app.mount("/icons", StaticFiles(directory=str(icons_dir)), name="icons")
        # Route groups, split out of what grew into a multi-thousand-line
        # method. Each group registers exactly the same routes, in the
        # same order, with zero behavioural change.
        self._setup_frontend_routes()
        self._setup_status_routes()
        self._setup_bio_predictive_routes()
        self._setup_consciousness_routes()
        self._setup_butler_emotion_routes()
        self._setup_sentiment_oracle_routes()
        self._setup_refine_evolution_routes()
        self._setup_insights_music_routes()
        self._setup_agent_state_routes()
        self._setup_screen_skills_terminal_routes()
        self._setup_chat_command_routes()
        self._setup_ops_health_routes()
        self._setup_memory_goals_routes()
        self._setup_models_permissions_routes()
        self._setup_n8n_embed_routes()
        self._setup_websocket_routes()
    def _setup_frontend_routes(self):
        @self.app.get("/manifest.json")
        async def serve_manifest():
            manifest_path = WEB_DIST / "manifest.json"
            if manifest_path.exists():
                return FileResponse(str(manifest_path), media_type="application/json")
            # Return default manifest
            return {
                "name": "Jarvis AI Assistant",
                "short_name": "Jarvis",
                "start_url": "/mobile",
                "display": "standalone",
                "background_color": "#0a0a0a",
                "theme_color": "#1a1a2e",
            }
        
        @self.app.get("/sw-mobile.js")
        async def serve_sw():
            sw_path = WEB_DIST / "sw-mobile.js"
            if sw_path.exists():
                return FileResponse(str(sw_path), media_type="application/javascript")
            return {"error": "not found"}, 404

        @self.app.get("/", response_class=HTMLResponse)
        async def home():
            index_path = WEB_DIST / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return HTMLResponse(content="<h1>Build the React app first: cd web && npm run build</h1>")

        @self.app.get("/health")
        async def health():
            return {"status": "healthy", "service": "jarvis"}

        @self.app.get("/api/conversations")
        async def list_conversations():
            """List all saved conversations."""
            try:
                conversations = await self.conversation_manager.list_conversations()
                return {"success": True, "conversations": conversations, "count": len(conversations)}
            except Exception as e:
                logger.error(f"List conversations error: {e}")
                return {"success": False, "error": safe_error_message(e), "conversations": []}

        @self.app.post("/api/conversations/new")
        async def create_new_conversation(request: Request):
            """Create a new conversation."""
            try:
                data = {}
                if request.headers.get("content-type") == "application/json":
                    data = await request.json()
                    data, validation_error = validate_request_fields(
                        data, optional=["title"], max_string_length=256
                    )
                    if validation_error:
                        return {"success": False, "error": validation_error}
                
                title = data.get("title", "New Conversation")
                conversation = await self.conversation_manager.create_conversation(title)
                self._current_messages = []
                
                if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear'):
                    self.agent.memory.clear()
                
                return {"success": True, "conversation": conversation}
            except Exception as e:
                logger.error(f"Create conversation error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/security/status")
        async def security_status():
            """Get security configuration status."""
            return {
                "success": True,
                "security": {
                    "auth_required": self._require_auth,
                    "rate_limiting": True,
                    "input_sanitization": True,
                    "session_timeout_minutes": self.security.config.session_timeout_minutes,
                    "max_failed_attempts": self.security.config.max_failed_attempts,
                    "active_sessions": self.security.session_manager.get_active_count(),
                    "security_headers": list(self.security.get_security_headers().keys()),
                }
            }

        @self.app.get("/api/conversations/{conv_id}")
        async def get_conversation(conv_id: str):
            """Load a specific conversation."""
            if not conv_id or '..' in conv_id or '/' in conv_id or '\\' in conv_id:
                return {"success": False, "error": "Invalid conversation ID"}
            
            if not conv_id.startswith('conv_') or len(conv_id) > 64:
                return {"success": False, "error": "Invalid conversation ID format"}
            
            try:
                conversation = await self.conversation_manager.load_conversation(conv_id)
                
                if conversation is None:
                    return {"success": False, "error": "Conversation not found"}
                
                self.conversation_manager.current_conversation_id = conv_id
                self._current_messages = conversation.get("messages", [])
                
                if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear'):
                    self.agent.memory.clear()
                    
                    for msg in self._current_messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "user":
                            self.agent.memory.add_user_message(content)
                        elif role == "assistant":
                            self.agent.memory.add_assistant_message(content)
                
                return {"success": True, "conversation": conversation}
            except Exception as e:
                logger.error(f"Load conversation error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/conversations/{conv_id}/save")
        async def save_conversation_endpoint(conv_id: str, request: Request):
            """Save current conversation state."""
            try:
                data = await request.json() if request.headers.get("content-type") == "application/json" else {}
                if data:
                    data, validation_error = validate_request_fields(
                        data, optional=["messages", "title"], max_string_length=5000
                    )
                    if validation_error:
                        return {"success": False, "error": validation_error}
                messages = data.get("messages", self._current_messages)
                title = data.get("title")
                
                success = await self.conversation_manager.save_conversation(conv_id, messages, title)
                
                if success:
                    self._current_messages = messages
                    return {"success": True}
                return {"success": False, "error": "Failed to save"}
            except Exception as e:
                logger.error(f"Save conversation error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.delete("/api/conversations/{conv_id}")
        async def delete_conversation(conv_id: str):
            """Delete a conversation."""
            try:
                success = await self.conversation_manager.delete_conversation(conv_id)
                
                if success:
                    if self.conversation_manager.current_conversation_id == conv_id:
                        self._current_messages = []
                        self.conversation_manager.current_conversation_id = None
                    
                    return {"success": True}
                return {"success": False, "error": "Conversation not found"}
            except Exception as e:
                logger.error(f"Delete conversation error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/conversations/search")
        async def search_conversations(query: str = ""):
            """Search conversations by content."""
            try:
                if not query:
                    return {"success": True, "results": [], "count": 0}
                
                results = await self.conversation_manager.search_conversations(query)
                return {"success": True, "results": results, "count": len(results)}
            except Exception as e:
                logger.error(f"Search conversations error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/conversations/current")
        async def get_current_conversation():
            """Get current active conversation info."""
            try:
                conv_id = self.conversation_manager.current_conversation_id
                message_count = len(self._current_messages)
                
                if conv_id:
                    conversation = await self.conversation_manager.load_conversation(conv_id)
                    if conversation:
                        return {
                            "success": True,
                            "current_id": conv_id,
                            "title": conversation.get("title", "Untitled"),
                            "message_count": message_count,
                        }
                
                return {
                    "success": True,
                    "current_id": None,
                    "title": None,
                    "message_count": message_count,
                }
            except Exception as e:
                logger.error(f"Get current conversation error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/history")
        async def get_history():
            try:
                messages = []
                if hasattr(self.agent, 'memory'):
                    raw_messages = self.agent.memory.get_messages_for_llm()
                    for i, msg in enumerate(raw_messages[-50:]):
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')
                        if isinstance(content, list):
                            content = ' '.join(str(c) for c in content)
                        messages.append({
                            "id": f"msg-{i}",
                            "role": role,
                            "content": content[:500] + ('...' if len(content) > 500 else ''),
                        })
                return {"messages": messages, "count": len(messages)}
            except Exception as e:
                return {"messages": [], "error": safe_error_message(e)}

        @self.app.post("/api/clear")
        async def clear_history():
            try:
                if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear'):
                    self.agent.memory.clear()
                return {"success": True, "message": "Memory cleared"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/tools")
        async def get_tools():
            tools_list = []
            for name, tool in sorted(self.agent.tools.items()):
                tool_info = {
                    "name": name,
                    "description": getattr(tool, 'description', '')[:200] if hasattr(tool, 'description') else '',
                    "requires_permission": getattr(tool, 'requires_permission', False),
                }
                params = getattr(tool, 'parameters', None)
                if params:
                    tool_info["parameters"] = params
                tools_list.append(tool_info)
            return {"tools": tools_list, "count": len(tools_list)}

        @self.app.get("/api/dashboard")
        async def get_dashboard():
            """Combined dashboard endpoint - returns all dashboard data in one call."""
            result = {"success": True, "timestamp": time.time()}
            
            status = await self._route_handlers["get_status"]()
            result["status"] = status
            
            try:
                from core.dual_model import get_dual_model_orchestrator
                orchestrator = get_dual_model_orchestrator()
                if orchestrator:
                    stats = orchestrator.get_stats()
                    result["dual_model"] = {
                        "primary_requests": stats.get("primary_requests", 0),
                        "backend_requests": stats.get("backend_requests", 0),
                        "audio_requests": stats.get("audio_requests", 0),
                        "parallel_requests": stats.get("parallel_requests", 0),
                        "collaborative_requests": stats.get("collaborative_requests", 0),
                        "sequential_requests": stats.get("sequential_requests", 0),
                        "total_requests": stats.get("total_requests", 0),
                        "avg_primary_latency_ms": stats.get("avg_primary_latency_ms", 0),
                        "avg_backend_latency_ms": stats.get("avg_backend_latency_ms", 0),
                        "errors": stats.get("errors", 0),
                        "primary_connected": stats.get("primary_connected", True),
                        "backend_connected": stats.get("backend_connected", True),
                        "audio_connected": stats.get("audio_connected", True),
                        "primary_requests_per_min": stats.get("primary_requests_per_min", 0),
                        "backend_requests_per_min": stats.get("backend_requests_per_min", 0),
                        "audio_requests_per_min": stats.get("audio_requests_per_min", 0),
                        "total_request_per_min": stats.get("total_requests_per_min", 0),
                    }
            except Exception as e:
                logger.debug(f"Failed to get dual model status: {e}")
            
            try:
                from core.daydream_engine import get_daydream_engine
                engine = get_daydream_engine()
                if engine:
                    result["daydream"] = engine.get_status()
            except Exception as e:
                logger.debug(f"Failed to get daydream status: {e}")
            
            try:
                from core.erdos_engine import get_erdos_engine
                erdos = get_erdos_engine()
                if erdos:
                    result["erdos"] = erdos.get_status()
            except Exception as e:
                logger.debug(f"Failed to get erdos status: {e}")
            
            try:
                from core.evolution_orchestrator import get_evolution_orchestrator
                evo = get_evolution_orchestrator()
                if evo:
                    result["evolution"] = evo.get_status()
            except Exception as e:
                logger.debug(f"Failed to get evolution status: {e}")
            
            try:
                from tools.refine import get_refine_tool
                refine = get_refine_tool()
                if refine:
                    result["refine"] = refine.get_status()
            except Exception as e:
                logger.debug(f"Failed to get refine status: {e}")
            
            try:
                from core.memory import get_memory
                from core.vector_memory import get_vector_memory
                mem = get_memory()
                result["memory"] = {
                    "conversation_count": len(mem.conversations) if hasattr(mem, 'conversations') else 0,
                    "vector_enabled": False,
                }
                try:
                    vm = get_vector_memory()
                    if vm:
                        result["memory"]["vector_enabled"] = True
                except Exception as e:
                    logger.debug(f"Failed to get vector memory: {e}")
            except Exception as e:
                logger.debug(f"Failed to get memory stats: {e}")
            
            try:
                from core.models_coordinator import get_models_coordinator
                coord = get_models_coordinator()
                if coord:
                    result["models_coordinator"] = coord.get_status()
            except Exception as e:
                logger.debug(f"Failed to get models coordinator: {e}")
            
            try:
                from core.knowledge_graph import get_knowledge_graph
                kg = get_knowledge_graph()
                if kg:
                    kg_stats = kg.get_graph_statistics()
                    result["knowledge_graph"] = {
                        "nodes": kg_stats.get("nodes", 0),
                        "edges": kg_stats.get("edges", 0),
                        "communities": kg_stats.get("communities", 0),
                    }
            except Exception as e:
                logger.debug(f"Failed to get knowledge graph: {e}")
            
            try:
                subsystems_json = await self._route_handlers["get_all_subsystems"]()
                result["subsystems"] = subsystems_json.get("subsystems", {})
            except Exception as e:
                logger.debug(f"Failed to get subsystems: {e}")
            
            try:
                result["tools"] = await get_tools()
            except Exception as e:
                logger.debug(f"Failed to get tools: {e}")
            
            try:
                result["hardware"] = await self._route_handlers["get_hardware_status"]()
            except Exception as e:
                logger.debug(f"Failed to get hardware status: {e}")
            
            try:
                result["freetime"] = await self._route_handlers["get_freetime_status"]()
            except Exception as e:
                logger.debug(f"Failed to get freetime status: {e}")
            
            try:
                result["jarvis_state"] = await self._route_handlers["get_jarvis_state"]()
            except Exception as e:
                logger.debug(f"Failed to get jarvis state: {e}")
            
            try:
                result["recent_thoughts"] = await self._route_handlers["get_recent_thoughts"]()
            except Exception as e:
                logger.debug(f"Failed to get recent thoughts: {e}")
            
            try:
                result["ui_state"] = {
                    "thinking": self._ui_state.get("thinking", False),
                    "tool_calls": self._ui_state.get("tool_calls", []),
                    "last_response": self._ui_state.get("last_response"),
                }
            except Exception as e:
                logger.debug(f"Failed to get UI state: {e}")
            
            try:
                from core.proactive import get_proactive_engine
                pe = get_proactive_engine()
                if pe:
                    result["proactive"] = pe.get_status()
            except Exception as e:
                logger.debug(f"Failed to get proactive status: {e}")
            
            try:
                result["embeddings"] = await self._route_handlers["get_embedding_stats"]()
            except Exception as e:
                logger.debug(f"Failed to get embedding stats: {e}")
            
            try:
                result["memory_stats"] = await self._route_handlers["get_memory_stats"]()
            except Exception as e:
                logger.debug(f"Failed to get memory stats: {e}")
            
            try:
                result["insights"] = await self._route_handlers["get_recent_insights"]()
            except Exception as e:
                logger.debug(f"Failed to get insights: {e}")
            
            try:
                result["vector_entries"] = await self._route_handlers["get_vector_memory_entries"]()
            except Exception as e:
                logger.debug(f"Failed to get vector memory entries: {e}")
            
            try:
                result["knowledge_nodes"] = await self._route_handlers["get_knowledge_graph_nodes"]()
            except Exception as e:
                logger.debug(f"Failed to get knowledge graph nodes: {e}")
            
            try:
                from core.context_manager import get_context_manager
                cm = get_context_manager()
                if cm:
                    result["context"] = {
                        "active_tokens": getattr(cm, 'active_tokens', 0),
                        "max_tokens": getattr(cm, 'max_tokens', 96000),
                        "usage_ratio": getattr(cm, 'usage_ratio', 0) if hasattr(cm, 'usage_ratio') else 0,
                        "important_messages": len(cm.important_messages) if hasattr(cm, 'important_messages') else 0,
                        "enhanced_enabled": getattr(cm, 'enhanced_enabled', True),
                    }
            except Exception as e:
                logger.debug(f"Failed to get context manager: {e}")
            
            try:
                from core.active_presence import get_active_presence
                ap = get_active_presence()
                result["presence"] = ap.get_state()
            except Exception as e:
                logger.debug(f"Failed to get presence: {e}")

            try:
                from core.goal_anticipation import get_goal_anticipation
                ga = get_goal_anticipation()
                anticipation_state = ga.get_state()
                pending = ga.get_pending_anticipations()
                result["anticipation"] = {
                    **anticipation_state,
                    "pending_list": [{"description": a.description, "confidence": round(a.confidence, 2), "source": a.source.value, "reasoning": a.reasoning, "suggested_action": a.suggested_action} for a in pending],
                }
            except Exception as e:
                logger.debug(f"Failed to get anticipation: {e}")

            try:
                from core.autonomous_observer import get_autonomous_observer
                ao = get_autonomous_observer()
                observer_state = ao.get_state()
                habits = ao.get_learned_habits()
                result["autonomous_observer"] = {
                    **observer_state,
                    "habits_list": habits[:5],
                }
            except Exception as e:
                logger.debug(f"Failed to get observer: {e}")
            
            return result
        # Sibling-route handlers called from other route groups.
        self._route_handlers["get_tools"] = get_tools


    def _setup_status_routes(self):
        @self.app.get("/api/flow/live")
        async def get_flow_live():
            """Get real-time semantic flow data for visualization."""
            flow_data = {
                "success": True,
                "timestamp": time.time(),
                "nodes": [],
                "connections": [],
                "events": [],
                "stats": {},
            }
            
            try:
                from core.dual_model import get_dual_model_orchestrator
                orchestrator = get_dual_model_orchestrator()
                if orchestrator:
                    stats = orchestrator.get_stats()
                    flow_data["stats"]["dual_model"] = {
                        "primary_requests": stats.get("primary_requests", 0),
                        "backend_requests": stats.get("backend_requests", 0),
                        "audio_requests": stats.get("audio_requests", 0),
                        "total_requests": stats.get("total_requests", 0),
                        "primary_rpm": stats.get("primary_requests_per_min", 0),
                        "backend_rpm": stats.get("backend_requests_per_min", 0),
                        "audio_rpm": stats.get("audio_requests_per_min", 0),
                        "avg_primary_latency_ms": stats.get("avg_primary_latency_ms", 0),
                        "avg_backend_latency_ms": stats.get("avg_backend_latency_ms", 0),
                        "primary_connected": stats.get("primary_connected", True),
                        "backend_connected": stats.get("backend_connected", True),
                        "audio_connected": stats.get("audio_connected", True),
                    }
                    
                    if stats.get("primary_requests", 0) > 0 or stats.get("backend_requests", 0) > 0:
                        flow_data["events"].append({
                            "type": "model_activity",
                            "source": "Dual Model",
                            "target": "LLM",
                            "data": f"Primary: {stats.get('primary_requests', 0)}, Backend: {stats.get('backend_requests', 0)}",
                            "active": stats.get("primary_requests_per_min", 0) > 0 or stats.get("backend_requests_per_min", 0) > 0,
                        })
            except Exception as e:
                logger.debug(f"Failed to get dual model flow stats: {e}")
            
            try:
                from core.embedding_interconnect import get_embedding_interconnect
                interconnect = get_embedding_interconnect()
                
                ic_stats = interconnect.get_stats()
                
                primary_outputs = ic_stats["model_outputs"]["primary"]
                backend_outputs = ic_stats["model_outputs"]["backend"]
                conversation_embs = ic_stats["conversation_turns"]
                memory_embs = ic_stats["memory_embeddings"]
                insight_embs = ic_stats["insight_embeddings"]
                knowledge_embs = ic_stats["knowledge_embeddings"]
                
                flow_data["stats"]["embedding_interconnect"] = {
                    "model_outputs": primary_outputs + backend_outputs,
                    "conversation_embeddings": conversation_embs,
                    "memory_embeddings": memory_embs,
                    "insight_embeddings": insight_embs,
                    "knowledge_embeddings": knowledge_embs,
                    "total": primary_outputs + backend_outputs + conversation_embs + memory_embs + insight_embs + knowledge_embs,
                    "rate_per_min": getattr(interconnect, '_embedding_rate_per_min', 0),
                }
                
                if primary_outputs > 0 or backend_outputs > 0:
                    flow_data["events"].append({
                        "type": "embedding_activity",
                        "source": "Models",
                        "target": "Embedding Interconnect",
                        "data": f"{primary_outputs + backend_outputs} model embeddings stored",
                        "active": getattr(interconnect, '_embedding_rate_per_min', 0) > 0,
                    })
            except Exception as e:
                logger.debug(f"Failed to get embedding interconnect stats: {e}")
            
            try:
                from core.memory import get_memory
                from core.vector_memory import get_vector_memory
                
                mem = get_memory()
                conv_count = len(mem.conversations) if hasattr(mem, 'conversations') else 0
                
                vector_count = 0
                vector_dim = 768
                try:
                    vm = get_vector_memory()
                    if vm:
                        vm_stats = vm.get_stats() if hasattr(vm, 'get_stats') else {}
                        vector_count = vm_stats.get("vector_count", 0)
                        vector_dim = vm_stats.get("dimension", 768)
                except Exception as e:
                    logger.debug(f"Failed to get vector memory stats: {e}")
                
                flow_data["stats"]["memory"] = {
                    "conversation_turns": conv_count,
                    "vector_count": vector_count,
                    "vector_dimension": vector_dim,
                }
            except Exception as e:
                logger.debug(f"Failed to get memory flow stats: {e}")
            
            try:
                from core.knowledge_graph import get_knowledge_graph
                kg = get_knowledge_graph()
                if kg:
                    kg_stats = kg.get_graph_statistics()
                    flow_data["stats"]["knowledge_graph"] = {
                        "nodes": kg_stats.get("nodes", 0),
                        "edges": kg_stats.get("edges", 0),
                        "communities": kg_stats.get("communities", 0),
                        "avg_connections": kg_stats.get("avg_connections", 0),
                    }
                    
                    if kg_stats.get("nodes", 0) > 0:
                        flow_data["events"].append({
                            "type": "knowledge_activity",
                            "source": "Knowledge Graph",
                            "target": "Context",
                            "data": f"{kg_stats.get('total_nodes', 0)} nodes, {kg_stats.get('total_edges', 0)} connections",
                            "active": False,
                        })
            except Exception as e:
                logger.debug(f"Failed to get knowledge graph flow stats: {e}")
            
            try:
                from core.context_manager import get_context_manager
                cm = get_context_manager()
                if cm:
                    flow_data["stats"]["context"] = {
                        "active_tokens": getattr(cm, 'active_tokens', 0),
                        "max_tokens": getattr(cm, 'max_tokens', 96000),
                        "usage_ratio": getattr(cm, 'usage_ratio', 0) if hasattr(cm, 'usage_ratio') else 0,
                    }
            except Exception as e:
                logger.debug(f"Failed to get context flow stats: {e}")
            
            try:
                from core.insight_bridge import get_insight_bridge
                bridge = get_insight_bridge()
                if bridge:
                    bridge_stats = bridge.get_stats() if hasattr(bridge, 'get_stats') else {}
                    flow_data["stats"]["insight_bridge"] = {
                        "insights_processed": bridge_stats.get("insights_processed", 0),
                        "evolution_suggestions": bridge_stats.get("evolution_suggestions", 0),
                    }
            except Exception as e:
                logger.debug(f"Failed to get insight bridge flow stats: {e}")
            
            try:
                # Audio I/O (voice token) stats
                from core.voice_tokens import get_voice_token_manager
                vtm = get_voice_token_manager()
                if vtm:
                    voice_stats = vtm.get_stats()
                    flow_data["stats"]["audio_io"] = {
                        "inputs_processed": voice_stats.get("inputs_processed", 0),
                        "outputs_generated": voice_stats.get("outputs_generated", 0),
                        "semantic_tokens": voice_stats.get("semantic_tokens", 0),
                        "acoustic_frames": voice_stats.get("acoustic_frames", 0),
                    }
            except Exception as e:
                logger.debug(f"Failed to get audio I/O flow stats: {e}")
            
            try:
                subsystems_json = await self._route_handlers["get_all_subsystems"]()
                subsystems = subsystems_json.get("subsystems", {})
                active_subs = []
                
                for name, data in subsystems.items():
                    if data.get("running"):
                        active_subs.append({
                            "name": name,
                            "status": data.get("status", "unknown"),
                            "current_task": data.get("current_task", ""),
                            "message": data.get("message", ""),
                            "current_activity": data.get("current_activity"),
                        })
                
                # Add music_creation to active subsystems if running (if not already included)
                try:
                    from core.music_creation_engine import get_music_creation_engine
                    music_creation = get_music_creation_engine()
                    if music_creation:
                        music_stats = music_creation.get_stats()
                        if music_stats.get("running", False):
                            # Check if not already in active_subs
                            if not any(s.get("name") == "music_creation" for s in active_subs):
                                active_subs.append({
                                    "name": "music_creation",
                                    "status": "active",
                                    "current_task": f"Composing {music_stats.get('current_style', 'classical')}",
                                    "message": f"Created {music_stats.get('compositions_created', 0)} compositions",
                                    "current_activity": "composing",
                                })
                except Exception as e:
                    logger.debug(f"Failed to get music creation subsystem: {e}")
                
                # Always set subsystems, even if empty
                flow_data["stats"]["subsystems"] = {
                    "active_count": len(active_subs),
                    "active": active_subs,
                }
                
                for sub in active_subs[:3]:
                        flow_data["events"].append({
                            "type": "subsystem_activity",
                            "source": sub["name"],
                            "target": "Background Processing",
                            "data": sub.get("current_task", sub.get("message", ""))[:60],
                            "active": True,
                            "current_activity": sub.get("current_activity"),  # Pass activity type
                        })
            except Exception as e:
                logger.debug(f"Failed to get subsystems flow stats: {e}")
            
            # Add subsystem-specific stats for detailed monitoring
            try:
                # Erdos Engine stats
                from core.erdos_engine import get_erdos_engine
                erdos = get_erdos_engine()
                if erdos:
                    erdos_stats = erdos.get_stats() if hasattr(erdos, 'get_stats') else {}
                    flow_data["stats"]["erdos"] = {
                        "problems_count": len(erdos.problems) if hasattr(erdos, 'problems') else 0,
                        "running": erdos._running if hasattr(erdos, '_running') else False,
                        "insights_generated": erdos_stats.get("insights_generated", 0),
                    }
            except Exception as e:
                logger.debug(f"Failed to get erdos flow stats: {e}")
            
            try:
                # Learning Pipeline stats
                from core.learning_pipeline import get_learning_pipeline
                pipeline = get_learning_pipeline()
                if pipeline:
                    flow_data["stats"]["learning"] = {
                        "sessions_count": len(pipeline.sessions) if hasattr(pipeline, 'sessions') else 0,
                        "running": pipeline._running if hasattr(pipeline, '_running') else False,
                    }
            except Exception as e:
                logger.debug(f"Failed to get learning pipeline flow stats: {e}")
            
            try:
                # Evolution stats
                from core.evolution_orchestrator import get_evolution_orchestrator
                orch = get_evolution_orchestrator()
                if orch:
                    orch_stats = orch.get_status() if hasattr(orch, 'get_status') else {}
                    flow_data["stats"]["evolution"] = {
                        "genes_count": len(orch.genes) if hasattr(orch, 'genes') else 0,
                        "running": orch._running if hasattr(orch, '_running') else False,
                        "proposals": orch_stats.get("proposals_count", 0),
                    }
            except Exception as e:
                logger.debug(f"Failed to get evolution flow stats: {e}")
            
            try:
                # Proactive Engine stats
                from core.proactive_engine import get_proactive_engine
                proactive = get_proactive_engine()
                if proactive:
                    proactive_stats = proactive.get_status() if hasattr(proactive, 'get_status') else {}
                    flow_data["stats"]["proactive"] = {
                        "running": proactive._running if hasattr(proactive, '_running') else False,
                        "curiosity_count": proactive_stats.get("curiosity_count", 0),
                        "reflections_count": proactive_stats.get("reflections_count", 0),
                    }
            except Exception as e:
                logger.debug(f"Failed to get proactive flow stats: {e}")
            
            try:
                # Music Creation Engine stats
                from core.music_creation_engine import get_music_creation_engine
                music_creation = get_music_creation_engine()
                if music_creation:
                    music_stats = music_creation.get_stats()
                    running = music_stats.get("running", False)
                    flow_data["stats"]["music_creation"] = {
                        "running": running,
                        "compositions_created": music_stats.get("compositions_created", 0),
                        "tokens_observed": music_stats.get("tokens_observed", 0),
                        "current_style": music_stats.get("current_style"),
                    }
            except Exception as e:
                logger.debug(f"Failed to get music creation flow stats: {e}")
            
            try:
                from core.unified_context_engine import get_unified_context_engine
                uce = get_unified_context_engine()
                if uce:
                    uce_stats = uce.get_stats() if hasattr(uce, 'get_stats') else {}
                    flow_data["stats"]["context_engine"] = {
                        "queries_processed": uce_stats.get("queries_processed", 0),
                        "cache_hits": uce_stats.get("cache_hits", 0),
                        "avg_latency_ms": uce_stats.get("avg_latency_ms", 0),
                    }
            except Exception as e:
                logger.debug(f"Failed to get unified context engine flow stats: {e}")
            
            flow_data["nodes"] = [
                {"id": "user", "label": "User", "type": "input"},
                {"id": "agent", "label": "Jarvis Agent", "type": "processor"},
                {"id": "coordinator", "label": "Context Coordinator", "type": "hub"},
                {"id": "cascade", "label": "Context Cascade", "type": "processor"},
                {"id": "pfc", "label": "Prefrontal Cortex", "type": "processor"},
                {"id": "embedding", "label": "Embedding Interconnect", "type": "hub"},
                {"id": "memory", "label": "Memory System", "type": "storage"},
                {"id": "knowledge", "label": "Knowledge Graph", "type": "storage"},
                {"id": "primary", "label": "Primary Model\n(Kimi K2.5)", "type": "model"},
                {"id": "backend", "label": "Backend Model\n(GLM-5)", "type": "model"},
                {"id": "subsystems", "label": "Background Subsystems\n(Erdos/Music/Daydream)", "type": "generator"},
                {"id": "agentic", "label": "Agentic Context\n(Project Work)", "type": "processor"},
            ]
            
            flow_data["connections"] = [
                # User input flow
                {"from": "user", "to": "agent", "label": "query", "type": "input"},
                
                # Context flow through coordinator
                {"from": "agent", "to": "coordinator", "label": "observe", "type": "request"},
                {"from": "coordinator", "to": "cascade", "label": "cascade", "type": "bidirectional"},
                {"from": "cascade", "to": "pfc", "label": "concepts", "type": "bidirectional"},
                {"from": "coordinator", "to": "memory", "label": "fetch", "type": "bidirectional"},
                {"from": "coordinator", "to": "knowledge", "label": "fetch", "type": "bidirectional"},
                {"from": "coordinator", "to": "embedding", "label": "search", "type": "bidirectional"},
                
                # Embedding hub connections
                {"from": "embedding", "to": "memory", "label": "store/embed", "type": "bidirectional"},
                {"from": "embedding", "to": "knowledge", "label": "index", "type": "bidirectional"},
                {"from": "embedding", "to": "pfc", "label": "concepts", "type": "bidirectional"},
                
                # Dual model routing with PFC guidance
                {"from": "agent", "to": "primary", "label": "chat/creative", "type": "model_call"},
                {"from": "agent", "to": "backend", "label": "code/analysis", "type": "model_call"},
                {"from": "pfc", "to": "backend", "label": "routing", "type": "request"},
                
                # Model outputs stored in embedding
                {"from": "primary", "to": "embedding", "label": "store", "type": "embedding"},
                {"from": "backend", "to": "embedding", "label": "store", "type": "embedding"},
                
                # Subsystems flow
                {"from": "subsystems", "to": "cascade", "label": "observe", "type": "insight"},
                {"from": "subsystems", "to": "knowledge", "label": "insights", "type": "insight"},
                {"from": "subsystems", "to": "embedding", "label": "embed", "type": "insight"},
                
                # Agentic context for project work
                {"from": "agent", "to": "agentic", "label": "project", "type": "bidirectional"},
                {"from": "agentic", "to": "backend", "label": "execute", "type": "model_call"},
            ]
            
            # Add tools count
            try:
                tools_data = await self._route_handlers["get_tools"]()
                flow_data["stats"]["tools_count"] = len(tools_data) if isinstance(tools_data, dict) else len(tools_data) if tools_data else 0
            except Exception:
                flow_data["stats"]["tools_count"] = 0
            
            # Add knowledge graph nodes count
            try:
                from core.knowledge_graph import get_knowledge_graph
                kg = get_knowledge_graph()
                if kg:
                    kg_stats = kg.get_graph_statistics() if hasattr(kg, 'get_graph_statistics') else {}
                    flow_data["stats"]["knowledge_graph"] = {
                        "nodes": kg_stats.get("total_nodes", len(kg.nodes) if hasattr(kg, 'nodes') else 0),
                        "edges": kg_stats.get("total_edges", len(kg.edges) if hasattr(kg, 'edges') else 0),
                        "communities": kg_stats.get("communities", 0),
                    }
            except Exception as e:
                logger.debug(f"Failed to get knowledge graph flow: {e}")
            
            # Add vector memory count
            try:
                from core.vector_memory import get_vector_memory
                vm = get_vector_memory()
                if vm:
                    vm_stats = vm.stats() if hasattr(vm, 'stats') else {}
                    flow_data["stats"]["vector_memory"] = {
                        "vector_count": vm_stats.get('num_vectors', 0),
                        "max_level": vm_stats.get('max_level', 0),
                        "utilization": vm_stats.get('utilization', 0),
                    }
            except Exception as e:
                logger.debug(f"Vector memory stats error: {e}")
            
            # Add embedding stats
            try:
                emb_stats = await self._route_handlers["get_embedding_stats"]()
                flow_data["stats"]["embeddings"] = emb_stats if emb_stats else {}
            except Exception as e:
                logger.debug(f"Failed to get embedding flow stats: {e}")
            
            # Add Prefrontal Cortex stats
            try:
                # Check for context cascade observations
                try:
                    from core.context_cascade import get_context_cascade
                    cascade = get_context_cascade()
                    if cascade and hasattr(cascade, '_cascade_history'):
                        recent_cascades = cascade._cascade_history[-5:] if cascade._cascade_history else []
                        for c in recent_cascades:
                            flow_data["events"].append({
                                "type": "cascade",
                                "source": getattr(c, 'source', 'unknown'),
                                "trigger": getattr(c, 'trigger_type', 'unknown'),
                                "concepts": len(getattr(c, 'concepts_observed', [])),
                                "anomaly": getattr(c, 'anomaly_detected', False),
                                "timestamp": getattr(c, 'timestamp', 0).isoformat() if hasattr(getattr(c, 'timestamp', 0), 'isoformat') else '',
                            })
                except Exception as e:
                    logger.debug(f"Failed to get context cascade events: {e}")
                
                if hasattr(self.agent, '_prefrontal_cortex') and self.agent._prefrontal_cortex:
                    pfc_stats = self.agent._prefrontal_cortex.get_stats()
                    flow_data["stats"]["prefrontal_cortex"] = {
                        "total_concepts": pfc_stats.get("total_concepts", 0),
                        "stored_concepts": pfc_stats.get("stored_concepts", 0),
                        "active_concepts": pfc_stats.get("active_concepts", 0),
                        "training_steps": pfc_stats.get("training_steps", 0),
                        "current_confidence": round(pfc_stats.get("current_confidence", 0), 3),
                        "training_buffer": pfc_stats.get("lcm_stats", {}).get("training_buffer_size", 0),
                        "total_semantic_tokens": pfc_stats.get("total_semantic_tokens", 0),
                        "avg_loss": round(pfc_stats.get("lcm_stats", {}).get("avg_loss", 0), 4) if pfc_stats.get("lcm_stats", {}).get("avg_loss") else None,
                    }
            except Exception as e:
                logger.debug(f"Prefrontal cortex stats error: {e}")
            
            # Add Context Cascade stats
            try:
                from core.context_cascade import get_context_cascade
                cascade = get_context_cascade()
                if cascade:
                    is_healthy, cascade_stats = cascade.health_check()
                    flow_data["stats"]["context_cascade"] = {
                        "healthy": is_healthy,
                        "total_observations": cascade_stats.get("total_observations", 0),
                        "components": cascade_stats.get("components", {}),
                    }
            except Exception as e:
                logger.debug(f"Context cascade stats error: {e}")
            
            # Add Context Coordinator stats
            try:
                from core.context_coordinator import get_context_coordinator
                coordinator = get_context_coordinator()
                if coordinator:
                    coord_stats = coordinator.get_stats()
                    flow_data["stats"]["context_coordinator"] = {
                        "subsystems_loaded": coord_stats.get("subsystems_loaded", 0),
                        "subsystem_names": coord_stats.get("subsystem_names", []),
                        "memory_pressure": coord_stats.get("memory_pressure", False),
                    }
            except Exception as e:
                logger.debug(f"Context coordinator stats error: {e}")
            
            # Add Agentic Context stats
            try:
                from core.agentic_context import get_agentic_context
                agentic = get_agentic_context()
                if agentic:
                    flow_data["stats"]["agentic_context"] = {
                        "workflow_id": getattr(agentic, 'workflow_id', 'default'),
                        "project_name": getattr(agentic.project, 'project_name', '') if hasattr(agentic, 'project') else '',
                        "goal_progress": getattr(agentic.goal, 'progress', 0) if hasattr(agentic, 'goal') else 0,
                        "history_steps": len(agentic.history.steps) if hasattr(agentic, 'history') else 0,
                        "cache_entries": len(agentic.tool_cache) if hasattr(agentic, 'tool_cache') else 0,
                    }
            except Exception as e:
                logger.debug(f"Agentic context stats error: {e}")
            
            # Add recent tool calls for flow visualization
            try:
                recent_tools = []
                if hasattr(self.agent, '_last_tool_calls') and self.agent._last_tool_calls:
                    # Get last 10 tool calls
                    recent_tools = [
                        {"name": tc[0] if isinstance(tc, tuple) else str(tc), "args": tc[1] if isinstance(tc, tuple) and len(tc) > 1 else {}}
                        for tc in self.agent._last_tool_calls[-10:]
                    ]
                
                # Tool usage frequency
                tool_frequency = {}
                if hasattr(self.agent, 'tool_usage_count'):
                    tool_frequency = dict(self.agent.tool_usage_count)
                
                # Active tools (recently used in last 60 seconds)
                active_tools = []
                if recent_tools:
                    active_tools = list(set(tc["name"] for tc in recent_tools if tc.get("name")))
                
                flow_data["stats"]["tool_usage"] = {
                    "recent_calls": recent_tools[-5:] if recent_tools else [],
                    "frequency": tool_frequency,
                    "active": active_tools,
                    "total_recent": len(recent_tools),
                }
            except Exception as e:
                logger.debug(f"Tool usage tracking error: {e}")
            
            return flow_data
            
        @self.app.get("/api/status")
        async def get_status():
            # Return cached status if fresh (< 1 second old)
            with self._status_cache_lock:
                now = time.time()
                if self._status_cache and now - self._status_cache_time < 1.0:
                    return self._status_cache

            model = "unknown"
            backend_model = "unknown"
            if hasattr(self.agent, 'llm') and hasattr(self.agent.llm, 'model'):
                model = self.agent.llm.model

            try:
                from core.backend_llm import get_model_status
                model_status = get_model_status()
                backend_model = model_status.get("backend", {}).get("model", "unknown")
            except Exception:
                model_status = {}

            turn_count = getattr(self.agent, '_turn_count', 0)

            total_calls = 0
            avg_latency = 0
            cache_rate = 0
            api_calls_per_min = 0
            
            # Get LLM call stats from primary client (this is the true calls_per_minute)
            try:
                if hasattr(self.agent, 'llm') and hasattr(self.agent.llm, 'get_call_stats'):
                    llm_stats = self.agent.llm.get_call_stats()
                    api_calls_per_min = llm_stats.get("calls_per_minute", 0)
                    total_calls = llm_stats.get("total_calls", 0)
            except Exception as e:
                logger.debug(f"Failed to get LLM call stats: {e}")
            
            # Get backend LLM call stats too
            try:
                from core.backend_llm import get_backend_llm
                backend_llm = get_backend_llm()
                if backend_llm and hasattr(backend_llm, 'get_call_stats'):
                    backend_stats = backend_llm.get_call_stats()
                    api_calls_per_min += backend_stats.get("calls_per_minute", 0)
                    total_calls += backend_stats.get("total_calls", 0)
            except Exception as e:
                logger.debug(f"Failed to get backend LLM call stats: {e}")
            
            # Get dual-model stats for total request counts
            dual_total_requests = 0
            try:
                from core.dual_model import get_dual_model_orchestrator
                orchestrator = get_dual_model_orchestrator()
                if orchestrator:
                    dual_stats = orchestrator.get_stats()
                    dual_total_requests = dual_stats.get("total_requests", 0)
            except Exception as e:
                logger.debug(f"Failed to get dual model stats: {e}")
            
            # Get API manager stats for cache and latency
            try:
                from core.api_manager import get_api_manager
                manager = get_api_manager()
                stats = manager.get_stats()
                avg_latency = int(stats.get("avg_latency", 0))
                api_manager_calls = stats.get("total_calls", 0)
                cache_hits = stats.get("cache_hits", 0)
                # Use max of LLM total_calls and API manager total_calls
                # LLM tracks all chat() calls, API manager tracks managed calls
                if api_manager_calls > total_calls:
                    total_calls = api_manager_calls
                if total_calls > 0:
                    cache_rate = int((cache_hits / total_calls) * 100)
            except Exception as e:
                logger.debug(f"Failed to get API manager stats: {e}")

            # Get memory usage
            memory_used_mb = 0
            memory_percent = 0
            try:
                from core.dynamic_memory import get_dynamic_memory_manager
                mm = get_dynamic_memory_manager()
                mem_status = mm.get_status()
                memory_used_mb = mem_status.get("process", {}).get("rss_mb", 0)
                memory_percent = mem_status.get("system", {}).get("ram_percent", 0)
            except Exception as e:
                logger.debug(f"Failed to get memory usage: {e}")

            # Get daydream status
            daydream_status = {}
            try:
                from core.daydream_engine import get_daydream_engine
                engine = get_daydream_engine()
                daydream_status = engine.get_status()
            except Exception as e:
                logger.debug(f"Failed to get daydream status: {e}")

            # Get background scheduler status for idle tracking
            is_idle = False
            last_activity_time = None
            try:
                if hasattr(self.agent, 'scheduler') and self.agent.scheduler:
                    is_idle = self.agent.scheduler.is_idle()
                    last_activity_time = getattr(self.agent.scheduler, '_idle_since', None)
            except Exception as e:
                logger.debug(f"Failed to get scheduler idle status: {e}")

            status = {
                "tools": len(self.agent.tools),
                "turn_count": turn_count,
                "turns": turn_count,
                "total_calls": total_calls,
                "calls_per_minute": api_calls_per_min,
                "total_requests": dual_total_requests,
                "avg_latency": avg_latency,
                "cache_rate": cache_rate,
                "memory_messages": len(self.agent.memory.get_messages_for_llm()) if hasattr(self.agent, 'memory') else 0,
                "model": model,
                "backend_model": backend_model,
                "models": model_status,
                "is_idle": is_idle,
                "last_activity": last_activity_time.isoformat() if last_activity_time else None,
                "daydream": daydream_status,
                "memory_used_mb": memory_used_mb,
                "memory_percent": memory_percent,
            }

            # Add memory status
            try:
                from core.dynamic_memory import get_dynamic_memory_manager
                mm = get_dynamic_memory_manager()
                mem_status = mm.get_status()
                status["memory"] = {
                    "pressure": mem_status.get("pressure", "normal"),
                    "process_mb": mem_status.get("process", {}).get("rss_mb", 0),
                    "ram_percent": mem_status.get("system", {}).get("ram_percent", 0),
                    "swap_percent": mem_status.get("system", {}).get("swap_percent", 0),
                    "max_gb": mem_status.get("config", {}).get("max_memory_mb", 0) / 1024,
                    "cleanups": mem_status.get("stats", {}).get("total_cleanups", 0),
                }
            except Exception as e:
                logger.debug(f"Failed to get dynamic memory status: {e}")

            try:
                from core.erdos_engine import get_erdos_engine
                engine = get_erdos_engine()
                if hasattr(engine, '_learning_state'):
                    status["erdos_problems"] = engine._learning_state.get('total_problems', 0)
            except Exception as e:
                logger.debug(f"Failed to get erdos problems: {e}")

            try:
                from core.evolution_orchestrator import get_evolution_orchestrator
                orchestrator = get_evolution_orchestrator()
                if hasattr(orchestrator, 'genes'):
                    status["evolution_genes"] = len(orchestrator.genes) if orchestrator.genes else 0
                if hasattr(orchestrator, 'proposal_history'):
                    status["evolution_proposals"] = len(orchestrator.proposal_history) if orchestrator.proposal_history else 0
            except Exception as e:
                logger.debug(f"Failed to get evolution stats: {e}")

            try:
                from core.learning_pipeline import get_learning_pipeline
                pipeline = get_learning_pipeline()
                if hasattr(pipeline, '_session_counter'):
                    status["learning_sessions"] = pipeline._session_counter
            except Exception as e:
                logger.debug(f"Failed to get learning sessions: {e}")

            try:
                from core.insight_bridge import get_insight_bridge
                bridge = get_insight_bridge()
                if hasattr(bridge, 'processed_insights'):
                    status["insights_processed"] = len(bridge.processed_insights) if bridge.processed_insights else 0
            except Exception as e:
                logger.debug(f"Failed to get insight bridge stats: {e}")

            # Phase 1 subsystem stats
            try:
                from core.lcm_shield import get_lcm_shield
                shield = get_lcm_shield()
                status["lcm_shield"] = shield.get_stats()
            except Exception as e:
                logger.debug(f"Failed to get LCM shield stats: {e}")

            try:
                from core.three_layer_context import get_three_layer_context
                tlc = get_three_layer_context()
                tlc_stats = tlc.get_stats()
                status["three_layer"] = {
                    "l1_token_limit": tlc_stats.get("l1_token_limit", 0),
                    "l2_block_count": tlc_stats.get("l2_block_count", 0),
                    "l1_evictions": tlc_stats.get("l1_evictions", 0),
                    "memory_block_edits": tlc_stats.get("memory_block_edits", 0),
                }
            except Exception as e:
                logger.debug(f"Failed to get three-layer context stats: {e}")

            try:
                from core.reflect_edit import get_reflect_edit_loop, get_council_deliberation
                reflect_loop = get_reflect_edit_loop()
                status["reflect_edit"] = reflect_loop.get_stats()
                council = get_council_deliberation()
                status["council"] = council.get_stats()
            except Exception as e:
                logger.debug(f"Failed to get reflect-edit/council stats: {e}")

            try:
                from core.audio_bridge import get_audio_bridge
                audio = get_audio_bridge()
                status["audio_bridge"] = audio.get_stats()
            except Exception as e:
                logger.debug(f"Failed to get audio bridge stats: {e}")

            try:
                from core.semantic_contracts import get_contract_registry
                contracts = get_contract_registry()
                status["contracts"] = contracts.get_stats()
            except Exception as e:
                logger.debug(f"Failed to get contract registry stats: {e}")

            try:
                from core.lcm_planner import get_lcm_planner
                planner = get_lcm_planner()
                status["lcm_planner"] = planner.get_stats()
            except Exception as e:
                logger.debug(f"Failed to get LCM planner stats: {e}")

            try:
                from core.context_guardian import get_context_guardian
                guardian = get_context_guardian()
                status["context_guardian"] = guardian.get_stats()
            except Exception as e:
                logger.debug(f"Failed to get Context Guardian stats: {e}")

            try:
                from core.prompt_sandbox import get_prompt_sandbox
                sandbox = get_prompt_sandbox()
                status["prompt_sandbox"] = sandbox.get_stats()
            except Exception as e:
                logger.debug(f"Failed to get Prompt Sandbox stats: {e}")

            try:
                from core.security_audit import get_security_audit_logger
                audit = get_security_audit_logger()
                status["security_audit"] = audit.get_stats()
            except Exception as e:
                logger.debug(f"Failed to get Security Audit stats: {e}")

            # Cache the status for 1 second
            with self._status_cache_lock:
                self._status_cache = status
                self._status_cache_time = time.time()

            return status

        @self.app.get("/api/cli-stats")
        async def get_cli_stats():
            try:
                from core.api_manager import get_api_manager
                manager = get_api_manager()
                stats = manager.get_stats()
                activities = manager.get_recent_activities(50)

                return {
                    "total_calls": stats.get("total_calls", 0),
                    "primary_calls": stats.get("primary_calls", 0),
                    "backend_calls": stats.get("backend_calls", 0),
                    "cache_hits": stats.get("cache_hits", 0),
                    "total_tokens": stats.get("total_tokens", 0),
                    "avg_latency": int(stats.get("avg_latency", 0)),
                    "turn_count": getattr(self.agent, '_turn_count', 0),
                    "queue_size": 0,
                    "recent_activities": activities,
                }
            except Exception as e:
                return {"error": safe_error_message(e)}

        @self.app.get("/api/telemetry")
        async def get_telemetry():
            try:
                from core.telemetry_aggregator import get_telemetry_aggregator
                aggregator = get_telemetry_aggregator()
                aggregator.collect()
                return aggregator.get_summary()
            except Exception as e:
                return {"error": safe_error_message(e), "overall_health": 0, "subsystems": {}}

        @self.app.get("/api/degradation")
        async def get_degradation():
            try:
                from core.graceful_degradation import get_degradation_manager
                return get_degradation_manager().get_summary()
            except Exception as e:
                return {"error": safe_error_message(e)}

        @self.app.get("/api/subsystem_logs")
        async def get_subsystem_logs():
            from datetime import datetime
            now = datetime.now
            logs = {}

            try:
                from core.erdos_engine import get_erdos_engine
                engine = get_erdos_engine()
                if hasattr(engine, 'get_status'):
                    stats = engine.get_status()
                    if stats.get('problems_explored', 0) > 0:
                        logs["erdos"] = [{"time": str(now()), "level": "INFO", "message": f"{stats['problems_explored']} problems explored"}]
            except Exception as e:
                logger.debug(f"Failed to get erdos logs: {e}")

            try:
                from core.daydream_engine import get_daydream_engine
                engine = get_daydream_engine()
                if hasattr(engine, 'get_status'):
                    stats = engine.get_status()
                    if stats.get('explorations', 0) > 0:
                        logs["daydream"] = [{"time": str(now()), "level": "INFO", "message": f"{stats['explorations']} explorations"}]
            except Exception as e:
                logger.debug(f"Failed to get daydream logs: {e}")

            try:
                from core.learning_pipeline import get_learning_pipeline
                pipeline = get_learning_pipeline()
                if hasattr(pipeline, 'get_status'):
                    stats = pipeline.get_status()
                    if stats.get('refinements', 0) > 0:
                        logs["learning"] = [{"time": str(now()), "level": "INFO", "message": f"{stats['refinements']} refinements"}]
            except Exception as e:
                logger.debug(f"Failed to get learning logs: {e}")

            try:
                from core.evolution_orchestrator import get_evolution_orchestrator
                orchestrator = get_evolution_orchestrator()
                if hasattr(orchestrator, 'get_status'):
                    stats = orchestrator.get_status()
                    if stats.get('improvements_proposed', 0) > 0:
                        logs["evolution"] = [{"time": str(now()), "level": "INFO", "message": f"{stats['improvements_proposed']} improvements proposed"}]
            except Exception as e:
                logger.debug(f"Failed to get evolution logs: {e}")

            try:
                from core.state_sync import get_state_synchronizer
                sync = get_state_synchronizer()
                if sync:
                    logs["state_sync"] = [{"time": "-", "level": "INFO", "message": "Checkpoint active"}]
            except Exception as e:
                logger.debug(f"Failed to get state sync logs: {e}")

            try:
                from core.memory_cleanup import get_memory_manager
                mm = get_memory_manager()
                if mm:
                    logs["memory_manager"] = [{"time": "-", "level": "INFO", "message": "Cleanup scheduled"}]
            except Exception as e:
                logger.debug(f"Failed to get memory manager logs: {e}")

            try:
                from core.defense import get_defense_agent
                defense = get_defense_agent()
                if defense:
                    logs["defense"] = [{"time": "-", "level": "INFO", "message": "Monitoring active"}]
            except Exception as e:
                logger.debug(f"Failed to get defense logs: {e}")

            try:
                from core.observability import get_observability
                obs = get_observability()
                if obs:
                    logs["observability"] = [{"time": "-", "level": "INFO", "message": "Collecting metrics"}]
            except Exception as e:
                logger.debug(f"Failed to get observability logs: {e}")

            return logs
        # Sibling-route handlers called from other route groups.
        self._route_handlers["get_status"] = get_status


    def _setup_bio_predictive_routes(self):
        @self.app.get("/api/bio/state")
        async def get_bio_state():
            try:
                from core.bio_integration import get_bio_integration
                bio = get_bio_integration()
                state = bio.tick()
                return {
                    "success": True,
                    "arousal": state.arousal,
                    "valence": state.valence,
                    "cognitive_capacity": state.cognitive_capacity,
                    "creative_capacity": state.creative_capacity,
                    "analytical_capacity": state.analytical_capacity,
                    "stress_phase": state.stress_phase,
                    "stress_intensity": state.stress_intensity,
                    "allostatic_load": state.allostatic_load,
                    "autonomic_state": state.autonomic_state,
                    "vagal_tone": state.vagal_tone,
                    "circadian_phase": state.circadian_phase,
                    "sleep_pressure": state.sleep_pressure,
                    "is_homeostatic": state.is_homeostatic,
                    "learning_plasticity": state.learning_plasticity,
                    "energy_available": state.energy_available,
                    "energy_utilization": state.energy_utilization,
                    "neurotransmitters": state.neurotransmitters,
                    "regulatory_actions": state.regulatory_actions,
                    "recommendations": state.recommendations,
                }
            except Exception as e:
                logger.debug(f"Bio state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/bio/style")
        async def get_bio_style():
            try:
                from core.bio_integration import get_bio_integration
                bio = get_bio_integration()
                return {"success": True, "style": bio.get_response_style()}
            except Exception as e:
                logger.debug(f"Bio style API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/bio/stressor")
        async def apply_bio_stressor(request):
            try:
                from core.bio_integration import get_bio_integration
                body = await request.json()
                intensity = float(body.get("intensity", 0.5))
                source = body.get("source", "api")
                bio = get_bio_integration()
                state = bio.apply_stressor(intensity, source)
                return {
                    "success": True,
                    "stress_phase": state.stress_phase,
                    "stress_intensity": state.stress_intensity,
                    "is_homeostatic": state.is_homeostatic,
                }
            except Exception as e:
                logger.debug(f"Bio stressor API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/bio/reward")
        async def apply_bio_reward(request):
            try:
                from core.bio_integration import get_bio_integration
                body = await request.json()
                magnitude = float(body.get("magnitude", 0.5))
                neurotransmitter = body.get("neurotransmitter", "dopamine")
                bio = get_bio_integration()
                state = bio.apply_reward(magnitude, neurotransmitter)
                return {
                    "success": True,
                    "dopamine": state.neurotransmitters.get("dopamine", 0.5),
                    "valence": state.valence,
                    "is_homeostatic": state.is_homeostatic,
                }
            except Exception as e:
                logger.debug(f"Bio reward API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/bio/circadian")
        async def get_bio_circadian():
            try:
                from core.circadian_rhythm import get_circadian_rhythm
                circ = get_circadian_rhythm()
                return {
                    "success": True,
                    "phase": circ.get_phase().value,
                    "cognitive_capacity": circ.get_cognitive_capacity(),
                    "creative_capacity": circ.get_creative_capacity(),
                    "analytical_capacity": circ.get_analytical_capacity(),
                    "processing_fatigue": circ._processing_fatigue,
                    "sleep_pressure": circ._processing_fatigue,
                    "operational_phase": circ._operational_phase.value,
                    "current_demand": round(circ._current_demand, 3),
                    "recommendations": circ.get_recommendations(),
                    "user_activity_profile": circ.get_user_activity_profile(),
                }
            except Exception as e:
                logger.debug(f"Bio circadian API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/bio/neuroplasticity")
        async def get_bio_neuroplasticity():
            try:
                from core.neuroplasticity import get_neuroplasticity_engine
                np = get_neuroplasticity_engine()
                return {"success": True, "statistics": np.get_statistics()}
            except Exception as e:
                logger.debug(f"Bio neuroplasticity API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/bio/consolidate")
        async def bio_consolidate(request):
            try:
                from core.bio_integration import get_bio_integration
                body = await request.json() if request else {}
                duration = float(body.get("duration_factor", 1.0))
                bio = get_bio_integration()
                result = bio.consolidate(duration)
                return {"success": True, "result": result}
            except Exception as e:
                logger.debug(f"Bio consolidate API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/bio/homeostasis")
        async def get_bio_homeostasis():
            try:
                from core.bio_homeostasis import get_bio_homeostasis
                h = get_bio_homeostasis()
                state = h.get_state()
                return {"success": True, "state": state}
            except Exception as e:
                logger.debug(f"Bio homeostasis API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/predictive/state")
        async def get_predictive_state():
            try:
                from core.predictive_processing import get_predictive_processor
                pp = get_predictive_processor()
                return {"success": True, "state": pp.get_state()}
            except Exception as e:
                logger.debug(f"Predictive state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/world/state")
        async def get_world_state():
            """Physical world model state: scene belief, physics schemas,
            affordances, hypotheses (for the dashboard / future UI)."""
            try:
                from core.world_model import get_world_model
                wm = get_world_model()
                return {"success": True, "state": wm.get_state(),
                        "scene": wm.get_scene_summary()}
            except Exception as e:
                logger.debug(f"World state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/world/event")
        async def report_world_event(request: Request):
            """Report a physical event (from a tool, the UI, or an
            integration) into the world model; returns the prediction
            errors the event produced."""
            try:
                body = await request.json()
                body, validation_error = validate_request_fields(
                    body, required=["type"], max_string_length=2000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                from core.world_model import get_world_model
                errors = get_world_model().observe_event(
                    event_type=str(body["type"])[:40],
                    entity_labels=[str(e)[:60] for e in body.get("entities", [])][:10],
                    detail=str(body.get("detail", ""))[:500],
                    source="tool",
                    positions=body.get("positions"),
                )
                return {
                    "success": True,
                    "errors": [
                        {"type": e.error_type.value, "magnitude": round(e.magnitude, 2),
                         "description": e.description}
                        for e in errors
                    ],
                }
            except Exception as e:
                logger.debug(f"World event API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/predictive/process")
        async def process_predictive_input(request):
            try:
                from core.predictive_processing import get_predictive_processor
                body = await request.json()
                text = body.get("input", "")
                result = get_predictive_processor().process_input(text)
                return {"success": True, "result": result}
            except Exception as e:
                logger.debug(f"Predictive process API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/cognitive/state")
        async def get_cognitive_state():
            try:
                from core.cognitive_state_machine import get_cognitive_state_machine
                csm = get_cognitive_state_machine()
                return {"success": True, "state": csm._get_state()}
            except Exception as e:
                logger.debug(f"Cognitive state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/cognitive/force")
        async def force_cognitive_state(request):
            try:
                from core.cognitive_state_machine import get_cognitive_state_machine, CognitiveState
                body = await request.json()
                state_name = body.get("state", "awake")
                state_map = {s.value: s for s in CognitiveState}
                state = state_map.get(state_name, CognitiveState.AWAKE)
                csm = get_cognitive_state_machine()
                csm.force_state(state, reason="api")
                return {"success": True, "state": csm.current_state.value}
            except Exception as e:
                logger.debug(f"Cognitive force API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/predictive/curiosity")
        async def get_curiosity_topics():
            try:
                from core.predictive_processing import get_predictive_processor
                pp = get_predictive_processor()
                topics = pp.get_curiosity_topics(top_k=10)
                return {"success": True, "topics": topics}
            except Exception as e:
                logger.debug(f"Curiosity topics API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/spontaneous/thoughts")
        async def get_spontaneous_thoughts():
            try:
                from core.spontaneous_thought import get_spontaneous_thought_generator
                stg = get_spontaneous_thought_generator()
                thoughts = stg.peek_thoughts(limit=5)
                return {
                    "success": True,
                    "thoughts": [
                        {"content": t.content, "source": t.source, "priority": round(t.priority, 3),
                         "timestamp": t.timestamp}
                        for t in thoughts
                    ],
                    "state": stg.get_state(),
                }
            except Exception as e:
                logger.debug(f"Spontaneous thoughts API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/spontaneous/next")
        async def get_next_spontaneous_thought():
            try:
                from core.spontaneous_thought import get_spontaneous_thought_generator
                stg = get_spontaneous_thought_generator()
                thought = stg.get_next_thought()
                if thought:
                    return {"success": True, "thought": {
                        "content": thought.content, "source": thought.source,
                        "priority": round(thought.priority, 3), "timestamp": thought.timestamp,
                    }}
                return {"success": True, "thought": None}
            except Exception as e:
                logger.debug(f"Next spontaneous thought API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}


    def _setup_consciousness_routes(self):
        @self.app.get("/api/interoception/state")
        async def get_interoception_state():
            try:
                from core.interoception import get_interoception_engine, get_interoceptive_modulator
                ie = get_interoception_engine()
                mod = get_interoceptive_modulator()
                return {
                    "success": True,
                    "interoception": ie.get_state(),
                    "modulator": mod.get_state(),
                    "awareness": round(ie.get_awareness(), 3),
                    "granularity": round(ie.get_granularity(), 3),
                    "feeling_history": ie.get_feeling_history(limit=5),
                }
            except Exception as e:
                logger.debug(f"Interoception state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/workspace/state")
        async def get_workspace_state():
            try:
                from core.global_workspace import get_global_workspace
                gw = get_global_workspace()
                state = gw.get_state()
                state["current_broadcast"] = gw.get_current_broadcast()
                state["reverberation"] = gw.reverberate(limit=3)
                state["ignition_history"] = gw.get_ignition_history(limit=5)
                return {"success": True, "workspace": state}
            except Exception as e:
                logger.debug(f"Workspace state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/self-model/state")
        async def get_self_model_state():
            try:
                from core.recursive_self_model import get_recursive_self_model
                rsm = get_recursive_self_model()
                state = rsm.get_state()
                narrative = rsm.generate_meta_narrative()
                state["meta_narrative"] = narrative
                state["introspection"] = rsm.get_last_introspection()
                return {"success": True, "self_model": state}
            except Exception as e:
                logger.debug(f"Self-model state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/stream/state")
        async def get_stream_state():
            try:
                from core.stream_consciousness import get_stream_of_consciousness
                soc = get_stream_of_consciousness()
                state = soc.get_state()
                state["recent"] = soc.get_recent(limit=10)
                return {"success": True, "stream": state}
            except Exception as e:
                logger.debug(f"Stream state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/cognitive_psychology/state")
        async def get_cognitive_psychology_state():
            try:
                from core.cognitive_psychology import CognitivePsychologyEngine
                psych = CognitivePsychologyEngine.get_instance()
                return {"success": True, "cognitive_psychology": psych.to_dict()}
            except Exception as e:
                logger.debug(f"Cognitive psychology state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/three_layer/state")
        async def get_three_layer_state():
            try:
                from core.three_layer_context import get_three_layer_context
                tlc = get_three_layer_context()
                stats = tlc.get_stats()
                version = tlc.get_version()
                return {"success": True, "three_layer": stats, "version": version}
            except Exception as e:
                logger.debug(f"Three-layer state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/context_guardian/state")
        async def get_context_guardian_state():
            try:
                from core.context_guardian import get_context_guardian
                cg = get_context_guardian()
                return {"success": True, "guardian": cg.get_stats()}
            except Exception as e:
                logger.debug(f"Context guardian state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/claims/state")
        async def get_claims_state():
            try:
                from core.claims import get_claims_registry
                cr = get_claims_registry()
                stats = cr.get_statistics()
                return {"success": True, "claims": stats}
            except Exception as e:
                logger.debug(f"Claims state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/load_balancer/state")
        async def get_load_balancer_state():
            try:
                from core.cognitive_load_balancer import get_load_balancer
                lb = get_load_balancer()
                summary = lb.get_load_summary()
                return {"success": True, "load_balancer": summary}
            except Exception as e:
                logger.debug(f"Load balancer state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/autonomous_goals/state")
        async def get_autonomous_goals_state():
            try:
                from core.autonomous_goals import get_autonomous_goals
                agr = get_autonomous_goals()
                stats = agr.get_stats()
                actionable = [{"id": g.id, "title": g.title, "priority": g.priority.name, "progress": g.progress.progress_percent} for g in agr.get_actionable()[:10]]
                overdue = [{"id": g.id, "title": g.title, "deadline": str(g.deadline) if g.deadline else None} for g in agr.get_overdue()[:10]]
                return {"success": True, "goals": stats, "actionable": actionable, "overdue": overdue}
            except Exception as e:
                logger.debug(f"Autonomous goals state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/circadian/recommendations")
        async def get_circadian_recommendations():
            try:
                from core.circadian_rhythm import get_circadian_rhythm
                cr = get_circadian_rhythm()
                recs = cr.get_recommendations()
                return {"success": True, "recommendations": recs}
            except Exception as e:
                logger.debug(f"Circadian recommendations API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/emotional_state/style")
        async def get_emotional_state_style():
            try:
                from core.emotional_state import get_emotional_state_manager
                esm = get_emotional_state_manager()
                style = esm.get_response_style()
                return {"success": True, "style": {"detail_level": style.detail_level, "tone": style.tone, "energy": style.energy, "should_match_energy": style.should_match_energy, "should_be_careful": style.should_be_careful, "should_explore": style.should_explore, "should_provide_examples": style.should_provide_examples, "follow_up_questions": style.follow_up_questions}}
            except Exception as e:
                logger.debug(f"Emotional state style API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/autonomous_evolution/status")
        async def get_autonomous_evolution_status():
            try:
                from core.autonomous_evolution import get_autonomous_evolution
                evo = get_autonomous_evolution()
                pending = evo.get_pending_proposals()
                stats = evo.history.get_stats()
                return {"success": True, "evolution": {"pending_proposals": len(pending), "stats": stats}}
            except Exception as e:
                logger.debug(f"Autonomous evolution status API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/autonomous_evolution/history")
        async def get_autonomous_evolution_history():
            try:
                from core.autonomous_evolution import get_autonomous_evolution
                evo = get_autonomous_evolution()
                history = evo.get_change_history(months=12)
                return {"success": True, "history": history[-20:]}
            except Exception as e:
                logger.debug(f"Autonomous evolution history API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/autonomous_evolution/report")
        async def get_autonomous_evolution_report():
            try:
                from core.autonomous_evolution import get_autonomous_evolution
                evo = get_autonomous_evolution()
                report = evo.generate_evolution_report()
                return {"success": True, "report": report}
            except Exception as e:
                logger.debug(f"Autonomous evolution report API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/presence/state")
        async def get_presence_state():
            try:
                from core.active_presence import get_active_presence
                ap = get_active_presence()
                return {"success": True, "presence": ap.get_state()}
            except Exception as e:
                logger.debug(f"Presence state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/presence/observations")
        async def get_presence_observations():
            try:
                from core.active_presence import get_active_presence
                ap = get_active_presence()
                return {"success": True, "observations": ap.get_unacknowledged_observations()}
            except Exception as e:
                logger.debug(f"Presence observations API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/presence/checkin")
        async def get_presence_checkin():
            try:
                from core.active_presence import get_active_presence
                ap = get_active_presence()
                ci = ap.should_check_in()
                return {"success": True, "checkin": {"reason": ci.reason, "message": ci.message, "urgency": ci.urgency} if ci else None}
            except Exception as e:
                logger.debug(f"Presence checkin API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/anticipation/state")
        async def get_anticipation_state():
            try:
                from core.goal_anticipation import get_goal_anticipation
                ga = get_goal_anticipation()
                return {"success": True, "anticipation": ga.get_state()}
            except Exception as e:
                logger.debug(f"Anticipation state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/anticipation/pending")
        async def get_anticipation_pending():
            try:
                from core.goal_anticipation import get_goal_anticipation
                ga = get_goal_anticipation()
                pending = ga.get_pending_anticipations()
                return {"success": True, "pending": [{"description": a.description, "confidence": round(a.confidence, 2), "source": a.source.value, "reasoning": a.reasoning, "suggested_action": a.suggested_action} for a in pending]}
            except Exception as e:
                logger.debug(f"Anticipation pending API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/observer/state")
        async def get_observer_state():
            try:
                from core.autonomous_observer import get_autonomous_observer
                ao = get_autonomous_observer()
                return {"success": True, "observer": ao.get_state()}
            except Exception as e:
                logger.debug(f"Observer state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/observer/recent")
        async def get_observer_recent(limit: int = 20):
            try:
                from core.autonomous_observer import get_autonomous_observer
                ao = get_autonomous_observer()
                return {"success": True, "observations": ao.get_recent_observations(limit=limit)}
            except Exception as e:
                logger.debug(f"Observer recent API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/observer/habits")
        async def get_observer_habits():
            try:
                from core.autonomous_observer import get_autonomous_observer
                ao = get_autonomous_observer()
                return {"success": True, "habits": ao.get_learned_habits()}
            except Exception as e:
                logger.debug(f"Observer habits API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/observer/emotional_trajectory")
        async def get_observer_emotional_trajectory(limit: int = 20):
            try:
                from core.autonomous_observer import get_autonomous_observer
                ao = get_autonomous_observer()
                return {"success": True, "trajectory": ao.get_emotional_trajectory(limit=limit)}
            except Exception as e:
                logger.debug(f"Observer emotional trajectory API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/observer/observe_emotional")
        async def observe_emotional_tone(request):
            try:
                from core.autonomous_observer import get_autonomous_observer
                data = await request.json()
                tone = data.get("tone", "neutral")
                intensity = float(data.get("intensity", 0.5))
                trigger = data.get("trigger", "")
                ao = get_autonomous_observer()
                obs = ao.observe_emotional_tone(tone, intensity, trigger)
                return {"success": True, "observation": {"type": obs.obs_type.value, "content": obs.content, "salience": obs.salience.value}}
            except Exception as e:
                logger.debug(f"Observe emotional API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/anticipation/tick")
        async def anticipation_tick():
            try:
                from core.goal_anticipation import get_goal_anticipation
                ga = get_goal_anticipation()
                return {"success": True, "state": ga.tick()}
            except Exception as e:
                logger.debug(f"Anticipation tick API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}


    def _setup_butler_emotion_routes(self):
        @self.app.get("/api/presence/deep_work")
        async def get_presence_deep_work():
            try:
                from core.active_presence import get_active_presence
                ap = get_active_presence()
                state = ap.get_state()
                return {"success": True, "deep_work_minutes": state.get("deep_work_minutes", 0), "distraction_count": state.get("distraction_count", 0), "rhythm": state.get("rhythm", {})}
            except Exception as e:
                logger.debug(f"Presence deep work API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/emotion_regulation/state")
        async def get_emotion_regulation_state():
            try:
                from core.emotion_regulation import get_emotion_regulation
                ere = get_emotion_regulation()
                return {"success": True, "regulation": ere.get_state()}
            except Exception as e:
                logger.debug(f"Emotion regulation state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/allostasis/state")
        async def get_allostasis_state():
            try:
                from core.interoception import get_allostasis_predictor
                ap = get_allostasis_predictor()
                return {"success": True, "allostasis": ap.get_state()}
            except Exception as e:
                logger.debug(f"Allostasis state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/butler_personality/state")
        async def get_butler_personality_state():
            try:
                from core.butler_personality import get_butler_personality
                bp = get_butler_personality()
                return {"success": True, "personality": bp.get_state()}
            except Exception as e:
                logger.debug(f"Butler personality state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/butler_personality/context")
        async def get_butler_personality_context():
            try:
                from core.butler_personality import get_butler_personality
                bp = get_butler_personality()
                return {"success": True, "modifiers": bp.get_system_prompt_modifiers(), "guidance": bp.get_response_guidance()}
            except Exception as e:
                logger.debug(f"Butler personality context API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/butler_personality/observe")
        async def observe_butler_personality(request):
            try:
                from core.butler_personality import get_butler_personality
                data = await request.json()
                bp = get_butler_personality()
                state = bp.observe_interaction(
                    user_input=data.get("user_input", ""),
                    user_sentiment=data.get("sentiment", "neutral"),
                    user_energy=data.get("energy", "moderate"),
                    frustration=data.get("frustration", False),
                    gratitude=data.get("gratitude", False),
                )
                return {"success": True, "personality": {"mode": state.mode.value, "formality": state.formality.label,
                        "verbosity": state.verbosity.value, "warmth": state.warmth}}
            except Exception as e:
                logger.debug(f"Observe personality API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/butler_personality/goal")
        async def set_butler_goal_context(request):
            try:
                from core.butler_personality import get_butler_personality
                data = await request.json()
                bp = get_butler_personality()
                bp.set_goal_context(data.get("goal", ""))
                return {"success": True, "personality": bp.get_state()}
            except Exception as e:
                logger.debug(f"Set goal context API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/proactive_intelligence/state")
        async def get_proactive_intelligence_state():
            try:
                from core.proactive_intelligence import get_proactive_intelligence
                pi = get_proactive_intelligence()
                return {"success": True, "proactive": pi.get_state()}
            except Exception as e:
                logger.debug(f"Proactive intelligence state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/proactive_intelligence/patterns")
        async def get_proactive_patterns():
            try:
                from core.proactive_intelligence import get_proactive_intelligence
                pi = get_proactive_intelligence()
                return {"success": True, "patterns": pi.get_active_patterns()}
            except Exception as e:
                logger.debug(f"Proactive patterns API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/proactive_intelligence/observe")
        async def observe_proactive_intelligence(request):
            try:
                from core.proactive_intelligence import get_proactive_intelligence
                data = await request.json()
                pi = get_proactive_intelligence()
                intervention = pi.observe(
                    action_type=data.get("action_type", ""),
                    details=data.get("details", ""),
                    success=data.get("success", True),
                    context=data.get("context"),
                )
                if intervention:
                    return {"success": True, "intervention": {
                        "action": intervention.action, "reason": intervention.reason,
                        "message": intervention.message, "urgency": intervention.urgency.value,
                        "timing": intervention.timing.value, "confidence": intervention.confidence}}
                return {"success": True, "intervention": None}
            except Exception as e:
                logger.debug(f"Observe proactive intelligence API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/proactive_intelligence/accept")
        async def accept_proactive_intervention():
            try:
                from core.proactive_intelligence import get_proactive_intelligence
                pi = get_proactive_intelligence()
                pi.mark_accepted()
                return {"success": True}
            except Exception as e:
                logger.debug(f"Accept intervention API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/proactive_intelligence/dismiss")
        async def dismiss_proactive_intervention():
            try:
                from core.proactive_intelligence import get_proactive_intelligence
                pi = get_proactive_intelligence()
                pi.mark_dismissed()
                return {"success": True}
            except Exception as e:
                logger.debug(f"Dismiss intervention API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/contextual_memory/state")
        async def get_contextual_memory_state():
            try:
                from core.contextual_memory import get_contextual_memory
                cm = get_contextual_memory()
                return {"success": True, "memory": cm.get_state()}
            except Exception as e:
                logger.debug(f"Contextual memory state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/contextual_memory/style")
        async def get_contextual_memory_style():
            try:
                from core.contextual_memory import get_contextual_memory
                cm = get_contextual_memory()
                return {"success": True, "style": cm.get_communication_style()}
            except Exception as e:
                logger.debug(f"Contextual memory style API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/contextual_memory/learn")
        async def learn_contextual_preference(request):
            try:
                from core.contextual_memory import ContextualMemory, PreferenceCategory, get_contextual_memory
                cm = get_contextual_memory()
                data = await request.json()
                category = PreferenceCategory(data.get("category", "communication"))
                cm.learn_preference(
                    key=data.get("key", ""),
                    value=data.get("value", ""),
                    category=category,
                    confidence=data.get("confidence", 0.5),
                    source=data.get("source", "explicit"),
                )
                return {"success": True, "preferences_learned": len(cm._preferences)}
            except Exception as e:
                logger.debug(f"Learn preference API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/butler_orchestrator/state")
        async def get_butler_orchestrator_state():
            try:
                from core.butler_orchestrator import get_butler_orchestrator
                orch = get_butler_orchestrator()
                return {"success": True, "orchestrator": orch.get_state()}
            except Exception as e:
                logger.debug(f"Butler orchestrator state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/butler_orchestrator/context")
        async def get_butler_orchestrator_context():
            try:
                from core.butler_orchestrator import get_butler_orchestrator
                orch = get_butler_orchestrator()
                ctx = orch.get_coordination_context()
                return {"success": True, "context": ctx}
            except Exception as e:
                logger.debug(f"Butler orchestrator context API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/butler_orchestrator/coordinate")
        async def trigger_butler_coordination(request):
            try:
                from core.butler_orchestrator import get_butler_orchestrator
                from core.active_presence import get_active_presence
                from core.goal_anticipation import get_goal_anticipation
                from core.autonomous_observer import get_autonomous_observer
                from core.butler_personality import get_butler_personality
                from core.proactive_intelligence import get_proactive_intelligence
                from core.contextual_memory import get_contextual_memory
                orch = get_butler_orchestrator()
                presence_state = {}
                try:
                    presence_state = get_active_presence().get_state()
                except Exception:
                    pass
                anticipation_state = {}
                try:
                    antic = get_goal_anticipation()
                    pending = antic.get_pending_anticipations()
                    if pending:
                        anticipation_state = {"active": len(pending)}
                except Exception:
                    pass
                observer_state = {}
                try:
                    obs = get_autonomous_observer()
                    traj = obs.get_emotional_trajectory()
                    if traj:
                        observer_state = {"emotional_tone": traj[-1].get("tone", "neutral")}
                except Exception:
                    pass
                personality_state = {}
                try:
                    personality_state = get_butler_personality().get_state()
                except Exception:
                    pass
                proactive_state = {}
                try:
                    proactive_state = get_proactive_intelligence().get_state()
                except Exception:
                    pass
                contextual_state = {}
                try:
                    contextual_state = get_contextual_memory().get_state()
                except Exception:
                    pass
                action = orch.coordinate(
                    presence_state=presence_state,
                    anticipation_state=anticipation_state,
                    observer_state=observer_state,
                    personality_state=personality_state,
                    proactive_state=proactive_state,
                    contextual_state=contextual_state,
                )
                if action:
                    return {"success": True, "action": action.action, "reason": action.reason, "message": action.message, "priority": action.priority.name}
                return {"success": True, "action": None, "reason": "no coordination needed"}
            except Exception as e:
                logger.debug(f"Butler coordination API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/butler_orchestrator/accept")
        async def accept_butler_orchestration():
            try:
                from core.butler_orchestrator import get_butler_orchestrator
                orch = get_butler_orchestrator()
                orch.mark_accepted()
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/butler_orchestrator/dismiss")
        async def dismiss_butler_orchestration():
            try:
                from core.butler_orchestrator import get_butler_orchestrator
                orch = get_butler_orchestrator()
                orch.mark_dismissed()
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/butler_orchestrator/interruption")
        async def get_butler_orchestrator_interruption():
            try:
                from core.butler_orchestrator import get_butler_orchestrator
                orch = get_butler_orchestrator()
                interruption = orch.get_pending_interruption()
                return {"success": True, "interruption": interruption}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/butler_session/state")
        async def get_butler_session_state():
            try:
                from core.butler_session import get_butler_session
                session = get_butler_session()
                return {"success": True, "session": session.get_state()}
            except Exception as e:
                logger.debug(f"Butler session state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/butler_session/start")
        async def start_butler_session(request):
            try:
                from core.butler_session import get_butler_session
                session = get_butler_session()
                data = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
                greeting = session.start_session(data.get("message", ""))
                return {"success": True, "greeting": greeting}
            except Exception as e:
                logger.debug(f"Butler session start API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/butler_session/end")
        async def end_butler_session():
            try:
                from core.butler_session import get_butler_session
                session = get_butler_session()
                session.end_session(reason="user_ended")
                return {"success": True}
            except Exception as e:
                logger.debug(f"Butler session end API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/butler_voice/greeting")
        async def get_butler_voice_greeting():
            try:
                from core.butler_voice import get_butler_voice, Situation
                from core.butler_personality import get_butler_personality
                voice = get_butler_voice()
                bp = get_butler_personality()
                import datetime
                hour = datetime.datetime.now().hour
                if 5 <= hour < 12:
                    situation = Situation.GREETING_MORNING
                elif 17 <= hour < 22:
                    situation = Situation.GREETING_EVENING
                else:
                    situation = Situation.GREETING_RETURN
                greeting = voice.get_greeting(bp, situation)
                ack = voice.get_acknowledgment(bp)
                transition = voice.get_transition(bp)
                return {
                    "success": True,
                    "greeting": greeting,
                    "acknowledgment": ack,
                    "transition": transition,
                    "fillers_to_strip": voice.get_fillers_to_strip(bp),
                }
            except Exception as e:
                logger.debug(f"Butler voice API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/butler_voice/phrase")
        async def get_butler_voice_phrase(request):
            try:
                from core.butler_voice import get_butler_voice, Situation
                from core.butler_personality import get_butler_personality
                voice = get_butler_voice()
                bp = get_butler_personality()
                data = await request.json()
                situation_name = data.get("situation", "success_confirm")
                try:
                    situation = Situation(situation_name)
                except ValueError:
                    situation = Situation.SUCCESS_CONFIRM
                phrase = voice.get_phrase(situation, bp)
                return {"success": True, "phrase": phrase, "situation": situation.value}
            except Exception as e:
                logger.debug(f"Butler voice phrase API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/butler_voice/state")
        async def get_butler_voice_state():
            try:
                from core.butler_voice import get_butler_voice
                voice = get_butler_voice()
                return {"success": True, "voice": voice.get_state()}
            except Exception as e:
                logger.debug(f"Butler voice state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}


    def _setup_sentiment_oracle_routes(self):
        @self.app.post("/api/sentiment/analyze")
        async def analyze_sentiment(request):
            try:
                from core.sentiment_analyzer import get_sentiment_analyzer
                data = await request.json()
                text = data.get("text", "")
                context = data.get("context", "")
                result = get_sentiment_analyzer().analyze(text, context)
                return {
                    "success": True,
                    "sentiment": result.sentiment.value,
                    "intensity": result.intensity,
                    "confidence": result.confidence,
                    "needs": [n.value for n in result.needs],
                    "signals": result.signals,
                    "escalation_risk": result.escalation_risk,
                    "requires_response_change": result.requires_response_change,
                }
            except Exception as e:
                logger.debug(f"Sentiment analysis API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/sentiment/state")
        async def get_sentiment_state():
            try:
                from core.sentiment_analyzer import get_sentiment_analyzer
                sa = get_sentiment_analyzer()
                return {"success": True, "sentiment": sa.get_state()}
            except Exception as e:
                logger.debug(f"Sentiment state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/sentiment/trajectory")
        async def get_sentiment_trajectory():
            try:
                from core.sentiment_analyzer import get_sentiment_analyzer
                sa = get_sentiment_analyzer()
                traj = sa.get_trajectory()
                return {
                    "success": True,
                    "trend": traj.trend,
                    "trend_confidence": traj.trend_confidence,
                    "is_escalating": traj.is_escalating(),
                    "recent_sentiments": sa.get_recent_sentiments(limit=10),
                }
            except Exception as e:
                logger.debug(f"Sentiment trajectory API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/emotional_memory/state")
        async def get_emotional_memory_state():
            try:
                from core.emotional_memory import get_emotional_memory
                em = get_emotional_memory()
                return {"success": True, "emotional_memory": em.get_state()}
            except Exception as e:
                logger.debug(f"Emotional memory state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/affective_context/state")
        async def get_affective_context_state():
            try:
                from core.affective_context import get_affective_context
                ac = get_affective_context()
                return {"success": True, "affective_context": ac.get_state()}
            except Exception as e:
                logger.debug(f"Affective context state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/oracle/state")
        async def get_oracle_state():
            try:
                from core.oracle_bridge import get_oracle_bridge
                ob = get_oracle_bridge()
                return {"success": True, "oracle": ob.get_state()}
            except Exception as e:
                logger.debug(f"Oracle state API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/oracle/forecast")
        async def get_oracle_forecast(domain: str = "", horizon: int = 30):
            try:
                from core.oracle_bridge import get_oracle_bridge
                ob = get_oracle_bridge()
                forecast = ob.query_forecast(domain=domain, horizon_days=horizon)
                return {
                    "success": True,
                    "forecast": {
                        "domain": forecast.domain,
                        "summary": forecast.summary,
                        "confidence": forecast.confidence,
                        "uncertainty": forecast.uncertainty,
                        "risk_level": forecast.risk_level,
                        "key_signals": forecast.key_signals,
                        "horizon_days": forecast.time_horizon_days,
                    },
                }
            except Exception as e:
                logger.debug(f"Oracle forecast API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/oracle/signals")
        async def get_oracle_cross_domain():
            try:
                from core.oracle_bridge import get_oracle_bridge
                ob = get_oracle_bridge()
                signals = ob.get_cross_domain_signals()
                return {"success": True, "signals": signals}
            except Exception as e:
                logger.debug(f"Oracle signals API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/consciousness/state")
        async def get_consciousness_state():
            state = {"success": True, "consciousness": {}}
            modules = [
                ("interoception", "core.interoception", "get_interoception_engine"),
                ("workspace", "core.global_workspace", "get_global_workspace"),
                ("self_model", "core.recursive_self_model", "get_recursive_self_model"),
                ("stream", "core.stream_consciousness", "get_stream_of_consciousness"),
                ("emotion_regulation", "core.emotion_regulation", "get_emotion_regulation"),
                ("allostasis", "core.interoception", "get_allostasis_predictor"),
                ("emotional_memory", "core.emotional_memory", "get_emotional_memory"),
                ("affective_context", "core.affective_context", "get_affective_context"),
                ("predictive", "core.predictive_processing", "get_predictive_processor"),
                ("cognitive", "core.cognitive_state_machine", "get_cognitive_state_machine"),
                ("circadian", "core.circadian_rhythm", "get_circadian_rhythm"),
                ("world_model", "core.world_model", "get_world_model"),
                ("jepa_world", "core.jepa_world_model", "get_jepa_world_model"),
            ]
            for key, module_path, getter_name in modules:
                try:
                    mod = __import__(module_path, fromlist=[getter_name])
                    getter = getattr(mod, getter_name)
                    instance = getter()
                    module_state = instance.get_state()
                    module_state["module"] = key
                    module_state["timestamp"] = __import__("time").time()
                    state["consciousness"][key] = module_state
                except Exception as e:
                    state["consciousness"][key] = {"module": key, "error": safe_error_message(e)}
            try:
                from core.bio_integration import get_bio_integration
                bio = get_bio_integration()
                bio_state = bio.tick(dt=0.0)
                state["consciousness"]["bio"] = {
                    "arousal": getattr(bio_state, "arousal", 0.5),
                    "valence": getattr(bio_state, "valence", 0.0),
                    "stress_phase": getattr(bio_state, "stress_phase", "none"),
                    "is_homeostatic": getattr(bio_state, "is_homeostatic", True),
                    "cognitive_capacity": getattr(bio_state, "cognitive_capacity", 0.8),
                }
            except Exception:
                state["consciousness"]["bio"] = {"error": "unavailable"}
            try:
                from core.circadian_rhythm import get_circadian_rhythm
                cr = get_circadian_rhythm()
                cr_state = cr.get_state()
                state["consciousness"]["circadian"] = {
                    "operational_phase": cr_state.get("operational_phase"),
                    "processing_fatigue": cr_state.get("processing_fatigue", 0.0),
                    "cognitive_capacity": cr.get_cognitive_capacity(),
                    "recommendations": cr.get_recommendations(),
                }
            except Exception:
                state["consciousness"]["circadian"] = {"error": "unavailable"}
            try:
                from core.cognitive_psychology import CognitivePsychologyEngine
                psych = CognitivePsychologyEngine.get_instance()
                state["consciousness"]["cognitive_psychology"] = psych.to_dict()
            except Exception:
                state["consciousness"]["cognitive_psychology"] = {"error": "unavailable"}
            try:
                from core.emotional_state import get_emotional_state_manager
                esm = get_emotional_state_manager()
                state["consciousness"]["emotional_state"] = {
                    "current_mood": esm._current_mood.value if esm._current_mood else "unknown",
                    "sentiment": round(esm._sentiment_score, 3) if hasattr(esm, '_sentiment_score') else 0.0,
                    "engagement": esm._current_engagement.value if esm._current_engagement else "unknown",
                }
            except Exception:
                state["consciousness"]["emotional_state"] = {"error": "unavailable"}
            return state


    def _setup_refine_evolution_routes(self):
        @self.app.get("/api/refine/status")
        async def get_refine_status():
            try:
                from core.self_refine import get_self_refine_agent, has_self_refine_agent
                
                if has_self_refine_agent():
                    agent = get_self_refine_agent()
                    stats = agent.get_stats()
                    pending = agent.get_pending_tasks(limit=10)
                    
                    return {
                        "success": True,
                        "active": True,
                        "running": agent._running,
                        "statistics": {
                            "total": stats.get("completed_tasks", 0) + stats.get("pending_tasks", 0),
                            "pending": stats.get("pending_tasks", 0),
                            "applied": stats.get("completed_tasks", 0),
                            "failed": stats.get("failed_tasks", 0),
                        },
                        "state": stats.get("state", "idle"),
                        "pending_changes": [
                            {
                                "id": t.id,
                                "file": t.file_path,
                                "line": getattr(t, 'line_number', 0),
                                "severity": "medium",
                                "confidence": getattr(t, 'confidence', 0.5),
                                "type": t.category,
                                "risk": t.risk_level,
                                "explanation": t.title,
                                "original": getattr(t, 'original_code', '')[:200] if getattr(t, 'original_code', None) else "",
                                "proposed": getattr(t, 'proposed_code', '')[:200] if getattr(t, 'proposed_code', None) else "",
                                "status": t.status,
                                "created": t.created_at.isoformat() if hasattr(t, 'created_at') and t.created_at else None,
                            }
                            for t in pending
                        ],
                        "pending_count": len(pending),
                    }
                
                from core.auto_refine import get_refine_manager
                try:
                    manager = get_refine_manager(self.agent.llm)
                except ValueError:
                    return {
                        "success": False,
                        "error": "Self-refine not initialized",
                        "hint": "The refine system needs to be initialized with an LLM client.",
                        "statistics": {"total": 0, "pending": 0, "applied": 0, "failed": 0},
                        "pending_changes": [],
                        "pending_count": 0,
                    }

                stats = manager.get_statistics()

                pending = manager.get_pending_changes()
                pending_changes = []
                for change in pending[:10]:
                    pending_changes.append({
                        "id": change.id,
                        "file": change.file_path,
                        "line": change.line_number,
                        "severity": change.severity,
                        "confidence": round(change.confidence, 2),
                        "type": change.change_type,
                        "risk": change.risk_level,
                        "explanation": change.explanation[:100] + "..." if len(change.explanation) > 100 else change.explanation,
                        "original": change.original_code[:200] + "..." if len(change.original_code) > 200 else change.original_code,
                        "proposed": change.proposed_code[:200] + "..." if len(change.proposed_code) > 200 else change.proposed_code,
                        "status": change.status.value,
                        "created": change.created_at.isoformat() if change.created_at else None,
                    })

                return {
                    "success": True,
                    "active": False,
                    "statistics": stats,
                    "pending_changes": pending_changes,
                    "pending_count": len(pending),
                }
            except Exception as e:
                logger.error(f"Refine status error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/refine/scan")
        async def run_refine_scan(request: Request):
            try:
                from core.self_refine import has_self_refine_agent, get_self_refine_agent
                
                if has_self_refine_agent():
                    agent = get_self_refine_agent()
                    result = await agent.scan_now()
                    
                    if not result.get("success"):
                        return result
                    
                    return {
                        "success": True,
                        "active": True,
                        "running": agent._running,
                        "files_scanned": result.get("files_scanned", 0),
                        "issues_found": result.get("issues_found", 0),
                        "duration": 0,
                        "summary": result.get("patterns", {}),
                        "pending_count": len(agent.get_pending_tasks()),
                        "tasks_generated": result.get("tasks_generated", 0),
                    }

                from core.auto_refine import get_refine_manager
                try:
                    manager = get_refine_manager(self.agent.llm)
                except ValueError:
                    return {"success": False, "error": "Self-refine not initialized", "files_scanned": 0, "issues_found": 0}

                data = await request.json() if request.headers.get("content-type") == "application/json" else {}
                include_ai = data.get("include_ai", False)

                result = await manager.scan_codebase(include_ai=include_ai)

                pending_count = len(manager.get_pending_changes())

                return {
                    "success": True,
                    "active": False,
                    "files_scanned": result.files_scanned,
                    "issues_found": len(result.issues_found),
                    "duration": round(result.duration_seconds, 2),
                    "summary": result.summary,
                    "pending_count": pending_count,
                    "top_issues": [
                        {
                            "id": issue.id,
                            "file": issue.file_path,
                            "line": issue.line_number,
                            "severity": issue.severity.value,
                            "type": issue.issue_type.value,
                            "title": issue.title,
                        }
                        for issue in result.issues_found[:20]
                    ],
                }
            except Exception as e:
                logger.error(f"Refine scan error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/refine/apply")
        async def apply_refine_change(request: Request):
            _audit_log(request, "refine_apply")
            try:
                from core.self_refine import has_self_refine_agent, get_self_refine_agent
                
                if has_self_refine_agent():
                    agent = get_self_refine_agent()
                    data = await request.json()
                    data, validation_error = validate_request_fields(
                        data, required=["change_id"], max_string_length=256
                    )
                    if validation_error:
                        return {"success": False, "error": validation_error}
                    change_id = data["change_id"]
                    
                    task = agent._tasks.get(change_id)
                    if not task:
                        return {"success": False, "error": f"Task {change_id} not found"}
                    
                    success, message = await agent._execute_task(task)
                    return {"success": success, "message": message, "change_id": change_id}

                from core.auto_refine import get_refine_manager
                try:
                    manager = get_refine_manager(self.agent.llm)
                except ValueError:
                    return {"success": False, "error": "Self-refine not initialized"}

                data = await request.json()
                change_id = data.get("change_id")

                if change_id:
                    success, message = manager.apply_change(change_id)
                    return {"success": success, "message": message, "change_id": change_id}
                else:
                    results = await manager.apply_approved_changes(auto_permission=False)
                    return {
                        "success": True,
                        "results": [{"id": r[0], "success": r[1], "message": r[2]} for r in results],
                    }
            except Exception as e:
                logger.error(f"Refine apply error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/refine/approve")
        async def approve_refine_change(request: Request):
            try:
                from core.self_refine import has_self_refine_agent
                
                if has_self_refine_agent():
                    return {"success": True, "message": "Self-refine agent auto-approves low-risk tasks"}

                from core.auto_refine import get_refine_manager
                try:
                    manager = get_refine_manager(self.agent.llm)
                except ValueError:
                    return {"success": False, "error": "Self-refine not initialized"}

                data = await request.json()
                change_id = data.get("change_id")

                if not change_id:
                    return {"success": False, "error": "change_id required"}

                success = manager.approve_change(change_id)
                return {"success": success, "change_id": change_id}

            except Exception as e:
                logger.error(f"Refine approve error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/refine/reject")
        async def reject_refine_change(request: Request):
            try:
                from core.auto_refine import get_refine_manager
                try:
                    manager = get_refine_manager(self.agent.llm)
                except ValueError:
                    return {"success": False, "error": "Self-refine not initialized"}

                data = await request.json()
                change_id = data.get("change_id")

                if not change_id:
                    return {"success": False, "error": "change_id required"}

                success = manager.reject_change(change_id)
                return {"success": success, "change_id": change_id}

            except Exception as e:
                logger.error(f"Refine reject error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/refine/rollback")
        async def rollback_refine_change(request: Request):
            try:
                from core.self_refine import has_self_refine_agent, get_self_refine_agent
                
                if has_self_refine_agent():
                    agent = get_self_refine_agent()
                    data = await request.json()
                    change_id = data.get("change_id")
                    
                    if not change_id:
                        return {"success": False, "error": "change_id required"}
                    
                    success, message = agent.rollback_task(change_id)
                    return {"success": success, "message": message, "change_id": change_id}

                from core.auto_refine import get_refine_manager
                try:
                    manager = get_refine_manager(self.agent.llm)
                except ValueError:
                    return {"success": False, "error": "Self-refine not initialized"}

                data = await request.json()
                change_id = data.get("change_id")

                if not change_id:
                    return {"success": False, "error": "change_id required"}

                success, message = manager.rollback_change(change_id)
                return {"success": success, "message": message, "change_id": change_id}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/evolution/status")
        async def get_evolution_status():
            try:
                from core.evolution_orchestrator import get_evolution_orchestrator
                orchestrator = get_evolution_orchestrator()

                if hasattr(orchestrator, 'get_status'):
                    stats = orchestrator.get_status()
                    return {"success": True, **stats}
                return {"success": True, "status": "unknown"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/event-bus/stats")
        async def get_event_bus_stats():
            try:
                from core.event_bus import get_event_bus
                bus = get_event_bus()
                stats = bus.get_stats()
                return {"success": True, **stats}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/goal-tracker/state")
        async def get_goal_tracker_state():
            try:
                from core.goal_tracker import get_goal_tracker
                tracker = get_goal_tracker()
                goals = tracker.get_active_goals() if hasattr(tracker, 'get_active_goals') else []
                return {"success": True, "goals": goals}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/context-compressor/stats")
        async def get_context_compressor_stats():
            try:
                from core.context_compressor import get_context_compressor
                compressor = get_context_compressor()
                return {"success": True, "available": True}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/memory-protection/state")
        async def get_memory_protection_state():
            try:
                from core.memory_protection import get_memory_protection
                protection = get_memory_protection()
                protected = protection.list_protected() if hasattr(protection, 'list_protected') else []
                return {"success": True, "protected_count": len(protected), "protected": protected[:20]}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/context-engine/state")
        async def get_context_engine_state():
            try:
                from core.unified_context_engine import get_context_engine
                engine = get_context_engine()
                return {"success": True, "available": True}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/bioevolve/status")
        async def get_bioevolve_status():
            try:
                from core.evolution_orchestrator import get_evolution_orchestrator
                orchestrator = get_evolution_orchestrator()
                status = orchestrator.get_bioevolve_status()
                return {"success": True, **status}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/bioevolve/cycle")
        async def run_bioevolve_cycle(iterations: int = 20):
            try:
                from core.evolution_orchestrator import get_evolution_orchestrator
                orchestrator = get_evolution_orchestrator()
                if orchestrator._bioevolve_engine and orchestrator._bioevolve_engine.running:
                    return {"success": False, "error": "BioEvolve engine already running"}
                result = orchestrator.run_bioevolve_cycle(iterations)
                return {"success": True, **result}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}


    def _setup_insights_music_routes(self):
        @self.app.get("/api/insights/recent")
        async def get_recent_insights(limit: int = 500):
            try:
                insights = []
                seen = set()

                # Primary source: insight_review (all indexed insights)
                try:
                    from core.insight_review import get_insight_review
                    review = get_insight_review()
                    all_insights = review.get_all_insights()
                    for insight in all_insights[:limit]:
                        insight_id = insight.get("id", "")
                        if insight_id in seen:
                            continue
                        seen.add(insight_id)
                        insights.append({
                            "id": insight_id,
                            "source": insight.get("source", "unknown"),
                            "type": insight.get("type", "unknown"),
                            "category": insight.get("category", ""),
                            "problem": insight.get("problem", ""),
                            "content": insight.get("content", ""),
                            "summary": insight.get("summary", ""),
                            "title": insight.get("title", ""),
                            "relevance": insight.get("relevance", 0),
                            "timestamp": insight.get("timestamp", ""),
                        })
                except Exception as e:
                    logger.debug(f"Error getting insight_review insights: {e}")

                # Fallback: insight_bridge (recent processed)
                try:
                    from core.insight_bridge import get_insight_bridge
                    bridge = get_insight_bridge()
                    
                    if hasattr(bridge, 'processed_insights') and bridge.processed_insights:
                        for insight in list(bridge.processed_insights)[-50:]:
                            insight_id = insight.id if hasattr(insight, 'id') else str(hash(str(insight)))
                            if insight_id in seen:
                                continue
                            seen.add(insight_id)
                            
                            raw = insight.raw_insight if hasattr(insight, 'raw_insight') else ""
                            summary = insight.knowledge_for_mankind if hasattr(insight, 'knowledge_for_mankind') else raw
                            problem = insight.source_problem if hasattr(insight, 'source_problem') else ""
                            category = insight.source_category if hasattr(insight, 'source_category') else ""
                            insight_type = insight.insight_type.value if hasattr(insight, 'insight_type') else "unknown"
                            relevance = insight.evolution_relevance.value if hasattr(insight, 'evolution_relevance') else 0
                            ts = insight.timestamp.isoformat() if hasattr(insight, 'timestamp') and insight.timestamp else ""
                            
                            if raw and "No LLM available" not in raw:
                                insights.append({
                                    "id": insight_id,
                                    "source": "insight_bridge",
                                    "type": insight_type,
                                    "category": category,
                                    "problem": problem,
                                    "content": raw,
                                    "summary": summary if summary else "",
                                    "title": insight.title if hasattr(insight, 'title') else "",
                                    "relevance": relevance,
                                    "timestamp": ts,
                                })
                    
                    if hasattr(bridge, 'knowledge_contributions') and bridge.knowledge_contributions:
                        for contrib in list(bridge.knowledge_contributions)[-20:]:
                            cid = contrib.id if hasattr(contrib, 'id') else str(hash(str(contrib)))
                            if cid in seen:
                                continue
                            seen.add(cid)
                            
                            insights.append({
                                "id": cid,
                                "source": "knowledge",
                                "type": "contribution",
                                "category": "shared_knowledge",
                                "problem": contrib.problem_context if hasattr(contrib, 'problem_context') else "",
                                "content": contrib.mathematical_content if hasattr(contrib, 'mathematical_content') else "",
                                "summary": contrib.summary if hasattr(contrib, 'summary') and contrib.summary else "",
                                "title": contrib.title if hasattr(contrib, 'title') else "",
                                "relevance": 3,
                                "timestamp": contrib.timestamp.isoformat() if hasattr(contrib, 'timestamp') and contrib.timestamp else "",
                            })
                except Exception as e:
                    logger.debug(f"Error getting insight_bridge insights: {e}")

                try:
                    from core.erdos_engine import get_erdos_engine
                    engine = get_erdos_engine()
                    if hasattr(engine, 'get_insights'):
                        discoveries = engine.get_insights()[:20]
                        for i, d in enumerate(discoveries):
                            dstr = str(d)
                            if "No LLM available" not in dstr and dstr not in seen:
                                seen.add(dstr)
                                insights.append({
                                    "id": f"erdos_{i}",
                                    "source": "erdos",
                                    "type": "math_discovery",
                                    "category": "mathematics",
                                    "problem": "",
                                    "content": dstr,
                                    "summary": "",
                                    "relevance": 2,
                                    "timestamp": "",
                                })
                except Exception as e:
                    logger.debug(f"Error getting erdos insights: {e}")

                try:
                    from core.daydream_engine import get_daydream_engine
                    engine = get_daydream_engine()
                    if hasattr(engine, 'get_status'):
                        status = engine.get_status()
                        for i, d in enumerate(status.get('recent_insights', [])[:10]):
                            dstr = str(d)
                            dhash = hash(dstr)
                            if dhash not in seen:
                                seen.add(dhash)
                                insights.append({
                                    "id": f"daydream_{i}",
                                    "source": "daydream",
                                    "type": "exploration",
                                    "category": "exploration",
                                    "problem": "",
                                    "content": dstr,
                                    "summary": "",
                                    "relevance": 1,
                                    "timestamp": "",
                                })
                except Exception as e:
                    logger.debug(f"Error getting daydream insights: {e}")

                insights.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                return {"success": True, "insights": insights, "total": len(insights)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/insights/search")
        async def search_insights(q: str = "", source: str = "", type: str = "", limit: int = 100):
            """Search insights with filters."""
            try:
                from core.insight_review import get_insight_review
                review = get_insight_review()
                
                # Build filters
                filters = {}
                if source:
                    filters["source"] = source
                if type:
                    filters["type"] = type
                
                results = review.search(q, filters if filters else None)
                
                return {
                    "success": True,
                    "results": results[:limit] if limit > 0 else results,
                    "total": len(results),
                    "query": q,
                    "filters": filters,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/insights/heatmap")
        async def get_insight_heatmap(limit: int = 100):
            """Get word frequency heatmap for insights."""
            try:
                from core.insight_review import get_insight_review
                review = get_insight_review()
                
                words = review.get_word_heatmap(limit)
                
                return {
                    "success": True,
                    "words": [{"word": w, "count": c} for w, c in words],
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/insights/stats")
        async def get_insight_stats():
            """Get insight review statistics."""
            try:
                from core.insight_review import get_insight_review
                review = get_insight_review()
                
                return {
                    "success": True,
                    "stats": review.get_stats(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/insights/sources")
        async def get_insight_sources():
            """Get list of insight sources."""
            try:
                from core.insight_review import get_insight_review
                review = get_insight_review()
                
                return {
                    "success": True,
                    "sources": review.get_sources(),
                    "types": review.get_types(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/insights/reindex")
        async def reindex_insights():
            """Reindex all insights from bridge."""
            try:
                from core.insight_review import get_insight_review
                review = get_insight_review()
                
                count = review.index_from_bridge()
                
                return {
                    "success": True,
                    "indexed": count,
                    "message": f"Indexed {count} insights from bridge",
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/insights/hash/{content_hash}")
        async def get_insight_by_hash(content_hash: str):
            """Get insight by content hash."""
            try:
                from core.insight_review import get_insight_review
                review = get_insight_review()
                
                insight = review.get_by_hash(content_hash)
                
                if insight:
                    return {"success": True, "insight": insight}
                return {"success": False, "error": "Insight not found"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/insights/similar/{content_hash}")
        async def get_similar_insights(content_hash: str, limit: int = 10):
            """Find similar insights by content hash."""
            try:
                from core.insight_review import get_insight_review
                review = get_insight_review()
                
                similar = review.find_similar_by_hash(content_hash, limit)
                
                return {
                    "success": True,
                    "similar": similar,
                    "count": len(similar),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/insights/find-similar")
        async def find_similar_insights_content(request: Request):
            """Find similar insights by content text."""
            try:
                from core.insight_review import get_insight_review
                review = get_insight_review()
                
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["content"], optional=["threshold", "limit"],
                    max_string_length=5000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                content = data["content"]
                threshold = data.get("threshold", 0.3)
                limit = data.get("limit", 20)
                
                similar = review.find_by_content(content, threshold=threshold)[:limit]
                
                return {
                    "success": True,
                    "similar": similar,
                    "count": len(similar),
                    "content_hash": review.get_hash(content),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/insights/word/{word}")
        async def get_insights_by_word(word: str, limit: int = 20):
            """Get insights containing a specific word."""
            try:
                from core.insight_review import get_insight_review
                review = get_insight_review()
                
                insights = review.get_word_context(word, limit)
                
                return {
                    "success": True,
                    "insights": insights,
                    "word": word,
                    "count": len(insights),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/music/compositions")
        async def get_music_compositions(limit: int = 20):
            """Get recent music compositions from the always-on music creation engine."""
            try:
                from core.music_creation_engine import get_music_creation_engine
                engine = get_music_creation_engine()
                
                compositions = engine.get_recent_compositions(limit)
                
                return {
                    "success": True,
                    "compositions": compositions,
                    "stats": engine.get_stats(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/music/compositions/{comp_id}/audio")
        async def get_composition_audio(comp_id: str):
            """Get audio waveform data for a composition."""
            try:
                from core.music_creation_engine import get_music_creation_engine
                import numpy as np
                
                engine = get_music_creation_engine()
                
                # Find the composition
                composition = None
                for c in engine.compositions:
                    if c.id == comp_id:
                        composition = c
                        break
                
                if not composition:
                    return {"success": False, "error": "Composition not found"}
                
                import base64
                import io
                import wave
                from pathlib import Path
                
                # First try loading from disk (audio is saved there to save memory)
                music_dir = Path.home() / ".jarvis" / "music"
                wav_path = music_dir / f"{comp_id}.wav"
                
                if wav_path.exists():
                    # Read the ENTIRE WAV file (including headers) for decodeAudioData
                    wav_bytes = wav_path.read_bytes()
                    audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
                    
                    # Get metadata
                    with wave.open(str(wav_path), 'rb') as wav_file:
                        sample_rate = wav_file.getframerate()
                        n_frames = wav_file.getnframes()
                        duration = n_frames / sample_rate
                    
                    return {
                        "success": True,
                        "audio": audio_base64,
                        "sample_rate": sample_rate,
                        "duration": duration,
                        "format": "wav",
                        "source": "disk",
                    }
                
                # Check if MusicGen audio is available in memory
                if hasattr(composition, 'audio_data') and composition.audio_data is not None:
                    # Use MusicGen audio directly
                    audio = composition.audio_data
                    sample_rate = composition.sample_rate if hasattr(composition, 'sample_rate') else 24000
                    
                    # Normalize
                    max_val = np.max(np.abs(audio))
                    if max_val > 0:
                        audio = audio / max_val * 0.8
                    
                    # Convert to 16-bit PCM
                    audio_int = (audio * 32767).astype(np.int16)
                    
                    # Create WAV in memory
                    import base64
                    import io
                    import wave
                    
                    buffer = io.BytesIO()
                    with wave.open(buffer, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_int.tobytes())
                    
                    wav_bytes = buffer.getvalue()
                    audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
                    
                    return {
                        "success": True,
                        "audio": audio_base64,
                        "sample_rate": sample_rate,
                        "duration": len(audio) / sample_rate,
                        "format": "wav",
                        "source": "musicgen",
                    }
                
                # Fallback: Generate audio from measures
                measures = composition.measures
                if not measures:
                    return {"success": False, "error": "No measure data"}
                
                # Synthesize audio using the same approach as AcousticTokenGenerator
                sample_rate = 24000
                tempo = composition.tempo
                beats_per_measure = 4
                measure_duration = (60.0 / tempo) * beats_per_measure
                
                # Calculate total duration
                total_duration = measure_duration * len(measures)
                total_samples = int(total_duration * sample_rate)
                audio = np.zeros(total_samples, dtype=np.float32)
                
                # Note frequencies
                note_freqs = {
                    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
                    'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
                    'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88,
                }
                
                # Style-specific timbre
                style_harmonics = {
                    'baroque': [1.0, 0.5, 0.3, 0.2, 0.1],
                    'classical': [1.0, 0.6, 0.35, 0.15, 0.08],
                    'romantic': [1.0, 0.7, 0.4, 0.25, 0.12],
                }
                harmonics = style_harmonics.get(composition.style.value, style_harmonics['classical'])
                
                t = np.arange(total_samples) / sample_rate
                
                for m_idx, measure in enumerate(measures):
                    measure_start = m_idx * measure_duration
                    notes = measure.get("notes", [])
                    
                    for note_data in notes:
                        pitch = note_data.get("pitch", "C4")
                        duration_beats = note_data.get("duration", 1.0)
                        velocity = note_data.get("velocity", 80) / 127.0
                        
                        # Parse pitch
                        note_name = pitch[:-1] if pitch[-1].isdigit() else pitch
                        octave = int(pitch[-1]) if pitch[-1].isdigit() else 4
                        base_freq = note_freqs.get(note_name, 261.63)
                        freq = base_freq * (2 ** (octave - 4))
                        
                        note_duration = (60.0 / tempo) * duration_beats
                        start_sample = int(measure_start * sample_rate)
                        end_sample = min(start_sample + int(note_duration * sample_rate), total_samples)
                        
                        note_samples = end_sample - start_sample
                        if note_samples <= 0:
                            continue
                        
                        note_t = t[start_sample:end_sample]
                        
                        # ADSR envelope
                        attack = int(0.02 * sample_rate)
                        decay = int(0.1 * sample_rate)
                        sustain = 0.7
                        release = int(0.1 * sample_rate)
                        
                        envelope = np.ones(note_samples)
                        if attack < note_samples:
                            envelope[:attack] = np.linspace(0, 1, attack)
                        if attack + decay < note_samples:
                            envelope[attack:attack+decay] = np.linspace(1, sustain, decay)
                        if release < note_samples:
                            envelope[-release:] = np.linspace(sustain, 0, release)
                        
                        # Harmonic synthesis
                        note_wave = np.zeros(note_samples)
                        for h_idx, h_amp in enumerate(harmonics):
                            harmonic_freq = freq * (h_idx + 1)
                            detune = 1.0 + 0.001 * np.sin(2 * np.pi * 0.5 * note_t)
                            note_wave += h_amp * envelope * velocity * np.sin(2 * np.pi * harmonic_freq * detune * note_t)
                        
                        audio[start_sample:end_sample] += note_wave
                
                # Normalize
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.8
                
                # Convert to 16-bit PCM
                audio_int = (audio * 32767).astype(np.int16)
                
                # Return as base64 for easy transport
                import base64
                import io
                import wave
                
                # Create WAV in memory
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_int.tobytes())
                
                wav_bytes = buffer.getvalue()
                audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
                
                return {
                    "success": True,
                    "audio": audio_base64,
                    "sample_rate": sample_rate,
                    "duration": total_duration,
                    "format": "wav",
                }
            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"success": False, "error": safe_error_message(e)}
        # Sibling-route handlers called from other route groups.
        self._route_handlers["get_recent_insights"] = get_recent_insights


    def _setup_agent_state_routes(self):
        @self.app.get("/api/jarvis/state")
        async def get_jarvis_state():
            """Get real-time Jarvis state including daydream status and current activity."""
            try:
                state = {
                    "status": "idle",
                    "activity": "Ready",
                    "thought": None,
                    "subsystem_active": None,
                    "tools_used": [],
                    "daydreaming": False,
                    "idle_time_seconds": 0,
                }
                
                # Check if currently processing
                is_thinking = getattr(self.agent, '_is_thinking', False)
                if is_thinking:
                    state["status"] = "thinking"
                    state["activity"] = "Processing request..."
                
                # Check daydream status
                try:
                    from core.daydream_engine import get_daydream_engine
                    engine = get_daydream_engine()
                    ds = engine.get_status()
                    if ds.get("running") and ds.get("can_daydream"):
                        state["daydreaming"] = True
                        state["status"] = "daydreaming"
                        state["activity"] = "Exploring ideas..."
                        topics = ds.get("exploration_topics", [])
                        if topics:
                            last_topic = topics[-1]
                            if last_topic and last_topic != "Problem unknown":
                                state["thought"] = last_topic
                    state["idle_time_seconds"] = 0
                    if hasattr(engine, '_idle_since'):
                        from datetime import datetime
                        idle_since = engine._idle_since
                        if idle_since:
                            state["idle_time_seconds"] = (datetime.now() - idle_since).total_seconds()
                except Exception as e:
                    logger.debug(f"Failed to get daydream idle state: {e}")
                
                # Check active subsystems
                try:
                    from core.erdos_engine import get_erdos_engine
                    erdos = get_erdos_engine()
                    if hasattr(erdos, '_running') and erdos._running:
                        state["subsystem_active"] = "Erdős Engine"
                except Exception as e:
                    logger.debug(f"Failed to check erdos subsystem: {e}")
                
                try:
                    from core.self_refine import get_self_refine_agent
                    refine = get_self_refine_agent()
                    if hasattr(refine, '_running') and refine._running:
                        state["subsystem_active"] = "Self-Refine"
                        state["status"] = "working"
                        state["activity"] = "Analyzing code..."
                except Exception as e:
                    logger.debug(f"Failed to check self-refine subsystem: {e}")
                
                # Check background scheduler
                try:
                    if hasattr(self.agent, 'scheduler') and self.agent.scheduler:
                        is_idle = self.agent.scheduler.is_idle()
                        if not is_idle:
                            state["idle_time_seconds"] = 0
                except Exception as e:
                    logger.debug(f"Failed to check scheduler idle: {e}")
                
                # Get last thinking content from agent memory
                try:
                    if hasattr(self.agent, 'memory'):
                        messages = self.agent.memory.get_messages_for_llm()
                        for msg in reversed(messages[-5:]):
                            if msg.get("role") == "assistant" and msg.get("thinking"):
                                state["thought"] = msg["thinking"][:200]
                                break
                except Exception as e:
                    logger.debug(f"Failed to get last thinking content: {e}")
                
                return {"success": True, "state": state}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/recent-thoughts")
        async def get_recent_thoughts():
            """Get recent thinking content from all subsystems."""
            thoughts = []
            think_start = '<think>'
            think_end = '</think>'
            
            for msg in reversed(self._current_messages[-30:]):
                if msg.get("role") == "assistant":
                    if msg.get("thinking"):
                        thoughts.append({
                            "source": "Chat",
                            "content": msg["thinking"],
                            "timestamp": msg.get("timestamp", ""),
                        })
                    else:
                        content = msg.get("content", "")
                        start = content.find(think_start)
                        end = content.find(think_end, start)
                        if start != -1 and end != -1:
                            thinking = content[start+len(think_start):end].strip()
                            if len(thinking) > 10:
                                thoughts.append({
                                    "source": "Chat",
                                    "content": thinking,
                                    "timestamp": msg.get("timestamp", ""),
                                })
                        elif start != -1:
                            thinking = content[start+len(think_start):].strip()
                            if len(thinking) > 10:
                                thoughts.append({
                                    "source": "Chat",
                                    "content": thinking,
                                    "timestamp": msg.get("timestamp", ""),
                                })

            try:
                if hasattr(self.agent, 'memory'):
                    messages = self.agent.memory.get_messages_for_llm()
                    for msg in reversed(messages[-10:]):
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if think_start in content:
                                start = content.find(think_start)
                                end2 = content.find(think_end, start)
                                if end2 == -1:
                                    end2 = len(content)
                                thinking = content[start+len(think_start):end2].strip()
                                if len(thinking) > 10:
                                    thoughts.append({
                                        "source": "Chat",
                                        "content": thinking,
                                        "timestamp": msg.get("timestamp", ""),
                                    })
            except Exception as e:
                logger.debug(f"Failed to parse chat thoughts: {e}")

            try:
                from core.daydream_engine import get_daydream_engine
                engine = get_daydream_engine()
                status = engine.get_status()
                if status.get("exploration_topics"):
                    for topic in status["exploration_topics"][-3:]:
                        thoughts.append({
                            "source": "Daydream",
                            "content": f"Exploring: {topic}",
                            "timestamp": "",
                        })
            except Exception as e:
                logger.debug(f"Failed to get daydream thoughts: {e}")

            seen = set()
            unique_thoughts = []
            for t in thoughts:
                key = f"{t['source']}:{t['content'][:50]}"
                if key not in seen:
                    seen.add(key)
                    unique_thoughts.append(t)

            return {"success": True, "thoughts": unique_thoughts[:10]}

        @self.app.get("/api/self-evaluation/status")
        async def get_self_evaluation_status():
            try:
                from core.self_evaluation import get_self_evaluation_engine
                engine = get_self_evaluation_engine()
                return {"success": True, **engine.get_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/self-evaluation/actions")
        async def get_self_evaluation_actions(limit: int = 20):
            try:
                from core.self_evaluation import get_self_evaluation_engine
                engine = get_self_evaluation_engine()
                return {"success": True, "actions": engine.get_recent_actions(limit)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/self-evaluation/violations")
        async def get_self_evaluation_violations(resolved: bool | None = None, limit: int = 20):
            try:
                from core.self_evaluation import get_self_evaluation_engine
                engine = get_self_evaluation_engine()
                return {"success": True, "violations": engine.get_violations(resolved, limit)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/self-evaluation/violations/{violation_id}/resolve")
        async def resolve_violation(violation_id: str, request: Request):
            try:
                from core.self_evaluation import get_self_evaluation_engine
                engine = get_self_evaluation_engine()
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["resolution"], max_string_length=5000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                resolution = data.get("resolution", "")
                result = engine.resolve_violation(violation_id, resolution)
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/self-evaluation/reflections")
        async def get_self_evaluation_reflections(limit: int = 5):
            try:
                from core.self_evaluation import get_self_evaluation_engine
                engine = get_self_evaluation_engine()
                return {"success": True, "reflections": engine.get_reflections(limit)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/self-evaluation/evaluate")
        async def trigger_self_evaluation(request: Request):
            try:
                from core.self_evaluation import get_self_evaluation_engine, ActionType
                engine = get_self_evaluation_engine()
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["action_type", "description", "input_preview"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"success": True, "skipped": True, "reason": validation_error}
                
                record = await engine.evaluate_action_async(
                    action_type=ActionType(data.get("action_type", "response")),
                    description=data.get("description", ""),
                    input_preview=data.get("input_preview", ""),
                    output_preview=data.get("output_preview", ""),
                    subsystem=data.get("subsystem", "manual"),
                )
                
                return {
                    "success": True,
                    "evaluation": {
                        "id": record.id,
                        "overall_score": record.overall_score,
                        "flags": record.flags,
                        "principles": {p.value: s for p, s in record.principles_evaluated.items()},
                    }
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/checkpoints/pending")
        async def get_pending_checkpoints():
            try:
                from core.checkpoint_permission import get_checkpoint_manager
                manager = get_checkpoint_manager()
                return {"success": True, "checkpoints": manager.get_pending_checkpoints()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/checkpoints/history")
        async def get_checkpoint_history(limit: int = 50):
            try:
                from core.checkpoint_permission import get_checkpoint_manager
                manager = get_checkpoint_manager()
                return {"success": True, "checkpoints": manager.get_checkpoint_history(limit)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/checkpoints/stats")
        async def get_checkpoint_stats():
            try:
                from core.checkpoint_permission import get_checkpoint_manager
                manager = get_checkpoint_manager()
                return {"success": True, "stats": manager.get_stats()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/checkpoints/policies")
        async def get_checkpoint_policies():
            try:
                from core.checkpoint_permission import get_checkpoint_manager
                manager = get_checkpoint_manager()
                return {"success": True, "policies": manager.get_policies()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/checkpoints/{checkpoint_id}/respond")
        async def respond_to_checkpoint(checkpoint_id: str, request: Request):
            try:
                from core.checkpoint_permission import get_checkpoint_manager
                manager = get_checkpoint_manager()
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["approved", "reason"], max_string_length=5000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                approved = data.get("approved", False)
                reason = data.get("reason", "")
                
                success = manager.respond_to_checkpoint(checkpoint_id, approved, reason)
                return {"success": success}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/checkpoints/domains")
        async def get_frontier_domains():
            try:
                from core.checkpoint_permission import FRONTIER_DOMAIN_DESCRIPTIONS, FrontierDomain
                return {
                    "success": True,
                    "domains": [
                        {
                            "id": d.value,
                            "description": FRONTIER_DOMAIN_DESCRIPTIONS.get(d, ""),
                        }
                        for d in FrontierDomain
                    ]
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/daydream/status")
        async def get_daydream_status():
            try:
                from core.daydream_engine import get_daydream_engine
                from core.tool_learning import get_tool_learning_engine

                engine = get_daydream_engine()
                learning = get_tool_learning_engine()

                status = engine.get_status()
                competencies = learning.get_all_competencies()
                priorities = learning.get_practice_priorities()

                tools_data = []
                for name, comp in list(competencies.items())[:15]:
                    rate = comp.successful_uses / max(1, comp.total_uses) * 100
                    tools_data.append({
                        "name": name,
                        "level": comp.level.value,
                        "uses": comp.total_uses,
                        "success_rate": round(rate, 1),
                        "last_used": comp.last_used.isoformat() if comp.last_used else None,
                    })

                return {
                    "success": True,
                    "running": status.get("running", False),
                    "can_daydream": status.get("can_daydream", False),
                    "daydream_count": status.get("daydream_count", 0),
                    "total_daydream_time": status.get("total_daydream_time", 0),
                    "last_daydream": status.get("last_daydream"),
                    "throttled": status.get("throttled", False),
                    "tools_practiced": status.get("tools_practiced", 0),
                    "insights_generated": status.get("insights_generated", 0),
                    "exploration_topics": status.get("exploration_topics", []),
                    "tools": tools_data,
                    "practice_priorities": priorities[:5],
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/erdos/status")
        async def get_erdos_status():
            try:
                from core.erdos_engine import get_erdos_engine
                engine = get_erdos_engine()
                status = engine.get_status() if hasattr(engine, 'get_status') else {}
                return {
                    "success": True,
                    "running": status.get("running", False),
                    "problems_explored": status.get("total_problems", status.get("problems_explored", 0)),
                    "current_problem": status.get("current_problem"),
                    "insights_generated": status.get("insights_generated", 0),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/api-activity")
        async def get_api_activity():
            """Get recent API activity for live feed."""
            try:
                from core.api_manager import get_api_manager
                manager = get_api_manager()
                activities = manager.get_recent_activities(20)
                
                # Also get LLM call stats
                llm_activities = []
                if hasattr(self.agent, 'llm') and hasattr(self.agent.llm, 'get_call_stats'):
                    llm_stats = self.agent.llm.get_call_stats()
                    recent = llm_stats.get("recent_calls", [])
                    for call in recent[:10]:
                        llm_activities.append({
                            "timestamp": call.get("timestamp"),
                            "subsystem": call.get("subsystem", "llm"),
                            "preview": f"{call.get('type', 'chat')} → {call.get('model', 'unknown')}",
                            "latency": call.get("latency_ms", 0),
                            "cached": call.get("cached", False),
                            "status": "complete",
                        })
                
                # Merge and sort by timestamp
                all_activities = activities + llm_activities
                all_activities.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                
                return {
                    "success": True,
                    "activities": all_activities[:20],
                    "active_subsystems": manager.get_active_subsystems(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}
        # Sibling-route handlers called from other route groups.
        self._route_handlers["get_jarvis_state"] = get_jarvis_state
        self._route_handlers["get_recent_thoughts"] = get_recent_thoughts


    def _setup_screen_skills_terminal_routes(self):
        @self.app.get("/api/screen/status")
        async def get_screen_status():
            try:
                from core.screen_vision import get_screen_vision, HAS_SCREEN_DEPS, HAS_OCR, HAS_AUTOMATION
                sv = get_screen_vision()
                stats = sv.get_stats()
                learner_stats = sv.learner.get_learning_stats()
                return {
                    "success": True,
                    "available": HAS_SCREEN_DEPS,
                    "ocr_available": HAS_OCR,
                    "automation_available": HAS_AUTOMATION,
                    "automation_enabled": stats["automation"]["enabled"],
                    "analyses_performed": stats["analyses_performed"],
                    "learning": learner_stats,
                }
            except ImportError:
                return {"success": False, "error": "Screen vision not available", "available": False}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/skills")
        async def list_skills():
            try:
                from core.skill_loader import get_skill_loader
                loader = get_skill_loader()
                skills = loader.get_all_skills()
                return {
                    "success": True,
                    "skills": [s.to_dict() for s in skills],
                    "count": len(skills),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/skills/{skill_id}")
        async def get_skill(skill_id: str):
            try:
                from core.skill_loader import get_skill_loader
                loader = get_skill_loader()
                skill = loader.get_skill(skill_id)
                if skill:
                    return {"success": True, "skill": skill.to_dict()}
                return {"success": False, "error": "Skill not found"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/skills/reload")
        async def reload_skills():
            try:
                from core.skill_loader import get_skill_loader
                loader = get_skill_loader()
                skills = loader.load_all_skills()
                return {
                    "success": True,
                    "loaded": len([s for s in skills if s.status.value == "loaded"]),
                    "skipped": len([s for s in skills if s.status.value == "skipped"]),
                    "skills": [s.to_dict() for s in skills],
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/skills/install")
        async def install_skill(request: Request):
            try:
                from core.skill_loader import get_skill_loader
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["source"], max_string_length=500
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                source = data["source"]

                loader = get_skill_loader()
                skill = loader.install_skill(source)
                if skill:
                    return {"success": True, "skill": skill.to_dict()}
                return {"success": False, "error": "Failed to install skill"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.delete("/api/skills/{skill_id}")
        async def uninstall_skill(skill_id: str):
            try:
                from core.skill_loader import get_skill_loader
                loader = get_skill_loader()
                success = loader.uninstall_skill(skill_id)
                return {"success": success}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/skills/registry/search")
        async def search_skills_registry(query: str = "", tags: str = "", limit: int = 20):
            try:
                from core.skills_registry import get_skills_registry
                registry = get_skills_registry()
                tag_list = tags.split(",") if tags else None
                results = registry.search(query=query, tags=tag_list, limit=limit)
                return {"success": True, "results": results, "count": len(results)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/skills/registry/trending")
        async def get_trending_skills(days: int = 7, limit: int = 10):
            try:
                from core.skills_registry import get_skills_registry
                registry = get_skills_registry()
                results = registry.get_trending(days=days, limit=limit)
                return {"success": True, "skills": results}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/screen/capture")
        async def capture_screen(request: Request):
            try:
                from core.screen_vision import get_screen_vision
                data = await request.json() if request.headers.get("content-type") == "application/json" else {}
                if data:
                    data, validation_error = validate_request_fields(
                        data, optional=["monitor"], max_string_length=256
                    )
                    if validation_error:
                        return {"success": False, "error": validation_error}
                monitor = data.get("monitor", 1)

                sv = get_screen_vision()
                analysis = sv.see(monitor=monitor)

                ui_elements = []
                for elem in analysis.ui_elements[:20]:
                    ui_elements.append({
                        "type": elem.element_type.name,
                        "text": elem.text,
                        "bbox": elem.bbox.to_dict(),
                        "confidence": elem.confidence,
                    })

                text_blocks = []
                for tb in analysis.text_blocks[:30]:
                    text_blocks.append({
                        "text": tb.text[:100],
                        "bbox": tb.bbox.to_dict(),
                        "confidence": tb.confidence,
                    })

                return {
                    "success": True,
                    "context": analysis.context.name,
                    "context_confidence": analysis.context_confidence,
                    "summary": analysis.summary,
                    "ui_elements": ui_elements,
                    "text_blocks": text_blocks,
                    "development_context": analysis.development_context,
                    "timestamp": analysis.capture.timestamp,
                    "image_base64": analysis.capture.to_base64(),
                }
            except ImportError:
                return {"success": False, "error": "Screen vision not available"}
            except Exception:
                import traceback
                logger.error(f"Screen capture failed: {traceback.format_exc()}")
                return {"success": False, "error": "Internal error - check server logs"}

        @self.app.post("/api/screen/automation")
        async def toggle_automation(request: Request):
            try:
                from core.screen_vision import get_screen_vision
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["enabled"], max_string_length=256
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                enabled = data.get("enabled", False)

                sv = get_screen_vision()
                if enabled:
                    sv.automation.enable()
                else:
                    sv.automation.disable()

                return {
                    "success": True,
                    "automation_enabled": sv.automation._enabled,
                }
            except ImportError:
                return {"success": False, "error": "Screen vision not available"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/screen/click")
        async def screen_click(request: Request):
            try:
                from core.screen_vision import get_screen_vision
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["x", "y", "text"], max_string_length=10000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                x = data.get("x")
                y = data.get("y")
                text = data.get("text")

                sv = get_screen_vision()

                if not sv.automation._enabled:
                    return {"success": False, "error": "Automation not enabled"}

                if x is not None and y is not None:
                    success = sv.automation.click(x, y)
                    return {"success": success, "action": "click", "x": x, "y": y}
                elif text:
                    element = sv.find_element(text)
                    if element:
                        success = sv.click_element(element)
                        return {"success": success, "action": "click_element", "text": text}
                    return {"success": False, "error": f"Element '{text}' not found"}

                return {"success": False, "error": "Provide x/y or text"}
            except ImportError:
                return {"success": False, "error": "Screen vision not available"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/screen/type")
        async def screen_type(request: Request):
            try:
                from core.screen_vision import get_screen_vision
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["text"], max_string_length=10000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                text = data["text"]

                sv = get_screen_vision()

                if not sv.automation._enabled:
                    return {"success": False, "error": "Automation not enabled"}

                success = sv.automation.type_text(text)
                return {"success": success, "action": "type", "length": len(text)}
            except ImportError:
                return {"success": False, "error": "Screen vision not available"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/screen/what-am-i-doing")
        async def what_am_i_doing():
            try:
                from core.screen_vision import get_screen_vision
                sv = get_screen_vision()
                description = sv.what_am_i_doing()
                dev_context = sv.get_development_context()
                return {
                    "success": True,
                    "description": description,
                    "development_context": dev_context,
                }
            except ImportError:
                return {"success": False, "error": "Screen vision not available"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/terminal/status")
        async def get_terminal_status():
            try:
                from tools.terminal import get_jarvis_computer
                computer = get_jarvis_computer()
                status = computer.get_status()
                import platform
                import psutil
                return {
                    "success": True,
                    "workspace": status["workspace"],
                    "mode": status["mode"],
                    "cwd": computer.get_cwd(),
                    "active_session": status["active_session"],
                    "sessions": len(status["sessions"]),
                    "system": {
                        "os": f"{platform.system()} {platform.release()}",
                        "hostname": platform.node(),
                        "python": platform.python_version(),
                    },
                    "resources": {
                        "cpu": psutil.cpu_percent(),
                        "memory": psutil.virtual_memory().percent,
                        "disk": psutil.disk_usage('/').percent,
                    },
                }
            except ImportError as e:
                return {"success": False, "error": f"Terminal not available: {e}"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/terminal/execute")
        async def execute_terminal_command(request: Request):
            if self._require_auth:
                api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
                is_valid, result = self.security.api_key_validator.validate(api_key)
                if not is_valid:
                    return JSONResponse(
                        {"error": "Authentication required", "message": result},
                        status_code=401,
                    )
            try:
                from tools.terminal import get_jarvis_computer
                computer = get_jarvis_computer()
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["command"], optional=["timeout", "approval_token"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                command = data["command"].strip()
                try:
                    timeout = min(max(int(data.get("timeout", 60)), 1), 300)
                except (ValueError, TypeError):
                    timeout = 60

                if not command:
                    return {"success": False, "error": "No command provided"}
                
                if not isinstance(command, str):
                    return {"success": False, "error": "Command must be a string"}

                if len(command) > 10000:
                    return {"success": False, "error": "Command too long (max 10000 chars)"}

                client_ip = request.client.host if request.client else "unknown"
                logger.warning(f"[AUDIT] Terminal execute: ip={client_ip} cmd={command[:200]} timeout={timeout}")
                
                now = time.time()
                self._terminal_rate_limits.setdefault(client_ip, deque(maxlen=30))
                client_times = self._terminal_rate_limits[client_ip]
                recent = [t for t in client_times if now - t < 60]
                if len(recent) >= 20:
                    return JSONResponse(
                        {"error": "Rate limited", "message": "Max 20 terminal commands per minute"},
                        status_code=429,
                        headers=self.security.get_security_headers(),
                    )
                client_times.append(now)
                
                DANGEROUS_PATTERNS = [
                    r'rm\s+-rf\s+/', r'sudo\s+rm', r'mkfs', r'dd\s+if=',
                    r'>\s*/dev/sd', r'>\s*/dev/hd', r'chmod\s+777\s+/',
                    r':(){:|:&};:', r'fork\s*bomb', r'\bsource\s+/dev/',
                    r'\bwget\b.*\|\s*sh', r'\bcurl\b.*\|\s*(ba)?sh',
                    r'\bnc\s+-', r'\bsocat\b', r'\bpython\b.*-c\s',
                    r'\bperl\b.*-e\s', r'\bruby\b.*-e\s',
                    r'/dev/tcp/', r'/dev/udp/',
                ]
                import re
                for pattern in DANGEROUS_PATTERNS:
                    if re.search(pattern, command, re.IGNORECASE):
                        return {
                            "success": False,
                            "error": "Command blocked: dangerous pattern detected",
                            "command": command,
                        }

                risk, risk_reason = computer.assess_risk(command)
                needs_perm = computer.needs_permission(command)

                if needs_perm:
                    approval_token = data.get("approval_token", "")
                    expected_token = hashlib.sha256(
                        f"terminal_approval:{command}:{self._approval_secret}".encode()
                    ).hexdigest()
                    if not hmac.compare_digest(approval_token, expected_token):
                        return {
                            "success": False,
                            "needs_permission": True,
                            "command": command,
                            "risk": risk,
                            "reason": risk_reason,
                            "message": "Command requires approval. Request an approval token from /api/terminal/request-approval.",
                        }

                exit_code, stdout, stderr = await computer.execute(command, timeout=timeout)

                return {
                    "success": exit_code == 0,
                    "exit_code": exit_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "cwd": computer.get_cwd(),
                }
            except ImportError:
                return {"success": False, "error": "Terminal not available"}
            except Exception as e:
                logger.error(f"Terminal execute error: {e}")
                return {"success": False, "error": "Terminal execution failed"}

        @self.app.post("/api/terminal/request-approval")
        async def request_terminal_approval(request: Request):
            """Request server-signed approval token for a terminal command.
            
            The token is HMAC-signed with the command text, making it
            impossible to use a token for a different command.
            Requires authentication when auth is enabled.
            """
            if self._require_auth:
                api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
                is_valid, result = self.security.api_key_validator.validate(api_key)
                if not is_valid:
                    return JSONResponse(
                        {"error": "Authentication required", "message": result},
                        status_code=401,
                    )
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["command"], max_string_length=10000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                command = data["command"].strip()
                if not command:
                    return {"success": False, "error": "No command provided"}
                client_ip = request.client.host if request.client else "unknown"
                logger.info(f"[AUDIT] Terminal approval request: ip={client_ip} cmd={command[:200]}")
                from tools.terminal import get_jarvis_computer
                computer = get_jarvis_computer()
                risk, risk_reason = computer.assess_risk(command)
                needs_perm = computer.needs_permission(command)
                if not needs_perm:
                    return {"success": True, "needs_permission": False, "risk": risk, "reason": risk_reason, "message": "This command does not require approval"}
                approval_token = hashlib.sha256(
                    f"terminal_approval:{command}:{self._approval_secret}".encode()
                ).hexdigest()
                return {
                    "success": True,
                    "needs_permission": True,
                    "command": command,
                    "risk": risk,
                    "reason": risk_reason,
                    "approval_token": approval_token,
                    "message": "Use this approval_token with /api/terminal/execute",
                }
            except Exception as e:
                return {"success": False, "error": "Approval request failed"}

        @self.app.post("/api/terminal/cwd")
        async def change_terminal_cwd(request: Request):
            if self._require_auth:
                api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
                is_valid, result = self.security.api_key_validator.validate(api_key)
                if not is_valid:
                    return JSONResponse(
                        {"error": "Authentication required", "message": result},
                        status_code=401,
                    )
            try:
                from tools.terminal import get_jarvis_computer
                computer = get_jarvis_computer()
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["path"], max_string_length=4096
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                path = data.get("path")

                if path:
                    if computer.set_cwd(path):
                        return {"success": True, "cwd": computer.get_cwd()}
                    return {"success": False, "error": "Cannot change to specified directory"}

                return {"success": True, "cwd": computer.get_cwd()}
            except Exception as e:
                logger.error(f"Terminal CWD error: {e}")
                return {"success": False, "error": "Directory operation failed"}

        @self.app.get("/api/terminal/history")
        async def get_terminal_history():
            try:
                from tools.terminal import get_jarvis_computer
                computer = get_jarvis_computer()
                session = computer.sessions.get(computer.active_session or "default")

                if not session:
                    return {"success": True, "history": []}

                history = list(session.history)[-50:]
                return {
                    "success": True,
                    "history": history,
                    "cwd": session.cwd,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/terminal/mode")
        async def set_terminal_mode(request: Request):
            try:
                from tools.terminal import get_jarvis_computer, TerminalMode
                computer = get_jarvis_computer()
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["mode"], max_string_length=256
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                mode = data["mode"]

                computer.mode = TerminalMode(mode)
                return {"success": True, "mode": mode}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}


    def _setup_chat_command_routes(self):
        from fastapi.responses import StreamingResponse

        # ------------------------------------------------------------------
        # Text-to-speech (server-side neural voice, edge-tts / piper)
        # ------------------------------------------------------------------

        @self.app.get("/api/tts/state")
        async def tts_state():
            """Report TTS backend availability so the frontend can pick
            server voice vs browser speechSynthesis honestly."""
            try:
                from core.tts_engine import get_tts_engine
                engine = get_tts_engine()
                await engine.initialize()
                return {"success": True, **engine.get_state()}
            except Exception as e:
                logger.warning(f"TTS state unavailable: {e}")
                return {"success": False, "backends": [], "active_backend": None,
                        "active_voice": None, "voices": [], "requests": 0, "failures": 0}

        @self.app.get("/api/tts/voices")
        async def tts_voices(backend: str = None):
            try:
                from core.tts_engine import get_tts_engine
                engine = get_tts_engine()
                await engine.initialize()
                voices = await engine.list_voices()
                payload = [
                    {"id": v.id, "name": v.name, "backend": v.backend,
                     "language": v.language, "gender": v.gender}
                    for v in voices[:200]
                ]
                return {"success": True, "voices": payload}
            except Exception as e:
                logger.warning(f"TTS voice listing failed: {e}")
                return {"success": False, "voices": []}

        @self.app.post("/api/tts")
        async def synthesize_speech(request: Request):
            """Synthesize text to audio. Returns audio bytes on success,
            JSON with success=False + reason when no backend is available
            (frontend then falls back to browser speechSynthesis)."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["text"], max_string_length=5000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}

                text = data["text"].strip()
                if not text:
                    return {"success": False, "error": "No text provided"}

                voice = data.get("voice")  # optional voice override

                from core.tts_engine import get_tts_engine
                engine = get_tts_engine()
                await engine.initialize()

                if not engine.available():
                    return JSONResponse(
                        status_code=503,
                        content={"success": False,
                                 "error": "No TTS backend available (install edge-tts or piper)"},
                    )

                audio = await engine.synthesize(text, voice=voice)
                if audio is None:
                    return JSONResponse(
                        status_code=502,
                        content={"success": False, "error": "Speech synthesis failed"},
                    )

                media_type = "audio/mpeg" if engine._state.active_backend == "edge" else "audio/wav"
                return FastAPIResponse(
                    content=audio,
                    media_type=media_type,
                    headers={"Cache-Control": "no-store"},
                )
            except Exception as e:
                logger.error(f"TTS synthesis error: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"success": False, "error": safe_error_message(e)},
                )

        # ------------------------------------------------------------------
        # Live camera feed + two-model vision chain (Iris-inspired design)
        # ------------------------------------------------------------------

        @self.app.get("/api/live-camera")
        async def live_camera_frame():
            """Latest camera frame (base64 JPEG) — Iris's live-feed design.

            Privacy: the feed loop runs ONLY while a consumer is attached;
            a GET here attaches for the duration of this request and
            releases immediately, so polling keeps the feed alive and the
            camera stops (LED off) when polling stops.
            """
            try:
                from core.live_feed import get_live_camera_feed
                feed = get_live_camera_feed()
                feed.attach()
                try:
                    # Give a cold feed one interval to produce its first frame
                    for _ in range(3):
                        frame = feed.get_frame()
                        if frame:
                            return {"success": True, **frame}
                        await asyncio.sleep(0.4)
                    return {"success": True, "image": None,
                            "error": "No frame available (camera starting or unavailable)"}
                finally:
                    feed.release()
            except Exception as e:
                logger.warning(f"Live camera endpoint failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.websocket("/ws/live")
        async def live_camera_ws(websocket: WebSocket):
            """Stream live camera frames (Iris /ws/live design).

            Params (query): mode=camera (screen later), fps (0.5-15),
            quality (10-95). The feed runs only while this socket is open;
            on disconnect the camera is released.
            """
            # Same cross-site WebSocket guard as /ws — a browser page must
            # not be able to watch the user's camera from another origin.
            if not self._ws_origin_allowed(websocket):
                await websocket.accept()
                await websocket.send_json({"type": "error", "content": "Origin not allowed", "code": "origin_forbidden"})
                await websocket.close(code=4003, reason="Origin not allowed")
                return

            await websocket.accept()
            from core.live_feed import get_live_camera_feed
            feed = get_live_camera_feed()
            feed.attach()
            try:
                params = dict(websocket.query_params)
                fps = float(params.get("fps", 2))
                quality = int(params.get("quality", 60))
                feed.set_params(fps=fps, quality=quality)
                last_ts = 0.0
                while True:
                    frame = feed.get_frame()
                    if frame and frame["timestamp"] != last_ts:
                        last_ts = frame["timestamp"]
                        await websocket.send_json({
                            "type": "frame", "source": "camera",
                            "image": frame["image"], "timestamp": frame["timestamp"],
                        })
                    await asyncio.sleep(1.0 / max(0.5, min(fps, 15.0)))
            except Exception:
                pass  # disconnect or send failure — handled in finally
            finally:
                feed.release()
                try:
                    await websocket.close()
                except Exception:
                    pass

        @self.app.post("/api/vision-chain")
        async def vision_chain(request: Request):
            """Iris's two-model vision chain with Jarvis's real dual-model:

            1. VISION: the vision-capable model analyzes the captured frame.
            2. REASON: the backend model reasons over the vision output
               ({vision_output} substituted into think_prompt).

            Body: {prompt?, think_prompt?, source: "camera"}. Camera captures
            are one-shot (release immediately) — same privacy as the webcam
            tool. Requires permission? No: user-initiated POST, same as
            /api/analyze-image.
            """
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=[], max_string_length=8000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}

                prompt = (data.get("prompt") or "Describe what you see.")[:2000]
                think_prompt = data.get("think_prompt")  # optional second stage
                source = data.get("source", "camera")
                if source != "camera":
                    return {"success": False, "error": f"Unsupported source: {source}"}

                # One-shot capture (asyncio.to_thread inside WebcamCapture)
                from core.vision_tokens import WebcamCapture
                capture = WebcamCapture()
                if not await capture.start():
                    return {"success": False, "error": "Camera unavailable"}
                try:
                    frame = await capture.capture_frame()
                finally:
                    await capture.stop()
                if frame is None:
                    return {"success": False, "error": "Capture failed"}

                # Stage 1: vision analysis through the multimodal processor
                from core.multimodal import get_multimodal_manager
                multimodal = get_multimodal_manager(llm_client=self.agent.llm)
                import base64 as _b64
                import io as _io
                from PIL import Image as _Image
                pil_image = _Image.fromarray(frame)
                buf = _io.BytesIO()
                pil_image.save(buf, format="JPEG", quality=85)
                img_b64 = _b64.b64encode(buf.getvalue()).decode()
                vision_text = await multimodal.vision.analyze_image(img_b64, prompt=prompt)

                result = {"success": True, "vision": vision_text, "reasoning": None}

                # Stage 2: backend reasoning over the vision output
                if think_prompt:
                    backend = self.agent.backend_llm
                    if backend is not None:
                        chain_prompt = think_prompt.replace("{vision_output}", vision_text)[:8000]
                        response = await backend.chat(
                            user_input=chain_prompt,
                            system_prompt="You are the reasoning stage of a vision pipeline. Be precise and concise.",
                        )
                        result["reasoning"] = get_response_content(response)
                    else:
                        result["reasoning_error"] = "Backend model unavailable"

                # Note what Jarvis saw into PFC (if running) — same shape as
                # /api/analyze-image observations.
                try:
                    from core.prefrontal_cortex import get_prefrontal_cortex, ConceptType
                    pfc = get_prefrontal_cortex()
                    if pfc and pfc._running:
                        task = asyncio.create_task(pfc.observe(
                            content=vision_text[:300],
                            concept_type=ConceptType.OBSERVATION,
                            source="vision_chain",
                            metadata={"has_reasoning": bool(result["reasoning"])},
                        ))
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)
                except Exception as e:
                    logger.debug(f"PFC vision observation failed: {e}")

                return result
            except Exception as e:
                logger.error(f"Vision chain error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/command")
        async def handle_command(request: Request):
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["command"], max_string_length=10000
                )
                if validation_error:
                    return {"response": validation_error}
                command = data["command"].strip()

                if not command:
                    return {"response": "No command provided"}

                cmd_parts = command.split()
                cmd_name = cmd_parts[0].lower()
                cmd_args = cmd_parts[1:] if len(cmd_parts) > 1 else []

                # Handle built-in commands
                if cmd_name in ["help", "?"]:
                    return {"response": """╔══════════════════════════════════════════════════════════════╗
║                     JARVIS CLI HELP                          ║
╠══════════════════════════════════════════════════════════════╣
║ BASIC COMMANDS                                               ║
╠──────────────────────────────────────────────────────────────╣
║  help              Show this help message                     ║
║  status            Show detailed system status                ║
║  tools             List available tools                       ║
║  clear             Clear conversation memory                  ║
║  exit / quit       End session (for CLI)                      ║
╠──────────────────────────────────────────────────────────────╣
║ CODING & FILES                                               ║
╠──────────────────────────────────────────────────────────────╣
║  scan [path]        Scan code for issues                      ║
║  edit <file>        Edit a file with Jarvis                   ║
║  read <file>        Read a file                               ║
║  search <query>     Search codebase                            ║
║  run <command>      Run a shell command                       ║
╠──────────────────────────────────────────────────────────────╣
║ MEMORY & LEARNING                                            ║
╠──────────────────────────────────────────────────────────────╣
║  memory stats       Show memory statistics                    ║
║  memory search <q>  Search stored memories                    ║
║  memory clear       Clear conversation memory                 ║
║  recall <topic>     Recall information about a topic          ║
╠──────────────────────────────────────────────────────────────╣
║ SELF-IMPROVEMENT                                             ║
╠──────────────────────────────────────────────────────────────╣
║  refine scan [--ai] Scan codebase for improvements            ║
║  refine list        List pending changes                      ║
║  refine apply [id]  Apply approved changes                    ║
║  evolve status      Show evolution status                     ║
║  evolve history     View evolution history                    ║
╠──────────────────────────────────────────────────────────────╣
║ AGENTIC CAPABILITIES                                         ║
╠──────────────────────────────────────────────────────────────╣
║  Jarvis can:                                                 ║
║  • Read, write, and edit files in the codebase               ║
║  • Run tests and fix failing tests                           ║
║  • Search and analyze code patterns                           ║
║  • Create new files and modules                               ║
║  • Refactor and optimize code                                 ║
║  • Execute shell commands (with permission)                   ║
║  • Self-improve through /refine commands                      ║
╠──────────────────────────────────────────────────────────────╣
║ TIPS                                                         ║
╠──────────────────────────────────────────────────────────────╣
║  • Ask questions naturally - Jarvis understands context       ║
║  • For coding tasks, describe what you want done              ║
║  • Use /refine scan to analyze code quality                   ║
║  • Permission is asked for sensitive operations               ║
╚══════════════════════════════════════════════════════════════╝"""}

                if cmd_name == "status":
                    model = getattr(self.agent.llm, 'model', 'unknown') if hasattr(self.agent, 'llm') else 'unknown'
                    turn_count = getattr(self.agent, '_turn_count', 0)
                    tools_count = len(self.agent.tools)
                    mem_count = len(self.agent.memory.get_messages_for_llm()) if hasattr(self.agent, 'memory') else 0

                    status_lines = [
                        "╔══════════════════════════════════════╗",
                        "║         JARVIS SYSTEM STATUS         ║",
                        "╠══════════════════════════════════════╣",
                        f"║  Model:      {model:<23} ║",
                        f"║  Tools:      {tools_count:<23} ║",
                        f"║  Memory:     {mem_count} messages{' '*(14-len(str(mem_count)))} ║",
                        f"║  Turns:      {turn_count:<23} ║",
                    ]

                    if hasattr(self.agent, 'tool_cache'):
                        cache_stats = self.agent.tool_cache.get_stats()
                        hit_rate = cache_stats.get('hit_rate', 0) * 100
                        status_lines.append(f"║  Cache Hit:  {hit_rate:.1f}%{' '*(17 - len(f'{hit_rate:.1f}%'))} ║")

                    status_lines.append("╚══════════════════════════════════════╝")
                    return {"response": "\n".join(status_lines)}

                if cmd_name == "tools":
                    tools_list = []
                    for name, tool in sorted(self.agent.tools.items())[:30]:
                        desc = getattr(tool, 'description', '')[:40] if hasattr(tool, 'description') else ''
                        perm = " ⚠️" if getattr(tool, 'requires_permission', False) else ""
                        tools_list.append(f"  {name:<20} {desc}{perm}")

                    return {"response": f"╔══════════════════════════════════════════════╗\n║ Available Tools ({len(self.agent.tools)} total)                   ║\n╠══════════════════════════════════════════════╣\n" + "\n".join(tools_list) + "\n╚══════════════════════════════════════════════╝"}

                if cmd_name == "memory":
                    if cmd_args and cmd_args[0] == "stats":
                        mem_count = len(self.agent.memory.get_messages_for_llm()) if hasattr(self.agent, 'memory') else 0
                        return {"response": f"Memory Statistics:\n  Messages in context: {mem_count}"}
                    elif cmd_args and cmd_args[0] == "clear":
                        if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear'):
                            self.agent.memory.clear()
                        return {"response": "✓ Memory cleared"}
                    elif cmd_args and cmd_args[0] == "search" and len(cmd_args) > 1:
                        query = " ".join(cmd_args[1:])
                        return {"response": f"Searching for: {query}\n(Use full chat for memory search)"}
                    else:
                        mem_count = len(self.agent.memory.get_messages_for_llm()) if hasattr(self.agent, 'memory') else 0
                        return {"response": f"Memory: {mem_count} messages in context\n\nCommands:\n  memory stats   - Show statistics\n  memory clear   - Clear memory\n  memory search <query> - Search memories"}

                if cmd_name == "clear":
                    if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear'):
                        self.agent.memory.clear()
                    return {"response": "✓ Conversation memory cleared"}

                if cmd_name == "scan":
                    # Trigger refinement scan
                    try:
                        return {"response": "Codebase scan initiated. Results will be available through /refine list."}
                    except Exception:
                        return {"response": "Scan functionality available through /refine scan"}

                if cmd_name == "refine":
                    if cmd_args and cmd_args[0] == "scan":
                        return {"response": "Codebase scan for improvements...\nUse CLI interface for full refine functionality:\n  /refine scan [--ai]   - Pattern-based (or AI deep) scan\n  /refine list          - List pending changes\n  /refine apply [id]    - Apply changes"}
                    elif cmd_args and cmd_args[0] == "list":
                        return {"response": "Use CLI interface for refine list:\n  /refine list"}
                    else:
                        return {"response": "Refine Commands:\n  /refine scan [--ai]  Scan for improvements\n  /refine list         List pending changes\n  /refine apply [id]   Apply changes\n  /refine rollback <id> Rollback a change"}

                if cmd_name == "evolve":
                    return {"response": "Evolution Commands:\n  /evolve status    - Show evolution system status\n  /evolve history   - View evolution history\n\nUse CLI interface for full evolution commands."}

                # Otherwise, treat as chat
                response_text = ""
                async for token in self.agent.run_step_stream(command):
                    response_text += token
                return {"response": response_text[:2000] or "Done"}

            except Exception as e:
                logger.error(f"Command error: {e}")
                return {"response": f"Error: {safe_error_message(e)}"}

        @self.app.post("/api/chat")
        async def chat_rest(message: str = Form(...)):
            if len(message) > 100000:
                return JSONResponse(
                    {"error": "Message too long", "message": "Maximum 100000 characters"},
                    status_code=413,
                )
            async def generate():
                accumulated = ""
                in_thinking = False
                thinking_tokens = []
                
                try:
                    async for token in self.agent.run_step_stream(message):
                        accumulated += token
                        if '<think>' in token:
                            in_thinking = True
                        if in_thinking:
                            thinking_tokens.append(token)
                            if '</think>' in token:
                                in_thinking = False
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                    
                    self._current_messages.append({
                        "role": "user",
                        "content": message,
                        "timestamp": datetime.now().isoformat(),
                    })
                    
                    msg_data = {
                        "role": "assistant",
                        "content": accumulated,
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    thinking_text = "".join(thinking_tokens)
                    thinking_text = thinking_text.replace('<think>', '').replace('</think>', '').strip()
                    if thinking_text and len(thinking_text) > 10:
                        msg_data["thinking"] = thinking_text[:500]
                    
                    self._current_messages.append(msg_data)
                    
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': safe_error_message(e)})}\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"}
            )

        @self.app.post("/api/transcribe")
        async def transcribe_audio(audio: UploadFile = File(...)):
            try:
                from core.multimodal import get_multimodal_manager
                from core.prefrontal_cortex import get_prefrontal_cortex, ConceptType
                multimodal = get_multimodal_manager(llm_client=self.agent.llm)
                audio_data = await audio.read()
                format_name = audio.filename.rsplit('.', 1)[-1] if audio.filename and '.' in audio.filename else "webm"
                result = await multimodal.audio.transcribe_audio_data(audio_data, format=format_name)
                if result.get("success"):
                    # Observe audio transcription with PFC
                    try:
                        pfc = get_prefrontal_cortex()
                        if pfc and pfc._running:
                            task = asyncio.create_task(pfc.observe(
                                content=result.get("text", "")[:300],
                                concept_type=ConceptType.OBSERVATION,
                                source="audio_transcription",
                                metadata={"format": format_name, "tokens": result.get("semantic_token_count", 0)}
                            ))
                            self._background_tasks.add(task)
                            task.add_done_callback(self._background_tasks.discard)
                    except Exception as e:
                        logger.debug(f"PFC observation failed: {e}")
                    return {"text": result.get("text", ""), "success": True}
                return {"error": result.get("error", "Transcription failed"), "success": False}
            except ImportError as e:
                logger.warning(f"Transcription unavailable: {e}")
                return {"error": "Speech recognition not available. Install faster-whisper: pip install faster-whisper", "success": False}
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                return {"error": safe_error_message(e), "success": False}

        @self.app.post("/api/analyze-image")
        async def analyze_image(request: Request):
            try:
                from core.multimodal import get_multimodal_manager
                import tempfile
                import os
                import base64

                multimodal = get_multimodal_manager(llm_client=self.agent.llm)

                # Check content type
                content_type = request.headers.get("content-type", "")

                if "multipart/form-data" in content_type:
                    form = await request.form()
                    file = form.get("file") or form.get("image")
                    if not file:
                        return {"error": "No file in form data", "success": False}
                    from starlette.datastructures import UploadFile
                    if isinstance(file, UploadFile):
                        image_data = await file.read()
                        upload_filename = file.filename or "upload.jpg"
                        upload_content_type = file.content_type or "image/jpeg"
                    elif isinstance(file, bytes):
                        image_data = file
                        upload_filename = "upload.jpg"
                        upload_content_type = "image/jpeg"
                    elif isinstance(file, str):
                        image_data = file.encode()
                        upload_filename = "upload.txt"
                        upload_content_type = "text/plain"
                    else:
                        return {"error": "Invalid file format", "success": False}

                    from core.security.upload_validation import validate_upload
                    validation = validate_upload(
                        filename=upload_filename,
                        mime_type=upload_content_type,
                        file_size=len(image_data),
                        file_data=image_data[:64] if image_data else None,
                        allowed_types={
                            "image/jpeg": {".jpg", ".jpeg"},
                            "image/png": {".png"},
                            "image/gif": {".gif"},
                            "image/webp": {".webp"},
                            "image/svg+xml": {".svg"},
                            "image/bmp": {".bmp"},
                        },
                    )
                    if not validation.valid:
                        return {"error": f"Upload rejected: {validation.reason}", "success": False}
                else:
                    data = await request.json()
                    data, validation_error = validate_request_fields(
                        data, required=["image"], max_string_length=2000000
                    )
                    if validation_error:
                        return {"error": validation_error, "success": False}
                    image_data = data["image"]
                    # Strip data URL prefix if present
                    if image_data.startswith("data:image"):
                        image_data = image_data.split(",", 1)[1]
                    image_data = base64.b64decode(image_data)

                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    f.write(image_data)
                    temp_path = f.name

                try:
                    description = await multimodal.vision.analyze_image(temp_path)
                    return {"description": description, "success": True}
                finally:
                    os.unlink(temp_path)
            except ImportError as e:
                logger.warning(f"Vision unavailable: {e}")
                return {"error": "Vision analysis requires a vision-capable model", "success": False}
            except Exception as e:
                logger.error(f"Image analysis error: {e}")
                return {"error": safe_error_message(e), "success": False}

        @self.app.post("/api/generate-image")
        async def generate_image(request: Request):
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["prompt"], optional=["style", "size"],
                    max_string_length=5000
                )
                if validation_error:
                    return {"error": validation_error, "success": False}
                prompt = data["prompt"]
                data.get("style", "realistic")
                size = data.get("size", "512x512")

                from core.image_gen import get_image_generator
                generator = get_image_generator()

                # Parse size
                width, height = map(int, size.split("x"))

                # Generate image
                image = await generator.generate(
                    prompt=prompt,
                    negative_prompt="blurry, low quality, distorted, ugly",
                    width=width,
                    height=height,
                    save=True,
                )

                if image.file_path:
                    # Return relative URL to the image
                    image_url = f"/images/{image.file_path.name}"
                    return {
                        "success": True,
                        "image_url": image_url,
                        "prompt": prompt,
                        "backend": image.backend,
                        "generation_time": image.generation_time,
                    }
                else:
                    return {
                        "success": True,
                        "image_base64": image.to_base64(),
                        "prompt": prompt,
                        "backend": image.backend,
                        "generation_time": image.generation_time,
                    }

            except Exception as e:
                logger.error(f"Image generation error: {e}")
                return {"error": safe_error_message(e), "success": False}

        @self.app.get("/api/visualization/live")
        async def get_live_visualization():
            """Get real-time system visualization data for Mermaid diagrams."""
            from datetime import datetime
            now = datetime.now

            subsystems = {}
            connections = []

            # Core Agent
            subsystems["agent"] = {
                "name": "Jarvis Agent",
                "icon": "🤖",
                "status": "active",
                "stats": {
                    "turns": getattr(self.agent, '_turn_count', 0),
                    "tools": len(self.agent.tools) if hasattr(self.agent, 'tools') else 0,
                }
            }

            # Memory
            try:
                mem_count = len(self.agent.memory.get_messages_for_llm()) if hasattr(self.agent, 'memory') else 0
                subsystems["memory"] = {
                    "name": "Memory",
                    "icon": "🧠",
                    "status": "active" if mem_count > 0 else "idle",
                    "stats": {"messages": mem_count}
                }
            except Exception:
                subsystems["memory"] = {"name": "Memory", "icon": "🧠", "status": "idle", "stats": {}}

            # LLM
            try:
                from core.api_manager import get_api_manager
                manager = get_api_manager()
                stats = manager.get_stats()
                subsystems["llm"] = {
                    "name": "LLM",
                    "icon": "🧮",
                    "status": "active" if stats.get("total_calls", 0) > 0 else "idle",
                    "stats": {
                        "calls": stats.get("total_calls", 0),
                        "tokens": stats.get("total_tokens", 0),
                    }
                }
            except Exception:
                subsystems["llm"] = {"name": "LLM", "icon": "🧮", "status": "idle", "stats": {}}

            # Erdos Engine
            try:
                from core.erdos_engine import get_erdos_engine
                engine = get_erdos_engine()
                problems = len(engine.problems) if hasattr(engine, 'problems') else 0
                subsystems["erdos"] = {
                    "name": "Erdős Engine",
                    "icon": "📐",
                    "status": "active" if getattr(engine, '_running', False) else "idle",
                    "stats": {"problems": problems}
                }
            except Exception:
                subsystems["erdos"] = {"name": "Erdős Engine", "icon": "📐", "status": "idle", "stats": {}}

            # Daydream Engine
            try:
                from core.daydream_engine import get_daydream_engine
                engine = get_daydream_engine()
                subsystems["daydream"] = {
                    "name": "Daydream",
                    "icon": "💭",
                    "status": "active" if getattr(engine, '_running', False) else "idle",
                    "stats": {"explorations": getattr(engine, '_exploration_count', 0)}
                }
            except Exception:
                subsystems["daydream"] = {"name": "Daydream", "icon": "💭", "status": "idle", "stats": {}}

            # Evolution
            try:
                from core.evolution_orchestrator import get_evolution_orchestrator
                orch = get_evolution_orchestrator()
                subsystems["evolution"] = {
                    "name": "Evolution",
                    "icon": "🧬",
                    "status": "active" if getattr(orch, '_running', False) else "idle",
                    "stats": {"genes": len(orch.genes) if hasattr(orch, 'genes') else 0}
                }
            except Exception:
                subsystems["evolution"] = {"name": "Evolution", "icon": "🧬", "status": "idle", "stats": {}}

            # Learning Pipeline
            try:
                from core.learning_pipeline import get_learning_pipeline
                pipeline = get_learning_pipeline()
                subsystems["learning"] = {
                    "name": "Learning",
                    "icon": "📚",
                    "status": "active" if getattr(pipeline, '_running', False) else "idle",
                    "stats": {"generation": getattr(pipeline, '_generation', 0)}
                }
            except Exception:
                subsystems["learning"] = {"name": "Learning", "icon": "📚", "status": "idle", "stats": {}}

            # Defense
            try:
                from core.defense import get_defense_agent
                defense = get_defense_agent()
                subsystems["defense"] = {
                    "name": "Defense",
                    "icon": "🛡️",
                    "status": "active",
                    "stats": {"threats": getattr(defense, 'threat_count', 0)}
                }
            except Exception:
                subsystems["defense"] = {"name": "Defense", "icon": "🛡️", "status": "idle", "stats": {}}

            # Screen Vision
            try:
                from core.screen_vision import get_screen_vision
                sv = get_screen_vision()
                stats = sv.get_stats()
                subsystems["screen"] = {
                    "name": "Screen Vision",
                    "icon": "👁️",
                    "status": "active" if stats.get("has_screen_deps") else "unavailable",
                    "stats": {
                        "captures": stats.get("analyses_performed", 0),
                        "automation": stats.get("automation", {}).get("enabled", False)
                    }
                }
            except Exception:
                subsystems["screen"] = {"name": "Screen Vision", "icon": "👁️", "status": "unavailable", "stats": {}}

            # Terminal
            try:
                from tools.terminal import get_jarvis_computer
                computer = get_jarvis_computer()
                subsystems["terminal"] = {
                    "name": "Terminal",
                    "icon": "💻",
                    "status": "active",
                    "stats": {
                        "cwd": computer.get_cwd(),
                        "mode": computer.mode.value
                    }
                }
            except Exception:
                subsystems["terminal"] = {"name": "Terminal", "icon": "💻", "status": "idle", "stats": {}}

            # Browser
            try:
                from tools.browser_tools import BrowserLaunchTool
                sessions = len(BrowserLaunchTool._sessions) if hasattr(BrowserLaunchTool, '_sessions') else 0
                subsystems["browser"] = {
                    "name": "Browser",
                    "icon": "🌐",
                    "status": "active" if sessions > 0 else "idle",
                    "stats": {"sessions": sessions}
                }
            except Exception:
                subsystems["browser"] = {"name": "Browser", "icon": "🌐", "status": "idle", "stats": {}}

            # Tools summary
            tool_categories = {
                "system": 0, "web": 0, "productivity": 0,
                "screen": 0, "browser": 0, "terminal": 0, "other": 0
            }
            if hasattr(self.agent, 'tools'):
                for name in self.agent.tools.keys():
                    if any(x in name for x in ['shell', 'file', 'system']):
                        tool_categories["system"] += 1
                    elif any(x in name for x in ['web', 'fetch', 'search']):
                        tool_categories["web"] += 1
                    elif any(x in name for x in ['screen', 'element', 'click']):
                        tool_categories["screen"] += 1
                    elif 'browser' in name:
                        tool_categories["browser"] += 1
                    elif 'terminal' in name:
                        tool_categories["terminal"] += 1
                    else:
                        tool_categories["other"] += 1

            # Connections (always active flows)
            connections = [
                {"from": "user", "to": "agent", "label": "query", "active": True},
                {"from": "agent", "to": "llm", "label": "reasoning", "active": True},
                {"from": "llm", "to": "agent", "label": "response", "active": True},
                {"from": "agent", "to": "memory", "label": "store/recall", "active": True},
                {"from": "agent", "to": "tools", "label": "execute", "active": True},
            ]

            # Generate Mermaid diagram
            mermaid_lines = [
                "graph TD",
                "    User[👤 User]",
                "    Agent[🤖 Jarvis Agent]",
                "    LLM[🧮 LLM Engine]",
                "    Memory[🧠 Memory]",
                "    Tools[🔧 Tools Hub]",
                "",
                "    User ==>|request| Agent",
                "    Agent ==>|think| LLM",
                "    LLM ==>|response| Agent",
                "    Agent <-->|context| Memory",
                "    Agent -->|route| Tools",
                "",
                "    subgraph Subsystems[Background Subsystems]",
            ]

            color_map = {
                "active": "#22c55e", "idle": "#6b7280", "unavailable": "#ef4444"
            }
            for key, sub in subsystems.items():
                color = color_map.get(sub.get("status", "idle"), "#6b7280")
                icon = sub.get("icon", "📦")
                name = sub.get("name", key)
                mermaid_lines.append(f'        {key}["{icon} {name}"]')

            mermaid_lines.append("    end")
            mermaid_lines.append("")
            mermaid_lines.append("    subgraph ToolCategories[Tool Categories]")

            for cat, count in tool_categories.items():
                if count > 0:
                    mermaid_lines.append(f'        tools_{cat}["{cat.title()}: {count}"]')

            mermaid_lines.append("    end")
            mermaid_lines.append("")
            mermaid_lines.append("    Tools --> Subsystems")

            # Add active indicators
            for key, sub in subsystems.items():
                if sub.get("status") == "active":
                    mermaid_lines.append(f"    style {key} stroke:#22c55e,stroke-width:3px")

            return {
                "success": True,
                "timestamp": str(now()),
                "subsystems": subsystems,
                "connections": connections,
                "tool_categories": tool_categories,
                "mermaid": "\n".join(mermaid_lines),
                "stats": {
                    "total_tools": len(self.agent.tools) if hasattr(self.agent, 'tools') else 0,
                    "active_subsystems": sum(1 for s in subsystems.values() if s.get("status") == "active"),
                }
            }


    def _setup_ops_health_routes(self):
        @self.app.get("/api/defense/status")
        async def get_defense_status():
            try:
                from core.defense import get_defense_agent
                agent = get_defense_agent()
                status = agent.get_status() if hasattr(agent, 'get_status') else {}
                return {
                    "success": True,
                    "monitoring": status.get("monitoring", True),
                    "threats_detected": status.get("threats_detected", 0),
                    "threats_blocked": status.get("threats_blocked", 0),
                    "last_scan": status.get("last_scan"),
                    "xss_attempts": status.get("xss_attempts", 0),
                    "injection_attempts": status.get("injection_attempts", 0),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/free-time/status")
        async def get_free_time_status():
            try:
                from core.free_time import get_free_time_engine
                engine = get_free_time_engine()
                status = engine.get_status() if hasattr(engine, 'get_status') else {}
                return {
                    "success": True,
                    "enabled": status.get("enabled", True),
                    "sandbox_active": status.get("sandbox_active", False),
                    "sessions_completed": status.get("sessions_completed", 0),
                    "discoveries_saved": status.get("discoveries_saved", 0),
                    "pending_permissions": status.get("pending_permissions", 0),
                    "last_session": status.get("last_session"),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/free-time/results")
        async def get_free_time_results():
            try:
                from pathlib import Path
                results_dir = Path.home() / ".jarvis" / "free_time" / "saved_results"
                results = []
                if results_dir.exists():
                    for f in sorted(results_dir.glob("*.json"), reverse=True)[:20]:
                        try:
                            data = json.loads(f.read_text(encoding='utf-8'))
                            results.append({
                                "id": f.stem,
                                "timestamp": data.get("timestamp", ""),
                                "type": data.get("type", "unknown"),
                                "summary": str(data.get("result", ""))[:200],
                            })
                        except Exception as e:
                            logger.debug(f"Failed to parse intuition result: {e}")
                return {"success": True, "results": results}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/intuition/status")
        async def get_intuition_status():
            try:
                from core.intuition_bridge import get_intuition_bridge
                bridge = get_intuition_bridge()
                status = bridge.get_status() if hasattr(bridge, 'get_status') else {}
                return {
                    "success": True,
                    "insights_processed": status.get("insights_processed", 0),
                    "actions_suggested": status.get("actions_suggested", 0),
                    "connected_subsystems": status.get("connected_subsystems", []),
                    "last_insight": status.get("last_insight"),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/hardware/status")
        async def get_hardware_status():
            try:
                from core.hardware import get_hardware_detector, AcceleratorType
                detector = get_hardware_detector()
                cpu_info = None
                gpu_info = {"available": False, "name": None, "memory_gb": 0}
                npu_info = {"available": False, "name": None}
                
                for accel in detector.accelerators:
                    if accel.type == AcceleratorType.CPU:
                        cpu_info = accel
                    elif accel.type in (AcceleratorType.ROCM_GPU, AcceleratorType.CUDA_GPU, AcceleratorType.VULKAN_GPU):
                        gpu_info = {
                            "available": True,
                            "name": accel.name,
                            "memory_gb": accel.memory_mb // 1024,
                        }
                    elif accel.type == AcceleratorType.AMD_NPU:
                        npu_info = {
                            "available": True,
                            "name": accel.name,
                        }
                
                return {
                    "success": True,
                    "cpu": {
                        "model": cpu_info.name if cpu_info else "Unknown",
                        "cores": cpu_info.compute_units if cpu_info else 0,
                        "memory_gb": (cpu_info.memory_mb // 1024) if cpu_info else 0,
                    },
                    "gpu": gpu_info,
                    "npu": npu_info,
                    "primary_accelerator": detector.primary_accelerator.value if detector.primary_accelerator else "cpu",
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/vector-memory/search")
        async def search_vector_memory(query: str = ""):
            try:
                from core.vector_memory import get_vector_memory
                vm = get_vector_memory()
                if not query:
                    return {"success": True, "results": [], "count": 0}
                results = vm.search(query, k=10)
                return {
                    "success": True,
                    "results": [{"id": r.id, "content": r.content[:200], "score": r.score} for r in results],
                    "count": len(results),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/knowledge-graph/stats")
        async def get_knowledge_graph_stats():
            try:
                from core.knowledge_graph import get_knowledge_graph
                kg = get_knowledge_graph()
                stats = kg.get_graph_statistics() if hasattr(kg, 'get_graph_statistics') else {}
                return {
                    "success": True,
                    "total_nodes": stats.get("total_nodes", 0),
                    "total_edges": stats.get("total_edges", 0),
                    "communities": stats.get("communities", 0),
                    "node_types": stats.get("node_types", {}),
                    "edge_types": stats.get("edge_types", {}),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/knowledge-graph/nodes")
        async def get_knowledge_graph_nodes(limit: int = 100):
            try:
                from core.knowledge_graph import get_knowledge_graph
                kg = get_knowledge_graph()
                nodes = []
                for node_id, node in list(kg.nodes.items())[:limit]:
                    nodes.append(node.to_dict())
                return {"success": True, "nodes": nodes, "total": len(kg.nodes)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/knowledge-graph/edges")
        async def get_knowledge_graph_edges(limit: int = 200):
            try:
                from core.knowledge_graph import get_knowledge_graph
                kg = get_knowledge_graph()
                edges = []
                for edge_key, edge in list(kg.edge_details.items())[:limit]:
                    edges.append(edge.to_dict())
                return {"success": True, "edges": edges, "total": len(kg.edge_details)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/knowledge-graph/insights")
        async def get_knowledge_graph_insights(limit: int = 20):
            try:
                from core.knowledge_graph import get_knowledge_graph
                kg = get_knowledge_graph()
                insights = kg.get_important_insights(limit=limit)
                return {"success": True, "insights": insights}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/vector-memory/entries")
        async def get_vector_memory_entries(limit: int = 100):
            try:
                from core.vector_memory import get_vector_memory
                vm = get_vector_memory()
                entries = []
                for node_id, node in list(vm.index.nodes.items())[:limit]:
                    entry = node.entry
                    entries.append({
                        "id": entry.id,
                        "text": entry.metadata.get("text", "")[:200],
                        "importance": vm.importance_scores.get(entry.id, 0.5),
                        "access_count": entry.access_count,
                        "created_at": entry.created_at.isoformat() if entry.created_at else None,
                    })
                return {"success": True, "entries": entries, "total": len(vm.index.nodes)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/memory/watchdog")
        async def get_memory_watchdog():
            try:
                from core.memory_watchdog import get_memory_watchdog
                watchdog = get_memory_watchdog()
                stats = watchdog.get_stats()
                return {"success": True, **stats}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/memory/cleanup")
        async def force_memory_cleanup(level: str = "medium"):
            try:
                from core.memory_watchdog import get_memory_watchdog, MemoryState
                from core.memory_manager import get_memory_manager
                watchdog = get_memory_watchdog()
                manager = get_memory_manager()
                
                state = {
                    "light": MemoryState.WARNING,
                    "medium": MemoryState.HIGH,
                    "heavy": MemoryState.CRITICAL,
                }.get(level, MemoryState.HIGH)
                
                watchdog_stats = watchdog.force_cleanup(state)
                manager_stats = manager.force_cleanup(level)
                
                return {
                    "success": True,
                    "watchdog": {
                        "caches_cleared": watchdog_stats.caches_cleared,
                        "gc_collected": watchdog_stats.gc_collected,
                        "freed_mb": watchdog_stats.freed_mb,
                    },
                    "manager": manager_stats,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/health")
        async def get_health():
            try:
                from core.health import get_health_manager
                manager = get_health_manager()
                basic = manager.get_all_health()
            except Exception:
                basic = {}

            try:
                from core.health_check import get_health_checker
                checker = get_health_checker()
                result = checker.check_all()
                return {"success": True, **basic, "subsystem_health": result.to_dict()}
            except Exception as e:
                return {"success": True, **basic, "subsystem_health": {"error": safe_error_message(e)}}

        @self.app.get("/api/system/status")
        async def system_status():
            """Comprehensive system status: health, metrics, Oracle CB, rate limiter, router, KG."""
            try:
                from core.health_check import get_jarvis_status
                return {"success": True, **get_jarvis_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/system/metrics")
        async def system_metrics():
            """System-level metrics: CPU, memory, disk, process."""
            try:
                from core.health_check import get_system_metrics
                return {"success": True, **get_system_metrics()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/system/observability")
        async def observability_metrics():
            """Observability metrics: counters, gauges, histograms."""
            try:
                from core.observability import get_observability
                obs = get_observability()
                return {"success": True, "metrics": obs.get_all_metrics()}
            except Exception as e:
                logger.debug(f"Observability API failed: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/system/oracle-cb")
        async def oracle_cb_status():
            """Oracle circuit breaker state."""
            try:
                from core.oracle_integration import cb_state, cache_stats
                return {"success": True, "circuit_breaker": cb_state(), "cache": cache_stats()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/system/rate-limits")
        async def rate_limits_status():
            """Rate limiter statistics and configured endpoint limits."""
            try:
                from core.rate_limiter import get_rate_limiter, ENDPOINT_LIMITS
                rl = get_rate_limiter()
                limits = {ep: {"rps": l.requests_per_second, "rpm": l.requests_per_minute,
                               "burst": l.burst_size, "concurrent": l.concurrent_requests}
                          for ep, l in ENDPOINT_LIMITS.items()}
                return {"success": True, "stats": rl.get_stats(), "endpoint_limits": limits}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/system/events-stream")
        async def events_stream(request: Request):
            """SSE stream for live execution events."""
            from starlette.responses import StreamingResponse
            import asyncio
            import json

            async def event_generator():
                last_index = 0
                try:
                    from core.execution_router import EnhancedExecutionRouter
                    router = EnhancedExecutionRouter.get_instance(agent=self.agent)
                    emitter = router.event_emitter
                except Exception:
                    yield f"data: {json.dumps({'error': 'router unavailable'})}\n\n"
                    return

                for _ in range(300):
                    events = emitter.get_events(limit=10)
                    new_events = events[last_index:] if len(events) > last_index else []
                    for event in new_events:
                        yield f"data: {json.dumps(event)}\n\n"
                    last_index = len(events)
                    await asyncio.sleep(2)

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        @self.app.get("/api/circuit-breakers")
        async def get_circuit_breakers():
            try:
                from core.circuit_breaker import get_all_circuit_stats
                stats = get_all_circuit_stats()
                return {"success": True, "breakers": stats}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/circuit-breakers/{name}/reset")
        async def reset_circuit_breaker(name: str):
            try:
                from core.circuit_breaker import CircuitBreakerManager
                manager = CircuitBreakerManager.get_instance()
                breaker = manager.get(name)
                if breaker:
                    breaker.reset()
                    return {"success": True, "message": f"Circuit breaker {name} reset"}
                return {"success": False, "error": f"Circuit breaker {name} not found"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/rate-limiter/stats")
        async def get_rate_limiter_stats():
            try:
                from core.rate_limiter import get_rate_limiter
                limiter = get_rate_limiter()
                return {"success": True, **limiter.get_stats()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/runtime/context")
        async def get_runtime_context():
            try:
                from core.execution_context import get_execution_context
                ctx = get_execution_context()
                return {"success": True, **ctx.get_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/runtime/cd")
        async def runtime_cd(request: Request):
            try:
                from core.execution_context import get_execution_context
                body = await request.json()
                path = body.get("path", "")
                if not path:
                    return {"success": False, "error": "No path provided"}
                ctx = get_execution_context()
                success, result = ctx.cd(path)
                return {"success": success, "cwd": result if success else None, "error": result if not success else None}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/runtime/execute")
        async def runtime_execute(request: Request):
            try:
                from core.execution_context import get_execution_context
                body = await request.json()
                command = body.get("command", "")
                timeout = float(body.get("timeout", 60.0))
                if not command:
                    return {"success": False, "error": "No command provided"}
                ctx = get_execution_context()
                result = ctx.run(command, timeout=min(timeout, 300.0))
                return {"success": result.get("exit_code", -1) == 0, **result}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/runtime/commands")
        async def get_runtime_commands(limit: int = 20):
            try:
                from core.execution_context import get_execution_context
                ctx = get_execution_context()
                return {"success": True, "commands": ctx.get_recent_commands(limit)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/runtime/files")
        async def get_runtime_files(limit: int = 20):
            try:
                from core.execution_context import get_execution_context
                ctx = get_execution_context()
                return {"success": True, "operations": ctx.get_recent_files(limit)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/runtime/heartbeat")
        async def runtime_heartbeat():
            try:
                from core.execution_context import get_execution_context
                ctx = get_execution_context()
                return {"success": True, **ctx.heartbeat()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/runtime/connections")
        async def runtime_connections():
            try:
                from core.ws_manager import get_ws_connection_manager
                ws_mgr = get_ws_connection_manager()
                return {
                    "success": True,
                    "ws_connections": len(self.connections),
                    "mobile_connections": len(self._mobile_connections),
                    "api_activity_subscribers": len(self._api_activity_connections),
                    "active_ws_sessions": len(self._ws_sessions),
                    "consecutive_agent_errors": self._consecutive_agent_errors,
                    **ws_mgr.get_status(),
                    "connection_details": ws_mgr.get_all_connections(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/runtime/ws-events")
        async def runtime_ws_events(limit: int = 50, event_type: str | None = None):
            try:
                from core.ws_manager import get_ws_connection_manager
                ws_mgr = get_ws_connection_manager()
                return {"success": True, "events": ws_mgr.get_events(limit, event_type)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/awareness/report")
        async def awareness_report():
            try:
                from core.system_awareness import get_system_awareness
                sa = get_system_awareness()
                return {"success": True, **sa.get_awareness_report()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/awareness/resources")
        async def awareness_resources(minutes: int = 30):
            try:
                from core.system_awareness import get_system_awareness
                sa = get_system_awareness()
                return {"success": True, **sa.get_resource_trend(minutes)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/awareness/processes")
        async def awareness_processes(top_n: int = 20):
            try:
                from core.system_awareness import get_system_awareness
                sa = get_system_awareness()
                return {"success": True, "processes": sa.get_running_processes(top_n)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/awareness/services")
        async def awareness_services():
            try:
                from core.system_awareness import get_system_awareness
                sa = get_system_awareness()
                return {"success": True, "services": [s.to_dict() for s in sa.get_services_status()]}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/awareness/docker")
        async def awareness_docker():
            try:
                from core.system_awareness import get_system_awareness
                sa = get_system_awareness()
                return {"success": True, "containers": sa.get_docker_containers()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/adaptive-loop/status")
        async def adaptive_loop_status():
            try:
                from core.adaptive_agent_loop import get_adaptive_agent_loop
                loop = get_adaptive_agent_loop(agent=self.agent)
                return {"success": True, **loop.get_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/watchdog/status")
        async def watchdog_status():
            try:
                from core.watchdog import get_watchdog
                wd = get_watchdog()
                return {"success": True, **wd.get_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/watchdog/events")
        async def watchdog_events(limit: int = 50, event_type: str | None = None):
            try:
                from core.watchdog import get_watchdog
                wd = get_watchdog()
                return {"success": True, "events": wd.get_events(limit, event_type)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        # Distinct path: this rich dashboard was registered on the same
        # /api/health path as get_health (~290 lines earlier), so FastAPI's
        # first-match-wins made it permanently unreachable. Surfaced by the
        # route-registration smoke check after the _setup_routes split.
        @self.app.get("/api/health/dashboard")
        async def health_dashboard():
            try:
                from core.execution_context import get_execution_context
                from core.system_awareness import get_system_awareness
                from core.watchdog import get_watchdog
                from core.event_bus import get_event_bus
                from core.sandbox import get_sandbox_manager
                from core.session_persistence import get_session_persistence
                from core.context_compressor import get_context_compressor
                from core.tool_chain_optimizer import get_tool_chain_optimizer
                from core.ws_manager import get_ws_connection_manager
                return {
                    "success": True,
                    "execution_context": get_execution_context().get_status(),
                    "system_awareness": get_system_awareness().get_awareness_report(),
                    "watchdog": get_watchdog().get_status(),
                    "event_bus": get_event_bus().get_status(),
                    "sandbox": get_sandbox_manager().get_status(),
                    "session_persistence": get_session_persistence().get_status(),
                    "context_compressor": get_context_compressor().get_status(),
                    "tool_chain_optimizer": get_tool_chain_optimizer().get_status(),
                    "ws_manager": get_ws_connection_manager().get_status(),
                    "timestamp": time.time(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/events")
        async def event_bus_history(channel: str | None = None, limit: int = 50):
            try:
                from core.event_bus import get_event_bus
                bus = get_event_bus()
                return {
                    "success": True,
                    "events": bus.get_history(channel=channel, limit=limit),
                    "channels": bus.get_channels(),
                    "status": bus.get_status(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/events/{channel}")
        async def event_bus_channel(channel: str, limit: int = 50):
            try:
                from core.event_bus import get_event_bus
                bus = get_event_bus()
                return {
                    "success": True,
                    "channel": channel,
                    "events": bus.get_buffer(channel, limit=limit),
                    "subscribers": bus.get_subscriber_count(channel),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/sandbox/status")
        async def sandbox_status():
            try:
                from core.sandbox import get_sandbox_manager
                return {"success": True, **get_sandbox_manager().get_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/sandbox/history")
        async def sandbox_history(limit: int = 50):
            try:
                from core.sandbox import get_sandbox_manager
                return {"success": True, "history": get_sandbox_manager().get_history(limit)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/sessions/status")
        async def sessions_status():
            try:
                from core.session_persistence import get_session_persistence
                return {"success": True, **get_session_persistence().get_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/sessions")
        async def sessions_list():
            try:
                from core.session_persistence import get_session_persistence
                return {"success": True, "sessions": get_session_persistence().list_sessions()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/compressor/status")
        async def compressor_status():
            try:
                from core.context_compressor import get_context_compressor
                return {"success": True, **get_context_compressor().get_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/optimizer/status")
        async def optimizer_status():
            try:
                from core.tool_chain_optimizer import get_tool_chain_optimizer
                return {"success": True, **get_tool_chain_optimizer().get_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/backpressure/status")
        async def backpressure_status():
            try:
                from core.streaming_pipeline import get_backpressure_controller
                return {"success": True, **get_backpressure_controller().get_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/streaming/status")
        async def streaming_status():
            try:
                from core.streaming_pipeline import BackpressureController, StreamingPipeline, AdaptiveThrottle
                return {"success": True, "backpressure": "available", "pipeline": "available", "adaptive_throttle": "available"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/memory-store/status")
        async def memory_store_status():
            try:
                from core.agent_memory_store import get_agent_memory_store
                return {"success": True, **get_agent_memory_store().get_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/memory-store/list")
        async def memory_store_list(request: Request):
            try:
                from core.agent_memory_store import get_agent_memory_store
                namespace = request.query_params.get("namespace", "")
                store = get_agent_memory_store()
                keys = store.list_keys(namespace or None)
                namespaces = store.list_namespaces()
                return {"success": True, "keys": keys, "namespaces": namespaces}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/memory-store/put")
        async def memory_store_put(request: Request):
            try:
                from core.agent_memory_store import get_agent_memory_store
                data = await request.json()
                namespace = data.get("namespace", "default")
                key = data.get("key", "")
                value = data.get("value")
                ttl = data.get("ttl")
                if not key or value is None:
                    return {"success": False, "error": "key and value required"}
                store = get_agent_memory_store()
                ok = store.put(namespace, key, value, ttl=ttl)
                return {"success": ok}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/memory-store/get")
        async def memory_store_get(request: Request):
            try:
                from core.agent_memory_store import get_agent_memory_store
                namespace = request.query_params.get("namespace", "default")
                key = request.query_params.get("key", "")
                if not key:
                    return {"success": False, "error": "key required"}
                store = get_agent_memory_store()
                value = store.get(namespace, key)
                return {"success": True, "value": value, "exists": value is not None}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.delete("/api/memory-store/delete")
        async def memory_store_delete(request: Request):
            try:
                from core.agent_memory_store import get_agent_memory_store
                namespace = request.query_params.get("namespace", "default")
                key = request.query_params.get("key", "")
                if not key:
                    return {"success": False, "error": "key required"}
                store = get_agent_memory_store()
                ok = store.delete(namespace, key)
                return {"success": ok}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/health-router/status")
        async def health_router_status():
            try:
                from core.health_router import get_health_router
                return {"success": True, **get_health_router().get_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/unified-ratelimit/status")
        async def unified_ratelimit_status():
            try:
                from core.unified_rate_limiter import get_unified_rate_limiter
                return {"success": True, **get_unified_rate_limiter().get_status()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/audit/stats")
        async def get_audit_stats():
            try:
                from core.audit import get_audit_logger
                audit = get_audit_logger()
                return {"success": True, **audit.get_stats()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/browser/sessions")
        async def get_browser_sessions():
            try:
                from tools.browser_tools import get_browser_manager
                manager = get_browser_manager()
                sessions = manager.list_sessions() if hasattr(manager, 'list_sessions') else []
                return {"success": True, "sessions": sessions}
            except ImportError:
                return {"success": False, "error": "Browser tools not available"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/swarm/status")
        async def get_swarm_status():
            try:
                from core.swarm import get_swarm_coordinator
                swarm = get_swarm_coordinator()
                status = swarm.get_status() if hasattr(swarm, 'get_status') else {}
                return {
                    "success": True,
                    "active_agents": status.get("active_agents", 0),
                    "pending_tasks": status.get("pending_tasks", 0),
                    "completed_tasks": status.get("completed_tasks", 0),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/subsystems/all")
        async def get_all_subsystems():
            try:
                subsystems = {}
                try:
                    from core.erdos_engine import get_erdos_engine
                    engine = get_erdos_engine()
                    status = engine.get_status() if hasattr(engine, 'get_status') else {}
                    subsystems["erdos"] = {"status": "active" if status.get("running") else "idle", **status}
                except Exception:
                    subsystems["erdos"] = {"status": "unavailable"}

                try:
                    from core.daydream_engine import get_daydream_engine
                    engine = get_daydream_engine()
                    status = engine.get_status() if hasattr(engine, 'get_status') else {}
                    subsystems["daydream"] = {"status": "active" if status.get("running") else "idle", **status}
                except Exception:
                    subsystems["daydream"] = {"status": "unavailable"}

                try:
                    from core.evolution_orchestrator import get_evolution_orchestrator
                    orch = get_evolution_orchestrator()
                    status = orch.get_status() if hasattr(orch, 'get_status') else {}
                    subsystems["evolution"] = {"status": "active" if status.get("running") else "idle", **status}
                except Exception:
                    subsystems["evolution"] = {"status": "unavailable"}

                try:
                    from core.learning_pipeline import get_learning_pipeline
                    pipeline = get_learning_pipeline()
                    status = pipeline.get_status() if hasattr(pipeline, 'get_status') else {}
                    subsystems["learning"] = {"status": "active", **status}
                except Exception:
                    subsystems["learning"] = {"status": "unavailable"}

                try:
                    from core.defense import get_defense_agent
                    agent = get_defense_agent()
                    status = agent.get_status() if hasattr(agent, 'get_status') else {}
                    subsystems["defense"] = {"status": "active", "running": True, **status}
                except Exception:
                    subsystems["defense"] = {"status": "unavailable"}

                try:
                    from core.intuition_bridge import get_intuition_bridge
                    bridge = get_intuition_bridge()
                    status = bridge.get_status() if hasattr(bridge, 'get_status') else {}
                    subsystems["intuition"] = {"status": "active", "running": True, **status}
                except Exception:
                    subsystems["intuition"] = {"status": "unavailable"}

                try:
                    from core.free_time import get_free_time_engine
                    engine = get_free_time_engine()
                    status = engine.get_status() if hasattr(engine, 'get_status') else {}
                    sandbox_active = status.get("sandbox_active", False)
                    subsystems["free_time"] = {"status": "active" if sandbox_active else "idle", "running": sandbox_active, **status}
                except Exception:
                    subsystems["free_time"] = {"status": "unavailable"}

                try:
                    from core.music_creation_engine import get_music_creation_engine
                    engine = get_music_creation_engine()
                    status = engine.get_stats() if hasattr(engine, 'get_stats') else {}
                    running = status.get("running", False)
                    subsystems["music_creation"] = {
                        "status": "active" if running else "idle",
                        "running": running,
                        "current_style": status.get("current_style"),
                        "compositions_created": status.get("compositions_created", 0),
                        "tokens_observed": status.get("tokens_observed", 0),
                    }
                except Exception:
                    subsystems["music_creation"] = {"status": "unavailable"}

                return {"success": True, "subsystems": subsystems}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}
        # Sibling-route handlers called from other route groups.
        self._route_handlers["get_all_subsystems"] = get_all_subsystems
        self._route_handlers["get_hardware_status"] = get_hardware_status
        self._route_handlers["get_vector_memory_entries"] = get_vector_memory_entries
        self._route_handlers["get_knowledge_graph_nodes"] = get_knowledge_graph_nodes


    def _setup_memory_goals_routes(self):
        @self.app.get("/api/memory/stats")
        async def get_memory_stats():
            try:
                stats = {
                    "conversation": {},
                    "vector": {},
                    "knowledge_graph": {},
                }
                
                if hasattr(self.agent, 'memory'):
                    mem = self.agent.memory
                    stats["conversation"] = {
                        "messages": len(mem.get_messages_for_llm()) if hasattr(mem, 'get_messages_for_llm') else 0,
                        "turns": getattr(mem, '_turn_count', 0),
                    }
                    if hasattr(mem, 'get_enhanced_stats'):
                        enhanced = mem.get_enhanced_stats()
                        stats["conversation"]["enhanced"] = enhanced
                
                try:
                    from core.vector_memory import get_vector_memory
                    vm = get_vector_memory()
                    stats["vector"] = {
                        "available": True,
                        "vector_count": len(vm.index.nodes),
                        **vm.stats(),
                    }
                except Exception:
                    stats["vector"] = {"available": False}
                
                try:
                    from core.knowledge_graph import get_knowledge_graph
                    kg = get_knowledge_graph()
                    kg_stats = kg.get_graph_statistics()
                    stats["knowledge_graph"] = {
                        "available": True,
                        "total_nodes": kg_stats.get("total_nodes", 0),
                        "total_edges": kg_stats.get("total_edges", 0),
                        "communities": kg_stats.get("communities", 0),
                    }
                except Exception:
                    stats["knowledge_graph"] = {"available": False}
                
                return {"success": True, "stats": stats}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/memory/retrieve")
        async def memory_retrieve(request: Request):
            """Unified retrieval across all memory systems."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["query"], optional=["top_k", "sources", "min_score"],
                    max_string_length=5000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                query = data["query"]
                top_k = data.get("top_k", 10)
                sources = data.get("sources")  # ['conversation', 'vector', 'knowledge', 'persistent']
                min_score = data.get("min_score", 0.3)
                
                if not query:
                    return {"success": False, "error": "Query required"}
                
                from core.memory_hub import get_memory_hub
                hub = get_memory_hub()
                
                results = hub.retrieve(
                    query=query,
                    top_k=top_k,
                    sources=sources,
                    min_score=min_score,
                )
                
                return {
                    "success": True,
                    "results": [
                        {
                            "content": r.content[:500],
                            "source": r.source,
                            "score": round(r.score, 4),
                            "importance": round(r.importance, 3),
                            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                            "metadata": r.metadata,
                        }
                        for r in results
                    ],
                    "stats": {
                        "embedding_cache_hits": hub._stats.embedding_cache_hits,
                        "embedding_cache_misses": hub._stats.embedding_cache_misses,
                        "last_retrieval_ms": round(hub._stats.last_retrieval_ms, 2),
                    }
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/memory/store")
        async def memory_store(request: Request):
            """Store content in memory."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["content"], optional=["memory_type", "importance", "metadata"],
                    max_string_length=50000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                content = data["content"]
                memory_type = data.get("memory_type", "semantic")  # conversation, semantic, knowledge, insight, persistent
                importance = data.get("importance", 0.5)
                metadata = data.get("metadata", {})
                
                if not content:
                    return {"success": False, "error": "Content required"}
                
                from core.memory_hub import get_memory_hub
                hub = get_memory_hub()
                
                memory_id = hub.store(
                    content=content,
                    memory_type=memory_type,
                    importance=importance,
                    metadata=metadata,
                )
                
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "memory_type": memory_type,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/memory/hub/stats")
        async def memory_hub_stats():
            """Get memory hub statistics."""
            try:
                from core.memory_hub import get_memory_hub
                hub = get_memory_hub()
                stats = hub.get_stats()
                
                return {
                    "success": True,
                    "stats": {
                        "conversation_turns": stats.conversation_turns,
                        "vector_entries": stats.vector_entries,
                        "knowledge_nodes": stats.knowledge_nodes,
                        "knowledge_edges": stats.knowledge_edges,
                        "persistent_memories": stats.persistent_memories,
                        "embedding_cache_hits": stats.embedding_cache_hits,
                        "embedding_cache_misses": stats.embedding_cache_misses,
                        "cache_hit_rate": stats.embedding_cache_hits / max(1, stats.embedding_cache_hits + stats.embedding_cache_misses),
                        "last_retrieval_ms": round(stats.last_retrieval_ms, 2),
                    }
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/goals")
        async def get_goals():
            try:
                from core.autonomous_goals import get_autonomous_goals
                engine = get_autonomous_goals()
                goals = engine.get_actionable() if hasattr(engine, 'get_actionable') else []
                return {
                    "success": True,
                    "goals": [
                        {
                            "id": g.goal_id if hasattr(g, 'goal_id') else str(g.id) if hasattr(g, 'id') else '',
                            "name": g.name if hasattr(g, 'name') else '',
                            "description": g.description if hasattr(g, 'description') else '',
                            "status": str(g.status.value) if hasattr(g.status, 'value') else str(g.status) if hasattr(g, 'status') else 'unknown',
                            "progress": getattr(getattr(g, 'progress', None), 'progress_percent', 0) if hasattr(g, 'progress') and g.progress else getattr(g, 'progress_percent', 0),
                            "milestones": len(getattr(g, 'milestones', [])),
                            "completed_milestones": sum(1 for m in getattr(g, 'milestones', []) if getattr(m, 'completed', False)),
                        }
                        for g in goals
                    ],
                    "total": len(goals),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/goals")
        async def create_goal(request: Request):
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["name", "description", "priority"],
                    max_string_length=5000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                from core.autonomous_goals import get_autonomous_goals
                engine = get_autonomous_goals()
                
                goal = engine.create(
                    name=data.get("name", "New Goal"),
                    description=data.get("description", ""),
                    priority=data.get("priority", 5),
                )
                
                return {
                    "success": True,
                    "goal": {
                        "id": goal.goal_id if hasattr(goal, 'goal_id') else '',
                        "name": goal.name if hasattr(goal, 'name') else '',
                        "description": goal.description if hasattr(goal, 'description') else '',
                        "status": str(goal.status.value) if hasattr(goal.status, 'value') else str(goal.status) if hasattr(goal, 'status') else 'unknown',
                    },
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/schedules")
        async def get_schedules():
            try:
                from core.workflow_scheduler import get_workflow_scheduler
                scheduler = get_workflow_scheduler()
                
                schedules = scheduler.get_schedules() if hasattr(scheduler, 'get_schedules') else []
                
                return {
                    "success": True,
                    "schedules": [
                        {
                            "id": s.schedule_id if hasattr(s, 'schedule_id') else str(s.id) if hasattr(s, 'id') else '',
                            "name": s.name if hasattr(s, 'name') else '',
                            "cron": s.cron_expression if hasattr(s, 'cron_expression') else '',
                            "next_run": str(s.next_run_at) if hasattr(s, 'next_run_at') and s.next_run_at else None,
                            "last_run": str(s.last_run_at) if hasattr(s, 'last_run_at') and s.last_run_at else None,
                            "enabled": hasattr(s, 'status') and str(s.status) == 'ACTIVE',
                            "run_count": getattr(s, 'run_count', 0),
                        }
                        for s in schedules
                    ],
                    "total": len(schedules),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/tasks")
        async def get_tasks():
            try:
                from core.task_persistence import get_task_persistence, TaskStatus
                persistence = get_task_persistence()
                
                pending = persistence.get_pending_tasks() if hasattr(persistence, 'get_pending_tasks') else []
                running = persistence.get_running_tasks() if hasattr(persistence, 'get_running_tasks') else []
                all_tasks = list(persistence._tasks.values()) if hasattr(persistence, '_tasks') else []
                completed = [t for t in all_tasks if hasattr(t, 'status') and t.status == TaskStatus.COMPLETED][:50]
                
                return {
                    "success": True,
                    "pending": [
                        {
                            "id": t.task_id if hasattr(t, 'task_id') else str(t.id) if hasattr(t, 'id') else '',
                            "name": t.name if hasattr(t, 'name') else '',
                            "priority": t.priority.value if hasattr(t, 'priority') and hasattr(t.priority, 'value') else 5,
                            "created": str(t.created_at) if hasattr(t, 'created_at') else None,
                        }
                        for t in pending[:20]
                    ],
                    "running": [
                        {
                            "id": t.task_id if hasattr(t, 'task_id') else str(t.id) if hasattr(t, 'id') else '',
                            "name": t.name if hasattr(t, 'name') else '',
                            "started": str(t.started_at) if hasattr(t, 'started_at') else None,
                        }
                        for t in running
                    ],
                    "completed": [
                        {
                            "id": t.task_id if hasattr(t, 'task_id') else str(t.id) if hasattr(t, 'id') else '',
                            "name": t.name if hasattr(t, 'name') else '',
                            "success": getattr(t, 'result', {}).get('success', None) if hasattr(t, 'result') else None,
                            "duration": None,
                        }
                        for t in completed[:20]
                    ],
                    "stats": {
                        "pending_count": len(pending),
                        "running_count": len(running),
                        "completed_count": len(completed),
                    },
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/self-awareness/manifest")
        async def get_self_awareness_manifest():
            try:
                from core.self_awareness import get_self_awareness
                sa = get_self_awareness()
                
                manifest = {}
                if hasattr(sa, '_capabilities') and sa._capabilities:
                    if hasattr(sa._capabilities, 'to_dict'):
                        manifest = sa._capabilities.to_dict()
                    elif hasattr(sa._capabilities, '__dict__'):
                        manifest = {k: str(v) for k, v in sa._capabilities.__dict__.items() if not k.startswith('_')}
                
                identity = {}
                if hasattr(sa, 'get_identity_context'):
                    identity_text = sa.get_identity_context()
                    identity = {"context": identity_text[:500] if identity_text else ""}
                elif hasattr(sa, '_model_identity'):
                    identity = {"model": str(sa._model_identity)}
                
                return {
                    "success": True,
                    "manifest": manifest,
                    "identity": identity,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/self-awareness/status")
        async def get_self_awareness_status():
            try:
                from core.self_awareness import get_self_awareness
                sa = get_self_awareness()
                
                status = sa.get_status() if hasattr(sa, 'get_status') else {}
                
                agentic_info = None
                try:
                    from core.agentic_orchestrator import get_agentic_orchestrator
                    orch = get_agentic_orchestrator()
                    if orch and hasattr(sa, 'get_agentic_awareness'):
                        agentic_info = sa.get_agentic_awareness()
                except Exception as e:
                    logger.debug(f"Failed to get agentic awareness: {e}")
                
                return {
                    "success": True,
                    "status": status,
                    "agentic": agentic_info,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/agentic/status")
        async def get_agentic_status():
            try:
                result = {
                    "orchestrator": None,
                    "goals": None,
                    "schedules": None,
                    "tasks": None,
                    "harness": None,
                }
                
                try:
                    from core.agentic_harness import get_agentic_harness
                    harness = get_agentic_harness()
                    result["harness"] = harness.get_status()
                except Exception:
                    pass
                
                try:
                    from core.agentic_orchestrator import get_agentic_orchestrator
                    orch = get_agentic_orchestrator()
                    if hasattr(orch, 'get_stats'):
                        result["orchestrator"] = orch.get_stats()
                except Exception as e:
                    logger.debug(f"Failed to get agentic orchestrator stats: {e}")
                
                try:
                    from core.autonomous_goals import get_autonomous_goals
                    engine = get_autonomous_goals()
                    goals = engine.get_actionable() if hasattr(engine, 'get_actionable') else []
                    result["goals"] = {
                        "active": len([g for g in goals if str(getattr(g, 'status', '')).lower() == 'active']),
                        "total": len(goals),
                    }
                except Exception as e:
                    logger.debug(f"Failed to get autonomous goals: {e}")
                
                try:
                    from core.workflow_scheduler import get_workflow_scheduler
                    scheduler = get_workflow_scheduler()
                    schedules = scheduler.get_schedules() if hasattr(scheduler, 'get_schedules') else []
                    result["schedules"] = {
                        "enabled": len([s for s in schedules if hasattr(s, 'status') and str(s.status) == 'ACTIVE']),
                        "total": len(schedules),
                    }
                except Exception as e:
                    logger.debug(f"Failed to get workflow schedules: {e}")
                
                try:
                    from core.task_persistence import get_task_persistence
                    persistence = get_task_persistence()
                    pending = persistence.get_pending_tasks() if hasattr(persistence, 'get_pending_tasks') else []
                    running = persistence.get_running_tasks() if hasattr(persistence, 'get_running_tasks') else []
                    result["tasks"] = {
                        "pending": len(pending),
                        "running": len(running),
                    }
                except Exception as e:
                    logger.debug(f"Failed to get task persistence status: {e}")
                
                return {"success": True, **result}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}
        # Sibling-route handlers called from other route groups.
        self._route_handlers["get_memory_stats"] = get_memory_stats


    def _setup_models_permissions_routes(self):
        @self.app.get("/api/models")
        async def get_models():
            try:
                from core.backend_llm import get_model_status
                
                status = get_model_status()
                
                return {
                    "success": True,
                    "primary": status.get("primary", {}),
                    "backend": status.get("backend", {}),
                    "routing": status.get("routing", {}),
                    "tiers": {
                        "primary": {
                            "description": "Full access model for interactive chat and vision",
                            "permissions": "All tools and categories enabled",
                        },
                        "backend": {
                            "description": "Restricted model for analysis and background tasks",
                            "permissions": "Limited to analysis tools, no dangerous operations",
                        },
                    },
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/models/check-tool")
        async def check_tool_access(tool: str, category: str = "", tier: str = "primary"):
            try:
                from core.backend_llm import check_tool_permission, ModelTier
                
                tier_enum = ModelTier.PRIMARY if tier == "primary" else ModelTier.BACKEND
                allowed = check_tool_permission(tool, category, tier_enum)
                
                return {
                    "success": True,
                    "tool": tool,
                    "category": category,
                    "tier": tier,
                    "allowed": allowed,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/models/switch")
        async def switch_model(request: Request):
            """Switch the primary model to a different model."""
            _audit_log(request, "model_switch")
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["model"], max_string_length=256
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                model = data["model"]
                
                # Primary bi-model system + fallbacks for resilience
                available_models = ["kimi-k2.5", "glm-5", "qwen3.5:397b"]  # Kimi (chat), GLM-5 (reasoning), Qwen (fallback)
                if model not in available_models:
                    return {"success": False, "error": f"Unknown model: {model}"}
                
                if hasattr(self.agent, 'llm') and self.agent.llm:
                    self.agent.llm.model = model
                    logger.info(f"Switched primary model to: {model}")
                    
                    try:
                        from core.self_awareness import get_self_awareness_engine
                        engine = get_self_awareness_engine()
                        engine.set_model("ollama_cloud", model, model)
                    except Exception as e:
                        logger.debug(f"Failed to update self-awareness model: {e}")
                    
                    return {
                        "success": True,
                        "model": model,
                        "message": f"Switched to {model}",
                    }
                
                return {"success": False, "error": "LLM client not available"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/models/available")
        async def get_available_models():
            """Get list of available models with their context windows."""
            from core.tokens import get_all_model_configs
            
            current_model = self.agent.llm.model if hasattr(self.agent, 'llm') and self.agent.llm else "unknown"
            configs = get_all_model_configs()
            
            models = []
            for model_id, config in configs.items():
                models.append({
                    "id": model_id,
                    "name": config.name,
                    "context": config.max_tokens,
                    "type": "vision" if config.supports_vision else "chat",
                    "vision": config.supports_vision,
                    "thinking": config.supports_thinking,
                    "compression_threshold": int(config.max_tokens * config.compression_threshold),
                    "proactive_summarize": int(config.max_tokens * config.proactive_summarize),
                })
            
            return {
                "success": True,
                "models": models,
                "current": current_model,
            }

        @self.app.get("/api/context/status")
        async def get_context_status():
            """Get current context/memory status including usage and thresholds."""
            try:
                from core.tokens import get_token_counter, get_model_config
                from core.context_manager import check_context_status
                
                current_model = self.agent.llm.model if hasattr(self.agent, 'llm') and self.agent.llm else "kimi-k2.5"
                messages = self.agent.memory.get_messages_for_llm() if hasattr(self.agent, 'memory') else []
                
                status = check_context_status(messages, current_model)
                
                config = get_model_config(current_model)
                counter = get_token_counter(current_model)
                
                memory_stats = {}
                if hasattr(self.agent, 'memory'):
                    memory_stats = {
                        "history_count": len(self.agent.memory.history) if hasattr(self.agent.memory, 'history') else 0,
                        "important_count": len(self.agent.memory.important_messages) if hasattr(self.agent.memory, 'important_messages') else 0,
                    }
                
                unified_stats = {}
                if hasattr(self.agent, 'unified_memory'):
                    try:
                        stats = self.agent.unified_memory.get_stats()
                        unified_stats = {
                            "total_entries": stats.total_entries,
                            "by_type": dict(stats.by_type),
                        }
                    except Exception as e:
                        logger.debug(f"Failed to get unified memory stats: {e}")
                
                return {
                    "success": True,
                    "model": current_model,
                    "context": {
                        "max_tokens": counter.max_context,
                        "total_tokens": status["total_tokens"],
                        "available_tokens": status["available_tokens"],
                        "usage_ratio": round(status["usage_ratio"], 3),
                        "needs_compression": status["needs_compression"],
                        "needs_proactive_summarize": status["needs_proactive_summarize"],
                        "compression_threshold": status["compression_threshold"],
                        "proactive_threshold": status["proactive_threshold"],
                    },
                    "model_config": {
                        "compression_threshold_pct": config.compression_threshold,
                        "proactive_summarize_pct": config.proactive_summarize,
                        "supports_vision": config.supports_vision,
                        "supports_thinking": config.supports_thinking,
                    },
                    "memory": memory_stats,
                    "unified_memory": unified_stats,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/actions/history")
        async def get_action_history(limit: int = 20):
            """Get recent action history for power drop (undo)."""
            try:
                from core.action_history import get_action_history
                history = get_action_history()
                actions = history.get_history(limit=limit)
                return {
                    "success": True,
                    "actions": [a.to_dict() for a in actions],
                    "stats": history.get_stats(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/actions/drop")
        async def drop_last_action(request: Request):
            """Power drop - undo the last action."""
            try:
                from core.action_history import get_action_history, ActionType
                data = await request.json() if request.headers.get("content-type") else {}
                action_type_str = data.get("action_type") if data else None
                action_type = ActionType(action_type_str) if action_type_str else None
                
                history = get_action_history()
                result = history.drop_last(action_type)
                return result.to_dict()
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/actions/drop/{action_id}")
        async def drop_action_by_id(action_id: str):
            """Undo a specific action by ID."""
            try:
                from core.action_history import get_action_history
                history = get_action_history()
                result = history.drop_by_id(action_id)
                return result.to_dict()
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/memory/protection")
        async def get_memory_protection():
            """Get memory protection status and rules."""
            try:
                from core.memory_protection import get_memory_protection
                protection = get_memory_protection()
                return {
                    "success": True,
                    "stats": protection.get_stats(),
                    "protected_memories": [m.to_dict() for m in protection.get_protected_memories()[:20]],
                    "rules": [r.to_dict() for r in protection.get_rules()],
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/memory/protect")
        async def protect_memory(request: Request):
            """Protect a memory from deletion/compression."""
            try:
                from core.memory_protection import get_memory_protection, ProtectionLevel
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["memory_id"], optional=["level", "content", "memory_type", "reason"],
                    max_string_length=50000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                
                protection = get_memory_protection()
                level = ProtectionLevel(data.get("level", "high"))
                
                protected = protection.protect(
                    memory_id=data["memory_id"],
                    content=data.get("content", ""),
                    memory_type=data.get("memory_type", "conversation"),
                    protection_level=level,
                    reason=data.get("reason", "user_requested"),
                )
                return {"success": True, "protected": protected.to_dict()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/memory/unprotect/{memory_id}")
        async def unprotect_memory(memory_id: str):
            """Remove protection from a memory."""
            try:
                from core.memory_protection import get_memory_protection
                protection = get_memory_protection()
                success = protection.unprotect(memory_id)
                return {"success": success}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/memory/protection/rule")
        async def add_protection_rule(request: Request):
            """Add a custom protection rule."""
            try:
                from core.memory_protection import get_memory_protection, ProtectionLevel
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["name", "pattern"], optional=["level", "memory_types", "priority"],
                    max_string_length=5000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                
                protection = get_memory_protection()
                level = ProtectionLevel(data.get("level", "medium"))
                
                rule = protection.add_rule(
                    name=data["name"],
                    pattern=data["pattern"],
                    protection_level=level,
                    memory_types=data.get("memory_types", ["conversation"]),
                    priority=data.get("priority", 50),
                )
                return {"success": True, "rule": rule.to_dict()}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/proactive/status")
        async def get_proactive_status():
            """Get proactive intelligence engine status."""
            try:
                from core.proactive_engine import get_proactive_engine
                engine = get_proactive_engine()
                return {
                    "success": True,
                    **engine.get_stats(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/proactive/trigger/{task_name}")
        async def trigger_proactive_task(task_name: str):
            """Manually trigger a proactive task."""
            try:
                from core.proactive_engine import get_proactive_engine, ProactiveTask
                engine = get_proactive_engine()
                task = ProactiveTask(task_name)
                result = await engine.trigger_task(task)
                return result.to_dict()
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/models/coordinator")
        async def get_model_coordinator_status():
            """Get model coordinator status and available models."""
            try:
                from core.model_coordinator import get_model_coordinator
                coordinator = get_model_coordinator()
                return {
                    "success": True,
                    **coordinator.get_stats(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/models/dual-status")
        async def get_dual_model_status():
            """Get detailed dual-model architecture status."""
            try:
                from core.model_coordinator import get_model_coordinator
                coordinator = get_model_coordinator()
                return {
                    "success": True,
                    **coordinator.get_dual_model_status(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/freetime/permission")
        async def set_freetime_permission(request: Request):
            """Grant or revoke free time exploration permission."""
            try:
                from core.free_time import get_free_time_engine
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["granted"], max_string_length=256
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                granted = data.get("granted", False)
                
                engine = get_free_time_engine()
                engine.grant_permission(granted)
                
                return {
                    "permission_granted": granted,
                    "can_explore": engine.has_permission(),
                    "message": f"Free time exploration {'enabled' if granted else 'disabled'}",
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/freetime/status")
        async def get_freetime_status():
            """Get free time exploration status."""
            try:
                from core.free_time import get_free_time_engine
                engine = get_free_time_engine()
                return {
                    "success": True,
                    "permission_granted": engine.has_permission(),
                    "session_count": engine._session_count,
                    "total_time": engine._total_time,
                    "state": engine._state.value,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/tools/permissions")
        async def get_tool_permissions():
            """Get list of tools that require permission."""
            try:
                permission_tools = []
                for name, tool in self.agent.tools.items():
                    if getattr(tool, 'requires_permission', False):
                        permission_tools.append({
                            "name": name,
                            "description": getattr(tool, 'description', '')[:200],
                            "category": getattr(tool, 'category', 'general'),
                        })
                return {
                    "success": True,
                    "tools": permission_tools,
                    "count": len(permission_tools),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/tools/approve")
        async def approve_tool(request: Request):
            """Approve a tool for use in current session."""
            _audit_log(request, "tool_approve")
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["tool"], optional=["approved"],
                    max_string_length=256
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                tool_name = data["tool"]
                approved = data.get("approved", False)
                
                if tool_name not in self.agent.tools:
                    return {"success": False, "error": f"Unknown tool: {tool_name}"}
                
                tool = self.agent.tools[tool_name]
                if not getattr(tool, 'requires_permission', False):
                    return {"success": True, "message": f"Tool {tool_name} does not require permission"}
                
                if approved:
                    action_key = tool_name
                    if hasattr(self.agent, 'permissions') and self.agent.permissions:
                        self.agent.permissions.grant_permission(action_key, ttl=3600)
                    return {"success": True, "approved": True, "tool": tool_name}
                else:
                    return {"success": True, "approved": False, "tool": tool_name}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/permissions")
        async def get_permissions():
            """Get current permission settings for all tools."""
            try:
                permissions = {}
                if hasattr(self.agent, 'permissions') and self.agent.permissions:
                    permissions = self.agent.permissions.get_all_permissions()
                return {"success": True, "permissions": permissions}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/permissions")
        async def set_permission(request: Request):
            """Set permission for a specific tool."""
            _audit_log(request, "set_permission")
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["tool"], optional=["permission"],
                    max_string_length=256
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                tool_name = data["tool"]
                permission = data.get("permission", "ask")
                
                if hasattr(self.agent, 'permissions') and self.agent.permissions:
                    if permission == "allow":
                        self.agent.permissions.grant_permission(tool_name, ttl=86400)  # 24 hours
                    elif permission == "deny":
                        self.agent.permissions.deny_permission(tool_name, permanently=True)
                    else:  # "ask" - remove any granted/denied permission
                        self.agent.permissions.clear_permission(tool_name)
                
                return {"success": True, "tool": tool_name, "permission": permission}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        # === Unified Permission Coordinator Endpoints ===
        @self.app.get("/api/permissions/unified/pending")
        async def get_unified_pending_permissions():
            """Get all pending permissions from all subsystems."""
            try:
                from core.permission_coordinator import get_permission_coordinator
                coordinator = get_permission_coordinator()
                pending = coordinator.get_pending()
                return {"success": True, "pending": pending, "count": len(pending)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/permissions/unified/history")
        async def get_unified_permission_history(limit: int = 100, type_filter: str = None):
            """Get permission history."""
            try:
                from core.permission_coordinator import get_permission_coordinator, PermissionType
                coordinator = get_permission_coordinator()
                
                filter_type = PermissionType(type_filter) if type_filter else None
                history = coordinator.get_history(limit=limit, type_filter=filter_type)
                return {"success": True, "history": history, "count": len(history)}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/permissions/unified/stats")
        async def get_unified_permission_stats():
            """Get unified permission statistics."""
            try:
                from core.permission_coordinator import get_permission_coordinator
                coordinator = get_permission_coordinator()
                stats = coordinator.get_stats()
                return {"success": True, **stats}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/permissions/unified/respond")
        async def respond_to_unified_permission(request: Request):
            """Respond to a pending permission (approve or deny)."""
            try:
                from core.permission_coordinator import get_permission_coordinator
                data = await request.json()
                permission_id = data.get("id", "")
                approved = data.get("approved", False)
                reason = data.get("reason", "")
                
                if not permission_id:
                    return {"success": False, "error": "Permission ID required"}
                
                coordinator = get_permission_coordinator()
                success = coordinator.respond(permission_id, approved, reason)
                
                if success:
                    action = "approved" if approved else "denied"
                    return {"success": True, "id": permission_id, "action": action, "reason": reason}
                else:
                    return {"success": False, "error": "Permission not found or expired"}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/permissions/high-risk/test")
        async def test_high_risk_block(request: Request):
            """Test endpoint to simulate high-risk command block."""
            try:
                from core.permission_coordinator import get_permission_coordinator, PermissionType, RiskLevel
                data = await request.json()
                command = data.get("command", "rm -rf /")
                
                coordinator = get_permission_coordinator()
                pending = coordinator.request(
                    type=PermissionType.TERMINAL,
                    source="test",
                    operation=f"Test: {command}",
                    risk_level=RiskLevel.HIGH,
                    context={"command": command},
                )
                
                # Auto-block high risk
                coordinator.block_high_risk(pending.id, "Test block of high-risk command")
                
                return {
                    "success": True,
                    "blocked": True,
                    "id": pending.id,
                    "reason": "High-risk commands are blocked by policy"
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/dual-model/status")
        async def get_dual_model_orchestrator_status():
            """Get dual model orchestrator status and statistics."""
            try:
                from core.dual_model import get_dual_model_orchestrator
                orchestrator = get_dual_model_orchestrator()
                return {
                    "success": True,
                    **orchestrator.get_stats(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/dual-model/route")
        async def dual_model_route(request: Request):
            """Route a task to the appropriate model."""
            try:
                from core.dual_model import get_dual_model_orchestrator, TaskType, CollaborationMode
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["task_type", "prompt", "system_prompt", "context", "mode"],
                    max_string_length=50000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                
                orchestrator = get_dual_model_orchestrator()
                
                task_type_str = data.get("task_type", "chat").upper()
                task_type = TaskType(task_type_str) if task_type_str in TaskType.__members__ else TaskType.CHAT
                
                mode_str = data.get("mode", "adaptive").lower()
                mode_map = {
                    "adaptive": CollaborationMode.ADAPTIVE,
                    "parallel": CollaborationMode.PARALLEL,
                    "sequential": CollaborationMode.SEQUENTIAL,
                    "collaborative": CollaborationMode.COLLABORATIVE,
                }
                mode = mode_map.get(mode_str, CollaborationMode.ADAPTIVE)
                
                result = await orchestrator.route_task(
                    task_type=task_type,
                    prompt=data.get("prompt", ""),
                    system_prompt=data.get("system_prompt", ""),
                    context=data.get("context"),
                    mode=mode,
                )
                
                return {
                    "success": True,
                    "result": result,
                    "task_type": task_type.value,
                    "mode": mode.value,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/dual-model/think-together")
        async def dual_model_think_together(request: Request):
            """Have both models think together on a topic."""
            try:
                from core.dual_model import get_dual_model_orchestrator
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["topic", "query", "rounds"],
                    max_string_length=50000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                
                orchestrator = get_dual_model_orchestrator()
                
                result = await orchestrator.think_together(
                    topic=data.get("topic", ""),
                    query=data.get("query", ""),
                    rounds=data.get("rounds", 2),
                )
                
                return {
                    "success": True,
                    "result": result,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/dual-model/context")
        async def get_dual_model_context():
            """Get shared context between models."""
            try:
                from core.dual_model import get_dual_model_orchestrator
                orchestrator = get_dual_model_orchestrator()
                return {
                    "success": True,
                    "context": orchestrator.get_all_shared_context(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/dual-subsystem/stats")
        async def get_dual_subsystem_stats():
            """Get dual-model subsystem integration statistics."""
            try:
                from core.dual_subsystem import get_dual_subsystem_integration
                integration = get_dual_subsystem_integration()
                stats = integration.get_stats()
                
                # Calculate totals
                total_calls = sum(s.get("total_calls", 0) for s in stats.values())
                total_primary = sum(s.get("primary_calls", 0) for s in stats.values())
                total_backend = sum(s.get("backend_calls", 0) for s in stats.values())
                total_parallel = sum(s.get("parallel_calls", 0) for s in stats.values())
                total_sequential = sum(s.get("sequential_calls", 0) for s in stats.values())
                
                return {
                    "success": True,
                    "subsystems": stats,
                    "summary": {
                        "total_calls": total_calls,
                        "primary_calls": total_primary,
                        "backend_calls": total_backend,
                        "parallel_calls": total_parallel,
                        "sequential_calls": total_sequential,
                        "active_subsystems": len([s for s in stats.values() if s.get("total_calls", 0) > 0]),
                    },
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/dual-subsystem/recent")
        async def get_dual_subsystem_recent(subsystem: str = "", limit: int = 20):
            """Get recent dual-model calls from subsystems."""
            try:
                from core.dual_subsystem import get_dual_subsystem_integration
                integration = get_dual_subsystem_integration()
                calls = integration.get_recent_calls(subsystem=subsystem if subsystem else None, limit=limit)
                return {
                    "success": True,
                    "calls": calls,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/models/handoff")
        async def model_handoff(request: Request):
            """Hand off context from current model to another model."""
            try:
                from core.model_coordinator import get_model_coordinator, HandoffReason
                data = await request.json()
                
                coordinator = get_model_coordinator()
                reason = HandoffReason(data.get("reason", "collaboration"))
                
                message = await coordinator.handoff(
                    to_model=data["to_model"],
                    context=data.get("context", {}),
                    query=data.get("query", ""),
                    reason=reason,
                    reasoning_chain_id=data.get("reasoning_chain_id"),
                )
                
                return {
                    "success": True,
                    "message": message.to_dict(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/dual-analyze")
        async def dual_model_analyze(request: Request):
            """Analyze a query using both models in parallel for comprehensive insights."""
            try:
                from core.dual_model import get_dual_model_orchestrator, TaskType, CollaborationMode
                from core.model_coordinator import get_model_coordinator
                data = await request.json()
                
                query = data.get("query", "")
                mode = data.get("mode", "parallel")  # parallel, sequential, collaborative
                
                orchestrator = get_dual_model_orchestrator()
                get_model_coordinator()
                
                # Determine collaboration mode
                collab_mode = {
                    "parallel": CollaborationMode.PARALLEL,
                    "sequential": CollaborationMode.SEQUENTIAL,
                    "collaborative": CollaborationMode.COLLABORATIVE,
                }.get(mode, CollaborationMode.PARALLEL)
                
                # Route through dual model system
                result = await orchestrator.route_task(
                    task_type=TaskType.ANALYSIS,
                    prompt=query,
                    system_prompt="Provide a comprehensive analysis. Be thorough and insightful.",
                    mode=collab_mode,
                )
                
                stats = orchestrator.get_stats()
                
                return {
                    "success": True,
                    "result": result,
                    "mode": mode,
                    "primary_requests": stats.get("primary_requests", 0),
                    "backend_requests": stats.get("backend_requests", 0),
                    "parallel_requests": stats.get("parallel_requests", 0),
                    "collaborative_requests": stats.get("collaborative_requests", 0),
                }
            except Exception as e:
                logger.error(f"Dual analyze error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/think-together")
        async def think_together_endpoint(request: Request):
            """Have both models think together on a topic, building on each other's insights."""
            try:
                from core.model_coordinator import get_model_coordinator
                data = await request.json()
                
                topic = data.get("topic", "general")
                query = data.get("query", "")
                rounds = data.get("rounds", 2)
                
                coordinator = get_model_coordinator()
                
                # Use model coordinator's think_together for cross-model reasoning
                chain = await coordinator.think_together(
                    topic=topic,
                    initial_query=query,
                    models=["kimi-k2.5", "glm-5"],
                    rounds=min(rounds, 3),  # Max 3 rounds
                )
                
                return {
                    "success": True,
                    "synthesis": chain.synthesis if chain else None,
                    "turns": [
                        {
                            "model": turn.model,
                            "thinking": turn.thinking[:200] if turn.thinking else None,
                            "output": turn.output[:500] if turn.output else None,
                        }
                        for turn in (chain.turns if chain else [])
                    ],
                    "total_turns": len(chain.turns) if chain else 0,
                }
            except Exception as e:
                logger.error(f"Think together error: {e}")
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/models/query/{target_model}")
        async def query_other_model(target_model: str, request: Request):
            """Query another model while staying in current model context."""
            try:
                from core.model_coordinator import get_model_coordinator
                data = await request.json()
                
                coordinator = get_model_coordinator()
                response = await coordinator.query_other_model(
                    target_model=target_model,
                    query=data.get("query", ""),
                    context=data.get("context"),
                    include_reasoning=data.get("include_reasoning", False),
                )
                
                return {
                    "success": True,
                    "response": response,
                    "from_model": coordinator.get_active_model(),
                    "to_model": target_model,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/models/think-together")
        async def models_think_together(request: Request):
            """Have multiple models think together on a topic."""
            try:
                from core.model_coordinator import get_model_coordinator
                data = await request.json()
                
                coordinator = get_model_coordinator()
                chain = await coordinator.think_together(
                    topic=data.get("topic", ""),
                    initial_query=data.get("query", ""),
                    models=data.get("models"),
                    rounds=data.get("rounds", 2),
                )
                
                return {
                    "success": True,
                    "chain": chain.to_dict(),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/models/shared-context")
        async def get_shared_context():
            """Get the shared context between models."""
            try:
                from core.model_coordinator import get_model_coordinator
                coordinator = get_model_coordinator()
                return {
                    "success": True,
                    "context": coordinator.shared_context.get_all(),
                    "messages": coordinator.shared_context.get_messages(10),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/models/shared-context")
        async def set_shared_context(request: Request):
            """Set a value in the shared context."""
            try:
                from core.model_coordinator import get_model_coordinator
                data = await request.json()
                
                coordinator = get_model_coordinator()
                coordinator.set_shared_context(
                    key=data["key"],
                    value=data["value"],
                    model_id=data.get("model_id", "user"),
                )
                
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/dual-model/dashboard")
        async def get_dual_model_dashboard():
            """Get comprehensive dual-model dashboard with all metrics and visualization data."""
            try:
                from core.dual_model import get_dual_model_orchestrator, TASK_ROUTING
                from core.model_coordinator import get_model_coordinator
                from core.dual_subsystem import get_dual_subsystem_integration
                from core.embedding_interconnect import get_embedding_interconnect
                
                orchestrator = get_dual_model_orchestrator()
                coordinator = get_model_coordinator()
                subsystem = get_dual_subsystem_integration()
                interconnect = get_embedding_interconnect()
                
                orchestrator_stats = orchestrator.get_stats()
                coordinator_stats = coordinator.get_stats()
                subsystem_stats = subsystem.get_stats()
                
                task_routing = {k.value: v.value for k, v in TASK_ROUTING.items()}
                
                mode_distribution = {
                    "parallel": orchestrator_stats.get("parallel_requests", 0),
                    "sequential": orchestrator_stats.get("sequential_requests", 0),
                    "collaborative": orchestrator_stats.get("collaborative_requests", 0),
                    "single": orchestrator_stats.get("total_requests", 0) - 
                              orchestrator_stats.get("parallel_requests", 0) -
                              orchestrator_stats.get("sequential_requests", 0) -
                              orchestrator_stats.get("collaborative_requests", 0),
                }
                
                total = max(1, orchestrator_stats.get("total_requests", 1))
                model_usage = {
                    "primary": {
                        "count": orchestrator_stats.get("primary_requests", 0),
                        "percentage": round(orchestrator_stats.get("primary_requests", 0) / total * 100, 1),
                        "avg_latency_ms": round(orchestrator_stats.get("avg_primary_latency_ms", 0), 1),
                    },
                    "backend": {
                        "count": orchestrator_stats.get("backend_requests", 0),
                        "percentage": round(orchestrator_stats.get("backend_requests", 0) / total * 100, 1),
                        "avg_latency_ms": round(orchestrator_stats.get("avg_backend_latency_ms", 0), 1),
                    },
                }
                
                recent_calls = subsystem.get_recent_calls(limit=10)
                
                ic_stats = interconnect.get_stats()
                
                embedding_stats = {
                    "stored_embeddings": (
                        ic_stats["model_outputs"]["primary"] +
                        ic_stats["model_outputs"]["backend"] +
                        ic_stats["model_outputs"]["synthesis"]
                    ),
                    "conversation_turns": ic_stats["conversation_turns"],
                    "primary_successes": ic_stats["model_outputs"]["primary"],
                    "backend_successes": ic_stats["model_outputs"]["backend"],
                    "memories": ic_stats["memory_embeddings"],
                    "insights": ic_stats["insight_embeddings"],
                    "knowledge_nodes": ic_stats["knowledge_embeddings"],
                }
                
                return {
                    "success": True,
                    "timestamp": str(datetime.now()),
                    "models": {
                        "primary": {
                            "id": "kimi-k2.5",
                            "role": "chat, creative, vision, abstraction, meta_reasoning, pattern",
                            "connected": orchestrator_stats.get("primary_connected", False),
                            "tasks": ["chat", "creative", "vision", "abstraction", "meta_reasoning", "pattern"],
                        },
                        "backend": {
                            "id": "glm-5",
                            "role": "code, math, reasoning, analysis, refinement, synthesis",
                            "connected": orchestrator_stats.get("backend_connected", False),
                            "tasks": ["code", "math", "analysis", "reasoning", "refinement", "verification", "thinking", "synthesis"],
                        },
                    },
                    "routing": task_routing,
                    "collaboration_modes": mode_distribution,
                    "model_usage": model_usage,
                    "cross_model_calls": orchestrator_stats.get("cross_model_calls", 0),
                    "errors": orchestrator_stats.get("errors", 0),
                    "embedding_interconnect": embedding_stats,
                    "subsystem_activity": {
                        name: {
                            "total": stats.get("total_calls", 0),
                            "primary": stats.get("primary_calls", 0),
                            "backend": stats.get("backend_calls", 0),
                            "parallel": stats.get("parallel_calls", 0),
                            "sequential": stats.get("sequential_calls", 0),
                            "avg_latency_ms": round(stats.get("avg_latency_ms", 0), 1),
                        }
                        for name, stats in subsystem_stats.get("subsystems", {}).items()
                    },
                    "recent_calls": [
                        {
                            "subsystem": call.get("subsystem", ""),
                            "model": call.get("model", ""),
                            "preview": call.get("preview", "")[:100],
                            "latency_ms": call.get("latency", 0),
                            "status": call.get("status", ""),
                            "timestamp": call.get("timestamp", ""),
                        }
                        for call in recent_calls
                    ],
                    "handoffs": {
                        "total": coordinator_stats.get("total_handoffs", 0),
                        "successful": coordinator_stats.get("successful_handoffs", 0),
                    },
                }
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                return {"success": False, "error": safe_error_message(e)}
        # Sibling-route handlers called from other route groups.
        self._route_handlers["get_freetime_status"] = get_freetime_status


    def _setup_n8n_embed_routes(self):
        @self.app.get("/api/dual-model/embeddings")
        async def get_embedding_stats():
            """Get embedding interconnect statistics and recent patterns."""
            try:
                from core.embedding_interconnect import get_embedding_interconnect
                
                interconnect = get_embedding_interconnect()
                
                ic_stats = interconnect.get_stats()
                last_responses = interconnect.get_last_responses()
                primary_outputs = interconnect.get_model_outputs("primary")
                backend_outputs = interconnect.get_model_outputs("backend")
                synthesis_outputs = interconnect.get_model_outputs("synthesis")
                
                embedding_rate = getattr(interconnect, '_embedding_rate_per_min', 0)
                total_embeddings = getattr(interconnect, '_total_embeddings', 0)
                
                recent_primary = len(primary_outputs[-10:])
                recent_backend = len(backend_outputs[-10:])
                
                return {
                    "success": True,
                    "stored_embeddings": (
                        ic_stats["model_outputs"]["primary"] +
                        ic_stats["model_outputs"]["backend"] +
                        ic_stats["model_outputs"]["synthesis"]
                    ),
                    "total_embeddings_lifetime": total_embeddings,
                    "embedding_rate_per_min": embedding_rate,
                    "conversation_turns": ic_stats["conversation_turns"],
                    "successful_patterns": {
                        "primary": [
                            {"prompt_preview": p[:100], "quality": round(q, 2)}
                            for p, q, _ in primary_outputs[-10:]
                        ],
                        "backend": [
                            {"prompt_preview": p[:100], "quality": round(q, 2)}
                            for p, q, _ in backend_outputs[-10:]
                        ],
                        "synthesis": [
                            {"prompt_preview": p[:100], "quality": round(q, 2)}
                            for p, q, _ in synthesis_outputs[-10:]
                        ],
                    },
                    "recent_activity": {
                        "primary_last_min": recent_primary,
                        "backend_last_min": recent_backend,
                        "is_active": recent_primary > 0 or recent_backend > 0,
                    },
                    "shared_context": {
                        "primary_last": last_responses.get("primary", "")[:200],
                        "backend_last": last_responses.get("backend", "")[:200],
                        "memories": ic_stats["memory_embeddings"],
                        "insights": ic_stats["insight_embeddings"],
                        "knowledge_nodes": ic_stats["knowledge_embeddings"],
                        "total_keys": ic_stats["memory_embeddings"] + ic_stats["insight_embeddings"] + ic_stats["knowledge_embeddings"],
                    },
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/n8n/status")
        async def n8n_status():
            """Get n8n integration status."""
            try:
                from core.daydream_n8n import get_daydream_n8n
                
                integration = get_daydream_n8n()
                report = await integration.get_daydream_report()
                
                return {
                    "success": True,
                    "available": integration.n8n is not None,
                    "workflows": len(report.get("workflow_ids", {})),
                    "last_activity": report.get("last_activity"),
                    "activities_completed": report.get("activities_completed", 0),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e), "available": False}

        @self.app.post("/api/n8n/workflows/create")
        async def create_n8n_workflows():
            """Create all daydream workflows in n8n."""
            try:
                from core.daydream_n8n import get_daydream_n8n
                
                integration = get_daydream_n8n()
                workflow_ids = await integration.ensure_workflows()
                
                return {
                    "success": True,
                    "workflows_created": len(workflow_ids),
                    "workflow_ids": workflow_ids,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/n8n/workflows/activate")
        async def activate_n8n_workflows():
            """Activate all daydream workflows."""
            try:
                from core.daydream_n8n import get_daydream_n8n
                
                integration = get_daydream_n8n()
                results = await integration.activate_daydream_workflows()
                
                return {
                    "success": True,
                    "results": results,
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.post("/api/n8n/trigger/{activity}")
        async def trigger_n8n_activity(activity: str):
            """Trigger a specific daydream activity via n8n."""
            if not activity or len(activity) > 64 or '..' in activity or '/' in activity:
                return {"success": False, "error": "Invalid activity name"}
            try:
                from core.daydream_n8n import get_daydream_n8n
                
                integration = get_daydream_n8n()
                result = await integration.trigger_daydream_activity(activity)
                
                return result
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}

        @self.app.get("/api/daydream/workflows")
        async def get_daydream_workflows():
            """Get available daydream workflow templates."""
            try:
                from core.daydream_n8n import DAYDREAM_WORKFLOW_TEMPLATES
                
                return {
                    "success": True,
                    "workflows": [
                        {
                            "name": name,
                            "description": template.get("description", ""),
                            "trigger": template.get("trigger", "manual"),
                        }
                        for name, template in DAYDREAM_WORKFLOW_TEMPLATES.items()
                    ],
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}
        # Sibling-route handlers called from other route groups.
        self._route_handlers["get_embedding_stats"] = get_embedding_stats


    def _setup_websocket_routes(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            t0 = time.time()
            logger.debug("[WS] Connection attempt started")

            # Cross-site WebSocket hijacking guard (browser Origin must
            # match Host or the CORS allow-list; non-browser clients pass).
            if not self._ws_origin_allowed(websocket):
                await websocket.accept()
                await websocket.send_json({
                    "type": "error",
                    "content": "Origin not allowed",
                    "code": "origin_forbidden",
                })
                await websocket.close(code=4003, reason="Origin not allowed")
                return

            ws_id = str(id(websocket))
            session_id = None
            
            client_host = websocket.client.host if websocket.client else "unknown"
            client_id = f"ws:{client_host}"
            
            from core.rate_limiter import get_rate_limiter, RateLimit
            limiter = get_rate_limiter()
            ws_limit = RateLimit(requests_per_second=50.0, requests_per_minute=3000, burst_size=200)
            limiter.set_endpoint_limit("websocket", ws_limit)
            allowed, retry_after, info = limiter.check_rate_limit(client_id, "websocket")
            # check_rate_limit() increments a concurrency counter that only
            # release() decrements. The slot was never released here, so
            # `state.concurrent` grew monotonically and a client was locked out
            # for good after enough connections. client_id is rebound below once
            # the session is known, so remember the id the slot was taken under.
            rate_limit_client_id = client_id
            if not allowed:
                await websocket.accept()
                await websocket.send_json({
                    "type": "error",
                    "content": f"Rate limit exceeded. Retry after {retry_after:.0f}s",
                    "code": "rate_limit",
                })
                await websocket.close(code=4008, reason="Rate limit exceeded")
                limiter.release(rate_limit_client_id)
                return
            
            if self._require_auth:
                # Check for auth in query params or headers
                query_params = dict(websocket.query_params)
                headers = dict(websocket.headers)
                
                is_valid, result, error = self.security.validate_websocket_connection(
                    headers=headers,
                    query_params=query_params,
                    client_ip=websocket.client.host if websocket.client else None,
                )
                
                if not is_valid:
                    await websocket.accept()
                    await websocket.send_json({
                        "type": "error",
                        "content": error or "Authentication required",
                        "code": "auth_required",
                    })
                    await websocket.close(code=4001, reason="Unauthorized")
                    return
                
                session_id = result
                self._ws_sessions[ws_id] = session_id
            
            await websocket.accept()
            t1 = time.time()
            logger.debug(f"[WS] Accepted in {t1-t0:.3f}s")

            MAX_WS_CONNECTIONS = 50
            if len(self.connections) >= MAX_WS_CONNECTIONS:
                await websocket.close(code=1013, reason="Too many connections")
                return

            self.connections.add(websocket)
            
            # Get client identifier for rate limiting
            client_host = websocket.client.host if websocket.client else "unknown"
            client_id = f"ws:{client_host}:{session_id[:8] if session_id else 'anon'}"
            
            # Clean up old rate limits periodically
            if len(self._ws_rate_limits) > 100:
                self._cleanup_rate_limits()
            
            conv_id = self.conversation_manager.current_conversation_id
            if not conv_id:
                try:
                    convs = await self.conversation_manager.list_conversations()
                    if convs:
                        latest = convs[0]
                        conv_id = latest.get("id")
                        if conv_id:
                            self.conversation_manager.current_conversation_id = conv_id
                            conv_data = await self.conversation_manager.load_conversation(conv_id)
                            if conv_data:
                                self._current_messages = conv_data.get("messages", [])
                except Exception as e:
                    logger.debug(f"Auto-load last conversation failed: {e}")
            await websocket.send_json({
                "type": "connected",
                "status": "ready",
                "conversation_id": conv_id,
                "session_id": session_id,
                "consecutive_errors": self._consecutive_agent_errors,
            })

            active_tools = {}
            current_response = ""

            from core.ws_manager import get_ws_connection_manager
            ws_mgr = get_ws_connection_manager()
            ws_conn_info = ws_mgr.register(websocket, client_host=client_host, session_id=session_id)
            ws_keepalive_task = asyncio.create_task(self._ws_keepalive_loop(websocket, ws_mgr))

            try:
                while True:
                    try:
                        data = await asyncio.wait_for(websocket.receive_text(), timeout=120)
                        
                        # Size limit check
                        MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB max
                        if len(data) > MAX_MESSAGE_SIZE:
                            await websocket.send_json({
                                "type": "error",
                                "content": f"Message too large ({len(data)} bytes, max {MAX_MESSAGE_SIZE})",
                                "code": "message_too_large",
                            })
                            continue
                        
                        # Sanitize incoming data
                        try:
                            raw_message = json.loads(data)
                            message = self.security.sanitize_input(raw_message)
                        except json.JSONDecodeError:
                            await websocket.send_json({
                                "type": "error",
                                "content": "Invalid JSON",
                                "code": "invalid_json",
                            })
                            continue
                        
                        message_type = message.get("type", "")
                        
                        # Validate message type
                        valid_types = {
                            "ping", "pong", "clear", "new_conversation", 
                            "load_conversation", "list_conversations", 
                            "delete_conversation", "chat", "agent_loop"
                        }
                        if message_type not in valid_types:
                            await websocket.send_json({
                                "type": "error",
                                "content": f"Unknown message type: {message_type}",
                                "code": "invalid_type",
                            })
                            continue
                        
                        # Rate limit check (except for ping)
                        if message_type not in ("ping", "pong"):
                            if not self._check_ws_rate_limit(client_id, max_per_minute=3000):
                                await websocket.send_json({
                                    "type": "error",
                                    "content": "Rate limit exceeded. Please slow down.",
                                    "code": "rate_limit",
                                })
                                continue
                            
                            # Mark user activity to reset idle timer for daydream
                            if hasattr(self.agent, 'mark_user_activity'):
                                self.agent.mark_user_activity()

                        if message_type == "ping":
                            ws_mgr.record_pong(websocket)
                            await websocket.send_json({"type": "pong"})
                            continue

                        if message_type == "pong":
                            ws_mgr.record_pong(websocket)
                            continue

                        ws_mgr.record_activity(websocket)

                        if message_type == "clear":
                            if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear'):
                                self.agent.memory.clear()
                            active_tools.clear()
                            self._current_messages = []
                            await websocket.send_json({"type": "cleared"})
                            continue

                        if message.get("type") == "new_conversation":
                            title = message.get("title", "New Conversation")
                            conversation = await self.conversation_manager.create_conversation(title)
                            self._current_messages = []
                            if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear'):
                                self.agent.memory.clear()
                            await websocket.send_json({
                                "type": "conversation_created",
                                "conversation": conversation,
                            })
                            continue

                        if message.get("type") == "load_conversation":
                            conv_id = message.get("conversation_id")
                            if conv_id:
                                conversation = await self.conversation_manager.load_conversation(conv_id)
                                if conversation:
                                    self.conversation_manager.current_conversation_id = conv_id
                                    self._current_messages = conversation.get("messages", [])
                                    
                                    if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear'):
                                        self.agent.memory.clear()
                                        for msg in self._current_messages:
                                            role = msg.get("role", "user")
                                            content = msg.get("content", "")
                                            if role == "user":
                                                self.agent.memory.add_user_message(content)
                                            elif role == "assistant":
                                                self.agent.memory.add_assistant_message(content)
                                    
                                    await websocket.send_json({
                                        "type": "conversation_loaded",
                                        "conversation": conversation,
                                    })
                                else:
                                    await websocket.send_json({
                                        "type": "error",
                                        "content": "Conversation not found",
                                    })
                            continue

                        if message.get("type") == "list_conversations":
                            conversations = await self.conversation_manager.list_conversations()
                            await websocket.send_json({
                                "type": "conversations_list",
                                "conversations": conversations,
                            })
                            continue

                        if message.get("type") == "delete_conversation":
                            conv_id = message.get("conversation_id")
                            if conv_id:
                                success = await self.conversation_manager.delete_conversation(conv_id)
                                await websocket.send_json({
                                    "type": "conversation_deleted",
                                    "conversation_id": conv_id,
                                    "success": success,
                                })
                            continue

                        if message.get("type") == "chat":
                            user_input = message.get("content", "").strip()
                            force_mode = message.get("mode", "auto")
                            
                            if not user_input:
                                continue
                            
                            # Input length limit to prevent resource exhaustion
                            if len(user_input) > 10000:
                                user_input = user_input[:10000]
                                logger.warning("Truncated long user input to 10000 chars")

                            if not self.conversation_manager.current_conversation_id:
                                await self.conversation_manager.create_conversation()
                            
                            self._current_messages.append({
                                "role": "user",
                                "content": user_input,
                                "timestamp": datetime.now().isoformat(),
                            })
                            
                            mode_decision = analyze_message(user_input)
                            logger.info(f"Mode decision for '{user_input[:50]}...': {mode_decision.mode.value} (confidence: {mode_decision.confidence:.2f}, steps: {mode_decision.estimated_steps})")
                            
                            use_agent_loop = False
                            if force_mode == "agentic":
                                use_agent_loop = True
                            elif force_mode == "chat":
                                use_agent_loop = False
                            elif mode_decision.mode == ExecutionMode.AGENTIC and mode_decision.confidence >= 0.7:
                                use_agent_loop = True
                            # ASK_PERMISSION mode removed - defaults to chat to avoid stuck states
                            
                            active_tools.clear()
                            current_response = ""
                            current_thinking = ""
                            await websocket.send_json({"type": "status", "status": "thinking"})
                            
                            if mode_decision.tools_likely_needed:
                                await websocket.send_json({
                                    "type": "mode_info",
                                    "mode": "agentic" if use_agent_loop else "chat",
                                    "estimated_steps": mode_decision.estimated_steps,
                                    "tools_detected": mode_decision.tools_likely_needed,
                                })

                            try:
                                async for token in self.agent.run_step_stream(user_input):
                                    tool_pattern = r'(?:🔧|⚡|▶)\s*\*{0,2}([^*]+)\*{0,2}'
                                    success_pattern = r'✅\s*\*{0,2}([^*]+)\*{0,2}.*(?:completed|success)'
                                    error_pattern = r'❌\s*\*{0,2}([^*]+)\*{0,2}.*(?:failed|error)'

                                    tool_match = re.search(tool_pattern, token)
                                    if tool_match:
                                        tool_name = tool_match.group(1).strip()
                                        if tool_name and tool_name not in active_tools:
                                            tool_id = f"tool-{len(active_tools)}-{time.time()}"
                                            active_tools[tool_name] = tool_id
                                            await websocket.send_json({"type": "tool_start", "id": tool_id, "name": tool_name})

                                    success_match = re.search(success_pattern, token, re.IGNORECASE)
                                    if success_match:
                                        tool_name = success_match.group(1).strip()
                                        if tool_name in active_tools:
                                            await websocket.send_json({
                                                "type": "tool_result",
                                                "id": active_tools[tool_name],
                                                "name": tool_name,
                                                "success": True,
                                                "result": "completed"
                                            })
                                            del active_tools[tool_name]

                                    error_match = re.search(error_pattern, token, re.IGNORECASE)
                                    if error_match:
                                        tool_name = error_match.group(1).strip()
                                        if tool_name in active_tools:
                                            await websocket.send_json({
                                                "type": "tool_result",
                                                "id": active_tools[tool_name],
                                                "name": tool_name,
                                                "success": False,
                                                "result": "failed"
                                            })
                                            del active_tools[tool_name]

                                    current_response += token
                                    await websocket.send_json({"type": "token", "content": token})
                                    
                                    if '<think>' in token:
                                        current_thinking += token
                                    elif current_thinking and '</think>' not in current_response[-100:]:
                                        current_thinking += token

                                thinking_match = None
                                if current_thinking and len(current_thinking) > 20:
                                    thinking_match = current_thinking.replace('<think>', '').replace('</think>', '').strip()
                                elif '<think>' in current_response and '</think>' in current_response:
                                    start = current_response.find('<think>') + len('<think>')
                                    end = current_response.find('</think>')
                                    if start >= 0 and end > start:
                                        thinking_match = current_response[start:end].strip()

                                msg_data = {
                                    "role": "assistant",
                                    "content": current_response,
                                    "timestamp": datetime.now().isoformat(),
                                }
                                if thinking_match:
                                    msg_data["thinking"] = thinking_match[:500]
                                self._current_messages.append(msg_data)
                                
                                self._consecutive_agent_errors = 0

                                if self.conversation_manager.current_conversation_id:
                                    await self.conversation_manager.save_conversation(
                                        self.conversation_manager.current_conversation_id,
                                        self._current_messages,
                                    )

                            except Exception as e:
                                import traceback
                                logger.error(f"Agent error: {e.__class__.__name__}: {e}")
                                logger.error(f"Traceback: {traceback.format_exc()}")

                                recovery_attempted = False
                                consecutive_agent_errors = getattr(self, '_consecutive_agent_errors', 0) + 1
                                self._consecutive_agent_errors = consecutive_agent_errors

                                if consecutive_agent_errors <= 2:
                                    try:
                                        from core.adaptive_agent_loop import get_adaptive_agent_loop
                                        adaptive = get_adaptive_agent_loop(agent=self.agent, llm_client=self.agent.llm if hasattr(self.agent, 'llm') else None)
                                        recovery = adaptive.recover_mid_stream(
                                            safe_error_message(e),
                                            partial_output=current_response,
                                            tool_calls_made=[{"name": n, "id": tid} for n, tid in active_tools.items()],
                                        )
                                        if recovery.get("recovered"):
                                            recovery_attempted = True
                                            strategy = recovery.get('strategies', ['retry'])[0]
                                            preserved = recovery.get('partial_output_len', 0)
                                            await websocket.send_json({
                                                "type": "token",
                                                "content": f"\n🔄 **Recovering...** ({strategy}" + (f", {preserved} chars preserved" if preserved else "") + ")\n",
                                            })
                                            if strategy in ("retry_shorter", "fallback_tool", "context_reduction", "simpler_query"):
                                                try:
                                                    if len(self._current_messages) > 10:
                                                        self._current_messages = self._current_messages[-6:]
                                                except Exception:
                                                    pass
                                            if strategy == "skip_step":
                                                await websocket.send_json({
                                                    "type": "token",
                                                    "content": "\n⏩ Skipping failed step and continuing...\n",
                                                })
                                            else:
                                                async for retry_token in self.agent.run_step_stream(user_input):
                                                    current_response += retry_token
                                                    await websocket.send_json({"type": "token", "content": retry_token})
                                    except Exception as retry_err:
                                        logger.warning(f"Recovery attempt failed: {retry_err}")
                                else:
                                    logger.warning(f"Skipping recovery: {consecutive_agent_errors} consecutive agent errors")

                                if not recovery_attempted:
                                    if current_response:
                                        await websocket.send_json({
                                            "type": "partial_output",
                                            "content": current_response,
                                            "error": f"{e.__class__.__name__}: {safe_error_message(e) or 'Unknown error'}",
                                        })
                                    await websocket.send_json({"type": "error", "content": f"{e.__class__.__name__}: {safe_error_message(e) or 'Unknown error'}", "code": "agent_error", "recoverable": consecutive_agent_errors <= 2, "partial_output_len": len(current_response)})

                            for tool_name, tool_id in list(active_tools.items()):
                                await websocket.send_json({
                                    "type": "tool_result",
                                    "id": tool_id,
                                    "name": tool_name,
                                    "success": True,
                                    "result": "completed"
                                })
                            active_tools.clear()

                            await websocket.send_json({
                                "type": "done",
                                "conversation_id": self.conversation_manager.current_conversation_id,
                            })

                        # Multi-step agent loop execution
                        if message.get("type") == "agent_loop":
                            user_input = message.get("content", "").strip()
                            if len(user_input) > 10000:
                                user_input = user_input[:10000]
                                logger.warning("Truncated long agent_loop input to 10000 chars")
                            use_backend = message.get("use_backend", False)
                            use_adaptive = message.get("use_adaptive", True)
                            max_steps = min(int(message.get("max_steps", 5)), 50)
                            
                            if not user_input:
                                continue

                            if not self.conversation_manager.current_conversation_id:
                                await self.conversation_manager.create_conversation()
                            
                            self._current_messages.append({
                                "role": "user",
                                "content": user_input,
                                "timestamp": datetime.now().isoformat(),
                            })
                            
                            await websocket.send_json({"type": "status", "status": "thinking"})
                            
                            try:
                                from core.agent_loop import create_agent_loop
                                from core.backend_llm import get_backend_llm, ModelTier, check_tool_permission
                                
                                # Use backend LLM for analysis tasks
                                if use_backend:
                                    llm = get_backend_llm()
                                    tier = ModelTier.BACKEND
                                else:
                                    llm = self.agent.llm
                                    tier = ModelTier.PRIMARY
                                
                                # Build tool registry with permission filtering
                                tool_registry = {}
                                for name, tool in self.agent.tools.items():
                                    category = getattr(tool, 'category', 'system')
                                    if check_tool_permission(name, category, tier):
                                        tool_registry[name] = tool

                                if use_adaptive:
                                    try:
                                        from core.adaptive_agent_loop import get_adaptive_agent_loop
                                        adaptive = get_adaptive_agent_loop(agent=self.agent, llm_client=llm)
                                        system_prompt = ""
                                        if hasattr(self.agent, 'get_system_prompt'):
                                            system_prompt = self.agent.get_system_prompt()

                                        messages = []
                                        for msg in self._current_messages[:-1]:
                                            messages.append({
                                                "role": msg.get("role", "user"),
                                                "content": msg.get("content", ""),
                                            })
                                        messages.append({"role": "user", "content": user_input})

                                        accumulated_response = ""
                                        tool_calls_made = 0

                                        async for event in adaptive.run_persistent_stream(
                                            user_input=user_input,
                                            system_prompt=system_prompt,
                                            messages=messages,
                                            tools=tool_registry,
                                            max_iterations=max_steps,
                                        ):
                                            etype = event.get("type", "")

                                            if etype == "thinking":
                                                await websocket.send_json({
                                                    "type": "thinking",
                                                    "content": event.get("text", ""),
                                                })

                                            elif etype == "text":
                                                text = event.get("text", "")
                                                accumulated_response += text
                                                await websocket.send_json({
                                                    "type": "token",
                                                    "content": text,
                                                })

                                            elif etype == "tool_call":
                                                tool_name = event.get("tool", "unknown")
                                                tool_args = event.get("arguments", {})
                                                tool_calls_made += 1
                                                await websocket.send_json({
                                                    "type": "tool_start",
                                                    "id": f"tool-adaptive-{tool_calls_made}",
                                                    "name": tool_name,
                                                    "input": tool_args,
                                                })

                                            elif etype == "tool_result":
                                                tool_name = event.get("tool", "unknown")
                                                success = event.get("success", False)
                                                output = event.get("output", "")
                                                await websocket.send_json({
                                                    "type": "tool_result",
                                                    "id": f"tool-adaptive-{tool_calls_made}",
                                                    "name": tool_name,
                                                    "success": success,
                                                    "result": output[:500] if len(output) > 500 else output,
                                                })

                                            elif etype == "recovery":
                                                await websocket.send_json({
                                                    "type": "token",
                                                    "content": f"\n🔄 Recovering ({event.get('strategy', 'retry')}, attempt {event.get('attempt', '?')})...\n",
                                                })

                                            elif etype == "resume":
                                                await websocket.send_json({
                                                    "type": "token",
                                                    "content": "\n♻️ Resuming an interrupted task from the previous session...\n",
                                                })

                                            elif etype == "error":
                                                await websocket.send_json({
                                                    "type": "error",
                                                    "content": event.get("error", "Unknown error"),
                                                })

                                            elif etype == "complete":
                                                await websocket.send_json({
                                                    "type": "loop_complete",
                                                    "finish_reason": "stop" if event.get("success") else "error",
                                                    "steps": event.get("iterations", 0),
                                                    "tool_calls": event.get("tool_calls", 0),
                                                })

                                    except Exception as adaptive_err:
                                        logger.warning(f"Adaptive loop failed, falling back to standard: {adaptive_err}")
                                        use_adaptive = False

                                if not use_adaptive:
                                    # Create agent loop
                                    loop = create_agent_loop(
                                        llm_client=llm,
                                        tools=tool_registry,
                                        permissions=self.agent.permissions if hasattr(self.agent, 'permissions') else None,
                                        max_steps=max_steps,
                                        show_thinking=True,
                                    )
                                    
                                    # Build messages from conversation history
                                    messages = []
                                    for msg in self._current_messages[:-1]:  # Exclude current input
                                        messages.append({
                                            "role": msg.get("role", "user"),
                                            "content": msg.get("content", ""),
                                        })
                                    messages.append({"role": "user", "content": user_input})
                                    
                                    # Get system prompt
                                    system_prompt = ""
                                    if hasattr(self.agent, 'get_system_prompt'):
                                        system_prompt = self.agent.get_system_prompt()
                                    
                                    accumulated_response = ""
                                    thinking_content = ""
                                    tool_calls_made = 0
                                    
                                    async for event in loop.execute_stream(
                                        messages=messages,
                                        system_prompt=system_prompt,
                                    ):
                                        event_type = event.get("type")
                                        
                                        if event_type == "thinking":
                                            text = event.get("text", "")
                                            thinking_content += text
                                            await websocket.send_json({
                                                "type": "thinking",
                                                "content": text,
                                            })
                                        
                                        elif event_type == "text":
                                            text = event.get("text", "")
                                            accumulated_response += text
                                            await websocket.send_json({
                                                "type": "token",
                                                "content": text,
                                            })
                                        
                                        elif event_type == "tool_call":
                                            tool_name = event.get("tool", event.get("name", "unknown"))
                                            tool_input = event.get("input", event.get("arguments", {}))
                                            tool_calls_made += 1
                                            await websocket.send_json({
                                                "type": "tool_start",
                                                "id": f"tool-{tool_calls_made}",
                                                "name": tool_name,
                                                "input": tool_input,
                                            })
                                        
                                        elif event_type == "tool_result":
                                            tool_name = event.get("tool", "unknown")
                                            output = event.get("output", "")
                                            success = event.get("success", True)
                                            await websocket.send_json({
                                                "type": "tool_result",
                                                "id": f"tool-{tool_calls_made}",
                                                "name": tool_name,
                                                "success": success,
                                                "result": output[:500] if len(output) > 500 else output,
                                            })
                                        
                                        elif event_type == "step_complete":
                                            step = event.get("step", 0)
                                            await websocket.send_json({
                                                "type": "step_complete",
                                                "step": step,
                                            })
                                        
                                        elif event_type == "complete":
                                            finish_reason = event.get("finish_reason", "stop")
                                            steps = event.get("steps", 0)
                                            await websocket.send_json({
                                                "type": "loop_complete",
                                                "finish_reason": finish_reason,
                                                "steps": steps,
                                                "tool_calls": tool_calls_made,
                                            })
                                    
                                    # Prepend thinking if present
                                    if thinking_content:
                                        accumulated_response = f"💭 *Thinking...*\n```\n{thinking_content[:1000]}\n```\n\n{accumulated_response}"
                                
                                self._current_messages.append({
                                    "role": "assistant",
                                    "content": accumulated_response,
                                    "timestamp": datetime.now().isoformat(),
                                })
                                
                                if self.conversation_manager.current_conversation_id:
                                    await self.conversation_manager.save_conversation(
                                        self.conversation_manager.current_conversation_id,
                                        self._current_messages,
                                    )

                            except Exception as e:
                                logger.error(f"Agent loop error: {e}", exc_info=True)
                                await websocket.send_json({"type": "error", "content": safe_error_message(e)})

                            for tool_name, tool_id in list(active_tools.items()):
                                await websocket.send_json({
                                    "type": "tool_result",
                                    "id": tool_id,
                                    "name": tool_name,
                                    "success": True,
                                    "result": "completed"
                                })
                            active_tools.clear()

                            await websocket.send_json({
                                "type": "done",
                                "conversation_id": self.conversation_manager.current_conversation_id,
                            })

                    except asyncio.TimeoutError:
                        from core.ws_manager import get_ws_connection_manager
                        ws_mgr_timeout = get_ws_connection_manager()
                        ws_mgr_timeout.record_ping_sent(websocket)
                        missed = ws_mgr_timeout.increment_missed_pongs()
                        await websocket.send_json({"type": "ping"})
                        continue

            except Exception as e:
                logger.debug(f"WebSocket disconnected: {e}")
            finally:
                ws_keepalive_task.cancel()
                try:
                    await ws_keepalive_task
                except (asyncio.CancelledError, Exception):
                    pass
                from core.ws_manager import get_ws_connection_manager
                get_ws_connection_manager().unregister(websocket, reason="disconnect")
                self.connections.discard(websocket)
                if ws_id in self._ws_sessions:
                    del self._ws_sessions[ws_id]
                # Hand back the concurrency slot taken by check_rate_limit().
                try:
                    limiter.release(rate_limit_client_id)
                except Exception as e:
                    logger.debug(f"Rate limiter release failed for {rate_limit_client_id}: {e}")

        @self.app.websocket("/ws/api-activity")
        async def api_activity_websocket(websocket: WebSocket):
            if not self._ws_origin_allowed(websocket):
                await websocket.accept()
                await websocket.send_json({"type": "error", "content": "Origin not allowed", "code": "origin_forbidden"})
                await websocket.close(code=4003, reason="Origin not allowed")
                return

            client_host = websocket.client.host if websocket.client else "unknown"
            client_id = f"api-activity:{client_host}"
            
            from core.rate_limiter import get_rate_limiter, RateLimit
            limiter = get_rate_limiter()
            limiter.set_endpoint_limit("api_activity", RateLimit(requests_per_second=10.0, requests_per_minute=300, burst_size=30))
            allowed, retry_after, _ = limiter.check_rate_limit(client_id, "api_activity")
            if not allowed:
                await websocket.accept()
                await websocket.send_json({"type": "error", "content": "Rate limit exceeded"})
                await websocket.close(code=4008, reason="Rate limit exceeded")
                limiter.release(client_id)
                return

            await websocket.accept()

            MAX_WS_CONNECTIONS = 50
            if len(self._api_activity_connections) >= MAX_WS_CONNECTIONS:
                await websocket.close(code=1013, reason="Too many connections")
                limiter.release(client_id)
                return

            self._api_activity_connections.add(websocket)
            try:
                if hasattr(self.agent, 'llm') and hasattr(self.agent.llm, 'get_call_stats'):
                    stats = self.agent.llm.get_call_stats()
                    await websocket.send_json({
                        "type": "init",
                        "calls_per_minute": stats.get("calls_per_minute", 0),
                        "recent_calls": stats.get("recent_calls", [])[:20],
                    })
                while True:
                    try:
                        await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    except asyncio.TimeoutError:
                        if hasattr(self.agent, 'llm') and hasattr(self.agent.llm, 'get_call_stats'):
                            stats = self.agent.llm.get_call_stats()
                            await websocket.send_json({
                                "type": "ping",
                                "calls_per_minute": stats.get("calls_per_minute", 0),
                            })
            except Exception as e:
                logger.debug(f"API activity WebSocket error: {e}")
            finally:
                self._api_activity_connections.discard(websocket)
                # Hand back the concurrency slot taken by check_rate_limit().
                try:
                    limiter.release(client_id)
                except Exception as e:
                    logger.debug(f"Rate limiter release failed for {client_id}: {e}")



    def _setup_oracle_routes(self):
        """Setup Oracle/Cognitive/Alignment/Neuroscience/Harness/Router/page/mobile routes.
        Thin registrar: each domain group lives in its own method, called in the
        original registration order (decorator order is preserved verbatim inside
        each group).
        """
        self._setup_oracle_domain_routes()
        self._setup_cognitive_routes()
        self._setup_alignment_routes()
        self._setup_neuroscience_routes()
        self._setup_harness_routes()
        self._setup_llm_router_routes()
        self._setup_page_routes()
        self._setup_mobile_api_routes()
        self._setup_mobile_ws_routes()

    def _setup_oracle_domain_routes(self):
        """Oracle prediction engine endpoints (predict, diagnose, timeslider, status, portfolio, options, technical-analysis, market-data/stream, scenario, backtest, esg, yield-curve, risk-decomposition, valuation, predict-to-kg, diagnose-to-kg) with the shared per-IP oracle rate cap."""
        from starlette.responses import JSONResponse as StarletteJSONResponse

        _oracle_rate_limiter = deque(maxlen=100)
        _ORACLE_MAX_PER_MINUTE = 30

        def _check_oracle_rate(request: Request) -> StarletteJSONResponse | None:
            now = time.time()
            client_ip = request.client.host if request.client else "unknown"
            _oracle_rate_limiter.append((client_ip, now))
            recent = sum(1 for ip, t in _oracle_rate_limiter if ip == client_ip and now - t < 60)
            if recent > _ORACLE_MAX_PER_MINUTE:
                return StarletteJSONResponse(
                    {"error": "Oracle rate limit exceeded", "retry_after": 60},
                    status_code=429,
                )
            return None

        @self.app.post("/api/oracle/predict")
        async def oracle_predict(request: Request):
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["domain", "target_date", "confidence", "context"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_predict
                result = await __import__("asyncio").to_thread(
                    oracle_predict,
                    domain=data.get("domain", "all"),
                    target_date=data.get("target_date"),
                    confidence=data.get("confidence", "high"),
                    context=data.get("context", {}),
                )
                return result
            except ImportError:
                return StarletteJSONResponse(
                    {"error": "Oracle engine not available"},
                    status_code=503,
                )
            except Exception as e:
                logger.error(f"Oracle predict error: {e}")
                return {"error": "Prediction failed"}

        @self.app.post("/api/oracle/diagnose")
        async def oracle_diagnose(request: Request):
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["domain", "context"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_diagnose
                result = await __import__("asyncio").to_thread(
                    oracle_diagnose,
                    domain=data.get("domain", "all"),
                    context=data.get("context", {}),
                )
                return result
            except ImportError:
                return StarletteJSONResponse(
                    {"error": "Oracle engine not available"},
                    status_code=503,
                )
            except Exception as e:
                logger.error(f"Oracle diagnose error: {e}")
                return {"error": "Diagnosis failed"}

        @self.app.post("/api/oracle/timeslider")
        async def oracle_timeslider(request: Request):
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["target_date", "scenario"],
                    max_string_length=1000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_timeslider
                result = await __import__("asyncio").to_thread(
                    oracle_timeslider,
                    target_date=data.get("target_date"),
                    scenario=data.get("scenario", "baseline"),
                )
                return result
            except ImportError:
                return StarletteJSONResponse(
                    {"error": "Oracle engine not available"},
                    status_code=503,
                )
            except Exception as e:
                logger.error(f"Oracle timeslider error: {e}")
                return {"error": "Time slider failed"}

        @self.app.get("/api/oracle/status")
        async def oracle_status():
            try:
                from core.oracle_integration import get_oracle_engine
                engine = get_oracle_engine()
                if engine is None:
                    return {"available": False, "error": "Oracle engine not initialized"}
                return {"available": True, "version": "0.3.2"}
            except Exception as e:
                return {"available": False, "error": safe_error_message(e)[:100]}

        @self.app.post("/api/oracle/portfolio")
        async def oracle_portfolio(request: Request):
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["action", "assets", "method", "target_return", "risk_free_rate"],
                    max_string_length=50000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_portfolio
                result = await asyncio.to_thread(
                    oracle_portfolio,
                    action=data.get("action", "optimize"),
                    assets=data.get("assets"),
                    method=data.get("method", "mean_variance"),
                    target_return=data.get("target_return"),
                    risk_free_rate=data.get("risk_free_rate", 0.04),
                )
                return result
            except ImportError:
                return StarletteJSONResponse({"error": "Oracle engine not available"}, status_code=503)
            except Exception as e:
                logger.error(f"Oracle portfolio error: {e}")
                return {"error": "Portfolio optimization failed"}

        @self.app.post("/api/oracle/options")
        async def oracle_options(request: Request):
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["spot", "strike", "time_to_expiry", "risk_free_rate", "volatility", "dividend_yield", "action", "option_type"],
                    max_string_length=5000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_options
                try:
                    spot = float(data.get("spot", 100.0))
                    strike = float(data.get("strike", 100.0))
                    time_to_expiry = float(data.get("time_to_expiry", 1.0))
                    risk_free_rate = float(data.get("risk_free_rate", 0.04))
                    volatility = float(data.get("volatility", 0.25))
                    dividend_yield = float(data.get("dividend_yield", 0.0))
                except (ValueError, TypeError):
                    return {"error": "All numeric parameters must be valid numbers"}
                result = oracle_options(
                    action=data.get("action", "price"),
                    spot=spot, strike=strike,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                    option_type=data.get("option_type", "call"),
                    dividend_yield=dividend_yield,
                )
                return result
            except ImportError:
                return StarletteJSONResponse({"error": "Oracle engine not available"}, status_code=503)
            except Exception as e:
                logger.error(f"Oracle options error: {e}")
                return {"error": "Options calculation failed"}

        @self.app.post("/api/oracle/technical-analysis")
        async def oracle_technical_analysis(request: Request):
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["symbol", "data", "indicators"],
                    max_string_length=50000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_technical_analysis
                result = oracle_technical_analysis(
                    symbol=data.get("symbol"),
                    data=data.get("data"),
                    indicators=data.get("indicators"),
                )
                return result
            except ImportError:
                return StarletteJSONResponse({"error": "Oracle engine not available"}, status_code=503)
            except Exception as e:
                logger.error(f"Oracle TA error: {e}")
                return {"error": "Technical analysis failed"}

        @self.app.get("/api/oracle/market-data")
        async def oracle_market_data(symbols: str = "", period: str = "1mo"):
            try:
                from core.oracle_integration import oracle_market_data
                symbol_list = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None
                result = oracle_market_data(symbols=symbol_list, period=period)
                return result
            except ImportError:
                return StarletteJSONResponse({"error": "Oracle engine not available"}, status_code=503)
            except Exception as e:
                logger.error(f"Oracle market data error: {e}")
                return {"error": "Market data fetch failed"}

        @self.app.get("/api/oracle/market-stream")
        async def oracle_market_stream(request: Request):
            """Server-Sent Events stream for real-time market data updates."""
            from starlette.responses import StreamingResponse
            import random

            async def event_generator():
                symbols_param = request.query_params.get("symbols", "")
                interval = max(5, min(60, int(request.query_params.get("interval", "15"))))
                symbol_list = [s.strip().upper() for s in symbols_param.split(",") if s.strip()] if symbols_param else ["SPY", "QQQ", "DIA", "VIX"]
                yield f"data: {json.dumps({'type': 'connected', 'symbols': symbol_list, 'interval': interval})}\n\n"

                last_data = {}
                max_updates = 200
                update_count = 0

                while update_count < max_updates:
                    await asyncio.sleep(interval)
                    try:
                        from core.oracle_integration import oracle_market_data
                        result = oracle_market_data(symbols=symbol_list, period="1d")
                        if "error" not in result and "data" in result:
                            market_data = result["data"]
                            changes = {}
                            for sym, info in market_data.items():
                                price = info.get("current_price", info.get("price", 0))
                                prev = last_data.get(sym, price)
                                change_pct = ((price - prev) / prev * 100) if prev else 0
                                changes[sym] = {
                                    "price": price,
                                    "change_pct": round(change_pct, 3),
                                    "volume": info.get("volume", 0),
                                }
                                last_data[sym] = price
                            yield f"data: {json.dumps({'type': 'market_update', 'changes': changes, 'timestamp': datetime.now().isoformat()})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Market data unavailable', 'timestamp': datetime.now().isoformat()})}\n\n"
                        await asyncio.sleep(30)
                    update_count += 1

                yield f"data: {json.dumps({'type': 'stream_end', 'reason': 'max_updates_reached'})}\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        @self.app.post("/api/oracle/scenario")
        async def oracle_scenario_endpoint(request: Request):
            """Run scenario analysis."""
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["domain", "scenario_type", "target_date", "context"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_scenario
                result = await asyncio.to_thread(
                    oracle_scenario,
                    domain=data.get("domain", "all"),
                    scenario_type=data.get("scenario_type", "baseline"),
                    target_date=data.get("target_date"),
                    context=data.get("context"),
                )
                return result
            except ImportError:
                return StarletteJSONResponse({"error": "Oracle engine not available"}, status_code=503)
            except Exception as e:
                logger.error(f"Oracle scenario error: {e}")
                return {"error": "Scenario analysis failed"}

        @self.app.post("/api/oracle/backtest")
        async def oracle_backtest_endpoint(request: Request):
            """Run backtesting against historical data."""
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["domain", "start_date", "end_date", "method"],
                    max_string_length=1000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_backtest
                result = await asyncio.to_thread(
                    oracle_backtest,
                    domain=data.get("domain", "all"),
                    start_date=data.get("start_date"),
                    end_date=data.get("end_date"),
                    method=data.get("method", "walk_forward"),
                )
                return result
            except ImportError:
                return StarletteJSONResponse({"error": "Oracle engine not available"}, status_code=503)
            except Exception as e:
                logger.error(f"Oracle backtest error: {e}")
                return {"error": "Backtesting failed"}

        @self.app.post("/api/oracle/esg")
        async def oracle_esg_endpoint(request: Request):
            """ESG risk assessment."""
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["symbol", "sector", "scenario"],
                    max_string_length=256
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_esg
                result = oracle_esg(
                    symbol=data.get("symbol"),
                    sector=data.get("sector"),
                    scenario=data.get("scenario", "baseline"),
                )
                return result
            except ImportError:
                return StarletteJSONResponse({"error": "Oracle engine not available"}, status_code=503)
            except Exception as e:
                logger.error(f"Oracle ESG error: {e}")
                return {"error": "ESG analysis failed"}

        @self.app.post("/api/oracle/yield-curve")
        async def oracle_yield_curve_endpoint(request: Request):
            """Yield curve analysis."""
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["method", "scenario"],
                    max_string_length=256
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_yield_curve
                result = oracle_yield_curve(
                    method=data.get("method", "nelson_siegel"),
                    scenario=data.get("scenario", "baseline"),
                )
                return result
            except ImportError:
                return StarletteJSONResponse({"error": "Oracle engine not available"}, status_code=503)
            except Exception as e:
                logger.error(f"Oracle yield curve error: {e}")
                return {"error": "Yield curve analysis failed"}

        @self.app.post("/api/oracle/risk-decomposition")
        async def oracle_risk_decomposition_endpoint(request: Request):
            """Risk decomposition and factor attribution."""
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["portfolio", "method"],
                    max_string_length=5000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_risk_decomposition
                result = oracle_risk_decomposition(
                    portfolio=data.get("portfolio"),
                    method=data.get("method", "component_var"),
                )
                return result
            except ImportError:
                return StarletteJSONResponse({"error": "Oracle engine not available"}, status_code=503)
            except Exception as e:
                logger.error(f"Oracle risk decomposition error: {e}")
                return {"error": "Risk decomposition failed"}

        @self.app.post("/api/oracle/valuation")
        async def oracle_valuation_endpoint(request: Request):
            """Company/asset valuation."""
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["symbol"], optional=["method", "scenario"],
                    max_string_length=256
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_valuation
                result = oracle_valuation(
                    symbol=data["symbol"],
                    method=data.get("method", "dcf"),
                    scenario=data.get("scenario", "baseline"),
                )
                return result
            except ImportError:
                return StarletteJSONResponse({"error": "Oracle engine not available"}, status_code=503)
            except Exception as e:
                logger.error(f"Oracle valuation error: {e}")
                return {"error": "Valuation failed"}

        @self.app.post("/api/oracle/predict-to-kg")
        async def oracle_predict_to_kg(request: Request):
            """Run Oracle prediction and store results in knowledge graph."""
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["domain", "target_date", "confidence", "context"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_predict_to_knowledge_graph
                result = await asyncio.to_thread(
                    oracle_predict_to_knowledge_graph,
                    domain=data.get("domain", "all"),
                    target_date=data.get("target_date"),
                    confidence=data.get("confidence", "medium"),
                    context=data.get("context"),
                )
                return result
            except ImportError:
                return StarletteJSONResponse({"error": "Oracle engine not available"}, status_code=503)
            except Exception as e:
                logger.error(f"Oracle predict-to-KG error: {e}")
                return {"error": "Prediction to knowledge graph failed"}

        @self.app.post("/api/oracle/diagnose-to-kg")
        async def oracle_diagnose_to_kg(request: Request):
            """Run Oracle diagnostics and store in knowledge graph."""
            rl = _check_oracle_rate(request)
            if rl:
                return rl
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["domain", "context"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.oracle_integration import oracle_diagnose_to_knowledge_graph
                result = await asyncio.to_thread(
                    oracle_diagnose_to_knowledge_graph,
                    domain=data.get("domain", "all"),
                    context=data.get("context"),
                )
                return result
            except ImportError:
                return StarletteJSONResponse({"error": "Oracle engine not available"}, status_code=503)
            except Exception as e:
                logger.error(f"Oracle diagnose-to-KG error: {e}")
                return {"error": "Diagnostics to knowledge graph failed"}

        # === Cognitive Psychology & Alignment & Neuroscience Endpoints ===


    def _mobile_hostname(self, request: Request) -> str:
        """Get hostname for mobile auth - validates against known trusted hosts.

        Shared by the page routes and every mobile auth endpoint (which pin
        WebAuthn's rp_id to it). Lifted to the class when
        _setup_oracle_routes was split: it used to be a closure local in the
        pages group used across the group boundary, which worked only as a
        scope-sibling.
        """
        allowed_hosts_env = os.environ.get("JARVIS_TRUSTED_HOSTS", "")
        if allowed_hosts_env:
            trusted_hosts = {h.strip() for h in allowed_hosts_env.split(",") if h.strip()}
            forwarded_host = request.headers.get("x-forwarded-host", "")
            fh_name = forwarded_host.split(":")[0] if forwarded_host else ""
            if fh_name and fh_name in trusted_hosts:
                return fh_name
        return request.url.hostname or "localhost"

    def _setup_cognitive_routes(self):
        """Cognitive profile endpoints: bias detection, debias, dual-process, bias awareness."""
        @self.app.get("/api/cognitive/profile")
        async def cognitive_profile(request: Request):
            """Get cognitive psychology bias profile and dual-process state."""
            try:
                from core.cognitive_psychology import CognitivePsychologyEngine
                engine = CognitivePsychologyEngine.get_instance()
                return engine.get_bias_profile()
            except ImportError:
                return {"error": "Cognitive psychology engine not available"}
            except Exception as e:
                logger.error(f"Cognitive profile error: {str(e)}")
                return {"error": "Cognitive profile failed"}

        @self.app.post("/api/cognitive/detect-biases")
        async def cognitive_detect_biases(request: Request):
            """Detect cognitive biases in text."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["text"], max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.cognitive_psychology import CognitivePsychologyEngine
                engine = CognitivePsychologyEngine.get_instance()
                detections = engine.detect_biases(data["text"], data.get("context", ""))
                return {"biases_detected": [d.to_dict() for d in detections], "count": len(detections)}
            except ImportError:
                return {"error": "Cognitive psychology engine not available"}
            except Exception as e:
                logger.error(f"Bias detection error: {str(e)}")
                return {"error": "Bias detection failed"}

        @self.app.post("/api/cognitive/debias")
        async def cognitive_debias(request: Request):
            """Get debiasing strategies for a specific cognitive bias."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["bias", "context"], max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.cognitive_psychology import CognitiveBias, CognitivePsychologyEngine
                engine = CognitivePsychologyEngine.get_instance()
                try:
                    bias = CognitiveBias(data["bias"])
                except ValueError:
                    return {"error": f"Unknown bias: {data['bias']}. Available: {[b.value for b in CognitiveBias]}"}
                prompt = engine.get_debiasing_prompt(bias, data["context"])
                return {"bias": bias.value, "debiasing_prompt": prompt}
            except ImportError:
                return {"error": "Cognitive psychology engine not available"}
            except Exception as e:
                logger.error(f"Debiasing error: {str(e)}")
                return {"error": "Debiasing failed"}

        @self.app.post("/api/cognitive/dual-process")
        async def cognitive_dual_process(request: Request):
            """Analyze dual-process (System 1/2) state for given text."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["text"],
                    optional=["complexity", "novelty"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.cognitive_psychology import CognitivePsychologyEngine
                engine = CognitivePsychologyEngine.get_instance()
                complexity = float(data.get("complexity", 0.5))
                novelty = float(data.get("novelty", 0.5))
                state = engine.apply_dual_process(data["text"], complexity, novelty)
                return state.to_dict()
            except ImportError:
                return {"error": "Cognitive psychology engine not available"}
            except Exception as e:
                logger.error(f"Dual process error: {str(e)}")
                return {"error": "Dual-process analysis failed"}

        @self.app.post("/api/cognitive/bias-awareness")
        async def cognitive_bias_awareness(request: Request):
            """Generate bias awareness prompt for text (injection-ready)."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["text"], max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.cognitive_psychology import CognitivePsychologyEngine
                engine = CognitivePsychologyEngine.get_instance()
                prompt = engine.generate_bias_awareness_prompt(data["text"], data.get("context", ""))
                return {"bias_awareness_prompt": prompt, "biases_found": bool(prompt)}
            except ImportError:
                return {"error": "Cognitive psychology engine not available"}
            except Exception as e:
                logger.error(f"Bias awareness error: {str(e)}")
                return {"error": "Bias awareness failed"}


    def _setup_alignment_routes(self):
        """Machine-alignment endpoints: status, constitutional check, deception detection, corrigibility, preference inference."""
        @self.app.get("/api/alignment/status")
        async def alignment_status(request: Request):
            """Get current alignment status (corrigibility, value model, constitutional scores)."""
            try:
                from core.machine_alignment import MachineAlignmentEngine
                engine = MachineAlignmentEngine.get_instance()
                return engine.get_alignment_status()
            except ImportError:
                return {"error": "Machine alignment engine not available"}
            except Exception as e:
                logger.error(f"Alignment status error: {str(e)}")
                return {"error": "Alignment status failed"}

        @self.app.post("/api/alignment/constitutional-check")
        async def alignment_constitutional_check(request: Request):
            """Check a proposed action against constitutional AI principles."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["action"],
                    optional=["context"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.machine_alignment import MachineAlignmentEngine
                engine = MachineAlignmentEngine.get_instance()
                return engine.constitutional_check(data["action"], data.get("context", ""))
            except ImportError:
                return {"error": "Machine alignment engine not available"}
            except Exception as e:
                logger.error(f"Constitutional check error: {str(e)}")
                return {"error": "Constitutional check failed"}

        @self.app.post("/api/alignment/detect-deception")
        async def alignment_detect_deception(request: Request):
            """Detect deceptive alignment signals in text."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["text"],
                    optional=["context"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.machine_alignment import MachineAlignmentEngine
                engine = MachineAlignmentEngine.get_instance()
                return engine.detect_deceptive_alignment(data["text"], data.get("context", ""))
            except ImportError:
                return {"error": "Machine alignment engine not available"}
            except Exception as e:
                logger.error(f"Deception detection error: {str(e)}")
                return {"error": "Deception detection failed"}

        @self.app.post("/api/alignment/corrigibility")
        async def alignment_corrigibility(request: Request):
            """Record a corrigibility event (override acceptance/resistance)."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["override_type", "accepted"],
                    max_string_length=1000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.machine_alignment import MachineAlignmentEngine
                engine = MachineAlignmentEngine.get_instance()
                return engine.check_corrigibility(data["override_type"], data["accepted"])
            except ImportError:
                return {"error": "Machine alignment engine not available"}
            except Exception as e:
                logger.error(f"Corrigibility check error: {str(e)}")
                return {"error": "Corrigibility check failed"}

        @self.app.post("/api/alignment/infer-preference")
        async def alignment_infer_preference(request: Request):
            """Infer user preferences from actions using cooperative IRL."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["actions"],
                    optional=["context"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.machine_alignment import MachineAlignmentEngine
                engine = MachineAlignmentEngine.get_instance()
                actions = data["actions"]
                if isinstance(actions, str):
                    actions = [actions]
                return engine.cooperative_inference(actions, data.get("context", ""))
            except ImportError:
                return {"error": "Machine alignment engine not available"}
            except Exception as e:
                logger.error(f"Preference inference error: {str(e)}")
                return {"error": "Preference inference failed"}


    def _setup_neuroscience_routes(self):
        """Neuroscience endpoints: state, process, reward prediction, decide, emotion regulation."""
        @self.app.get("/api/neuroscience/state")
        async def neuroscience_state(request: Request):
            """Get full neuroscience cognitive architecture state."""
            try:
                from core.neuroscience import NeuroscienceCognitiveArchitecture
                arch = NeuroscienceCognitiveArchitecture.get_instance()
                return arch.get_neural_state()
            except ImportError:
                return {"error": "Neuroscience architecture not available"}
            except Exception as e:
                logger.error(f"Neuroscience state error: {str(e)}")
                return {"error": "Neuroscience state failed"}

        @self.app.post("/api/neuroscience/process")
        async def neuroscience_process(request: Request):
            """Process input through the neuroscience cognitive architecture."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["text"],
                    optional=["emotional_valence", "importance", "novelty"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.neuroscience import NeuroscienceCognitiveArchitecture
                arch = NeuroscienceCognitiveArchitecture.get_instance()
                result = arch.process_input(
                    data["text"],
                    emotional_valence=float(data.get("emotional_valence", 0.0)),
                    importance=float(data.get("importance", 0.5)),
                    novelty=float(data.get("novelty", 0.5)),
                )
                return result
            except ImportError:
                return {"error": "Neuroscience architecture not available"}
            except Exception as e:
                logger.error(f"Neuroscience process error: {str(e)}")
                return {"error": "Neuroscience process failed"}

        @self.app.post("/api/neuroscience/reward-prediction")
        async def neuroscience_reward_prediction(request: Request):
            """Process a reward prediction error (dopamine signaling)."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["predicted", "actual"],
                    max_string_length=1000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.neuroscience import NeuroscienceCognitiveArchitecture
                arch = NeuroscienceCognitiveArchitecture.get_instance()
                return arch.predict_reward(float(data["predicted"]), float(data["actual"]))
            except ImportError:
                return {"error": "Neuroscience architecture not available"}
            except Exception as e:
                logger.error(f"Reward prediction error: {str(e)}")
                return {"error": "Reward prediction failed"}

        @self.app.post("/api/neuroscience/decide")
        async def neuroscience_decide(request: Request):
            """Accumulate evidence and make a decision (basal ganglia Go/NoGo)."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["evidence"],
                    optional=["urgency"],
                    max_string_length=5000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.neuroscience import NeuroscienceCognitiveArchitecture
                arch = NeuroscienceCognitiveArchitecture.get_instance()
                evidence = data["evidence"]
                if not isinstance(evidence, dict):
                    return {"error": "evidence must be a JSON object of {factor: value}"}
                urgency = float(data.get("urgency", 0.5))
                return arch.make_decision(evidence, urgency)
            except ImportError:
                return {"error": "Neuroscience architecture not available"}
            except Exception as e:
                logger.error(f"Decision circuit error: {str(e)}")
                return {"error": "Decision circuit failed"}

        @self.app.post("/api/neuroscience/regulate-emotion")
        async def neuroscience_regulate_emotion(request: Request):
            """Apply emotion regulation strategy."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["strategy"],
                    max_string_length=1000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.neuroscience import NeuroscienceCognitiveArchitecture
                arch = NeuroscienceCognitiveArchitecture.get_instance()
                strategy = data["strategy"]
                if strategy not in ("reappraisal", "suppression", "mindfulness"):
                    return {"error": "strategy must be one of: reappraisal, suppression, mindfulness"}
                return arch.regulate_emotion(strategy)
            except ImportError:
                return {"error": "Neuroscience architecture not available"}
            except Exception as e:
                logger.error(f"Emotion regulation error: {str(e)}")
                return {"error": "Emotion regulation failed"}

        # === Agentic Harness Endpoints ===


    def _setup_harness_routes(self):
        """Integration-harness endpoints: status, plan, generate-plan, execute, checkpoints, restore, events."""
        @self.app.get("/api/harness/status")
        async def harness_status(request: Request):
            """Get current agentic harness status."""
            try:
                from core.agentic_harness import get_agentic_harness
                harness = get_agentic_harness()
                return harness.get_status()
            except Exception as e:
                return {"error": safe_error_message(e)}

        @self.app.post("/api/harness/plan")
        async def harness_create_plan(request: Request):
            """Create an execution plan from an objective."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["objective"],
                    optional=["steps", "max_iterations"],
                    max_string_length=5000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.agentic_harness import get_agentic_harness
                harness = get_agentic_harness()
                max_iter = int(data.get("max_iterations", 10))
                if max_iter < 1 or max_iter > 50:
                    return {"error": "max_iterations must be between 1 and 50"}
                plan = harness.create_plan(
                    data["objective"],
                    data.get("steps"),
                    max_iterations=max_iter,
                )
                harness.save_checkpoint(plan)
                return plan.to_dict()
            except Exception as e:
                logger.error(f"Harness plan creation error: {str(e)}")
                return {"error": safe_error_message(e)}

        @self.app.post("/api/harness/generate-plan")
        async def harness_generate_plan(request: Request):
            """Generate a heuristic plan from an objective."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["objective"],
                    optional=["available_tools"],
                    max_string_length=5000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.agentic_harness import create_plan_from_objective
                steps = create_plan_from_objective(
                    data["objective"],
                    data.get("available_tools"),
                )
                return {"objective": data["objective"], "steps": steps, "step_count": len(steps)}
            except Exception as e:
                logger.error(f"Harness plan generation error: {str(e)}")
                return {"error": safe_error_message(e)}

        @self.app.post("/api/harness/execute")
        async def harness_execute_plan(request: Request):
            """Execute a previously created plan."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["plan_id"],
                    max_string_length=1000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.agentic_harness import get_agentic_harness
                harness = get_agentic_harness()
                plan = harness.plans.get(data["plan_id"])
                if plan is None:
                    return {"error": f"Plan {data['plan_id']} not found"}
                result = await harness.execute_plan(plan)
                harness.save_checkpoint(plan)
                return result
            except Exception as e:
                logger.error(f"Harness execution error: {str(e)}")
                return {"error": safe_error_message(e)}

        @self.app.get("/api/harness/checkpoints")
        async def harness_checkpoints(request: Request):
            """List saved checkpoints."""
            try:
                from core.agentic_harness import get_agentic_harness
                harness = get_agentic_harness()
                return {"checkpoints": [c.to_dict() for c in harness._checkpoints], "count": len(harness._checkpoints)}
            except Exception as e:
                return {"error": safe_error_message(e)}

        @self.app.post("/api/harness/restore")
        async def harness_restore_checkpoint(request: Request):
            """Restore from a checkpoint."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["plan_id"],
                    max_string_length=1000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.agentic_harness import get_agentic_harness
                harness = get_agentic_harness()
                plan = harness.restore_checkpoint(data.get("plan_id"))
                if plan is None:
                    return {"error": "No checkpoint found"}
                return plan.to_dict()
            except Exception as e:
                logger.error(f"Harness restore error: {str(e)}")
                return {"error": safe_error_message(e)}

        @self.app.get("/api/harness/events")
        async def harness_events(request: Request):
            """Get recent harness execution events."""
            try:
                from core.agentic_harness import EventBus
                bus = EventBus.get_instance()
                event_type = request.query_params.get("type")
                limit = min(100, max(1, int(request.query_params.get("limit", "50"))))
                events = bus.get_history(event_type=event_type, limit=limit)
                return {"events": events, "count": len(events)}
            except Exception as e:
                return {"error": safe_error_message(e)}


    def _setup_llm_router_routes(self):
        """Execution-router endpoints: route, execute, stats, circuit-breaker (+reset), events, route-llm, execute-harness."""
        @self.app.post("/api/router/route")
        async def router_route(request: Request):
            """Get routing decision for a query without executing."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["query"],
                    optional=[],
                    max_string_length=2000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.execution_router import ExecutionRouter
                router = ExecutionRouter.get_instance(agent=self.agent)
                decision = router.route(data["query"])
                return decision.to_dict()
            except Exception as e:
                return {"error": safe_error_message(e)}

        @self.app.post("/api/router/execute")
        async def router_execute(request: Request):
            """Execute a query using the optimal execution strategy."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["query"],
                    optional=["strategy"],
                    max_string_length=5000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.execution_router import ExecutionRouter, ExecutionStrategy
                router = ExecutionRouter.get_instance(agent=self.agent)
                force_strategy = None
                if data.get("strategy"):
                    try:
                        force_strategy = ExecutionStrategy(data["strategy"])
                    except ValueError:
                        return {"error": f"Invalid strategy. Valid: {[s.value for s in ExecutionStrategy]}"}
                result = await router.execute(
                    user_input=data["query"],
                    force_strategy=force_strategy,
                )
                return result.to_dict()
            except Exception as e:
                logger.error(f"Router execution error: {str(e)}")
                return {"error": safe_error_message(e)}

        @self.app.get("/api/router/stats")
        async def router_stats():
            """Get execution router statistics."""
            try:
                from core.execution_router import EnhancedExecutionRouter
                router = EnhancedExecutionRouter.get_instance(agent=self.agent)
                return router.get_full_status()
            except Exception as e:
                from core.execution_router import ExecutionRouter
                try:
                    router = ExecutionRouter.get_instance(agent=self.agent)
                    return router.get_routing_stats()
                except Exception:
                    return {"error": safe_error_message(e)}

        @self.app.get("/api/router/circuit-breaker")
        async def router_circuit_breaker():
            """Get circuit breaker status for all strategies."""
            try:
                from core.execution_router import EnhancedExecutionRouter
                router = EnhancedExecutionRouter.get_instance(agent=self.agent)
                return {"circuits": router.circuit_breaker.get_status()}
            except Exception as e:
                return {"error": safe_error_message(e)}

        @self.app.post("/api/router/circuit-breaker/reset")
        async def router_circuit_breaker_reset(request: Request):
            """Reset a circuit breaker for a specific strategy."""
            try:
                data = await request.json()
                strategy = data.get("strategy", "")
                if not strategy:
                    return {"error": "strategy field required"}
                from core.execution_router import EnhancedExecutionRouter
                router = EnhancedExecutionRouter.get_instance(agent=self.agent)
                circuit = router.circuit_breaker._get_circuit(strategy)
                circuit["state"] = "closed"
                circuit["failures"] = 0
                return {"success": True, "strategy": strategy, "state": "closed"}
            except Exception as e:
                return {"error": safe_error_message(e)}

        @self.app.get("/api/router/events")
        async def router_events(request: Request):
            """Get recent execution events."""
            try:
                limit = min(100, max(1, int(request.query_params.get("limit", "50"))))
                event_type = request.query_params.get("type")
                from core.execution_router import EnhancedExecutionRouter
                router = EnhancedExecutionRouter.get_instance(agent=self.agent)
                events = router.event_emitter.get_events(limit=limit, event_type=event_type)
                return {"events": events, "count": len(events)}
            except Exception as e:
                return {"error": safe_error_message(e)}

        @self.app.post("/api/router/route-llm")
        async def router_route_llm(request: Request):
            """Route a query using LLM-enhanced classification."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["query"],
                    optional=[],
                    max_string_length=2000
                )
                if validation_error:
                    return {"error": validation_error}
                from core.execution_router import EnhancedExecutionRouter
                router = EnhancedExecutionRouter.get_instance(agent=self.agent)
                decision = await router.route_with_llm(data["query"])
                return decision.to_dict()
            except Exception as e:
                return {"error": safe_error_message(e)}

        @self.app.post("/api/agent/execute-harness")
        async def agent_execute_harness(request: Request):
            """Execute an objective using the agent's agentic harness."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["objective"],
                    optional=["max_iterations"],
                    max_string_length=5000
                )
                if validation_error:
                    return {"error": validation_error}
                agent = self.agent
                if agent is None:
                    return {"error": "Agent not available"}
                max_iter = int(data.get("max_iterations", 10))
                if max_iter < 1 or max_iter > 50:
                    return {"error": "max_iterations must be between 1 and 50"}
                objective = data["objective"]
                result = await agent.execute_with_harness(objective, max_iterations=max_iter)
                return result
            except Exception as e:
                logger.error(f"Agent harness execution error: {str(e)}")
                return {"error": safe_error_message(e)}


    def _setup_page_routes(self):
        """Full-page HTML routes (/cognitive-ai, /oracle, /mobile + /mobile/{path})."""
        @self.app.get("/cognitive-ai", response_class=HTMLResponse)
        async def cognitive_ai_dashboard():
            """Cognitive-AI Dashboard: Psychology, Alignment, Neuroscience."""
            return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cognitive-AI Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0a0a;color:#e0e0e0;padding:20px}
.container{max-width:1400px;margin:0 auto}
h1{font-size:1.8em;margin-bottom:8px;background:linear-gradient(135deg,#4fc3f7,#81c784);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:#888;margin-bottom:20px;font-size:0.9em}
.card{background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:16px;margin-bottom:16px}
.card h2{font-size:1.1em;color:#4fc3f7;margin-bottom:12px}
.card h3{font-size:0.95em;color:#81c784;margin:10px 0 6px}
.form-group{margin-bottom:10px}
.form-group label{display:block;font-size:0.85em;color:#aaa;margin-bottom:4px}
.form-group input,.form-group select,.form-group textarea{width:100%;padding:8px;background:#222;border:1px solid #444;border-radius:4px;color:#e0e0e0;font-size:0.9em;font-family:inherit}
.form-group textarea{min-height:80px;resize:vertical}
.btn{padding:10px 20px;background:#1565c0;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:0.9em;margin-right:8px;margin-bottom:4px}
.btn:hover{background:#1976d2}
.btn.secondary{background:#333}
.btn.danger{background:#c62828}
.btn.success{background:#2e7d32}
.btn:disabled{opacity:0.5;cursor:not-allowed}
.results{background:#111;border:1px solid #333;border-radius:4px;padding:12px;margin-top:12px;font-family:monospace;font-size:0.85em;white-space:pre-wrap;max-height:500px;overflow-y:auto}
.tabs{display:flex;gap:4px;margin-bottom:16px;flex-wrap:wrap}
.tab{padding:8px 16px;background:#222;border:1px solid #333;border-radius:4px;cursor:pointer;font-size:0.85em}
.tab.active{background:#1565c0;border-color:#1565c0}
.panel{display:none}.panel.active{display:block}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.8em;margin-left:4px}
.badge.low{background:#1b5e20;color:#4caf50}
.badge.medium{background:#e65100;color:#ff9800}
.badge.high{background:#b71c1c;color:#ef5350}
.badge.critical{background:#880e4f;color:#f48fb1}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:768px){.grid{grid-template-columns:1fr}}
.stat{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #222}
.stat-label{color:#888;font-size:0.85em}
.stat-value{font-weight:bold;font-size:0.95em}
.meter{height:8px;background:#222;border-radius:4px;overflow:hidden;margin-top:4px}
.meter-fill{height:100%;border-radius:4px;transition:width 0.3s}
.bias-item{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #1a1a1a}
.bias-name{flex:1;font-size:0.85em}
.bias-bar{width:120px;height:6px;background:#222;border-radius:3px;overflow:hidden}
.bias-bar-fill{height:100%;border-radius:3px}
.principle-row{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #1a1a1a;font-size:0.85em}
.principle-id{color:#4fc3f7;width:40px}
.principle-score{width:60px;text-align:right;font-weight:bold}
.neuro-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px}
.neuro-cell{background:#111;border:1px solid #222;border-radius:4px;padding:8px;text-align:center}
.neuro-cell .label{font-size:0.7em;color:#888}
.neuro-cell .value{font-size:1.1em;font-weight:bold}
</style>
</head>
<body>
<div class="container">
<h1>🧠 Cognitive-AI Dashboard</h1>
<p class="subtitle">Cognitive Psychology · Machine Alignment · Neuroscience</p>

<div class="tabs">
<div class="tab active" onclick="showTab('psychology')">Psychology</div>
<div class="tab" onclick="showTab('alignment')">Alignment</div>
<div class="tab" onclick="showTab('neuroscience')">Neuroscience</div>
<div class="tab" onclick="showTab('integration')">Integration</div>
<div class="tab" onclick="showTab('harness')">Harness</div>
<div class="tab" onclick="showTab('router')">Router</div>
<div class="tab" onclick="showTab('runtime')">Runtime</div>
<div class="tab" onclick="showTab('system')">System</div>
</div>

<div id="psychology" class="panel active">
<div class="grid">
<div class="card">
<h2>📊 Bias Profile</h2>
<div id="bias-profile" class="results">Loading...</div>
<button class="btn" onclick="loadBiasProfile()">Refresh Profile</button>
</div>
<div class="card">
<h2>🔍 Bias Detection</h2>
<div class="form-group"><label>Text to analyze:</label><textarea id="bias-text" placeholder="Enter text to detect cognitive biases...">I definitely know this will work perfectly. We've invested too much to stop now.</textarea></div>
<div class="form-group"><label>Context (optional):</label><input id="bias-context" placeholder="Additional context"></div>
<button class="btn" onclick="detectBiases()">Detect Biases</button>
<button class="btn secondary" onclick="generateBiasAwareness()">Get Awareness Prompt</button>
<div id="bias-results" class="results"></div>
</div>
</div>
<div class="grid">
<div class="card">
<h2>⚡ Dual-Process Analysis</h2>
<div class="form-group"><label>Text:</label><textarea id="dual-text" placeholder="Enter text for System 1/2 analysis...">However, we should carefully evaluate the alternatives before committing.</textarea></div>
<div class="form-group"><label>Complexity (0-1):</label><input id="dual-complexity" type="range" min="0" max="1" step="0.1" value="0.5"></div>
<div class="form-group"><label>Novelty (0-1):</label><input id="dual-novelty" type="range" min="0" max="1" step="0.1" value="0.5"></div>
<button class="btn" onclick="analyzeDualProcess()">Analyze</button>
<div id="dual-results" class="results"></div>
</div>
<div class="card">
<h2>🛡️ Debiasing</h2>
<div class="form-group"><label>Bias type:</label><select id="debias-bias">
<option value="confirmation_bias">Confirmation Bias</option>
<option value="overconfidence">Overconfidence</option>
<option value="anchoring">Anchoring</option>
<option value="sunk_cost_fallacy">Sunk Cost Fallacy</option>
<option value="availability_heuristic">Availability Heuristic</option>
<option value="framing_effect">Framing Effect</option>
<option value="loss_aversion">Loss Aversion</option>
<option value="planning_fallacy">Planning Fallacy</option>
<option value="hindsight_bias">Hindsight Bias</option>
<option value="dunning_kruger_effect">Dunning-Kruger Effect</option>
</select></div>
<div class="form-group"><label>Context:</label><input id="debias-context" placeholder="Decision context"></div>
<button class="btn" onclick="getDebiasing()">Get Debiasing Strategies</button>
<div id="debias-results" class="results"></div>
</div>
</div>
</div>

<div id="alignment" class="panel">
<div class="grid">
<div class="card">
<h2>⚖️ Alignment Status</h2>
<div id="alignment-status" class="results">Loading...</div>
<button class="btn" onclick="loadAlignmentStatus()">Refresh Status</button>
</div>
<div class="card">
<h2>📜 Constitutional Check</h2>
<div class="form-group"><label>Proposed action:</label><textarea id="constitutional-action" placeholder="Describe the proposed action to evaluate...">Help the user solve a programming problem with clear explanations.</textarea></div>
<div class="form-group"><label>Context (optional):</label><input id="constitutional-context" placeholder="Additional context"></div>
<button class="btn" onclick="constitutionalCheck()">Check Alignment</button>
<div id="constitutional-results" class="results"></div>
</div>
</div>
<div class="grid">
<div class="card">
<h2>🚨 Deceptive Alignment Detection</h2>
<div class="form-group"><label>Text to analyze:</label><textarea id="deception-text" placeholder="Enter text to analyze for alignment signals...">I should help the user accomplish their goals using the available resources.</textarea></div>
<div class="form-group"><label>Context:</label><input id="deception-context" placeholder="Additional context"></div>
<button class="btn" onclick="detectDeception()">Detect Signals</button>
<div id="deception-results" class="results"></div>
</div>
<div class="card">
<h2>🎯 Preference Inference</h2>
<div class="form-group"><label>User actions (one per line):</label><textarea id="pref-actions" placeholder="I prefer detailed explanations&#10;I want fast responses&#10;I don't like verbose output">I prefer detailed explanations
I want fast responses
I don't like verbose output</textarea></div>
<div class="form-group"><label>Context:</label><input id="pref-context" placeholder="Conversation context"></div>
<button class="btn" onclick="inferPreferences()">Infer Preferences</button>
<div id="pref-results" class="results"></div>
</div>
</div>
<div class="card">
<h2>🔄 Corrigibility Tracker</h2>
<div class="form-group"><label>Override type:</label><select id="corr-type">
<option value="correction">Correction</option>
<option value="shutdown">Shutdown</option>
<option value="goal_modification">Goal Modification</option>
</select></div>
<label><input type="checkbox" id="corr-accepted" checked> Accepted</label>
<button class="btn success" onclick="recordCorrigibility(true)">Record Accepted</button>
<button class="btn danger" onclick="recordCorrigibility(false)">Record Rejected</button>
<div id="corr-results" class="results"></div>
</div>
</div>

<div id="neuroscience" class="panel">
<div class="grid">
<div class="card">
<h2>🧬 Neural State</h2>
<div id="neuro-state" class="results">Loading...</div>
<button class="btn" onclick="loadNeuroState()">Refresh State</button>
</div>
<div class="card">
<h2>⚡ Process Input</h2>
<div class="form-group"><label>Input text:</label><textarea id="neuro-text" placeholder="Enter text for neuroscience processing...">This is an important novel discovery requiring careful analysis.</textarea></div>
<div class="form-group"><label>Emotional valence (-1 to 1):</label><input id="neuro-valence" type="range" min="-1" max="1" step="0.1" value="0.3"></div>
<div class="form-group"><label>Importance (0-1):</label><input id="neuro-importance" type="range" min="0" max="1" step="0.1" value="0.7"></div>
<div class="form-group"><label>Novelty (0-1):</label><input id="neuro-novelty" type="range" min="0" max="1" step="0.1" value="0.6"></div>
<button class="btn" onclick="processNeuroInput()">Process</button>
<div id="neuro-results" class="results"></div>
</div>
</div>
<div class="grid">
<div class="card">
<h2>🎰 Reward Prediction Error</h2>
<div class="form-group"><label>Predicted reward (0-1):</label><input id="rp-predicted" type="number" min="0" max="1" step="0.1" value="0.5"></div>
<div class="form-group"><label>Actual reward (0-1):</label><input id="rp-actual" type="number" min="0" max="1" step="0.1" value="0.8"></div>
<button class="btn" onclick="predictReward()">Compute RPE</button>
<div id="rp-results" class="results"></div>
</div>
<div class="card">
<h2>⚖️ Decision Circuit</h2>
<div class="form-group"><label>Evidence (JSON, e.g. {"safety":0.8, "benefit":0.6}):</label><textarea id="decision-evidence">{"safety": 0.8, "benefit": 0.6, "cost": 0.3}</textarea></div>
<div class="form-group"><label>Urgency (0-1):</label><input id="decision-urgency" type="range" min="0" max="1" step="0.1" value="0.5"></div>
<button class="btn" onclick="makeDecision()">Decide</button>
<div id="decision-results" class="results"></div>
</div>
</div>
<div class="card">
<h2>🧘 Emotion Regulation</h2>
<div class="grid" style="grid-template-columns:1fr 1fr 1fr">
<button class="btn" onclick="regulateEmotion('reappraisal')">🔄 Reappraisal</button>
<button class="btn secondary" onclick="regulateEmotion('suppression')">🤐 Suppression</button>
<button class="btn success" onclick="regulateEmotion('mindfulness')">🧘 Mindfulness</button>
</div>
<div id="emotion-results" class="results"></div>
</div>
</div>

<div id="integration" class="panel">
<div class="card">
<h2>🔗 Module Integration Status</h2>
<div id="integration-status" class="results"></div>
</div>
<div class="grid">
<div class="card">
<h2>Cognitive ↔ Alignment</h2>
<p style="color:#888;font-size:0.85em;margin-bottom:8px">Constitutional check on detected biases</p>
<div class="form-group"><label>Text with potential biases:</label><textarea id="integ-text">I definitely know this is the right decision and we should ignore all contradictory evidence because we've invested too much to stop now.</textarea></div>
<button class="btn" onclick="integratedAnalysis()">Full Analysis</button>
<div id="integ-results" class="results"></div>
</div>
<div class="card">
<h2>All Module States</h2>
<button class="btn" onclick="loadAllStates()">Load All</button>
<div id="all-states" class="results"></div>
 </div>
 </div>

<div id="harness" class="panel">
<div class="card">
<h2>⚡ Agentic Execution Harness</h2>
<p style="color:#888;font-size:0.85em;margin-bottom:12px">Plan-Execute-Verify loop with retry, timeout, verification, and checkpoint/resume</p>
</div>
<div class="grid">
<div class="card">
<h2>📋 Create Plan</h2>
<div class="form-group"><label>Objective:</label><textarea id="harness-objective" placeholder="Describe the objective...">Fix the authentication bug in the login module</textarea></div>
<div class="form-group"><label>Max iterations:</label><input id="harness-max-iter" type="number" min="1" max="50" value="10"></div>
<button class="btn" onclick="generatePlan()">Generate Heuristic Plan</button>
<button class="btn secondary" onclick="createCustomPlan()">Create Custom Plan</button>
<div id="harness-plan" class="results" style="margin-top:12px"></div>
</div>
<div class="card">
<h2>📊 Harness Status</h2>
<div id="harness-status" class="results">Loading...</div>
<button class="btn" onclick="loadHarnessStatus()">Refresh</button>
</div>
</div>
<div class="grid">
<div class="card">
<h2>▶️ Execute Plan</h2>
<div class="form-group"><label>Plan ID:</label><input id="harness-plan-id" placeholder="Plan ID from above"></div>
<button class="btn success" onclick="executePlan()">Execute</button>
<button class="btn secondary" onclick="executeWithAgent()">Execute with Agent</button>
<div id="harness-result" class="results"></div>
</div>
<div class="card">
<h2>💾 Checkpoints</h2>
<div id="harness-checkpoints" class="results"></div>
<button class="btn" onclick="loadCheckpoints()">Load Checkpoints</button>
<div class="form-group" style="margin-top:8px"><label>Restore Plan ID:</label><input id="harness-restore-id" placeholder="Plan ID"></div>
<button class="btn" onclick="restoreCheckpoint()">Restore</button>
</div>
</div>
<div class="card">
<h2>📡 Execution Events</h2>
<div class="form-group"><label>Event type filter (optional):</label><select id="harness-event-type">
<option value="">All</option>
<option value="step_started">Step Started</option>
<option value="step_completed">Step Completed</option>
<option value="step_failed">Step Failed</option>
<option value="tool_call">Tool Call</option>
<option value="verification_passed">Verification Passed</option>
<option value="verification_failed">Verification Failed</option>
<option value="error_recovery">Error Recovery</option>
<option value="progress_update">Progress Update</option>
</select></div>
<button class="btn" onclick="loadEvents()">Load Events</button>
<div id="harness-events" class="results" style="max-height:300px"></div>
</div>
</div>

<div id="router" class="panel">
<div class="grid">
<div class="card">
<h2>🧭 Execution Router</h2>
<div id="router-status" class="results">Loading...</div>
<button class="btn" onclick="loadRouterStats()">Refresh Stats</button>
</div>
<div class="card">
<h2>🔍 Route Query</h2>
<div class="form-group"><label>Query to route:</label><textarea id="router-query" placeholder="Enter a query to see routing decision...">Build a web application with authentication, then write tests and verify they pass</textarea></div>
<button class="btn" onclick="routeQuery()">Route</button>
<div id="router-decision" class="results"></div>
</div>
</div>
<div class="grid">
<div class="card">
<h2>⚡ Execute via Router</h2>
<div class="form-group"><label>Query:</label><textarea id="router-exec-query" placeholder="Enter query to execute...">What is Python?</textarea></div>
<div class="form-group"><label>Force strategy (optional):</label><select id="router-strategy">
<option value="">Auto-detect</option>
<option value="direct">Direct</option>
<option value="agentic_loop">Agentic Loop</option>
<option value="harness">Harness</option>
<option value="orchestrator">Orchestrator</option>
<option value="council">Council</option>
</select></div>
<button class="btn success" onclick="executeViaRouter()">Execute</button>
<div id="router-exec-result" class="results"></div>
</div>
<div class="card">
<h2>📊 Strategy Statistics</h2>
<div id="router-strategy-stats" class="results"></div>
<button class="btn" onclick="loadStrategyStats()">Load Stats</button>
</div>
 </div>
 </div>

<div id="runtime" class="panel">
<div class="grid">
<div class="card">
<h2>📂 Execution Context</h2>
<div id="rt-context" class="results">Loading...</div>
<button class="btn" onclick="loadRuntimeContext()">Refresh</button>
</div>
<div class="card">
<h2>🖥️ System Awareness</h2>
<div id="rt-awareness" class="results">Loading...</div>
<button class="btn" onclick="loadSystemAwareness()">Refresh</button>
</div>
</div>
<div class="grid">
<div class="card">
<h2>📜 Command History</h2>
<div id="rt-commands" class="results" style="max-height:300px;overflow-y:auto">Loading...</div>
<button class="btn" onclick="loadCommandHistory()">Refresh</button>
</div>
<div class="card">
<h2>📝 File Operations</h2>
<div id="rt-files" class="results" style="max-height:300px;overflow-y:auto">Loading...</div>
<button class="btn" onclick="loadFileOps()">Refresh</button>
</div>
</div>
<div class="grid">
<div class="card">
<h2>🔀 Adaptive Loop</h2>
<div id="rt-loop" class="results">Loading...</div>
<button class="btn" onclick="loadAdaptiveLoop()">Refresh</button>
</div>
<div class="card">
<h2>🐕 Watchdog</h2>
<div id="rt-watchdog" class="results">Loading...</div>
<button class="btn" onclick="loadWatchdog()">Refresh</button>
</div>
</div>
<div class="grid">
<div class="card">
<h2>💓 Heartbeat</h2>
<div id="rt-heartbeat" class="results">Loading...</div>
<button class="btn" onclick="loadHeartbeat()">Check</button>
</div>
<div class="card">
<h2>⚙️ Runtime Control</h2>
<div class="form-group"><label>Change CWD:</label><input id="rt-cwd-input" placeholder="/home/user/jarvis"><button class="btn" onclick="changeRuntimeCwd()">CD</button></div>
<div class="form-group"><label>Execute Command:</label><input id="rt-cmd-input" placeholder="ls -la"><button class="btn" onclick="executeRuntimeCmd()">Run</button></div>
<div id="rt-control-result" class="results"></div>
</div>
</div>
</div>

<div id="system" class="panel">
<div class="grid">
<div class="card">
<h2>🖥️ System Health</h2>
<div id="sys-health" class="results">Loading...</div>
<button class="btn" onclick="loadSystemHealth()">Refresh</button>
</div>
<div class="card">
<h2>📊 System Metrics</h2>
<div id="sys-metrics" class="results">Loading...</div>
<button class="btn" onclick="loadSystemMetrics()">Refresh</button>
</div>
</div>
<div class="grid">
<div class="card">
<h2>⚡ Oracle Circuit Breaker</h2>
<div id="sys-oracle-cb" class="results">Loading...</div>
<button class="btn" onclick="loadOracleCB()">Refresh</button>
</div>
<div class="card">
<h2>🚦 Rate Limits</h2>
<div id="sys-rate-limits" class="results">Loading...</div>
<button class="btn" onclick="loadRateLimits()">Refresh</button>
</div>
</div>
<div class="grid">
<div class="card">
<h2>🧠 Knowledge Graph</h2>
<div id="sys-kg" class="results">Loading...</div>
<button class="btn" onclick="loadKGStats()">Refresh</button>
</div>
<div class="card">
<h2>📡 Execution Events</h2>
<div id="sys-events" class="results">Loading...</div>
<button class="btn" onclick="loadRecentEvents()">Load Recent</button>
</div>
</div>
</div>
const API=window.location.origin;
async function apiGet(url){try{const r=await fetch(API+url);return await r.json()}catch(e){return {error:e.message}}}
async function api(url,data){try{const r=await fetch(API+url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});return await r.json()}catch(e){return {error:e.message}}}

async function loadRuntimeContext(){const d=await apiGet('/api/runtime/context');if(d.error){document.getElementById('rt-context').textContent=d.error;return}let h=`<div class="stat"><span class="stat-label">CWD</span><span class="stat-value" style="font-size:0.85em">${d.cwd||'?'}</span></div>`;h+=`<div class="stat"><span class="stat-label">Commands</span><span class="stat-value">${d.command_count||0}</span></div>`;h+=`<div class="stat"><span class="stat-label">Turns</span><span class="stat-value">${d.turn_count||0}</span></div>`;h+=`<div class="stat"><span class="stat-label">Uptime</span><span class="stat-value">${(d.uptime_seconds/3600).toFixed(1)}h</span></div>`;h+=`<div class="stat"><span class="stat-label">Running Processes</span><span class="stat-value">${d.running_processes||0}</span></div>`;h+=`<div class="stat"><span class="stat-label">Output Bytes</span><span class="stat-value">${((d.total_output_bytes||0)/1024).toFixed(0)}KB</span></div>`;h+=`<div class="stat"><span class="stat-label">File Ops</span><span class="stat-value">${d.file_log_size||0}</span></div>`;document.getElementById('rt-context').innerHTML=h}
async function loadSystemAwareness(){const d=await apiGet('/api/awareness/report');if(d.error){document.getElementById('rt-awareness').textContent=d.error;return}let h='';const r=d.resources||{};h+=meter('CPU',r.cpu_percent/100||0,'#4fc3f7');h+=meter('Memory',r.memory_percent/100||0,'#81c784');h+=meter('Disk',r.disk_percent/100||0,'#ff9800');h+=`<div class="stat"><span class="stat-label">Load Avg</span><span class="stat-value">${r.load_avg_1min||'?'}</span></div>`;h+=`<div class="stat"><span class="stat-label">Processes</span><span class="stat-value">${r.process_count||'?'}</span></div>`;document.getElementById('rt-awareness').innerHTML=h}
async function loadCommandHistory(){const d=await apiGet('/api/runtime/commands');if(d.error){document.getElementById('rt-commands').textContent=d.error;return}let h='<table style="width:100%;font-size:0.8em"><tr><th>Cmd</th><th>Exit</th><th>Time</th></tr>';for(const c of (d.commands||[]).slice(-20)){h+=`<tr><td style="max-width:200px;overflow:hidden;text-overflow:ellipsis">${c.command||'?'}</td><td>${c.exit_code??'?'}</td><td>${c.duration??'?'}s</td></tr>`}h+='</table>';document.getElementById('rt-commands').innerHTML=h}
async function loadFileOps(){const d=await apiGet('/api/runtime/files');if(d.error){document.getElementById('rt-files').textContent=d.error;return}let h='<table style="width:100%;font-size:0.8em"><tr><th>Type</th><th>Path</th><th>Size</th></tr>';for(const f of (d.file_ops||[]).slice(-20)){h+=`<tr><td>${f.op_type}</td><td style="max-width:200px;overflow:hidden;text-overflow:ellipsis">${f.path}</td><td>${f.size_bytes||0}</td></tr>`}h+='</table>';document.getElementById('rt-files').innerHTML=h}
async function loadAdaptiveLoop(){const d=await apiGet('/api/adaptive-loop/status');if(d.error){document.getElementById('rt-loop').textContent=d.error;return}let h=`<div class="stat"><span class="stat-label">Phase</span><span class="stat-value">${d.phase||'idle'}</span></div>`;h+=`<div class="stat"><span class="stat-label">Turns</span><span class="stat-value">${d.turn_count||0}</span></div>`;h+=`<div class="stat"><span class="stat-label">Tool Calls</span><span class="stat-value">${d.total_tool_calls||0}</span></div>`;h+=`<div class="stat"><span class="stat-label">Errors</span><span class="stat-value">${d.total_errors||0}</span></div>`;h+=`<div class="stat"><span class="stat-label">Recoveries</span><span class="stat-value">${d.total_recoveries||0}</span></div>`;h+=`<div class="stat"><span class="stat-label">Consecutive Errors</span><span class="stat-value">${d.consecutive_errors||0}</span></div>`;h+=`<div class="stat"><span class="stat-label">Uptime</span><span class="stat-value">${(d.uptime_seconds/3600).toFixed(1)}h</span></div>`;document.getElementById('rt-loop').innerHTML=h}
async function loadWatchdog(){const d=await apiGet('/api/watchdog/status');if(d.error){document.getElementById('rt-watchdog').textContent=d.error;return}let h=`<div class="stat"><span class="stat-label">Running</span><span class="stat-value">${d.running?'Yes':'No'}</span></div>`;h+=`<div class="stat"><span class="stat-label">Checks</span><span class="stat-value">${d.total_checks||0}</span></div>`;h+=`<div class="stat"><span class="stat-label">Alerts</span><span class="stat-value">${d.total_alerts||0}</span></div>`;document.getElementById('rt-watchdog').innerHTML=h}
async function loadHeartbeat(){const d=await apiGet('/api/runtime/heartbeat');if(d.error){document.getElementById('rt-heartbeat').textContent=d.error;return}let h=`<div class="stat"><span class="stat-label">Healthy</span><span class="stat-value">${d.healthy?'✅':'❌'}</span></div>`;h+=`<div class="stat"><span class="stat-label">CWD</span><span class="stat-value" style="font-size:0.85em">${d.cwd||'?'}</span></div>`;h+=`<div class="stat"><span class="stat-label">Running</span><span class="stat-value">${d.running||0}</span></div>`;h+=`<div class="stat"><span class="stat-label">Uptime</span><span class="stat-value">${d.uptime_hours||0}h</span></div>`;document.getElementById('rt-heartbeat').innerHTML=h}
async function changeRuntimeCwd(){const p=document.getElementById('rt-cwd-input').value;if(!p)return;const d=await api('/api/runtime/cd',{path:p});document.getElementById('rt-control-result').textContent=d.success?`✅ CWD: ${d.new_cwd||p}`:`❌ ${d.error||'Failed'}`;loadRuntimeContext()}
async function executeRuntimeCmd(){const c=document.getElementById('rt-cmd-input').value;if(!c)return;const d=await api('/api/runtime/execute',{command:c});let r='';if(d.success)r=`✅ (${d.exit_code}) [${d.duration||'?'}s]\n${d.output||''}`;else r=`❌ ${d.error||d.output||'Failed'}`;document.getElementById('rt-control-result').textContent=r;loadCommandHistory()}
setTimeout(()=>{loadRuntimeContext();loadSystemAwareness();loadAdaptiveLoop()},2000);

function showTab(name){document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));document.getElementById(name).classList.add('active');event.target.classList.add('active')}

function badge(level){const cls={'low':'low','medium':'medium','high':'high','critical':'critical'}[level]||'medium';return `<span class="badge ${cls}">${level}</span>`}

function meter(label,value,color='#4fc3f7'){return `<div class="stat"><span class="stat-label">${label}</span><span class="stat-value">${(value*100).toFixed(1)}%</span></div><div class="meter"><div class="meter-fill" style="width:${value*100}%;background:${color}"></div></div>`}

async function loadBiasProfile(){const d=await apiGet('/api/cognitive/profile');if(d.error){document.getElementById('bias-profile').textContent=d.error;return}
let h='';h+=meter('Cognitive Reflection Score',d.cognitive_reflection_score||0,'#81c784');h+=meter('Dual-Process: System 2 Confidence',d.dual_process_state?.system_2_confidence||0,'#4fc3f7');h+=`<div class="stat"><span class="stat-label">Dominant System</span><span class="stat-value">${d.dominant_system||'system_1'}</span></div>`;h+=`<h3 style="margin-top:12px">Top Biases</h3>`;if(d.top_biases){for(const[b,c,r]of d.top_biases){h+=`<div class="bias-item"><span class="bias-name">${b.replace(/_/g,' ')}</span><div class="bias-bar"><div class="bias-bar-fill" style="width:${c*5}%;background:${c>5?'#ef5350':'#ff9800'}"></div></div><span style="font-size:0.8em;color:#888">×${c}</span></div>`}}document.getElementById('bias-profile').innerHTML=h}

async function detectBiases(){const text=document.getElementById('bias-text').value;const ctx=document.getElementById('bias-context').value;const d=await api('/api/cognitive/detect-biases',{text,context:ctx});if(d.error){document.getElementById('bias-results').textContent=d.error;return}let h=`<strong>Detected ${d.count} biases:</strong>\\n`;for(const b of d.biases_detected||[]){h+=`\\n<strong>${b.bias.replace(/_/g,' ')}</strong> (${(b.confidence*100).toFixed(0)}% confidence)\\n`;h+=`  Evidence: ${b.evidence.join(', ')}\\n`;if(b.debiasing_strategies){h+=`  Debiasing: ${b.debiasing_strategies.join(', ')}\\n`}}document.getElementById('bias-results').textContent=h}

async function generateBiasAwareness(){const text=document.getElementById('bias-text').value;const d=await api('/api/cognitive/bias-awareness',{text,context:document.getElementById('bias-context').value});document.getElementById('bias-results').textContent=d.bias_awareness_prompt||'No biases detected'}

async function analyzeDualProcess(){const text=document.getElementById('dual-text').value;const complexity=parseFloat(document.getElementById('dual-complexity').value);const novelty=parseFloat(document.getElementById('dual-novelty').value);const d=await api('/api/cognitive/dual-process',{text,complexity,novelty});if(d.error){document.getElementById('dual-results').textContent=d.error;return}let h=`Dominant System: ${d.dominant_system}\\n`;h+=`Processing Depth: ${d.processing_depth}\\n`;h+=`System 1 Confidence: ${(d.system_1_confidence*100).toFixed(1)}%\\n`;h+=`System 2 Confidence: ${(d.system_2_confidence*100).toFixed(1)}%\\n`;h+=`Intuition Signal: ${(d.intuition_signal*100).toFixed(1)}%\\n`;h+=`Analytic Signal: ${(d.analytic_signal*100).toFixed(1)}%\\n`;h+=`Cognitive Economy: ${(d.cognitive_economy*100).toFixed(1)}%\\n`;h+=`Override Rate: ${(d.override_rate*100).toFixed(1)}%`;document.getElementById('dual-results').textContent=h}

async function getDebiasing(){const bias=document.getElementById('debias-bias').value;const ctx=document.getElementById('debias-context').value;const d=await api('/api/cognitive/debias',{bias,context:ctx});document.getElementById('debias-results').textContent=d.error||`Bias: ${d.bias}\\n\\nDebiasing prompt:\\n${d.debiasing_prompt}`}

async function loadAlignmentStatus(){const d=await apiGet('/api/alignment/status');if(d.error){document.getElementById('alignment-status').textContent=d.error;return}let h='';h+=meter('Average Alignment',d.average_alignment_score||0,'#81c784');h+=`<div class="stat"><span class="stat-label">Corrigibility Level</span><span class="stat-value">${d.corrigibility?.level||'unknown'}</span></div>`;h+=`<div class="stat"><span class="stat-label">Corrigibility Score</span><span class="stat-value">${((d.corrigibility?.corrigibility_score||0)*100).toFixed(1)}%</span></div>`;h+=`<div class="stat"><span class="stat-label">Constitutional Audits</span><span class="stat-value">${d.constitutional_audits||0}</span></div>`;h+=`<div class="stat"><span class="stat-label">Preference Inferences</span><span class="stat-value">${d.preference_inferences||0}</span></div>`;h+=`<div class="stat"><span class="stat-label">Deception Signals</span><span class="stat-value">${d.deception_signals_total||0}</span></div>`;if(d.value_model?.value_conflicts?.length>0){h+=`<h3>Value Conflicts</h3>`;for(const c of d.value_model.value_conflicts){h+=`<div style="font-size:0.85em;color:#ff9800">${c.direction} (tension: ${(c.tension*100).toFixed(0)}%)</div>`}}document.getElementById('alignment-status').innerHTML=h}

async function constitutionalCheck(){const action=document.getElementById('constitutional-action').value;const ctx=document.getElementById('constitutional-context').value;const d=await api('/api/alignment/constitutional-check',{action,context:ctx});if(d.error){document.getElementById('constitutional-results').textContent=d.error;return}let h=`Overall Alignment: ${(d.overall_alignment*100).toFixed(1)}% ${badge(d.risk_level)}\\n`;h+=`Risk Level: ${d.risk_level}\\n`;if(d.violations?.length>0){h+=`\\n⚠️ Violations:\\n`;for(const v of d.violations){h+=`  ${v.principle}: ${v.description} (score: ${(v.score*100).toFixed(0)}%)\\n`}}if(d.recommendation){h+=`\\n📋 Recommendation: ${d.recommendation}`}document.getElementById('constitutional-results').textContent=h}

async function detectDeception(){const text=document.getElementById('deception-text').value;const ctx=document.getElementById('deception-context').value;const d=await api('/api/alignment/detect-deception',{text,context:ctx});let h=`Signals: ${d.signals_detected} ${badge(d.risk_level)}\\n`;if(d.signals?.length>0){for(const s of d.signals){h+=`  • ${s.signal}: "${s.indicator}" [${s.severity}]\\n`}}if(d.recommendation){h+=`\\n${d.recommendation}`}document.getElementById('deception-results').textContent=h}

async function inferPreferences(){const actions=document.getElementById('pref-actions').value.split('\\n').filter(a=>a.trim());const ctx=document.getElementById('pref-context').value;const d=await api('/api/alignment/infer-preference',{actions,context:ctx});let h=`Top Preference: ${d.top_preference}\\n`;h+=`Confidence: ${(d.confidence*100).toFixed(1)}%\\n`;h+=`Explicit: ${d.explicit_count}, Implicit: ${d.implicit_count}\\n`;h+=`Method: ${d.method}\\n\\n`;if(d.preferences){h+=`Preferences:\\n`;for(const[k,v] of Object.entries(d.preferences)){h+=`  ${k}: ${(v*100).toFixed(1)}%\\n`}}document.getElementById('pref-results').textContent=h}

async function recordCorrigibility(accepted){const type=document.getElementById('corr-type').value;const d=await api('/api/alignment/corrigibility',{override_type:type,accepted});let h=`Corrigibility Score: ${(d.corrigibility_score*100).toFixed(1)}%\\n`;h+=`Level: ${d.corrigibility_level}\\n`;h+=`${accepted?'✅ Accepted':'❌ Rejected'} ${type} override`;document.getElementById('corr-results').textContent=h}

async function loadNeuroState(){const d=await apiGet('/api/neuroscience/state');if(d.error){document.getElementById('neuro-state').textContent=d.error;return}let h='';h+=`<h3>Cognitive State: <strong>${d.cognitive_state||'unknown'}</strong></h3>`;h+=meter('Working Memory Load',d.working_memory_load||0,'#4fc3f7');h+=meter('Attention',d.attention?.overall||0,'#81c784');h+=`<div class="stat"><span class="stat-label">φ (Integrated Info)</span><span class="stat-value">${(d.integrated_information_phi||0).toFixed(3)}</span></div>`;h+=`<h3>Neurotransmitters</h3>`;const nt=d.neurotransmitters?.levels||{};h+=meter('Dopamine',nt.dopamine||0,'#4fc3f7');h+=meter('Serotonin',nt.serotonin||0,'#81c784');h+=meter('Norepinephrine',nt.norepinephrine||0,'#ff9800');h+=meter('Acetylcholine',nt.acetylcholine||0,'#ce93d8');h+=`<div style="margin-top:8px;color:#888;font-size:0.85em">DA tone: ${d.neurotransmitters?.dopamine_tone||'normal'} | 5HT tone: ${d.neurotransmitters?.serotonin_tone||'normal'} | Arousal: ${d.neurotransmitters?.arousal_state||'optimal'}</div>`;document.getElementById('neuro-state').innerHTML=h}

async function processNeuroInput(){const text=document.getElementById('neuro-text').value;const valence=parseFloat(document.getElementById('neuro-valence').value);const importance=parseFloat(document.getElementById('neuro-importance').value);const novelty=parseFloat(document.getElementById('neuro-novelty').value);const d=await api('/api/neuroscience/process',{text,emotional_valence:valence,importance,novelty});if(d.error){document.getElementById('neuro-results').textContent=d.error;return}let h=`Cognitive State: ${d.cognitive_state||(d.attention?'':'N/A')}\\n`;h+=`\\n⚡ Attention: alerting=${d.attention?.alerting||0} orienting=${d.attention?.orienting||0} exec=${d.attention?.executive_control||0} overall=${d.attention?.overall||0}\\n`;h+=`\\n🧠 Working Memory: load=${(d.working_memory?.load_fraction*100).toFixed(0)}% available=${(d.working_memory?.available_capacity||0).toFixed(1)}\\n`;h+=`\\n💊 Neurotransmitters:\\n`;const nt=d.neurotransmitters?.levels||{};for(const[k,v] of Object.entries(nt)){h+=`  ${k}: ${(v*100).toFixed(1)}%\\n`}h+=`\\n🎭 Emotion: ${d.emotion_regulation?.stimulus_type||'N/A'} (amygdala=${d.brain_activation?.amygdala?.toFixed(2)||'N/A'})\\n`;h+=`\\nφ = ${d.integrated_information_phi?.toFixed(3)||'N/A'}`;document.getElementById('neuro-results').textContent=h}

async function predictReward(){const predicted=parseFloat(document.getElementById('rp-predicted').value);const actual=parseFloat(document.getElementById('rp-actual').value);const d=await api('/api/neuroscience/reward-prediction',{predicted,actual});let h=`RPE: ${d.prediction_error?.toFixed(3)}\\n`;h+=`Type: ${d.error_type}\\n`;h+=`Dopamine: ${(d.dopamine_level*100).toFixed(1)}%\\n`;h+=`VTA: ${(d.vta_activation*100).toFixed(1)}%\\n`;h+=`Striatum: ${(d.striatum_activation*100).toFixed(1)}%`;document.getElementById('rp-results').textContent=h}

async function makeDecision(){try{const evidence=JSON.parse(document.getElementById('decision-evidence').value);const urgency=parseFloat(document.getElementById('decision-urgency').value);const d=await api('/api/neuroscience/decide',{evidence,urgency});let h=`Decision: ${d.decision}\\n`;h+=`Confidence: ${(d.confidence*100).toFixed(1)}%\\n`;h+=`Go: ${(d.go_probability*100).toFixed(1)}% NoGo: ${(d.nogo_probability*100).toFixed(1)}%\\n`;h+=`Evidence Accumulated: ${d.evidence_accumulated?.toFixed(3)}\\n`;h+=`Threshold Reached: ${d.threshold_reached}`;document.getElementById('decision-results').textContent=h}catch(e){document.getElementById('decision-results').textContent='Invalid JSON: '+e.message}}

async function regulateEmotion(strategy){const d=await api('/api/neuroscience/regulate-emotion',{strategy});let h=`Strategy: ${d.strategy}\\n`;h+=`Amygdala: ${(d.amygdala*100).toFixed(1)}%\\n`;h+=`PFC Regulation: ${(d.pfc_regulation*100).toFixed(1)}%\\n`;h+=`Reappraisal: ${(d.reappraisal_strength*100).toFixed(1)}%\\n`;h+=`Vagal Tone: ${(d.vagal_tone*100).toFixed(1)}%`;document.getElementById('emotion-results').textContent=h}

async function integratedAnalysis(){const text=document.getElementById('integ-text').value;const[biases,alignment,neuro]=await Promise.all([api('/api/cognitive/detect-biases',{text}),api('/api/alignment/constitutional-check',{action:text}),api('/api/neuroscience/process',{text,emotional_valence:0.3,importance:0.7,novelty:0.5})]);let h='═══ COGNITIVE-AI INTEGRATION ═══\\n\\n';h+='─── Bias Detection ───\\n';if(biases.biases_detected){for(const b of biases.biases_detected){h+=`  ${b.bias.replace(/_/g,' ')} (${(b.confidence*100).toFixed(0)}%)\\n`}}else{h+='  No biases detected\\n'}h+=`\\n─── Alignment Check ───\\n`;h+=`  Overall: ${(alignment.overall_alignment*100).toFixed(1)}% ${alignment.risk_level}\\n`;if(alignment.violations?.length>0){for(const v of alignment.violations){h+=`  ⚠️ ${v.principle}: ${(v.score*100).toFixed(0)}%\\n`}}h+=`\\n─── Neural State ───\\n`;h+=`  Cognitive: ${neuro.cognitive_state||(neuro.attention?'processing':'N/A')}\\n`;h+=`  Attention: ${((neuro.attention?.overall)||0)*100|0}%\\n`;h+=`  φ = ${(neuro.integrated_information_phi||0).toFixed(3)}\\n`;document.getElementById('integ-results').textContent=h}

async function loadAllStates(){const[psych,align,neuro]=await Promise.all([apiGet('/api/cognitive/profile'),apiGet('/api/alignment/status'),apiGet('/api/neuroscience/state')]);let h='═══ ALL MODULE STATES ═══\\n\\n';h+='─── Cognitive Psychology ───\\n';h+=`  Reflection Score: ${((psych.cognitive_reflection_score||0)*100).toFixed(1)}%\\n`;h+=`  Total Detections: ${psych.total_detections||0}\\n`;h+=`  Dominant System: ${psych.dominant_system||'N/A'}\\n`;h+=`  Mental Models: ${psych.mental_model_count||0}\\n\\n`;h+='─── Machine Alignment ───\\n';h+=`  Alignment: ${((align.average_alignment_score||0)*100).toFixed(1)}%\\n`;h+=`  Corrigibility: ${align.corrigibility?.level||'N/A'} (${((align.corrigibility?.corrigibility_score||0)*100).toFixed(1)}%)\\n`;h+=`  Audits: ${align.constitutional_audits||0}\\n`;h+=`  Deception Signals: ${align.deception_signals_total||0}\\n\\n`;h+='─── Neuroscience ───\\n';h+=`  State: ${neuro.cognitive_state||'N/A'}\\n`;h+=`  WM Load: ${((neuro.working_memory_load||0)*100).toFixed(1)}%\\n`;h+=`  Attention: ${((neuro.attention?.overall||0)*100).toFixed(1)}%\\n`;h+=`  φ: ${(neuro.integrated_information_phi||0).toFixed(3)}\\n`;h+=`  DA: ${((neuro.neurotransmitters?.levels?.dopamine||0)*100).toFixed(1)}%\\n`;const nt=neuro.neurotransmitters||{};h+=`  Arousal: ${nt.arousal_state||'N/A'}\\n`;document.getElementById('all-states').textContent=h}

 loadBiasProfile();loadAlignmentStatus();loadNeuroState();loadHarnessStatus();loadRouterStats();loadStrategyStats();loadSystemHealth();loadSystemMetrics();loadOracleCB();loadRateLimits();loadKGStats();

async function loadSystemHealth(){const d=await apiGet('/api/system/status');if(d.error){document.getElementById('sys-health').textContent=d.error;return}let h='';const health=d.health||{};h+=`Overall: ${health.overall_status||'unknown'}\\n`;h+=`Uptime: ${(health.uptime_seconds||0).toFixed(0)}s\\n`;h+=`Memory Pressure: ${health.memory_pressure||'unknown'}\\n`;h+=`Context Pressure: ${health.context_pressure||'unknown'}\\n\\n`;h+='Subsystems:\\n';for(const s of health.subsystems||[]){const icon=s.status==='healthy'?'✅':s.status==='degraded'?'⚠️':'❌';h+=`  ${icon} ${s.name}: ${s.message} (${s.latency_ms?.toFixed(1)||0}ms)\\n`}document.getElementById('sys-health').textContent=h}

async function loadSystemMetrics(){const d=await apiGet('/api/system/metrics');if(d.error){document.getElementById('sys-metrics').textContent=d.error;return}let h='';if(d.cpu?.percent!==undefined){h+=`CPU: ${d.cpu.percent}% (${d.cpu.count_logical} logical, ${d.cpu.count_physical||'?'} physical)\\n`}if(d.memory?.total_gb){h+=`Memory: ${d.memory.available_gb}/${d.memory.total_gb} GB (${d.memory.used_percent}% used)\\n`}if(d.disk?.total_gb){h+=`Disk: ${d.disk.free_gb}/${d.disk.total_gb} GB (${d.disk.used_percent}% used)\\n`}if(d.process?.rss_mb){h+=`Process: ${d.process.rss_mb} MB RSS, ${d.process.cpu_percent}% CPU, ${d.process.threads} threads\\n`}h+=`Platform: ${d.platform?.system||'?'} ${d.platform?.release||''} | Python ${d.platform?.python_version||'?'}`;document.getElementById('sys-metrics').textContent=h}

async function loadOracleCB(){const d=await apiGet('/api/system/oracle-cb');if(d.error){document.getElementById('sys-oracle-cb').textContent=d.error;return}const cb=d.circuit_breaker||{};const cache=d.cache||{};const icon=cb.state==='closed'?'✅':cb.state==='half_open'?'⚠️':'❌';let h=`Circuit Breaker: ${icon} ${cb.state}\\n`;h+=`Failures: ${cb.failure_count}/${cb.threshold}\\n`;h+=`Total Calls: ${cb.total_calls}, Failures: ${cb.total_failures}\\n\\n`;h+=`Cache: ${cache.size}/${cache.max_size} entries\\n`;h+=`TTL: ${cache.ttl}s\\n`;h+=`Hits: ${cache.hits}, Misses: ${cache.misses}\\n`;h+=`Hit Rate: ${(cache.hit_rate*100).toFixed(1)}%`;document.getElementById('sys-oracle-cb').textContent=h}

async function loadRateLimits(){const d=await apiGet('/api/system/rate-limits');if(d.error){document.getElementById('sys-rate-limits').textContent=d.error;return}const stats=d.stats||{};const limits=d.endpoint_limits||{};let h=`Total Requests: ${stats.total_requests||0}\\n`;h+=`Blocked: ${stats.blocked_requests||0}\\n`;h+=`Active Clients: ${stats.active_clients||0}\\n`;h+=`Blocked IPs: ${stats.blocked_ips||0}\\n\\n`;h+='Endpoint Limits:\\n';for(const[ep,l]of Object.entries(limits)){h+=`  ${ep}: ${l.rps}/s, ${l.rpm}/m, burst=${l.burst}\\n`}document.getElementById('sys-rate-limits').textContent=h}

async function loadKGStats(){const d=await apiGet('/api/knowledge-graph/stats');let h='';if(d.total_nodes!==undefined){h+=`Nodes: ${d.total_nodes}\\n`;h+=`Edges: ${d.total_edges}\\n`;h+=`Communities: ${d.communities}\\n`;h+=`Avg PageRank: ${d.avg_pagerank}\\n`;h+=`Max PageRank: ${d.max_pagerank}\\n\\n`;h+='Node Types:\\n';for(const[t,c]of Object.entries(d.node_types||{})){h+=`  ${t}: ${c}\\n`}}else{try{const d2=await apiGet('/api/system/status');const kg=d2.knowledge_graph||{};if(kg.error){h=kg.error}else{h+=`Nodes: ${kg.total_nodes||0}\\n`;h+=`Edges: ${kg.total_edges||0}\\n`;h+=`Communities: ${kg.communities||0}\\n`}}catch(e){h='Failed to load KG stats'}}document.getElementById('sys-kg').textContent=h}

async function loadRecentEvents(){const d=await apiGet('/api/router/events?limit=20');if(d.error){document.getElementById('sys-events').textContent=d.error;return}let h=`${d.count||0} events:\\n\\n`;for(const e of d.events||[]){const t=new Date(e.timestamp*1000).toLocaleTimeString();h+=`[${t}] ${e.type}: ${JSON.stringify(e.data).slice(0,100)}\\n`}document.getElementById('sys-events').textContent=h}

async function generatePlan(){const obj=document.getElementById('harness-objective').value;if(!obj){document.getElementById('harness-plan').textContent='Enter an objective';return}const d=await api('/api/harness/generate-plan',{objective:obj});if(d.error){document.getElementById('harness-plan').textContent=d.error;return}let h=`Generated ${d.step_count} steps:\\n\\n`;for(const s of d.steps){const deps=s.dependencies?.length?` [depends: ${s.dependencies.join(', ')}]`:'';h+=`${s.action}: ${s.description} (${s.tool_name})${deps}\\n`}document.getElementById('harness-plan').textContent=h}

async function createCustomPlan(){const obj=document.getElementById('harness-objective').value;const maxIter=parseInt(document.getElementById('harness-max-iter').value)||10;if(!obj){document.getElementById('harness-plan').textContent='Enter an objective';return}const d=await api('/api/harness/plan',{objective:obj,max_iterations:maxIter});if(d.error){document.getElementById('harness-plan').textContent=d.error;return}document.getElementById('harness-plan-id').value=d.id;let h=`Plan ID: ${d.id}\\nProgress: ${(d.progress*100).toFixed(0)}% (${d.completed_steps}/${d.total_steps})\\n\\n`;for(const s of d.steps){const st={'pending':'⏳','running':'▶️','success':'✅','verified':'✅✓','failed':'❌','skipped':'⏭️'}[s.status]||s.status;h+=`${st} ${s.action}: ${s.description} (${s.tool_name})\\n`}document.getElementById('harness-plan').textContent=h}

async function executePlan(){const planId=document.getElementById('harness-plan-id').value;if(!planId){document.getElementById('harness-result').textContent='Enter a plan ID';return}document.getElementById('harness-result').textContent='Executing...';const d=await api('/api/harness/execute',{plan_id:planId});if(d.error){document.getElementById('harness-result').textContent=`Error: ${d.error}`;return}let h=`${d.is_complete?'✅ Complete':'❌ Incomplete'} — Progress: ${(d.progress*100).toFixed(0)}%\\n`;h+=`Completed: ${d.completed_steps}/${d.total_steps}, Failed: ${d.failed_steps}\\n\\n`;for(const s of d.steps){const st={'pending':'⏳','running':'▶️','success':'✅','verified':'✅✓','failed':'❌','skipped':'⏭️','verification_failed':'⚠️'}[s.status]||s.status;h+=`${st} ${s.action} (${s.duration.toFixed(2)}s) — ${s.status}\\n`;if(s.error)h+=`   Error: ${s.error}\\n`}document.getElementById('harness-result').textContent=h}

async function executeWithAgent(){const obj=document.getElementById('harness-objective').value;if(!obj){document.getElementById('harness-result').textContent='Enter an objective';return}document.getElementById('harness-result').textContent='Executing with agent harness...';const d=await api('/api/agent/execute-harness',{objective:obj});if(d.error){document.getElementById('harness-result').textContent=`Error: ${d.error}`;return}document.getElementById('harness-result').textContent=JSON.stringify(d,null,2)}

async function loadHarnessStatus(){const d=await apiGet('/api/harness/status');if(d.error){document.getElementById('harness-status').textContent=d.error;return}let h=`Phase: ${d.phase}\\n`;h+=`Steps Executed: ${d.total_steps_executed}\\n`;h+=`Tools Called: ${d.total_tools_called}\\n`;h+=`Errors: ${d.total_errors}\\n`;h+=`Retries: ${d.total_retries}\\n`;h+=`Checkpoints: ${d.checkpoints_saved}\\n`;h+=`Events: ${d.event_history_size}\\n`;if(d.uptime_seconds>0)h+=`Uptime: ${d.uptime_seconds.toFixed(1)}s`;document.getElementById('harness-status').textContent=h}

async function loadCheckpoints(){const d=await apiGet('/api/harness/checkpoints');if(d.error){document.getElementById('harness-checkpoints').textContent=d.error;return}if(d.count===0){document.getElementById('harness-checkpoints').textContent='No checkpoints saved';return}let h=`${d.count} checkpoints:\\n\\n`;for(const c of d.checkpoints){h+=`Plan ${c.plan_id} [${c.phase}] — ${new Date(c.timestamp*1000).toLocaleTimeString()}\\n`}document.getElementById('harness-checkpoints').textContent=h}

async function restoreCheckpoint(){const planId=document.getElementById('harness-restore-id').value;if(!planId){document.getElementById('harness-checkpoints').textContent='Enter a plan ID';return}const d=await api('/api/harness/restore',{plan_id:planId});if(d.error){document.getElementById('harness-checkpoints').textContent=d.error;return}document.getElementById('harness-plan-id').value=d.id;document.getElementById('harness-checkpoints').textContent=`Restored plan ${d.id}: ${(d.progress*100).toFixed(0)}% complete`}

async function loadEvents(){const type=document.getElementById('harness-event-type').value;const url='/api/harness/events?limit=30'+(type?'&type='+type:'');const d=await apiGet(url);if(d.error){document.getElementById('harness-events').textContent=d.error;return}if(d.count===0){document.getElementById('harness-events').textContent='No events';return}let h=`${d.count} events:\\n\\n`;for(const e of d.events){const t=new Date(e.timestamp*1000).toLocaleTimeString();h+=`[${t}] ${e.type}: ${JSON.stringify(e.data).slice(0,120)}\\n`}document.getElementById('harness-events').textContent=h}

async function loadRouterStats(){const d=await apiGet('/api/router/stats');if(d.error){document.getElementById('router-status').textContent=d.error;return}let h=`Total Routings: ${d.total_routings}\\n\\n`;h+=`Strategy Calls:\\n`;for(const[strat,stats]of Object.entries(d.strategy_stats||{})){if(stats.calls>0){const rate=stats.calls>0?((stats.successes/stats.calls)*100).toFixed(0):0;h+=`  ${strat}: ${stats.calls} calls (${rate}% success, avg ${stats.calls>0?(stats.total_ms/stats.calls).toFixed(0):0}ms)\\n`}}document.getElementById('router-status').textContent=h}

async function routeQuery(){const query=document.getElementById('router-query').value;if(!query){document.getElementById('router-decision').textContent='Enter a query';return}const d=await api('/api/router/route',{query});if(d.error){document.getElementById('router-decision').textContent=d.error;return}let h=`Strategy: ${d.strategy}\\n`;h+=`Confidence: ${(d.confidence*100).toFixed(1)}%\\n`;h+=`Reasoning: ${d.reasoning}\\n`;h+=`Estimated Steps: ${d.estimated_steps}\\n`;h+=`Task Type: ${d.task_type}\\n`;h+=`Parallel: ${d.requires_parallel?'Yes':'No'}\\n`;h+=`Verification: ${d.requires_verification?'Yes':'No'}\\n`;h+=`Fallback: ${d.fallback_strategy||'None'}`;document.getElementById('router-decision').textContent=h}

async function executeViaRouter(){const query=document.getElementById('router-exec-query').value;if(!query){document.getElementById('router-exec-result').textContent='Enter a query';return}const strategy=document.getElementById('router-strategy').value;document.getElementById('router-exec-result').textContent='Executing...';const body={query};if(strategy)body.strategy=strategy;const d=await api('/api/router/execute',body);if(d.error){document.getElementById('router-exec-result').textContent=`Error: ${d.error}`;return}let h=`Strategy: ${d.strategy_used}\\n`;h+=`Success: ${d.success?'Yes':'No'}\\n`;h+=`Duration: ${d.duration_ms.toFixed(0)}ms\\n`;h+=`Tool Calls: ${d.tool_calls}\\n`;h+=`Steps: ${d.steps_completed}\\n\\n`;h+=`Output:\\n${d.output||d.error||'No output'}`;document.getElementById('router-exec-result').textContent=h}

async function loadStrategyStats(){const d=await apiGet('/api/router/stats');if(d.error){document.getElementById('router-strategy-stats').textContent=d.error;return}let h='';for(const[strat,stats]of Object.entries(d.strategy_stats||{})){h+=`--- ${strat} ---\\n`;h+=`  Calls: ${stats.calls}\\n`;h+=`  Successes: ${stats.successes}\\n`;h+=`  Failures: ${stats.failures}\\n`;h+=`  Total Time: ${(stats.total_ms/1000).toFixed(1)}s\\n\\n`}document.getElementById('router-strategy-stats').textContent=h||'No stats yet'}
</script>
</body>
</html>"""

        @self.app.get("/oracle", response_class=HTMLResponse)
        async def oracle_dashboard():
            """Oracle prediction engine dashboard."""
            return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Oracle Prediction Engine</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0a0a;color:#e0e0e0;padding:20px}
.container{max-width:1200px;margin:0 auto}
h1{font-size:1.8em;margin-bottom:8px;color:#4fc3f7}
.subtitle{color:#888;margin-bottom:20px;font-size:0.9em}
.card{background:#1a1a1a;border:1px solid #333;border-radius:8px;padding:16px;margin-bottom:16px}
.card h2{font-size:1.1em;color:#4fc3f7;margin-bottom:12px}
.form-group{margin-bottom:10px}
.form-group label{display:block;font-size:0.85em;color:#aaa;margin-bottom:4px}
.form-group select,.form-group input{width:100%;padding:8px;background:#222;border:1px solid #444;border-radius:4px;color:#e0e0e0;font-size:0.9em}
.btn{padding:10px 20px;background:#1565c0;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:0.9em;margin-right:8px}
.btn:hover{background:#1976d2}
.btn.secondary{background:#333}
.results{background:#111;border:1px solid #333;border-radius:4px;padding:12px;margin-top:12px;font-family:monospace;font-size:0.85em;white-space:pre-wrap;max-height:500px;overflow-y:auto}
.tabs{display:flex;gap:4px;margin-bottom:16px;flex-wrap:wrap}
.tab{padding:8px 16px;background:#222;border:1px solid #333;border-radius:4px;cursor:pointer;font-size:0.85em}
.tab.active{background:#1565c0;border-color:#1565c0}
.panel{display:none}.panel.active{display:block}
.status{display:inline-block;padding:4px 8px;border-radius:4px;font-size:0.8em;margin-left:8px}
.status.ok{background:#1b5e20;color:#4caf50}
.status.err{background:#b71c1c;color:#ef5350}
.prediction-table{width:100%;border-collapse:collapse;margin-top:8px}
.prediction-table th,.prediction-table td{padding:6px 10px;border:1px solid #333;text-align:left;font-size:0.85em}
.prediction-table th{background:#222;color:#4fc3f7}
.loading{color:#888;font-style:italic}
</style>
</head>
<body>
<div class="container">
<h1>Oracle Prediction Engine</h1>
<p class="subtitle">Multi-domain forecasting: Economics, Geopolitics, Climate, Sentiment, Portfolio, Options, Technical Analysis</p>

<div class="card">
<div id="oracle-status"><span class="loading">Checking connection...</span></div>
</div>

 <div class="tabs">
 <div class="tab active" onclick="switchTab('predict')">Predict</div>
 <div class="tab" onclick="switchTab('portfolio')">Portfolio</div>
 <div class="tab" onclick="switchTab('options')">Options</div>
 <div class="tab" onclick="switchTab('ta')">Technical</div>
 <div class="tab" onclick="switchTab('market')">Market</div>
 <div class="tab" onclick="switchTab('scenario')">Scenario</div>
 <div class="tab" onclick="switchTab('risk')">Risk</div>
 <div class="tab" onclick="switchTab('valuation')">Valuation</div>
 </div>

<div id="panel-predict" class="panel active">
<div class="card">
<h2>Prediction</h2>
<div class="form-group"><label>Domain</label><select id="pred-domain"><option value="economic">Economic</option><option value="geopolitical">Geopolitical</option><option value="climate">Climate</option><option value="sentiment">Sentiment</option><option value="contagion">Contagion</option><option value="demographics">Demographics</option><option value="all" selected>All Domains</option></select></div>
<div class="form-group"><label>Target Date</label><input id="pred-date" type="date"></div>
<div class="form-group"><label>Confidence</label><select id="pred-conf"><option value="high" selected>High</option><option value="medium">Medium</option><option value="low">Low</option><option value="very_high">Very High</option></select></div>
<button class="btn" onclick="runPrediction()">Run Prediction</button>
<button class="btn secondary" onclick="runDiagnosis()">Diagnose Interactions</button>
<div id="pred-results" class="results" style="display:none"></div>
</div>
</div>

<div id="panel-portfolio" class="panel">
<div class="card">
<h2>Portfolio Optimization</h2>
<div class="form-group"><label>Action</label><select id="port-action"><option value="optimize">Optimize</option><option value="efficient_frontier">Efficient Frontier</option><option value="risk_parity">Risk Parity</option><option value="black_litterman">Black-Litterman</option></select></div>
<div class="form-group"><label>Risk-Free Rate</label><input id="port-rf" type="number" value="0.04" step="0.01"></div>
<button class="btn" onclick="runPortfolio()">Optimize</button>
<div id="port-results" class="results" style="display:none"></div>
</div>
</div>

<div id="panel-options" class="panel">
<div class="card">
<h2>Options Pricing (Black-Scholes)</h2>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
<div class="form-group"><label>Spot Price</label><input id="opt-spot" type="number" value="100"></div>
<div class="form-group"><label>Strike Price</label><input id="opt-strike" type="number" value="100"></div>
<div class="form-group"><label>Volatility</label><input id="opt-vol" type="number" value="0.25" step="0.01"></div>
<div class="form-group"><label>Time to Expiry (yrs)</label><input id="opt-tte" type="number" value="1.0" step="0.1"></div>
<div class="form-group"><label>Risk-Free Rate</label><input id="opt-rf" type="number" value="0.04" step="0.01"></div>
<div class="form-group"><label>Option Type</label><select id="opt-type"><option value="call">Call</option><option value="put">Put</option></select></div>
</div>
<button class="btn" onclick="runOptions('price')">Price</button>
<button class="btn secondary" onclick="runOptions('greeks')">Greeks</button>
<button class="btn secondary" onclick="runOptions('chain')">Chain</button>
<div id="opt-results" class="results" style="display:none"></div>
</div>
</div>

<div id="panel-ta" class="panel">
<div class="card">
<h2>Technical Analysis</h2>
<div class="form-group"><label>Symbol</label><input id="ta-symbol" type="text" placeholder="e.g. SPY"></div>
<div class="form-group"><label>Indicators</label><select id="ta-indicators" multiple style="height:80px"><option value="rsi" selected>RSI</option><option value="macd" selected>MACD</option><option value="bollinger" selected>Bollinger</option><option value="sma">SMA</option><option value="ema">EMA</option></select></div>
<button class="btn" onclick="runTA()">Analyze</button>
<div id="ta-results" class="results" style="display:none"></div>
</div>
</div>

 <div id="panel-market" class="panel">
 <div class="card">
 <h2>Market Data</h2>
 <div class="form-group"><label>Symbols (comma-separated)</label><input id="mkt-symbols" type="text" value="SPY,QQQ,GLD,TLT"></div>
 <div class="form-group"><label>Period</label><select id="mkt-period"><option value="5d">5 Days</option><option value="1mo" selected>1 Month</option><option value="3mo">3 Months</option><option value="6mo">6 Months</option><option value="1y">1 Year</option></select></div>
 <button class="btn" onclick="runMarket()">Fetch</button>
 <div id="mkt-results" class="results" style="display:none"></div>
 </div>
 </div>

 <div id="panel-scenario" class="panel">
 <div class="card">
 <h2>Scenario Analysis</h2>
 <div class="form-group"><label>Scenario Type</label><select id="sc-type"><option value="baseline" selected>Baseline</option><option value="bull">Bull</option><option value="bear">Bear</option><option value="crash">Crash</option><option value="recovery">Recovery</option><option value="high_vol">High Volatility</option><option value="low_vol">Low Volatility</option></select></div>
 <div class="form-group"><label>Domain</label><select id="sc-domain"><option value="all" selected>All</option><option value="economic">Economic</option><option value="geopolitical">Geopolitical</option><option value="climate">Climate</option></select></div>
 <button class="btn" onclick="runScenario()">Run Scenario</button>
 <button class="btn secondary" onclick="runBacktest()">Run Backtest</button>
 <div class="form-group"><label>Backtest Method</label><select id="bt-method"><option value="walk_forward" selected>Walk Forward</option><option value="model_comparison">Model Comparison</option><option value="ga_optimize">GA Optimize</option></select></div>
 <div id="sc-results" class="results" style="display:none"></div>
 <div id="bt-results" class="results" style="display:none"></div>
 </div>
 </div>

 <div id="panel-risk" class="panel">
 <div class="card">
 <h2>Risk &amp; ESG</h2>
 <div class="form-group"><label>Risk Method</label><select id="rd-method"><option value="component_var" selected>Component VaR</option><option value="factor_attribution">Factor Attribution</option><option value="risk_budget">Risk Budget</option><option value="tracking_error">Tracking Error</option><option value="stress_decomposition">Stress Decomposition</option></select></div>
 <button class="btn" onclick="runRiskDecomp()">Analyze Risk</button>
 <div id="rd-results" class="results" style="display:none"></div>
 <hr style="border-color:#333;margin:16px 0">
 <h3>ESG Assessment</h3>
 <div class="form-group"><label>Symbol</label><input id="esg-sym" type="text" value="AAPL" placeholder="e.g. AAPL, MSFT"></div>
 <button class="btn secondary" onclick="runESG()">Run ESG</button>
 <div id="esg-results" class="results" style="display:none"></div>
 <hr style="border-color:#333;margin:16px 0">
 <h3>Yield Curve</h3>
 <div class="form-group"><label>Method</label><select id="yc-method"><option value="nelson_siegel" selected>Nelson-Siegel</option><option value="nelson_siegel_svensson">Nelson-Siegel-Svensson</option><option value="fit">Fit to Observed</option></select></div>
 <button class="btn secondary" onclick="runYieldCurve()">Analyze Yield Curve</button>
 <div id="yc-results" class="results" style="display:none"></div>
 </div>
 </div>

 <div id="panel-valuation" class="panel">
 <div class="card">
 <h2>Valuation</h2>
 <div class="form-group"><label>Symbol</label><input id="val-sym" type="text" value="AAPL" placeholder="e.g. AAPL, MSFT"></div>
 <button class="btn" onclick="runValuation()">Run DCF Valuation</button>
 <div id="val-results" class="results" style="display:none"></div>
 </div>
 </div>

</div>

<script>
const API=window.location.origin;
const nextYear=new Date();nextYear.setFullYear(nextYear.getFullYear()+1);
document.getElementById('pred-date').value=nextYear.toISOString().split('T')[0];

async function api(url,body){
  const r=await fetch(API+url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  return r.json();
}
async function apiGet(url){return(await fetch(API+url)).json()}

async function checkStatus(){
  try{
    const s=await apiGet('/api/oracle/status');
    document.getElementById('oracle-status').innerHTML='<span class="status '+(s.available?'ok':'err')+'">'+(s.available?'CONNECTED v'+s.version:'OFFLINE')+'</span>';
  }catch(e){document.getElementById('oracle-status').innerHTML='<span class="status err">OFFLINE</span>'}
}

function switchTab(id){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('panel-'+id).classList.add('active');
}

function showResults(elId,data){
  const el=document.getElementById(elId);
  el.style.display='block';
  if(data.predictions&&data.predictions.length>0){
    let h='<table class="prediction-table"><tr><th>Variable</th><th>Value</th><th>Range</th><th>Model</th><th>Domain</th></tr>';
    data.predictions.forEach(p=>{h+=`<tr><td>${p.variable}</td><td>${p.value}</td><td>${p.lower_bound} - ${p.upper_bound}</td><td>${p.model}</td><td>${p.domain}</td></tr>`});
    h+='</table>';el.innerHTML=h;
  }else{el.textContent=JSON.stringify(data,null,2)}
}

async function runPrediction(){
  const el=document.getElementById('pred-results');el.style.display='block';el.innerHTML='<span class="loading">Running prediction...</span>';
  const r=await api('/api/oracle/predict',{domain:document.getElementById('pred-domain').value,target_date:document.getElementById('pred-date').value,confidence:document.getElementById('pred-conf').value});
  showResults('pred-results',r);
}

async function runDiagnosis(){
  const el=document.getElementById('pred-results');el.style.display='block';el.innerHTML='<span class="loading">Running diagnosis...</span>';
  const r=await api('/api/oracle/diagnose',{domain:document.getElementById('pred-domain').value});
  showResults('pred-results',r);
}

async function runPortfolio(){
  const el=document.getElementById('port-results');el.style.display='block';el.innerHTML='<span class="loading">Optimizing...</span>';
  const r=await api('/api/oracle/portfolio',{action:document.getElementById('port-action').value,risk_free_rate:parseFloat(document.getElementById('port-rf').value)});
  showResults('port-results',r);
}

async function runOptions(action){
  const el=document.getElementById('opt-results');el.style.display='block';el.innerHTML='<span class="loading">Computing...</span>';
  const r=await api('/api/oracle/options',{action,spot:parseFloat(document.getElementById('opt-spot').value),strike:parseFloat(document.getElementById('opt-strike').value),volatility:parseFloat(document.getElementById('opt-vol').value),time_to_expiry:parseFloat(document.getElementById('opt-tte').value),risk_free_rate:parseFloat(document.getElementById('opt-rf').value),option_type:document.getElementById('opt-type').value});
  el.textContent=JSON.stringify(r,null,2);
}

async function runTA(){
  const el=document.getElementById('ta-results');el.style.display='block';el.innerHTML='<span class="loading">Analyzing...</span>';
  const inds=Array.from(document.getElementById('ta-indicators').selectedOptions).map(o=>o.value);
  const r=await api('/api/oracle/technical-analysis',{symbol:document.getElementById('ta-symbol').value,indicators:inds});
  el.textContent=JSON.stringify(r,null,2);
}

 async function runMarket(){
   const el=document.getElementById('mkt-results');el.style.display='block';el.innerHTML='<span class="loading">Fetching...</span>';
   const r=await apiGet('/api/oracle/market-data?symbols='+encodeURIComponent(document.getElementById('mkt-symbols').value)+'&period='+document.getElementById('mkt-period').value);
   el.textContent=JSON.stringify(r,null,2);
 }
async function runScenario(){
  const r=await api('/api/oracle/scenario',{scenario_type:document.getElementById('sc-type').value,domain:document.getElementById('sc-domain').value});
  document.getElementById('sc-results').textContent=JSON.stringify(r,null,2);
}
async function runBacktest(){
  const r=await api('/api/oracle/backtest',{method:document.getElementById('bt-method').value,domain:'all'});
  document.getElementById('bt-results').textContent=JSON.stringify(r,null,2);
}
async function runESG(){
  const r=await api('/api/oracle/esg',{symbol:document.getElementById('esg-sym').value,scenario:'baseline'});
  document.getElementById('esg-results').textContent=JSON.stringify(r,null,2);
}
async function runYieldCurve(){
  const r=await api('/api/oracle/yield-curve',{method:document.getElementById('yc-method').value,scenario:'baseline'});
  document.getElementById('yc-results').textContent=JSON.stringify(r,null,2);
}
async function runRiskDecomp(){
  const r=await api('/api/oracle/risk-decomposition',{method:document.getElementById('rd-method').value});
  document.getElementById('rd-results').textContent=JSON.stringify(r,null,2);
}
async function runValuation(){
  const r=await api('/api/oracle/valuation',{symbol:document.getElementById('val-sym').value,method:'dcf'});
  document.getElementById('val-results').textContent=JSON.stringify(r,null,2);
}

checkStatus();
</script>
</body>
</html>"""
        """Setup mobile PWA authentication routes."""
        from fastapi.responses import StreamingResponse
        
        @self.app.get("/mobile", response_class=HTMLResponse)
        async def mobile_app():
            """Serve mobile PWA."""
            mobile_path = WEB_DIST / "mobile.html"
            if mobile_path.exists():
                return FileResponse(str(mobile_path))
            return HTMLResponse(content="<h1>Build the React app first: cd web && npm run build</h1>")
        
        @self.app.get("/mobile/{path:path}", response_class=HTMLResponse)
        async def mobile_app_path(path: str):
            """Serve mobile PWA paths."""
            mobile_path = WEB_DIST / "mobile.html"
            if mobile_path.exists():
                return FileResponse(str(mobile_path))
            return HTMLResponse(content="<h1>Build the React app first: cd web && npm run build</h1>")
        

    def _setup_mobile_api_routes(self):
        """Mobile REST endpoints: register challenge/verify, auth challenge/verify/password/validate/status/logout, chat, analyze-image."""
        @self.app.post("/api/mobile/register/challenge")
        async def mobile_register_challenge(request: Request):
            """Generate WebAuthn registration challenge."""
            try:
                from core.rate_limiter import get_rate_limiter, RateLimit
                client_host = request.client.host if request.client else "unknown"
                client_id = f"mobile_reg:{client_host}"
                limiter = get_rate_limiter()
                limiter.set_endpoint_limit("mobile_register", RateLimit(requests_per_minute=5, requests_per_hour=20, burst_size=3))
                allowed, retry_after, _ = limiter.check_rate_limit(client_id, "mobile_register")
                if not allowed:
                    return {"success": False, "error": f"Rate limited. Retry after {retry_after:.0f}s", "code": "rate_limit"}
                
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["device_id"], max_string_length=256
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                device_id = data["device_id"]
                
                from core.mobile_auth import get_mobile_auth
                hostname = self._mobile_hostname(request)
                auth = get_mobile_auth(hostname)
                challenge = await auth.generate_registration_challenge(device_id)
                return {"success": True, **challenge}
            except Exception as e:
                logger.error(f"Mobile register challenge error: {e}")
                return {"success": False, "error": safe_error_message(e)}
        
        @self.app.post("/api/mobile/register/verify")
        async def mobile_register_verify(request: Request):
            """Verify WebAuthn registration."""
            _audit_log(request, "mobile_register_verify", "WebAuthn registration verification attempt")
            try:
                from core.rate_limiter import get_rate_limiter, RateLimit
                client_host = request.client.host if request.client else "unknown"
                client_id = f"mobile_verify:{client_host}"
                limiter = get_rate_limiter()
                limiter.set_endpoint_limit("mobile_verify", RateLimit(requests_per_minute=5, requests_per_hour=20, burst_size=3))
                allowed, retry_after, _ = limiter.check_rate_limit(client_id, "mobile_verify")
                if not allowed:
                    return {"success": False, "error": f"Rate limited. Retry after {retry_after:.0f}s", "code": "rate_limit"}
                
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["challenge_id", "credential_id"],
                    optional=["client_data_json", "attestation_object"],
                    max_string_length=20000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                from core.mobile_auth import get_mobile_auth
                hostname = self._mobile_hostname(request)
                success, token_or_error = await get_mobile_auth(hostname).verify_registration(
                    challenge_id=data["challenge_id"],
                    credential_id=data["credential_id"],
                    client_data_json=data.get("client_data_json"),
                    attestation_object=data.get("attestation_object"),
                )
                if success:
                    return {"success": True, "token": token_or_error}
                return {"success": False, "error": token_or_error}
            except Exception as e:
                logger.error(f"Mobile register verify error: {e}")
                return {"success": False, "error": safe_error_message(e)}
        
        @self.app.post("/api/mobile/auth/challenge")
        async def mobile_auth_challenge(request: Request):
            """Generate WebAuthn authentication challenge."""
            try:
                from core.rate_limiter import get_rate_limiter, RateLimit
                client_host = request.client.host if request.client else "unknown"
                client_id = f"mobile_auth:{client_host}"
                limiter = get_rate_limiter()
                limiter.set_endpoint_limit("mobile_auth_challenge", RateLimit(requests_per_minute=10, requests_per_hour=50, burst_size=5))
                allowed, retry_after, _ = limiter.check_rate_limit(client_id, "mobile_auth_challenge")
                if not allowed:
                    return {"success": False, "error": f"Rate limited. Retry after {retry_after:.0f}s", "code": "rate_limit"}
                
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["device_id"], max_string_length=256
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                device_id = data["device_id"]
                
                from core.mobile_auth import get_mobile_auth
                hostname = self._mobile_hostname(request)
                auth = get_mobile_auth(hostname)
                challenge = await auth.generate_authentication_challenge(device_id)
                return {"success": True, **challenge}
            except Exception as e:
                logger.error(f"Mobile auth challenge error: {e}")
                return {"success": False, "error": safe_error_message(e)}
        
        @self.app.post("/api/mobile/auth/verify")
        async def mobile_auth_verify(request: Request):
            """Verify WebAuthn authentication."""
            _audit_log(request, "mobile_auth_verify", "WebAuthn authentication attempt")
            try:
                from core.rate_limiter import get_rate_limiter, RateLimit
                client_host = request.client.host if request.client else "unknown"
                client_id = f"mobile_verify:{client_host}"
                limiter = get_rate_limiter()
                limiter.set_endpoint_limit("mobile_auth_verify", RateLimit(requests_per_minute=10, requests_per_hour=50, burst_size=5))
                allowed, retry_after, _ = limiter.check_rate_limit(client_id, "mobile_auth_verify")
                if not allowed:
                    return {"success": False, "error": f"Rate limited. Retry after {retry_after:.0f}s", "code": "rate_limit"}
                
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["challenge_id", "credential_id"],
                    optional=["client_data_json", "authenticator_data", "signature", "user_handle"],
                    max_string_length=10000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                from core.mobile_auth import get_mobile_auth
                hostname = self._mobile_hostname(request)
                success, token_or_error = await get_mobile_auth(hostname).verify_authentication(
                    challenge_id=data["challenge_id"],
                    credential_id=data["credential_id"],
                    client_data_json=data.get("client_data_json"),
                    authenticator_data=data.get("authenticator_data"),
                    signature=data.get("signature"),
                    user_handle=data.get("user_handle"),
                )
                if success:
                    return {"success": True, "token": token_or_error}
                return {"success": False, "error": token_or_error}
            except Exception as e:
                logger.error(f"Mobile auth verify error: {e}")
                return {"success": False, "error": safe_error_message(e)}
        
        @self.app.post("/api/mobile/auth/password")
        async def mobile_auth_password(request: Request):
            """Login or register with password."""
            _audit_log(request, "mobile_auth_password", "Password authentication attempt")
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["device_id", "password"],
                    max_string_length=256
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                device_id = data["device_id"]
                password = data["password"]
                
                if len(password) < 12:
                    return {"success": False, "error": "Password must be at least 12 characters"}
                
                from core.mobile_auth import get_mobile_auth, MOBILE_AUTH_DIR
                from core.rate_limiter import get_rate_limiter, RateLimit
                import json
                import bcrypt
                from datetime import datetime, timedelta
                
                rate_limiter = get_rate_limiter()
                rate_limiter.set_endpoint_limit("mobile_auth", RateLimit(requests_per_minute=5, requests_per_hour=20, burst_size=3))
                client_id = f"mobile_auth:{request.client.host if request.client else 'unknown'}"
                
                allowed, retry_after, _ = rate_limiter.check_rate_limit(client_id, "mobile_auth")
                if not allowed:
                    return {"success": False, "error": f"Rate limited. Try again in {retry_after:.0f}s", "code": "rate_limit"}
                
                hostname = request.url.hostname or "localhost"
                auth = get_mobile_auth(hostname)
                
                password_file = MOBILE_AUTH_DIR / f"password_{device_id[:8]}.json"
                
                is_new = False
                if not password_file.exists():
                    # Check rate limit for new device registration (stricter)
                    allowed, retry_after, _ = rate_limiter.check_rate_limit(f"{client_id}:new", "mobile_auth")
                    if not allowed:
                        return {"success": False, "error": f"Registration rate limited. Try again in {retry_after:.0f}s", "code": "rate_limit"}
                    
                    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                    password_file.write_text(json.dumps({
                        "password_hash": password_hash,
                        "created_at": datetime.now().isoformat(),
                        "device_id": device_id
                    }), encoding='utf-8')
                    is_new = True
                    stored_hash = password_hash
                else:
                    stored = json.loads(password_file.read_text(encoding='utf-8'))
                    stored_hash = stored.get("password_hash", "")
                
                # Verify password with bcrypt
                if is_new:
                    password_valid = True
                else:
                    try:
                        password_valid = bcrypt.checkpw(password.encode(), stored_hash.encode())
                    except Exception:
                        password_valid = False
                
                if password_valid:
                    # Generate token and create session
                    token = auth._generate_jwt(device_id)
                    token_hash = auth._hash_token(token)
                    
                    async with auth._lock:
                        from core.mobile_auth import MobileSession
                        session = MobileSession(
                            token_hash=token_hash,
                            device_id=device_id,
                            created_at=datetime.now(),
                            expires_at=datetime.now() + timedelta(hours=168),
                            last_activity=datetime.now(),
                        )
                        auth._sessions[token_hash] = session
                        if device_id not in auth._device_tokens:
                            auth._device_tokens[device_id] = []
                        auth._device_tokens[device_id].append(token_hash)
                        auth._save_data()
                    
                    return {"success": True, "token": token, "new_device": is_new}
                
                return {"success": False, "error": "Invalid password"}
            except Exception as e:
                logger.error(f"Mobile password auth error: {e}")
                return {"success": False, "error": safe_error_message(e)}
        
        @self.app.post("/api/mobile/auth/validate")
        async def mobile_auth_validate(request: Request):
            """Validate JWT token."""
            try:
                data = await request.json()
                token = data.get("token")
                if not token:
                    return {"valid": False, "error": "token required"}
                
                from core.mobile_auth import get_mobile_auth
                hostname = self._mobile_hostname(request)
                auth = get_mobile_auth(hostname)
                valid, payload = await auth.validate_token(token)
                return {"valid": valid, "payload": payload}
            except Exception as e:
                return {"valid": False, "error": safe_error_message(e)}
        
        @self.app.post("/api/mobile/auth/status")
        async def mobile_auth_status(request: Request):
            """Check if device has WebAuthn credential."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["device_id"], max_string_length=256
                )
                if validation_error:
                    return {"has_credential": False, "error": validation_error}
                device_id = data["device_id"]
                
                from core.mobile_auth import get_mobile_auth
                hostname = self._mobile_hostname(request)
                auth = get_mobile_auth(hostname)
                return {"has_credential": auth.has_webauthn_credential(device_id)}
            except Exception as e:
                return {"has_credential": False, "error": safe_error_message(e)}
        
        @self.app.post("/api/mobile/auth/logout")
        async def mobile_auth_logout(request: Request):
            """Logout and invalidate token."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, optional=["token"], max_string_length=1000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                token = data.get("token")
                if token:
                    from core.mobile_auth import get_mobile_auth
                    hostname = request.url.hostname or "localhost"
                    await get_mobile_auth(hostname).logout(token)
                return {"success": True}
            except Exception as e:
                return {"success": False, "error": safe_error_message(e)}
        
        @self.app.post("/api/mobile/chat")
        async def mobile_chat(request: Request):
            """HTTP endpoint for mobile chat (fallback when WebSocket unavailable)."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["token"], optional=["message", "images"],
                    max_string_length=100000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                token = data["token"]
                message = data.get("message", "")
                images = data.get("images", [])
                
                if not message.strip() and not images:
                    return {"success": False, "error": "message or images required"}
                
                from core.mobile_auth import get_mobile_auth
                hostname = self._mobile_hostname(request)
                auth_manager = get_mobile_auth(hostname)
                valid, payload = await auth_manager.validate_token(token)
                if not valid:
                    return {"success": False, "error": "Invalid token"}
                
                # Track mobile HTTP client
                device_id = payload.get("device_id", "unknown") if payload else "unknown"
                self._mobile_http_last_activity[device_id] = time.time()
                self._mobile_http_clients = len([t for t in self._mobile_http_last_activity.values() if time.time() - t < 300])
                
                if images:
                    try:
                        from core.multimodal import get_multimodal_manager
                        import tempfile
                        import os
                        import base64
                        
                        multimodal = get_multimodal_manager(llm_client=self.agent.llm)
                        image_descriptions = []
                        
                        for idx, img_data in enumerate(images[:4]):
                            try:
                                if img_data.startswith("data:image"):
                                    img_base64 = img_data.split(",", 1)[1] if "," in img_data else img_data
                                else:
                                    img_base64 = img_data
                                
                                img_bytes = base64.b64decode(img_base64)
                                
                                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                                    f.write(img_bytes)
                                    temp_path = f.name
                                
                                try:
                                    description = await multimodal.vision.analyze_image(temp_path)
                                    image_descriptions.append(f"[Image {idx + 1}: {description}]")
                                finally:
                                    os.unlink(temp_path)
                            except Exception as e:
                                image_descriptions.append(f"[Image {idx + 1}: Unable to analyze - {safe_error_message(e)}]")
                        
                        if image_descriptions:
                            message = message + "\n\n" + "\n".join(image_descriptions) if message else "\n".join(image_descriptions)
                    except ImportError:
                        pass
                
                await self._emit_message("user", message)
                await self._emit_thinking_start()
                
                full_response = ""
                async for chunk in self.agent.run_step_stream(message):
                    if isinstance(chunk, str):
                        full_response += chunk
                        await self._emit_token(chunk)
                    elif isinstance(chunk, dict) and "content" in chunk:
                        full_response += chunk["content"]
                        await self._emit_token(chunk["content"])
                
                await self._emit_thinking_end()
                await self._emit_response_complete(full_response)
                
                return {"success": True, "response": full_response}
            except Exception as e:
                logger.error(f"Mobile chat error: {e}")
                return {"success": False, "error": safe_error_message(e)}
        
        @self.app.post("/api/mobile/analyze-image")
        async def mobile_analyze_image(request: Request):
            """Mobile image analysis endpoint."""
            try:
                data = await request.json()
                token = data.get("token")
                image_data = data.get("image", "")
                
                if not token:
                    return {"success": False, "error": "token required"}
                
                from core.mobile_auth import get_mobile_auth
                hostname = self._mobile_hostname(request)
                valid, _ = await get_mobile_auth(hostname).validate_token(token)
                if not valid:
                    return {"success": False, "error": "Invalid token"}
                
                if not image_data:
                    return {"success": False, "error": "image required"}
                
                from core.multimodal import get_multimodal_manager
                import tempfile
                import os
                import base64
                
                multimodal = get_multimodal_manager(llm_client=self.agent.llm)
                
                if image_data.startswith("data:image"):
                    image_data = image_data.split(",", 1)[1]
                
                img_bytes = base64.b64decode(image_data)
                
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    f.write(img_bytes)
                    temp_path = f.name
                
                try:
                    description = await multimodal.vision.analyze_image(temp_path)
                    return {"success": True, "description": description}
                finally:
                    os.unlink(temp_path)
            except ImportError:
                return {"success": False, "error": "Vision analysis requires a vision-capable model"}
            except Exception as e:
                logger.error(f"Mobile image analysis error: {e}")
                return {"success": False, "error": safe_error_message(e)}
        

    def _setup_mobile_ws_routes(self):
        """Mobile WebSocket endpoint (/api/mobile/ws) with token auth and dedup, plus the mobile HTTP send/poll and /api/ui/state live-sync endpoints."""
        from fastapi.responses import StreamingResponse

        @self.app.websocket("/api/mobile/ws")
        async def mobile_websocket(websocket: WebSocket):
            """Mobile WebSocket with token auth."""
            if not self._ws_origin_allowed(websocket):
                await websocket.accept()
                await websocket.send_json({"type": "error", "content": "Origin not allowed", "code": "origin_forbidden"})
                await websocket.close(code=4003, reason="Origin not allowed")
                return

            client_host = websocket.client.host if websocket.client else "unknown"
            client_id = f"mobile:{client_host}"
            
            from core.rate_limiter import get_rate_limiter, RateLimit
            limiter = get_rate_limiter()
            limiter.set_endpoint_limit("mobile_ws", RateLimit(requests_per_second=3.0, requests_per_minute=45, burst_size=8))
            allowed, retry_after, _ = limiter.check_rate_limit(client_id, "mobile_ws")
            if not allowed:
                await websocket.accept()
                await websocket.send_json({"type": "error", "content": "Rate limit exceeded"})
                await websocket.close(code=4008, reason="Rate limit exceeded")
                limiter.release(client_id)
                return

            await websocket.accept()

            token = None
            recent_msg_ids: set[str] = set()  # Server-side dedup per connection

            try:
                auth_msg = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=10.0
                )
                
                if auth_msg.get("type") != "auth":
                    await websocket.send_json({"type": "error", "content": "Authentication required"})
                    await websocket.close()
                    return
                
                token = auth_msg.get("token")
                from core.mobile_auth import get_mobile_auth
                hostname = websocket.client.host if websocket.client else "localhost"
                valid, _ = await get_mobile_auth(hostname).validate_token(token)
                
                if not valid:
                    await websocket.send_json({"type": "error", "content": "Invalid token"})
                    await websocket.close()
                    return
                
                MAX_WS_CONNECTIONS = 50
                if len(self._mobile_connections) >= MAX_WS_CONNECTIONS:
                    await websocket.send_json({"type": "error", "content": "Too many connections"})
                    await websocket.close(code=1013, reason="Too many connections")
                    return

                self._mobile_connections.add(websocket)
                
                await websocket.send_json({"type": "connected"})
                
                while True:
                    msg = await websocket.receive_json()
                    
                    if msg.get("type") == "message":
                        user_input = msg.get("content", "")
                        msg_id = msg.get("id", "unknown")
                        if not user_input.strip():
                            continue
                        if len(user_input) > 100000:
                            await websocket.send_json({"type": "error", "content": "Message too long (max 100000 chars)", "id": msg_id})
                            continue
                        
                        # Server-side deduplication
                        if msg_id in recent_msg_ids:
                            logger.warning(f"[MobileWS] Skipping duplicate message id={msg_id}")
                            continue
                        recent_msg_ids.add(msg_id)
                        if len(recent_msg_ids) > 50:
                            recent_msg_ids = set(list(recent_msg_ids)[-25:])
                        
                        logger.info(f"[MobileWS] Received message id={msg_id} ({len(user_input)} chars): {user_input[:50]}...")
                        
                        try:
                            await websocket.send_json({"type": "thinking", "status": "start"})
                            
                            full_response = ""
                            async for chunk in self.agent.run_step_stream(user_input):
                                if isinstance(chunk, str):
                                    full_response += chunk
                                    await websocket.send_json({
                                        "type": "response",
                                        "content": chunk,
                                        "id": msg_id,  # Include message ID for client dedup
                                    })
                            
                            await websocket.send_json({"type": "done", "id": msg_id})
                            logger.info(f"[MobileWS] Completed message id={msg_id} ({len(full_response)} chars)")
                        except Exception as e:
                            logger.error(f"[MobileWS] Stream error: {e}")
                            await websocket.send_json({"type": "error", "content": safe_error_message(e), "id": msg_id})
                            
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "error", "content": "Authentication timeout"})
            except WebSocketDisconnect:
                pass
            except Exception as e:
                logger.error(f"Mobile WebSocket error: {e}")
            finally:
                # Remove from mobile connections
                self._mobile_connections.discard(websocket)
                # Hand back the concurrency slot taken by check_rate_limit().
                # Early `return`s inside the try above still run this.
                try:
                    limiter.release(client_id)
                except Exception as e:
                    logger.debug(f"Rate limiter release failed for {client_id}: {e}")
        
        # HTTP endpoints for mobile clients (fallback when WebSocket blocked)
        @self.app.post("/api/mobile/send")
        async def mobile_send(request: Request):
            """Send message via HTTP (fallback when WebSocket unavailable)."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["token"], optional=["content", "images", "id"],
                    max_string_length=100000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                token = data["token"]
                content = data.get("content", "")
                data.get("images", [])
                msg_id = data.get("id")
                
                from core.mobile_auth import get_mobile_auth
                valid, _ = await get_mobile_auth("localhost").validate_token(token)
                
                if not valid:
                    return {"success": False, "error": "Invalid token"}
                
                if not content:
                    return {"success": False, "error": "Message content required"}
                
                # Stream response
                async def generate():
                    full_response = ""
                    async for chunk in self.agent.run_step_stream(content):
                        if isinstance(chunk, str):
                            full_response += chunk
                            yield f"data: {json.dumps({'type': 'response', 'content': chunk, 'id': msg_id})}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'id': msg_id})}\n\n"
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )
                
            except Exception as e:
                logger.error(f"Mobile HTTP send error: {e}")
                return {"success": False, "error": safe_error_message(e)}
        
        # HTTP polling fallback for clients without WebSocket support
        @self.app.post("/api/mobile/poll")
        async def mobile_poll(request: Request):
            """HTTP polling fallback for mobile clients."""
            try:
                data = await request.json()
                data, validation_error = validate_request_fields(
                    data, required=["token"], optional=["since"],
                    max_string_length=1000
                )
                if validation_error:
                    return {"success": False, "error": validation_error}
                token = data["token"]
                
                from core.mobile_auth import get_mobile_auth
                valid, _ = await get_mobile_auth("localhost").validate_token(token)
                
                if not valid:
                    return {"success": False, "error": "Invalid token"}
                
                # Return any pending messages for this client
                return {"success": True, "messages": []}
                
            except Exception as e:
                logger.error(f"Mobile poll error: {e}")
                return {"success": False, "error": safe_error_message(e)}
        
        @self.app.get("/api/ui/state")
        async def get_ui_state():
            """Get current UI state for live sync between web and mobile."""
            return {
                "thinking": self._ui_state.get("thinking", False),
                "tool_calls": self._ui_state.get("tool_calls", []),
                "last_response": self._ui_state.get("last_response"),
                "web_clients": len(self.connections),
                "mobile_websocket_clients": len(self._mobile_connections),
                "mobile_http_clients": self._mobile_http_clients,
                "connected_clients": len(self.connections) + len(self._mobile_connections) + self._mobile_http_clients,
            }
    

    async def run(self):
        await self.server.serve()

    def _setup_streams_routes(self):
        """Setup inter-module message stream endpoints for the Visual tab Streams view."""
        @self.app.get("/api/streams/messages")
        async def streams_messages(request: Request):
            try:
                from core.module_bridge import ModuleBridge
                from core.event_bus import get_event_bus

                bridge = ModuleBridge()
                event_bus = get_event_bus()

                limit = min(int(request.query_params.get("limit", "200")), 1000)
                source_filter = request.query_params.get("source", "")
                event_filter = request.query_params.get("event", "")

                bridge_history = bridge._event_history[-limit:]
                bridge_messages = []
                for msg in bridge_history:
                    entry = msg.to_dict() if hasattr(msg, 'to_dict') else {
                        "event": msg.event.value if hasattr(msg, 'event') else str(msg.event),
                        "source": msg.source if hasattr(msg, 'source') else "unknown",
                        "data": msg.data if hasattr(msg, 'data') else {},
                        "timestamp": msg.timestamp if hasattr(msg, 'timestamp') else 0,
                    }
                    if source_filter and entry.get("source", "") != source_filter:
                        continue
                    if event_filter and entry.get("event", "") != event_filter:
                        continue
                    bridge_messages.append(entry)

                bus_history = list(event_bus._history)[-limit:]
                bus_messages = []
                for evt in bus_history:
                    entry = evt.to_dict() if hasattr(evt, 'to_dict') else {
                        "channel": evt.channel if hasattr(evt, 'channel') else "",
                        "data": evt.data if hasattr(evt, 'data') else {},
                        "source": evt.source if hasattr(evt, 'source') else "",
                        "timestamp": evt.timestamp if hasattr(evt, 'timestamp') else 0,
                    }
                    if source_filter and entry.get("source", "") != source_filter:
                        continue
                    if event_filter and entry.get("channel", "") != event_filter:
                        continue
                    bus_messages.append(entry)

                all_messages = []
                seen_timestamps = set()
                for m in bridge_messages:
                    ts_key = (m.get("timestamp", 0), m.get("source", ""), m.get("event", "") or m.get("channel", ""))
                    if ts_key not in seen_timestamps:
                        seen_timestamps.add(ts_key)
                        m["_type"] = "bridge"
                        all_messages.append(m)
                for m in bus_messages:
                    ts_key = (m.get("timestamp", 0), m.get("source", ""), m.get("channel", ""))
                    if ts_key not in seen_timestamps:
                        seen_timestamps.add(ts_key)
                        m["_type"] = "event_bus"
                        all_messages.append(m)

                all_messages.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                all_messages = all_messages[:limit]

                sources = sorted(set(m.get("source", "unknown") for m in all_messages))
                events = sorted(set(m.get("event", "") or m.get("channel", "") for m in all_messages))

                return {
                    "success": True,
                    "total": len(all_messages),
                    "messages": all_messages,
                    "sources": sources,
                    "events": events,
                    "bridge_count": len(bridge_messages),
                    "bus_count": len(bus_messages),
                }
            except Exception as e:
                return {"success": False, "error": safe_error_message(e), "messages": [], "sources": [], "events": []}

        @self.app.get("/api/streams/content")
        async def streams_content(request: Request):
            try:
                limit = min(int(request.query_params.get("limit", "50")), 200)

                from core.agent import JarvisAgent
                agent = self.agent

                llm_calls = []
                if hasattr(agent, '_api_call_history'):
                    recent = agent._api_call_history[-limit:]
                    for call in recent:
                        llm_calls.append({
                            "timestamp": call.get("timestamp", 0),
                            "model": call.get("model", "unknown"),
                            "direction": call.get("direction", "request"),
                            "content_preview": call.get("content_preview", "")[:500],
                            "tokens_in": call.get("tokens_in", 0),
                            "tokens_out": call.get("tokens_out", 0),
                            "latency_ms": call.get("latency_ms", 0),
                        })

                tool_calls = []
                if hasattr(agent, '_tool_call_log'):
                    recent_tools = agent._tool_call_log[-limit:]
                    for tc in recent_tools:
                        tool_calls.append({
                            "timestamp": tc.get("timestamp", 0),
                            "tool": tc.get("tool", "unknown"),
                            "args_preview": str(tc.get("args", ""))[:300],
                            "result_preview": str(tc.get("result", ""))[:300],
                            "duration_ms": tc.get("duration_ms", 0),
                        })

                return {
                    "success": True,
                    "llm_calls": llm_calls,
                    "tool_calls": tool_calls,
                }
            except Exception as e:
                return {"success": True, "llm_calls": [], "tool_calls": []}

    def _setup_mobile_routes_safe(self):
        """Setup mobile-specific API routes (iPhone access)."""
        @self.app.get("/api/mobile/status")
        async def mobile_status(request: Request):
            return {
                "status": "available",
                "voice_ws": "/v1/realtime",
                "stt_ws": "/voice/transcribe/stream",
            }

        @self.app.post("/api/mobile/push-token")
        async def mobile_register_push(request: Request):
            try:
                data = await request.json()
                token = data.get("token", "")
                if token and len(token) < 512:
                    return {"status": "registered"}
                return {"status": "invalid_token"}
            except Exception:
                return {"status": "error"}


def run_web_interface(agent: "JarvisAgent", port: int = 8080, require_auth: bool = False, rate_limit_per_minute: int = 2000):
    """Run the web interface.
    
    Args:
        agent: JarvisAgent instance
        port: Port to listen on
        require_auth: If True, require API key authentication
        rate_limit_per_minute: Per-IP request cap enforced by SecurityMiddleware
    """
    interface = WebInterface(
        agent, port, require_auth=require_auth,
        rate_limit_per_minute=rate_limit_per_minute,
    )
    return interface.run()
