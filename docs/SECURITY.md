# Jarvis Security Implementation

## Overview

Jarvis now includes comprehensive security middleware to protect against:
- Unauthorized access
- Injection attacks (XSS, SQL, command, path traversal)
- Data extraction
- Rate limiting bypass
- Session hijacking

## Components

### 1. Authentication Middleware (`core/auth_middleware.py`)

| Component | Description |
|-----------|-------------|
| `SecurityConfig` | Configuration class for all security settings |
| `InputSanitizer` | Pattern-based sanitization for injection attacks |
| `APIKeyValidator` | Timing-safe API key validation |
| `SessionManager` | Session lifecycle with lockout protection |
| `SecurityMiddleware` | Coordinator for all security features |

### 2. Input Sanitization Patterns

**XSS Protection:**
- `<script>`, `<iframe>`, `<object>`, `<embed>`, `<form>` tags
- `javascript:` URLs
- `onerror`, `onload` event handlers
- `<img>` and `<svg>` with event handlers

**SQL Injection Protection:**
- `OR/AND '1'='1'` patterns
- `UNION SELECT` attacks
- Comment injection (`--`, `/* */`)
- `DROP/DELETE/INSERT` with semicolons
- `BENCHMARK`, `SLEEP`, `WAITFOR` time-based attacks

**Command Injection Protection:**
- Shell metacharacters: `;`, `&`, `|`, `$`, `` ` ``
- Command substitution: `$(...)`, `` `...` ``
- Dangerous commands: `rm -rf`, `chmod`, `chown`, `nc`, `curl`, `wget`

**Path Traversal Protection:**
- `../` sequences
- URL-encoded variants: `%2e%2e%2f`, `%2e%2e/`
- Double-encoded: `%252e%252e`

### 3. Security Headers

All responses include:
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
Content-Security-Policy: default-src 'self'; ...
```

### 4. Rate Limiting

- Per-client WebSocket rate limiting
- 120 requests per minute (configurable)
- Automatic cleanup of stale rate limit entries

### 5. Session Management

- Secure session tokens with HMAC signatures
- Configurable timeout (default 60 minutes)
- Failed attempt tracking with lockout
- Maximum lockout duration (default 15 minutes)

## Configuration

Add to `config.yaml`:

```yaml
security:
  require_auth: false  # Set to true to require API key
  api_key: null        # Or set JARVIS_API_KEY environment variable
  session_timeout_minutes: 60
  max_failed_attempts: 5
  lockout_duration_minutes: 15
  rate_limit_requests_per_minute: 120
  max_request_size_mb: 10
  xss_protection: true
  csrf_protection: true
```

## Usage

### Enable Authentication

```bash
# Set environment variable
export JARVIS_API_KEY="your-secure-random-key-here"

# Or in config.yaml
security:
  require_auth: true
  api_key: "your-secure-key"
```

### Client Authentication

**REST API:**
```bash
curl -H "X-API-Key: your-key" http://localhost:8080/api/conversations
curl -H "Authorization: Bearer your-key" http://localhost:8080/api/chat
```

**WebSocket:**
```javascript
// Via query parameter
const ws = new WebSocket('ws://localhost:8080/ws?api_key=your-key');

// Or after connection (not recommended)
ws.send(JSON.stringify({type: 'auth', api_key: 'your-key'}));
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/security/status` | View current security configuration |

## Integration Points

### Web Interface
- Security middleware in `interfaces/web_react.py`
- All REST endpoints protected when `require_auth=true`
- All WebSocket messages sanitized
- Rate limiting on all non-ping messages

### LLM Client
- Circuit breaker integration in `core/llm.py`
- Success/failure tracking for model availability
- Automatic failover on repeated failures

### Agent Loop
- Input sanitization for user messages
- Tool result sanitization
- Thinking content filtering

## Testing

Run security tests:
```bash
python -c "
from core.auth_middleware import SecurityMiddleware, SecurityConfig, InputSanitizer

sanitizer = InputSanitizer()
test = '<script>alert(1)</script>hello'
print(f'Sanitized: {sanitizer.sanitize_string(test)}')
# Output: Sanitized: hello
"
```