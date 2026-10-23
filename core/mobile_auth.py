import json
import uuid
import hashlib
from core.helpers import atomic_write_json, read_json_with_integrity
import base64
import secrets
import hmac
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
import logging
import asyncio

logger = logging.getLogger(__name__)

MOBILE_AUTH_DIR = Path.home() / ".jarvis" / "mobile_auth"
MOBILE_AUTH_DIR.mkdir(parents=True, exist_ok=True)

JWT_SECRET_FILE = MOBILE_AUTH_DIR / "jwt_secret"
TOKENS_FILE = MOBILE_AUTH_DIR / "tokens.json"
WEBAUTHN_CREDENTIALS_FILE = MOBILE_AUTH_DIR / "webauthn_credentials.json"

PWA_ICON_DIR = Path(__file__).parent.parent / "web" / "public" / "icons"


@dataclass
class MobileSession:
    """Mobile PWA session."""
    token_hash: str
    device_id: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    user_agent: str = ""
    ip_address: str = ""
    credential_id: Optional[str] = None  # WebAuthn credential


@dataclass
class WebAuthnCredential:
    """Stored WebAuthn credential for FaceID/TouchID."""
    credential_id: str
    public_key: str
    sign_count: int
    device_id: str
    created_at: datetime
    aaguid: str = ""
    transports: list = field(default_factory=list)


@dataclass
class WebAuthnChallenge:
    """Active WebAuthn challenge."""
    challenge: str
    device_id: str
    created_at: datetime
    expires_at: datetime


class MobileAuthManager:
    """Secure authentication manager for mobile PWA.
    
    Features:
    - JWT tokens for session management
    - WebAuthn/FaceID biometric authentication
    - Password fallback
    - Secure token storage
    - Device fingerprinting
    """
    
    TOKEN_EXPIRY_HOURS = 168  # 1 week
    CHALLENGE_EXPIRY_SECONDS = 60
    MAX_TOKENS_PER_DEVICE = 3
    MIN_PASSWORD_LENGTH = 12
    _LOGIN_WINDOW_SECONDS = 300  # 5 min
    _MAX_LOGIN_ATTEMPTS = 8
    _LOCKOUT_SECONDS = 900  # 15 min
    
    RP_NAME = "Jarvis AI Assistant"
    
    def __init__(self, hostname: str = "localhost"):
        self.hostname = hostname
        # For WebAuthn, rp_id must be a valid domain - IP addresses don't work
        # Tailscale provides domain like: your-host.your-tailnet.ts.net
        if hostname == "localhost" or hostname.startswith("127."):
            self.rp_id = "localhost"
            self.is_secure_context = True  # localhost is allowed for WebAuthn
        elif hostname.startswith("100.") or hostname.replace(".", "").isdigit():
            # Tailscale IP (100.x.x.x) or other IPs - not valid for WebAuthn
            # Fall back to localhost which IS valid for WebAuthn
            self.rp_id = "localhost"
            self.is_secure_context = True
        else:
            # Valid domain (e.g., your-host.your-tailnet.ts.net)
            self.rp_id = hostname.split(":")[0]  # Remove port if present
            self.is_secure_context = True
        self._jwt_secret: Optional[bytes] = None
        self._sessions: dict[str, MobileSession] = {}
        self._credentials: dict[str, WebAuthnCredential] = {}
        self._challenges: dict[str, WebAuthnChallenge] = {}
        self._device_tokens: dict[str, list[str]] = {}  # device_id -> [token_hashes]
        self._failed_login_attempts: dict[str, list[float]] = {}  # brute-force protection
        self._lockouts: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._sync_lock = threading.RLock()
        self._webauthn_challenge = secrets.token_bytes(32)
        
        self._init_jwt_secret()
        self._load_data()
        self._ensure_icons()
    
    def _init_jwt_secret(self):
        """Initialize or load JWT secret."""
        if JWT_SECRET_FILE.exists():
            self._jwt_secret = JWT_SECRET_FILE.read_bytes()
        else:
            self._jwt_secret = secrets.token_bytes(64)
            JWT_SECRET_FILE.write_bytes(self._jwt_secret)
            JWT_SECRET_FILE.chmod(0o600)
            logger.info("Generated new JWT secret")
    
    def _load_data(self):
        """Load persisted auth data."""
        with self._sync_lock:
            if TOKENS_FILE.exists():
                try:
                    data = read_json_with_integrity(TOKENS_FILE)
                    for token_hash, session_data in data.get("sessions", {}).items():
                        session_data["created_at"] = datetime.fromisoformat(session_data["created_at"])
                        session_data["expires_at"] = datetime.fromisoformat(session_data["expires_at"])
                        session_data["last_activity"] = datetime.fromisoformat(session_data["last_activity"])
                        self._sessions[token_hash] = MobileSession(**session_data)
                    logger.info(f"Loaded {len(self._sessions)} mobile sessions")
                except Exception as e:
                    logger.error(f"Failed to load tokens: {e}")

            if WEBAUTHN_CREDENTIALS_FILE.exists():
                try:
                    data = read_json_with_integrity(WEBAUTHN_CREDENTIALS_FILE)
                    for cred_id, cred_data in data.items():
                        cred_data["created_at"] = datetime.fromisoformat(cred_data["created_at"])
                        self._credentials[cred_id] = WebAuthnCredential(**cred_data)
                    logger.info(f"Loaded {len(self._credentials)} WebAuthn credentials")
                except Exception as e:
                    logger.error(f"Failed to load WebAuthn credentials: {e}")
    
    def _save_data(self):
        """Persist auth data."""
        try:
            with self._sync_lock:
                sessions_data = {}
                for token_hash, session in self._sessions.items():
                    if datetime.now() < session.expires_at:
                        sessions_data[token_hash] = {
                            **asdict(session),
                            "created_at": session.created_at.isoformat(),
                            "expires_at": session.expires_at.isoformat(),
                            "last_activity": session.last_activity.isoformat(),
                        }

                creds_data = {}
                for cred_id, cred in self._credentials.items():
                    creds_data[cred_id] = {
                        **asdict(cred),
                        "created_at": cred.created_at.isoformat(),
                    }

            atomic_write_json(TOKENS_FILE, {
                "sessions": sessions_data,
                "updated": datetime.now().isoformat(),
            })
            TOKENS_FILE.chmod(0o600)

            atomic_write_json(WEBAUTHN_CREDENTIALS_FILE, creds_data)
            WEBAUTHN_CREDENTIALS_FILE.chmod(0o600)
        except Exception as e:
            logger.error(f"Failed to save auth data: {e}")
    
    def _ensure_icons(self):
        """Generate PWA icons if they don't exist."""
        PWA_ICON_DIR.mkdir(parents=True, exist_ok=True)
        
        icon_sizes = [72, 96, 128, 144, 152, 192, 384, 512]
        
        for size in icon_sizes:
            icon_path = PWA_ICON_DIR / f"icon-{size}.png"
            if not icon_path.exists():
                self._generate_icon(size, icon_path)
    
    def _generate_icon(self, size: int, path: Path):
        """Generate a simple SVG-based icon."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.debug("PIL not available, creating placeholder icon")
            path.write_bytes(b"")
            return
        
        img = Image.new('RGBA', (size, size), (26, 26, 46, 255))
        draw = ImageDraw.Draw(img)
        
        center = size // 2
        radius = int(size * 0.35)
        
        draw.ellipse(
            [center - radius, center - radius, center + radius, center + radius],
            fill=(88, 166, 255, 255),
            outline=(100, 200, 255, 255),
            width=max(2, size // 64)
        )
        
        font_size = int(size * 0.35)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        text = "J"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.text(
            (center - text_width // 2, center - text_height // 2 - font_size // 10),
            text,
            fill=(255, 255, 255, 255),
            font=font
        )
        
        img.save(path, 'PNG')
        logger.debug(f"Generated icon {path.name}")
    
    def _hash_token(self, token: str) -> str:
        """Hash a token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()
    
    def _generate_jwt(self, device_id: str, credential_id: Optional[str] = None) -> str:
        """Generate a JWT token."""
        now = datetime.now()
        expires = now + timedelta(hours=self.TOKEN_EXPIRY_HOURS)
        
        header = base64.urlsafe_b64encode(json.dumps({
            "alg": "HS256",
            "typ": "JWT"
        }).encode()).decode().rstrip('=')
        
        payload = base64.urlsafe_b64encode(json.dumps({
            "device_id": device_id,
            "iat": int(now.timestamp()),
            "exp": int(expires.timestamp()),
            "jti": str(uuid.uuid4()),
            "credential_id": credential_id,
        }).encode()).decode().rstrip('=')
        
        message = f"{header}.{payload}"
        if self._jwt_secret is None:
            raise ValueError("JWT secret not initialized")
        signature = base64.urlsafe_b64encode(
            hmac.new(self._jwt_secret, message.encode(), hashlib.sha256).digest()
        ).decode().rstrip('=')
        
        return f"{message}.{signature}"
    
    def _verify_jwt(self, token: str) -> Optional[dict]:
        """Verify and decode a JWT token."""
        try:
            parts = token.split('.')
            if len(parts) != 3:
                return None
            
            header, payload, signature = parts
            
            if self._jwt_secret is None:
                return None
            
            expected_sig = base64.urlsafe_b64encode(
                hmac.new(self._jwt_secret, f"{header}.{payload}".encode(), hashlib.sha256).digest()
            ).decode().rstrip('=')
            
            if not hmac.compare_digest(signature, expected_sig):
                return None
            
            while len(payload) % 4:
                payload += '='
            
            decoded = json.loads(base64.urlsafe_b64decode(payload))
            
            if datetime.now().timestamp() > decoded.get("exp", 0):
                return None
            
            return decoded
        except Exception as e:
            logger.debug(f"JWT verification failed: {e}")
            return None
    
    async def generate_registration_challenge(self, device_id: str) -> dict:
        """Generate WebAuthn registration challenge for FaceID setup."""
        challenge = secrets.token_urlsafe(32)
        challenge_id = str(uuid.uuid4())
        
        self._challenges[challenge_id] = WebAuthnChallenge(
            challenge=challenge,
            device_id=device_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.CHALLENGE_EXPIRY_SECONDS),
        )
        
        result = {
            "challenge_id": challenge_id,
            "challenge": base64.urlsafe_b64encode(challenge.encode()).decode().rstrip('='),
            "rp": {
                "id": self.rp_id,
                "name": self.RP_NAME,
            },
            "user": {
                "id": base64.urlsafe_b64encode(device_id.encode()).decode().rstrip('='),
                "name": f"jarvis-{device_id[:8]}",
                "displayName": "Jarvis User",
            },
            "pubKeyCredParams": [
                {"type": "public-key", "alg": -7},   # ES256
                {"type": "public-key", "alg": -257}, # RS256
            ],
            "authenticatorSelection": {
                "authenticatorAttachment": "platform",
                "userVerification": "required",
                "residentKey": "preferred",
            },
            "timeout": self.CHALLENGE_EXPIRY_SECONDS * 1000,
            "attestation": "direct",
            "is_secure_context": self.is_secure_context,
        }
        
        # Warn if not secure context
        if not self.is_secure_context:
            logger.warning(f"WebAuthn may not work on non-HTTPS connection (hostname={self.hostname}, rp_id={self.rp_id})")
        
        return result
    
    async def verify_registration(
        self,
        challenge_id: str,
        credential_id: str,
        client_data_json: str,
        attestation_object: str,
    ) -> tuple[bool, Optional[str]]:
        """Verify WebAuthn registration response."""
        async with self._lock:
            challenge = self._challenges.get(challenge_id)
            if not challenge:
                return False, "Challenge not found"
            
            if datetime.now() > challenge.expires_at:
                del self._challenges[challenge_id]
                return False, "Challenge expired"
            
            try:
                client_data = json.loads(base64.urlsafe_b64decode(client_data_json + '=='))
                
                if client_data.get("type") != "webauthn.create":
                    return False, "Invalid client data type"
                
                client_origin = client_data.get("origin", "")
                if client_origin and self.rp_id:
                    from urllib.parse import urlparse
                    client_host = urlparse(client_origin).hostname or ""
                    expected_host = urlparse(self.rp_id).hostname if "://" in self.rp_id else self.rp_id
                    if client_host and expected_host and client_host != expected_host:
                        logger.warning(f"WebAuthn registration origin mismatch: {client_origin} != {self.rp_id}")
                        return False, "Origin mismatch"
                
                challenge_b64 = base64.urlsafe_b64encode(challenge.challenge.encode()).decode().rstrip('=')
                if client_data.get("challenge") != challenge_b64:
                    return False, "Challenge mismatch"
                
                attestation_data = base64.urlsafe_b64decode(attestation_object + '==')
                
                cred = WebAuthnCredential(
                    credential_id=credential_id,
                    public_key=base64.urlsafe_b64encode(attestation_data[:64]).decode(),
                    sign_count=0,
                    device_id=challenge.device_id,
                    created_at=datetime.now(),
                )
                
                self._credentials[credential_id] = cred
                
                token = self._generate_jwt(challenge.device_id, credential_id)
                token_hash = self._hash_token(token)
                
                session = MobileSession(
                    token_hash=token_hash,
                    device_id=challenge.device_id,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(hours=self.TOKEN_EXPIRY_HOURS),
                    last_activity=datetime.now(),
                    credential_id=credential_id,
                )
                
                self._sessions[token_hash] = session
                
                if challenge.device_id not in self._device_tokens:
                    self._device_tokens[challenge.device_id] = []
                self._device_tokens[challenge.device_id].append(token_hash)
                
                del self._challenges[challenge_id]
                self._save_data()
                
                logger.info(f"WebAuthn registration successful for device {challenge.device_id[:8]}")
                return True, token
                
            except Exception as e:
                logger.error(f"WebAuthn registration verification failed: {e}")
                return False, "Registration verification failed"
    
    async def generate_authentication_challenge(self, device_id: str) -> dict:
        """Generate WebAuthn authentication challenge."""
        challenge = secrets.token_urlsafe(32)
        challenge_id = str(uuid.uuid4())
        
        self._challenges[challenge_id] = WebAuthnChallenge(
            challenge=challenge,
            device_id=device_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.CHALLENGE_EXPIRY_SECONDS),
        )
        
        allow_credentials = []
        for cred in self._credentials.values():
            if cred.device_id == device_id:
                allow_credentials.append({
                    "type": "public-key",
                    "id": cred.credential_id,
                })
        
        return {
            "challenge_id": challenge_id,
            "challenge": base64.urlsafe_b64encode(challenge.encode()).decode().rstrip('='),
            "allowCredentials": allow_credentials,
            "userVerification": "required",
            "timeout": self.CHALLENGE_EXPIRY_SECONDS * 1000,
            "rpId": self.rp_id,
        }
    
    async def verify_authentication(
        self,
        challenge_id: str,
        credential_id: str,
        client_data_json: str,
        authenticator_data: str,
        signature: str,
        user_handle: Optional[str] = None,
    ) -> tuple[bool, Optional[str]]:
        """Verify WebAuthn authentication response (FaceID login)."""
        async with self._lock:
            challenge = self._challenges.get(challenge_id)
            if not challenge:
                return False, "Challenge not found"
            
            if datetime.now() > challenge.expires_at:
                del self._challenges[challenge_id]
                return False, "Challenge expired"
            
            cred = self._credentials.get(credential_id)
            if not cred:
                return False, "Unknown credential"
            
            if cred.device_id != challenge.device_id:
                return False, "Credential device mismatch"
            
            try:
                client_data = json.loads(base64.urlsafe_b64decode(client_data_json + '=='))
                
                if client_data.get("type") != "webauthn.get":
                    return False, "Invalid client data type"
                
                client_origin = client_data.get("origin", "")
                if client_origin and self.rp_id:
                    from urllib.parse import urlparse
                    client_host = urlparse(client_origin).hostname or ""
                    expected_host = urlparse(self.rp_id).hostname if "://" in self.rp_id else self.rp_id
                    if client_host and expected_host and client_host != expected_host:
                        logger.warning(f"WebAuthn origin mismatch: {client_origin} != {self.rp_id}")
                        return False, "Origin mismatch"
                
                if not signature or not authenticator_data:
                    return False, "Missing signature or authenticator data"
                
                try:
                    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
                    from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicKey
                    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
                    from cryptography.hazmat.primitives.asymmetric import padding
                    from cryptography.hazmat.primitives import hashes
                    
                    pub_key_bytes = base64.urlsafe_b64decode(cred.public_key + '==')
                    
                    auth_data_bytes = base64.urlsafe_b64decode(authenticator_data + '==')
                    sig_bytes = base64.urlsafe_b64decode(signature + '==')
                    
                    client_data_hash = hashlib.sha256(client_data_json.encode()).digest()
                    
                    verification_data = auth_data_bytes + client_data_hash
                    
                    try:
                        pub_key = Ed25519PublicKey.from_public_bytes(pub_key_bytes)
                        pub_key.verify(sig_bytes, verification_data)
                        logger.info("WebAuthn Ed25519 signature verified successfully")
                    except Exception as ed25519_err:
                        logger.debug(f"Ed25519 verification failed, trying ECDSA: {ed25519_err}")
                        try:
                            from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicKey, ECDSA, SECP256R1
                            from cryptography.hazmat.primitives.serialization import load_der_public_key
                            pub_key_der = load_der_public_key(pub_key_bytes)
                            pub_key_der.verify(sig_bytes, verification_data, ECDSA(hashes.SHA256()))
                            logger.info("WebAuthn ECDSA signature verified successfully")
                        except Exception as ecdsa_err:
                            logger.warning(f"WebAuthn signature verification FAILED: Ed25519={ed25519_err}, ECDSA={ecdsa_err}")
                            return False, "Signature verification failed"
                    
                except ImportError:
                    logger.error("cryptography library not available - CANNOT verify WebAuthn signature. Authentication DENIED.")
                    return False, "Signature verification unavailable"
                except Exception as sig_err:
                    logger.error(f"WebAuthn signature verification error: {sig_err}")
                    return False, "Signature verification error"
                
                if len(auth_data_bytes) >= 37:
                    new_sign_count = int.from_bytes(auth_data_bytes[33:37], 'big')
                    if new_sign_count <= cred.sign_count and cred.sign_count > 0:
                        logger.warning(f"WebAuthn sign_count replay detected: got {new_sign_count}, expected > {cred.sign_count}")
                        return False, "Possible authenticator clone detected"
                    cred.sign_count = new_sign_count
                
                token = self._generate_jwt(challenge.device_id, credential_id)
                token_hash = self._hash_token(token)
                
                session = MobileSession(
                    token_hash=token_hash,
                    device_id=challenge.device_id,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(hours=self.TOKEN_EXPIRY_HOURS),
                    last_activity=datetime.now(),
                    credential_id=credential_id,
                )
                
                self._sessions[token_hash] = session
                
                if challenge.device_id not in self._device_tokens:
                    self._device_tokens[challenge.device_id] = []
                self._device_tokens[challenge.device_id].append(token_hash)
                
                while len(self._device_tokens[challenge.device_id]) > self.MAX_TOKENS_PER_DEVICE:
                    old_hash = self._device_tokens[challenge.device_id].pop(0)
                    if old_hash in self._sessions:
                        del self._sessions[old_hash]
                
                del self._challenges[challenge_id]
                self._save_data()
                
                logger.info(f"WebAuthn authentication successful for device {challenge.device_id[:8]}")
                return True, token
                
            except Exception as e:
                logger.error(f"WebAuthn authentication verification failed: {e}")
                return False, "Authentication verification failed"
    
    async def login_with_password(self, device_id: str, password: str, stored_password_hash: str) -> tuple[bool, Optional[str]]:
        """Login with password (fallback if FaceID unavailable).
        
        The stored_password_hash must be a bcrypt hash generated by
        the web interface registration endpoint.
        """
        async with self._lock:
            now = time.time()
            attempts = self._failed_login_attempts.get(device_id, [])
            attempts = [t for t in attempts if now - t < self._LOGIN_WINDOW_SECONDS]
            self._failed_login_attempts[device_id] = attempts
            
            if len(attempts) >= self._MAX_LOGIN_ATTEMPTS:
                return False, "Too many attempts — try again later"
            
            lockout_key = f"lockout:{device_id}"
            if lockout_key in self._lockouts:
                if now < self._lockouts[lockout_key]:
                    return False, "Account temporarily locked"
                del self._lockouts[lockout_key]
        
        try:
            import bcrypt as _bcrypt
            if isinstance(stored_password_hash, str):
                stored_password_hash = stored_password_hash.encode('utf-8')
            if _bcrypt.checkpw(password.encode('utf-8'), stored_password_hash):
                pass
            else:
                async with self._lock:
                    self._failed_login_attempts[device_id].append(now)
                return False, "Invalid password"
        except ImportError:
            logger.error("bcrypt not available - password login DENIED")
            return False, "Password authentication unavailable"
        except Exception:
            async with self._lock:
                self._failed_login_attempts[device_id].append(now)
            return False, "Invalid password"
        
        async with self._lock:
            self._failed_login_attempts.pop(device_id, None)
        
        token = self._generate_jwt(device_id)
        token_hash = self._hash_token(token)
        
        async with self._lock:
            session = MobileSession(
                token_hash=token_hash,
                device_id=device_id,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=self.TOKEN_EXPIRY_HOURS),
                last_activity=datetime.now(),
            )
            
            self._sessions[token_hash] = session
            
            if device_id not in self._device_tokens:
                self._device_tokens[device_id] = []
            self._device_tokens[device_id].append(token_hash)
            
            while len(self._device_tokens[device_id]) > self.MAX_TOKENS_PER_DEVICE:
                old_hash = self._device_tokens[device_id].pop(0)
                if old_hash in self._sessions:
                    del self._sessions[old_hash]
            
            self._save_data()
        
        logger.info(f"Password login successful for device {device_id[:8]}")
        return True, token
    
    async def validate_token(self, token: str) -> tuple[bool, Optional[dict]]:
        """Validate a JWT token and update activity."""
        payload = self._verify_jwt(token)
        if not payload:
            return False, None
        
        token_hash = self._hash_token(token)
        
        async with self._lock:
            session = self._sessions.get(token_hash)
            if not session:
                return False, None
            
            if datetime.now() > session.expires_at:
                del self._sessions[token_hash]
                return False, None
            
            session.last_activity = datetime.now()
            self._save_data()
        
        return True, payload
    
    async def logout(self, token: str) -> bool:
        """Invalidate a token."""
        token_hash = self._hash_token(token)
        
        async with self._lock:
            if token_hash in self._sessions:
                session = self._sessions[token_hash]
                if session.device_id in self._device_tokens:
                    try:
                        self._device_tokens[session.device_id].remove(token_hash)
                    except ValueError:
                        pass
                del self._sessions[token_hash]
                self._save_data()
                return True
        
        return False
    
    def has_webauthn_credential(self, device_id: str) -> bool:
        """Check if device has WebAuthn credential registered."""
        with self._sync_lock:
            for cred in self._credentials.values():
                if cred.device_id == device_id:
                    return True
        return False
    
    def get_active_sessions_count(self) -> int:
        """Get count of active (non-expired) sessions."""
        with self._sync_lock:
            now = datetime.now()
            return sum(1 for s in self._sessions.values() if s.expires_at > now)
    
    def cleanup_expired(self):
        """Remove expired sessions and challenges."""
        with self._sync_lock:
            now = datetime.now()

            for token_hash, session in list(self._sessions.items()):
                if session.expires_at < now:
                    del self._sessions[token_hash]

            for challenge_id, challenge in list(self._challenges.items()):
                if challenge.expires_at < now:
                    del self._challenges[challenge_id]

        self._save_data()


_mobile_auth_by_host: dict[str, MobileAuthManager] = {}
_mobile_auth_lock = threading.Lock()


def get_mobile_auth(hostname: str = "localhost") -> MobileAuthManager:
    """Get mobile auth manager for a specific hostname."""
    with _mobile_auth_lock:
        if hostname not in _mobile_auth_by_host:
            _mobile_auth_by_host[hostname] = MobileAuthManager(hostname)
        return _mobile_auth_by_host[hostname]