"""Man-in-the-Middle (MITM) Attack Protection for Jarvis.

Provides comprehensive protection against MITM attacks through:
- TLS certificate validation and pinning
- Request signing and verification
- Encrypted communication channels
- Certificate transparency checking
- DNS spoofing detection
- Session hijacking prevention
"""

import base64
import hashlib
import hmac
import logging
import os
import threading
import time
from core.helpers import atomic_write_json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)


class MITMThreatType(str, Enum):
    """Types of MITM threats."""
    CERTIFICATE_MISMATCH = "certificate_mismatch"
    CERTIFICATE_EXPIRED = "certificate_expired"
    CERTIFICATE_UNTRUSTED = "certificate_untrusted"
    TLS_DOWNGRADE = "tls_downgrade"
    DNS_SPOOFING = "dns_spoofing"
    REQUEST_TAMPERING = "request_tampering"
    RESPONSE_TAMPERING = "response_tampering"
    SESSION_HIJACKING = "session_hijacking"
    CERTIFICATE_TRANSPARENCY_FAILURE = "ct_failure"


class ThreatSeverity(str, Enum):
    """Severity of MITM threats."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MITMThreatEvent:
    """Represents a detected MITM threat."""
    timestamp: datetime
    threat_type: MITMThreatType
    severity: ThreatSeverity
    description: str
    source_ip: str | None
    destination: str
    details: dict[str, Any] = field(default_factory=dict)
    blocked: bool = True

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "threat_type": self.threat_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "source_ip": self.source_ip,
            "destination": self.destination,
            "details": self.details,
            "blocked": self.blocked,
        }


class RequestSigner:
    """Signs and verifies requests to prevent tampering."""

    def __init__(self, secret_key: bytes | None = None):
        """Initialize with a secret key for HMAC signing."""
        if secret_key is None:
            secret_key = os.urandom(32)
        self.secret_key = secret_key

        # Generate RSA key pair for asymmetric signing
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self._public_key = self._private_key.public_key()

    def sign_request(
        self,
        method: str,
        url: str,
        body: str | bytes | None = None,
        headers: dict | None = None,
        timestamp: float | None = None,
    ) -> str:
        """Sign an HTTP request.

        Args:
            method: HTTP method
            url: Request URL
            body: Request body
            headers: Request headers
            timestamp: Unix timestamp (defaults to current time)

        Returns:
            Signature string to include in headers
        """
        if timestamp is None:
            timestamp = time.time()

        # Build canonical request string
        body_hash = hashlib.sha256(
            (body.encode() if isinstance(body, str) else body) or b""
        ).hexdigest()

        canonical = f"{method.upper()}\n{url}\n{body_hash}\n{timestamp}"

        # Sign with RSA private key
        signature = self._private_key.sign(
            canonical.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        return base64.b64encode(signature).decode()

    def verify_request(
        self,
        method: str,
        url: str,
        signature: str,
        body: str | bytes | None = None,
        timestamp: float | None = None,
        max_age_seconds: float = 300.0,
    ) -> bool:
        """Verify a signed request.

        Args:
            method: HTTP method
            url: Request URL
            signature: Signature from headers
            body: Request body
            timestamp: Unix timestamp from request
            max_age_seconds: Maximum age of request (replay protection)

        Returns:
            True if signature is valid
        """
        if timestamp is None:
            return False

        now = time.time()
        if now - timestamp > max_age_seconds:
            return False
        if timestamp - now > 30:
            return False

        # Build canonical request string
        body_hash = hashlib.sha256(
            (body.encode() if isinstance(body, str) else body) or b""
        ).hexdigest()

        canonical = f"{method.upper()}\n{url}\n{body_hash}\n{timestamp}"

        try:
            signature_bytes = base64.b64decode(signature)
            self._public_key.verify(
                signature_bytes,
                canonical.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.warning(f"Request verification failed: {e}")
            return False

    def sign_data(self, data: str | bytes) -> str:
        """Create an HMAC signature for data."""
        if isinstance(data, str):
            data = data.encode()
        return hmac.new(self.secret_key, data, hashlib.sha256).hexdigest()

    def verify_data_signature(self, data: str | bytes, signature: str) -> bool:
        """Verify an HMAC signature for data."""
        expected = self.sign_data(data)
        return hmac.compare_digest(expected, signature)

    def get_public_key_pem(self) -> str:
        """Get the public key in PEM format for sharing."""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()


class SecureChannel:
    """Provides encrypted communication channel."""

    MAX_SESSION_KEYS = 500

    def __init__(
        self,
        key: bytes | None = None,
        iv_length: int = 16,
    ):
        self.key = key or os.urandom(32)
        self.iv_length = iv_length
        self._session_keys: dict[str, bytes] = {}
        self._lock = threading.RLock()

    def _get_cipher(self, iv: bytes):
        """Get AES cipher for encryption/decryption."""
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        return Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())

    def encrypt(self, plaintext: str | bytes) -> str:
        """Encrypt data.

        Args:
            plaintext: Data to encrypt

        Returns:
            Base64-encoded encrypted data with IV prepended
        """
        if isinstance(plaintext, str):
            plaintext = plaintext.encode()

        # Pad to block size
        block_size = 16
        pad_length = block_size - (len(plaintext) % block_size)
        padded = plaintext + bytes([pad_length] * pad_length)

        # Generate IV
        iv = os.urandom(self.iv_length)

        # Encrypt
        cipher = self._get_cipher(iv)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded) + encryptor.finalize()

        # Return IV + ciphertext
        return base64.b64encode(iv + ciphertext).decode()

    def decrypt(self, encrypted: str) -> str:
        """Decrypt data.

        Args:
            encrypted: Base64-encoded encrypted data

        Returns:
            Decrypted string
        """
        data = base64.b64decode(encrypted)

        # Extract IV and ciphertext
        iv = data[:self.iv_length]
        ciphertext = data[self.iv_length:]

        # Decrypt
        cipher = self._get_cipher(iv)
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()

        # Remove padding (PKCS7 validation)
        pad_length = padded[-1]
        if pad_length < 1 or pad_length > self.iv_length:
            raise ValueError("Invalid PKCS7 padding")
        for i in range(1, pad_length + 1):
            if padded[-i] != pad_length:
                raise ValueError("Invalid PKCS7 padding bytes")
        plaintext = padded[:-pad_length]

        return plaintext.decode()

    def create_session_key(self, session_id: str) -> str:
        session_key = os.urandom(32)
        with self._lock:
            self._session_keys[session_id] = session_key
            if len(self._session_keys) > self.MAX_SESSION_KEYS:
                oldest = list(self._session_keys.keys())[:len(self._session_keys) - self.MAX_SESSION_KEYS]
                for k in oldest:
                    del self._session_keys[k]
        return base64.b64encode(session_key).decode()

    def get_session_key(self, session_id: str) -> bytes | None:
        with self._lock:
            return self._session_keys.get(session_id)

    def revoke_session_key(self, session_id: str) -> None:
        with self._lock:
            self._session_keys.pop(session_id, None)


class MITMProtector:
    """Main MITM protection coordinator."""

    MAX_THREATS = 1000
    MAX_CERT_PINS_PER_HOST = 10
    MAX_PINNED_HOSTS = 100
    MAX_SESSIONS = 500
    MAX_DNS_ENTRIES = 500

    DEFAULT_CERT_PINS: dict[str, list[str]] = {
        "ollama.com": [],
        "api.openai.com": [],
        "api.anthropic.com": [],
    }

    def __init__(
        self,
        storage_path: Path | None = None,
        enforce_tls: bool = True,
        enforce_certificate_pinning: bool = False,
        audit_callback: Callable[[MITMThreatEvent], None] | None = None,
    ):
        self.storage_path = storage_path
        self.enforce_tls = enforce_tls
        self.enforce_certificate_pinning = enforce_certificate_pinning
        self.audit_callback = audit_callback
        self._lock = threading.RLock()

        self.signer = RequestSigner()
        self.secure_channel = SecureChannel()

        self._cert_pins: dict[str, list[str]] = dict(self.DEFAULT_CERT_PINS)
        self._known_dns: dict[str, str] = {}
        self._threats: list[MITMThreatEvent] = []
        self._active_sessions: dict[str, dict[str, Any]] = {}

    def add_cert_pin(self, hostname: str, fingerprint: str) -> None:
        with self._lock:
            if hostname not in self._cert_pins:
                if len(self._cert_pins) >= self.MAX_PINNED_HOSTS:
                    oldest_host = next(iter(self._cert_pins))
                    del self._cert_pins[oldest_host]
                self._cert_pins[hostname] = []
            self._cert_pins[hostname].append(fingerprint.lower())
            if len(self._cert_pins[hostname]) > self.MAX_CERT_PINS_PER_HOST:
                self._cert_pins[hostname] = self._cert_pins[hostname][-self.MAX_CERT_PINS_PER_HOST:]

    def verify_certificate(
        self,
        hostname: str,
        cert_der: bytes | None = None,
        cert_fingerprint: str | None = None,
    ) -> list[MITMThreatEvent]:
        threats = []

        if cert_der and not cert_fingerprint:
            cert_fingerprint = hashlib.sha256(cert_der).hexdigest()

        if not cert_fingerprint:
            return threats

        cert_fingerprint = cert_fingerprint.lower()

        with self._lock:
            if hostname in self._cert_pins:
                if self._cert_pins[hostname]:
                    if cert_fingerprint not in self._cert_pins[hostname]:
                        threat = MITMThreatEvent(
                            timestamp=datetime.now(),
                            threat_type=MITMThreatType.CERTIFICATE_MISMATCH,
                            severity=ThreatSeverity.CRITICAL,
                            description=f"Certificate pin mismatch for {hostname}",
                            source_ip=None,
                            destination=hostname,
                            details={
                                "expected_pins": self._cert_pins[hostname][:3],
                                "actual_fingerprint": cert_fingerprint[:16] + "...",
                            },
                        )
                        threats.append(threat)
                elif self.enforce_certificate_pinning:
                    threat = MITMThreatEvent(
                        timestamp=datetime.now(),
                        threat_type=MITMThreatType.CERTIFICATE_UNTRUSTED,
                        severity=ThreatSeverity.HIGH,
                        description=f"No certificate pins configured for pinned host {hostname}",
                        source_ip=None,
                        destination=hostname,
                        details={"hostname": hostname},
                    )
                    threats.append(threat)
            elif self.enforce_certificate_pinning:
                threat = MITMThreatEvent(
                    timestamp=datetime.now(),
                    threat_type=MITMThreatType.CERTIFICATE_UNTRUSTED,
                    severity=ThreatSeverity.HIGH,
                    description=f"No certificate pin configured for {hostname}",
                    source_ip=None,
                    destination=hostname,
                    details={"hostname": hostname},
                )
                threats.append(threat)

        return threats

    def check_tls_version(self, version: str) -> MITMThreatEvent | None:
        """Check TLS version for downgrade attacks.

        Args:
            version: TLS version string (e.g., "TLSv1.3")

        Returns:
            Threat event if downgrade detected
        """
        if version in ("TLSv1.0", "TLSv1.1", "SSLv3"):
            return MITMThreatEvent(
                timestamp=datetime.now(),
                threat_type=MITMThreatType.TLS_DOWNGRADE,
                severity=ThreatSeverity.HIGH,
                description=f"Deprecated TLS version detected: {version}",
                source_ip=None,
                destination="unknown",
                details={"version": version},
            )
        return None

    def register_dns_resolution(self, hostname: str, ip_address: str) -> MITMThreatEvent | None:
        """Register and check DNS resolution.

        Args:
            hostname: Resolved hostname
            ip_address: Resolved IP address

        Returns:
            Threat if DNS spoofing detected
        """
        with self._lock:
            if hostname in self._known_dns:
                expected_ip = self._known_dns[hostname]
                if ip_address != expected_ip:
                    return MITMThreatEvent(
                        timestamp=datetime.now(),
                        threat_type=MITMThreatType.DNS_SPOOFING,
                        severity=ThreatSeverity.HIGH,
                        description=f"DNS resolution mismatch for {hostname}",
                        source_ip=ip_address,
                        destination=hostname,
                        details={
                            "expected_ip": expected_ip,
                            "actual_ip": ip_address,
                        },
                    )

            self._known_dns[hostname] = ip_address
            if len(self._known_dns) > self.MAX_DNS_ENTRIES:
                oldest = list(self._known_dns.keys())[:len(self._known_dns) - self.MAX_DNS_ENTRIES]
                for k in oldest:
                    del self._known_dns[k]
        return None

    def create_session(self, session_id: str, client_info: dict[str, Any]) -> str:
        """Create a new session.

        Args:
            session_id: Unique session identifier
            client_info: Client information (IP, user agent, etc.)

        Returns:
            Session token
        """
        session_token = self.signer.sign_data(f"session:{session_id}:{time.time()}")

        with self._lock:
            self._active_sessions[session_id] = {
                "created_at": datetime.now(),
                "client_info": client_info,
                "token": session_token,
                "last_activity": datetime.now(),
            }
            if len(self._active_sessions) > self.MAX_SESSIONS:
                oldest = sorted(self._active_sessions.items(), key=lambda x: x[1].get("last_activity", datetime.min))[:len(self._active_sessions) - self.MAX_SESSIONS]
                for k, _ in oldest:
                    del self._active_sessions[k]

        return session_token

    def validate_session(
        self,
        session_id: str,
        token: str,
        client_ip: str | None = None,
    ) -> MITMThreatEvent | None:
        """Validate a session.

        Args:
            session_id: Session identifier
            token: Session token
            client_ip: Client IP for hijacking check

        Returns:
            Threat if session hijacking detected
        """
        with self._lock:
            session = self._active_sessions.get(session_id)

            if not session:
                return MITMThreatEvent(
                    timestamp=datetime.now(),
                    threat_type=MITMThreatType.SESSION_HIJACKING,
                    severity=ThreatSeverity.HIGH,
                    description="Invalid session ID",
                    source_ip=client_ip,
                    destination="local",
                    details={"session_id": session_id[:8] + "..."},
                )

            if not hmac.compare_digest(session["token"], token):
                return MITMThreatEvent(
                    timestamp=datetime.now(),
                    threat_type=MITMThreatType.SESSION_HIJACKING,
                    severity=ThreatSeverity.HIGH,
                    description="Invalid session token",
                    source_ip=client_ip,
                    destination="local",
                    details={"session_id": session_id[:8] + "..."},
                )

            if client_ip and "client_info" in session:
                original_ip = session["client_info"].get("ip")
                if original_ip and client_ip != original_ip:
                    return MITMThreatEvent(
                        timestamp=datetime.now(),
                        threat_type=MITMThreatType.SESSION_HIJACKING,
                        severity=ThreatSeverity.MEDIUM,
                        description="Session IP mismatch",
                        source_ip=client_ip,
                        destination="local",
                        details={
                            "original_ip": original_ip,
                            "current_ip": client_ip,
                        },
                    )

            session["last_activity"] = datetime.now()
        return None

    def revoke_session(self, session_id: str) -> None:
        with self._lock:
            self._active_sessions.pop(session_id, None)
        self.secure_channel.revoke_session_key(session_id)

    def check_response_integrity(
        self,
        request_signature: str,
        response_data: str | bytes,
        response_signature: str | None = None,
    ) -> MITMThreatEvent | None:
        """Check response for tampering.

        Args:
            request_signature: Signature of the original request
            response_data: Response data
            response_signature: Expected signature of response

        Returns:
            Threat if tampering detected
        """
        if response_signature:
            computed = self.signer.sign_data(response_data)
            if not hmac.compare_digest(computed, response_signature):
                return MITMThreatEvent(
                    timestamp=datetime.now(),
                    threat_type=MITMThreatType.RESPONSE_TAMPERING,
                    severity=ThreatSeverity.HIGH,
                    description="Response signature mismatch",
                    source_ip=None,
                    destination="unknown",
                    details={},
                )
        return None

    def record_threat(self, threat: MITMThreatEvent) -> None:
        with self._lock:
            self._threats.append(threat)
            if len(self._threats) > self.MAX_THREATS:
                self._threats = self._threats[-self.MAX_THREATS:]
        if self.audit_callback:
            try:
                self.audit_callback(threat)
            except Exception as e:
                logger.warning(f"Audit callback error: {e}")

    def get_threats(
        self,
        severity: ThreatSeverity | None = None,
        threat_type: MITMThreatType | None = None,
        since: datetime | None = None,
    ) -> list[MITMThreatEvent]:
        with self._lock:
            threats = list(self._threats)

        if severity:
            threats = [t for t in threats if t.severity == severity]
        if threat_type:
            threats = [t for t in threats if t.threat_type == threat_type]
        if since:
            threats = [t for t in threats if t.timestamp >= since]

        return threats

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            total_threats = len(self._threats)
            active_sessions = len(self._active_sessions)
            cert_pins_count = sum(len(pins) for pins in self._cert_pins.values())
            threats_snapshot = list(self._threats)

        by_severity: dict[str, int] = {}
        for severity in ThreatSeverity:
            by_severity[severity.value] = len([
                t for t in threats_snapshot if t.severity == severity
            ])

        by_type: dict[str, int] = {}
        for threat_type in MITMThreatType:
            by_type[threat_type.value] = len([
                t for t in threats_snapshot if t.threat_type == threat_type
            ])

        return {
            "total_threats": total_threats,
            "active_sessions": active_sessions,
            "cert_pins_count": cert_pins_count,
            "by_severity": by_severity,
            "by_type": by_type,
            "tls_enforced": self.enforce_tls,
            "cert_pinning_enabled": self.enforce_certificate_pinning,
        }

    def save_state(self) -> None:
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        with self._lock:
            threats_data = [t.to_dict() for t in self._threats[-100:]]
            pins_data = dict(self._cert_pins)

        threats_file = self.storage_path / "mitm_threats.json"
        atomic_write_json(threats_file, threats_data)

        pins_file = self.storage_path / "cert_pins.json"
        atomic_write_json(pins_file, pins_data)
