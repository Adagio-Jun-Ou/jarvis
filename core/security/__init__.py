"""Jarvis Security Suite - Comprehensive security module.

This module provides multi-layered security for Jarvis:
- Data Leak Protection (DLP): Prevents sensitive data exfiltration
- Unauthorized Data Branch Detection: Monitors data flows for anomalies
- MITM Protection: TLS validation, request signing, encryption
- Stack Bug Defense: Input validation, buffer protection, safe memory
- SSRF Protection: Outbound URL validation for internal-network access
- Session Security: Session lifecycle and login-attempt tracking
"""

from .dlp import DataLeakProtector, SensitiveDataPattern, DataLeakEvent, LeakSeverity
from .branch_detection import DataBranchMonitor, DataFlowEvent, BranchAnomaly, AnomalySeverity, AnomalyType, BranchType, FlowDirection
from .mitm_protection import MITMProtector, SecureChannel, RequestSigner, MITMThreatType, ThreatSeverity
from .stack_defense import StackDefense, InputValidator, SafeBuffer, VulnerabilityType, DefenseAction
from .audit import SecurityAuditor, AuditEvent, AuditLevel, AuditCategory
from .upload_validation import validate_upload, sanitize_filename, UploadValidation, UploadValidationResult
from .ssrf_protection import (
    SSRFProtector,
    SSRFValidationResult,
    get_ssrf_protector,
    validate_outbound_url,
    resolve_validated_ips,
    sanitize_request_headers,
)
from .session_security import (
    SessionSecurityManager,
    SessionInfo,
    LoginAttempt,
    get_session_security,
)

__all__ = [
    # DLP
    "DataLeakProtector",
    "SensitiveDataPattern",
    "DataLeakEvent",
    "LeakSeverity",
    # Branch Detection
    "DataBranchMonitor",
    "DataFlowEvent",
    "BranchAnomaly",
    "AnomalySeverity",
    "AnomalyType",
    "BranchType",
    "FlowDirection",
    # MITM Protection
    "MITMProtector",
    "SecureChannel",
    "RequestSigner",
    "MITMThreatType",
    "ThreatSeverity",
    # Stack Defense
    "StackDefense",
    "InputValidator",
    "SafeBuffer",
    "VulnerabilityType",
    "DefenseAction",
    # Audit
    "SecurityAuditor",
    "AuditEvent",
    "AuditLevel",
    "AuditCategory",
    # Upload Validation
    "validate_upload",
    "sanitize_filename",
    "UploadValidation",
    "UploadValidationResult",
    # SSRF Protection
    "SSRFProtector",
    "SSRFValidationResult",
    "get_ssrf_protector",
    "validate_outbound_url",
    "resolve_validated_ips",
    "sanitize_request_headers",
    # Session Security
    "SessionSecurityManager",
    "SessionInfo",
    "LoginAttempt",
    "get_session_security",
]
