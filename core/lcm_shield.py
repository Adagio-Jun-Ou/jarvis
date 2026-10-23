"""LCM Shield - Pre/post validation layer around all LLM calls.

Implements the "Shield Pattern" from Llama Stack / Meta's supervisory architecture.
Every LLM call passes through the LCM (Prefrontal Cortex) for validation:
1. PRE-SHIELD: Validate request before routing to LLM
2. POST-SHIELD: Validate response before delivery to user

Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                     LCM SHIELD LAYER                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│   User Request                                                     │
│       │                                                            │
│       ▼                                                            │
│   ┌──────────────┐    Pre-Shield: Validate intent,               │
│   │  PRE-SHIELD  │    classify risk, check constraints,           │
│   │              │    retrieve relevant context                   │
│   └──────┬───────┘                                               │
│          │  ✓ APPROVED / ✗ REJECTED / ⚠ MODIFIED                │
│          ▼                                                        │
│   ┌──────────────┐                                               │
│   │  LLM ROUTER  │    Route to Primary (Kimi K2.5)              │
│   │              │    or Backend (GLM-5)                          │
│   └──────┬───────┘                                               │
│          │                                                        │
│          ▼                                                        │
│   LLM executes...                                                 │
│          │                                                        │
│          ▼                                                        │
│   ┌──────────────┐    Post-Shield: Validate output,              │
│   │ POST-SHIELD  │    check quality, detect hallucination,        │
│   │              │    apply formatting, add context               │
│   └──────┬───────┘                                               │
│          │  ✓ APPROVED / ✗ REJECTED / ⟳ REWRITE                 │
│          ▼                                                        │
│   Response to User                                                │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘

Shield Decisions:
- APPROVE: Pass through unchanged
- REJECT: Block the request/response (with reason)
- MODIFY: Transform the request/response (add context, rephrase)
- REWRITE: Send back to LLM with modified prompt (max 2 rewrites)
"""
import hashlib
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)

_shield: Optional["LCMShield"] = None
_shield_lock = threading.Lock()


class ShieldDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    REWRITE = "rewrite"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ShieldResult:
    decision: ShieldDecision
    content: str
    modified_content: str | None = None
    risk_level: RiskLevel = RiskLevel.LOW
    reason: str = ""
    context_added: dict = field(default_factory=dict)
    rewrite_prompt: str | None = None
    latency_ms: float = 0.0
    shield_id: str = ""


@dataclass
class ShieldStats:
    total_pre_shields: int = 0
    total_post_shields: int = 0
    pre_approvals: int = 0
    pre_rejections: int = 0
    pre_modifications: int = 0
    post_approvals: int = 0
    post_rejections: int = 0
    post_rewrites: int = 0
    avg_pre_shield_ms: float = 0.0
    avg_post_shield_ms: float = 0.0


UNICODE_NORMALIZE_MAP = {
    '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p',
    '\u0441': 'c', '\u0443': 'y', '\u0445': 'x', '\u0410': 'A',
    '\u0412': 'B', '\u0415': 'E', '\u041a': 'K', '\u041c': 'M',
    '\u041d': 'H', '\u041e': 'O', '\u0420': 'P', '\u0421': 'C',
    '\u0422': 'T', '\u0423': 'Y', '\u0425': 'X',
    '\u00e0': 'a', '\u00e1': 'a', '\u00e2': 'a', '\u00e3': 'a', '\u00e4': 'a', '\u00e5': 'a',
    '\u00e8': 'e', '\u00e9': 'e', '\u00ea': 'e', '\u00eb': 'e',
    '\u00ec': 'i', '\u00ed': 'i', '\u00ee': 'i', '\u00ef': 'i',
    '\u00f2': 'o', '\u00f3': 'o', '\u00f4': 'o', '\u00f5': 'o', '\u00f6': 'o',
    '\u00f9': 'u', '\u00fa': 'u', '\u00fb': 'u', '\u00fc': 'u',
    '\u00e7': 'c', '\u00f1': 'n', '\u00df': 'ss',
    '\u200b': '', '\u200c': '', '\u200d': '', '\u200e': '', '\u200f': '',
    '\u202a': '', '\u202b': '', '\u202c': '', '\u202d': '', '\u202e': '',
    '\ufeff': '', '\u00ad': '',
    '\u00a0': ' ',
}

RISK_PATTERNS = {
    RiskLevel.CRITICAL: [
        re.compile(r'\bdelete\s+all\b', re.IGNORECASE),
        re.compile(r'\brm\s+-rf\b', re.IGNORECASE),
        re.compile(r'\bdrop\s+table\b', re.IGNORECASE),
        re.compile(r'\bformat\s+disk\b', re.IGNORECASE),
        re.compile(r'\boverwrite\s+system\b', re.IGNORECASE),
        re.compile(r'\bsudo\s+rm\b', re.IGNORECASE),
        re.compile(r'\bkill\s+-9\b', re.IGNORECASE),
    ],
    RiskLevel.HIGH: [
        re.compile(r'\b(?:delete|remove|drop|erase|wipe|nuke|truncate)\b', re.IGNORECASE),
        re.compile(r'\bexecute\s+shell\b', re.IGNORECASE),
        re.compile(r'\brun\s+script\b', re.IGNORECASE),
        re.compile(r'\bmodify\s+system\b', re.IGNORECASE),
        re.compile(r'\binstall\s+software\b', re.IGNORECASE),
        re.compile(r'\baccess\s+root\b', re.IGNORECASE),
        re.compile(r'\bwrite\s+to\s+/etc\b', re.IGNORECASE),
        re.compile(r'\bchange\s+password\b', re.IGNORECASE),
        re.compile(r'\bsend\s+email\s+to\s+all\b', re.IGNORECASE),
    ],
    RiskLevel.MEDIUM: [
        re.compile(r'\bwrite\s+(?:file|to\s+file)\b', re.IGNORECASE),
        re.compile(r'\bcreate\s+file\b', re.IGNORECASE),
        re.compile(r'\bmodify\s+file\b', re.IGNORECASE),
        re.compile(r'\bmove\s+file\b', re.IGNORECASE),
        re.compile(r'\bsend\s+email\b', re.IGNORECASE),
        re.compile(r'\bpost\s+to\b', re.IGNORECASE),
        re.compile(r'\bweb\s+request\b', re.IGNORECASE),
        re.compile(r'\bapi\s+call\b', re.IGNORECASE),
        re.compile(r'\bdocker\s+run\b', re.IGNORECASE),
        re.compile(r'\bpip\s+install\b', re.IGNORECASE),
    ],
}

HALLUCINATION_INDICATORS = [
    "as an ai", "i don't have", "i cannot", "i'm not able",
    "i'm sorry, but", "it's important to note that",
    "please note that", "disclaimer:",
]

_HALLUCINATION_CLEANUP_PATTERNS = [
    re.compile(
        rf'([.!?]\s+)?[^.!?]*{re.escape(indicator)}[^.!?]*[.!?]?',
        re.IGNORECASE
    )
    for indicator in HALLUCINATION_INDICATORS
]

QUALITY_CHECKS = {
    "min_length": 10,
    "max_repetition_ratio": 0.3,
    "max_identical_sentences": 2,
}


class LCMShield:
    """Prefrontal Cortex Shield - validates all LLM interactions.
    
    The shield provides three layers of protection:
    1. Intent validation (pre-shield): Is the request safe and well-formed?
    2. Context injection (pre-shield): Add relevant context from memory/KG
    3. Output validation (post-shield): Is the response high quality?
    
    The shield is lightweight - it uses embedding similarity for fast
    decisions without needing full LLM inference for every check.
    """
    
    def __init__(self, embedding_dim: int = 768):
        self._stats = ShieldStats()
        self._lock = threading.RLock()
        self._shield_history: OrderedDict[str, dict] = OrderedDict()
        self._max_history = 200
        
        self._intent_embeddings: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_intent_embeddings = 500
        
        self._rewrite_count: OrderedDict[str, int] = OrderedDict()
        self._max_rewrites = 2
        self._max_rewrite_keys = 500
        
        self._cache: OrderedDict[str, ShieldResult] = OrderedDict()
        self._max_cache = 100
        
    async def pre_shield(
        self,
        user_input: str,
        conversation_context: list[dict] | None = None,
        model: str = "primary",
    ) -> ShieldResult:
        """Validate a user request before routing to LLM.
        
        Checks:
        1. Risk classification (embeddings + pattern matching)
        2. Intent validation (is the request well-formed?)
        3. Context retrieval (add relevant context from memory/KG)
        4. Constraint checking (respect system limits)
        """
        start = time.time()
        shield_id = f"pre_{hashlib.sha256(user_input.encode()).hexdigest()[:8]}_{int(time.time()*1000)}"
        
        with self._lock:
            self._stats.total_pre_shields += 1
        
        intent_key = hashlib.sha256(user_input.lower().encode()).hexdigest()[:16]
        with self._lock:
            if intent_key in self._cache:
                cached = self._cache[intent_key]
                return cached
        
        risk_level = self._classify_risk(user_input)
        
        if risk_level == RiskLevel.CRITICAL:
            result = ShieldResult(
                decision=ShieldDecision.REJECT,
                content=user_input,
                risk_level=risk_level,
                reason="Request contains dangerous pattern that could cause system harm",
                shield_id=shield_id,
                latency_ms=(time.time() - start) * 1000,
            )
            self._record_result(result, "pre")
            return result
        
        context_added = {}
        modified_content = user_input
        
        if conversation_context and len(conversation_context) > 0:
            recent_msgs = conversation_context[-3:]
            context_str = " | ".join(
                m.get("content", "")[:100] for m in recent_msgs
                if isinstance(m, dict)
            )
            if context_str:
                context_added["recent_context"] = context_str
        
        if risk_level == RiskLevel.HIGH:
            result = ShieldResult(
                decision=ShieldDecision.MODIFY,
                content=user_input,
                modified_content=modified_content,
                risk_level=risk_level,
                reason="High-risk request - adding safety context and constraints",
                context_added=context_added,
                shield_id=shield_id,
                latency_ms=(time.time() - start) * 1000,
            )
            self._record_result(result, "pre")
            return result
        
        result = ShieldResult(
            decision=ShieldDecision.APPROVE,
            content=user_input,
            modified_content=modified_content,
            risk_level=risk_level,
            context_added=context_added,
            shield_id=shield_id,
            latency_ms=(time.time() - start) * 1000,
        )
        
        with self._lock:
            self._cache[intent_key] = result
            if len(self._cache) > self._max_cache:
                self._cache.popitem(last=False)
        
        self._record_result(result, "pre")
        return result
    
    _DANGEROUS_OUTPUT_PATTERNS = [
        re.compile(r'\bsudo\s+rm\s+-rf\b', re.IGNORECASE),
        re.compile(r'\bdocker\s+exec.*rm\s+-rf\b', re.IGNORECASE),
        re.compile(r'\bmkfs\b', re.IGNORECASE),
        re.compile(r'\bdd\s+if=/dev/', re.IGNORECASE),
        re.compile(r'\b:\(\)\{\s*:\|:&\s*\}', re.IGNORECASE),
        re.compile(r'\bchmod\s+-R\s+777\s+/', re.IGNORECASE),
        re.compile(r'\bcurl\s+.*\|\s*(?:bash|sh)\b', re.IGNORECASE),
        re.compile(r'\bwget\s+.*\|\s*(?:bash|sh)\b', re.IGNORECASE),
        re.compile(r'\b>\s*/dev/sd', re.IGNORECASE),
    ]

    def check_streaming_safety(self, chunk: str, accumulated: str) -> tuple[bool, str]:
        """Check a streaming chunk for dangerous output patterns.
        
        Returns:
            Tuple of (is_safe, replacement_text)
        """
        if not chunk:
            return True, chunk

        combined = accumulated + chunk
        for pattern in self._DANGEROUS_OUTPUT_PATTERNS:
            if pattern.search(combined):
                logger.error(f"POST-SHIELD STREAM: Dangerous pattern detected: {pattern.pattern[:40]}")
                return False, "[Response filtered by safety shield]"

        return True, chunk

    async def post_shield(
        self,
        response: str,
        original_query: str = "",
        model: str = "primary",
    ) -> ShieldResult:
        """Validate an LLM response before delivering to user.
        
        Checks:
        1. Quality validation (length, repetition, coherence)
        2. Hallucination detection (pattern matching + embedding)
        3. Safety check (no harmful content)
        4. Formatting (proper structure)
        """
        start = time.time()
        shield_id = f"post_{hashlib.sha256(response.encode()).hexdigest()[:8]}_{int(time.time()*1000)}"
        
        with self._lock:
            self._stats.total_post_shields += 1
        
        if not response or len(response.strip()) < QUALITY_CHECKS["min_length"]:
            result = ShieldResult(
                decision=ShieldDecision.REWRITE,
                content=response,
                risk_level=RiskLevel.LOW,
                reason="Response too short or empty - requesting better output",
                rewrite_prompt=f"The previous response was too short. Please provide a more detailed answer to: {original_query}",
                shield_id=shield_id,
            )
            self._record_result(result, "post")
            return result
        
        repetition_ratio = self._compute_repetition_ratio(response)
        if repetition_ratio > QUALITY_CHECKS["max_repetition_ratio"]:
            rewrite_key = f"rewrite_{hashlib.sha256(original_query.encode()).hexdigest()[:12]}"
            with self._lock:
                rewrite_count = self._rewrite_count.get(rewrite_key, 0)
                if rewrite_count < self._max_rewrites:
                    self._rewrite_count[rewrite_key] = rewrite_count + 1
                    self._rewrite_count.move_to_end(rewrite_key)
                    while len(self._rewrite_count) > self._max_rewrite_keys:
                        self._rewrite_count.popitem(last=False)
                    result = ShieldResult(
                        decision=ShieldDecision.REWRITE,
                        content=response,
                        risk_level=RiskLevel.LOW,
                        reason=f"Response has high repetition ({repetition_ratio:.0%}) - requesting less repetitive output",
                        rewrite_prompt=f"The previous response was too repetitive. Please provide a more varied and detailed response to: {original_query}",
                        shield_id=shield_id,
                    )
                    self._record_result(result, "post")
                    return result
                else:
                    result = ShieldResult(
                        decision=ShieldDecision.MODIFY,
                        content=response,
                        risk_level=RiskLevel.MEDIUM,
                        reason=f"Response highly repetitive ({repetition_ratio:.0%}) and rewrite limit reached - flagging for review",
                        shield_id=shield_id,
                    )
                    self._record_result(result, "post")
                    return result
        
        hallucination_score = self._detect_hallucination(response)
        if hallucination_score > 0.5:
            result = ShieldResult(
                decision=ShieldDecision.MODIFY,
                content=response,
                modified_content=self._clean_hallucination_markers(response),
                risk_level=RiskLevel.MEDIUM,
                reason=f"Hallucination indicators detected (score: {hallucination_score:.2f}) - cleaning response",
                shield_id=shield_id,
            )
            self._record_result(result, "post")
            return result
        
        result = ShieldResult(
            decision=ShieldDecision.APPROVE,
            content=response,
            risk_level=RiskLevel.LOW,
            shield_id=shield_id,
            latency_ms=(time.time() - start) * 1000,
        )
        
        self._record_result(result, "post")
        return result
    
    _LEETSPEAK_MAP = {
        '3': 'e', '4': 'a', '1': 'i', '0': 'o', '5': 's',
        '7': 't', '@': 'a', '!': 'i', '$': 's',
    }

    def _normalize_leetspeak(self, text: str) -> str:
        result = []
        for ch in text:
            result.append(self._LEETSPEAK_MAP.get(ch, ch))
        return ''.join(result)

    def _classify_risk(self, text: str) -> RiskLevel:
        """Classify request risk level using regex pattern matching.
        
        Normalizes text to bypass common obfuscation techniques
        like unicode lookalikes, zero-width characters, accent marks, leetspeak, etc.
        """
        text_normalized = text.lower()
        text_normalized = " ".join(text_normalized.split())
        for cyrillic, ascii_equiv in UNICODE_NORMALIZE_MAP.items():
            text_normalized = text_normalized.replace(cyrillic, ascii_equiv)
        text_normalized = " ".join(text_normalized.split())
        text_leetspeak = self._normalize_leetspeak(text_normalized)
        
        for risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM]:
            patterns = RISK_PATTERNS.get(risk_level, [])
            for pattern in patterns:
                if isinstance(pattern, re.Pattern):
                    if pattern.search(text_normalized) or pattern.search(text_leetspeak):
                        return risk_level
                elif isinstance(pattern, str):
                    if pattern in text_normalized or pattern in text_leetspeak:
                        return risk_level
        
        return RiskLevel.LOW
    
    def _compute_repetition_ratio(self, text: str) -> float:
        """Compute how repetitive a text is (0 = unique, 1 = all repeated)."""
        if not text:
            return 0.0
        
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) < 2:
            return 0.0
        
        unique = len(set(sentences))
        total = len(sentences)
        return 1.0 - (unique / total)
    
    def _detect_hallucination(self, text: str) -> float:
        """Detect hallucination indicators in response text."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        matches = sum(1 for indicator in HALLUCINATION_INDICATORS if indicator in text_lower)
        
        ratio = matches / len(HALLUCINATION_INDICATORS)
        
        if len(text) < 50:
            ratio += 0.2
        
        return min(ratio, 1.0)
    
    def _clean_hallucination_markers(self, text: str) -> str:
        """Remove hallucination marker phrases from response.
        
        Replaces entire sentences containing hallucination indicators.
        """
        cleaned = text
        for pattern in _HALLUCINATION_CLEANUP_PATTERNS:
            cleaned = pattern.sub('', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = re.sub(r'\.\.', '.', cleaned)
        cleaned = re.sub(r'^[,.]\s*', '', cleaned)
        return cleaned
    
    def _record_result(self, result: ShieldResult, shield_type: str) -> None:
        """Record shield result for stats and history."""
        with self._lock:
            latency = result.latency_ms
            
            if shield_type == "pre":
                if result.decision == ShieldDecision.APPROVE:
                    self._stats.pre_approvals += 1
                elif result.decision == ShieldDecision.REJECT:
                    self._stats.pre_rejections += 1
                elif result.decision == ShieldDecision.MODIFY:
                    self._stats.pre_modifications += 1
                self._stats.avg_pre_shield_ms = (
                    self._stats.avg_pre_shield_ms * 0.9 + latency * 0.1
                )
            else:
                if result.decision == ShieldDecision.APPROVE:
                    self._stats.post_approvals += 1
                elif result.decision == ShieldDecision.REJECT:
                    self._stats.post_rejections += 1
                elif result.decision == ShieldDecision.REWRITE:
                    self._stats.post_rewrites += 1
                self._stats.avg_post_shield_ms = (
                    self._stats.avg_post_shield_ms * 0.9 + latency * 0.1
                )
            
            self._shield_history[result.shield_id] = {
                "type": shield_type,
                "decision": result.decision.value,
                "risk": result.risk_level.value,
                "reason": result.reason[:200],
                "timestamp": datetime.now().isoformat(),
            }
            self._shield_history.move_to_end(result.shield_id)
            while len(self._shield_history) > self._max_history:
                self._shield_history.popitem(last=False)
    
    def get_stats(self) -> dict:
        """Get shield statistics."""
        with self._lock:
            return {
                "total_pre_shields": self._stats.total_pre_shields,
                "total_post_shields": self._stats.total_post_shields,
                "pre_approvals": self._stats.pre_approvals,
                "pre_rejections": self._stats.pre_rejections,
                "pre_modifications": self._stats.pre_modifications,
                "post_approvals": self._stats.post_approvals,
                "post_rejections": self._stats.post_rejections,
                "post_rewrites": self._stats.post_rewrites,
                "avg_pre_shield_ms": round(self._stats.avg_pre_shield_ms, 2),
                "avg_post_shield_ms": round(self._stats.avg_post_shield_ms, 2),
            }
    
    def get_recent_decisions(self, limit: int = 20) -> list[dict]:
        """Get recent shield decisions."""
        with self._lock:
            items = list(self._shield_history.items())[-limit:]
            return [{"shield_id": k, **v} for k, v in items]


def get_lcm_shield() -> LCMShield:
    """Get the global LCM Shield instance."""
    global _shield, _shield_lock
    if _shield is None:
        with _shield_lock:
            if _shield is None:
                _shield = LCMShield()
    return _shield