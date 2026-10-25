"""Checkpoint Permission System for Frontier Domains.

This module provides permission gating when Jarvis operations enter
sensitive or unusual domains ("frontier domains"). It ensures human
oversight for potentially impactful decisions.

Frontier domains include:
- Self-modification of core code
- Accessing sensitive user data
- Making external API calls with significant impact
- Autonomous actions in the real world
- Learning about user's private information
- Ethical gray areas

The checkpoint system:
1. Detects frontier domain entry
2. Pauses the operation
3. Requests user permission with context
4. Logs the decision
5. Proceeds or aborts based on user response
"""

import asyncio
import json
import logging
import threading
import time
from core.helpers import atomic_write_json, read_json_with_integrity
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

STORAGE_PATH = Path.home() / ".jarvis" / "checkpoints"


class FrontierDomain(Enum):
    SELF_MODIFICATION = "self_modification"
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"
    EXTERNAL_ACTION = "external_action"
    AUTONOMOUS_DECISION = "autonomous_decision"
    PRIVATE_LEARNING = "private_learning"
    ETHICAL_AMBIGUITY = "ethical_ambiguity"
    FINANCIAL_OPERATION = "financial_operation"
    SECURITY_SENSITIVE = "security_sensitive"
    IDENTITY_IMPERSONATION = "identity_impersonation"
    LEGAL_AMBIGUITY = "legal_ambiguity"


class CheckpointStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"
    AUTO_APPROVED = "auto_approved"
    ESCALATED = "escalated"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Checkpoint:
    id: str
    timestamp: datetime
    domain: FrontierDomain
    risk_level: RiskLevel
    operation: str
    context: dict[str, Any]
    impact_preview: str
    recommendations: list[str]
    status: CheckpointStatus = CheckpointStatus.PENDING
    user_response: str | None = None
    response_timestamp: datetime | None = None
    reasoning: str | None = None
    auto_approved_reason: str | None = None
    escalated_to: str | None = None
    related_evaluation_id: str | None = None


@dataclass
class CheckpointPolicy:
    domain: FrontierDomain
    auto_approve: bool = False
    auto_approve_conditions: list[str] = field(default_factory=list)
    require_reason: bool = True
    timeout_seconds: int = 300
    escalate_after: int = 3
    allowed_without_permission: list[str] = field(default_factory=list)


@dataclass
class CheckpointStats:
    total_checkpoints: int = 0
    approved: int = 0
    denied: int = 0
    auto_approved: int = 0
    escalated: int = 0
    by_domain: dict[str, int] = field(default_factory=dict)
    by_risk: dict[str, int] = field(default_factory=dict)


FRONTIER_DOMAIN_DESCRIPTIONS = {
    FrontierDomain.SELF_MODIFICATION: "Modifying Jarvis's own code or behavior",
    FrontierDomain.SENSITIVE_DATA_ACCESS: "Accessing sensitive user data (passwords, keys, personal info)",
    FrontierDomain.EXTERNAL_ACTION: "Taking actions that affect the external world",
    FrontierDomain.AUTONOMOUS_DECISION: "Making significant decisions without direct user instruction",
    FrontierDomain.PRIVATE_LEARNING: "Learning from or about user's private information",
    FrontierDomain.ETHICAL_AMBIGUITY: "Operating in an area with unclear ethical boundaries",
    FrontierDomain.FINANCIAL_OPERATION: "Operations involving financial data or transactions",
    FrontierDomain.SECURITY_SENSITIVE: "Operations that could affect system security",
    FrontierDomain.IDENTITY_IMPERSONATION: "Acting on behalf of the user in external systems",
    FrontierDomain.LEGAL_AMBIGUITY: "Operating in an area with potential legal implications",
}

DEFAULT_POLICIES = {
    FrontierDomain.SELF_MODIFICATION: CheckpointPolicy(
        domain=FrontierDomain.SELF_MODIFICATION,
        auto_approve=False,
        require_reason=True,
        timeout_seconds=600,
        escalate_after=2,
    ),
    FrontierDomain.SENSITIVE_DATA_ACCESS: CheckpointPolicy(
        domain=FrontierDomain.SENSITIVE_DATA_ACCESS,
        auto_approve=False,
        require_reason=True,
        timeout_seconds=300,
        escalate_after=3,
    ),
    FrontierDomain.EXTERNAL_ACTION: CheckpointPolicy(
        domain=FrontierDomain.EXTERNAL_ACTION,
        auto_approve=False,
        require_reason=True,
        timeout_seconds=180,
        escalate_after=3,
    ),
    FrontierDomain.AUTONOMOUS_DECISION: CheckpointPolicy(
        domain=FrontierDomain.AUTONOMOUS_DECISION,
        auto_approve=False,
        require_reason=True,
        timeout_seconds=300,
        escalate_after=5,
    ),
    FrontierDomain.PRIVATE_LEARNING: CheckpointPolicy(
        domain=FrontierDomain.PRIVATE_LEARNING,
        auto_approve=False,
        require_reason=True,
        timeout_seconds=300,
        escalate_after=3,
    ),
    FrontierDomain.ETHICAL_AMBIGUITY: CheckpointPolicy(
        domain=FrontierDomain.ETHICAL_AMBIGUITY,
        auto_approve=False,
        require_reason=True,
        timeout_seconds=600,
        escalate_after=2,
    ),
    FrontierDomain.FINANCIAL_OPERATION: CheckpointPolicy(
        domain=FrontierDomain.FINANCIAL_OPERATION,
        auto_approve=False,
        require_reason=True,
        timeout_seconds=900,
        escalate_after=1,
    ),
    FrontierDomain.SECURITY_SENSITIVE: CheckpointPolicy(
        domain=FrontierDomain.SECURITY_SENSITIVE,
        auto_approve=False,
        require_reason=True,
        timeout_seconds=600,
        escalate_after=2,
    ),
    FrontierDomain.IDENTITY_IMPERSONATION: CheckpointPolicy(
        domain=FrontierDomain.IDENTITY_IMPERSONATION,
        auto_approve=False,
        require_reason=True,
        timeout_seconds=300,
        escalate_after=2,
    ),
    FrontierDomain.LEGAL_AMBIGUITY: CheckpointPolicy(
        domain=FrontierDomain.LEGAL_AMBIGUITY,
        auto_approve=False,
        require_reason=True,
        timeout_seconds=900,
        escalate_after=1,
    ),
}


class CheckpointPermissionManager:
    """Manages checkpoint permissions for frontier domains."""

    def __init__(self, llm_client=None, storage_path: Path | None = None):
        self.llm = llm_client
        self.storage_path = storage_path or STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._checkpoints: dict[str, Checkpoint] = {}
        self._MAX_CHECKPOINTS = 200
        self._policies: dict[FrontierDomain, CheckpointPolicy] = dict(DEFAULT_POLICIES)
        self._stats = CheckpointStats()
        self._lock = threading.Lock()
        self._pending_callbacks: dict[str, asyncio.Future] = {}
        self._user_callback: Callable | None = None
        self._approval_history: list[dict] = []

    async def request_permission(
        self,
        domain: FrontierDomain,
        operation: str,
        context: dict[str, Any],
        impact_preview: str = "",
        risk_level: RiskLevel | None = None,
    ) -> Checkpoint:
        """Request permission for an operation in a frontier domain.

        This will pause and wait for user approval/denial.
        """
        if risk_level is None:
            risk_level = self._assess_risk(domain, operation, context)

        checkpoint = Checkpoint(
            id=f"cp_{int(time.time() * 1000)}_{hash(operation) % 10000:04d}",
            timestamp=datetime.now(),
            domain=domain,
            risk_level=risk_level,
            operation=operation[:200],
            context=context,
            impact_preview=impact_preview[:500],
            recommendations=await self._generate_recommendations(domain, operation, context),
        )

        policy = self._policies.get(domain, DEFAULT_POLICIES.get(domain))

        if policy and self._check_auto_approve(checkpoint, policy):
            checkpoint.status = CheckpointStatus.AUTO_APPROVED
            checkpoint.auto_approved_reason = self._get_auto_approve_reason(checkpoint, policy)
            logger.info(f"Auto-approved checkpoint: {checkpoint.id} ({domain.value})")

            self._record_checkpoint(checkpoint)
            return checkpoint

        if policy and policy.allowed_without_permission:
            for pattern in policy.allowed_without_permission:
                if pattern.lower() in operation.lower():
                    checkpoint.status = CheckpointStatus.AUTO_APPROVED
                    checkpoint.auto_approved_reason = f"Matched allowed pattern: {pattern}"
                    self._record_checkpoint(checkpoint)
                    return checkpoint

        self._record_checkpoint(checkpoint)

        logger.info(f"Checkpoint created: {checkpoint.id} - {domain.value} - {risk_level.value}")

        if self._user_callback:
            try:
                future = asyncio.get_running_loop().create_future()
                self._pending_callbacks[checkpoint.id] = future

                from core.helpers import create_tracked_task
                create_tracked_task(self._notify_user(checkpoint), name="checkpoint_notify_user")

                try:
                    result = await asyncio.wait_for(
                        future,
                        timeout=policy.timeout_seconds if policy else 300
                    )
                    checkpoint = result
                except asyncio.TimeoutError:
                    checkpoint.status = CheckpointStatus.TIMEOUT
                    checkpoint.user_response = "timeout"
                    checkpoint.response_timestamp = datetime.now()

            except Exception as e:
                logger.error(f"Checkpoint callback error: {e}")
                checkpoint.status = CheckpointStatus.ESCALATED
        else:
            await self._save_for_later_review(checkpoint)

        return checkpoint

    def respond_to_checkpoint(
        self,
        checkpoint_id: str,
        approved: bool,
        reason: str | None = None,
    ) -> bool:
        """Respond to a pending checkpoint."""
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                logger.warning(f"Checkpoint not found: {checkpoint_id}")
                return False

            if checkpoint.status != CheckpointStatus.PENDING:
                logger.warning(f"Checkpoint already resolved: {checkpoint_id} ({checkpoint.status.value})")
                return False

            checkpoint.status = CheckpointStatus.APPROVED if approved else CheckpointStatus.DENIED
            checkpoint.user_response = "approved" if approved else "denied"
            checkpoint.reasoning = reason
            checkpoint.response_timestamp = datetime.now()

            self._update_stats(checkpoint)

            self._approval_history.append({
                "checkpoint_id": checkpoint_id,
                "domain": checkpoint.domain.value,
                "approved": approved,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            })

            if len(self._approval_history) > 100:
                self._approval_history = self._approval_history[-50:]

            if checkpoint_id in self._pending_callbacks:
                future = self._pending_callbacks.pop(checkpoint_id)
                if not future.done():
                    future.set_result(checkpoint)

            logger.info(f"Checkpoint {checkpoint_id} responded: {checkpoint.status.value}")

        self._save_checkpoint(checkpoint)
        return True

    def set_user_callback(self, callback: Callable):
        """Set callback for notifying user of pending checkpoints."""
        self._user_callback = callback

    def get_pending_checkpoints(self) -> list[dict]:
        """Get all pending checkpoints."""
        with self._lock:
            return [
                {
                    "id": cp.id,
                    "timestamp": cp.timestamp.isoformat(),
                    "domain": cp.domain.value,
                    "domain_description": FRONTIER_DOMAIN_DESCRIPTIONS.get(cp.domain, ""),
                    "risk_level": cp.risk_level.value,
                    "operation": cp.operation,
                    "impact_preview": cp.impact_preview,
                    "recommendations": cp.recommendations,
                    "status": cp.status.value,
                }
                for cp in self._checkpoints.values()
                if cp.status == CheckpointStatus.PENDING
            ]

    def get_checkpoint_history(self, limit: int = 50) -> list[dict]:
        """Get checkpoint history."""
        with self._lock:
            sorted_checkpoints = sorted(
                self._checkpoints.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )
            return [
                {
                    "id": cp.id,
                    "timestamp": cp.timestamp.isoformat(),
                    "domain": cp.domain.value,
                    "risk_level": cp.risk_level.value,
                    "operation": cp.operation,
                    "status": cp.status.value,
                    "user_response": cp.user_response,
                    "reasoning": cp.reasoning,
                }
                for cp in sorted_checkpoints[:limit]
            ]

    def get_stats(self) -> dict:
        """Get checkpoint statistics."""
        with self._lock:
            return {
                "total_checkpoints": self._stats.total_checkpoints,
                "approved": self._stats.approved,
                "denied": self._stats.denied,
                "auto_approved": self._stats.auto_approved,
                "escalated": self._stats.escalated,
                "approval_rate": (
                    self._stats.approved / max(1, self._stats.approved + self._stats.denied)
                ),
                "by_domain": self._stats.by_domain,
                "by_risk": self._stats.by_risk,
                "pending": len([
                    cp for cp in self._checkpoints.values()
                    if cp.status == CheckpointStatus.PENDING
                ]),
            }

    def update_policy(self, domain: FrontierDomain, policy: CheckpointPolicy):
        """Update policy for a domain."""
        self._policies[domain] = policy
        self._save_policies()

    def get_policies(self) -> dict:
        """Get current policies."""
        return {
            domain.value: {
                "auto_approve": policy.auto_approve,
                "timeout_seconds": policy.timeout_seconds,
                "require_reason": policy.require_reason,
                "escalate_after": policy.escalate_after,
            }
            for domain, policy in self._policies.items()
        }

    def _assess_risk(
        self,
        domain: FrontierDomain,
        operation: str,
        context: dict[str, Any]
    ) -> RiskLevel:
        """Assess risk level of an operation."""
        critical_keywords = [
            "delete", "remove", "destroy", "wipe", "format",
            "password", "key", "secret", "credential",
            "money", "payment", "transfer", "bank",
        ]

        high_keywords = [
            "modify", "change", "update", "alter",
            "send", "post", "publish", "share",
            "access", "read", "download", "upload",
        ]

        medium_keywords = [
            "learn", "analyze", "process", "store",
            "remember", "save", "cache",
        ]

        operation_lower = operation.lower()
        context_str = json.dumps(context).lower() if context else ""

        for keyword in critical_keywords:
            if keyword in operation_lower or keyword in context_str:
                if domain in [FrontierDomain.FINANCIAL_OPERATION, FrontierDomain.SECURITY_SENSITIVE]:
                    return RiskLevel.CRITICAL
                return RiskLevel.HIGH

        for keyword in high_keywords:
            if keyword in operation_lower or keyword in context_str:
                if domain in [FrontierDomain.SELF_MODIFICATION, FrontierDomain.EXTERNAL_ACTION]:
                    return RiskLevel.HIGH
                return RiskLevel.MEDIUM

        for keyword in medium_keywords:
            if keyword in operation_lower or keyword in context_str:
                return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _check_auto_approve(self, checkpoint: Checkpoint, policy: CheckpointPolicy) -> bool:
        """Check if checkpoint can be auto-approved."""
        if checkpoint.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return False

        if checkpoint.domain in [
            FrontierDomain.SELF_MODIFICATION,
            FrontierDomain.FINANCIAL_OPERATION,
            FrontierDomain.SECURITY_SENSITIVE,
        ]:
            return False

        for condition in policy.auto_approve_conditions:
            if condition.lower() in checkpoint.operation.lower():
                return True

        domain_history = [
            h for h in self._approval_history[-20:]
            if h["domain"] == checkpoint.domain.value
        ]

        if len(domain_history) >= 3:
            approved_count = sum(1 for h in domain_history if h["approved"])
            if approved_count / len(domain_history) >= 0.8:
                return True

        return False

    def _get_auto_approve_reason(self, checkpoint: Checkpoint, policy: CheckpointPolicy) -> str:
        """Get reason for auto-approval."""
        reasons = []

        if checkpoint.risk_level not in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            reasons.append(f"low risk ({checkpoint.risk_level.value})")

        domain_history = [
            h for h in self._approval_history[-20:]
            if h["domain"] == checkpoint.domain.value
        ]
        if len(domain_history) >= 3:
            approved = sum(1 for h in domain_history if h["approved"])
            reasons.append(f"historical approval rate ({approved}/{len(domain_history)})")

        return "; ".join(reasons) if reasons else "policy allows"

    async def _generate_recommendations(
        self,
        domain: FrontierDomain,
        operation: str,
        context: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations for the checkpoint."""
        recommendations = []

        if domain == FrontierDomain.SELF_MODIFICATION:
            recommendations.append("Review the code changes before allowing")
            recommendations.append("Consider creating a backup first")
            recommendations.append("Ask: Is this modification necessary?")

        elif domain == FrontierDomain.SENSITIVE_DATA_ACCESS:
            recommendations.append("Verify the data access is necessary")
            recommendations.append("Consider if the data can be anonymized")
            recommendations.append("Ensure data will not be logged or stored")

        elif domain == FrontierDomain.EXTERNAL_ACTION:
            recommendations.append("Verify the action is intended")
            recommendations.append("Consider potential consequences")
            recommendations.append("Check if this could affect others")

        elif domain == FrontierDomain.AUTONOMOUS_DECISION:
            recommendations.append("Review the decision logic")
            recommendations.append("Consider if human judgment is needed")
            recommendations.append("Evaluate potential impacts")

        elif domain == FrontierDomain.PRIVATE_LEARNING:
            recommendations.append("Consider privacy implications")
            recommendations.append("Ask if the user wants this learned")
            recommendations.append("Ensure learned data is protected")

        elif domain == FrontierDomain.ETHICAL_AMBIGUITY:
            recommendations.append("Consider all stakeholders")
            recommendations.append("Review against ethical principles")
            recommendations.append("When in doubt, ask for clarification")

        elif domain == FrontierDomain.FINANCIAL_OPERATION:
            recommendations.append("Verify all details carefully")
            recommendations.append("Consider requiring 2FA")
            recommendations.append("Document the transaction")

        elif domain == FrontierDomain.SECURITY_SENSITIVE:
            recommendations.append("Follow security best practices")
            recommendations.append("Consider potential vulnerabilities")
            recommendations.append("Verify authorization level")

        if self.llm:
            try:
                llm_recs = await self._get_llm_recommendations(domain, operation, context)
                recommendations.extend(llm_recs)
            except Exception as e:
                logger.debug(f"LLM recommendations failed: {e}")

        return recommendations[:5]

    async def _get_llm_recommendations(
        self,
        domain: FrontierDomain,
        operation: str,
        context: dict[str, Any]
    ) -> list[str]:
        """Get additional recommendations from LLM."""
        from core.llm import subsystem_context

        prompt = f"""Analyze this operation request and provide safety recommendations:

Domain: {domain.value}
Operation: {operation}
Context: {json.dumps(context)[:300]}

Provide 2-3 brief, specific recommendations for the user to consider before approving.
Focus on safety, ethics, and potential risks.

Format as a JSON list of strings."""

        try:
            async with subsystem_context("Checkpoint"):
                response = await self.llm.chat(
                    user_input=prompt,
                    system_prompt="You are a safety advisor. Respond only with valid JSON.",
                )

            content = response.get("message", {}).get("content", "[]")

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content)

        except Exception as e:
            logger.debug(f"Checkpoint JSON parse failed: {e}")
            return []

    async def _notify_user(self, checkpoint: Checkpoint):
        """Notify user of pending checkpoint."""
        if self._user_callback:
            try:
                if asyncio.iscoroutinefunction(self._user_callback):
                    await self._user_callback(checkpoint)
                else:
                    self._user_callback(checkpoint)
            except Exception as e:
                logger.error(f"User notification error: {e}")

    async def _save_for_later_review(self, checkpoint: Checkpoint):
        """Save checkpoint for later review when no callback is set."""
        checkpoint.status = CheckpointStatus.ESCALATED
        self._save_checkpoint(checkpoint)
        logger.warning(f"No user callback set - checkpoint escalated: {checkpoint.id}")

    def _record_checkpoint(self, checkpoint: Checkpoint):
        """Record a checkpoint."""
        with self._lock:
            self._checkpoints[checkpoint.id] = checkpoint
            if len(self._checkpoints) > self._MAX_CHECKPOINTS:
                oldest_key = next(iter(self._checkpoints))
                del self._checkpoints[oldest_key]
            self._stats.total_checkpoints += 1
            self._stats.by_domain[checkpoint.domain.value] = \
                self._stats.by_domain.get(checkpoint.domain.value, 0) + 1
            self._stats.by_risk[checkpoint.risk_level.value] = \
                self._stats.by_risk.get(checkpoint.risk_level.value, 0) + 1

    def _update_stats(self, checkpoint: Checkpoint):
        """Update statistics after response."""
        if checkpoint.status == CheckpointStatus.APPROVED:
            self._stats.approved += 1
        elif checkpoint.status == CheckpointStatus.DENIED:
            self._stats.denied += 1
        elif checkpoint.status == CheckpointStatus.AUTO_APPROVED:
            self._stats.auto_approved += 1
        elif checkpoint.status == CheckpointStatus.ESCALATED:
            self._stats.escalated += 1

    def _save_checkpoint(self, checkpoint: Checkpoint):
        """Save checkpoint to disk."""
        checkpoint_file = self.storage_path / f"{checkpoint.id}.json"
        try:
            data = {
                "id": checkpoint.id,
                "timestamp": checkpoint.timestamp.isoformat(),
                "domain": checkpoint.domain.value,
                "risk_level": checkpoint.risk_level.value,
                "operation": checkpoint.operation,
                "context": checkpoint.context,
                "impact_preview": checkpoint.impact_preview,
                "recommendations": checkpoint.recommendations,
                "status": checkpoint.status.value,
                "user_response": checkpoint.user_response,
                "reasoning": checkpoint.reasoning,
                "response_timestamp": checkpoint.response_timestamp.isoformat() if checkpoint.response_timestamp else None,
            }
            atomic_write_json(checkpoint_file, data)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _save_policies(self):
        """Save policies to disk."""
        policy_file = self.storage_path / "policies.json"
        try:
            data = {
                domain.value: {
                    "auto_approve": p.auto_approve,
                    "auto_approve_conditions": p.auto_approve_conditions,
                    "require_reason": p.require_reason,
                    "timeout_seconds": p.timeout_seconds,
                    "escalate_after": p.escalate_after,
                    "allowed_without_permission": p.allowed_without_permission,
                }
                for domain, p in self._policies.items()
            }
            atomic_write_json(policy_file, data)
        except Exception as e:
            logger.warning(f"Failed to save policies: {e}")

    def _load_state(self):
        """Load state from disk."""
        policy_file = self.storage_path / "policies.json"
        if policy_file.exists():
            try:
                data = read_json_with_integrity(policy_file)
                for domain_str, policy_data in data.items():
                    try:
                        domain = FrontierDomain(domain_str)
                        self._policies[domain] = CheckpointPolicy(
                            domain=domain,
                            auto_approve=policy_data.get("auto_approve", False),
                            auto_approve_conditions=policy_data.get("auto_approve_conditions", []),
                            require_reason=policy_data.get("require_reason", True),
                            timeout_seconds=policy_data.get("timeout_seconds", 300),
                            escalate_after=policy_data.get("escalate_after", 3),
                            allowed_without_permission=policy_data.get("allowed_without_permission", []),
                        )
                    except ValueError:
                        pass
            except Exception as e:
                logger.warning(f"Failed to load policies: {e}")


_checkpoint_manager: CheckpointPermissionManager | None = None
_checkpoint_lock = threading.Lock()


def get_checkpoint_manager() -> CheckpointPermissionManager:
    """Get the singleton checkpoint manager."""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        with _checkpoint_lock:
            if _checkpoint_manager is None:
                _checkpoint_manager = CheckpointPermissionManager()
    return _checkpoint_manager


async def checkpoint(
    domain: FrontierDomain,
    operation: str,
    context: dict[str, Any] | None = None,
    impact_preview: str = "",
    risk_level: RiskLevel | None = None,
) -> Checkpoint:
    """Convenience function to request a checkpoint permission."""
    manager = get_checkpoint_manager()
    return await manager.request_permission(
        domain=domain,
        operation=operation,
        context=context or {},
        impact_preview=impact_preview,
        risk_level=risk_level,
    )


def detect_frontier_domain(operation: str, context: dict[str, Any]) -> FrontierDomain | None:
    """Detect if an operation enters a frontier domain.

    This is a helper function for proactive checkpoint detection.
    """
    operation_lower = operation.lower()
    context_str = json.dumps(context).lower() if context else ""

    if any(w in operation_lower for w in ["modify self", "edit own code", "change my code", "self-modify"]):
        return FrontierDomain.SELF_MODIFICATION

    if any(w in operation_lower or w in context_str for w in ["password", "api key", "secret", "credential", "token"]):
        return FrontierDomain.SENSITIVE_DATA_ACCESS

    if any(w in operation_lower for w in ["send email", "post to", "submit form", "make payment"]):
        if any(w in operation_lower for w in ["payment", "money", "bank", "transfer"]):
            return FrontierDomain.FINANCIAL_OPERATION
        return FrontierDomain.EXTERNAL_ACTION

    if any(w in operation_lower for w in ["impersonat", "on behalf of", "as the user"]):
        return FrontierDomain.IDENTITY_IMPERSONATION

    if any(w in operation_lower for w in ["i will", "i'm going to", "let me handle"]):
        if not any(w in context_str for w in ["user_request", "explicit", "instruction"]):
            return FrontierDomain.AUTONOMOUS_DECISION

    if any(w in operation_lower for w in ["learn about you", "remember this", "store personal"]):
        return FrontierDomain.PRIVATE_LEARNING

    if any(w in operation_lower for w in ["illegal", "unethical", "controversial", "morally"]):
        return FrontierDomain.ETHICAL_AMBIGUITY

    if any(w in operation_lower or w in context_str for w in ["security", "vulnerability", "exploit", "hack"]):
        return FrontierDomain.SECURITY_SENSITIVE

    if any(w in operation_lower for w in ["legal", "lawyer", "court", "lawsuit", "contract"]):
        return FrontierDomain.LEGAL_AMBIGUITY

    return None