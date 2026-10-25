"""Claims System - Work ownership and coordination for multi-agent systems.

Provides claim-based ownership of tasks, files, and resources to prevent
conflicts in multi-agent environments. Implements Byzantine fault tolerance
for distributed consensus.
"""
import hashlib
import logging
import threading
import time
from collections import defaultdict, deque
from core.helpers import atomic_write_json, read_json_with_integrity
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    FILE = "file"
    TASK = "task"
    RESOURCE = "resource"
    MODULE = "module"
    KNOWLEDGE = "knowledge"


class ClaimStatus(Enum):
    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"
    CONTESTED = "contested"
    REVOKED = "revoked"


class ClaimPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Claim:
    id: str
    claim_type: ClaimType
    resource_id: str
    agent_id: str
    status: ClaimStatus = ClaimStatus.ACTIVE
    priority: ClaimPriority = ClaimPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    heartbeat_at: datetime = field(default_factory=datetime.now)
    signatures: list[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        if self.status != ClaimStatus.ACTIVE:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.claim_type.value,
            "resource": self.resource_id,
            "agent": self.agent_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "created": self.created_at.isoformat(),
            "expires": self.expires_at.isoformat() if self.expires_at else None,
            "heartbeat": self.heartbeat_at.isoformat(),
            "signatures": len(self.signatures),
        }


@dataclass
class ClaimProposal:
    proposal_id: str
    claim: Claim
    proposer_id: str
    votes: dict[str, bool] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    quorum: int = 3

    def has_quorum(self) -> bool:
        return len(self.votes) >= self.quorum

    def is_approved(self) -> bool:
        if not self.has_quorum():
            return False
        approvals = sum(1 for v in self.votes.values() if v)
        return approvals > len(self.votes) / 2


class ByzantineConsensus:
    """Byzantine fault-tolerant consensus for claim validation."""

    MINIMUM_NODES = 4
    MAXIMUM_BYZANTINE = 1

    MAX_PENDING_PROPOSALS = 100

    def __init__(self, node_id: str):
        self.node_id = node_id
        self._pending_proposals: dict[str, ClaimProposal] = {}
        self._node_registry: set[str] = set()
        self._lock = threading.RLock()

    def register_node(self, node_id: str) -> None:
        with self._lock:
            self._node_registry.add(node_id)

    def unregister_node(self, node_id: str) -> None:
        with self._lock:
            self._node_registry.discard(node_id)

    def create_proposal(self, claim: Claim, quorum: int = 3) -> ClaimProposal:
        with self._lock:
            proposal_id = hashlib.sha256(
                f"{claim.id}:{claim.agent_id}:{time.time()}".encode()
            ).hexdigest()[:16]

            proposal = ClaimProposal(
                proposal_id=proposal_id,
                claim=claim,
                proposer_id=self.node_id,
                quorum=quorum,
            )

            self._pending_proposals[proposal_id] = proposal
            # Evict oldest proposals if unbounded
            while len(self._pending_proposals) > self.MAX_PENDING_PROPOSALS:
                oldest_pid = next(iter(self._pending_proposals))
                del self._pending_proposals[oldest_pid]
            return proposal

    def vote(self, proposal_id: str, voter_id: str, approve: bool) -> bool:
        with self._lock:
            if proposal_id not in self._pending_proposals:
                return False

            proposal = self._pending_proposals[proposal_id]
            proposal.votes[voter_id] = approve

            if proposal.has_quorum():
                if proposal.is_approved():
                    proposal.claim.status = ClaimStatus.ACTIVE
                else:
                    proposal.claim.status = ClaimStatus.REVOKED

            return True

    def get_proposal(self, proposal_id: str) -> ClaimProposal | None:
        with self._lock:
            return self._pending_proposals.get(proposal_id)

    def get_decision(self, proposal_id: str) -> dict | None:
        """Get the current decision status for a proposal.
        
        Returns:
            Dict with 'approved', 'votes', 'quorum_reached', 'vote_count', or None if not found.
        """
        with self._lock:
            proposal = self._pending_proposals.get(proposal_id)
            if proposal is None:
                return None
            
            return {
                "approved": proposal.is_approved() if proposal.has_quorum() else None,
                "votes": dict(proposal.votes),
                "quorum_reached": proposal.has_quorum(),
                "vote_count": len(proposal.votes),
                "quorum": proposal.quorum,
            }

    def cleanup_expired_proposals(self, max_age_minutes: int = 30) -> int:
        with self._lock:
            cutoff = datetime.now() - timedelta(minutes=max_age_minutes)
            expired = [
                pid for pid, p in self._pending_proposals.items()
                if p.created_at < cutoff
            ]
            for pid in expired:
                del self._pending_proposals[pid]
            return len(expired)


class ClaimsRegistry:
    """Registry for managing claims on resources."""

    DEFAULT_TTL_MINUTES = 30
    MAX_CLAIMS = 1000
    MAX_HISTORY = 500

    def __init__(self, storage_path: Path | None = None, agent_id: str = "jarvis"):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "claims"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.agent_id = agent_id

        self._claims: dict[str, Claim] = {}
        self._resource_claims: dict[str, str] = {}
        self._agent_claims: dict[str, list[str]] = defaultdict(list)
        self._claim_history: deque[dict] = deque(maxlen=self.MAX_HISTORY)
        self._MAX_AGENT_CLAIMS = 100
        self._MAX_CLAIM_HISTORY = 200

        self._claim_counter = 0
        self._lock = threading.RLock()

        self._consensus = ByzantineConsensus(agent_id)

        self._load_state()

    def _load_state(self) -> None:
        state_file = self.storage_path / "claims_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self._claim_counter = data.get("counter", 0)
                logger.info(f"Loaded claims state: {self._claim_counter} claims created")
            except Exception as e:
                logger.warning(f"Failed to load claims state: {e}")

    def _save_state(self) -> None:
        state_file = self.storage_path / "claims_state.json"
        try:
            atomic_write_json(state_file, {
                    "counter": self._claim_counter,
                    "active_claims": len(self._claims),
                    "last_updated": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.warning(f"Failed to save claims state: {e}")

    def claim(
        self,
        claim_type: ClaimType,
        resource_id: str,
        priority: ClaimPriority = ClaimPriority.NORMAL,
        ttl_minutes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Claim | None:
        with self._lock:
            if resource_id in self._resource_claims:
                existing_id = self._resource_claims[resource_id]
                existing = self._claims.get(existing_id)
                if existing and existing.is_valid():
                    if existing.agent_id == self.agent_id:
                        existing.heartbeat_at = datetime.now()
                        return existing
                    return None

            if len(self._claims) >= self.MAX_CLAIMS:
                self._evict_expired_claims()
                if len(self._claims) >= self.MAX_CLAIMS:
                    self._evict_low_priority_claims(10)

            self._claim_counter += 1
            claim_id = f"claim_{self._claim_counter:06d}"

            ttl = ttl_minutes or self.DEFAULT_TTL_MINUTES
            expires_at = datetime.now() + timedelta(minutes=ttl)

            claim = Claim(
                id=claim_id,
                claim_type=claim_type,
                resource_id=resource_id,
                agent_id=self.agent_id,
                priority=priority,
                expires_at=expires_at,
                metadata=metadata or {},
            )

            self._claims[claim_id] = claim
            self._resource_claims[resource_id] = claim_id
            self._agent_claims[self.agent_id].append(claim_id)

            self._save_state()

            logger.info(f"Claimed {claim_type.value}:{resource_id} for {self.agent_id}")

            return claim

    def release(self, claim_id: str) -> bool:
        with self._lock:
            if claim_id not in self._claims:
                return False

            claim = self._claims[claim_id]

            if claim.agent_id != self.agent_id:
                return False

            claim.status = ClaimStatus.RELEASED

            if claim.resource_id in self._resource_claims:
                del self._resource_claims[claim.resource_id]

            self._add_to_history(claim, "released")

            del self._claims[claim_id]

            self._save_state()

            logger.info(f"Released claim {claim_id}")

            return True

    def heartbeat(self, claim_id: str) -> bool:
        with self._lock:
            if claim_id not in self._claims:
                return False

            claim = self._claims[claim_id]
            claim.heartbeat_at = datetime.now()

            return True

    def renew(self, claim_id: str, ttl_minutes: int | None = None) -> bool:
        with self._lock:
            if claim_id not in self._claims:
                return False

            claim = self._claims[claim_id]

            if claim.agent_id != self.agent_id:
                return False

            ttl = ttl_minutes or self.DEFAULT_TTL_MINUTES
            claim.expires_at = datetime.now() + timedelta(minutes=ttl)
            claim.heartbeat_at = datetime.now()

            return True

    def contest(
        self,
        claim_id: str,
        reason: str,
        require_consensus: bool = True,
    ) -> ClaimProposal | None:
        with self._lock:
            if claim_id not in self._claims:
                return None

            claim = self._claims[claim_id]
            claim.status = ClaimStatus.CONTESTED
            claim.metadata["contest_reason"] = reason

            if require_consensus:
                proposal = self._consensus.create_proposal(claim)
                logger.info(f"Contested claim {claim_id}, proposal {proposal.proposal_id}")
                return proposal

            return None

    def vote_on_contest(
        self,
        proposal_id: str,
        voter_id: str,
        approve: bool,
    ) -> bool:
        with self._lock:
            return self._consensus.vote(proposal_id, voter_id, approve)

    def get_claim(self, claim_id: str) -> Claim | None:
        with self._lock:
            return self._claims.get(claim_id)

    def get_resource_claim(self, resource_id: str) -> Claim | None:
        with self._lock:
            claim_id = self._resource_claims.get(resource_id)
            if claim_id:
                return self._claims.get(claim_id)
            return None

    def is_claimed(self, resource_id: str) -> bool:
        with self._lock:
            claim_id = self._resource_claims.get(resource_id)
            if claim_id:
                claim = self._claims.get(claim_id)
                return claim is not None and claim.is_valid()
            return False

    def get_agent_claims(self, agent_id: str | None = None) -> list[Claim]:
        with self._lock:
            aid = agent_id or self.agent_id
            claim_ids = self._agent_claims.get(aid, [])
            # Clean up stale claim IDs
            valid_ids = [cid for cid in claim_ids if cid in self._claims]
            if len(valid_ids) < len(claim_ids):
                self._agent_claims[aid] = valid_ids
            if len(self._agent_claims[aid]) > self._MAX_AGENT_CLAIMS:
                self._agent_claims[aid] = self._agent_claims[aid][-self._MAX_AGENT_CLAIMS:]
            return [
                self._claims[cid]
                for cid in valid_ids
                if cid in self._claims
            ]

    def cleanup_expired(self) -> int:
        with self._lock:
            return self._evict_expired_claims()

    def _evict_expired_claims(self) -> int:
        expired = [
            cid for cid, claim in self._claims.items()
            if not claim.is_valid()
        ]

        for cid in expired:
            claim = self._claims[cid]
            claim.status = ClaimStatus.EXPIRED
            self._add_to_history(claim, "expired")

            if claim.resource_id in self._resource_claims:
                del self._resource_claims[claim.resource_id]

            del self._claims[cid]

        if expired:
            logger.info(f"Evicted {len(expired)} expired claims")

        return len(expired)

    def _evict_low_priority_claims(self, count: int) -> int:
        sorted_claims = sorted(
            self._claims.items(),
            key=lambda x: (x[1].priority.value, x[1].created_at)
        )

        evicted = 0
        for cid, claim in sorted_claims[:count]:
            if claim.agent_id != self.agent_id:
                continue

            self._add_to_history(claim, "evicted")

            if claim.resource_id in self._resource_claims:
                del self._resource_claims[claim.resource_id]

            del self._claims[cid]
            evicted += 1

        return evicted

    def _add_to_history(self, claim: Claim, action: str) -> None:
        self._claim_history.append({
            "claim_id": claim.id,
            "resource": claim.resource_id,
            "agent": claim.agent_id,
            "action": action,
            "timestamp": datetime.now().isoformat(),
        })

    def claim_file(
        self,
        file_path: str,
        purpose: str,
        ttl_minutes: int | None = None,
    ) -> Claim | None:
        return self.claim(
            ClaimType.FILE,
            file_path,
            metadata={"purpose": purpose},
            ttl_minutes=ttl_minutes,
        )

    def claim_task(
        self,
        task_id: str,
        description: str,
        priority: ClaimPriority = ClaimPriority.NORMAL,
    ) -> Claim | None:
        return self.claim(
            ClaimType.TASK,
            task_id,
            priority=priority,
            metadata={"description": description},
        )

    def claim_resource(
        self,
        resource_name: str,
        reason: str,
    ) -> Claim | None:
        return self.claim(
            ClaimType.RESOURCE,
            resource_name,
            ClaimPriority.NORMAL,
            metadata={"reason": reason},
        )

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            by_type: dict[str, int] = defaultdict(int)
            by_status: dict[str, int] = defaultdict(int)
            by_agent: dict[str, int] = defaultdict(int)

            for claim in self._claims.values():
                by_type[claim.claim_type.value] += 1
                by_status[claim.status.value] += 1
                by_agent[claim.agent_id] += 1

            return {
                "active_claims": len(self._claims),
                "by_type": dict(by_type),
                "by_status": dict(by_status),
                "by_agent": dict(by_agent),
                "history_size": len(self._claim_history),
                "total_created": self._claim_counter,
            }


_claims_registry: ClaimsRegistry | None = None
_claims_lock = threading.Lock()


def get_claims_registry() -> ClaimsRegistry:
    global _claims_registry
    if _claims_registry is None:
        with _claims_lock:
            if _claims_registry is None:
                _claims_registry = ClaimsRegistry()
    return _claims_registry


def claim_resource(
    resource_type: ClaimType,
    resource_id: str,
    **kwargs,
) -> Claim | None:
    return get_claims_registry().claim(resource_type, resource_id, **kwargs)


def release_claim(claim_id: str) -> bool:
    return get_claims_registry().release(claim_id)


def is_resource_claimed(resource_id: str) -> bool:
    return get_claims_registry().is_claimed(resource_id)
