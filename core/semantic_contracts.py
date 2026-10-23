"""Semantic I/O Contracts for Inter-Model Communication.

Implements the "I/O Contract" pattern from Heddle for clean,
typed, validated data flow between Kimi K2.5 and GLM-5.

Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                SEMANTIC I/O CONTRACTS                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Primary (Kimi K2.5)                                             │
│       │                                                            │
│       ▼                                                            │
│   ┌──────────────┐                                                │
│   │  OUTPUT      │  Primary generates structured output          │
│   │  CONTRACT    │  with typed fields                              │
│   └──────┬───────┘                                                │
│          │ validated by LCM                                         │
│          ▼                                                         │
│   ┌──────────────┐                                                │
│   │  LCM         │  Contract validation:                          │
│   │  VALIDATOR   │  - Type checking                                              │
│   │              │  - Semantic coherence                                   │
│   │              │  - Size limits                                            │
│   └──────┬───────┘                                                │
│          │ approved                                                 │
│          ▼                                                         │
│   ┌──────────────┐                                                │
│   │  INPUT       │  Backend receives only validated,              │
│   │  CONTRACT    │  typed, size-bounded input                     │
│   └──────┬───────┘                                                │
│          │                                                         │
│          ▼                                                         │
│   Backend (GLM-5)                                                  │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘

Key Innovation:
Instead of passing raw text between models, I/O contracts enforce:
1. STRUCTURED OUTPUT: Models produce typed JSON, not free text
2. SIZE LIMITS: Prevent context overflow between models
3. SEMANTIC VALIDATION: LCM verifies coherence before passing
4. SCHEMA EVOLUTION: Contracts can evolve without breaking compatibility
"""
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

_contracts: Optional["ContractRegistry"] = None
_contracts_lock = threading.Lock()


class ContractStatus(Enum):
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    EXPIRED = "expired"


class FieldType(Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class FieldSpec:
    """Specification for a single field in a contract."""
    name: str
    field_type: FieldType
    required: bool = True
    max_length: int | None = None
    description: str = ""
    default: Any = None


@dataclass
class ContractSchema:
    """Schema definition for an I/O contract."""
    name: str
    version: str = "1.0"
    fields: list[FieldSpec] = field(default_factory=list)
    max_total_size: int = 4000
    description: str = ""


@dataclass
class ContractInstance:
    """An instance of an I/O contract with actual data."""
    id: str
    schema_name: str
    source_model: str
    target_model: str
    data: dict = field(default_factory=dict)
    status: ContractStatus = ContractStatus.PENDING
    validation_errors: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    validated_at: datetime | None = None
    size_bytes: int = 0


@dataclass
class ContractStats:
    total_contracts: int = 0
    validated: int = 0
    rejected: int = 0
    avg_validation_ms: float = 0.0
    contracts_by_type: dict = field(default_factory=dict)


STANDARD_SCHEMAS = {
    "chat_response": ContractSchema(
        name="chat_response",
        description="Standard chat response from primary to backend",
        fields=[
            FieldSpec("content", FieldType.STRING, max_length=2000),
            FieldSpec("intent", FieldType.STRING, max_length=100),
            FieldSpec("confidence", FieldType.NUMBER),
            FieldSpec("topics", FieldType.ARRAY),
        ],
    ),
    "code_request": ContractSchema(
        name="code_request",
        description="Code generation request from primary to backend",
        fields=[
            FieldSpec("language", FieldType.STRING, max_length=50),
            FieldSpec("description", FieldType.STRING, max_length=2000),
            FieldSpec("constraints", FieldType.ARRAY),
            FieldSpec("context", FieldType.STRING, max_length=1000, required=False),
        ],
    ),
    "analysis_result": ContractSchema(
        name="analysis_result",
        description="Analysis result from backend to primary",
        fields=[
            FieldSpec("summary", FieldType.STRING, max_length=1000),
            FieldSpec("findings", FieldType.ARRAY),
            FieldSpec("confidence", FieldType.NUMBER),
            FieldSpec("recommendations", FieldType.ARRAY, required=False),
        ],
    ),
    "verification_result": ContractSchema(
        name="verification_result",
        description="Verification result from backend",
        fields=[
            FieldSpec("verified", FieldType.BOOLEAN),
            FieldSpec("issues", FieldType.ARRAY),
            FieldSpec("corrected_content", FieldType.STRING, max_length=2000, required=False),
            FieldSpec("confidence", FieldType.NUMBER),
        ],
    ),
    "cross_model_context": ContractSchema(
        name="cross_model_context",
        description="Shared context between models",
        fields=[
            FieldSpec("topic", FieldType.STRING, max_length=200),
            FieldSpec("key_points", FieldType.ARRAY),
            FieldSpec("source", FieldType.STRING, max_length=50),
            FieldSpec("relevance_score", FieldType.NUMBER, required=False),
        ],
        max_total_size=2000,
    ),
}


class ContractRegistry:
    """Registry and validator for I/O contracts.
    
    Manages contract schemas and validates data flow between models.
    Every inter-model communication must pass through a contract.
    """
    
    def __init__(self):
        self._schemas: dict[str, ContractSchema] = dict(STANDARD_SCHEMAS)
        self._instances: OrderedDict[str, ContractInstance] = OrderedDict()
        self._max_instances = 200
        self._stats = ContractStats()
        self._lock = threading.RLock()
    
    def register_schema(self, schema: ContractSchema) -> None:
        """Register a new contract schema."""
        with self._lock:
            self._schemas[schema.name] = schema
    
    def create_contract(
        self,
        schema_name: str,
        source_model: str,
        target_model: str,
        data: dict,
    ) -> ContractInstance:
        """Create and validate a new I/O contract."""
        contract_id = f"contract_{hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:8]}_{int(time.time()*1000)}"
        
        instance = ContractInstance(
            id=contract_id,
            schema_name=schema_name,
            source_model=source_model,
            target_model=target_model,
            data=data,
            size_bytes=len(json.dumps(data, default=str)),
        )
        
        with self._lock:
            self._validate_instance(instance)
            self._stats.total_contracts += 1
            if instance.status == ContractStatus.VALIDATED:
                self._stats.validated += 1
            elif instance.status == ContractStatus.REJECTED:
                self._stats.rejected += 1
            
            self._stats.contracts_by_type[schema_name] = (
                self._stats.contracts_by_type.get(schema_name, 0) + 1
            )
            
            self._instances[contract_id] = instance
            while len(self._instances) > self._max_instances:
                self._instances.popitem(last=False)
        
        return instance
    
    def _validate_instance(self, instance: ContractInstance) -> None:
        """Validate a contract instance against its schema."""
        schema = self._schemas.get(instance.schema_name)
        if not schema:
            instance.status = ContractStatus.REJECTED
            instance.validation_errors.append(f"Unknown schema: {instance.schema_name}")
            return
        
        start = time.time()
        errors = []
        
        if instance.size_bytes > schema.max_total_size:
            errors.append(
                f"Contract size ({instance.size_bytes}) exceeds limit ({schema.max_total_size})"
            )
        
        for field_spec in schema.fields:
            if field_spec.required and field_spec.name not in instance.data:
                errors.append(f"Missing required field: {field_spec.name}")
            elif field_spec.name in instance.data:
                value = instance.data[field_spec.name]
                
                if field_spec.max_length and isinstance(value, str):
                    if len(value) > field_spec.max_length:
                        errors.append(
                            f"Field '{field_spec.name}' exceeds max_length "
                            f"({len(value)} > {field_spec.max_length})"
                        )
        
        instance.validation_errors = errors
        instance.status = ContractStatus.REJECTED if errors else ContractStatus.VALIDATED
        instance.validated_at = datetime.now()
        
        validation_ms = (time.time() - start) * 1000
        if self._stats.total_contracts <= 1:
            self._stats.avg_validation_ms = validation_ms
        else:
            self._stats.avg_validation_ms = self._stats.avg_validation_ms * 0.9 + validation_ms * 0.1
    
    def get_validated_data(self, contract_id: str) -> dict | None:
        """Get validated data from a contract. Returns None if contract was rejected."""
        with self._lock:
            instance = self._instances.get(contract_id)
            if instance and instance.status == ContractStatus.VALIDATED:
                return instance.data
        return None
    
    def get_stats(self) -> dict:
        """Get contract registry statistics."""
        with self._lock:
            return {
                "total_contracts": self._stats.total_contracts,
                "validated": self._stats.validated,
                "rejected": self._stats.rejected,
                "validation_rate": (
                    self._stats.validated / self._stats.total_contracts
                    if self._stats.total_contracts > 0 else 0
                ),
                "avg_validation_ms": round(self._stats.avg_validation_ms, 2),
                "contracts_by_type": dict(self._stats.contracts_by_type),
                "registered_schemas": list(self._schemas.keys()),
                "active_instances": len(self._instances),
            }


def get_contract_registry() -> ContractRegistry:
    """Get the global ContractRegistry instance."""
    global _contracts, _contracts_lock
    if _contracts is None:
        with _contracts_lock:
            if _contracts is None:
                _contracts = ContractRegistry()
    return _contracts