"""Jarvis Bootstrap Module.

Provides structured initialization with:
- Sequential stage execution
- Dependency ordering
- Health checks per stage
- Rollback on failure
- Progress tracking
"""

from .stages import (
    BootstrapManager,
    BootstrapStage,
    StageResult,
    StageStatus,
    create_default_bootstrap,
    get_bootstrap_manager,
)

__all__ = [
    "BootstrapManager",
    "BootstrapStage",
    "StageResult",
    "StageStatus",
    "create_default_bootstrap",
    "get_bootstrap_manager",
]