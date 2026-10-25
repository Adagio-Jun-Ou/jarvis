"""System-wide Prefrontal Cortex observer utility.

This module provides a simple async function that all subsystems can call
to report their activity to the Prefrontal Cortex for learning.

Usage in any subsystem:
    from core.pfc_observer import observe_event
    
    await observe_event(
        event_type="insight",
        content="New pattern discovered",
        source="erdos",
        metadata={"problem_id": "...", "strategy": "..."}
    )
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Cache the PFC reference
_pfc = None

async def observe_event(
    event_type: str,
    content: str,
    source: str,
    metadata: Optional[dict] = None,
) -> bool:
    """Report an event to the Prefrontal Cortex.
    
    This is a lightweight async function that all subsystems can call.
    It safely handles PFC availability and errors.
    
    Args:
        event_type: Type of event (insight, discovery, error, etc.)
        content: The content/summary of the event
        source: Which subsystem generated this (erdos, daydream, learning, etc.)
        metadata: Optional additional context
    
    Returns:
        bool: True if observation was recorded, False otherwise
    """
    global _pfc
    
    # Try to get PFC if not cached
    if _pfc is None:
        try:
            from core.prefrontal_cortex import get_prefrontal_cortex
            _pfc = get_prefrontal_cortex()
        except Exception as e:
            logger.debug(f"PFC not available: {e}")
            return False
    
    # Check if PFC is running
    if _pfc is None or not getattr(_pfc, '_running', False):
        return False
    
    # Import concept types
    try:
        from core.prefrontal_cortex import ConceptType
        
        # Map event types to ConceptType
        type_map = {
            "insight": ConceptType.INSIGHT,
            "discovery": ConceptType.INSIGHT,
            "error": ConceptType.ERROR,
            "observation": ConceptType.OBSERVATION,
            "decision": ConceptType.DECISION,
            "memory": ConceptType.MEMORY,
            "subsystem_event": ConceptType.SUBSYSTEM_EVENT,
        }
        
        concept_type = type_map.get(event_type, ConceptType.SUBSYSTEM_EVENT)
        
        # Create the observation
        await _pfc.observe(
            content=content,
            concept_type=concept_type,
            source=source,
            metadata=metadata or {},
        )
        
        return True
        
    except Exception as e:
        logger.debug(f"PFC observation failed: {e}")
        return False


async def observe_subsystem_event(
    subsystem: str,
    event: str,
    details: Optional[dict] = None,
) -> bool:
    """Convenience function for subsystem events.
    
    Args:
        subsystem: Name of subsystem (erdos, daydream, evolution, etc.)
        event: Event description
        details: Optional details dict
    
    Returns:
        bool: True if observation was recorded
    """
    return await observe_event(
        event_type="subsystem_event",
        content=f"{subsystem}: {event}",
        source=f"subsystem:{subsystem}",
        metadata=details,
    )


async def observe_insight(
    insight: str,
    source: str,
    metadata: Optional[dict] = None,
) -> bool:
    """Convenience function for insights.
    
    Args:
        insight: The insight content
        source: Which subsystem generated this
        metadata: Optional context
    
    Returns:
        bool: True if observation was recorded
    """
    return await observe_event(
        event_type="insight",
        content=insight,
        source=source,
        metadata=metadata,
    )
