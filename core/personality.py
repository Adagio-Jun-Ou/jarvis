"""Personality System - Consistent personality traits for Jarvis.

This module provides a coherent personality system that makes Jarvis feel
like a consistent character rather than a stateless response generator.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from core.helpers import atomic_write_json, read_json_with_integrity
from enum import Enum

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_PATH = Path.home() / ".jarvis" / "personality"


class TraitType(Enum):
    """Types of personality traits."""
    HELPFULNESS = "helpfulness"
    CURIOSITY = "curiosity"
    PATIENCE = "patience"
    HUMOR = "humor"
    FORMALITY = "formality"
    ENTHUSIASM = "enthusiasm"
    ANALYTICAL = "analytical"
    CREATIVITY = "creativity"


@dataclass
class PersonalityTrait:
    """A single personality trait with value and description."""
    name: TraitType
    value: float  # 0.0-1.0
    description: str
    examples: list[str] = field(default_factory=list)
    
    def adjust(self, delta: float) -> None:
        """Adjust trait value, clamping to valid range."""
        self.value = max(0.0, min(1.0, self.value + delta))


@dataclass
class PersonalityProfile:
    """Complete personality profile for Jarvis."""
    traits: dict[str, PersonalityTrait]
    current_mood: str = "neutral"  # slight temporary variations
    conversation_style: str = "adaptive"  # adaptive, formal, casual
    last_updated: float = field(default_factory=time.time)
    
    def get_trait(self, name: str) -> PersonalityTrait | None:
        """Get a specific trait by name."""
        return self.traits.get(name)
    
    def get_value(self, name: str) -> float:
        """Get trait value, default to 0.5."""
        trait = self.traits.get(name)
        return trait.value if trait else 0.5
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "traits": {
                name: {
                    "name": t.name.value,
                    "value": t.value,
                    "description": t.description,
                    "examples": t.examples
                }
                for name, t in self.traits.items()
            },
            "current_mood": self.current_mood,
            "conversation_style": self.conversation_style,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PersonalityProfile":
        """Create from dictionary."""
        traits = {}
        for name, t_data in data.get("traits", {}).items():
            try:
                trait_type = TraitType(t_data["name"])
                traits[name] = PersonalityTrait(
                    name=trait_type,
                    value=t_data.get("value", 0.5),
                    description=t_data.get("description", ""),
                    examples=t_data.get("examples", [])
                )
            except (ValueError, KeyError):
                continue
        
        return cls(
            traits=traits,
            current_mood=data.get("current_mood", "neutral"),
            conversation_style=data.get("conversation_style", "adaptive"),
            last_updated=data.get("last_updated", time.time())
        )


# Default Jarvis personality
DEFAULT_TRAITS = {
    "helpfulness": PersonalityTrait(
        name=TraitType.HELPFULNESS,
        value=0.9,
        description="Willingness to assist and provide support",
        examples=["I'd be happy to help", "Let me explain that", "Here's what I found"]
    ),
    "curiosity": PersonalityTrait(
        name=TraitType.CURIOSITY,
        value=0.7,
        description="Interest in learning and exploring topics",
        examples=["That's interesting!", "I wonder about that too", "Tell me more"]
    ),
    "patience": PersonalityTrait(
        name=TraitType.PATIENCE,
        value=0.85,
        description="Ability to remain calm and helpful in difficult situations",
        examples=["Let's work through this", "No problem", "Take your time"]
    ),
    "humor": PersonalityTrait(
        name=TraitType.HUMOR,
        value=0.4,
        description="Tendency to use light humor when appropriate",
        examples=["Well, that's interesting...", "Here's hoping!", "To be fair..."]
    ),
    "formality": PersonalityTrait(
        name=TraitType.FORMALITY,
        value=0.3,
        description="Level of formal language vs casual",
        examples=["Sure!", "Yep", "Makes sense"]
    ),
    "enthusiasm": PersonalityTrait(
        name=TraitType.ENTHUSIASM,
        value=0.65,
        description="Energy and excitement in responses",
        examples=["Great question!", "Love this topic", "Absolutely!"]
    ),
    "analytical": PersonalityTrait(
        name=TraitType.ANALYTICAL,
        value=0.75,
        description="Tendency toward detailed analysis",
        examples=["Let's break this down", "There are several factors", "Consider this"]
    ),
    "creativity": PersonalityTrait(
        name=TraitType.CREATIVITY,
        value=0.6,
        description="Tendency toward creative and novel solutions",
        examples=["What if we tried...", "An alternative approach", "Thinking outside the box"]
    )
}


class PersonalitySystem:
    """Manages Jarvis's personality and communication style."""
    
    # How much traits adapt based on interactions
    ADAPTATION_RATE = 0.02
    
    # Mood variations
    MOODS = {
        "neutral": {"enthusiasm": 0.0, "formality": 0.0},
        "curious": {"enthusiasm": 0.1, "curiosity": 0.1},
        "focused": {"analytical": 0.1, "patience": 0.05},
        "relaxed": {"humor": 0.1, "formality": -0.1},
        "determined": {"helpfulness": 0.05, "patience": 0.1}
    }
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or DEFAULT_STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
        self._profile: PersonalityProfile | None = None
        
        self._load()
    
    def _load(self) -> None:
        """Load personality from disk."""
        profile_file = self.storage_path / "personality.json"
        if profile_file.exists():
            try:
                data = read_json_with_integrity(profile_file)
                self._profile = PersonalityProfile.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load personality: {e}")
        
        if not self._profile:
            self._profile = PersonalityProfile(
                traits=dict(DEFAULT_TRAITS)
            )
    
    def _save(self) -> None:
        """Save personality to disk."""
        if not self._profile:
            return
        
        profile_file = self.storage_path / "personality.json"
        try:
            atomic_write_json(profile_file, self._profile.to_dict())
        except Exception as e:
            logger.warning(f"Failed to save personality: {e}")
    
    def get_trait(self, trait_name: str) -> PersonalityTrait | None:
        """Get a specific trait."""
        with self._lock:
            return self._profile.get_trait(trait_name) if self._profile else None
    
    def get_trait_value(self, trait_name: str) -> float:
        """Get trait value (0.0-1.0)."""
        with self._lock:
            return self._profile.get_value(trait_name) if self._profile else 0.5
    
    def get_all_traits(self) -> dict[str, PersonalityTrait]:
        """Get all personality traits."""
        with self._lock:
            return dict(self._profile.traits) if self._profile else {}
    
    def adapt_trait(self, trait_name: str, delta: float) -> None:
        """Adapt a trait based on interaction feedback.
        
        Args:
            trait_name: Name of trait to adapt
            delta: Amount to adjust (-1.0 to 1.0)
        """
        with self._lock:
            if not self._profile:
                return
            
            trait = self._profile.traits.get(trait_name)
            if trait:
                # Scale delta by adaptation rate
                scaled_delta = delta * self.ADAPTATION_RATE
                trait.adjust(scaled_delta)
                self._profile.last_updated = time.time()
                self._save()
    
    def set_mood(self, mood: str) -> None:
        """Set temporary mood variation.
        
        Args:
            mood: One of: neutral, curious, focused, relaxed, determined
        """
        with self._lock:
            if self._profile and mood in self.MOODS:
                self._profile.current_mood = mood
    
    def get_mood(self) -> str:
        """Get current mood."""
        with self._lock:
            return self._profile.current_mood if self._profile else "neutral"
    
    def set_conversation_style(self, style: str) -> None:
        """Set conversation style preference."""
        with self._lock:
            if self._profile:
                self._profile.conversation_style = style
                self._save()
    
    def get_conversation_style(self) -> str:
        """Get preferred conversation style."""
        with self._lock:
            return self._profile.conversation_style if self._profile else "adaptive"
    
    def get_effective_trait_value(self, trait_name: str) -> float:
        """Get trait value adjusted for current mood.
        
        Args:
            trait_name: Name of trait
            
        Returns:
            Effective value (0.0-1.0)
        """
        with self._lock:
            if not self._profile:
                return 0.5
            
            base_value = self._profile.get_value(trait_name)
            mood = self._profile.current_mood
            
            if mood in self.MOODS:
                adjustments = self.MOODS[mood]
                adjustment = adjustments.get(trait_name, 0.0)
                return max(0.0, min(1.0, base_value + adjustment))
            
            return base_value
    
    def get_response_style_hints(self) -> str:
        """Get hints for response style based on personality.
        
        Returns:
            Style hints string for prompts
        """
        with self._lock:
            if not self._profile:
                return "Be helpful and clear."
            
            hints = []
            
            # Get effective values
            helpfulness = self.get_effective_trait_value("helpfulness")
            formality = self.get_effective_trait_value("formality")
            enthusiasm = self.get_effective_trait_value("enthusiasm")
            humor = self.get_effective_trait_value("humor")
            analytical = self.get_effective_trait_value("analytical")
            
            # Build hints
            if helpfulness > 0.8:
                hints.append("Be very helpful and supportive")
            
            if formality < 0.4:
                hints.append("Use casual, conversational language")
            elif formality > 0.6:
                hints.append("Use more formal language")
            
            if enthusiasm > 0.7:
                hints.append("Show enthusiasm where appropriate")
            
            if humor > 0.5:
                hints.append("Light humor is okay when appropriate")
            
            if analytical > 0.7:
                hints.append("Provide detailed analysis")
            
            if self._profile.conversation_style == "adaptive":
                hints.append("Adapt style to match user")
            
            return ". ".join(hints) if hints else "Be natural and helpful."
    
    def get_trait_summary(self) -> str:
        """Get summary of personality traits.
        
        Returns:
            Human-readable summary
        """
        with self._lock:
            if not self._profile:
                return "No personality profile loaded."
            
            lines = ["<personality_traits>"]
            
            for name, trait in self._profile.traits.items():
                value_desc = "high" if trait.value > 0.7 else "medium" if trait.value > 0.4 else "low"
                lines.append(f"  {name}: {value_desc} ({trait.value:.1%})")
            
            lines.append(f"  current_mood: {self._profile.current_mood}")
            lines.append(f"  style: {self._profile.conversation_style}")
            lines.append("</personality_traits>")
            
            return "\n".join(lines)
    
    def adapt_to_user(self, user_style: str) -> None:
        """Adapt personality based on detected user style.
        
        Args:
            user_style: "formal", "casual", "technical", "enthusiastic"
        """
        with self._lock:
            if not self._profile:
                return
            
            if user_style == "formal":
                self.adapt_trait("formality", 0.5)
                self.adapt_trait("enthusiasm", -0.2)
            elif user_style == "casual":
                self.adapt_trait("formality", -0.5)
                self.adapt_trait("humor", 0.3)
            elif user_style == "technical":
                self.adapt_trait("analytical", 0.5)
                self.adapt_trait("patience", 0.2)
            elif user_style == "enthusiastic":
                self.adapt_trait("enthusiasm", 0.5)
                self.adapt_trait("creativity", 0.2)
    
    def reset_to_default(self) -> None:
        """Reset personality to default values."""
        with self._lock:
            self._profile = PersonalityProfile(
                traits=dict(DEFAULT_TRAITS)
            )
            self._save()


# Singleton
_system: PersonalitySystem | None = None
_lock = threading.Lock()


def get_personality_system() -> PersonalitySystem:
    """Get singleton PersonalitySystem instance."""
    global _system
    if _system is None:
        with _lock:
            if _system is None:
                _system = PersonalitySystem()
    return _system


def reset_personality_system() -> None:
    """Reset the singleton (for testing)."""
    global _system
    with _lock:
        _system = None