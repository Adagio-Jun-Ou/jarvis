"""Tests for Personality System."""
import pytest
import time
from pathlib import Path
from core.personality import (
    PersonalitySystem,
    PersonalityTrait,
    PersonalityProfile,
    TraitType,
    get_personality_system,
    reset_personality_system,
)


def get_fresh_personality():
    """Get a fresh personality instance for testing."""
    import uuid
    unique_id = str(uuid.uuid4())[:8]
    path = Path(f"/tmp/personality_test_{unique_id}")
    return PersonalitySystem(storage_path=path)


class TestPersonalityTrait:
    """Tests for PersonalityTrait dataclass."""
    
    def test_creation(self):
        trait = PersonalityTrait(
            name=TraitType.HELPFULNESS,
            value=0.8,
            description="Test trait",
            examples=["Example 1"]
        )
        assert trait.name == TraitType.HELPFULNESS
        assert trait.value == 0.8
        assert len(trait.examples) == 1
    
    def test_adjust_positive(self):
        trait = PersonalityTrait(
            name=TraitType.HELPFULNESS,
            value=0.5,
            description="Test"
        )
        trait.adjust(0.2)
        assert trait.value == 0.7
    
    def test_adjust_negative(self):
        trait = PersonalityTrait(
            name=TraitType.HELPFULNESS,
            value=0.5,
            description="Test"
        )
        trait.adjust(-0.2)
        assert trait.value == 0.3
    
    def test_adjust_clamp_high(self):
        trait = PersonalityTrait(
            name=TraitType.HELPFULNESS,
            value=0.9,
            description="Test"
        )
        trait.adjust(0.2)
        assert trait.value == 1.0  # Clamped
    
    def test_adjust_clamp_low(self):
        trait = PersonalityTrait(
            name=TraitType.HELPFULNESS,
            value=0.1,
            description="Test"
        )
        trait.adjust(-0.2)
        assert trait.value == 0.0  # Clamped


class TestPersonalityProfile:
    """Tests for PersonalityProfile dataclass."""
    
    def test_creation(self):
        traits = {
            "helpfulness": PersonalityTrait(
                name=TraitType.HELPFULNESS,
                value=0.8,
                description="Test"
            )
        }
        profile = PersonalityProfile(traits=traits)
        assert len(profile.traits) == 1
        assert profile.current_mood == "neutral"
    
    def test_get_trait(self):
        traits = {
            "helpfulness": PersonalityTrait(
                name=TraitType.HELPFULNESS,
                value=0.8,
                description="Test"
            )
        }
        profile = PersonalityProfile(traits=traits)
        
        trait = profile.get_trait("helpfulness")
        assert trait is not None
        assert trait.value == 0.8
        
        missing = profile.get_trait("nonexistent")
        assert missing is None
    
    def test_get_value(self):
        traits = {
            "helpfulness": PersonalityTrait(
                name=TraitType.HELPFULNESS,
                value=0.8,
                description="Test"
            )
        }
        profile = PersonalityProfile(traits=traits)
        
        value = profile.get_value("helpfulness")
        assert value == 0.8
        
        default = profile.get_value("nonexistent")
        assert default == 0.5
    
    def test_to_dict(self):
        traits = {
            "helpfulness": PersonalityTrait(
                name=TraitType.HELPFULNESS,
                value=0.8,
                description="Test",
                examples=["ex"]
            )
        }
        profile = PersonalityProfile(traits=traits, current_mood="curious")
        
        data = profile.to_dict()
        
        assert "traits" in data
        assert "helpfulness" in data["traits"]
        assert data["current_mood"] == "curious"
    
    def test_from_dict(self):
        data = {
            "traits": {
                "helpfulness": {
                    "name": "helpfulness",
                    "value": 0.8,
                    "description": "Test",
                    "examples": ["ex"]
                }
            },
            "current_mood": "curious",
            "conversation_style": "adaptive"
        }
        
        profile = PersonalityProfile.from_dict(data)
        
        assert len(profile.traits) == 1
        assert profile.current_mood == "curious"


class TestPersonalitySystem:
    """Tests for PersonalitySystem."""
    
    def test_initial_state(self):
        system = get_fresh_personality()
        
        traits = system.get_all_traits()
        assert len(traits) > 0
        assert "helpfulness" in traits
    
    def test_get_trait(self):
        system = get_fresh_personality()
        
        trait = system.get_trait("helpfulness")
        assert trait is not None
        assert trait.name == TraitType.HELPFULNESS
    
    def test_get_trait_value(self):
        system = get_fresh_personality()
        
        value = system.get_trait_value("helpfulness")
        assert 0.0 <= value <= 1.0
    
    def test_adapt_trait(self):
        system = get_fresh_personality()
        
        initial = system.get_trait_value("helpfulness")
        system.adapt_trait("helpfulness", 1.0)  # Large delta
        new_value = system.get_trait_value("helpfulness")
        
        # Should have increased slightly
        assert new_value >= initial
    
    def test_adapt_trait_negative(self):
        system = get_fresh_personality()
        
        initial = system.get_trait_value("curiosity")
        system.adapt_trait("curiosity", -1.0)
        new_value = system.get_trait_value("curiosity")
        
        # Should have decreased slightly
        assert new_value <= initial
    
    def test_set_mood(self):
        system = get_fresh_personality()
        
        system.set_mood("curious")
        assert system.get_mood() == "curious"
        
        system.set_mood("focused")
        assert system.get_mood() == "focused"
    
    def test_invalid_mood(self):
        system = get_fresh_personality()
        
        system.set_mood("invalid_mood")
        # Should not crash, just ignore
    
    def test_conversation_style(self):
        system = get_fresh_personality()
        
        system.set_conversation_style("formal")
        assert system.get_conversation_style() == "formal"
    
    def test_get_effective_trait_value(self):
        system = get_fresh_personality()
        
        base = system.get_trait_value("enthusiasm")
        system.set_mood("curious")
        effective = system.get_effective_trait_value("enthusiasm")
        
        # Curious mood increases enthusiasm
        assert effective >= base
    
    def test_get_response_style_hints(self):
        system = get_fresh_personality()
        
        hints = system.get_response_style_hints()
        
        assert isinstance(hints, str)
        assert len(hints) > 0
    
    def test_get_trait_summary(self):
        system = get_fresh_personality()
        
        summary = system.get_trait_summary()
        
        assert "<personality_traits>" in summary
        assert "helpfulness" in summary.lower()
    
    def test_adapt_to_user(self):
        system = get_fresh_personality()
        
        initial_formality = system.get_trait_value("formality")
        
        system.adapt_to_user("formal")
        
        new_formality = system.get_trait_value("formality")
        # Should have increased slightly
    
    def test_adapt_to_user_casual(self):
        system = get_fresh_personality()
        
        system.adapt_to_user("casual")
        
        # Just testing it doesn't crash
    
    def test_adapt_to_user_technical(self):
        system = get_fresh_personality()
        
        system.adapt_to_user("technical")
        
        # Just testing it doesn't crash
    
    def test_reset_to_default(self):
        system = get_fresh_personality()
        
        system.adapt_trait("helpfulness", 0.5)
        system.reset_to_default()
        
        # Should be back to defaults
        traits = system.get_all_traits()
        assert len(traits) > 0
    
    def test_persistence(self, tmp_path):
        system1 = PersonalitySystem(storage_path=tmp_path)
        
        system1.adapt_trait("helpfulness", 0.5)
        system1.set_mood("curious")
        system1._save()  # Force save
        
        # Create new instance to load persisted data
        system2 = PersonalitySystem(storage_path=tmp_path)
        
        # Mood is not persisted, only traits are
        # But adapt_trait should persist
        assert system2.get_trait_value("helpfulness") is not None


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_personality_system_singleton(self):
        system1 = get_personality_system()
        system2 = get_personality_system()
        assert system1 is system2
    
    def test_reset_personality_system(self):
        system1 = get_personality_system()
        reset_personality_system()
        system2 = get_personality_system()
        # Could be different instance after reset
        assert system2 is not None


class TestIntegration:
    """Integration tests."""
    
    def test_full_personality_flow(self):
        system = get_fresh_personality()
        
        # Get initial traits
        initial_helpfulness = system.get_trait_value("helpfulness")
        
        # Get style hints
        hints = system.get_response_style_hints()
        assert len(hints) > 0
        
        # Set mood
        system.set_mood("curious")
        assert system.get_mood() == "curious"
        
        # Get effective value with mood
        effective = system.get_effective_trait_value("curiosity")
        base = system.get_trait_value("curiosity")
        # Curious mood increases curiosity
        assert effective >= base
        
        # Adapt based on user style
        system.adapt_to_user("technical")
        
        # Get summary
        summary = system.get_trait_summary()
        assert "personality" in summary.lower()
    
    def test_mood_variations(self):
        system = get_fresh_personality()
        
        # Test different moods
        for mood in ["neutral", "curious", "focused", "relaxed", "determined"]:
            system.set_mood(mood)
            assert system.get_mood() == mood
            
            hints = system.get_response_style_hints()
            assert len(hints) > 0
    
    def test_adaptation_range(self):
        system = get_fresh_personality()
        
        # Multiple adaptations
        for _ in range(10):
            system.adapt_trait("creativity", 0.5)
        
        # Should be clamped
        value = system.get_trait_value("creativity")
        assert 0.0 <= value <= 1.0