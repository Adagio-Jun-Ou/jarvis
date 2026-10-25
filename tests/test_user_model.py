"""Tests for User Model."""
import pytest
import time
from core.user_model import (
    UserModelManager,
    UserModelData,
    UserPreference,
    ExpertiseArea,
    UserGoal,
    CommunicationStyle,
    DetailLevel,
    get_user_model_manager,
    reset_user_model,
    analyze_user_preferences,
    get_user_context_for_prompt,
)


class TestUserPreference:
    """Tests for UserPreference dataclass."""
    
    def test_creation(self):
        """Test creating a preference."""
        pref = UserPreference(
            key="like_python",
            value="python",
            confidence=0.8,
            source="explicit",
            timestamp=time.time(),
        )
        assert pref.key == "like_python"
        assert pref.value == "python"
        assert pref.confidence == 0.8
        assert pref.access_count == 0


class TestExpertiseArea:
    """Tests for ExpertiseArea dataclass."""
    
    def test_creation(self):
        """Test creating an expertise area."""
        area = ExpertiseArea(
            domain="python",
            level="intermediate",
            evidence=["Understands decorators", "Uses type hints"],
            confidence=0.7,
        )
        assert area.domain == "python"
        assert area.level == "intermediate"
        assert len(area.evidence) == 2


class TestUserGoal:
    """Tests for UserGoal dataclass."""
    
    def test_creation(self):
        """Test creating a goal."""
        goal = UserGoal(
            description="Build a web app",
            status="active",
            first_mentioned=time.time(),
            last_mentioned=time.time(),
        )
        assert goal.description == "Build a web app"
        assert goal.status == "active"
        assert goal.mention_count == 1


class TestUserModelManager:
    """Tests for UserModelManager."""
    
    def setup_method(self):
        reset_user_model()
    
    def teardown_method(self):
        reset_user_model()
    
    def test_singleton(self):
        """Test singleton pattern."""
        manager1 = get_user_model_manager()
        manager2 = get_user_model_manager()
        assert manager1 is manager2
    
    def test_initial_state(self):
        """Test initial state."""
        manager = get_user_model_manager()
        
        assert manager._data.interaction_count == 0
        assert manager._data.communication_style == CommunicationStyle.MIXED
        assert manager._data.detail_preference == DetailLevel.MEDIUM
    
    def test_update_from_input(self):
        """Test updating from input."""
        reset_user_model()  # Reset singleton for clean state
        manager = get_user_model_manager()
        
        manager.update_from_input("I prefer Python over JavaScript")
        
        assert manager._data.interaction_count == 1
        assert len(manager._data.preferences) > 0
    
    def test_detect_technical_terms(self):
        """Test technical term detection."""
        manager = get_user_model_manager()
        
        manager.update_from_input("I need to implement an async function with a callback")
        
        assert manager._data.uses_technical_terms is True
    
    def test_detect_interests(self):
        """Test interest detection."""
        manager = get_user_model_manager()
        
        manager.update_from_input("I love programming in Python and JavaScript")
        
        interests = manager.get_interests()
        assert "python" in interests
        assert "javascript" in interests
    
    def test_detect_goals(self):
        """Test goal detection."""
        manager = get_user_model_manager()
        
        manager.update_from_input("I want to build a web application")
        
        goals = manager.get_active_goals()
        assert len(goals) > 0
        assert "build" in goals[0]["description"].lower() or "web" in goals[0]["description"].lower()
    
    def test_detect_preferences_explicit(self):
        """Test explicit preference detection."""
        manager = get_user_model_manager()
        
        manager.update_from_input("I prefer dark mode for coding")
        
        prefs = manager.get_preferences()
        assert len(prefs) > 0
    
    def test_detect_communication_style_formal(self):
        """Test formal style detection."""
        manager = get_user_model_manager()
        
        manager.update_from_input("Could you please help me with this task? I would appreciate your assistance.")
        
        style, _ = manager.get_communication_style()
        assert style == CommunicationStyle.FORMAL
    
    def test_detect_communication_style_casual(self):
        """Test casual style detection."""
        manager = get_user_model_manager()
        
        manager.update_from_input("Hey! Wanna help me code this? It's gonna be cool!")
        manager.update_from_input("Yeah, that's awesome! Gonna try it!")
        manager.update_from_input("Nope, not gonna work!")
        
        style, _ = manager.get_communication_style()
        assert style == CommunicationStyle.CASUAL
    
    def test_detect_emojis(self):
        """Test emoji detection."""
        manager = get_user_model_manager()
        
        manager.update_from_input("I love this! 😊 It's great! 🎉")
        
        assert manager._data.uses_emojis is True
    
    def test_track_topics(self):
        """Test topic tracking."""
        manager = get_user_model_manager()
        
        manager.update_from_input("I'm working on a Python project with machine learning")
        manager.update_from_input("The Python code uses machine learning algorithms")
        
        topics = manager._data.topics_discussed
        assert "python" in topics
        assert "machine" in topics or "learning" in topics
    
    def test_get_user_context(self):
        """Test getting user context."""
        manager = get_user_model_manager()
        
        manager.update_from_input("I prefer Python")
        manager.update_from_input("I'm interested in machine learning")
        
        context = manager.get_user_context()
        
        # Should have some context
        assert isinstance(context, str)
    
    def test_predict_interest(self):
        """Test interest prediction."""
        manager = get_user_model_manager()
        
        # Learn interests
        manager.update_from_input("I love Python programming")
        manager.update_from_input("I'm learning about machine learning in Python")
        
        # Predict interest
        score = manager.predict_interest("python web development")
        assert score > 0
        
        score = manager.predict_interest("unrelated random topic")
        assert score >= 0
    
    def test_update_goal_progress(self):
        """Test goal progress update."""
        manager = get_user_model_manager()
        
        manager.update_from_input("I'm working on building a web app")
        manager.update_goal_progress("building a web app", "Finished the backend")
        
        for goal in manager._data.active_goals:
            if "web" in goal.description.lower():
                assert len(goal.progress_notes) > 0
    
    def test_complete_goal(self):
        """Test completing a goal."""
        manager = get_user_model_manager()
        
        manager.update_from_input("I want to finish my project")
        manager.complete_goal("finish my project")
        
        for goal in manager._data.active_goals:
            if "project" in goal.description.lower():
                assert goal.status == "completed"
    
    def test_message_length_tracking(self):
        """Test message length tracking."""
        manager = get_user_model_manager()
        
        manager.update_from_input("Short message")
        manager.update_from_input("This is a longer message with more words")
        manager.update_from_input("Medium length message here")
        
        assert manager._data.average_message_length > 0
    
    def test_persistence(self, tmp_path):
        """Test persistence to disk."""
        from pathlib import Path
        
        manager = UserModelManager(storage_path=tmp_path)
        
        # Update model
        manager.update_from_input("I prefer Python")
        
        # Force save
        manager._save()
        
        # Create new manager to load from disk
        manager2 = UserModelManager(storage_path=tmp_path)
        
        assert manager2._data.interaction_count == manager._data.interaction_count
        assert len(manager2._data.preferences) == len(manager._data.preferences)


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def setup_method(self):
        reset_user_model()
    
    def teardown_method(self):
        reset_user_model()
    
    def test_analyze_user_preferences(self):
        """Test analyze_user_preferences function."""
        prefs = analyze_user_preferences("I prefer dark mode")
        
        assert isinstance(prefs, dict)
        assert len(prefs) > 0
    
    def test_get_user_context_for_prompt(self):
        """Test get_user_context_for_prompt function."""
        analyze_user_preferences("I'm interested in Python")
        
        context = get_user_context_for_prompt()
        
        assert isinstance(context, str)
    
    def test_reset_user_model(self):
        """Test reset function."""
        manager = get_user_model_manager()
        manager.update_from_input("I love Python")
        
        assert manager._data.interaction_count > 0
        
        reset_user_model()
        
        manager2 = get_user_model_manager()
        # New instance should have fresh data
        assert manager2._data.interaction_count == 0


class TestUserModelData:
    """Tests for UserModelData serialization."""
    
    def test_to_dict(self):
        """Test serialization to dict."""
        data = UserModelData(
            user_id="test",
            created_at=time.time(),
            last_interaction=time.time(),
            interaction_count=5,
        )
        
        d = data.to_dict()
        
        assert d["user_id"] == "test"
        assert d["interaction_count"] == 5
        assert "created_at" in d
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "user_id": "test",
            "created_at": time.time(),
            "last_interaction": time.time(),
            "interaction_count": 5,
            "preferences": {},
            "communication_style": "technical",
            "interests": {"python": 0.8},
        }
        
        data = UserModelData.from_dict(d)
        
        assert data.user_id == "test"
        assert data.interaction_count == 5
        assert data.interests["python"] == 0.8
        assert data.communication_style == CommunicationStyle.TECHNICAL
    
    def test_round_trip(self):
        """Test round trip serialization."""
        data = UserModelData(
            user_id="test",
            created_at=time.time(),
            last_interaction=time.time(),
            interaction_count=5,
            preferences={"pref1": UserPreference("pref1", "value", 0.8, "explicit", time.time())},
            interests={"python": 0.9, "javascript": 0.7},
        )
        
        d = data.to_dict()
        data2 = UserModelData.from_dict(d)
        
        assert data2.user_id == data.user_id
        assert data2.interaction_count == data.interaction_count
        assert data2.interests == data.interests


class TestIntegration:
    """Integration tests for user model."""
    
    def setup_method(self):
        reset_user_model()
    
    def teardown_method(self):
        reset_user_model()
    
    def test_full_conversation_flow(self):
        """Test full conversation learning flow."""
        manager = get_user_model_manager()
        
        # User messages
        manager.update_from_input("I prefer Python over JavaScript")
        manager.update_from_input("Can you explain async/await in detail?")
        manager.update_from_input("I'm working on a machine learning project")
        manager.update_from_input("Could you help me with this implementation?")
        
        # Check what was learned
        assert manager._data.interaction_count == 4
        assert "python" in manager._data.interests
        assert manager._data.uses_technical_terms is True
        assert len(manager._data.active_goals) > 0
        
        # Get context
        context = manager.get_user_context()
        assert "python" in context.lower() or "machine" in context.lower()
    
    def test_multiple_preferences(self):
        """Test learning multiple preferences."""
        manager = get_user_model_manager()
        
        manager.update_from_input("I prefer dark mode")
        manager.update_from_input("I like concise explanations")
        manager.update_from_input("I always use type hints")
        
        prefs = manager.get_preferences()
        assert len(prefs) >= 2
    
    def test_goal_evolution(self):
        """Test goal tracking over time."""
        manager = get_user_model_manager()
        
        # First mention
        manager.update_from_input("I want to build a web app")
        goals_before = manager.get_active_goals()
        
        # Update progress - goal is stored as "a web app"
        manager.update_goal_progress("web app", "Created initial structure")
        
        # Check it's still active
        goals_after = manager.get_active_goals()
        assert len(goals_after) > 0
        
        # Complete
        manager.complete_goal("web app")
        
        # Check it's completed
        for goal in manager._data.active_goals:
            if "web" in goal.description.lower():
                assert goal.status == "completed"
    
    def test_interest_prediction(self):
        """Test predicting user interest."""
        manager = get_user_model_manager()
        
        # Train on interests
        manager.update_from_input("I love Python programming")
        manager.update_from_input("I'm learning about web development")
        manager.update_from_input("Python is my favorite language")
        
        # Predict
        python_score = manager.predict_interest("Python tutorials")
        random_score = manager.predict_interest("basketball")
        
        assert python_score > random_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])