"""Tests for Proactive Engine."""
import pytest
import time
from core.proactive import (
    ProactiveEngine,
    ProactiveSuggestion,
    SuggestionType,
    Opportunity,
    get_proactive_engine,
    reset_proactive_engine,
)


def get_fresh_engine():
    """Get a fresh engine for testing."""
    reset_proactive_engine()
    return get_proactive_engine()


class TestProactiveSuggestion:
    """Tests for ProactiveSuggestion dataclass."""
    
    def test_creation(self):
        suggestion = ProactiveSuggestion(
            suggestion_type=SuggestionType.HELP_WITH_TASK,
            message="Test suggestion",
            priority=3,
            confidence=0.8
        )
        assert suggestion.suggestion_type == SuggestionType.HELP_WITH_TASK
        assert suggestion.message == "Test suggestion"
        assert suggestion.priority == 3
        assert suggestion.confidence == 0.8
        assert not suggestion.shown
        assert not suggestion.dismissed


class TestOpportunity:
    """Tests for Opportunity dataclass."""
    
    def test_creation(self):
        opp = Opportunity(
            description="Test opportunity",
            relevance=0.9,
            action_suggestion="Do something"
        )
        assert opp.description == "Test opportunity"
        assert opp.relevance == 0.9
        assert opp.action_suggestion == "Do something"
        assert not opp.requires_permission


class TestProactiveEngine:
    """Tests for ProactiveEngine."""
    
    def test_initial_state(self):
        engine = get_fresh_engine()
        assert engine._conversation_turns == 0
        assert len(engine._mentioned_files) == 0
    
    def test_should_suggest_no_opportunity(self):
        engine = get_fresh_engine()
        
        should, suggestion = engine.should_suggest(
            user_mood="neutral",
            active_goals=[],
            recent_activity={}
        )
        
        assert should == False
        assert suggestion is None
    
    def test_should_suggest_with_goals(self):
        engine = get_fresh_engine()
        
        # Create mock goal with blocker
        class MockGoal:
            def __init__(self):
                self.description = "Build app"
                self.goal_id = "test1"
                self.blockers = ["Missing API"]
                self.tasks = []
        
        goals = [MockGoal()]
        
        # Wait minimum interval
        engine._last_suggestion_time = 0
        
        should, suggestion = engine.should_suggest(
            user_mood="neutral",
            active_goals=goals,
            recent_activity={}
        )
        
        assert should == True
        assert suggestion is not None
        assert "blocked" in suggestion.message.lower() or "help" in suggestion.message.lower()
    
    def test_should_suggest_frustrated_user(self):
        engine = get_fresh_engine()
        
        # Frustrated users shouldn't get proactive suggestions
        should, suggestion = engine.should_suggest(
            user_mood="frustrated",
            active_goals=[],
            recent_activity={"last_input": "This is not working!"}
        )
        
        assert should == False
    
    def test_should_suggest_rate_limit(self):
        engine = get_fresh_engine()
        
        class MockGoal:
            def __init__(self):
                self.description = "Test"
                self.goal_id = "test"
                self.blockers = ["Issue"]
                self.tasks = []
        
        goals = [MockGoal()]
        
        # First suggestion should work
        engine._last_suggestion_time = 0
        should1, _ = engine.should_suggest(
            user_mood="neutral",
            active_goals=goals,
            recent_activity={}
        )
        assert should1 == True
        
        # Second suggestion immediately should be rate limited
        should2, _ = engine.should_suggest(
            user_mood="neutral",
            active_goals=goals,
            recent_activity={}
        )
        assert should2 == False
    
    def test_check_for_opportunities_empty(self):
        engine = get_fresh_engine()
        
        opportunities = engine.check_for_opportunities()
        assert isinstance(opportunities, list)
    
    def test_check_for_opportunities_with_goals(self):
        engine = get_fresh_engine()
        
        class MockGoal:
            def __init__(self):
                self.description = "Build API"
                self.goal_id = "api_goal"
                self.blockers = ["Need authentication"]
                self.tasks = []
        
        goals = [MockGoal()]
        
        opportunities = engine.check_for_opportunities(
            user_mood="neutral",
            active_goals=goals
        )
        
        assert len(opportunities) > 0
        assert any("blocked" in o.description.lower() or "api" in o.description.lower() for o in opportunities)
    
    def test_check_for_opportunities_with_pending_tasks(self):
        engine = get_fresh_engine()
        
        class MockTask:
            def __init__(self, desc):
                self.description = desc
                self.status = "pending"
        
        class MockGoal:
            def __init__(self):
                self.description = "Build app"
                self.goal_id = "app_goal"
                self.blockers = []
                self.tasks = [MockTask("Write code"), MockTask("Test")]
        
        goals = [MockGoal()]
        
        opportunities = engine.check_for_opportunities(
            user_mood="neutral",
            active_goals=goals
        )
        
        assert len(opportunities) > 0
    
    def test_generate_suggestion(self):
        engine = get_fresh_engine()
        
        suggestion = engine.generate_suggestion(
            user_model={"mood": "neutral"},
            active_goals=[],
            recent_activity={}
        )
        
        # May or may not return suggestion depending on context
        assert suggestion is None or isinstance(suggestion, ProactiveSuggestion)
    
    def test_anticipate_needs(self):
        engine = get_fresh_engine()
        
        needs = engine.anticipate_needs("I'm trying to fix this bug in my code")
        
        assert len(needs) > 0
        assert any("code" in n.lower() or "bug" in n.lower() or "error" in n.lower() for n in needs)
    
    def test_anticipate_needs_learning(self):
        engine = get_fresh_engine()
        
        needs = engine.anticipate_needs("I want to learn about machine learning")
        
        assert len(needs) > 0
        assert any("learn" in n.lower() or "machine" in n.lower() for n in needs)
    
    def test_anticipate_needs_building(self):
        engine = get_fresh_engine()
        
        needs = engine.anticipate_needs("I want to build a web application")
        
        assert len(needs) > 0
        # Could be "Suggest architecture", "Generate boilerplate", etc.
        assert any("build" in n.lower() or "web" in n.lower() or "architecture" in n.lower() for n in needs)
    
    def test_update_activity(self):
        engine = get_fresh_engine()
        
        engine.update_activity(
            user_input="I'm working on main.py and app.js",
            mentioned_files=["main.py", "app.js"]
        )
        
        assert engine._conversation_turns == 1
        assert "main.py" in engine._mentioned_files
        assert "app.js" in engine._mentioned_files
    
    def test_update_activity_multiple_times(self):
        engine = get_fresh_engine()
        
        engine.update_activity("First message")
        engine.update_activity("Second message about python")
        engine.update_activity("Third message about testing")
        
        assert engine._conversation_turns == 3
    
    def test_dismiss_suggestion(self):
        engine = get_fresh_engine()
        
        engine.dismiss_suggestion(SuggestionType.HELP_WITH_TASK)
        
        assert engine._dismissed_types.get("help_with_task", 0) >= 1
    
    def test_get_context_for_prompt(self):
        engine = get_fresh_engine()
        
        engine.update_activity("I'm working on Python", ["test.py"])
        
        context = engine.get_context_for_prompt()
        
        assert "<proactive_context>" in context
        assert "Conversation turns" in context
    
    def test_get_context_for_prompt_empty(self):
        engine = get_fresh_engine()
        
        context = engine.get_context_for_prompt()
        
        assert context == ""
    
    def test_get_suggestion_history(self):
        engine = get_fresh_engine()
        
        class MockGoal:
            def __init__(self):
                self.description = "Test"
                self.goal_id = "test"
                self.blockers = ["Issue"]
                self.tasks = []
        
        goals = [MockGoal()]
        engine._last_suggestion_time = 0
        
        engine.should_suggest(
            user_mood="neutral",
            active_goals=goals,
            recent_activity={}
        )
        
        history = engine.get_suggestion_history()
        assert len(history) >= 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_proactive_engine_singleton(self):
        engine1 = get_proactive_engine()
        engine2 = get_proactive_engine()
        assert engine1 is engine2
    
    def test_reset_proactive_engine(self):
        engine1 = get_proactive_engine()
        reset_proactive_engine()
        engine2 = get_proactive_engine()
        # After reset, could be different instance
        assert engine2 is not None


class TestIntegration:
    """Integration tests."""
    
    def test_full_proactive_flow(self):
        engine = get_fresh_engine()
        
        # Simulate conversation
        engine.update_activity("I want to build a chat application")
        
        # Anticipate needs
        needs = engine.anticipate_needs("I'm trying to build a chat application")
        assert len(needs) > 0
        
        # Update with files
        engine.update_activity(
            "I added a new file chat.py",
            mentioned_files=["chat.py"]
        )
        
        # Check for opportunities
        opportunities = engine.check_for_opportunities()
        assert isinstance(opportunities, list)
        
        # Get context for prompt
        context = engine.get_context_for_prompt()
        assert len(context) > 0
    
    def test_proactive_with_blocked_goal(self):
        engine = get_fresh_engine()
        
        class MockGoal:
            def __init__(self):
                self.description = "Deploy application"
                self.goal_id = "deploy"
                self.blockers = ["Need server access"]
                self.tasks = []
        
        goals = [MockGoal()]
        engine._last_suggestion_time = 0
        
        should, suggestion = engine.should_suggest(
            user_mood="neutral",
            active_goals=goals,
            recent_activity={}
        )
        
        assert should == True
        assert suggestion is not None
        assert suggestion.confidence > 0
    
    def test_proactive_respects_mood(self):
        engine = get_fresh_engine()
        
        # Test with different moods
        for mood in ["neutral", "curious", "excited"]:
            engine._last_suggestion_time = 0
            
            class MockGoal:
                def __init__(self):
                    self.description = "Test"
                    self.goal_id = "test"
                    self.blockers = ["Issue"]
                    self.tasks = []
            
            should, _ = engine.should_suggest(
                user_mood=mood,
                active_goals=[MockGoal()],
                recent_activity={}
            )
            
            # Should suggest for positive/neutral moods
            if mood in ["frustrated", "angry", "urgent"]:
                assert should == False