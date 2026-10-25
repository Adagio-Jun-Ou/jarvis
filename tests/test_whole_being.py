"""Tests for Live State and Whole Being Integration."""
import pytest
import threading
import time
from unittest.mock import MagicMock, patch

from core.live_state import (
    LiveState,
    LiveStateData,
    SubsystemState,
    SubsystemStatus,
    MemoryState,
    BackgroundDiscoveries,
    NarrativeState,
    get_live_state,
    reset_live_state,
)
from core.context_enrichment import (
    get_daydream_state,
    get_erdos_state,
    get_music_state,
    get_learning_state,
    get_all_subsystem_states,
    get_active_subsystems,
    get_recent_discoveries,
    enrich_context_with_subsystems,
    get_subsystem_activity_summary,
    build_rich_context,
)
from core.background_bridge import (
    BackgroundBridge,
    Discovery,
    ConversationContext,
    get_background_bridge,
    reset_background_bridge,
)
from core.narrative_thread import (
    NarrativeThread,
    NarrativeState,
    SessionSummary,
    LearningMilestone,
    FocusArea,
    get_narrative_thread,
    reset_narrative_thread,
)


class TestLiveState:
    """Tests for LiveState class."""
    
    def setup_method(self):
        reset_live_state()
    
    def teardown_method(self):
        reset_live_state()
    
    def test_singleton(self):
        """Test that get_live_state returns singleton."""
        state1 = get_live_state()
        state2 = get_live_state()
        assert state1 is state2
    
    def test_initial_state(self):
        """Test initial state is empty."""
        state = get_live_state()
        data = state.get_state()
        
        assert data.identity == "Jarvis"
        assert data.model == "kimi-k2.5"
        assert data.current_activity == ""
        assert len(data.subsystems) == 0
    
    def test_register_subsystem_getter(self):
        """Test registering subsystem getter."""
        state = get_live_state()
        
        state.register_subsystem_getter("test", lambda: {
            "status": "active",
            "activity": "testing",
            "discoveries": ["test discovery"],
        })
        
        state.update_from_subsystems()
        
        assert "test" in state._data.subsystems
        assert state._data.subsystems["test"].status == SubsystemStatus.ACTIVE
        assert state._data.subsystems["test"].current_activity == "testing"
    
    def test_update_from_agent(self):
        """Test updating from agent."""
        state = get_live_state()
        
        agent = MagicMock()
        agent.config = MagicMock()
        agent.config.identity = {"name": "TestJarvis"}
        agent.llm = MagicMock()
        agent.llm.model = "test-model"
        agent.tools = {"tool1": MagicMock(), "tool2": MagicMock()}
        agent.memory = MagicMock()
        agent.memory.get_history.return_value = []
        
        state.update_from_agent(agent)
        
        data = state.get_state()
        assert data.model == "test-model"
        assert len(data.capabilities) == 2
    
    def test_get_context_string(self):
        """Test context string generation."""
        state = get_live_state()
        
        # Set some state
        state._data.current_activity = "Composing music"
        state._data.discoveries.music_compositions = ["Test composition in C"]
        state._data.narrative.current_focus = "Testing"
        
        context = state.get_context_string()
        
        assert "Composing music" in context
        assert "Test composition" in context
        assert "Testing" in context
    
    def test_set_mood(self):
        """Test setting mood."""
        state = get_live_state()
        
        state.set_mood("curious", ["music", "math"])
        
        assert state._data.mood == "curious"
        assert "music" in state._data.focus_areas
        assert "math" in state._data.focus_areas
    
    def test_update_narrative(self):
        """Test updating narrative."""
        state = get_live_state()
        
        state.update_narrative(focus="Testing narrative", progress="Made good progress")
        
        assert state._data.narrative.current_focus == "Testing narrative"
        assert "Made good progress" in state._data.narrative.recent_progress


class TestContextEnrichment:
    """Tests for context enrichment functions."""
    
    def test_get_daydream_state(self):
        """Test daydream state returns valid structure."""
        state = get_daydream_state()
        assert "status" in state
        assert "activity" in state
        assert "discoveries" in state
    
    def test_get_erdos_state(self):
        """Test erdos state returns valid structure."""
        state = get_erdos_state()
        assert "status" in state
        assert "activity" in state
        assert "discoveries" in state
    
    def test_get_music_state(self):
        """Test music state returns valid structure."""
        state = get_music_state()
        assert "status" in state
        assert "activity" in state
        assert "discoveries" in state
    
    def test_get_all_subsystem_states(self):
        """Test getting all subsystem states."""
        states = get_all_subsystem_states(use_cache=False)
        
        assert "daydream" in states
        assert "erdos" in states
        assert "music" in states
        assert "learning" in states
        assert "evolution" in states
    
    def test_get_active_subsystems(self):
        """Test getting active subsystems."""
        active = get_active_subsystems()
        # Most subsystems will be idle/unavailable, so this should be empty or short
        assert isinstance(active, list)
    
    def test_enrich_context_with_subsystems(self):
        """Test enriching context."""
        base = "You are Jarvis."
        enriched = enrich_context_with_subsystems(base)
        
        # Should at least return base context
        assert "You are Jarvis." in enriched
    
    def test_get_subsystem_activity_summary(self):
        """Test activity summary."""
        summary = get_subsystem_activity_summary()
        # Should return something, even if "idle"
        assert isinstance(summary, str)
    
    def test_build_rich_context(self):
        """Test building rich context."""
        agent = MagicMock()
        agent.prompt_manager = MagicMock()
        agent.prompt_manager.get_base_prompt.return_value = "Base prompt"
        agent.tools = {}
        agent.memory = MagicMock()
        agent.memory.get_history.return_value = []
        
        context = build_rich_context(agent, include_memory=False)
        
        assert "Base prompt" in context


class TestBackgroundBridge:
    """Tests for BackgroundBridge class."""
    
    def setup_method(self):
        reset_background_bridge()
    
    def teardown_method(self):
        reset_background_bridge()
    
    def test_singleton(self):
        """Test that get_background_bridge returns singleton."""
        bridge1 = get_background_bridge()
        bridge2 = get_background_bridge()
        assert bridge1 is bridge2
    
    def test_parse_conversation_context_casual(self):
        """Test parsing casual conversation."""
        bridge = get_background_bridge()
        
        context = bridge.parse_conversation_context("How are you today?")
        
        assert context.is_casual is True
        assert context.about_self is True
    
    def test_parse_conversation_context_topic(self):
        """Test parsing topic-specific conversation."""
        bridge = get_background_bridge()
        
        context = bridge.parse_conversation_context("Can you help me with music composition?")
        
        assert "music" in context.topics
    
    def test_should_share_activity_casual(self):
        """Test should share for casual conversation."""
        bridge = get_background_bridge()
        
        should_share, reason = bridge.should_share_activity("How are you?")
        
        # Should share if asking about Jarvis
        assert should_share is True
        assert reason == "user_asked_about_self"
    
    def test_collect_discoveries(self):
        """Test collecting discoveries."""
        bridge = get_background_bridge()
        
        discoveries = bridge.collect_discoveries()
        
        # Should return a list (empty if no subsystems running)
        assert isinstance(discoveries, list)
    
    def test_format_discovery_mention(self):
        """Test formatting discovery mentions."""
        bridge = get_background_bridge()
        
        discovery = Discovery(
            source="music",
            content="Composed Test Piece in C",
            timestamp=time.time(),
            relevance_keywords=["music"],
            importance=0.7
        )
        
        # Natural style
        natural = bridge.format_discovery_mention(discovery, style="natural")
        assert "music" in natural.lower() or "composed" in natural.lower()
        
        # Brief style
        brief = bridge.format_discovery_mention(discovery, style="brief")
        assert brief == "Composed Test Piece in C"
    
    def test_generate_background_context(self):
        """Test generating background context."""
        bridge = get_background_bridge()
        
        # Should return empty string for non-matching message
        context = bridge.generate_background_context("What is 2+2?")
        
        # Empty if not triggered
        assert isinstance(context, str)
    
    def test_get_background_summary(self):
        """Test getting background summary."""
        bridge = get_background_bridge()
        
        summary = bridge.get_background_summary()
        
        assert isinstance(summary, str)


class TestNarrativeThread:
    """Tests for NarrativeThread class."""
    
    def setup_method(self):
        reset_narrative_thread()
    
    def teardown_method(self):
        reset_narrative_thread()
    
    def test_singleton(self):
        """Test that get_narrative_thread returns singleton."""
        thread1 = get_narrative_thread()
        thread2 = get_narrative_thread()
        assert thread1 is thread2
    
    def test_start_end_session(self):
        """Test session lifecycle."""
        thread = get_narrative_thread()
        
        # Reset state
        initial_sessions = thread._state.total_sessions
        
        thread.start_session("test-session")
        assert thread._current_session is not None
        assert thread._current_session.session_id == "test-session"
        
        thread.end_session(key_points=["point1", "point2"])
        assert thread._current_session is None
        assert thread._state.total_sessions == initial_sessions + 1
    
    def test_record_turn(self):
        """Test recording turns."""
        thread = get_narrative_thread()
        
        thread.start_session()
        thread.record_turn("Tell me about music", tools_used=["web_search"])
        
        assert thread._current_session.turn_count == 1
        assert "music" in thread._current_session.topics
        assert "web_search" in thread._current_session.tools_used
        
        thread.end_session()
    
    def test_record_discovery(self):
        """Test recording discoveries."""
        thread = get_narrative_thread()
        
        thread._state.background_highlights.clear()
        initial_count = len(thread._state.background_highlights)
        
        thread.start_session()
        thread.record_discovery("Found cool pattern", "daydream")
        
        assert len(thread._current_session.discoveries) == 1
        assert len(thread._state.background_highlights) == initial_count + 1
        
        thread.end_session()
    
    def test_update_focus(self):
        """Test updating focus."""
        thread = get_narrative_thread()
        
        # Reset state for clean test
        thread._state.focus_areas = []
        
        thread.update_focus("python programming", relevance=0.8)
        
        assert thread._state.current_focus == "python programming"
        assert len(thread._state.focus_areas) == 1
        assert thread._state.focus_areas[0].relevance == 0.8
    
    def test_add_learning(self):
        """Test adding learning milestones."""
        thread = get_narrative_thread()
        
        # Clear previous milestones for clean test
        thread._state.milestones = []
        
        thread.add_learning("code", "Learned about async patterns")
        
        assert len(thread._state.milestones) == 1
        assert thread._state.milestones[0].category == "code"
        assert "Learning" in thread._state.learning_trajectory or "learning" in thread._state.learning_trajectory.lower()
    
    def test_get_context_string(self):
        """Test context string generation."""
        thread = get_narrative_thread()
        
        thread.update_focus("testing")
        thread.add_learning("test", "learned testing patterns")
        
        context = thread.get_context_string()
        
        assert "testing" in context
        assert "learned" in context.lower() or "Learning" in context
    
    def test_persistence(self, tmp_path):
        """Test state persistence."""
        thread = NarrativeThread(storage_path=tmp_path)
        
        thread.start_session("persist-test")
        thread.update_focus("persistence")
        thread.end_session(key_points=["tested persistence"])
        
        # Create new instance - should load state
        thread2 = NarrativeThread(storage_path=tmp_path)
        
        assert thread2._state.total_sessions == 1
        assert thread2._state.current_focus == "persistence"


class TestIntegration:
    """Integration tests for whole being system."""
    
    def setup_method(self):
        reset_live_state()
        reset_background_bridge()
        reset_narrative_thread()
    
    def teardown_method(self):
        reset_live_state()
        reset_background_bridge()
        reset_narrative_thread()
    
    def test_full_state_flow(self):
        """Test complete state flow."""
        # Initialize state
        live_state = get_live_state()
        narrative = get_narrative_thread()
        bridge = get_background_bridge()
        
        # Start session
        narrative.start_session("integration-test")
        
        # Simulate subsystem state
        live_state.register_subsystem_getter("test", lambda: {
            "status": "active",
            "activity": "testing integration",
            "discoveries": ["test discovery"],
        })
        live_state.update_from_subsystems()
        
        # Record activity
        narrative.record_turn("Test message", ["test_tool"])
        narrative.update_focus("integration testing")
        narrative.add_learning("test", "learned integration patterns")
        
        # End session
        narrative.end_session(key_points=["integration tested"])
        
        # Verify state
        data = live_state.get_state()
        assert "test" in data.subsystems
        
        context = narrative.get_context_string()
        assert "integration" in context.lower()
    
    def test_context_enrichment_integration(self):
        """Test context enrichment integration."""
        live_state = get_live_state()
        
        # Register subsystems
        live_state.register_subsystem_getter("daydream", get_daydream_state)
        live_state.register_subsystem_getter("music", get_music_state)
        
        # Update state
        live_state.update_from_subsystems()
        live_state.update_discoveries()
        
        # Get context
        context = live_state.get_context_string()
        
        # Should have some context even if empty subsystems
        assert isinstance(context, str)
    
    def test_background_to_foreground(self):
        """Test background to foreground flow."""
        bridge = get_background_bridge()
        narrative = get_narrative_thread()
        
        # Record some background activity
        narrative.start_session()
        narrative.record_discovery("Exploring music theory", "daydream")
        narrative.end_session()
        
        # Check if it would share for appropriate message
        should_share, reason = bridge.should_share_activity("What have you been up to?")
        
        # Should share because it matches a trigger phrase
        assert should_share is True
        # Reason could be user_asked_about_self or trigger_phrase
        assert reason in ("user_asked_about_self", "trigger_phrase")


class TestSubsystemState:
    """Tests for SubsystemState dataclass."""
    
    def test_creation(self):
        """Test creating subsystem state."""
        state = SubsystemState(
            name="test",
            status=SubsystemStatus.ACTIVE,
            current_activity="testing",
            recent_discoveries=["discovery1"],
            metrics={"count": 5}
        )
        
        assert state.name == "test"
        assert state.status == SubsystemStatus.ACTIVE
        assert len(state.recent_discoveries) == 1


class TestDiscovery:
    """Tests for Discovery dataclass."""
    
    def test_creation(self):
        """Test creating discovery."""
        discovery = Discovery(
            source="test",
            content="Test discovery",
            timestamp=time.time(),
            relevance_keywords=["test", "discovery"],
            importance=0.75
        )
        
        assert discovery.source == "test"
        assert discovery.importance == 0.75


class TestEndToEndFlow:
    """End-to-end tests for whole being integration."""
    
    def test_live_state_to_context(self):
        """Test live state integrates into context."""
        from core.live_state import get_live_state
        
        state = get_live_state()
        
        # Register a test subsystem
        state.register_subsystem_getter("test", lambda: {
            "status": "active",
            "activity": "testing end-to-end",
            "discoveries": ["test discovery"],
        })
        
        # Update state
        state.update_from_subsystems()
        
        # Get context
        context = state.get_context_string()
        
        assert "testing end-to-end" in context
    
    def test_narrative_persists(self):
        """Test narrative thread persists."""
        from core.narrative_thread import get_narrative_thread
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            thread = get_narrative_thread()
            thread.storage_path = Path(tmpdir)
            
            # Start session, add learning
            thread.start_session("test-persist")
            thread.update_focus("testing persistence")
            thread.add_learning("test", "learned about persistence")
            thread.end_session()
            
            # Get context
            context = thread.get_context_string()
            
            assert "testing persistence" in context or "persistence" in context.lower()
    
    def test_background_bridge_relevance(self):
        """Test background bridge relevance matching."""
        bridge = get_background_bridge()
        
        # Collect discoveries
        bridge.collect_discoveries()
        
        # Test relevance matching
        should_share, reason = bridge.should_share_activity("What have you been working on?")
        
        # Should share for this question
        assert should_share is True
    
    def test_background_bridge_natural_mention(self):
        """Test natural mention formatting."""
        bridge = get_background_bridge()
        
        # Test with different messages
        mention1 = bridge.format_natural_mention("What have you been doing?")
        mention2 = bridge.format_natural_mention("What is 2+2?")
        
        # Either returns something or empty string
        assert isinstance(mention1, str)
        assert isinstance(mention2, str)
    
    def test_background_bridge_injection(self):
        """Test response injection."""
        bridge = get_background_bridge()
        
        # Test injection context
        context = bridge.get_injection_context("What are you working on?")
        assert isinstance(context, str)
        
        # Test should inject
        should = bridge.should_inject_in_response("What have you been doing?", "I'm doing well.")
        assert isinstance(should, bool)
        
        # Test inject into response
        response = bridge.inject_into_response("What have you been doing?", "I'm doing well.")
        assert isinstance(response, str)
    
    def test_context_enrichment_builds(self):
        """Test context enrichment builds rich context."""
        from core.context_enrichment import get_all_subsystem_states, get_subsystem_activity_summary
        
        # Get all states
        states = get_all_subsystem_states(use_cache=False)
        
        # Should have all subsystems
        assert "daydream" in states
        assert "erdos" in states
        assert "music" in states
        
        # Get summary
        summary = get_subsystem_activity_summary()
        
        # Should return something
        assert isinstance(summary, str)
    
    def test_unified_memory_search(self):
        """Test unified memory search across all systems."""
        from core.context_enrichment import unified_memory_search, format_unified_results
        
        # Search for something
        results = unified_memory_search("test", limit=3)
        
        # Should return a list
        assert isinstance(results, list)
        
        # Each result should have source and content
        for result in results:
            assert "source" in result
            assert "content" in result
        
        # Format results
        formatted = format_unified_results(results, max_items=3)
        
        # Should return a string
        assert isinstance(formatted, str)
    
    def test_format_live_context_integration(self):
        """Test format_live_context_for_prompt integration."""
        from core.context_enrichment import format_live_context_for_prompt
        
        # Create mock agent with required attributes
        class MockConfig:
            identity = {"name": "TestJarvis"}
        
        class MockLLM:
            model = "test-model"
        
        class MockAgent:
            prompt_manager = None
            tools = {}
            memory = None
            _live_state = None
            _narrative_thread = None
            config = MockConfig()
            llm = MockLLM()
        
        agent = MockAgent()
        
        # Get context
        context = format_live_context_for_prompt(agent, "test query")
        
        # Should return string
        assert isinstance(context, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])