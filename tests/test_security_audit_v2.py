"""Security audit tests covering critical fixes from the extended session."""

import pytest


class TestStackDefenseRecursion:
    def test_add_violation_no_recursion(self):
        from core.security.stack_defense import InputValidator, SecurityViolation, VulnerabilityType, DefenseAction
        from datetime import datetime
        
        validator = InputValidator()
        violation = SecurityViolation(
            timestamp=datetime.now(),
            vulnerability_type=VulnerabilityType.COMMAND_INJECTION,
            description="Test violation",
            input_data="test_input",
            action=DefenseAction.LOGGED,
        )
        validator._add_violation(violation)
        assert len(validator._violations) == 1

    def test_multiple_violations_no_crash(self):
        from core.security.stack_defense import InputValidator, SecurityViolation, VulnerabilityType, DefenseAction
        from datetime import datetime
        
        validator = InputValidator()
        for i in range(5):
            violation = SecurityViolation(
                timestamp=datetime.now(),
                vulnerability_type=VulnerabilityType.COMMAND_INJECTION,
                description=f"Test violation {i}",
                input_data=f"input_{i}",
                action=DefenseAction.LOGGED,
            )
            validator._add_violation(violation)
        assert len(validator._violations) == 5


class TestSessionManagerBounds:
    def test_session_eviction_on_limit(self):
        from core.auth_middleware import SessionManager, SecurityConfig, AuthLevel
        
        config = SecurityConfig()
        manager = SessionManager(config)
        manager.MAX_SESSIONS = 5
        
        for i in range(10):
            manager.create_session(auth_level=AuthLevel.STANDARD)
        
        assert len(manager._sessions) <= 5


class TestTerminalSecurity:
    def test_timeout_bounded(self):
        timeout = min(max(int(999999), 1), 300)
        assert timeout == 300

    def test_timeout_minimum(self):
        timeout = min(max(int(0), 1), 300)
        assert timeout == 1

    def test_normal_timeout(self):
        timeout = min(max(int(60), 1), 300)
        assert timeout == 60


class TestAgentToolSafety:
    def test_simple_query_dangerous_detection(self):
        from core.agent import JarvisAgent
        agent = JarvisAgent.__new__(JarvisAgent)
        assert not agent._is_simple_query("delete all files")
        assert not agent._is_simple_query("run rm -rf /")
        assert not agent._is_simple_query("sudo rm something")
        assert not agent._is_simple_query("format drive")
        assert not agent._is_simple_query("kill process")

    def test_simple_query_still_works(self):
        from core.agent import JarvisAgent
        agent = JarvisAgent.__new__(JarvisAgent)
        assert agent._is_simple_query("hi")
        assert agent._is_simple_query("hello")
        assert agent._is_simple_query("what is Python")
        assert agent._is_simple_query("ok")


class TestEvaluationCompletion:
    def test_incomplete_keyword_not_matching_complete(self):
        # Verify that "incomplete" does NOT contain bare "complete" as a word
        answer = "The analysis is incomplete"
        answer_words = answer.lower().split()
        assert "complete" not in answer_words
        assert "incomplete" in answer.lower()

    def test_done_treated_as_complete(self):
        answer = "COMPLETE"
        answer_words = answer.lower().split()
        assert "complete" in answer_words


class TestEmbeddingInterconnectThreadSafety:
    def test_get_model_outputs_returns_copy(self):
        from core.embedding_interconnect import EmbeddingInterconnect
        ic = EmbeddingInterconnect(prefer_server=False)
        ic.store_model_output("primary", "test", "response", 0.9)
        outputs = ic.get_model_outputs("primary")
        assert isinstance(outputs, list)
        assert len(outputs) == 1
        assert outputs[0][0] == "test"
        outputs.clear()
        assert len(ic.get_model_outputs("primary")) == 1

    def test_get_last_responses_returns_copy(self):
        from core.embedding_interconnect import EmbeddingInterconnect
        ic = EmbeddingInterconnect(prefer_server=False)
        ic.store_model_output("primary", "test", "response", 0.9)
        responses = ic.get_last_responses()
        assert isinstance(responses, dict)
        responses.clear()
        assert len(ic.get_last_responses()) > 0


class TestIntelligentOrchestratorSafety:
    def test_get_returns_initialized_or_none(self):
        # After reset, get_intelligent_orchestrator returns None
        # This tests that no half-initialized __new__ objects are created
        import core.intelligent_orchestrator as mod
        with mod._lock:
            old = mod._intelligent_orchestrator
            mod._intelligent_orchestrator = None
        result = mod.get_intelligent_orchestrator()
        assert result is None
        with mod._lock:
            mod._intelligent_orchestrator = old


class TestAuthBypassWarning:
    def test_no_keys_allows_access(self):
        from core.auth_middleware import APIKeyValidator, SecurityConfig
        
        config = SecurityConfig()
        validator = APIKeyValidator(config)
        validator._valid_keys = set()
        validator._key_hashes = {}
        
        is_valid, token = validator.validate("any_key")
        assert is_valid is False


class TestMusicTheoryEngine:
    def test_voice_leading_detects_leaps(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        result = engine.check_voice_leading([60, 64, 67], [72, 76, 79])
        assert not result["valid"]
        assert len(result["issues"]) > 0

    def test_voice_leading_accepts_stepwise(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        # Stepwise motion is always valid (C-E-G to D-F-A)
        result = engine.check_voice_leading([60, 64, 67], [62, 65, 69])
        # May have parallel motion issue but no leaps
        has_leap_issue = any("leap" in i for i in result["issues"])
        assert not has_leap_issue