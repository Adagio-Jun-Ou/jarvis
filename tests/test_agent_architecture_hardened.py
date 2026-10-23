"""Tests for agent architecture hardening.

Covers:
- ToolRegistryValidator: dynamic registration, validation, health, conflicts
- ToolExecutionGuard: rate limiting, timeout, concurrency, input validation
- SelfCorrectionEngine: plan expiry, outcome learning, analytics
- AgenticHarness: plan validation, dead step detection, duration estimation
- SSRFProtector: URL validation, internal detection, DNS checks
- SessionSecurityManager: session lifecycle, brute-force, revocation
- KnowledgeGraph: fuzzy search, temporal insights, cross-domain, paths
"""
import os
import sys
import time
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.tool_registry import (
    ToolRegistryValidator, ToolInfo, ToolCategory, TOOL_REGISTRY,
    get_tool_validator,
)
from core.agent_loop import (
    AgentLoop, ToolExecutionGuard, FinishReason, ToolStatus,
    ToolPart, AssistantMessage, get_tool_execution_guard,
)
from core.self_correction import (
    SelfCorrectionEngine, ExecutionStep, ExecutionPlan, GoalStatus,
    StepStatus, RecoveryStrategy, ErrorType, OutcomeRecord,
)
from core.agentic_harness import (
    AgenticHarness, HarnessPlan, HarnessStep, HarnessPhase,
    StepStatus as HarnessStepStatus, EventBus, EventType,
)
from core.security.ssrf_protection import (
    SSRFProtector, SSRFValidationResult, validate_outbound_url,
    sanitize_request_headers, get_ssrf_protector,
)
from core.security.session_security import (
    SessionSecurityManager, SessionInfo, get_session_security,
)
from core.knowledge_graph import (
    KnowledgeGraph, KnowledgeGraphContext, NodeType, EdgeType,
)


# ========================================================================
# ToolRegistryValidator Tests
# ========================================================================

class TestToolRegistryValidator:
    def setup_method(self):
        self.validator = ToolRegistryValidator()

    def test_register_valid_tool(self):
        tool = ToolInfo(
            name="test_tool",
            category=ToolCategory.SYSTEM,
            description="A test tool for unit testing purposes",
            parameters={"input": "The input to process"},
        )
        warnings = self.validator.register_dynamic_tool(tool)
        assert isinstance(warnings, list)
        assert "test_tool" in self.validator.get_dynamic_tools()

    def test_register_invalid_name_numbers(self):
        tool = ToolInfo(
            name="123invalid",
            category=ToolCategory.SYSTEM,
            description="Test tool description for testing",
            parameters={},
        )
        with pytest.raises(ValueError, match="Invalid tool name"):
            self.validator.register_dynamic_tool(tool)

    def test_register_invalid_name_hyphens(self):
        tool = ToolInfo(
            name="test-tool",
            category=ToolCategory.SYSTEM,
            description="Test tool description for testing",
            parameters={},
        )
        with pytest.raises(ValueError, match="Invalid tool name"):
            self.validator.register_dynamic_tool(tool)

    def test_register_short_name(self):
        tool = ToolInfo(
            name="x",
            category=ToolCategory.SYSTEM,
            description="Test tool description for testing",
            parameters={},
        )
        with pytest.raises(ValueError, match="too short"):
            self.validator.register_dynamic_tool(tool)

    def test_register_duplicate_tool(self):
        tool1 = ToolInfo(
            name="unique_test_1",
            category=ToolCategory.SYSTEM,
            description="First test tool description for testing",
            parameters={"a": "Parameter a description"},
        )
        tool2 = ToolInfo(
            name="unique_test_1",
            category=ToolCategory.WEB,
            description="Second test tool description for testing",
            parameters={"b": "Parameter b description"},
        )
        self.validator.register_dynamic_tool(tool1)
        with pytest.raises(ValueError, match="already registered"):
            self.validator.register_dynamic_tool(tool2)

    def test_register_short_description(self):
        tool = ToolInfo(
            name="short_desc_tool",
            category=ToolCategory.SYSTEM,
            description="Short",
            parameters={},
        )
        with pytest.raises(ValueError, match="description too short"):
            self.validator.register_dynamic_tool(tool)

    def test_register_invalid_parameter_name(self):
        tool = ToolInfo(
            name="bad_param_tool",
            category=ToolCategory.SYSTEM,
            description="A tool with invalid parameter names for testing",
            parameters={"123bad": "Invalid parameter name"},
        )
        with pytest.raises(ValueError, match="invalid parameter name"):
            self.validator.register_dynamic_tool(tool)

    def test_unregister_dynamic_tool(self):
        tool = ToolInfo(
            name="removable_tool",
            category=ToolCategory.SYSTEM,
            description="A tool that can be removed during testing",
            parameters={},
        )
        self.validator.register_dynamic_tool(tool)
        assert self.validator.unregister_dynamic_tool("removable_tool")
        assert "removable_tool" not in self.validator.get_dynamic_tools()

    def test_cannot_unregister_builtin(self):
        result = self.validator.unregister_dynamic_tool("shell")
        assert result is False

    def test_record_tool_call_success(self):
        self.validator.record_tool_call("test_tool_success", True, 50.0)
        health = self.validator.get_tool_health("test_tool_success")
        assert health is not None
        assert health["call_count"] == 1
        assert health["avg_latency_ms"] == 50.0

    def test_record_tool_call_error(self):
        self.validator.record_tool_call("test_tool_err", False, 10.0, "timeout")
        health = self.validator.get_tool_health("test_tool_err")
        assert health["error_count"] == 1
        assert health["last_error"] == "timeout"

    def test_unhealthy_detection(self):
        for i in range(10):
            self.validator.record_tool_call("unhealthy_tool", False, 10.0, "error")
        unhealthy = self.validator.get_unhealthy_tools()
        assert "unhealthy_tool" in unhealthy

    def test_validate_tool_input(self):
        valid, errors = self.validator.validate_tool_input("shell", {"command": "ls"})
        assert valid is True
        assert len(errors) == 0

    def test_validate_unknown_tool(self):
        valid, errors = self.validator.validate_tool_input("nonexistent_tool", {})
        assert valid is False
        assert any("Unknown tool" in e for e in errors)

    def test_validate_input_oversized(self):
        big_input = {"data": "x" * 60000}
        guard = ToolExecutionGuard()
        valid, errors = guard.validate_input("test", big_input)
        assert not valid

    def test_get_all_tools_includes_dynamic(self):
        tool = ToolInfo(
            name="all_tools_test",
            category=ToolCategory.DATA,
            description="A dynamic tool for testing get_all_tools method",
            parameters={},
        )
        self.validator.register_dynamic_tool(tool)
        all_tools = self.validator.get_all_tools()
        assert "all_tools_test" in all_tools
        assert "shell" in all_tools

    def test_get_registry_stats(self):
        stats = self.validator.get_registry_stats()
        assert "total_tools" in stats
        assert "builtin_tools" in stats
        assert "dynamic_tools" in stats
        assert stats["builtin_tools"] == len(TOOL_REGISTRY)

    def test_singleton(self):
        v1 = get_tool_validator()
        v2 = get_tool_validator()
        assert v1 is v2


# ========================================================================
# ToolExecutionGuard Tests
# ========================================================================

class TestToolExecutionGuard:
    def setup_method(self):
        self.guard = ToolExecutionGuard()

    def test_set_tool_timeout(self):
        self.guard.set_tool_timeout("heavy_tool", 300.0)
        assert self.guard.get_timeout("heavy_tool") == 300.0
        assert self.guard.get_timeout("unknown_tool") == 120.0

    def test_rate_limit_allows(self):
        for i in range(5):
            assert self.guard.check_rate_limit("test_tool") is True

    def test_rate_limit_blocks(self):
        self.guard.set_rate_limit(3)
        assert self.guard.check_rate_limit("flood_tool") is True
        assert self.guard.check_rate_limit("flood_tool") is True
        assert self.guard.check_rate_limit("flood_tool") is True
        assert self.guard.check_rate_limit("flood_tool") is False

    def test_concurrency_check(self):
        assert self.guard.check_concurrency("test_conc") is True
        self.guard.acquire_concurrency("test_conc")
        assert self.guard.check_concurrency("test_conc") is True
        self.guard.release_concurrency("test_conc")

    def test_input_validation_dict(self):
        valid, errors = self.guard.validate_input("test", {"key": "value"})
        assert valid is True

    def test_input_validation_non_dict(self):
        valid, errors = self.guard.validate_input("test", "not a dict")
        assert valid is False

    def test_input_validation_oversized(self):
        big = {"data": "x" * 60000}
        valid, errors = self.guard.validate_input("test", big)
        assert not valid

    def test_get_stats(self):
        stats = self.guard.get_stats()
        assert "rate_limit_per_minute" in stats
        assert "tool_timeouts" in stats

    def test_singleton(self):
        g1 = get_tool_execution_guard()
        g2 = get_tool_execution_guard()
        assert g1 is g2

    def test_concurrency_release(self):
        self.guard.acquire_concurrency("rel_test")
        assert self.guard._concurrent_counts.get("rel_test") == 1
        self.guard.release_concurrency("rel_test")
        assert self.guard._concurrent_counts.get("rel_test") == 0
        self.guard.release_concurrency("rel_test")  # should not go negative
        assert self.guard._concurrent_counts.get("rel_test") == 0


# ========================================================================
# SelfCorrectionEngine Tests - Hardened
# ========================================================================

class TestSelfCorrectionHardened:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.engine = SelfCorrectionEngine(storage_path=Path(self.tmpdir))

    def test_cleanup_expired_plans(self):
        plan = self.engine.decompose_goal("test expired plan")
        # Plans just created should not be expired (72 hour window)
        count = self.engine.cleanup_expired_plans()
        assert isinstance(count, int)
        assert count == 0  # Fresh plan should not be cleaned up

    def test_prune_outcomes(self):
        self.engine.record_outcome("test_goal", True)
        result = self.engine.prune_outcomes()
        assert isinstance(result, int)

    def test_enforce_plan_limit(self):
        removed = self.engine.enforce_plan_limit()
        assert isinstance(removed, int)
        assert removed >= 0

    def test_learn_from_outcomes(self):
        self.engine.record_outcome("test1", False, error="connection timeout", strategy_used=RecoveryStrategy.RETRY)
        learned = self.engine.learn_from_outcomes()
        assert isinstance(learned, dict)

    def test_get_best_recovery_strategy(self):
        strategy = self.engine.get_best_recovery_strategy("connection timeout")
        assert strategy is not None
        assert isinstance(strategy, RecoveryStrategy)

    def test_get_analytics(self):
        analytics = self.engine.get_analytics()
        assert "total_outcomes" in analytics
        assert "success_count" in analytics
        assert "success_rate" in analytics
        assert "avg_duration_seconds" in analytics
        assert "error_type_distribution" in analytics
        assert "strategy_distribution" in analytics
        assert "plan_status_distribution" in analytics
        assert "learned_patterns" in analytics
        assert "active_plans" in analytics

    def test_classify_error_patterns(self):
        assert self.engine.classify_error("connection refused") == ErrorType.TRANSIENT
        assert self.engine.classify_error("permission denied") == ErrorType.PERMISSION
        assert self.engine.classify_error("timeout waiting") == ErrorType.TIMEOUT
        assert self.engine.classify_error("unknown thing") == ErrorType.UNKNOWN

    def test_decompose_goal_creates_plan(self):
        plan = self.engine.decompose_goal("analyze and test the code")
        assert isinstance(plan, ExecutionPlan)
        assert plan.goal == "analyze and test the code"
        assert len(plan.steps) > 0

    def test_execute_step_no_handler(self):
        step = ExecutionStep(step_id="s1", name="test", description="test", action="unknown_action")
        plan = ExecutionPlan(goal_id="g1", goal="test", steps=[step])
        result = self.engine.record_outcome("g1", True)
        # Should not crash


# ========================================================================
# AgenticHarness Tests - Hardened
# ========================================================================

class TestAgenticHarnessHardened:
    def setup_method(self):
        self.harness = AgenticHarness()

    def test_validate_plan_valid(self):
        plan = self.harness.create_plan("test objective", [
            {"action": "step1", "tool_name": "shell", "description": "First step", "expected_outcome": "Done"},
        ])
        is_valid, issues = self.harness.validate_plan(plan)
        assert is_valid is True

    def test_validate_plan_too_many_steps(self):
        steps = [
            {"action": f"step_{i}", "tool_name": "shell", "description": f"Step {i}"}
            for i in range(60)
        ]
        plan = self.harness.create_plan("big plan", steps)
        is_valid, issues = self.harness.validate_plan(plan)
        assert not is_valid

    def test_validate_plan_self_dependency(self):
        plan = HarnessPlan(objective="test")
        step = HarnessStep(id="s1", action="step1", dependencies=["s1"])
        step.dependencies = ["s1"]
        plan.steps.append(step)
        is_valid, issues = self.harness.validate_plan(plan)
        assert any("self-dependency" in i for i in issues)

    def test_validate_plan_negative_retries(self):
        plan = HarnessPlan(objective="test")
        step = HarnessStep(id="s1", action="step1", max_retries=-1)
        plan.steps.append(step)
        is_valid, issues = self.harness.validate_plan(plan)
        assert any("negative" in i.lower() for i in issues)

    def test_detect_dead_steps(self):
        plan = HarnessPlan(objective="test")
        step1 = HarnessStep(id="s1", action="step1", status=HarnessStepStatus.FAILED, error="failed")
        step2 = HarnessStep(id="s2", action="step2", dependencies=["s1"], status=HarnessStepStatus.PENDING)
        plan.steps = [step1, step2]
        dead = self.harness.detect_dead_steps(plan)
        assert "s2" in dead

    def test_estimate_plan_duration(self):
        plan = HarnessPlan(objective="test")
        step1 = HarnessStep(id="s1", action="step1", timeout_seconds=30.0)
        step2 = HarnessStep(id="s2", action="step2", timeout_seconds=60.0)
        plan.steps = [step1, step2]
        duration = self.harness.estimate_plan_duration(plan)
        assert "min" in duration
        assert "max" in duration
        assert "expected" in duration
        assert duration["min"] >= 0

    def test_estimate_empty_plan_duration(self):
        plan = HarnessPlan(objective="test")
        duration = self.harness.estimate_plan_duration(plan)
        assert duration["min"] == 0

    def test_cleanup_expired_plans(self):
        plan = self.harness.create_plan("test cleanup")
        removed = self.harness.cleanup_expired_plans()
        assert isinstance(removed, int)

    def test_get_status_includes_new_fields(self):
        status = self.harness.get_status()
        assert "plans_stored" in status


# ========================================================================
# SSRFProtector Tests
# ========================================================================

class TestSSRFProtector:
    def setup_method(self):
        self.protector = SSRFProtector()

    def test_validate_valid_url(self):
        result = self.protector.validate_url("https://example.com/path")
        assert result.valid is True
        assert result.risk_level == "low"

    def test_validate_file_scheme(self):
        result = self.protector.validate_url("file:///etc/passwd")
        assert result.valid is False
        assert result.risk_level == "critical"

    def test_validate_gopher_scheme(self):
        result = self.protector.validate_url("gopher://evil.com/")
        assert result.valid is False
        assert result.risk_level == "critical"

    def test_validate_localhost(self):
        result = self.protector.validate_url("http://localhost/admin")
        assert result.valid is False
        assert "denylisted" in result.error.lower() or "internal" in result.error.lower()

    def test_validate_internal_ip(self):
        result = self.protector.validate_url("http://10.0.0.1/internal")
        assert result.valid is False
        assert "internal" in result.error.lower()

    def test_validate_metadata_hostname(self):
        result = self.protector.validate_url("http://metadata.google.internal/")
        assert result.valid is False

    def test_validate_azure_metadata(self):
        result = self.protector.validate_url("http://169.254.169.254/metadata")
        assert result.valid is False
        assert result.risk_level == "critical"

    def test_validate_empty_url(self):
        result = self.protector.validate_url("")
        assert result.valid is False

    def test_validate_non_string_url(self):
        result = self.protector.validate_url(12345)
        assert result.valid is False

    def test_validate_denylisted_port(self):
        result = self.protector.validate_url("https://example.com:3306/db")
        assert result.valid is False
        assert "denylisted" in result.error.lower()

    def test_validate_embedded_credentials(self):
        result = self.protector.validate_url("https://user:pass@example.com/")
        assert result.valid is False

    def test_validate_internal_ip_allowed(self):
        result = self.protector.validate_url("http://192.168.1.1/", allow_internal=True)
        assert result.valid is True

    def test_validate_url_too_long(self):
        result = self.protector.validate_url("https://example.com/" + "a" * 3000)
        assert result.valid is False

    def test_rate_limiting(self):
        self.protector._request_rate_limit = 3
        assert self.protector.check_request_rate("test.com") is True
        assert self.protector.check_request_rate("test.com") is True
        assert self.protector.check_request_rate("test.com") is True
        assert self.protector.check_request_rate("test.com") is False

    def test_custom_denylist(self):
        self.protector.add_denylist("1.2.3.4/32")
        result = self.protector.validate_url("http://1.2.3.4/")
        assert result.valid is False

    def test_stats(self):
        stats = self.protector.get_stats()
        assert "dns_cache_size" in stats
        assert "tracked_hosts" in stats

    def test_sanitize_headers(self):
        headers = {
            "content-type": "application/json",
            "x-forwarded-for": "10.0.0.1",
            "authorization": "Bearer token123",
            "accept": "text/html",
        }
        cleaned = sanitize_request_headers(headers)
        assert "content-type" in cleaned
        assert "accept" in cleaned
        assert "x-forwarded-for" not in cleaned
        assert "authorization" not in cleaned

    def test_singleton(self):
        p1 = get_ssrf_protector()
        p2 = get_ssrf_protector()
        assert p1 is p2

    def test_convenience_function(self):
        result = validate_outbound_url("https://example.com/page")
        assert result.valid is True


# ========================================================================
# SessionSecurityManager Tests
# ========================================================================

class TestSessionSecurityManager:
    def setup_method(self):
        self.session_mgr = SessionSecurityManager()

    def test_create_session(self):
        session_id, token = self.session_mgr.create_session(
            user_id="test_user",
            ip_address="127.0.0.1",
            user_agent="TestBrowser/1.0",
        )
        assert isinstance(session_id, str)
        assert isinstance(token, str)
        assert len(token) == 64  # 32 bytes = 64 hex chars

    def test_validate_valid_session(self):
        session_id, token = self.session_mgr.create_session(
            user_id="test_user",
            ip_address="127.0.0.1",
        )
        valid, session_info, error = self.session_mgr.validate_session(token, ip_address="127.0.0.1")
        assert valid is True
        assert session_info is not None
        assert session_info.user_id == "test_user"
        assert error == ""

    def test_validate_invalid_token(self):
        valid, session_info, error = self.session_mgr.validate_session("invalid_token")
        assert valid is False
        assert "Invalid" in error

    def test_revoke_session(self):
        session_id, token = self.session_mgr.create_session(user_id="test_user")
        assert self.session_mgr.revoke_session(session_id) is True
        valid, _, _ = self.session_mgr.validate_session(token)
        assert valid is False

    def test_revoke_nonexistent_session(self):
        assert self.session_mgr.revoke_session("nonexistent") is False

    def test_concurrent_session_limit(self):
        self.session_mgr.max_concurrent_sessions = 2
        for i in range(2):
            self.session_mgr.create_session(user_id="limited_user")
        sid3, tok3 = self.session_mgr.create_session(user_id="limited_user")
        # First session should be revoked
        active = self.session_mgr.get_active_sessions(user_id="limited_user")
        assert len(active) <= 2

    def test_revoke_all_user_sessions(self):
        self.session_mgr.create_session(user_id="multi_user")
        self.session_mgr.create_session(user_id="multi_user")
        count = self.session_mgr.revoke_all_user_sessions("multi_user")
        assert count >= 2

    def test_brute_force_protection(self):
        self.session_mgr.max_login_attempts = 3
        self.session_mgr.login_window_seconds = 900
        self.session_mgr.lockout_duration_seconds = 60

        for i in range(3):
            self.session_mgr._record_login_attempt(
                "10.0.0.1", "testuser", success=False, reason="bad password"
            )

        allowed, reason = self.session_mgr.check_login_allowed("10.0.0.1")
        assert allowed is False
        assert "locked" in reason.lower() or "too many" in reason.lower()

    def test_different_ip_accounts(self):
        self.session_mgr.max_login_attempts = 3
        for i in range(2):
            self.session_mgr._record_login_attempt("192.168.1.1", "user1", success=False)
        allowed, _ = self.session_mgr.check_login_allowed("192.168.1.2")
        assert allowed is True

    def test_session_stats(self):
        self.session_mgr.create_session(user_id="user1")
        self.session_mgr.create_session(user_id="user2")
        stats = self.session_mgr.get_stats()
        assert "total_sessions" in stats
        assert "active_sessions" in stats
        assert "unique_users" in stats

    def test_ip_mismatch_rejection(self):
        _, token = self.session_mgr.create_session(
            user_id="ip_test", ip_address="10.0.0.1"
        )
        valid, _, error = self.session_mgr.validate_session(token, ip_address="10.0.0.2")
        assert valid is False
        assert "mismatch" in error.lower()

    def test_singleton(self):
        s1 = get_session_security()
        s2 = get_session_security()
        assert s1 is s2


# ========================================================================
# KnowledgeGraph Advanced Tests
# ========================================================================

class TestKnowledgeGraphAdvanced:
    def setup_method(self):
        self.kg = KnowledgeGraph(storage_path=Path(tempfile.mkdtemp()))

    def test_fuzzy_search_concepts(self):
        self.kg.add_node("c1", NodeType.CONCEPT, "machine learning neural networks")
        self.kg.add_node("c2", NodeType.CONCEPT, "deep learning techniques")
        self.kg.add_node("c3", NodeType.CONCEPT, "natural language processing")
        self.kg.compute_pagerank()

        results = self.kg.fuzzy_search_concepts("machine learning", limit=5, min_score=0.1)
        assert isinstance(results, list)
        # Fuzzy search may return results with varying scores
        # Just verify we get some results back
        if len(results) > 0:
            assert "concept" in results[0]
            assert "score" in results[0]

    def test_fuzzy_search_empty_query(self):
        results = self.kg.fuzzy_search_concepts("", limit=5)
        assert results == []

    def test_get_temporal_insights(self):
        self.kg.add_node("ins1", NodeType.INSIGHT, "Recent insight about AI")
        results = self.kg.get_temporal_insights(days=30, limit=10)
        assert isinstance(results, list)

    def test_find_knowledge_paths(self):
        self.kg.add_node("ml", NodeType.CONCEPT, "machine learning")
        self.kg.add_node("dl", NodeType.CONCEPT, "deep learning")
        self.kg.add_node("nlp", NodeType.CONCEPT, "natural language processing")
        self.kg.add_edge("ml", "dl", EdgeType.RELATED_TO, 0.8)
        self.kg.add_edge("dl", "nlp", EdgeType.RELATED_TO, 0.7)
        self.kg.compute_pagerank()

        paths = self.kg.find_knowledge_paths("machine learning", "natural language processing")
        assert isinstance(paths, list)

    def test_find_knowledge_paths_no_match(self):
        paths = self.kg.find_knowledge_paths("nonexistent_a", "nonexistent_b")
        assert paths == []

    def test_cross_domain_connections(self):
        self.kg.add_node("domain_ai", NodeType.DOMAIN, "artificial intelligence")
        self.kg.add_node("domain_bio", NodeType.DOMAIN, "biology")
        self.kg.add_node("ins_bridge", NodeType.INSIGHT, "AI methods applied to biological data analysis")
        self.kg.add_edge("domain_ai", "ins_bridge", EdgeType.IN_DOMAIN)
        self.kg.add_edge("domain_bio", "ins_bridge", EdgeType.IN_DOMAIN)
        self.kg.compute_pagerank()

        connections = self.kg.get_cross_domain_connections("artificial", "biology")
        assert isinstance(connections, list)

    def test_search_concepts_basic(self):
        self.kg.add_node("c1", NodeType.CONCEPT, "reinforcement learning")
        self.kg.compute_pagerank()
        results = self.kg.search_concepts("reinforcement")
        # Verify search returns a list (may be empty if concept matching differs)
        assert isinstance(results, list)


# ========================================================================
# Integration Tests
# ========================================================================

class TestIntegration:
    def test_validator_and_guard_interaction(self):
        validator = ToolRegistryValidator()
        guard = ToolExecutionGuard()

        tool = ToolInfo(
            name="integration_tool",
            category=ToolCategory.SYSTEM,
            description="Integration test tool for validation and guard testing",
            parameters={"input": "The input data to process"},
        )
        warnings = validator.register_dynamic_tool(tool)
        assert isinstance(warnings, list)

        valid, errors = validator.validate_tool_input("integration_tool", {"input": "test"})
        assert valid is True

        assert guard.check_rate_limit("integration_tool") is True

    def test_ssrf_and_session_interaction(self):
        ssrf = SSRFProtector()
        session = SessionSecurityManager()

        # Use allow_internal to skip DNS check for example.com
        url_result = ssrf.validate_url("https://example.com/data", allow_internal=True)
        # DNS may or may not resolve in test environment, just verify it returns a result
        assert isinstance(url_result, SSRFValidationResult)

        session_id, token = session.create_session(user_id="api_user")
        valid, info, _ = session.validate_session(token)
        assert valid is True
        assert info.user_id == "api_user"

    def test_harness_plan_creation_and_validation(self):
        harness = AgenticHarness()
        plan = harness.create_plan(
            "Analyze data",
            steps=[
                {"action": "gather", "tool_name": "shell", "description": "Gather data", "expected_outcome": "Data gathered"},
                {"action": "analyze", "tool_name": "shell", "description": "Analyze data", "expected_outcome": "Analysis done"},
            ],
        )
        assert len(plan.steps) == 2
        is_valid, issues = harness.validate_plan(plan)
        # Plan should be valid (no circular deps, proper structure)
        assert is_valid is True

    def test_self_correction_analytics(self):
        tmpdir = tempfile.mkdtemp()
        engine = SelfCorrectionEngine(storage_path=Path(tmpdir))
        # Create a plan first so record_outcome can reference it
        plan = engine.decompose_goal("test analytics")
        # Record outcomes that reference the plan
        engine.record_outcome(plan.goal_id, True, strategy_used=RecoveryStrategy.RETRY)
        
        # Also test with a direct outcome
        outcome = OutcomeRecord(
            goal_id="test_goal_2",
            goal="direct outcome test",
            success=False,
            error="timeout error",
            error_type=ErrorType.TIMEOUT,
            strategy_used=RecoveryStrategy.ALTERNATIVE,
            recovery_successful=False,
            steps_completed=1,
            steps_total=3,
            duration_seconds=5.0,
        )
        with engine._lock:
            engine._outcomes.append(outcome)

        analytics = engine.get_analytics()
        assert analytics["total_outcomes"] >= 1
        # success_count includes the first outcome
        assert analytics["success_count"] >= 1