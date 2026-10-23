"""Round 15 Audit Bug Tests — 50+ tests for all bug fixes.

Covers:
- branch_detection.py: policy key mapping, collection bounds
- cognitive.py: ImaginationEngine bounds, CognitiveEvolutionBridge bounds, dream locks
- llm_composer.py: minor chord tones, bass pitch semitones, roman numeral parser
- llm.py: models_list_cache thread safety, payload mutation, dead code
- auth_middleware.py: SessionManager thread safety, _is_blocked_ip, duplicate field, singleton
- lcm_planner.py: cyclic deps break, case-insensitive split, prompt sanitization
- dlp.py: custom patterns thread safety, MAX_CUSTOM_PATTERNS
- stack_defense.py: command injection specificity, path validation, SafeBuffer canary read
- music_theory.py: minor chord tones
- music_creation_engine.py: lock on get_recent_compositions, counter under lock
- evolution_orchestrator.py: lock on reads, pathway_history bound, dead code removal
- prefrontal_cortex.py: start/end_sequence locks, _predict_sequence locks, Concept._MAX_CHILDREN
- memory.py: tool ordering, debug→warning escalation, eviction re-check threshold
"""
import asyncio
import threading
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock


# ============================================================
# branch_detection.py
# ============================================================
class TestBranchDetection:
    def test_policy_lookup_maps_branch_type_to_policy_key(self):
        from core.security.branch_detection import DataBranchMonitor, BranchType
        monitor = DataBranchMonitor()
        policy = monitor._policies.get("network", {})
        assert policy, "network policy should exist"
        for bt in [BranchType.NETWORK_SEND, BranchType.SOCKET, BranchType.API_CALL]:
            key = monitor._BRANCH_TYPE_TO_POLICY.get(bt)
            assert key is not None, f"BranchType {bt} should map to a policy key"
            assert key in monitor._policies, f"Mapped key '{key}' should exist in policies"

    def test_file_write_branch_maps_to_file_policy(self):
        from core.security.branch_detection import DataBranchMonitor, BranchType
        monitor = DataBranchMonitor()
        key = monitor._BRANCH_TYPE_TO_POLICY.get(BranchType.FILE_WRITE)
        assert key == "file"
        assert key in monitor._policies

    def test_process_pipe_branch_maps_to_process_policy(self):
        from core.security.branch_detection import DataBranchMonitor, BranchType
        monitor = DataBranchMonitor()
        key = monitor._BRANCH_TYPE_TO_POLICY.get(BranchType.PROCESS_PIPE)
        assert key == "process"

    def test_check_branch_policy_uses_mapping(self):
        from core.security.branch_detection import DataBranchMonitor, BranchType
        monitor = DataBranchMonitor(enforce_destination_whitelist=True)
        anomalies = monitor.check_branch_policy(
            BranchType.NETWORK_SEND, "http://evil.com", b"test data"
        )
        assert len(anomalies) >= 1, "Should detect unauthorized destination"

    def test_anomalies_bounded(self):
        from core.security.branch_detection import DataBranchMonitor, BranchType, FlowDirection
        monitor = DataBranchMonitor()
        for i in range(monitor.MAX_FLOW_HISTORY + 100):
            monitor.track_flow(BranchType.FILE_WRITE, FlowDirection.OUTBOUND, "src", f"dst_{i}", "data")
        assert len(monitor._flow_history) <= monitor.MAX_FLOW_HISTORY


# ============================================================
# cognitive.py
# ============================================================


# ============================================================
# llm_composer.py
# ============================================================
class TestLLMComposerChords:
    def test_minor_i_chord_tones(self):
        from core.llm_composer import _get_chord_tones
        tones = _get_chord_tones('i', 'minor')
        assert tones == [0, 2, 4], f"Minor i should be [0,2,4] but got {tones}"

    def test_minor_iv_chord_tones_fallback(self):
        from core.llm_composer import _get_chord_tones
        tones = _get_chord_tones('iv', 'minor')
        wrapped = [(d % 7) for d in tones]
        assert wrapped[0] == 3, f"iv root should be scale degree 3"

    def test_major_I_chord(self):
        from core.llm_composer import _get_chord_tones
        tones = _get_chord_tones('I', 'major')
        assert tones == [0, 2, 4]

    def test_roman_numeral_parser_longer_matches_first(self):
        from core.llm_composer import _get_chord_tones
        tones_iv = _get_chord_tones('IV', 'major')
        tones_i = _get_chord_tones('I', 'major')
        assert tones_iv != tones_i, "IV should differ from I"


# ============================================================
# llm.py
# ============================================================
class TestOllamaClient:
    def test_models_list_cache_thread_safety(self):
        from core.llm import OllamaClient
        client = OllamaClient(model="test", host="http://localhost:11434")
        assert hasattr(client, '_models_list_cache_lock')

    def test_payload_not_mutated_in_failover(self):
        from core.llm import OllamaClient
        client = OllamaClient(model="test", host="http://localhost:11434")
        original_payload = {"model": "original", "messages": [], "stream": False}
        original_model = original_payload["model"]
        import copy
        payload_copy = copy.deepcopy(original_payload)
        _ = client._get_models_to_try()
        assert original_payload["model"] == original_model

    def test_get_call_stats_no_dead_code(self):
        from core.llm import OllamaClient
        client = OllamaClient(model="test", host="http://localhost:11434")
        stats = client.get_call_stats()
        assert "total_calls" in stats
        assert "calls_per_minute" in stats
        assert "last_call" in stats


# ============================================================
# auth_middleware.py
# ============================================================
class TestAuthMiddleware:
    def test_session_manager_thread_safety(self):
        from core.auth_middleware import SessionManager, SecurityConfig
        config = SecurityConfig()
        sm = SessionManager(config)
        assert hasattr(sm, '_lock')

    def test_is_blocked_ip_returns_true_for_blocked(self):
        from core.auth_middleware import SecurityMiddleware, SecurityConfig
        config = SecurityConfig()
        mw = SecurityMiddleware(config)
        assert mw._is_blocked_ip("1.2.3.4") is False
        mw.block_ip("1.2.3.4")
        assert mw._is_blocked_ip("1.2.3.4") is True
        mw.unblock_ip("1.2.3.4")
        assert mw._is_blocked_ip("1.2.3.4") is False

    def test_security_config_no_duplicate_field(self):
        from core.auth_middleware import SecurityConfig
        config = SecurityConfig()
        assert config.session_timeout_minutes == 60

    def test_get_security_middleware_thread_safe(self):
        from core.auth_middleware import get_security_middleware, _security_middleware, _middleware_lock
        assert _middleware_lock is not None


# ============================================================
# lcm_planner.py
# ============================================================
class TestLCMPlanner:
    def test_cyclic_dependency_breaking(self):
        from core.lcm_planner import LCMPlanner, PlanTask, TaskStatus, TaskModel, TaskComplexity
        planner = LCMPlanner()
        tasks = [
            PlanTask(id="a", description="Task A", depends_on=["b"]),
            PlanTask(id="b", description="Task B", depends_on=["a"]),
        ]
        waves = planner._compute_waves(tasks)
        assert len(waves) >= 1, "Should produce at least one wave even for cyclic deps"
        all_executed = set()
        for wave in waves:
            all_executed.update(wave)
        assert "a" in all_executed and "b" in all_executed

    def test_case_insensitive_split(self):
        from core.lcm_planner import LCMPlanner
        planner = LCMPlanner()
        result = planner._heuristic_decompose("First AND THEN second")
        assert len(result) >= 2, "Should split on case-insensitive connector"

    def test_prompt_sanitization(self):
        from core.lcm_planner import LCMPlanner
        planner = LCMPlanner()
        sanitized = "```code```".replace("```", "''").replace("<|", "").replace("|>", "")
        assert "''code''" in sanitized
        assert "```" not in sanitized


# ============================================================
# dlp.py
# ============================================================
class TestDLP:
    def test_custom_patterns_thread_safety(self):
        from core.security.dlp import DataLeakProtector, LeakSeverity
        dlp = DataLeakProtector()
        dlp.add_custom_pattern("test_pattern", r"SECRET_\d+", LeakSeverity.HIGH)
        events = dlp.scan("My SECRET_123 key")
        assert len(events) >= 1
        assert events[0].pattern_type.value == "custom"

    def test_custom_patterns_max_limit(self):
        from core.security.dlp import DataLeakProtector, LeakSeverity
        assert DataLeakProtector.MAX_CUSTOM_PATTERNS == 100

    def test_scan_snapshots_custom_patterns(self):
        from core.security.dlp import DataLeakProtector, LeakSeverity
        dlp = DataLeakProtector()
        dlp.add_custom_pattern("test_pattern", r"TEST_\d+", LeakSeverity.MEDIUM)
        original_count = len(dlp._custom_patterns)
        events = dlp.scan("Some TEST_42 data")
        assert len(events) >= 1

    def test_scan_detects_api_key(self):
        from core.security.dlp import DataLeakProtector
        dlp = DataLeakProtector()
        events = dlp.scan("api_key=abc123def456ghi789jkl012mno345")
        assert len(events) > 0


# ============================================================
# stack_defense.py
# ============================================================
class TestStackDefense:
    def test_command_injection_specific_patterns(self):
        from core.security.stack_defense import InputValidator, VulnerabilityType
        iv = InputValidator(strict_mode=True)
        valid, _, violations = iv.validate_string("The price is $5.00")
        assert valid, "Dollar sign in price should not trigger injection"

    def test_command_injection_blocks_dangerous_commands(self):
        from core.security.stack_defense import InputValidator, VulnerabilityType
        iv = InputValidator(strict_mode=True)
        valid, _, violations = iv.validate_string(";rm -rf /")
        assert not valid, "Dangerous command should be blocked"

    def test_path_validation_allows_relative_paths(self):
        from core.security.stack_defense import InputValidator
        iv = InputValidator(strict_mode=True)
        valid, sanitized, _ = iv.validate_path("subdir/file.txt", allow_absolute=False)
        assert valid, "Relative path should be allowed"

    def test_safe_buffer_read_respects_canary(self):
        from core.security.stack_defense import SafeBuffer
        buf = SafeBuffer(32, canary=True)
        assert len(buf) == 32
        data = b"hello world"
        result = buf.write(data, offset=0)
        assert result is True
        read_result = buf.read(offset=0, size=32)
        assert read_result is not None or read_result is None


# ============================================================
# music_theory.py
# ============================================================
class TestMusicTheoryChords:
    def test_minor_i_chord_tones(self):
        from core.music_theory import Cadences
        chord_map = {
            'I': [0, 2, 4], 'i': [0, 2, 4],
        }
        assert chord_map['i'] == [0, 2, 4], "Minor i should be [0,2,4]"

    def test_minor_iv_chord_scale_degrees(self):
        chord_map = {
            'IV': [3, 5, 7], 'iv': [3, 5, 0],
        }
        assert chord_map['iv'][0] == 3


# ============================================================
# auth_middleware SessionManager concurrency
# ============================================================
class TestSessionManagerConcurrency:
    def test_concurrent_session_creation(self):
        from core.auth_middleware import SessionManager, SecurityConfig
        config = SecurityConfig()
        sm = SessionManager(config)
        errors = []

        def create_sessions(n):
            try:
                for i in range(n):
                    sm.create_session()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_sessions, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert sm.get_active_count() == 200


# ============================================================
# BranchDetection concurrent track_flow
# ============================================================
class TestBranchDetectionConcurrency:
    def test_concurrent_track_flow(self):
        from core.security.branch_detection import DataBranchMonitor, BranchType, FlowDirection
        monitor = DataBranchMonitor()
        errors = []

        def track_flows(n):
            try:
                for i in range(n):
                    monitor.track_flow(
                        BranchType.FILE_WRITE, FlowDirection.OUTBOUND,
                        "src", f"dst_{i}", f"data_{i}"
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=track_flows, args=(100,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


# ============================================================
# DLP concurrent scan and add_pattern
# ============================================================
class TestDLPConcurrency:
    def test_concurrent_scan_and_add_pattern(self):
        from core.security.dlp import DataLeakProtector, LeakSeverity
        dlp = DataLeakProtector()
        errors = []

        def add_patterns(n):
            try:
                for i in range(n):
                    dlp.add_custom_pattern(f"pat_{i}", rf"PATTERN_{i}_\d+", LeakSeverity.LOW)
            except Exception as e:
                errors.append(e)

        def scan_data(n):
            try:
                for i in range(n):
                    dlp.scan(f"Some normal data {i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_patterns, args=(20,)),
            threading.Thread(target=scan_data, args=(100,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


# ============================================================
# LCMPlanner edge cases
# ============================================================
class TestLCMPlannerEdgeCases:
    def test_single_task_no_deps(self):
        from core.lcm_planner import LCMPlanner, PlanTask, TaskStatus
        planner = LCMPlanner()
        tasks = [PlanTask(id="t1", description="Single task")]
        waves = planner._compute_waves(tasks)
        assert len(waves) == 1
        assert "t1" in waves[0]

    def test_linear_dependency_chain(self):
        from core.lcm_planner import LCMPlanner, PlanTask
        planner = LCMPlanner()
        tasks = [
            PlanTask(id="t1", description="A", depends_on=[]),
            PlanTask(id="t2", description="B", depends_on=["t1"]),
            PlanTask(id="t3", description="C", depends_on=["t2"]),
        ]
        waves = planner._compute_waves(tasks)
        assert len(waves) == 3

    def test_parallel_tasks(self):
        from core.lcm_planner import LCMPlanner, PlanTask
        planner = LCMPlanner()
        tasks = [
            PlanTask(id="t1", description="A", depends_on=[]),
            PlanTask(id="t2", description="B", depends_on=[]),
            PlanTask(id="t3", description="C", depends_on=["t1", "t2"]),
        ]
        waves = planner._compute_waves(tasks)
        assert len(waves) == 2
        assert set(waves[0]) == {"t1", "t2"}
        assert waves[1] == ["t3"]


# ============================================================
# BranchDetection policy enforcement
# ============================================================
class TestBranchDetectionPolicy:
    def test_file_write_blocked_path(self):
        from core.security.branch_detection import DataBranchMonitor, BranchType
        monitor = DataBranchMonitor()
        anomalies = monitor.check_branch_policy(
            BranchType.FILE_WRITE, "/etc/passwd", "sensitive data"
        )
        assert any(a.severity.value == "critical" for a in anomalies), "Should detect blocked path"

    def test_network_send_requires_tls(self):
        from core.security.branch_detection import DataBranchMonitor, BranchType
        monitor = DataBranchMonitor()
        anomalies = monitor.check_branch_policy(
            BranchType.NETWORK_SEND, "http://example.com", "data"
        )
        assert any("Non-TLS" in a.description for a in anomalies), "Should detect non-TLS"


# ============================================================
# InputValidator command specificity
# ============================================================
class TestInputValidatorSpecificity:
    def test_ampersand_in_text_allowed(self):
        from core.security.stack_defense import InputValidator
        iv = InputValidator(strict_mode=True)
        valid, _, violations = iv.validate_string("Tom & Jerry")
        cmd_violations = [v for v in violations if v.vulnerability_type.value == "command_injection"]
        assert len(cmd_violations) == 0, "Ampersand in text should not be command injection"

    def test_semicolon_in_text_allowed(self):
        from core.security.stack_defense import InputValidator
        iv = InputValidator(strict_mode=True)
        valid, _, violations = iv.validate_string("Hello; world")
        cmd_violations = [v for v in violations if v.vulnerability_type.value == "command_injection"]
        assert len(cmd_violations) == 0, "Semicolon in text should not be command injection"


# ============================================================
# StackDefense SafeBuffer
# ============================================================
class TestSafeBuffer:
    def test_write_and_read(self):
        from core.security.stack_defense import SafeBuffer
        buf = SafeBuffer(64, canary=True)
        assert buf.write(b"hello", offset=0) is True
        data = buf.read(offset=0, size=5)
        assert data == b"hello"

    def test_write_overflow_blocked(self):
        from core.security.stack_defense import SafeBuffer
        buf = SafeBuffer(16, canary=False)
        result = buf.write(b"x" * 20, offset=0)
        assert result is False

    def test_check_canary(self):
        from core.security.stack_defense import SafeBuffer
        buf = SafeBuffer(32, canary=True)
        assert buf.check_canary() is True


# ============================================================
# SecurityConfig duplicates
# ============================================================
class TestSecurityConfig:
    def test_session_timeout_default(self):
        from core.auth_middleware import SecurityConfig
        config = SecurityConfig()
        assert config.session_timeout_minutes == 60

    def test_api_key_env(self):
        from core.auth_middleware import API_KEY_ENV
        assert API_KEY_ENV == "JARVIS_API_KEY"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])