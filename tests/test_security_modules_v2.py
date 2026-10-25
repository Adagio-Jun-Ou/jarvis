"""Tests for security module fixes (Session 3): branch_detection, mitm_protection, stack_defense, dlp."""

import struct
import time
import threading
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from core.security.branch_detection import (
    DataBranchMonitor, BranchType, FlowDirection,
    AnomalyType, AnomalySeverity, DataFlowEvent,
)
from core.security.stack_defense import SafeBuffer, InputValidator, StackDefense
from core.security.dlp import DataLeakProtector, SensitiveDataPattern, LeakSeverity


class TestBranchDetectionDestinationWhitelist:
    def test_empty_destinations_allows_all_by_default(self):
        m = DataBranchMonitor(authorized_destinations=set())
        anomalies = m.check_branch_policy(
            BranchType.NETWORK_SEND, "https://evil.com", "data"
        )
        unauthorized = [a for a in anomalies if a.anomaly_type == AnomalyType.UNAUTHORIZED_DESTINATION]
        assert len(unauthorized) == 0

    def test_enforce_whitelist_blocks_unknown(self):
        m = DataBranchMonitor(
            authorized_destinations={"https://good.com"},
            enforce_destination_whitelist=True,
        )
        anomalies = m.check_branch_policy(
            BranchType.NETWORK_SEND, "https://evil.com", "data"
        )
        unauthorized = [a for a in anomalies if a.anomaly_type == AnomalyType.UNAUTHORIZED_DESTINATION]
        assert len(unauthorized) == 1

    def test_enforce_whitelist_allows_known(self):
        m = DataBranchMonitor(
            authorized_destinations={"https://good.com"},
            enforce_destination_whitelist=True,
        )
        anomalies = m.check_branch_policy(
            BranchType.NETWORK_SEND, "https://good.com", "data"
        )
        unauthorized = [a for a in anomalies if a.anomaly_type == AnomalyType.UNAUTHORIZED_DESTINATION]
        assert len(unauthorized) == 0

    def test_enforce_whitelist_with_empty_set_blocks_all(self):
        m = DataBranchMonitor(
            authorized_destinations=set(),
            enforce_destination_whitelist=True,
        )
        anomalies = m.check_branch_policy(
            BranchType.NETWORK_SEND, "https://any.com", "data"
        )
        unauthorized = [a for a in anomalies if a.anomaly_type == AnomalyType.UNAUTHORIZED_DESTINATION]
        assert len(unauthorized) == 1


class TestBranchDetectionThreadSafety:
    def test_concurrent_track_flow(self):
        m = DataBranchMonitor()
        errors = []

        def worker(i):
            try:
                for j in range(50):
                    m.track_flow(
                        BranchType.SOCKET, FlowDirection.OUTBOUND,
                        f"src_{i}", f"dst_{i}", f"data_{i}_{j}"
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(m._flow_history) == 200

    def test_concurrent_check_and_track(self):
        m = DataBranchMonitor()
        errors = []

        def tracker():
            try:
                for j in range(50):
                    m.track_flow(
                        BranchType.SOCKET, FlowDirection.OUTBOUND,
                        "src", "dst", f"data_{j}"
                    )
            except Exception as e:
                errors.append(e)

        def checker():
            try:
                for _ in range(50):
                    m.check_branch_policy(BranchType.SOCKET, "dst", "data")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=tracker),
            threading.Thread(target=checker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestBranchDetectionBounds:
    def test_hash_keys_bounded(self):
        m = DataBranchMonitor()
        for i in range(m.MAX_HASH_KEYS + 100):
            m.track_flow(
                BranchType.SOCKET, FlowDirection.OUTBOUND,
                "src", "dst", f"unique_data_{i}"
            )
        assert len(m._data_hashes) <= m.MAX_HASH_KEYS + 1

    def test_destination_keys_bounded(self):
        m = DataBranchMonitor()
        for i in range(m.MAX_DESTINATION_KEYS + 100):
            m.track_flow(
                BranchType.SOCKET, FlowDirection.OUTBOUND,
                "src", f"dst_{i}", "data"
            )
        assert len(m._destinations) <= m.MAX_DESTINATION_KEYS + 1

    def test_volume_keys_bounded(self):
        m = DataBranchMonitor()
        for i in range(m.MAX_VOLUME_KEYS + 100):
            m.track_flow(
                BranchType.SOCKET, FlowDirection.OUTBOUND,
                f"src_{i}", "dst", "data"
            )
        assert len(m._volume_window) <= m.MAX_VOLUME_KEYS + 1


class TestMITMReplayProtection:
    def test_rejects_future_timestamps(self):
        from core.security.mitm_protection import RequestSigner
        signer = RequestSigner()
        sig = signer.sign_request("GET", "/test", timestamp=time.time() + 600)
        result = signer.verify_request("GET", "/test", sig, timestamp=time.time() + 600)
        assert result is False

    def test_rejects_expired_timestamps(self):
        from core.security.mitm_protection import RequestSigner
        signer = RequestSigner()
        old_time = time.time() - 600
        sig = signer.sign_request("GET", "/test", timestamp=old_time)
        result = signer.verify_request("GET", "/test", sig, timestamp=old_time)
        assert result is False

    def test_accepts_valid_timestamps(self):
        from core.security.mitm_protection import RequestSigner
        signer = RequestSigner()
        now = time.time()
        sig = signer.sign_request("GET", "/test", timestamp=now)
        result = signer.verify_request("GET", "/test", sig, timestamp=now)
        assert result is True


class TestMITMCertPinEnforcement:
    def test_empty_pins_no_enforcement(self):
        from core.security.mitm_protection import MITMProtector
        p = MITMProtector(enforce_certificate_pinning=False)
        threats = p.verify_certificate("api.openai.com", cert_der=b"der_data")
        assert len(threats) == 0

    def test_enforce_pins_unknown_host(self):
        from core.security.mitm_protection import MITMProtector
        p = MITMProtector(enforce_certificate_pinning=True)
        threats = p.verify_certificate("unknown.host.com", cert_der=b"der_data")
        untrusted = [t for t in threats if t.threat_type.value == "certificate_untrusted"]
        assert len(untrusted) == 1

    def test_enforce_pins_known_host_empty_pins(self):
        from core.security.mitm_protection import MITMProtector, MITMThreatType
        p = MITMProtector(enforce_certificate_pinning=True)
        threats = p.verify_certificate("api.openai.com", cert_der=b"der_data")
        untrusted = [t for t in threats if t.threat_type == MITMThreatType.CERTIFICATE_UNTRUSTED]
        assert len(untrusted) == 1


class TestMITMSessionBounds:
    def test_sessions_bounded(self):
        from core.security.mitm_protection import MITMProtector
        p = MITMProtector()
        for i in range(p.MAX_SESSIONS + 100):
            p.create_session(f"session_{i}", {"ip": "127.0.0.1"})
        assert len(p._active_sessions) <= p.MAX_SESSIONS


class TestMITMDNSBounds:
    def test_dns_entries_bounded(self):
        from core.security.mitm_protection import MITMProtector
        p = MITMProtector()
        for i in range(p.MAX_DNS_ENTRIES + 100):
            p.register_dns_resolution(f"host{i}.example.com", f"10.0.0.{i % 256}")
        assert len(p._known_dns) <= p.MAX_DNS_ENTRIES


class TestMITMThreadSafety:
    def test_concurrent_cert_pins(self):
        from core.security.mitm_protection import MITMProtector
        p = MITMProtector()
        errors = []

        def worker(i):
            try:
                p.add_cert_pin(f"host{i}.com", f"abc123{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestStackDefenseCanary:
    def test_canary_detects_corruption(self):
        buf = SafeBuffer(64, canary=True)
        assert buf.check_canary() is True
        corrupt_data = b'\x00' * 8
        buf._buffer[buf._canary_position:buf._canary_position + 8] = corrupt_data
        assert buf.check_canary() is False

    def test_canary_write_protection(self):
        buf = SafeBuffer(64, canary=True)
        assert buf._canary_position > 0
        result = buf.write(b'\xff' * 64, offset=0)
        assert result is False

    def test_small_buffer_canary(self):
        buf = SafeBuffer(4, canary=True)
        assert buf.check_canary() is True

    def test_canary_preserved_on_valid_write(self):
        buf = SafeBuffer(64, canary=True)
        canary_pos = buf._canary_position
        result = buf.write(b'\x41' * 8, offset=0)
        assert result is True
        assert buf.check_canary() is True


class TestStackDefenseRegexInjection:
    def test_allowed_chars_regex_injection_blocked(self):
        v = InputValidator(strict_mode=True)
        is_valid, sanitized, violations = v.validate_string(
            "test]", allowed_chars="a-z]"
        )
        assert len(violations) > 0 or not is_valid

    def test_allowed_chars_normal_use(self):
        v = InputValidator(strict_mode=False)
        is_valid, sanitized, violations = v.validate_string(
            "hello", allowed_chars="a-z"
        )
        blocked = [v for v in violations if v.action.value == "blocked"]
        assert len(blocked) == 0


class TestStackDefenseHexOverflowPattern:
    def test_hex_overflow_detected(self):
        v = InputValidator(strict_mode=True)
        hex_str = "\\x41\\x42\\x43\\x44\\x45\\x46\\x47\\x48\\x49\\x4a"
        is_valid, _, violations = v.validate_string(hex_str)
        overflow = [v for v in violations if v.vulnerability_type.value == "buffer_overflow"]
        assert len(overflow) > 0


class TestStackDefenseNoTripleSearch:
    def test_pattern_match_in_violation_input_data(self):
        v = InputValidator(strict_mode=True, max_length=100)
        is_valid, _, violations = v.validate_string("; rm -rf /")
        assert len(violations) > 0
        for viol in violations:
            if viol.vulnerability_type.value == "command_injection":
                assert viol.input_data != ""
                assert len(viol.input_data) <= 50
                break


class TestDLPcustomPatternType:
    def test_custom_pattern_uses_custom_type(self):
        dlp = DataLeakProtector()
        dlp.add_custom_pattern("project_id", r'PROJ-[A-Z]{3}-\d{4}', LeakSeverity.HIGH)
        events = dlp.scan("Here is PROJ-ABC-1234 in text")
        custom_events = [e for e in events if e.pattern_type == SensitiveDataPattern.CUSTOM]
        assert len(custom_events) == 1
        assert custom_events[0].context.get("custom_pattern") == "project_id"

    def test_custom_pattern_not_secret_key(self):
        dlp = DataLeakProtector()
        dlp.add_custom_pattern("project_id", r'PROJ-[A-Z]{3}-\d{4}', LeakSeverity.HIGH)
        events = dlp.scan("PROJ-ABC-1234")
        secret_key_events = [e for e in events if e.pattern_type == SensitiveDataPattern.SECRET_KEY]
        assert len(secret_key_events) == 0


class TestDLPCustomPatternRedaction:
    def test_custom_pattern_redacted_in_output(self):
        dlp = DataLeakProtector()
        dlp.add_custom_pattern("project_id", r'PROJ-[A-Z]{3}-\d{4}', LeakSeverity.HIGH)
        text = "The project is PROJ-ABC-1234 and active"
        allowed, sanitized, events = dlp.check_output(text, "test", "output")
        assert "[REDACTED]" in sanitized
        assert "PROJ-ABC-1234" not in sanitized


class TestDLPFinditerVsFindall:
    def test_pattern_with_groups(self):
        dlp = DataLeakProtector()
        dlp.add_custom_pattern("test_pattern", r'(password|secret)\s*=\s*["\']([^"\']+)["\']', LeakSeverity.HIGH)
        events = dlp.scan('password = "mysecret123"')
        assert len(events) >= 1
        for e in events:
            assert e.data_preview != ""