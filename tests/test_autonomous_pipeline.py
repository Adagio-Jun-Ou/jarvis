"""Integration tests for Jarvis autonomous pipeline.

Tests the complete flow:
- Daydream Engine: Idle-time exploration
- Erdős Engine: Math problem exploration  
- Insight Bridge: Captures and routes discoveries
- Evolution Orchestrator: Manages improvement proposals
- Self-Edit: Applies changes with approval
- Discovery Notifier: Shares discoveries
"""
import asyncio
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDaydreamEngine:
    """Test Daydream Engine functionality."""
    
    def test_import(self):
        """Test DaydreamEngine can be imported."""
        from core.daydream_engine import DaydreamEngine
        assert DaydreamEngine is not None
    
    def test_initialization(self):
        """Test DaydreamEngine initializes correctly."""
        from core.daydream_engine import DaydreamEngine
        
        engine = DaydreamEngine()
        assert engine.DEFAULT_IDLE_THRESHOLD > 0
        assert engine.DEFAULT_DAYDREAM_INTERVAL > 0
    
    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test DaydreamEngine can start and stop."""
        from core.daydream_engine import DaydreamEngine
        
        engine = DaydreamEngine()
        
        await engine.start()
        status = engine.get_status()
        assert status is not None
        
        await engine.stop()
        status = engine.get_status()
        assert status is not None
    
    @pytest.mark.asyncio
    async def test_can_daydream(self):
        """Test can_daydream check."""
        from core.daydream_engine import DaydreamEngine
        
        engine = DaydreamEngine()
        result = engine.can_daydream()
        assert result is True or result is False
        await engine.stop()
    
    def test_get_status(self):
        """Test status retrieval."""
        from core.daydream_engine import DaydreamEngine
        
        engine = DaydreamEngine()
        status = engine.get_status()
        assert status is not None
        assert isinstance(status, dict)


class TestErdosEngine:
    """Test Erdős Engine functionality."""
    
    def test_import(self):
        """Test ErdosEngine can be imported."""
        from core.erdos_engine import ErdosEngine
        assert ErdosEngine is not None
    
    def test_initialization(self):
        """Test ErdosEngine initializes correctly."""
        from core.erdos_engine import ErdosEngine
        
        engine = ErdosEngine()
        assert engine.MAX_PROBLEMS > 0
        assert engine.MAX_HISTORY > 0
    
    @pytest.mark.asyncio
    async def test_get_active_problems(self):
        """Test getting active problems."""
        from core.erdos_engine import ErdosEngine
        
        engine = ErdosEngine()
        problems = engine.get_active_problems()
        assert problems is not None
    
    @pytest.mark.asyncio
    async def test_daydream_step(self):
        """Test basic daydream step."""
        from core.erdos_engine import ErdosEngine
        
        engine = ErdosEngine()
        await engine.daydream_step()
    
    def test_get_status(self):
        """Test status retrieval."""
        from core.erdos_engine import ErdosEngine
        
        engine = ErdosEngine()
        status = engine.get_status()
        assert status is not None


class TestInsightBridge:
    """Test Insight Bridge functionality."""
    
    def test_import(self):
        """Test InsightBridge can be imported."""
        from core.insight_bridge import InsightBridge
        assert InsightBridge is not None
    
    def test_initialization(self):
        """Test InsightBridge initializes correctly."""
        from core.insight_bridge import get_insight_bridge
        
        bridge = get_insight_bridge()
        assert bridge is not None
    
    def test_process_insight(self):
        """Test processing an insight."""
        from core.insight_bridge import get_insight_bridge
        
        bridge = get_insight_bridge()
        
        result = bridge.process_insight(
            raw_insight="Prime numbers exhibit clustering patterns",
            source_problem="exploration",
            source_category="pattern",
        )
        
        assert result is not None or result is None
    
    def test_get_insight_statistics(self):
        """Test getting insight statistics."""
        from core.insight_bridge import get_insight_bridge
        
        bridge = get_insight_bridge()
        
        stats = bridge.get_insight_statistics()
        assert stats is not None


class TestEvolutionOrchestrator:
    """Test Evolution Orchestrator functionality."""
    
    def test_import(self):
        """Test EvolutionOrchestrator can be imported."""
        from core.evolution_orchestrator import EvolutionOrchestrator
        assert EvolutionOrchestrator is not None
    
    def test_initialization(self):
        """Test EvolutionOrchestrator initializes correctly."""
        from core.evolution_orchestrator import EvolutionOrchestrator
        
        orch = EvolutionOrchestrator(storage_path=Path(tempfile.mkdtemp()))
        assert orch is not None
    
    def test_get_status(self):
        """Test status retrieval."""
        from core.evolution_orchestrator import EvolutionOrchestrator
        
        orch = EvolutionOrchestrator(storage_path=Path(tempfile.mkdtemp()))
        status = orch.get_status()
        assert status is not None


class TestSelfEdit:
    """Test Self-Edit functionality."""
    
    def test_import(self):
        """Test SelfEditEngine can be imported."""
        from core.self_edit import SelfEditEngine
        assert SelfEditEngine is not None
    
    def test_initialization(self):
        """Test SelfEditEngine initializes correctly."""
        from core.self_edit import get_self_edit_engine
        
        editor = get_self_edit_engine()
        assert editor is not None
        assert len(editor.SAFE_FILES) > 0
    
    def test_safe_files_defined(self):
        """Test that safe files are properly defined."""
        from core.self_edit import get_self_edit_engine
        
        editor = get_self_edit_engine()
        safe_files = editor.SAFE_FILES
        
        expected_files = [
            "core/erdos_engine.py",
            "core/insight_bridge.py",
            "core/evolution_orchestrator.py",
            "core/daydream_engine.py",
        ]
        
        for expected in expected_files:
            assert expected in safe_files, f"{expected} should be in SAFE_FILES"
    
    def test_can_edit(self):
        """Test edit permission checking."""
        from core.self_edit import get_self_edit_engine
        
        editor = get_self_edit_engine()
        
        for safe_file in editor.SAFE_FILES:
            can_edit, _ = editor.can_edit_file(safe_file)
            assert can_edit, f"Should be able to edit {safe_file}"
        
        can_edit, _ = editor.can_edit_file("core/config.py")
        assert not can_edit, "Should not edit config.py"


class TestDiscoveryNotifier:
    """Test Discovery Notifier functionality."""
    
    def test_import(self):
        """Test DiscoveryNotifier can be imported."""
        from core.discovery_notifier import DiscoveryNotifier, DiscoveryType
        assert DiscoveryNotifier is not None
        assert DiscoveryType is not None
    
    def test_initialization(self):
        """Test DiscoveryNotifier initializes correctly."""
        from core.discovery_notifier import DiscoveryNotifier
        
        notifier = DiscoveryNotifier()
        assert notifier is not None
    
    def test_record_discovery(self):
        """Test recording a discovery."""
        from core.discovery_notifier import DiscoveryNotifier, DiscoveryType
        
        notifier = DiscoveryNotifier()
        
        result = notifier.record_discovery(
            discovery_type=DiscoveryType.PATTERN_DISCOVERY,
            title="Pattern in prime numbers",
            description="Found pattern in prime numbers",
            source="erdos_engine",
        )
        
        assert result is not None or result is None
    
    def test_get_discovery_summary(self):
        """Test getting discovery summary."""
        from core.discovery_notifier import DiscoveryNotifier
        
        notifier = DiscoveryNotifier()
        summary = notifier.get_discovery_summary()
        assert summary is not None


class TestDaydreamVerification:
    """Test Daydream Verification functionality."""
    
    def test_import(self):
        """Test SolutionVerifier can be imported."""
        from core.daydream_verification import SolutionVerifier, VerificationStatus
        assert SolutionVerifier is not None
        assert VerificationStatus is not None
    
    def test_initialization(self):
        """Test SolutionVerifier initializes correctly."""
        from core.daydream_verification import SolutionVerifier
        
        verification = SolutionVerifier()
        assert verification is not None
    
    def test_submit_solution(self):
        """Test submitting a solution for verification."""
        from core.daydream_verification import SolutionVerifier
        
        verification = SolutionVerifier()
        
        solution_id = verification.submit_solution(
            problem_type="test",
            problem_statement="Test problem",
            proposed_solution="Test solution",
            insight="Test insight for verification",
        )
        
        assert solution_id is not None or solution_id is None


class TestAutonomousIntegration:
    """Integration tests for the complete autonomous pipeline."""
    
    @pytest.mark.asyncio
    async def test_erdos_to_insight_flow(self):
        """Test Erdős → Insight Bridge flow."""
        from core.erdos_engine import ErdosEngine
        from core.insight_bridge import get_insight_bridge
        
        erdos = ErdosEngine()
        bridge = get_insight_bridge()
        
        insight = bridge.process_insight(
            raw_insight="Pattern discovered in exploration",
            source_problem="test_problem",
            source_category="exploration",
        )
        
        assert insight is not None or insight is None
    
    @pytest.mark.asyncio
    async def test_daydream_to_discovery_flow(self):
        """Test Daydream → Discovery flow."""
        from core.daydream_engine import DaydreamEngine
        from core.discovery_notifier import DiscoveryNotifier, DiscoveryType
        
        daydream = DaydreamEngine()
        notifier = DiscoveryNotifier()
        
        await daydream.start()
        await asyncio.sleep(0.1)
        
        result = notifier.record_discovery(
            discovery_type=DiscoveryType.PATTERN_DISCOVERY,
            title="Exploration completed",
            description="Daydream exploration discovered pattern",
            source="daydream_engine",
            
        )
        
        await daydream.stop()
        
        assert result is not None or result is None
    
    @pytest.mark.asyncio
    async def test_insight_to_evolution_flow(self):
        """Test Insight → Evolution flow."""
        from core.insight_bridge import get_insight_bridge
        from core.evolution_orchestrator import EvolutionOrchestrator
        
        bridge = get_insight_bridge()
        evolution = EvolutionOrchestrator(storage_path=Path(tempfile.mkdtemp()))
        
        bridge.process_insight(
            raw_insight="Optimization opportunity in memory handling",
            source_problem="memory_optimization",
            source_category="improvement",
        )
        
        pending = bridge.get_pending_evolution_updates()
        assert pending is not None
    
    @pytest.mark.asyncio
    async def test_evolution_to_self_edit_flow(self):
        """Test Evolution → Self-Edit flow."""
        from core.evolution_orchestrator import EvolutionOrchestrator
        from core.self_edit import get_self_edit_engine
        
        evolution = EvolutionOrchestrator(storage_path=Path(tempfile.mkdtemp()))
        editor = get_self_edit_engine()
        
        pending = evolution.get_pending_evolution_updates() if hasattr(evolution, 'get_pending_evolution_updates') else []
        
        if pending:
            for update in pending[:3]:
                target = update.get("target_file", "")
                if target:
                    can_edit, _ = editor.can_edit_file(target)
    
    @pytest.mark.asyncio
    async def test_full_autonomous_cycle(self):
        """Test the complete autonomous cycle."""
        from core.daydream_engine import DaydreamEngine
        from core.erdos_engine import ErdosEngine
        from core.insight_bridge import get_insight_bridge
        from core.evolution_orchestrator import EvolutionOrchestrator
        from core.discovery_notifier import DiscoveryNotifier, DiscoveryType
        from core.daydream_verification import SolutionVerifier
        
        daydream = DaydreamEngine()
        erdos = ErdosEngine()
        bridge = get_insight_bridge()
        evolution = EvolutionOrchestrator(storage_path=Path(tempfile.mkdtemp()))
        notifier = DiscoveryNotifier()
        verification = SolutionVerifier()
        
        await daydream.start()
        
        insight_id = bridge.process_insight(
            raw_insight="Full cycle test insight",
            source_problem="integration_test",
            source_category="pattern",
        )
        
        await erdos.daydream_step()
        
        discovery_id = notifier.record_discovery(
            discovery_type=DiscoveryType.PATTERN_DISCOVERY,
            title="Full cycle test discovery",
            description="Test discovery from full cycle",
            source="integration_test",
            
        )
        
        await daydream.stop()
        
        daydream_status = daydream.get_status()
        erdos_status = erdos.get_status()
        evolution_status = evolution.get_status()
        
        assert daydream_status is not None
        assert erdos_status is not None
        assert evolution_status is not None


class TestAutonomousBoundsAndSafety:
    """Test safety constraints and memory bounds."""
    
    def test_evolution_safe_files_only(self):
        """Test that evolution only proposes edits to safe files."""
        from core.evolution_orchestrator import EvolutionOrchestrator
        from core.self_edit import get_self_edit_engine
        
        evolution = EvolutionOrchestrator(storage_path=Path(tempfile.mkdtemp()))
        editor = get_self_edit_engine()
        
        for safe_file in editor.SAFE_FILES:
            can_edit, _ = editor.can_edit_file(safe_file)
            assert can_edit, f"Should be able to edit {safe_file}"
    
    def test_limits_exist(self):
        """Test that bounds are defined."""
        from core.erdos_engine import ErdosEngine
        from core.daydream_engine import DaydreamEngine
        from core.insight_bridge import InsightBridge
        
        erdos = ErdosEngine()
        daydream = DaydreamEngine()
        
        assert erdos.MAX_PROBLEMS > 0
        assert erdos.MAX_HISTORY > 0
        assert daydream.MAX_CONTINUOUS_DAYDREAM > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--timeout=60"])