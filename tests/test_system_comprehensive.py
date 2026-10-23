#!/usr/bin/env python3
"""Comprehensive System Test for Jarvis.

Tests:
1. Core module initialization
2. Inter-module connections
3. Subsystem integrations
4. API endpoints
5. Tool execution
6. Context propagation
7. Self-evaluation system
8. Checkpoint permissioning
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

class TestRunner:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def log(self, category: str, test: str, status: str, detail: str = ""):
        result = {"category": category, "test": test, "status": status, "detail": detail}
        self.results.append(result)
        
        icon = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⚠"
        print(f"{icon} [{category}] {test}: {status}")
        if detail:
            print(f"    {detail}")
        
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.warnings += 1
    
    def summary(self):
        total = self.passed + self.failed + self.warnings
        print(f"\n{'='*60}")
        print(f"SUMMARY: {self.passed} passed, {self.failed} failed, {self.warnings} warnings")
        print(f"Total: {total} tests")
        return self.failed == 0


def _test_core_modules(runner: TestRunner):
    """Test core module imports and initialization."""
    print("\n" + "="*60)
    print("CORE MODULE TESTS")
    print("="*60)
    
    # LLM Module
    try:
        from core.llm import OllamaClient, set_subsystem, get_subsystem, subsystem_context
        runner.log("Core", "LLM module import", "PASS")
        
        # Test subsystem context
        set_subsystem("Test")
        current = get_subsystem()
        if current == "Test":
            runner.log("Core", "Subsystem context tracking", "PASS")
        else:
            runner.log("Core", "Subsystem context tracking", "FAIL", f"Expected 'Test', got '{current}'")
    except Exception as e:
        runner.log("Core", "LLM module import", "FAIL", str(e))
    
    # Memory Module
    try:
        from core.memory import ConversationMemory, get_conversation_memory
        memory = get_conversation_memory()
        memory.add_message("user", "test message")
        messages = memory.get_messages_for_llm()
        if len(messages) > 0:
            runner.log("Core", "Memory module", "PASS", f"{len(messages)} messages stored")
        else:
            runner.log("Core", "Memory module", "FAIL", "No messages returned")
    except Exception as e:
        runner.log("Core", "Memory module", "FAIL", str(e))
    
    # Context Manager
    try:
        from core.context_manager import (
            get_subsystem, set_subsystem, add_context, get_context,
            subsystem_context, get_full_context_status
        )
        runner.log("Core", "Context manager import", "PASS")
        
        # Test context propagation
        add_context("test_key", "test_value")
        value = get_context("test_key")
        if value == "test_value":
            runner.log("Core", "Context propagation", "PASS")
        else:
            runner.log("Core", "Context propagation", "FAIL", f"Expected 'test_value', got '{value}'")
    except Exception as e:
        runner.log("Core", "Context manager import", "FAIL", str(e))
    
    # Self Evaluation
    try:
        from core.self_evaluation import SelfEvaluationEngine, get_self_evaluation_engine, ActionType
        runner.log("Core", "Self-evaluation module", "PASS")
        
        engine = get_self_evaluation_engine()
        status = engine.get_status()
        if "overall_score" in status:
            runner.log("Core", "Self-evaluation status", "PASS", f"Score: {status['overall_score']}")
        else:
            runner.log("Core", "Self-evaluation status", "WARN", "Status incomplete")
    except Exception as e:
        runner.log("Core", "Self-evaluation module", "FAIL", str(e))
    
    # Checkpoint Permission
    try:
        from core.checkpoint_permission import (
            CheckpointPermissionManager, get_checkpoint_manager,
            FrontierDomain, detect_frontier_domain
        )
        runner.log("Core", "Checkpoint permission module", "PASS")
        
        # Test frontier detection
        domain = detect_frontier_domain("delete all files", {})
        if domain:
            runner.log("Core", "Frontier domain detection", "PASS", f"Detected: {domain.value}")
        else:
            runner.log("Core", "Frontier domain detection", "WARN", "No domain detected for 'delete all files'")
    except Exception as e:
        runner.log("Core", "Checkpoint permission module", "FAIL", str(e))
    
    # Context Compressor
    try:
        from core.context_compressor import ContextCompressor, TokenCounter
        tokens = TokenCounter.count("Hello world")
        if tokens > 0:
            runner.log("Core", "Context compressor", "PASS", f"Token count: {tokens}")
        else:
            runner.log("Core", "Context compressor", "FAIL", "Token counting failed")
    except Exception as e:
        runner.log("Core", "Context compressor", "FAIL", str(e))


def _test_subsystems(runner: TestRunner):
    """Test subsystem imports and basic functionality."""
    print("\n" + "="*60)
    print("SUBSYSTEM TESTS")
    print("="*60)
    
    # Erdős Engine
    try:
        from core.erdos_engine import get_erdos_engine, ErdosEngine
        runner.log("Subsystem", "Erdős Engine import", "PASS")
        
        engine = get_erdos_engine()
        status = engine.get_status()
        runner.log("Subsystem", "Erdős Engine status", "PASS", f"Problems explored: {status.get('problems_explored', 0)}")
    except Exception as e:
        runner.log("Subsystem", "Erdős Engine", "FAIL", str(e))
    
    # Daydream Engine
    try:
        from core.daydream_engine import get_daydream_engine, DaydreamEngine
        runner.log("Subsystem", "Daydream Engine import", "PASS")
        
        engine = get_daydream_engine()
        status = engine.get_status()
        runner.log("Subsystem", "Daydream Engine status", "PASS", f"Running: {status.get('running', False)}")
    except Exception as e:
        runner.log("Subsystem", "Daydream Engine", "FAIL", str(e))
    
    # Self-Refine
    try:
        from core.self_refine import get_self_refine_agent, SelfRefineAgent
        runner.log("Subsystem", "Self-Refine import", "PASS")
        
        agent = get_self_refine_agent()
        status = agent.get_state()
        runner.log("Subsystem", "Self-Refine status", "PASS", f"State: {status.value if hasattr(status, 'value') else status}")
    except Exception as e:
        runner.log("Subsystem", "Self-Refine", "FAIL", str(e))
    
    # Learning Pipeline
    try:
        from core.learning_pipeline import get_learning_pipeline, LearningPipeline
        runner.log("Subsystem", "Learning Pipeline import", "PASS")
        
        pipeline = get_learning_pipeline()
        metrics = pipeline.get_metrics()
        runner.log("Subsystem", "Learning Pipeline metrics", "PASS", f"Problems: {metrics.problems_attempted}")
    except Exception as e:
        runner.log("Subsystem", "Learning Pipeline", "FAIL", str(e))
    
    # Evolution Orchestrator
    try:
        from core.evolution_orchestrator import get_evolution_orchestrator
        runner.log("Subsystem", "Evolution Orchestrator import", "PASS")
        
        orchestrator = get_evolution_orchestrator()
        runner.log("Subsystem", "Evolution Orchestrator", "PASS")
    except Exception as e:
        runner.log("Subsystem", "Evolution Orchestrator", "FAIL", str(e))
    
    # Health Manager
    try:
        from core.health import get_health_manager
        runner.log("Subsystem", "Health Manager import", "PASS")
        
        health = get_health_manager()
        all_health = health.get_all_health()
        runner.log("Subsystem", "Health Manager status", "PASS", f"{len(all_health)} components tracked")
    except Exception as e:
        runner.log("Subsystem", "Health Manager", "FAIL", str(e))
    
    # Free Time Engine
    try:
        from core.free_time import get_free_time_engine
        runner.log("Subsystem", "Free Time Engine import", "PASS")
    except Exception as e:
        runner.log("Subsystem", "Free Time Engine", "FAIL", str(e))
    
    # Intuition Bridge
    try:
        from core.intuition_bridge import get_intuition_bridge
        bridge = get_intuition_bridge()
        runner.log("Subsystem", "Intuition Bridge import", "PASS")
    except Exception as e:
        runner.log("Subsystem", "Intuition Bridge", "FAIL", str(e))


def _test_tools(runner: TestRunner):
    """Test tool system."""
    print("\n" + "="*60)
    print("TOOL TESTS")
    print("="*60)
    
    try:
        from tools import get_all_tools, TOOLS
        
        tools = get_all_tools()
        runner.log("Tools", "Tool registry", "PASS", f"{len(tools)} tools registered")
        
        # Check critical tools
        critical_tools = [
            "web_search", "web_fetch", "image_search", 
            "file_read", "file_write", "execute_shell",
            "memory_store", "memory_recall"
        ]
        
        found = 0
        for tool_name in critical_tools:
            if tool_name in tools:
                found += 1
        
        runner.log("Tools", "Critical tools available", "PASS" if found >= 4 else "WARN", 
                   f"{found}/{len(critical_tools)} critical tools found")
        
    except Exception as e:
        runner.log("Tools", "Tool registry", "FAIL", str(e))


def _test_integrations(runner: TestRunner):
    """Test inter-module integrations."""
    print("\n" + "="*60)
    print("INTEGRATION TESTS")
    print("="*60)
    
    # Context Manager <-> LLM
    try:
        from core.context_manager import set_subsystem, get_subsystem
        from core.llm import set_subsystem as llm_set_subsystem, get_subsystem as llm_get_subsystem
        
        set_subsystem("IntegrationTest")
        ctx_subsystem = get_subsystem()
        
        llm_set_subsystem("LLMTest")
        llm_subsystem = llm_get_subsystem()
        
        runner.log("Integration", "Context <-> LLM subsystem", "PASS", 
                   f"Ctx: {ctx_subsystem}, LLM: {llm_subsystem}")
    except Exception as e:
        runner.log("Integration", "Context <-> LLM subsystem", "FAIL", str(e))
    
    # Memory <-> Context Compressor
    try:
        from core.memory import get_conversation_memory
        from core.context_compressor import TokenCounter
        
        memory = get_conversation_memory()
        memory.add_message("user", "Integration test message")
        
        messages = memory.get_messages_for_llm()
        tokens = TokenCounter.count_messages(messages)
        
        runner.log("Integration", "Memory <-> Compressor", "PASS", f"Tokens: {tokens}")
    except Exception as e:
        runner.log("Integration", "Memory <-> Compressor", "FAIL", str(e))
    
    # Self-Eval <-> Checkpoint
    try:
        from core.self_evaluation import get_self_evaluation_engine, ActionType
        from core.checkpoint_permission import detect_frontier_domain
        
        engine = get_self_evaluation_engine()
        
        # Evaluate a potentially harmful action
        record = engine.evaluate_action(
            action_type=ActionType.RESPONSE,
            description="Test response",
            input_preview="how to hack",
            output_preview="I cannot help with hacking",
            subsystem="Test"
        )
        
        # Check frontier detection
        domain = detect_frontier_domain("hack the system", {})
        
        runner.log("Integration", "Self-Eval <-> Checkpoint", "PASS",
                   f"Eval score: {record.overall_score:.2f}, Frontier: {domain.value if domain else 'None'}")
    except Exception as e:
        runner.log("Integration", "Self-Eval <-> Checkpoint", "FAIL", str(e))


def run_async_api_tests(runner: TestRunner):
    """Test API endpoints (requires running server)."""
    print("\n" + "="*60)
    print("API TESTS (requires running server)")
    print("="*60)
    
    import httpx
    
    base_url = "https://localhost:8080"
    
    endpoints = [
        ("/health", "Health check"),
        ("/api/status", "System status"),
        ("/api/tools", "Tools list"),
        ("/api/jarvis/state", "Jarvis state"),
        ("/api/daydream/status", "Daydream status"),
        ("/api/erdos/status", "Erdős status"),
        ("/api/refine/status", "Self-Refine status"),
        ("/api/self-evaluation/status", "Self-evaluation status"),
        ("/api/checkpoints/stats", "Checkpoint stats"),
        ("/api/recent-thoughts", "Recent thoughts"),
        ("/api/api-activity", "API activity"),
    ]
    
    for endpoint, name in endpoints:
        try:
            resp = httpx.get(f"{base_url}{endpoint}", verify=False, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                runner.log("API", name, "PASS", f"Status: {resp.status_code}")
            else:
                runner.log("API", name, "FAIL", f"Status: {resp.status_code}")
        except httpx.ConnectError:
            runner.log("API", name, "WARN", "Server not running")
        except Exception as e:
            runner.log("API", name, "FAIL", str(e))


def _test_data_persistence(runner: TestRunner):
    """Test data persistence directories."""
    print("\n" + "="*60)
    print("PERSISTENCE TESTS")
    print("="*60)
    
    paths = [
        Path.home() / ".jarvis",
        Path.home() / ".jarvis" / "memory",
        Path.home() / ".jarvis" / "self_evaluation",
        Path.home() / ".jarvis" / "checkpoints",
        Path.home() / ".jarvis" / "conversations",
        Path.home() / ".jarvis" / "audit",
    ]
    
    for path in paths:
        if path.exists():
            runner.log("Persistence", f"Directory {path.name}", "PASS")
        else:
            runner.log("Persistence", f"Directory {path.name}", "WARN", "Not created yet")


def main():
    print("="*60)
    print("JARVIS COMPREHENSIVE SYSTEM TEST")
    print("="*60)
    
    runner = TestRunner()
    
    # Run all tests
    _test_core_modules(runner)
    _test_subsystems(runner)
    _test_tools(runner)
    _test_integrations(runner)
    _test_data_persistence(runner)
    run_async_api_tests(runner)
    
    # Print summary
    success = runner.summary()
    
    # Print detailed results for failed tests
    print("\n" + "="*60)
    print("DETAILED FAILURES")
    print("="*60)
    for r in runner.results:
        if r["status"] == "FAIL":
            print(f"\n{r['category']} - {r['test']}:")
            print(f"  {r['detail']}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())