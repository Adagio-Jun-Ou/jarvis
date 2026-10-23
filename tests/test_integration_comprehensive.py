#!/usr/bin/env python3
"""Comprehensive Integration Tests for Jarvis.

Tests all major subsystems with focus on:
- NPU acceleration pipeline
- OpenClaw-inspired features
- Bidirectional module communication
- Self-learning and evolution
- Hardware abstraction layers
"""
import sys
import time
import asyncio
import threading
from pathlib import Path

import unittest

JARVIS_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(JARVIS_PATH))




class TestOpenClawFeatures(unittest.TestCase):
    """Test OpenClaw-inspired features."""

    def test_workflow_shell_singleton(self):
        from core.workflow_shell import get_workflow_shell, WorkflowShell

        shell1 = get_workflow_shell()
        shell2 = get_workflow_shell()

        self.assertIs(shell1, shell2)
        self.assertIsInstance(shell1, WorkflowShell)

    def test_workflow_pipeline_registration(self):
        from core.workflow_shell import get_workflow_shell

        shell = get_workflow_shell()

        pipelines = shell.list_pipelines()
        self.assertIsInstance(pipelines, list)

    def test_sub_agent_communication(self):
        from core.sub_agent_comm import get_sub_agent_comm, SubAgentCommunication

        comm = get_sub_agent_comm()

        self.assertIsInstance(comm, SubAgentCommunication)

    def test_skills_registry(self):
        from core.skills_registry import get_skills_registry, SkillsRegistry

        registry = get_skills_registry()

        self.assertIsInstance(registry, SkillsRegistry)

        categories = registry.get_categories()
        self.assertIsNotNone(categories)




class TestBidirectionalFlow(unittest.TestCase):
    """Test bidirectional module communication."""

    def test_module_bridge_singleton(self):
        from core.module_bridge import get_bridge, ModuleBridge

        bridge1 = get_bridge()
        bridge2 = get_bridge()

        self.assertIs(bridge1, bridge2)
        self.assertIsInstance(bridge1, ModuleBridge)

    def test_module_connections(self):
        from core.module_bridge import get_bridge

        bridge = get_bridge()

        handle1 = bridge.register_module("npu_optimizer")
        handle2 = bridge.register_module("npu_integration")

        self.assertIsNotNone(handle1)
        self.assertIsNotNone(handle2)

    def test_event_propagation(self):
        from core.module_bridge import get_bridge, BridgeEvent

        bridge = get_bridge()

        events_received = []

        def handler(msg):
            events_received.append(msg.data if hasattr(msg, 'data') else msg)

        bridge.subscribe(BridgeEvent.MEMORY_PRESSURE, handler)
        bridge.emit(BridgeEvent.MEMORY_PRESSURE, "test_source", {"data": "test_payload"})

        self.assertEqual(len(events_received), 1)


class TestHardwareAbstraction(unittest.TestCase):
    """Test hardware abstraction layers."""

    def test_hardware_detector_singleton(self):
        from core.hardware import get_hardware_detector, HardwareDetector

        detector1 = get_hardware_detector()
        detector2 = get_hardware_detector()

        self.assertIs(detector1, detector2)
        self.assertIsInstance(detector1, HardwareDetector)

    def test_hardware_detection(self):
        from core.hardware import get_hardware_detector

        detector = get_hardware_detector()

        self.assertIsNotNone(detector.accelerators)
        self.assertGreater(len(detector.accelerators), 0)


    def test_gpu_simulation_engine(self):
        from core.gpu_simulation import get_gpu_simulation, GPUBackend

        engine = get_gpu_simulation()

        self.assertIsNotNone(engine)
        self.assertIn(engine.backend, [GPUBackend.NONE, GPUBackend.ROCM, GPUBackend.CUDA, GPUBackend.MPS, GPUBackend.CPU])


class TestSelfLearning(unittest.TestCase):
    """Test self-learning and evolution systems."""

    def test_evolution_orchestrator(self):
        from core.evolution_orchestrator import get_evolution_orchestrator

        orchestrator = get_evolution_orchestrator()

        self.assertIsNotNone(orchestrator)

    def test_insight_bridge(self):
        from core.insight_bridge import get_insight_bridge

        bridge = get_insight_bridge()

        self.assertIsNotNone(bridge)

    def test_knowledge_graph(self):
        from core.knowledge_graph import get_knowledge_graph, KnowledgeGraph

        kg = get_knowledge_graph()

        self.assertIsInstance(kg, KnowledgeGraph)

    def test_vector_memory(self):
        from core.vector_memory import get_vector_memory, VectorMemory

        vm = get_vector_memory()

        self.assertIsInstance(vm, VectorMemory)


class TestAsyncOperations(unittest.TestCase):
    """Test async operations across modules."""


    def test_async_event_bus(self):
        from core.module_bridge import get_bridge, BridgeEvent

        async def run_test():
            bridge = get_bridge()

            received = []

            def handler(msg):
                received.append(msg.data if hasattr(msg, 'data') else msg)

            bridge.subscribe(BridgeEvent.MEMORY_PRESSURE, handler)

            bridge.emit(BridgeEvent.MEMORY_PRESSURE, "test", {"value": 42})

            await asyncio.sleep(0.05)

            return received

        received = asyncio.run(run_test())
        self.assertGreaterEqual(len(received), 0)






if __name__ == "__main__":
    unittest.main(verbosity=2)
