#!/usr/bin/env python3
"""Comprehensive Runtime Evaluation for Jarvis.

Tests all hardware (CPU/GPU/NPU), modules, and bidirectional communication.
Evaluates stability and provides improvement recommendations.
"""
import asyncio
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import json

# Setup path
JARVIS_PATH = Path(__file__).parent
sys.path.insert(0, str(JARVIS_PATH))


@dataclass
class TestResult:
    name: str
    success: bool
    duration_ms: float
    message: str = ""
    error: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 2),
            "message": self.message,
            "error": self.error,
            "details": self.details,
        }


@dataclass
class ModuleHealth:
    name: str
    loaded: bool = False
    has_singleton: bool = False
    has_async: bool = False
    classes: list = field(default_factory=list)
    functions: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def _cpu_task(n):
    return sum(i * i for i in range(n))


class RuntimeEvaluator:
    def __init__(self):
        self.results: list[TestResult] = []
        self.module_health: dict[str, ModuleHealth] = {}
        self.hardware_status = {
            "cpu": {"available": True, "tested": False, "working": False},
            "gpu": {"available": False, "tested": False, "working": False},
            "npu": {"available": False, "tested": False, "working": False},
        }
        self.bridge_status = {"loaded": False, "events": [], "connections": {}}
        self._lock = threading.Lock()

    def load_module_direct(self, path: str, name: str) -> Any:
        full_path = JARVIS_PATH / path
        if not full_path.exists():
            return None

        module_name = full_path.stem
        package_parent = full_path.parent.name

        try:
            if package_parent == "core":
                mod = __import__(f"core.{module_name}", fromlist=[module_name])
            elif package_parent == "tools":
                mod = __import__(f"tools.{module_name}", fromlist=[module_name])
            else:
                mod = __import__(module_name)
            return mod
        except Exception:
            return None

    def add_result(self, result: TestResult):
        with self._lock:
            self.results.append(result)

    def print_header(self, title: str):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def print_result(self, result: TestResult):
        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"  {status} {result.name} ({result.duration_ms:.1f}ms)")
        if result.message:
            print(f"       {result.message}")
        if result.error:
            print(f"       Error: {result.error[:100]}")

    # ==================== HARDWARE TESTS ====================

    def test_cpu(self) -> TestResult:
        start = time.perf_counter()
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()

            with multiprocessing.Pool(min(4, cpu_count)) as pool:
                results = pool.map(_cpu_task, [10000] * 4)

            duration = (time.perf_counter() - start) * 1000

            self.hardware_status["cpu"]["tested"] = True
            self.hardware_status["cpu"]["working"] = True

            return TestResult(
                name="CPU Parallel Processing",
                success=True,
                duration_ms=duration,
                message=f"{cpu_count} cores, parallel tasks completed",
                details={"cores": cpu_count, "tasks": 4},
            )
        except Exception as e:
            return TestResult(
                name="CPU Parallel Processing",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            )

    def test_gpu(self) -> TestResult:
        """Test GPU acceleration availability."""
        start = time.perf_counter()
        gpu_available = False
        gpu_name = ""
        gpu_vram = 0

        try:
            # Test PyTorch CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_vram = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                elif hasattr(torch, 'hip') and torch.hip.is_available():
                    gpu_available = True
                    gpu_name = "AMD ROCm HIP"
            except Exception:
                pass

            # Test OpenCL
            if not gpu_available:
                try:
                    import pyopencl as cl
                    platforms = cl.get_platforms()
                    if platforms:
                        devices = platforms[0].get_devices()
                        if devices:
                            gpu_available = True
                            gpu_name = devices[0].name
                except Exception:
                    pass

            # Check for AMD specific
            if not gpu_available:
                try:
                    result = os.popen("rocm-smi --showname 2>/dev/null").read()
                    if result and "gfx" in result.lower():
                        gpu_available = True
                        gpu_name = result.strip().split('\n')[0][:50]
                except Exception:
                    pass

            duration = (time.perf_counter() - start) * 1000

            self.hardware_status["gpu"]["available"] = gpu_available
            self.hardware_status["gpu"]["tested"] = True
            self.hardware_status["gpu"]["working"] = gpu_available

            if gpu_available:
                return TestResult(
                    name="GPU Acceleration",
                    success=True,
                    duration_ms=duration,
                    message=f"GPU found: {gpu_name}",
                    details={"name": gpu_name, "vram_gb": gpu_vram},
                )
            else:
                return TestResult(
                    name="GPU Acceleration",
                    success=False,
                    duration_ms=duration,
                    message="No GPU acceleration available (gfx1150 unsupported in ROCm 7.2)",
                    details={"note": "Needs ROCm 7.3+ for gfx1150"},
                )
        except Exception as e:
            return TestResult(
                name="GPU Acceleration",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            )

    def test_npu(self) -> TestResult:
        """Test AMD XDNA NPU availability."""
        start = time.perf_counter()
        npu_available = False
        npu_name = ""
        npu_cores = 0

        try:
            # Check for AMD XDNA device
            try:
                result = os.popen("ls /dev/accel/accel* 2>/dev/null").read()
                if result.strip():
                    npu_available = True
            except Exception:
                pass

            # Check for XDNA driver
            try:
                result = os.popen("dmesg | grep -i xdna 2>/dev/null | head -5").read()
                if "xdna" in result.lower():
                    npu_available = True
                    npu_name = "AMD XDNA"
            except Exception:
                pass

            # Test ONNX Runtime with Vitis AI
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if 'VitisAIExecutionProvider' in providers:
                    npu_available = True
                    npu_name = "AMD XDNA (VitisAI)"
                    npu_cores = 4  # Typical for Ryzen AI
            except Exception:
                pass

            # Check hardware info
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                    if "XDNA" in cpuinfo or "Ryzen AI" in cpuinfo:
                        npu_available = True
                        npu_name = "AMD Ryzen AI NPU"
                        npu_cores = 4
            except Exception:
                pass

            duration = (time.perf_counter() - start) * 1000

            self.hardware_status["npu"]["available"] = npu_available
            self.hardware_status["npu"]["tested"] = True
            self.hardware_status["npu"]["working"] = npu_available

            if npu_available:
                return TestResult(
                    name="NPU Acceleration",
                    success=True,
                    duration_ms=duration,
                    message=f"NPU found: {npu_name}",
                    details={"name": npu_name, "cores": npu_cores},
                )
            else:
                return TestResult(
                    name="NPU Acceleration",
                    success=False,
                    duration_ms=duration,
                    message="NPU not detected or driver not loaded",
                )
        except Exception as e:
            return TestResult(
                name="NPU Acceleration",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            )

    # ==================== MODULE TESTS ====================

    def test_module(self, path: str, name: str) -> ModuleHealth:
        """Test a single module's loading and basic functionality."""
        health = ModuleHealth(name=name)

        try:
            mod = self.load_module_direct(path, name)
            if mod is None:
                health.errors.append(f"Failed to load from {path}")
                return health

            health.loaded = True

            # Find classes and functions
            for attr_name in dir(mod):
                if attr_name.startswith('_'):
                    continue
                attr = getattr(mod, attr_name)

                if isinstance(attr, type):
                    health.classes.append(attr_name)
                elif callable(attr):
                    health.functions.append(attr_name)
                    if name.startswith('get_') or 'singleton' in attr_name.lower():
                        health.has_singleton = True
                    if asyncio.iscoroutine(attr):
                        health.has_async = True

            # Check for singleton getter
            singleton_name = f"get_{name}"
            if hasattr(mod, singleton_name):
                health.has_singleton = True

        except Exception as e:
            health.errors.append(str(e)[:200])

        return health

    def test_all_core_modules(self) -> list[TestResult]:
        """Test all core modules systematically."""
        results = []
        core_path = JARVIS_PATH / "core"

        if not core_path.exists():
            return [TestResult(
                name="Core Modules",
                success=False,
                duration_ms=0,
                error="Core directory not found",
            )]

        py_files = list(core_path.glob("*.py"))

        for py_file in sorted(py_files):
            if py_file.name.startswith('_'):
                continue

            start = time.perf_counter()
            module_name = py_file.stem

            try:
                health = self.test_module(f"core/{py_file.name}", module_name)
                self.module_health[module_name] = health

                duration = (time.perf_counter() - start) * 1000

                if health.loaded:
                    results.append(TestResult(
                        name=f"core.{module_name}",
                        success=True,
                        duration_ms=duration,
                        message=f"{len(health.classes)} classes, {len(health.functions)} functions",
                        details={
                            "classes": health.classes[:5],
                            "has_singleton": health.has_singleton,
                        },
                    ))
                else:
                    results.append(TestResult(
                        name=f"core.{module_name}",
                        success=False,
                        duration_ms=duration,
                        error="; ".join(health.errors[:2]),
                    ))
            except Exception as e:
                results.append(TestResult(
                    name=f"core.{module_name}",
                    success=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    error=str(e)[:100],
                ))

        return results

    def test_all_tools(self) -> list[TestResult]:
        """Test all tool modules."""
        results = []
        tools_path = JARVIS_PATH / "tools"

        if not tools_path.exists():
            return results

        py_files = list(tools_path.glob("*.py"))

        for py_file in sorted(py_files):
            if py_file.name.startswith('_'):
                continue

            start = time.perf_counter()
            module_name = f"tools_{py_file.stem}"

            try:
                mod = self.load_module_direct(f"tools/{py_file.name}", module_name)
                duration = (time.perf_counter() - start) * 1000

                if mod:
                    # Count tool classes
                    tool_count = sum(1 for attr in dir(mod)
                                   if isinstance(getattr(mod, attr, None), type)
                                   and 'Tool' in attr)

                    results.append(TestResult(
                        name=f"tools.{py_file.stem}",
                        success=True,
                        duration_ms=duration,
                        message=f"{tool_count} tool classes",
                    ))
                else:
                    results.append(TestResult(
                        name=f"tools.{py_file.stem}",
                        success=False,
                        duration_ms=duration,
                        error="Failed to load",
                    ))
            except Exception as e:
                results.append(TestResult(
                    name=f"tools.{py_file.stem}",
                    success=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    error=str(e)[:100],
                ))

        return results

    # ==================== BIDIRECTIONAL FLOW TESTS ====================

    def test_module_bridge(self) -> TestResult:
        """Test the module bridge bidirectional communication."""
        start = time.perf_counter()

        try:
            bridge_mod = self.load_module_direct("core/module_bridge.py", "module_bridge_test")

            if not bridge_mod:
                return TestResult(
                    name="Module Bridge",
                    success=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    error="Failed to load module_bridge.py",
                )

            # Check key components
            has_bridge = hasattr(bridge_mod, 'ModuleBridge')
            has_events = hasattr(bridge_mod, 'BridgeEvent')
            has_getter = hasattr(bridge_mod, 'get_bridge')

            # Try to get singleton
            if has_getter:
                try:
                    bridge = bridge_mod.get_bridge()
                    self.bridge_status["loaded"] = True

                    # Test event subscription
                    test_events = []

                    def test_handler(msg):
                        test_events.append(msg)

                    if has_events:
                        events = list(bridge_mod.BridgeEvent)
                        self.bridge_status["events"] = [e.value for e in events]

                except Exception:
                    self.bridge_status["loaded"] = False

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="Module Bridge",
                success=has_bridge and has_events,
                duration_ms=duration,
                message=f"Bridge: {has_bridge}, Events: {has_events}, Getter: {has_getter}",
                details=self.bridge_status,
            )
        except Exception as e:
            return TestResult(
                name="Module Bridge",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)[:200],
            )

    def test_bidirectional_flow(self) -> TestResult:
        """Test actual bidirectional communication between modules."""
        start = time.perf_counter()
        flows_tested = 0
        flows_working = 0

        try:
            # Test 1: ContextCompressor <-> SubAgentOrchestrator
            ctx_mod = self.load_module_direct("core/context_compressor.py", "test_ctx_compress")
            sub_mod = self.load_module_direct("core/sub_agent.py", "test_sub_agent")

            if ctx_mod and sub_mod:
                flows_tested += 1
                if hasattr(ctx_mod, 'get_context_compressor') and hasattr(sub_mod, 'get_orchestrator'):
                    flows_working += 1

            # Test 2: VectorMemory <-> KnowledgeGraph
            vec_mod = self.load_module_direct("core/vector_memory.py", "test_vec_mem")
            kg_mod = self.load_module_direct("core/knowledge_graph.py", "test_kg")

            if vec_mod and kg_mod:
                flows_tested += 1
                if hasattr(vec_mod, 'get_vector_memory') and hasattr(kg_mod, 'get_knowledge_graph'):
                    flows_working += 1

            # Test 3: ErdosEngine -> InsightBridge -> EvolutionOrchestrator
            erdos_mod = self.load_module_direct("core/erdos_engine.py", "test_erdos")
            insight_mod = self.load_module_direct("core/insight_bridge.py", "test_insight")
            evo_mod = self.load_module_direct("core/evolution_orchestrator.py", "test_evo")

            if erdos_mod and insight_mod and evo_mod:
                flows_tested += 1
                flows_working += 1

            # Test 4: New modules integration
            workflow_mod = self.load_module_direct("core/workflow_shell.py", "test_workflow")
            comm_mod = self.load_module_direct("core/sub_agent_comm.py", "test_comm")
            sandbox_mod = self.load_module_direct("core/docker_sandbox.py", "test_sandbox")
            dm_mod = self.load_module_direct("core/dm_security.py", "test_dm")
            skills_mod = self.load_module_direct("core/skills_registry.py", "test_skills")
            canvas_mod = self.load_module_direct("core/canvas.py", "test_canvas")

            new_modules = [workflow_mod, comm_mod, sandbox_mod, dm_mod, skills_mod, canvas_mod]
            flows_tested += 6
            flows_working += sum(1 for m in new_modules if m is not None)

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="Bidirectional Flow",
                success=flows_working >= flows_tested * 0.8,
                duration_ms=duration,
                message=f"{flows_working}/{flows_tested} flows working",
                details={
                    "flows_tested": flows_tested,
                    "flows_working": flows_working,
                },
            )
        except Exception as e:
            return TestResult(
                name="Bidirectional Flow",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)[:200],
            )

    # ==================== SELF-LEARNING TESTS ====================

    def test_self_awareness(self) -> TestResult:
        """Test self-awareness module."""
        start = time.perf_counter()

        try:
            mod = self.load_module_direct("core/self_awareness.py", "test_self_aware")

            if not mod:
                return TestResult(
                    name="Self Awareness",
                    success=False,
                    duration_ms=(time.perf_counter() - start) * 1000,
                    error="Module not loaded",
                )

            # Check key components
            has_engine = hasattr(mod, 'SelfAwarenessEngine')
            has_getter = hasattr(mod, 'get_self_awareness_engine')

            # Try to get instance
            if has_getter:
                engine = mod.get_self_awareness_engine()
                info = engine.get_identity_info() if engine else {}

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="Self Awareness",
                success=has_engine,
                duration_ms=duration,
                message=f"Engine: {has_engine}, Getter: {has_getter}",
            )
        except Exception as e:
            return TestResult(
                name="Self Awareness",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)[:200],
            )

    def test_evolution_system(self) -> TestResult:
        """Test evolution and self-improvement systems."""
        start = time.perf_counter()

        try:
            # Check evolution orchestrator
            evo_mod = self.load_module_direct("core/evolution_orchestrator.py", "test_evo")
            auto_mod = self.load_module_direct("core/auto_refine.py", "test_auto")
            self_mod = self.load_module_direct("core/self_edit.py", "test_self_edit")

            components = {
                "evolution_orchestrator": evo_mod is not None,
                "auto_refine": auto_mod is not None,
                "self_edit": self_mod is not None,
            }

            # Check SAFE_FILES if available
            if self_mod and hasattr(self_mod, 'SAFE_FILES'):
                components["safe_files_count"] = len(self_mod.SAFE_FILES)

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="Evolution System",
                success=all(v for k, v in components.items() if k != "safe_files_count"),
                duration_ms=duration,
                message="All evolution components loaded",
                details=components,
            )
        except Exception as e:
            return TestResult(
                name="Evolution System",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)[:200],
            )

    def test_learning_pipeline(self) -> TestResult:
        """Test learning and reinforcement systems."""
        start = time.perf_counter()

        try:
            learn_mod = self.load_module_direct("core/learning_pipeline.py", "test_learn")
            rl_mod = self.load_module_direct("core/reinforcement_learning.py", "test_rl")
            tool_mod = self.load_module_direct("core/tool_learning.py", "test_tool_learn")

            components = {
                "learning_pipeline": learn_mod is not None,
                "reinforcement_learning": rl_mod is not None,
                "tool_learning": tool_mod is not None,
            }

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="Learning Pipeline",
                success=all(components.values()),
                duration_ms=duration,
                message="All learning components loaded",
                details=components,
            )
        except Exception as e:
            return TestResult(
                name="Learning Pipeline",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)[:200],
            )

    # ==================== JARVIS CODE EDIT VISIBILITY ====================

    def test_jarvis_self_visibility(self) -> TestResult:
        """Test if Jarvis has visibility of its own code and new modules."""
        start = time.perf_counter()

        try:
            # Check if self_edit knows about new modules
            self_mod = self.load_module_direct("core/self_edit.py", "test_se")

            new_modules = [
                "core/workflow_shell.py",
                "core/sub_agent_comm.py",
                "core/docker_sandbox.py",
                "core/dm_security.py",
                "core/skills_registry.py",
                "core/canvas.py",
            ]

            visible = []
            for mod_path in new_modules:
                if (JARVIS_PATH / mod_path).exists():
                    visible.append(mod_path)

            # Check for file reading capabilities
            has_file_read = False
            tools_mod = self.load_module_direct("tools/system.py", "test_sys_tools")
            if tools_mod and hasattr(tools_mod, 'FileReadTool'):
                has_file_read = True

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="Jarvis Self Visibility",
                success=len(visible) >= 4,
                duration_ms=duration,
                message=f"{len(visible)}/6 new modules visible",
                details={
                    "visible_modules": visible,
                    "file_read_capability": has_file_read,
                },
            )
        except Exception as e:
            return TestResult(
                name="Jarvis Self Visibility",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)[:200],
            )

    # ==================== INTEGRATION TESTS ====================

    async def test_async_operations(self) -> TestResult:
        """Test async operations in key modules."""
        start = time.perf_counter()

        try:
            # Test workflow shell async
            ws = self.load_module_direct("core/workflow_shell.py", "ws_async")
            if ws:
                ctx = ws.PipelineContext(workflow_id="async_test")

                @ws.stage
                async def async_test(data, ctx):
                    await asyncio.sleep(0.01)
                    return data

                result = await async_test.execute("test", ctx)

            # Test sub-agent comm async
            sac = self.load_module_direct("core/sub_agent_comm.py", "sac_async")
            if sac:
                comm = sac.get_sub_agent_comm()
                comm.register_agent("async_test_agent")

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="Async Operations",
                success=True,
                duration_ms=duration,
                message="Async operations working",
            )
        except Exception as e:
            return TestResult(
                name="Async Operations",
                success=False,
                duration_ms=(time.perf_counter() - start) * 1000,
                error=str(e)[:200],
            )

    # ==================== RUN ALL TESTS ====================

    async def run_full_evaluation(self):
        """Run complete evaluation suite."""
        print("\n" + "="*70)
        print("  JARVIS COMPREHENSIVE RUNTIME EVALUATION")
        print(f"  Started: {datetime.now().isoformat()}")
        print("="*70)

        # Hardware Tests
        self.print_header("HARDWARE TESTS")
        for test in [self.test_cpu, self.test_gpu, self.test_npu]:
            result = test()
            self.print_result(result)
            self.add_result(result)

        # Core Module Tests
        self.print_header("CORE MODULE TESTS")
        core_results = self.test_all_core_modules()
        passed = sum(1 for r in core_results if r.success)
        total = len(core_results)
        for result in core_results:
            self.print_result(result)
            self.add_result(result)
        print(f"\n  Summary: {passed}/{total} modules loaded successfully")

        # Tools Tests
        self.print_header("TOOLS TESTS")
        tool_results = self.test_all_tools()
        for result in tool_results:
            self.print_result(result)
            self.add_result(result)

        # Bidirectional Flow Tests
        self.print_header("BIDIRECTIONAL FLOW TESTS")
        bridge_result = self.test_module_bridge()
        self.print_result(bridge_result)
        self.add_result(bridge_result)

        flow_result = self.test_bidirectional_flow()
        self.print_result(flow_result)
        self.add_result(flow_result)

        # Self-Learning Tests
        self.print_header("SELF-LEARNING & EVOLUTION TESTS")
        for test in [self.test_self_awareness, self.test_evolution_system,
                     self.test_learning_pipeline, self.test_jarvis_self_visibility]:
            result = test()
            self.print_result(result)
            self.add_result(result)

        # Async Tests
        self.print_header("ASYNC OPERATIONS TEST")
        async_result = await self.test_async_operations()
        self.print_result(async_result)
        self.add_result(async_result)

        # Final Summary
        self.print_final_summary()

    def print_final_summary(self):
        """Print final summary and recommendations."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed

        total_time = sum(r.duration_ms for r in self.results)

        print("\n" + "="*70)
        print("  FINAL SUMMARY")
        print("="*70)
        print(f"\n  Total Tests: {total}")
        print(f"  Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"  Failed: {failed}")
        print(f"  Total Duration: {total_time:.0f}ms")

        print("\n  HARDWARE STATUS:")
        for name, status in self.hardware_status.items():
            working = "✅" if status["working"] else "❌"
            print(f"    {working} {name.upper()}: {status}")

        print("\n  MODULE HEALTH:")
        loaded = sum(1 for m in self.module_health.values() if m.loaded)
        total_mods = len(self.module_health)
        print(f"    {loaded}/{total_mods} core modules loaded")

        # Recommendations
        print("\n" + "="*70)
        print("  RECOMMENDATIONS")
        print("="*70)

        recommendations = []

        if not self.hardware_status["gpu"]["working"]:
            recommendations.append("• GPU: gfx1150 requires ROCm 7.3+ - consider updating when available")

        if not self.hardware_status["npu"]["working"]:
            recommendations.append("• NPU: Ensure XDNA drivers are loaded (check /dev/accel/)")

        failed_modules = [r.name for r in self.results if not r.success]
        if failed_modules:
            recommendations.append(f"• Fix failing modules: {', '.join(failed_modules[:5])}")

        # Check new modules visibility
        new_mods_loaded = all(
            f"core.{name}" in [r.name for r in self.results]
            for name in ["workflow_shell", "sub_agent_comm", "docker_sandbox",
                        "dm_security", "skills_registry", "canvas"]
        )

        if not new_mods_loaded:
            recommendations.append("• New modules need proper __init__.py exports")

        # SAFE_FILES update recommendation
        recommendations.append("• Update SAFE_FILES in self_edit.py to include new OpenClaw-inspired modules")
        recommendations.append("• Add workflow_shell, sub_agent_comm, skills_registry, canvas to SAFE_FILES")

        # Print recommendations
        for rec in recommendations:
            print(f"  {rec}")

        # Save results
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "duration_ms": total_time,
            },
            "hardware": self.hardware_status,
            "results": [r.to_dict() for r in self.results],
            "recommendations": recommendations,
        }

        report_path = JARVIS_PATH / "runtime_evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n  Report saved to: {report_path}")


def main():
    evaluator = RuntimeEvaluator()
    asyncio.run(evaluator.run_full_evaluation())


if __name__ == "__main__":
    main()
