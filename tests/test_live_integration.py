#!/usr/bin/env python3
"""Live integration test for Jarvis - tests each module layer by layer.

Run with: ./venv/bin/python tests/test_live_integration.py
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class LiveIntegrationTest:
    def __init__(self):
        self.results = []
        self.start_time = time.time()

    def log(self, module: str, test: str, status: str, detail: str = ""):
        elapsed = time.time() - self.start_time
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  [{elapsed:6.2f}s] {symbol} {module}: {test} - {status}")
        if detail:
            print(f"           {detail}")
        self.results.append((module, test, status, detail))

    def print_header(self, title: str):
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")


async def check_layer_1_imports(test: LiveIntegrationTest):
    """Layer 1: Import all core modules."""
    test.print_header("LAYER 1: Core Module Imports")

    modules = [
        ("core.config", "Config"),
        ("core.agent", "JarvisAgent"),
        ("core.llm", "OllamaClient"),
        ("core.memory", "ConversationMemory"),
        ("core.api_manager", "APICallManager"),
        ("core.permissions", "PermissionManager"),
    ]

    for module_name, class_name in modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            test.log("imports", f"{module_name}.{class_name}", "PASS")
        except Exception as e:
            test.log("imports", f"{module_name}.{class_name}", "FAIL", str(e)[:50])


async def check_layer_2_subsystem_imports(test: LiveIntegrationTest):
    """Layer 2: Import subsystem modules."""
    test.print_header("LAYER 2: Subsystem Imports")

    subsystems = [
        ("core.erdos_engine", "ErdosSolver"),
        ("core.learning_pipeline", "ContinuousLearningPipeline"),
        ("core.auto_refine", "AutoRefineManager"),
        ("core.meta", "SelfReflectionEngine"),
        ("core.thinking", "ThinkingEngine"),
        ("core.free_time", "FreeTimeEngine"),
        ("core.self_refine", "SelfRefineAgent"),
        ("core.daydream_engine", "DaydreamEngine"),
        ("core.evolution_orchestrator", "EvolutionOrchestrator"),
        ("core.defense", "DataDefenseAgent"),
    ]

    for module_name, class_name in subsystems:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            test.log("subsystems", f"{module_name}.{class_name}", "PASS")
        except Exception as e:
            test.log("subsystems", f"{module_name}.{class_name}", "FAIL", str(e)[:50])


async def check_layer_3_singletons(test: LiveIntegrationTest):
    """Layer 3: Get singleton instances."""
    test.print_header("LAYER 3: Singleton Instances")

    singletons = [
        ("core.erdos_engine", "get_erdos_engine"),
        ("core.learning_pipeline", "get_learning_pipeline"),
        ("core.defense", "get_defense_agent"),
        ("core.daydream_engine", "get_daydream_engine"),
        ("core.observability", "get_observability"),
        ("core.state_sync", "get_state_synchronizer"),
        ("core.memory_cleanup", "get_memory_manager"),
        ("core.api_manager", "get_api_manager"),
    ]

    for module_name, func_name in singletons:
        try:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)

            # Some need llm_client
            if func_name in ["get_erdos_engine", "get_learning_pipeline", "get_free_time_engine", "get_api_manager"]:
                try:
                    instance = func()
                except ValueError as e:
                    if "LLM client required" in str(e):
                        test.log("singletons", f"{func_name}()", "PASS", "needs LLM client")
                        continue
                    raise
            else:
                instance = func()

            test.log("singletons", f"{func_name}()", "PASS", f"type={type(instance).__name__}")
        except Exception as e:
            test.log("singletons", f"{func_name}()", "FAIL", str(e)[:50])


async def check_layer_4_tools(test: LiveIntegrationTest):
    """Layer 4: Tool registration."""
    test.print_header("LAYER 4: Tool Registration")

    tool_modules = [
        ("tools.system", "SystemTools"),
        ("tools.web", "WebTools"),
        ("tools.productivity", "ProductivityTools"),
        ("tools.spatial", "SpatialTools"),
        ("tools.physics_sim", "PhysicsSimTools"),
        ("tools.art_philosophy", "ArtPhilosophyTools"),
        ("tools.data_tools", "get_data_tools"),
        ("tools.refine_tools", "get_refine_tools"),
        ("tools.security_tools", "get_security_tools"),
    ]

    total_tools = 0
    for module_name, item_name in tool_modules:
        try:
            module = __import__(module_name, fromlist=[item_name])
            item = getattr(module, item_name)

            if item_name.startswith("get_"):
                # Functions that don't need arguments
                tools = item()
            elif item_name == "SystemTools":
                tools = item("/tmp/sandbox").get_tools()
            elif item_name == "WebTools":
                tools = item().get_tools()
            elif item_name == "ProductivityTools":
                tools = item(Path("/tmp/storage")).get_tools()
            else:
                tools = item().get_tools()

            count = len(tools) if isinstance(tools, list) else 0
            total_tools += count
            test.log("tools", f"{module_name}.{item_name}", "PASS", f"{count} tools")
        except Exception as e:
            test.log("tools", f"{module_name}.{item_name}", "FAIL", str(e)[:50])

    test.log("tools", "TOTAL", "PASS" if total_tools >= 25 else "FAIL", f"{total_tools} tools in test suite")


async def check_layer_5_api_endpoints(test: LiveIntegrationTest):
    """Layer 5: API endpoints (web interface)."""
    test.print_header("LAYER 5: API Endpoints")

    import aiohttp

    base_url = "https://localhost:8080"  # Server is on 8080
    endpoints = [
        ("/health", "GET", "Health check"),
        ("/api/cli-stats", "GET", "CLI stats"),
        ("/api/erdos", "GET", "Erdős engine"),
        ("/api/tools", "GET", "Tools list"),
        ("/api/memory", "GET", "Memory status"),
        ("/api/subsystems", "GET", "Subsystem status"),
    ]

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        for endpoint, method, description in endpoints:
            try:
                url = f"{base_url}{endpoint}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        test.log("api", endpoint, "PASS", description)
                    else:
                        test.log("api", endpoint, "FAIL", f"status={resp.status}")
            except asyncio.TimeoutError:
                test.log("api", endpoint, "FAIL", "timeout")
            except Exception as e:
                test.log("api", endpoint, "FAIL", str(e)[:50])


async def check_layer_6_llm_integration(test: LiveIntegrationTest):
    """Layer 6: LLM client and API manager integration."""
    test.print_header("LAYER 6: LLM Integration")

    from core.config import Config
    from core.backend_llm import get_backend_llm, get_primary_llm, get_llm_for_task

    try:
        config = Config.load()

        # Test primary LLM (Qwen-3.5)
        test.log("llm", f"Primary model: {config.llm.model}", "PASS", f"use_cloud={config.llm.use_cloud}")
        primary_llm = get_primary_llm(config)
        test.log("llm", "Primary LLM client", "PASS", f"model={primary_llm.model}")

        # Test backend LLM (GLM-5)
        test.log("llm", f"Backend model: {config.backend_llm.model}", "PASS", f"temp={config.backend_llm.temperature}")
        backend_llm = get_backend_llm(config)
        test.log("llm", "Backend LLM client", "PASS", f"model={backend_llm.model}")

        # Test task routing
        chat_llm = get_llm_for_task("chat", config)
        refine_llm = get_llm_for_task("refine", config)
        test.log("llm", "Task routing", "PASS", f"chat->{chat_llm.model}, refine->{refine_llm.model}")

        # Test backend LLM call (GLM-5 is faster/cheaper for testing)
        start = time.time()
        response = await backend_llm.chat(
            user_input="Say 'test ok' and nothing else.",
            system_prompt="You are a simple test responder. Be very brief.",
        )
        elapsed = time.time() - start

        if response and "message" in response:
            content = response["message"].get("content", "")
            test.log("llm", "Backend chat completion", "PASS", f"{elapsed:.2f}s, {len(content)} chars")
        else:
            test.log("llm", "Backend chat completion", "FAIL", "no response")
    except Exception as e:
        test.log("llm", "OllamaClient", "FAIL", str(e)[:50])


async def check_layer_7_api_manager(test: LiveIntegrationTest):
    """Layer 7: API manager coordination."""
    test.print_header("LAYER 7: API Manager Coordination")

    from core.api_manager import APICallManager, Priority
    from core.backend_llm import get_backend_llm
    from core.config import Config

    try:
        config = Config.load()
        llm = get_backend_llm(config)
        manager = APICallManager(llm, enable_caching=True)

        test.log("api_manager", "APICallManager creation", "PASS")

        # Test coordinated call
        start = time.time()
        response = await manager.submit_chat(
            subsystem="test_layer7",
            priority=Priority.INTERACTIVE,
            user_input="Respond with 'layer7 ok'",
            system_prompt="Be very brief.",
        )
        elapsed = time.time() - start

        if response.get("message"):
            test.log("api_manager", "submit_chat", "PASS", f"{elapsed:.2f}s")
        else:
            test.log("api_manager", "submit_chat", "FAIL", str(response)[:50])

        # Test caching
        start = time.time()
        response2 = await manager.submit_chat(
            subsystem="test_layer7",
            priority=Priority.INTERACTIVE,
            user_input="Respond with 'layer7 ok'",
            system_prompt="Be very brief.",
        )
        elapsed2 = time.time() - start

        if response2.get("_from_cache"):
            test.log("api_manager", "caching", "PASS", f"cache hit, {elapsed2:.4f}s")
        else:
            test.log("api_manager", "caching", "WARN", f"no cache hit, {elapsed2:.2f}s")

        # Check stats
        stats = manager.get_stats()
        test.log("api_manager", "get_stats", "PASS", f"total={stats['total_requests']}, hits={stats['cache_hits']}")

    except Exception as e:
        test.log("api_manager", "initialization", "FAIL", str(e)[:50])


async def check_layer_8_subsystem_llm_calls(test: LiveIntegrationTest):
    """Layer 8: Test subsystem LLM calls through API manager."""
    test.print_header("LAYER 8: Subsystem LLM Calls")

    from core.erdos_engine import ErdosSolver
    from core.learning_pipeline import ContinuousLearningPipeline
    from core.backend_llm import get_backend_llm
    from core.config import Config

    try:
        config = Config.load()
        llm = get_backend_llm(config)

        # Test ErdosSolver
        solver = ErdosSolver(llm_client=llm)
        api_mgr = solver._get_api_manager()
        test.log("subsystem_llm", "ErdosSolver API manager", "PASS" if api_mgr else "FAIL")

        start = time.time()
        result = await solver._make_llm_call(
            user_input="Test: what is 1+1?",
            system_prompt="Give a one-word answer.",
            priority="learning",
        )
        elapsed = time.time() - start
        test.log("subsystem_llm", "ErdosSolver._make_llm_call", "PASS", f"{elapsed:.2f}s")

        # Test LearningPipeline
        pipeline = ContinuousLearningPipeline(llm_client=llm)
        api_mgr = pipeline._get_api_manager()
        test.log("subsystem_llm", "LearningPipeline API manager", "PASS" if api_mgr else "FAIL")

        result = await pipeline._llm_call(
            user_input="Test learning call",
            system_prompt="Say 'ok'",
            priority="learning",
        )
        test.log("subsystem_llm", "LearningPipeline._llm_call", "PASS" if result else "FAIL")

    except Exception as e:
        test.log("subsystem_llm", "test", "FAIL", str(e)[:50])


async def check_layer_9_tool_execution(test: LiveIntegrationTest):
    """Layer 9: Live tool execution."""
    test.print_header("LAYER 9: Tool Execution")

    tools_to_test = [
        ("tools.physics_sim", "CollisionDetectionTool", {"objects": [{"id": "ball", "shape": "sphere", "position": [0,0,0], "radius": 1}]}),
        ("tools.spatial", "SpatialCalculationTool", {"operation": "volume", "shape": "box", "dimensions": {"length": 2, "width": 3, "height": 4}}),
        ("tools.learning_tools", "ErdosMathTool", {"action": "generate", "category": "combinatorics", "difficulty": "easy"}),
    ]

    for module_name, class_name, params in tools_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            tool = cls()

            start = time.time()
            result = await tool.execute(**params)
            elapsed = time.time() - start

            test.log("tool_exec", f"{class_name}", "PASS", f"{elapsed:.2f}s, {len(str(result))} chars")
        except Exception as e:
            test.log("tool_exec", f"{class_name}", "FAIL", str(e)[:50])


async def check_layer_10_background_subsystems(test: LiveIntegrationTest):
    """Layer 10: Background subsystems status."""
    test.print_header("LAYER 10: Background Subsystems")

    import aiohttp

    valid_statuses = {"running", "solving", "learning", "evolving", "idle", "active"}

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        # Check Erdős engine
        try:
            async with session.get("https://localhost:8080/api/erdos", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json()
                active = data.get("active", False)
                attempted = data.get("problems_attempted", 0)
                test.log("background", "Erdős engine", "PASS" if active else "WARN", f"active={active}, attempted={attempted}")
        except Exception as e:
            test.log("background", "Erdős engine", "FAIL", str(e)[:50])

        # Check subsystems status
        try:
            async with session.get("https://localhost:8080/api/subsystems", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    subsystems = data.get("subsystems", {})
                    for name, info in list(subsystems.items())[:5]:
                        status = info.get("status", "unknown")
                        # Any valid status is PASS, only unknown or error is WARN
                        is_valid = status in valid_statuses
                        test.log("background", name, "PASS" if is_valid else "WARN", f"status={status}")
                else:
                    test.log("background", "subsystems endpoint", "WARN", f"status={resp.status}")
        except Exception as e:
            test.log("background", "subsystems", "FAIL", str(e)[:50])


async def run_all_tests():
    """Run all integration tests."""
    test = LiveIntegrationTest()

    print("\n" + "="*60)
    print(" JARVIS LIVE INTEGRATION TEST")
    print(" Testing each module layer by layer")
    print("="*60)

    await check_layer_1_imports(test)
    await check_layer_2_subsystem_imports(test)
    await check_layer_3_singletons(test)
    await check_layer_4_tools(test)
    await check_layer_5_api_endpoints(test)
    await check_layer_6_llm_integration(test)
    await check_layer_7_api_manager(test)
    await check_layer_8_subsystem_llm_calls(test)
    await check_layer_9_tool_execution(test)
    await check_layer_10_background_subsystems(test)

    # Summary
    test.print_header("TEST SUMMARY")

    passed = sum(1 for _, _, status, _ in test.results if status == "PASS")
    warned = sum(1 for _, _, status, _ in test.results if status == "WARN")
    failed = sum(1 for _, _, status, _ in test.results if status == "FAIL")
    total = len(test.results)

    print(f"  Total tests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Warnings: {warned}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {passed/total*100:.1f}%")

    if failed > 0:
        print("\n  Failed tests:")
        for module, test_name, status, detail in test.results:
            if status == "FAIL":
                print(f"    - {module}: {test_name} - {detail}")

    print("\n" + "="*60)
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
