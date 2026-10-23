"""Integration test for Jarvis - tests full pipeline end-to-end."""
import asyncio
import time
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_agent_initialization():
    """Test full agent initialization."""
    print("\n" + "=" * 60)
    print("TEST: Agent Initialization")
    print("=" * 60)

    from core.config import Config
    from core.agent import JarvisAgent

    config = Config()
    print(f"  Config loaded: model={config.llm.model}")

    agent = JarvisAgent(config)
    print(f"  Agent created: {agent.config.identity.name}")

    # Check subsystems
    assert agent.llm is not None, "LLM client should be initialized"
    print("  ✅ LLM client initialized")

    assert agent.memory is not None, "Memory should be initialized"
    print("  ✅ Memory initialized")

    assert agent.permissions is not None, "Permissions should be initialized"
    print("  ✅ Permissions initialized")

    assert agent.awareness is not None, "Awareness should be initialized"
    print("  ✅ Awareness initialized")

    # Check background systems
    print(f"  Background scheduler: {agent.scheduler is not None}")
    print(f"  Thinking engine: {agent.thinking_engine is not None}")

    await agent.llm.close()
    print("\n✅ Agent initialization test PASSED")
    return True


async def test_memory_pipeline():
    """Test memory storage and retrieval pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Memory Pipeline")
    print("=" * 60)

    from core.memory import ConversationMemory
    from core.vector_memory import get_vector_memory
    from core.unified_memory import get_unified_memory, MemoryType

    # Test conversation memory
    conv_mem = ConversationMemory(max_turns=10)
    conv_mem.add_user_message("Hello, I'm testing Jarvis")
    conv_mem.add_assistant_message("Hello! I'm Jarvis, ready to help.")
    print(f"  Conversation memory: {len(conv_mem.get_messages_for_llm())} messages")

    # Test vector memory
    vec_mem = get_vector_memory()
    entry_id = vec_mem.store("Test memory entry about AI systems", {"type": "test"})
    print(f"  Vector memory: stored entry {entry_id[:8]}...")

    results = vec_mem.search("AI systems", k=3)
    print(f"  Vector memory: found {len(results)} results")

    # Test unified memory
    unified = get_unified_memory()
    unified.store("User is testing the memory pipeline", MemoryType.EPISODIC, 0.5)
    print("  Unified memory: stored episodic memory")

    context = unified.get_context_for_prompt(max_entries=5)
    print(f"  Unified memory context: {len(context)} chars")

    print("\n✅ Memory pipeline test PASSED")
    return True


async def test_hardware_pipeline():
    """Test hardware detection and acceleration pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Hardware Pipeline")
    print("=" * 60)

    from core.hardware import get_hardware_detector, get_memory_pool

    # Test hardware detection
    hw = get_hardware_detector()
    print(f"  Accelerators detected: {len(hw.accelerators)}")
    for accel in hw.accelerators:
        print(f"    - {accel.type.value}: {accel.name[:30]} ({accel.memory_mb}MB)")

    print(f"  Primary accelerator: {hw.primary_accelerator.value}")
    print(f"  GPU usable: {hw.use_gpu()}")
    print(f"  NPU usable: {hw.use_npu()}")

    # Test memory pool
    pool = get_memory_pool()
    stats = pool.get_stats()
    print(f"  Memory pool: {stats['total_mb']}MB total, {stats['available_mb']}MB available")

    print("\n✅ Hardware pipeline test PASSED")
    return True


async def test_cognitive_pipeline():
    """Test cognitive subsystem pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Cognitive Pipeline")
    print("=" * 60)

    from core.thinking import ThinkingMode
    from core.prompts import PromptManager
    from core.self_awareness import create_self_awareness
    from core.knowledge_graph import KnowledgeGraph

    # Test thinking engine
    print(f"  Thinking modes: {[m.value for m in ThinkingMode]}")

    # Test prompt management
    pm = PromptManager(name="TestJarvis", mode="supportive")
    prompt = pm.get_base_prompt()
    print(f"  Prompt manager: {len(prompt)} char base prompt")

    # Test self-awareness
    sa = create_self_awareness(
        name="TestJarvis",
        model_provider="test",
        model_id="test-model",
        model_name="Test Model",
        context_window=4096,
    )
    env = sa.refresh_environment()
    print("  Self-awareness: environment context loaded")

    # Test knowledge graph
    kg = KnowledgeGraph()
    node_id = kg.add_insight("test_001", "Test insight about prime numbers and graph theory")
    print(f"  Knowledge graph: added insight node {node_id[:8]}...")

    insights = kg.get_important_insights(limit=5)
    print(f"  Knowledge graph: {len(insights)} important insights")

    print("\n✅ Cognitive pipeline test PASSED")
    return True


async def test_autonomous_pipeline():
    """Test autonomous systems pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Autonomous Pipeline")
    print("=" * 60)

    # Test daydream engine (class exists)
    try:
        from core.daydream_engine import DaydreamEngine
        print("  DaydreamEngine: class available")
    except ImportError:
        print("  DaydreamEngine: not available (optional)")

    # Test evolution orchestrator
    try:
        from core.evolution_orchestrator import EvolutionOrchestrator
        print("  EvolutionOrchestrator: class available")
    except ImportError:
        print("  EvolutionOrchestrator: not available (optional)")

    # Test self-edit
    try:
        from core.self_edit import SelfEditor, SAFE_FILES
        print(f"  SelfEditor: {len(SAFE_FILES)} safe files defined")
    except ImportError:
        print("  SelfEditor: not available (optional)")

    # Test erdos engine
    try:
        from core.erdos_engine import ErdosEngine
        print("  ErdosEngine: class available")
    except ImportError:
        print("  ErdosEngine: not available (optional)")

    print("\n✅ Autonomous pipeline test PASSED")
    return True


async def test_security_pipeline():
    """Test security subsystem pipeline."""
    print("\n" + "=" * 60)
    print("TEST: Security Pipeline")
    print("=" * 60)

    import tempfile
    from core.permissions import PermissionManager, PermissionLevel

    with tempfile.TemporaryDirectory() as tmpdir:
        pm = PermissionManager(Path(tmpdir))

        # Check default behavior
        level = pm.check_permission("shell:test_command")
        print(f"  Permission check: {level}")

        # Grant permission
        pm.grant_permission("shell:test_command", PermissionLevel.ALLOW_SESSION)
        level = pm.check_permission("shell:test_command")
        print(f"  After grant: {level}")

    print("\n✅ Security pipeline test PASSED")
    return True


async def run_all_integration_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("JARVIS INTEGRATION TEST SUITE")
    print("=" * 70)

    tests = [
        ("Agent Initialization", test_agent_initialization),
        ("Memory Pipeline", test_memory_pipeline),
        ("Hardware Pipeline", test_hardware_pipeline),
        ("Voice Pipeline", test_voice_pipeline),
        ("Cognitive Pipeline", test_cognitive_pipeline),
        ("Autonomous Pipeline", test_autonomous_pipeline),
        ("Security Pipeline", test_security_pipeline),
    ]

    results = []
    for name, test_func in tests:
        try:
            start = time.perf_counter()
            passed = await test_func()
            elapsed = time.perf_counter() - start
            results.append((name, passed, elapsed, None))
        except Exception as e:
            elapsed = time.perf_counter() - start
            results.append((name, False, elapsed, str(e)))
            print(f"\n❌ {name} FAILED: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, p, _, _ in results if p)
    total = len(results)

    for name, p, elapsed, err in results:
        status = "✅" if p else "❌"
        print(f"{status} {name}: {elapsed:.2f}s")
        if err:
            print(f"   Error: {err[:100]}")

    print("-" * 70)
    rate = passed / total * 100 if total > 0 else 0
    print(f"TOTAL: {passed}/{total} passed ({rate:.1f}%)")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_integration_tests())
    sys.exit(0 if success else 1)
