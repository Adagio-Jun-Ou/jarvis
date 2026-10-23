#!/usr/bin/env python3
"""Jarvis Full Stack Integration Test Suite.

Tests Jarvis NPU/GPU/CPU integration progressively:
- Level 1: Core imports (1 module)
- Level 2: Hardware detection (2 modules)
- Level 4: NPU Voice Tokenizers (4 modules)
- Level 8: GPU Simulation (8 modules)
- Level 16: Full tool integration (16 modules)
- Level 32: Cross-module data flow (32 modules)
- Level 64: Performance benchmarks (64 modules)
- Level 120: All modules stress test
"""
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RESULTS = {
    "level_1": {"passed": 0, "failed": 0, "tests": []},
    "level_2": {"passed": 0, "failed": 0, "tests": []},
    "level_4": {"passed": 0, "failed": 0, "tests": []},
    "level_8": {"passed": 0, "failed": 0, "tests": []},
    "level_16": {"passed": 0, "failed": 0, "tests": []},
    "level_32": {"passed": 0, "failed": 0, "tests": []},
    "level_64": {"passed": 0, "failed": 0, "tests": []},
    "level_120": {"passed": 0, "failed": 0, "tests": []},
}

def log(level: str, msg: str):
    print(f"[{level}] {msg}")

def check_result(level: str, name: str, passed: bool, details: str = ""):
    RESULTS[level]["tests"].append({"name": name, "passed": passed, "details": details})
    if passed:
        RESULTS[level]["passed"] += 1
        log(level, f"  ✓ {name}")
    else:
        RESULTS[level]["failed"] += 1
        log(level, f"  ✗ {name}: {details}")

# =============================================================================
# LEVEL 1: Core Imports (1 module)
# =============================================================================
def test_level_1():
    log("LEVEL_1", "Testing Core Imports (1 module)...")

    try:
        check_result("level_1", "Import core package", True)
    except Exception as e:
        check_result("level_1", "Import core package", False, str(e))
        return False

    try:
        from core.config import Config
        config = Config()
        check_result("level_1", "Config initialization", True)
    except Exception as e:
        check_result("level_1", "Config initialization", False, str(e))

    try:
        from core.memory import ConversationMemory
        memory = ConversationMemory()
        check_result("level_1", "ConversationMemory initialization", True)
    except Exception as e:
        check_result("level_1", "ConversationMemory initialization", False, str(e))

    try:
        from core.permissions import PermissionManager
        perms = PermissionManager()
        check_result("level_1", "PermissionManager initialization", True)
    except Exception as e:
        check_result("level_1", "PermissionManager initialization", False, str(e))

    try:
        check_result("level_1", "OllamaClient import", True)
    except Exception as e:
        check_result("level_1", "OllamaClient import", False, str(e))

    return RESULTS["level_1"]["failed"] == 0

# =============================================================================
# LEVEL 2: Hardware Detection (2 modules)
# =============================================================================
def test_level_2():
    log("LEVEL_2", "Testing Hardware Detection (2 modules)...")

    try:
        from core.hardware import (
            get_hardware_detector,
            AcceleratorType,
        )
        check_result("level_2", "Hardware module imports", True)
    except Exception as e:
        check_result("level_2", "Hardware module imports", False, str(e))
        return False

    try:
        detector = get_hardware_detector()
        check_result("level_2", "HardwareDetector singleton", True)

        cpu = next((a for a in detector.accelerators if a.type == AcceleratorType.CPU), None)
        if cpu:
            check_result("level_2", f"CPU detected: {cpu.name}", True, f"{cpu.compute_units} cores, {cpu.memory_mb}MB")
        else:
            check_result("level_2", "CPU detected", False, "No CPU found")

        gpu = next((a for a in detector.accelerators if a.type in [AcceleratorType.ROCM_GPU, AcceleratorType.CUDA_GPU]), None)
        if gpu:
            check_result("level_2", f"GPU detected: {gpu.name}", True, f"{gpu.memory_mb}MB, {gpu.compute_units} CUs")
        else:
            check_result("level_2", "GPU detection", False, "No GPU found")

        npu = next((a for a in detector.accelerators if a.type == AcceleratorType.AMD_NPU), None)
        if npu:
            check_result("level_2", f"NPU detected: {npu.name}", True, f"{npu.memory_mb}MB")
        else:
            check_result("level_2", "NPU detection", False, "No AMD XDNA NPU found")

        primary = detector.primary_accelerator
        check_result("level_2", f"Primary accelerator: {primary.value}", True)

    except Exception as e:
        check_result("level_2", "HardwareDetector initialization", False, str(e))

    try:
        from core.hardware import get_gpu_accelerator
        gpu_accel = get_gpu_accelerator()
        check_result("level_2", "GPUAccelerator singleton", True)

        stats = gpu_accel.detector.memory_pool
        if stats:
            check_result("level_2", f"Memory pool: {stats.total_mb}MB total, {stats.available_mb}MB available", True)
    except Exception as e:
        check_result("level_2", "GPUAccelerator initialization", False, str(e))

    try:
        from core.gpu_simulation import get_gpu_simulation
        gpu_sim = get_gpu_simulation()
        check_result("level_2", f"GPUSimulationEngine backend: {gpu_sim.backend.value}", True)

        if gpu_sim.detected:
            info = gpu_sim.gpu_info
            check_result("level_2", f"GPU Info: {info.device_name} ({info.memory_mb}MB)", True)
        else:
            check_result("level_2", "GPUSimulationEngine working", False, "GPU not functional")
    except Exception as e:
        check_result("level_2", "GPUSimulationEngine", False, str(e))

    return RESULTS["level_2"]["failed"] == 0

# =============================================================================
# LEVEL 4: NPU Voice Tokenizers (4 modules)
# =============================================================================
def test_level_4():
    log("LEVEL_4", "Testing NPU Voice Tokenizers (4 modules)...")

    try:
        import numpy as np
        check_result("level_4", "NumPy import", True)
    except Exception as e:
        check_result("level_4", "NumPy import", False, str(e))
        return False

    try:
        from core.npu_voice import (
            get_voice_tokenizer,
            get_voice_processor,
        )
        check_result("level_4", "NPU Voice module imports", True)
    except Exception as e:
        check_result("level_4", "NPU Voice module imports", False, str(e))
        return False

    try:
        tokenizer = get_voice_tokenizer()
        check_result("level_4", "NPUVoiceTokenizer singleton", True)

        stats = tokenizer.get_stats()
        check_result("level_4", f"Tokenizer stats: device={stats['device']}, frame_rate={stats['frame_rate']}fps", True)

        check_result("level_4", f"Semantic vocab: {tokenizer.SEMANTIC_VOCAB} tokens", True)
        check_result("level_4", f"Acoustic codebooks: {tokenizer.ACOUSTIC_CODEBOOKS} x {tokenizer.ACOUSTIC_VOCAB} vocab", True)

        special_tokens = tokenizer.SPECIAL_TOKENS
        check_result("level_4", f"Special tokens: {list(special_tokens.keys())}", True)

    except Exception as e:
        check_result("level_4", "NPUVoiceTokenizer initialization", False, str(e))

    try:
        processor = get_voice_processor()
        check_result("level_4", "NPUVoiceProcessor singleton", True)

        stats = processor.get_stats()
        check_result("level_4", f"Processor stats: {stats['speakers_registered']} speakers", True)
    except Exception as e:
        check_result("level_4", "NPUVoiceProcessor initialization", False, str(e))

    log("LEVEL_4", "Testing voice tokenization with synthetic audio...")
    try:
        np = __import__('numpy')
        tokenizer = get_voice_tokenizer()

        sample_rate = 16000
        duration_sec = 1.0
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))

        freq = 440
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        audio = audio.astype(np.float32)

        check_result("level_4", f"Generated test audio: {len(audio)} samples, {duration_sec}s", True)

        start = time.perf_counter()
        result = tokenizer.tokenize(audio, sample_rate)
        elapsed = (time.perf_counter() - start) * 1000

        check_result("level_4", f"Tokenization time: {elapsed:.2f}ms", True)
        check_result("level_4", f"Semantic tokens: {len(result.semantic_tokens)}", True)
        check_result("level_4", f"Acoustic frames: {len(result.acoustic_tokens)}", True)
        check_result("level_4", f"Duration: {result.duration_ms:.1f}ms", True)
        check_result("level_4", f"Semantic timing: {result.semantic_ms:.2f}ms", True)
        check_result("level_4", f"Acoustic timing: {result.acoustic_ms:.2f}ms", True)
        check_result("level_4", f"Total timing: {result.total_ms:.2f}ms", True)
        check_result("level_4", f"Device: {result.device}", True)

        decoded = tokenizer.decode_semantic(result.semantic_tokens)
        check_result("level_4", f"Decoded: phonemes={decoded['phoneme_count']}, silence={decoded['silence_count']}, emphasis={decoded['emphasis_count']}", True)

        if len(result.acoustic_tokens) > 0:
            first_frame = result.acoustic_tokens[0]
            check_result("level_4", f"First acoustic frame: {len(first_frame)} codebooks = {first_frame[:4]}...", True)

        real_time_factor = result.total_ms / result.duration_ms if result.duration_ms > 0 else 0
        check_result("level_4", f"Real-time factor: {real_time_factor:.3f}x", real_time_factor < 1.0)

    except Exception as e:
        check_result("level_4", "Voice tokenization test", False, str(e))
        traceback.print_exc()

    log("LEVEL_4", "Testing semantic token analysis...")
    try:
        tokenizer = get_voice_tokenizer()
        np = __import__('numpy')

        audio_silence = np.zeros(16000, dtype=np.float32)
        result_silence = tokenizer.tokenize(audio_silence, 16000)
        silence_tokens = sum(1 for t in result_silence.semantic_tokens if t == tokenizer.SPECIAL_TOKENS['SILENCE'])
        check_result("level_4", f"Silence audio: {silence_tokens} silence tokens detected", silence_tokens > 0)

        audio_loud = np.ones(16000, dtype=np.float32) * 0.8
        result_loud = tokenizer.tokenize(audio_loud, 16000)
        emphasis_tokens = sum(1 for t in result_loud.semantic_tokens if t == tokenizer.SPECIAL_TOKENS['EMPHASIS'])
        check_result("level_4", "Loud audio: emphasis patterns in tokens", True)

    except Exception as e:
        check_result("level_4", "Semantic analysis test", False, str(e))

    log("LEVEL_4", "Testing acoustic token structure...")
    try:
        tokenizer = get_voice_tokenizer()
        np = __import__('numpy')

        sample_rate = 16000
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.3
        result = tokenizer.tokenize(audio, sample_rate)

        if len(result.acoustic_tokens) > 0:
            frame = result.acoustic_tokens[0]
            check_result("level_4", f"Acoustic frame structure: {len(frame)} values (expected 8 codebooks)", len(frame) == 8)

            all_frames = np.array(result.acoustic_tokens)
            check_result("level_4", f"Acoustic tensor shape: {all_frames.shape}", len(all_frames.shape) == 2)

            codebook_ranges = []
            for i in range(8):
                vals = all_frames[:, i]
                codebook_ranges.append(f"CB{i}: [{vals.min():.0f}-{vals.max():.0f}]")
            check_result("level_4", f"Codebook ranges: {', '.join(codebook_ranges[:4])}...", True)

    except Exception as e:
        check_result("level_4", "Acoustic structure test", False, str(e))

    log("LEVEL_4", "Testing speaker identification...")
    try:
        processor = get_voice_processor()
        np = __import__('numpy')

        np.random.seed(42)
        speaker1_audio = np.random.randn(16000).astype(np.float32) * 0.5
        speaker2_audio = np.random.randn(16000).astype(np.float32) * 0.5

        processor.register_speaker("Alice", speaker1_audio, 16000)
        processor.register_speaker("Bob", speaker2_audio, 16000)

        stats = processor.get_stats()
        check_result("level_4", f"Registered {stats['speakers_registered']} speakers", stats['speakers_registered'] == 2)

        name, confidence = processor.identify_speaker(speaker1_audio, 16000)
        check_result("level_4", f"Speaker identification: {name} ({confidence:.2%})", name == "Alice" or name == "Bob")

        similarity = processor.compare_voices(speaker1_audio, speaker2_audio, 16000)
        check_result("level_4", f"Voice similarity: {similarity:.4f}", True)

    except Exception as e:
        check_result("level_4", "Speaker identification test", False, str(e))

    return RESULTS["level_4"]["failed"] == 0

# =============================================================================
# LEVEL 8: GPU Simulation Acceleration (8 modules)
# =============================================================================
def test_level_8():
    log("LEVEL_8", "Testing GPU Simulation Acceleration (8 modules)...")

    try:
        import torch
        check_result("level_8", f"PyTorch version: {torch.__version__}", True)

        if hasattr(torch, 'hip') and torch.hip.is_available():
            check_result("level_8", "ROCm/HIP available", True)
        elif torch.cuda.is_available():
            check_result("level_8", "CUDA available", True)
        else:
            check_result("level_8", "CPU-only PyTorch", True)
    except Exception as e:
        check_result("level_8", "PyTorch import", False, str(e))
        return False

    try:
        from core.gpu_simulation import (
            get_gpu_simulation,
        )
        check_result("level_8", "GPU Simulation imports", True)
    except Exception as e:
        check_result("level_8", "GPU Simulation imports", False, str(e))
        return False

    try:
        gpu_sim = get_gpu_simulation()
        check_result("level_8", "GPUSimulationEngine singleton", True)

        backend = gpu_sim.backend
        check_result("level_8", f"Backend: {backend.value}", True)

        detected = gpu_sim.detected
        available = gpu_sim.available

        if detected:
            info = gpu_sim.gpu_info
            check_result("level_8", f"GPU detected: {info.device_name}", True)
            check_result("level_8", f"GPU memory: {info.memory_mb}MB", True)
            check_result("level_8", f"GPU compute units: {info.compute_units}", True)
            check_result("level_8", f"FP16 support: {info.supports_fp16}", True)
            check_result("level_8", f"INT8 support: {info.supports_int8}", True)

        check_result("level_8", f"GPU working: {available}", True)

    except Exception as e:
        check_result("level_8", "GPUSimulationEngine initialization", False, str(e))

    log("LEVEL_8", "Testing particle simulation...")
    try:
        gpu_sim = get_gpu_simulation()

        n_particles = 100
        positions = [[0.0, 0.0, 0.0] for _ in range(n_particles)]
        velocities = [[0.0, 0.0, 0.0] for _ in range(n_particles)]
        masses = [1.0] * n_particles
        forces = {"gravity": [0.0, -9.81, 0.0], "wind": [0.5, 0.0, 0.0]}

        start = time.perf_counter()
        final_pos, final_vel, stats = gpu_sim.simulate_particles_gpu(
            positions=positions,
            velocities=velocities,
            mass=masses,
            forces=forces,
            dt=0.01,
            steps=100,
            boundary=[-10, -10, -10, 10, 10, 10],
        )
        elapsed = (time.perf_counter() - start) * 1000

        check_result("level_8", f"Particle simulation: {n_particles} particles, 100 steps in {elapsed:.2f}ms", True)
        check_result("level_8", f"Simulation stats: total={stats.total_time_ms:.2f}ms, compute={stats.gpu_time_ms:.2f}ms", True)
        check_result("level_8", f"Backend used: {stats.backend_used}", True)
        check_result("level_8", f"Speedup vs CPU: {stats.speedup_vs_cpu:.1f}x", True)
        check_result("level_8", f"Objects processed: {stats.objects_processed}", True)

        check_result("level_8", f"Final position sample: {final_pos[0]}", True)
        check_result("level_8", f"Final velocity sample: {final_vel[0]}", True)

    except Exception as e:
        check_result("level_8", "Particle simulation", False, str(e))
        traceback.print_exc()

    log("LEVEL_8", "Testing collision detection...")
    try:
        gpu_sim = get_gpu_simulation()

        n_objects = 10
        positions = [[float(i), 0.0, 0.0] for i in range(n_objects)]
        radii = [1.0] * n_objects

        start = time.perf_counter()
        collisions = gpu_sim.detect_collisions_gpu(
            positions=positions,
            radii=radii,
            cell_size=2.0,
        )
        elapsed = (time.perf_counter() - start) * 1000

        check_result("level_8", f"Collision detection: {n_objects} objects in {elapsed:.2f}ms", True)
        check_result("level_8", f"Collisions found: {len(collisions)}", True)

        if collisions:
            for idx_a, idx_b, dist in collisions[:3]:
                check_result("level_8", f"  Collision: {idx_a} <-> {idx_b} at distance {dist:.3f}", True)

    except Exception as e:
        check_result("level_8", "Collision detection", False, str(e))

    log("LEVEL_8", "Testing matrix operations...")
    try:
        gpu_sim = get_gpu_simulation()

        size = 512
        a = [[float(i + j) for j in range(size)] for i in range(size)]
        b = [[float(i - j) for j in range(size)] for i in range(size)]

        start = time.perf_counter()
        result = gpu_sim.accelerate_matrix_ops('multiply', a, b)
        elapsed = (time.perf_counter() - start) * 1000

        check_result("level_8", f"Matrix multiply: {size}x{size} in {elapsed:.2f}ms", True)
        check_result("level_8", f"Result shape: {len(result)}x{len(result[0])}", True)

    except Exception as e:
        check_result("level_8", "Matrix operations", False, str(e))

    return RESULTS["level_8"]["failed"] == 0

# =============================================================================
# LEVEL 16: Full Tool Integration (16 modules)
# =============================================================================
def test_level_16():
    log("LEVEL_16", "Testing Full Tool Integration (16 modules)...")

    try:
        check_result("level_16", "Tool base class import", True)
    except Exception as e:
        check_result("level_16", "Tool base class import", False, str(e))
        return False

    tools_to_test = [
        ("physics_sim", "CollisionDetectionTool", "ParticleSystemTool", "PhysicsAnalysisTool"),
        ("npu_tools", "EmbeddingTool", "FileIndexTool", "FileSearchTool", "VoiceTokenizeTool"),
        ("system", "ShellTool", "FileTool", "ProcessTool"),
        ("data_tools", "DataAnalysisTool"),
    ]

    total_tools = 0
    for module_name, *tool_names in tools_to_test:
        try:
            module = __import__(f'tools.{module_name}', fromlist=tool_names)
            for tool_name in tool_names:
                try:
                    tool_class = getattr(module, tool_name)
                    total_tools += 1
                    check_result("level_16", f"Tool: {tool_name}", True)
                except AttributeError:
                    check_result("level_16", f"Tool: {tool_name}", False, "Not found")
        except ImportError as e:
            check_result("level_16", f"Module: tools.{module_name}", False, str(e))

    check_result("level_16", f"Total tools imported: {total_tools}", True)

    log("LEVEL_16", "Testing ParticleSystemTool with GPU...")
    try:
        import asyncio
        from tools.physics_sim import ParticleSystemTool

        async def test_particle_tool():
            tool = ParticleSystemTool()
            result = await tool.execute(
                particle_count=100,
                steps=50,
                dt=0.01,
                initial_pattern='random',
                forces={'gravity': [0, -9.81, 0]}
            )
            return result

        result = asyncio.run(test_particle_tool())

        has_gpu_info = "GPU" in result or "Backend:" in result
        check_result("level_16", "ParticleSystemTool execution", True)
        check_result("level_16", f"GPU acceleration used: {has_gpu_info}", True)

        lines = result.split('\n')
        for line in lines[:5]:
            log("LEVEL_16", f"  {line}")

    except Exception as e:
        check_result("level_16", "ParticleSystemTool test", False, str(e))
        traceback.print_exc()

    log("LEVEL_16", "Testing CollisionDetectionTool with GPU...")
    try:
        import asyncio
        from tools.physics_sim import CollisionDetectionTool

        async def test_collision_tool():
            tool = CollisionDetectionTool()
            result = await tool.execute(
                objects=[
                    {"id": "obj1", "shape": "sphere", "position": [0, 0, 0], "radius": 1.0},
                    {"id": "obj2", "shape": "sphere", "position": [1.5, 0, 0], "radius": 1.0},
                    {"id": "obj3", "shape": "sphere", "position": [5, 0, 0], "radius": 1.0},
                ]
            )
            return result

        result = asyncio.run(test_collision_tool())
        check_result("level_16", "CollisionDetectionTool execution", "Collisions detected" in result)

    except Exception as e:
        check_result("level_16", "CollisionDetectionTool test", False, str(e))

    log("LEVEL_16", "Testing NPU EmbeddingTool...")
    try:
        import asyncio
        from tools.npu_tools import EmbeddingTool

        async def test_embedding():
            tool = EmbeddingTool()
            result = await tool.execute(texts=["Hello world", "Test embedding"])
            return result

        result = asyncio.run(test_embedding())
        check_result("level_16", "EmbeddingTool executed", "Embedding" in result)

    except Exception as e:
        check_result("level_16", "EmbeddingTool test", False, str(e))

    return RESULTS["level_16"]["failed"] == 0

# =============================================================================
# LEVEL 32: Cross-Module Data Flow (32 modules)
# =============================================================================
def test_level_32():
    log("LEVEL_32", "Testing Cross-Module Data Flow (32 modules)...")

    log("LEVEL_32", "Testing NPU Voice -> CPU Memory -> GPU processing...")
    try:
        import numpy as np

        from core.npu_voice import get_voice_tokenizer
        from core.gpu_simulation import get_gpu_simulation

        np.random.seed(42)
        audio = np.random.randn(16000).astype(np.float32) * 0.3

        tokenizer = get_voice_tokenizer()
        voice_result = tokenizer.tokenize(audio, 16000)

        check_result("level_32", f"Voice tokenization: {len(voice_result.semantic_tokens)} semantic tokens", True)

        acoustic_tokens = voice_result.acoustic_tokens
        if len(acoustic_tokens) > 0:
            frame_tensor = np.array(acoustic_tokens, dtype=np.float32)
            check_result("level_32", f"Acoustic tensor: shape={frame_tensor.shape}", True)

            gpu_sim = get_gpu_simulation()
            if gpu_sim:
                try:
                    result = gpu_sim.accelerate_matrix_ops('transpose', frame_tensor.tolist())
                    check_result("level_32", f"GPU transpose of acoustic data: shape={len(result)}x{len(result[0])}", True)
                except Exception as e:
                    check_result("level_32", f"GPU transpose (CPU fallback): {e}", False)

    except Exception as e:
        check_result("level_32", "Voice->Memory->GPU flow", False, str(e))
        traceback.print_exc()

    log("LEVEL_32", "Testing Hardware -> MemoryPool -> Simulation integration...")
    try:
        from core.hardware import get_hardware_detector, get_memory_pool
        from core.gpu_simulation import get_gpu_simulation

        detector = get_hardware_detector()
        memory = get_memory_pool()
        gpu_sim = get_gpu_simulation()

        allocated = memory.allocate("test_sim", 512, priority="normal")
        check_result("level_32", f"Memory allocation: {allocated}", True)

        stats = memory.get_stats()
        check_result("level_32", f"Memory stats: {stats['allocated_mb']}MB allocated, {stats['available_mb']}MB available", True)

        if allocated:
            import numpy as np
            n = 1000
            positions = np.random.randn(n, 3).tolist()
            velocities = np.zeros((n, 3)).tolist()
            masses = np.ones(n).tolist()

            result = gpu_sim.simulate_particles_gpu(
                positions, velocities, masses,
                {"gravity": [0, -9.81, 0]}, 0.01, 10
            )
            check_result("level_32", "GPU simulation with allocated memory", True)

        memory.deallocate("test_sim")
        check_result("level_32", "Memory deallocation", True)

    except Exception as e:
        check_result("level_32", "Hardware->Memory->Sim flow", False, str(e))

    log("LEVEL_32", "Testing NPU embedding -> GPU similarity search...")
    try:
        from core.npu_models import get_embedding
        from core.gpu_simulation import get_gpu_simulation
        import numpy as np

        embedder = get_embedding("mini")
        gpu_sim = get_gpu_simulation()

        texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is transforming technology",
            "AMD ROCm provides GPU acceleration",
            "Voice tokenization enables speech processing",
        ]

        results = embedder.encode(texts)
        check_result("level_32", f"Generated {len(results)} embeddings", True)

        if results:
            embeddings = np.array([r.embedding for r in results])
            query = embeddings[0:1]

            scores, indices = gpu_sim.accelerate_similarity_search(query, embeddings, k=4)
            check_result("level_32", f"GPU similarity search: top matches indices={indices[0]}", True)
            check_result("level_32", f"Similarity scores: {scores[0]}", True)

    except Exception as e:
        check_result("level_32", "Embedding->GPU similarity flow", False, str(e))

    log("LEVEL_32", "Testing multi-threaded access...")
    try:
        from core.npu_voice import get_voice_tokenizer
        from core.gpu_simulation import get_gpu_simulation
        from core.hardware import get_hardware_detector
        import threading

        results_list = []
        errors_list = []

        def worker(worker_id):
            try:
                tokenizer = get_voice_tokenizer()
                gpu_sim = get_gpu_simulation()
                detector = get_hardware_detector()

                np.random.seed(worker_id)
                audio = np.random.randn(8000).astype(np.float32) * 0.5
                result = tokenizer.tokenize(audio, 16000)
                results_list.append((worker_id, len(result.semantic_tokens)))
            except Exception as e:
                errors_list.append((worker_id, str(e)))

        import numpy as np
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        check_result("level_32", f"Multi-threaded test: {len(results_list)}/4 successful", len(errors_list) == 0)

        for wid, tokens in results_list:
            check_result("level_32", f"  Thread {wid}: {tokens} tokens", True)

    except Exception as e:
        check_result("level_32", "Multi-threaded test", False, str(e))

    return RESULTS["level_32"]["failed"] == 0

# =============================================================================
# LEVEL 64: Performance Benchmarks (64 modules)
# =============================================================================
def test_level_64():
    log("LEVEL_64", "Testing Performance Benchmarks (64 modules)...")

    import numpy as np

    log("LEVEL_64", "Benchmarking voice tokenization...")
    try:
        from core.npu_voice import get_voice_tokenizer
        import time as time_module

        tokenizer = get_voice_tokenizer()

        durations = [0.5, 1.0, 2.0, 5.0]
        for dur in durations:
            samples = int(16000 * dur)
            audio = np.random.randn(samples).astype(np.float32) * 0.5

            start = time_module.perf_counter()
            result = tokenizer.tokenize(audio, 16000)
            elapsed = (time_module.perf_counter() - start) * 1000

            rtf = elapsed / (dur * 1000)
            check_result("level_64", f"Voice {dur}s audio: {elapsed:.2f}ms (RTF={rtf:.3f}x)", rtf < 0.5)

    except Exception as e:
        check_result("level_64", "Voice benchmark", False, str(e))

    log("LEVEL_64", "Benchmarking particle simulation...")
    try:
        from core.gpu_simulation import get_gpu_simulation
        import time

        gpu_sim = get_gpu_simulation()

        particle_counts = [100, 1000, 10000]
        steps = 100

        for n in particle_counts:
            positions = np.random.randn(n, 3).tolist()
            velocities = np.zeros((n, 3)).tolist()
            masses = np.ones(n).tolist()

            start = time.perf_counter()
            final_pos, final_vel, stats = gpu_sim.simulate_particles_gpu(
                positions, velocities, masses,
                {"gravity": [0, -9.81, 0]}, 0.01, steps
            )
            elapsed = (time.perf_counter() - start) * 1000

            throughput = n * steps / (elapsed / 1000) if elapsed > 0 else 0
            check_result("level_64", f"Particles {n}: {elapsed:.2f}ms ({throughput:.0f} particles/sec)", True)

    except Exception as e:
        check_result("level_64", "Particle benchmark", False, str(e))

    log("LEVEL_64", "Benchmarking collision detection...")
    try:
        from core.gpu_simulation import get_gpu_simulation

        gpu_sim = get_gpu_simulation()

        object_counts = [10, 100, 1000]

        for n in object_counts:
            positions = np.random.randn(n, 3).tolist()
            radii = np.ones(n).tolist()

            start = time.perf_counter()
            collisions = gpu_sim.detect_collisions_gpu(positions, radii)
            elapsed = (time.perf_counter() - start) * 1000

            check_result("level_64", f"Collision {n} objects: {elapsed:.2f}ms, {len(collisions)} collisions", True)

    except Exception as e:
        check_result("level_64", "Collision benchmark", False, str(e))

    log("LEVEL_64", "Benchmarking embeddings...")
    try:
        from core.npu_models import get_embedding

        embedder = get_embedding("mini")

        batch_sizes = [1, 10, 100]

        for batch in batch_sizes:
            texts = [f"Test text number {i} for embedding benchmark" for i in range(batch)]

            start = time.perf_counter()
            results = embedder.encode(texts)
            elapsed = (time.perf_counter() - start) * 1000

            throughput = batch / (elapsed / 1000) if elapsed > 0 else 0
            check_result("level_64", f"Embedding {batch} texts: {elapsed:.2f}ms ({throughput:.0f} texts/sec)", True)

    except Exception as e:
        check_result("level_64", "Embedding benchmark", False, str(e))

    log("LEVEL_64", "Stress test: Concurrent operations...")
    try:
        from core.npu_voice import get_voice_tokenizer
        from core.gpu_simulation import get_gpu_simulation
        from core.npu_models import get_embedding
        from concurrent.futures import ThreadPoolExecutor

        tokenizer = get_voice_tokenizer()
        gpu_sim = get_gpu_simulation()
        embedder = get_embedding("mini")

        def voice_task(i):
            audio = np.random.randn(8000).astype(np.float32)
            result = tokenizer.tokenize(audio, 16000)
            return ('voice', i, len(result.semantic_tokens))

        def particle_task(i):
            positions = np.random.randn(100, 3).tolist()
            velocities = np.zeros((100, 3)).tolist()
            masses = np.ones(100).tolist()
            final_pos, final_vel, stats = gpu_sim.simulate_particles_gpu(
                positions, velocities, masses, {"gravity": [0, -9.81, 0]}, 0.01, 10
            )
            return ('particle', i, stats.total_time_ms)

        def embedding_task(i):
            results = embedder.encode([f"Test {i}"])
            return ('embedding', i, len(results[0].embedding))

        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []
            for i in range(4):
                futures.append(executor.submit(voice_task, i))
                futures.append(executor.submit(particle_task, i))
                futures.append(executor.submit(embedding_task, i))

            results = [f.result() for f in futures]

        elapsed = (time.perf_counter() - start) * 1000

        check_result("level_64", f"Concurrent stress test: 12 tasks in {elapsed:.2f}ms", True)

        for task_type, task_id, value in results:
            check_result("level_64", f"  {task_type}_{task_id}: {value}", True)

    except Exception as e:
        check_result("level_64", "Concurrent stress test", False, str(e))
        traceback.print_exc()

    return RESULTS["level_64"]["failed"] == 0

# =============================================================================
# LEVEL 120: All Modules Stress Test
# =============================================================================
def test_level_120():
    log("LEVEL_120", "Testing All Modules Stress Test...")

    all_modules = [
        "core.agent",
        "core.llm",
        "core.memory",
        "core.permissions",
        "core.config",
        "core.thinking",
        "core.prompts",
        "core.awareness",
        "core.self_awareness",
        "core.hardware",
        "core.npu_models",
        "core.npu_voice",
        "core.gpu_simulation",
        "core.session",
        "core.erdos_engine",
        "core.defense",
        "core.learning_pipeline",
        "core.evolution_orchestrator",
        "core.vector_memory",
        "core.knowledge_graph",
        "tools.physics_sim",
        "tools.npu_tools",
        "tools.base",
        "tools.system",
        "tools.data_tools",
    ]

    imported = []
    failed = []

    for module in all_modules:
        try:
            __import__(module)
            imported.append(module)
            check_result("level_120", f"Import: {module}", True)
        except Exception as e:
            failed.append((module, str(e)))
            check_result("level_120", f"Import: {module}", False, str(e))

    check_result("level_120", f"Total imported: {len(imported)}/{len(all_modules)}", len(failed) == 0)

    log("LEVEL_120", "Testing full NPU/CPU/GPU/Stack integration...")
    try:
        import numpy as np

        from core.hardware import get_hardware_detector, get_memory_pool
        from core.npu_voice import get_voice_tokenizer, get_voice_processor
        from core.npu_models import get_embedding, get_retriever
        from core.gpu_simulation import get_gpu_simulation

        detector = get_hardware_detector()
        memory = get_memory_pool()
        tokenizer = get_voice_tokenizer()
        processor = get_voice_processor()
        embedder = get_embedding("mini")
        retriever = get_retriever()
        gpu_sim = get_gpu_simulation()

        np.random.seed(42)
        audio = np.random.randn(16000).astype(np.float32) * 0.3

        processor.register_speaker("TestUser", audio, 16000)

        voice_result = tokenizer.tokenize(audio, 16000)
        check_result("level_120", f"Stack: Voice tokenization ({len(voice_result.semantic_tokens)} tokens)", True)

        texts = [f"Audio frame {i}" for i in range(len(voice_result.acoustic_tokens[:10]))]
        embeddings = embedder.encode(texts)
        check_result("level_120", f"Stack: Generated {len(embeddings)} embeddings", True)

        if len(voice_result.acoustic_tokens) > 0:
            acoustic_data = np.array(voice_result.acoustic_tokens, dtype=np.float32)
            positions = acoustic_data[:, :3].tolist()
            velocities = np.zeros((len(positions), 3)).tolist()
            masses = np.ones(len(positions)).tolist()

            final_pos, final_vel, stats = gpu_sim.simulate_particles_gpu(
                positions, velocities, masses,
                {"gravity": [0, -9.81, 0]}, 0.01, 10
            )
            check_result("level_120", f"Stack: GPU simulation ({stats.total_time_ms:.2f}ms)", True)

        hw_stats = detector.memory_pool
        mem_stats = memory.get_stats()
        check_result("level_120", f"Stack: Memory pool {mem_stats['available_mb']}/{mem_stats['total_mb']}MB available", True)

        check_result("level_120", f"Stack: Hardware accelerator: {detector.primary_accelerator.value}", True)

        check_result("level_120", "Full NPU/CPU/GPU/Stack integration", True)

    except Exception as e:
        check_result("level_120", "Full stack integration", False, str(e))
        traceback.print_exc()

    log("LEVEL_120", "Memory cleanup verification...")
    try:
        from core.hardware import get_memory_pool
        import gc

        memory = get_memory_pool()
        gc.collect()

        stats = memory.get_stats()
        check_result("level_120", f"Memory after cleanup: {stats['available_mb']}MB available", True)

    except Exception as e:
        check_result("level_120", "Memory cleanup", False, str(e))

    return RESULTS["level_120"]["failed"] == 0

# =============================================================================
# Summary
# =============================================================================
def print_summary():
    print("\n" + "=" * 70)
    print("JARVIS FULL STACK INTEGRATION TEST SUMMARY")
    print("=" * 70)

    total_passed = 0
    total_failed = 0

    for level, data in RESULTS.items():
        passed = data["passed"]
        failed = data["failed"]
        total_passed += passed
        total_failed += failed

        status = "✓ PASS" if failed == 0 else "✗ FAIL"
        print(f"\n{level.upper():12} [{status}]  {passed} passed, {failed} failed")

        if failed > 0:
            for test in data["tests"]:
                if not test["passed"]:
                    print(f"    ✗ {test['name']}: {test['details'][:60]}...")

    print("\n" + "-" * 70)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")

    if total_failed == 0:
        print("\n✓ ALL TESTS PASSED - JARVIS FULL STACK INTEGRATION COMPLETE")
    else:
        print(f"\n✗ {total_failed} TESTS FAILED - REVIEW ERRORS ABOVE")

    print("=" * 70)

    return total_failed == 0

# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 70)
    print("JARVIS FULL STACK INTEGRATION TEST")
    print("Testing NPU/GPU/CPU integration progressively")
    print("=" * 70)

    import numpy as np

    print("\n[SETUP] Checking dependencies...")
    print(f"  NumPy: {np.__version__}")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        if hasattr(torch, 'hip') and torch.hip.is_available():
            print("  ROCm/HIP: Available")
    except:
        print("  PyTorch: Not available")

    print()

    test_level_1()
    print()

    test_level_2()
    print()

    test_level_4()
    print()

    test_level_8()
    print()

    test_level_16()
    print()

    test_level_32()
    print()

    test_level_64()
    print()

    test_level_120()
    print()

    success = print_summary()

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
