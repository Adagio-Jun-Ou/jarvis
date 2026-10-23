"""Hardware acceleration layer for Jarvis.

Optimized for AMD Ryzen AI 9 HX 370:
- Radeon 890M (RDNA 3.5) GPU via ROCm/Vulkan
- XDNA 2 NPU for AI acceleration
- 32GB unified RAM (minimal VRAM allocation for iGPU)
- Unified memory architecture (CPU/GPU share system RAM)

Provides:
- GPU/NPU detection and initialization
- Unified memory pool management
- Accelerated operations (matrix multiply, vector search, etc.)
- Automatic fallback to CPU when GPU unavailable
"""
import os
import time
import threading
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, List, Dict, Tuple
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

HAS_NUMPY = False
HAS_TORCH = False
HAS_NUMBA = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None

try:
    import numba
    from numba import jit, cuda, vectorize
    HAS_NUMBA = True
except ImportError:
    numba = None
    jit = None
    cuda = None


class AcceleratorType(Enum):
    CPU = "cpu"
    ROCM_GPU = "rocm_gpu"
    CUDA_GPU = "cuda_gpu"
    VULKAN_GPU = "vulkan_gpu"
    AMD_NPU = "amd_npu"
    APPLE_METAL = "apple_metal"


UNSUPPORTED_GFX_ARCHS = {"gfx1150", "gfx1200", "gfx1201"}


@dataclass
class AcceleratorInfo:
    type: AcceleratorType
    name: str
    memory_mb: int
    compute_units: int
    supports_fp16: bool
    supports_int8: bool
    supports_bf16: bool
    vendor: str
    device_id: Optional[int] = None
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryPool:
    total_mb: int
    available_mb: int
    gpu_allocated_mb: int
    cpu_allocated_mb: int
    unified: bool

    def can_allocate(self, size_mb: int) -> bool:
        return self.available_mb >= size_mb


class HardwareDetector:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.accelerators: List[AcceleratorInfo] = []
        self.primary_accelerator: Optional[AcceleratorType] = None
        self.memory_pool: Optional[MemoryPool] = None
        self._torch_device = None
        self._numba_cuda_available = False

        self._detect_hardware()
        self._setup_memory_pool()
        self._log_detection_results()

    def _detect_hardware(self):
        self.accelerators.append(self._detect_cpu())

        rocm_accel = self._detect_rocm()
        if rocm_accel:
            self.accelerators.append(rocm_accel)

        cuda_accel = self._detect_cuda()
        if cuda_accel:
            self.accelerators.append(cuda_accel)

        vulkan_accel = self._detect_vulkan()
        if vulkan_accel:
            self.accelerators.append(vulkan_accel)

        npu_accel = self._detect_amd_npu()
        if npu_accel:
            self.accelerators.append(npu_accel)

        metal_accel = self._detect_apple_metal()
        if metal_accel:
            self.accelerators.append(metal_accel)

        self._select_primary_accelerator()

    def _detect_cpu(self) -> AcceleratorInfo:
        cpu_count = os.cpu_count() or 8
        try:
            with open('/proc/meminfo', 'r', encoding='utf-8') as f:
                meminfo = f.read()
                total_kb = 0
                for line in meminfo.split('\n'):
                    if line.startswith('MemTotal:'):
                        total_kb = int(line.split()[1])
                        break
                total_mb = total_kb // 1024
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            logger.debug("Could not read /proc/meminfo, using default memory")
            total_mb = 32768

        return AcceleratorInfo(
            type=AcceleratorType.CPU,
            name=platform.processor() or "Unknown CPU",
            memory_mb=total_mb,
            compute_units=cpu_count,
            supports_fp16=True,
            supports_int8=True,
            supports_bf16=True,
            vendor=platform.machine(),
        )

    def _detect_rocm(self) -> Optional[AcceleratorInfo]:
        if not HAS_TORCH:
            return None

        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                device_name = props.name if hasattr(props, 'name') else ""
                gcn_arch = getattr(props, 'gcnArchName', '') or ''

                is_amd = (
                    'AMD' in device_name or
                    'Radeon' in device_name or
                    'gfx' in gcn_arch or
                    (hasattr(torch, 'hip') and torch.hip.is_available())
                )

                if is_amd:
                    arch_name = gcn_arch if gcn_arch else "unknown"

                    if arch_name in UNSUPPORTED_GFX_ARCHS:
                        logger.warning(f"GPU {arch_name} detected but lacks ROCm kernel support, using CPU fallback")
                        self._torch_device = torch.device('cpu')
                        return AcceleratorInfo(
                            type=AcceleratorType.ROCM_GPU,
                            name=f"{device_name} ({arch_name}) [UNSUPPORTED - CPU fallback]",
                            memory_mb=props.total_memory // (1024 * 1024),
                            compute_units=props.multi_processor_count if hasattr(props, 'multi_processor_count') else 16,
                            supports_fp16=False,
                            supports_int8=False,
                            supports_bf16=False,
                            vendor="AMD",
                            device_id=0,
                            properties={"backend": "rocm", "gcn_arch": arch_name, "usable": False},
                        )

                    device = torch.device('cuda:0')
                    self._torch_device = device

                    compute_units = props.multi_processor_count if hasattr(props, 'multi_processor_count') else 16

                    return AcceleratorInfo(
                        type=AcceleratorType.ROCM_GPU,
                        name=f"{device_name} ({arch_name})" if arch_name != "unknown" else device_name,
                        memory_mb=props.total_memory // (1024 * 1024),
                        compute_units=compute_units,
                        supports_fp16=True,
                        supports_int8=True,
                        supports_bf16=True,
                        vendor="AMD",
                        device_id=0,
                        properties={"backend": "rocm", "gcn_arch": arch_name, "usable": True},
                    )
        except Exception as e:
            logger.debug(f"ROCm detection failed: {e}")

        rocm_path = Path("/opt/rocm")
        if rocm_path.exists():
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    vram_mb = 512
                    for line in lines:
                        if 'Total' in line or 'VRAM' in line:
                            parts = line.split()
                            for i, p in enumerate(parts):
                                if p.isdigit() and i > 0:
                                    vram_mb = int(p) // (1024 * 1024) if int(p) > 1024 else int(p)

                    return AcceleratorInfo(
                        type=AcceleratorType.ROCM_GPU,
                        name="AMD Radeon (ROCm)",
                        memory_mb=vram_mb,
                        compute_units=16,
                        supports_fp16=True,
                        supports_int8=True,
                        supports_bf16=True,
                        vendor="AMD",
                        device_id=0,
                    )
            except Exception as e:
                logger.debug(f"rocm-smi detection failed: {e}")

        return None

    def _detect_cuda(self) -> Optional[AcceleratorInfo]:
        if not HAS_TORCH:
            return None

        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                device_name = props.name if hasattr(props, 'name') else ""
                gcn_arch = getattr(props, 'gcnArchName', '') or ''

                is_amd = (
                    'AMD' in device_name or
                    'Radeon' in device_name or
                    'gfx' in gcn_arch or
                    (hasattr(torch, 'hip') and torch.hip.is_available())
                )

                if not is_amd:
                    device = torch.device('cuda:0')
                    self._torch_device = device

                    return AcceleratorInfo(
                        type=AcceleratorType.CUDA_GPU,
                        name=props.name,
                        memory_mb=props.total_memory // (1024 * 1024),
                        compute_units=props.multi_processor_count,
                        supports_fp16=True,
                        supports_int8=True,
                        supports_bf16=True,
                        vendor="NVIDIA",
                        device_id=0,
                    )
        except Exception as e:
            logger.debug(f"CUDA detection failed: {e}")

        return None

    def _detect_vulkan(self) -> Optional[AcceleratorInfo]:
        vulkan_libs = [
            "libvulkan.so",
            "libvulkan.so.1",
            "/usr/lib/x86_64-linux-gnu/libvulkan.so",
            "/usr/lib/libvulkan.so",
        ]

        for lib_path in vulkan_libs:
            try:
                if Path(lib_path).exists() or Path(f"/usr/lib/{lib_path}").exists():
                    result = subprocess.run(
                        ["vulkaninfo", "--summary"],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode == 0 and "AMD" in result.stdout:
                        return AcceleratorInfo(
                            type=AcceleratorType.VULKAN_GPU,
                            name="AMD GPU (Vulkan)",
                            memory_mb=0,
                            compute_units=0,
                            supports_fp16=True,
                            supports_int8=False,
                            supports_bf16=False,
                            vendor="AMD",
                            properties={"backend": "vulkan"},
                        )
            except Exception as e:
                logger.debug(f"Vulkan GPU detection failed: {e}")

        return None

    def _detect_amd_npu(self) -> Optional[AcceleratorInfo]:
        """Detect AMD XDNA NPU via /dev/accel interface."""
        accel_path = Path("/dev/accel/accel0")
        if accel_path.exists():
            try:
                vendor_path = Path("/sys/class/accel/accel0/device/vendor")
                device_path = Path("/sys/class/accel/accel0/device/device")

                if vendor_path.exists() and device_path.exists():
                    vendor_id = int(vendor_path.read_text(encoding='utf-8').strip(), 16)
                    device_id = int(device_path.read_text(encoding='utf-8').strip(), 16)

                    if vendor_id == 0x1022 and device_id == 0x17f0:
                        return AcceleratorInfo(
                            type=AcceleratorType.AMD_NPU,
                            name="AMD XDNA NPU (Strix)",
                            memory_mb=512,
                            compute_units=4,
                            supports_fp16=True,
                            supports_int8=True,
                            supports_bf16=False,
                            vendor="AMD",
                            device_id=device_id,
                            properties={"backend": "xdna", "device_path": str(accel_path)},
                        )
            except Exception as e:
                logger.debug(f"Error detecting AMD NPU: {e}")

        npu_paths = [
            "/dev/amdxdna",
            "/sys/class/misc/amdxdna",
            "/opt/amd/xrt",
        ]

        for path in npu_paths:
            if Path(path).exists():
                return AcceleratorInfo(
                    type=AcceleratorType.AMD_NPU,
                    name="AMD XDNA NPU",
                    memory_mb=512,
                    compute_units=4,
                    supports_fp16=True,
                    supports_int8=True,
                    supports_bf16=False,
                    vendor="AMD",
                    properties={"backend": "xdna"},
                )

        return None

    def _detect_apple_metal(self) -> Optional[AcceleratorInfo]:
        if platform.system() != "Darwin":
            return None

        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=10
            )
            if "Apple" in result.stdout and ("M1" in result.stdout or "M2" in result.stdout or "M3" in result.stdout):
                return AcceleratorInfo(
                    type=AcceleratorType.APPLE_METAL,
                    name="Apple Silicon GPU",
                    memory_mb=16384,
                    compute_units=8,
                    supports_fp16=True,
                    supports_int8=True,
                    supports_bf16=True,
                    vendor="Apple",
                )
        except Exception as e:
            logger.debug(f"MPS detection failed: {e}")

        return None

    def _select_primary_accelerator(self):
        priority = [
            AcceleratorType.AMD_NPU,
            AcceleratorType.ROCM_GPU,
            AcceleratorType.CUDA_GPU,
            AcceleratorType.APPLE_METAL,
            AcceleratorType.VULKAN_GPU,
            AcceleratorType.CPU,
        ]

        accel_types = {a.type for a in self.accelerators}

        for accel_type in priority:
            if accel_type in accel_types:
                self.primary_accelerator = accel_type
                break

        if self.primary_accelerator is None:
            self.primary_accelerator = AcceleratorType.CPU

    def _setup_memory_pool(self):
        cpu_accel = next((a for a in self.accelerators if a.type == AcceleratorType.CPU), None)
        if not cpu_accel:
            cpu_accel = self.accelerators[0]

        gpu_accel = next(
            (a for a in self.accelerators if a.type in [AcceleratorType.ROCM_GPU, AcceleratorType.CUDA_GPU]),
            None
        )

        if gpu_accel and gpu_accel.memory_mb > 0:
            total_mb = cpu_accel.memory_mb
            self.memory_pool = MemoryPool(
                total_mb=total_mb,
                available_mb=total_mb - 2048,
                gpu_allocated_mb=0,
                cpu_allocated_mb=2048,
                unified=True,
            )
        else:
            total_mb = cpu_accel.memory_mb
            self.memory_pool = MemoryPool(
                total_mb=total_mb,
                available_mb=total_mb - 4096,
                gpu_allocated_mb=0,
                cpu_allocated_mb=4096,
                unified=True,
            )

    def _log_detection_results(self):
        logger.info("=" * 50)
        logger.info("Hardware Detection Results")
        logger.info("=" * 50)

        for accel in self.accelerators:
            logger.info(f"  {accel.type.value.upper()}: {accel.name}")
            logger.info(f"    Memory: {accel.memory_mb} MB")
            logger.info(f"    Compute Units: {accel.compute_units}")
            logger.info(f"    FP16: {accel.supports_fp16}, INT8: {accel.supports_int8}")

        logger.info(f"\nPrimary Accelerator: {self.primary_accelerator.value}")
        logger.info(f"Memory Pool: {self.memory_pool.total_mb} MB total, {self.memory_pool.available_mb} MB available")
        logger.info(f"Unified Memory: {self.memory_pool.unified}")
        logger.info("=" * 50)

    def get_torch_device(self) -> "torch.device":
        if self._torch_device is not None:
            return self._torch_device

        if HAS_TORCH:
            if torch.cuda.is_available():
                self._torch_device = torch.device('cuda')
                return self._torch_device

        if HAS_TORCH:
            self._torch_device = torch.device('cpu')
        return self._torch_device

    def get_optimal_batch_size(self, model_size_mb: int, tokens_per_sample: int = 512) -> int:
        """Calculate optimal batch size based on available memory."""
        if not self.memory_pool:
            return 1

        available_mb = self.memory_pool.available_mb
        overhead_mb = 256
        usable_mb = available_mb - overhead_mb

        mb_per_token = 0.0005 * tokens_per_sample
        mb_per_sample = model_size_mb * 0.1 + mb_per_token * tokens_per_sample

        batch_size = max(1, int(usable_mb / mb_per_sample))
        batch_size = min(batch_size, 64)

        return batch_size

    def use_gpu(self) -> bool:
        rocm_accel = next((a for a in self.accelerators if a.type == AcceleratorType.ROCM_GPU), None)
        if rocm_accel and rocm_accel.properties.get("usable", True):
            return True
        cuda_accel = next((a for a in self.accelerators if a.type == AcceleratorType.CUDA_GPU), None)
        if cuda_accel:
            return True
        metal_accel = next((a for a in self.accelerators if a.type == AcceleratorType.APPLE_METAL), None)
        if metal_accel:
            return True
        return False

    def use_npu(self) -> bool:
        return self.primary_accelerator == AcceleratorType.AMD_NPU


class GPUAccelerator:
    _instance = None
    _lock = threading.Lock()
    MAX_CACHE_ENTRIES = 100

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.detector = HardwareDetector()
        self._tensor_cache: Dict[str, Any] = {}
        self._compile_cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()

        self._init_gpu_kernels()

    def _init_gpu_kernels(self):
        if not HAS_NUMBA:
            logger.debug("Numba not available, GPU kernels disabled")
            return

        if not self.detector.use_gpu():
            logger.debug("No GPU detected, GPU kernels disabled")
            return

        try:
            if cuda and cuda.is_available():
                self._numba_cuda_available = True
                logger.info("Numba CUDA/HIP kernels initialized")
        except Exception as e:
            logger.warning(f"GPU kernel init failed: {e}")
            self._numba_cuda_available = False

    def accelerate_matrix_multiply(self, a, b) -> Any:
        if not HAS_NUMPY:
            return None

        a_arr = np.asarray(a) if not isinstance(a, np.ndarray) else a
        b_arr = np.asarray(b) if not isinstance(b, np.ndarray) else b

        if HAS_TORCH and self._torch_available():
            device = self.detector.get_torch_device()
            a_tensor = torch.from_numpy(a_arr).to(device)
            b_tensor = torch.from_numpy(b_arr).to(device)
            result = torch.matmul(a_tensor, b_tensor)
            return result.cpu().numpy()

        if HAS_NUMBA and self._numba_cuda_available:
            try:
                return self._cuda_matmul(a_arr, b_arr)
            except Exception as e:
                logger.debug(f"CUDA matmul failed, falling back to numpy: {e}")

        return np.matmul(a_arr, b_arr)

    def _torch_available(self) -> bool:
        return HAS_TORCH and self.detector._torch_device is not None

    def accelerate_similarity_search(self, query: np.ndarray, vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if not HAS_NUMPY:
            return np.array([]), np.array([])

        query = np.asarray(query, dtype=np.float32)
        vectors = np.asarray(vectors, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if HAS_TORCH and self._torch_available():
            device = self.detector.get_torch_device()
            query_t = torch.from_numpy(query).to(device)
            vectors_t = torch.from_numpy(vectors).to(device)

            query_norm = query_t / (torch.norm(query_t, dim=1, keepdim=True) + 1e-8)
            vectors_norm = vectors_t / (torch.norm(vectors_t, dim=1, keepdim=True) + 1e-8)

            similarities = torch.mm(query_norm, vectors_norm.T)

            scores, indices = torch.topk(similarities, k=min(k, vectors.shape[0]), dim=1)

            return scores.cpu().numpy(), indices.cpu().numpy()

        scores = np.dot(query, vectors.T)
        norms = np.linalg.norm(query, axis=1, keepdims=True) * np.linalg.norm(vectors, axis=1)
        similarities = scores / (norms + 1e-8)

        top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
        top_k_scores = np.take_along_axis(similarities, top_k_indices, axis=1)

        return top_k_scores, top_k_indices

    def accelerate_softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        if not HAS_NUMPY:
            return x

        x = np.asarray(x, dtype=np.float32)

        if HAS_TORCH and self._torch_available():
            device = self.detector.get_torch_device()
            x_tensor = torch.from_numpy(x).to(device)
            result = torch.softmax(x_tensor, dim=axis)
            return result.cpu().numpy()

        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def accelerate_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not HAS_NUMPY:
            return query

        query = np.asarray(query, dtype=np.float32)
        key = np.asarray(key, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)

        d_k = query.shape[-1]
        scores = np.matmul(query, key.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        attention_weights = self.accelerate_softmax(scores, axis=-1)
        output = np.matmul(attention_weights, value)

        return output

    def accelerate_layer_norm(
        self,
        x: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        eps: float = 1e-5,
    ) -> np.ndarray:
        if not HAS_NUMPY:
            return x

        x = np.asarray(x, dtype=np.float32)
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)

        return weight * x_norm + bias


class NPUAccelerator:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.detector = HardwareDetector()
        self.available = self.detector.use_npu()

        if self.available:
            logger.info("AMD XDNA NPU detected - enabling AI acceleration")

    def accelerate_inference(self, model_input: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            return None

        return None

    def quantize_int8(self, tensor: np.ndarray) -> Tuple[np.ndarray, float, float]:
        if not HAS_NUMPY:
            return tensor, 0.0, 1.0

        tensor = np.asarray(tensor, dtype=np.float32)
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        
        if max_val == min_val:
            return np.zeros_like(tensor, dtype=np.int8), 1.0, float(min_val)
        
        scale = (max_val - min_val) / 255.0
        zero_point = min_val

        quantized = np.round((tensor - zero_point) / scale).astype(np.int8)
        return quantized, scale, zero_point


class MemoryPoolManager:
    _instance = None
    _lock = threading.Lock()
    MAX_ALLOCATIONS = 100

    def __new__(cls, config: Any = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                    cls._instance._config = config
        return cls._instance

    def __init__(self, config: Any = None):
        if self._initialized:
            return
        self._initialized = True

        self.detector = HardwareDetector()
        self.pool = self.detector.memory_pool

        self._allocations: Dict[str, Tuple[int, str]] = {}
        self._total_allocated_mb = 0
        self._alloc_lock = threading.Lock()

        if config and hasattr(config, 'hardware'):
            self._reserved_mb = config.hardware.reserved_system_mb
            self._max_cache_mb = config.hardware.max_cache_mb
            self._gpu_memory_mb = config.hardware.gpu_memory_mb
        else:
            self._reserved_mb = 4096
            self._max_cache_mb = 8192
            self._gpu_memory_mb = 4096

        self._cache: Dict[str, Tuple[Any, int, float]] = {}
        self._cache_timestamps: Dict[str, float] = {}

    def allocate(self, name: str, size_mb: int, priority: str = "normal") -> bool:
        with self._alloc_lock:
            if len(self._allocations) >= self.MAX_ALLOCATIONS:
                self._evict_lru_allocations()
            effective_available = self.pool.available_mb - self._reserved_mb

            if size_mb > effective_available:
                if priority == "critical":
                    self._evict_lru(size_mb - effective_available)
                    effective_available = self.pool.available_mb - self._reserved_mb
                    if size_mb > effective_available:
                        return False
                else:
                    return False

            self._allocations[name] = (size_mb, priority)
            self._total_allocated_mb += size_mb
            self.pool.cpu_allocated_mb = self._total_allocated_mb
            self.pool.available_mb = self.pool.total_mb - self._total_allocated_mb

            return True

    def deallocate(self, name: str):
        with self._alloc_lock:
            if name in self._allocations:
                size_mb, _ = self._allocations[name]
                del self._allocations[name]
                self._total_allocated_mb -= size_mb
                self.pool.cpu_allocated_mb = self._total_allocated_mb
                self.pool.available_mb = self.pool.total_mb - self._total_allocated_mb

    def _evict_lru_allocations(self):
        low_priority = [
            (name, size_mb) for name, (size_mb, priority) in self._allocations.items()
            if priority != "critical"
        ]
        low_priority.sort(key=lambda x: x[0])
        for name, size_mb in low_priority[:len(low_priority) // 2]:
            del self._allocations[name]
            self._total_allocated_mb -= size_mb
            self.pool.cpu_allocated_mb = self._total_allocated_mb
            self.pool.available_mb = self.pool.total_mb - self._total_allocated_mb

    def cache_put(self, key: str, data: Any, size_mb: int) -> bool:
        with self._alloc_lock:
            current_cache_mb = sum(s for _, s, _ in self._cache.values())

            if current_cache_mb + size_mb > self._max_cache_mb:
                self._evict_cache_lru(size_mb)

            self._cache[key] = (data, size_mb, time.time())
            self._cache_timestamps[key] = time.time()
            return True

    def cache_get(self, key: str) -> Optional[Any]:
        with self._alloc_lock:
            if key in self._cache:
                data, size_mb, _ = self._cache[key]
                self._cache[key] = (data, size_mb, time.time())
                self._cache_timestamps[key] = time.time()
                return data
        return None

    def _evict_lru(self, needed_mb: int):
        with self._alloc_lock:
            freed_mb = 0
            sorted_allocs = sorted(
                self._allocations.items(),
                key=lambda x: (0 if x[1][1] == "low" else 1 if x[1][1] == "normal" else 2, x[0])
            )

            for name, (size_mb, priority) in sorted_allocs:
                if priority != "critical" and freed_mb < needed_mb:
                    del self._allocations[name]
                    self._total_allocated_mb -= size_mb
                    freed_mb += size_mb
            self.pool.cpu_allocated_mb = self._total_allocated_mb
            self.pool.available_mb = self.pool.total_mb - self._total_allocated_mb

    def _evict_cache_lru(self, needed_mb: int):
        freed_mb = 0
        keys_to_delete = []
        sorted_cache = sorted(
            self._cache.items(),
            key=lambda x: x[1][2]
        )

        for key, (_, size_mb, _) in sorted_cache:
            if freed_mb < needed_mb:
                keys_to_delete.append(key)
                freed_mb += size_mb
            else:
                break

        for key in keys_to_delete:
            del self._cache[key]
            self._cache_timestamps.pop(key, None)

    def get_stats(self) -> Dict[str, Any]:
        with self._alloc_lock:
            cache_size_mb = sum(s for _, s, _ in self._cache.values())

            return {
                "total_mb": self.pool.total_mb,
                "available_mb": self.pool.available_mb,
                "allocated_mb": self._total_allocated_mb,
                "cache_size_mb": cache_size_mb,
                "allocations": len(self._allocations),
                "cache_entries": len(self._cache),
                "unified_memory": self.pool.unified,
            }


def get_hardware_detector() -> HardwareDetector:
    return HardwareDetector()


def get_gpu_accelerator() -> GPUAccelerator:
    return GPUAccelerator()


def get_npu_accelerator() -> NPUAccelerator:
    return NPUAccelerator()


def get_memory_pool(config: Any = None) -> MemoryPoolManager:
    return MemoryPoolManager(config)


_accurate_dot = None
if HAS_NUMPY:
    def _accurate_dot_cpu(a, b):
        return np.dot(a, b)
    _accurate_dot = _accurate_dot_cpu


def accelerated_function(fallback: Callable):
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                accelerator = get_gpu_accelerator()
                if accelerator.detector.use_gpu():
                    return func(*args, **kwargs)
            except Exception as e:
                logger.debug(f"GPU acceleration failed, using fallback: {e}")

            return fallback(*args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    "HardwareDetector",
    "GPUAccelerator",
    "NPUAccelerator",
    "MemoryPoolManager",
    "AcceleratorType",
    "AcceleratorInfo",
    "MemoryPool",
    "get_hardware_detector",
    "get_gpu_accelerator",
    "get_npu_accelerator",
    "get_memory_pool",
    "accelerated_function",
    "HAS_NUMPY",
    "HAS_TORCH",
    "HAS_NUMBA",
]
