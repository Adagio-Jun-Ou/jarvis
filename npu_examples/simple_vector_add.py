"""Simple MLIR-AIE example for AMD XDNA NPU.

This example demonstrates how to create and compile a simple vector addition
kernel for the AMD XDNA NPU using MLIR-AIE.

Prerequisites:
1. MLIR-AIE installed (pip install mlir-aie)
2. XDNA driver loaded (kernel 6.12+)
3. Set environment:
   export PATH=$PATH:/path/to/mlir_aie/bin
   export PYTHONPATH=$PYTHONPATH:/path/to/mlir_aie/python

Running:
    cd /home/user/jarvis
    source venv/bin/activate
    export PATH=$PATH:$VIRTUAL_ENV/lib/python3.13/site-packages/mlir_aie/bin
    export PYTHONPATH=$PYTHONPATH:$VIRTUAL_ENV/lib/python3.13/site-packages/mlir_aie/python
    python npu_examples/simple_vector_add.py
"""
import os
import sys

VIRTUAL_ENV = os.environ.get('VIRTUAL_ENV', '')
if VIRTUAL_ENV:
    mlir_aie_python = os.path.join(VIRTUAL_ENV, 'lib', 'python3.13', 'site-packages', 'mlir_aie', 'python')
    if mlir_aie_python not in sys.path:
        sys.path.insert(0, mlir_aie_python)
    mlir_aie_bin = os.path.join(VIRTUAL_ENV, 'lib', 'python3.13', 'site-packages', 'mlir_aie', 'bin')
    if mlir_aie_bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = mlir_aie_bin + ':' + os.environ.get('PATH', '')

import time
from pathlib import Path

MLIR_AIE_AVAILABLE = False
try:
    from aie.ir import Context, Location, Module
    from aie.dialects import aie, aiex, aievec
    from aie.dialects.aie import (
        AIEDevice,
        Core,
        CoreOp,
        Device,
        Lock,
        LockAction,
        LockOp,
        MemOp,
        MemRefType,
        ObjectFifo,
        ObjectFifoType,
        Tile,
        TileOp,
        external_func,
       func,
    )
    MLIR_AIE_AVAILABLE = True
except ImportError as e:
    print(f"MLIR-AIE not available: {e}")
    print("Install with: pip install mlir-aie")
    print("And set PYTHONPATH to include mlir_aie/python")


def check_npu_available():
    """Check if NPU device is available."""
    npu_device = Path("/dev/accel/accel0")
    if npu_device.exists():
        print("✓ NPU device found: /dev/accel/accel0")
        return True
    print("✗ NPU device not found")
    print("  Kernel 6.12+ with amdxdna driver required")
    return False


def check_xrt_available():
    """Check if XRT (Xilinx Runtime) is available."""
    try:
        result = os.popen("xrt-smi examine 2>/dev/null").read()
        if "NPU" in result or "Strix" in result:
            print("✓ XRT available, NPU detected")
            return True
        print("XRT available but NPU not visible")
    except Exception:
        print("✗ XRT not available")
        print("  Build and install xdna-driver/xrt from https://github.com/Xilinx/mlir-aie")
    return False


def create_simple_aie_module():
    """Create a simple MLIR-AIE module for demonstration.
    
    This creates a minimal AIE configuration that could be used
    for vector operations on the NPU.
    """
    if not MLIR_AIE_AVAILABLE:
        return None
    
    try:
        with Context() as ctx:
            ctx.allow_unregistered_dialects = True
            
            with Location.unknown(ctx):
                module = Module.create()
                
                dev = AIEDevice.npu1_4col
                
                with Location.unknown(ctx):
                    
                    device = Device(dev)
                    
                    print("Created MLIR-AIE module for NPU1 4-column device")
                    print(f"Module: {module}")
                    
                    return str(module)
                    
    except Exception as e:
        print(f"Error creating module: {e}")
        return None


def get_npu_info():
    """Print NPU information and MLIR-AIE status."""
    print("=" * 60)
    print("AMD XDNA NPU Status")
    print("=" * 60)
    print()
    
    npu_ok = check_npu_available()
    print()
    
    xrt_ok = check_xrt_available()
    print()
    
    print(f"MLIR-AIE Python API: {'✓ Available' if MLIR_AIE_AVAILABLE else '✗ Not available'}")
    print()
    
    if VIRTUAL_ENV:
        mlir_bin = os.path.join(VIRTUAL_ENV, 'lib', 'python3.13', 'site-packages', 'mlir_aie', 'bin')
        aie_opt = os.path.join(mlir_bin, 'aie-opt')
        aie_translate = os.path.join(mlir_bin, 'aie-translate')
        print(f"aie-opt: {'✓ ' + aie_opt if os.path.exists(aie_opt) else '✗ Not found'}")
        print(f"aie-translate: {'✓ ' + aie_translate if os.path.exists(aie_translate) else '✗ Not found'}")
    print()
    
    if MLIR_AIE_AVAILABLE:
        print("=" * 60)
        print("Creating Sample MLIR-AIE Module")
        print("=" * 60)
        print()
        module_str = create_simple_aie_module()
        if module_str:
            print("Module created successfully!")
    
    print()
    print("=" * 60)
    print("Next Steps for Full NPU Acceleration")
    print("=" * 60)
    print()
    print("1. For custom kernels:")
    print("   - Use the IRON API for simplified NPU programming")
    print("   - See: https://xilinx.github.io/mlir-aie/programming_guide")
    print()
    print("2. For ONNX model inference on Windows:")
    print("   - Install Ryzen AI Software: https://github.com/amd/RyzenAI-SW")
    print("   - Use VitisAIExecutionProvider with ONNX Runtime")
    print()
    print("3. For LLM inference:")
    print("   - Use amd-shark-ai: https://github.com/nod-ai/amd-shark-ai")
    print("   - Supports llama, phi-3, qwen, and more")
    print()
    print("4. Documentation:")
    print("   - MLIR-AIE: https://xilinx.github.io/mlir-aie/")
    print("   - Ryzen AI: https://ryzenai.docs.amd.com/")
    print()


if __name__ == "__main__":
    get_npu_info()