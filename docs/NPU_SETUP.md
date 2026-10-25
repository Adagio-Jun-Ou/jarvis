# AMD XDNA NPU Setup Guide for Jarvis

This guide explains how to enable AMD XDNA NPU acceleration on Linux.

## Current Status

✓ **Hardware**: AMD XDNA NPU (device 0x17f0) detected
✓ **Kernel Driver**: `amdxdna` loaded (kernel 6.12+)
✗ **XRT**: Not installed - required for NPU execution
✗ **MLIR-AIE Python**: Installed but needs XRT for execution

## Hardware Information

- **Device**: AMD XDNA NPU (Strix Point)
- **Device ID**: 0x17f0
- **Compute Units**: 4 AI cores
- **Memory**: 512MB
- **Kernel Driver**: amdxdna (included in Linux 6.12+)

## Full NPU Setup (Ubuntu 24.04/24.10)

### Step 1: Install XRT and XDNA Driver

```bash
# Clone MLIR-AIE repository (contains driver build scripts)
cd ~
git clone https://github.com/Xilinx/mlir-aie.git
cd mlir-aie

# Build and install XDNA driver + XRT
# This requires sudo access
bash ./utils/build_drivers.sh

# Reboot after installation
sudo reboot
```

### Step 2: Verify XRT Installation

After reboot:
```bash
source /opt/xilinx/xrt/setup.sh
xrt-smi examine
```

You should see your NPU listed:
```
Devices present
BDF             :  Name
------------------------------------
[0000:xx:xx.x]  :  NPU Strix
```

### Step 3: Install MLIR-AIE Python API

Already installed in Jarvis venv:
```bash
cd /home/user/jarvis
source venv/bin/activate

# MLIR-AIE is already installed:
# mlir-aie 1.2.1 at /home/user/jarvis/venv/lib/python3.13/site-packages/mlir_aie/
```

### Step 4: Set Environment Variables

Add to your `~/.bashrc`:
```bash
# XRT
source /opt/xilinx/xrt/setup.sh

# MLIR-AIE
export PATH="$PATH:/home/user/jarvis/venv/lib/python3.13/site-packages/mlir_aie/bin"
export PYTHONPATH="$PYTHONPATH:/home/user/jarvis/venv/lib/python3.13/site-packages/mlir_aie/python"
```

## Running NPU Examples

```bash
cd /home/user/jarvis
source venv/bin/activate
source /opt/xilinx/xrt/setup.sh  # After XRT installation
python npu_examples/simple_vector_add.py
```

## Alternative: Ryzen AI Software (Windows Only)

If you're running on Windows:
1. Download: https://github.com/amd/RyzenAI-SW
2. Install `ryzen-ai-lt-1.7.0.exe`
3. Use `VitisAIExecutionProvider` with ONNX Runtime

This provides direct ONNX model acceleration without custom compilation.

## Supported Models for NPU

### Embedding Models (Low latency)
- all-MiniLM-L6-v2
- all-mpnet-base-v2
- bge-small-en, bge-base-en

### Vision Models
- ResNet50, MobileNetV2
- EfficientNet-B0
- YOLOv8s

### Speech Models
- Whisper-tiny, Whisper-base

### LLMs (via amd-shark-ai)
- Phi-2, Phi-3-mini
- TinyLlama-1.1B
- Qwen2-0.5B, Qwen2-1.5B

## Resources

- [MLIR-AIE Documentation](https://xilinx.github.io/mlir-aie/)
- [IRON Programming Guide](https://xilinx.github.io/mlir-aie/programming_guide)
- [Ryzen AI Software](https://github.com/amd/RyzenAI-SW)
- [AMD-SHARK AI](https://github.com/nod-ai/amd-shark-ai)
- [MLIR-AIE GitHub](https://github.com/Xilinx/mlir-aie)

## Troubleshooting

### NPU not visible in xrt-smi
- Ensure BIOS has NPU/IPU enabled
- Update to latest BIOS
- Check kernel version: `uname -r` (should be 6.11+)

### aie-opt not found
```bash
export PATH="$PATH:/home/user/jarvis/venv/lib/python3.13/site-packages/mlir_aie/bin"
```

### ModuleNotFoundError: aie.ir
```bash
export PYTHONPATH="$PYTHONPATH:/home/user/jarvis/venv/lib/python3.13/site-packages/mlir_aie/python"
```

### Permission denied on /dev/accel/accel0
```bash
# Add user to render group
sudo usermod -a -G render $USER
# Log out and back in
```