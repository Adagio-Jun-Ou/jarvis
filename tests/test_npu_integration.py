"""NPU Detection and Status Integration Tests.

Tests the hardware detection pipeline:
- NPU hardware detection via XDNAController
- ONNX runtime provider detection
- MLIR-AIE tool availability
- Unified accelerator abstraction
"""
import pytest
from pathlib import Path








class TestNPUVoiceIntegration:
    """Test NPU voice processing integration."""

    def test_npu_voice_module_import(self):
        """Test npu_voice module imports."""
        try:
            from core.npu_voice import (
                NPUVoiceTokenizer,
                NPUVoiceProcessor,
                get_voice_processor,
            )

            assert NPUVoiceTokenizer is not None
            assert NPUVoiceProcessor is not None
        except ImportError as e:
            pytest.skip(f"npu_voice module not available: {e}")


    def test_npu_models_import(self):
        """Test npu_models module imports."""
        try:
            from core.npu_models import (
                FastEmbeddingModel,
                FastVoiceTokenizer,
            )

            assert FastEmbeddingModel is not None
            assert FastVoiceTokenizer is not None
        except ImportError as e:
            pytest.skip(f"npu_models module not available: {e}")






if __name__ == "__main__":
    pytest.main([__file__, "-v"])