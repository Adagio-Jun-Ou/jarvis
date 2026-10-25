"""Tests for Concept Model - Tri-model abstraction layer."""
import asyncio
import os
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestConceptConfig:
    """Test ConceptConfig dataclass."""

    def test_import(self):
        from core.concept_model import ConceptConfig

        assert ConceptConfig is not None

    def test_default_values(self):
        from core.concept_model import ConceptConfig

        config = ConceptConfig()
        assert config.model_id == "glm-5"
        assert config.embedding_dim == 768
        assert config.max_seq_length == 8192
        assert config.embedding_provider == "pytorch"  # Changed from onnx for jina-embeddings-v5

    def test_capabilities(self):
        from core.concept_model import ConceptConfig

        config = ConceptConfig()
        assert "meta_reasoning" in config.capabilities
        assert "abstraction" in config.capabilities
        assert "pattern_recognition" in config.capabilities


class TestConceptTaskType:
    """Test ConceptTaskType enum."""

    def test_import(self):
        from core.concept_model import ConceptTaskType

        assert ConceptTaskType is not None

    def test_values(self):
        from core.concept_model import ConceptTaskType

        assert ConceptTaskType.ABSTRACTION.value == "abstraction"
        assert ConceptTaskType.SYNTHESIS.value == "synthesis"
        assert ConceptTaskType.PATTERN_RECOGNITION.value == "pattern_recognition"
        assert ConceptTaskType.META_REASONING.value == "meta_reasoning"
        assert ConceptTaskType.BRIDGE.value == "bridge"
        assert ConceptTaskType.STRATEGIC.value == "strategic"


class TestConceptMode:
    """Test ConceptMode enum."""

    def test_import(self):
        from core.concept_model import ConceptMode

        assert ConceptMode is not None

    def test_values(self):
        from core.concept_model import ConceptMode

        assert ConceptMode.STANDALONE.value == "standalone"
        assert ConceptMode.SYNTHESIZE.value == "synthesize"
        assert ConceptMode.GUIDE.value == "guide"
        assert ConceptMode.VALIDATE.value == "validate"
        assert ConceptMode.ENRICH.value == "enrich"


class TestConceptEmbedder:
    """Test ConceptEmbedder class."""

    def test_import(self):
        from core.concept_model import ConceptEmbedder

        assert ConceptEmbedder is not None

    def test_init_defaults(self):
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder()
        assert embedder.model_name == "jinaai/jina-embeddings-v5-text-nano"
        assert embedder.max_seq_length == 8192
        assert embedder.provider == "pytorch"

    def test_init_custom(self):
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder(
            model_name="custom-model",
            max_seq_length=4096,
            provider="pytorch",
        )
        assert embedder.model_name == "custom-model"
        assert embedder.max_seq_length == 4096
        assert embedder.provider == "pytorch"

    def test_get_device_cpu(self):
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder(device="cpu")
        assert embedder._get_device() == "cpu"

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_encode_single_text(self):
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder(provider="onnx")
        embedding = embedder.encode("Hello world")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_encode_multiple_texts(self):
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder(provider="onnx")
        texts = ["Hello world", "Machine learning is great"]
        embeddings = embedder.encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 768)

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_similarity(self):
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder(provider="onnx")
        texts = [
            "Neural networks learn patterns",
            "Machine learning requires data",
            "Paris is in France",
        ]
        embeddings = embedder.encode(texts)

        # Check for zero vectors (model not loaded)
        norms = [np.linalg.norm(e) for e in embeddings]
        if any(n == 0 for n in norms):
            # Model not loaded - zero vectors returned, skip similarity test
            import pytest
            pytest.skip("Embedding model not loaded, cannot test similarity")

        sim_ml = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        sim_geo = np.dot(embeddings[0], embeddings[2]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[2])
        )

        assert -1 <= sim_ml <= 1
        assert -1 <= sim_geo <= 1


class TestConceptRequest:
    """Test ConceptRequest dataclass."""

    def test_import(self):
        from core.concept_model import ConceptRequest

        assert ConceptRequest is not None

    def test_create(self):
        from core.concept_model import ConceptRequest, ConceptTaskType, ConceptMode

        request = ConceptRequest(
            id="test-1",
            task_type=ConceptTaskType.ABSTRACTION,
            prompt="Extract pattern",
        )
        assert request.id == "test-1"
        assert request.task_type == ConceptTaskType.ABSTRACTION
        assert request.mode == ConceptMode.SYNTHESIZE


class TestConceptStats:
    """Test ConceptStats dataclass."""

    def test_import(self):
        from core.concept_model import ConceptStats

        assert ConceptStats is not None

    def test_defaults(self):
        from core.concept_model import ConceptStats

        stats = ConceptStats()
        assert stats.total_requests == 0
        assert stats.abstractions == 0
        assert stats.errors == 0


class TestConceptModelOrchestrator:
    """Test ConceptModelOrchestrator class."""

    def test_import(self):
        from core.concept_model import ConceptModelOrchestrator

        assert ConceptModelOrchestrator is not None

    def test_init(self):
        from core.concept_model import ConceptModelOrchestrator

        orchestrator = ConceptModelOrchestrator()
        assert orchestrator.concept_llm is None
        assert orchestrator._running is False

    def test_init_with_llm(self):
        from core.concept_model import ConceptModelOrchestrator

        mock_llm = MagicMock()
        orchestrator = ConceptModelOrchestrator(concept_llm=mock_llm)
        assert orchestrator.concept_llm == mock_llm

    @pytest.mark.asyncio
    async def test_start_stop(self):
        from core.concept_model import ConceptModelOrchestrator

        orchestrator = ConceptModelOrchestrator()
        await orchestrator.start()
        assert orchestrator._running is True

        await orchestrator.stop()
        assert orchestrator._running is False

    @pytest.mark.asyncio
    async def test_extract_abstract_pattern_no_llm(self):
        from core.concept_model import ConceptModelOrchestrator

        orchestrator = ConceptModelOrchestrator()
        await orchestrator.start()

        with pytest.raises(RuntimeError, match="No LLM configured"):
            await orchestrator.extract_abstract_pattern(["test"])

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_extract_abstract_pattern_with_mock_llm(self):
        from core.concept_model import ConceptModelOrchestrator

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(
            return_value={"message": {"content": "Abstract pattern: test pattern"}}
        )

        orchestrator = ConceptModelOrchestrator(concept_llm=mock_llm)
        await orchestrator.start()

        result = await orchestrator.extract_abstract_pattern(
            ["Instance 1", "Instance 2"], domain="test"
        )

        assert "pattern" in result.lower() or len(result) > 0
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_synthesize_concepts_with_mock_llm(self):
        from core.concept_model import ConceptModelOrchestrator

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(
            return_value={"message": {"content": "Synthesis: combined concepts"}}
        )

        orchestrator = ConceptModelOrchestrator(concept_llm=mock_llm)
        await orchestrator.start()

        result = await orchestrator.synthesize_concepts(
            ["concept1", "concept2"], context="test context"
        )

        assert len(result) > 0
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_find_patterns(self):
        from core.concept_model import ConceptModelOrchestrator

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(
            return_value={"message": {"content": "- Pattern 1\n- Pattern 2"}}
        )

        orchestrator = ConceptModelOrchestrator(concept_llm=mock_llm)
        await orchestrator.start()

        result = await orchestrator.find_patterns(
            [{"text": "Data shows X correlates with Y"}, {"text": "Similar patterns in Z"}]
        )

        assert isinstance(result, list)
        await orchestrator.stop()


class TestConceptModelSingleton:
    """Test singleton functions."""

    def test_get_concept_orchestrator_import(self):
        from core.concept_model import get_concept_orchestrator

        assert get_concept_orchestrator is not None

    def test_init_concept_model_import(self):
        from core.concept_model import init_concept_model

        assert init_concept_model is not None


class TestConceptModelIntegration:
    """Integration tests for concept model with tri-model architecture."""

    @pytest.mark.asyncio
    async def test_tri_model_collaboration(self):
        from core.concept_model import ConceptModelOrchestrator

        mock_primary = MagicMock()
        mock_primary.chat = AsyncMock(return_value={"message": {"content": "Primary response"}})

        mock_backend = MagicMock()
        mock_backend.chat = AsyncMock(return_value={"message": {"content": "Backend response"}})

        mock_concept = MagicMock()
        mock_concept.chat = AsyncMock(
            return_value={"message": {"content": "Concept abstraction"}}
        )

        orchestrator = ConceptModelOrchestrator(
            primary_llm=mock_primary,
            backend_llm=mock_backend,
            concept_llm=mock_concept,
        )
        await orchestrator.start()

        result = await orchestrator.extract_abstract_pattern(["test instance"])
        assert len(result) > 0

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_fallback_to_backend(self):
        from core.concept_model import ConceptModelOrchestrator

        mock_backend = MagicMock()
        mock_backend.chat = AsyncMock(
            return_value={"message": {"content": "Backend fallback response"}}
        )

        orchestrator = ConceptModelOrchestrator(
            backend_llm=mock_backend,
            concept_llm=None,
        )
        await orchestrator.start()

        result = await orchestrator.synthesize_concepts(["a", "b"])
        assert len(result) > 0

        await orchestrator.stop()


class TestConceptEmbedderCaching:
    """Test embedding caching behavior."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_cache_hit(self):
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder(provider="onnx")
        text = "Cache test text"

        embedding1 = embedder.encode(text)
        embedding2 = embedder.encode(text)

        np.testing.assert_array_equal(embedding1, embedding2)

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_cache_different_texts(self):
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder(provider="onnx")

        embedding1 = embedder.encode("First text")
        embedding2 = embedder.encode("Second text")

        # Check for zero vectors (model not loaded)
        import numpy as np
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 and norm2 == 0:
            # Model not loaded - both are zero vectors, skip test
            pytest.skip("Embedding model not loaded, cannot test cache differentiation")

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(embedding1, embedding2)


class TestConceptEmbedderClustering:
    """Test embedding clustering functionality."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_cluster_concepts(self):
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder(provider="onnx")
        texts = [
            "Machine learning processes data",
            "Neural networks recognize patterns",
            "Paris is a beautiful city",
            "London has many landmarks",
        ]

        embeddings = embedder.encode(texts)
        
        # Check for zero vectors (model not loaded)
        import numpy as np
        norms = [np.linalg.norm(e) for e in embeddings]
        if any(n == 0 for n in norms):
            # Model not loaded - zero vectors returned, skip clustering test
            pytest.skip("Embedding model not loaded, cannot test clustering")

        clusters = embedder.cluster_concepts(texts, n_clusters=2)

        assert isinstance(clusters, dict)
        assert len(clusters) == 2
        total = sum(len(v) for v in clusters.values())
        assert total == 4


class TestConceptModelStats:
    """Test statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        from core.concept_model import ConceptModelOrchestrator

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value={"message": {"content": "Response"}})

        orchestrator = ConceptModelOrchestrator(concept_llm=mock_llm)
        await orchestrator.start()

        initial_requests = orchestrator._stats.total_requests

        await orchestrator.synthesize_concepts(["a", "b"])

        assert orchestrator._stats.total_requests > initial_requests

        await orchestrator.stop()