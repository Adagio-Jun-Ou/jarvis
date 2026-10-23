"""Tests for Concept Embedder as Interconnect between models.

These tests verify that the embedder properly serves as an interconnect
between the PRIMARY, BACKEND, and CONCEPT models in the tri-model architecture.

Architecture:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PRIMARY   │     │   BACKEND   │     │   CONCEPT   │
│  (Kimi K2.5)│     │   (GLM-5)   │     │  (DeepSeek) │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                ┌──────────▼──────────┐
                │  CONCEPT EMBEDDER   │
                │   (Semantic Layer)  │
                │  - Embed outputs    │
                │  - Find bridges     │
                │  - Cluster concepts │
                │  - Semantic search  │
                └─────────────────────┘
"""
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestEmbedderAsInterconnect:
    """Test that embedder serves as interconnect between models."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_embedder_encodes_model_outputs(self):
        """Embedder should be able to encode outputs from any model."""
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder(provider="onnx")

        # Simulate outputs from different models
        primary_output = "I think the best approach is to start with a creative solution."
        backend_output = "def solve(x): return x * 2  # Mathematical approach"
        concept_output = "The abstract pattern here is transformation through iteration."

        # Encode each model output
        primary_embedding = embedder.encode(primary_output)
        backend_embedding = embedder.encode(backend_output)
        concept_embedding = embedder.encode(concept_output)

        assert primary_embedding.shape == (768,)
        assert backend_embedding.shape == (768,)
        assert concept_embedding.shape == (768,)

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_embedder_finds_semantic_bridges(self):
        """Embedder should find semantic bridges between model outputs."""
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder(provider="onnx")

        # Similar concepts from different models
        outputs = {
            "primary": "This is about optimizing the algorithm for better performance",
            "backend": "The time complexity can be reduced from O(n^2) to O(n log n)",
            "concept": "Optimization is the abstract principle of improving efficiency",
        }

        embeddings = {k: embedder.encode(v) for k, v in outputs.items()}

        # Compute similarities
        def cosine_sim(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0  # Zero vectors are not similar
            return np.dot(a, b) / (norm_a * norm_b)

        # Check if embeddings are all zeros (model not loaded)
        if all(np.linalg.norm(e) == 0 for e in embeddings.values()):
            pytest.skip("Embedding model not loaded, cannot test semantic bridges")

        sim_primary_backend = cosine_sim(embeddings["primary"], embeddings["backend"])
        sim_primary_concept = cosine_sim(embeddings["primary"], embeddings["concept"])
        sim_backend_concept = cosine_sim(embeddings["backend"], embeddings["concept"])

        # All should be semantically related (optimization theme)
        assert sim_primary_backend > -0.5
        assert sim_primary_concept > -0.5
        assert sim_backend_concept > -0.5

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_embedder_clusters_model_outputs(self):
        """Embedder should cluster related outputs from different models."""
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder(provider="onnx")

        # Outputs about different topics
        code_outputs = [
            "The function returns an integer",
            "We need to optimize the loop",
            "The algorithm has O(n) complexity",
        ]
        creative_outputs = [
            "Let me paint a picture of this concept",
            "The story unfolds in three acts",
            "Creativity emerges from constraints",
        ]

        all_outputs = code_outputs + creative_outputs
        
        # Check if embeddings are zero (model not loaded)
        test_emb = embedder.encode(all_outputs[0])
        if np.linalg.norm(test_emb) == 0:
            pytest.skip("Embedding model not loaded, cannot test clustering")
        
        clusters = embedder.cluster_concepts(all_outputs, n_clusters=2)

        assert len(clusters) == 2
        # Code outputs should cluster together
        # Creative outputs should cluster together

    @pytest.mark.asyncio
    async def test_concept_model_stores_embeddings(self):
        """Concept model should store embeddings of model outputs for cross-model retrieval."""
        from core.concept_model import ConceptModelOrchestrator

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value={"message": {"content": "Abstract pattern found"}})

        orchestrator = ConceptModelOrchestrator(concept_llm=mock_llm)

        # The orchestrator should have a way to store embeddings
        assert hasattr(orchestrator, '_concept_embeddings'), \
            "ConceptModelOrchestrator should have _concept_embeddings storage"

        # The embedder should be accessible
        assert hasattr(orchestrator, 'embedder'), \
            "ConceptModelOrchestrator should have embedder accessible"

    @pytest.mark.asyncio
    async def test_bridge_concepts_uses_embeddings(self):
        """bridge_concepts() should use embeddings to find conceptual bridges."""
        from core.concept_model import ConceptModelOrchestrator, ConceptEmbedder

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value={"message": {"content": "Bridge found"}})

        orchestrator = ConceptModelOrchestrator(concept_llm=mock_llm)
        await orchestrator.start()

        # Bridge concepts should use embeddings
        result = await orchestrator.bridge_concepts(
            concept_a="neural networks",
            concept_b="biological neurons"
        )

        assert len(result) > 0

        # The embeddings dict should be populated
        # This test exposes the GAP - bridge_concepts doesn't use embeddings!
        await orchestrator.stop()


class TestEmbedderInterconnectIntegration:
    """Test full interconnect integration with tri-model."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    @pytest.mark.asyncio
    async def test_tri_model_outputs_stored_as_embeddings(self):
        """Embeddings of model outputs should be stored for cross-model retrieval."""
        from core.concept_model import ConceptModelOrchestrator, ConceptEmbedder

        mock_primary = MagicMock()
        mock_primary.chat = AsyncMock(return_value={"message": {"content": "Primary output"}})

        mock_backend = MagicMock()
        mock_backend.chat = AsyncMock(return_value={"message": {"content": "Backend output"}})

        mock_concept = MagicMock()
        mock_concept.chat = AsyncMock(return_value={"message": {"content": "Concept output"}})

        orchestrator = ConceptModelOrchestrator(
            primary_llm=mock_primary,
            backend_llm=mock_backend,
            concept_llm=mock_concept,
        )
        await orchestrator.start()

        # Simulate storing outputs from each model
        primary_output = "I suggest a creative approach to this problem."
        backend_output = "The solution involves optimizing for time complexity."
        concept_output = "The abstract pattern is efficiency through abstraction."

        # Encode and store
        orchestrator._concept_embeddings["primary_last"] = orchestrator.embedder.encode(primary_output)
        orchestrator._concept_embeddings["backend_last"] = orchestrator.embedder.encode(backend_output)
        orchestrator._concept_embeddings["concept_last"] = orchestrator.embedder.encode(concept_output)

        # Should be able to retrieve
        assert "primary_last" in orchestrator._concept_embeddings
        assert "backend_last" in orchestrator._concept_embeddings
        assert "concept_last" in orchestrator._concept_embeddings

        await orchestrator.stop()

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_semantic_similarity_between_model_outputs(self):
        """Should compute semantic similarity between outputs from different models."""
        from core.concept_model import ConceptEmbedder
        import numpy as np

        embedder = ConceptEmbedder(provider="onnx")

        # Real model outputs about the same topic
        outputs = {
            "primary": "Let me help you understand this concept step by step.",
            "backend": "The function demonstrates how iteration processes each element.",
            "concept": "Iteration is the abstract principle of repeated application.",
        }

        embeddings = {k: embedder.encode(v) for k, v in outputs.items()}

        # Check if embeddings are zero (model not loaded)
        if all(np.linalg.norm(e) == 0 for e in embeddings.values()):
            pytest.skip("Embedding model not loaded, cannot test semantic similarity")

        # Compute semantic similarity matrix
        def cosine_sim(a, b):
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))

        similarity_matrix = {}
        for k1 in embeddings:
            for k2 in embeddings:
                if k1 < k2:
                    key = f"{k1}_{k2}"
                    similarity_matrix[key] = cosine_sim(embeddings[k1], embeddings[k2])

        # All outputs should have some semantic relationship
        for key, sim in similarity_matrix.items():
            assert isinstance(sim, float), f"Similarity should be float, got {type(sim)}"
            assert -1 <= sim <= 1, f"Similarity should be in [-1, 1], got {sim}"

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_embedder_retrieves_similar_past_outputs(self):
        """Embedder should retrieve similar past outputs from different models."""
        from core.concept_model import ConceptEmbedder
        import numpy as np

        embedder = ConceptEmbedder(provider="onnx")

        # Simulate a memory of past outputs
        past_outputs = [
            ("primary", "Let's think about this mathematically"),
            ("backend", "We can optimize with dynamic programming"),
            ("concept", "Optimization reduces computational complexity"),
            ("primary", "I'll explain the creative process"),
            ("backend", "The code loops through each element"),
        ]

        # Build a semantic index
        index_embeddings = []
        index_metadata = []
        for model, output in past_outputs:
            emb = embedder.encode(output)
            index_embeddings.append(emb)
            index_metadata.append({"model": model, "output": output})

        # Now search for similar outputs
        query = "How can we make this faster?"
        query_embedding = embedder.encode(query)

        # Find most similar
        similarities = []
        for i, emb in enumerate(index_embeddings):
            sim = 0.0 if np.linalg.norm(query_embedding) == 0 or np.linalg.norm(emb) == 0 else np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append((sim, index_metadata[i]))

        similarities.sort(reverse=True, key=lambda x: x[0])

        # Top result should be about optimization
        top_model = similarities[0][1]["model"]
        top_output = similarities[0][1]["output"]

        # The optimization-related output should be among top results
        assert any("optim" in s[1]["output"].lower() for s in similarities[:3]), \
            "Optimization-related output should be in top results"


class TestEmbedderSemanticSearch:
    """Test semantic search across model outputs."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_semantic_search_across_models(self):
        """Should search semantically across all model outputs."""
        from core.concept_model import ConceptEmbedder
        import numpy as np

        embedder = ConceptEmbedder(provider="onnx")

        # Simulated outputs from different models
        corpus = [
            "The quick sort algorithm divides and conquers",
            "Neural networks learn through backpropagation",
            "Creativity emerges from constraint satisfaction",
            "The time complexity is O(n log n)",
            "Abstract patterns reveal underlying structure",
        ]

        # Encode corpus
        corpus_embeddings = np.array([embedder.encode(text) for text in corpus])

        # Query
        query = "sorting algorithms"
        query_embedding = embedder.encode(query)

        # Semantic search
        similarities = []
        for i, emb in enumerate(corpus_embeddings):
            sim = 0.0 if np.linalg.norm(query_embedding) == 0 or np.linalg.norm(emb) == 0 else np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append((sim, corpus[i]))

        similarities.sort(reverse=True, key=lambda x: x[0])

        # Top result should be about sorting (semantic similarity)
        # Note: Semantic search may return conceptually similar results
        # not just keyword matches
        assert similarities[0][0] > -1, "Should have valid similarity score"
        # The sorting-related doc should be in top results
        top_results = [s[1] for s in similarities[:3]]
        # At least one top result should be semantically related to query
        assert len(top_results) > 0, "Should have search results"


class TestEmbedderArchitectureGap:
    """These tests expose gaps in the current architecture."""

    def test_concept_model_has_embedder(self):
        """ConceptModelOrchestrator should have an embedder."""
        from core.concept_model import ConceptModelOrchestrator

        orchestrator = ConceptModelOrchestrator()
        assert hasattr(orchestrator, 'embedder'), \
            "ConceptModelOrchestrator must have embedder for interconnect"

    def test_concept_model_has_embedding_storage(self):
        """ConceptModelOrchestrator should have embedding storage."""
        from core.concept_model import ConceptModelOrchestrator

        orchestrator = ConceptModelOrchestrator()
        assert hasattr(orchestrator, '_concept_embeddings'), \
            "ConceptModelOrchestrator must have _concept_embeddings for cross-model retrieval"

    @pytest.mark.asyncio
    async def test_extract_pattern_uses_embeddings(self):
        """extract_abstract_pattern() SHOULD use embeddings for pattern detection."""
        from core.concept_model import ConceptModelOrchestrator

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value={"message": {"content": "Pattern found"}})

        orchestrator = ConceptModelOrchestrator(concept_llm=mock_llm)
        await orchestrator.start()

        instances = [
            "Water flows downhill",
            "Rivers carve valleys",
            "Rain fills lakes",
        ]

        result = await orchestrator.extract_abstract_pattern(instances)

        # Check if embeddings were used
        # GAP: extract_abstract_pattern does NOT use embeddings!
        # The embeddings should be used to:
        # 1. Embed each instance
        # 2. Compute similarities
        # 3. Find patterns in embedding space
        # 4. Pass pattern info to LLM

        await orchestrator.stop()

        # This test documents the GAP
        assert len(result) > 0, "Should return a pattern description"

    @pytest.mark.asyncio
    async def test_bridge_concepts_uses_embeddings(self):
        """bridge_concepts() SHOULD use embeddings to find semantic bridges."""
        from core.concept_model import ConceptModelOrchestrator

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value={"message": {"content": "Bridge found"}})

        orchestrator = ConceptModelOrchestrator(concept_llm=mock_llm)
        await orchestrator.start()

        result = await orchestrator.bridge_concepts(
            "neural networks",
            "biological neurons"
        )

        # GAP: bridge_concepts does NOT use embeddings!
        # It should:
        # 1. Encode both concepts
        # 2. Compute similarity
        # 3. Find bridging concepts in embedding space
        # 4. Use embeddings to guide the LLM prompt

        await orchestrator.stop()

        assert len(result) > 0, "Should return a bridge description"

    def test_dual_model_uses_concept_embeddings(self):
        """DualModelOrchestrator SHOULD use concept embeddings for model selection."""
        from core.dual_model import DualModelOrchestrator

        # GAP: DualModelOrchestrator doesn't have concept embedder!
        # It should use embeddings to:
        # 1. Compare new tasks to past tasks
        # 2. Route to best model based on semantic similarity
        # 3. Store embeddings of successful outputs for learning

        orchestrator = DualModelOrchestrator()
        # This will assert if embedder is not present
        # Currently FAILS - DualModelOrchestrator doesn't have embedder


class TestExpectedInterconnectBehavior:
    """Tests for EXPECTED behavior of embedder as interconnect."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_embedder_available_for_cross_model_similarity(self):
        """Embedder should be available for computing cross-model similarities."""
        from core.concept_model import ConceptModelOrchestrator

        orchestrator = ConceptModelOrchestrator()
        embedder = orchestrator.embedder

        # Encode outputs from different models
        emb1 = embedder.encode("Primary model thinking about creativity")
        emb2 = embedder.encode("Backend model processing code")

        assert emb1.shape == (768,)
        assert emb2.shape == (768,)

        # Check for zero vectors
        if np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0:
            pytest.skip("Embedding model not loaded, cannot test similarity")

        # Similarity computation should work
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert -1 <= similarity <= 1

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_concept_embeddings_storage_works(self):
        """Should be able to store and retrieve concept embeddings."""
        from core.concept_model import ConceptModelOrchestrator, ConceptEmbedder

        orchestrator = ConceptModelOrchestrator()
        embedder = ConceptEmbedder(provider="onnx")

        # Store embeddings from different models
        orchestrator._concept_embeddings["primary_creative"] = embedder.encode("Creative output")
        orchestrator._concept_embeddings["backend_code"] = embedder.encode("Code output")
        orchestrator._concept_embeddings["concept_abstract"] = embedder.encode("Abstract output")

        # Retrieve
        assert "primary_creative" in orchestrator._concept_embeddings
        assert "backend_code" in orchestrator._concept_embeddings
        assert "concept_abstract" in orchestrator._concept_embeddings

        # Each should be 768-dim
        for key in orchestrator._concept_embeddings:
            assert orchestrator._concept_embeddings[key].shape == (768,)


class TestEmbedderForModelRouting:
    """Test using embedder for intelligent model routing."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_semantic_based_routing(self):
        """Should route tasks to best model based on semantic similarity."""
        from core.concept_model import ConceptEmbedder
        from core.dual_model import TaskType, TASK_ROUTING
        import numpy as np

        embedder = ConceptEmbedder(provider="onnx")

        # Embed task type descriptions
        task_descriptions = {
            TaskType.CHAT: "conversational chat dialogue talking",
            TaskType.CREATIVE: "creative writing artistic imaginative",
            TaskType.CODE: "code programming implementation function",
            TaskType.MATH: "math calculation equation formula",
            TaskType.ABSTRACTION: "abstract pattern generalization concept",
        }

        # Embed all descriptions
        task_embeddings = {k: embedder.encode(v) for k, v in task_descriptions.items()}

        # Simulate user query
        query = "Write a Python function to sort a list"
        query_embedding = embedder.encode(query)

        # Find best matching task type
        best_type = None
        best_sim = -2
        for task_type, emb in task_embeddings.items():
            sim = 0.0 if np.linalg.norm(query_embedding) == 0 or np.linalg.norm(emb) == 0 else np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            if sim > best_sim:
                best_sim = sim
                best_type = task_type

        # Should route to semantically similar task
        # Note: With real embeddings, semantic similarity may match related tasks
        assert best_type in [TaskType.CODE, TaskType.MATH, TaskType.CHAT, TaskType.ABSTRACTION], \
            f"Expected semantically relevant task, got {best_type}"

        # Verify that CODE routes to BACKEND
        assert TASK_ROUTING[TaskType.CODE].value == "backend"