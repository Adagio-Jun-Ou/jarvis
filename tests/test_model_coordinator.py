"""Comprehensive tests for Model Coordinator and Inter-Model Communication.

Tests the connective layer between Kimi K2.5 and GLM-5 that enables
models to communicate THROUGH the Jarvis stack, not over it.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.model_coordinator import (
    ModelCoordinator,
    ModelRole,
    HandoffReason,
    MessageType,
    ModelConfig,
    InterModelMessage,
    ReasoningChain,
    SharedContext,
    get_model_coordinator,
    handoff_to_backend,
)


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_model_config_creation(self):
        """Test creating a model configuration."""
        config = ModelConfig(
            model_id="test-model",
            role=ModelRole.PRIMARY,
            capabilities=["chat", "reasoning"],
            context_window=128000,
        )
        
        assert config.model_id == "test-model"
        assert config.role == ModelRole.PRIMARY
        assert "chat" in config.capabilities
        assert config.context_window == 128000
        assert config.supports_thinking is False
        assert config.supports_vision is False
    
    def test_model_config_with_all_fields(self):
        """Test model config with all fields."""
        config = ModelConfig(
            model_id="vision-model",
            role=ModelRole.VISION,
            capabilities=["vision", "image"],
            context_window=32768,
            supports_thinking=True,
            supports_vision=True,
            preferred_tasks=["ocr", "image_analysis"],
            temperature=0.5,
            max_tokens=2048,
        )
        
        assert config.supports_vision is True
        assert config.supports_thinking is True
        assert "ocr" in config.preferred_tasks


class TestSharedContext:
    """Test SharedContext for inter-model communication."""
    
    def test_shared_context_set_get(self):
        """Test setting and getting values in shared context."""
        context = SharedContext()
        
        context.set("user_name", "Alice", model_id="kimi-k2.5")
        context.set("task_type", "code_review", model_id="glm-5")
        
        assert context.get("user_name") == "Alice"
        assert context.get("task_type") == "code_review"
        assert context.get("nonexistent") is None
    
    def test_shared_context_get_all(self):
        """Test getting all shared context."""
        context = SharedContext()
        
        context.set("key1", "value1")
        context.set("key2", "value2")
        
        all_context = context.get_all()
        assert all_context["key1"] == "value1"
        assert all_context["key2"] == "value2"
    
    def test_shared_context_message_log(self):
        """Test message logging in shared context."""
        context = SharedContext()
        
        msg = InterModelMessage(
            id="test_msg_1",
            message_type=MessageType.QUERY,
            from_model="kimi-k2.5",
            to_model="glm-5",
            context={},
            query="What is 2+2?",
        )
        
        context.log_message(msg)
        
        messages = context.get_messages(limit=10)
        assert len(messages) == 1
        assert messages[0]["from_model"] == "kimi-k2.5"
        assert messages[0]["to_model"] == "glm-5"
    
    def test_reasoning_chain(self):
        """Test reasoning chain for collaborative thinking."""
        context = SharedContext()
        
        chain = context.create_reasoning_chain("math problem")
        
        chain.add_turn("kimi-k2.5", "Hmm, let me think...", "I think it's 4")
        chain.add_turn("glm-5", "Checking...", "Confirmed, 2+2=4")
        
        assert len(chain.turns) == 2
        assert chain.turns[0]["model"] == "kimi-k2.5"
        assert chain.turns[1]["model"] == "glm-5"


class TestInterModelMessage:
    """Test InterModelMessage dataclass."""
    
    def test_message_creation(self):
        """Test creating an inter-model message."""
        msg = InterModelMessage(
            id="msg_123",
            message_type=MessageType.HANDOFF,
            from_model="kimi-k2.5",
            to_model="glm-5",
            context={"topic": "coding"},
            query="Please review this code",
            handoff_reason=HandoffReason.SPECIALIZATION,
        )
        
        assert msg.id == "msg_123"
        assert msg.message_type == MessageType.HANDOFF
        assert msg.from_model == "kimi-k2.5"
        assert msg.to_model == "glm-5"
        assert msg.handoff_reason == HandoffReason.SPECIALIZATION
    
    def test_message_to_dict(self):
        """Test serializing message to dictionary."""
        msg = InterModelMessage(
            id="msg_456",
            message_type=MessageType.QUERY,
            from_model="glm-5",
            to_model="kimi-k2.5",
            context={},
            query="Test query",
            response="Test response",
            response_from="kimi-k2.5",
        )
        
        data = msg.to_dict()
        
        assert data["id"] == "msg_456"
        assert data["message_type"] == "query"
        assert data["response"] == "Test response"


class TestModelCoordinator:
    """Test ModelCoordinator class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM client with a concrete (string) model id."""
        llm = AsyncMock()
        llm.model = "test-primary-model"
        llm.chat = AsyncMock(return_value={"message": {"content": "Test response"}})
        return llm

    @pytest.fixture
    def coordinator(self, mock_llm):
        """Create a ModelCoordinator with mock LLMs.

        Same client for both roles -> both ids resolve to the mock's model.
        """
        return ModelCoordinator(
            primary_llm=mock_llm,
            backend_llm=mock_llm,
        )

    def test_coordinator_initialization(self, coordinator):
        """Test coordinator resolves live model ids from the clients."""
        assert coordinator.primary_model_id == "test-primary-model"
        # Same client instance for both roles -> backend follows primary
        assert coordinator.backend_model_id == "test-primary-model"
        assert coordinator.get_active_model() == "test-primary-model"
        # Template ids are replaced, not left behind
        assert "kimi-k2.5" not in coordinator.models
        assert "glm-5" not in coordinator.models
        assert "test-primary-model" in coordinator.models

    def test_get_model_for_task(self, coordinator):
        """Test task routing to appropriate models."""
        chat_model = coordinator.get_model_for_task("chat")
        code_model = coordinator.get_model_for_task("code")
        vision_model = coordinator.get_model_for_task("vision")

        assert chat_model in ["test-primary-model", "qwen3.5:397b"]
        assert code_model in ["test-primary-model"]
        # Vision routes to primary
        assert vision_model == "test-primary-model"

    def test_get_llm_for_model(self, coordinator, mock_llm):
        """Test getting LLM client for a model."""
        llm = coordinator.get_llm_for_model("test-primary-model")
        assert llm is mock_llm
    
    def test_set_shared_context(self, coordinator):
        """Test setting shared context through coordinator."""
        coordinator.set_shared_context("test_key", "test_value", "system")
        
        value = coordinator.get_shared_context("test_key")
        assert value == "test_value"
    
    def test_register_model(self, coordinator):
        """Test registering a new model."""
        new_config = ModelConfig(
            model_id="new-model",
            role=ModelRole.SPECIALIZED,
            capabilities=["custom"],
            context_window=64000,
        )
        
        coordinator.register_model(new_config)
        
        assert "new-model" in coordinator.models
        assert coordinator.models["new-model"].role == ModelRole.SPECIALIZED
    
    def test_get_stats(self, coordinator):
        """Test getting coordinator statistics."""
        stats = coordinator.get_stats()
        
        assert "active_model" in stats
        assert "available_models" in stats
        assert "total_handoffs" in stats
        assert "cross_model_calls" in stats
    
    @pytest.mark.asyncio
    async def test_handoff(self, coordinator):
        """Test model handoff."""
        message = await coordinator.handoff(
            to_model=coordinator.backend_model_id,
            context={"task": "code review"},
            query="Review this code",
            reason=HandoffReason.SPECIALIZATION,
        )

        assert message.from_model == coordinator.primary_model_id
        assert message.to_model == coordinator.backend_model_id
        assert message.handoff_reason == HandoffReason.SPECIALIZATION
        assert coordinator.get_active_model() == coordinator.backend_model_id
    
    @pytest.mark.asyncio
    async def test_query_other_model(self, coordinator, mock_llm):
        """Test querying another model."""
        response = await coordinator.query_other_model(
            target_model=coordinator.backend_model_id,
            query="What is 2+2?",
            context={"mode": "math"},
        )

        assert "Test response" in response
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_think_together(self, coordinator, mock_llm):
        """Test collaborative thinking across models."""
        chain = await coordinator.think_together(
            topic="math problem",
            initial_query="What is 2*3?",
            models=[coordinator.primary_model_id],
            rounds=1,
        )

        assert chain.topic == "math problem"
        assert len(chain.turns) >= 1
        assert chain.complete is True


class TestHandoffReasons:
    """Test different handoff scenarios."""
    
    def test_handoff_reasons(self):
        """Test all handoff reason types."""
        reasons = [
            HandoffReason.CAPABILITY,
            HandoffReason.SPECIALIZATION,
            HandoffReason.LOAD_BALANCE,
            HandoffReason.FALLBACK,
            HandoffReason.COLLABORATION,
            HandoffReason.USER_REQUEST,
        ]
        
        for reason in reasons:
            assert reason.value in [
                "capability", 
                "specialization", 
                "load_balance",
                "fallback",
                "collaboration",
                "user_request",
            ]


class TestMessageTypes:
    """Test inter-model message types."""
    
    def test_message_types(self):
        """Test all message types."""
        types = [
            MessageType.HANDOFF,
            MessageType.QUERY,
            MessageType.SYNTHESIZE,
            MessageType.VERIFY,
            MessageType.REFINE,
            MessageType.THINK_TOGETHER,
        ]
        
        for msg_type in types:
            assert isinstance(msg_type.value, str)


class TestIntegration:
    """Integration tests for model coordinator."""
    
    @pytest.fixture
    def coordinator(self):
        """Create coordinator with real shared context."""
        return ModelCoordinator()
    
    def test_shared_context_persistence(self, coordinator):
        """Test that shared context persists across operations."""
        coordinator.set_shared_context("persistent_key", "persistent_value")
        
        # Perform some operations
        stats = coordinator.get_stats()
        
        # Context should still be there
        assert coordinator.get_shared_context("persistent_key") == "persistent_value"
    
    def test_reasoning_chain_lifecycle(self, coordinator):
        """Test full reasoning chain lifecycle."""
        # Create chain
        chain = coordinator.shared_context.create_reasoning_chain("test topic")
        
        # Add turns
        chain.add_turn("model_a", "thinking", "output")
        chain.add_turn("model_b", "thinking 2", "output 2")
        
        # Complete and synthesize
        chain.complete = True
        chain.synthesis = "Final synthesis"
        
        # Retrieve
        retrieved = coordinator.shared_context.get_reasoning_chain(chain.id)
        assert retrieved is not None
        assert len(retrieved.turns) == 2


class TestGlobalCoordinator:
    """Test global coordinator singleton."""
    
    def test_get_model_coordinator_singleton(self):
        """Test that get_model_coordinator returns singleton."""
        coordinator1 = get_model_coordinator(primary_llm=MagicMock())
        coordinator2 = get_model_coordinator()
        assert coordinator1 is coordinator2


class TestConceptModel:
    """Test concept model integration - now uses dual-model architecture."""
    
    def test_model_role_concept_removed(self):
        """Test that CONCEPT role no longer exists in ModelRole enum (dual-model)."""
        assert not hasattr(ModelRole, 'CONCEPT')
    
    def test_concept_tasks_route_correctly(self):
        """Test that concept tasks route correctly between primary and backend."""
        coordinator = ModelCoordinator()

        # Abstraction, meta_reasoning, pattern go to primary
        assert coordinator.get_model_for_task("abstraction") == coordinator.primary_model_id
        assert coordinator.get_model_for_task("meta_reasoning") == coordinator.primary_model_id
        assert coordinator.get_model_for_task("pattern") == coordinator.primary_model_id
        # Synthesis goes to backend
        assert coordinator.get_model_for_task("synthesis") == coordinator.backend_model_id
        # Vision goes to primary
        assert coordinator.get_model_for_task("vision") == coordinator.primary_model_id

    def test_get_dual_model_status(self):
        """Test dual-model status reporting."""
        coordinator = ModelCoordinator()
        status = coordinator.get_dual_model_status()

        assert "enabled" in status
        assert "primary" in status
        assert "backend" in status
        assert status["backend"]["model"] == coordinator.backend_model_id
        assert "synthesis" in status["backend"]["tasks"]
        assert "abstraction" in status["primary"]["tasks"]
        assert "vision" in status["primary"]["tasks"]
    
    def test_primary_handles_meta_reasoning(self):
        """Test that primary LLM handles meta-reasoning tasks."""
        mock_primary = MagicMock()
        mock_primary.model = "primary-x"
        coordinator = ModelCoordinator(primary_llm=mock_primary)

        llm = coordinator.get_llm_for_model(coordinator.primary_model_id)
        assert llm == mock_primary


class TestDualModelStatus:
    """Test dual-model architecture status."""
    
    def test_connected_models_primary_and_backend(self):
        """Test that connected models list includes primary and backend."""
        mock_primary = MagicMock()
        mock_primary.model = "primary-x"
        mock_backend = MagicMock()
        mock_backend.model = "backend-y"

        coordinator = ModelCoordinator(
            primary_llm=mock_primary,
            backend_llm=mock_backend,
        )

        connected = coordinator.get_connected_models()
        assert coordinator.backend_model_id == "backend-y"
        assert "backend-y" in connected
    
    def test_dual_model_status_has_required_keys(self):
        """Test that get_dual_model_status has required keys."""
        coordinator = ModelCoordinator()
        status = coordinator.get_dual_model_status()
        
        assert "enabled" in status
        assert "primary" in status
        assert "backend" in status
        assert "routing" in status
        assert "collaboration" in status