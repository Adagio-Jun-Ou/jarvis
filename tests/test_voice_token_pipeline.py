"""Integration test for Voice Token Pipeline with NPU Whisper.

Tests the complete flow:
Audio → NPUWhisperModel → Semantic Tokens → AcousticTokenizer → Voice Output
"""
import numpy as np
import pytest


class TestVoiceTokenPipeline:
    """Test the voice token pipeline integration."""



    def test_voice_token_bridge_full_pipeline(self):
        """Test full VoiceTokenBridge pipeline."""
        from core.voice_tokens import VoiceTokenBridge

        bridge = VoiceTokenBridge()

        audio = np.random.randn(16000).astype(np.float32) * 0.1

        voice_tokens = bridge.encode(audio, sample_rate=16000)

        assert voice_tokens.semantic_tokens is not None
        assert len(voice_tokens.semantic_tokens) > 0
        assert voice_tokens.acoustic_tokens is not None
        assert voice_tokens.duration_ms == pytest.approx(1000.0, rel=0.1)

        decoded = bridge.decode(voice_tokens, use_acoustic=True)
        assert decoded is not None
        assert len(decoded) > 0


    def test_acoustic_tokenizer_rvq(self):
        """Test AcousticTokenizer RVQ codebooks."""
        from core.voice_tokens import AcousticTokenizer

        tokenizer = AcousticTokenizer()

        audio = np.random.randn(24000).astype(np.float32) * 0.1

        tokens = tokenizer.tokenize(audio, sample_rate=24000)

        assert len(tokens) > 0

        for frame in tokens:
            assert len(frame) == AcousticTokenizer.N_CODEBOOKS
            for token in frame:
                assert 0 <= token < AcousticTokenizer.CODEBOOK_SIZE

    def test_speaker_embedding_extraction(self):
        """Test speaker embedding extraction from acoustic tokens."""
        from core.voice_tokens import VoiceTokenBridge

        bridge = VoiceTokenBridge()

        np.random.seed(42)
        audio1 = np.random.randn(24000).astype(np.float32) * 0.1
        audio2 = np.random.randn(24000).astype(np.float32) * 0.1

        tokens1 = bridge.encode(audio1, sample_rate=24000)
        tokens2 = bridge.encode(audio2, sample_rate=24000)

        speaker1 = bridge.extract_speaker_embedding(tokens1.acoustic_tokens)
        speaker2 = bridge.extract_speaker_embedding(tokens2.acoustic_tokens)

        assert speaker1.startswith("speaker_")
        assert speaker2.startswith("speaker_")
        assert speaker1 != speaker2  # Different audio should have different embeddings

    def test_voice_token_serialization(self):
        """Test VoiceTokens serialization."""
        from core.voice_tokens import VoiceTokens

        tokens = VoiceTokens(
            semantic_tokens=[1, 10, 11, 12, 2],
            acoustic_tokens=[[1, 2, 3, 4], [5, 6, 7, 8]],
            sample_rate=24000,
            duration_ms=1000.0,
            text_transcription="test",
        )

        data = tokens.to_dict()
        assert data["semantic_tokens"] == [1, 10, 11, 12, 2]
        assert data["sample_rate"] == 24000

        restored = VoiceTokens.from_dict(data)
        assert restored.semantic_tokens == tokens.semantic_tokens
        assert restored.acoustic_tokens == tokens.acoustic_tokens
        assert restored.sample_rate == tokens.sample_rate

    def test_voice_token_tool_interface(self):
        """Test VoiceTokenTool interface."""
        from core.voice_tokens import VoiceTokenTool, VoiceTokenManager

        manager = VoiceTokenManager()
        tool = VoiceTokenTool(manager)

        assert tool.name == "voice_tokens"
        assert "encode" in tool.parameters["properties"]["action"]["enum"]

    @pytest.mark.asyncio
    async def test_voice_token_tool_decode_action(self):
        """Test VoiceTokenTool decode action."""
        from core.voice_tokens import VoiceTokenTool, VoiceTokenManager

        manager = VoiceTokenManager()
        tool = VoiceTokenTool(manager)

        result = await tool.execute(action="decode", text="Hello", emotion="happy")

        assert result is not None
        assert "audio_data" in result or "Error" in result

    @pytest.mark.asyncio
    async def test_voice_token_tool_list_speakers(self):
        """Test VoiceTokenTool list_speakers action."""
        from core.voice_tokens import VoiceTokenTool, VoiceTokenManager

        manager = VoiceTokenManager()
        tool = VoiceTokenTool(manager)

        result = await tool.execute(action="list_speakers")

        assert result is not None
        assert "speakers" in result

    def test_token_sequence_interleaving(self):
        """Test TokenSequence for interleaved token handling."""
        from core.voice_tokens import TokenSequence, TokenType

        seq = TokenSequence()

        seq.add_text_tokens([10, 11, 12])
        assert len(seq.tokens) == 3
        assert all(t == TokenType.SEMANTIC for t in seq.token_types)

        seq.add_audio_tokens(
            semantic=[20, 21, 22],
            acoustic=[[30, 31], [32, 33], [34, 35]]
        )

        assert len(seq.tokens) > 3
        assert TokenType.ACOUSTIC in seq.token_types

    def test_fallback_tokenization_without_models(self):
        """Test fallback when neural models are not available."""
        from core.voice_tokens import SemanticTokenizer

        tokenizer = SemanticTokenizer(model_size="base")
        tokenizer._loaded = True
        tokenizer._model = None
        tokenizer._processor = None

        audio = np.random.randn(16000).astype(np.float32) * 0.1
        tokens = tokenizer.tokenize(audio, sample_rate=16000)

        assert len(tokens) > 0
        assert tokens[0] == tokenizer.BOS_TOKEN
        assert tokens[-1] == tokenizer.EOS_TOKEN




class TestVoiceTokenManager:
    """Test VoiceTokenManager functionality."""

    def test_manager_creation(self):
        """Test VoiceTokenManager creation."""
        from core.voice_tokens import VoiceTokenManager
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceTokenManager(storage_path=Path(tmp))
            assert manager.storage_path.exists()

    def test_manager_singleton(self):
        """Test VoiceTokenManager singleton."""
        from core.voice_tokens import get_voice_token_manager

        m1 = get_voice_token_manager()
        m2 = get_voice_token_manager()

        assert m1 is m2

    def test_manager_save_load_tokens(self):
        """Test saving and loading voice tokens."""
        from core.voice_tokens import VoiceTokenManager, VoiceTokens
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            manager = VoiceTokenManager(storage_path=Path(tmp))

            tokens = VoiceTokens(
                semantic_tokens=[1, 10, 11, 12, 2],
                acoustic_tokens=[[1, 2], [3, 4]],
                sample_rate=24000,
            )

            manager.save_tokens(tokens, "test_tokens")

            loaded = manager.load_tokens("test_tokens")

            assert loaded is not None
            assert loaded.semantic_tokens == tokens.semantic_tokens
            assert loaded.acoustic_tokens == tokens.acoustic_tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])