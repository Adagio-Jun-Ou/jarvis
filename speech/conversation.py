"""Conversational Speech Model for end-to-end speech-to-speech conversation.

This module implements a speech-to-speech conversation model that:
1. Encodes input speech to semantic + acoustic tokens
2. Generates response tokens using the LLM
3. Decodes tokens back to speech with voice preservation
"""

import asyncio
import logging
import numpy as np
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, AsyncIterator
import json

from .tokenizer import SpeechTokenizer, SpeechTokens
from .codec import CodecConfig
from .vocoder import NeuralVocoder, VocoderConfig

logger = logging.getLogger(__name__)

# Strong references to in-flight background tasks. asyncio only holds a weak
# reference to a running task, so without this a task can be garbage-collected
# mid-flight and its work silently dropped.
_background_tasks: set[asyncio.Task] = set()


def _spawn_tracked(coro, name: str) -> asyncio.Task:
    """Schedule a coroutine on the running loop, keeping a strong reference."""
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


@dataclass
class ConversationConfig:
    """Configuration for conversational speech model."""
    # Audio settings
    sample_rate: int = 16000

    # Token settings
    semantic_vocab_size: int = 500
    acoustic_vocab_size: int = 1024
    num_codebooks: int = 8

    # Conversation settings
    max_context_turns: int = 10
    voice_preservation: bool = True
    response_style: str = "natural"  # natural, formal, casual

    # Speaker settings
    default_speaker: str = "jarvis"
    speaker_embedding_dim: int = 256


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    speaker: str  # "user" or "assistant"
    text: str
    audio_path: Optional[str] = None
    tokens: Optional[SpeechTokens] = None
    speaker_embedding: Optional[np.ndarray] = None


class SpeakerManager:
    """Manages speaker embeddings and voice profiles."""

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "speakers"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.speakers: dict[str, np.ndarray] = {}
        self._load_speakers()

    def _load_speakers(self) -> None:
        """Load saved speaker profiles."""
        profile_file = self.storage_path / "profiles.json"
        if profile_file.exists():
            with open(profile_file, encoding='utf-8') as f:
                data = json.load(f)
                for name in data.get("speakers", []):
                    if not name.replace("_", "").replace("-", "").replace(" ", "").isalnum():
                        continue
                    embedding_path = self.storage_path / f"{name}.npy"
                    if embedding_path.exists():
                        try:
                            resolved = embedding_path.resolve()
                            if not str(resolved).startswith(str(self.storage_path.resolve())):
                                continue
                        except (OSError, ValueError):
                            continue
                        self.speakers[name] = np.load(resolved)

    def save_speaker(self, name: str, embedding: np.ndarray) -> None:
        """Save a speaker profile."""
        if not name.replace("_", "").replace("-", "").replace(" ", "").isalnum():
            raise ValueError(f"Invalid speaker name: {name}")
        self.speakers[name] = embedding
        np.save(self.storage_path / f"{name}.npy", embedding)

        profile_file = self.storage_path / "profiles.json"
        with open(profile_file, "w", encoding='utf-8') as f:
            json.dump({"speakers": list(self.speakers.keys())}, f)

    def get_speaker(self, name: str) -> Optional[np.ndarray]:
        """Get a speaker embedding by name."""
        return self.speakers.get(name)

    def extract_speaker_embedding(self, tokens: SpeechTokens) -> np.ndarray:
        """Extract speaker embedding from acoustic tokens."""
        # Use the mean of acoustic tokens as a simple embedding
        # In production, use a dedicated speaker encoder
        acoustic = tokens.acoustic
        embedding = np.mean(acoustic, axis=-1).flatten()

        # Pad or truncate to fixed size
        target_size = 256
        if len(embedding) < target_size:
            embedding = np.pad(embedding, (0, target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]

        return embedding


class ConversationalSpeechModel:
    """End-to-end conversational speech model.

    This model enables natural speech-to-speech conversation by:
    1. Understanding spoken input (speech-to-semantic-tokens)
    2. Generating response (LLM in token space)
    3. Synthesizing speech (tokens-to-speech with voice)
    """

    def __init__(
        self,
        config: Optional[ConversationConfig] = None,
        llm_client=None,
        tokenizer: Optional[SpeechTokenizer] = None,
        vocoder: Optional[NeuralVocoder] = None,
    ):
        self.config = config or ConversationConfig()
        self.llm_client = llm_client
        self.tokenizer = tokenizer or SpeechTokenizer(
            CodecConfig(
                sample_rate=self.config.sample_rate,
                semantic_vocab_size=self.config.semantic_vocab_size,
                acoustic_vocab_size=self.config.acoustic_vocab_size,
                num_codebooks=self.config.num_codebooks,
            )
        )
        self.vocoder = vocoder or NeuralVocoder(
            VocoderConfig(sample_rate=self.config.sample_rate)
        )
        self.speaker_manager = SpeakerManager()

        self.conversation_history: deque[ConversationTurn] = deque(maxlen=500)
        self.current_speaker_embedding: Optional[np.ndarray] = None
        self._loaded = False

    async def load(self) -> bool:
        """Load all model components."""
        tokenizer_loaded = await self.tokenizer.load()
        vocoder_loaded = await self.vocoder.load()
        self._loaded = tokenizer_loaded and vocoder_loaded

        # Load default Jarvis voice
        jarvis_embedding = self.speaker_manager.get_speaker("jarvis")
        if jarvis_embedding is None:
            self.current_speaker_embedding = np.random.randn(256).astype(np.float32)
            norm = np.linalg.norm(self.current_speaker_embedding)
            if norm > 1e-8:
                self.current_speaker_embedding = self.current_speaker_embedding / norm
        else:
            self.current_speaker_embedding = jarvis_embedding

        return self._loaded

    async def process_speech(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> tuple[str, SpeechTokens]:
        """Process input speech to text and tokens.

        Args:
            audio: Input audio waveform
            sample_rate: Sample rate of audio

        Returns:
            Tuple of (transcribed text, speech tokens)
        """
        if not self._loaded:
            await self.load()

        # Encode speech to tokens
        tokens = await self.tokenizer.encode(audio, sample_rate)

        # Extract speaker embedding for voice preservation
        if self.config.voice_preservation:
            self.current_speaker_embedding = self.speaker_manager.extract_speaker_embedding(
                tokens
            )

        # Transcribe using LLM or ASR
        # For now, use a simple approach - in production, use Whisper or similar
        text = await self._transcribe(tokens)

        return text, tokens

    async def _transcribe(self, tokens: SpeechTokens) -> str:
        """Transcribe speech tokens to text."""
        # This would use a speech-to-text model in production
        # For now, return a placeholder
        if self.llm_client is not None:
            # Use the LLM to help with transcription context
            pass
        return "[Transcribed speech]"

    async def generate_response(
        self,
        user_text: str,
        user_tokens: Optional[SpeechTokens] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a text response using the LLM.

        Args:
            user_text: Transcribed user speech
            user_tokens: Optional speech tokens for context
            system_prompt: Optional system prompt override

        Returns:
            Generated response text
        """
        if self.llm_client is None:
            return "I'm here to help. Could you please repeat that?"

        # Build conversation context
        context = self._build_context()

        # Generate response
        response = await self.llm_client.chat(
            user_input=user_text,
            system_prompt=system_prompt or self._get_system_prompt(),
            conversation_history=context,
        )

        return response.get("message", {}).get("content", "")

    def _get_system_prompt(self) -> str:
        """Get the system prompt for conversation."""
        style_prompts = {
            "natural": "Respond naturally and conversationally.",
            "formal": "Respond in a formal, professional manner.",
            "casual": "Respond in a friendly, casual style.",
        }

        return f"""You are Jarvis, an advanced AI assistant with voice capabilities.
You engage in natural, flowing conversations with users.

{style_prompts.get(self.config.response_style, style_prompts['natural'])}

Keep responses concise and engaging. Use natural speech patterns including
occasional conversational fillers like "hmm", "let me think", etc.

You have access to tools and can perform various tasks through speech."""

    def _build_context(self) -> list[dict]:
        """Build conversation context from history."""
        context = []
        for turn in list(self.conversation_history)[-self.config.max_context_turns:]:
            context.append({
                "role": turn.speaker,
                "content": turn.text,
            })
        return context

    async def synthesize_speech(
        self,
        text: str,
        speaker_embedding: Optional[np.ndarray] = None,
        style: Optional[str] = None
    ) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            speaker_embedding: Optional speaker embedding for voice control
            style: Optional style override

        Returns:
            Audio waveform
        """
        if not self._loaded:
            await self.load()

        # Use provided embedding or current speaker
        embedding = speaker_embedding or self.current_speaker_embedding

        # Generate tokens from text (would use TTS model in production)
        tokens = await self._text_to_tokens(text, embedding)

        # Decode tokens to audio
        audio = await self.tokenizer.decode(tokens, embedding)

        return audio

    async def _text_to_tokens(
        self,
        text: str,
        speaker_embedding: np.ndarray
    ) -> SpeechTokens:
        """Convert text to speech tokens.

        This is a simplified implementation. In production, this would use:
        1. A text-to-semantic model (like Bark, VALL-E)
        2. A semantic-to-acoustic model conditioned on speaker embedding
        """
        # Estimate token counts based on text length
        # Roughly 10 phonemes per second at 50 semantic tokens/sec
        num_semantic = max(1, len(text.split()) * 4)  # ~4 tokens per word

        # Generate semantic tokens (would come from TTS model)
        # For now, create placeholder tokens
        semantic = np.random.randint(
            0, self.config.semantic_vocab_size, size=num_semantic
        ).astype(np.int32)

        # Generate acoustic tokens with speaker conditioning
        # In production, this would be conditioned on speaker_embedding
        num_frames = int(num_semantic * 1.5)  # More acoustic frames
        acoustic = np.random.randint(
            0, self.config.acoustic_vocab_size,
            size=(self.config.num_codebooks, num_frames)
        ).astype(np.int32)

        return SpeechTokens(
            semantic=semantic,
            acoustic=acoustic,
            duration=num_semantic / 50.0,  # 50 semantic tokens per second
            sample_rate=self.config.sample_rate
        )

    async def conversation_turn(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        callback: Optional[Callable[[str, np.ndarray], None]] = None
    ) -> tuple[str, np.ndarray]:
        """Process a complete conversation turn.

        Args:
            audio: Input audio from user
            sample_rate: Sample rate of input audio
            callback: Optional callback for streaming responses

        Returns:
            Tuple of (response text, response audio)
        """
        # Process input speech
        user_text, user_tokens = await self.process_speech(audio, sample_rate)

        # Store user turn
        self.conversation_history.append(ConversationTurn(
            speaker="user",
            text=user_text,
            tokens=user_tokens,
        ))

        # Generate response
        response_text = await self.generate_response(user_text, user_tokens)

        # Synthesize response speech
        response_audio = await self.synthesize_speech(response_text)

        # Store assistant turn
        self.conversation_history.append(ConversationTurn(
            speaker="assistant",
            text=response_text,
        ))

        # Callback for streaming
        if callback:
            callback(response_text, response_audio)

        return response_text, response_audio

    async def stream_conversation(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        sample_rate: int = 16000,
    ) -> AsyncIterator[tuple[str, np.ndarray]]:
        """Stream conversation with continuous audio input/output.

        Args:
            audio_stream: Async iterator of audio chunks
            sample_rate: Sample rate of input audio

        Yields:
            Tuples of (text chunk, audio chunk)
        """
        audio_buffer = []

        async for chunk in audio_stream:
            audio_buffer.append(chunk)

            # Detect end of speech (simplified)
            # In production, use VAD
            if len(np.concatenate(audio_buffer)) > sample_rate * 2:  # 2 seconds
                full_audio = np.concatenate(audio_buffer)
                audio_buffer = []

                # Process turn
                text, audio = await self.conversation_turn(full_audio, sample_rate)

                yield text, audio

    def set_voice(self, speaker_name: str) -> bool:
        """Set the current voice by speaker name.

        Args:
            speaker_name: Name of the speaker profile to use

        Returns:
            True if voice was set successfully
        """
        embedding = self.speaker_manager.get_speaker(speaker_name)
        if embedding is not None:
            self.current_speaker_embedding = embedding
            return True
        return False

    async def learn_voice_async(self, audio: np.ndarray, name: str, sample_rate: int = 16000) -> None:
        """Learn a voice profile from audio samples (async version)."""
        tokens = await self.tokenizer.encode(audio, sample_rate)
        embedding = self.speaker_manager.extract_speaker_embedding(tokens)
        self.speaker_manager.save_speaker(name, embedding)

    def learn_voice(self, audio: np.ndarray, name: str, sample_rate: int = 16000) -> None:
        """Learn a voice profile from audio samples (sync wrapper).

        Prefer learn_voice_async() when called from async contexts.
        """
        try:
            asyncio.get_running_loop()
            _spawn_tracked(
                self.learn_voice_async(audio, name, sample_rate),
                name=f"learn_voice:{name}",
            )
            logger.warning("learn_voice() called from async context — use learn_voice_async() instead")
        except RuntimeError:
            tokens = asyncio.run(self.tokenizer.encode(audio, sample_rate))
            embedding = self.speaker_manager.extract_speaker_embedding(tokens)
            self.speaker_manager.save_speaker(name, embedding)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()

    def save_conversation(self, path: str) -> None:
        """Save conversation history to file."""
        data = [
            {
                "speaker": turn.speaker,
                "text": turn.text,
            }
            for turn in self.conversation_history
        ]
        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load_conversation(self, path: str) -> None:
        """Load conversation history from file."""
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        self.conversation_history = [
            ConversationTurn(speaker=t["speaker"], text=t["text"])
            for t in data
        ]
