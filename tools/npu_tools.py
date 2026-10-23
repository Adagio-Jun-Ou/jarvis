"""NPU-powered tools for Jarvis - fast embeddings, voice processing, and file retrieval."""
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from .base import Tool

logger = logging.getLogger(__name__)


class EmbeddingTool(Tool):
    """Tool for fast text embeddings."""

    name = "embed"
    description = "Generate text embeddings for semantic search or similarity. Fast CPU-based encoding."
    parameters = {
        "type": "object",
        "properties": {
            "texts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of texts to embed",
            },
            "model": {
                "type": "string",
                "enum": ["mini", "small", "bge"],
                "description": "Model: mini (fastest, 384d), small (better, 768d), bge (retrieval, 384d)",
            },
        },
        "required": ["texts"],
    }
    requires_permission = False

    def __init__(self, model_type: str = "mini"):
        self.model_type = model_type
        self._model = None

    def _get_model(self):
        if self._model is None:
            from core.npu_models import get_embedding
            self._model = get_embedding(self.model_type)
        return self._model

    async def execute(self, texts: list[str], model: Optional[str] = None) -> str:
        m = self._get_model()
        if model:
            from core.npu_models import FastEmbeddingModel
            m = FastEmbeddingModel(model)

        results = m.encode(texts)

        output = ["## Embeddings\n"]
        for r in results:
            output.append(f"- \"{r.text[:40]}...\"" if len(r.text) > 40 else f"- \"{r.text}\"")
            output.append(f"  Dims: {len(r.embedding)}, Tokens: {r.tokens}, Time: {r.inference_ms:.1f}ms")

        return "\n".join(output)


class FileIndexTool(Tool):
    """Tool for indexing files with embeddings."""

    name = "index_files"
    description = "Index files or directories for semantic search. Chunks and embeds each file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File or directory path",
            },
            "chunk_size": {
                "type": "integer",
                "description": "Words per chunk (default 512)",
            },
            "clear": {
                "type": "boolean",
                "description": "Clear existing index first",
            },
        },
        "required": ["path"],
    }
    requires_permission = False

    def __init__(self):
        self._retriever = None

    def _get(self):
        if self._retriever is None:
            from core.npu_models import get_retriever
            self._retriever = get_retriever()
        return self._retriever

    async def execute(self, path: str, chunk_size: int = 512, clear: bool = False) -> str:
        r = self._get()
        if clear:
            r.clear()

        target = Path(path)
        if not target.exists():
            return f"Path not found: {path}"

        if target.is_dir():
            count = r.index_directory(target, chunk_size=chunk_size)
        else:
            count = r.index_file(target, chunk_size=chunk_size)

        s = r.stats()
        return f"Indexed {count} chunks from {s['files']} files. Total: {s['chunks']} chunks, {s['size_mb']:.1f}MB"


class FileSearchTool(Tool):
    """Tool for semantic search over indexed files."""

    name = "search_files"
    description = "Search indexed files semantically. Returns matching chunks with scores."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results (default 5)",
            },
        },
        "required": ["query"],
    }
    requires_permission = False

    def __init__(self):
        self._retriever = None

    def _get(self):
        if self._retriever is None:
            from core.npu_models import get_retriever
            self._retriever = get_retriever()
        return self._retriever

    async def execute(self, query: str, top_k: int = 5) -> str:
        r = self._get()
        results = r.search(query, top_k=top_k)

        if not results:
            s = r.stats()
            return f"No matches. Index has {s['chunks']} chunks. Use index_files to add files."

        out = [f"## Results for: {query}\n"]
        for res in results:
            out.append(f"**{res['path']}** (score: {res['score']:.2f})")
            out.append(f"{res['preview']}\n")

        return "\n".join(out)


class ContextRetrievalTool(Tool):
    """Tool for getting formatted context."""

    name = "get_context"
    description = "Get relevant context from indexed files for a query. Use to augment LLM context."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Max context tokens (default 4000)",
            },
        },
        "required": ["query"],
    }
    requires_permission = False

    def __init__(self):
        self._retriever = None

    def _get(self):
        if self._retriever is None:
            from core.npu_models import get_retriever
            self._retriever = get_retriever()
        return self._retriever

    async def execute(self, query: str, max_tokens: int = 4000) -> str:
        r = self._get()
        ctx = r.get_context(query, max_tokens=max_tokens)
        if not ctx:
            return "No context found. Use index_files first."
        return ctx


class VoiceTokenizeTool(Tool):
    """Tool for NPU-optimized voice tokenization."""

    name = "tokenize_voice"
    description = "Process audio into semantic and acoustic tokens using NPU-optimized tokenizer. Fast voice processing for conversational AI."
    parameters = {
        "type": "object",
        "properties": {
            "audio_data": {
                "type": "string",
                "description": "Base64-encoded audio (WAV format)",
            },
        },
        "required": ["audio_data"],
    }
    requires_permission = False

    async def execute(self, audio_data: str) -> str:
        import base64
        import io
        import wave

        try:
            audio_bytes = base64.b64decode(audio_data)
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0
                sr = wf.getframerate()
        except Exception as e:
            return f"Audio decode error: {e}"

        from core.npu_voice import get_voice_tokenizer
        voice = get_voice_tokenizer()
        result = voice.tokenize(audio, sr)

        decoded = voice.decode_semantic(result.semantic_tokens)

        return f"""## Voice Tokenization Complete

**Timing:**
- Duration: {result.duration_ms:.1f}ms
- Semantic: {result.semantic_ms:.2f}ms
- Acoustic: {result.acoustic_ms:.2f}ms
- Total: {result.total_ms:.2f}ms
- Device: {result.device}

**Tokens:**
- Semantic: {len(result.semantic_tokens)} tokens (vocab: {voice.SEMANTIC_VOCAB})
- Acoustic: {len(result.acoustic_tokens)} frames x 8 codebooks (vocab: {voice.ACOUSTIC_VOCAB})

**Speech Analysis:**
- Phonemes: {decoded['phoneme_count']}
- Silence segments: {decoded['silence_count']}
- Emphasis markers: {decoded['emphasis_count']}

**Sample acoustic frame:** `{result.acoustic_tokens[0] if result.acoustic_tokens else 'N/A'}`"""


class IndexStatsTool(Tool):
    """Tool for index statistics."""

    name = "index_stats"
    description = "Get statistics about the file index."
    parameters = {"type": "object", "properties": {}, "required": []}
    requires_permission = False

    async def execute(self) -> str:
        from core.npu_models import get_retriever
        r = get_retriever()
        s = r.stats()
        return f"Index: {s['files']} files, {s['chunks']} chunks, {s['dim']}d embeddings ({s['model']}), {s['size_mb']:.1f}MB"


class RegisterSpeakerTool(Tool):
    """Tool for registering a speaker for identification."""

    name = "register_speaker"
    description = "Register a speaker from audio for later identification. Creates voice profile from acoustic tokens."
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Speaker name",
            },
            "audio_data": {
                "type": "string",
                "description": "Base64-encoded audio sample (WAV)",
            },
        },
        "required": ["name", "audio_data"],
    }
    requires_permission = False

    async def execute(self, name: str, audio_data: str) -> str:
        import base64
        import io
        import wave

        try:
            audio_bytes = base64.b64decode(audio_data)
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0
                sr = wf.getframerate()
        except Exception as e:
            return f"Audio decode error: {e}"

        from core.npu_voice import get_voice_processor
        processor = get_voice_processor()
        processor.register_speaker(name, audio, sr)

        return f"Speaker '{name}' registered successfully."


class IdentifySpeakerTool(Tool):
    """Tool for identifying a speaker from audio."""

    name = "identify_speaker"
    description = "Identify a speaker from audio. Returns matching speaker name and confidence score."
    parameters = {
        "type": "object",
        "properties": {
            "audio_data": {
                "type": "string",
                "description": "Base64-encoded audio (WAV)",
            },
        },
        "required": ["audio_data"],
    }
    requires_permission = False

    async def execute(self, audio_data: str) -> str:
        import base64
        import io
        import wave

        try:
            audio_bytes = base64.b64decode(audio_data)
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0
                sr = wf.getframerate()
        except Exception as e:
            return f"Audio decode error: {e}"

        from core.npu_voice import get_voice_processor
        processor = get_voice_processor()
        name, confidence = processor.identify_speaker(audio, sr)

        if name == "unknown":
            return "Speaker not recognized. Register speakers first with 'register_speaker'."

        return f"Identified: {name} (confidence: {confidence:.2%})"


class VoiceStatsTool(Tool):
    """Tool for voice processor statistics."""

    name = "voice_stats"
    description = "Get statistics about the NPU voice processor."
    parameters = {"type": "object", "properties": {}, "required": []}
    requires_permission = False

    async def execute(self) -> str:
        from core.npu_voice import get_voice_processor
        p = get_voice_processor()
        s = p.get_stats()
        return f"Voice Processor: {s['device']}, {s['speakers_registered']} speakers registered, {s['frame_rate']} fps"


class ConversationalSpeechTool(Tool):
    """Tool for conversational speech with tone/intent detection."""

    name = "process_speech"
    description = "Process speech with tone and intent detection. Returns semantic, acoustic, and conversational tokens for natural voice synthesis."
    parameters = {
        "type": "object",
        "properties": {
            "audio_data": {
                "type": "string",
                "description": "Base64-encoded audio (WAV format)",
            },
            "context": {
                "type": "string",
                "description": "Optional conversation context for better prosody",
            },
        },
        "required": ["audio_data"],
    }
    requires_permission = False

    async def execute(self, audio_data: str, context: Optional[str] = None) -> str:
        import base64
        import io
        import wave

        try:
            audio_bytes = base64.b64decode(audio_data)
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
                audio = audio.astype(np.float32) / 32768.0
                sr = wf.getframerate()
        except Exception as e:
            return f"Audio decode error: {e}"

        from core.conversational_speech import get_conversational_speech_model
        csm = get_conversational_speech_model()
        result = csm.process_speech(audio, sr, context_text=context)

        tone_intent = result.get('tone_intent', {})
        speaker = result.get('speaker', {})

        return f"""## Conversational Speech Analysis

**Tone & Intent:**
- Primary: {tone_intent.get('primary', 'neutral')}
- Confidence: {tone_intent.get('confidence', 0):.0%}
- Prosocial: {tone_intent.get('prosocial', False)}

**Speaker:**
- Identity: {speaker.get('identity', 'unknown')}
- Confidence: {speaker.get('confidence', 0):.0%}

**Tokens:**
- Semantic: {len(result.get('semantic_tokens', []))} tokens
- Acoustic: {len(result.get('acoustic_tokens', []))} frames
- Tone tokens: {result.get('tone_tokens', [])}

**Prosody:**
- Rate: {result.get('prosody', {}).get('rate', 1.0):.2f}x
- Energy: {result.get('prosody', {}).get('energy', 0.5):.2f}
- Pitch range: {result.get('prosody', {}).get('pitch_range', 0):.0f}Hz"""


class SynthesizeSpeechTool(Tool):
    """Tool for synthesizing speech with controlled tone."""

    name = "synthesize_speech"
    description = "Synthesize speech from text with controlled tone and intent. Returns audio as base64-encoded WAV."
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to synthesize",
            },
            "tone": {
                "type": "string",
                "enum": ["neutral", "friendly", "professional", "empathetic", "excited", "calm", "questioning"],
                "description": "Tone for speech synthesis",
            },
            "speaker_id": {
                "type": "string",
                "description": "Optional speaker identity to emulate",
            },
        },
        "required": ["text"],
    }
    requires_permission = False

    async def execute(self, text: str, tone: Optional[str] = None, speaker_id: Optional[str] = None) -> str:
        import base64
        import io
        import wave

        from core.conversational_speech import get_conversational_speech_model, ToneIntent

        csm = get_conversational_speech_model()

        tone_enum = ToneIntent.NEUTRAL
        if tone:
            tone_upper = tone.upper()
            for t in ToneIntent:
                if t.name == tone_upper:
                    tone_enum = t
                    break

        result = csm.synthesize(text, tone=tone_enum, speaker_id=speaker_id)

        if 'error' in result:
            return f"Synthesis error: {result['error']}"

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(result['sample_rate'])
            audio_int = (result['audio'] * 32767).astype(np.int16)
            wf.writeframes(audio_int.tobytes())

        audio_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return f"""## Speech Synthesized

**Text:** "{text[:50]}{'...' if len(text) > 50 else ''}"
**Tone:** {tone_enum.name}
**Duration:** {result['duration_ms']:.0f}ms
**Sample rate:** {result['sample_rate']}Hz

**Audio:** (base64 WAV)
{audio_b64[:100]}..."""


class ToneAnalysisTool(Tool):
    """Tool for analyzing text tone/intent."""

    name = "analyze_tone"
    description = "Analyze the tone and intent of text for conversational context."
    parameters = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to analyze",
            },
        },
        "required": ["text"],
    }
    requires_permission = False

    async def execute(self, text: str) -> str:
        from core.conversational_speech import get_conversational_speech_model
        csm = get_conversational_speech_model()
        tone = csm.classify_tone(text)
        prosody = csm.get_prosody_for_tone(tone)

        return f"""## Tone Analysis

**Input:** "{text[:80]}{'...' if len(text) > 80 else ''}"
**Detected tone:** {tone.name}
**Suggested prosody:**
- Rate: {prosody.get('rate', 1.0):.2f}x
- Energy: {prosody.get('energy', 0.5):.2f}
- Pitch range: {prosody.get('pitch_range', 0):.0f}Hz"""


class RetrievalTools:
    """Collection of retrieval and voice tools."""

    def __init__(self, default_model: str = "mini"):
        self.tools = [
            EmbeddingTool(model_type=default_model),
            FileIndexTool(),
            FileSearchTool(),
            ContextRetrievalTool(),
            VoiceTokenizeTool(),
            RegisterSpeakerTool(),
            IdentifySpeakerTool(),
            IndexStatsTool(),
            VoiceStatsTool(),
            ConversationalSpeechTool(),
            SynthesizeSpeechTool(),
            ToneAnalysisTool(),
        ]

    def get_tools(self) -> list[Tool]:
        return self.tools
