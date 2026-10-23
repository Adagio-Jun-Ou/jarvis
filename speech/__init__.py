"""Jarvis Conversational Speech Module.

This module implements a speech-to-speech conversation model using:
- Semantic tokens: Capture linguistic content (phonemes, words)
- Acoustic tokens: Capture speaker characteristics, prosody, emotion

Architecture inspired by VALL-E, SpeechT5, and similar models.
"""

from .tokenizer import SpeechTokenizer
from .codec import AcousticCodec, SemanticCodec
from .vocoder import NeuralVocoder
from .conversation import ConversationalSpeechModel

__all__ = [
    "SpeechTokenizer",
    "AcousticCodec",
    "SemanticCodec",
    "NeuralVocoder",
    "ConversationalSpeechModel",
]
