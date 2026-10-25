"""Voice interface for Jarvis with wake word detection and conversational speech."""
import asyncio
import logging
import queue
from typing import Any

from core.agent import JarvisAgent
from core.permissions import PermissionLevel
from utils.audio import AudioHandler

logger = logging.getLogger(__name__)


class VoiceInterface:
    """Voice interface with wake word detection and conversational speech."""

    def __init__(
        self,
        agent: JarvisAgent,
        wake_word: str = "hey jarvis",
        use_conversational_speech: bool = False,
        config: Any = None,
    ):
        self.agent = agent
        self.wake_word = wake_word.lower()
        self.running = False
        self.audio_handler = AudioHandler()
        self.tts_engine = None  # TTSEngine, or legacy pyttsx3 engine, or "espeak"
        self.piper_voice = None  # Retained for API compatibility; Piper lives in TTSEngine now
        self.stt_model = None
        self.wakeword_model = None
        self.audio_queue: queue.Queue = queue.Queue()
        self._audio_available = self.audio_handler._audio_available
        self.config = config or {}

        # Conversational speech model (optional)
        self.use_conversational_speech = use_conversational_speech
        self.speech_model = None

    async def start(self) -> None:
        """Start the voice interface."""
        self.running = True

        print("Initializing voice components...")

        # Initialize components
        print("  Loading TTS...")
        await self._initialize_tts()
        print("  Loading STT...")
        await self._initialize_stt()
        print("  Loading Wake Word...")
        await self._initialize_wakeword()

        # Initialize conversational speech model if enabled
        if self.use_conversational_speech:
            print("  Loading Conversational Speech...")
            await self._initialize_speech_model()

        # Check if LLM is ready
        print("  Checking LLM connection...")
        if not await self.agent.is_ready():
            await self._speak("Error: Cannot connect to LLM. Make sure it's running.")
            return

        print("Voice components ready!")
        await self._speak("Jarvis ready. Say hey Jarvis to activate.")

        # Main loop
        await self._run_loop()

    async def _initialize_tts(self) -> None:
        """Initialize text-to-speech with the shared TTSEngine (edge-tts neural,
        piper local, honest text-only degradation)."""
        try:
            from core.tts_engine import get_tts_engine
            self.tts_engine = get_tts_engine(
                config=self.config if isinstance(self.config, dict) else {}
            )
            await self.tts_engine.initialize()
            if self.tts_engine.available():
                state = self.tts_engine.get_state()
                print(f"  TTS ready: {state['active_backend']} ({state['active_voice']})")
            else:
                print("  TTS unavailable (no backend) — responses will be text-only")
        except Exception as e:
            print(f"  TTS init error: {e}")
            self.tts_engine = None

        # Legacy pyttsx3/espeak fallbacks remain only if TTSEngine has no backend
        if self.tts_engine and self.tts_engine.available():
            return

        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.piper_voice = None
            self.tts_engine.setProperty("rate", 160)
            self.tts_engine.setProperty("volume", 1.0)
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                voice_name = voice.name.lower()
                if 'female' in voice_name or 'en-us' in voice.id.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    print(f"  Using pyttsx3 voice: {voice.name}")
                    break
        except ImportError:
            print("  No TTS backend available (install edge-tts: pip install edge-tts)")

    async def _initialize_stt(self) -> None:
        """Initialize speech-to-text model with Whisper large-v3 for best accuracy."""
        try:
            from faster_whisper import WhisperModel

            # Use large-v3 for best accuracy (falls back to medium if memory issues)
            model_size = "large-v3"
            device = "cpu"
            compute_type = "int8"

            print(f"  Loading Whisper {model_size} (this may take a moment)...")
            try:
                self.stt_model = WhisperModel(model_size, device=device, compute_type=compute_type)
                print(f"  ✅ Whisper {model_size} loaded")
            except Exception as e:
                # Fall back to medium if large-v3 fails (memory constraints)
                print(f"  Large-v3 failed ({e}), trying medium...")
                try:
                    self.stt_model = WhisperModel("medium", device=device, compute_type=compute_type)
                    print("  ✅ Whisper medium loaded")
                except Exception as e2:
                    # Final fallback to base
                    print(f"  Medium failed, using base: {e2}")
                    self.stt_model = WhisperModel("base", device=device, compute_type=compute_type)
                    print("  ✅ Whisper base loaded")

        except ImportError:
            print("Warning: faster-whisper not installed. Voice input will not work.")
            print("Install with: pip install faster-whisper")

    async def _initialize_wakeword(self) -> None:
        """Initialize wake word detection."""
        try:
            import openwakeword
            from openwakeword import Model

            # Load the default model which includes "hey jarvis"
            self.wakeword_model = Model()
        except ImportError:
            print("Warning: openwakeword not installed. Wake word detection disabled.")
            print("Install with: pip install openwakeword")

    async def _initialize_speech_model(self) -> None:
        """Initialize conversational speech model."""
        try:
            from speech.conversation import ConversationalSpeechModel
            from dataclasses import dataclass

            @dataclass
            class ConversationConfig:
                sample_rate: int = 16000
                voice_preservation: bool = True
                response_style: str = "natural"

            # Create speech model with LLM client
            config = ConversationConfig(
                sample_rate=16000,
                voice_preservation=True,
                response_style="natural",
            )

            # Use the agent's LLM client
            self.speech_model = ConversationalSpeechModel(
                config=config,
                llm_client=self.agent.llm,
            )

            loaded = await self.speech_model.load()
            if loaded:
                print("Conversational speech model initialized.")
            else:
                print("Warning: Could not fully initialize speech model.")
                self.speech_model = None

        except ImportError as e:
            print(f"Warning: Speech model not available: {e}")
            self.speech_model = None
        except Exception as e:
            print(f"Warning: Error initializing speech model: {e}")
            self.speech_model = None

    async def _speak(self, text: str) -> None:
        """Speak text: TTSEngine (edge/piper neural) -> legacy pyttsx3/espeak -> text."""
        print(f"Jarvis: {text}")

        loop = asyncio.get_event_loop()

        # Primary: shared TTSEngine (neural voice, MP3/WAV bytes)
        if self.tts_engine and hasattr(self.tts_engine, "synthesize"):
            try:
                audio = await self.tts_engine.synthesize(text)
                if audio is not None:
                    await self._play_tts_bytes(loop, audio, self.tts_engine._state.active_backend)
                    return
                # synthesis failed — fall through to legacy engines below
            except Exception as e:
                print(f"TTSEngine error, falling back: {e}")

        # Neural conversational speech model (optional)
        if self.speech_model and self.use_conversational_speech:
            try:
                result = await self.speech_model.synthesize_speech(text)
                if result and isinstance(result, dict):
                    audio = result.get('audio')
                    if audio is not None:
                        await self.audio_handler.play_audio(audio)
                        return
            except Exception as e:
                print(f"Neural TTS error, falling back: {e}")

        # Legacy: pyttsx3 engine object
        if self.tts_engine and hasattr(self.tts_engine, "say"):
            def speak_sync():
                try:
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                except Exception as e:
                    print(f"[TTS error: {e}] {text}")
            await loop.run_in_executor(None, speak_sync)
            return

        # Legacy: espeak binaries
        if self.tts_engine == "espeak":
            import subprocess
            import shutil
            if shutil.which("espeak-ng"):
                await loop.run_in_executor(None, lambda: subprocess.run(
                    ["espeak-ng", "-v", "en-us+f3", "-s", "160", text],
                    capture_output=True,
                ))
            elif shutil.which("espeak"):
                await loop.run_in_executor(None, lambda: subprocess.run(
                    ["espeak", "-v", "en-us+f3", "-s", "160", text],
                    capture_output=True,
                ))
            else:
                print(f"[TTS] {text}")

    async def _play_tts_bytes(self, loop: asyncio.AbstractEventLoop, audio: bytes, backend: str) -> None:
        """Play synthesized audio bytes on the local machine (cross-platform)."""
        import os
        import subprocess
        import tempfile

        suffix = ".mp3" if backend == "edge" else ".wav"

        if hasattr(self.audio_handler, 'play_wav') and suffix == ".wav":
            await self.audio_handler.play_wav(audio)
            return

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio)
            temp_path = f.name
        try:
            import shutil
            import sys as _sys
            if _sys.platform == "win32":
                # PowerShell media playback (available on every Windows box)
                ps = (
                    "Add-Type -AssemblyName presentationCore; "
                    f"$p = New-Object System.Windows.Media.MediaPlayer; "
                    f"$p.Open([Uri]::Create('{temp_path}')); "
                    "Start-Sleep -Milliseconds 200; "
                    "$p.Play(); Start-Sleep -Seconds 30; $p.Close()"
                )
                await loop.run_in_executor(None, lambda: subprocess.run(
                    ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                    capture_output=True, timeout=60,
                ))
            elif os.path.exists('/usr/bin/paplay'):
                await loop.run_in_executor(None, lambda: subprocess.run(['paplay', temp_path], capture_output=True))
            elif os.path.exists('/usr/bin/aplay'):
                await loop.run_in_executor(None, lambda: subprocess.run(['aplay', '-q', temp_path], capture_output=True))
            elif shutil.which("ffplay"):
                await loop.run_in_executor(None, lambda: subprocess.run(
                    ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', temp_path],
                    capture_output=True,
                ))
            else:
                print(f"[TTS audio saved: {temp_path}]")
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    async def _listen_for_wakeword(self) -> bool:
        """Listen for wake word activation."""
        if self.wakeword_model is None:
            # If no wake word model, just wait for user input
            return True

        print(f"Listening for '{self.wake_word}'...")

        try:

            # Record short audio chunks and check for wake word
            chunk_duration = 1.0  # seconds

            while self.running:
                audio_data = await self.audio_handler.record_audio(duration=chunk_duration)

                if audio_data is not None:
                    # Check for wake word
                    prediction = self.wakeword_model.predict(audio_data)

                    # Check if wake word detected
                    if hasattr(prediction, 'hey_jarvis'):
                        if prediction.hey_jarvis > 0.5:
                            return True

                    # Also check for manual activation via key press
                    import sys
                    import select
                    if select.select([sys.stdin], [], [], 0)[0]:
                        line = sys.stdin.readline().strip()
                        if line.lower() == self.wake_word:
                            return True

        except Exception as e:
            print(f"Wake word detection error: {e}")
            return True  # Fall back to always active

        return False

    async def _listen_for_speech(self, timeout: float = 10.0) -> str | None:
        """Listen for user speech and return transcribed text."""
        # If audio not available, fall back to text input
        if not self.audio_handler._audio_available:
            print("[Voice mode - text input (audio unavailable)]")
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                line = await loop.run_in_executor(None, input, "You (text): ")
                return line.strip() if line else None
            except EOFError:
                return None

        if self.stt_model is None:
            print("Speech recognition not available.")
            return None

        print("Listening...")

        # Record audio with VAD
        audio_data = await self.audio_handler.record_audio_with_vad(
            timeout=timeout,
            silence_duration=2.0,
        )

        if audio_data is None:
            return None

        # If using conversational speech model, process through it
        if self.speech_model and self.use_conversational_speech:
            try:
                text, tokens = await self.speech_model.process_speech(audio_data)
                return text
            except Exception as e:
                print(f"Speech model error: {e}")
                # Fall back to Whisper

        # Transcribe with Whisper
        segments, _ = self.stt_model.transcribe(audio_data)

        # Combine segments
        text = " ".join(segment.text for segment in segments)
        return text.strip() if text else None

    async def _run_loop(self) -> None:
        """Main interaction loop."""
        # Check if audio is available
        audio_available = self.audio_handler._audio_available if hasattr(self.audio_handler, '_audio_available') else False

        if not audio_available:
            print("\n[Voice mode - Audio not available, using text input]")
            print("Type your commands. Type 'exit' to quit.\n")

            # Text-based fallback loop
            import sys
            import asyncio

            while self.running:
                try:
                    # Get text input
                    print("You: ", end="", flush=True)

                    # Non-blocking input using asyncio
                    loop = asyncio.get_event_loop()
                    user_input = await loop.run_in_executor(None, sys.stdin.readline)
                    user_input = user_input.strip()

                    if not user_input:
                        continue

                    # Check for exit commands
                    if user_input.lower() in ["exit", "quit", "goodbye", "stop"]:
                        print("Jarvis: Goodbye!")
                        self.running = False
                        continue

                    # Process with agent
                    response = await self.agent.run_step(user_input)
                    print(f"Jarvis: {response}\n")

                except KeyboardInterrupt:
                    self.running = False
                except Exception as e:
                    print(f"Error: {e}")
            return

        # Full voice loop (when audio is available)
        while self.running:
            try:
                # Wait for wake word
                activated = await self._listen_for_wakeword()

                if not activated or not self.running:
                    continue

                # Play activation sound
                await self._speak("Yes?")

                # Listen for command
                user_input = await self._listen_for_speech()

                if user_input is None or not user_input.strip():
                    continue

                print(f"You: {user_input}")

                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "goodbye", "stop"]:
                    await self._speak("Goodbye!")
                    self.running = False
                    continue

                # Process with agent
                response = await self.agent.run_step(user_input)

                # Handle permission requests
                if response.startswith("[PERMISSION_REQUIRED]"):
                    # For voice, default to asking for confirmation
                    await self._speak("This action requires permission. Say yes to proceed, or no to cancel.")
                    confirmation = await self._listen_for_speech()

                    if confirmation and "yes" in confirmation.lower():
                        # Grant temporary permission and retry
                        parts = response.replace("[PERMISSION_REQUIRED]", "").strip()
                        if "Tool:" in parts:
                            tool_name = parts.split("Tool:")[1].split(",")[0].strip()
                            await self.agent.grant_permission(
                                tool_name, {},
                                PermissionLevel.ALLOW_ONCE,
                            )
                            response = await self.agent.run_step(f"Retry: {user_input}")
                    else:
                        await self._speak("Action cancelled.")
                        continue

                # Speak response
                await self._speak(response)

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                print(f"Error: {e}")
                from core.helpers import safe_error_message
                await self._speak(f"Sorry, I encountered an error: {safe_error_message(e)}")

    async def stop(self) -> None:
        """Stop the voice interface."""
        self.running = False

    def set_voice(self, voice_name: str) -> bool:
        """Set the TTS voice.

        Args:
            voice_name: Name of the voice to use

        Returns:
            True if voice was set successfully
        """
        if self.speech_model:
            return self.speech_model.set_voice(voice_name)

        # Shared TTSEngine (edge/piper)
        if self.tts_engine and hasattr(self.tts_engine, "set_voice"):
            try:
                import asyncio as _aio
                return _aio.get_event_loop().run_until_complete(
                    self.tts_engine.set_voice(voice_name)
                )
            except Exception as e:
                logger.warning(f"Failed to set TTSEngine voice: {e}")
                return False

        # Legacy pyttsx3 engine
        if self.tts_engine and hasattr(self.tts_engine, "getProperty"):
            try:
                voices = self.tts_engine.getProperty('voices')
                for voice in voices:
                    if voice_name.lower() in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        return True
            except Exception as e:
                logger.warning(f"Failed to set voice: {e}")

        return False

    def learn_voice(self, audio_samples: Any, name: str) -> None:
        """Learn a voice profile from audio samples.

        Args:
            audio_samples: Audio samples to learn from
            name: Name to save the voice profile as
        """
        if self.speech_model:
            self.speech_model.learn_voice(audio_samples, name)
