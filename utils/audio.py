"""Audio utilities for Jarvis voice interface."""
import queue

import numpy as np


class AudioHandler:
    """Handles audio input/output for voice interface."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self._audio_available = self._check_audio()

    def _check_audio(self) -> bool:
        """Check if audio devices are available."""
        try:
            import sounddevice as sd
            return True
        except ImportError:
            print("Warning: sounddevice not installed. Audio features disabled.")
            print("Install with: pip install sounddevice")
            return False
        except OSError as e:
            print(f"Warning: Audio system not available: {e}")
            print("Audio features disabled. Install PortAudio for audio support.")
            return False

    async def record_audio(self, duration: float) -> np.ndarray | None:
        """Record audio for a specified duration."""
        if not self._audio_available:
            return None

        try:
            import sounddevice as sd

            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
            )
            sd.wait()

            return recording.flatten()

        except Exception as e:
            print(f"Error recording audio: {e}")
            return None

    async def record_audio_with_vad(
        self,
        timeout: float = 30.0,
        silence_duration: float = 2.0,
        min_speech_duration: float = 0.5,
    ) -> np.ndarray | None:
        """Record audio with Voice Activity Detection.

        Records until silence is detected for silence_duration seconds.
        """
        if not self._audio_available:
            return None

        try:
            import sounddevice as sd

            audio_chunks = []
            silence_start = None
            speech_start = None
            total_duration = 0.0

            print("Listening for speech...")

            # Use a callback-based approach
            q = queue.Queue()

            def audio_callback(indata, frames, time, status):
                q.put(indata.copy())

            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                blocksize=self.chunk_size,
            )

            with stream:
                while total_duration < timeout:
                    try:
                        chunk = q.get(timeout=0.1)
                        audio_chunks.append(chunk)

                        # Simple VAD based on energy
                        energy = np.abs(chunk).mean()

                        # Threshold for speech detection
                        is_speech = energy > 0.02

                        chunk_duration = len(chunk) / self.sample_rate
                        total_duration += chunk_duration

                        if is_speech:
                            silence_start = None
                            if speech_start is None:
                                speech_start = total_duration
                        else:
                            if speech_start is not None:
                                if silence_start is None:
                                    silence_start = total_duration

                                if silence_start is not None:
                                    silence_len = total_duration - silence_start
                                    if silence_len >= silence_duration:
                                        # Check if we had enough speech
                                        speech_len = silence_start - speech_start
                                        if speech_len >= min_speech_duration:
                                            break
                                        # Otherwise, reset and continue
                                        speech_start = None
                                        silence_start = None

                    except queue.Empty:
                        continue

            if not audio_chunks:
                return None

            # Combine chunks
            audio_data = np.concatenate(audio_chunks, axis=0)
            return audio_data.flatten()

        except Exception as e:
            print(f"Error in VAD recording: {e}")
            return None

    async def play_audio(self, audio_data: np.ndarray) -> None:
        """Play audio data."""
        if not self._audio_available:
            return

        try:
            import sounddevice as sd

            sd.play(audio_data, self.sample_rate)
            sd.wait()

        except Exception as e:
            print(f"Error playing audio: {e}")

    def list_devices(self) -> list[dict]:
        """List available audio devices."""
        if not self._audio_available:
            return []

        try:
            import sounddevice as sd

            devices = []
            for i, dev in enumerate(sd.query_devices()):
                devices.append({
                    "id": i,
                    "name": dev["name"],
                    "inputs": dev["max_input_channels"],
                    "outputs": dev["max_output_channels"],
                    "default": dev.get("default", False),
                })
            return devices

        except Exception as e:
            print(f"Error listing devices: {e}")
            return []
