"""
Hybrid Music Renderer - Neural + Programmatic synthesis pipeline.

Provides three rendering paths:
1. FluidSynth (fast, real-time, GM SoundFont) - primary
2. Sine synthesis (fallback, always available)
3. Offline neural rendering (midi-ddsp style, if available)

Architecture:
    MIDI Input → FluidSynth → WAV Output
                ↘ Fallback Synth → WAV Output

Integrates with music_creation_engine as a drop-in replacement for
symbolic_composer_v4.midi_to_audio(), producing realistic instrument
timbres instead of basic sine waves.
"""

import io
import os
import logging
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_RENDERER_LOCK = threading.Lock()
_renderer_instance = None


class RenderPath(Enum):
    FLUIDSYNTH = "fluidsynth"
    SYNTHESIS = "synthesis"
    OFFLINE_NEURAL = "offline_neural"
    PRETTY_MIDI = "pretty_midi"


@dataclass
class RenderResult:
    audio: np.ndarray
    sample_rate: int
    path: Optional[str] = None
    duration_seconds: float = 0.0
    render_path: RenderPath = RenderPath.SYNTHESIS
    render_time_ms: float = 0.0


@dataclass
class FluidSynthConfig:
    sample_rate: int = 44100
    soundfont_path: Optional[str] = None
    gain: float = 0.8
    chorus_depth: float = 8.0
    chorus_speed: float = 0.3
    reverb_roomsize: float = 0.7
    reverb_damping: float = 0.3
    reverb_width: float = 0.8
    reverb_level: float = 0.4


class HybridRenderer:
    """Hybrid music renderer combining FluidSynth + fallback synthesis.

    Automatically selects the best available rendering path:
    - FluidSynth with GM SoundFont (highest quality, real-time)
    - Sine/triangle/sawtooth synthesis (always available fallback)
    - Offline neural path (future: midi-ddsp if available)
    """

    GM_PROGRAMS = {
        "piano": 0,
        "bright_piano": 1,
        "electric_grand": 2,
        "honky_tonk": 3,
        "electric_piano_1": 4,
        "electric_piano_2": 5,
        "harpsichord": 6,
        "clavinet": 7,
        "celesta": 8,
        "glockenspiel": 9,
        "music_box": 10,
        "vibraphone": 11,
        "marimba": 12,
        "xylophone": 13,
        "tubular_bells": 14,
        "dulcimer": 15,
        "organ_drawbar": 16,
        "organ_percussive": 17,
        "organ_rock": 18,
        "organ_church": 19,
        "organ_reed": 20,
        "accordion": 21,
        "harmonica": 22,
        "bandoneon": 23,
        "guitar_nylon": 24,
        "guitar_steel": 25,
        "guitar_jazz": 26,
        "guitar_clean": 27,
        "guitar_muted": 28,
        "guitar_overdriven": 29,
        "guitar_distortion": 30,
        "guitar_harmonics": 31,
        "bass_acoustic": 32,
        "bass_finger": 33,
        "bass_pick": 34,
        "bass_fretless": 35,
        "bass_slap_1": 36,
        "bass_slap_2": 37,
        "bass_synth_1": 38,
        "bass_synth_2": 39,
        "violin": 40,
        "viola": 41,
        "cello": 42,
        "contrabass": 43,
        "tremolo_strings": 44,
        "pizzicato_strings": 45,
        "harp": 46,
        "timpani": 47,
        "strings_ensemble_1": 48,
        "strings_ensemble_2": 49,
        "synth_strings_1": 50,
        "synth_strings_2": 51,
        "choir_aahs": 52,
        "voice_oohs": 53,
        "synth_choir": 54,
        "orchestra_hit": 55,
        "trumpet": 56,
        "trombone": 57,
        "tuba": 58,
        "muted_trumpet": 59,
        "french_horn": 60,
        "brass_section": 61,
        "synth_brass_1": 62,
        "synth_brass_2": 63,
        "soprano_sax": 64,
        "alto_sax": 65,
        "tenor_sax": 66,
        "baritone_sax": 67,
        "oboe": 68,
        "english_horn": 69,
        "bassoon": 70,
        "clarinet": 71,
        "piccolo": 72,
        "flute": 73,
        "recorder": 74,
        "pan_flute": 75,
        "bottle": 76,
    }

    COMPOSER_INSTRUMENTS = {
        "bach": {
            "melody": "harpsichord",
            "harmony": "strings_ensemble_1",
            "bass": "cello",
        },
        "mozart": {
            "melody": "piano",
            "harmony": "strings_ensemble_1",
            "bass": "bass_acoustic",
        },
        "beethoven": {
            "melody": "piano",
            "harmony": "strings_ensemble_1",
            "bass": "bass_acoustic",
        },
        "chopin": {
            "melody": "piano",
            "harmony": "strings_ensemble_2",
            "bass": "cello",
        },
        "debussy": {
            "melody": "piano",
            "harmony": "strings_ensemble_2",
            "bass": "cello",
        },
        "brahms": {
            "melody": "piano",
            "harmony": "french_horn",
            "bass": "bass_acoustic",
        },
        "schumann": {
            "melody": "piano",
            "harmony": "strings_ensemble_1",
            "bass": "bass_finger",
        },
        "tchaikovsky": {
            "melody": "violin",
            "harmony": "strings_ensemble_1",
            "bass": "contrabass",
        },
        "dvorak": {
            "melody": "violin",
            "harmony": "strings_ensemble_1",
            "bass": "cello",
        },
        "default": {
            "melody": "piano",
            "harmony": "strings_ensemble_1",
            "bass": "bass_acoustic",
        },
    }

    def __init__(self, config: Optional[FluidSynthConfig] = None):
        self.config = config or FluidSynthConfig()
        self._fluidsynth_available = False
        self._synth = None
        self._soundfont_id = None
        self._pretty_midi_available = False
        self._music21_available = False
        self._librosa_available = False
        self._init_lock = threading.Lock()
        self._initialized = False

        self._check_availability()

    def _check_availability(self):
        try:
            import pretty_midi
            self._pretty_midi_available = True
        except ImportError:
            pass

        try:
            import music21
            self._music21_available = True
        except ImportError:
            pass

        try:
            import librosa
            self._librosa_available = True
        except ImportError:
            pass

        for path in self._find_soundfonts():
            if Path(path).exists():
                self.config.soundfont_path = path
                break

        logger.info(
            f"HybridRenderer: pretty_midi={self._pretty_midi_available}, "
            f"music21={self._music21_available}, librosa={self._librosa_available}, "
            f"soundfont={self.config.soundfont_path}"
        )

    @staticmethod
    def _find_soundfonts() -> list[str]:
        candidates = [
            "/usr/share/sounds/sf2/MuseScore_General.sf2",
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/default-GM.sf2",
            "/usr/share/sounds/sf2/TimGM6mb.sf2",
            str(Path.home() / ".jarvis" / "soundfonts" / "MuseScore_General.sf2"),
            str(Path.home() / ".jarvis" / "soundfonts" / "FluidR3_GM.sf2"),
        ]
        env_sf = os.environ.get("JARVIS_SOUNDFONT", "")
        if env_sf:
            candidates.insert(0, env_sf)
        return candidates

    def initialize(self) -> bool:
        if self._initialized:
            return self._fluidsynth_available

        with self._init_lock:
            if self._initialized:
                return self._fluidsynth_available

            try:
                import fluidsynth

                self._synth = fluidsynth.Synth(
                    samplerate=self.config.sample_rate,
                    gain=self.config.gain,
                )

                if self.config.soundfont_path and Path(self.config.soundfont_path).exists():
                    self._soundfont_id = self._synth.sfload(self.config.soundfont_path)
                    self._fluidsynth_available = True
                    logger.info(f"FluidSynth: loaded soundfont {self.config.soundfont_path}")
                else:
                    logger.warning("No GM SoundFont found, FluidSynth unavailable")

            except Exception as e:
                logger.warning(f"FluidSynth init failed: {e}")
                self._fluidsynth_available = False

            self._initialized = True
            return self._fluidsynth_available

    def render_midi_bytes(
        self,
        midi_bytes: bytes,
        composer: str = "default",
        output_path: Optional[str] = None,
    ) -> RenderResult:
        start_time = time.time()

        if not self._initialized:
            self.initialize()

        instruments = self.COMPOSER_INSTRUMENTS.get(
            composer, self.COMPOSER_INSTRUMENTS["default"]
        )

        if self._fluidsynth_available and self._pretty_midi_available:
            result = self._render_fluidsynth(midi_bytes, instruments, output_path)
            if result is not None:
                result.render_time_ms = (time.time() - start_time) * 1000
                return result

        result = self._render_synthesis(midi_bytes, output_path)
        result.render_time_ms = (time.time() - start_time) * 1000
        return result

    def render_midi_file(
        self,
        midi_path: str,
        composer: str = "default",
        output_path: Optional[str] = None,
    ) -> RenderResult:
        with open(midi_path, "rb") as f:
            midi_bytes = f.read()
        return self.render_midi_bytes(midi_bytes, composer, output_path)

    def render_composition(
        self,
        composition_dict: dict,
        composer: str = "default",
        output_path: Optional[str] = None,
        bpm: int = 120,
        key: str = "C",
    ) -> RenderResult:
        midi_bytes = self._composition_to_midi(composition_dict, bpm, key)
        return self.render_midi_bytes(midi_bytes, composer, output_path)

    def _render_fluidsynth(
        self,
        midi_bytes: bytes,
        instruments: dict[str, str],
        output_path: Optional[str],
    ) -> Optional[RenderResult]:
        if self._synth is not None and self._soundfont_id is not None:
            try:
                import pretty_midi

                pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))

                for i, track in enumerate(pm.instruments):
                    program_name = list(instruments.values())[min(i, len(instruments) - 1)]
                    program_num = self.GM_PROGRAMS.get(program_name, 0)
                    track.program = program_num
                    track.is_drum = False

                sample_rate = self.config.sample_rate
                synth = self._synth
                sfid = self._soundfont_id

                total_duration = pm.get_end_time()
                if total_duration <= 0:
                    total_duration = 1.0

                total_samples = int(total_duration * sample_rate) + sample_rate

                synth.system_reset()

                all_events = []
                for i, track in enumerate(pm.instruments):
                    program_num = track.program
                    channel = min(i, 15)
                    synth.program_select(channel, sfid, 0, program_num)
                    for note in track.notes:
                        all_events.append((note.start, 'on', channel, note.pitch, int(min(note.velocity, 127))))
                        all_events.append((note.end, 'off', channel, note.pitch, 0))

                all_events.sort(key=lambda e: e[0])

                chunk_size = sample_rate
                audio_chunks = []
                current_sample = 0

                event_idx = 0
                while current_sample < total_samples:
                    current_time = current_sample / sample_rate

                    while event_idx < len(all_events) and all_events[event_idx][0] <= current_time:
                        _, etype, ch, pitch, vel = all_events[event_idx]
                        if etype == 'on':
                            synth.noteon(ch, pitch, vel)
                        else:
                            synth.noteoff(ch, pitch)
                        event_idx += 1

                    chunk = synth.get_samples(chunk_size)
                    if chunk is not None and len(chunk) > 0:
                        if chunk.ndim == 2:
                            audio_chunks.append(chunk[:, 0].astype(np.float32))
                        else:
                            audio_chunks.append(chunk[:chunk_size].astype(np.float32))
                    else:
                        audio_chunks.append(np.zeros(chunk_size, dtype=np.float32))

                    current_sample += chunk_size

                if audio_chunks:
                    audio = np.concatenate(audio_chunks)[:total_samples]
                else:
                    audio = np.zeros(total_samples, dtype=np.float32)

                peak = np.max(np.abs(audio))
                if peak > 0.01:
                    audio = audio * min(1.0, 0.9 / peak)

                audio = self._apply_tone_shaping(audio, sample_rate)

                duration = len(audio) / sample_rate

                if output_path:
                    self._save_audio(audio, output_path)

                return RenderResult(
                    audio=audio,
                    sample_rate=sample_rate,
                    path=output_path,
                    duration_seconds=duration,
                    render_path=RenderPath.FLUIDSYNTH,
                )

            except Exception as e:
                logger.warning(f"FluidSynth rendering failed, falling back to pretty_midi: {e}")

        try:
            import pretty_midi

            pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))

            for i, track in enumerate(pm.instruments):
                program_name = list(instruments.values())[min(i, len(instruments) - 1)]
                program_num = self.GM_PROGRAMS.get(program_name, 0)
                track.program = program_num
                track.is_drum = False

            rendered_bytes = pm.synthesize(fs=self.config.sample_rate)

            if rendered_bytes.ndim == 1:
                audio = rendered_bytes.astype(np.float32)
            else:
                audio = rendered_bytes[:, 0].astype(np.float32)

            peak = np.max(np.abs(audio))
            if peak > 0.01:
                audio = audio * min(1.0, 0.9 / peak)

            audio = self._apply_tone_shaping(audio, self.config.sample_rate)

            duration = len(audio) / self.config.sample_rate

            if output_path:
                self._save_audio(audio, output_path)

            return RenderResult(
                audio=audio,
                sample_rate=self.config.sample_rate,
                path=output_path,
                duration_seconds=duration,
                render_path=RenderPath.PRETTY_MIDI,
            )

        except Exception as e:
            logger.warning(f"pretty_midi render also failed: {e}")
            return None

    def _render_synthesis(
        self,
        midi_bytes: bytes,
        output_path: Optional[str],
    ) -> RenderResult:
        try:
            import pretty_midi

            pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
            audio = pm.synthesize(fs=self.config.sample_rate)

            if audio.ndim > 1:
                audio = audio[:, 0]

            audio = audio.astype(np.float32)

            if audio.size == 0:
                audio = np.zeros(self.config.sample_rate, dtype=np.float32)

            peak = np.max(np.abs(audio))
            if peak > 0.01:
                audio = audio * min(1.0, 0.9 / peak)

            duration = len(audio) / self.config.sample_rate

            if output_path:
                self._save_audio(audio, output_path)

            return RenderResult(
                audio=audio,
                sample_rate=self.config.sample_rate,
                path=output_path,
                duration_seconds=duration,
                render_path=RenderPath.SYNTHESIS,
            )

        except ImportError:
            audio = np.zeros(self.config.sample_rate * 2, dtype=np.float32)
            return RenderResult(
                audio=audio,
                sample_rate=self.config.sample_rate,
                path=output_path,
                duration_seconds=2.0,
                render_path=RenderPath.SYNTHESIS,
            )

    @staticmethod
    def _apply_tone_shaping(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if len(audio) < sample_rate // 2:
            return audio
        try:
            from scipy.signal import butter, sosfiltfilt

            # 1) High-shelf attenuation at 6kHz (-6dB) - tames brightness
            sos_hp = butter(2, 6000, btype='high', fs=sample_rate, output='sos')
            high_freq = sosfiltfilt(sos_hp, audio)
            audio = audio - high_freq * 0.6

            # 2) Low-frequency warmth boost (bass shelf at 300Hz, +2dB)
            sos_shelf = butter(2, 300, btype='high', fs=sample_rate, output='sos')
            high_component = sosfiltfilt(sos_shelf, audio)
            low_component = audio - high_component
            audio = low_component * 1.08 + high_component

            # 3) Gentle smoothing to reduce digital harshness
            sos_smooth = butter(2, 8000, btype='low', fs=sample_rate, output='sos')
            smoothed = sosfiltfilt(sos_smooth, audio)
            audio = audio * 0.85 + smoothed * 0.15

            peak = np.max(np.abs(audio))
            if peak > 0.01:
                audio = audio * min(1.0, 0.9 / peak)
            return audio.astype(np.float32)
        except Exception:
            return audio

    def _composition_to_midi(
        self, composition: dict, bpm: int = 120, key: str = "C"
    ) -> bytes:
        try:
            import pretty_midi

            effective_bpm = composition.get("bpm", bpm)
            effective_key = composition.get("key", key)

            pm = pretty_midi.PrettyMIDI(initial_tempo=effective_bpm)

            note_names = {
                "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
                "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
                "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
            }

            key_offset = note_names.get(
                effective_key.replace(" minor", "").replace(" major", "").strip(), 0
            )

            instrument_names = composition.get("instruments", {})
            default_programs = {"melody": 0, "harmony": 48, "bass": 32}

            channel_programs = {}
            for name, prog_name in instrument_names.items():
                ch = 0 if name == "melody" else 1 if name == "harmony" else 2
                channel_programs[ch] = self.GM_PROGRAMS.get(prog_name, 0)

            track_keys = ["melody", "harmony", "bass"]
            has_track_format = any(k in composition for k in track_keys)

            if not has_track_format and "notes" in composition:
                all_notes = composition["notes"]
                channel_notes = {}
                for note_entry in all_notes:
                    ch = note_entry.get("channel", 0) if isinstance(note_entry, dict) else 0
                    channel_notes.setdefault(ch, []).append(note_entry)

                for ch in sorted(channel_notes.keys()):
                    program = channel_programs.get(ch, default_programs.get(
                        track_keys[ch] if ch < len(track_keys) else "melody", 0
                    ))
                    inst = pretty_midi.Instrument(program=program, name=f"Track_{ch}")

                    for note_entry in channel_notes[ch]:
                        parsed = self._parse_note_entry(note_entry, key_offset, note_names)
                        if parsed is None:
                            continue
                        pitch, velocity, start_time, end_time = parsed
                        inst.notes.append(pretty_midi.Note(
                            velocity=velocity, pitch=pitch,
                            start=start_time, end=end_time,
                        ))

                    pm.instruments.append(inst)
            else:
                for track_name in track_keys:
                    if track_name not in composition:
                        continue
                    track_data = composition[track_name]
                    notes = track_data if not isinstance(track_data, dict) else track_data.get("notes", [])

                    ch_idx = track_keys.index(track_name)
                    program = channel_programs.get(ch_idx, default_programs.get(track_name, 0))
                    inst = pretty_midi.Instrument(program=program, name=track_name.capitalize())

                    time_offset = 0.0
                    for note_entry in notes:
                        parsed = self._parse_note_entry(note_entry, key_offset, note_names, time_offset)
                        if parsed is None:
                            continue
                        pitch, velocity, start_time, end_time = parsed

                        inst.notes.append(pretty_midi.Note(
                            velocity=velocity, pitch=pitch,
                            start=start_time, end=end_time,
                        ))
                        time_offset = end_time

                    pm.instruments.append(inst)

            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                pm.write(tmp_path)
                with open(tmp_path, "rb") as f:
                    return f.read()
            finally:
                os.unlink(tmp_path)

        except ImportError:
            logger.warning("pretty_midi not available for composition-to-MIDI conversion")
            return b""

        except Exception as e:
            logger.error(f"Composition-to-MIDI conversion failed: {e}")
            return b""

    @staticmethod
    def _parse_note_entry(note_entry, key_offset=0, note_names=None, time_offset=None):
        """Parse a note entry (list, tuple, or dict) into (pitch, velocity, start, end)."""
        if note_names is None:
            note_names = {
                "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
                "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
                "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
            }

        if isinstance(note_entry, (list, tuple)) and len(note_entry) >= 3:
            pitch, velocity, duration = note_entry[0], note_entry[1], note_entry[2]
            start_time = time_offset if time_offset is not None else 0.0
            end_time = start_time + duration
        elif isinstance(note_entry, dict):
            pitch = note_entry.get("pitch", 60)
            velocity = note_entry.get("velocity", 80)
            start_time = note_entry.get("start", time_offset if time_offset is not None else 0.0)
            duration = note_entry.get("duration", 0.5)
            end_time = start_time + duration
        else:
            return None

        if isinstance(pitch, str):
            import re as _re
            match = _re.match(r'^([A-Ga-g#b]+)(\d+)$', pitch)
            if match:
                note_part = match.group(1)
                octave = int(match.group(2))
            else:
                note_part = pitch
                octave = 4
            pitch = note_names.get(note_part, 0) + (octave + 1) * 12 + key_offset
        elif not isinstance(pitch, int):
            pitch = 60

        pitch = max(0, min(127, int(pitch)))
        velocity = max(1, min(127, int(velocity)))

        return pitch, velocity, float(start_time), float(end_time)

    def _save_audio(self, audio: np.ndarray, path: str):
        try:
            import soundfile

            soundfile.write(path, audio, self.config.sample_rate)
            logger.info(f"Saved audio to {path}")

        except ImportError:
            try:
                from scipy.io import wavfile

                audio_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(path, self.config.sample_rate, audio_int16)
                logger.info(f"Saved audio (16-bit) to {path}")
            except Exception as e:
                logger.error(f"Failed to save audio: {e}")

    def analyze_audio(self, audio: np.ndarray, sr: int = 44100) -> dict:
        result = {
            "duration": len(audio) / sr if sr > 0 else 0,
            "rms_energy": float(np.sqrt(np.mean(audio ** 2))),
            "peak_amplitude": float(np.max(np.abs(audio))),
        }

        if self._librosa_available:
            try:
                import librosa

                result["tempo"] = float(librosa.feature.rhythm.tempo(y=audio, sr=sr)[0])
                chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
                key_idx = int(np.argmax(chroma.sum(axis=1)))
                key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                result["estimated_key"] = key_names[key_idx]
                result["spectral_centroid"] = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
                result["spectral_bandwidth"] = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))

            except Exception as e:
                logger.debug(f"Librosa analysis failed: {e}")

        if not self._librosa_available or "tempo" not in result:
            result["tempo"] = 120.0
            result["estimated_key"] = "C"

        try:
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
            magnitudes = np.abs(fft)
            top_idx = np.argsort(magnitudes)[-5:][::-1]
            result["dominant_frequencies"] = [float(freqs[i]) for i in top_idx if i < len(freqs)]
        except Exception:
            result["dominant_frequencies"] = []

        return result

    def get_status(self) -> dict:
        return {
            "initialized": self._initialized,
            "fluidsynth_available": self._fluidsynth_available,
            "pretty_midi_available": self._pretty_midi_available,
            "music21_available": self._music21_available,
            "librosa_available": self._librosa_available,
            "soundfont_path": self.config.soundfont_path,
            "sample_rate": self.config.sample_rate,
            "render_paths": [
                path.value for path in RenderPath
                if (path != RenderPath.FLUIDSYNTH or self._fluidsynth_available)
                and (path != RenderPath.OFFLINE_NEURAL)
            ],
        }

    def shutdown(self):
        if self._synth is not None:
            try:
                self._synth.delete()
            except Exception as e:
                logger.debug(f"Synth cleanup failed: {e}")
            self._synth = None
        self._initialized = False
        self._fluidsynth_available = False


def get_hybrid_renderer() -> HybridRenderer:
    global _renderer_instance
    if _renderer_instance is None:
        with _RENDERER_LOCK:
            if _renderer_instance is None:
                _renderer_instance = HybridRenderer()
                _renderer_instance.initialize()
    return _renderer_instance


def reset_hybrid_renderer():
    global _renderer_instance
    with _RENDERER_LOCK:
        if _renderer_instance is not None:
            _renderer_instance.shutdown()
            _renderer_instance = None