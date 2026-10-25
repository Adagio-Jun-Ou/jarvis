"""Tests for hybrid music renderer."""

import io
import os
import tempfile
import threading

import numpy as np
import pytest


class TestHybridRendererInit:
    def test_import(self):
        from core.music_hybrid_renderer import HybridRenderer
        assert HybridRenderer is not None

    def test_check_availability(self):
        from core.music_hybrid_renderer import HybridRenderer

        renderer = HybridRenderer()
        status = renderer.get_status()
        assert "initialized" in status
        assert "pretty_midi_available" in status
        assert "music21_available" in status
        assert "librosa_available" in status

    def test_soundfont_search(self):
        from core.music_hybrid_renderer import HybridRenderer

        paths = HybridRenderer._find_soundfonts()
        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_gm_programs_exist(self):
        from core.music_hybrid_renderer import HybridRenderer

        assert "piano" in HybridRenderer.GM_PROGRAMS
        assert "harpsichord" in HybridRenderer.GM_PROGRAMS
        assert "violin" in HybridRenderer.GM_PROGRAMS
        assert "trumpet" in HybridRenderer.GM_PROGRAMS
        assert "flute" in HybridRenderer.GM_PROGRAMS
        assert len(HybridRenderer.GM_PROGRAMS) >= 50

    def test_composer_instruments_exist(self):
        from core.music_hybrid_renderer import HybridRenderer

        assert "bach" in HybridRenderer.COMPOSER_INSTRUMENTS
        assert "mozart" in HybridRenderer.COMPOSER_INSTRUMENTS
        assert "chopin" in HybridRenderer.COMPOSER_INSTRUMENTS
        assert "default" in HybridRenderer.COMPOSER_INSTRUMENTS
        assert "melody" in HybridRenderer.COMPOSER_INSTRUMENTS["bach"]
        assert "harmony" in HybridRenderer.COMPOSER_INSTRUMENTS["bach"]
        assert "bass" in HybridRenderer.COMPOSER_INSTRUMENTS["bach"]


class TestCompositionToMidi:
    def test_simple_composition(self):
        from core.music_hybrid_renderer import HybridRenderer, get_hybrid_renderer

        renderer = get_hybrid_renderer()

        composition = {
            "melody": [[60, 80, 0.5], [64, 80, 0.5], [67, 80, 0.5]],
            "harmony": [[64, 60, 1.5]],
            "bass": [[48, 70, 1.5]],
        }

        midi_bytes = renderer._composition_to_midi(composition, bpm=120, key="C")
        if renderer._pretty_midi_available:
            assert len(midi_bytes) > 0
            assert midi_bytes[:4] == b'MThd'
        else:
            assert len(midi_bytes) >= 0

    def test_composition_with_dict_notes(self):
        from core.music_hybrid_renderer import get_hybrid_renderer

        renderer = get_hybrid_renderer()

        composition = {
            "melody": [
                {"pitch": 60, "velocity": 80, "start": 0.0, "duration": 0.5},
                {"pitch": 64, "velocity": 75, "start": 0.5, "duration": 0.5},
            ],
        }

        midi_bytes = renderer._composition_to_midi(composition, bpm=100, key="C")
        if renderer._pretty_midi_available:
            assert len(midi_bytes) > 0

    def test_flat_composition_with_channels(self):
        from core.music_hybrid_renderer import get_hybrid_renderer

        renderer = get_hybrid_renderer()

        composition = {
            "notes": [
                {"pitch": 72, "start": 0.0, "end": 0.5, "velocity": 90, "channel": 0},
                {"pitch": 60, "start": 0.0, "end": 1.0, "velocity": 55, "channel": 1},
                {"pitch": 64, "start": 0.0, "end": 1.0, "velocity": 50, "channel": 1},
                {"pitch": 48, "start": 0.0, "end": 1.0, "velocity": 70, "channel": 2},
            ],
            "bpm": 120,
            "key": "C",
            "instruments": {
                "melody": "piano",
                "harmony": "strings_ensemble_1",
                "bass": "bass_acoustic",
            },
        }

        midi_bytes = renderer._composition_to_midi(composition)
        if renderer._pretty_midi_available:
            assert len(midi_bytes) > 100, "Flat composition should produce substantial MIDI"
            assert midi_bytes[:4] == b'MThd'

    def test_flat_composition_produces_notes(self):
        from core.music_hybrid_renderer import get_hybrid_renderer

        renderer = get_hybrid_renderer()
        if not renderer._pretty_midi_available:
            pytest.skip("pretty_midi not available")

        composition = {
            "notes": [
                {"pitch": 60, "start": 0.0, "end": 0.5, "velocity": 80, "channel": 0},
                {"pitch": 64, "start": 0.5, "end": 1.0, "velocity": 80, "channel": 0},
            ],
            "bpm": 120,
        }

        midi_bytes = renderer._composition_to_midi(composition)
        assert len(midi_bytes) > 0
        note_ons = sum(1 for i in range(len(midi_bytes)) if i + 2 < len(midi_bytes) and midi_bytes[i + 1] & 0xF0 == 0x90)
        assert note_ons > 0, "Flat composition should produce note-on events"

    def test_flat_composition_renders_to_audio(self):
        from core.music_hybrid_renderer import get_hybrid_renderer

        renderer = get_hybrid_renderer()
        if not renderer._fluidsynth_available:
            pytest.skip("FluidSynth not available")

        composition = {
            "notes": [
                {"pitch": 72, "start": 0.0, "end": 0.5, "velocity": 90, "channel": 0},
                {"pitch": 71, "start": 0.5, "end": 0.75, "velocity": 75, "channel": 0},
                {"pitch": 69, "start": 0.75, "end": 1.0, "velocity": 80, "channel": 0},
                {"pitch": 67, "start": 1.0, "end": 1.5, "velocity": 85, "channel": 0},
            ],
            "bpm": 100,
            "key": "C",
        }

        midi_bytes = renderer._composition_to_midi(composition)
        result = renderer.render_midi_bytes(midi_bytes, composer="chopin")
        assert result.audio is not None
        assert len(result.audio) > 0
        assert result.duration_seconds > 0.5
        rms = np.sqrt(np.mean(result.audio ** 2))
        assert rms > 0.001, "Rendered audio should not be silence"

    def test_empty_composition(self):
        from core.music_hybrid_renderer import get_hybrid_renderer

        renderer = get_hybrid_renderer()

        composition = {}
        midi_bytes = renderer._composition_to_midi(composition)
        assert isinstance(midi_bytes, (bytes, bytearray))


class TestRenderPath:
    def test_render_result_dataclass(self):
        from core.music_hybrid_renderer import RenderResult, RenderPath

        result = RenderResult(
            audio=np.zeros(44100, dtype=np.float32),
            sample_rate=44100,
            duration_seconds=1.0,
            render_path=RenderPath.SYNTHESIS,
        )
        assert result.sample_rate == 44100
        assert result.duration_seconds == 1.0
        assert result.render_path == RenderPath.SYNTHESIS

    def test_render_path_enum(self):
        from core.music_hybrid_renderer import RenderPath

        assert RenderPath.FLUIDSYNTH.value == "fluidsynth"
        assert RenderPath.SYNTHESIS.value == "synthesis"
        assert RenderPath.OFFLINE_NEURAL.value == "offline_neural"


class TestAudioAnalysis:
    def test_analyze_silence(self):
        from core.music_hybrid_renderer import get_hybrid_renderer

        renderer = get_hybrid_renderer()
        silence = np.zeros(44100, dtype=np.float32)
        analysis = renderer.analyze_audio(silence, sr=44100)

        assert "duration" in analysis
        assert "rms_energy" in analysis
        assert "peak_amplitude" in analysis
        assert analysis["duration"] == pytest.approx(1.0, abs=0.01)
        assert analysis["rms_energy"] == pytest.approx(0.0, abs=0.01)

    def test_analyze_sine_wave(self):
        from core.music_hybrid_renderer import get_hybrid_renderer

        renderer = get_hybrid_renderer()
        t = np.linspace(0, 1, 44100, dtype=np.float32)
        sine = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        analysis = renderer.analyze_audio(sine, sr=44100)
        assert analysis["duration"] == pytest.approx(1.0, abs=0.01)
        assert analysis["peak_amplitude"] > 0.3
        assert len(analysis.get("dominant_frequencies", [])) > 0


class TestSaveAudio:
    def test_save_wav(self):
        from core.music_hybrid_renderer import get_hybrid_renderer

        renderer = get_hybrid_renderer()
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, 22050)).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            renderer._save_audio(audio, output_path)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            os.unlink(output_path)


class TestSingletonPattern:
    def test_get_hybrid_renderer_singleton(self):
        from core.music_hybrid_renderer import get_hybrid_renderer

        r1 = get_hybrid_renderer()
        r2 = get_hybrid_renderer()
        assert r1 is r2

    def test_reset_hybrid_renderer(self):
        from core.music_hybrid_renderer import get_hybrid_renderer, reset_hybrid_renderer

        r1 = get_hybrid_renderer()
        reset_hybrid_renderer()
        r2 = get_hybrid_renderer()
        assert r1 is not r2

    def test_thread_safety(self):
        from core.music_hybrid_renderer import get_hybrid_renderer, reset_hybrid_renderer

        reset_hybrid_renderer()
        results = []

        def get_renderer():
            r = get_hybrid_renderer()
            results.append(id(r))

        threads = [threading.Thread(target=get_renderer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(set(results)) == 1


class TestFluidSynthConfig:
    def test_default_config(self):
        from core.music_hybrid_renderer import FluidSynthConfig

        config = FluidSynthConfig()
        assert config.sample_rate == 44100
        assert config.gain == 0.8
        assert config.soundfont_path is None

    def test_custom_config(self):
        from core.music_hybrid_renderer import FluidSynthConfig

        config = FluidSynthConfig(
            sample_rate=48000,
            gain=0.9,
            soundfont_path="/test/path.sf2",
        )
        assert config.sample_rate == 48000
        assert config.gain == 0.9
        assert config.soundfont_path == "/test/path.sf2"


class TestFluidSynthRendering:
    @pytest.mark.requires_host_env
    def test_fluidsynth_renders_midi(self):
        from core.music_hybrid_renderer import get_hybrid_renderer, reset_hybrid_renderer, RenderPath
        import pretty_midi

        reset_hybrid_renderer()
        renderer = get_hybrid_renderer()
        status = renderer.get_status()

        if not status["fluidsynth_available"]:
            pytest.skip("FluidSynth not available")

        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        piano = pretty_midi.Instrument(program=0, name="Piano")
        for i, pitch in enumerate([60, 62, 64, 65, 67]):
            note = pretty_midi.Note(
                velocity=80, pitch=pitch, start=i * 0.5, end=i * 0.5 + 0.4
            )
            piano.notes.append(note)
        pm.instruments.append(piano)

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            pm.write(f.name)
            midi_bytes = open(f.name, "rb").read()

        result = renderer.render_midi_bytes(midi_bytes, composer="chopin")
        assert result.render_path == RenderPath.FLUIDSYNTH
        assert result.audio is not None
        assert len(result.audio) > 0
        assert result.duration_seconds > 2.0
        assert result.sample_rate == 44100

    @pytest.mark.requires_host_env
    def test_composer_instrument_mapping(self):
        from core.music_hybrid_renderer import get_hybrid_renderer, reset_hybrid_renderer, RenderPath
        import pretty_midi

        reset_hybrid_renderer()
        renderer = get_hybrid_renderer()

        if not renderer.get_status()["fluidsynth_available"]:
            pytest.skip("FluidSynth not available")

        for composer in ["bach", "mozart", "chopin"]:
            pm = pretty_midi.PrettyMIDI(initial_tempo=100)
            melody = pretty_midi.Instrument(program=0, name="Melody")
            melody.notes.append(pretty_midi.Note(velocity=70, pitch=60, start=0.0, end=1.0))
            pm.instruments.append(melody)

            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                pm.write(f.name)
                midi_bytes = open(f.name, "rb").read()

            result = renderer.render_midi_bytes(midi_bytes, composer=composer)
            assert result.render_path == RenderPath.FLUIDSYNTH
            assert len(result.audio) > 0
    def test_render_path_pretty_midi_exists(self):
        from core.music_hybrid_renderer import RenderPath
        assert hasattr(RenderPath, 'PRETTY_MIDI')
        assert RenderPath.PRETTY_MIDI.value == "pretty_midi"
