"""Tests for music tool transformations."""
import pytest
import numpy as np


class TestThemeTransformations:
    def test_invert_theme(self):
        from tools.music import MusicGenerator
        gen = MusicGenerator()
        theme = ["C4", "E4", "G4", "C5"]
        inverted = gen._invert_theme(theme)
        assert inverted[0] == "C4", "First note should stay the same (pivot)"
        assert len(inverted) == len(theme), "Inversion should preserve note count"
        assert inverted != theme, "Inversion should change at least some notes"

    def test_invert_empty_theme(self):
        from tools.music import MusicGenerator
        gen = MusicGenerator()
        assert gen._invert_theme([]) == []
        assert gen._invert_theme(["C4"]) == ["C4"]

    def test_sequence_theme(self):
        from tools.music import MusicGenerator
        gen = MusicGenerator()
        theme = ["C4", "E4", "G4"]
        seq = gen._sequence_theme(theme, intervals=2)
        assert len(seq) > len(theme), "Sequence should be longer than original"

    def test_diminish_theme(self):
        from tools.music import MusicGenerator
        gen = MusicGenerator()
        theme = ["C4", "G4", "C5"]
        dim = gen._diminish_theme(theme)
        assert len(dim) == len(theme), "Diminution should preserve note count"
        assert dim[0] == theme[0], "First note (pivot) should stay"

    def test_augment_theme(self):
        from tools.music import MusicGenerator
        gen = MusicGenerator()
        theme = ["C4", "E4", "G4"]
        aug = gen._augment_theme(theme)
        assert len(aug) == len(theme), "Augmentation should preserve note count"
        assert aug[0] == theme[0], "First note (pivot) should stay"

    def test_note_parsing(self):
        from tools.music import MusicGenerator
        gen = MusicGenerator()
        assert gen._parse_note("C4") == ("C", 4)
        assert gen._parse_note("F#5") == ("F#", 5)
        assert gen._parse_note("Bb3") == ("Bb", 3)

    def test_note_to_midi_roundtrip(self):
        from tools.music import MusicGenerator
        gen = MusicGenerator()
        for note in ["C4", "D4", "E4", "F4", "G4", "A4", "B4"]:
            midi = gen._note_to_midi(note)
            reconstructed = gen._midi_to_note(midi)
            assert reconstructed == note, f"Roundtrip failed: {note} → {midi} → {reconstructed}"

    def test_note_to_midi_values(self):
        from tools.music import MusicGenerator
        gen = MusicGenerator()
        assert gen._note_to_midi("C4") == 60
        assert gen._note_to_midi("A4") == 69
        assert gen._note_to_midi("C5") == 72


class TestChordToneBugFixes:
    """Test that chord tone parsing returns correct scale degrees."""

    def test_basic_major_chords(self):
        from core.llm_composer import _get_chord_tones
        assert _get_chord_tones('I', 'major') == [0, 2, 4]
        assert _get_chord_tones('IV', 'major') == [3, 5, 0]
        assert _get_chord_tones('V', 'major') == [4, 6, 1]

    def test_minor_chords(self):
        from core.llm_composer import _get_chord_tones
        result = _get_chord_tones('ii', 'major')
        assert set(d % 7 for d in result) == {1, 3, 5}
        result = _get_chord_tones('vi', 'major')
        assert set(d % 7 for d in result) == {0, 2, 5}

    def test_seventh_chords(self):
        from core.llm_composer import _get_chord_tones
        result = _get_chord_tones('V7', 'major')
        assert set(d % 7 for d in result) == {1, 3, 4, 6}
        result = _get_chord_tones('I7', 'major')
        assert set(d % 7 for d in result) == {0, 2, 4, 6}

    def test_flat_chord_bVI(self):
        from core.llm_composer import _get_chord_tones
        result = _get_chord_tones('bVI', 'major')
        assert set(d % 7 for d in result) == {0, 3, 5}, f"bVI should be [5,0,3] mod7, got {result}"

    def test_flat_chord_bII(self):
        from core.llm_composer import _get_chord_tones
        result = _get_chord_tones('bII', 'major')
        assert set(d % 7 for d in result) == {1, 3, 5}, f"bII should be Neapolitan, got {result}"

    def test_neapolitan_N6(self):
        from core.llm_composer import _get_chord_tones
        result = _get_chord_tones('N6', 'major')
        assert set(d % 7 for d in result) == {1, 3, 6}, f"N6 should be Neapolitan 6th, got {result}"

    def test_secondary_dominant_V7_over_V(self):
        from core.llm_composer import _get_chord_tones
        result = _get_chord_tones('V7/V', 'major')
        assert len(result) >= 3, f"V7/V should have at least 3 tones, got {result}"

    def test_inversion_I6(self):
        from core.llm_composer import _get_chord_tones
        result = _get_chord_tones('I6', 'major')
        assert 2 in result, f"I6 first inversion should contain 3rd (degree 2), got {result}"

    def test_diminished_vii(self):
        from core.llm_composer import _get_chord_tones
        result = _get_chord_tones('vii°', 'major')
        assert set(d % 7 for d in result) == {1, 3, 6}, f"vii° should be diminished, got {result}"

    def test_alt_chord_V7alt(self):
        from core.llm_composer import _get_chord_tones
        result = _get_chord_tones('V7alt', 'major')
        assert len(result) >= 3, f"V7alt should have tones, got {result}"
        assert set(d % 7 for d in result) == {1, 3, 4, 6}, f"V7alt should equal V7 tones, got {result}"


class TestChordNameStripping:
    """Test that chord names are correctly stripped for lookup."""

    def test_V7alt_stripping(self):
        chord = 'V7alt'
        chord_raw = chord.split('/')[0]
        chord_without_dim = chord_raw.split('°')[0]
        if 'maj7' in chord_without_dim:
            result = chord_without_dim.replace('maj7', '').replace('7', '')
        elif 'alt' in chord_without_dim:
            result = chord_without_dim.replace('alt', '').replace('7', '')
        else:
            import re
            result = re.sub(r'[764](?:/\d+)?$', '', chord_without_dim)
        assert result == 'V', f"V7alt should strip to V, got '{result}'"

    def test_II7_stripping(self):
        import re
        chord = 'ii7'
        chord_raw = chord.split('/')[0]
        chord_without_dim = chord_raw.split('°')[0]
        result = re.sub(r'[764](?:/\d+)?$', '', chord_without_dim)
        assert result == 'ii', f"ii7 should strip to ii, got '{result}'"

    def test_I6_stripping(self):
        import re
        chord = 'I6'
        chord_raw = chord.split('/')[0]
        chord_without_dim = chord_raw.split('°')[0]
        result = re.sub(r'[764](?:/\d+)?$', '', chord_without_dim)
        assert result == 'I', f"I6 should strip to I, got '{result}'"

    def test_underscore_suffix_stripping(self):
        import re
        for chord_name in ['I_1', 'V7_2', 'ii°_3']:
            clean = re.sub(r'_\d+$', '', chord_name)
            assert '_' not in clean, f"Should strip _N suffix: {chord_name} -> {clean}"

    def test_bVI_preserved(self):
        chord = 'bVI'
        chord_raw = chord.split('/')[0]
        chord_without_dim = chord_raw.split('°')[0]
        assert chord_without_dim == 'bVI', f"bVI should be preserved, got '{chord_without_dim}'"


class TestBassLineGeneration:
    """Test bass line generation in llm_composer."""

    def test_bass_pitch_in_range(self):
        from core.llm_composer import _generate_bass_line, _generate_functional_harmony
        harmony = _generate_functional_harmony(['I', 'IV', 'V', 'I'], 0, 'major', 4)
        bass = _generate_bass_line(harmony, 0)
        for note in bass:
            assert 36 <= note['pitch'] <= 60, f"Bass pitch {note['pitch']} outside range [36, 60]"

    def test_bass_has_duration(self):
        from core.llm_composer import _generate_bass_line, _generate_functional_harmony
        harmony = _generate_functional_harmony(['I', 'IV', 'V', 'I'], 0, 'major', 4)
        bass = _generate_bass_line(harmony, 0)
        for note in bass:
            assert note['duration'] > 0, f"Bass note has zero/negative duration: {note['duration']}"

    def test_bass_has_velocity(self):
        from core.llm_composer import _generate_bass_line, _generate_functional_harmony
        harmony = _generate_functional_harmony(['I', 'IV', 'V', 'I'], 0, 'major', 4)
        bass = _generate_bass_line(harmony, 0)
        for note in bass:
            assert 1 <= note['velocity'] <= 127, f"Bass velocity {note['velocity']} outside range"

    def test_bass_different_keys(self):
        from core.llm_composer import _generate_bass_line, _generate_functional_harmony
        for key_root in [0, 2, 5, 7]:
            harmony = _generate_functional_harmony(['I', 'IV', 'V', 'I'], key_root, 'major', 4)
            bass = _generate_bass_line(harmony, key_root)
            for note in bass:
                assert 36 <= note['pitch'] <= 60, f"Key {key_root}: bass pitch {note['pitch']} out of range"


class TestToneShaping:
    """Test tone shaping in hybrid renderer."""

    def test_tone_shaping_applies_all_stages(self):
        from core.music_hybrid_renderer import HybridRenderer
        r = HybridRenderer()
        audio = np.random.uniform(-0.5, 0.5, 88200).astype(np.float32)
        result = r._apply_tone_shaping(audio, 44100)
        assert result.shape == audio.shape
        assert result.dtype == np.float32
        assert not np.allclose(result, audio), "Tone shaping should modify the audio"

    def test_tone_shaping_short_audio_passthrough(self):
        from core.music_hybrid_renderer import HybridRenderer
        r = HybridRenderer()
        short = np.random.uniform(-0.5, 0.5, 1000).astype(np.float32)
        result = r._apply_tone_shaping(short, 44100)
        assert np.array_equal(result, short), "Short audio should pass through unchanged"

    def test_tone_shaping_preserves_peaks(self):
        from core.music_hybrid_renderer import HybridRenderer
        r = HybridRenderer()
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 88200)).astype(np.float32) * 0.8
        result = r._apply_tone_shaping(audio, 44100)
        peak = np.max(np.abs(result))
        assert peak <= 1.0, f"Peak {peak} exceeds 1.0 after tone shaping"
        assert peak > 0.01, "Audio should not be silent after tone shaping"


class TestCompositionRendering:
    """Test composition-to-audio rendering pipeline."""

    def test_harmony_chord_dict_preserved(self):
        from core.music_hybrid_renderer import get_hybrid_renderer
        from core.midi_humanizer import apply_humanization_to_composition

        phrases = [{
            'melody_notes': [{'pitch': 72, 'beat': 0, 'duration': 2, 'velocity': 80}],
            'harmony_notes': [
                {'chord': 'C', 'beat': 0, 'duration': 2, 'velocity': 55},
                {'chord': 'G', 'beat': 2, 'duration': 2, 'velocity': 55},
            ],
            'bass_notes': [{'pitch': 48, 'beat': 0, 'duration': 2, 'velocity': 50}],
            'dynamics': 'mf',
            'duration': 4.0,
            'section': 'A',
        }]
        result = apply_humanization_to_composition(phrases, 'classical', 90)
        assert len(result) == 1
        for h in result[0]['harmony_notes']:
            assert 'chord' in h, f"Harmony note missing 'chord' key: {h}"

    def test_empty_tracks_render(self):
        from core.music_hybrid_renderer import get_hybrid_renderer
        renderer = get_hybrid_renderer()
        comp = {
            'bpm': 120, 'key': 'C',
            'tracks': [
                {'name': 'melody', 'notes': []},
                {'name': 'harmony', 'notes': []},
                {'name': 'bass', 'notes': []},
            ]
        }
        result = renderer.render_composition(comp)
        assert result.audio is not None
        assert len(result.audio) > 0

    @pytest.mark.requires_host_env
    def test_chord_with_underscore_renders(self):
        from core.music_hybrid_renderer import get_hybrid_renderer
        renderer = get_hybrid_renderer()
        comp = {
            'bpm': 110, 'key': 'C',
            'tracks': [
                {'name': 'melody', 'notes': [{'pitch': 60, 'beat': 0, 'duration': 2, 'velocity': 80}]},
                {'name': 'harmony', 'notes': [{'chord': 'I_1', 'beat': 0, 'duration': 2, 'velocity': 55}]},
                {'name': 'bass', 'notes': [{'pitch': 36, 'beat': 0, 'duration': 2, 'velocity': 50}]},
            ]
        }
        midi_bytes = renderer._composition_to_midi(comp, 110, 'C')
        assert len(midi_bytes) > 0, "Chord with underscore should still render"

    @pytest.mark.requires_host_env
    def test_museScoresoundfont_is_primary(self):
        from core.music_hybrid_renderer import HybridRenderer
        paths = HybridRenderer._find_soundfonts()
        assert paths[0] == "/usr/share/sounds/sf2/MuseScore_General.sf2", \
            "MuseScore General should be primary soundfont"
        # Verify it actually exists
        import os
        assert os.path.exists(paths[0]), f"MuseScore soundfont not found at {paths[0]}"


class TestMelodyGeneration:
    """Test melody generation edge cases."""

    def test_melody_notes_in_range(self):
        from core.llm_composer import _generate_smooth_melody, SCALES
        scale = SCALES['major']
        for emotion in ['peaceful', 'joyful', 'dramatic']:
            melody = _generate_smooth_melody(
                scale=scale, key_root=0, mode='major',
                num_bars=4, pitch_range=(60, 72),
                emotion=emotion, style='classical',
                section='A', phrase_position=0.0,
            )
            for note in melody:
                assert 60 <= note['pitch'] <= 72, \
                    f"Note pitch {note['pitch']} outside range [60,72] for {emotion}"

    def test_melody_has_positive_duration(self):
        from core.llm_composer import _generate_smooth_melody, SCALES
        scale = SCALES['major']
        melody = _generate_smooth_melody(
            scale=scale, key_root=0, mode='major',
            num_bars=4, pitch_range=(60, 84),
            emotion='peaceful', style='classical',
            section='A', phrase_position=0.0,
        )
        for note in melody:
            assert note['duration'] >= 0.5, f"Duration {note['duration']} too short"

    def test_melody_velocity_in_range(self):
        from core.llm_composer import _generate_smooth_melody, SCALES
        scale = SCALES['major']
        melody = _generate_smooth_melody(
            scale=scale, key_root=0, mode='major',
            num_bars=4, pitch_range=(60, 84),
            emotion='peaceful', style='classical',
            section='A', phrase_position=0.0,
        )
        for note in melody:
            assert 1 <= note['velocity'] <= 127, \
                f"Velocity {note['velocity']} out of range"


class TestHarmonicProgression:
    """Test harmonic progression generation."""

    def test_progression_has_correct_length(self):
        from core.llm_composer import _generate_functional_harmony
        for n in [4, 8]:
            prog = _generate_functional_harmony(['I', 'IV', 'V', 'I'], 0, 'major', n)
            assert len(prog) == n, f"Expected {n} chords, got {len(prog)}"

    def test_progression_has_beat_positions(self):
        from core.llm_composer import _generate_functional_harmony
        prog = _generate_functional_harmony(['I', 'IV', 'V', 'I'], 0, 'major', 4)
        for h in prog:
            assert 'beat' in h, f"Harmony missing 'beat': {h}"
            assert 'duration' in h, f"Harmony missing 'duration': {h}"
            assert 'chord' in h, f"Harmony missing 'chord': {h}"

    def test_progression_beats_are_monotonic(self):
        from core.llm_composer import _generate_functional_harmony
        prog = _generate_functional_harmony(['I', 'IV', 'V', 'I'], 0, 'major', 4)
        beats = [h['beat'] for h in prog]
        for i in range(1, len(beats)):
            assert beats[i] >= beats[i-1], f"Beats not monotonic: {beats}"

    def test_progression_duration_sum_reasonable(self):
        from core.llm_composer import _generate_functional_harmony
        prog = _generate_functional_harmony(['I', 'IV', 'V', 'I'], 0, 'major', 4)
        total_dur = sum(h['duration'] for h in prog)
        assert 8 <= total_dur <= 32, f"Total duration {total_dur} unreasonable for 4 chords"
