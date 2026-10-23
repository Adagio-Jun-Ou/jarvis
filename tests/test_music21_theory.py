"""Tests for music21 theory engine."""

import pytest


class TestMusic21TheoryEngineInit:
    def test_import(self):
        from core.music21_theory_engine import Music21TheoryEngine
        assert Music21TheoryEngine is not None

    def test_available_flag(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        assert isinstance(engine.available, bool)

    def test_status(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        status = engine.get_status()
        assert "available" in status
        assert "backend" in status


class TestKeyValidation:
    def test_valid_major_keys(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        for key in ["C", "G", "D", "A", "E", "F", "Bb", "Eb"]:
            assert engine.validate_key(key), f"{key} should be valid"

    def test_validate_key_with_mode(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        assert engine.validate_key("C major"), "'C major' should be valid"
        assert engine.validate_key("G minor"), "'G minor' should be valid"

    def test_invalid_key(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        assert not engine.validate_key("Z")


class TestScaleGeneration:
    def test_c_major_scale(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        scale = engine.get_scale("C", "major")
        assert len(scale) >= 7

    def test_a_minor_scale(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        scale = engine.get_scale("A", "minor")
        assert len(scale) >= 5


class TestChordGeneration:
    def test_I_chord(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        chord = engine.get_chord("I", "C", "major")
        assert len(chord) >= 3

    def test_V_chord(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        chord = engine.get_chord("V", "C", "major")
        assert len(chord) >= 3

    def test_diatonic_chords(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        chords = engine.get_diatonic_chords("C", "major")
        assert len(chords) >= 7


class TestMelodicTransformations:
    def test_invert_melody(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        melody = [60, 62, 64, 65, 67]
        inverted = engine.invert_melody(melody, axis=64)
        assert len(inverted) == len(melody)
        assert inverted[0] == 68  # 2*64 - 60

    def test_transpose_melody(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        melody = [60, 62, 64]
        transposed = engine.transpose_melody(melody, 5)
        assert transposed == [65, 67, 69]

    def test_retrograde_melody(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        melody = [60, 62, 64, 67]
        retrograde = engine.retrograde_melody(melody)
        assert retrograde == [67, 64, 62, 60]

    def test_augment_melody(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        melody = [60, 62, 64]
        augmented = engine.augment_melody(melody, factor=2)
        assert augmented == [60, 60, 62, 62, 64, 64]

    def test_diminish_melody(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        melody = [60, 62, 64, 65, 67, 69]
        diminished = engine.diminish_melody(melody, factor=2)
        assert diminished == [60, 64, 67]


class TestVoiceLeading:
    def test_valid_voice_leading(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        result = engine.check_voice_leading([60, 64, 67], [62, 65, 69])
        assert "valid" in result
        assert "issues" in result

    def test_large_leap_detected(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        result = engine.check_voice_leading([60, 64, 67], [72, 76, 79])
        assert len(result["issues"]) > 0

    def test_empty_chords(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        result = engine.check_voice_leading([], [])
        assert result["valid"]


class TestCadences:
    def test_perfect_cadence(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        cadence = engine.generate_cadence("C", "perfect")
        assert len(cadence) == 2
        assert len(cadence[0]) >= 3  # V chord
        assert len(cadence[1]) >= 3  # I chord

    def test_plagal_cadence(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        cadence = engine.generate_cadence("C", "plagal")
        assert len(cadence) == 2

    def test_deceptive_cadence(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        cadence = engine.generate_cadence("C", "deceptive")
        assert len(cadence) == 2


class TestIntervals:
    def test_interval_names(self):
        from core.music21_theory_engine import get_theory_engine
        engine = get_theory_engine()
        assert engine.get_interval(60, 60) == "P1" or engine.get_interval(60, 60) == "d0"
        interval = engine.get_interval(60, 67)
        assert "5" in interval  # Perfect 5th


class TestSingletonPattern:
    def test_singleton(self):
        from core.music21_theory_engine import get_theory_engine
        e1 = get_theory_engine()
        e2 = get_theory_engine()
        assert e1 is e2

    def test_reset(self):
        from core.music21_theory_engine import get_theory_engine, reset_theory_engine
        e1 = get_theory_engine()
        reset_theory_engine()
        e2 = get_theory_engine()
        assert e1 is not e2