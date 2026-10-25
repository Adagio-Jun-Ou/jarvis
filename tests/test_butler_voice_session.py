"""Tests for Butler Voice and Butler Session modules."""
import tempfile
import threading
import time

import pytest

from core.butler_voice import (
    ButlerVoice, Situation, get_butler_voice,
    _GREETINGS, _SITUATION_PHRASES, _TRANSITIONS, _ACKNOWLEDGMENTS,
    _FILLER_REDUCTION,
)
from core.butler_session import ButlerSession, get_butler_session
from core.butler_personality import (
    ButlerPersonality, PersonaMode, Formality, Verbosity, EnergyLevel,
)


class TestButlerVoiceInitialization:
    def test_voice_creates(self):
        voice = ButlerVoice()
        state = voice.get_state()
        assert "phrases_generated" in state
        assert "unique_phrases" in state

    def test_singleton(self):
        v1 = get_butler_voice()
        v2 = get_butler_voice()
        assert v1 is v2


class TestButlerVoiceGreetings:
    def test_greeting_no_args(self):
        voice = ButlerVoice()
        greeting = voice.get_greeting()
        assert isinstance(greeting, str)
        assert len(greeting) > 0

    def test_greeting_butler_formal(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        bp._state.mode = PersonaMode.BUTLER
        bp._state.formality = Formality.FORMAL
        greeting = voice.get_greeting(personality=bp)
        assert isinstance(greeting, str)
        assert len(greeting) > 0

    def test_greeting_collaborator_casual(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        bp._state.mode = PersonaMode.COLLABORATOR
        bp._state.formality = Formality.CASUAL
        greeting = voice.get_greeting(personality=bp)
        assert isinstance(greeting, str)
        assert len(greeting) > 0

    def test_greeting_mentor_professional(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        bp._state.mode = PersonaMode.MENTOR
        bp._state.formality = Formality.PROFESSIONAL
        greeting = voice.get_greeting(personality=bp)
        assert isinstance(greeting, str)
        assert len(greeting) > 0

    def test_greeting_companion_intimate(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        bp._state.mode = PersonaMode.COMPANION
        bp._state.formality = Formality.INTIMATE
        greeting = voice.get_greeting(personality=bp)
        assert isinstance(greeting, str)
        assert len(greeting) > 0

    def test_greeting_variety(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        bp._state.mode = PersonaMode.BUTLER
        bp._state.formality = Formality.PROFESSIONAL
        greetings = set()
        for _ in range(20):
            g = voice.get_greeting(personality=bp)
            greetings.add(g)
        assert len(greetings) > 1  # Should get variety


class TestButlerVoiceSituations:
    def test_error_responding(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        for v in [Verbosity.TERSE, Verbosity.CONCISE, Verbosity.MODERATE, Verbosity.DETAILED]:
            bp._state.verbosity = v
            phrase = voice.get_phrase(Situation.ERROR_RESPONDING, bp)
            assert isinstance(phrase, str)
            assert len(phrase) > 0

    def test_success_confirm(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        bp._state.verbosity = Verbosity.TERSE
        phrase = voice.get_phrase(Situation.SUCCESS_CONFIRM, bp)
        assert isinstance(phrase, str)
        assert len(phrase) > 0

    def test_stuck_help(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        bp._state.verbosity = Verbosity.MODERATE
        phrase = voice.get_phrase(Situation.STUCK_HELP, bp)
        assert isinstance(phrase, str)
        assert len(phrase) > 0

    def test_frustration_empathy(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        phrase = voice.get_phrase(Situation.FRUSTRATION_EMPATHY, bp)
        assert isinstance(phrase, str)
        assert len(phrase) > 0

    def test_deep_work_respect(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        phrase = voice.get_phrase(Situation.DEEP_WORK_RESPECT, bp)
        assert isinstance(phrase, str)

    def test_break_suggest(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        phrase = voice.get_phrase(Situation.BREAK_SUGGEST, bp)
        assert isinstance(phrase, str)

    def test_curiosity_encourage(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        phrase = voice.get_phrase(Situation.CURIOSITY_ENCOURAGE, bp)
        assert isinstance(phrase, str)

    def test_all_situations_have_phrases(self):
        for situation in Situation:
            for verbosity in [Verbosity.TERSE, Verbosity.CONCISE, Verbosity.MODERATE, Verbosity.DETAILED]:
                phrases = _SITUATION_PHRASES.get(situation, {}).get(verbosity, [])
                assert len(phrases) > 0, f"No phrases for {situation.value}/{verbosity.value}"

    def test_verbosity_affects_phrase_length(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        terse_phrases = []
        detailed_phrases = []
        for _ in range(10):
            bp._state.verbosity = Verbosity.TERSE
            voice._phrase_history.clear()
            terse_phrases.append(voice.get_phrase(Situation.LEARNING_EXPLAIN, bp))
            bp._state.verbosity = Verbosity.DETAILED
            voice._phrase_history.clear()
            detailed_phrases.append(voice.get_phrase(Situation.LEARNING_EXPLAIN, bp))
        avg_terse = sum(len(p) for p in terse_phrases) / len(terse_phrases)
        avg_detailed = sum(len(p) for p in detailed_phrases) / len(detailed_phrases)
        assert avg_detailed >= avg_terse

    def test_avoids_recent_repeats(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        bp._state.mode = PersonaMode.BUTLER
        bp._state.formality = Formality.PROFESSIONAL
        phrases = []
        for _ in range(10):
            g = voice.get_greeting(personality=bp)
            phrases.append(g)
        unique_count = len(set(phrases))
        assert unique_count >= 2

    def test_history_clears_on_exhaustion(self):
        voice = ButlerVoice()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())
        bp._state.mode = PersonaMode.BUTLER
        bp._state.formality = Formality.FORMAL
        formal_greetings = _GREETINGS[PersonaMode.BUTLER][Formality.FORMAL]
        for _ in range(len(formal_greetings) + 1):
            g = voice.get_greeting(personality=bp)
            assert isinstance(g, str)
            assert len(g) > 0


class TestButlerSessionStart:
    def test_session_start(self):
        session = ButlerSession()
        greeting = session.start_session("hello")
        assert isinstance(greeting, str)
        assert len(greeting) > 0

    def test_session_state_after_start(self):
        session = ButlerSession()
        session.start_session()
        state = session.get_state()
        assert state["in_session"] is True
        assert state["session_count"] >= 1
        assert state["turn_count"] == 0

    def test_session_start_creates_new(self):
        session = ButlerSession()
        session.start_session("first")
        session.start_session("second")
        state = session.get_state()
        assert state["session_count"] >= 2

    def test_singleton(self):
        s1 = get_butler_session()
        s2 = get_butler_session()
        assert s1 is s2


class TestButlerSessionTracking:
    def test_record_turn(self):
        session = ButlerSession()
        session.start_session()
        session.record_turn("hello world", "neutral")
        session.record_turn("fix the bug", "frustrated")
        state = session.get_state()
        assert state["turn_count"] == 2

    def test_record_error(self):
        session = ButlerSession()
        session.start_session()
        session.record_error()
        session.record_error()
        state = session.get_state()
        assert state["errors"] == 2

    def test_record_tool_use(self):
        session = ButlerSession()
        session.start_session()
        session.record_tool_use()
        session.record_tool_use()
        session.record_tool_use()
        state = session.get_state()
        assert state["tools_used"] == 3

    def test_record_mode_change(self):
        session = ButlerSession()
        session.start_session()
        session.record_mode_change("butler")
        session.record_mode_change("mentor")
        session.record_mode_change("mentor")
        state = session.get_state()
        assert state["peak_mode"] == "mentor"

    def test_topics_tracked(self):
        session = ButlerSession()
        session.start_session()
        session.record_turn("debug the authentication module")
        session.record_turn("implement the caching feature")
        state = session.get_state()
        assert len(state["topics"]) > 0


class TestButlerSessionEnd:
    def test_end_session(self):
        session = ButlerSession()
        session.start_session()
        session.record_turn("test")
        session.end_session(reason="user_ended")
        state = session.get_state()
        assert state["in_session"] is False

    def test_end_session_saves_history(self):
        session = ButlerSession()
        session.start_session("test")
        session.record_turn("hello")
        session.record_turn("world")
        session.end_session()
        assert len(session._session_history) >= 1

    def test_productivity_assessment(self):
        session = ButlerSession()
        session.start_session()
        for _ in range(10):
            session.record_turn("productive work")
        session.end_session()
        last = session._session_history[-1]
        assert last.was_productive is True


class TestButlerSessionIdle:
    def test_no_idle_check_without_session(self):
        session = ButlerSession()
        result = session.check_idle()
        assert result is None

    def test_idle_check_with_recent_activity(self):
        session = ButlerSession()
        session.start_session()
        session._last_activity = time.time() - 10
        result = session.check_idle()
        assert result is None

    def test_idle_check_with_idle(self):
        session = ButlerSession()
        session.start_session()
        session._last_activity = time.time() - 3600
        session._last_idle_warning = 0
        result = session.check_idle()
        assert result is not None
        assert "idle" in result.lower() or "minutes" in result.lower()


class TestButlerSessionResume:
    def test_resume_context_empty(self):
        session = ButlerSession()
        ctx = session.get_resume_context()
        assert ctx == ""

    def test_resume_context_after_session(self):
        session = ButlerSession()
        session.start_session()
        for _ in range(6):
            session.record_turn("coding python")
        session.end_session()
        ctx = session.get_resume_context()
        assert len(ctx) > 0


class TestButlerSessionDecay:
    def test_memory_decay(self):
        session = ButlerSession()
        session.start_session()
        session.record_turn("test", "neutral")
        old_count = len(session._events)
        session.apply_memory_decay()
        assert len(session._events) <= old_count

    def test_decay_preserves_recent(self):
        session = ButlerSession()
        session.start_session()
        session.record_turn("recent", "neutral")
        count_before = len(session._events)
        session.apply_memory_decay()
        assert len(session._events) == count_before


class TestButlerSessionThreadSafety:
    def test_concurrent_access(self):
        session = ButlerSession()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    session.start_session("test")
                    session.record_turn("test", "neutral")
                    session.get_state()
                    session.end_session()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(errors) == 0