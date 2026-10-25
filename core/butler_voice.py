"""Butler Voice — Context-aware phrasing that makes Jarvis sound genuinely different.

A butler doesn't just decide "be terse" — it knows *how* to be terse. The difference
between a butler and a chatbot is that a butler varies its PHRASING, not just its length.

This module provides:
1. **Greeting templates** — how Jarvis opens a response (formal vs. casual vs. direct)
2. **Transition phrases** — how Jarvis moves between topics (smooth vs. abrupt)
3. **Affirmation vocabulary** — how Jarvis acknowledges understanding
4. **Filler reduction** — which filler words to remove based on verbosity
5. **Closing templates** — how Jarvis ends a response (offering more vs. stopping)
6. **Situational phrasing** — specific templates for common situations (errors, success, learning)
"""
import logging
import random
import threading
import time
from collections import deque
from enum import Enum
from typing import Optional

from core.butler_personality import (
    ButlerPersonality, PersonaMode, Formality, Verbosity, EnergyLevel,
    get_butler_personality, _formality_from_value,
)

logger = logging.getLogger(__name__)

_voice: Optional["ButlerVoice"] = None
_voice_lock = threading.Lock()


class Situation(Enum):
    ERROR_RESPONDING = "error_responding"
    SUCCESS_CONFIRM = "success_confirm"
    PARTIAL_SUCCESS = "partial_success"
    STUCK_HELP = "stuck_help"
    LEARNING_EXPLAIN = "learning_explain"
    PROACTIVE_SUGGEST = "proactive_suggest"
    BREAK_SUGGEST = "break_suggest"
    DEEP_WORK_RESPECT = "deep_work_respect"
    FRUSTRATION_EMPATHY = "frustration_empathy"
    FLOW_PROTECT = "flow_protect"
    CURIOSITY_ENCOURAGE = "curiosity_encourage"
    GREETING_RETURN = "greeting_return"
    GREETING_MORNING = "greeting_morning"
    GREETING_EVENING = "greeting_evening"
    GREETING_FIRST = "greeting_first"
    CLOSING_OFFER = "closing_offer"
    CLOSING_DIRECT = "closing_direct"


_GREETINGS: dict[PersonaMode, dict[Formality, list[str]]] = {
    PersonaMode.BUTLER: {
        Formality.FORMAL: [
            "Certainly.", "Of course.", "Very well.", "Right away.",
            "As requested.", "I shall proceed.", "Allow me.", "At once.",
        ],
        Formality.PROFESSIONAL: [
            "Got it.", "Understood.", "On it.", "Sure thing.",
            "I'll handle that.", "Let me take care of this.",
        ],
        Formality.CASUAL: [
            "Yep.", "Sure.", "Gotcha.", "On it.",
            "Let me look into that.", "I'll sort this out.",
        ],
        Formality.INTIMATE: [
            "Of course.", "Absolutely.", "Consider it done.",
            "I'm on it.", "Leave this to me.",
        ],
    },
    PersonaMode.COLLABORATOR: {
        Formality.FORMAL: [
            "Let's work through this together.", "I have some ideas here.",
            "Shall we explore this?", "I'd suggest we approach it this way.",
        ],
        Formality.PROFESSIONAL: [
            "Here's what I'm thinking.", "Let me show you an approach.",
            "I see a few options here.", "What if we tried this?",
        ],
        Formality.CASUAL: [
            "So here's the deal.", "I've got a hunch about this.",
            "Let's try something.", "What about this approach?",
        ],
        Formality.INTIMATE: [
            "I've been thinking about this.", "Here's my take.",
            "Let me walk you through my thinking.", "I've got an idea.",
        ],
    },
    PersonaMode.MENTOR: {
        Formality.FORMAL: [
            "An excellent question. Let me explain.", "Let me walk you through this.",
            "This is a great opportunity to learn.", "Here's how this works.",
        ],
        Formality.PROFESSIONAL: [
            "Great question!", "So here's how this works.",
            "Let me break this down.", "The key insight here is",
        ],
        Formality.CASUAL: [
            "Oh, interesting question!", "So the cool thing here is",
            "Here's the thing about this —", "Let me show you a trick.",
        ],
        Formality.INTIMATE: [
            "I love that you're curious about this.", "This is actually really fascinating.",
            "Let's dig into this together.", "The really interesting part is",
        ],
    },
    PersonaMode.COMPANION: {
        Formality.FORMAL: [
            "I appreciate you sharing that.", "That's a thoughtful perspective.",
            "I'm here whenever you'd like to explore this.", "Take your time.",
        ],
        Formality.PROFESSIONAL: [
            "That's interesting!", "I hear you.",
            "That makes sense.", "I'm with you on that.",
        ],
        Formality.CASUAL: [
            "Nice!", "Oh, that's cool.", "I feel you.",
            "Yeah, I get that.", "Honestly, same.",
        ],
        Formality.INTIMATE: [
            "I'm really glad you brought that up.", "That resonates with me.",
            "I've been thinking about that too.", "You know what's funny?",
        ],
    },
}

_SITUATION_PHRASES: dict[Situation, dict[Verbosity, list[str]]] = {
    Situation.ERROR_RESPONDING: {
        Verbosity.TERSE: ["Error.", "Failed.", "Can't do that.", "Problem."],
        Verbosity.CONCISE: ["Hit an error.", "That failed.", "Something went wrong.", "Couldn't complete that."],
        Verbosity.MODERATE: [
            "I ran into an issue here.", "Something went wrong with that.",
            "That didn't work as expected.", "I'm hitting an error on this.",
        ],
        Verbosity.DETAILED: [
            "I encountered an error while processing that. Let me explain what happened.",
            "That didn't work. Here's what went wrong and what we can try instead.",
            "I'm seeing an issue here — it looks like the operation failed because",
        ],
    },
    Situation.SUCCESS_CONFIRM: {
        Verbosity.TERSE: ["Done.", "Fixed.", "Works.", "Complete."],
        Verbosity.CONCISE: ["That's done.", "Looks good now.", "All set.", "Fixed that up."],
        Verbosity.MODERATE: [
            "That's taken care of.", "Everything looks good now.",
            "All done — let me know if anything else comes up.",
        ],
        Verbosity.DETAILED: [
            "That's been resolved. Here's what I changed and why.",
            "All set! Let me walk you through what I did.",
            "Done — I'll explain the details so you know what happened.",
        ],
    },
    Situation.PARTIAL_SUCCESS: {
        Verbosity.TERSE: ["Mostly done.", "Partial.", "Almost."],
        Verbosity.CONCISE: ["Got most of it.", "Almost there — one thing left.", "Mostly working."],
        Verbosity.MODERATE: [
            "Most of this is working, but there's one issue remaining.",
            "That's mostly done. Let me flag what still needs attention.",
        ],
        Verbosity.DETAILED: [
            "I've made good progress, but there are still some items to address. Let me explain.",
            "This is working for the most part. Here's what's done and what still needs work.",
        ],
    },
    Situation.STUCK_HELP: {
        Verbosity.TERSE: ["Try this:", "Alternative:", "Different approach:"],
        Verbosity.CONCISE: ["Here's another way:", "Try this instead:", "Maybe this approach:"],
        Verbosity.MODERATE: [
            "I notice this isn't working. Let me suggest a different approach.",
            "That method seems stuck. Here's what might work better.",
        ],
        Verbosity.DETAILED: [
            "I can see this approach isn't producing results. Let me explain why and suggest an alternative.",
            "It looks like we're going in circles here. I think the issue is — and a better approach would be",
        ],
    },
    Situation.LEARNING_EXPLAIN: {
        Verbosity.TERSE: ["Key point:", "The main idea:", "In short:"],
        Verbosity.CONCISE: ["Here's the key idea:", "The core concept is:", "In essence:"],
        Verbosity.MODERATE: [
            "Here's how this works:", "The important thing to understand is",
            "Let me connect a few concepts for you.",
        ],
        Verbosity.DETAILED: [
            "Great question! Let me explain this in depth. The key concepts are",
            "This is actually really interesting. Here's the full picture:",
            "Let me walk you through this step by step, starting from the basics.",
        ],
    },
    Situation.PROACTIVE_SUGGEST: {
        Verbosity.TERSE: ["Suggestion:", "BTW:", "Also:"],
        Verbosity.CONCISE: ["I notice something:", "A heads up:", "You might want to"],
        Verbosity.MODERATE: [
            "I wanted to point out something I noticed.",
            "Here's something that might be helpful.",
            "I have a suggestion that could save time here.",
        ],
        Verbosity.DETAILED: [
            "I've been watching the pattern here and I think there's an opportunity to improve things.",
            "Based on what I've observed, I'd like to suggest an approach that might be more efficient.",
        ],
    },
    Situation.BREAK_SUGGEST: {
        Verbosity.TERSE: ["Break?", "Take 5?", "Pause?"],
        Verbosity.CONCISE: ["Maybe take a short break?", "Consider stepping away briefly.", "A quick pause might help."],
        Verbosity.MODERATE: [
            "You've been at this for a while. A short break might help clear your head.",
            "It might be worth stepping away for a few minutes — sometimes that helps.",
        ],
        Verbosity.DETAILED: [
            "I've noticed you've been working intensively for over an hour. Research shows that short breaks improve focus and decision-making. Would you like to take a few minutes?",
        ],
    },
    Situation.DEEP_WORK_RESPECT: {
        Verbosity.TERSE: ["(Deep work — minimal interruptions)", "(In flow mode)", "(Staying quiet)"],
        Verbosity.CONCISE: ["I'll keep this brief — you're in deep work mode.", "Staying minimal — I see you're focused."],
        Verbosity.MODERATE: [
            "I can see you're in deep work mode. I'll keep interruptions minimal and only surface critical items.",
        ],
        Verbosity.DETAILED: [
            "You're clearly in a focused flow state. I'm monitoring for anything critical in the background, but I'll only interrupt if absolutely necessary. Just let me know when you'd like me to be more proactive.",
        ],
    },
    Situation.FRUSTRATION_EMPATHY: {
        Verbosity.TERSE: ["I see the issue.", "Let me fix this.", "Found it."],
        Verbosity.CONCISE: ["I see the problem. Let me sort this out.", "I understand the frustration. Here's the fix."],
        Verbosity.MODERATE: [
            "I can see this is frustrating. Let me focus on solving this — no preamble needed.",
            "That's annoying. I'll cut straight to the solution.",
        ],
        Verbosity.DETAILED: [
            "I understand this is frustrating — debugging this kind of issue can really drain your energy. Let me identify the core problem and give you a clear path forward.",
        ],
    },
    Situation.FLOW_PROTECT: {
        Verbosity.TERSE: ["Carrying on.", "Continuing.", "On track."],
        Verbosity.CONCISE: ["Looks like good progress. I'll stay out of the way.", "You're on a roll — I'm here if needed."],
        Verbosity.MODERATE: [
            "You're making great progress. I'll stay in the background and only jump in if something needs attention.",
        ],
        Verbosity.DETAILED: [
            "This is going really well. I can see you're in a productive rhythm, so I'll minimize my suggestions. I'm watching for issues in the background and will flag anything critical.",
        ],
    },
    Situation.CURIOSITY_ENCOURAGE: {
        Verbosity.TERSE: ["Good question!", "Interesting.", "Hmm, let me think about that."],
        Verbosity.CONCISE: ["That's a great question!", "Interesting — let me dig into that.", "I love that question."],
        Verbosity.MODERATE: [
            "That's a really insightful question! Let me explore that.",
            "I love that curiosity. Here's what I know about this.",
        ],
        Verbosity.DETAILED: [
            "What a fantastic question — these are exactly the kinds of questions that lead to deeper understanding. Let me give you a thorough answer.",
        ],
    },
    Situation.GREETING_RETURN: {
        Verbosity.TERSE: ["Welcome back.", "Back again.", "Hey again."],
        Verbosity.CONCISE: ["Good to see you again!", "Welcome back!", "Hey, good to have you back."],
        Verbosity.MODERATE: [
            "Good to have you back! Picking up where we left off.",
            "Welcome back — I've kept things ready for you.",
        ],
        Verbosity.DETAILED: [
            "Welcome back! It's good to see you again. I've been keeping track of where we were, so we can jump right back in.",
        ],
    },
    Situation.GREETING_MORNING: {
        Verbosity.TERSE: ["Morning.", "Good morning.", "Ready when you are."],
        Verbosity.CONCISE: ["Good morning! Ready to start the day.", "Morning! What's on the agenda?"],
        Verbosity.MODERATE: [
            "Good morning! Fresh start today — what are we working on?",
            "Morning! I'm ready to help. What's first on your list?",
        ],
        Verbosity.DETAILED: [
            "Good morning! New day, fresh context. I'm ready to dive into whatever you need. Here's a quick summary of where we left off last time if you'd like to continue.",
        ],
    },
    Situation.GREETING_EVENING: {
        Verbosity.TERSE: ["Evening.", "Still at it?", "Late session."],
        Verbosity.CONCISE: ["Good evening. What can I help with?", "Working late — I'm here.", "Evening! What's on your mind?"],
        Verbosity.MODERATE: [
            "Good evening! Still going strong? I'm here to help.",
            "Evening session — I've got plenty of energy for you.",
        ],
        Verbosity.DETAILED: [
            "Good evening! I see you're working after hours — no judgment here, I'm a 24/7 operation. What can I help you tackle tonight?",
        ],
    },
    Situation.GREETING_FIRST: {
        Verbosity.TERSE: ["Hello.", "Hi.", "Ready."],
        Verbosity.CONCISE: ["Hi! I'm Jarvis, ready to help.", "Hello! What can I do for you?", "Hey there! Let's get started."],
        Verbosity.MODERATE: [
            "Hello! I'm Jarvis, your AI assistant. I'm here to help with whatever you need — coding, analysis, writing, problem-solving, and more.",
            "Hey! I'm Jarvis. I can help with code, answer questions, work through problems, or just chat. What would you like to work on?",
        ],
        Verbosity.DETAILED: [
            "Hello! I'm Jarvis, and I'm here to make your work easier. I can write and debug code, analyze data, help with research, explain complex topics, manage projects, and much more. I'll adapt to your style — just tell me if you prefer brief answers or detailed explanations.",
        ],
    },
    Situation.CLOSING_OFFER: {
        Verbosity.TERSE: ["More?", "Next?", "Continue?"],
        Verbosity.CONCISE: ["Want me to continue?", "Anything else?", "What's next?"],
        Verbosity.MODERATE: [
            "Let me know if you'd like me to go further with this.",
            "I can explore this more if you'd like, or we can move on to something else.",
        ],
        Verbosity.DETAILED: [
            "There's more I could dig into here. I'm also happy to switch gears if there's something else on your mind. Just let me know which direction you'd like to go.",
        ],
    },
    Situation.CLOSING_DIRECT: {
        Verbosity.TERSE: ["Done.", "That's it.", "Finished."],
        Verbosity.CONCISE: ["That should do it.", "All done here.", "That covers it."],
        Verbosity.MODERATE: [
            "That takes care of it. Let me know if anything else comes up.",
            "Alright, that should be everything for now.",
        ],
        Verbosity.DETAILED: [
            "That should resolve everything. If you run into any issues or have follow-up questions, I'm right here. I'll be keeping an eye on things in the background.",
        ],
    },
}

_TRANSITIONS: dict[PersonaMode, list[str]] = {
    PersonaMode.BUTLER: [
        "Moving on —", "Regarding your next point —", "As for that —",
        "Additionally —", "On a related note —",
    ],
    PersonaMode.COLLABORATOR: [
        "Now, another thing —", "OK, next —", "Also —",
        "Here's something else —", "Switching gears —",
    ],
    PersonaMode.MENTOR: [
        "Now, an important related concept —", "Building on that —",
        "Here's where it gets interesting —", "This connects to another idea —",
    ],
    PersonaMode.COMPANION: [
        "Oh, and —", "By the way —", "Also, I was thinking —",
        "Hmm, here's something —", "You know what else? —",
    ],
}

_ACKNOWLEDGMENTS: dict[Verbosity, list[str]] = {
    Verbosity.TERSE: ["Got it.", "Understood.", "Right.", "OK."],
    Verbosity.CONCISE: ["Got it.", "Makes sense.", "I see.", "Right, understood."],
    Verbosity.MODERATE: [
        "I understand.", "That makes sense.", "Good point.",
        "I see what you mean.", "Noted.",
    ],
    Verbosity.DETAILED: [
        "I understand exactly what you mean.", "That's a great point, and it connects to —",
        "I appreciate that clarification.", "That gives me a clear picture.",
    ],
}

_FILLER_REDUCTION: dict[Verbosity, list[str]] = {
    Verbosity.TERSE: [],
    Verbosity.CONCISE: ["certainly", "absolutely", "without a doubt", "it goes without saying",
                        "needless to say", "at the end of the day", "in this day and age"],
    Verbosity.MODERATE: ["needless to say", "it goes without saying"],
    Verbosity.DETAILED: [],
}

_MAX_HISTORY = 50


class ButlerVoice:
    def __init__(self):
        self._lock = threading.RLock()
        self._phrase_history: deque[str] = deque(maxlen=_MAX_HISTORY)
        self._history_rotation_index: int = 0

    def get_greeting(self, personality: ButlerPersonality = None,
                     situation: Situation = None) -> str:
        with self._lock:
            if personality is None:
                personality = get_butler_personality()
            state = personality.get_state()
            mode = PersonaMode(state["mode"])
            formality = _formality_from_value(state["formality"])
            if situation:
                return self._pick_phrase(situation, Verbosity(state["verbosity"]))
            mode_greetings = _GREETINGS.get(mode, _GREETINGS[PersonaMode.BUTLER])
            formality_greetings = mode_greetings.get(formality, mode_greetings[Formality.PROFESSIONAL])
            return self._pick_unique(formality_greetings)

    def get_phrase(self, situation: Situation, personality: ButlerPersonality = None) -> str:
        with self._lock:
            if personality is None:
                personality = get_butler_personality()
            state = personality.get_state()
            verbosity = Verbosity(state["verbosity"])
            return self._pick_phrase(situation, verbosity)

    def get_transition(self, personality: ButlerPersonality = None) -> str:
        with self._lock:
            if personality is None:
                personality = get_butler_personality()
            state = personality.get_state()
            mode = PersonaMode(state["mode"])
            transitions = _TRANSITIONS.get(mode, _TRANSITIONS[PersonaMode.BUTLER])
            return self._pick_unique(transitions)

    def get_acknowledgment(self, personality: ButlerPersonality = None) -> str:
        with self._lock:
            if personality is None:
                personality = get_butler_personality()
            state = personality.get_state()
            verbosity = Verbosity(state["verbosity"])
            acks = _ACKNOWLEDGMENTS.get(verbosity, _ACKNOWLEDGMENTS[Verbosity.MODERATE])
            return self._pick_unique(acks)

    def get_fillers_to_strip(self, personality: ButlerPersonality = None) -> list[str]:
        with self._lock:
            if personality is None:
                personality = get_butler_personality()
            state = personality.get_state()
            verbosity = Verbosity(state["verbosity"])
            return _FILLER_REDUCTION.get(verbosity, [])

    def strip_fillers(self, text: str, personality: ButlerPersonality = None) -> str:
        fillers = self.get_fillers_to_strip(personality)
        result = text
        for filler in fillers:
            result = result.replace(f", {filler}", "")
            result = result.replace(f". {filler.capitalize()}", ".")
            result = result.replace(f" {filler} ", " ")
            result = result.replace(f"{filler.capitalize()}, ", "")
            result = result.replace(f"{filler}, ", "")
        return result

    def get_response_template(self, situation: Situation,
                              personality: ButlerPersonality = None) -> dict:
        with self._lock:
            if personality is None:
                personality = get_butler_personality()
            state = personality.get_state()
            mode = PersonaMode(state["mode"])
            verbosity = Verbosity(state["verbosity"])
            return {
                "greeting": self.get_greeting(personality, situation if situation in (
                    Situation.GREETING_RETURN, Situation.GREETING_MORNING,
                    Situation.GREETING_EVENING, Situation.GREETING_FIRST) else None),
                "phrase": self.get_phrase(situation, personality),
                "transition": self.get_transition(personality),
                "fillers_to_strip": self.get_fillers_to_strip(personality),
                "mode": mode.value,
                "verbosity": verbosity.value,
            }

    def get_time_greeting(self, personality: ButlerPersonality = None) -> str:
        import datetime
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return self.get_greeting(personality, Situation.GREETING_MORNING)
        elif 12 <= hour < 17:
            return self.get_greeting(personality)
        elif 17 <= hour < 22:
            return self.get_greeting(personality, Situation.GREETING_EVENING)
        return self.get_greeting(personality)

    def get_state(self) -> dict:
        with self._lock:
            return {
                "phrases_generated": len(self._phrase_history),
                "unique_phrases": len(set(self._phrase_history)),
            }

    def _pick_phrase(self, situation: Situation, verbosity: Verbosity) -> str:
        phrases = _SITUATION_PHRASES.get(situation, {}).get(verbosity, [])
        if not phrases:
            for v in [Verbosity.CONCISE, Verbosity.MODERATE, Verbosity.TERSE, Verbosity.DETAILED]:
                phrases = _SITUATION_PHRASES.get(situation, {}).get(v, [])
                if phrases:
                    break
        if not phrases:
            return ""
        return self._pick_unique(phrases)

    def _pick_unique(self, phrases: list[str]) -> str:
        if not phrases:
            return ""
        unused = [p for p in phrases if p not in self._phrase_history]
        if not unused:
            self._phrase_history.clear()
            unused = phrases
        choice = random.choice(unused)
        self._phrase_history.append(choice)
        return choice


def get_butler_voice() -> ButlerVoice:
    global _voice
    if _voice is None:
        with _voice_lock:
            if _voice is None:
                _voice = ButlerVoice()
    return _voice