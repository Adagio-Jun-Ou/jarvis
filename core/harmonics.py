"""
Harmonic Series Physics - The Mathematics of Sound.

The harmonic series explains WHY harmony works:
1. A vibrating string/air column produces fundamental + overtones
2. Overtones follow: f_n = f_0 * n (integer multiples)
3. Simple ratios (2:1, 3:2, 4:3) sound consonant
4. Complex ratios sound dissonant
5. This is the PHYSICAL basis of all Western harmony

Key insights:
- Octave (2:1) - Most consonant, first overtone
- Fifth (3:2) - Next most consonant, second overtone
- Fourth (4:3) - Strong consonance, third overtone inversion
- Major third (5:4) - Fifth overtone, Renaissance "discovery"
- Minor third (6:5) - Sixth overtone, minor mode
- Tritone (7:5 or 45:32) - Seventh overtone, "diabolus in musica"

This module teaches Jarvis to compose using harmonic principles.
"""
from dataclasses import dataclass
import math
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class HarmonicInterval:
    """An interval with its harmonic series properties."""
    name: str
    ratio: tuple  # Frequency ratio (numerator, denominator)
    harmonics: tuple  # Which harmonics produce this (e.g., (2,3) for fifth)
    consonance: float  # 0-1, based on ratio simplicity
    description: str
    emotional_quality: str
    era_acceptance: dict  # When this interval was accepted in Western music


# The Harmonic Series (overtones)
# Fundamental: f_0
# 2nd harmonic: 2*f_0 (octave)
# 3rd harmonic: 3*f_0 (octave + fifth)
# 4th harmonic: 4*f_0 (two octaves)
# 5th harmonic: 5*f_0 (two octaves + major third)
# 6th harmonic: 6*f_0 (two octaves + fifth)
# 7th harmonic: 7*f_0 (two octaves + minor seventh, "out of tune")
# 8th harmonic: 8*f_0 (three octaves)
# 9th harmonic: 9*f_0 (three octaves + major second)
# 10th harmonic: 10*f_0 (three octaves + major third)

HARMONIC_SERIES = {
    1: {"name": "Fundamental", "ratio": (1, 1), "cents": 0, "role": "root"},
    2: {"name": "Octave", "ratio": (2, 1), "cents": 1200, "role": "most_consonant"},
    3: {"name": "Fifth", "ratio": (3, 2), "cents": 702, "role": "perfect"},
    4: {"name": "Fourth", "ratio": (4, 3), "cents": 498, "role": "perfect"},
    5: {"name": "Major third", "ratio": (5, 4), "cents": 386, "role": "consonant"},
    6: {"name": "Minor third", "ratio": (6, 5), "cents": 316, "role": "consonant"},
    7: {"name": "Minor seventh", "ratio": (7, 4), "cents": 969, "role": "complex"},
    8: {"name": "Major second", "ratio": (9, 8), "cents": 204, "role": "imperfect"},
    9: {"name": "Major seventh", "ratio": (15, 8), "cents": 1088, "role": "tense"},
    10: {"name": "Tritone", "ratio": (7, 5), "cents": 583, "role": "dissonant"},
}

# Intervals derived from harmonic series
INTERVALS_FROM_HARMONICS = {
    "perfect_octave": HarmonicInterval(
        name="Perfect Octave",
        ratio=(2, 1),
        harmonics=(1, 2),
        consonance=1.0,
        description="The most consonant interval - exact doubling of frequency",
        emotional_quality="completion, finality, resolution",
        era_acceptance={"medieval": "essential", "renaissance": "essential", "baroque": "essential", "modern": "essential"}
    ),
    "perfect_fifth": HarmonicInterval(
        name="Perfect Fifth",
        ratio=(3, 2),
        harmonics=(2, 3),
        consonance=0.95,
        description="Second harmonic creates perfect fifth with fundamental",
        emotional_quality="strength, stability, openness",
        era_acceptance={"medieval": "essential", "renaissance": "essential", "baroque": "essential", "modern": "essential"}
    ),
    "perfect_fourth": HarmonicInterval(
        name="Perfect Fourth",
        ratio=(4, 3),
        harmonics=(3, 4),
        consonance=0.90,
        description="Third harmonic creates fourth with second harmonic",
        emotional_quality="transitional, modal",
        era_acceptance={"medieval": "essential", "renaissance": "consonant", "baroque": "consonant", "modern": "essential"}
    ),
    "major_third": HarmonicInterval(
        name="Major Third",
        ratio=(5, 4),
        harmonics=(4, 5),
        consonance=0.80,
        description="Fifth harmonic creates major third - 'discovered' in Renaissance",
        emotional_quality="bright, happy, stable",
        era_acceptance={"medieval": "dissonant", "renaissance": "consonant", "baroque": "essential", "modern": "essential"}
    ),
    "minor_third": HarmonicInterval(
        name="Minor Third",
        ratio=(6, 5),
        harmonics=(5, 6),
        consonance=0.75,
        description="Sixth harmonic creates minor third with fifth harmonic",
        emotional_quality="sad, tense, serious",
        era_acceptance={"medieval": "dissonant", "renaissance": "consonant", "baroque": "essential", "modern": "essential"}
    ),
    "major_sixth": HarmonicInterval(
        name="Major Sixth",
        ratio=(5, 3),
        harmonics=(3, 5),
        consonance=0.75,
        description="Fifth harmonic creates sixth with third harmonic",
        emotional_quality="hopeful, open, yearning",
        era_acceptance={"medieval": "imperfect", "renaissance": "consonant", "baroque": "essential", "modern": "essential"}
    ),
    "minor_sixth": HarmonicInterval(
        name="Minor Sixth",
        ratio=(8, 5),
        harmonics=(5, 8),
        consonance=0.70,
        description="Inversion of major third through octave",
        emotional_quality="mysterious, unresolved",
        era_acceptance={"medieval": "imperfect", "renaissance": "consonant", "baroque": "consonant", "modern": "essential"}
    ),
    "minor_seventh": HarmonicInterval(
        name="Minor Seventh",
        ratio=(7, 4),
        harmonics=(4, 7),
        consonance=0.55,
        description="Seventh harmonic - naturally 'out of tune', blue note",
        emotional_quality="bluesy, jazzy, unresolved",
        era_acceptance={"medieval": "dissonant", "renaissance": "dissonant", "baroque": "dissonant", "modern": "consonant"}
    ),
    "major_seventh": HarmonicInterval(
        name="Major Seventh",
        ratio=(15, 8),
        harmonics=(8, 15),
        consonance=0.45,
        description="Strong tension seeking resolution to octave",
        emotional_quality="longing, anticipation, dream",
        era_acceptance={"medieval": "dissonant", "renaissance": "dissonant", "baroque": "dissonant", "modern": "consonant"}
    ),
    "major_second": HarmonicInterval(
        name="Major Second",
        ratio=(9, 8),
        harmonics=(8, 9),
        consonance=0.35,
        description="Ninth harmonic creates this imperfect consonance",
        emotional_quality="movement, transition, stepwise",
        era_acceptance={"medieval": "imperfect", "renaissance": "imperfect", "baroque": "dissonant", "modern": "consonant"}
    ),
    "minor_second": HarmonicInterval(
        name="Minor Second",
        ratio=(16, 15),
        harmonics=(15, 16),
        consonance=0.25,
        description="Tightest interval, strongest dissonance",
        emotional_quality="tension, pain, urgency",
        era_acceptance={"medieval": "dissonant", "renaissance": "dissonant", "baroque": "dissonant", "modern": "dissonant"}
    ),
    "tritone": HarmonicInterval(
        name="Tritone",
        ratio=(7, 5),
        harmonics=(5, 7),
        consonance=0.20,
        description="Seventh harmonic creates this 'devil's interval'",
        emotional_quality="unstable, sinister, dramatic",
        era_acceptance={"medieval": "forbidden", "renaissance": "avoided", "baroque": "tension", "modern": "essential"}
    ),
}

# Why chords work: stacking thirds from harmonic series
CHORD_HARMONICS = {
    "major_triad": {
        "notes": ["root", "major_third", "fifth"],
        "ratios": [(1, 1), (5, 4), (3, 2)],
        "harmonics": [1, 5, 3],
        "description": "Harmonics 4, 5, 6 superimposed - most consonant chord",
        "consonance": 0.90,
        "stability": "stable",
    },
    "minor_triad": {
        "notes": ["root", "minor_third", "fifth"],
        "ratios": [(1, 1), (6, 5), (3, 2)],
        "harmonics": [5, 6, 9],
        "description": "Harmonics 10, 12, 15 - naturally darker",
        "consonance": 0.85,
        "stability": "less_stable",
    },
    "dominant_seventh": {
        "notes": ["root", "major_third", "fifth", "minor_seventh"],
        "ratios": [(1, 1), (5, 4), (3, 2), (7, 4)],
        "harmonics": [4, 5, 6, 7],
        "description": "Harmonics 4-7, the seventh harmonic creates tension",
        "consonance": 0.50,
        "stability": "unstable_seeks_resolution",
    },
    "major_seventh": {
        "notes": ["root", "major_third", "fifth", "major_seventh"],
        "ratios": [(1, 1), (5, 4), (3, 2), (15, 8)],
        "harmonics": [8, 10, 12, 15],
        "description": "Extended harmonic series, dreamy quality",
        "consonance": 0.65,
        "stability": "soft_tension",
    },
    "minor_seventh": {
        "notes": ["root", "minor_third", "fifth", "minor_seventh"],
        "ratios": [(1, 1), (6, 5), (3, 2), (7, 4)],
        "harmonics": [10, 12, 15, 21],
        "description": "Jazz standard, harmonics 10, 12, 15, 21",
        "consonance": 0.60,
        "stability": "soft_tension",
    },
}

# Scale construction from harmonics
# Each scale relates to specific harmonic relationships

SCALES_FROM_HARMONICS = {
    # === NATURAL SCALES FROM HARMONIC SERIES ===
    "harmonic_series_scale": {
        "name": "Harmonic Series Scale",
        "intervals": [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8],
        "semitones": [0, 2, 4, 5, 7, 9, 11],
        "ratios": [(1,1), (9,8), (5,4), (4,3), (3,2), (5,3), (15,8)],
        "harmonics_used": [1, 9, 5, 4, 3, 5, 15],
        "description": "Intervals derived directly from harmonic series",
        "quality": "naturally consonant, ancient quality",
        "emotional_character": "pure, resolved, fundamental",
        "consonance_score": 0.92,
        "era_origin": "ancient",
    },
    "just_major": {
        "name": "Just Major Scale",
        "intervals": [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8],
        "semitones": [0, 2, 4, 5, 7, 9, 11],
        "ratios": [(1,1), (9,8), (5,4), (4,3), (3,2), (5,3), (15,8)],
        "harmonics_used": [1, 9, 5, 4, 3, 5, 15],
        "description": "Pure intervals from harmonic series - most consonant chords",
        "quality": "perfectly in tune thirds, cannot modulate",
        "emotional_character": "bright, stable, resolved",
        "consonance_score": 0.90,
        "best_for": "static harmony, drones, perfect intervals",
        "era_origin": "ancient",
        "harmonic_insight": "Major triad I is harmonics 4-5-6, V is 3-4-5, IV is 6-8-9",
    },
    "just_minor": {
        "name": "Just Minor Scale (Natural)",
        "intervals": [1, 6/5, 4/3, 3/2, 8/5, 9/5, 16/9],
        "semitones": [0, 2, 3, 5, 7, 8, 10],
        "ratios": [(1,1), (6,5), (4,3), (3,2), (8,5), (9,5), (16,9)],
        "harmonics_used": [1, 6, 4, 3, 8, 9, 16],
        "description": "Minor mode from harmonic series - harmonics 10,12,15",
        "quality": "naturally somber, darker consonance",
        "emotional_character": "melancholic, introspective, serious",
        "consonance_score": 0.85,
        "best_for": "introspective pieces, minor key works",
        "era_origin": "ancient",
        "harmonic_insight": "Minor triad uses harmonics 10-12-15, naturally darker than major",
    },
    "harmonic_minor": {
        "name": "Harmonic Minor Scale",
        "intervals": [1, 9/8, 6/5, 4/3, 3/2, 8/5, 15/8],
        "semitones": [0, 2, 3, 5, 7, 8, 11],
        "ratios": [(1,1), (9,8), (6,5), (4,3), (3,2), (8,5), (15,8)],
        "harmonics_used": [1, 9, 6, 4, 3, 8, 15],
        "description": "Named 'harmonic' because V chord is major (leading tone)",
        "quality": "tense leading tone, strong resolution",
        "emotional_character": "dramatic, exotic, classical minor",
        "consonance_score": 0.75,
        "best_for": "classical minor works, dramatic resolution",
        "era_origin": "renaissance",
        "harmonic_insight": "Raised 7th creates leading tone, major V with dominant function",
        "distinctive_interval": "Augmented second (6-7) = Eastern/exotic sound",
    },
    "melodic_minor": {
        "name": "Melodic Minor Scale",
        "intervals_ascending": [1, 9/8, 4/3, 3/2, 5/3, 15/8],
        "intervals_descending": [1, 9/8, 6/5, 4/3, 3/2, 8/5, 16/9],
        "semitones_asc": [0, 2, 3, 5, 7, 9, 11],
        "semitones_desc": [0, 2, 3, 5, 7, 8, 10],
        "description": "Ascends like major, descends like natural minor",
        "quality": "versatile, avoids awkward augmented second",
        "emotional_character": "fluid, jazz-influenced, sophisticated",
        "consonance_score": 0.78,
        "best_for": "jazz, classical melody writing",
        "era_origin": "baroque",
        "harmonic_insight": "Compromise for smooth voice leading",
    },

    # === MODES - Different harmonizations of diatonic scale ===
    "dorian": {
        "name": "Dorian Mode",
        "semitones": [0, 2, 3, 5, 7, 9, 10],
        "intervals": [1, 9/8, 6/5, 4/3, 3/2, 5/3, 16/9],
        "description": "Minor with raised 6th - symmetric around center",
        "quality": "balanced minor, neither sad nor happy",
        "emotional_character": "ancient, modal, folk-like",
        "consonance_score": 0.82,
        "best_for": "folk music, jazz improvisation",
        "era_origin": "ancient_greek",
        "harmonic_insight": "i-IV is minor-minor, unique balanced quality",
    },
    "phrygian": {
        "name": "Phrygian Mode",
        "semitones": [0, 1, 3, 5, 7, 8, 10],
        "intervals": [1, 16/15, 6/5, 4/3, 3/2, 8/5, 16/9],
        "description": "Minor with flattened 2nd - Spanish/Flamenco quality",
        "quality": "exotic tension from lowered second",
        "emotional_character": "Spanish, mysterious, dark",
        "consonance_score": 0.72,
        "best_for": "flamenco, dark atmospheres",
        "era_origin": "ancient_greek",
        "harmonic_insight": "Half-step from tonic creates distinctive tension",
    },
    "lydian": {
        "name": "Lydian Mode",
        "semitones": [0, 2, 4, 6, 7, 9, 11],
        "intervals": [1, 9/8, 5/4, 45/32, 3/2, 5/3, 15/8],
        "description": "Major with raised 4th - dreamy quality",
        "quality": "floating, unresolved, dream-like",
        "emotional_character": "dreamy, floating, magical",
        "consonance_score": 0.80,
        "best_for": "film scores, dream sequences, fantasy",
        "era_origin": "ancient_greek",
        "harmonic_insight": "Raised 4th avoids resolution, creates floating sensation",
    },
    "mixolydian": {
        "name": "Mixolydian Mode",
        "semitones": [0, 2, 4, 5, 7, 9, 10],
        "intervals": [1, 9/8, 5/4, 4/3, 3/2, 5/3, 16/9],
        "description": "Major with flattened 7th - blues/rock quality",
        "quality": "major with bluesy flattened seventh",
        "emotional_character": "bluesy, rock, dominant",
        "consonance_score": 0.83,
        "best_for": "rock, blues, dominant function",
        "era_origin": "ancient_greek",
        "harmonic_insight": "Flat 7 creates blues dominant feel, V chord is minor",
    },
    "locrian": {
        "name": "Locrian Mode",
        "semitones": [0, 1, 3, 5, 6, 8, 10],
        "intervals": [1, 16/15, 6/5, 4/3, 7/5, 8/5, 16/9],
        "description": "Most dissonant mode - tritone from start",
        "quality": "unstable, avoiding resolution",
        "emotional_character": "dark, ominous, unresolved",
        "consonance_score": 0.55,
        "best_for": "horror, tension, avant-garde",
        "era_origin": "theoretical",
        "harmonic_insight": "Contains tritone from root, i chord is diminished",
    },

    # === PENTATONIC - Most harmonic of all scales ===
    "major_pentatonic": {
        "name": "Major Pentatonic",
        "semitones": [0, 2, 4, 7, 9],
        "intervals": [1, 9/8, 5/4, 3/2, 5/3],
        "ratios": [(1,1), (9,8), (5,4), (3,2), (5,3)],
        "harmonics_used": [1, 9, 5, 3, 5],
        "description": "Only most consonant intervals - no semitones",
        "quality": "universally consonant, no wrong notes",
        "emotional_character": "peaceful, folk, nursery",
        "consonance_score": 0.95,
        "best_for": "folk, children's music, improvisation",
        "era_origin": "ancient_universal",
        "harmonic_insight": "Removes 4th and 7th - only perfect and major intervals remain",
        "universal": "Found in virtually every world music tradition",
    },
    "minor_pentatonic": {
        "name": "Minor Pentatonic",
        "semitones": [0, 3, 5, 7, 10],
        "intervals": [1, 6/5, 4/3, 3/2, 16/9],
        "description": "Minor form of pentatonic - blues foundation",
        "quality": "blues foundation, soulful",
        "emotional_character": "soulful, blues, rock",
        "consonance_score": 0.88,
        "best_for": "blues, rock improvisation",
        "era_origin": "ancient_universal",
        "harmonic_insight": "Used in blues with added flat 5 and flat 7",
    },
    "blues_scale": {
        "name": "Blues Scale",
        "semitones": [0, 3, 5, 6, 7, 10],
        "intervals": [1, 6/5, 4/3, 7/5, 3/2, 16/9],
        "description": "Minor pentatonic with 'blue note' (flat 5)",
        "quality": "soulful tension from tritone",
        "emotional_character": "bluesy, soulful, American",
        "consonance_score": 0.72,
        "best_for": "blues, jazz, rock",
        "era_origin": "african_american",
        "harmonic_insight": "Flat 5 (tritone) creates 'blue' tension, bends toward harmonic series",
    },

    # === SPECIALTY SCALES ===
    "whole_tone": {
        "name": "Whole Tone Scale",
        "semitones": [0, 2, 4, 6, 8, 10],
        "intervals": [1, 9/8, 5/4, 7/5, 9/5, 16/9],
        "description": "All whole steps - no harmonic gravity",
        "quality": "floating, rootless, ambiguous",
        "emotional_character": "dreamy, ambiguous, impressionistic",
        "consonance_score": 0.68,
        "best_for": "Debussy, dream sequences, unresolved",
        "era_origin": "impressionist",
        "harmonic_insight": "No perfect fifth (3:2) = no tonal center = floating sensation",
    },
    "chromatic": {
        "name": "Chromatic Scale",
        "semitones": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "description": "All twelve semitones - no harmonic bias",
        "quality": "neutral, atonal potential",
        "emotional_character": "neutral, wandering, atonal",
        "consonance_score": 0.50,
        "best_for": "chromatic passages, modulation",
        "era_origin": "universal",
        "harmonic_insight": "Contains all intervals, no preference for harmonic relationships",
    },
    "octatonic": {
        "name": "Octatonic Scale (Diminished)",
        "semitones": [0, 1, 3, 4, 6, 7, 9, 10],
        "intervals": [1, 16/15, 6/5, 5/4, 7/5, 3/2, 16/9],
        "description": "Alternating whole and half steps - symmetric",
        "quality": "tense, unstable, symmetric",
        "emotional_character": "tense, mysterious, Russian",
        "consonance_score": 0.60,
        "best_for": "Rimsky-Korsakov, tension, transitions",
        "era_origin": "russian_modernist",
        "harmonic_insight": "Creates diminished 7th chords, no stable resolution",
    },

    # === HISTORICAL TUNING SYSTEMS ===
    "pythagorean": {
        "name": "Pythagorean Tuning",
        "intervals": [1, 9/8, 81/64, 4/3, 3/2, 27/16, 243/128],
        "semitones": [0, 2, 4, 5, 7, 9, 11],
        "description": "Perfect fifths stacked - ancient Greek",
        "quality": "pure fifths, sharp thirds",
        "emotional_character": "ancient, medieval, chant-like",
        "consonance_score": 0.82,
        "best_for": "medieval chant, pure fifth emphasis",
        "era_origin": "ancient_greek",
        "harmonic_insight": "Thirds too sharp (81:64 instead of 5:4), wolf interval",
    },
    "equal_temperament": {
        "name": "Equal Temperament",
        "semitones": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "intervals": [1, 1.0595, 1.1225, 1.1892, 1.2599, 1.3348, 1.4142, 1.4983, 1.5874, 1.6818, 1.7818, 1.8877],
        "description": "All semitones equal - modern compromise",
        "quality": "all keys equally out of tune",
        "emotional_character": "versatile, modern, neutral",
        "consonance_score": 0.78,
        "best_for": "key modulation, modern music",
        "era_origin": "baroque_compromise",
        "harmonic_insight": "Thirds 14 cents sharp, fifths 2 cents flat - imperceptible compromise",
    },

    # === WORLD MUSIC SCALES ===
    "persian_dastgah": {
        "name": "Persian Dastgah (Shur)",
        "semitones": [0, 1, 4, 5, 7, 8, 11],
        "description": "Persian modal system with quarter tones",
        "quality": "microtonal, Middle Eastern",
        "emotional_character": "Persian, exotic, mystical",
        "consonance_score": 0.70,
        "best_for": "Middle Eastern music",
        "era_origin": "persian",
        "harmonic_insight": "Quarter tones create tension not in Western system",
    },
    "raga_bhairavi": {
        "name": "Raga Bhairavi (Indian)",
        "semitones": [0, 1, 3, 5, 7, 8, 10],
        "description": "One of the oldest Indian ragas",
        "quality": "devotional, morning raga",
        "emotional_character": "devotional, peaceful, morning",
        "consonance_score": 0.73,
        "best_for": "Indian classical, meditation",
        "era_origin": "ancient_indian",
        "harmonic_insight": "Microtonal ornamentation enriches harmonic series",
    },
}

# Scale degree functions from harmonic perspective
SCALE_DEGREE_HARMONICS = {
    1: {"name": "Tonic", "harmonic": "Fundamental (1st)", "function": "home, resolution", "stability": 1.0},
    2: {"name": "Supertonic", "harmonic": "9th harmonic divided by 8", "function": "pre-dominant", "stability": 0.6},
    3: {"name": "Mediant", "harmonic": "5th harmonic", "function": "tonic expansion", "stability": 0.75},
    4: {"name": "Subdominant", "harmonic": "4th harmonic", "function": "pre-dominant", "stability": 0.85},
    5: {"name": "Dominant", "harmonic": "3rd harmonic / 2", "function": "dominant tension", "stability": 0.95},
    6: {"name": "Submediant", "harmonic": "6th harmonic", "function": "relative minor", "stability": 0.70},
    7: {"name": "Leading tone", "harmonic": "7th harmonic", "function": "leads to tonic", "stability": 0.40},
}

# Why scales evoke emotions - harmonic relationships
SCALE_EMOTION_RELATIONSHIPS = {
    "major": {
        "harmonic_basis": "Built from harmonics 4-5-6-8, all consonant",
        "emotion": "bright, happy, resolved",
        "physics": "All intervals from simple ratios (4:5:6)",
    },
    "minor": {
        "harmonic_basis": "Uses harmonics 10-12-15, darker harmonics",
        "emotion": "sad, serious, introspective",
        "physics": "Higher harmonics = more complex = darker",
    },
    "harmonic_minor": {
        "harmonic_basis": "Adds leading tone (7th harmonic) for strong resolution",
        "emotion": "dramatic, exotic, classical",
        "physics": "7th harmonic creates maximum tension toward tonic",
    },
    "dorian": {
        "harmonic_basis": "Symmetric around tritone, balanced harmonics",
        "emotion": "ancient, modal, neither major nor minor",
        "physics": "No strong tension toward any harmonic center",
    },
    "pentatonic": {
        "harmonic_basis": "Only simplest ratios: 1:1, 9:8, 5:4, 3:2, 5:3",
        "emotion": "peaceful, universal, childlike",
        "physics": "All intervals from first 6 harmonics = pure consonance",
    },
    "blues": {
        "harmonic_basis": "Minor pentatonic + 7th harmonic (blue note)",
        "emotion": "soulful, American, African diaspora",
        "physics": "Blue notes bend toward 7th harmonic position",
    },
    "whole_tone": {
        "harmonic_basis": "No harmonic series anchors (no 3:2 fifth)",
        "emotion": "floating, dreamy, unresolved",
        "physics": "Lacks perfect fifth = no tonal gravity",
    },
}


class HarmonicSeries:
    """
    The mathematical foundation of harmony.
    
    The harmonic series explains why we hear intervals the way we do:
    - Simple ratios (small integers) = consonant
    - Complex ratios (large integers) = dissonant
    
    This is NOT arbitrary cultural convention - it's PHYSICS.
    """
    
    def __init__(self, fundamental_hz: float = 440.0):
        self.fundamental = fundamental_hz
        self.overtones = self._generate_overtones(16)
        
    def _generate_overtones(self, n: int) -> list:
        """Generate first n harmonics of fundamental."""
        return [self.fundamental * (i + 1) for i in range(n)]
    
    def get_interval_ratio(self, harmonic1: int, harmonic2: int) -> tuple:
        """Get the ratio between two harmonics."""
        if harmonic1 > harmonic2:
            harmonic1, harmonic2 = harmonic2, harmonic1
        return (harmonic2, harmonic1)
    
    def compute_consonance(self, ratio: tuple) -> float:
        """
        Compute consonance of an interval based on ratio simplicity.
        
        The simpler the ratio, the more consonant.
        Consonance = 1 / log(product of ratio parts)
        """
        numerator, denominator = ratio
        simplicity = numerator * denominator
        
        if simplicity <= 2:
            return 1.0
        elif simplicity <= 6:
            return 0.95
        elif simplicity <= 12:
            return 0.80
        elif simplicity <= 30:
            return 0.60
        else:
            return max(0.1, 1.0 / math.log(simplicity + 1))
    
    def get_harmonics_in_chord(self, chord_type: str) -> dict:
        """Get which harmonics are present in a chord type."""
        return CHORD_HARMONICS.get(chord_type, CHORD_HARMONICS["major_triad"])
    
    def find_common_harmonics(self, pitch1: int, pitch2: int) -> list:
        """
        Find which harmonics two pitches share.
        
        Two notes are consonant if they share many harmonics.
        This is why simple ratios sound good together.
        """
        freq1 = self.fundamental * (2 ** ((pitch1 - 69) / 12))
        freq2 = self.fundamental * (2 ** ((pitch2 - 69) / 12))
        
        harmonics1 = set()
        harmonics2 = set()
        
        for i in range(1, 17):
            harmonics1.add(round(freq1 * i, 2))
            harmonics2.add(round(freq2 * i, 2))
        
        common = harmonics1.intersection(harmonics2)
        return sorted(list(common))[:8]
    
    def calculate_beating(self, freq1: float, freq2: float) -> float:
        """
        Calculate beating frequency between two near frequencies.
        
        Beat frequency = |f1 - f2|
        More beating = more dissonance
        """
        return abs(freq1 - freq2)
    
    def get_interval_quality(self, semitones: int) -> dict:
        """Get the harmonic quality of an interval in semitones."""
        interval_map = {
            0: ("perfect_unison", 1.0, "identity", 1),
            1: ("minor_second", 0.25, "tension", 16),
            2: ("major_second", 0.35, "imperfect", 9),
            3: ("minor_third", 0.75, "consonant", 6),
            4: ("major_third", 0.80, "consonant", 5),
            5: ("perfect_fourth", 0.90, "perfect", 4),
            6: ("tritone", 0.20, "dissonant", 7),
            7: ("perfect_fifth", 0.95, "perfect", 3),
            8: ("minor_sixth", 0.70, "imperfect", 8),
            9: ("major_sixth", 0.75, "consonant", 5),
            10: ("minor_seventh", 0.55, "imperfect", 7),
            11: ("major_seventh", 0.45, "tension", 15),
            12: ("perfect_octave", 1.0, "identity", 2),
        }
        
        semitones = semitones % 12
        name, consonance, quality, harmonic = interval_map.get(semitones, ("unknown", 0.5, "unknown", 1))
        
        return {
            "name": name,
            "consonance": consonance,
            "quality": quality,
            "harmonic": harmonic,
        }


class HarmonicComposer:
    """
    Compose using harmonic series principles.
    
    This teaches Jarvis to compose MUSICALLY, not just theoretically.
    """
    
    def __init__(self):
        self.series = HarmonicSeries()
        
    def choose_consonant_melody_note(self, base_pitch: int, 
                                       era: str = "baroque",
                                       consonance_threshold: float = 0.6) -> list:
        """
        Choose a melody note with good consonance.
        
        Uses harmonic series to select notes that naturally
        fit with the base pitch.
        """
        candidates = []
        
        for interval_semitones in range(1, 12):
            quality = self.series.get_interval_quality(interval_semitones)
            
            if quality["consonance"] >= consonance_threshold:
                pitch = base_pitch + interval_semitones
                
                candidates.append({
                    "pitch": pitch,
                    "interval": quality["name"],
                    "consonance": quality["consonance"],
                    "reason": f"Harmonic {quality['harmonic']} ratio",
                })
                
                # Also check descending
                candidates.append({
                    "pitch": base_pitch - interval_semitones,
                    "interval": quality["name"],
                    "consonance": quality["consonance"],
                    "reason": f"Harmonic {quality['harmonic']} ratio descending",
                })
        
        # For baroque/classical, prefer perfect and major intervals
        if era in ["baroque", "classical"]:
            candidates.sort(key=lambda x: (
                -x["consonance"],
                0 if "perfect" in x["interval"] or "major" in x["interval"] else 1
            ))
        
        return candidates[:5]
    
    def build_harmonic_chord(self, root: int, 
                             chord_type: str = "major_triad",
                             era: str = "baroque") -> dict:
        """
        Build a chord using harmonic series ratios.
        
        Returns chord notes with justification from harmonics.
        """
        chord_info = self.series.get_harmonics_in_chord(chord_type)
        notes = []
        ratios = chord_info["ratios"]
        harmonics = chord_info["harmonics"]
        
        for i, (note_name, ratio) in enumerate(zip(chord_info["notes"], ratios)):
            num, denom = ratio
            semitones = round(12 * math.log2(num / denom))
            pitch = root + semitones
            
            notes.append({
                "pitch": pitch,
                "name": note_name,
                "ratio": ratio,
                "harmonic_number": harmonics[i] if i < len(harmonics) else 1,
                "frequency_ratio": f"{num}:{denom}",
            })
        
        return {
            "root": root,
            "chord_type": chord_type,
            "notes": notes,
            "consonance": chord_info["consonance"],
            "stability": chord_info["stability"],
            "reason": chord_info["description"],
        }
    
    def create_harmonic_progression(self, key_root: int, 
                                       era: str = "baroque",
                                       length: int = 4) -> list:
        """
        Create a progression based on harmonic relationships.
        
        Uses harmonic series to choose chords that naturally
        follow from each other.
        """
        # Key relationships from harmonics:
        # I (tonic) - root
        # V (dominant) - 3rd harmonic divided by 2 = fifth
        # IV (subdominant) - 4th harmonic divided by 4 = fourth
        
        fifth_up = (key_root + 7) % 12 + (key_root // 12) * 12
        fourth_up = (key_root + 5) % 12 + (key_root // 12) * 12
        
        progressions = {
            "baroque": [
                {"degree": "I", "harmonic": "Fundamental", "pitch": key_root},
                {"degree": "IV", "harmonic": "4th harmonic", "pitch": fourth_up},
                {"degree": "V", "harmonic": "3rd harmonic", "pitch": fifth_up},
                {"degree": "I", "harmonic": "Return to fundamental", "pitch": key_root},
            ],
            "classical": [
                {"degree": "I", "harmonic": "Fundamental", "pitch": key_root},
                {"degree": "vi", "harmonic": "Relative minor", "pitch": key_root + 9},
                {"degree": "ii", "harmonic": "Supertonic", "pitch": key_root + 2},
                {"degree": "V", "harmonic": "3rd harmonic", "pitch": fifth_up},
                {"degree": "I", "harmonic": "Resolution", "pitch": key_root},
            ],
            "romantic": [
                {"degree": "i", "harmonic": "Minor fundamental", "pitch": key_root},
                {"degree": "VI", "harmonic": "Major relative", "pitch": key_root + 8},
                {"degree": "iv", "harmonic": "Minor fourth", "pitch": key_root + 5},
                {"degree": "V", "harmonic": "Dominant tension", "pitch": fifth_up},
                {"degree": "i", "harmonic": "Return", "pitch": key_root},
            ],
        }
        
        return progressions.get(era, progressions["baroque"])[:length]
    
    def explain_why_interval_works(self, interval_name: str) -> str:
        """
        Explain WHY an interval sounds the way it does.
        
        This teaches Jarvis the physical basis of harmony.
        """
        interval = INTERVALS_FROM_HARMONICS.get(interval_name.replace(" ", "_").lower())
        
        if not interval:
            return f"Unknown interval: {interval_name}"
        
        ratio = f"{interval.ratio[0]}:{interval.ratio[1]}"
        simplicity = interval.ratio[0] * interval.ratio[1]
        
        explanation = f"""
{interval.name}:
  Frequency ratio: {ratio}
  Harmonics: {interval.harmonics[0]} and {interval.harmonics[1]}
  Consonance: {interval.consonance:.2f} (ratio simplicity: {simplicity})
  
WHY it sounds {interval.emotional_quality}:
  {interval.description}
  
ERA acceptance: {interval.era_acceptance}
"""
        return explanation
    
    def suggest_harmonic_intervals(self, melody_note: int, 
                                     era: str = "baroque",
                                     context: str = "harmony") -> list:
        """
        Suggest intervals for harmony that fit naturally with melody.
        
        Uses consonance thresholds appropriate to era.
        """
        consonance_thresholds = {
            "medieval": 0.85,  # Only perfect intervals
            "renaissance": 0.75,  # Add thirds
            "baroque": 0.70,  # Sixths acceptable
            "romantic": 0.50,  # Sevenths acceptable
            "modern": 0.25,  # Seconds acceptable
            "jazz": 0.20,  # Tritones acceptable
        }
        
        threshold = consonance_thresholds.get(era, 0.70)
        
        suggestions = []
        for semitones in range(1, 12):
            quality = self.series.get_interval_quality(semitones)
            
            if quality["consonance"] >= threshold:
                # Get era acceptance safely
                interval_key = quality["name"].replace(" ", "_").lower()
                interval_info = INTERVALS_FROM_HARMONICS.get(interval_key)
                era_acceptance = interval_info.era_acceptance if interval_info else {"modern": "acceptable"}
                
                suggestions.append({
                    "pitch": melody_note + semitones,
                    "interval": quality["name"],
                    "consonance": quality["consonance"],
                    "harmonic_reason": f"Harmonic {quality['harmonic']} resonance",
                    "era_appropriate": era in era_acceptance,
                })
                
                if semitones != 6:  # Avoid tritone in both directions
                    suggestions.append({
                        "pitch": melody_note - semitones,
                        "interval": quality["name"] + " (below)",
                        "consonance": quality["consonance"],
                        "harmonic_reason": f"Harmonic {quality['harmonic']} resonance",
                        "era_appropriate": True,
                    })
        
        suggestions.sort(key=lambda x: -x["consonance"])
        return suggestions[:6]
    
    def compute_chord_tension(self, pitches: list) -> dict:
        """
        Compute the tension/resolution tendency of a chord.
        
        Based on harmonic relationships between all note pairs.
        """
        if not pitches:
            return {"tension": 0, "resolution_tendency": "stable"}
        
        pair_consonances = []
        
        for i, p1 in enumerate(pitches):
            for p2 in pitches[i+1:]:
                interval = abs(p1 - p2) % 12
                quality = self.series.get_interval_quality(interval)
                pair_consonances.append(quality["consonance"])
        
        avg_consonance = sum(pair_consonances) / len(pair_consonances) if pair_consonances else 0.5
        
        if avg_consonance >= 0.80:
            return {"tension": 1 - avg_consonance, "resolution_tendency": "stable", "stability": "consonant"}
        elif avg_consonance >= 0.60:
            return {"tension": 1 - avg_consonance, "resolution_tendency": "mild", "stability": "imperfect"}
        elif avg_consonance >= 0.40:
            return {"tension": 1 - avg_consonance, "resolution_tendency": "strong", "stability": "unstable"}
        else:
            return {"tension": 1 - avg_consonance, "resolution_tendency": "must_resolve", "stability": "dissonant"}


class HarmonicPractice:
    """
    Practice exercises to internalize harmonic understanding.
    
    These exercises teach WHY harmony works, not just rules.
    """
    
    def __init__(self):
        self.composer = HarmonicComposer()
        self.exercises = self._create_exercises()
    
    def _create_exercises(self) -> list:
        """Create practice exercises for harmonic understanding."""
        return [
            {
                "name": "Harmonic Series Intervals",
                "difficulty": 1,
                "description": "Hear and identify intervals from harmonic series",
                "exercise": "Listen to fundamental + each harmonic (2-8). Identify the interval and its ratio.",
                "concepts": ["octave (2:1)", "fifth (3:2)", "fourth (4:3)", "major third (5:4)"],
                "harmonic_reason": "These intervals come directly from physics - not culture.",
            },
            {
                "name": "Consonance Ranking",
                "difficulty": 2,
                "description": "Order intervals by consonance using harmonic ratios",
                "exercise": "Given intervals, rank by consonance using ratio simplicity.",
                "concepts": ["simpler ratio = more consonant", "product of ratio numbers"],
                "harmonic_reason": "Simpler ratios share more harmonics = less beating = consonance.",
            },
            {
                "name": "Chord Construction from Harmonics",
                "difficulty": 2,
                "description": "Build major/minor triads using harmonic series",
                "exercise": "Stack harmonics 4, 5, 6 to create major triad. Explain WHY it's consonant.",
                "concepts": ["major triad = harmonics 4, 5, 6", "minor triad = harmonics 10, 12, 15"],
                "harmonic_reason": "Major triad is literally overtones 4, 5, 6 - naturally occurring.",
            },
            {
                "name": "The Forbidden Tritone",
                "difficulty": 3,
                "description": "Understand why the tritone was 'diabolus in musica'",
                "exercise": "Compare fifth (3:2) to tritone (7:5). Why is one consonant, one dissonant?",
                "concepts": ["7th harmonic is 'out of tune'", "tritone doesn't fit harmonic series"],
                "harmonic_reason": "Perfect fifth = harmonics 2&3 share many overtones. Tritone = harmonics 5&7 share few.",
            },
            {
                "name": "Why Dominant Seventh Resolves",
                "difficulty": 3,
                "description": "Understand V7 -> I using harmonic tension",
                "exercise": "Build V7 from harmonics 4, 5, 6, 7. Where does tension come from?",
                "concepts": ["7th harmonic creates tension", "seeks resolution to fundamental"],
                "harmonic_reason": "V7 includes 7th harmonic which naturally 'wants' to resolve.",
            },
            {
                "name": "Just Intonation vs Equal Temperament",
                "difficulty": 4,
                "description": "Compare pure harmonics to tempered scale",
                "exercise": "Play major third in just (5:4) vs equal temperament (400 cents). Which is 'in tune'?",
                "concepts": ["Just = harmonic series ratios", "ET = compromise for modulation"],
                "harmonic_reason": "Just intonation matches physical harmonics. ET allows key changes.",
            },
            {
                "name": "Compose with Harmonic Ratios",
                "difficulty": 4,
                "description": "Create melodies using only harmonic series intervals",
                "exercise": "Compose 8 bars using only notes from harmonic series above C.",
                "concepts": ["octave, fifth, major third, minor seventh", "all naturally consonant"],
                "harmonic_reason": "This is pre-cultural music - harmony that 'works' across all cultures.",
            },
            {
                "name": "Seventh Harmonic Exploration",
                "difficulty": 5,
                "description": "The 'blue note' and harmonic seventh",
                "exercise": "Compare harmonic seventh (7:4 = 969 cents) vs equal tempered minor seventh (1000 cents).",
                "concepts": ["7th harmonic is 'blue'", "used in blues and jazz", 'natural "bend"'],
                "harmonic_reason": "Blue notes bend toward the 7th harmonic - natural intonation.",
            },
        ]
    
    def get_next_exercise(self, mastery: dict) -> dict:
        """Get the next exercise based on current mastery."""
        for exercise in self.exercises:
            if mastery.get(exercise["name"], 0) < 0.8:
                return exercise
        return self.exercises[0]
    
    def evaluate_practice(self, exercise_name: str, response: dict) -> dict:
        """Evaluate a practice response."""
        exercise = next((e for e in self.exercises if e["name"] == exercise_name), None)
        
        if not exercise:
            return {"success": False, "feedback": "Exercise not found"}
        
        concept_score = sum(1 for concept in exercise["concepts"] if concept in str(response)) / len(exercise["concepts"])
        
        feedback = []
        
        if concept_score >= 0.7:
            feedback.append("Good understanding of harmonic concepts!")
        else:
            feedback.append(f"Review concepts: {', '.join(exercise['concepts'])}")
        
        if "harmonic" in str(response).lower():
            feedback.append("Excellent - you connected to the harmonic series!")
        
        return {
            "success": concept_score >= 0.7,
            "concept_score": concept_score,
            "feedback": " ".join(feedback),
            "harmonic_reason": exercise["harmonic_reason"],
        }


_composer = None
_composer_lock = threading.Lock()
_practice = None
_practice_lock = threading.Lock()


def get_harmonic_series() -> HarmonicSeries:
    """Get harmonic series instance."""
    return HarmonicSeries()


def get_harmonic_composer() -> HarmonicComposer:
    """Get harmonic composer instance."""
    global _composer
    if _composer is None:
        with _composer_lock:
            if _composer is None:
                _composer = HarmonicComposer()
            return _composer
    return _composer


def get_harmonic_practice() -> HarmonicPractice:
    """Get harmonic practice instance."""
    global _practice
    if _practice is None:
        with _practice_lock:
            if _practice is None:
                _practice = HarmonicPractice()
            return _practice
    return _practice


def get_scale_info(scale_name: str) -> dict:
    """Get detailed information about a scale from harmonic perspective.
    
    Args:
        scale_name: Name of scale (e.g., 'major', 'harmonic_minor', 'dorian')
    
    Returns:
        Dictionary with scale properties, harmonic basis, and usage
    """
    # Normalize name
    scale_key = scale_name.lower().replace(' ', '_').replace('-', '_')
    
    # Common scale name mappings
    scale_aliases = {
        'major': 'just_major',
        'minor': 'just_minor',
        'natural_minor': 'just_minor',
        'ionian': 'just_major',
        'aeolian': 'just_minor',
    }
    
    if scale_key in scale_aliases:
        scale_key = scale_aliases[scale_key]
    
    scale_info = SCALES_FROM_HARMONICS.get(scale_key)
    
    if not scale_info:
        # Try to find by partial match
        for key in SCALES_FROM_HARMONICS:
            if scale_key in key or key in scale_key:
                scale_info = SCALES_FROM_HARMONICS[key]
                break
    
    return scale_info or SCALES_FROM_HARMONICS['just_major']


def get_consonant_scale_degrees(scale_name: str, threshold: float = 0.7) -> list:
    """Get scale degrees ranked by consonance from harmonic perspective.
    
    Args:
        scale_name: Name of scale
        threshold: Minimum consonance (0-1)
    
    Returns:
        List of (degree, consonance, harmonic_reason) tuples
    """
    scale_info = get_scale_info(scale_name)
    semitones = scale_info.get('semitones', [0, 2, 4, 5, 7, 9, 11])
    
    series = HarmonicSeries()
    degrees = []
    
    for i, semi in enumerate(semitones):
        quality = series.get_interval_quality(semi)
        if quality['consonance'] >= threshold:
            degree_info = SCALE_DEGREE_HARMONICS.get(i + 1, {"name": f"Degree {i+1}"})
            degrees.append({
                'degree': i + 1,
                'semitones': semi,
                'name': degree_info.get('name', f'Degree {i+1}'),
                'consonance': quality['consonance'],
                'stability': degree_info.get('stability', 0.5),
                'harmonic_reason': f"Harmonic {quality['harmonic']}",
            })
    
    return sorted(degrees, key=lambda x: -x['consonance'])


def explain_scale_emotion(scale_name: str) -> str:
    """Explain why a scale evokes certain emotions from harmonic perspective.
    
    Args:
        scale_name: Name of scale
    
    Returns:
        Human-readable explanation of the scale's emotional character
    """
    # Normalize
    base_name = scale_name.lower().replace(' ', '_').replace('-', '_')
    if 'minor' in base_name and 'harmonic' in base_name:
        base_name = 'harmonic_minor'
    elif 'minor' in base_name:
        base_name = 'minor'
    elif 'major' in base_name:
        base_name = 'major'
    elif 'pentatonic' in base_name and 'minor' not in base_name:
        base_name = 'pentatonic'
    elif 'blues' in base_name:
        base_name = 'blues'
    elif 'dorian' in base_name:
        base_name = 'dorian'
    elif 'whole_tone' in base_name:
        base_name = 'whole_tone'
    
    relationship = SCALE_EMOTION_RELATIONSHIPS.get(base_name, SCALE_EMOTION_RELATIONSHIPS['major'])
    scale_info = get_scale_info(scale_name)
    
    explanation = f"""{scale_info.get('name', scale_name)} - {scale_info.get('emotional_character', 'Neutral')}

Harmonic Basis:
{relationship['harmonic_basis']}

Emotional Character: {relationship['emotion']}

Physics: {relationship['physics']}

Consonance Score: {scale_info.get('consonance_score', 0.5):.2f}/1.0
Best For: {scale_info.get('best_for', 'General use')}
"""
    return explanation


def get_scale_for_emotion(emotion: str) -> list:
    """Get scales that match a desired emotional character.
    
    Args:
        emotion: Desired emotion (e.g., 'sad', 'happy', 'dramatic', 'peaceful')
    
    Returns:
        List of matching scales sorted by appropriateness
    """
    emotion_keywords = {
        'sad': ['minor', 'harmonic_minor', 'locrian'],
        'happy': ['major', 'lydian', 'major_pentatonic'],
        'dramatic': ['harmonic_minor', 'phrygian', 'minor'],
        'peaceful': ['major_pentatonic', 'major', 'lydian'],
        'mysterious': ['dorian', 'phrygian', 'whole_tone'],
        'bluesy': ['blues_scale', 'minor_pentatonic', 'mixolydian'],
        'exotic': ['phrygian', 'harmonic_minor', 'persian_dastgah'],
        'tense': ['locrian', 'octatonic', 'whole_tone'],
        'dreamy': ['lydian', 'whole_tone', 'major'],
        'resolved': ['major', 'major_pentatonic', 'just_major'],
        'dark': ['minor', 'locrian', 'phrygian'],
        'bright': ['major', 'lydian', 'major_pentatonic'],
        'folk': ['dorian', 'major_pentatonic', 'mixolydian'],
        'classical': ['major', 'harmonic_minor', 'minor'],
        'baroque': ['major', 'minor', 'harmonic_minor'],
    }
    
    emotion_lower = emotion.lower()
    matches = []
    
    for key, keywords in emotion_keywords.items():
        if emotion_lower in key or key in emotion_lower:
            matches = keywords
            break
    
    # Get full scale info for matches
    results = []
    for scale_key in matches:
        if scale_key in SCALES_FROM_HARMONICS:
            scale = SCALES_FROM_HARMONICS[scale_key]
            results.append({
                'name': scale.get('name', scale_key),
                'key': scale_key,
                'emotional_character': scale.get('emotional_character', ''),
                'consonance_score': scale.get('consonance_score', 0.5),
                'best_for': scale.get('best_for', ''),
            })
    
    return sorted(results, key=lambda x: -x['consonance_score'])


def generate_harmonic_melody(scale_name: str, length: int = 8, root: int = 60) -> list:
    """Generate a melody using harmonic consonance principles.
    
    Uses harmonic series weights to choose notes that naturally sound good together.
    
    Args:
        scale_name: Name of scale to use
        length: Number of notes to generate
        root: MIDI root note (default: C4=60)
    
    Returns:
        List of MIDI pitches with harmonic justification
    """
    scale_info = get_scale_info(scale_name)
    semitones = scale_info.get('semitones', [0, 2, 4, 5, 7, 9, 11])
    scale_info.get('consonance_score', 0.7)
    
    series = HarmonicSeries()
    melody = []
    
    # Weight toward tonic (most stable)
    for i in range(length):
        # Start on tonic
        if i == 0:
            melody.append({
                'pitch': root,
                'degree': 1,
                'consonance': 1.0,
                'reason': 'Tonic - most stable',
            })
        # End on tonic
        elif i == length - 1:
            melody.append({
                'pitch': root,
                'degree': 1,
                'consonance': 1.0,
                'reason': 'Resolution to tonic',
            })
        else:
            # Choose scale degree weighted by consonance
            import random
            
            # Prefer higher consonance degrees
            weights = []
            for semi in semitones:
                quality = series.get_interval_quality(semi)
                weights.append(quality['consonance'])
            
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
            
            # Choose degree
            degree_idx = random.choices(range(len(semitones)), weights=weights)[0]
            semi = semitones[degree_idx]
            quality = series.get_interval_quality(semi)
            
            # Octave choice for variety
            octave = random.choice([0, 12, -12]) if i > 1 and i < length - 2 else 0
            
            melody.append({
                'pitch': root + semi + octave,
                'degree': degree_idx + 1,
                'consonance': quality['consonance'],
                'reason': f"{quality['name']} - harmonic {quality['harmonic']}",
            })
    
    return melody