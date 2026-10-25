"""
LLM-Powered Composition System - Sophisticated Music Creation

Master Techniques Integration:
- BACH: Inventions with real counterpoint, subject/answer structure
- MOZART: Sonata form with proper exposition/development/recapitulation  
- BEETHOVEN: Motivic development, theme transformation

Creates longer, fuller pieces with:
- Extended forms (sonata, rondo, theme-variations, fugue)
- Sophisticated chord progressions from master analysis
- Proper development sections
- Theme elaboration and transformation
- Musical intelligence for structure
- HARMONIC SERIES AWARENESS - understands WHY intervals work
- COUNTERPOINT RULES - voice leading and dissonance resolution
- NON-CHORD TONES - passing, neighbor, suspension, appoggiatura
- CADENCE TYPES - perfect, plagal, deceptive, half
"""

import logging
import random as _random_module
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

_rng = _random_module.Random()

logger = logging.getLogger(__name__)

# Import harmonic knowledge
try:
    from core.harmonics import get_harmonic_composer, INTERVALS_FROM_HARMONICS, CHORD_HARMONICS
    HARMONICS_AVAILABLE = True
except ImportError:
    HARMONICS_AVAILABLE = False

# Import master-inspired composition
try:
    from core.master_inspired_composer import compose_master_piece, BachInvention, MozartSonata
    MASTER_AVAILABLE = True
except ImportError:
    MASTER_AVAILABLE = False

# Import advanced music theory
try:
    from core.music_theory import (
        Counterpoint, NonChordTones, Cadences, ThemeDevelopment, 
        PeriodStructure, SequencePatterns
    )
    THEORY_AVAILABLE = True
except ImportError:
    THEORY_AVAILABLE = False

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Flat to sharp conversion for key parsing
FLAT_TO_SHARP = {
    'Cb': 'B', 'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    'cb': 'B', 'db': 'C#', 'eb': 'D#', 'fb': 'E', 'gb': 'F#', 'ab': 'G#', 'bb': 'A#',
}

# Extended scales
SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
    'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
}


# SOPHISTICATED PROGRESSIONS FROM MASTER ANALYSIS
# Each progression includes: progression, function description, era, usage context
MASTER_PROGRESSIONS = {
    # BAROQUE - Bach, Handel, Vivaldi
    'baroque': {
        'authentic_cadence': {
            'progression': ['I', 'IV', 'V7', 'I'],
            'function': 'tonic-predominant-dominant-tonic resolution',
            'usage': 'End of sections, strong resolution',
            'voice_leading': 'Keep common tones, resolve leading tone up',
        },
        'circle_of_fifths': {
            'progression': ['I', 'IV', 'vii°', 'iii', 'vi', 'ii', 'V', 'I'],
            'function': 'Sequential harmonic motion through circle',
            'usage': 'Modulatory sections, continuous development',
            'voice_leading': 'Roots move by descending fifth',
        },
        'bach_chorale': {
            'progression': ['I', 'I6', 'ii6', 'V7', 'I'],
            'function': 'Embellished tonic progression',
            'usage': 'Opening phrases, hymn style',
            'voice_leading': 'Smooth bass line, prepared dissonances',
        },
        'bach_passacaglia': {
            'progression': ['i', 'VII', 'i', 'VII', 'i', 'V', 'i', 'V'],
            'function': 'Ground bass pattern',
            'usage': 'Variations over repeating bass',
            'voice_leading': 'Consistent bass, varying upper voices',
        },
        'fugue_subject': {
            'progression': ['I', 'V', 'vi', 'V7', 'I'],
            'function': 'Subject-answer-countersubject outline',
            'usage': 'Fugue exposition',
            'voice_leading': 'Subject enters in tonic, answer in dominant',
        },
    },
    
    # CLASSICAL - Mozart, Haydn
    'classical': {
        'sonata_exposition': {
            'progression': ['I', 'I', 'V7/V', 'V', 'V', 'V7', 'I6/4', 'V7', 'I'],
            'function': 'Sonata exposition: first theme, transition, second theme, closing',
            'usage': 'Opening of sonata form movements',
            'voice_leading': 'Clear theme separation, modulatory bridge',
        },
        'alberti_bass_accompaniment': {
            'progression': ['I', 'V6', 'vi', 'IV', 'I6', 'V6', 'I', 'V7'],
            'function': 'Broken chord accompaniment pattern',
            'usage': 'Piano sonatas, accompaniment texture',
            'voice_leading': 'Low-high-middle-high pattern in left hand',
        },
        'mozart_cadence': {
            'progression': ['I6', 'ii6', 'I6/4', 'V7', 'I'],
            'function': 'Classical cadential 6-4',
            'usage': 'End of phrases, cadenzas',
            'voice_leading': '6/4 chord delays resolution to V7',
        },
        'haydn_surprise': {
            'progression': ['I', 'V', 'I', 'IV', 'ii6', 'V', 'vi', 'V7/V', 'V', 'I'],
            'function': 'Extended phrase with deceptive motion',
            'usage': 'Symphonic themes, longer phrases',
            'voice_leading': 'Surprise chord for dramatic effect',
        },
        'rondo_theme': {
            'progression': ['I', 'V', 'vi', 'iii', 'IV', 'I', 'ii', 'V7', 'I'],
            'function': 'Rounded binary with return',
            'usage': 'Rondo main theme',
            'voice_leading': 'Balanced phrases, clear return',
        },
    },
    
    # ROMANTIC - Beethoven, Chopin, Brahms, Schumann
    'romantic': {
        'beethoven_pathetique': {
            'progression': ['i', 'VII', 'V7/V', 'V7', 'i', 'VI', 'VII7', 'i'],
            'function': 'Dramatic minor opening',
            'usage': 'Sonata opening, emotional intensity',
            'voice_leading': 'Strong dissonances, dramatic resolution',
        },
        'beethoven_motif_development': {
            'progression': ['I', 'III', 'vi', 'I', 'ii', 'IV6', 'V7', 'I'],
            'function': 'Motivic development through keys',
            'usage': 'Development sections',
            'voice_leading': 'Mediant relationships, remote keys',
        },
        'chopin_nocturne': {
            'progression': ['I', 'I6', 'ii°7', 'V7', 'I', 'IV', 'ii°7/V', 'V7', 'I'],
            'function': 'Chromatic embellishment, bel canto style',
            'usage': 'Nocturne opening phrases',
            'voice_leading': 'Rich harmonies, suspensions, chromatic passing',
        },
        'schumann_dream': {
            'progression': ['I', 'vi', 'IV', 'ii', 'I6', 'V7', 'I', 'V7/V', 'V', 'I'],
            'function': 'Lyric phrase with inner voice motion',
            'usage': 'Character pieces, lieder',
            'voice_leading': 'Important inner voices, chromatic mediants',
        },
        'brahms_hemiola': {
            'progression': ['I', 'i', 'I', 'vi', 'IV', 'ii', 'V', 'I'],
            'function': 'Major-minor mixture with rhythmic displacement',
            'usage': 'Symphonic themes, romantic ambiguity',
            'voice_leading': '3-against-2 rhythm, smooth modulations',
        },
        'brahms_shift': {
            'progression': ['I', 'vi', 'bVI', 'IV', 'bVII', 'V7', 'I'],
            'function': 'Third-related harmonic shifts',
            'usage': 'Late romantic harmonic palette',
            'voice_leading': 'Seamless key changes by third',
        },
        'liszt_thematic': {
            'progression': ['I', 'bII6', 'V7', 'vi', 'V7/IV', 'IV', 'V7/V', 'V7', 'I'],
            'function': 'Thematic transformation progressions',
            'usage': 'Symphonic poems, thematic metamorphosis',
            'voice_leading': 'Chromatic mediants, augmented sixth',
        },
    },
    
    # IMPRESSIONIST - Debussy, Ravel
    'impressionist': {
        'debussy_whole_tone': {
            'progression': ['I+', 'V7alt', 'vi°7', 'II7', 'V7b9', 'I+'],
            'function': 'Whole tone and extended harmony',
            'usage': 'Impressionist color',
            'voice_leading': 'Parallel motion, planing',
        },
        'debussy_parallel': {
            'progression': ['I', 'III', 'IV', 'ii', 'V9', 'I'],
            'function': 'Parallel chord motion',
            'usage': 'Impressionist texture',
            'voice_leading': 'Chords slide in parallel',
        },
        'ravel_bitonal': {
            'progression': ['I7', 'bII7', 'ii7', 'V7', 'I'],
            'function': 'Bitonal and extended chord sequences',
            'usage': 'Impressionist/modern harmony',
            'voice_leading': 'Extended chords as sonorities',
        },
    },
    
    # MODERN - Stravinsky, Gershwin
    'modern': {
        'stravinsky_bitonal': {
            'progression': ['I', 'bIII', 'V7', 'i', 'VII', 'bVI', 'V', 'i'],
            'function': 'Polytonal layering',
            'usage': 'Modern ballet, orchestral',
            'voice_leading': 'Layers in different keys',
        },
        'gershwin_jazz': {
            'progression': ['I6', 'vi7', 'ii7', 'V7', 'I6/9', 'vi7#5', 'ii7', 'V7alt'],
            'function': 'Jazz-influenced classical',
            'usage': 'Rhapsody, concerto themes',
            'voice_leading': 'Extended chords, guide tone voice leading',
        },
        'blues_form': {
            'progression': ['I7', 'IV7', 'I7', 'V7', 'IV7', 'I7', 'V7', 'I7'],
            'function': '12-bar blues structure',
            'usage': 'Jazz-influenced classical',
            'voice_leading': 'Dominant quality throughout',
        },
    },
}

# EXTENDED FORMS WITH MEASURE COUNTS
EXTENDED_FORMS = {
    'sonata': {
        'total_measures': 120,
        'sections': {
            'exposition': {
                'measures': 40,
                'subsections': [
                    {'first_theme': 16, 'key': 'tonic'},
                    {'transition': 8, 'key': 'modulating'},
                    {'second_theme': 12, 'key': 'dominant'},
                    {'closing': 4, 'key': 'dominant'}
                ]
            },
            'development': {
                'measures': 35,
                'subsections': [
                    {'theme_fragmentation': 12},
                    {'sequence': 10},
                    {'retransition': 8},
                    {'climax': 5}
                ]
            },
            'recapitulation': {
                'measures': 40,
                'subsections': [
                    {'first_theme': 16, 'key': 'tonic'},
                    {'transition_modified': 8, 'key': 'tonic'},
                    {'second_theme': 12, 'key': 'tonic'},
                    {'coda': 4, 'key': 'tonic'}
                ]
            },
        },
        'emotional_arc': ['confident', 'tension', 'struggle', 'resolution'],
    },
    
    'rondo': {
        'total_measures': 80,
        'sections': {
            'A': {'measures': 16, 'key': 'tonic', 'repeat': 3},
            'B': {'measures': 16, 'key': 'dominant'},
            'C': {'measures': 16, 'key': 'subdominant'},
            'D': {'measures': 16, 'key': 'relative_major'},
        },
        'pattern': ['A', 'B', 'A', 'C', 'A', 'D', 'A'],
        'emotional_arc': ['joyful', 'contrasting', 'return', 'variety', 'return', 'dramatic', 'triumphant'],
    },
    
    'theme_variations': {
        'total_measures': 96,
        'sections': {
            'theme': {'measures': 24, 'key': 'tonic'},
            'var1': {'measures': 24, 'character': 'simple_variation'},
            'var2': {'measures': 24, 'character': 'ornamented'},
            'var3': {'measures': 24, 'character': 'thematic_transformation'},
        },
        'emotional_arc': ['theme', 'development', 'elaboration', 'transformation'],
    },
    
    'fugue': {
        'total_measures': 60,
        'sections': {
            'exposition': {'measures': 20, 'voices': 3},
            'episode_1': {'measures': 10, 'character': 'sequential'},
            'middle_entry': {'measures': 15, 'character': 'subject_returns'},
            'episode_2': {'measures': 8, 'character': 'development'},
            'coda': {'measures': 7, 'character': 'resolution'},
        },
        'emotional_arc': ['subject', 'development', 'tension', 'resolution'],
    },
    
    'invention': {
        'total_measures': 32,
        'sections': {
            'exposition': {'measures': 8, 'voices': 2},
            'development': {'measures': 16, 'sequences': 3},
            'cadence': {'measures': 8, 'key': 'tonic'},
        },
        'emotional_arc': ['theme', 'development', 'resolution'],
    },
    
    'nocturne': {
        'total_measures': 48,
        'sections': {
            'A': {'measures': 16, 'character': 'lyrical'},
            'B': {'measures': 16, 'character': 'agitated'},
            'A_prime': {'measures': 16, 'character': 'ornamented_return'},
        },
        'emotional_arc': ['peaceful', 'agitated', 'resolved_beauty'],
    },
    
    'prelude': {
        'total_measures': 40,
        'sections': {
            'opening': {'measures': 8, 'character': 'thematic'},
            'development': {'measures': 20, 'character': 'free'},
            'resolution': {'measures': 12, 'character': 'cadential'},
        },
        'emotional_arc': ['opening', 'exploration', 'resolution'],
    },
}

# DYNAMIC SHAPING FOR LONGER PIECES
DYNAMIC_SHAPES = {
    'sonata': {
        'exposition': ['mp', 'mf', 'f', 'mf'],
        'development': ['mf', 'f', 'ff', 'f'],
        'recapitulation': ['mp', 'mf', 'f', 'ff'],
    },
    'rondo': {
        'A': ['mf', 'mf'],
        'B': ['f'],
        'C': ['mp'],
        'D': ['ff'],
    },
    'theme_variations': {
        'theme': ['mp', 'mf'],
        'var1': ['mf', 'f'],
        'var2': ['f', 'mf'],
        'var3': ['ff', 'mp'],
    },
}


@dataclass
class CompositionPlan:
    """LLM-generated plan for the composition."""
    title: str
    style: str
    form: str
    key: str
    mode: str
    tempo: int
    total_measures: int
    emotional_arc: List[str]
    sections: List[Dict]
    themes: List[Dict]
    progressions: List[Dict]


def _generate_smooth_melody(
    scale: List[int],
    key_root: int,
    mode: str,
    num_bars: int,
    pitch_range: Tuple[int, int],
    emotion: str = 'peaceful',
    style: str = 'classical',
    section: str = 'A',
    phrase_position: float = 0.0,
    starting_pitch: Optional[int] = None,
    chord_progression: Optional[List[Dict]] = None,
    theme_pattern: Optional[List] = None,
    theme_state: str = 'stating'
) -> List[Dict]:
    """Generate MUSICAL melody with sophistication.
    
    Principles from Bach/Mozart/Beethoven:
    1. Stepwise motion (70%) - smooth, flowing
    2. Skips of 3rd/4th (20%) - for interest
    3. Leaps of 5th/6th (10%) - dramatic moments
    4. Varied rhythms - quarter, half, dotted, syncopation
    5. ornamentation for baroque
    6. CHORD TONES - melody follows harmony
    7. THEME STATE - stating, developing, or resolving
    """
    melody = []
    current_beat = 0.0
    min_pitch, max_pitch = pitch_range
    
    # CHORD AWARENESS: Get current chord based on beat
    def get_current_chord(beat: float) -> Tuple[str, List[int]]:
        """Get chord for current beat and its chord tones."""
        if chord_progression:
            for chord_info in chord_progression:
                chord_start = chord_info.get('beat', 0)
                chord_dur = chord_info.get('duration', 4.0)
                if chord_start <= beat < chord_start + chord_dur:
                    chord_tones = chord_info.get('chord_scale_degrees', [0, 2, 4])
                    return (chord_info.get('chord', 'I'), chord_tones)
        # Default: tonic chord
        return ('I', [0, 2, 4])
    
    # FULL RANGE - Use entire pitch_range passed in (could be 8 octaves)
    # But keep MELODIC motion smooth - stepwise with occasional skips
    
    # Ensure valid MIDI range
    melody_low = max(21, min_pitch)   # A0 minimum
    melody_high = min(108, max_pitch)  # C8 maximum
    
    if melody_high <= melody_low:
        melody_high = melody_low + 24  # At least 2 octaves
    
    # Place tonic in CENTER of the range (not at bottom)
    range_center = (melody_low + melody_high) // 2
    tonic_octave = (range_center // 12) * 12
    tonic_midi = tonic_octave + key_root
    
    # Center the melody around tonic - DO NOT force tonic above melody_low!
    # The melody should span around the tonic, not force tonic into a narrow range
    if tonic_midi < melody_low:
        # Add octaves until tonic is in range
        while tonic_midi < melody_low:
            tonic_midi += 12
    if tonic_midi > melody_high:
        # Subtract octaves if tonic is too high
        while tonic_midi > melody_high:
            tonic_midi -= 12
    
    # SOPHISTICATED PATTERNS with varied rhythms and intervals
    # Format: (scale_degree, duration, interval_type)
    # interval_type: 's' = step, 'k' = skip (3rd/4th), 'l' = leap (5th+)
    
    # OPENING PATTERNS - Multiple distinct opening styles for variety
    # Each has unique character and starting gesture
    opening_patterns = {
        # BACH-STYLE: Rising arpeggio opening
        'bach_rise': [
            (0, 'q'), (2, 'q'), (4, 'q'), (2, 'q'),
            (4, 'q'), (5, 'q'), (4, 'q'), (2, 'q'),
            (0, 'h'), (4, 'h'),
            (2, 'q'), (0, 'h'),
        ],
        # MOZART-STYLE: Graceful turn opening  
        'mozart_grace': [
            (0, 'd'), (2, 'e'), (4, 'e'), (5, 'e'),
            (4, 'q'), (2, 'q'), (0, 'h'),
            (2, 'q'), (4, 'q'), (5, 'q'), (4, 'q'),
            (2, 'h'), (0, 'h'),
        ],
        # BEETHOVEN-STYLE: Bold assertive opening
        'beethoven_bold': [
            (0, 'h'), (4, 'q'), (2, 'q'),
            (0, 'q'), (2, 'q'), (4, 'h'),
            (5, 'q'), (4, 'q'), (2, 'q'), (0, 'q'),
            (2, 'h'), (0, 'h'),
        ],
        # CHOPIN-STYLE: Lyrical flowing opening
        'chopin_lyric': [
            (4, 'h'), (2, 'q'), (0, 'q'), (2, 'q'),
            (4, 'd'), (5, 'e'), (4, 'e'),
            (2, 'h'), (0, 'q'), (2, 'q'),
            (4, 'h'), (0, 'h'),
        ],
        # DEBUSSY-STYLE: Impressionistic opening
        'debussy_float': [
            (0, 'h'), (4, 'h'),
            (5, 'q'), (4, 'q'), (2, 'h'),
            (0, 'q'), (4, 'q'), (2, 'q'), (0, 'q'),
            (4, 'h'), (0, 'h'),
        ],
        # JAZZ-STYLE: Syncopated opening
        'jazz_syncopated': [
            (0, 'd'), (4, 'e'), (2, 'e'),
            (0, 'q'), (2, 'q'), (4, 'q'), (2, 'q'),
            (0, 'q'), (4, 'q'), (5, 'q'), (4, 'q'),
            (2, 'h'), (0, 'h'),
        ],
        # FOLK-STYLE: Simple memorable opening
        'folk_simple': [
            (0, 'q'), (2, 'q'), (4, 'q'), (5, 'q'),
            (4, 'q'), (2, 'q'), (0, 'h'),
            (2, 'q'), (4, 'q'), (2, 'q'), (0, 'q'),
            (4, 'h'), (0, 'h'),
        ],
        # ROMANTIC-STYLE: Emotional sweeping opening
        'romantic_sweep': [
            (0, 'q'), (4, 'q'), (5, 'q'), (6, 'q'),
            (5, 'q'), (4, 'q'), (2, 'q'), (0, 'q'),
            (2, 'q'), (4, 'h'),
            (0, 'q'), (2, 'q'), (0, 'h'),
        ],
    }
    
    # SECONDARY PATTERNS - For development and contrasting sections
    secondary_patterns = {
        # BEETHOVEN-STYLE Transition: Driving, sequential
        'transition': [
            (4, 'q'), (2, 'q'), (0, 'q'), (2, 'q'),
            (4, 'q'), (5, 'q'), (4, 'q'), (2, 'q'),
            (0, 'h'), (4, 'h'),
            (2, 'q'), (0, 'hd'),
        ],
        # CHOPIN-STYLE Fragment: Emotional, varied
        'fragment': [
            (5, 'q'), (4, 'e'), (5, 'e'), (4, 'h'),
            (2, 'q'), (0, 'h'),
            (4, 'q'), (5, 'q'), (6, 'h'),
            (4, 'q'), (2, 'q'), (0, 'hd'),
        ],
        # Sequence: Build tension
        'sequence_up': [
            (0, 'q'), (2, 'q'), (4, 'q'),
            (2, 'q'), (4, 'q'), (5, 'q'),
            (4, 'q'), (5, 'q'), (6, 'q'),
            (7, 'h'), (5, 'q'), (4, 'h'),
        ],
        # Modulation: Key change preparation
        'modulation': [
            (2, 'q'), (4, 'q'), (5, 'q'), (4, 'q'),
            (2, 'q'), (4, 'q'), (6, 'q'), (5, 'q'),
            (4, 'h'), (2, 'h'),
            (0, 'hd'),
        ],
        # Return: Homeward bound
        'return': [
            (4, 'h'), (2, 'h'),
            (0, 'h'), (2, 'q'), (4, 'q'),
            (2, 'h'), (4, 'q'), (2, 'q'),
            (0, 'hd'),
        ],
        # Development: Explore new territory
        'development': [
            (2, 'q'), (4, 'q'), (5, 'q'), (6, 'q'),
            (5, 'q'), (4, 'q'), (2, 'q'), (4, 'q'),
            (0, 'h'), (2, 'h'),
            (4, 'q'), (2, 'q'), (0, 'h'),
        ],
    }
    
    # CLOSING PATTERNS - For endings
    closing_patterns = {
        # Coda: Grand conclusion
        'coda': [
            (0, 'h'), (4, 'h'),
            (2, 'h'), (5, 'h'),
            (4, 'h'), (2, 'q'), (0, 'q'),
            (0, 'w'),
        ],
        # Cadence: Simple resolve
        'cadence': [
            (4, 'q'), (2, 'q'), (0, 'h'),
            (2, 'q'), (4, 'q'), (5, 'h'),
            (4, 'q'), (2, 'q'), (0, 'hd'),
        ],
        # Extended close
        'extended_close': [
            (0, 'q'), (4, 'q'), (0, 'q'), (4, 'q'),
            (2, 'q'), (0, 'q'), (2, 'q'), (4, 'q'),
            (0, 'h'), (4, 'h'),
            (0, 'w'),
        ],
    }
    
    patterns_by_section = {
        'exposition': ['opening', 'secondary', 'transition', 'closing'],
        'development': ['secondary', 'sequence_up', 'fragment'],
        'recapitulation': ['opening', 'transition', 'closing'],
        'A': ['opening'],
        'B': ['secondary'],
        'C': ['secondary', 'transition'],
    }
    
    # Duration mapping
    dur_map = {
        'e': 0.5,      # eighth
        'q': 1.0,      # quarter  
        'd': 1.5,      # dotted quarter
        'h': 2.0,      # half
        'hd': 3.0,     # dotted half
        'w': 4.0,      # whole (for final notes)
        'sq': 0.25,    # sixteenth (for ornaments only)
    }
    
    # GET SECTION PATTERNS WITH RANDOMIZATION
    section_patterns = patterns_by_section.get(section.lower(), ['opening'])
    pattern_idx = min(int(phrase_position * len(section_patterns)), len(section_patterns) - 1)
    pattern_type = section_patterns[pattern_idx]
    
    # THEME DEVELOPMENT: Store theme for section continuity
    # First phrase establishes the theme, subsequent phrases develop it
    is_first_phrase_of_section = (phrase_position < 0.2)
    is_last_phrase_of_section = (phrase_position > 0.8)
    
    # DETERMINE PATTERN BASED ON POSITION
    if is_first_phrase_of_section:
        # First phrase: establish theme with random pattern
        if pattern_type == 'opening':
            pattern_name = _rng.choice(list(opening_patterns.keys()))
            base_pattern = opening_patterns[pattern_name]
        elif pattern_type == 'secondary':
            pattern_name = _rng.choice(list(secondary_patterns.keys()))
            base_pattern = secondary_patterns[pattern_name]
        elif pattern_type == 'closing':
            pattern_name = _rng.choice(list(closing_patterns.keys()))
            base_pattern = closing_patterns[pattern_name]
        else:
            pattern_name = _rng.choice(list(opening_patterns.keys()))
            base_pattern = opening_patterns[pattern_name]
        # Store theme for subsequent phrases
    else:
        # Subsequent phrases: develop the theme, don't create new random pattern
        # Use opening pattern with less variation for continuity
        if pattern_type == 'opening':
            pattern_name = _rng.choice(list(opening_patterns.keys()))
            base_pattern = opening_patterns[pattern_name]
        elif pattern_type == 'secondary':
            pattern_name = _rng.choice(list(secondary_patterns.keys()))
            base_pattern = secondary_patterns[pattern_name]
        elif pattern_type == 'closing':
            pattern_name = _rng.choice(list(closing_patterns.keys()))
            base_pattern = closing_patterns[pattern_name]
        else:
            pattern_name = _rng.choice(list(opening_patterns.keys()))
            base_pattern = opening_patterns[pattern_name]
    
    # APPLY MOTIVIC DEVELOPMENT (not just random variation)
    # Development techniques from classical composition:
    # 1. Sequence - repeat at different pitch level
    # 2. Fragmentation - use only part of theme
    # 3. Extension - add notes to theme
    # 4. Inversion - mirror intervals
    # 5. Retrograde - reverse order
    
    if is_first_phrase_of_section and _rng.random() < 0.25:
        # First phrase: can vary to establish character
        if _rng.random() < 0.4:
            base_pattern = [(6 - deg, r) for deg, r in base_pattern]
            base_pattern = [(deg % 7, r) for deg, r in base_pattern]
        elif _rng.random() < 0.5:
            base_pattern = list(reversed(base_pattern))
        else:
            # Transpose theme up or down
            shift = _rng.choice([2, 4, 5])  # Third, fifth, or fourth
            base_pattern = [((deg + shift) % 7, r) for deg, r in base_pattern]
    elif not is_first_phrase_of_section and not is_last_phrase_of_section:
        # Middle phrases: develop through fragmentation or extension
        if _rng.random() < 0.30:
            # Fragmentation: use first half of theme only
            base_pattern = base_pattern[:len(base_pattern)//2 + 2]
            # Then extend with pattern completion
            pattern_len = len(base_pattern)
            for i in range(len(base_pattern)):
                deg, r = base_pattern[i % pattern_len]
                base_pattern.append((deg, r))
                if len(base_pattern) >= 12:
                    break
    
    # MINIMAL RHYTHMIC VARIATION (5% only, preserve theme)
    if _rng.random() < 0.05 and not is_last_phrase_of_section:
        rhythm_variations = {'q': 'd', 'h': 'hd'}
        new_pattern = []
        for deg, r in base_pattern:
            if r in rhythm_variations and _rng.random() < 0.2:
                new_pattern.append((deg, rhythm_variations[r]))
            else:
                new_pattern.append((deg, r))
        base_pattern = new_pattern
    
    # REPEAT PATTERN TO FILL BAR COUNT
    # Each bar = 4 beats, so num_bars * 4 = total beats needed
    # Pattern covers ~15 beats, repeat it to fill the required duration
    total_beats_needed = num_bars * 4
    base_pattern_beats = sum(dur_map.get(r, 1.0) for _, r in base_pattern)
    
    # Repeat pattern enough times to fill the phrase
    repeats_needed = max(1, int(total_beats_needed / base_pattern_beats) + 1)
    steps = base_pattern * repeats_needed
    
    # Trim to approximate duration (may be slightly shorter or longer)
    current_beats = 0.0
    steps_to_use = []
    for step in steps:
        dur = dur_map.get(step[1], 1.0)
        if current_beats + dur <= total_beats_needed * 1.1:  # Allow 10% over
            steps_to_use.append(step)
            current_beats += dur
    steps = steps_to_use
    
    # Velocity by emotion and style
    vel_map = {
        'peaceful': (50, 68), 'calm': (45, 65),
        'joyful': (65, 82), 'dramatic': (68, 88),
        'triumphant': (72, 95), 'agitated': (65, 82),
    }
    vel_min, vel_max = vel_map.get(emotion, (60, 78))
    
    # Add style-appropriate ornamentation
    
    # Track position for MUSICAL PHRASE SHAPE
    # Classical phrase: rise -> peak -> fall (like a sentence)
    phrase_progress = 0.0  # 0.0 = start, 1.0 = end
    notes_in_phrase = len(steps)
    note_idx = 0
    
    # CRITICAL: Calculate how much of range to use
    # Real melodies span 1-2 octaves, not just 5 notes
    range_span = melody_high - melody_low
    _usable_range = int(range_span * 0.7)
    
    phrase_peak = 0.55 + _rng.random() * 0.2  # Peak at 55-75% through
    
    # Initialize position - use starting_pitch if provided for smooth transitions
    start_transpose = 0  # How many scale degrees to transpose the pattern
    if starting_pitch is not None and starting_pitch >= melody_low and starting_pitch <= melody_high:
        # Find the scale degree of the starting pitch
        pitch_from_tonic = (starting_pitch - key_root) % 12
        # Find closest scale degree
        scale_distances = [abs(pitch_from_tonic - s) for s in scale]
        closest_degree = scale_distances.index(min(scale_distances))
        current_degree = closest_degree
        _starting_octave = (starting_pitch // 12) * 12
        # Transpose pattern so it starts near this degree
        pattern_first_degree = base_pattern[0][0] if base_pattern else 0
        start_transpose = closest_degree - pattern_first_degree
    else:
        # Start on a stable tone (tonic, mediant, or dominant)
        current_degree = _rng.choice([0, 2, 4])
        _center_octave = (melody_low + melody_high) // 2 // 12 * 12
    
    # Track range used
    
    # MELODIC GUIDANCE: Track previous note for stepwise motion
    prev_degree = None
    prev_pitch = None
    stepwise_motion_count = 0
    leap_count = 0
    
    # MELODIC SHAPE MODELING
    # Real melodies have shape: arch (rise-fall), cascade (fall), or ascent
    # Choose shape based on phrase position and emotion
    if emotion == 'dramatic' or emotion == 'triumphant':
        phrase_shapes = ['arch', 'ascent', 'arch']  # Building intensity
    elif emotion == 'peaceful' or emotion == 'calm':
        phrase_shapes = ['arch', 'cascade', 'arch']  # Gentle flow
    else:
        phrase_shapes = ['arch', 'arch', 'cascade']  # Mixed
    
    current_shape = _rng.choice(phrase_shapes)
    phrase_peak = 0.6 if current_shape == 'arch' else (0.8 if current_shape == 'ascent' else 0.3)
    
    for step_degree, rhythm in steps:
        # Apply transposition for continuity with previous phrase
        transposed_degree = (step_degree + start_transpose) % 7
        
        phrase_progress = note_idx / max(1, notes_in_phrase - 1)
        
        # MELODIC SHAPE: Apply arch/cascade/ascent shape
        if current_shape == 'arch':
            # Rise to peak, then fall
            if phrase_progress < phrase_peak:
                target_register = phrase_progress * 5  # Rise
            else:
                target_register = (1 - phrase_progress) * 5  # Fall
        elif current_shape == 'ascent':
            # Gradual rise throughout
            target_register = phrase_progress * 6
        else:  # cascade
            # Fall throughout
            target_register = (1 - phrase_progress) * 5
        
        # Apply register offset based on melodic shape
        register_offset = int(target_register)
        
        # MELODIC SMOOTHNESS: Enforce 65% steps, 35% leaps dynamically
        total_notes = stepwise_motion_count + leap_count
        if total_notes > 0:
            current_step_ratio = stepwise_motion_count / total_notes
            if current_step_ratio < 0.60:
                step_probability = 0.85  # Force more steps
            elif current_step_ratio < 0.65:
                step_probability = 0.75
            elif current_step_ratio > 0.70:
                step_probability = 0.40  # Allow more leaps
            else:
                step_probability = 0.65
        else:
            step_probability = 0.65
        
        if prev_degree is not None and _rng.random() < step_probability:
            # Stepwise motion - move by 1 or 2 scale degrees
            degree_distance = (transposed_degree - prev_degree) % 7
            if degree_distance > 4:
                degree_distance = 7 - degree_distance
            
            if degree_distance <= 2:
                current_degree = transposed_degree
            else:
                # Step toward the transposed degree
                if transposed_degree > prev_degree:
                    current_degree = (prev_degree + _rng.choice([1, 2])) % 7
                else:
                    current_degree = (prev_degree - _rng.choice([1, 2])) % 7
            stepwise_motion_count += 1
        else:
            current_degree = transposed_degree
            leap_count += 1
        
        # MINIMAL ornamentation (3% for stability)
        if _rng.random() < 0.03:
            # Only apply ornamentation if not in last 3 notes
            remaining_notes = notes_in_phrase - note_idx - 1
            if remaining_notes > 3:
                current_degree = (current_degree + _rng.choice([-1, 1])) % 7
        
        # CHORD AWARENESS: Strong coordination with harmony
        # Get current chord and its tones
        chord_info = get_current_chord(current_beat)
        chord_tones = chord_info[1]
        chord_info[0]
        
        # DOWNBEAT: Beat positions 0, 4, 8, 12... (start of each measure)
        is_downbeat = (int(current_beat) % 4) == 0
        is_strong_beat = (int(current_beat) % 4) in [0, 2]  # Beats 1 and 3
        
        # PHRASE END: Last 3-4 notes should resolve properly
        remaining_notes = notes_in_phrase - note_idx - 1
        is_phrase_end = remaining_notes <= 2
        is_final_note = remaining_notes == 0
        is_penultimate = remaining_notes == 1
        
        if is_phrase_end:
            # STABLE RESOLUTION PROGRESSION
            # Last 3 notes: approach → dominant → tonic
            # Or: approach → leading tone → tonic
            
            # Get current chord for resolution
            current_chord_tones = chord_tones
            
            if is_final_note:
                # FINAL NOTE: Must resolve to tonic (degree 0)
                # This is the most important resolution point
                if 0 in current_chord_tones:
                    current_degree = 0  # Strong resolution to tonic
                else:
                    # If tonic not in chord, find nearest stable tone
                    stable_tones = [0, 4]  # Tonic and dominant are most stable
                    for stable in stable_tones:
                        if stable in current_chord_tones:
                            current_degree = stable
                            break
                    else:
                        # Fallback: resolve to nearest chord tone
                        distances = [(abs((ct - current_degree) % 7), ct) for ct in current_chord_tones]
                        current_degree = min(distances)[1]
                        
            elif is_penultimate:
                # PENULTIMATE NOTE: Prepare resolution
                # Should be leading tone (6) or dominant (4) or supertonic (1)
                # depending on the chord
                
                # Prefer motion toward resolution
                if mode == 'major':
                    # In major: degree 6 → 0 (leading tone resolution)
                    # or degree 4 → 0 (dominant resolution)
                    if _rng.random() < 0.6:
                        # Leading tone approach (stronger resolution)
                        current_degree = 6  # Leading tone will resolve to tonic
                    else:
                        # Dominant approach
                        current_degree = 4
                else:
                    # In minor: degree 7 → 0 (raised leading tone)
                    current_degree = 5  # Will become leading tone
                    
            else:
                # Third-to-last note: any chord tone to prepare
                if current_degree not in current_chord_tones:
                    distances = [(abs((ct - current_degree) % 7), ct) for ct in current_chord_tones]
                    current_degree = min(distances)[1]
        
        # DOWNBEAT RULE: Strong beats must be chord tones
        if is_downbeat or is_strong_beat:
            if current_degree not in chord_tones:
                # Find nearest chord tone
                distances = [(abs((ct - current_degree) % 7), ct) for ct in chord_tones]
                nearest = min(distances)[1]
                # Move to chord tone by step (smooth voice leading)
                if abs(nearest - current_degree) <= 2:
                    current_degree = nearest
                elif nearest > current_degree:
                    current_degree = (current_degree + 1) % 7
                else:
                    current_degree = (current_degree - 1) % 7
        else:
            # WEAK BEAT: Can use non-chord tones (passing, neighbor)
            # Non-chord tone handling - only on weak beats
            if chord_tones and _rng.random() < 0.70:
                # 70% prefer chord tones even on weak beats
                if current_degree not in chord_tones:
                    distances = [(abs((ct - current_degree) % 7), ct) for ct in chord_tones]
                    nearest_chord_tone = min(distances)[1]
                    current_degree = nearest_chord_tone
        
        # LEADING TONE RESOLUTION: Degree 6 should resolve to tonic (0)
        if prev_degree is not None and prev_degree == 6:
            # After leading tone, resolve to tonic
            if _rng.random() < 0.85:  # 85% resolve to tonic (increased from 70%)
                current_degree = 0
        
        # Wrap degree
        current_degree = current_degree % 7
        
        # Calculate absolute pitch
        semitones = scale[current_degree]
        midi_pitch = tonic_midi + semitones + register_offset
        
        # SCALE VALIDATION: Ensure all notes are in the key's scale
        # The register_offset can push notes outside the scale
        chromatic_pitch = (midi_pitch - key_root) % 12
        if chromatic_pitch not in scale:
            distances = [abs(s - chromatic_pitch) for s in scale]
            nearest_scale_note = scale[distances.index(min(distances))]
            midi_pitch = (midi_pitch // 12) * 12 + key_root + nearest_scale_note
            if midi_pitch < melody_low:
                midi_pitch += 12
            elif midi_pitch > melody_high:
                midi_pitch -= 12
        
        # Track previous for next iteration
        prev_degree = current_degree
        
        # Apply range constraints - allow wide range
        while midi_pitch < melody_low:
            midi_pitch += 12
        while midi_pitch > melody_high:
            midi_pitch -= 12
        
        # SPECIAL HANDLING FOR FIRST NOTE: Ensure continuity with previous phrase
        # If this is the first note and we have a starting pitch, get close to it
        if prev_pitch is None and starting_pitch is not None:
            # Find the octave closest to starting_pitch
            pitch_diff = midi_pitch - starting_pitch
            if abs(pitch_diff) > 6:  # More than a tritone away
                # Try adjacent octaves
                candidates = [midi_pitch, midi_pitch + 12, midi_pitch - 12]
                candidates = [p for p in candidates if melody_low <= p <= melody_high]
                if candidates:
                    midi_pitch = min(candidates, key=lambda p: abs(p - starting_pitch))
        
        # MELODIC SMOOTHNESS: Smooth octave jumps caused by range wrapping
        if prev_pitch is not None:
            interval = abs(midi_pitch - prev_pitch)
            if interval > 4:  # More than a third - try adjacent octaves
                candidates = [midi_pitch]
                alt_up = midi_pitch + 12
                alt_down = midi_pitch - 12
                if melody_low <= alt_up <= melody_high:
                    candidates.append(alt_up)
                if melody_low <= alt_down <= melody_high:
                    candidates.append(alt_down)
                if len(candidates) > 1:
                    midi_pitch = min(candidates, key=lambda p: abs(p - prev_pitch))
        
        prev_pitch = midi_pitch
        
        # Get duration
        duration = dur_map.get(rhythm, 1.0)
        
        # PHRASE END: Longer notes for stability
        if is_phrase_end:
            duration = max(duration, 1.5)  # At least 1.5 beats at phrase end
            if note_idx == notes_in_phrase - 1:
                duration = max(duration, 2.0)  # Last note at least 2 beats
        
        # AVOID SKIDDISH: Minimum duration for short notes
        # Very short notes (< 0.5) sound "skiddish" - avoid them
        if duration < 0.5:
            duration = 0.5  # Minimum duration
        
        # Velocity based on phrase position
        is_stable = current_degree in [0, 2, 4]
        is_downbeat = (len(melody) % 4) == 0
        is_climax = abs(phrase_progress - phrase_peak) < 0.15
        
        vel = vel_min + int((vel_max - vel_min) * (0.7 if is_stable else 0.5))
        if is_downbeat:
            vel = min(vel_max + 5, vel + 8)
        if is_climax:
            vel = min(vel_max + 10, vel + 10)
        
        melody.append({
            'pitch': midi_pitch,
            'duration': duration,
            'velocity': vel,
            'beat': current_beat,
            'scale_degree': current_degree,
        })
        
        current_beat += duration
        note_idx += 1
    
    return melody


def _generate_functional_harmony(
    progression: List[str],
    key_root: int,
    mode: str,
    num_chords: int,
    phrase_position: float = 0.0,
    is_cadence: bool = False
) -> List[Dict]:
    """Generate harmony with functional understanding and proper form.
    
    Creates chord progressions with:
    - Proper cadences (V-I, V7-I, etc.)
    - Functional harmony (tonic, predominant, dominant)
    - Variety through chord substitutions
    """
    
    harmony = []
    
    # Function mapping with more detail
    function_map = {
        'I': 'tonic', 'i': 'tonic', 'I6': 'tonic', 'I6/4': 'tonic',
        'ii': 'predominant', 'ii6': 'predominant', 'ii°': 'predominant', 'ii°7': 'predominant',
        'iii': 'tonic_substitute', 
        'IV': 'predominant', 'IV6': 'predominant', 'iv': 'predominant',
        'V': 'dominant', 'V7': 'dominant', 'V7/V': 'secondary_dominant', 'V6': 'dominant',
        'vi': 'tonic_substitute', 'VI': 'tonic_substitute', 'vi7': 'tonic_substitute',
        'vii°': 'dominant', 'vii°7': 'dominant',
        'bII6': 'predominant', 'bVI': 'predominant', 'N6': 'predominant',
        'Imaj7': 'tonic', 'IVmaj7': 'predominant', 'V7alt': 'dominant',
    }
    
    # Inversion mapping
    
    # Chord substitutions for variety
    substitute_chords = {
        'I': ['I', 'I6', 'Imaj7', 'vi'],
        'IV': ['IV', 'IV6', 'ii6', 'ii'],
        'V': ['V', 'V7', 'V6', 'vii°7'],
        'ii': ['ii', 'ii6', 'IV'],
        'vi': ['vi', 'IV'],
    }
    
    # Create structured progression based on phrase position
    # Early phrase: Establish tonic
    # Mid phrase: Develop with predominant/dominant
    # Late phrase (cadence): Resolve V-I
    
    if is_cadence and phrase_position > 0.7:
        # CADENCE FORMULA: Strong V-I resolution
        structured_prog = ['V7', 'I']
        if phrase_position < 0.85:
            # Add predominant before V
            structured_prog = ['ii', 'V7', 'I']
    else:
        # Use the progression with form
        structured_prog = list(progression)
    
    # HARMONIC RHYTHM: Vary chord durations based on phrase position
    # Early: Longer chords (4 beats) to establish tonic
    # Mid: Shorter chords (2 beats) for development
    # End: Longer chords (4+ beats) for cadence
    
    # Build the harmony
    chords_to_use = structured_prog[:num_chords]
    while len(chords_to_use) < num_chords:
        chords_to_use.extend(structured_prog)
    chords_to_use = chords_to_use[:num_chords]
    
    # Previous chord for voice leading
    prev_chord_degrees = None
    
    for i, chord in enumerate(chords_to_use):
        # Parse chord numeral - strip suffixes while preserving b-prefix
        chord_raw = chord.split('/')[0]  # Remove secondary dominant /V etc.
        chord_without_dim = chord_raw.split('°')[0]  # Remove diminished mark
        if 'maj7' in chord_without_dim:
            chord_numeral = chord_without_dim.replace('maj7', '').replace('7', '')
        elif 'alt' in chord_without_dim:
            chord_numeral = chord_without_dim.replace('alt', '').replace('7', '')
        else:
            # Strip 7, 6, 4, #5 suffixes but preserve b-prefix and °
            import re as _re_numeral
            chord_numeral = _re_numeral.sub(r'[764](?:/\d+)?$', '', chord_without_dim)
            chord_numeral = _re_numeral.sub(r'#5$', '', chord_numeral)
            # Ensure b-prefix preserved: bVI, bII, bVII
            if not chord_numeral:
                chord_numeral = chord_raw
        
        chord_func = function_map.get(chord_numeral, 'tonic')
        
        # Substitute for variety (but not at cadences)
        if not is_cadence and chord_numeral in substitute_chords and _rng.random() < 0.15:
            chord = _rng.choice(substitute_chords[chord_numeral])
            chord_numeral = chord.split('/')[0].split('°')[0].replace('7', '').replace('6', '')
            chord_func = function_map.get(chord_numeral, 'tonic')
        
        # HARMONIC RHYTHM: Context-aware duration
        if is_cadence and i == len(chords_to_use) - 1:
            duration = 4.0  # Final chord holds
        elif is_cadence:
            duration = 2.0  # Cadence chords move faster
        elif i == 0:
            duration = 4.0  # First chord establishes
        elif phrase_position < 0.3:
            duration = 4.0  # Opening: slower harmonic rhythm
        elif phrase_position > 0.7:
            duration = 2.0  # Approaching cadence: faster
        else:
            duration = 4.0  # Development: moderate
        
        # DETERMINE INVERSION for smooth bass voice leading
        inversion = 'root'
        current_chord_degrees = _get_chord_tones(chord_numeral, mode)
        
        if prev_chord_degrees and i > 0:
            common_tones = set(prev_chord_degrees) & set(current_chord_degrees)
            
            prev_root = prev_chord_degrees[0]
            curr_root = current_chord_degrees[0]
            leap_size = abs(curr_root - prev_root) % 7
            
            if leap_size > 2 and _rng.random() < 0.5:
                inversion = 'first'
            elif len(common_tones) >= 2:
                inversion = 'root'
        
        # Update previous for next iteration
        prev_chord_degrees = current_chord_degrees
        
        harmony.append({
            'chord': chord,
            'function': chord_func,
            'duration': duration,
            'inversion': inversion,
            'beat': i * duration if i == 0 else harmony[-1]['beat'] + harmony[-1]['duration'],
            'chord_scale_degrees': current_chord_degrees,  # Correct chord tones
            'numeral': chord_numeral,  # Store for debugging
        })
    
    return harmony


def _get_chord_tones(chord_numeral: str, mode: str) -> List[int]:
    """Get the scale degrees that belong to a chord.
    
    For C major:
    - I = C-E-G = degrees [0, 2, 4] (root, third, fifth)
    - V = G-B-D = degrees [4, 6, 1] (dominant, leading tone, supertonic)
    
    IMPORTANT: Scale degrees wrap within [0-6], then converted to actual pitches
    using the scale intervals.
    """
    # Chord intervals from root (in scale degrees)
    # Major triad: root, major 3rd (2 steps), perfect 5th (4 steps from root in scale)
    # Minor triad: root, minor 3rd (1.5 steps = scale degrees +1.5), perfect 5th
    # Diminished: root, minor 3rd, diminished 5th
    
    # Scale degree positions from root of chord
    chord_intervals = {
        # MAJOR CHORDS
        'I': [0, 2, 4],           # C-E-G in C major: tonic triad
        'ii': [1, 3, 5],          # D-F-A in C major: minor supertonic
        'iii': [2, 4, 6],         # E-G-B in C major: minor mediant
        'IV': [3, 5, 7],          # F-A-C: but 7 wraps to 0, so [3, 5, 0]
        'V': [4, 6, 8],           # G-B-D: but 8 wraps to 1, so [4, 6, 1]
        'vi': [5, 7, 9],          # A-C-E: wraps to [5, 0, 2]
        'vii°': [6, 8, 10],       # B-D-F: diminished, wraps to [6, 1, 3]
        
        # SEVENTH CHORDS
        'V7': [4, 6, 1, 3],       # G-B-D-F: dominant 7th
        'I7': [0, 2, 4, 6],       # Major 7th
        'ii7': [1, 3, 5, 7],      # Minor 7th (wraps)
        'vi7': [5, 7, 2, 4],      # Minor 7th on vi
        
        # INVERSIONS (different bass, same tones)
        'I6': [2, 4, 7],          # First inversion: E-G-C (3-5-1)
        'IV6': [5, 7, 10],        # First inversion: A-C-F
        
        # MINOR CHORDS (for minor keys)
        'i': [0, 2, 4],           # Minor tonic (flat 3rd, perfect 5th in minor)
        'iv': [3, 5, 8],          # Minor subdominant
        'v': [4, 6, 9],           # Minor dominant (rare, usually V from harmonic minor)
        'VII': [5, 7, 10],        # Major VII in minor
        
        # DIMINISHED
        'ii°': [1, 3, 6],          # Diminished supertonic
        'vii°7': [6, 1, 3, 6],     # Diminished 7th
        
        # FLAT/NAPOLITAN CHORDS (modal mixture and chromatic)
        'bII': [1, 3, 5],          # Neapolitan (Db major in C): flat 2nd degree
        'bVI': [5, 0, 3],          # Flat 6th (Ab major in C): bVI degree
        'bVII': [6, 1, 3],         # Flat 7th (Bb major in C): bVII degree
        'N6': [1, 3, 6],           # Neapolitan 6th
        'V7/V': [4, 6, 1, 3],     # Secondary dominant of V (= II7)
        'I6/4': [0, 2, 4],        # Cadential 6/4 (same as I triad)
        'I6/9': [0, 2, 4, 6],     # Added 9th
        'I+': [0, 2, 4],           # Augmented (fall back to major)
        'vi7#5': [5, 0, 2, 4],    # vi7 with raised 5th
        'V7alt': [4, 6, 1, 3],    # Altered dominant (same tones as V7)
        'Imaj7': [0, 2, 4, 6],    # Major 7th on I
        'IVmaj7': [3, 5, 0, 2],   # Major 7th on IV
        'III': [2, 4, 6],          # Major mediant (in minor)
    }
    
    # Helper: wrap scale degrees to 0-6
    def wrap_degrees(degrees):
        return [(d % 7) for d in degrees]
    
    # Check direct match first
    if chord_numeral in chord_intervals:
        return wrap_degrees(chord_intervals[chord_numeral])
    
    # Parse roman numeral
    numerals_major = dict(sorted({'I': 0, 'ii': 1, 'iii': 2, 'IV': 3, 'V': 4, 'vi': 5, 'vii': 6}.items(), key=lambda x: len(x[0]), reverse=True))
    numerals_minor = dict(sorted({'i': 0, 'ii': 1, 'III': 2, 'iv': 3, 'V': 4, 'VI': 5, 'VII': 6, 'vii': 6}.items(), key=lambda x: len(x[0]), reverse=True))
    
    root = 0
    
    # Try major roman numerals
    for numeral, val in numerals_major.items():
        if chord_numeral.startswith(numeral) or chord_numeral == numeral:
            root = val
            break
    
    # Try minor roman numerals
    for numeral, val in numerals_minor.items():
        if chord_numeral.startswith(numeral) or chord_numeral == numeral:
            root = val
            if numeral.islower():
                pass
            break
    
    # Build triad from root
    # In major: I, IV, V are major; ii, iii, vi are minor; vii° is diminished
    # In minor: i, iv are minor; V and VII are major; ii° is diminished
    
    is_diminished = '°' in chord_numeral or chord_numeral in ['vii', 'ii°', 'vii°']
    
    # Determine chord quality based on scale degree and mode
    # Scale degrees: 0=root, 2=third, 4=fifth (in scale degrees)
    if mode == 'minor':
        # Natural minor: i, iv are minor; III, VI, VII are major; ii°, v, vi are minor/dim
        if root in [0, 3]:  # i, iv
            return wrap_degrees([root, root + 2, root + 4])
        elif root in [2, 5, 6]:  # III, VI, VII (from natural minor)
            return wrap_degrees([root, root + 2, root + 4])  # Major triad
        elif is_diminished:
            return wrap_degrees([root, root + 2, root + 5])  # Diminished
        else:
            # Harmonic minor V (major)
            return wrap_degrees([root, root + 2, root + 4])
    else:
        # Major mode
        if root in [0, 3, 4]:  # I, IV, V are major
            return wrap_degrees([root, root + 2, root + 4])
        elif root in [1, 2, 5]:  # ii, iii, vi are minor
            return wrap_degrees([root, root + 2, root + 4])  # Same intervals, different quality in key
        elif is_diminished or root == 6:  # vii° is diminished
            return wrap_degrees([root, root + 2, root + 5])
        else:
            return wrap_degrees([root, root + 2, root + 4])


def _generate_bass_line(harmony: List[Dict], key_root: int) -> List[Dict]:
    """Generate bass line from harmony with rhythmic variety."""
    
    bass = []
    
    degree_map = {
        'I': 0, 'i': 0, 'Imaj7': 0, 'I6': 0,
        'ii': 1, 'ii°': 1, 'ii6': 1, 'ii7': 1,
        'iii': 2, 'III': 2,
        'IV': 3, 'iv': 3, 'IV6': 3, 'IVmaj7': 3,
        'V': 4, 'V7': 4, 'v': 4, 'V6': 4, 'V7alt': 4,
        'vi': 5, 'VI': 5, 'vi7': 5, 'vi7#5': 5,
        'vii°': 6, 'vii°7': 6, 'VII': 6, 'VII7': 6,
        'bII': 1, 'bVI': 5, 'bII6': 1, 'bVII': 6,
        'N6': 1, 'I6/4': 0, 'I6/9': 0, 'I+': 0, 'I7': 0,
    }
    
    minor_chords = {'i', 'ii', 'ii6', 'ii°', 'iii', 'vi', 'vi7', 'iv', 'vii°', 'vii°7', 'N6'}
    diminished_chords = {'ii°', 'vii°', 'vii°7'}
    
    semitone_offsets = {0: 0, 1: 2, 2: 4, 3: 5, 4: 7, 5: 9, 6: 11}
    
    for h in harmony:
        chord_raw = h['chord']
        chord = chord_raw.split('/')[0].split('°')[0].split('7')[0].replace('maj7', '').replace('alt', '').replace('#5', '')
        degree = degree_map.get(chord_raw, degree_map.get(chord, 0))
        
        root_pitch = key_root + semitone_offsets.get(degree, degree * 2) + 48
        root_pitch = max(36, min(60, root_pitch))
        
        if h['inversion'] == 'first':
            third_semitones = 3 if (chord_raw in minor_chords or chord_raw in diminished_chords) else 4
            root_pitch = (root_pitch + third_semitones) % 12 + 48
        elif h['inversion'] == 'second':
            root_pitch = (root_pitch + 7) % 12 + 48
        
        chord_dur = h['duration']
        chord_beat = h['beat']
        
        fifth_pitch = min(root_pitch + 7, 60)
        if fifth_pitch < root_pitch:
            fifth_pitch = root_pitch + 7
        
        pattern = _rng.choice(['half', 'quarter', 'root_fifth', 'walking'])
        
        if pattern == 'half' or chord_dur <= 2.0:
            bass.append({
                'pitch': root_pitch,
                'duration': chord_dur * 0.85,
                'velocity': 50 + _rng.randint(-5, 10),
                'beat': chord_beat
            })
        elif pattern == 'quarter':
            beat_in_chord = 0.0
            q_len = chord_dur / 2.0
            bass.append({
                'pitch': root_pitch,
                'duration': q_len * 0.85,
                'velocity': 50 + _rng.randint(0, 8),
                'beat': chord_beat + beat_in_chord
            })
            beat_in_chord += q_len
            bass.append({
                'pitch': root_pitch,
                'duration': q_len * 0.85,
                'velocity': 45 + _rng.randint(0, 8),
                'beat': chord_beat + beat_in_chord
            })
        elif pattern == 'root_fifth':
            half = chord_dur / 2.0
            bass.append({
                'pitch': root_pitch,
                'duration': half * 0.85,
                'velocity': 52 + _rng.randint(0, 8),
                'beat': chord_beat
            })
            bass.append({
                'pitch': fifth_pitch,
                'duration': half * 0.85,
                'velocity': 45 + _rng.randint(0, 8),
                'beat': chord_beat + half
            })
        elif pattern == 'walking':
            third_pitch = min(root_pitch + 4, 60)
            q_len = chord_dur / 4.0
            notes_in_pattern = [root_pitch, third_pitch, fifth_pitch, root_pitch]
            for j, note_pitch in enumerate(notes_in_pattern):
                bass.append({
                    'pitch': max(36, min(60, note_pitch)),
                    'duration': q_len * 0.75,
                    'velocity': 45 + _rng.randint(0, 12),
                    'beat': chord_beat + j * q_len
                })
    
    return bass


def _get_expression_mark(emotion: str, is_cadence: bool) -> str:
    """Get expression marking for section."""
    marks = {
        'peaceful': 'dolce',
        'calm': 'tranquillo',
        'energetic': 'energico',
        'dramatic': 'drammatico',
        'joyful': 'gioioso',
        'struggle': 'con forza',
        'resolution': 'risoluto',
        'triumphant': 'trionfante',
        'agitated': 'agitato',
        'confident': 'con brio',
        'theme': 'espressivo',
        'development': 'agitato',
        'exploration': 'cantabile',
    }
    
    if is_cadence:
        return 'pocò rit.'
    
    return marks.get(emotion, 'espressivo')


def _get_section_character(section_name: str, form: str) -> str:
    """Get character description for section."""
    characters = {
        'exposition': 'primary thematic material',
        'development': 'thematic transformation',
        'recapitulation': 'thematic return',
        'A': 'main theme',
        'B': 'contrasting theme',
        'C': 'secondary contrasting',
        'D': 'dramatic contrast',
        'theme': 'lyrical presentation',
        'var1': 'simple variation',
        'var2': 'ornamented',
        'var3': 'transformation',
        'coda': 'concluding',
        'episode': 'developmental',
    }
    return characters.get(section_name, 'thematic')


def get_extended_form(form_name: str) -> Optional[Dict]:
    """Get form structure by name."""
    return EXTENDED_FORMS.get(form_name)


def get_progressions_for_style(style: str) -> List[Dict]:
    """Get progressions for a given style."""
    style_progressions = MASTER_PROGRESSIONS.get(style, MASTER_PROGRESSIONS['classical'])
    return [
        {
            'name': name,
            'chords': prog['progression'],
            'function': prog['function'],
            'usage': prog.get('usage', ''),
        }
        for name, prog in style_progressions.items()
    ]


def compose_extended_piece(
    style: str = 'classical',
    form: str = 'sonata',
    key: str = 'C_major',
    mood: str = 'peaceful',
    tempo: int = 90,
    num_phrases: int = None
) -> Dict:
    """Compose an extended piece with sophisticated structure.
    
    Creates longer pieces with:
    - Extended forms (sonata, rondo, theme-variations, fugue)
    - Proper harmonic progressions
    - Theme development and transformation
    - Musical structure with emotional arc
    
    THEME DEVELOPMENT STRUCTURE:
    - Exposition: Theme A stated → Theme A developed → Theme A transitions
    - Development: Theme fragments → Sequences → Modulations
    - Recapitulation: Theme A returns → Theme A resolved
    
    Args:
        style: Musical style (baroque, classical, romantic, modern)
        form: Musical form (sonata, rondo, theme_variations, fugue, invention)
        key: Key with mode (C_major, A_minor, etc.)
        mood: Emotional character (peaceful, joyful, dramatic, etc.)
        tempo: Tempo in BPM
        num_phrases: Override number of phrases (optional)
    
    Returns:
        Dict with phrases, key, mode, tempo, total_measures, etc.
    """
    
    # Parse key
    key_parts = key.split('_')
    key_root_name = key_parts[0]
    key_mode = key_parts[1].lower() if len(key_parts) > 1 else 'major'
    
    # Convert flat names to sharp (e.g., 'Eb' -> 'D#')
    if key_root_name in FLAT_TO_SHARP:
        key_root_name = FLAT_TO_SHARP[key_root_name]
    
    # Key root to MIDI
    key_root_midi = NOTES.index(key_root_name)
    
    # Get scale
    scale_name = 'major' if key_mode == 'major' else 'harmonic_minor'
    scale = SCALES.get(scale_name, SCALES['major'])
    
    # Get form structure
    form_data = EXTENDED_FORMS.get(form, EXTENDED_FORMS['sonata'])
    total_measures = form_data.get('total_measures', 64)
    sections_data = form_data.get('sections', {})
    emotional_arc = form_data.get('emotional_arc', ['peaceful'])
    
    # Get progressions for style
    style_progressions = MASTER_PROGRESSIONS.get(style, MASTER_PROGRESSIONS['classical'])
    
    # Build title
    form_caps = form.replace('_', ' ').title()
    style_caps = style.capitalize()
    title = f"{style_caps} {form_caps} in {key_root_name} {key_mode.capitalize()}"
    
    # Determine number of phrases
    if num_phrases is None:
        num_phrases = max(4, total_measures // 8)  # ~8 measures per phrase
    
    # Get chord progressions
    progression_names = list(style_progressions.keys())
    
    # Initialize tracking
    all_phrases = []
    themes_used = []
    current_beat = 0.0
    last_phrase_end_pitch = None  # Track for smooth transitions
    
    # THEME STATE - Track main theme for development
    # In classical composition, themes are:
    # 1. STATED clearly in exposition (first 1-2 phrases)
    # 2. DEVELOPED through sequences, fragmentation, modulation
    # 3. VARIED with ornamentation, inversion, augmentation
    # 4. RESOLVED in recapitulation
    main_theme_pattern = None  # Store opening pattern
    theme_state = 'stating'  # stating -> developing -> transitioning -> resolving
    
    # Melody range (2 octaves) - centered around middle C for brightness
    melody_low = 60 + key_root_midi % 12  # Middle C area (C4 up)
    melody_high = melody_low + 24  # 2 octaves, bright singing range
    
    # Bass range - in proper bass register (not sub-bass rumble)
    bass_low = 48 + key_root_midi % 12  # C3 area, strong but clear
    _bass_high = bass_low + 12
    
    # Generate phrases for each section
    section_names = list(sections_data.keys())
    
    for section_idx, section_name in enumerate(section_names):
        section_info = sections_data[section_name]
        measures = section_info.get('measures', 16)
        section_key = section_info.get('key', 'tonic')
        
        # Determine key for this section
        if section_key == 'dominant':
            section_root = (key_root_midi + 7) % 12
        elif section_key == 'relative_major':
            section_root = (key_root_midi + 3) % 12 if key_mode == 'minor' else key_root_midi
        elif section_key == 'subdominant':
            section_root = (key_root_midi + 5) % 12
        else:
            section_root = key_root_midi
        
        # THEME STATE TRANSITIONS
        # Exposition: state -> develop -> transition
        # Development: fragment -> sequence -> modulate
        # Recapitulation: return -> resolve
        
        if section_name == 'exposition' or section_name in ['A', 'theme']:
            theme_state = 'stating' if section_idx == 0 else 'developing'
        elif section_name == 'development' or section_name == 'B':
            theme_state = 'developing'
        elif section_name == 'recapitulation' or section_name == 'A_prime':
            theme_state = 'resolving'
        elif section_name in ['C', 'episode']:
            theme_state = 'contrasting'
        
        # Get emotional character for section
        emotion_idx = min(section_idx, len(emotional_arc) - 1)
        emotion = emotional_arc[emotion_idx]
        if mood != 'peaceful':
            # Blend with overall mood
            emotion = emotion if _rng.random() > 0.5 else mood
        
        # Get progressions
        prog_name = _rng.choice(progression_names)
        progression_data = style_progressions[prog_name]
        chords = progression_data['progression']
        
        # SECTION END DETERMINATION
        is_last_section = (section_idx == len(section_names) - 1)
        
        # Generate phrases for this section
        phrases_per_section = max(2, measures // 8)
        
        for phrase_idx in range(phrases_per_section):
            phrase_name = f"{section_name}_{phrase_idx + 1}"
            
            # Determine if this is a cadence phrase (end of section)
            is_cadence = phrase_idx == phrases_per_section - 1
            is_section_end = is_cadence and not is_last_section
            is_final_cadence = is_cadence and is_last_section
            
            phrase_position = (section_idx * phrases_per_section + phrase_idx) / max(1, len(section_names) * phrases_per_section - 1)
            
            # CADENCE TYPE SELECTION
            # Final cadence: Perfect authentic (V-I, strongest)
            # Section end: Strong cadence but not final
            # Mid-phrase: Weak cadence (half)
            if is_final_cadence:
                target_cadence = 'perfect_authentic'
            elif is_section_end:
                target_cadence = 'plagal' if _rng.random() < 0.3 else 'perfect_authentic'
            elif is_cadence:
                # Section-internal cadence: weaker
                target_cadence = 'half' if _rng.random() < 0.6 else 'deceptive'
            else:
                target_cadence = 'half'
            
            # PERIOD STRUCTURE
            if THEORY_AVAILABLE and phrases_per_section >= 2:
                try:
                    phrase_char = PeriodStructure.get_phrase_character(phrase_idx, phrases_per_section)
                    if phrase_char.get('cadence') == 'weak':
                        target_cadence = 'half'
                except Exception as e:
                    logger.debug(f"Phrase cadence check failed: {e}")
            
            # Choose progression - generate enough chords for full phrase
            phrase_measures = measures // phrases_per_section
            phrase_beats = phrase_measures * 4  # 4 beats per measure
            
            # SELECT APPROPRIATE CADENCE BASED ON PHRASE POSITION
            if is_cadence or is_section_end:
                # CADENCE PROGRESSIONS
                if THEORY_AVAILABLE:
                    try:
                        cadence_chords = Cadences.get_cadence_chords(
                            target_cadence, section_root, scale, key_mode
                        )
                        # Replace end of progression with cadence
                        phrase_chords = chords[:min(2, len(chords))]
                        phrase_chords.extend([c['chord'] for c in cadence_chords])
                    except Exception:
                        phrase_chords = chords[:min(4, len(chords))]
                else:
                    # Fallback cadence progressions
                    if target_cadence == 'perfect_authentic':
                        phrase_chords = chords[:2] + ['V7', 'I']
                    elif target_cadence == 'plagal':
                        phrase_chords = chords[:2] + ['IV', 'I']
                    elif target_cadence == 'half':
                        phrase_chords = chords[:2] + ['ii', 'V']
                    elif target_cadence == 'deceptive':
                        phrase_chords = chords[:2] + ['V', 'vi']
                    else:
                        phrase_chords = chords[:min(4, len(chords))]
            else:
                phrase_chords = chords[:min(4, len(chords))]
            
            # Generate enough chord changes to span the phrase
            num_chords_needed = max(4, int(phrase_beats / 4))
            while len(phrase_chords) < num_chords_needed:
                phrase_chords = phrase_chords + [chords[i % len(chords)] for i in range(len(chords))]
                if len(phrase_chords) > num_chords_needed * 2:
                    break
            phrase_chords = phrase_chords[:num_chords_needed]
            
            # GENERATE HARMONY FIRST (so melody can follow chords)
            harmony_notes = _generate_functional_harmony(
                progression=phrase_chords,
                key_root=section_root,
                mode=key_mode,
                num_chords=len(phrase_chords),
                phrase_position=phrase_position,
                is_cadence=is_cadence
            )
            
            # Generate melody with chord awareness
            phrase_measures = measures // phrases_per_section
            pitch_range = (melody_low, melody_high)
            
            # THEME DEVELOPMENT: Pass theme pattern and state
            # First phrase establishes theme, subsequent develop it
            melody_notes = _generate_smooth_melody(
                scale=scale,
                key_root=section_root,
                mode=key_mode,
                num_bars=phrase_measures,
                pitch_range=pitch_range,
                emotion=emotion,
                style=style,
                section=section_name,
                phrase_position=phrase_position,
                starting_pitch=last_phrase_end_pitch,
                chord_progression=harmony_notes,
                theme_pattern=main_theme_pattern,  # Pass saved theme
                theme_state=theme_state  # stating, developing, contrasting, resolving
            )
            
            # Save main theme from first phrase
            if main_theme_pattern is None and section_idx == 0 and phrase_idx == 0:
                # Extract pattern from melody notes
                if melody_notes:
                    main_theme_pattern = [(n.get('scale_degree', 0), 'q') for n in melody_notes[:8]]
            
            # Track ending pitch for next phrase
            if melody_notes:
                last_phrase_end_pitch = melody_notes[-1].get('pitch', None)
            
            # NOTE: Do NOT offset beats here - music_creation_engine handles global timing
            # Keep notes in local phrase time (starting at beat 0)
            
            # Calculate phrase duration - use even measures (multiples of 4 beats)
            phrase_duration = float(phrase_measures) * 4.0
            
            # Generate harmony with clean 4-beat chords
            harmony_notes = _generate_functional_harmony(
                progression=phrase_chords,
                key_root=section_root,
                mode=key_mode,
                num_chords=len(phrase_chords),
                phrase_position=phrase_position,
                is_cadence=is_cadence
            )
            
            # Ensure harmony fits evenly in phrase (no fractional beats)
            # Each chord should be exactly 4 beats
            _num_chord_slots = int(phrase_duration / 4.0)
            for i, h in enumerate(harmony_notes):
                h['beat'] = i * 4.0  # Clean integer beats: 0, 4, 8, 12...
                h['duration'] = 4.0   # Exactly 4 beats per chord
            
            # Generate bass line from harmony (bass inherits correct beats from harmony)
            bass_notes = _generate_bass_line(harmony_notes, section_root)
            
            # Build phrase
            phrase = {
                'name': phrase_name,
                'section': section_name,
                'measures': phrase_measures,
                'key': NOTES[section_root],
                'mode': key_mode,
                'theme_type': 'primary' if section_idx == 0 else 'secondary' if section_idx == 1 else 'development',
                'is_cadence': phrase_idx == phrases_per_section - 1,
                'melody_notes': melody_notes,
                'harmony_notes': harmony_notes,
                'bass_notes': bass_notes,
                'dynamics': _get_expression_mark(emotion, phrase_idx == phrases_per_section - 1),
                'start_beat': current_beat,
                'duration': phrase_duration,
            }
            
            all_phrases.append(phrase)
            current_beat += phrase_duration  # Sequential phrases, no gap (music_creation_engine handles timing)
            
            # Track themes
            if section_idx == 0 and phrase_idx == 0:
                themes_used.append('primary_theme')
            elif section_idx == 1 and phrase_idx == 0:
                themes_used.append('secondary_theme')
    
    # Build composition result
    result = {
        'title': title,
        'phrases': all_phrases,
        'key': f"{key_root_name}_{key_mode}",
        'mode': key_mode,
        'tempo': tempo,
        'total_measures': total_measures,
        'total_duration': current_beat,
        'form': form,
        'style': style,
        'form_analysis': {
            'sections': [{'name': s, 'measures': sections_data[s].get('measures', 16)} for s in section_names],
            'emotional_arc': emotional_arc,
            'themes_used': themes_used,
        },
        'structure': {
            'form': form,
            'total_measures': total_measures,
            'phrases': len(all_phrases),
        },
        'notes': [],  # Will be filled by caller
        'chords': [],  # Will be filled by caller
    }
    
    # Collect all notes for convenience
    all_notes = []
    for phrase in all_phrases:
        for note in phrase.get('melody_notes', []):
            note['phrase'] = phrase['name']
            all_notes.append(note)
    result['notes'] = all_notes
    
    logger.info(f"[LLMComposer] Composed {title}: {len(all_phrases)} phrases, {total_measures} measures")
    
    return result


# Export functions
__all__ = [
    'compose_extended_piece',
    'get_extended_form',
    'get_progressions_for_style',
    'MASTER_PROGRESSIONS',
    'EXTENDED_FORMS',
]
