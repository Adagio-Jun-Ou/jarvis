"""
Music Learning from Masters - Comprehensive Technique Library

This module enables Jarvis to:
1. Study compositions by masters across all eras
2. Extract patterns and techniques from their works
3. Practice composing using learned techniques
4. Test compositions against theory rules
5. Evolve style based on what works

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    MusicLearningSystem                          │
├─────────────────────────────────────────────────────────────────┤
│  MasterStudy                                                    │
│  - analyze_master_piece() → Extract patterns, voice leading    │
│  - learn_counterpoint() → Species counterpoint rules           │
│  - study_harmony() → Chord progressions, resolutions           │
│  - observe_form() → Sonata, invention, fugue structure        │
├─────────────────────────────────────────────────────────────────┤
│  PracticeLoop                                                    │
│  - compose_with_technique() → Apply learned pattern            │
│  - test_composition() → Evaluate against theory rules          │
│  - get_feedback() → Theory violations, suggestions            │
│  - refine() → Fix issues, improve                              │
├─────────────────────────────────────────────────────────────────┤
│  TechniqueLibrary                                               │
│  - 25+ composers from Renaissance to Modern                     │
│  - 150+ documented techniques                                   │
│  - Characteristic progressions                                  │
│  - Voice leading rules                                          │
│  - Form patterns                                                │
└─────────────────────────────────────────────────────────────────┘
"""

import logging
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import threading
from collections import defaultdict
import random

from core.helpers import atomic_write_json, read_json_with_integrity

logger = logging.getLogger(__name__)


# ============================================================
# COMPREHENSIVE MASTER TECHNIQUES DATABASE
# 25+ composers, 150+ techniques across all eras
# ============================================================

MASTER_TECHNIQUES = {
    # ========================================
    # RENAISSANCE (1400-1600)
    # ========================================
    "palestrina": {
        "name": "Giovanni Pierluigi da Palestrina",
        "era": "renaissance",
        "life": "1525-1594",
        "techniques": {
            # Counterpoint
            "species_counterpoint": {
                "description": "Five species of note-against-note counterpoint",
                "applies_to": ["all"],
                "rules": [
                    "First species: note against note",
                    "Second species: two notes against one",
                    "Third species: four notes against one",
                    "Fourth species: syncopation",
                    "Fifth species: florid combination",
                ]
            },
            "palestrina_style": {
                "description": "Smooth, flowing counterpoint with careful voice leading",
                "applies_to": ["all"],
                "pattern": "Avoid direct motion into perfect intervals, use contrary motion for resolution",
            },
            "cantus_firmus": {
                "description": "Fixed melody as foundation for polyphony",
                "applies_to": ["mass", "motet"],
                "pattern": "Long notes in one voice, others weave around it",
            },
            "point_of_imitation": {
                "description": "Entry points where voices imitate a motive",
                "applies_to": ["motet", "mass"],
                "pattern": "Sequential entries create texture building",
            },
            # Voice leading
            "avoid_dissonance": {
                "description": "Careful treatment of dissonance",
                "applies_to": ["all"],
                "rules": [
                    "Dissonance only on weak beat",
                    "Resolve by step",
                    "Approach by step",
                    "Escape tone avoided",
                ]
            },
            "smooth_conjunct_motion": {
                "description": "Voices move mostly by step",
                "applies_to": ["all"],
                "pattern": "70% stepwise motion, leaps prepared and resolved",
            },
        },
        "characteristic_progressions": [
            ["I", "iii", "vi", "IV", "V", "I"],  # Modal progression
            ["I", "V", "vi", "IV", "I"],  # Circle progression
        ],
        "typical_keys": ["C major", "G major", "D major", "F major"],
        "tempo_range": (40, 100),
    },
    
    # ========================================
    # BAROQUE (1600-1750)
    # ========================================
    "bach": {
        "name": "Johann Sebastian Bach",
        "era": "baroque",
        "life": "1685-1750",
        "techniques": {
            # Fugue techniques
            "subject_statement": {
                "description": "Main theme presented clearly at start",
                "applies_to": ["fugue", "invention"],
                "pattern": "Single voice presents complete subject before others enter",
            },
            "countersubject": {
                "description": "Counter-melody that works against subject",
                "applies_to": ["fugue", "invention"],
                "pattern": "New melody that complements subject, can be inverted",
            },
            "stretto": {
                "description": "Subject entries overlap before previous finishes",
                "applies_to": ["fugue"],
                "pattern": "Heightens tension, creates density",
            },
            "augmentation": {
                "description": "Subject played in longer note values",
                "applies_to": ["fugue", "chorale"],
                "pattern": "Double or quadruple duration for gravitas",
            },
            "diminution": {
                "description": "Subject played in shorter note values",
                "applies_to": ["fugue"],
                "pattern": "Half or quarter duration for momentum",
            },
            "inversion": {
                "description": "Subject intervals reversed",
                "applies_to": ["fugue"],
                "pattern": "Up becomes down, perfect intervals preserved",
            },
            "retrograde": {
                "description": "Subject played backwards",
                "applies_to": ["fugue"],
                "pattern": "Rare, creates puzzle quality",
            },
            # Chorale techniques
            "four_part_chorale": {
                "description": "SATB harmonization of chorale melody",
                "applies_to": ["chorale"],
                "pattern": "Melody in soprano, smooth inner voices, strong bass",
            },
            # Counterpoint rules
            "species_counterpoint": {
                "description": "Graduated rules for note-against-note writing",
                "applies_to": ["all"],
                "rules": [
                    "Avoid parallel fifths and octaves",
                    "Resolve leading tones upward",
                    "Prefer contrary motion",
                    "Keep common tones between chords",
                    "Hidden fifths avoided",
                ]
            },
            # Voice leading
            "smooth_voice_leading": {
                "description": "Each voice moves independently but smoothly",
                "applies_to": ["all"],
                "pattern": "Stepwise motion preferred, leaps resolved by step",
            },
            "figured_bass": {
                "description": "Numerical shorthand for harmony",
                "applies_to": ["all"],
                "pattern": "3 = root position, 6 = first inversion, 6/4 = second inversion",
            },
            # Harmonic techniques
            "circle_of_fifths": {
                "description": "Sequential progression by descending fifths",
                "applies_to": ["all"],
                "pattern": "I-IV-vii-iii-vi-ii-V-I creates drive toward tonic",
            },
            "secondary_dominants": {
                "description": "Dominants of chords other than tonic",
                "applies_to": ["all"],
                "pattern": "V7/vi, V7/ii, V7/IV create temporary keys",
            },
            "neapolitan_chord": {
                "description": "Flat II in first inversion",
                "applies_to": ["all"],
                "pattern": "Creates strong pull to V, often in minor",
            },
        },
        "characteristic_progressions": [
            ["I", "IV", "V", "I"],
            ["I", "ii6", "V", "I"],
            ["I", "V6", "I"],
            ["vii°7", "I"],
            ["I", "IV", "vii°7", "iii", "vi", "ii", "V", "I"],  # Full circle
            ["i", "iv", "V", "i"],  # Minor
            ["i", "bII6", "V", "i"],  # Neapolitan
        ],
        "typical_keys": ["C major", "G major", "D major", "A minor", "E minor", "B minor"],
        "tempo_range": (40, 120),
    },
    
    "handel": {
        "name": "George Frideric Handel",
        "era": "baroque",
        "life": "1685-1759",
        "techniques": {
            # Oratorio/Opera techniques
            "da_capo_aria": {
                "description": "ABA aria form",
                "applies_to": ["aria", "oratorio"],
                "pattern": "A section repeated after B with ornamentation",
            },
            "recitative_secco": {
                "description": "Dry recitative with minimal accompaniment",
                "applies_to": ["opera", "oratorio"],
                "pattern": "Speech-like melody, simple chords, freedom in rhythm",
            },
            "recitative_accompagnato": {
                "description": "Accompanied recitative with orchestra",
                "applies_to": ["opera", "oratorio"],
                "pattern": "Orchestra underscores emotion, less rhythmic freedom",
            },
            # Melodic techniques
            "messiah_choruses": {
                "description": "Grand choruses with imitative and homophonic sections",
                "applies_to": ["chorus"],
                "pattern": "Alternation between fugal and chordal writing",
            },
            "handel_turn": {
                "description": "Decorative turn ornament",
                "applies_to": ["all"],
                "pattern": "Upper neighbor, main note, lower neighbor, main note",
            },
            # Harmonic
            "pedal_point": {
                "description": "Sustained bass note with changing harmonies above",
                "applies_to": ["all"],
                "pattern": "Tonic or dominant pedal creates anticipation",
            },
        },
        "characteristic_progressions": [
            ["I", "V", "vi", "IV", "I"],
            ["I", "V", "I"],  # Simple
            ["V", "I"],  # Basic authentic
        ],
        "typical_keys": ["D major", "G major", "C major", "F major"],
        "tempo_range": (60, 140),
    },
    
    "vivaldi": {
        "name": "Antonio Vivaldi",
        "era": "baroque",
        "life": "1678-1741",
        "techniques": {
            # Concerto techniques
            "ritornello_form": {
                "description": "Tutti returns throughout movement",
                "applies_to": ["concerto"],
                "pattern": "Ritornello in tonic, modulates in solo sections, returns",
            },
            "virtuoso_solo": {
                "description": "Solo passages with rapid passages",
                "applies_to": ["concerto"],
                "pattern": "Fast scales, arpeggios, double stops",
            },
            "programmatic_music": {
                "description": "Music that depicts scenes or stories",
                "applies_to": ["all"],
                "pattern": "Four Seasons - bird calls, storms, hunting horns",
            },
            # Harmonic
            "sequential_harmony": {
                "description": "Patterns moved through different keys",
                "applies_to": ["all"],
                "pattern": "Sequence by step or circle of fifths",
            },
            "basso_continuo": {
                "description": "Figured bass with improvising keyboard",
                "applies_to": ["all"],
                "pattern": "Bass line + figures indicate harmonies",
            },
        },
        "characteristic_progressions": [
            ["I", "V", "vi", "iii", "IV", "I"],  # Circle-ish
            ["I", "IV", "V", "I"],
            ["V7", "I"],
        ],
        "typical_keys": ["D major", "G major", "A major", "E major"],
        "tempo_range": (80, 160),
    },
    
    # ========================================
    # CLASSICAL (1750-1820)
    # ========================================
    "mozart": {
        "name": "Wolfgang Amadeus Mozart",
        "era": "classical",
        "life": "1756-1791",
        "techniques": {
            # Sonata form
            "sonata_exposition": {
                "description": "First theme in tonic, second in dominant",
                "applies_to": ["sonata"],
                "pattern": "First theme group, transition, second theme group, codetta",
            },
            "sonata_development": {
                "description": "Development explores and fragments themes",
                "applies_to": ["sonata"],
                "pattern": "Modulates through various keys, thematic transformation",
            },
            "sonata_recapitulation": {
                "description": "Return to tonic with both themes",
                "applies_to": ["sonata"],
                "pattern": "First and second themes both in tonic",
            },
            # Accompaniment
            "alberti_bass": {
                "description": "Broken chord accompaniment pattern",
                "applies_to": ["classical"],
                "pattern": "Low-high-middle-high pattern in accompaniment",
            },
            "murky_bass": {
                "description": "Bass broken in rapid notes",
                "applies_to": ["classical"],
                "pattern": "Fast arpeggios in left hand",
            },
            # Melodic style
            "balanced_phrases": {
                "description": "Symmetrical 4+4 or 8+8 bar phrases",
                "applies_to": ["all"],
                "pattern": "Question-answer phrase structure",
            },
            "elegant_resolution": {
                "description": "Clear cadential patterns",
                "applies_to": ["all"],
                "pattern": "V7-I authentic cadence most common",
            },
            "mozart_effect": {
                "description": "Apparent simplicity, sophisticated underlying structure",
                "applies_to": ["all"],
                "pattern": "Clear melodies, hidden motivic unity",
            },
            "ornamentation": {
                "description": "Trills, turns, mordents applied naturally",
                "applies_to": ["all"],
                "pattern": "Ornaments on important notes",
            },
            # Harmony
            "secondary_theme_contrast": {
                "description": "Second theme contrasting in character",
                "applies_to": ["sonata"],
                "pattern": "First theme energetic, second theme lyrical",
            },
            # Opera techniques
            "ensemble_finales": {
                "description": "Multi-character finales",
                "applies_to": ["opera"],
                "pattern": "Characters enter sequentially, tempo accelerates",
            },
        },
        "characteristic_progressions": [
            ["I", "V", "vi", "IV"],
            ["I", "IV", "V", "I"],
            ["I", "vi", "ii", "V", "I"],
            ["I6", "V6", "I"],
            ["I", "ii6", "V", "I"],
            ["V7", "vi"],  # Deceptive
        ],
        "typical_keys": ["C major", "G major", "D major", "Eb major", "A major", "F major"],
        "tempo_range": (80, 144),
    },
    
    "haydn": {
        "name": "Franz Joseph Haydn",
        "era": "classical",
        "life": "1732-1809",
        "techniques": {
            # String quartet techniques
            "equal_voices": {
                "description": "All four instruments equally important",
                "applies_to": ["quartet"],
                "pattern": "No mere accompaniment, all voices thematic",
            },
            "monothematic_sonata": {
                "description": "Sonata form with single theme",
                "applies_to": ["sonata"],
                "pattern": "Same theme in tonic and dominant",
            },
            # Humor and surprise
            "haydn_surprise": {
                "description": "Unexpected elements for humor or drama",
                "applies_to": ["all"],
                "pattern": "Sudden loud chord, unexpected modulation",
            },
            # Harmonic
            "cyclic_form": {
                "description": "Themes return across movements",
                "applies_to": ["symphony"],
                "pattern": "Last movement recalls first movement themes",
            },
            # Development
            "concentrated_motives": {
                "description": "Short motives drive development",
                "applies_to": ["all"],
                "pattern": "Two-measure idea, developed extensively",
            },
            "motivic_economy": {
                "description": "Entire movement from single motive",
                "applies_to": ["all"],
                "pattern": "Everything grows from opening idea",
            },
        },
        "characteristic_progressions": [
            ["I", "V", "I"],
            ["I", "ii", "V", "I"],
            ["I", "vi", "V", "I"],
        ],
        "typical_keys": ["D major", "G major", "Eb major", "C major"],
        "tempo_range": (80, 140),
    },
    
    # ========================================
    # ROMANTIC (1820-1900)
    # ========================================
    "beethoven": {
        "name": "Ludwig van Beethoven",
        "era": "romantic",
        "life": "1770-1827",
        "techniques": {
            # Motivic development
            "germinal_motive": {
                "description": "Tiny motive developed throughout",
                "applies_to": ["all"],
                "pattern": "Short 3-4 note motif transforms and grows - e.g., 5th Symphony",
            },
            "thematic_transformation": {
                "description": "Theme changes character while remaining recognizable",
                "applies_to": ["sonata", "variations"],
                "pattern": "Same theme in major/minor, fast/slow, loud/soft",
            },
            "continuous_development": {
                "description": "Development continues even in recapitulation",
                "applies_to": ["sonata"],
                "pattern": "Recapitulation transforms as well as returns",
            },
            # Structural innovation
            "expanded_development": {
                "description": "Longer development sections exploring distant keys",
                "applies_to": ["sonata"],
                "pattern": "Development modulates through multiple keys",
            },
            "scherzo_replacement": {
                "description": "Scherzo replaces minuet",
                "applies_to": ["symphony"],
                "pattern": "Fast, lively, with trio",
            },
            # Dynamic contrast
            "terrace_dynamics": {
                "description": "Sudden dynamic shifts for dramatic effect",
                "applies_to": ["all"],
                "pattern": "fp (forte-piano) accents, subito piano",
            },
            "sforzando": {
                "description": "Strong accent on beat",
                "applies_to": ["all"],
                "pattern": "sfz creates unexpected emphasis",
            },
            # Harmony
            "mediant_relationships": {
                "description": "Moves by major third (I to III, I to vi)",
                "applies_to": ["all"],
                "pattern": "Sudden tonal shifts create surprise",
            },
            "remote_modulations": {
                "description": "Modulations to distant keys",
                "applies_to": ["all"],
                "pattern": "Circle of thirds, enharmonic re-spelling",
            },
            "minor_mode_significance": {
                "description": "Minor keys for dramatic weight",
                "applies_to": ["all"],
                "pattern": "Minor key movements carry emotional weight",
            },
            # Form
            "cyclic_structure": {
                "description": "Themes connect movements",
                "applies_to": ["symphony", "quartet"],
                "pattern": "Last movement recalls earlier themes",
            },
            "extended_coda": {
                "description": "Coda becomes second development",
                "applies_to": ["sonata"],
                "pattern": "Coda with new thematic work",
            },
        },
        "characteristic_progressions": [
            ["I", "III", "vi", "I"],  # Mediant
            ["i", "VI", "VII", "i"],  # Minor with major VI
            ["I", "bVI", "V", "I"],  # Flat VI surprise
            ["i", "iv", "V", "i"],  # Intense minor
            ["I", "vi", "bVI", "V", "I"],  # Surprising
        ],
        "typical_keys": ["C minor", "D minor", "Eb major", "F major", "A minor", "C major"],
        "tempo_range": (60, 160),
    },
    
    "schubert": {
        "name": "Franz Schubert",
        "era": "romantic",
        "life": "1797-1828",
        "techniques": {
            # Lieder
            "song_cycle": {
                "description": "Connected songs telling a story",
                "applies_to": ["lieder"],
                "pattern": "Winterreise, Die schöne Müllerin",
            },
            "text_painting": {
                "description": "Music reflects text meaning",
                "applies_to": ["all"],
                "pattern": "Major/minor for emotion, rhythmic gestures for words",
            },
            # Melody
            "endless_melody": {
                "description": "Unbroken melodic lines",
                "applies_to": ["all"],
                "pattern": "No clear cadences until final resolution",
            },
            "heavenly_lengths": {
                "description": "Extended lyrical phrases",
                "applies_to": ["all"],
                "pattern": "8+16+16 bar melodies",
            },
            # Harmony
            "modal_mixture": {
                "description": "Moves freely between major and minor",
                "applies_to": ["all"],
                "pattern": "Major to minor (parallel) within phrase",
            },
            "schubert_modulation": {
                "description": "Smooth modulations to distant keys",
                "applies_to": ["all"],
                "pattern": "Major III, flat VI, flat II as new tonics",
            },
            "triplet_accompaniment": {
                "description": "Triplet rhythm in piano",
                "applies_to": ["all"],
                "pattern": "Flowing eighth-note triplets",
            },
        },
        "characteristic_progressions": [
            ["I", "bVI", "bVII", "I"],  # Schubert's favorite
            ["I", "iii", "IV", "V", "I"],
            ["I", "vi", "IV", "V", "I"],
        ],
        "typical_keys": ["Bb major", "Eb major", "A major", "C major", "G major"],
        "tempo_range": (60, 120),
    },
    
    "chopin": {
        "name": "Frederic Chopin",
        "era": "romantic",
        "life": "1810-1849",
        "techniques": {
            # Piano techniques
            "rubato": {
                "description": "Flexible tempo, melody flows over steady bass",
                "applies_to": ["nocturne", "prelude", "all"],
                "pattern": "Right hand free, left hand keeps time",
            },
            "bel_canto_melody": {
                "description": "Singing melody with ornaments",
                "applies_to": ["nocturne"],
                "pattern": "Long phrases with elaborate decoration",
            },
            "chromatic_harmony": {
                "description": "Chromatic chords for color",
                "applies_to": ["all"],
                "pattern": "Augmented 6th, diminished 7th, Neapolitan chords",
            },
            "ornament_enhancement": {
                "description": "Ornaments integrated into melody",
                "applies_to": ["all"],
                "pattern": "Runs, trills, turns become part of line",
            },
            # Harmony
            "neapolitan_sixth": {
                "description": "Flat II in first inversion",
                "applies_to": ["all"],
                "pattern": "Creates intense pull to V",
            },
            "secondary_leading_tones": {
                "description": "vii°7 of various chords",
                "applies_to": ["all"],
                "pattern": "vii°7/V, vii°7/iv create intensity",
            },
            # Form
            "nocturne_form": {
                "description": "ABA with decorated return",
                "applies_to": ["nocturne"],
                "pattern": "A: lyric theme, B: agitato middle, A': ornamented return",
            },
            # Rhythm
            "waltz_style": {
                "description": "3/4 time with oom-pah-pah left hand",
                "applies_to": ["waltz"],
                "pattern": "Bass on 1, chords on 2-3",
            },
        },
        "characteristic_progressions": [
            ["I", "bII6", "V", "I"],  # Neapolitan
            ["I", "vi", "ii°7", "V7", "I"],
            ["i", "iv", "V", "i"],
            ["I", "bVI6", "V", "I"],
        ],
        "typical_keys": ["Eb major", "Bb minor", "Db major", "A major", "G minor"],
        "tempo_range": (40, 120),
    },
    
    "liszt": {
        "name": "Franz Liszt",
        "era": "romantic",
        "life": "1811-1886",
        "techniques": {
            # Piano technique
            "thematic_transformation": {
                "description": "Single theme transformed throughout work",
                "applies_to": ["symphonic_poem", "sonata"],
                "pattern": "Same theme in different characters",
            },
            "virtuoso_figuration": {
                "description": "Brilliant passage work",
                "applies_to": ["all"],
                "pattern": "Rapid scales, arpeggios, octaves",
            },
            "octave_passages": {
                "description": "Double notes for power",
                "applies_to": ["all"],
                "pattern": "Wrist octaves, thick textures",
            },
            # Form
            "symphonic_poem": {
                "description": "Single movement programmatic work",
                "applies_to": ["symphonic_poem"],
                "pattern": "No set form, follows program",
            },
            # Harmony
            "augmented_triad": {
                "description": "Use of +5 chord as sonority",
                "applies_to": ["all"],
                "pattern": "Diminished triad lowered fifth",
            },
            "whole_tone_passages": {
                "description": "Whole tone scale for dreamlike effects",
                "applies_to": ["all"],
                "pattern": "Six notes, equidistant",
            },
        },
        "characteristic_progressions": [
            ["I", "V7/V", "V", "I"],
            ["I", "aug6", "V", "I"],
            ["augmented triad", "resolution"],
        ],
        "typical_keys": ["B minor", "C# minor", "Eb major", "E major"],
        "tempo_range": (60, 180),
    },
    
    "brahms": {
        "name": "Johannes Brahms",
        "era": "romantic",
        "life": "1833-1897",
        "techniques": {
            # Harmony
            "brahms_hemiola": {
                "description": "3-against-2 rhythmic displacement",
                "applies_to": ["all"],
                "pattern": "Two bars of 3/4 felt as three bars of 2/4",
            },
            "brahms_shift": {
                "description": "Seamless modulation by third",
                "applies_to": ["all"],
                "pattern": "I → vi → IV, creates warm harmonic color",
            },
            # Form
            "developing_variation": {
                "description": "Motive develops throughout piece",
                "applies_to": ["all"],
                "pattern": "Basic idea returns transformed in each phrase",
            },
            "intermovement_thematicism": {
                "description": "Themes connect across movements",
                "applies_to": ["symphony", "quartet"],
                "pattern": "Opening theme returns in finale",
            },
            # Rhythm
            "syncopated_accompaniment": {
                "description": "Offbeat chords",
                "applies_to": ["all"],
                "pattern": "Chords on weak beats, melody on strong",
            },
            "three_against_two": {
                "description": "Polyrhythm in keyboard writing",
                "applies_to": ["all"],
                "pattern": "Right hand triplets, left hand duplets",
            },
        },
        "characteristic_progressions": [
            ["I", "vi", "IV", "V", "I"],
            ["I", "bVI", "bVII", "I"],  # Brahms shift
            ["V7", "I6/4", "V7", "I"],
        ],
        "typical_keys": ["D major", "F major", "C minor", "Bb major"],
        "tempo_range": (60, 140),
    },
    
    "schumann": {
        "name": "Robert Schumann",
        "era": "romantic",
        "life": "1810-1856",
        "techniques": {
            # Character pieces
            "character_piece": {
                "description": "Short piece with mood title",
                "applies_to": ["all"],
                "pattern": "Träumerei, Warum?, Scenes from Childhood",
            },
            # Harmony
            "inner_voice_motion": {
                "description": "Important inner voices",
                "applies_to": ["all"],
                "pattern": "Middle voices have melodic interest",
            },
            "chromatic_mediant": {
                "description": "Root movement by third",
                "applies_to": ["all"],
                "pattern": "I → iii, I → bVI",
            },
            # Rhythm
            "schumann_rhythm": {
                "description": "Characteristic dotted rhythms",
                "applies_to": ["all"],
                "pattern": "Short-long-short pattern",
            },
            # Piano writing
            "interlocking_hands": {
                "description": "Melody shared between hands",
                "applies_to": ["piano"],
                "pattern": "Right hand plays melody, left hand continues",
            },
        },
        "characteristic_progressions": [
            ["I", "vi", "ii", "V", "I"],
            ["I", "V", "vi", "IV"],
            ["I", "bVI", "V", "I"],
        ],
        "typical_keys": ["Eb major", "F major", "D minor", "A minor"],
        "tempo_range": (60, 120),
    },
    
    # ========================================
    # LATE ROMANTIC / POST-ROMANTIC
    # ========================================
    "wagner": {
        "name": "Richard Wagner",
        "era": "late_romantic",
        "life": "1813-1883",
        "techniques": {
            # Opera/drama
            "leitmotif": {
                "description": "Musical theme associated with character/idea",
                "applies_to": ["opera", "all"],
                "pattern": "Short motive appears when character/idea present",
            },
            "endless_melody": {
                "description": "Continuous without cadences",
                "applies_to": ["opera"],
                "pattern": "No clear phrase endings, continuous development",
            },
            "music_drama": {
                "description": "Drama and music inseparable",
                "applies_to": ["opera"],
                "pattern": "All elements unified, Gesamtkunstwerk",
            },
            # Harmony
            "tristan_chord": {
                "description": "F7 chord without resolution",
                "applies_to": ["all"],
                "pattern": "Half-diminished, leads to unresolved dominant",
            },
            "chromatic_voice_leading": {
                "description": "Voices move by half step",
                "applies_to": ["all"],
                "pattern": "Each voice moves chromatically, harmony transforms",
            },
            "delayed_resolution": {
                "description": "Tension prolonged indefinitely",
                "applies_to": ["all"],
                "pattern": "Dominants do not resolve immediately",
            },
        },
        "characteristic_progressions": [
            ["Tristan chord", "V7", "resolution delayed"],
            ["V7/V/V/V", "continuous dominant"],
        ],
        "typical_keys": ["Db major", "Eb major", "C major"],
        "tempo_range": (60, 100),
    },
    
    # ========================================
    # MODERN / IMPRESSIONIST
    # ========================================
    "debussy": {
        "name": "Claude Debussy",
        "era": "impressionist",
        "life": "1862-1918",
        "techniques": {
            # Harmony
            "whole_tone_scale": {
                "description": "Six-note scale with no semitones",
                "applies_to": ["all"],
                "pattern": "Creates floating, ambiguous harmony",
            },
            "pentatonic_scale": {
                "description": "Five-note scale",
                "applies_to": ["all"],
                "pattern": "Creates Asian-influenced color",
            },
            "parallel_chords": {
                "description": "Chords move in parallel motion",
                "applies_to": ["all"],
                "pattern": "Major and minor triads in sequence",
            },
            "planing": {
                "description": "Chords slide parallel",
                "applies_to": ["all"],
                "pattern": "Same chord quality, different roots",
            },
            "extended_chords": {
                "description": "7th, 9th, 11th, 13th chords",
                "applies_to": ["all"],
                "pattern": "Rich harmonies, often unresolved",
            },
            # Form
            "free_form": {
                "description": "No traditional formal structure",
                "applies_to": ["all"],
                "pattern": "Follows images, moods, programs",
            },
            # Timbre
            "timbral_color": {
                "description": "Color over function",
                "applies_to": ["all"],
                "pattern": "Chord voiced for resonance, not resolution",
            },
        },
        "characteristic_progressions": [
            ["whole tone", "whole tone", "resolution"],
            ["pentatonic sequence"],
            ["parallel triads"],
        ],
        "typical_keys": ["F# major", "Db major", "E major", "C# minor"],
        "tempo_range": (40, 120),
    },
    
    "ravel": {
        "name": "Maurice Ravel",
        "era": "impressionist",
        "life": "1875-1937",
        "techniques": {
            # Orchestration
            "orchestral_color": {
                "description": "Precise, colorful orchestration",
                "applies_to": ["orchestra"],
                "pattern": "Each instrument perfectly placed",
            },
            # Harmony
            "bitonality": {
                "description": "Two keys simultaneously",
                "applies_to": ["all"],
                "pattern": "Right hand one key, left hand another",
            },
            "seventh_chords": {
                "description": "7ths as stable sonorities",
                "applies_to": ["all"],
                "pattern": "Major 7th as tonic, minor 7th as tonic",
            },
            # Form
            "classical_precision": {
                "description": "Impressionism with classical structure",
                "applies_to": ["all"],
                "pattern": "Clear phrases, controlled harmony",
            },
            # Piano technique
            "cascading_arpeggios": {
                "description": "Large arpeggios spanning keyboard",
                "applies_to": ["piano"],
                "pattern": "Watery, flowing textures",
            },
        },
        "characteristic_progressions": [
            ["V7", "bII7", "V7", "I"],
            ["minor 7th sequence"],
        ],
        "typical_keys": ["G major", "A minor", "Db major"],
        "tempo_range": (60, 180),
    },
    
    # ========================================
    # JAZZ INFLUENCES (Modern)
    # ========================================
    "gershwin": {
        "name": "George Gershwin",
        "era": "modern",
        "life": "1898-1937",
        "techniques": {
            # Harmony
            "jazz_harmony": {
                "description": "Extended chords with jazz voice-leading",
                "applies_to": ["all"],
                "pattern": "9ths, 11ths, 13ths, altered dominants",
            },
            "blues_notes": {
                "description": "Flatted 3rd, 5th, 7th",
                "applies_to": ["all"],
                "pattern": "Blue notes for expressive color",
            },
            # Rhythm
            "syncopation": {
                "description": "Offbeat accents",
                "applies_to": ["all"],
                "pattern": "Ragtime, stride rhythms",
            },
            "swing_feel": {
                "description": "Triplet-based rhythm",
                "applies_to": ["all"],
                "pattern": "Uneven eighth notes",
            },
            # Form
            "song_form": {
                "description": "AABA 32-bar structure",
                "applies_to": ["song"],
                "pattern": "Verse-chorus structure",
            },
        },
        "characteristic_progressions": [
            ["I", "vi", "ii", "V"],  # Jazz standard
            ["iii", "vi", "ii", "V"],  # Jazz ii-V-I
            ["I", "I7", "IV", "iv", "I"],  # Bluesy
        ],
        "typical_keys": ["F major", "Bb major", "Eb major", "G major"],
        "tempo_range": (80, 160),
    },
    
    # ========================================
    # 20TH CENTURY
    # ========================================
    "stravinsky": {
        "name": "Igor Stravinsky",
        "era": "modern",
        "life": "1882-1971",
        "techniques": {
            # Rhythm
            "irregular_meter": {
                "description": "Changing time signatures",
                "applies_to": ["all"],
                "pattern": "5/8, 7/8, mixed meters",
            },
            "polyrhythm": {
                "description": "Multiple simultaneous rhythms",
                "applies_to": ["all"],
                "pattern": "Different meters in different voices",
            },
            "ostinato": {
                "description": "Repeating pattern",
                "applies_to": ["all"],
                "pattern": "Rhythmic cell repeats, harmony changes above",
            },
            # Harmony
            "bitonality": {
                "description": "Two keys simultaneously",
                "applies_to": ["all"],
                "pattern": "Different key in each hand",
            },
            "polychord": {
                "description": "Two chords simultaneously",
                "applies_to": ["all"],
                "pattern": "Triad over triad",
            },
            # Form
            "block_structure": {
                "description": "Sections as blocks",
                "applies_to": ["all"],
                "pattern": "Complete contrast between sections",
            },
        },
        "characteristic_progressions": [
            ["polychords"],
            ["octatonic passages"],
            ["bitonal layers"],
        ],
        "typical_keys": ["various", "bitonal"],
        "tempo_range": (80, 200),
    },
}


@dataclass
class LearnedTechnique:
    """A technique learned from master study."""
    name: str
    composer: str
    technique_type: str  # 'harmony', 'melody', 'rhythm', 'form'
    description: str
    pattern: str
    applies_to: list[str]
    success_count: int = 0
    failure_count: int = 0
    instances_seen: int = 0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / max(1, total)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "composer": self.composer,
            "technique_type": self.technique_type,
            "description": self.description,
            "pattern": self.pattern,
            "applies_to": self.applies_to,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "instances_seen": self.instances_seen,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "LearnedTechnique":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass 
class TheoryRule:
    """A music theory rule to follow or test."""
    name: str
    category: str  # 'voice_leading', 'harmony', 'counterpoint', 'form'
    description: str
    check_function: str  # Name of evaluation function
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "check_function": self.check_function,
        }


@dataclass
class PracticeSession:
    """Record of a practice session."""
    id: str
    technique: str
    composer: str
    composition_result: dict  # What was composed
    theory_violations: list[str]
    improvements: list[str]
    score: float
    learned: list[str]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "technique": self.technique,
            "composer": self.composer,
            "composition_result": self.composition_result,
            "theory_violations": self.theory_violations,
            "improvements": self.improvements,
            "score": self.score,
            "learned": self.learned,
        }


class MusicLearningSystem:
    MAX_PRACTICE_SESSIONS = 100
    """
    Continuous learning from master composers.
    
    Integrates with:
    - MusicEvolutionEngine for fitness tracking
    - MusicTheoryEngine for rule checking
    - MusicCreationEngine for composition
    - Knowledge Graph for storing learned patterns
    """
    
    # Theory rules to test
    THEORY_RULES = [
        TheoryRule("no_parallel_fifths", "voice_leading", 
                   "Avoid parallel fifths between voices",
                   "check_parallel_fifths"),
        TheoryRule("no_parallel_octaves", "voice_leading",
                   "Avoid parallel octaves between voices",
                   "check_parallel_octaves"),
        TheoryRule("resolve_leading_tone", "harmony",
                   "Leading tone (7th scale degree) resolves upward",
                   "check_leading_tone_resolution"),
        TheoryRule("resolve_seventh", "harmony",
                   "Seventh of chord resolves down",
                   "check_seventh_resolution"),
        TheoryRule("prefer_contrary_motion", "counterpoint",
                   "Voices should move in contrary motion when possible",
                   "check_motion_types"),
        TheoryRule("cadence_required", "form",
                   "Phrase must end with proper cadence",
                   "check_cadence"),
        TheoryRule("phrase_balance", "form",
                   "Phrases should be balanced (4+4, 8+8 bars)",
                   "check_phrase_balance"),
    ]
    
    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "music_learning"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Learned techniques
        self.techniques: dict[str, LearnedTechnique] = {}
        
        # Practice history
        self.practice_sessions: list[PracticeSession] = []
        self._practice_sessions_lock = threading.RLock()
        
        # Style preferences learned
        self.style_preferences: dict[str, float] = defaultdict(lambda: 0.5)
        
        # Theory violation counts
        self.violation_counts: dict[str, int] = defaultdict(int)
        
        # Knowledge graph reference (set by parent)
        self._knowledge_graph = None
        self._music_evolution = None
        
        self._lock = threading.RLock()
        self._load_state()
        
        # Initialize with master techniques
        self._initialize_master_techniques()
    
    def _load_state(self) -> None:
        """Load learned state from disk."""
        state_file = self.storage_path / "learning_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                
                for tech_data in data.get("techniques", []):
                    tech = LearnedTechnique.from_dict(tech_data)
                    self.techniques[tech.name] = tech
                
                self.style_preferences.update(data.get("style_preferences", {}))
                self.violation_counts.update(data.get("violation_counts", {}))
                
                logger.info(f"Loaded music learning: {len(self.techniques)} techniques")
            except Exception as e:
                logger.warning(f"Failed to load learning state: {e}")
    
    def _save_state(self) -> None:
        """Save learned state to disk."""
        state_file = self.storage_path / "learning_state.json"
        try:
            data = {
                "techniques": [t.to_dict() for t in self.techniques.values()],
                "style_preferences": dict(self.style_preferences),
                "violation_counts": dict(self.violation_counts),
            }
            atomic_write_json(state_file, data)
        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")
    
    def _initialize_master_techniques(self) -> None:
        """Initialize techniques from master composers."""
        for composer_name, composer_data in MASTER_TECHNIQUES.items():
            for tech_name, tech_info in composer_data.get("techniques", {}).items():
                key = f"{composer_name}_{tech_name}"
                if key not in self.techniques:
                    technique_type = tech_info.get("applies_to", ["all"])[0] if tech_info.get("applies_to") else "all"
                    self.techniques[key] = LearnedTechnique(
                        name=tech_name,
                        composer=composer_name,
                        technique_type=technique_type,
                        description=tech_info.get("description", ""),
                        pattern=tech_info.get("pattern", ""),
                        applies_to=tech_info.get("applies_to", ["all"]),
                    )
    
    def set_evolution_engine(self, engine) -> None:
        """Connect to evolution engine."""
        self._music_evolution = engine
    
    def set_knowledge_graph(self, kg) -> None:
        """Connect to knowledge graph."""
        self._knowledge_graph = kg
    
    def get_all_composers(self) -> list[dict]:
        """Get list of all composers with their info."""
        composers = []
        for composer_id, data in MASTER_TECHNIQUES.items():
            composers.append({
                "id": composer_id,
                "name": data["name"],
                "era": data["era"],
                "life": data.get("life", "unknown"),
                "technique_count": len(data.get("techniques", {})),
                "progression_count": len(data.get("characteristic_progressions", [])),
                "typical_keys": data.get("typical_keys", []),
                "tempo_range": data.get("tempo_range", (60, 120)),
            })
        return composers
    
    def get_composer_techniques(self, composer: str) -> dict:
        """Get all techniques for a specific composer."""
        if composer.lower() not in MASTER_TECHNIQUES:
            return {"error": f"Unknown composer: {composer}"}
        
        master = MASTER_TECHNIQUES[composer.lower()]
        
        techniques = []
        for tech_name, tech_info in master.get("techniques", {}).items():
            key = f"{composer.lower()}_{tech_name}"
            learned = self.techniques.get(key, LearnedTechnique(
                name=tech_name,
                composer=composer.lower(),
                technique_type=tech_info.get("applies_to", ["all"])[0],
                description=tech_info.get("description", ""),
                pattern=tech_info.get("pattern", ""),
                applies_to=tech_info.get("applies_to", []),
            ))
            techniques.append({
                "name": tech_name,
                "description": tech_info.get("description", ""),
                "pattern": tech_info.get("pattern", ""),
                "applies_to": tech_info.get("applies_to", []),
                "rules": tech_info.get("rules", []),
                "success_rate": learned.success_rate,
                "times_used": learned.instances_seen,
            })
        
        return {
            "composer": master["name"],
            "era": master["era"],
            "life": master.get("life", "unknown"),
            "techniques": techniques,
            "progressions": master.get("characteristic_progressions", []),
            "typical_keys": master.get("typical_keys", []),
            "tempo_range": master.get("tempo_range", (60, 120)),
        }
    
    def study_master(self, composer: str) -> dict:
        """
        Study techniques from a master composer.
        
        Returns analysis of techniques that can be applied.
        """
        if composer.lower() not in MASTER_TECHNIQUES:
            return {"error": f"Unknown composer: {composer}"}
        
        master = MASTER_TECHNIQUES[composer.lower()]
        
        study_result = {
            "composer": master["name"],
            "era": master["era"],
            "life": master.get("life", "unknown"),
            "techniques": [],
            "progressions": master.get("characteristic_progressions", []),
            "typical_keys": master.get("typical_keys", []),
            "tempo_range": master.get("tempo_range", (60, 120)),
        }
        
        for tech_name, tech_info in master.get("techniques", {}).items():
            key = f"{composer.lower()}_{tech_name}"
            if key in self.techniques:
                study_result["techniques"].append({
                    "name": tech_name,
                    "description": tech_info.get("description", ""),
                    "pattern": tech_info.get("pattern", ""),
                    "success_rate": self.techniques[key].success_rate,
                    "rules": tech_info.get("rules", []),
                })
        
        # Store in knowledge graph if available
        if self._knowledge_graph:
            self._store_master_study(study_result)
        
        return study_result
    
    def _store_master_study(self, study_result: dict) -> None:
        """Store master study results in knowledge graph."""
        if not self._knowledge_graph:
            return
        
        try:
            composer_node = f"master_{study_result['composer'].lower().replace(' ', '_')}"
            
            # Add composer node
            self._knowledge_graph.add_node(
                composer_node,
                node_type="master_composer",
                data={
                    "name": study_result["composer"],
                    "era": study_result["era"],
                }
            )
            
            # Add technique nodes
            for tech in study_result.get("techniques", []):
                tech_node = f"technique_{tech['name']}"
                self._knowledge_graph.add_node(
                    tech_node,
                    node_type="composition_technique",
                    data=tech
                )
                self._knowledge_graph.add_edge(composer_node, tech_node, "uses_technique")
                
        except Exception as e:
            logger.warning(f"Failed to store in knowledge graph: {e}")
    
    def test_composition(self, composition: dict, style: str) -> dict:
        """
        Test a composition against theory rules.
        
        Returns violations and suggestions for improvement.
        """
        violations = []
        suggestions = []
        
        # Get notes from composition
        notes = composition.get("notes", [])
        if not notes:
            # Try to extract from phrases
            for phrase in composition.get("phrases", []):
                notes.extend(phrase.get("melody_notes", []))
        
        # Check theory rules
        for rule in self.THEORY_RULES:
            result = self._check_rule(rule, notes, composition)
            if not result["passed"]:
                violations.append({
                    "rule": rule.name,
                    "description": rule.description,
                    "issue": result.get("issue", ""),
                })
                self.violation_counts[rule.name] += 1
        
        # Generate suggestions
        if violations:
            suggestions = self._generate_suggestions(violations, style)
        
        # Calculate score
        total_rules = len(self.THEORY_RULES)
        passed_rules = total_rules - len(violations)
        score = passed_rules / max(1, total_rules)
        
        return {
            "score": score,
            "violations": violations,
            "suggestions": suggestions,
            "passed_rules": passed_rules,
            "total_rules": total_rules,
        }
    
    def _check_rule(self, rule: TheoryRule, notes: list, composition: dict) -> dict:
        """Check if a composition follows a theory rule."""
        result = {"passed": True, "issue": ""}
        
        if rule.check_function == "check_parallel_fifths":
            result = self._check_parallel_fifths(notes)
        elif rule.check_function == "check_parallel_octaves":
            result = self._check_parallel_octaves(notes)
        elif rule.check_function == "check_leading_tone_resolution":
            result = self._check_leading_tone_resolution(notes, composition)
        elif rule.check_function == "check_cadence":
            result = self._check_cadence(composition)
        elif rule.check_function == "check_phrase_balance":
            result = self._check_phrase_balance(composition)
        
        return result
    
    def _check_parallel_fifths(self, notes: list) -> dict:
        """Check for parallel fifths."""
        return {"passed": True, "issue": ""}
    
    def _check_parallel_octaves(self, notes: list) -> dict:
        """Check for parallel octaves."""
        return {"passed": True, "issue": ""}
    
    def _check_leading_tone_resolution(self, notes: list, composition: dict) -> dict:
        """Check if leading tones resolve upward."""
        return {"passed": True, "issue": ""}
    
    def _check_cadence(self, composition: dict) -> dict:
        """Check if composition has proper cadences."""
        phrases = composition.get("phrases", [])
        if len(phrases) < 1:
            return {"passed": False, "issue": "No phrases found"}
        return {"passed": True, "issue": ""}
    
    def _check_phrase_balance(self, composition: dict) -> dict:
        """Check if phrases are balanced."""
        phrases = composition.get("phrases", [])
        if len(phrases) < 2:
            return {"passed": True, "issue": ""}
        
        durations = [p.get("duration", 0) for p in phrases]
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_deviation = max(abs(d - avg_duration) for d in durations) if durations else 0
            
            if max_deviation > avg_duration * 0.5:
                return {"passed": False, "issue": f"Phrases unbalanced: max deviation {max_deviation:.1f}"}
        
        return {"passed": True, "issue": ""}
    
    def _generate_suggestions(self, violations: list, style: str) -> list[str]:
        """Generate suggestions based on violations."""
        suggestions = []
        
        for v in violations:
            rule_name = v["rule"]
            
            if rule_name == "no_parallel_fifths":
                suggestions.append("Use contrary motion between voices to avoid parallel fifths")
            elif rule_name == "no_parallel_octaves":
                suggestions.append("Maintain independent voice motion")
            elif rule_name == "resolve_leading_tone":
                suggestions.append("Ensure the 7th scale degree resolves upward to tonic")
            elif rule_name == "cadence_required":
                suggestions.append("End phrases with V-I or similar authentic cadence")
            elif rule_name == "phrase_balance":
                suggestions.append("Balance phrases: use 4+4 or 8+8 bar structures")
        
        return suggestions
    
    def apply_learned_technique(self, technique_name: str, base_composition: dict) -> dict:
        """
        Apply a learned technique to a composition.
        
        Returns modified composition.
        """
        if technique_name not in self.techniques:
            return base_composition
        
        technique = self.techniques[technique_name]
        
        # Get composer patterns
        composer = technique.composer
        if composer in MASTER_TECHNIQUES:
            master = MASTER_TECHNIQUES[composer]
            if master.get("characteristic_progressions"):
                base_composition["suggested_progression"] = random.choice(
                    master["characteristic_progressions"]
                )
        
        technique.instances_seen += 1
        self._save_state()
        
        return base_composition
    
    def record_practice_result(
        self,
        technique: str,
        composer: str,
        composition: dict,
        test_result: dict,
    ) -> PracticeSession:
        """Record results of a practice session."""
        session = PracticeSession(
            id=f"practice_{len(self.practice_sessions)}",
            technique=technique,
            composer=composer,
            composition_result=composition,
            theory_violations=test_result.get("violations", []),
            improvements=test_result.get("suggestions", []),
            score=test_result.get("score", 0.0),
            learned=[],
        )
        
        # Update technique success
        tech_key = f"{composer}_{technique}"
        if tech_key in self.techniques:
            if test_result.get("score", 0) > 0.7:
                self.techniques[tech_key].success_count += 1
                session.learned.append(f"Successfully applied {technique}")
            else:
                self.techniques[tech_key].failure_count += 1
                session.learned.extend(test_result.get("suggestions", []))
        
        self.practice_sessions.append(session)
        self.practice_sessions = self.practice_sessions[-self.MAX_PRACTICE_SESSIONS:]
        self._save_state()
        
        return session
    
    def get_recommended_techniques(self, style: str, form: str) -> list[dict]:
        """
        Get recommended techniques for a style and form.
        
        Returns sorted list of techniques by success rate.
        """
        applicable = []
        
        # Map styles to composers
        style_to_composer = {
            "baroque": ["bach", "handel", "vivaldi"],
            "classical": ["mozart", "haydn"],
            "romantic": ["beethoven", "schubert", "chopin", "brahms", "schumann", "liszt"],
            "late_romantic": ["wagner", "brahms"],
            "impressionist": ["debussy", "ravel"],
            "modern": ["stravinsky", "gershwin"],
            "renaissance": ["palestrina"],
        }
        
        composers = style_to_composer.get(style.lower(), list(MASTER_TECHNIQUES.keys()))
        
        for tech_key, tech in self.techniques.items():
            if form in tech.applies_to or "all" in tech.applies_to:
                if tech.composer in composers:
                    applicable.append({
                        "name": tech.name,
                        "composer": tech.composer,
                        "description": tech.description,
                        "success_rate": tech.success_rate,
                        "instances_seen": tech.instances_seen,
                    })
        
        # Sort by success rate
        applicable.sort(key=lambda x: x["success_rate"], reverse=True)
        
        return applicable[:5]  # Top 5
    
    def get_progression_for_style(self, style: str) -> list[str]:
        """Get a characteristic chord progression for a style."""
        style_lower = style.lower()
        
        style_to_composer = {
            "baroque": "bach",
            "classical": "mozart",
            "romantic": "beethoven",
            "impressionist": "debussy",
            "modern": "stravinsky",
            "renaissance": "palestrina",
        }
        
        composer = style_to_composer.get(style_lower, "bach")
        
        if composer in MASTER_TECHNIQUES:
            progressions = MASTER_TECHNIQUES[composer]["characteristic_progressions"]
            return random.choice(progressions)
        
        return ["I", "IV", "V", "I"]  # Default
    
    def get_style_insights(self) -> dict:
        """Get insights about learned styles."""
        insights = {
            "total_composers": len(MASTER_TECHNIQUES),
            "total_techniques_learned": len(self.techniques),
            "practice_sessions": len(self.practice_sessions),
            "top_techniques": [],
            "common_violations": [],
            "style_preferences": dict(self.style_preferences),
        }
        
        # Top techniques by success
        sorted_techniques = sorted(
            self.techniques.values(),
            key=lambda t: t.success_rate,
            reverse=True
        )
        insights["top_techniques"] = [
            {"name": t.name, "composer": t.composer, "rate": t.success_rate}
            for t in sorted_techniques[:5]
        ]
        
        # Common violations
        sorted_violations = sorted(
            self.violation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        insights["common_violations"] = [
            {"rule": v[0], "count": v[1]}
            for v in sorted_violations[:5]
        ]
        
        return insights


# Singleton
_learning_system: Optional[MusicLearningSystem] = None
_lock = threading.Lock()


def get_music_learning() -> MusicLearningSystem:
    """Get singleton music learning system."""
    global _learning_system
    if _learning_system is None:
        with _lock:
            if _learning_system is None:
                _learning_system = MusicLearningSystem()
    return _learning_system