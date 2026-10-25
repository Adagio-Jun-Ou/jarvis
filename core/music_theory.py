"""
Native Music Theory System for Jarvis.

Provides deep music theory knowledge for composition:
- Harmony and voice leading
- Counterpoint rules
- Form and structure
- Thematic development
- Orchestration
"""

from enum import Enum
from dataclasses import dataclass
import random


class Key(Enum):
    """Musical keys with their characteristics."""
    C_MAJOR = ("C major", "bright, triumphant, noble")
    G_MAJOR = ("G major", "pastoral, serene, rustic")
    D_MAJOR = ("D major", "triumphant, victorious, jubilant")
    A_MAJOR = ("A major", "bright, joyful, pastoral")
    E_MAJOR = ("E major", "brilliant, majestic, ecstatic")
    B_MAJOR = ("B major", "radiant, noble, exalted")
    F_MAJOR = ("F major", "pastoral, calm, peaceful")
    BB_MAJOR = ("Bb major", "noble, majestic, dignified")
    EB_MAJOR = ("Eb major", "heroic, majestic, solemn")
    AB_MAJOR = ("Ab major", "warm, tender, expressive")
    
    A_MINOR = ("A minor", "melancholic, tender, longing")
    E_MINOR = ("E minor", "somber, dramatic, tragic")
    B_MINOR = ("B minor", "solitary, mournful, pathetic")
    FS_MINOR = ("F# minor", "passionate, tragic, intense")
    CS_MINOR = ("C# minor", "tragic, passionate, despairing")
    D_MINOR = ("D minor", "serious, dramatic, solemn")
    G_MINOR = ("G minor", "melancholic, tragic, dramatic")
    C_MINOR = ("C minor", "tragic, dramatic, passionate")
    F_MINOR = ("F minor", "deeply tragic, dark, intense")


@dataclass
class ChordProgression:
    """A chord progression with functional harmony."""
    name: str
    chords: list
    era: str
    emotional_character: str
    voice_leading_rules: list
    examples: list


@dataclass
class FormStructure:
    """Musical form with section details."""
    name: str
    sections: list
    proportions: list
    key_relations: list
    characteristics: list


@dataclass
class ThematicDevelopment:
    """How themes are developed in music."""
    technique: str
    description: str
    examples: list
    emotional_effect: str


# Progressions by era
PROGRESSIONS = {
    "baroque": [
        ChordProgression(
            name="Circle of Fifths",
            chords=["I", "IV", "vii", "iii", "vi", "ii", "V", "I"],
            era="baroque",
            emotional_character="dignified, sequential, flowing",
            voice_leading_rules=[
                "Resolve leading tones upward",
                "Avoid parallel fifths and octaves",
                "Keep common tones between chords",
                "Bass moves by 4th/5th or stepwise"
            ],
            examples=["Pachelbel Canon", "Bach Well-Tempered Clavier"]
        ),
    ],
    "classical": [
        ChordProgression(
            name="Sonata Exposition",
            chords=["I", "I", "V/V", "V", "V", "V", "V", "I"],
            era="classical",
            emotional_character="balanced, dramatic",
            voice_leading_rules=[
                "First theme in tonic",
                "Bridge modulates to dominant",
                "Second theme in dominant",
                "Codetta confirms key"
            ],
            examples=["Mozart Symphonies", "Beethoven Sonatas"]
        ),
        ChordProgression(
            name="I-vi-ii-V-I",
            chords=["I", "vi", "ii", "V", "I"],
            era="classical",
            emotional_character="balanced, satisfying",
            voice_leading_rules=[
                "Each chord leads naturally to next",
                "Predominant leads to dominant",
                "Dominant resolves to tonic"
            ],
            examples=["Mozart Eine Kleine Nachtmusik", "Haydn Quartets"]
        ),
    ],
    "romantic": [
        ChordProgression(
            name="Chopin Nocturne",
            chords=["i", "VI", "iv", "V7", "i", "vii/V", "V", "i"],
            era="romantic",
            emotional_character="nostalgic, yearning",
            voice_leading_rules=[
                "Rich chromatic inner voices",
                "Delayed resolution",
                "Rubato interpretation",
                "Arpeggiated accompaniment"
            ],
            examples=["Chopin Nocturnes", "Field Nocturnes"]
        ),
    ],
}


# Forms
MUSICAL_FORMS = {
    "binary": FormStructure(
        name="Binary Form",
        sections=["A", "B"],
        proportions=[0.5, 0.5],
        key_relations=["tonic", "dominant/relative"],
        characteristics=[
            "Two sections, each repeated",
            "Section A establishes tonic",
            "Section B moves away and returns",
            "Common in Baroque dance movements"
        ]
    ),
    "ternary": FormStructure(
        name="Ternary Form",
        sections=["A", "B", "A"],
        proportions=[0.35, 0.30, 0.35],
        key_relations=["tonic", "contrasting", "tonic"],
        characteristics=[
            "Three-part structure",
            "A section establishes main theme",
            "B section provides contrast",
            "Return to A creates balance"
        ]
    ),
    "sonata": FormStructure(
        name="Sonata Form",
        sections=["Exposition", "Development", "Recapitulation"],
        proportions=[0.30, 0.35, 0.35],
        key_relations=["tonic/dominant", "modulatory", "tonic"],
        characteristics=[
            "Most sophisticated instrumental form",
            "Two contrasting themes",
            "Development explores themes",
            "Recapitulation resolves tension"
        ]
    ),
    "rondo": FormStructure(
        name="Rondo Form",
        sections=["A", "B", "A", "C", "A", "B", "A"],
        proportions=[0.15, 0.12, 0.15, 0.16, 0.15, 0.12, 0.15],
        key_relations=["tonic", "contrasting", "tonic"],
        characteristics=[
            "Refrain returns between episodes",
            "Episodes provide contrast",
            "Common for finales"
        ]
    ),
    "fugue": FormStructure(
        name="Fugue",
        sections=["Exposition", "Episode", "Development", "Climax", "Coda"],
        proportions=[0.20, 0.15, 0.30, 0.20, 0.15],
        key_relations=["tonic/dominant", "various", "tonic"],
        characteristics=[
            "Polyphonic texture",
            "Subject treated imitatively",
            "Countersubject accompanies subject",
            "Stretto for climax"
        ]
    ),
    "theme_variations": FormStructure(
        name="Theme and Variations",
        sections=["Theme", "Var1", "Var2", "Var3", "Var4", "Var5", "Coda"],
        proportions=[0.15, 0.12, 0.13, 0.13, 0.13, 0.13, 0.21],
        key_relations=["tonic throughout"],
        characteristics=[
            "Theme stated simply at beginning",
            "Each variation transforms theme",
            "Progressively more elaborate",
            "Final variation often grandest"
        ]
    ),
}


# Thematic techniques
THEMATIC_TECHNIQUES = [
    ThematicDevelopment(
        technique="Repetition",
        description="Exact restatement of theme",
        examples=["Beethoven Symphony 5", "Mozart Symphony 40"],
        emotional_effect="Reinforces idea, creates familiarity"
    ),
    ThematicDevelopment(
        technique="Sequence",
        description="Theme repeated at different pitch levels",
        examples=["Bach Inventions", "Beethoven development sections"],
        emotional_effect="Builds tension, creates direction"
    ),
    ThematicDevelopment(
        technique="Inversion",
        description="Theme with intervals reversed",
        examples=["Bach Art of Fugue", "Beethoven Eroica"],
        emotional_effect="Intellectual, creates variety"
    ),
    ThematicDevelopment(
        technique="Augmentation",
        description="Theme in longer note values",
        examples=["Bach fugues", "Wagner operas"],
        emotional_effect="Grand, majestic, solemn"
    ),
    ThematicDevelopment(
        technique="Diminution",
        description="Theme in shorter note values",
        examples=["Bach fugues", "Beethoven scherzos"],
        emotional_effect="Agitated, excited, virtuosic"
    ),
    ThematicDevelopment(
        technique="Fragmentation",
        description="Theme broken into smaller pieces",
        examples=["Beethoven development sections", "Mahler symphonies"],
        emotional_effect="Building tension, development"
    ),
]


class MusicTheoryEngine:
    """Provides music theory knowledge for composition prompts."""
    
    # Composer vocabulary by era
    COMPOSER_VOCAB = {
        "bach": ["contrapuntal", "fugal exposition", "species counterpoint",
                 "sequential development", "ornamented melodies", "terraced dynamics"],
        "mozart": ["elegant melodies", "balanced phrases", "Alberti bass",
                   "clear cadences", "operatic drama", "symphonic development"],
        "beethoven": ["motivic development", "structural innovation", "dynamic contrasts",
                      "expanded form", "emotional range", "theme transformation"],
        "schumann": ["lyrical melodies", "literary inspiration", "chromatic harmony",
                     "interlocking rhythms", "emotional depth", "character pieces"],
        "brahms": ["classical forms", "romantic harmony", "developing variation",
                   "syncopated rhythm", "thick textures", "hemiola"],
        "chopin": ["cantabile melody", "rubato", "bravura passages",
                   "national influence", "chromatic embellishment", "nocturne style"],
        "debussy": ["impressionist harmony", "whole tone scales", "parallel chords",
                    "tone color", "fluid rhythm", "pentatonic influence"],
        "dvorak": ["folk elements", "slavonic dances", "lyrical themes",
                   "modal mixture", "dumka form", "national style"],
        "tchaikovsky": ["melodic genius", "emotional intensity", "ballet influence",
                        "orchestral color", "russian soul", "dramatic climaxes"],
    }
    
    def get_composer_prompt(self, composer, form, key, character, duration):
        """Generate a detailed prompt using music theory."""
        
        composer_lower = composer.lower()
        era = self._get_era(composer_lower)
        form_info = MUSICAL_FORMS.get(form.lower().replace("-", "_"))
        progressions = PROGRESSIONS.get(era, PROGRESSIONS["classical"])
        progression = progressions[0]  # Use first progression for era
        
        # Get key description
        key_desc = "expressive"
        for k in Key:
            if key.upper() in k.value[0].upper():
                key_desc = k.value[1]
                break
        
        # Get composer vocabulary
        vocab = self.COMPOSER_VOCAB.get(composer_lower, ["expressive melody", "clear harmony"])
        
        # Build prompt parts
        parts = []
        
        # Form
        if form_info:
            parts.append(form_info.name + " form")
            parts.append("Sections: " + ", ".join(form_info.sections))
        
        # Harmony
        parts.append("Harmony: " + progression.name + " progression")
        parts.append("Voice leading: " + progression.voice_leading_rules[0])
        
        # Key and character  
        key_desc = key_desc if key_desc else "expressive"
        parts.append("Key: " + key + " (" + key_desc + "), " + character + " mood")
        
        # Thematic development
        technique = random.choice(THEMATIC_TECHNIQUES)
        parts.append("Use " + technique.technique + ": " + technique.description)
        
        # Instrumentation
        if era == "baroque":
            parts.append("Instrumentation: harpsichord or strings, contrapuntal texture")
        elif era == "classical":
            parts.append("Instrumentation: piano or string quartet, clear textures")
        elif era == "romantic":
            parts.append("Instrumentation: piano, lush harmonies, rubato")
        
        # Style
        parts.append("Style elements: " + ", ".join(vocab[:3]))
        
        # Duration guidance
        if duration < 60:
            parts.append("Structure: compact form with clear phrases")
        elif duration < 120:
            parts.append("Structure: developed form with contrasting sections")
        else:
            parts.append("Structure: extended form with full development")
        
        return ", ".join(parts)
    
    def _get_era(self, composer):
        """Map composer to era."""
        if composer in ["bach", "handel", "vivaldi", "scarlatti"]:
            return "baroque"
        elif composer in ["mozart", "haydn", "clementi", "beethoven"]:
            return "classical"
        elif composer in ["schumann", "brahms", "chopin", "liszt", "dvorak", "tchaikovsky"]:
            return "romantic"
        elif composer in ["debussy", "ravel"]:
            return "romantic"
        return "classical"
    
    def get_form_sections(self, form, duration):
        """Get form sections with durations."""
        form_info = MUSICAL_FORMS.get(form.lower().replace("-", "_"), MUSICAL_FORMS["ternary"])
        sections = []
        for section, proportion in zip(form_info.sections, form_info.proportions):
            sections.append((section, duration * proportion))
        return sections
    
    def get_voice_leading_rules(self, era):
        """Get voice leading rules for era."""
        if era == "baroque":
            return ["Avoid parallel fifths and octaves", "Resolve leading tones upward"]
        elif era == "classical":
            return ["Clear voice leading", "Alberti bass accompaniment"]
        elif era == "romantic":
            return ["Rich chromatic voice leading", "Expressive inner voices"]
        return ["Smooth voice leading"]


class Counterpoint:
    """
    Fux Species Counterpoint Rules (1725)
    
    Foundation of Western voice leading:
    1. First Species: Note-against-note (most consonant)
    2. Second Species: Two notes against one
    3. Third Species: Four notes against one
    4. Fourth Species: Suspensions
    5. Fifth Species: Florid counterpoint
    
    Core rules enforced:
    - No parallel fifths or octaves
    - Prefer contrary motion
    - Resolve leading tones upward
    - Consonances on strong beats
    """
    
    PERFECT_INTERVALS = [0, 7, 12]  # Unison, fifth, octave
    CONSONANT_INTERVALS = [0, 3, 4, 5, 7, 8, 9, 12]  # Thirds, fifths, sixths, octaves
    DISSONANT_INTERVALS = [1, 2, 6, 10, 11]  # Seconds, tritone, sevenths
    
    @staticmethod
    def check_parallel_fifths_or_octaves(voice1: list, voice2: list) -> list:
        """Detect forbidden parallel motion."""
        errors = []
        for i in range(len(voice1) - 1):
            int1 = abs(voice1[i] - voice2[i]) % 12
            int2 = abs(voice1[i+1] - voice2[i+1]) % 12
            
            if int1 in [0, 7] and int1 == int2:
                if voice1[i] != voice1[i+1]:
                    errors.append({
                        'type': f'parallel_{"octaves" if int1 == 0 else "fifths"}',
                        'positions': [i, i+1],
                        'intervals': [int1, int2],
                    })
        return errors
    
    @staticmethod
    def resolve_leading_tone(notes: list, scale: list) -> list:
        """Ensure leading tone (degree 6 in minor, 7 in major) resolves to tonic."""
        leading_tone_idx = 6 if len(scale) == 7 else 6
        resolved = []
        
        for i, note in enumerate(notes):
            deg = note.get('scale_degree', 0) if isinstance(note, dict) else 0
            
            if deg == leading_tone_idx and i < len(notes) - 1:
                next_deg = notes[i+1].get('scale_degree', 0) if isinstance(notes[i+1], dict) else 0
                if next_deg != 0:
                    resolved.append({**note, 'scale_degree': 0} if isinstance(note, dict) else note)
                    continue
            
            resolved.append(note)
        
        return resolved
    
    @staticmethod
    def get_recommended_intervals(degree1: int, degree2: int, scale: list) -> list:
        """Get recommended consonant intervals between voices."""
        pitch1 = scale[degree1]
        pitch2 = scale[degree2]
        interval = abs(pitch1 - pitch2) % 12
        
        # Thirds and sixths are most melodic
        if interval in [3, 4, 8, 9]:
            return [(degree2, 'consonant', 'melodic')]
        # Fifths and octaves are strong
        elif interval in [0, 7]:
            return [(degree2, 'consonant', 'strong')]
        # Seconds and sevenths are dissonant
        else:
            return [(degree2, 'dissonant', 'prepare_and_resolve')]


class NonChordTones:
    """
    Non-Chord Tones (Embellishments)
    
    PASSING TONE: Step between chord tones (fills gap)
    NEIGHBOR TONE: Step away and back (decorates)
    SUSPENSION: Prepared dissonance resolved down by step
    APPOGGIATURA: Unprepared dissonance resolved down by step
    ESCAPE TONE: Step away, skip opposite direction
    ANTICIPATION: Early arrival of next chord tone
    PEDAL TONE: Sustained note with changing harmonies
    """
    
    TYPES = {
        'passing': {
            'approach': 'step', 'departure': 'step_same_dir', 'strong_beat': False,
            'preparation': None, 'resolution': 'chord_tone',
        },
        'neighbor': {
            'approach': 'step', 'departure': 'step_opposite_dir', 'strong_beat': False,
            'preparation': None, 'resolution': 'same_tone',
        },
        'suspension': {
            'approach': 'same_note', 'departure': 'step_down', 'strong_beat': True,
            'preparation': 'chord_tone', 'resolution': 'chord_tone_down',
        },
        'appoggiatura': {
            'approach': 'skip', 'departure': 'step_down', 'strong_beat': True,
            'preparation': None, 'resolution': 'chord_tone_down',
        },
        'escape': {
            'approach': 'step', 'departure': 'skip_opposite', 'strong_beat': False,
            'preparation': None, 'resolution': 'skip_to_chord_tone',
        },
        'anticipation': {
            'approach': 'any', 'departure': 'same_note', 'strong_beat': False,
            'preparation': None, 'resolution': 'next_chord_tone',
        },
        'pedal': {
            'approach': 'same_note', 'departure': 'same_note', 'strong_beat': True,
            'preparation': 'chord_tone', 'resolution': 'sustained',
        },
    }
    
    @staticmethod
    def add_passing_tone(notes: list, pos: int, direction: int) -> dict:
        """Add passing tone between notes[pos] and notes[pos+1]."""
        if pos >= len(notes) - 1:
            return None
        
        curr_pitch = notes[pos]['pitch']
        next_pitch = notes[pos+1]['pitch']
        
        # Fill gap with stepwise motion
        if abs(next_pitch - curr_pitch) > 2:
            return None  # Too large for passing tone
        
        passing_pitch = curr_pitch + (1 if next_pitch > curr_pitch else -1)
        
        return {
            'pitch': passing_pitch,
            'duration': 0.5,
            'velocity': max(50, notes[pos]['velocity'] - 5),
            'beat': notes[pos]['beat'] + notes[pos]['duration'],
            'is_nct': True,
            'nct_type': 'passing',
        }
    
    @staticmethod
    def add_neighbor_tone(note: dict, scale: list, direction: str = 'upper') -> dict:
        """Add neighbor tone decoration."""
        pitch = note['pitch']
        step = 1 if direction == 'upper' else -1
        neighbor_pitch = pitch + step
        
        return {
            'pitch': neighbor_pitch,
            'duration': note['duration'] / 2,
            'velocity': max(50, note['velocity'] - 3),
            'beat': note['beat'],
            'is_nct': True,
            'nct_type': 'neighbor',
        }
    
    @staticmethod
    def create_suspension(harmony_pos: int, prev_chord_degree: int, 
                          curr_chord_degree: int, scale: list) -> dict:
        """Create a suspension from previous chord tone."""
        # Suspended note is from previous chord, resolves to current chord
        suspended_pitch = scale[prev_chord_degree]
        resolved_pitch = scale[curr_chord_degree]
        
        return {
            'suspended': suspended_pitch,
            'resolved': resolved_pitch,
            'preparation_beat': harmony_pos - 0.5,
            'suspension_beat': harmony_pos,
            'resolution_beat': harmony_pos + 0.5,
        }


class Cadences:
    """
    Cadence Types - Musical Punctuation
    
    PERFECT AUTHENTIC (V-I): Strongest resolution, final cadence
    IMPERFECT AUTHENTIC (V-I with 3rd or 5th in melody): Slightly weaker
    PLAGAL (IV-I): "Amen" cadence, softer resolution
    HALF (ii-V or IV-V): Leaves phrase open, expects continuation
    DECEPTIVE (V-vi): Surprise, avoids resolution, extends phrase
    PHRYGIAN (iv6-V in minor): Dramatic, Baroque style
    """
    
    TYPES = {
        'perfect_authentic': {
            'chords': ['V7', 'I'],
            'bass_degrees': [4, 0],
            'melody_ends_on': [0, 2],  # Tonic or third
            'emotional': 'final resolution, conclusive',
            'placement': 'final cadence only',
            'strength': 1.0,
        },
        'imperfect_authentic': {
            'chords': ['V', 'I'],
            'bass_degrees': [4, 0],
            'melody_ends_on': [4],  # Fifth in melody
            'emotional': 'resolving but less final',
            'placement': 'mid-phrase or final',
            'strength': 0.85,
        },
        'plagal': {
            'chords': ['IV', 'I'],
            'bass_degrees': [3, 0],
            'melody_ends_on': [0, 2, 4],
            'emotional': 'soft, religious (Amen)',
            'placement': 'final cadence after V-I',
            'strength': 0.7,
        },
        'half': {
            'chords': ['ii', 'V'],
            'bass_degrees': [1, 4],
            'melody_ends_on': [2, 4],  # Third or fifth of V
            'emotional': 'open, questioning, expects continuation',
            'placement': 'antecedent phrase ending',
            'strength': 0.5,
        },
        'deceptive': {
            'chords': ['V', 'vi'],
            'bass_degrees': [4, 5],
            'melody_ends_on': [0, 2, 4],
            'emotional': 'surprise, extends phrase',
            'placement': 'unexpected resolution',
            'strength': 0.4,
        },
        'phrygian': {
            'chords': ['iv6', 'V'],
            'bass_degrees': [3, 4],  # iv6 has 6th in bass
            'melody_ends_on': [0, 2, 4],
            'emotional': 'dramatic, Baroque minor key',
            'placement': 'minor key cadences',
            'strength': 0.75,
        },
    }
    
    @staticmethod
    def get_cadence_chords(cadence_type: str, key_root: int, scale: list, mode: str) -> list:
        """Get chord progression for cadence type."""
        if cadence_type not in Cadences.TYPES:
            cadence_type = 'perfect_authentic'
        
        cadence = Cadences.TYPES[cadence_type]
        chords = []
        beat = 0
        
        chord_tones_map = {
            'I': [0, 2, 4], 'i': [0, 2, 4],
            'ii': [1, 3, 5], 'ii°': [1, 3, 6],
            'iii': [2, 4, 6], 'III': [2, 5, 7],  # Minor III
            'IV': [3, 5, 7],             'iv': [3, 5, 0],
            'V': [4, 6, 1], 'V7': [4, 6, 1, 3],
            'vi': [5, 7, 2], 'VI': [5, 8, 0],  # Minor VI
            'vii°': [6, 1, 3],
            'iv6': [5, 8, 0],  # iv6 has flat-6 in bass
        }
        
        for chord_name in cadence['chords']:
            numeral = chord_name.replace('7', '').replace('°', '').replace('6', '')
            tones = chord_tones_map.get(chord_name, chord_tones_map.get(numeral, [0, 2, 4]))
            
            chords.append({
                'chord': chord_name,
                'chord_tones': tones,
                'duration': 2.0,
                'beat': beat,
                'cadence_type': cadence_type,
                'strength': cadence['strength'],
            })
            beat += 2.0
        
        return chords


class ThemeDevelopment:
    """
    Thematic Development Techniques
    
    Classical composers developed themes through:
    - AUGMENTATION: Double durations (slower, grander)
    - DIMINUTION: Halve durations (faster, more urgent)
    - INVERSION: Reverse intervals (up becomes down)
    - RETROGRADE: Reverse order (last becomes first)
    - SEQUENCE: Repeat at different pitches
    - FRAGMENTATION: Use portions of theme
    - STRETTO: Overlapping entries in fugue
    """
    
    @staticmethod
    def augment(notes: list) -> list:
        """Double all durations - theme becomes grander."""
        return [{**n, 'duration': n['duration'] * 2} for n in notes]
    
    @staticmethod
    def diminish(notes: list) -> list:
        """Halve all durations - theme becomes more urgent."""
        return [{**n, 'duration': max(0.25, n['duration'] / 2)} for n in notes]
    
    @staticmethod
    def invert(notes: list) -> list:
        """Mirror intervals - ascending becomes descending."""
        if len(notes) < 2:
            return notes
        
        inverted = [notes[0].copy()]
        prev_pitch = notes[0]['pitch']
        
        for i in range(1, len(notes)):
            interval = notes[i]['pitch'] - notes[i-1]['pitch']
            inverted_pitch = prev_pitch - interval
            
            inverted.append({**notes[i], 'pitch': inverted_pitch})
            prev_pitch = inverted_pitch
        
        return inverted
    
    @staticmethod
    def retrograde(notes: list) -> list:
        """Reverse order - theme played backwards."""
        reversed_notes = [n.copy() for n in reversed(notes)]
        
        beat = 0
        for n in reversed_notes:
            n['beat'] = beat
            beat += n['duration']
        
        return reversed_notes
    
    @staticmethod
    def sequence(notes: list, intervals: list) -> list:
        """Repeat theme at different pitch levels."""
        result = []
        current_beat = 0
        sum(n['duration'] for n in notes)
        
        for interval in intervals:
            for note in notes:
                result.append({
                    **note,
                    'pitch': note['pitch'] + interval,
                    'beat': current_beat,
                })
                current_beat += note['duration']
        
        return result
    
    @staticmethod
    def fragment(notes: list, start: int, length: int) -> list:
        """Extract portion of theme."""
        fragment_notes = notes[start:start+length]
        
        beat = 0
        for n in fragment_notes:
            n['beat'] = beat
            beat += n['duration']
        
        return fragment_notes
    
    @staticmethod
    def apply_development(notes: list, technique: str) -> list:
        """Apply a development technique."""
        techniques = {
            'augment': ThemeDevelopment.augment,
            'diminish': ThemeDevelopment.diminish,
            'invert': ThemeDevelopment.invert,
            'retrograde': ThemeDevelopment.retrograde,
        }
        
        if technique in techniques:
            return techniques[technique](notes)
        
        return notes


class PeriodStructure:
    """
    Period Structure - Question and Answer
    
    ANTECEDENT (Question phrase):
    - Ends with weak cadence (half or imperfect)
    - Creates sense of openness
    - Typically 4-8 measures
    
    CONSEQUENT (Answer phrase):
    - Begins similarly to antecedent
    - Ends with strong cadence (perfect authentic)
    - Provides resolution
    - Same length as antecedent
    
    PARALLEL PERIOD: Both phrases begin the same
    CONTRASTING PERIOD: Different beginnings
    """
    
    @staticmethod
    def create_period(num_bars: int = 4) -> dict:
        """Create period structure with antecedent and consequent."""
        return {
            'antecedent': {
                'num_bars': num_bars,
                'ends_weakly': True,
                'target_cadence': 'half',
                'character': 'questioning',
            },
            'consequent': {
                'num_bars': num_bars,
                'ends_weakly': False,
                'target_cadence': 'perfect_authentic',
                'character': 'resolving',
            },
        }
    
    @staticmethod
    def get_phrase_character(phrase_num: int, total_phrases: int) -> dict:
        """Get character for phrase position in period."""
        if total_phrases == 2:
            if phrase_num == 0:
                return {'type': 'antecedent', 'tension': 'open', 'cadence': 'weak'}
            else:
                return {'type': 'consequent', 'tension': 'resolved', 'cadence': 'strong'}
        else:
            # Longer structure
            position = phrase_num / total_phrases
            if position < 0.3:
                return {'type': 'opening', 'tension': 'establishing', 'cadence': 'any'}
            elif position < 0.7:
                return {'type': 'development', 'tension': 'building', 'cadence': 'weak'}
            else:
                return {'type': 'closing', 'tension': 'resolving', 'cadence': 'strong'}


class SequencePatterns:
    """
    Sequential Patterns - Development Backbone
    
    CIRCLE OF FIFTHS: Descending 5ths (I-IV-vii°-iii-vi-ii-V-I)
    ASCENDING: Each repetition up one step
    DESCENDING: Each repetition down one step
    OMNIBUS: Chromatic sequence using diminished chords
    """
    
    PATTERNS = {
        'circle_of_fifths': {
            'degrees': [0, 3, 6, 2, 5, 1, 4],
            'description': 'Strongest harmonic sequence',
            'usage': 'Development sections, modulations',
        },
        'ascending_step': {
            'degrees': [0, 1, 2, 3, 4],
            'description': 'Rising sequence, momentum',
            'usage': 'Building intensity',
        },
        'descending_step': {
            'degrees': [4, 3, 2, 1, 0],
            'description': 'Falling sequence',
            'usage': 'Transitions, wind-downs',
        },
        'descending_third': {
            'degrees': [0, 5, 3, 1, 6, 4, 2],
            'description': 'Expressive, Romantic era',
            'usage': 'Emotional development',
        },
    }
    
    @staticmethod
    def generate_sequence(pattern_type: str, num_repeats: int, mode: str) -> list:
        """Generate chord sequence from pattern."""
        if pattern_type not in SequencePatterns.PATTERNS:
            pattern_type = 'circle_of_fifths'
        
        pattern = SequencePatterns.PATTERNS[pattern_type]
        degrees = pattern['degrees']
        
        major_numerals = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']
        minor_numerals = ['i', 'ii°', 'III', 'iv', 'V', 'VI', 'VII']
        
        numerals = minor_numerals if mode == 'minor' else major_numerals
        
        sequence = []
        for _ in range(num_repeats):
            for deg in degrees:
                if deg < len(numerals):
                    sequence.append(numerals[deg])
        
        return sequence


_engine = None
import threading as _threading
_engine_lock = _threading.Lock()


def get_music_theory_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = MusicTheoryEngine()
            return _engine
    return _engine


def get_counterpoint_rules():
    """Get counterpoint rules module."""
    return Counterpoint


def get_cadence_types():
    """Get cadence types."""
    return Cadences.TYPES


def get_non_chord_tones():
    """Get non-chord tone definitions."""
    return NonChordTones.TYPES
