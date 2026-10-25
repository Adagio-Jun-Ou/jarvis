"""
Symbolic Music Composition Engine v4 - Proper Phrasing & Harmony

Key fixes from v3:
1. RESTS AT PHRASE BOUNDARIES - Not random, but at musical cadence points
2. CHORDS + MELODY - Harmony supports melody, not just solo line
3. VOICE LEADING - Proper chord connections
4. STRUCTURED PHRASING - 4-bar phrases, 8-bar periods, proper cadences
"""

import random
import logging
import threading
from dataclasses import dataclass
from typing import Optional
import io

logger = logging.getLogger(__name__)

_mutex_lock = threading.Lock()

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALES = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
}

# Chord types - what notes make up each chord
CHORDS = {
    'major': [0, 4, 7],
    'minor': [0, 3, 7],
    'diminished': [0, 3, 6],
    'major7': [0, 4, 7, 11],
    'minor7': [0, 3, 7, 10],
    'dominant7': [0, 4, 7, 10],
}

# Chord progressions with BEATS per chord - not just symbols
# Format: [(degree, beats), ...] - degree is 1-7 in scale
# Extended with master-inspired progressions for variety

PROGRESSIONS = {
    'major': {
        # === BASIC CADENCES ===
        'authentic': [(1, 4), (5, 4), (1, 4)],              # I - V - I (basic resolution)
        'plagal': [(1, 4), (4, 4), (1, 4)],                  # I - IV - I (amen)
        'half_authentic': [(1, 4), (4, 2), (5, 4)],          # I - IV - V (half cadence)
        
        # === CLASSICAL ERA ===
        'classical': [(1, 4), (4, 2), (2, 2), (5, 4), (1, 4)],        # I - IV - ii - V - I
        'classical_extended': [(1, 4), (6, 2), (2, 2), (5, 4), (1, 4)], # I - vi - ii - V - I
        'mozart_style': [(1, 4), (5, 2), (6, 2), (4, 2), (5, 2), (1, 4)], # I - V - vi - IV - V - I
        'haydn_style': [(1, 4), (1, 2), (4, 2), (2, 2), (5, 4), (1, 4)], # I - I6 - IV - ii - V - I
        'circle_of_fifths': [(1, 4), (4, 2), (2, 2), (5, 2), (1, 4)],    # I - IV - ii - V - I
        
        # === BAROQUE ERA ===
        'baroque_cadence': [(1, 4), (4, 2), (5, 4), (1, 4)],            # I - IV - V - I
        'baroque_sequence': [(1, 2), (5, 2), (6, 2), (3, 2), (4, 2), (2, 2), (5, 4), (1, 4)], # Descending 5ths
        'bach_chorale': [(1, 4), (4, 2), (5, 2), (6, 2), (3, 2), (4, 2), (5, 4), (1, 4)],
        'baroque_romanesca': [(1, 4), (5, 4), (6, 4), (3, 4), (4, 2), (5, 2), (1, 4)], # I-V-vi-iii-IV-V-I
        'passamezzo': [(1, 4), (1, 2), (4, 2), (1, 4), (5, 2), (4, 2), (5, 2), (1, 4)], # Ground bass
        
        # === ROMANTIC ERA ===
        'romantic': [(1, 4), (6, 2), (4, 2), (5, 4), (1, 4)],          # I - vi - IV - V - I
        'schubert_style': [(1, 4), (6, 4), (3, 2), (5, 2), (1, 4)],     # I - vi - iii - V - I
        'chopin_style': [(1, 4), (2, 2), (4, 2), (6, 2), (5, 4), (1, 4)], # I - ii - IV - vi - V - I
        'brahms_style': [(1, 4), (6, 2), (2, 2), (4, 2), (5, 2), (1, 4)], # I - vi - ii - IV - V - I
        
        # === MODERN/POP ===
        'pop_standard': [(1, 4), (6, 4), (4, 4), (5, 4)],              # I - vi - IV - V (50s)
        'pop_modern': [(1, 4), (5, 4), (6, 4), (4, 4)],                # I - V - vi - IV (modern)
        'jazz_ii_V_I': [(2, 2), (5, 2), (1, 4)],                       # ii - V - I
        'jazz_turnaround': [(1, 2), (6, 2), (2, 2), (5, 2)],           # I - vi - ii - V
        'rhythm_changes': [(1, 4), (6, 2), (2, 2), (5, 4)],           # A section of Rhythm Changes
        
        # === DEVELOPMENTAL ===
        'circle_progression': [(1, 4), (4, 2), (2, 2), (5, 2), (1, 2), (6, 2), (2, 2), (5, 4)], # Extended circle
        'sequential': [(1, 2), (5, 2), (6, 2), (3, 2), (4, 2), (2, 2), (5, 2), (1, 2)], # Sequence by fifths
        'deceptive_resolution': [(1, 4), (4, 2), (5, 2), (6, 4)],      # I - IV - V - vi (deceptive)
        'modal_interchange': [(1, 4), (4, 2), (2, 2), (5, 2), (1, 2), (b3 := 3, 2), (4, 2), (1, 4)], # With bIII
        
        # === SOPHISTICATED ===
        'secondary_dominants': [(1, 4), (5, 2), (5, 2), (6, 2), (5, 2), (1, 4)], # V7/V and V7/vi
        'applied_dominants': [(1, 2), (5, 2), (2, 2), (5, 2), (3, 2), (5, 2), (4, 2), (5, 2), (1, 4)],
        'diminished_passing': [(1, 4), (1, 2), (2, 2), (4, 2), (5, 4), (1, 4)], # With diminished passing
        'neapolitan': [(1, 4), (4, 2), (2, 2), (5, 4), (1, 4)],        # Classical with N6
    },
    'minor': {
        # === BASIC CADENCES ===
        'authentic': [(1, 4), (5, 4), (1, 4)],              # i - V - i
        'plagal': [(1, 4), (4, 4), (1, 4)],                  # i - iv - i
        'half': [(1, 4), (4, 2), (5, 4)],                    # i - iv - V (half cadence)
        
        # === BAROQUE MINOR ===
        'andalusian': [(1, 4), (7, 4), (6, 4), (5, 4)],     # i - VII - VI - V (famous progression)
        'baroque_minor': [(1, 4), (4, 2), (5, 4), (1, 4)],  # i - iv - V - i
        'bach_minor': [(1, 4), (5, 2), (6, 2), (7, 2), (3, 2), (4, 2), (5, 4), (1, 4)],
        'passamezzo_antico': [(1, 4), (5, 2), (3, 2), (1, 4), (1, 2), (4, 2), (5, 2), (1, 4)], # Minor ground
        
        # === CLASSICAL MINOR ===
        'classical_minor': [(1, 4), (6, 2), (2, 2), (5, 4), (1, 4)],     # i - VI - ii° - V - i
        'beethoven_minor': [(1, 4), (4, 2), (5, 2), (1, 2), (5, 2), (1, 4)], # Dramatic minor
        'mozart_minor': [(1, 4), (4, 2), (2, 2), (5, 4), (1, 4)],         # i - iv - ii° - V - i
        
        # === ROMANTIC MINOR ===
        'romantic_minor': [(1, 4), (6, 4), (3, 4), (7, 4), (1, 4)],       # i - VI - III - VII - i
        'chopin_minor': [(1, 4), (6, 2), (4, 2), (5, 2), (1, 2), (7, 2), (1, 4)], # Dramatic Chopin
        'brahms_minor': [(1, 4), (4, 2), (6, 2), (5, 2), (1, 4)],        # i - iv - VI - V - i
        'dorian_mode': [(1, 4), (7, 4), (4, 4), (3, 4), (1, 4)],          # Modal, not functional
        
        # === SOPHISTICATED MINOR ===
        'circle_minor': [(1, 4), (4, 2), (2, 2), (5, 4), (1, 4)],        # Minor circle
        'harmonic_minor': [(1, 4), (6, 2), (2, 2), (5, 4), (1, 4)],       # Emphasizing V7-i resolution
        'melodic_minor': [(1, 4), (4, 2), (2, 2), (5, 2), (1, 4)],        # Melodic minor ascent
        'chromatic_minor': [(1, 4), (1, 2), (2, 2), (3, 2), (4, 2), (5, 4), (1, 4)], # Chromatic mediants
        'jazz_minor': [(2, 2), (5, 2), (1, 4)],                            # iiø - V - i
        'phrygian': [(1, 4), (2, 4), (3, 4), (5, 4), (1, 4)],             # Flamenco progression
    },
    'harmonic_minor': {
        'authentic': [(1, 4), (5, 4), (1, 4)],
        'flamenco': [(1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (1, 4)],    # i - bII - bIII - iv - V - i
        'classical': [(1, 4), (6, 2), (7, 2), (5, 4), (1, 4)],           # i - VI - viio7 - V7 - i
        'romantic': [(1, 4), (6, 4), (5, 4), (1, 4)],                     # i - VI - V - i
    },
    'dorian': {
        'dorian_folk': [(1, 4), (7, 4), (4, 4), (3, 4), (1, 4)],          # Modal dorian
        'dorian_rock': [(1, 4), (7, 4), (4, 4), (7, 4)],                   # i - bVII - IV - bVII
        'dorian_jazz': [(2, 2), (5, 2), (1, 4)],                           # ii - V - i (in Dorian mode)
    },
    'mixolydian': {
        'dominant_blues': [(1, 4), (7, 4), (4, 4), (1, 4)],                # Mixolydian blues
        'rock_dominant': [(1, 4), (4, 4), (1, 4), (5, 4)],                 # I - IV - I - bVII
        'folk_dominant': [(1, 4), (5, 4), (4, 4), (1, 4)],                 # I - V - IV - I
    },
    'phrygian': {
        'phrygian_flamenco': [(1, 4), (2, 4), (3, 4), (4, 4), (1, 4)],     # i - bII - bIII - iv - i
        'phrygian_metal': [(1, 4), (2, 4), (1, 4), (5, 4)],                # i - bII - i - V
    },
}

# Time signatures - structured phrasing
TIME_SIGNATURES = {
    '4/4': {'beats_per_measure': 4, 'strong_beats': [1, 3]},
    '3/4': {'beats_per_measure': 3, 'strong_beats': [1]},
    '2/4': {'beats_per_measure': 2, 'strong_beats': [1]},
}

DYNAMICS = {'pp': 20, 'p': 40, 'mp': 55, 'mf': 75, 'f': 95, 'ff': 115}


@dataclass
class Note:
    pitch: int
    duration: float
    velocity: int = 80
    start: float = 0.0
    voice: int = 0  # 0 = melody, 1 = harmony/bass
    
    def __repr__(self):
        octave = self.pitch // 12 - 1
        note_name = NOTES[self.pitch % 12]
        return f"{note_name}{octave}"


@dataclass  
class Chord:
    """A chord with its notes and duration."""
    degree: int  # 1-7 in scale
    notes: list[int]  # MIDI pitches
    duration: float  # In beats
    start: float  # When it starts


@dataclass
class Phrase:
    """Musical phrase - melody AND harmony together."""
    melody_notes: list[Note]
    harmony_notes: list[Note]  # Chord tones
    bass_notes: list[Note]     # Bass line
    start_beat: float
    duration: float
    dynamics: str = 'mf'
    is_cadence: bool = False  # Does this phrase end with cadence?
    section: str = 'A'         # Which section (A, B, C)
    chord_name: str = 'I'      # Chord symbol for this phrase


@dataclass
class Composition:
    title: str
    tempo: int
    key: str
    mode: str
    time_signature: tuple
    phrases: list[Phrase]
    total_duration: float
    form: str = 'ternary'
    
    def to_symbolic_tokens(self) -> list[dict]:
        tokens = []
        
        tokens.append({
            "type": "structure",
            "content": {"form": self.form, "phrases": len(self.phrases)}
        })
        tokens.append({
            "type": "tonality", 
            "content": {"key": self.key, "mode": self.mode, "tempo": self.tempo}
        })
        
        for i, phrase in enumerate(self.phrases):
            # Melody notes
            melody_data = []
            for note in phrase.melody_notes[:16]:
                if note.velocity > 0:
                    melody_data.append({
                        "pitch": f"{NOTES[note.pitch % 12]}{note.pitch // 12 - 1}",
                        "duration": note.duration,
                        "beat": round(note.start, 2)
                    })
            
            # Chord info
            chord_notes = []
            for note in phrase.harmony_notes[:8]:
                if note.velocity > 0:
                    chord_notes.append(f"{NOTES[note.pitch % 12]}{note.pitch // 12 - 1}")
            
            # Bass info
            bass_notes = []
            for note in phrase.bass_notes[:4]:
                if note.velocity > 0:
                    bass_notes.append(f"{NOTES[note.pitch % 12]}{note.pitch // 12 - 1}")
            
            tokens.append({
                "type": "phrase",
                "content": {
                    "phrase_num": i + 1,
                    "melody": melody_data,
                    "chords": chord_notes[:3],  # Top 3 chord tones
                    "bass": bass_notes[:2],  # Bass notes
                    "dynamics": phrase.dynamics,
                    "cadence": phrase.is_cadence
                }
            })
        
        return tokens
    
    def to_midi(self) -> bytes:
        from midiutil import MIDIFile
        
        # 3 tracks: melody, harmony, bass
        midi = MIDIFile(numTracks=3)
        midi.addTrackName(0, 0, f"{self.title} - Melody")
        midi.addTrackName(1, 0, f"{self.title} - Harmony")
        midi.addTrackName(2, 0, f"{self.title} - Bass")
        midi.addTempo(0, 0, self.tempo)
        midi.addTempo(1, 0, self.tempo)
        midi.addTempo(2, 0, self.tempo)
        
        for phrase in self.phrases:
            DYNAMICS.get(phrase.dynamics, 80)
            
            # Melody track (voice 0)
            for note in phrase.melody_notes:
                if note.velocity > 0:
                    midi.addNote(0, 0, note.pitch, note.start, note.duration, note.velocity)
            
            # Harmony track (voice 1) - softer
            for note in phrase.harmony_notes:
                if note.velocity > 0:
                    midi.addNote(1, 0, note.pitch, note.start, note.duration, int(note.velocity * 0.6))
            
            # Bass track (voice 2)
            for note in phrase.bass_notes:
                if note.velocity > 0:
                    midi.addNote(2, 0, note.pitch, note.start, note.duration, int(note.velocity * 0.8))
        
        buffer = io.BytesIO()
        midi.writeFile(buffer)
        buffer.seek(0)
        return buffer.read()


class EnhancedComposer:
    """
    Compose with proper structure:
    - Melody + Harmony + Bass (3 voices)
    - Rests only at phrase/cadence boundaries
    - Proper chord progressions
    """
    
    def __init__(self, spontaneity: float = 0.2):
        self.spontaneity = spontaneity
    
    def get_scale(self, root: int, mode: str) -> list[int]:
        pattern = SCALES.get(mode, SCALES['major'])
        return [root + interval for interval in pattern]
    
    def get_chord_from_degree(self, degree: int, scale: list[int], mode: str) -> tuple[str, list[int]]:
        """Get chord from scale degree (1-7)."""
        degree = max(1, min(7, degree))
        root = scale[degree - 1]
        
        # Chord quality by degree and mode
        if mode == 'major':
            qualities = {1: 'major', 2: 'minor', 3: 'minor', 4: 'major', 5: 'major', 6: 'minor', 7: 'diminished'}
            numerals = {1: 'I', 2: 'ii', 3: 'iii', 4: 'IV', 5: 'V', 6: 'vi', 7: 'vii°'}
        else:  # minor
            qualities = {1: 'minor', 2: 'diminished', 3: 'major', 4: 'minor', 5: 'major', 6: 'major', 7: 'diminished'}
            numerals = {1: 'i', 2: 'ii°', 3: 'III', 4: 'iv', 5: 'V', 6: 'VI', 7: 'vii°'}
        
        quality = qualities.get(degree, 'major')
        chord_pattern = CHORDS.get(quality, CHORDS['major'])
        notes = [root + interval for interval in chord_pattern]
        
        return numerals[degree], notes
    
    def generate_melody_for_chord(
        self,
        chord_notes: list[int],
        scale: list[int],
        beats: float,
        start_beat: float,
        base_octave: int = 4,
        is_cadence: bool = False
    ) -> list[Note]:
        """Generate melody that fits over chord tones."""
        notes = []
        current_beat = start_beat
        
        # Rhythm patterns - musical, not random
        if is_cadence:
            # Cadence: longer notes toward resolution
            rhythm_pattern = [1.0, 1.0, 2.0]  # End on longer note
        else:
            # Regular phrase: varied rhythm
            rhythm_pattern = [0.5, 0.5, 1.0, 0.5, 0.5, 1.0]
        
        rhythm_idx = 0
        prev_pitch = chord_notes[0] + base_octave * 12
        
        while current_beat < start_beat + beats:
            # Choose duration
            duration = rhythm_pattern[rhythm_idx % len(rhythm_pattern)]
            rhythm_idx += 1
            
            # Ensure we don't exceed phrase boundary
            if current_beat + duration > start_beat + beats:
                duration = start_beat + beats - current_beat
                if duration <= 0:
                    break
            
            # Melody: prefer chord tones, with passing tones
            if random.random() < 0.7:  # 70% chord tones
                # Choose from chord tones
                chord_tone = random.choice(chord_notes)
                pitch = chord_tone + base_octave * 12
            else:
                # Passing tone - scale degree not in chord
                passing_tones = [s for s in scale if s % 12 not in [c % 12 for c in chord_notes]]
                if passing_tones:
                    pitch = random.choice(passing_tones) + base_octave * 12
                else:
                    pitch = prev_pitch
            
            # Stepwise motion preference
            if random.random() < 0.6 and abs(pitch - prev_pitch) > 2:
                # Step toward chord tone instead of leap
                direction = 1 if pitch > prev_pitch else -1
                step = min(2, abs(pitch - prev_pitch))
                pitch = prev_pitch + direction * step
            
            # Velocity - phrase shape
            position_in_phrase = (current_beat - start_beat) / beats
            if position_in_phrase < 0.3:
                velocity = 75  # Start softer
            elif position_in_phrase < 0.7:
                velocity = 85  # Build
            else:
                velocity = 70 if is_cadence else 80  # Resolve or maintain
            
            note = Note(
                pitch=pitch,
                duration=duration,
                velocity=velocity,
                start=current_beat,
                voice=0
            )
            notes.append(note)
            
            current_beat += duration
            prev_pitch = pitch
        
        return notes
    
    def generate_harmony_for_chord(
        self,
        chord_notes: list[int],
        beats: float,
        start_beat: float,
        base_octave: int = 3
    ) -> list[Note]:
        """Generate harmony (chord tones) sustained for chord duration."""
        notes = []
        
        # Harmony plays chord tones - sustained
        for i, note_degree in enumerate(chord_notes):
            # Different octaves for voice leading
            octave = base_octave + (i > 0)  # Spread across octaves
            pitch = note_degree + octave * 12
            
            note = Note(
                pitch=pitch,
                duration=beats,
                velocity=55,  # Soft accompaniment
                start=start_beat,
                voice=1
            )
            notes.append(note)
        
        return notes
    
    def generate_bass_for_chord(
        self,
        chord_notes: list[int],
        beats: float,
        start_beat: float,
        is_cadence: bool = False
    ) -> list[Note]:
        """Generate bass line from chord root."""
        notes = []
        
        root = chord_notes[0]  # Root determines bass
        bass_octave = 2  # Low bass
        
        if is_cadence:
            # Cadence: bass holds root
            note = Note(
                pitch=root + bass_octave * 12,
                duration=beats,
                velocity=70,
                start=start_beat,
                voice=2
            )
            notes.append(note)
        else:
            # Regular: bass plays root on beat 1, maybe 5th
            # Beat 1: Root
            note1 = Note(
                pitch=root + bass_octave * 12,
                duration=beats * 0.5,
                velocity=70,
                start=start_beat,
                voice=2
            )
            notes.append(note1)
            
            # Optional: 5th on beat 3 (in 4/4)
            if beats >= 4 and random.random() < 0.5:
                fifth = chord_notes[2] if len(chord_notes) > 2 else root + 7
                note2 = Note(
                    pitch=fifth + bass_octave * 12,
                    duration=beats * 0.5,
                    velocity=60,
                    start=start_beat + beats * 0.5,
                    voice=2
                )
                notes.append(note2)
            else:
                # Hold root
                note2 = Note(
                    pitch=root + bass_octave * 12,
                    duration=beats * 0.5,
                    velocity=65,
                    start=start_beat + beats * 0.5,
                    voice=2
                )
                notes.append(note2)
        
        return notes
    
    def compose(
        self,
        style: str = 'classical',
        form: str = 'ternary',
        key: str = 'C',
        mode: str = 'major',
        tempo: int = 120,
        length_bars: int = 32
    ) -> Composition:
        """Compose with melody, harmony, and bass - proper structure."""
        
        # Parse key
        key = key.replace('_major', '').replace('_minor', '')
        if 'minor' in key.lower():
            mode = 'minor'
            key = key.replace(' minor', '').replace('_minor', '')
        key_root = NOTES.index(key) if key in NOTES else 0
        
        scale = self.get_scale(key_root, mode)
        
        # Choose chord progressions based on style and mode
        progressions = PROGRESSIONS.get(mode, PROGRESSIONS['major'])
        
        # Style-appropriate progression selection
        style_progression_map = {
            'baroque': ['baroque_cadence', 'baroque_sequence', 'bach_chorale', 'passamezzo', 'circle_of_fifths'],
            'classical': ['classical', 'classical_extended', 'mozart_style', 'haydn_style', 'circle_of_fifths'],
            'romantic': ['romantic', 'schubert_style', 'chopin_style', 'brahms_style', 'deceptive_resolution'],
            'modern': ['pop_standard', 'pop_modern', 'jazz_ii_V_I', 'jazz_turnaround'],
            'jazz': ['jazz_ii_V_I', 'jazz_turnaround', 'rhythm_changes', 'secondary_dominants'],
        }
        
        preferred_names = style_progression_map.get(style, list(progressions.keys()))
        
        # Form structure - different progressions per section
        if form == 'binary':
            sections = ['A', 'B']
            measures_per_section = length_bars // 2
        elif form == 'ternary':
            sections = ['A', 'B', 'A']
            measures_per_section = length_bars // 3
        elif form == 'rondo':
            sections = ['A', 'B', 'A', 'C', 'A']
            measures_per_section = length_bars // 5
        else:
            sections = ['A', 'B', 'A']
            measures_per_section = length_bars // 3
        
        # Assign progressions to sections - different for contrasting sections
        section_progressions = {}
        seen_sections = {}
        for section in sections:
            if section not in seen_sections:
                # Filter available progressions
                available = [n for n in preferred_names if n in progressions]
                if not available:
                    available = list(progressions.keys())
                
                # A sections: stable progressions, B/C sections: developmental
                if section == 'A':
                    # Stable, tonic-establishing progressions
                    stable_names = ['authentic', 'classical', 'classical_extended', 'circle_of_fifths',
                                   'baroque_cadence', 'pop_standard', 'circle_minor']
                    stable_available = [n for n in stable_names if n in progressions]
                    prog_name = random.choice(stable_available) if stable_available else random.choice(available)
                elif section == 'B':
                    # Developmental, modulating progressions
                    dev_names = ['sequential', 'circle_progression', 'secondary_dominants', 
                                'deceptive_resolution', 'romantic', 'baroque_sequence',
                                'andalusian', 'chromatic_minor']
                    dev_available = [n for n in dev_names if n in progressions]
                    prog_name = random.choice(dev_available) if dev_available else random.choice(available)
                else:  # C or other contrasting sections
                    # Varied, contrasting progressions
                    # Fall back to regular progressions if not available
                    prog_name = random.choice(available)
                
                section_progressions[section] = progressions[prog_name]
                seen_sections[section] = prog_name
            else:
                # Return sections use same progression
                section_progressions[section] = section_progressions[section]
        
        phrases = []
        current_beat = 0.0
        beats_per_measure = 4  # 4/4 time
        
        for section_idx, section in enumerate(sections):
            # Dynamic level by section
            if section == 'A':
                section_dynamics = 'mf' if section_idx == 0 else 'mp'  # Quieter return
            else:  # B or C section
                section_dynamics = 'f'
            
            # Get progression for this section
            progression = section_progressions[section]
            
            # Number of phrase units in this section
            phrase_units = max(2, measures_per_section // 4)  # 4-measure phrases
            
            for phrase_idx in range(phrase_units):
                # Get chords for this phrase (subset of progression)
                phrase_measures = measures_per_section // phrase_units
                phrase_beats = phrase_measures * beats_per_measure
                
                # Is this the last phrase of a section? (cadence)
                is_cadence = (phrase_idx == phrase_units - 1)
                
                # Get chord for this phrase - cycle through progression with variation
                chord_idx = phrase_idx % len(progression)
                chord_degree, chord_beats = progression[chord_idx]
                
                numeral, chord_notes = self.get_chord_from_degree(chord_degree, scale, mode)
                
                # Generate melody over chord
                melody = self.generate_melody_for_chord(
                    chord_notes=chord_notes,
                    scale=scale,
                    beats=phrase_beats,
                    start_beat=current_beat,
                    base_octave=4,
                    is_cadence=is_cadence
                )
                
                # Generate harmony (chord tones sustained)
                harmony = self.generate_harmony_for_chord(
                    chord_notes=chord_notes,
                    beats=phrase_beats,
                    start_beat=current_beat,
                    base_octave=3
                )
                
                # Generate bass
                bass = self.generate_bass_for_chord(
                    chord_notes=chord_notes,
                    beats=phrase_beats,
                    start_beat=current_beat,
                    is_cadence=is_cadence
                )
                
                # Dynamic shape
                dynamics = section_dynamics
                if is_cadence:
                    dynamics = 'mp' if dynamics == 'mf' else 'p'  # Softer resolution
                
                phrase = Phrase(
                    melody_notes=melody,
                    harmony_notes=harmony,
                    bass_notes=bass,
                    start_beat=current_beat,
                    duration=phrase_beats,
                    dynamics=dynamics,
                    is_cadence=is_cadence,
                    section=section,
                    chord_name=numeral
                )
                
                phrases.append(phrase)
                
                # Move to next phrase - NO RANDOM RESTS, just move forward
                current_beat += phrase_beats
                
                # Structured pause only at section ends
                if is_cadence and section_idx < len(sections) - 1:
                    # Fermata/breath at end of section - ONE beat pause
                    current_beat += 1.0
        
        total_duration = current_beat
        
        composition = Composition(
            title=f"{style.capitalize()} {form.capitalize()} in {key} {mode}",
            tempo=tempo,
            key=key,
            mode=mode,
            time_signature=(4, 4),
            phrases=phrases,
            total_duration=total_duration,
            form=form
        )
        
        return composition


def compose_piece(
    style: str = 'classical',
    form: str = 'ternary',
    key: str = 'C_major',
    mood: str = 'peaceful',
    tempo: int = 120
) -> Composition:
    """Main composition - now with harmony and structured phrasing."""
    
    # Spontaneity by mood
    spontaneity = {'peaceful': 0.15, 'calm': 0.15, 'energetic': 0.3, 'dramatic': 0.35}.get(mood, 0.2)
    
    composer = EnhancedComposer(spontaneity=spontaneity)
    
    mode = 'minor' if 'minor' in key.lower() else 'major'
    
    composition = composer.compose(
        style=style,
        form=form,
        key=key.replace('_major', '').replace('_minor', ''),
        mode=mode,
        tempo=tempo,
        length_bars=24
    )
    
    return composition


def midi_to_audio(midi_bytes: bytes, sample_rate: int = 44100) -> tuple:
    """Convert MIDI to audio using FluidSynth.
    
    Thread-safe: Uses a mutex lock to prevent race conditions when
    FluidSynth spawns external processes concurrently.
    """
    from midi2audio import FluidSynth
    import tempfile
    import wave
    import numpy as np
    import os
    
    audio_result = [None, None]
    audio_error = [None]
    
    def _convert():
        try:
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as midi_file:
                midi_file.write(midi_bytes)
                midi_path = midi_file.name
            
            fs = FluidSynth(
                sound_font='/usr/share/sounds/sf2/TimGM6mb.sf2',
                sample_rate=sample_rate
            )
            
            wav_path = midi_path.replace('.mid', '.wav')
            fs.midi_to_audio(midi_path, wav_path)
            
            with wave.open(wav_path, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                
                audio_data = wav.readframes(frames)
                
                if sample_width == 2:
                    audio = np.frombuffer(audio_data, dtype=np.int16)
                elif sample_width == 4:
                    audio = np.frombuffer(audio_data, dtype=np.int32)
                else:
                    audio = np.frombuffer(audio_data, dtype=np.uint8)
                
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
                
                if channels == 2:
                    audio = audio.reshape(-1, 2).mean(axis=1)
            
            os.unlink(midi_path)
            os.unlink(wav_path)
            
            audio_result[0] = audio
            audio_result[1] = rate
            
        except Exception as e:
            audio_error[0] = e
            try:
                if 'midi_path' in dir():
                    os.unlink(midi_path)
                if 'wav_path' in dir():
                    os.unlink(wav_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp files: {e}")
    
    with _mutex_lock:
        _convert()
    
    if audio_error[0]:
        raise audio_error[0]
    
    return audio_result[0], audio_result[1]


_composer: Optional[EnhancedComposer] = None

def get_composer() -> EnhancedComposer:
    global _composer
    if _composer is None:
        _composer = EnhancedComposer()
    return _composer