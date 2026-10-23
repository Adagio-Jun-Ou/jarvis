"""MIDI Humanization - Making music sound natural, not mechanical.

Key concepts:
1. TIMING: Notes don't start exactly on beat - slight variations
2. VELOCITY: Not all notes same volume - natural dynamics
3. ARTICULATION: Notes have shape - attack, sustain, release
4. PHRASE SHAPE: Music breathes - natural swells and fades
5. RUBATO: Tempo breathes - slight speeding and slowing
6. MICROTIMING: Swing, groove, human feel

Human players:
- Never start exactly on beat (±5-20ms variation)
- Never hit same velocity twice
- Shape phrases naturally
- Add subtle timing variations
- Breath between phrases
- Vibrate sustained notes (vibrato simulation)
"""
import logging
import random
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Human playing characteristics by style/era
PLAYING_STYLES = {
    'baroque': {
        'timing_variation': 0.015,  # Fairly precise
        'velocity_variation': 8,
        'rubato_amount': 0.02,
        'articulation': 'detached',  # Notes slightly separated
        'phrase_shape': 'terraced',  # Dynamics in steps, not swells
        'vibrato': 'rare',  # Used sparingly
        'typical_velocity': (65, 90),
        'staccato_probability': 0.3,
        'legato_probability': 0.4,
    },
    'classical': {
        'timing_variation': 0.02,
        'velocity_variation': 12,
        'rubato_amount': 0.03,
        'articulation': 'balanced',
        'phrase_shape': 'arched',  # Natural swells
        'vibrato': 'moderate',
        'typical_velocity': (55, 95),
        'staccato_probability': 0.2,
        'legato_probability': 0.5,
    },
    'romantic': {
        'timing_variation': 0.03,  # More freedom
        'velocity_variation': 18,
        'rubato_amount': 0.05,  # Significant rubato
        'articulation': 'connected',
        'phrase_shape': 'long_arched',
        'vibrato': 'frequent',
        'typical_velocity': (50, 100),
        'staccato_probability': 0.1,
        'legato_probability': 0.65,
    },
    'modern': {
        'timing_variation': 0.025,
        'velocity_variation': 15,
        'rubato_amount': 0.03,
        'articulation': 'varied',
        'phrase_shape': 'angular',
        'vibrato': 'varied',
        'typical_velocity': (40, 105),
        'staccato_probability': 0.25,
        'legato_probability': 0.35,
    },
    'jazz': {
        'timing_variation': 0.04,  # Swing feel
        'velocity_variation': 20,
        'rubato_amount': 0.04,
        'articulation': 'varied',
        'phrase_shape': 'syncopated',
        'vibrato': 'expressive',
        'typical_velocity': (45, 100),
        'staccato_probability': 0.15,
        'legato_probability': 0.55,
    },
}

# Velocity curves for natural dynamics
VELOCITY_CURVES = {
    'crescendo': lambda x: int(55 + 35 * (x ** 0.7)),  # Gradual swell
    'decrescendo': lambda x: int(90 - 35 * (x ** 0.7)),  # Gradual fade
    'accent': lambda x: int(75 + 15 * math.sin(x * math.pi)),  # Peak in middle
    'natural_phrase': lambda x: int(55 + 30 * math.sin(x * math.pi)),  # Arch shape
    'flat': lambda x: 68,  # No shape
    'dramatic': lambda x: int(50 + 40 * (x ** 2)),  # Exponential growth
}

@dataclass
class HumanizedNote:
    """A note with human characteristics."""
    pitch: int
    start_time: float  # Humanized start time
    duration: float  # Actual duration (may differ from notated)
    velocity: int  # Humanized velocity
    articulation: str  # 'legato', 'staccato', 'tenuto', 'accent', 'normal'
    timing_offset: float  # How much earlier/later than beat
    end_time: float  # Calculated end time
    is_breath_point: bool  # Should have slight pause before
    phrase_position: float  # 0-1, where in phrase (for shaping)


class MIDIHumanizer:
    """
    Applies human characteristics to MIDI-like note data.
    
    Making music sound PLAYED, not programmed.
    """
    
    def __init__(self):
        self.style_profiles = PLAYING_STYLES
        
    def humanize_phrase(
        self,
        notes: list[dict],
        style: str = 'classical',
        tempo: int = 90,
        emotional_intensity: float = 0.5,
        phrase_position: str = 'middle',  # 'start', 'middle', 'end', 'climax'
    ) -> list[HumanizedNote]:
        """
        Apply humanization to a phrase of notes.
        
        Args:
            notes: List of note dictionaries with pitch, duration, beat, velocity
            style: Musical style (baroque, classical, romantic, modern)
            tempo: BPM for timing calculations
            emotional_intensity: How intense the emotional content (0-1)
            phrase_position: Where this phrase is in the larger phrase structure
        
        Returns:
            List of HumanizedNote objects
        """
        
        profile = self.style_profiles.get(style, self.style_profiles['classical'])
        humanized = []
        
        # Phrase timing offset (rubato)
        phrase_rubato = self._calculate_phrase_rubato(phrase_position, tempo, profile)
        
        # Get velocity curve for phrase shape
        phrase_curve = self._get_velocity_curve(profile['phrase_shape'], emotional_intensity)
        
        # Calculate phrase timing start
        phrase_start = notes[0].get('beat', 0) if notes else 0
        
        # Find breath points (ends of sub-phrases)
        breath_points = self._find_breath_points(notes)
        
        for i, note in enumerate(notes):
            # Position in phrase (0 to 1)
            pos_in_phrase = (note.get('beat', 0) - phrase_start) / max(1.0, (notes[-1].get('beat', 0) + notes[-1].get('duration', 1.0) - phrase_start)) if len(notes) > 1 else 0.5
            
            # Base values
            base_velocity = note.get('velocity', 80)
            base_duration = note.get('duration', 1.0)
            base_beat = note.get('beat', 0)
            
            # Apply velocity curve
            curve_velocity = phrase_curve(pos_in_phrase)
            base_velocity = int(base_velocity * 0.5 + curve_velocity * 0.5)
            
            # Add velocity variation
            velocity_variation = int(random.gauss(0, profile['velocity_variation']))
            final_velocity = max(30, min(127, base_velocity + velocity_variation))
            
            # Calculate timing
            beat_in_seconds = 60.0 / tempo
            timing_variation = random.gauss(0, profile['timing_variation'] * beat_in_seconds)
            
            # Apply rubato
            rubato_offset = phrase_rubato * (pos_in_phrase - 0.5)  # Negative at start, positive at end
            
            # Is this a breath point?
            is_breath = i > 0 and i in breath_points
            if is_breath:
                timing_variation -= 0.02 * beat_in_seconds  # Slight pause before breath point
            
            # Start time with all variations
            start_time = base_beat * beat_in_seconds + timing_variation + rubato_offset
            
            # Duration adjustments
            duration_adj = 1.0
            
            # Articulation
            articulation = 'normal'
            r = random.random()
            if r < profile['staccato_probability']:
                articulation = 'staccato'
                duration_adj = 0.5
            elif r < profile['staccato_probability'] + profile['legato_probability']:
                articulation = 'legato'
                duration_adj = 1.05
            
            # Apply duration
            actual_duration = base_duration * beat_in_seconds * duration_adj
            
            # Add small random duration variation (human players aren't perfect)
            duration_variation = random.gauss(0, 0.02 * beat_in_seconds)
            actual_duration = max(0.1, actual_duration + duration_variation)
            
            # Accent in climax
            if phrase_position == 'climax' and pos_in_phrase > 0.5:
                final_velocity = min(110, final_velocity + 5)
                if random.random() < 0.3:
                    articulation = 'accent'
            
            # Soft ending for phrases
            if phrase_position == 'end':
                final_velocity = int(final_velocity * 0.85)
                if random.random() < 0.5:
                    articulation = 'legato'
            
            humanized_note = HumanizedNote(
                pitch=note.get('pitch', 60),
                start_time=start_time,
                duration=actual_duration,
                velocity=final_velocity,
                articulation=articulation,
                timing_offset=timing_variation + rubato_offset,
                end_time=start_time + actual_duration,
                is_breath_point=is_breath,
                phrase_position=pos_in_phrase,
            )
            
            humanized.append(humanized_note)
        
        return humanized
    
    def humanize_melody_line(
        self,
        notes: list[dict],
        style: str = 'classical',
        tempo: int = 90,
        is_accompaniment: bool = False,
    ) -> list[HumanizedNote]:
        """
        Humanize a melody line with natural shaping.
        Melody gets more rubato and dynamic shaping.
        """
        
        self.style_profiles.get(style, self.style_profiles['classical'])
        
        if not notes:
            return []
        
        # Find phrase boundaries
        phrase_boundaries = self._find_phrase_boundaries(notes)
        
        all_humanized = []
        phrase_start_idx = 0
        
        for boundary_idx in phrase_boundaries + [len(notes)]:
            phrase_notes = notes[phrase_start_idx:boundary_idx]
            
            if len(phrase_notes) <= 1:
                phrase_start_idx = boundary_idx
                continue
            
            # Determine phrase position
            phrases_count = len(phrase_boundaries) + 1
            current_phrase_num = len([b for b in phrase_boundaries if b < boundary_idx])
            
            if current_phrase_num == 0:
                phrase_pos = 'start'
            elif current_phrase_num == phrases_count - 1:
                phrase_pos = 'end'
            elif current_phrase_num == phrases_count // 2:
                phrase_pos = 'climax'
            else:
                phrase_pos = 'middle'
            
            # Humanize this phrase
            humanized_phrase = self.humanize_phrase(
                phrase_notes,
                style=style,
                tempo=tempo,
                emotional_intensity=0.6 if phrase_pos == 'climax' else 0.4,
                phrase_position=phrase_pos,
            )
            
            # Add timing offset for phrase separation
            for note in humanized_phrase:
                note.start_time += phrase_start_idx * (60.0 / tempo) * 0.01  # Tiny offset between phrases
            
            all_humanized.extend(humanized_phrase)
            phrase_start_idx = boundary_idx
        
        return all_humanized
    
    def humanize_accompaniment(
        self,
        notes: list[dict],
        style: str = 'classical',
        tempo: int = 90,
        pattern_type: str = 'alberti',  # 'alberti', 'chord', 'arpeggio', 'bass'
    ) -> list[HumanizedNote]:
        """
        Humanize accompaniment patterns.
        Less rubato than melody, steady but not mechanical.
        """
        
        profile = self.style_profiles.get(style, self.style_profiles['classical'])
        
        if not notes:
            return []
        
        humanized = []
        beat_in_seconds = 60.0 / tempo
        
        for i, note in enumerate(notes):
            base_velocity = note.get('velocity', 70)
            
            # Accompaniment is softer than melody
            base_velocity = int(base_velocity * 0.6)
            
            # Add slight variation
            velocity_variation = int(random.gauss(0, profile['velocity_variation'] * 0.5))
            final_velocity = max(30, min(100, base_velocity + velocity_variation))
            
            # Steadier timing for accompaniment
            timing_variation = random.gauss(0, profile['timing_variation'] * beat_in_seconds * 0.5)
            
            # Beat position
            beat = note.get('beat', i)
            start_time = beat * beat_in_seconds + timing_variation
            
            # Duration depends on pattern
            base_duration = note.get('duration', 1.0)
            
            if pattern_type == 'staccato':
                actual_duration = base_duration * beat_in_seconds * 0.5
                articulation = 'staccato'
            elif pattern_type == 'legato':
                actual_duration = base_duration * beat_in_seconds * 1.05
                articulation = 'legato'
            else:
                actual_duration = base_duration * beat_in_seconds
                articulation = 'normal'
            
            humanized_note = HumanizedNote(
                pitch=note.get('pitch', 60),
                start_time=start_time,
                duration=actual_duration,
                velocity=final_velocity,
                articulation=articulation,
                timing_offset=timing_variation,
                end_time=start_time + actual_duration,
                is_breath_point=False,
                phrase_position=0.5,  # Middle of range for accompaniment
            )
            
            humanized.append(humanized_note)
        
        return humanized
    
    def humanize_bass_line(
        self,
        notes: list[dict],
        style: str = 'classical',
        tempo: int = 90,
    ) -> list[HumanizedNote]:
        """
        Humanize bass line.
        Bass provides foundation - steady but present.
        """
        
        profile = self.style_profiles.get(style, self.style_profiles['classical'])
        
        if not notes:
            return []
        
        humanized = []
        beat_in_seconds = 60.0 / tempo
        
        for i, note in enumerate(notes):
            # Bass is steady and supportive, not overpowering
            base_velocity = note.get('velocity', 50)
            
            # Slight dynamic shape based on harmonic role
            # Strong beats slightly louder
            beat_position = note.get('beat', i) % 4  # Within measure
            if beat_position < 0.1:  # Downbeat
                base_velocity = int(base_velocity * 1.05)
            
            velocity_variation = int(random.gauss(0, profile['velocity_variation'] * 0.4))
            final_velocity = max(35, min(80, base_velocity + velocity_variation))
            
            # Bass timing is steady
            timing_variation = random.gauss(0, profile['timing_variation'] * beat_in_seconds * 0.3)
            
            beat = note.get('beat', i)
            start_time = beat * beat_in_seconds + timing_variation
            
            # Bass notes sustain
            base_duration = note.get('duration', 2.0)  # Longer by default
            actual_duration = base_duration * beat_in_seconds * 0.95  # Slight separation
            
            humanized_note = HumanizedNote(
                pitch=note.get('pitch', 48),
                start_time=start_time,
                duration=actual_duration,
                velocity=final_velocity,
                articulation='legato',
                timing_offset=timing_variation,
                end_time=start_time + actual_duration,
                is_breath_point=False,
                phrase_position=0.5,
            )
            
            humanized.append(humanized_note)
        
        return humanized
    
    def _calculate_phrase_rubato(
        self,
        phrase_position: str,
        tempo: int,
        profile: dict,
    ) -> float:
        """
        Calculate rubato offset for a phrase.
        Returns offset in beats (positive = rushing, negative = holding back).
        """
        
        beat_in_seconds = 60.0 / tempo
        rubato_amount = profile['rubato_amount'] * beat_in_seconds
        
        if phrase_position == 'start':
            # Starting phrases often hold slightly
            return -rubato_amount * 0.5
        elif phrase_position == 'climax':
            # Climax may rush slightly
            return rubato_amount * 0.3
        elif phrase_position == 'end':
            # End phrases often slow down
            return -rubato_amount * 1.5
        else:
            # Middle phrases might rush slightly
            return rubato_amount * 0.2
    
    def _get_velocity_curve(self, shape: str, intensity: float) -> callable:
        """
        Get velocity curve function for phrase shape.
        """
        
        base_curve = VELOCITY_CURVES.get(shape, VELOCITY_CURVES['natural_phrase'])
        
        def scaled_curve(x: float) -> int:
            base_velocity = base_curve(x)
            # Scale by intensity
            if intensity > 0.6:
                # Intense phrases have wider dynamic range
                return int(base_velocity * 0.8 + intensity * 30)
            else:
                return int(base_velocity * 0.9 + intensity * 20)
        
        return scaled_curve
    
    def _find_breath_points(self, notes: list[dict]) -> list[int]:
        """
        Find natural breath points in a phrase.
        These are places where a singer/instrumentalist would naturally breathe.
        """
        
        breath_points = []
        
        if len(notes) < 4:
            return breath_points
        
        # Look for:
        # - Longer notes followed by shorter
        # - Large intervals
        # - Ends of four-bar segments
        
        beats = [n.get('beat', i) for i, n in enumerate(notes)]
        durations = [n.get('duration', 1.0) for n in notes]
        pitches = [n.get('pitch', 60) for n in notes]
        
        for i in range(1, len(notes) - 1):
            # Longer note followed by shorter
            if durations[i] > durations[i+1] * 1.5:
                breath_points.append(i + 1)
                continue
            
            # Large interval
            if abs(pitches[i] - pitches[i+1]) > 7:  # More than fifth
                if random.random() < 0.4:
                    breath_points.append(i + 1)
                continue
            
            # Every 4 beats approximately
            if beats[i] > 0 and beats[i] % 4 < 0.5:
                if random.random() < 0.3:
                    breath_points.append(i)
        
        return list(set(breath_points))[:2]  # Max 2 breath points per phrase
    
    def _find_phrase_boundaries(self, notes: list[dict]) -> list[int]:
        """
        Find where phrases naturally break.
        """
        
        if len(notes) < 4:
            return [len(notes)]
        
        boundaries = []
        beats = [n.get('beat', i) for i, n in enumerate(notes)]
        durations = [n.get('duration', 1.0) for n in notes]
        
        total_duration = beats[-1] + durations[-1] if beats else 4
        
        # Phrases typically 4-8 beats
        # Find natural break points
        for i, (beat, dur) in enumerate(zip(beats, durations)):
            # Check if this is a natural phrase end
            # (at 4-beat boundary and followed by longer gap)
            if i < len(beats) - 1:
                next_beat = beats[i + 1]
                gap = next_beat - (beat + dur)
                
                # Natural break: on beat boundary with gap, relative to total duration
                gap_ratio = gap / max(total_duration, 1.0)
                if (beat + dur) % 4 < 0.5 and gap_ratio > 0.02:
                    boundaries.append(i + 1)
        
        # Ensure we have reasonable phrase lengths
        if boundaries:
            # Validate: phrases should be 4-16 beats
            valid_boundaries = []
            last_bound = 0
            for b in boundaries:
                if b - last_bound >= 2 and beats[min(b, len(beats)-1)] - beats[last_bound] <= 16:
                    valid_boundaries.append(b)
                    last_bound = b
            return valid_boundaries
        
        # Default: split into 4-beat phrases
        return [i for i in range(4, len(notes), 4) if i < len(notes)]


def apply_humanization_to_composition(
    phrases: list[dict],
    style: str = 'classical',
    tempo: int = 90,
) -> list[dict]:
    """
    Apply humanization to an entire composition.
    
    Args:
        phrases: List of phrase dictionaries from composer
        style: Musical style
        tempo: BPM
    
    Returns:
        Humanized phrases with timing and velocity adjustments
    """
    
    humanizer = MIDIHumanizer()
    humanized_phrases = []
    
    # Determine phrase positions
    total_phrases = len(phrases)
    
    for phrase_idx, phrase in enumerate(phrases):
        # Position in composition
        if phrase_idx < total_phrases * 0.25:
            phrase_position = 'start'
        elif phrase_idx > total_phrases * 0.75:
            phrase_position = 'end'
        elif total_phrases * 0.4 <= phrase_idx < total_phrases * 0.6:
            phrase_position = 'climax'
        else:
            phrase_position = 'middle'
        
        # Humanize melody
        melody_notes = phrase.get('melody_notes', [])
        if melody_notes:
            humanized_melody = humanizer.humanize_melody_line(
                melody_notes,
                style=style,
                tempo=tempo,
            )
        else:
            humanized_melody = []
        
        # Humanize harmony - preserve chord structure (don't convert to pitch events)
        harmony_notes = phrase.get('harmony_notes', [])
        if harmony_notes:
            humanized_harmony = []
            for idx, h in enumerate(harmony_notes):
                humanized_h = dict(h)
                humanized_h['beat'] = max(0.0, float(h.get('beat', idx)) + random.gauss(0, 0.02))
                humanized_h['duration'] = max(0.5, float(h.get('duration', 4.0)))
                humanized_h['velocity'] = max(20, min(80, int(h.get('velocity', 48)) + random.randint(-3, 3)))
                humanized_harmony.append(humanized_h)
        else:
            humanized_harmony = []
        
        # Humanize bass
        bass_notes = phrase.get('bass_notes', [])
        if bass_notes:
            humanized_bass = humanizer.humanize_bass_line(
                bass_notes,
                style=style,
                tempo=tempo,
            )
        else:
            humanized_bass = []
        
        beat_in_seconds = 60.0 / tempo
        
        # Convert back to dict format (convert from seconds to beats)
        # Ensure all values are valid (no negative beats or non-positive durations)
        humanized_phrase = {
            'melody_notes': [
                {
                    'pitch': max(0, min(127, n.pitch)),  # Clamp pitch to valid MIDI range
                    'duration': max(0.1, n.duration / beat_in_seconds),  # Ensure positive duration
                    'beat': max(0.0, n.start_time / beat_in_seconds),   # No negative beats
                    'velocity': max(1, min(127, n.velocity)),  # Valid velocity
                    'articulation': n.articulation,
                    'timing_offset': n.timing_offset,
                    'is_breath_point': n.is_breath_point,
                }
                for n in humanized_melody
            ],
            'harmony_notes': humanized_harmony,
            'bass_notes': [
                {
                    'pitch': max(0, min(127, n.pitch)),
                    'duration': max(0.1, n.duration / beat_in_seconds),
                    'beat': max(0.0, n.start_time / beat_in_seconds),
                    'velocity': max(1, min(127, n.velocity)),
                    'articulation': n.articulation,
                }
                for n in humanized_bass
            ],
            'dynamics': phrase.get('dynamics', 'mf'),
            'duration': phrase.get('duration', 4.0),
            'section': phrase.get('section', 'unknown'),
            'phrase_position': phrase_position,
        }
        
        humanized_phrases.append(humanized_phrase)
    
    logger.info(f"[Humanizer] Humanized {len(humanized_phrases)} phrases for {style} style at {tempo} BPM")
    
    return humanized_phrases


def get_humanizer() -> MIDIHumanizer:
    """Get humanizer singleton."""
    global _humanizer
    if '_humanizer' not in globals():
        _humanizer = MIDIHumanizer()
    return _humanizer
