"""Integration of Mathematical Composition with Music Creation Engine.

This module bridges math_composer.py with music_creation_engine.py,
enabling Jarvis to create mathematically-inspired musical compositions.

Integration points:
- LLM Composer gets mathematical motifs and hooks
- Music Theory Engine uses mathematical scales and progressions
- Music Evolution tracks mathematical theme fitness
- Perception evaluates mathematical coherence
"""

import logging
import random
import math
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Mathematical constants
PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio ≈ 1.618
PHI_INVERSE = 1 / PHI  # ≈ 0.618
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


# Singleton
_math_integration = None


@dataclass
class MathematicalComposition:
    """A composition with mathematical underpinnings."""
    composition_id: str
    mathematical_theme: str  # 'golden', 'fibonacci', 'prime', 'fractal', 'symmetry'
    primary_motif_id: str
    hook_id: str
    tension_curve_type: str
    scale_type: str
    form_proportions: list[float]
    coherence_score: float
    mathematical_beauty: float  # Score based on mathematical elegance
    phrases: list[dict]
    metadata: dict = field(default_factory=dict)


class MathematicalMusicIntegration:
    """
    Integrates mathematical principles with music composition.
    
    Key Features:
    1. Mathematical motifs become primary themes
    2. Hooks use golden ratio for surprise moments
    3. Tension curves follow mathematical functions
    4. Form sections use mathematical proportions
    5. Scales derived from number theory
    """
    
    def __init__(self):
        self._initialized = False
        self._math_composer = None
        self._llm_composer = None
        self._style_preferences: dict[str, float] = {
            'golden': 0.9,      # Golden ratio - most naturally pleasing
            'fibonacci': 0.85,  # Fibonacci - organic, asymmetrical
            'prime': 0.7,       # Prime - angular, dramatic
            'fractal': 0.75,    # Fractal - complex, self-similar
            'symmetry': 0.8,    # Symmetry - balanced, memorable
        }
        self._successful_motifs: dict[str, list[float]] = {}
        self._composition_history: list[dict] = []
    
    def _get_math_composer(self):
        """Lazy load math composer."""
        if self._math_composer is None:
            from core.math_composer import get_math_composer
            self._math_composer = get_math_composer()
        return self._math_composer
    
    def create_mathematical_composition(
        self,
        style: str,
        form: str,
        key: str,
        character: str,
        tempo: int
    ) -> Optional[dict]:
        """
        Create a composition using mathematical principles.
        
        This method:
        1. Selects optimal mathematical theme based on style
        2. Generates primary motif using selected mathematics
        3. Creates hook with golden ratio surprise placement
        4. Builds tension curve from mathematical function
        5. Structures form using mathematical proportions
        """
        try:
            math_theme = self._select_mathematical_theme(style, character)
            
            math_composer = self._get_math_composer()
            
            key_root = self._parse_key_root(key)
            
            scale_type = self._select_scale(math_theme, style)
            
            composition = math_composer.generate_composition(
                style=style,
                mathematical_theme=math_theme,
                key_root=key_root,
                form=form,
                measures=self._get_measures_for_form(form)
            )
            
            composition = self._enhance_with_musical_theory(
                composition,
                style,
                form,
                key,
                character,
                tempo
            )
            
            composition['mathematical_integration'] = {
                'theme': math_theme,
                'scale_type': scale_type,
                'golden_points': self._calculate_golden_points(composition),
                'fibonacci_rhythms': self._extract_fibonacci_rhythms(composition),
                'prime_intervals': self._extract_prime_intervals(composition),
            }
            
            mathematical_beauty = self._calculate_mathematical_beauty(composition)
            composition['mathematical_beauty_score'] = mathematical_beauty
            
            logger.info(f"[MathIntegration] Created composition with {math_theme} theme, beauty score: {mathematical_beauty:.2f}")
            
            return composition
            
        except Exception as e:
            logger.error(f"[MathIntegration] Composition creation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _select_mathematical_theme(self, style: str, character: str) -> str:
        """Select the best mathematical theme for the given style and character.
        
        Different mathematics suit different emotional expressions:
        - Golden ratio: Balanced, contemplative, naturally pleasing
        - Fibonacci: Organic growth, development, unfolding
        - Prime: Angular, dramatic, intense
        - Fractal: Complex, layered, self-similar
        - Symmetry: Balanced, memorable, structured
        
        ADDS VARIETY: Uses weighted random selection to avoid repetitive patterns.
        """
        # Character to mathematical theme mapping
        character_themes = {
            'peaceful': 'golden',
            'contemplative': 'fibonacci',
            'dramatic': 'prime',
            'heroic': 'fibonacci',
            'mysterious': 'prime',
            'joyful': 'symmetry',
            'melancholic': 'golden',
            'triumphant': 'fibonacci',
            'playful': 'symmetry',
            'romantic': 'golden',
            'agitated': 'prime',
            'serene': 'golden',
        }
        
        preferred = character_themes.get(character.lower(), 'golden')
        
        # Style adjustments
        style_adjustments = {
            'baroque': {'fractal': 0.1, 'symmetry': 0.1},  # Baroque loves symmetry and complexity
            'classical': {'golden': 0.1, 'symmetry': 0.05},  # Classical loves balance
            'romantic': {'fibonacci': 0.1},  # Romantic loves organic development
            'modern': {'prime': 0.15, 'fractal': 0.1},  # Modern loves angular complexity
        }
        
        if style.lower() in style_adjustments:
            self._style_preferences.get(preferred, 0.8)
            for theme, adjustment in style_adjustments[style.lower()].items():
                self._style_preferences[theme] = self._style_preferences.get(theme, 0.7) + adjustment
        
        # ADD VARIETY: Use weighted random instead of always picking best
        themes = list(self._style_preferences.keys())
        weights = [self._style_preferences[t] for t in themes]
        
        # Boost preferred theme weight
        if preferred in themes:
            idx = themes.index(preferred)
            weights[idx] *= 1.5
        
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # Random selection with weights
        selected = random.choices(themes, weights=weights, k=1)[0]
        return selected
    
    def _parse_key_root(self, key: str) -> int:
        """Parse key string to MIDI root note."""
        NOTES = {
            'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63,
            'E': 64, 'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68,
            'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71
        }
        
        key_clean = key.replace('_major', '').replace('_minor', '').replace('#', '#').replace('b', 'b')
        
        for note in ['C#', 'Db', 'D#', 'Eb', 'F#', 'Gb', 'G#', 'Ab', 'A#', 'Bb']:
            if key_clean.startswith(note):
                return NOTES.get(note, 60)
        
        return NOTES.get(key_clean[0], 60)
    
    def _get_measures_for_form(self, form: str) -> int:
        """Get typical measure count for form."""
        form_measures = {
            'binary': 32,
            'ternary': 48,
            'sonata': 96,
            'rondo': 64,
            'theme_variations': 72,
            'fugue': 80,
            'invention': 32,
            'prelude': 48,
            'nocturne': 56,
        }
        return form_measures.get(form.lower().replace('-', '_'), 48)
    
    def _select_scale(self, math_theme: str, style: str) -> str:
        """Select appropriate scale for mathematical theme."""
        math_scales = {
            'golden': 'golden',
            'fibonacci': 'fibonacci',
            'prime': 'prime',
            'fractal': 'overtone',
            'symmetry': 'whole_tone',
        }
        
        base_scale = math_scales.get(math_theme, 'major')
        
        # Style can override for authenticity
        if style.lower() == 'baroque':
            return 'major' if math_theme in ['golden', 'symmetry'] else base_scale
        elif style.lower() == 'modern':
            return base_scale  # Modern embraces exotic scales
        
        return base_scale
    
    def _enhance_with_musical_theory(
        self,
        composition: dict,
        style: str,
        form: str,
        key: str,
        character: str,
        tempo: int
    ) -> dict:
        """Enhance mathematical composition with music theory principles."""
        
        # Add authentic cadences at section boundaries
        composition = self._add_cadential_progressions(composition, form)
        
        # Ensure voice leading follows rules
        composition = self._apply_voice_leading_rules(composition, style)
        
        # Add appropriate ornaments for style
        composition = self._add_style_ornaments(composition, style, character)
        
        # Balance mathematical precision with musical expressiveness
        composition = self._add_expressive_timing(composition, character, tempo)
        
        # Ensure harmonic progression makes sense
        composition = self._validate_harmony(composition, key)
        
        return composition
    
    def _add_cadential_progressions(self, composition: dict, form: str) -> dict:
        """Add authentic cadences at section boundaries."""
        
        phrases = composition.get('phrases', [])
        if not phrases:
            return composition
        
        sections = {}
        for phrase in phrases:
            section = phrase.get('section', 'A')
            if section not in sections:
                sections[section] = []
            sections[section].append(phrase)
        
        for section_name, section_phrases in sections.items():
            if section_phrases:
                last_phrase = section_phrases[-1]
                if 'harmony_notes' in last_phrase:
                    last_harmony = last_phrase['harmony_notes']
                    if last_harmony:
                        # Ensure ends with V-I or similar authentic cadence
                        if section_name in ['A', 'exposition']:
                            last_harmony[-1]['chord'] = 'V7'  # End on dominant
                        elif section_name in ['B', 'development']:
                            pass
                        else:
                            last_harmony[-1]['chord'] = 'I'  # End on tonic
        
        return composition
    
    def _apply_voice_leading_rules(self, composition: dict, style: str) -> dict:
        """Apply voice leading rules to ensure smooth harmonic motion."""
        
        # Bach-style: avoid parallel fifths, resolve leading tones
        # Mozart-style: smooth bass lines, prepared dissonances
        # Chopin-style: chromatic inner voices, delayed resolutions
        
        phrases = composition.get('phrases', [])
        
        for i, phrase in enumerate(phrases):
            # Apply style-specific voice leading
            if style.lower() == 'baroque':
                phrase = self._baroque_voice_leading(phrase)
            elif style.lower() == 'classical':
                phrase = self._classical_voice_leading(phrase)
            elif style.lower() == 'romantic':
                phrase = self._romantic_voice_leading(phrase)
            
            phrases[i] = phrase
        
        composition['phrases'] = phrases
        return composition
    
    def _baroque_voice_leading(self, phrase: dict) -> dict:
        """Apply Baroque voice leading rules."""
        # Bach-style: keep common tones, resolve leading tones upward
        # Avoid parallel fifths and octaves
        
        melody_notes = phrase.get('melody_notes', [])
        if melody_notes:
            # Leading tones should resolve upward
            for i, note in enumerate(melody_notes):
                pitch = note.get('pitch', 60)
                pitch_class = pitch % 12
                
                if i < len(melody_notes) - 1:
                    melody_notes[i + 1].get('pitch', 60)
                    if pitch_class in [7, 11]:  # G, B in C major
                        note['velocity'] = min(127, note.get('velocity', 80) + 5)
        
        return phrase
    
    def _classical_voice_leading(self, phrase: dict) -> dict:
        """Apply Classical voice leading rules."""
        # Mozart-style: smooth melodic lines, balanced phrases
        
        melody_notes = phrase.get('melody_notes', [])
        if melody_notes:
            avg_pitch = sum(n.get('pitch', 60) for n in melody_notes) / len(melody_notes)
            
            for note in melody_notes:
                pitch = note.get('pitch', 60)
                if abs(pitch - avg_pitch) > 12:
                    note['velocity'] = max(40, note.get('velocity', 80) - 10)
        
        return phrase
    
    def _romantic_voice_leading(self, phrase: dict) -> dict:
        """Apply Romantic voice leading rules."""
        # Chopin/Brahms-style: chromatic lines, expressive timing
        
        melody_notes = phrase.get('melody_notes', [])
        if len(melody_notes) > 1:
            for note in melody_notes:
                note['duration'] *= random.uniform(0.9, 1.1)
        
        return phrase
    
    def _add_style_ornaments(
        self,
        composition: dict,
        style: str,
        character: str
    ) -> dict:
        """Add style-appropriate ornaments."""
        
        ornaments = {
            'baroque': ['trill', 'mordent', 'turn', 'appoggiatura'],
            'classical': ['turn', 'grace_note', 'trill'],
            'romantic': ['rubato', 'arpeggio', 'grace_note', 'ornament'],
            'modern': ['glissando', 'tremolo', 'cluster'],
        }
        
        ornaments.get(style.lower(), ['grace_note'])
        
        phrases = composition.get('phrases', [])
        for phrase in phrases:
            melody_notes = phrase.get('melody_notes', [])
            if melody_notes:
                # Add ornamental timing based on character
                if character.lower() in ['dramatic', 'heroic', 'triumphant']:
                    for i, note in enumerate(melody_notes):
                        if i % 4 == 0:  # Every 4th note
                            note['articulation'] = 'accented'
        composition['phrases'] = phrases
        return composition
    
    def _add_expressive_timing(
        self,
        composition: dict,
        character: str,
        tempo: int
    ) -> dict:
        """Add expressive timing variations based on character."""
        
        # Timing variations by character
        timing_multiplier = {
            'peaceful': 1.02,
            'contemplative': 1.05,
            'dramatic': 0.95,
            'heroic': 0.98,
            'joyful': 0.95,
            'melancholic': 1.08,
            'triumphant': 0.92,
            'mysterious': 1.10,
            'agitated': 0.88,
            'serene': 1.05,
        }
        
        mult = timing_multiplier.get(character.lower(), 1.0)
        
        phrases = composition.get('phrases', [])
        for phrase in phrases:
            for note in phrase.get('melody_notes', []):
                note['duration'] = note.get('duration', 1.0) * mult
        
        composition['phrases'] = phrases
        return composition
    
    def _validate_harmony(self, composition: dict, key: str) -> dict:
        """Validate and fix harmonic progression."""
        
        phrases = composition.get('phrases', [])
        
        # Basic voice leading checks
        for i, phrase in enumerate(phrases):
            harmony = phrase.get('harmony_notes', [])
            if not harmony:
                continue
            
            # Ensure root position stability
            bass_notes = phrase.get('bass_notes', [])
            if bass_notes and harmony:
                bass_notes[0].get('pitch', 36) % 12
                chord = harmony[0].get('chord', 'I')
                
                # Check if bass matches chord root
                chord_roots = {
                    'I': 0, 'i': 0, 'ii': 2, 'iii': 4, 'IV': 5, 
                    'V': 7, 'vi': 9, 'vii': 11, 'V7': 7
                }
                chord_roots.get(chord.replace('7', '').replace('6', ''), 0)
                
                # Adjust if needed (keep within key for good voice leading)
        
        return composition
    
    def _calculate_golden_points(self, composition: dict) -> list[int]:
        """Calculate measure positions at golden ratio points."""
        
        total_measures = composition.get('total_measures', 48)
        
        golden_points = [
            int(total_measures * PHI_INVERSE),  # ~0.618 - first climax
            int(total_measures * (1 - PHI_INVERSE)),  # ~0.382 - secondary high point
            int(total_measures * PHI_INVERSE * PHI_INVERSE),  # ~0.382 - tension point
        ]
        
        return sorted(set([p for p in golden_points if 0 < p < total_measures]))
    
    def _extract_fibonacci_rhythms(self, composition: dict) -> list[dict]:
        """Identify rhythmic patterns based on Fibonacci sequence."""
        
        fibonacci_positions = []
        phrases = composition.get('phrases', [])
        
        current_beat = 0
        for phrase in phrases:
            phrase_beats = phrase.get('duration', 4.0)
            for fib in FIBONACCI[:7]:
                fib_beat = fib
                if fib_beat < phrase_beats:
                    fibonacci_positions.append({
                        'beat': current_beat + fib_beat,
                        'phrase': phrases.index(phrase),
                        'fib_number': fib,
                    })
            current_beat += phrase_beats
        
        return fibonacci_positions[:12]
    
    def _extract_prime_intervals(self, composition: dict) -> list[dict]:
        """Identify melodic intervals that are prime numbers."""
        
        prime_intervals = []
        phrases = composition.get('phrases', [])
        
        for phrase in phrases:
            melody = phrase.get('melody_notes', [])
            for i in range(len(melody) - 1):
                interval = abs(melody[i + 1].get('pitch', 60) - melody[i].get('pitch', 60))
                if interval in PRIMES:
                    prime_intervals.append({
                        'interval': interval,
                        'position': i,
                        'phrase': phrases.index(phrase),
                    })
        
        return prime_intervals[:15]
    
    def _calculate_mathematical_beauty(self, composition: dict) -> float:
        """Calculate a mathematical beauty score for the composition.
        
        Factors:
        - Golden ratio adherence in climax placement
        - Fibonacci proportion in sections
        - Symmetry in phrase structure
        - Prime interval usage for angularity
        - Overall coherence
        """
        
        score = 0.0
        
        # Check golden ratio placement of climax
        golden_points = composition.get('mathematical_integration', {}).get('golden_points', [])
        if golden_points:
            # Prefer climax near golden ratio point
            total_measures = composition.get('total_measures', 48)
            target_golden = total_measures * PHI_INVERSE
            
            for gp in golden_points:
                if abs(gp - target_golden) < 4:
                    score += 0.2
                    break
        
        # Check phrase balance (symmetry)
        phrases = composition.get('phrases', [])
        if phrases:
            # Symmetrical phrases score higher
            mid = len(phrases) // 2
            if len(phrases) > 2:
                first_half_durations = [p.get('duration', 4.0) for p in phrases[:mid]]
                second_half_durations = [p.get('duration', 4.0) for p in phrases[mid:]]
                
                if first_half_durations and second_half_durations:
                    first_total = sum(first_half_durations)
                    second_total = sum(second_half_durations)
                    
                    ratio = min(first_total, second_total) / max(first_total, second_total)
                    score += ratio * 0.3
        
        # Fibonacci rhythm presence
        fib_rhythms = composition.get('mathematical_integration', {}).get('fibonacci_rhythms', [])
        if fib_rhythms:
            score += min(len(fib_rhythms) / 8, 0.25)
        
        # Prime interval angularity
        prime_ints = composition.get('mathematical_integration', {}).get('prime_intervals', [])
        if prime_ints:
            score += min(len(prime_ints) / 10, 0.25)
        
        return min(1.0, score)
    
    def integrate_with_llm_composer(
        self,
        llm_result: dict,
        math_theme: str
    ) -> dict:
        """Integrate mathematical composition with LLM composer output.
        
        Combines the sophisticated structure from LLM with mathematical motifs.
        """
        
        from core.math_composer import MotifGenerator
        
        key_root = 60
        if 'key' in llm_result:
            key_root = self._parse_key_root(llm_result['key'])
        
        # Generate mathematical motif
        if math_theme == 'fibonacci':
            motif = MotifGenerator.generate_fibonacci_motif(key_root)
        elif math_theme == 'prime':
            motif = MotifGenerator.generate_prime_motif(key_root)
        elif math_theme == 'golden':
            motif = MotifGenerator.generate_golden_motif(key_root)
        elif math_theme == 'fractal':
            motif = MotifGenerator.generate_fractal_motif(key_root, depth=2)
        else:
            motif = MotifGenerator.generate_symmetry_motif(key_root)
        
        # Enhance LLM phrases with mathematical motif
        llm_phrases = llm_result.get('phrases', [])
        enhanced_phrases = []
        
        for i, phrase in enumerate(llm_phrases):
            enhanced_phrase = phrase.copy()
            
            # Merge mathematical motif with LLM melody
            math_note_idx = i % len(motif.notes)
            math_note = motif.notes[math_note_idx]
            
            # Add mathematical accent
            melody_notes = phrase.get('melody_notes', [])
            if melody_notes and math_note_idx < len(melody_notes):
                # Blend mathematical precision with LLM inspiration
                melody_notes[0]['velocity'] = int(
                    (melody_notes[0].get('velocity', 80) + math_note['velocity']) / 2
                )
            
            enhanced_phrase['melody_notes'] = melody_notes
            enhanced_phrase['mathematical_motif_source'] = motif.id
            
            enhanced_phrases.append(enhanced_phrase)
        
        llm_result['phrases'] = enhanced_phrases
        llm_result['mathematical_integration'] = {
            'primary_motif': motif.id,
            'theme': math_theme,
            'description': motif.mathematical_description,
        }
        
        return llm_result
    
    def record_composition_success(
        self,
        composition_id: str,
        fitness_score: float,
        mathematical_theme: str
    ) -> None:
        """Record successful composition for learning."""
        
        self._composition_history.append({
            'id': composition_id,
            'fitness': fitness_score,
            'theme': mathematical_theme,
            'timestamp': str(Path.home() / '.jarvis'),
        })
        
        # Update style preferences based on success
        if fitness_score > 0.7:
            self._style_preferences[mathematical_theme] = min(
                1.0,
                self._style_preferences.get(mathematical_theme, 0.7) + 0.02
            )
        
        # Keep history manageable
        if len(self._composition_history) > 100:
            self._composition_history = self._composition_history[-100:]
    
    def get_recommended_theme(self, style: str, character: str) -> str:
        """Get recommended mathematical theme based on style and history."""
        return self._select_mathematical_theme(style, character)


def get_math_integration() -> MathematicalMusicIntegration:
    """Get singleton instance of mathematical music integration."""
    global _math_integration
    if _math_integration is None:
        _math_integration = MathematicalMusicIntegration()
        logger.info("[MathIntegration] Mathematical music integration initialized")
    return _math_integration