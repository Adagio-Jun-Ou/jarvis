"""Mathematical Music Composition - Creating Music Through Mathematical Beauty.

This module transforms mathematical concepts into musical elements:
- Golden Ratio (φ) for phrase proportions and climaxes
- Fibonacci sequences for rhythm and thematic development  
- Prime numbers for syncopation and irregular groupings
- Fractals for self-similar themes at different scales
- Symmetry groups for palindromic forms and mirror structures
- Chaos theory for controlled randomness
- Number theory for rhythmic and pitch patterns

Created compositions exhibit:
- Mathematical elegance: Pleasing proportions from natural constants
- Structural integrity: Coherent development through logical sequences
- Surprising variety: Emergent complexity from simple rules
- Emotional depth: Math-derived tension/release curves
"""
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Mathematical Constants
PHI = (1 + math.sqrt(5)) / 2  # Golden Ratio ≈ 1.618
PHI_INVERSE = 1 / PHI  # ≈ 0.618
E = math.e  # Euler's number ≈ 2.718
PI = math.pi  # ≈ 3.14159
SQRT2 = math.sqrt(2)  # ≈ 1.414

# Precompute Fibonacci sequence
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Precompute prime numbers
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


@dataclass
class MathematicalMotif:
    """A motif derived from mathematical principles."""
    id: str
    generator: str  # 'fibonacci', 'prime', 'golden', 'fractal', 'symmetry'
    notes: list[dict]
    intervals: list[int]
    rhythm: list[float]
    duration_pattern: list[float]
    mathematical_description: str
    parent_motif: Optional[str] = None
    transformations: list[str] = field(default_factory=list)


@dataclass
class Hook:
    """A memorable hook derived from mathematical principles."""
    id: str
    motif: MathematicalMotif
    repetition_pattern: list[int]
    surprise_position: float  # Fraction through melody (golden ratio point)
    catchiness_score: float
    variation_techniques: list[str]


@dataclass
class TensionCurve:
    """Tension/release curve derived from mathematical functions."""
    points: list[tuple[float, float]]  # (time, tension)
    mathematical_basis: str
    climax_time: float
    resolution_type: str


class MathematicalScales:
    """Generate scales from mathematical principles."""
    
    SCALES = {}
    
    @classmethod
    def fibonacci_scale(cls, root: int = 60, octaves: int = 2) -> list[int]:
        """Generate scale using Fibonacci intervals mod 12.
        
        The Fibonacci sequence mod 12 produces: 1, 1, 2, 3, 5, 8, 1, 9, 10, 7, 5, 0...
        This creates an exotic, mathematically-derived scale.
        """
        pitches = [root]
        for i, fib in enumerate(FIBONACCI[:8]):
            interval = fib % 12
            next_pitch = root + interval + (i // 7) * 12
            if octaves * 12 + root > next_pitch >= root:
                if next_pitch not in pitches:
                    pitches.append(next_pitch)
        return sorted(set(pitches))
    
    @classmethod
    def prime_scale(cls, root: int = 60, octaves: int = 2) -> list[int]:
        """Generate scale using prime numbers mod 12.
        
        Primes mod 12: 2, 3, 5, 7, 11 → distinctive, angular sound.
        Used by composers like Messiaen in modes of limited transposition.
        """
        pitches = [root]
        for p in PRIMES[:8]:
            interval = p % 12
            pitch = root + interval
            if octaves * 12 + root > pitch >= root:
                if pitch not in pitches:
                    pitches.append(pitch)
        return sorted(pitches)
    
    @classmethod
    def overtone_scale(cls, root: int = 60, partials: int = 12) -> list[int]:
        """Generate scale from harmonic series.
        
        Frequencies of harmonic series correspond to pitch relationships.
        Creates consonant, natural-sounding scales.
        """
        pitches = [root]
        for n in range(2, partials + 1):
            pitch_class = int(12 * math.log2(n)) % 12
            pitch = root + pitch_class
            if 0 <= pitch_class < 12:
                if pitch not in pitches:
                    pitches.append(pitch)
        return sorted(set(pitches))[:12]
    
    @classmethod
    def golden_scale(cls, root: int = 60, notes: int = 7) -> list[int]:
        """Generate scale using golden ratio intervals.
        
        Each interval is φ mod 12 ≈ 1.618 ≈ 1.5 (minor 3rd) semitones.
        Creates balanced, asymmetric scale.
        """
        pitches = [root]
        phi_interval = PHI * 2  # ≈ 3.236, use floor(3) or ceil(4)
        current = root
        for i in range(notes - 1):
            current += int(phi_interval * (i + 1) / 2)
            pitch_in_octave = root + (current - root) % 12
            if pitch_in_octave not in pitches:
                pitches.append(pitch_in_octave)
        return sorted(pitches)


class MotifGenerator:
    """Generate musically meaningful motifs from mathematical principles."""
    
    @staticmethod
    def generate_fibonacci_motif(
        key_root: int = 60,
        scale: list[int] = None,
        mood: str = "contemplative"
    ) -> MathematicalMotif:
        """Generate motif using Fibonacci sequence for rhythm and intervals.
        
        Fibonacci provides natural asymmetry that sounds musical.
        """
        if scale is None:
            scale = [0, 2, 4, 5, 7, 9, 11]
        
        notes = []
        intervals = []
        
        # ADD VARIETY: Start at different Fibonacci positions
        start_idx = random.randint(0, 5)
        fib_rhythms = [FIBONACCI[(start_idx + i) % len(FIBONACCI)] for i in range(5)]
        rhythm_values = [f / 8.0 for f in fib_rhythms]
        
        # ADD VARIETY: Random starting pitch within range
        octave_offset = random.choice([0, 12, 24])
        key_root + octave_offset
        
        # ADD VARIETY: Different interval patterns
        random.choice([1, -1])
        
        for i, rhythm in enumerate(rhythm_values):
            fib_interval = FIBONACCI[(start_idx + i) % len(FIBONACCI)] % len(scale)
            scale_degree = scale[fib_interval % len(scale)]
            pitch = key_root + scale_degree + (i % 3) * 12 + octave_offset
            
            # ADD VARIETY: Dynamic velocity patterns
            velocity_patterns = [
                70 + int(30 * (1 - i / len(rhythm_values))),
                80 + int(20 * math.sin(i * math.pi / 3)),
                60 + int(40 * (i / len(rhythm_values))),
            ]
            velocity = velocity_patterns[i % len(velocity_patterns)]
            
            notes.append({
                'pitch': pitch,
                'duration': rhythm,
                'velocity': velocity,
                'beat': sum(rhythm_values[:i])
            })
            intervals.append(fib_interval)
        
        return MathematicalMotif(
            id=f"fib_motif_{random.randint(0, 9999):04d}",
            generator='fibonacci',
            notes=notes,
            intervals=intervals,
            rhythm=rhythm_values,
            duration_pattern=rhythm_values,
            mathematical_description=f"Fibonacci motif starting at F[{start_idx}]",
            transformations=[]
        )
    
    @staticmethod
    def generate_prime_motif(
        key_root: int = 60,
        scale: list[int] = None,
        mood: str = "angular"
    ) -> MathematicalMotif:
        """Generate motif using prime numbers for angular, striking sound.
        
        Prime numbers create irregular, memorable patterns.
        """
        if scale is None:
            scale = [0, 2, 4, 5, 7, 9, 11]
        
        notes = []
        intervals = []
        
        # ADD VARIETY: Random subset of primes
        start_prime_idx = random.randint(0, 5)
        primes_subset = PRIMES[start_prime_idx:start_prime_idx + 6]
        rhythm_values = [p / 7.0 for p in primes_subset]
        
        # ADD VARIETY: Different octave placements
        base_octave = random.choice([0, 1, 2])
        
        for i, rhythm in enumerate(rhythm_values):
            prime_idx = i % len(primes_subset)
            prime_val = primes_subset[prime_idx]
            scale_degree = scale[prime_val % len(scale)]
            pitch = key_root + scale_degree + base_octave * 12
            
            # ADD VARIETY: Different velocity curves
            vel_curves = [
                60 + int(40 * math.sin(i * math.pi / len(rhythm_values))),
                50 + int(50 * (1 - i / len(rhythm_values))),
                70 + int(30 * random.random()),
            ]
            velocity = vel_curves[i % len(vel_curves)]
            
            notes.append({
                'pitch': pitch,
                'duration': rhythm,
                'velocity': velocity,
                'beat': sum(rhythm_values[:i])
            })
            intervals.append(prime_val % 7)
        
        return MathematicalMotif(
            id=f"prime_motif_{random.randint(0, 9999):04d}",
            generator='prime',
            notes=notes,
            intervals=intervals,
            rhythm=rhythm_values,
            duration_pattern=rhythm_values,
            mathematical_description=f"Prime motif using primes[{start_prime_idx}:+6]",
            transformations=[]
        )
    
    @staticmethod
    def generate_golden_motif(
        key_root: int = 60,
        scale: list[int] = None,
        mood: str = "balanced"
    ) -> MathematicalMotif:
        """Generate motif with golden ratio proportions.
        
        Golden ratio creates naturally pleasing proportions.
        """
        if scale is None:
            scale = [0, 2, 4, 5, 7, 9, 11]
        
        # ADD VARIETY: Different numbers of notes (4-8)
        num_notes = random.randint(4, 8)
        
        # ADD VARIETY: Different phi power combinations
        phi_powers = [int((PHI ** ((i + random.randint(0, 2)) % 4)) * 2) % 7 for i in range(num_notes)]
        
        # ADD VARIETY: Shuffle intervals sometimes
        if random.random() < 0.3:
            random.shuffle(phi_powers)
        
        total_duration = 4.0
        golden_durations = []
        for i in range(num_notes):
            if i < num_notes * PHI_INVERSE:
                golden_durations.append(total_duration * PHI_INVERSE / (num_notes * PHI_INVERSE))
            else:
                golden_durations.append(total_duration * (1 - PHI_INVERSE) / (num_notes * (1 - PHI_INVERSE)))
        
        # ADD VARIETY: Different duration patterns
        if random.random() < 0.4:
            # Invert the duration pattern
            golden_durations = list(reversed(golden_durations))
        
        notes = []
        beat = 0
        base_octave = random.choice([0, 12, 24])
        
        for i, (interval, dur) in enumerate(zip(phi_powers, golden_durations)):
            scale_degree = scale[interval % len(scale)]
            pitch = key_root + scale_degree + base_octave + 12 * (1 + i // 3)
            
            climax_factor = 1 - abs(2 * i / num_notes - 1)
            velocity = 65 + int(35 * climax_factor)
            
            notes.append({
                'pitch': pitch,
                'duration': dur,
                'velocity': velocity,
                'beat': beat
            })
            beat += dur
        
        return MathematicalMotif(
            id=f"golden_motif_{random.randint(0, 9999):04d}",
            generator='golden',
            notes=notes,
            intervals=phi_powers,
            rhythm=golden_durations,
            duration_pattern=golden_durations,
            mathematical_description=f"Golden ratio motif with {num_notes} notes",
            transformations=[]
        )
    
    @staticmethod
    def generate_fractal_motif(
        key_root: int = 60,
        scale: list[int] = None,
        depth: int = 2
    ) -> MathematicalMotif:
        """Generate self-similar motif at multiple scales (fractal).
        
        Creates coherence through self-similarity at different time scales.
        """
        if scale is None:
            scale = [0, 2, 4, 5, 7, 9, 11]
        
        base_pattern = [
            scale[0] + key_root,
            scale[2] + key_root + 12,
            scale[4] + key_root + 12,
            scale[0] + key_root + 24,
        ]
        
        notes = []
        base_dur = 1.0
        
        def add_fractal_notes(pattern, current_beat, duration, level):
            nonlocal notes
            
            if level == 0:
                for pitch in pattern:
                    notes.append({
                        'pitch': pitch,
                        'duration': duration,
                        'velocity': 70 + level * 10,
                        'beat': current_beat
                    })
                    current_beat += duration
                return current_beat
            
            for i, pitch in enumerate(pattern):
                sub_pattern = [
                    scale[(scale.index((pitch - key_root) % 12) + j) % len(scale)] + key_root
                    for j in range(4)
                ]
                current_beat = add_fractal_notes(sub_pattern, current_beat, duration / 2, level - 1)
            
            return current_beat
        
        add_fractal_notes(base_pattern, 0, base_dur, depth)
        
        intervals = [n['pitch'] - key_root for n in notes]
        rhythm = [n['duration'] for n in notes]
        
        return MathematicalMotif(
            id=f"fractal_motif_{depth}_{hash(tuple(intervals[:8])) % 10000:04d}",
            generator='fractal',
            notes=notes,
            intervals=intervals,
            rhythm=rhythm,
            duration_pattern=rhythm,
            mathematical_description=f"Self-similar pattern at {depth} scales - same structure repeats",
            transformations=['self_similar', 'recursive']
        )
    
    @staticmethod
    def generate_symmetry_motif(
        key_root: int = 60,
        scale: list[int] = None,
        symmetry_type: str = "palindrome"
    ) -> MathematicalMotif:
        """Generate motif with symmetry (palindrome, mirror, rotation).
        
        Creates balanced, memorable structures through symmetry.
        """
        if scale is None:
            scale = [0, 2, 4, 5, 7, 9, 11]
        
        palindrome_half = [scale[0], scale[2], scale[4], scale[5]]
        full_pattern = palindrome_half + palindrome_half[::-1]
        
        notes = []
        for i, degree in enumerate(full_pattern):
            pitch = key_root + degree + 12
            duration = 0.5 if i in [0, 3, 4, 7] else 0.25
            velocity = 80 if i in [3, 4] else 70
            
            notes.append({
                'pitch': pitch,
                'duration': duration,
                'velocity': velocity,
                'beat': sum(n['duration'] for n in notes)
            })
        
        intervals = full_pattern
        
        return MathematicalMotif(
            id=f"symmetry_motif_{hash(tuple(intervals)) % 10000:04d}",
            generator='symmetry',
            notes=notes,
            intervals=intervals,
            rhythm=[n['duration'] for n in notes],
            duration_pattern=[n['duration'] for n in notes],
            mathematical_description="Palindromic pitch structure - first half mirror of second half",
            transformations=['palindrome', 'mirror']
        )


class HookCreator:
    """Create memorable hooks from mathematical principles."""
    
    @staticmethod
    def create_hook(
        motif: MathematicalMotif,
        key_root: int = 60,
        repetition_scheme: str = "golden"
    ) -> Hook:
        """Create a memorable hook from a motif.
        
        Hook design uses:
        - Repetition at golden ratio intervals
        - Surprise element at attention peak
        - Variation for interest
        """
        if repetition_scheme == "golden":
            repeats = [1, 2, 3, 2, 1]
            [i * PHI_INVERSE for i in range(5)]
        elif repetition_scheme == "fibonacci":
            repeats = [1, 1, 2, 3, 5]
            [FIBONACCI[i] / 13 for i in range(5)]
        else:
            repeats = [1, 2, 3, 2, 1]
            [i / 4 for i in range(5)]
        
        surprise_position = PHI_INVERSE
        
        variation_techniques = [
            "sequence_up",
            "sequence_down",
            "inversion",
            "augmentation",
            "diminution",
            "fragmentation",
            "ornamentation",
            "reharmonization"
        ]
        
        catchiness = 0.5
        if motif.generator in ['golden', 'fibonacci']:
            catchiness += 0.2
        if 'palindrome' in motif.transformations:
            catchiness += 0.1
        if len(motif.notes) <= 8:
            catchiness += 0.1
        catchiness = min(1.0, catchiness + random.uniform(-0.1, 0.1))
        
        return Hook(
            id=f"hook_{motif.id}",
            motif=motif,
            repetition_pattern=repeats,
            surprise_position=surprise_position,
            catchiness_score=catchiness,
            variation_techniques=variation_techniques
        )


class TensionCurveGenerator:
    """Generate tension/release curves from mathematical functions."""
    
    @staticmethod
    def exponential_tension(
        duration: float,
        peak_time: float = None,
        steepness: float = 2.0
    ) -> TensionCurve:
        """Exponential tension curve using e^x.
        
        Builds tension exponentially then resolves.
        """
        if peak_time is None:
            peak_time = duration * PHI_INVERSE * 0.8
        
        num_points = 20
        points = []
        
        for i in range(num_points):
            t = duration * i / (num_points - 1)
            
            if t < peak_time:
                normalized_t = t / peak_time
                tension = math.expm1(steepness * normalized_t) / math.expm1(steepness)
            else:
                normalized_t = (t - peak_time) / (duration - peak_time)
                tension = 1.0 / (1.0 + math.exp(5 * (normalized_t - 0.5)))
            
            points.append((t, tension))
        
        return TensionCurve(
            points=points,
            mathematical_basis=f"Exponential rise (e^{steepness}x) with logistic decay",
            climax_time=peak_time,
            resolution_type="exponential"
        )
    
    @staticmethod
    def sinusoidal_tension(
        duration: float,
        num_waves: int = 2,
        modulation_depth: float = 0.3
    ) -> TensionCurve:
        """Sinusoidal tension curve using sine waves.
        
        Creates waves of tension/release.
        """
        num_points = 20
        points = []
        
        for i in range(num_points):
            t = duration * i / (num_points - 1)
            
            envelope = math.sin(math.pi * t / duration)
            
            wave = envelope * 0.6 + modulation_depth * math.sin(2 * math.pi * num_waves * t / duration + math.pi/4)
            
            wave = max(0, min(1, (wave + 1) / 2)) * 0.8 + 0.1
            
            points.append((t, wave))
        
        return TensionCurve(
            points=points,
            mathematical_basis=f"Sine wave with {num_waves} tension cycles",
            climax_time=duration / (num_waves + 1),
            resolution_type="sinusoidal"
        )
    
    @staticmethod
    def fibonacci_tension(duration: float) -> TensionCurve:
        """Tension curve following Fibonacci ratios.
        
        Natural rise/fall following Fibonacci proportions.
        """
        fib_ratios = [FIBONACCI[i] / FIBONACCI[i+1] for i in range(1, 7)]
        
        num_points = 20
        points = []
        
        sum(FIBONACCI[1:7])
        
        for i in range(num_points):
            t = duration * i / (num_points - 1)
            
            position = t / duration
            
            tension = 0
            for j, ratio in enumerate(fib_ratios):
                wave = math.sin(math.pi * position / ratio)
                tension += wave * (PHI_INVERSE ** j)
            
            tension = (tension / len(fib_ratios) + 1) / 2
            tension = max(0.1, min(0.95, tension))
            
            points.append((t, tension))
        
        climax_time = duration * FIBONACCI[5] / sum(FIBONACCI[1:7])
        
        return TensionCurve(
            points=points,
            mathematical_basis="Fibonacci ratio superposition",
            climax_time=climax_time,
            resolution_type="fibonacci"
        )


class SectionStructureGenerator:
    """Generate section structures from mathematical proportions."""
    
    @staticmethod
    def golden_sectioning(total_measures: int, num_sections: int = 3) -> list[tuple[str, int, float]]:
        """Divide form using golden ratio for optimal balance.
        
        Golden section creates naturally pleasing proportions.
        """
        if num_sections == 3:
            section_a = total_measures * PHI_INVERSE
            section_b = total_measures * PHI_INVERSE * PHI_INVERSE
            section_c = total_measures - section_a - section_b
            
            return [
                ('A', int(section_a), PHI_INVERSE),
                ('B', int(section_b), PHI_INVERSE * PHI_INVERSE),
                ('A\'', int(section_c), 1 - PHI_INVERSE - PHI_INVERSE * PHI_INVERSE),
            ]
        elif num_sections == 5:
            base = PHI_INVERSE
            sizes = [
                base,
                base * PHI_INVERSE,
                base * PHI_INVERSE * PHI_INVERSE,
                base * PHI_INVERSE * PHI_INVERSE * PHI_INVERSE,
                1 - base * PHI_INVERSE**0 - base * PHI_INVERSE**1 - base * PHI_INVERSE**2 - base * PHI_INVERSE**3
            ]
            total = sum(sizes)
            sizes = [s / total for s in sizes]
            
            measure_sizes = [int(s * total_measures) for s in sizes]
            remaining = total_measures - sum(measure_sizes)
            measure_sizes[-1] += remaining
            
            return [
                (f'Section_{i+1}', measure_sizes[i], sizes[i])
                for i in range(num_sections)
            ]
        else:
            return SectionStructureGenerator.fibonacci_sectioning(total_measures, num_sections)
    
    @staticmethod
    def fibonacci_sectioning(total_measures: int, num_sections: int) -> list[tuple[str, int, float]]:
        """Divide form using Fibonacci proportions.
        
        Creates distinctive asymmetrical balance.
        """
        fib_slice = FIBONACCI[:num_sections]
        fib_sum = sum(fib_slice)
        proportions = [f / fib_sum for f in fib_slice]
        
        measure_sizes = [int(p * total_measures) for p in proportions]
        remaining = total_measures - sum(measure_sizes)
        if remaining > 0:
            golden_idx = int(len(measure_sizes) * PHI_INVERSE)
            measure_sizes[golden_idx] += remaining
        
        return [
            (f'Section_{i+1}', measure_sizes[i], proportions[i])
            for i in range(num_sections)
        ]
    
    @staticmethod
    def prime_sectioning(total_measures: int, num_sections: int) -> list[tuple[str, int, float]]:
        """Divide form using prime proportions.
        
        Creates angular, distinctive sections.
        """
        prime_slice = PRIMES[:num_sections]
        prime_sum = sum(prime_slice)
        proportions = [p / prime_sum for p in prime_slice]
        
        measure_sizes = [int(p * total_measures) for p in proportions]
        remaining = total_measures - sum(measure_sizes)
        if remaining > 0:
            measure_sizes[-1] += remaining
        
        return [
            (f'Section_{i+1}', measure_sizes[i], proportions[i])
            for i in range(num_sections)
        ]


class ScaleGenerator:
    """Generate exotic and innovative scales from mathematics."""
    
    EXOTIC_SCALES = {
        'fibonacci': {
            'intervals': [0, 1, 2, 3, 5, 8, 10],
            'description': 'Fibonacci sequence mod 12 - exotic, angular',
            'mood': 'mysterious, otherworldly'
        },
        'prime': {
            'intervals': [0, 2, 3, 5, 7, 11],
            'description': 'Prime numbers mod 12 - angular, striking',
            'mood': 'tense, dramatic'
        },
        'overtone': {
            'intervals': [0, 12, 19, 24, 28, 31],
            'description': 'Harmonic series intervals - natural, resonant',
            'mood': 'open, organum-like'
        },
        'golden': {
            'intervals': [0, 2, 3, 5, 7, 9, 10],
            'description': 'Golden ratio derived - balanced, asymmetric',
            'mood': 'balanced, mysterious'
        },
        'whole_tone': {
            'intervals': [0, 2, 4, 6, 8, 10],
            'description': 'Whole tone scale - dreamy, ambiguous',
            'mood': 'dreamy, floating'
        },
        'octatonic': {
            'intervals': [0, 2, 3, 5, 6, 8, 9, 11],
            'description': 'Alternating whole/half - exotic, symmetric',
            'mood': 'mysterious, symmetrical'
        },
        'pentatonic_golden': {
            'intervals': [0, 2, 4, 7, 9],
            'description': 'Pentatonic with golden proportion spacing',
            'mood': 'peaceful, balanced'
        },
        'messiaen_mode_2': {
            'intervals': [0, 1, 3, 4, 6, 7, 9, 10],
            'description': 'Messiaen mode of limited transposition',
            'mood': 'transcendent, mystical'
        },
    }
    
    @staticmethod
    def get_scale_intervals(scale_name: str) -> list[int]:
        """Get interval pattern for a named scale."""
        if scale_name in ScaleGenerator.EXOTIC_SCALES:
            return ScaleGenerator.EXOTIC_SCALES[scale_name]['intervals']
        
        base_scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'harmonic_minor': [0, 2, 3, 5, 7, 8, 11],
            'melodic_minor': [0, 2, 3, 5, 7, 9, 11],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'phrygian': [0, 1, 3, 5, 7, 8, 10],
            'lydian': [0, 2, 4, 6, 7, 9, 11],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10],
            'blues': [0, 3, 5, 6, 7, 10],
            'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
        return base_scales.get(scale_name, base_scales['major'])


class RhythmGenerator:
    """Generate rhythmic patterns from mathematical sequences."""
    
    @staticmethod
    def fibonacci_rhythm(num_beats: int = 8) -> list[float]:
        """Generate rhythm following Fibonacci durations.
        
        Creates organic, asymmetric rhythm.
        """
        rhythms = []
        total_beats = 0
        i = 0
        
        while total_beats < num_beats:
            beat_val = FIBONACCI[i % len(FIBONACCI)] / 8.0
            if total_beats + beat_val <= num_beats:
                rhythms.append(beat_val)
                total_beats += beat_val
            i += 1
            if i > 20:
                break
        
        remainder = num_beats - sum(rhythms)
        if remainder > 0.001:
            rhythms.append(remainder)
        
        return rhythms
    
    @staticmethod
    def polyrhythm(base_beats: int = 3, against_beats: int = 2, cycles: int = 3) -> list[tuple[int, int]]:
        """Generate polyrhythmic pattern.
        
        Creates complex interlocking rhythms.
        """
        pattern = []
        lcm = base_beats * against_beats // math.gcd(base_beats, against_beats)
        
        for cycle in range(cycles):
            for i in range(lcm):
                base_hit = (i * base_beats) % lcm == 0
                against_hit = (i * against_beats) % lcm == 0
                if base_hit or against_hit:
                    pattern.append((i, base_hit * 1 + against_hit * 2))
        
        return pattern
    
    @staticmethod
    def prime_rhythm(num_beats: int = 16) -> list[float]:
        """Generate rhythm with prime number subdivisions.
        
        Creates syncopated, unpredictable rhythm.
        """
        primes = PRIMES[:8]
        rhythms = []
        
        for p in primes:
            sub = num_beats / p
            if sub >= 0.5:
                rhythms.extend([sub] * 2)
        
        total = sum(rhythms)
        if total < num_beats:
            rhythms.append(num_beats - total)
        elif total > num_beats:
            ratio = num_beats / total
            rhythms = [r * ratio for r in rhythms]
        
        return rhythms[:12]


class MathCompositionEngine:
    """Main engine for mathematically-inspired composition."""
    
    def __init__(self):
        self.motif_generator = MotifGenerator()
        self.hook_creator = HookCreator()
        self.tension_generator = TensionCurveGenerator()
        self.section_generator = SectionStructureGenerator()
        self.scale_generator = ScaleGenerator()
        self.rhythm_generator = RhythmGenerator()
        self._motif_cache: dict[str, MathematicalMotif] = {}
    
    def generate_composition(
        self,
        style: str = "classical",
        mathematical_theme: str = "golden",
        key_root: int = 60,
        form: str = "sonata",
        measures: int = 48
    ) -> dict:
        """Generate a complete composition using mathematical principles.
        
        Args:
            style: Musical style ('baroque', 'classical', 'romantic', 'modern')
            mathematical_theme: Primary mathematical inspiration ('golden', 'fibonacci', 'prime', 'fractal')
            key_root: MIDI root note (60 = middle C)
            form: Formal structure ('sonata', 'ternary', 'rondo', 'theme_variations')
            measures: Total number of measures
        
        Returns:
            Dictionary with composition data including notes, harmony, structure
        """
        logger.info(f"[MathComposition] Generating {style} composition with {mathematical_theme} theme")
        
        scale_intervals = self.scale_generator.get_scale_intervals('major')
        if mathematical_theme in ['fibonacci', 'prime', 'golden', 'octatonic', 'messiaen_mode_2']:
            scale_intervals = self.scale_generator.get_scale_intervals(mathematical_theme)
        
        primary_motif = self._motif_cache.get(mathematical_theme)
        if primary_motif is None:
            if mathematical_theme == 'fibonacci':
                primary_motif = MotifGenerator.generate_fibonacci_motif(key_root, scale_intervals)
            elif mathematical_theme == 'prime':
                primary_motif = MotifGenerator.generate_prime_motif(key_root, scale_intervals)
            elif mathematical_theme == 'fractal':
                primary_motif = MotifGenerator.generate_fractal_motif(key_root, scale_intervals, 2)
            elif mathematical_theme == 'golden':
                primary_motif = MotifGenerator.generate_golden_motif(key_root, scale_intervals)
            else:
                primary_motif = MotifGenerator.generate_symmetry_motif(key_root, scale_intervals)
            
            self._motif_cache[mathematical_theme] = primary_motif
        
        hook = HookCreator.create_hook(primary_motif, key_root, mathematical_theme)
        
        tension_curve = TensionCurveGenerator.fibonacci_tension(measures * 4.0)
        
        if form == 'sonata':
            sections = self.section_generator.golden_sectioning(measures, 3)
        elif form == 'rondo':
            sections = self.section_generator.fibonacci_sectioning(measures, 5)
        else:
            sections = self.section_generator.golden_sectioning(measures, 3)
        
        composition_data = self._build_composition(
            primary_motif=primary_motif,
            hook=hook,
            tension_curve=tension_curve,
            sections=sections,
            scale=scale_intervals,
            key_root=key_root,
            style=style,
            mathematical_theme=mathematical_theme
        )
        
        composition_data['mathematical_metadata'] = {
            'generator': mathematical_theme,
            'motif_description': primary_motif.mathematical_description,
            'tension_basis': tension_curve.mathematical_basis,
            'hook_catchiness': hook.catchiness_score,
            'scales_used': [mathematical_theme] if mathematical_theme in ScaleGenerator.EXOTIC_SCALES else ['major'],
        }
        
        return composition_data
    
    def _build_composition(
        self,
        primary_motif: MathematicalMotif,
        hook: Hook,
        tension_curve: TensionCurve,
        sections: list[tuple[str, int, float]],
        scale: list[int],
        key_root: int,
        style: str,
        mathematical_theme: str
    ) -> dict:
        """Build complete composition from mathematical elements."""
        
        all_phrases = []
        current_beat = 0.0
        
        for section_idx, (section_name, section_measures, proportion) in enumerate(sections):
            tension_at_section = tension_curve.points[min(section_idx, len(tension_curve.points) - 1)][1]
            
            section_motif = self._transform_motif(
                primary_motif,
                section_idx,
                tension_at_section,
                mathematical_theme
            )
            
            phrases = self._generate_section_phrases(
                section_motif=section_motif,
                hook=hook,
                measures=section_measures,
                scale=scale,
                key_root=key_root,
                tension=tension_at_section,
                section_name=section_name,
                style=style
            )
            
            for phrase in phrases:
                if 'start_beat' in phrase:
                    phrase['start_beat'] += current_beat
                else:
                    phrase['start_beat'] = current_beat
                all_phrases.append(phrase)
            
            current_beat += section_measures * 4.0
        
        return {
            'phrases': all_phrases,
            'total_measures': sum(s[1] for s in sections),
            'key': f'C_{mathematical_theme}',
            'style': style,
            'form': 'mathematical_inspired',
            'sections': sections,
            'primary_motif': primary_motif.id,
        }
    
    def _transform_motif(
        self,
        base_motif: MathematicalMotif,
        section_index: int,
        tension: float,
        theme: str
    ) -> MathematicalMotif:
        """Transform motif for a section using mathematical variations."""
        
        transformations = {
            0: [],
            1: ['sequence_up'],
            2: ['inversion'],
            3: ['augmentation', 'sequence_up'],
        }
        
        transform_list = transformations.get(section_index % 4, [])
        
        if tension > 0.7 and 'fragmentation' not in transform_list:
            transform_list.append('fragmentation')
        
        new_notes = []
        for note in base_motif.notes:
            new_note = note.copy()
            
            if 'sequence_up' in transform_list:
                new_note['pitch'] = note['pitch'] + 2 + (section_index % 3)
            elif 'sequence_down' in transform_list:
                new_note['pitch'] = note['pitch'] - 2
            
            if 'inversion' in transform_list:
                center_pitch = 60
                new_note['pitch'] = center_pitch - (note['pitch'] - center_pitch)
            
            if 'augmentation' in transform_list:
                new_note['duration'] = note['duration'] * 1.5
            elif 'diminution' in transform_list:
                new_note['duration'] = note['duration'] * 0.5
            
            if 'fragmentation' in transform_list:
                if random.random() > 0.5:
                    new_note['duration'] *= 0.5
            
            new_notes.append(new_note)
        
        new_intervals = [n['pitch'] - 60 for n in new_notes]
        
        return MathematicalMotif(
            id=f"{base_motif.id}_transformed_{section_index}",
            generator=base_motif.generator,
            notes=new_notes,
            intervals=new_intervals,
            rhythm=[n['duration'] for n in new_notes],
            duration_pattern=base_motif.duration_pattern,
            mathematical_description=f"{base_motif.mathematical_description} + {', '.join(transform_list)}",
            parent_motif=base_motif.id,
            transformations=transform_list
        )
    
    def _generate_section_phrases(
        self,
        section_motif: MathematicalMotif,
        hook: Hook,
        measures: int,
        scale: list[int],
        key_root: int,
        tension: float,
        section_name: str,
        style: str
    ) -> list[dict]:
        """Generate phrases for a section."""
        
        phrases = []
        beats_per_phrase = 4
        num_phrases = max(1, measures // 2)
        
        for phrase_idx in range(num_phrases):
            phrase_beats = beats_per_phrase
            
            motif_notes = section_motif.notes[phrase_idx % len(section_motif.notes):]
            if not motif_notes:
                motif_notes = section_motif.notes
            
            melody_notes = []
            for i, motif_note in enumerate(motif_notes[:4]):
                scale_degree = scale[(section_motif.intervals[i % len(section_motif.intervals)]) % len(scale)]
                pitch = key_root + scale_degree + 12
                duration = motif_note['duration']
                velocity = int(motif_note['velocity'] * (0.8 + tension * 0.4))
                
                melody_notes.append({
                    'pitch': pitch,
                    'duration': min(duration, 2.0),
                    'velocity': velocity,
                    'beat': sum(m['duration'] for m in melody_notes)
                })
            
            chord_types = ['I', 'V7', 'vi', 'IV']
            chord_idx = phrase_idx % len(chord_types)
            harmony = self._generate_harmony(chord_types[chord_idx], scale, key_root, phrase_beats)
            
            bass_notes = self._generate_bass(chord_types[chord_idx], scale, key_root, phrase_beats)
            
            phrase = {
                'melody_notes': melody_notes,
                'harmony_notes': harmony,
                'bass_notes': bass_notes,
                'dynamics': 'mf' if tension < 0.5 else 'f' if tension < 0.75 else 'ff',
                'duration': phrase_beats,
                'section': section_name,
                'tension_level': tension,
            }
            phrases.append(phrase)
        
        return phrases
    
    def _generate_harmony(
        self,
        chord_type: str,
        scale: list[int],
        key_root: int,
        beats: float
    ) -> list[dict]:
        """Generate harmony for a chord."""
        
        chord_degrees = {
            'I': [0, 2, 4],
            'V7': [4, 6, 8, 11],
            'vi': [5, 7, 9],
            'IV': [3, 5, 7],
            'ii': [1, 3, 5],
            'iii': [2, 4, 6],
        }
        
        degrees = chord_degrees.get(chord_type, [0, 2, 4])
        
        harmony = []
        for i, deg in enumerate(degrees):
            pitch = key_root + scale[deg % len(scale)] + 48
            harmony.append({
                'chord': chord_type if i == 0 else chord_type + '_' + str(i),
                'pitch': pitch,
                'duration': beats,
                'beat': 0.0,
            })
        
        return harmony
    
    def _generate_bass(
        self,
        chord_type: str,
        scale: list[int],
        key_root: int,
        beats: float
    ) -> list[dict]:
        """Generate bass line for a chord."""
        
        chord_roots = {
            'I': 0,
            'V7': 4,
            'vi': 5,
            'IV': 3,
            'ii': 1,
            'iii': 2,
        }
        
        root_degree = chord_roots.get(chord_type, 0)
        root_pitch = key_root + scale[root_degree % len(scale)] + 36
        
        return [{
            'pitch': root_pitch,
            'duration': beats,
            'velocity': 60,
            'beat': 0.0,
        }]


def get_math_composer() -> MathCompositionEngine:
    """Get singleton instance of math composition engine."""
    global _math_composer
    if '_math_composer' not in globals():
        _math_composer = MathCompositionEngine()
    return _math_composer