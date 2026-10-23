"""Music Evolution System for Creative Composition.

This module adds evolutionary learning to music creation, enabling
compositions to improve over time through fitness evaluation,
theme inheritance, and style adaptation.

Key Features:
- Composition fitness evaluation
- Theme inheritance and mutation
- Style parameter evolution
- PFC prediction integration for style guidance
- Cross-pollination of successful patterns

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    MusicEvolutionEngine                          │
├─────────────────────────────────────────────────────────────────┤
│  FitnessEvaluation                                               │
│  - theory_adherence() → Voice leading, harmonic resolution       │
│  - originality_score() → Pattern novelty vs repetition          │
│  - structural_coherence() → Form, development logic           │
│  - aesthetic_quality() → PFC confidence in predictions          │
├─────────────────────────────────────────────────────────────────┤
│  ThemeInheritance                                                │
│  - extract_themes() → Identify successful motifs               │
│  - mutate_theme() → Inversion, transposition, fragmentation    │
│  - cross_pollinate() → Combine themes from compositions        │
│  - store_successful() → Knowledge graph for reuse              │
├─────────────────────────────────────────────────────────────────┤
│  StyleEvolution                                                  │
│  - adapt_parameters() → Weight styles by success rate          │
│  - predict_next_style() → Use PFC predictions                  │
│  - evolve_spontaneity() → Adjust surprise probability         │
│  - learn_preferences() → Track listener/creator feedback       │
├─────────────────────────────────────────────────────────────────┤
│  CreativePressure                                                │
│  - survival_of_fittest() → Top compositions reproduce         │
│  - population_management() → Keep best, discard worst          │
│  - quality_threshold() → Minimum fitness to survive           │
│  - generation_tracking() → Lineage of successful works        │
└─────────────────────────────────────────────────────────────────┘
"""
import hashlib
import logging
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from collections import defaultdict

from core.helpers import atomic_write_json, read_json_with_integrity

logger = logging.getLogger(__name__)


class EvolutionPhase(Enum):
    """Phases of musical evolution."""
    INITIALIZATION = "initialization"
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    REFINEMENT = "refinement"


@dataclass
class MusicalTheme:
    """A melodic/harmonic theme that can evolve."""
    id: str
    notes: list[dict]  # List of {pitch, duration, velocity}
    harmony: list[str]  # Chord progression
    rhythm: list[float]  # Duration pattern
    style: str
    fitness: float = 0.0
    generation: int = 0
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "notes": self.notes,
            "harmony": self.harmony,
            "rhythm": self.rhythm,
            "style": self.style,
            "fitness": self.fitness,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MusicalTheme":
        return cls(
            id=data["id"],
            notes=data["notes"],
            harmony=data["harmony"],
            rhythm=data["rhythm"],
            style=data["style"],
            fitness=data.get("fitness", 0.0),
            generation=data.get("generation", 0),
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CompositionRecord:
    """Record of a completed composition."""
    id: str
    themes: list[str]  # Theme IDs used
    style: str
    key: str
    tempo: int
    duration: float
    created_at: datetime
    fitness: float
    tokens: list[dict] = field(default_factory=list)  # Symbolic/spectral tokens
    pfc_confidence: float = 0.0
    listener_rating: float | None = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "themes": self.themes,
            "style": self.style,
            "key": self.key,
            "tempo": self.tempo,
            "duration": self.duration,
            "created_at": self.created_at.isoformat(),
            "fitness": self.fitness,
            "tokens": self.tokens,
            "pfc_confidence": self.pfc_confidence,
            "listener_rating": self.listener_rating,
            "metadata": self.metadata,
        }


@dataclass
class StyleParameters:
    """Evolved style parameters."""
    # Base style weights (must sum to 1.0)
    baroque: float = 0.25
    classical: float = 0.35
    romantic: float = 0.30
    modern: float = 0.10
    
    # Composer preferences
    bach_affinity: float = 0.3
    mozart_affinity: float = 0.3
    beethoven_affinity: float = 0.25
    modern_affinity: float = 0.15
    
    # Spontaneity parameters
    surprise_probability: float = 0.15
    cross_style_pollination: float = 0.10
    
    # Generation context
    generation: int = 0
    successful_compositions: int = 0
    total_compositions: int = 0
    
    def to_dict(self) -> dict:
        return {
            "baroque": self.baroque,
            "classical": self.classical,
            "romantic": self.romantic,
            "modern": self.modern,
            "bach_affinity": self.bach_affinity,
            "mozart_affinity": self.mozart_affinity,
            "beethoven_affinity": self.beethoven_affinity,
            "modern_affinity": self.modern_affinity,
            "surprise_probability": self.surprise_probability,
            "cross_style_pollination": self.cross_style_pollination,
            "generation": self.generation,
            "successful_compositions": self.successful_compositions,
            "total_compositions": self.total_compositions,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StyleParameters":
        float_fields = {
            'baroque', 'classical', 'romantic', 'modern',
            'bach_affinity', 'mozart_affinity', 'beethoven_affinity', 'modern_affinity',
            'surprise_probability', 'cross_style_pollination',
        }
        int_fields = {'generation', 'successful_compositions', 'total_compositions'}
        filtered = {}
        for k, v in data.items():
            if not hasattr(cls, k):
                continue
            if k in float_fields:
                filtered[k] = float(v) if v is not None else getattr(cls, k)
            elif k in int_fields:
                filtered[k] = int(v) if v is not None else getattr(cls, k)
            else:
                filtered[k] = v
        return cls(**filtered)


class MusicEvolutionEngine:
    """Engine for evolving musical compositions.
    
    This system enables creative evolution by:
    1. Evaluating composition fitness
    2. Inheriting and mutating successful themes
    3. Evolving style parameters based on success
    4. Using PFC predictions to guide composition
    5. Managing a population of compositions
    
    Usage:
        engine = MusicEvolutionEngine()
        
        # After creating a composition
        fitness = engine.evaluate_fitness(composition)
        engine.record_composition(composition, fitness)
        
        # Get evolved parameters for next composition
        params = engine.get_evolved_parameters()
        
        # Extract and mutate successful themes
        themes = engine.get_successful_themes(min_fitness=0.7)
        new_theme = engine.mutate_theme(random.choice(themes))
    """
    
    MIN_FITNESS_THRESHOLD = 0.3
    MAX_POPULATION_SIZE = 100
    ELITE_SIZE = 10
    MUTATION_RATE = 0.3
    CROSSOVER_RATE = 0.4
    
    # Theory rules for fitness evaluation
    FUNCTIONAL_HARMONY_TENSION = {
        "I": 0, "ii": 2, "iii": 1, "IV": 1, "V": 3, "vi": 1, "vii°": 4,
    }
    
    RESOLUTION_PREFERENCES = {
        "V": ["I", "vi"],
        "vii°": ["I", "vi"],
        "ii": ["V", "I"],
        "IV": ["V", "I"],
    }
    
    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "music_evolution"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Population
        self.compositions: dict[str, CompositionRecord] = {}
        self.themes: dict[str, MusicalTheme] = {}
        self.style_parameters = StyleParameters()
        
        # Evolution state
        self.phase = EvolutionPhase.INITIALIZATION
        self.generation = 0
        self.best_fitness = 0.0
        self.avg_fitness = 0.0
        
        # Learning
        self.style_success_rates: dict[str, list[float]] = defaultdict(list)
        self.key_success_rates: dict[str, list[float]] = defaultdict(list)
        self.theme_lineage: dict[str, list[str]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing state
        self._load_state()
    
    def _load_state(self) -> None:
        """Load evolution state from disk."""
        state_file = self.storage_path / "evolution_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                
                self.style_parameters = StyleParameters.from_dict(data.get("style_parameters", {}))
                self.generation = data.get("generation", 0)
                self.best_fitness = data.get("best_fitness", 0.0)
                
                # Load compositions
                for comp_data in data.get("compositions", []):
                    comp = CompositionRecord(
                        id=comp_data["id"],
                        themes=comp_data["themes"],
                        style=comp_data["style"],
                        key=comp_data["key"],
                        tempo=float(comp_data["tempo"]) if comp_data.get("tempo") else 120.0,
                        duration=float(comp_data["duration"]) if comp_data.get("duration") else 0.0,
                        created_at=datetime.fromisoformat(comp_data["created_at"]),
                        fitness=float(comp_data["fitness"]) if comp_data.get("fitness") is not None else 0.5,
                    )
                    self.compositions[comp.id] = comp
                
                # Load themes
                for theme_data in data.get("themes", []):
                    theme = MusicalTheme.from_dict(theme_data)
                    self.themes[theme.id] = theme
                
                logger.info(f"Loaded evolution state: {len(self.compositions)} compositions, {len(self.themes)} themes")
                
            except Exception as e:
                logger.warning(f"Failed to load evolution state: {e}")
    
    def _save_state(self) -> None:
        """Save evolution state to disk."""
        state_file = self.storage_path / "evolution_state.json"
        try:
            data = {
                "style_parameters": self.style_parameters.to_dict(),
                "generation": self.generation,
                "best_fitness": self.best_fitness,
                "compositions": [c.to_dict() for c in self.compositions.values()],
                "themes": [t.to_dict() for t in self.themes.values()],
            }
            atomic_write_json(state_file, data)
        except Exception as e:
            logger.error(f"Failed to save evolution state: {e}")
    
    def evaluate_fitness(
        self,
        composition: dict,
        tokens: list[dict] | None = None,
        pfc_confidence: float = 0.0,
    ) -> float:
        """Evaluate fitness of a composition.
        
        Fitness is a weighted combination of:
        - Theory adherence (30%): Voice leading, harmonic resolution
        - Originality (25%): Pattern novelty vs repetition
        - Structural coherence (25%): Form and development
        - PFC confidence (20%): Quality prediction from learned patterns
        
        Args:
            composition: Composition data (measures, style, key, etc.)
            tokens: Symbolic/spectral tokens generated
            pfc_confidence: Confidence from PFC prediction
            
        Returns:
            Fitness score 0.0-1.0
        """
        with self._lock:
            # Theory adherence (30%)
            theory_score = self._evaluate_theory_adherence(composition)
            
            # Originality (25%)
            originality_score = self._evaluate_originality(composition)
            
            # Structural coherence (25%)
            structure_score = self._evaluate_structure(composition)
            
            # PFC confidence (20%) - use provided or default
            pfc_score = pfc_confidence if pfc_confidence > 0 else 0.5
            
            # Weighted combination
            fitness = (
                theory_score * 0.30 +
                originality_score * 0.25 +
                structure_score * 0.25 +
                pfc_score * 0.20
            )
            
            return max(0.0, min(1.0, fitness))
    
    def _evaluate_theory_adherence(self, composition: dict) -> float:
        """Evaluate adherence to music theory rules.
        
        Checks:
        - Voice leading (smooth transitions)
        - Harmonic resolution (tension/release patterns)
        - Scale adherence (notes in key)
        - Rhythmic coherence (groove consistency)
        """
        score = 0.0
        
        measures = composition.get("measures", [])
        if not measures:
            return 0.5
        
        # Check harmonic progression
        harmony = composition.get("harmony", [])
        if harmony:
            tension_sum = 0
            resolution_count = 0
            
            for i, chord in enumerate(harmony):
                # Get tension value
                tension = self.FUNCTIONAL_HARMONY_TENSION.get(chord, 1)
                tension_sum += tension
                
                # Check if resolves properly
                if i < len(harmony) - 1:
                    next_chord = harmony[i + 1]
                    if chord in self.RESOLUTION_PREFERENCES:
                        if next_chord in self.RESOLUTION_PREFERENCES[chord]:
                            resolution_count += 1
            
            # Tension management score
            if len(harmony) > 1:
                avg_tension = tension_sum / len(harmony)
                resolution_score = resolution_count / (len(harmony) - 1)
                score += (1 - avg_tension / 4) * 0.4  # Moderate tension
                score += resolution_score * 0.3
        
        # Check melodic smoothness (voice leading)
        notes = composition.get("notes", [])
        if notes:
            intervals = []
            for i in range(len(notes) - 1):
                interval = abs(notes[i + 1].get("pitch", 60) - notes[i].get("pitch", 60))
                intervals.append(interval)
            
            if intervals:
                # Prefer smaller intervals (smoother voice leading)
                smooth_intervals = sum(1 for i in intervals if i <= 4)  # Major third or less
                smoothness = smooth_intervals / len(intervals)
                score += smoothness * 0.3
        
        return min(1.0, score)
    
    def _evaluate_originality(self, composition: dict) -> float:
        """Evaluate originality of composition.
        
        Higher score for:
        - Novel patterns not seen in population
        - Unique rhythmic combinations
        - Unexpected harmonic progressions
        """
        with self._lock:
            if not self.compositions:
                return 0.5  # Neutral score when no history
            
            # Compare with existing compositions
            current_harmony = tuple(composition.get("harmony", []))
            tuple(
                (n.get("pitch", 0) % 12, n.get("duration", 0))
                for n in composition.get("notes", [])
            )
            
            # Count how many times similar patterns appear
            similar_count = 0
            for existing in self.compositions.values():
                existing_harmony = tuple(existing.metadata.get("harmony", []))
                if current_harmony[:4] == existing_harmony[:4]:  # First 4 chords match
                    similar_count += 1
            
            # Lower similarity = higher originality
            if self.compositions:
                similarity = similar_count / len(self.compositions)
                originality = 1 - similarity
            else:
                originality = 0.5
            
            return originality
    
    def _evaluate_structure(self, composition: dict) -> float:
        """Evaluate structural coherence.
        
        Checks:
        - Form coherence (A-B-A, sonata, etc.)
        - Development logic
        - Balance of repetition and variation
        """
        score = 0.0
        
        measures = composition.get("measures", [])
        if not measures:
            return 0.5
        
        num_measures = len(measures)
        
        # Check for standard forms
        # Binary form (A-B)
        if num_measures >= 8:
            half = num_measures // 2
            first_half = measures[:half]
            second_half = measures[half:]
            
            # Check if there's contrast
            # (Simplified: check if second half has different harmonic content)
            first_harmony = [m.get("harmony", "") for m in first_half[:4]]
            second_harmony = [m.get("harmony", "") for m in second_half[:4]]
            
            if first_harmony != second_harmony:
                score += 0.3  # Has contrast
            
            # Check for return (ternary A-B-A)
            if num_measures >= 12:
                third = num_measures // 3
                first_third = measures[:third]
                last_third = measures[-third:]
                
                first_harmony = [m.get("harmony", "") for m in first_third]
                last_harmony = [m.get("harmony", "") for m in last_third]
                
                if first_harmony == last_harmony:
                    score += 0.3  # Has return
        
        # Balance of repetition and variation
        if num_measures >= 4:
            # Count phrase repetitions
            phrases = []
            for i in range(0, num_measures, 4):
                phrase = tuple(m.get("harmony", "") for m in measures[i:i+4])
                phrases.append(phrase)
            
            unique_phrases = len(set(phrases))
            total_phrases = len(phrases)
            
            if total_phrases > 0:
                # Good balance: some repetition, some variety
                variety_ratio = unique_phrases / total_phrases
                if 0.3 <= variety_ratio <= 0.7:
                    score += 0.4
        
        return min(1.0, score)
    
    def record_composition(
        self,
        composition: dict,
        fitness: float,
        tokens: list[dict] | None = None,
        pfc_confidence: float = 0.0,
    ) -> str:
        """Record a composition and extract themes.
        
        Args:
            composition: Composition data
            fitness: Fitness score
            tokens: Symbolic/spectral tokens
            pfc_confidence: PFC prediction confidence
            
        Returns:
            Composition ID
        """
        with self._lock:
            # Generate ID
            comp_id = hashlib.sha256(
                f"{composition.get('style', '')}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
            
            # Create record
            record = CompositionRecord(
                id=comp_id,
                themes=[],  # Will be populated
                style=composition.get("style", "classical"),
                key=composition.get("key", "C major"),
                tempo=composition.get("tempo", 120),
                duration=composition.get("duration", 30.0),
                created_at=datetime.now(),
                fitness=fitness,
                tokens=tokens or [],
                pfc_confidence=pfc_confidence,
                metadata={
                    "harmony": composition.get("harmony", []),
                    "form": composition.get("form", "A-B-A"),
                },
            )
            
            self.compositions[comp_id] = record
            
            # Extract themes if fitness is good
            if fitness >= 0.6:
                themes = self._extract_themes(composition, fitness)
                for theme in themes:
                    self.themes[theme.id] = theme
                    record.themes.append(theme.id)
            
            # Update style success rates
            style = composition.get("style", "classical")
            self.style_success_rates[style].append(fitness)
            key = composition.get("key", "C major")
            self.key_success_rates[key].append(fitness)
            
            # Update best fitness
            if fitness > self.best_fitness:
                self.best_fitness = fitness
            
            # Update generation
            self.style_parameters.total_compositions += 1
            if fitness >= 0.7:
                self.style_parameters.successful_compositions += 1
            
            # Population management
            self._manage_population()
            
            # Save state
            self._save_state()
            
            return comp_id
    
    def _extract_themes(self, composition: dict, fitness: float) -> list[MusicalTheme]:
        """Extract thematic material from a successful composition."""
        themes = []
        
        notes = composition.get("notes", [])
        harmony = composition.get("harmony", [])
        rhythm = [n.get("duration", 0.5) for n in notes]
        style = composition.get("style", "classical")
        
        if len(notes) < 4:
            return themes
        
        # Extract melodic phrases (groups of 4-8 notes)
        for i in range(0, len(notes) - 4, 4):
            phrase_notes = notes[i:i+8]
            theme_id = hashlib.sha256(
                f"{style}_{i}_{datetime.now().timestamp()}".encode()
            ).hexdigest()[:8]
            
            theme = MusicalTheme(
                id=theme_id,
                notes=phrase_notes,
                harmony=harmony[i//4:i//4+2] if i//4 < len(harmony) else [],
                rhythm=rhythm[i:i+8],
                style=style,
                fitness=fitness,
                generation=self.generation,
            )
            themes.append(theme)
        
        return themes
    
    def _manage_population(self) -> None:
        """Manage composition population.
        
        Keep the best, discard the worst to maintain quality.
        """
        if len(self.compositions) <= self.MAX_POPULATION_SIZE:
            return
        
        # Sort by fitness
        sorted_comps = sorted(
            self.compositions.values(),
            key=lambda c: c.fitness,
            reverse=True
        )
        
        # Keep top compositions
        keep_ids = {c.id for c in sorted_comps[:self.MAX_POPULATION_SIZE]}
        
        # Remove low-fitness compositions
        to_remove = [cid for cid in self.compositions if cid not in keep_ids]
        for cid in to_remove:
            del self.compositions[cid]
            
            # Also remove associated themes
            for tid in list(self.themes.keys()):
                if self.themes[tid].generation < self.generation:
                    del self.themes[tid]
        
        logger.debug(f"Population management: removed {len(to_remove)} low-fitness compositions")
    
    def get_evolved_parameters(self) -> StyleParameters:
        """Get style parameters evolved from successful compositions.
        
        Uses fitness-weighted averaging to adjust style preferences.
        """
        with self._lock:
            if not self.compositions:
                return self.style_parameters
            
            # Calculate style success rates
            style_scores: dict[str, list[float]] = defaultdict(list)
            
            for comp in self.compositions.values():
                style_scores[comp.style].append(comp.fitness)
            
            # Average fitness per style
            style_avg: dict[str, float] = {}
            for style, scores in style_scores.items():
                style_avg[style] = sum(scores) / len(scores)
            
            # Adjust parameters based on success
            if style_avg:
                # Normalize to weights
                total = sum(style_avg.values())
                if total > 0:
                    # Gentle adjustment towards successful styles
                    learning_rate = 0.1
                    
                    if "baroque" in style_avg:
                        target = style_avg["baroque"] / total
                        self.style_parameters.baroque = (
                            (1 - learning_rate) * self.style_parameters.baroque +
                            learning_rate * target
                        )
                    
                    if "classical" in style_avg:
                        target = style_avg["classical"] / total
                        self.style_parameters.classical = (
                            (1 - learning_rate) * self.style_parameters.classical +
                            learning_rate * target
                        )
                    
                    if "romantic" in style_avg:
                        target = style_avg["romantic"] / total
                        self.style_parameters.romantic = (
                            (1 - learning_rate) * self.style_parameters.romantic +
                            learning_rate * target
                        )
                    
                    if "modern" in style_avg:
                        target = style_avg["modern"] / total
                        self.style_parameters.modern = (
                            (1 - learning_rate) * self.style_parameters.modern +
                            learning_rate * target
                        )
            
            # Adjust surprise probability based on success
            high_fitness_comps = [c for c in self.compositions.values() if c.fitness >= 0.7]
            if high_fitness_comps:
                # If high-fitness compositions had cross-pollination, increase it
                cross_poll_count = sum(
                    1 for c in high_fitness_comps
                    if c.metadata.get("cross_style_pollination", False)
                )
                cross_poll_rate = cross_poll_count / len(high_fitness_comps)
                
                self.style_parameters.cross_style_pollination = (
                    0.8 * self.style_parameters.cross_style_pollination +
                    0.2 * cross_poll_rate
                )
            
            self._save_state()
            
            return self.style_parameters
    
    def get_successful_themes(self, min_fitness: float = 0.7, limit: int = 10) -> list[MusicalTheme]:
        """Get themes from successful compositions for inheritance.
        
        Args:
            min_fitness: Minimum fitness threshold
            limit: Maximum number of themes to return
            
        Returns:
            List of successful themes
        """
        with self._lock:
            # Get successful compositions
            {
                c.id for c in self.compositions.values()
                if c.fitness >= min_fitness
            }
            
            # Get themes from successful compositions
            themes = [
                self.themes[tid] for tid in self.themes
                if self.themes[tid].fitness >= min_fitness
            ]
            
            # Sort by fitness, return top
            themes.sort(key=lambda t: t.fitness, reverse=True)
            
            return themes[:limit]
    
    def mutate_theme(self, theme: MusicalTheme, mutation_strength: float = 0.3) -> MusicalTheme:
        """Create a mutated version of a theme.
        
        Mutation operations:
        - Transposition (shift all pitches)
        - Inversion (flip melodic contour)
        - Retrograde (reverse order)
        - Fragmentation (take portion)
        - Rhythmic variation
        
        Args:
            theme: Theme to mutate
            mutation_strength: How much to change (0-1)
            
        Returns:
            New mutated theme
        """
        new_notes = list(theme.notes)  # Copy
        new_harmony = list(theme.harmony)
        new_rhythm = list(theme.rhythm)
        
        # Apply random mutations
        mutation_choice = random.random()
        
        if mutation_choice < 0.25:
            # Transposition
            semitones = random.randint(-5, 5)
            new_notes = [
                {**n, "pitch": n.get("pitch", 60) + semitones}
                for n in new_notes
            ]
        
        elif mutation_choice < 0.5:
            # Inversion (flip melodic contour)
            base_pitch = new_notes[0].get("pitch", 60) if new_notes else 60
            new_notes = [
                {**n, "pitch": base_pitch - (n.get("pitch", 60) - base_pitch)}
                for n in new_notes
            ]
        
        elif mutation_choice < 0.75:
            # Retrograde (reverse)
            new_notes = list(reversed(new_notes))
            new_rhythm = list(reversed(new_rhythm))
        
        else:
            # Fragmentation (take portion)
            if len(new_notes) > 4:
                start = random.randint(0, len(new_notes) // 2)
                end = start + random.randint(4, len(new_notes) - start)
                new_notes = new_notes[start:end]
                new_rhythm = new_rhythm[start:end]
        
        # Create new theme
        new_id = hashlib.sha256(
            f"mutant_{theme.id}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:8]
        
        return MusicalTheme(
            id=new_id,
            notes=new_notes,
            harmony=new_harmony,
            rhythm=new_rhythm,
            style=theme.style,
            fitness=0.0,  # Unknown until evaluated
            generation=self.generation + 1,
            parent_id=theme.id,
        )
    
    def cross_pollinate_themes(self, theme1: MusicalTheme, theme2: MusicalTheme) -> MusicalTheme:
        """Create new theme by combining two themes.
        
        Crossover operations:
        - Take first half from theme1, second from theme2
        - Combine harmonic progressions
        - Blend rhythmic patterns
        
        Args:
            theme1: First parent theme
            theme2: Second parent theme
            
        Returns:
            New combined theme
        """
        # Crossover point
        crossover = len(theme1.notes) // 2
        
        # Combine notes
        new_notes = theme1.notes[:crossover] + theme2.notes[crossover:]
        
        # Combine harmony
        new_harmony = theme1.harmony[:len(theme1.harmony)//2] + theme2.harmony[len(theme2.harmony)//2:]
        
        # Blend rhythm
        new_rhythm = theme1.rhythm[:crossover] + theme2.rhythm[crossover:]
        
        # Choose style from higher-fitness parent
        style = theme1.style if theme1.fitness > theme2.fitness else theme2.style
        
        new_id = hashlib.sha256(
            f"cross_{theme1.id}_{theme2.id}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:8]
        
        return MusicalTheme(
            id=new_id,
            notes=new_notes,
            harmony=new_harmony,
            rhythm=new_rhythm,
            style=style,
            fitness=0.0,
            generation=self.generation + 1,
            parent_id=f"{theme1.id}+{theme2.id}",
        )
    
    def adapt_to_pfc_prediction(
        self,
        prediction: dict,
        current_params: StyleParameters | None = None,
    ) -> StyleParameters:
        """Adapt style parameters based on PFC predictions.
        
        Uses PFC's prediction of what would be successful to bias
        style selection towards predicted successful styles.
        
        Args:
            prediction: PFC prediction with recommended style/confidence
            current_params: Current style parameters
            
        Returns:
            Adapted style parameters
        """
        params = current_params or self.style_parameters
        
        # Extract PFC recommendation
        recommended_style = prediction.get("style", "classical")
        confidence = prediction.get("confidence", 0.5)
        
        # If confidence is high, bias strongly towards prediction
        if confidence >= 0.7:
            # Strong bias
            adjustment = min(0.2, (confidence - 0.5) * 0.4)
            
            if recommended_style == "baroque":
                params.baroque = min(0.5, params.baroque + adjustment)
            elif recommended_style == "classical":
                params.classical = min(0.6, params.classical + adjustment)
            elif recommended_style == "romantic":
                params.romantic = min(0.5, params.romantic + adjustment)
            elif recommended_style == "modern":
                params.modern = min(0.3, params.modern + adjustment)
        
        elif confidence >= 0.5:
            # Moderate bias
            adjustment = (confidence - 0.5) * 0.2
            
            if recommended_style == "baroque":
                params.baroque = min(0.4, params.baroque + adjustment)
            elif recommended_style == "classical":
                params.classical = min(0.5, params.classical + adjustment)
            elif recommended_style == "romantic":
                params.romantic = min(0.4, params.romantic + adjustment)
            elif recommended_style == "modern":
                params.modern = min(0.25, params.modern + adjustment)
        
        # Normalize weights
        total = params.baroque + params.classical + params.romantic + params.modern
        if total > 0:
            params.baroque /= total
            params.classical /= total
            params.romantic /= total
            params.modern /= total
        
        return params
    
    def get_recommended_style(self, pfc_prediction: dict | None = None) -> tuple[str, StyleParameters]:
        """Get style recommendation for next composition.
        
        Combines:
        - Evolved parameters from successful compositions
        - PFC prediction (if available)
        - Exploration vs exploitation balance
        
        Args:
            pfc_prediction: Optional PFC prediction
            
        Returns:
            Tuple of (style_name, style_parameters)
        """
        with self._lock:
            params = self.get_evolved_parameters()
            
            # Incorporate PFC prediction if available
            if pfc_prediction:
                params = self.adapt_to_pfc_prediction(pfc_prediction, params)
            
            # Determine exploration vs exploitation
            exploration_rate = 1.0 / (1.0 + self.style_parameters.generation * 0.1)
            
            if random.random() < exploration_rate:
                # Exploration: choose randomly weighted by parameters
                r = random.random()
                if r < params.baroque:
                    style = "baroque"
                elif r < params.baroque + params.classical:
                    style = "classical"
                elif r < params.baroque + params.classical + params.romantic:
                    style = "romantic"
                else:
                    style = "modern"
            else:
                # Exploitation: choose most successful style
                style_rates = {s: sum(scores) / len(scores) if scores else 0.5 
                              for s, scores in self.style_success_rates.items()}
                
                if style_rates:
                    style = max(style_rates.keys(), key=lambda s: style_rates.get(s, 0))
                else:
                    # Default to classical
                    style = "classical"
            
            return style, params
    
    def increment_generation(self) -> int:
        """Increment evolution generation.
        
        Called after each batch of compositions to track evolution progress.
        
        Returns:
            New generation number
        """
        with self._lock:
            self.generation += 1
            self.style_parameters.generation = self.generation
            self._save_state()
            return self.generation
    
    def get_statistics(self) -> dict:
        """Get evolution statistics."""
        with self._lock:
            if not self.compositions:
                return {
                    "generation": self.generation,
                    "total_compositions": 0,
                    "total_themes": len(self.themes),
                    "best_fitness": 0,
                    "avg_fitness": 0,
                }
            
            fitnesses = [c.fitness for c in self.compositions.values()]
            
            return {
                "generation": self.generation,
                "phase": self.phase.value,
                "total_compositions": len(self.compositions),
                "total_themes": len(self.themes),
                "best_fitness": max(fitnesses),
                "avg_fitness": sum(fitnesses) / len(fitnesses),
                "style_parameters": self.style_parameters.to_dict(),
                "style_success_rates": {
                    s: sum(scores) / len(scores) if scores else 0
                    for s, scores in self.style_success_rates.items()
                },
                "key_success_rates": {
                    k: sum(scores) / len(scores) if scores else 0
                    for k, scores in self.key_success_rates.items()
                },
            }


# Singleton
_evolution_engine: MusicEvolutionEngine | None = None
_evolution_lock = threading.Lock()


def get_evolution_engine(storage_path: Path | None = None) -> MusicEvolutionEngine:
    """Get or create evolution engine singleton."""
    global _evolution_engine
    if _evolution_engine is None:
        with _evolution_lock:
            if _evolution_engine is None:
                _evolution_engine = MusicEvolutionEngine(storage_path)
    return _evolution_engine
