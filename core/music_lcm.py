"""Music LCM Integration - Large Concept Model for Musical Creativity.

This module uses the Prefrontal Cortex LCM to generate and guide musical
compositions by operating on musical concepts rather than just notes.

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Musical Concept Model                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   COMPOSITION → CONCEPT EXTRACTION → LCM PREDICTION → COMPOSITION           │
│                                                                              │
│   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐          │
│   │   Composition   │    │  Concept       │    │   LCM Core      │          │
│   │   (Notes/Form)  │───▶│  Encoder       │───▶│   Prediction    │          │
│   │                 │    │                │    │                 │          │
│   └─────────────────┘    └─────────────────┘    └────────┬────────┘          │
│                                                         │                     │
│   ┌─────────────────┐    ┌─────────────────┐    ┌───────▼────────┐           │
│   │   New Section   │    │  Concept       │    │  Predicted      │           │
│   │   Generation    │◀───│  Decoder       │◀───│  Musical       │           │
│   │                 │    │                │    │  Concepts      │           │
│   └─────────────────┘    └─────────────────┘    └────────────────┘           │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│   Musical Concepts:                                                          │
│   - Structural: "exposition", "developmental_sequence", "climax"            │
│   - Harmonic: "tension_rising", "resolution_approaching", "modulation"      │
│   - Melodic: "ascending_theme", "ornamented_variation", "sequence"         │
│   - Emotional: "dramatic_contrast", "peaceful_resolution", "tension"        │
│   - Rhythmic: "driving_rhythm", "lyrical_flow", "syncopation"              │
└─────────────────────────────────────────────────────────────────────────────┘

The LCM learns patterns of musical concepts that lead to successful compositions,
then predicts what musical concept should come next, guiding the composer.
"""
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Singleton
_music_lcm = None
_lcm_lock = threading.Lock()


# Musical concept vocabulary for LCM
MUSICAL_CONCEPTS = {
    # Structural concepts
    "exposition_start": "Beginning of main theme presentation",
    "exposition_transition": "Transition in exposition",
    "exposition_close": "Closing of exposition",
    "development_open": "Opening developmental material",
    "development_climax": "Climax of development",
    "development_transition": "Transition in development",
    "recapitulation_return": "Return to main themes",
    "recapitulation_variant": "Variation in recapitulation",
    "coda_begin": "Beginning of concluding section",
    "coda_climax": "Final climax before resolution",
    
    # Harmonic concepts
    "tonic_establishment": "Establishing home key",
    "dominant_preparation": "Preparing dominant",
    "modulation_start": "Beginning key change",
    "modulation_confirm": "Confirming new key",
    "deceptive_resolution": "Deceptive cadence/resolution",
    "harmonic_tension": "Building harmonic tension",
    "harmonic_resolution": "Resolving harmonic tension",
    "secondary_dominant": "Applying secondary dominant",
    
    # Melodic concepts
    "theme_statement": "Primary theme statement",
    "theme_development": "Developing thematic material",
    "theme_transformation": "Transforming theme",
    "theme_fragmentation": "Breaking down theme",
    "melodic_sequence": "Sequential repetition",
    "melodic_climax": "Melodic peak",
    "melodic_descent": "Descending motion",
    "ornamentation": "Adding ornaments",
    
    # Emotional concepts
    "tension_build": "Building emotional tension",
    "tension_release": "Releasing tension",
    "dramatic_contrast": "Sudden emotional shift",
    "peaceful_resolution": "Calm resolution",
    "triumphant_statement": "Bold, confident statement",
    "mysterious_passage": "Unclear, questioning material",
    "playful_interruption": "Light, unexpected element",
    
    # Rhythmic concepts
    "driving_rhythm": "Persistent rhythmic pattern",
    "rhythmic_displacement": "Shifting rhythmic accents",
    "rhythmic_augmentation": "Lengthening note values",
    "rhythmic_diminution": "Shortening note values",
    "syncopation": "Off-beat accents",
    "pulse_establishment": "Setting tempo feel",
    "pulse_disruption": "Disrupting steady pulse",
    
    # Textural concepts
    "texture_thicken": "Adding voices/layers",
    "texture_thin": "Reducing voices/layers",
    "texture_contrast": "Sudden texture change",
    "homophonic_passage": "Chordal texture",
    "polyphonic_passage": "Contrapuntal texture",
    "monophonic_passage": "Single melody",
    
    # Dynamic concepts
    "crescendo": "Growing louder",
    "diminuendo": "Growing softer",
    "sudden_forte": "Sudden loud",
    "sudden_piano": "Sudden soft",
    "terraced_dynamics": "Step-wise dynamic changes",
}


@dataclass
class MusicalConcept:
    """A musical concept that can be processed by LCM."""
    id: str
    concept_type: str  # From MUSICAL_CONCEPTS
    embedding: np.ndarray = None
    context: dict = field(default_factory=dict)
    
    # Musical parameters this concept implies
    suggested_key: Optional[str] = None
    suggested_tempo: Optional[int] = None
    suggested_dynamics: Optional[str] = None
    suggested_register: Optional[str] = None  # high, middle, low
    suggested_density: Optional[int] = None  # 1-5 voice density
    
    # Structural role
    section_position: Optional[float] = None  # 0-1 position in section
    tension_level: Optional[float] = None  # 0-1 tension
    resolution_expectation: Optional[float] = None  # 0-1 resolution expected
    
    # Parent lineage
    parent_concept_id: Optional[str] = None
    composition_id: Optional[str] = None
    
    # Learning signal
    effectiveness: float = 0.0  # How well did this concept work?


@dataclass  
class MusicalConceptSequence:
    """A sequence of musical concepts forming a composition arc."""
    concepts: list[MusicalConcept]
    style: str
    form: str
    key: str
    outcome_quality: float = 0.0  # Overall composition quality
    outcome_coherence: float = 0.0
    outcome_development: float = 0.0


class MusicalConceptEncoder:
    """Encode musical structures into concepts for LCM."""
    
    def __init__(self):
        self.concept_vocabulary = MUSICAL_CONCEPTS
        self.concept_embeddings = {}
        
    def extract_concepts_from_composition(
        self,
        composition_data: dict,
        style: str,
        form: str
    ) -> list[MusicalConcept]:
        """Extract musical concepts from a composition.
        
        Args:
            composition_data: Musical data (themes, harmony, dynamics)
            style: Musical style
            form: Form type
            
        Returns:
            List of MusicalConcept objects
        """
        concepts = []
        
        # Extract from phrases
        phrases = composition_data.get("phrases", [])
        composition_data.get("sections", [])
        composition_data.get("themes", [])
        
        # Track position
        total_phrases = len(phrases) if phrases else 1
        
        for phrase_idx, phrase in enumerate(phrases):
            position = phrase_idx / total_phrases
            section = phrase.get("section", "unknown")
            
            # Determine structural concept
            structural_concept = self._get_structural_concept(
                position, section, form
            )
            
            # Determine harmonic concept from chord progression
            harmonic_concept = self._get_harmonic_concept(
                phrase.get("harmony_notes", []),
                position
            )
            
            # Determine melodic concept
            melodic_concept = self._get_melodic_concept(
                phrase.get("melody_notes", []),
                position
            )
            
            # Determine emotional concept from dynamics/tension
            self._get_emotional_concept(
                phrase.get("dynamics", "mf"),
                position
            )
            
            # Create concept
            concept_id = f"concept_{section}_{phrase_idx}"
            
            concept = MusicalConcept(
                id=concept_id,
                concept_type=structural_concept,
                context={
                    "style": style,
                    "form": form,
                    "section": section,
                    "position": position,
                },
                section_position=position or 0.0,
                suggested_dynamics=phrase.get("dynamics", "mf"),
                composition_id=composition_data.get("id"),
            )
            
            concepts.append(concept)
            
            # Add secondary concepts
            if harmonic_concept and harmonic_concept != structural_concept:
                concepts.append(MusicalConcept(
                    id=f"{concept_id}_harm",
                    concept_type=harmonic_concept,
                    context={"type": "harmonic"},
                    parent_concept_id=concept_id,
                ))
            
            if melodic_concept and melodic_concept != structural_concept:
                concepts.append(MusicalConcept(
                    id=f"{concept_id}_mel",
                    concept_type=melodic_concept,
                    context={"type": "melodic"},
                    parent_concept_id=concept_id,
                ))
        
        return concepts
    
    def _get_structural_concept(self, position: float, section: str, form: str) -> str:
        """Determine structural concept from position and section."""
        form_lower = form.lower()
        
        if "sonata" in form_lower:
            if "exposition" in section.lower():
                if position < 0.3:
                    return "exposition_start"
                elif position < 0.7:
                    return "exposition_transition"
                else:
                    return "exposition_close"
            elif "development" in section.lower():
                if position < 0.5:
                    return "development_open"
                elif position < 0.8:
                    return "development_climax"
                else:
                    return "development_transition"
            elif "recapitulation" in section.lower():
                if position < 0.5:
                    return "recapitulation_return"
                else:
                    return "recapitulation_variant"
        
        elif "ternary" in form_lower or "aba" in form_lower:
            if position < 0.33:
                return "theme_statement"
            elif position < 0.66:
                return "theme_development" if "B" in section else "theme_statement"
            else:
                return "theme_statement" if "B" not in section else "theme_development"
        
        elif "rondo" in form_lower:
            if "A" in section:
                return "theme_statement"
            else:
                return "theme_development"
        
        # Default
        if position < 0.25:
            return "exposition_start"
        elif position < 0.75:
            return "development_open"
        else:
            return "recapitulation_return"
    
    def _get_harmonic_concept(self, harmony_notes: list, position: float) -> str:
        """Determine harmonic concept from chord progression."""
        if not harmony_notes:
            return "tonic_establishment"
        
        chords = [h.get("chord", "I") for h in harmony_notes[:4]]
        
        # Analyze progression
        has_secondary = any("/" in c for c in chords)
        has_dominant = "V" in chords[0] or "V7" in chords[0]
        ends_tonic = chords[-1] in ["I", "i"] if chords else True
        
        if has_secondary:
            return "secondary_dominant"
        elif has_dominant and not ends_tonic:
            return "harmonic_tension"
        elif ends_tonic and position > 0.7:
            return "harmonic_resolution"
        elif position < 0.3:
            return "tonic_establishment"
        else:
            return "harmonic_tension"
    
    def _get_melodic_concept(self, melody_notes: list, position: float) -> str:
        """Determine melodic concept from melody pattern."""
        if not melody_notes:
            return "theme_statement"
        
        # Analyze melody characteristics
        pitches = [n.get("pitch", 60) for n in melody_notes]
        
        if not pitches:
            return "theme_statement"
        
        # Calculate contour
        ascending = sum(1 for i in range(len(pitches)-1) if pitches[i+1] > pitches[i])
        descending = sum(1 for i in range(len(pitches)-1) if pitches[i+1] < pitches[i])
        
        # Calculate range
        pitch_range = max(pitches) - min(pitches) if pitches else 0
        
        if ascending > descending * 1.5:
            return "melodic_sequence" if position < 0.5 else "melodic_climax"
        elif descending > ascending * 1.5:
            return "melodic_descent"
        elif pitch_range < 6:
            return "ornamentation"
        else:
            return "theme_statement"
    
    def _get_emotional_concept(self, dynamics: str, position: float) -> str:
        """Determine emotional concept from dynamics."""
        dynamic_level = {
            "pp": 0.1, "p": 0.3, "mp": 0.4, "mf": 0.5,
            "f": 0.7, "ff": 0.9
        }.get(dynamics, 0.5)
        
        if dynamic_level > 0.7:
            if position > 0.7:
                return "triumphant_statement"
            else:
                return "tension_build"
        elif dynamic_level < 0.4:
            if position > 0.7:
                return "peaceful_resolution"
            else:
                return "mysterious_passage"
        else:
            return "theme_statement"


class MusicalConceptDecoder:
    """Decode LCM predictions back into musical parameters."""
    
    def __init__(self):
        self.concept_to_params = self._build_concept_params()
    
    def _build_concept_params(self) -> dict:
        """Map concepts to musical parameters."""
        return {
            "exposition_start": {
                "key_implication": "tonic",
                "tempo_range": (100, 140),
                "density": 3,
                "register": "middle",
                "tension": 0.3,
            },
            "exposition_transition": {
                "key_implication": "modulate",
                "tempo_range": (100, 140),
                "density": 3,
                "register": "middle",
                "tension": 0.5,
            },
            "development_open": {
                "key_implication": "foreign",
                "tempo_range": (90, 130),
                "density": 4,
                "register": "varied",
                "tension": 0.6,
            },
            "development_climax": {
                "key_implication": "foreign",
                "tempo_range": (110, 150),
                "density": 5,
                "register": "high",
                "tension": 0.9,
            },
            "recapitulation_return": {
                "key_implication": "tonic",
                "tempo_range": (100, 140),
                "density": 3,
                "register": "middle",
                "tension": 0.4,
            },
            "coda_climax": {
                "key_implication": "tonic",
                "tempo_range": (120, 160),
                "density": 5,
                "register": "high",
                "tension": 0.85,
            },
            "harmonic_tension": {
                "key_implication": "any",
                "tempo_range": None,
                "density": None,
                "register": None,
                "tension": 0.7,
            },
            "harmonic_resolution": {
                "key_implication": "tonic",
                "tempo_range": None,
                "density": 2,
                "register": "middle",
                "tension": 0.2,
            },
            "melodic_climax": {
                "key_implication": None,
                "tempo_range": None,
                "density": None,
                "register": "high",
                "tension": 0.8,
            },
            "melodic_descent": {
                "key_implication": None,
                "tempo_range": None,
                "density": None,
                "register": "descending",
                "tension": 0.4,
            },
            "tension_build": {
                "key_implication": None,
                "tempo_range": (110, 150),
                "density": 4,
                "register": "rising",
                "tension": 0.7,
            },
            "peaceful_resolution": {
                "key_implication": "tonic",
                "tempo_range": (60, 100),
                "density": 2,
                "register": "low",
                "tension": 0.2,
            },
            "triumphant_statement": {
                "key_implication": "tonic",
                "tempo_range": (120, 160),
                "density": 5,
                "register": "high",
                "tension": 0.6,
            },
        }
    
    def decode_concept_to_guidance(
        self,
        concept: str,
        current_context: dict
    ) -> dict:
        """Convert predicted concept into compositional guidance.
        
        Args:
            concept: Predicted concept type
            current_context: Current composition state
            
        Returns:
            Dictionary of musical parameters to apply
        """
        # Get base parameters for this concept
        params = self.concept_to_params.get(concept, {
            "tension": 0.5,
        })
        
        guidance = {
            "concept": concept,
            "tension_target": params.get("tension", 0.5),
        }
        
        # Key guidance
        if params.get("key_implication") == "tonic":
            guidance["key_guidance"] = "return_to_tonic"
        elif params.get("key_implication") == "foreign":
            guidance["key_guidance"] = "modulate_foreign"
        elif params.get("key_implication") == "modulate":
            guidance["key_guidance"] = "modulate_related"
        
        # Tempo guidance
        if params.get("tempo_range"):
            guidance["tempo_range"] = params["tempo_range"]
        
        # Density guidance
        if params.get("density"):
            guidance["voice_density"] = params["density"]
        
        # Register guidance
        if params.get("register") == "high":
            guidance["register_guidance"] = "prefer_upper"
        elif params.get("register") == "low":
            guidance["register_guidance"] = "prefer_lower"
        elif params.get("register") == "middle":
            guidance["register_guidance"] = "prefer_middle"
        
        # Dynamic guidance based on tension
        tension = params.get("tension", 0.5)
        if tension > 0.7:
            guidance["dynamics_guidance"] = "f_or_ff"
        elif tension > 0.5:
            guidance["dynamics_guidance"] = "mf_or_f"
        elif tension > 0.3:
            guidance["dynamics_guidance"] = "mp_or_mf"
        else:
            guidance["dynamics_guidance"] = "p_or_mp"
        
        return guidance


class MusicLCM:
    """Integrate LCM with music composition."""
    
    def __init__(self):
        self.encoder = MusicalConceptEncoder()
        self.decoder = MusicalConceptDecoder()
        self.pfc = None  # Will be set on first use
        
        # Concept history for prediction
        self.concept_history: list[list[MusicalConcept]] = []
        self.MAX_HISTORY = 100
        
        # Learning data
        self.successful_sequences: list[dict] = []
        self.failed_sequences: list[dict] = []
        
        # Concept effectiveness tracking
        self.concept_effectiveness: dict[str, float] = defaultdict(float)
    
    def _get_pfc(self):
        """Get PFC instance lazily."""
        if self.pfc is None:
            from core.prefrontal_cortex import get_prefrontal_cortex
            self.pfc = get_prefrontal_cortex()
        return self.pfc
    
    def _record_concept_history(self, concepts: list) -> None:
        """Add concepts to history with bound enforcement."""
        self.concept_history.append(concepts)
        if len(self.concept_history) > self.MAX_HISTORY:
            self.concept_history = self.concept_history[-self.MAX_HISTORY:]
    
    def encode_composition(
        self,
        composition_data: dict,
        style: str,
        form: str
    ) -> list[MusicalConcept]:
        """Encode a composition into concept sequence."""
        return self.encoder.extract_concepts_from_composition(
            composition_data, style, form
        )
    
    def predict_next_concept(
        self,
        current_concepts: list[MusicalConcept],
        context: dict
    ) -> tuple[str, float]:
        """Predict what musical concept should come next.
        
        Uses LCM to predict based on concept history.
        
        Args:
            current_concepts: Current concept sequence
            context: Composition context (style, form, position)
            
        Returns:
            (predicted_concept_type, confidence)
        """
        pfc = self._get_pfc()
        
        if not pfc or not pfc._running:
            # Fallback: use heuristics
            return self._heuristic_prediction(current_concepts, context)
        
        try:
            # Convert concepts to PFC format
            from core.prefrontal_cortex import Concept, ConceptType
            
            pfc_concepts = []
            for mc in current_concepts[-10:]:  # Last 10 concepts
                concept = Concept(
                    id=mc.id,
                    embedding=mc.embedding if mc.embedding is not None else np.zeros(768),
                    concept_type=ConceptType.MUSIC_TOKEN,
                    content=f"{mc.concept_type}:{mc.section_position:.2f}",
                    source="music_composition",
                )
                pfc_concepts.append(concept)
            
            # Add context as a concept
            context_concept = Concept(
                id="context",
                embedding=np.zeros(768),
                concept_type=ConceptType.INTENTION,
                content=f"style:{context.get('style', 'classical')},form:{context.get('form', 'ternary')},position:{context.get('position', 0.5):.2f}",
                source="music_context",
            )
            pfc_concepts.append(context_concept)
            
            # Get prediction from PFC
            predicted_embedding, predicted_types, confidence = pfc.lcm.predict_next(
                pfc_concepts
            )
            
            # Map predicted types to musical concepts
            music_types = [t for t in predicted_types if t[0] == ConceptType.MUSIC_TOKEN]
            
            if music_types:
                # Extract concept type from content
                # Content format: "concept_type:position"
                best_type = music_types[0]
                concept_str = best_type[0].value if hasattr(best_type[0], 'value') else str(best_type[0])
                return concept_str, confidence
            
            # Fallback to heuristic
            return self._heuristic_prediction(current_concepts, context)
            
        except Exception as e:
            logger.warning(f"LCM prediction failed: {e}")
            return self._heuristic_prediction(current_concepts, context)
    
    def _heuristic_prediction(
        self,
        current_concepts: list[MusicalConcept],
        context: dict
    ) -> tuple[str, float]:
        """Heuristic prediction when LCM is unavailable."""
        if not current_concepts:
            return "exposition_start", 0.5
        
        # Get last concept
        last = current_concepts[-1]
        position = last.section_position or 0.5
        form = context.get("form", "ternary")
        
        # Predict based on position and form
        if position < 0.3:
            if "development" in last.concept_type:
                return "development_open", 0.7
            return "theme_statement", 0.7
        elif position < 0.6:
            return "theme_development", 0.7
        elif position < 0.8:
            if "sonata" in form.lower():
                return "development_climax", 0.6
            return "theme_development", 0.6
        else:
            return "peaceful_resolution", 0.7
    
    def decode_to_composition_params(
        self,
        predicted_concept: str,
        current_context: dict
    ) -> dict:
        """Convert predicted concept into compositional parameters."""
        return self.decoder.decode_concept_to_guidance(
            predicted_concept, current_context
        )
    
    def learn_from_composition(
        self,
        concepts: list[MusicalConcept],
        quality_score: float,
        style: str
    ) -> None:
        """Learn from a completed composition's success/failure.
        
        Updates concept effectiveness tracking for future predictions.
        """
        # Record sequence
        sequence_data = {
            "concepts": [c.concept_type for c in concepts],
            "quality": quality_score,
            "style": style,
            "timestamp": datetime.now().isoformat(),
        }
        
        if quality_score > 0.6:
            self.successful_sequences.append(sequence_data)
            # Update effectiveness
            for concept in concepts:
                self.concept_effectiveness[concept.concept_type] = (
                    0.9 * self.concept_effectiveness[concept.concept_type] +
                    0.1 * quality_score
                )
        else:
            self.failed_sequences.append(sequence_data)
        
        # Trim history
        if len(self.successful_sequences) > 100:
            self.successful_sequences = self.successful_sequences[-100:]
        if len(self.failed_sequences) > 100:
            self.failed_sequences = self.failed_sequences[-100:]
        
        # Train PFC on successful sequences
        if quality_score > 0.7 and quality_score > 0.6:
            try:
                pfc = self._get_pfc()
                if pfc and pfc._running:
                    # Create concept sequence for PFC training
                    from core.prefrontal_cortex import Concept, ConceptType
                    
                    pfc_concepts = []
                    for concept in concepts:
                        c = Concept(
                            id=concept.id,
                            embedding=concept.embedding if concept.embedding is not None else np.zeros(768),
                            concept_type=ConceptType.MUSIC_TOKEN,
                            content=concept.concept_type,
                            source=f"composition_{style}",
                            reward=quality_score,
                        )
                        pfc_concepts.append(c)
                    
                    # Store in PFC memory
                    for c in pfc_concepts:
                        pfc.observe(
                            content=c.content,
                            concept_type=c.concept_type,
                            source=c.source,
                        )
                        
            except Exception as e:
                logger.debug(f"PFC training from composition failed: {e}")
    
    def get_concept_sequence_for_section(
        self,
        section: str,
        form: str,
        position: float,
        style: str
    ) -> list[str]:
        """Get recommended concept sequence for a section.
        
        Based on successful compositions in history.
        """
        # Filter successful sequences by style and form
        relevant = [
            s for s in self.successful_sequences
            if s["style"] == style
        ]
        
        if not relevant:
            # Default concept sequence based on theory
            return self._default_concept_sequence(section, form, position)
        
        # Extract concepts from successful sequences
        # Prioritize concepts that helped in similar sections
        concept_counts = defaultdict(int)
        
        for seq in relevant:
            concepts = seq["concepts"]
            for concept in concepts:
                # Weight by position similarity
                seq_position = concepts.index(concept) / max(1, len(concepts))
                position_weight = 1.0 - abs(position - seq_position)
                concept_counts[concept] += position_weight
        
        # Return sorted by effectiveness
        sorted_concepts = sorted(
            concept_counts.keys(),
            key=lambda c: concept_counts[c],
            reverse=True
        )
        
        return sorted_concepts[:5] if sorted_concepts else self._default_concept_sequence(section, form, position)
    
    def _default_concept_sequence(self, section: str, form: str, position: float) -> list[str]:
        """Default concept sequence based on music theory."""
        if "sonata" in form.lower():
            if "exposition" in section.lower():
                return ["exposition_start", "tonic_establishment", "theme_statement"]
            elif "development" in section.lower():
                return ["development_open", "harmonic_tension", "development_climax"]
            else:
                return ["recapitulation_return", "theme_statement", "harmonic_resolution"]
        elif "ternary" in form.lower():
            if position < 0.33:
                return ["theme_statement", "tonic_establishment"]
            elif position < 0.67:
                return ["theme_development", "harmonic_tension", "development_climax"]
            else:
                return ["theme_statement", "harmonic_resolution", "peaceful_resolution"]
        else:
            return ["theme_statement", "theme_development", "harmonic_resolution"]
    
    def get_next_concept_suggestion(
        self,
        current_position: float,
        form: str,
        style: str,
        current_concepts: list[MusicalConcept]
    ) -> dict:
        """Get suggestion for next musical concept.
        
        Main entry point for LCM-guided composition.
        
        Args:
            current_position: Position in composition (0-1)
            form: Musical form
            style: Style
            current_concepts: Concepts used so far
            
        Returns:
            Dictionary with concept suggestion and parameters
        """
        # Predict next concept
        predicted_concept, confidence = self.predict_next_concept(
            current_concepts,
            {"style": style, "form": form, "position": current_position}
        )
        
        # Decode to parameters
        params = self.decode_to_composition_params(
            predicted_concept,
            {"style": style, "form": form, "position": current_position}
        )
        
        # Add confidence
        params["confidence"] = confidence
        
        # Get alternatives
        alternatives = self.get_concept_sequence_for_section(
            section="current",
            form=form,
            position=current_position,
            style=style
        )
        params["alternative_concepts"] = alternatives[:3]
        
        return params


def get_music_lcm() -> MusicLCM:
    """Get singleton MusicLCM instance."""
    global _music_lcm
    if _music_lcm is None:
        with _lcm_lock:
            if _music_lcm is None:
                _music_lcm = MusicLCM()
    return _music_lcm