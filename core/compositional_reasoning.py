"""Compositional Reasoning System - Explains WHY musical choices are made.

This module provides structured reasoning for each composition, explaining:
1. Key selection and emotional character
2. Form choice and structural rationale
3. Progression selection and harmonic logic
4. Melodic decisions and motivic development
5. Texture and register assignments
6. Master composer influences

The reasoning is stored with each composition for LCM learning and
provides context for future compositional decisions.
"""
import logging
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Key emotional associations from music theory and history
KEY_ASSOCIATIONS = {
    "C_major": {
        "emotional": "Pure, innocent, triumphal, simple",
        "historical": "Used by Mozart for clarity, Bach for preludes",
        "character": "Pastoral, bright, open",
        "examples": ["Bach Prelude in C", "Mozart Sonata K.545", "Beethoven Symphony 1"],
        "difficulty": 1.0,  # No sharps/flats
        "common_modulations": ["G major", "F major", "A minor"],
    },
    "C_minor": {
        "emotional": "Tragic, heroic, fate-driven, struggle",
        "historical": "Beethoven's 'fate' key, used for dramatic struggles",
        "character": "Dark, serious, powerful",
        "examples": ["Beethoven Sonata Pathetique", "Mozart Piano Concerto 24", "Bach Passacaglia"],
        "difficulty": 1.5,
        "common_modulations": ["Eb major", "G major", "Ab major"],
    },
    "G_major": {
        "emotional": "Pastoral, bright, rustic, joyful",
        "historical": "Baroque trumpet key, pastoral tradition",
        "character": "Rustic, cheerful, natural",
        "examples": ["Mozart Eine kleine Nachtmusik", "Bach Brandenburg 3", "Beethoven Piano Concerto 4"],
        "difficulty": 1.2,
        "common_modulations": ["D major", "E minor", "C major"],
    },
    "D_major": {
        "emotional": "Triumphant, majestic, celebratory",
        "historical": "Trumpet key, baroque celebration",
        "character": "Majestic, ceremonial, bright",
        "examples": ["Bach Brandenburg 5", "Mozart Haffner Symphony", "Beethoven Symphony 2"],
        "difficulty": 2.0,
        "common_modulations": ["A major", "B minor", "G major"],
    },
    "D_minor": {
        "emotional": "Serious, intense, dramatic, doom-laden",
        "historical": "Used for Requiems and serious works",
        "character": "Intense, dramatic, severe",
        "examples": ["Bach Chaconne", "Mozart Requiem", "Beethoven Symphony 9 'Ode to Joy' chorus"],
        "difficulty": 2.0,
        "common_modulations": ["F major", "A minor", "Bb major"],
    },
    "A_major": {
        "emotional": "Bright, confident, warm",
        "historical": "Favoured by Mozart for clarity",
        "character": "Warm, lyrical, bright",
        "examples": ["Mozart Piano Concerto 23", "Beethoven Symphony 7", "Schubert Piano Sonata"],
        "difficulty": 3.0,  # 3 sharps
        "common_modulations": ["E major", "F# minor", "D major"],
    },
    "A_minor": {
        "emotional": "Poignant, melancholy, reflective",
        "historical": "Used for expressive solo works",
        "character": "Introspective, sad, deep",
        "examples": ["Bach Solo Violin Sonata", "Mozart Piano Sonata K.310", "Beethoven Fur Elise"],
        "difficulty": 2.5,
        "common_modulations": ["C major", "E minor", "G major"],
    },
    "E_major": {
        "emotional": "Bright, extroverted, heroic",
        "historical": "Liszt's heroic key",
        "character": "Brilliant, warm, heroic",
        "examples": ["Beethoven Sonata 'Em' ", "Liszt Hungarian Rhapsody", "Chopin Etudes"],
        "difficulty": 4.0,  # 4 sharps
        "common_modulations": ["B major", "G# minor", "A major"],
    },
    "E_minor": {
        "emotional": "Dramatic, passionate, fate",
        "historical": "Works on fate and destiny",
        "character": "Passionate, dramatic, serious",
        "examples": ["Bach Violin Concerto", "Mendelssohn Violin Concerto", "Chopin Piano Concerto 1"],
        "difficulty": 3.5,
        "common_modulations": ["G major", "B minor", "C major"],
    },
    "Eb_major": {
        "emotional": "Noble, heroic, majestic",
        "historical": "Beethoven's 'Eroica' key",
        "character": "Heroic, noble, grand",
        "examples": ["Beethoven Eroica Symphony", "Mozart Symphony 39", "Haydn Sonata"],
        "difficulty": 3.0,  # 3 flats
        "common_modulations": ["Bb major", "C minor", "Ab major"],
    },
    "Bb_major": {
        "emotional": "Warm, open, pastoral",
        "historical": "Wind instrument favorite",
        "character": "Pastoral, warm, relaxed",
        "examples": ["Mozart Piano Concerto 27", "Beethoven Symphony 4", "Schubert Unfinished"],
        "difficulty": 2.0,  # 2 flats
        "common_modulations": ["F major", "G minor", "Eb major"],
    },
    "F#_major": {
        "emotional": "Brilliant, sharp, intense",
        "historical": "Liszt's heroic key, Chopin's bravura",
        "character": "Brilliant, virtuosic, intense",
        "examples": ["Chopin Barcarolle", "Liszt Hungarian Rhapsodies"],
        "difficulty": 5.0,  # Very sharp
        "common_modulations": ["Db major", "B minor", "G# minor"],
    },
}

# Form characteristics and rationale
FORM_ASSOCIATIONS = {
    "binary": {
        "description": "Two-part form (AB) with modulation and return",
        "historical_origin": "Baroque dance movements, early keyboard music",
        "structural_logic": "Statement (A) and contrast/departure (B) with return",
        "compositional_benefits": ["Clear structure", "Modulation opportunity", "Balanced proportions"],
        "master_examples": ["Bach Two-Part Inventions", "Scarlatti Sonatas", "Handel Dance Movements"],
        "typical_length_bars": "16-32",
        "key_relationship": "B section typically modulates to dominant or relative major",
        "structural_benefits": "Short, clear, good for developmental exploration",
        "development_potential": "Moderate - limited time for development",
    },
    "ternary": {
        "description": "Three-part form (ABA) with return",
        "historical_origin": "Song form, character pieces",
        "structural_logic": "Statement (A), departure (B), return (A')",
        "compositional_benefits": ["Strong return", "Emotional contrast", "Formal clarity"],
        "master_examples": ["Chopin Nocturnes", "Schumann Character Pieces", "Brahms Intermezzos"],
        "typical_length_bars": "24-48",
        "key_relationship": "B section often contrasts in mode or key",
        "structural_benefits": "Satisfying return, clear emotional arc",
        "development_potential": "Moderate to high - B section can develop",
    },
    "sonata": {
        "description": "Sonata form with exposition, development, recapitulation",
        "historical_origin": "Classical period, perfected by Haydn and Mozart",
        "structural_logic": "Exposition (present themes in tonic/dominant), Development (explore), Recapitulation (resolve in tonic)",
        "compositional_benefits": ["Maximum development", "Dramatic arc", "Large-scale structure"],
        "master_examples": ["Mozart Piano Sonatas", "Beethoven Symphonies", "Haydn String Quartets"],
        "typical_length_bars": "80-200+",
        "key_relationship": "Exposition: tonic → dominant/relative major. Development: explore keys. Recapitulation: all in tonic",
        "structural_benefits": "Complete dramatic arc, motivic development, tonal journey",
        "development_potential": "Highest - dedicated development section",
    },
    "rondo": {
        "description": "Recurring theme with contrasting episodes (ABACA or ABACABA)",
        "historical_origin": "Baroque ritornello, Classical finales",
        "structural_logic": "Main theme returns after each contrasting episode",
        "compositional_benefits": ["Memorable theme", "Variety within unity", "Popular appeal"],
        "master_examples": ["Mozart Piano Concertos", "Beethoven Rondo a Capriccio", "Haydn String Quartet Finales"],
        "typical_length_bars": "60-120",
        "key_relationship": "A sections in tonic, episodes in contrasting keys",
        "structural_benefits": "Clear return structure, episodic variety",
        "development_potential": "Moderate - episodes can develop but limited",
    },
    "theme_variations": {
        "description": "Theme followed by varied repetitions",
        "historical_origin": "Renaissance divisions, Baroque chaconne",
        "structural_logic": "Each variation transforms the theme while keeping its essence",
        "compositional_benefits": ["Transformation showcase", "Unity and variety", "Progressive revelation"],
        "master_examples": ["Bach Goldberg Variations", "Beethoven Diabelli", "Brahms Haydn Variations"],
        "typical_length_bars": "Varies widely",
        "key_relationship": "Usually stays in tonic, may modulate in later variations",
        "structural_benefits": "Systematic exploration of theme's possibilities",
        "development_potential": "High - each variation is a development",
        "transformation_types": ["ornamental", "harmonic", "rhythmic", "contrapuntal", "textural"],
    },
    "invention": {
        "description": "Two-voice contrapuntal work with motivic development",
        "historical_origin": "J.S. Bach's teaching pieces",
        "structural_logic": "Single motive developed through imitation and invertible counterpoint",
        "compositional_benefits": ["Counterpoint mastery", "Motivic unity", "Technical development"],
        "master_examples": ["Bach Two-Part Inventions", "Bach Three-Part Sinfonias"],
        "typical_length_bars": "16-32",
        "key_relationship": "Modulates through closely related keys",
        "structural_benefits": "Imitation, tight structure, clear logic",
        "development_potential": "High - contrapuntal development throughout",
    },
    "fugue": {
        "description": "Polyphonic form with subject entries and development",
        "historical_origin": "Baroque, perfected by Bach",
        "structural_logic": "Exposition, development, stretto, pedal point, coda",
        "compositional_benefits": ["Contrapuntal mastery", "Structural complexity", "Intellectual depth"],
        "master_examples": ["Bach Well-Tempered Clavier", "Art of Fugue"],
        "typical_length_bars": "50-200+",
        "key_relationship": "Exposition covers tonic and dominant, development explores",
        "structural_benefits": "Maximum contrapuntal development",
        "development_potential": "Highest - contrapuntal development throughout",
    },
    "prelude": {
        "description": "Free-form introduction or standalone piece",
        "historical_origin": "Lute preludes, organ introductions",
        "structural_logic": "Improvisatory, sets mood or key",
        "compositional_benefits": ["Freedom", "Mood setting", "Key exploration"],
        "master_examples": ["Bach Well-Tempered Clavier", "Chopin Preludes", "Debussy Preludes"],
        "typical_length_bars": "8-40",
        "key_relationship": "Establishes key or explores freely",
        "structural_benefits": "Flexible, mood-focused",
        "development_potential": "Variable - can be minimal or extensive",
    },
    "nocturne": {
        "description": "Character piece evoking night atmosphere",
        "historical_origin": "Field, developed by Chopin",
        "structural_logic": "Lyrical melody over arpeggiated accompaniment",
        "compositional_benefits": ["Expressive freedom", "Melodic focus", "Atmospheric"],
        "master_examples": ["Chopin Nocturnes", "Field Nocturnes", "Faure Nocturnes"],
        "typical_length_bars": "40-80",
        "key_relationship": "May modulate to contrasting keys in middle section",
        "structural_benefits": "Clear melody-accompaniment texture, emotional depth",
        "development_potential": "Moderate - focus on melody, not development",
    },
}

# Style characteristics and rationale  
STYLE_ASSOCIATIONS = {
    "baroque": {
        "era": "1600-1750",
        "key_composers": ["Bach", "Handel", "Vivaldi", "Scarlatti"],
        "stylistic_traits": [
            "Contrapuntal texture",
            "Basso continuo",
            "Ornamentation",
            "Terraced dynamics",
            "Motor rhythms",
            "Figured bass",
        ],
        "harmonic_practices": [
            "Functional harmony establishment",
            "Circle of fifths progressions",
            "Secondary dominants",
            "Modulation to closely related keys",
        ],
        "melodic_characteristics": [
            "Stepwise motion with occasional leaps",
            "Sequence-driven development",
            "Motivic transformation",
            "Ornamentation (trills, mordents, turns)",
        ],
        "rhythmic_characteristics": [
            "Continuous motor rhythm",
            "Consistent tempo",
            "Regular phrase lengths",
        ],
        "formal_conventions": {
            "binary": "Most common for dance movements",
            "ternary": "Aria form",
            "fugue": "Contrapuntal apex",
            "invention": "Teaching pieces",
        },
        "compositional_reasoning": {
            "texture_rationale": "Polyphony demonstrates contrapuntal skill and creates harmonic richness through independent voices",
            "harmony_rationale": "Functional harmony provides structural clarity; circle of fifths creates smooth voice leading",
            "melody_rationale": "Stepwise motion ensures singable lines; sequences provide development without losing unity",
            "rhythm_rationale": "Motor rhythm creates forward momentum and structural cohesion",
        },
    },
    "classical": {
        "era": "1750-1820",
        "key_composers": ["Mozart", "Haydn", "Early Beethoven", "Clementi"],
        "stylistic_traits": [
            "Homophonic texture",
            "Clear phrases",
            "Balanced structure",
            "Gradual dynamics",
            "Alberti bass",
            "Sonata form",
        ],
        "harmonic_practices": [
            "Clear harmonic rhythm",
            "Symmetrical phrases",
            "Predominant-dominant-tonic logic",
            "Secondary dominants for color",
        ],
        "melodic_characteristics": [
            "Balanced phrases (4+4)",
            "Singing quality",
            "Grace notes",
            "Clear phrasing",
        ],
        "rhythmic_characteristics": [
            "Clear meter",
            "Phrase clarity",
            "Regular phrase lengths",
        ],
        "formal_conventions": {
            "sonata": "Large-scale structure with development",
            "ternary": "Minuet and trio",
            "rondo": "Finale form",
            "theme_variations": "Second movement",
        },
        "compositional_reasoning": {
            "texture_rationale": "Homophony provides clear melody with supportive harmony; Alberti bass maintains momentum",
            "harmony_rationale": "Clear harmonic rhythm lets melody shine; V-I relationships create strong tonal center",
            "melody_rationale": "Balanced phrases create symmetry; singing quality ensures memorability",
            "rhythm_rationale": "Clear meter provides structural foundation; regular phrases create expectation",
        },
    },
    "romantic": {
        "era": "1820-1900",
        "key_composers": ["Chopin", "Brahms", "Schumann", "Liszt", "Late Beethoven"],
        "stylistic_traits": [
            "Expressive harmony",
            "Chromaticism",
            "Rubato",
            "Expanded forms",
            "Programmatic elements",
            "Virtuosity",
        ],
        "harmonic_practices": [
            "Chromatic modulation",
            "Enharmonic reinterpretation",
            "Extended tertian harmony",
            "Mediant relationships",
        ],
        "melodic_characteristics": [
            "Long-breathed phrases",
            "Ornamented lines",
            "Vocal inspiration",
            "Emotional depth",
        ],
        "rhythmic_characteristics": [
            "Rubato",
            "Tempo flexibility",
            "Cross-rhythms",
        ],
        "formal_conventions": {
            "sonata": "Expanded development",
            "theme_variations": "Character variations",
            "nocturne": "Character piece",
            "ballade": "Narrative form",
        },
        "compositional_reasoning": {
            "texture_rationale": "Thicker textures express emotional intensity; varied accompaniment patterns support melody",
            "harmony_rationale": "Chromaticism creates emotional tension; modulation explores distant keys for expressive effect",
            "melody_rationale": "Long phrases express sustained emotion; ornaments add emotional nuance",
            "rhythm_rationale": "Rubato allows expressive timing; flexibility mirrors emotional content",
        },
    },
    "modern": {
        "era": "1900-present",
        "key_composers": ["Debussy", "Stravinsky", "Bartok", "Schoenberg"],
        "stylistic_traits": [
            "Extended harmony",
            "Irregular meter",
            "Tone clusters",
            "Atonality options",
            "World music influence",
        ],
        "harmonic_practices": [
            "Non-functional harmony",
            "Polytonality",
            "Tone rows",
            "Pandiatonicism",
        ],
        "melodic_characteristics": [
            "Angular lines",
            "Wide intervals",
            "Fragmentation",
            "Non-traditional scales",
        ],
        "rhythmic_characteristics": [
            "Irregular meter",
            "Accent displacement",
            "Polyrhythm",
        ],
        "formal_conventions": {
            "free": "No standardized forms",
            "sonata": "Deconstructed",
            "theme_variations": "Abstract variations",
        },
        "compositional_reasoning": {
            "texture_rationale": "Varied textures create interest without traditional harmony; layering creates color",
            "harmony_rationale": "Non-functional harmony creates novel sounds; extended techniques expand possibilities",
            "melody_rationale": "Angular lines break from tradition; fragmentation reflects modern sensibility",
            "rhythm_rationale": "Irregular meter creates unpredictability; complexity reflects modern complexity",
        },
    },
}


@dataclass
class CompositionalDecision:
    """A single compositional decision with reasoning."""
    decision_type: str  # 'key', 'form', 'progression', 'melody', 'rhythm', 'texture'
    choice: str  # What was chosen
    alternatives_considered: list[str]  # Other options
    rationale: str  # Why this choice
    master_reference: str  # Which composer/work inspired this
    expected_effect: str  # What effect this should create
    confidence: float  # How confident in this choice (0-1)


@dataclass
class CompositionalReasoning:
    """Complete reasoning for a composition."""
    composition_id: str
    style: str
    form: str
    key: str
    
    # Primary decisions
    key_reasoning: CompositionalDecision
    form_reasoning: CompositionalDecision
    
    # Structural reasoning
    structural_decisions: list[CompositionalDecision] = field(default_factory=list)
    
    # Progression reasoning per section
    progression_reasoning: dict[str, CompositionalDecision] = field(default_factory=dict)
    
    # Melodic reasoning per theme
    melody_reasoning: dict[str, CompositionalDecision] = field(default_factory=dict)
    
    # Master influences
    master_influences: list[str] = field(default_factory=list)
    
    # Overall artistic intent
    artistic_intent: str = ""
    
    # Learning points for future
    lessons_learned: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "composition_id": self.composition_id,
            "style": self.style,
            "form": self.form,
            "key": self.key,
            "key_reasoning": {
                "choice": self.key_reasoning.choice,
                "rationale": self.key_reasoning.rationale,
                "master_reference": self.key_reasoning.master_reference,
            },
            "form_reasoning": {
                "choice": self.form_reasoning.choice,
                "rationale": self.form_reasoning.rationale,
                "master_reference": self.form_reasoning.master_reference,
            },
            "structural_decisions": [
                {
                    "type": d.decision_type,
                    "choice": d.choice,
                    "rationale": d.rationale,
                    "master_reference": d.master_reference,
                }
                for d in self.structural_decisions
            ],
            "progression_reasoning": {
                section: {
                    "choice": d.choice,
                    "rationale": d.rationale,
                }
                for section, d in self.progression_reasoning.items()
            },
            "melody_reasoning": {
                theme: {
                    "choice": d.choice,
                    "rationale": d.rationale,
                }
                for theme, d in self.melody_reasoning.items()
            },
            "master_influences": self.master_influences,
            "artistic_intent": self.artistic_intent,
            "lessons_learned": self.lessons_learned,
        }


class CompositionalReasoner:
    """Generate detailed reasoning for compositional choices."""
    
    def __init__(self):
        self.key_data = KEY_ASSOCIATIONS
        self.form_data = FORM_ASSOCIATIONS
        self.style_data = STYLE_ASSOCIATIONS
    
    def generate_key_reasoning(
        self,
        chosen_key: str,
        style: str,
        emotional_character: str
    ) -> CompositionalDecision:
        """Generate reasoning for key choice."""
        
        # Normalize key name
        key_normalized = chosen_key.replace("_major", "").replace("_minor", "")
        is_minor = "minor" in chosen_key.lower()
        key_full = f"{key_normalized}_{'minor' if is_minor else 'major'}"
        
        # Get key associations
        key_info = self.key_data.get(key_full, {})
        
        # Determine rationale
        rationale_parts = []
        
        # Emotional match
        emotional = key_info.get("emotional", "")
        if emotional:
            rationale_parts.append(f"Key {key_full} has {emotional} character, matching the {emotional_character} mood")
        
        # Historical context
        examples = key_info.get("examples", [])
        if examples:
            rationale_parts.append(f"Historically used in {', '.join(examples[:2])}")
        
        # Difficulty/practicality
        difficulty = key_info.get("difficulty", 2.0)
        if difficulty < 2:
            rationale_parts.append("Good key for readability (few accidentals)")
        elif difficulty > 3:
            rationale_parts.append("Chosen for distinctive color")
        
        # Modulation possibilities
        modulations = key_info.get("common_modulations", [])
        if modulations:
            rationale_parts.append(f"Can effectively modulate to {', '.join(modulations[:3])}")
        
        rationale = "; ".join(rationale_parts) if rationale_parts else f"Key {key_full} chosen for its character"
        
        # Choose master reference
        master_ref = examples[0] if examples else f"Typical {style} usage"
        
        return CompositionalDecision(
            decision_type="key",
            choice=key_full,
            alternatives_considered=self._get_alternative_keys(emotional_character, is_minor),
            rationale=rationale,
            master_reference=master_ref,
            expected_effect=key_info.get("character", "Clear tonal center"),
            confidence=0.8 if key_info else 0.5,
        )
    
    def generate_form_reasoning(
        self,
        chosen_form: str,
        style: str,
        intended_length: int,
        development_needs: str
    ) -> CompositionalDecision:
        """Generate reasoning for form choice."""
        
        form_normalized = chosen_form.lower().replace("_", "")
        form_info = self.form_data.get(form_normalized, self.form_data.get(chosen_form.lower(), {}))
        
        rationale_parts = []
        
        # Structural logic
        structural_logic = form_info.get("structural_logic", "")
        if structural_logic:
            rationale_parts.append(structural_logic)
        
        # Benefits for this style
        benefits = form_info.get("compositional_benefits", [])
        if benefits:
            rationale_parts.append(f"Provides {', '.join(benefits[:2])}")
        
        # Master examples
        examples = form_info.get("master_examples", [])
        
        # Development potential
        dev_potential = form_info.get("development_potential", "")
        if development_needs == "high" and "Highest" in dev_potential or "High" in dev_potential:
            rationale_parts.append(f"High development potential: {dev_potential}")
        
        # Length match
        typical_length = form_info.get("typical_length_bars", "")
        rationale_parts.append(f"Typical length: {typical_length} bars")
        
        # Key relationship
        key_rel = form_info.get("key_relationship", "")
        if key_rel:
            rationale_parts.append(f"Key structure: {key_rel}")
        
        rationale = "; ".join(rationale_parts) if rationale_parts else f"Form {chosen_form} chosen for structural clarity"
        
        return CompositionalDecision(
            decision_type="form",
            choice=chosen_form,
            alternatives_considered=self._get_alternative_forms(development_needs),
            rationale=rationale,
            master_reference=examples[0] if examples else f"Standard {style} form",
            expected_effect=form_info.get("structural_benefits", "Clear structure"),
            confidence=0.85,
        )
    
    def generate_progression_reasoning(
        self,
        progression: list[str],
        section: str,
        style: str,
        position: float
    ) -> CompositionalDecision:
        """Generate reasoning for chord progression choice."""
        
        # Analyze progression
        progression_str = " - ".join(progression)
        
        rationale_parts = []
        
        # Functional analysis
        has_i_v = "I" in progression[:2] and "V" in progression[:3]
        has_circle = any(ii in progression for ii in ["ii", "iii", "vi"])
        has_modulation = any("/" in chord for chord in progression)
        
        if section == "exposition" or position < 0.3:
            if has_i_v:
                rationale_parts.append("Establishes tonal center with I-V relationship")
            if has_circle:
                rationale_parts.append("Circle progression creates smooth voice leading")
            
        elif section == "development" or 0.3 <= position < 0.7:
            if has_modulation:
                rationale_parts.append("Secondary dominants create developmental tension")
            rationale_parts.append("Development section explores harmonic territory")
            
        else:  # Recapitulation/coda
            rationale_parts.append("Resolution section returns to stability")
        
        # Style-specific reasoning
        style_info = self.style_data.get(style.lower(), {})
        harmonic_practices = style_info.get("harmonic_practices", [])
        if harmonic_practices:
            rationale_parts.append(f"Follows {style} practice: {harmonic_practices[0]}")
        
        rationale = "; ".join(rationale_parts) if rationale_parts else "Functional harmony progression"
        
        return CompositionalDecision(
            decision_type="progression",
            choice=progression_str,
            alternatives_considered=[],
            rationale=rationale,
            master_reference=f"Standard {style} progression",
            expected_effect="Harmonic progression guides emotional journey",
            confidence=0.75,
        )
    
    def generate_melody_reasoning(
        self,
        theme_type: str,
        style: str,
        phrase_shape: str,
        register: str
    ) -> CompositionalDecision:
        """Generate reasoning for melodic choices."""
        
        rationale_parts = []
        
        # Phrase shape reasoning
        if "arch" in phrase_shape.lower():
            rationale_parts.append("Arch shape (rise-fall) creates natural musical sentence")
        elif "ascending" in phrase_shape.lower():
            rationale_parts.append("Ascending motion builds energy and direction")
        elif "descending" in phrase_shape.lower():
            rationale_parts.append("Descending motion creates relaxation and resolution")
        
        # Register reasoning
        register_reasons = {
            "low": "Low register (cello range) provides foundation and warmth",
            "middle": "Middle register (piano range) offers balance and clarity",
            "high": "High register (piano RH) provides brightness and presence",
        }
        rationale_parts.append(register_reasons.get(register, "Register chosen for clarity"))
        
        # Style-specific melody traits
        style_info = self.style_data.get(style.lower(), {})
        melodic_traits = style_info.get("melodic_characteristics", [])
        if melodic_traits:
            rationale_parts.append(f"{style.title()} style: {melodic_traits[0]}")
        
        # Theme type reasoning
        theme_reasons = {
            "primary": "Primary theme establishes main musical idea",
            "secondary": "Secondary theme provides contrast and development",
            "development": "Development transforms motives through sequence",
        }
        if theme_type in theme_reasons:
            rationale_parts.append(theme_reasons[theme_type])
        
        rationale = "; ".join(rationale_parts) if rationale_parts else "Melodic choices serve structural function"
        
        examples = style_info.get("master_examples", [])
        
        return CompositionalDecision(
            decision_type="melody",
            choice=f"{theme_type} theme in {register} register",
            alternatives_considered=[],
            rationale=rationale,
            master_reference=examples[0] if examples else f"{style} melodic style",
            expected_effect="Melody guides emotional narrative",
            confidence=0.70,
        )
    
    def generate_complete_reasoning(
        self,
        composition_id: str,
        style: str,
        form: str,
        key: str,
        progressions: dict[str, list[str]],
        emotional_character: str,
        themes: list[str]
    ) -> CompositionalReasoning:
        """Generate complete reasoning for a composition."""
        
        # Key reasoning
        key_reasoning = self.generate_key_reasoning(key, style, emotional_character)
        
        # Form reasoning
        development_needs = "high" if form in ["sonata", "fugue", "invention"] else "moderate"
        form_reasoning = self.generate_form_reasoning(form, style, 60, development_needs)
        
        # Create reasoning object
        reasoning = CompositionalReasoning(
            composition_id=composition_id,
            style=style,
            form=form,
            key=key,
            key_reasoning=key_reasoning,
            form_reasoning=form_reasoning,
        )
        
        # Progression reasoning per section
        for section, prog in progressions.items():
            position = {"exposition": 0.25, "development": 0.5, "recapitulation": 0.75}.get(section, 0.5)
            reasoning.progression_reasoning[section] = self.generate_progression_reasoning(
                prog, section, style, position
            )
        
        # Melody reasoning
        for i, theme in enumerate(themes):
            theme_type = "primary" if i == 0 else "secondary" if i == 1 else "development"
            reasoning.melody_reasoning[theme] = self.generate_melody_reasoning(
                theme_type, style, "arch", "middle"
            )
        
        # Master influences
        style_info = self.style_data.get(style.lower(), {})
        reasoning.master_influences = style_info.get("key_composers", [])[:2]
        
        # Artistic intent
        key_info = self.key_data.get(key.replace("_", ""), {})
        emotional = key_info.get("emotional", emotional_character)
        form_info = self.form_data.get(form.lower(), {})
        structural = form_info.get("structural_logic", "")
        
        reasoning.artistic_intent = (
            f"This composition in {style} style uses {form} form {structural.lower()}. "
            f"The key of {key} evokes {emotional.lower()}. "
            f"Following {reasoning.master_influences[0] if reasoning.master_influences else 'traditional'} style, "
            f"the work aims to {emotional_character.lower()}."
        )
        
        return reasoning
    
    def _get_alternative_keys(self, emotional_character: str, is_minor: bool) -> list[str]:
        """Get alternative keys for similar emotional character."""
        alternatives = []
        
        # Based on emotional character
        if "dramatic" in emotional_character.lower() or "tragic" in emotional_character.lower():
            alternatives = ["C_minor", "D_minor", "F#_minor"] if is_minor else ["C_minor", "Bb_major"]
        elif "bright" in emotional_character.lower() or "joyful" in emotional_character.lower():
            alternatives = ["G_major", "D_major", "A_major"] if not is_minor else ["A_minor"]
        elif "peaceful" in emotional_character.lower() or "calm" in emotional_character.lower():
            alternatives = ["F_major", "Bb_major", "Eb_major"] if not is_minor else ["A_minor"]
        else:
            alternatives = ["C_major", "G_major"] if not is_minor else ["A_minor", "D_minor"]
        
        return alternatives
    
    def _get_alternative_forms(self, development_needs: str) -> list[str]:
        """Get alternative forms for similar development needs."""
        if development_needs == "high":
            return ["sonata", "fugue", "theme_variations"]
        elif development_needs == "moderate":
            return ["ternary", "rondo", "prelude"]
        else:
            return ["binary", "prelude", "nocturne"]
    
    def explain_master_reference(
        self,
        master: str,
        technique: str,
        context: str
    ) -> str:
        """Explain how a master composer's work influenced this decision."""
        
        explanations = {
            "Bach": f"In {master}'s tradition: {technique} exemplifies contrapuntal mastery through {context}",
            "Mozart": f"Following {master}'s classical style: {technique} achieves {context} through balanced phrases",
            "Beethoven": f"{master}'s influence: {technique} shows dramatic development through {context}",
            "Chopin": f"{master}'s romantic style: {technique} expresses {context} through lyrical melody",
            "Debussy": f"{master}'s impressionist approach: {technique} creates {context} through color",
            "Brahms": f"{master}'s classical-romantic synthesis: {technique} balances {context} with formal rigor",
        }
        
        return explanations.get(master, f"{master} provides precedent for {technique} through {context}")


_reasoner: CompositionalReasoner | None = None
_reasoner_lock = threading.Lock()


def get_reasoner() -> CompositionalReasoner:
    """Get singleton compositional reasoner."""
    global _reasoner
    if _reasoner is None:
        with _reasoner_lock:
            if _reasoner is None:
                _reasoner = CompositionalReasoner()
    return _reasoner