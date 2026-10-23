"""Music Self-Learning System - Continuous Improvement Through Self-Perception.

This module creates a closed learning loop:
1. COMPOSE → Create music
2. PERCEIVE → "Hear" the music (analyze)
3. EVALUATE → Score against criteria
4. REFINE → Apply learned techniques
5. REMEMBER → Store successful patterns
6. EVOLVE → Update parameters based on success

Key Features:
- Real-time feedback integration during composition
- Technique effectiveness tracking (which techniques improve scores)
- Progressive difficulty advancement
- Cross-style learning transfer
- Self-critique and improvement cycles
- Multi-model orchestration (LLM + LCM + Embeddings)

Architecture:
┌─────────────────────────────────────────────────────────────────────┐
│                    SelfLearningLoop                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Perceive                                                            │
│  - Analyze composition (key, texture, form)                          │
│  - Detect issues (muddy, weak, thin, clashing)                      │
│  - Calculate quality scores                                          │
│  - LLM semantic analysis                                             │
│  - Embedding pattern matching                                         │
├─────────────────────────────────────────────────────────────────────┤
│  Evaluate                                                            │
│  - Compare against master patterns                                    │
│  - Score by technique (counterpoint, harmony, melody, form)          │
│  - Identify specific improvements                                     │
│  - LCM concept encoding                                               │
│  - LLM reasoning                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  Refine                                                              │
│  - Apply targeted techniques from master practice                     │
│  - Adjust parameters based on feedback                                │
│  - Iterate until quality threshold met                                │
│  - LLM suggestions                                                    │
│  - LCM concept guidance                                               │
│  - Embedding pattern lookup                                           │
├─────────────────────────────────────────────────────────────────────┤
│  Remember                                                            │
│  - Store successful patterns in knowledge base                        │
│  - Track technique effectiveness                                      │
│  - Update stylistic preferences                                       │
│  - Train LCM concepts                                                 │
│  - Store embeddings                                                   │
│  - Link knowledge graph                                               │
├─────────────────────────────────────────────────────────────────────┤
│  Evolve                                                              │
│  - Increase technique weights that improve scores                     │
│  - Decrease weights for failing techniques                            │
│  - Adapt style preferences based on success                           │
└─────────────────────────────────────────────────────────────────────┘
"""
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import defaultdict
import random

from core.helpers import atomic_write_json, read_json_with_integrity

logger = logging.getLogger(__name__)

# Singleton
_self_learning_system = None
_lock = threading.Lock()


@dataclass
class TechniqueEffectiveness:
    """Track how effective each technique is for improving compositions."""
    technique_id: str
    technique_name: str
    style: str  # Which style it applies to
    
    # Effectiveness scores
    applications: int = 0
    successful_applications: int = 0
    avg_score_before: float = 0.0
    avg_score_after: float = 0.0
    improvement_rate: float = 0.0  # avg_score_after / avg_score_before
    
    # Context tracking
    best_contexts: list = field(default_factory=list)  # When it works best
    worst_contexts: list = field(default_factory=list)  # When it fails
    
    def to_dict(self) -> dict:
        return {
            "technique_id": self.technique_id,
            "technique_name": self.technique_name,
            "style": self.style,
            "applications": self.applications,
            "successful_applications": self.successful_applications,
            "avg_score_before": self.avg_score_before,
            "avg_score_after": self.avg_score_after,
            "improvement_rate": self.improvement_rate,
            "best_contexts": self.best_contexts,
            "worst_contexts": self.worst_contexts,
        }


@dataclass
class CompositionCritique:
    """Detailed self-critique of a composition."""
    composition_id: str
    style: str
    form: str
    key: str
    
    # Scores by aspect
    melody_score: float = 0.0
    harmony_score: float = 0.0
    rhythm_score: float = 0.0
    form_score: float = 0.0
    overall_score: float = 0.0
    
    # Detected issues
    issues: list = field(default_factory=list)
    
    # Improvement suggestions
    suggestions: list = field(default_factory=list)
    
    # Applied techniques
    techniques_applied: list = field(default_factory=list)
    techniques_successful: list = field(default_factory=list)
    
    # Learning data
    patterns_identified: list = field(default_factory=list)
    master_techniques_referenced: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "composition_id": self.composition_id,
            "style": self.style,
            "form": self.form,
            "key": self.key,
            "melody_score": self.melody_score,
            "harmony_score": self.harmony_score,
            "rhythm_score": self.rhythm_score,
            "form_score": self.form_score,
            "overall_score": self.overall_score,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "techniques_applied": self.techniques_applied,
            "techniques_successful": self.techniques_successful,
            "patterns_identified": self.patterns_identified,
            "master_techniques_referenced": self.master_techniques_referenced,
        }


@dataclass
class LearningSession:
    """A learning session tracking progress."""
    session_id: str
    started: datetime
    style: str
    
    # Progress tracking
    compositions_created: int = 0
    average_score_start: float = 0.0
    average_score_current: float = 0.0
    improvement_delta: float = 0.0
    
    # Technique progress
    techniques_practiced: dict = field(default_factory=dict)
    techniques_mastered: list = field(default_factory=list)
    
    # Insights generated
    insights: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started": self.started.isoformat(),
            "style": self.style,
            "compositions_created": self.compositions_created,
            "average_score_start": self.average_score_start,
            "average_score_current": self.average_score_current,
            "improvement_delta": self.improvement_delta,
            "techniques_practiced": self.techniques_practiced,
            "techniques_mastered": self.techniques_mastered,
            "insights": self.insights,
        }


class MusicSelfLearning:
    """Self-learning system for continuous musical improvement."""
    
    def __init__(self):
        self.storage_path = Path.home() / ".jarvis" / "music_learning"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Technique effectiveness tracking
        self.technique_effectiveness: dict[str, TechniqueEffectiveness] = {}
        
        # Recent critiques for pattern learning
        self.recent_critiques: list[CompositionCritique] = []
        self.MAX_CRITIQUES = 100
        
        # Successful patterns to reuse
        self.successful_patterns: dict[str, list] = defaultdict(list)
        self.MAX_PATTERNS_PER_STYLE = 50
        
        # Learning sessions
        self.current_session: Optional[LearningSession] = None
        self.session_history: list[LearningSession] = []
        self.MAX_SESSIONS = 50
        
        # Technique application tracking
        self.technique_queue: list[dict] = []  # Techniques to try
        self.recently_failed: list[str] = []   # Techniques that didn't work
        
        # Style-specific learning rates
        self.style_learning_rates: dict[str, float] = {
            "baroque": 0.15,
            "classical": 0.12,
            "romantic": 0.10,
            "modern": 0.08,
            "jazz": 0.10,
        }
        
        # Master technique references
        self.master_techniques = self._init_master_techniques()
        
        # Load saved state
        self._load_state()
    
    def _init_master_techniques(self) -> dict:
        """Initialize master composer techniques for reference."""
        return {
            "bach": {
                "counterpoint": [
                    "species_counterpoint_first",
                    "species_counterpoint_second",
                    "invertible_counterpoint",
                    "canonic_imitation",
                    "fugue_exposition",
                ],
                "harmony": [
                    "functional_harmony_bach",
                    "modulation_through_pivot",
                    "circle_of_fifths_progression",
                ],
                "melody": [
                    "motivic_development",
                    "sequence_transposition",
                    "inversion_augmentation",
                ],
            },
            "mozart": {
                "form": [
                    "sonata_exposition",
                    "sonata_development",
                    "sonata_recapitulation",
                    "balanced_phrases",
                ],
                "harmony": [
                    "alberti_bass",
                    "secondary_dominants",
                    "delayed_resolution",
                ],
                "melody": [
                    "singing_style",
                    "grace_notes",
                    "elegant_phrasing",
                ],
            },
            "beethoven": {
                "motivic": [
                    "motivic_fragments",
                    "rhythmic_displacement",
                    "dynamic_contrast",
                    "sudden_forte",
                ],
                "form": [
                    "expanded_sonata",
                    "continuous_development",
                    "dramatic_climax",
                ],
                "harmony": [
                    "remote_modulation",
                    "diminished_sevenths",
                    "harmonic_surprise",
                ],
            },
            "chopin": {
                "melody": [
                    "cantilena_melody",
                    "ornamented_lines",
                    "rubato_phrasing",
                    "bel_canto_style",
                ],
                "harmony": [
                    "chromatic_harmony",
                    "enharmonic_modulation",
                    "extended_tertiary",
                ],
                "form": [
                    "nocturne_form",
                    "ballade_structure",
                ],
            },
            "debussy": {
                "harmony": [
                    "parallel_sevenths",
                    "whole_tone_passages",
                    "non_functional_harmony",
                    "planing_chords",
                ],
                "melody": [
                    "pentatonic_flavor",
                    "modal_melody",
                    "ambiguous_phrasing",
                ],
                "texture": [
                    "layered_textures",
                    "pedal_point",
                    "wash_effect",
                ],
            },
        }
    
    def perceive(self, composition_data: dict, perception_features: dict) -> CompositionCritique:
        """Analyze a composition and generate detailed critique.
        
        This is step 1 of the learning loop: PERCEIVE.
        
        Args:
            composition_data: Composition metadata (style, form, key)
            perception_features: Features from music_perception analysis
            
        Returns:
            CompositionCritique with detailed analysis
        """
        critique = CompositionCritique(
            composition_id=composition_data.get("id", "unknown"),
            style=composition_data.get("style", "classical"),
            form=composition_data.get("form", "ternary"),
            key=composition_data.get("key", "C"),
        )
        
        # Extract scores from perception
        critique.melody_score = perception_features.get("melodic_quality", 0.5)
        critique.harmony_score = perception_features.get("harmonic_quality", 0.5)
        critique.rhythm_score = perception_features.get("rhythmic_quality", 0.5)
        critique.form_score = perception_features.get("structural_quality", 0.5)
        
        # Calculate overall
        critique.overall_score = (
            critique.melody_score * 0.3 +
            critique.harmony_score * 0.25 +
            critique.rhythm_score * 0.2 +
            critique.form_score * 0.25
        )
        
        # Generate specific issues and suggestions
        issues = []
        suggestions = []
        
        # Melody issues
        if critique.melody_score < 0.6:
            issues.append("melody_weak")
            
            melodic_range = perception_features.get("melodic_range", (60, 72))
            range_span = melodic_range[1] - melodic_range[0] if melodic_range else 0
            
            if range_span < 8:
                suggestions.append({
                    "issue": "narrow_range",
                    "suggestion": "Expand melodic range - use 1.5-2 octaves minimum",
                    "technique": "range_expansion",
                    "master_reference": "bach.melody.motivic_development",
                })
            elif range_span > 24:
                suggestions.append({
                    "issue": "excessive_range",
                    "suggestion": "Focus melodic range - stay within 2 octaves",
                    "technique": "range_focus",
                    "master_reference": "mozart.melody.singing_style",
                })
            
            step_ratio = perception_features.get("step_motion_ratio", 0.5)
            if step_ratio < 0.5:
                suggestions.append({
                    "issue": "too_many_leaps",
                    "suggestion": "Use more stepwise motion (65% steps, 30% skips, 5% leaps)",
                    "technique": "smooth_motion",
                    "master_reference": "mozart.melody.elegant_phrasing",
                })
        
        # Harmony issues
        if critique.harmony_score < 0.6:
            issues.append("harmony_weak")
            
            coherence = perception_features.get("coherence_score", 0.5)
            if coherence < 0.5:
                suggestions.append({
                    "issue": "harmonic_incoherence",
                    "suggestion": "Use functional progressions - tonic, dominant, subdominant relationships",
                    "technique": "functional_progression",
                    "master_reference": "bach.harmony.functional_harmony_bach",
                })
            
            resolution = perception_features.get("resolution_score", 0.5)
            if resolution < 0.5:
                suggestions.append({
                    "issue": "poor_resolution",
                    "suggestion": "Ensure phrases end on stable harmonies (tonic or dominant)",
                    "technique": "cadence_clarity",
                    "master_reference": "mozart.form.balanced_phrases",
                })
        
        # Rhythm issues
        if critique.rhythm_score < 0.6:
            issues.append("rhythm_weak")
            
            rhythm_variety = perception_features.get("rhythm_variety", 0.5)
            if rhythm_variety < 0.4:
                suggestions.append({
                    "issue": "monotonous_rhythm",
                    "suggestion": "Vary rhythm - use quarter, eighth, half, dotted notes",
                    "technique": "rhythm_variety",
                    "master_reference": "beethoven.motivic.rhythmic_displacement",
                })
        
        # Form issues
        if critique.form_score < 0.6:
            issues.append("form_weak")
            
            sections = perception_features.get("sections_detected", 1)
            expected_sections = {
                "binary": 2, "ternary": 3, "sonata": 3,
                "rondo": 5, "invention": 1, "prelude": 1,
            }.get(critique.form, 2)
            
            if sections < expected_sections - 1:
                suggestions.append({
                    "issue": "insufficient_sections",
                    "suggestion": f"Form requires {expected_sections} sections, only {sections} detected",
                    "technique": "form_clarity",
                    "master_reference": "mozart.form.sonata_exposition",
                })
        
        critique.issues = issues
        critique.suggestions = suggestions
        
        # Store patterns identified
        critique.patterns_identified = perception_features.get("patterns", [])
        
        # Store critique
        self.recent_critiques.append(critique)
        if len(self.recent_critiques) > self.MAX_CRITIQUES:
            self.recent_critiques = self.recent_critiques[-self.MAX_CRITIQUES:]
        
        return critique
    
    def suggest_improvements(self, critique: CompositionCritique) -> list[dict]:
        """Based on critique, suggest specific master techniques to apply.
        
        This is step 2: EVALUATE and prepare for REFINEMENT.
        
        Args:
            critique: The composition critique
            
        Returns:
            List of techniques to apply with expected improvement
        """
        suggestions = []
        style = critique.style
        
        # Get style-appropriate techniques
        style_techniques = self.master_techniques.get(style, self.master_techniques.get("classical", {}))
        
        # Prioritize by score deficit
        score_deficits = [
            ("melody", 1.0 - critique.melody_score),
            ("harmony", 1.0 - critique.harmony_score),
            ("rhythm", 1.0 - critique.rhythm_score),
            ("form", 1.0 - critique.form_score),
        ]
        score_deficits.sort(key=lambda x: x[1], reverse=True)
        
        # For each deficit, suggest techniques
        for aspect, deficit in score_deficits:
            if deficit < 0.2:  # Small deficit, skip
                continue
            
            # Get techniques for this aspect
            aspect_key = aspect  # melody, harmony, rhythm, form
            techniques = style_techniques.get(aspect_key, [])
            
            if not techniques:
                # Try general techniques
                for master_style, master_techs in self.master_techniques.items():
                    if aspect_key in master_techs:
                        techniques.extend(master_techs[aspect_key])
            
            if techniques:
                # Pick technique based on effectiveness tracking
                best_technique = self._pick_best_technique(techniques, aspect_key, style)
                
                if best_technique:
                    # Check if recently failed
                    if best_technique in self.recently_failed:
                        # Try second best
                        best_technique = self._pick_best_technique(
                            [t for t in techniques if t not in self.recently_failed],
                            aspect_key, style
                        )
                    
                    if best_technique:
                        suggestions.append({
                            "technique": best_technique,
                            "aspect": aspect_key,
                            "expected_improvement": deficit * 0.3,  # Expect 30% of deficit
                            "master_reference": f"{style}.{aspect_key}.{best_technique}",
                            "priority": len(suggestions) + 1,
                        })
        
        # Limit to top 3 techniques per composition
        return suggestions[:3]
    
    def _pick_best_technique(self, techniques: list, aspect: str, style: str) -> Optional[str]:
        """Pick the most effective technique for the given context."""
        if not techniques:
            return None
        
        # Check effectiveness tracking
        best_technique = None
        best_effectiveness = -1
        
        for technique in techniques:
            tech_id = f"{style}_{aspect}_{technique}"
            if tech_id in self.technique_effectiveness:
                effectiveness = self.technique_effectiveness[tech_id].improvement_rate
                if effectiveness > best_effectiveness:
                    best_effectiveness = effectiveness
                    best_technique = technique
        
        # If no tracked effectiveness, pick randomly
        if best_technique is None and techniques:
            best_technique = random.choice(techniques)
        
        return best_technique
    
    def record_technique_result(
        self,
        technique: str,
        aspect: str,
        style: str,
        score_before: float,
        score_after: float,
        success: bool
    ) -> None:
        """Record the result of applying a technique.
        
        This is step 5: REMEMBER - track what works.
        
        Args:
            technique: The technique applied
            aspect: Which aspect (melody, harmony, rhythm, form)
            style: Musical style
            score_before: Score before technique
            score_after: Score after technique
            success: Whether improvement was significant
        """
        tech_id = f"{style}_{aspect}_{technique}"
        
        if tech_id not in self.technique_effectiveness:
            self.technique_effectiveness[tech_id] = TechniqueEffectiveness(
                technique_id=tech_id,
                technique_name=technique,
                style=style,
            )
        
        tech = self.technique_effectiveness[tech_id]
        tech.applications += 1
        
        if success:
            tech.successful_applications += 1
        
        # Update averages
        n = tech.applications
        tech.avg_score_before = (tech.avg_score_before * (n - 1) + score_before) / n
        tech.avg_score_after = (tech.avg_score_after * (n - 1) + score_after) / n
        
        # Calculate improvement rate
        if tech.avg_score_before > 0:
            tech.improvement_rate = (tech.avg_score_after - tech.avg_score_before) / tech.avg_score_before
        
        # Track contexts
        context = {
            "aspect": aspect,
            "style": style,
            "score_before": score_before,
            "score_after": score_after,
            "success": success,
        }
        
        if success:
            tech.best_contexts.append(context)
            # Keep only top contexts
            if len(tech.best_contexts) > 10:
                tech.best_contexts.sort(key=lambda x: x["score_after"], reverse=True)
                tech.best_contexts = tech.best_contexts[:10]
        else:
            tech.worst_contexts.append(context)
            if len(tech.worst_contexts) > 5:
                self.recently_failed.append(tech_id)
                # Keep recently_failed limited
                if len(self.recently_failed) > 20:
                    self.recently_failed = self.recently_failed[-20:]
        
        logger.info(f"[SelfLearning] Technique {technique}: "
                   f"improvement_rate={tech.improvement_rate:.2f}, "
                   f"success_rate={tech.successful_applications}/{tech.applications}")
    
    def get_recommended_techniques(self, style: str, current_scores: dict) -> list[dict]:
        """Get recommended techniques for a composition in progress.
        
        This enables real-time technique application during composition.
        
        Args:
            style: Musical style
            current_scores: Current quality scores by aspect
            
        Returns:
            List of techniques ordered by expected improvement
        """
        recommendations = []
        
        # Find weak aspects
        for aspect, score in current_scores.items():
            if score < 0.7:  # Below threshold
                # Find best techniques for this aspect
                aspect_techniques = []
                
                for tech_id, tech in self.technique_effectiveness.items():
                    if tech.style == style and aspect in tech.technique_name:
                        expected_improvement = (1.0 - score) * tech.improvement_rate
                        aspect_techniques.append({
                            "technique": tech.technique_name,
                            "aspect": aspect,
                            "expected_improvement": expected_improvement,
                            "success_rate": tech.successful_applications / max(1, tech.applications),
                        })
                
                # Sort by expected improvement
                aspect_techniques.sort(key=lambda x: x["expected_improvement"], reverse=True)
                recommendations.extend(aspect_techniques[:2])
        
        # If no tracked techniques, use master defaults
        if not recommendations:
            style_techniques = self.master_techniques.get(style, {})
            for aspect in ["melody", "harmony", "form"]:
                if aspect in style_techniques and current_scores.get(aspect, 1.0) < 0.7:
                    default_tech = style_techniques[aspect][0]
                    recommendations.append({
                        "technique": default_tech,
                        "aspect": aspect,
                        "expected_improvement": 0.2,
                        "success_rate": 0.5,
                        "source": "master_default",
                    })
        
        return recommendations[:5]  # Top 5
    
    def store_successful_pattern(
        self,
        style: str,
        pattern_type: str,
        pattern_data: dict,
        score: float
    ) -> None:
        """Store a pattern that worked well.
        
        Args:
            style: Musical style
            pattern_type: Type of pattern (melody, harmony, rhythm, form)
            pattern_data: The pattern data
            score: Quality score achieved
        """
        key = f"{style}_{pattern_type}"
        
        # Only store if score is good
        if score < 0.6:
            return
        
        pattern_entry = {
            "data": pattern_data,
            "score": score,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.successful_patterns[key].append(pattern_entry)
        
        # Keep best patterns
        if len(self.successful_patterns[key]) > self.MAX_PATTERNS_PER_STYLE:
            self.successful_patterns[key].sort(key=lambda x: x["score"], reverse=True)
            self.successful_patterns[key] = self.successful_patterns[key][:self.MAX_PATTERNS_PER_STYLE]
    
    def get_successful_patterns(self, style: str, pattern_type: str, limit: int = 5) -> list:
        """Retrieve successful patterns to reuse.
        
        Args:
            style: Musical style
            pattern_type: Type of pattern
            limit: Maximum patterns to return
            
        Returns:
            List of successful patterns ordered by score
        """
        key = f"{style}_{pattern_type}"
        patterns = self.successful_patterns.get(key, [])
        
        # Sort by score and return top
        patterns.sort(key=lambda x: x["score"], reverse=True)
        return patterns[:limit]
    
    def start_learning_session(self, style: str) -> LearningSession:
        """Start a new learning session.
        
        Args:
            style: Focus style for this session
            
        Returns:
            New learning session
        """
        session = LearningSession(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            started=datetime.now(),
            style=style,
        )
        
        self.current_session = session
        return session
    
    def end_learning_session(self) -> Optional[dict]:
        """End current learning session and record progress.
        
        Returns:
            Session summary or None if no session active
        """
        if not self.current_session:
            return None
        
        session = self.current_session
        
        # Calculate improvement
        if session.compositions_created > 0:
            session.improvement_delta = session.average_score_current - session.average_score_start
        
        # Store session
        self.session_history.append(session)
        if len(self.session_history) > self.MAX_SESSIONS:
            self.session_history = self.session_history[-self.MAX_SESSIONS:]
        
        # Save state
        self._save_state()
        
        summary = {
            "session_id": session.session_id,
            "style": session.style,
            "compositions_created": session.compositions_created,
            "improvement_delta": session.improvement_delta,
            "techniques_mastered": session.techniques_mastered,
        }
        
        self.current_session = None
        return summary
    
    def update_learning_progress(self, score: float, techniques_used: list) -> None:
        """Update progress in current learning session.
        
        Args:
            score: Quality score of composition
            techniques_used: Techniques applied
        """
        if not self.current_session:
            return
        
        self.current_session.compositions_created += 1
        
        # Update average scores
        n = self.current_session.compositions_created
        if n == 1:
            self.current_session.average_score_start = score
            self.current_session.average_score_current = score
        else:
            self.current_session.average_score_current = (
                self.current_session.average_score_current * (n - 1) + score
            ) / n
        
        # Track techniques
        for tech in techniques_used:
            tech_name = tech if isinstance(tech, str) else tech.get("technique", "unknown")
            if tech_name not in self.current_session.techniques_practiced:
                self.current_session.techniques_practiced[tech_name] = 0
            self.current_session.techniques_practiced[tech_name] += 1
            
            # Mark as mastered if used successfully many times
            if self.current_session.techniques_practiced[tech_name] >= 3:
                if tech_name not in self.current_session.techniques_mastered:
                    self.current_session.techniques_mastered.append(tech_name)
    
    def get_learning_stats(self) -> dict:
        """Get statistics about learning progress.
        
        Returns:
            Dictionary with learning statistics
        """
        # Calculate overall improvement
        total_applications = sum(t.applications for t in self.technique_effectiveness.values())
        total_successes = sum(t.successful_applications for t in self.technique_effectiveness.values())
        
        # Most effective techniques
        effective_techniques = sorted(
            self.technique_effectiveness.values(),
            key=lambda t: t.improvement_rate,
            reverse=True
        )[:10]
        
        # Recent improvement trend
        recent_sessions = self.session_history[-5:] if self.session_history else []
        trend = "improving" if len(recent_sessions) > 1 and recent_sessions[-1].improvement_delta > 0 else "stable"
        
        return {
            "total_techniques_tracked": len(self.technique_effectiveness),
            "total_applications": total_applications,
            "overall_success_rate": total_successes / max(1, total_applications),
            "most_effective_techniques": [
                {
                    "technique": t.technique_name,
                    "style": t.style,
                    "improvement_rate": t.improvement_rate,
                    "applications": t.applications,
                }
                for t in effective_techniques
            ],
            "successful_patterns_stored": sum(len(patterns) for patterns in self.successful_patterns.values()),
            "learning_sessions": len(self.session_history),
            "improvement_trend": trend,
            "current_session": self.current_session.to_dict() if self.current_session else None,
        }
    
    def _save_state(self) -> None:
        """Save learning state to disk."""
        try:
            state_file = self.storage_path / "learning_state.json"
            state = {
                "technique_effectiveness": {
                    k: v.to_dict() for k, v in self.technique_effectiveness.items()
                },
                "successful_patterns": dict(self.successful_patterns),
                "session_history": [s.to_dict() for s in self.session_history],
                "recently_failed": self.recently_failed,
            }
            
            atomic_write_json(state_file, state)
            
        except Exception as e:
            logger.warning(f"Failed to save learning state: {e}")
    
    def _load_state(self) -> None:
        """Load learning state from disk."""
        try:
            state_file = self.storage_path / "learning_state.json"
            
            if not state_file.exists():
                return
            
            state = read_json_with_integrity(state_file)
            
            # Restore technique effectiveness
            for tech_id, tech_data in state.get("technique_effectiveness", {}).items():
                self.technique_effectiveness[tech_id] = TechniqueEffectiveness(
                    technique_id=tech_data["technique_id"],
                    technique_name=tech_data["technique_name"],
                    style=tech_data["style"],
                    applications=tech_data.get("applications", 0),
                    successful_applications=tech_data.get("successful_applications", 0),
                    avg_score_before=tech_data.get("avg_score_before", 0.0),
                    avg_score_after=tech_data.get("avg_score_after", 0.0),
                    improvement_rate=tech_data.get("improvement_rate", 0.0),
                    best_contexts=tech_data.get("best_contexts", []),
                    worst_contexts=tech_data.get("worst_contexts", []),
                )
            
            # Restore patterns
            for key, patterns in state.get("successful_patterns", {}).items():
                self.successful_patterns[key] = patterns
            
            # Restore sessions
            for session_data in state.get("session_history", []):
                session = LearningSession(
                    session_id=session_data["session_id"],
                    started=datetime.fromisoformat(session_data["started"]),
                    style=session_data["style"],
                    compositions_created=session_data.get("compositions_created", 0),
                    average_score_start=session_data.get("average_score_start", 0.0),
                    average_score_current=session_data.get("average_score_current", 0.0),
                    improvement_delta=session_data.get("improvement_delta", 0.0),
                    techniques_practiced=session_data.get("techniques_practiced", {}),
                    techniques_mastered=session_data.get("techniques_mastered", []),
                    insights=session_data.get("insights", []),
                )
                self.session_history.append(session)
            
            self.recently_failed = state.get("recently_failed", [])
            
            logger.info(f"[SelfLearning] Loaded {len(self.technique_effectiveness)} technique records, "
                       f"{len(self.session_history)} sessions")
            
        except Exception as e:
            logger.warning(f"Failed to load learning state: {e}")
    
    async def perceive_orchestrated(
        self,
        composition_data: dict,
        perception_features: dict
    ) -> tuple:
        """Multi-model perception using LLM + LCM + Embeddings.
        
        This extends the basic perceive() with:
        1. LLM semantic analysis
        2. LCM concept extraction
        3. Embedding pattern matching
        4. Combined scoring
        
        Args:
            composition_data: Composition metadata
            perception_features: Features from music_perception
            
        Returns:
            (CompositionCritique, PerceptionResult from orchestrated learning)
        """
        # Start with basic perception
        critique = self.perceive(composition_data, perception_features)
        
        try:
            # Get orchestrated learning system
            from core.orchestrated_learning import get_orchestrated_learning
            
            orchestrated = get_orchestrated_learning()
            
            # Run multi-model perception
            perception_result = await orchestrated.perceive(
                content=composition_data,
                domain="music",
                context=perception_features
            )
            
            # Enhance critique with orchestrated findings
            if perception_result.issues_detected:
                for issue in perception_result.issues_detected[:3]:
                    if issue not in critique.issues:
                        critique.issues.append(issue)
            
            if perception_result.strengths_found:
                for strength in perception_result.strengths_found[:3]:
                    if strength not in critique.suggestions:
                        critique.suggestions.append({
                            "issue": "identified_strength",
                            "suggestion": f"Continue: {strength}",
                            "technique": "reinforce_success",
                            "master_reference": "orchestrated_learning",
                        })
            
            # Update overall score with orchestrated quality
            critique.overall_score = (
                critique.overall_score * 0.6 +
                perception_result.overall_quality * 0.4
            )
            
            # Store concepts for LCM learning
            critique.patterns_identified.extend(perception_result.concepts_extracted)
            
            logger.info(f"[SelfLearning] Orchestrated perception: "
                       f"llm_score={perception_result.llm_analysis.get('quality_score', 0.5):.2f}, "
                       f"embedding_matches={len(perception_result.embedding_matches)}, "
                       f"concepts={len(perception_result.concepts_extracted)}")
            
            return critique, perception_result
            
        except Exception as e:
            logger.debug(f"Orchestrated perception failed, using basic: {e}")
            return critique, None
    
    async def evaluate_orchestrated(
        self,
        composition_data: dict,
        critique: CompositionCritique,
        perception_result
    ) -> dict:
        """Deep evaluation using LLM reasoning + LCM concepts.
        
        Args:
            composition_data: Composition data
            critique: Basic critique
            perception_result: From orchestrated perception
            
        Returns:
            Evaluation dict with reasoning and concepts
        """
        try:
            from core.orchestrated_learning import get_orchestrated_learning
            
            orchestrated = get_orchestrated_learning()
            
            if perception_result is None:
                # Create minimal perception result
                from core.orchestrated_learning import PerceptionResult
                perception_result = PerceptionResult(
                    llm_analysis={"quality_score": critique.overall_score},
                    embedding_matches=[],
                    perception_scores={"quality": critique.overall_score},
                    concepts_extracted=[],
                    issues_detected=critique.issues,
                    strengths_found=[s.get("suggestion", "") for s in critique.suggestions],
                    overall_quality=critique.overall_score,
                )
            
            evaluation = await orchestrated.evaluate(
                content=composition_data,
                perception=perception_result,
                domain="music"
            )
            
            # Enhance critique with evaluation
            critique.master_techniques_referenced.extend(evaluation.best_practices[:3])
            
            logger.info(f"[SelfLearning] Evaluation: "
                       f"improvement_potential={evaluation.improvement_potential:.2f}, "
                       f"concepts={len(evaluation.concept_sequence)}")
            
            return {
                "llm_reasoning": evaluation.llm_reasoning,
                "concept_sequence": evaluation.concept_sequence,
                "quality_breakdown": evaluation.quality_breakdown,
                "improvement_potential": evaluation.improvement_potential,
                "best_practices": evaluation.best_practices,
            }
            
        except Exception as e:
            logger.debug(f"Orchestrated evaluation failed: {e}")
            return {}
    
    async def refine_orchestrated(
        self,
        critique: CompositionCritique,
        evaluation: dict,
        composition_data: dict
    ) -> list[dict]:
        """Create refinement plan using multi-model intelligence.
        
        Args:
            critique: Composition critique
            evaluation: Evaluation dict from orchestrated evaluation
            composition_data: Composition data
            
        Returns:
            List of refinement suggestions
        """
        # Start with basic suggestions
        suggestions = self.suggest_improvements(critique)
        
        try:
            from core.orchestrated_learning import get_orchestrated_learning, EvaluationResult
            
            orchestrated = get_orchestrated_learning()
            
            # Create evaluation result
            if evaluation:
                eval_result = EvaluationResult(
                    llm_reasoning=evaluation.get("llm_reasoning", ""),
                    concept_sequence=evaluation.get("concept_sequence", []),
                    quality_breakdown=evaluation.get("quality_breakdown", {"overall": critique.overall_score}),
                    improvement_potential=evaluation.get("improvement_potential", 0.3),
                    master_comparison={},
                    best_practices=evaluation.get("best_practices", []),
                )
            else:
                eval_result = None
            
            if eval_result:
                refinement = await orchestrated.refine(
                    content=composition_data,
                    evaluation=eval_result,
                    domain="music"
                )
                
                # Combine basic and orchestrated suggestions
                combined = suggestions.copy()
                
                for llm_sug in refinement.llm_suggestions[:3]:
                    combined.append({
                        "issue": llm_sug.get("suggestion", ""),
                        "suggestion": llm_sug.get("suggestion", ""),
                        "technique": llm_sug.get("technique", "llm_suggested"),
                        "expected_improvement": llm_sug.get("expected_impact", 0.1),
                        "source": "orchestrated",
                    })
                
                # Add LCM guidance
                if refinement.lcm_guidance:
                    lcm_concept = refinement.lcm_guidance.get("concept", "")
                    if lcm_concept:
                        combined.append({
                            "issue": "concept_guidance",
                            "suggestion": f"Consider next concept: {lcm_concept}",
                            "technique": lcm_concept,
                            "expected_improvement": refinement.expected_improvement,
                            "source": "lcm",
                        })
                
                # Prioritize all suggestions
                combined.sort(key=lambda x: x.get("expected_improvement", 0), reverse=True)
                
                logger.info(f"[SelfLearning] Refined suggestions: "
                           f"llm={len(refinement.llm_suggestions)}, "
                           f"lcm_concept={refinement.lcm_guidance.get('concept', 'none')}, "
                           f"total={len(combined)}")
                
                return combined[:5]
            
        except Exception as e:
            logger.debug(f"Orchestrated refinement failed: {e}")
        
        return suggestions
    
    async def remember_orchestrated(
        self,
        composition_data: dict,
        critique: CompositionCritique,
        perception_result,
        evaluation: dict,
        refinement_suggestions: list[dict],
        final_score: float
    ) -> None:
        """Store learning in multi-system memory.
        
        Args:
            composition_data: Original composition
            critique: Critique
            perception_result: From orchestrated perception
            evaluation: From orchestrated evaluation
            refinement_suggestions: Applied refinements
            final_score: Final quality score achieved
        """
        try:
            from core.orchestrated_learning import get_orchestrated_learning, RefinementPlan
            
            orchestrated = get_orchestrated_learning()
            
            # Create minimal objects if needed
            if perception_result is None:
                from core.orchestrated_learning import PerceptionResult
                perception_result = PerceptionResult(
                    llm_analysis={"quality_score": final_score},
                    embedding_matches=[],
                    perception_scores={"quality": final_score},
                    concepts_extracted=[],
                    issues_detected=critique.issues,
                    strengths_found=[],
                    overall_quality=final_score,
                )
            
            if evaluation is None:
                evaluation = {
                    "concept_sequence": [],
                    "quality_breakdown": {"overall": final_score},
                    "llm_reasoning": "",
                    "improvement_potential": 0.3,
                    "best_practices": [],
                }
            
            from core.orchestrated_learning import EvaluationResult
            eval_result = EvaluationResult(
                llm_reasoning=evaluation.get("llm_reasoning", ""),
                concept_sequence=evaluation.get("concept_sequence", []),
                quality_breakdown=evaluation.get("quality_breakdown", {"overall": final_score}),
                improvement_potential=evaluation.get("improvement_potential", 0.3),
                master_comparison={},
                best_practices=evaluation.get("best_practices", []),
            )
            
            refinement = RefinementPlan(
                llm_suggestions=refinement_suggestions,
                lcm_guidance={},
                embedding_patterns=[],
                techniques_to_apply=[s.get("technique", "") for s in refinement_suggestions],
                priority_order=[],
                expected_improvement=0.1,
            )
            
            # Store in orchestrated learning
            await orchestrated.remember(
                content=composition_data,
                perception=perception_result,
                evaluation=eval_result,
                refinement=refinement,
                outcome_score=final_score,
                domain="music"
            )
            
            logger.info(f"[SelfLearning] Remembered with orchestrated: score={final_score:.2f}")
            
        except Exception as e:
            logger.debug(f"Orchestrated remember failed: {e}")


def get_self_learning() -> MusicSelfLearning:
    """Get singleton instance of self-learning system."""
    global _self_learning_system
    if _self_learning_system is None:
        with _lock:
            if _self_learning_system is None:
                _self_learning_system = MusicSelfLearning()
    return _self_learning_system