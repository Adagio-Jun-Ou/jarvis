"""Orchestrated Self-Learning - Unified LLM + LCM + Embedding Integration.

This module orchestrates multiple AI systems for deep self-learning:

1. PERCEIVE: Multi-model perception (LLM analysis + perception engine + embedding comparison)
2. EVALUATE: Deep analysis (LLM reasoning + LCM concept encoding + pattern matching)
3. REFINE: Intelligent improvement (LLM suggestions + LCM concept guidance + embedding lookup)
4. REMEMBER: Persistent learning (LCM concept training + embedding storage + knowledge graph)
5. EVOLVE: Continuous improvement (technique tracking + parameter adaptation)

Key Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OrchestratedSelfLearning                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│   │ Primary LLM │    │ Backend LLM │    │   LCM/PFC   │                    │
│   │ (Analysis)  │    │ (Reasoning) │    │ (Concepts) │                    │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                    │
│          │                  │                   │                            │
│          └──────────────────┼───────────────────┘                            │
│                             │                                                │
│                    ┌────────▼────────┐                                      │
│                    │   PERCEPTION    │                                      │
│                    │   - LLM analyze │                                      │
│                    │   - Embed query │                                      │
│                    │   - Patterns    │                                      │
│                    └────────┬────────┘                                      │
│                             │                                                │
│                    ┌────────▼────────┐                                      │
│                    │    EVALUATION   │                                      │
│                    │   - LLM reason │                                      │
│                    │   - LCM encode  │                                      │
│                    │   - Score       │                                      │
│                    └────────┬────────┘                                      │
│                             │                                                │
│                    ┌────────▼────────┐                                      │
│                    │    REFINEMENT   │                                      │
│                    │   - LLM suggest │                                      │
│                    │   - LCM guide   │                                      │
│                    │   - Embed lookup│                                      │
│                    └────────┬────────┘                                      │
│                             │                                                │
│                    ┌────────▼────────┐                                      │
│                    │    MEMORY      │                                      │
│                    │   - LCM train   │                                      │
│                    │   - Embed store │                                      │
│                    │   - Graph link  │                                      │
│                    └─────────────────┘                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

The orchestration ensures:
1. Both LLMs contribute their unique strengths
2. LCM provides concept-level guidance
3. Embeddings enable pattern matching and retrieval
4. Knowledge graph connects related insights
"""
import asyncio
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
from core.helpers import atomic_write_json, read_json_with_integrity

logger = logging.getLogger(__name__)

# Singleton
_orchestrated_learning = None
_orchestration_lock = threading.Lock()


@dataclass
class PerceptionResult:
    """Result of multi-model perception."""
    llm_analysis: dict  # Analysis from LLMs
    embedding_matches: list[dict]  # Similar patterns from embeddings
    perception_scores: dict  # Scores from perception engine
    concepts_extracted: list[str]  # LCM concepts found
    issues_detected: list[str]  # Problems identified
    strengths_found: list[str]  # What worked well
    overall_quality: float  # Combined score 0-1


@dataclass
class EvaluationResult:
    """Result of deep evaluation."""
    llm_reasoning: str  # LLM's detailed reasoning
    concept_sequence: list[dict]  # LCM concept sequence
    quality_breakdown: dict  # Scores by aspect
    improvement_potential: float  # How much can improve
    master_comparison: dict  # Comparison to master works
    best_practices: list[str]  # Relevant best practices


@dataclass
class RefinementPlan:
    """Plan for improvement."""
    llm_suggestions: list[dict]  # LLM-generated suggestions
    lcm_guidance: dict  # LCM concept guidance
    embedding_patterns: list[dict]  # Similar successful patterns
    techniques_to_apply: list[str]  # Specific techniques
    priority_order: list[str]  # What to apply first
    expected_improvement: float  # Expected score delta


@dataclass
class MemoryEntry:
    """Stored learning memory."""
    concept_sequence: list[dict]  # LCM concepts
    embedding: np.ndarray  # Embedding vector
    quality_score: float  # How well it worked
    technique_results: dict  # Which techniques helped
    llm_insights: list[str]  # LLM-generated insights
    timestamp: datetime = field(default_factory=datetime.now)


class OrchestratedSelfLearning:
    """Orchestrated learning using LLM + LCM + Embeddings."""
    
    def __init__(self):
        self.storage_path = Path.home() / ".jarvis" / "orchestrated_learning"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Learning memory
        self.memory: list[MemoryEntry] = []
        self.MAX_MEMORY = 1000
        
        # Concept effectiveness (from LCM)
        self.concept_effectiveness: dict[str, float] = defaultdict(float)
        
        # Technique effectiveness (tracked over time)
        self.technique_effectiveness: dict[str, dict] = defaultdict(lambda: {
            "applications": 0,
            "successes": 0,
            "avg_improvement": 0.0,
        })
        
        # Embedding pattern clusters
        self.successful_embeddings: list[tuple[np.ndarray, dict]] = []
        self.failed_embeddings: list[tuple[np.ndarray, dict]] = []
        
        # LLM clients (lazy init)
        self._primary_llm = None
        self._backend_llm = None
        self._pfc = None
        self._embeddings = None
        self._knowledge_graph = None
        
        # Domain-specific learners (lazy init)
        self._music_learner = None
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Load saved state
        self._load_state()
    
    def _get_primary_llm(self):
        """Get primary LLM client (Kimi K2.5) for semantic understanding."""
        if self._primary_llm is None:
            try:
                from core.backend_llm import get_primary_llm
                self._primary_llm = get_primary_llm()
            except Exception as e:
                logger.debug(f"Primary LLM unavailable: {e}")
        return self._primary_llm
    
    def _get_backend_llm(self):
        """Get backend LLM (GLM-5) for reasoning tasks."""
        if self._backend_llm is None:
            try:
                from core.backend_llm import call_backend
                self._backend_llm = call_backend
            except Exception as e:
                logger.debug(f"Backend LLM unavailable: {e}")
        return self._backend_llm
    
    def _get_pfc(self):
        """Get Prefrontal Cortex (LCM)."""
        if self._pfc is None:
            try:
                from core.prefrontal_cortex import get_prefrontal_cortex
                self._pfc = get_prefrontal_cortex()
            except Exception as e:
                logger.debug(f"PFC unavailable: {e}")
        return self._pfc
    
    def _get_embeddings(self):
        """Get embedding interconnect."""
        if self._embeddings is None:
            try:
                from core.embedding_interconnect import get_embedding_interconnect
                self._embeddings = get_embedding_interconnect()
            except Exception as e:
                logger.debug(f"Embeddings unavailable: {e}")
        return self._embeddings
    
    def _get_knowledge_graph(self):
        """Get knowledge graph."""
        if self._knowledge_graph is None:
            try:
                from core.knowledge_graph import get_knowledge_graph
                self._knowledge_graph = get_knowledge_graph()
            except Exception as e:
                logger.debug(f"Knowledge graph unavailable: {e}")
        return self._knowledge_graph
    
    async def perceive(
        self,
        content: dict,
        domain: str = "music",
        context: dict | None = None
    ) -> PerceptionResult:
        """Multi-model perception of content.
        
        Uses:
        1. LLM analysis for semantic understanding
        2. Perception engine for domain-specific features
        3. Embedding lookup for similar patterns
        4. LCM concept extraction
        
        Args:
            content: The content to perceive (composition, code, etc.)
            domain: Domain type ("music", "code", "insight", "general")
            context: Optional context
            
        Returns:
            PerceptionResult with multi-model analysis
        """
        with self._lock:
            llm_analysis = {}
            embedding_matches = []
            perception_scores = {}
            concepts_extracted = []
            issues_detected = []
            strengths_found = []
            
            # 1. Get LLM analysis
            llm_result = await self._llm_analyze(content, domain, "perception")
            if llm_result:
                llm_analysis = llm_result
                issues_detected.extend(llm_result.get("issues", []))
                strengths_found.extend(llm_result.get("strengths", []))
            
            # 2. Get perception engine scores (domain-specific)
            perception_result = await self._perception_analyze(content, domain)
            if perception_result:
                perception_scores = perception_result.get("scores", {})
                issues_detected.extend(perception_result.get("issues", []))
                strengths_found.extend(perception_result.get("strengths", []))
            
            # 3. Get embedding matches for similar patterns
            embedding_matches = await self._embedding_lookup(content, domain)
            
            # 4. Extract LCM concepts
            concepts_extracted = await self._extract_concepts(content, domain)
            
            # 5. Calculate overall quality
            scores_list = []
            
            if perception_scores.get("quality") is not None:
                scores_list.append(perception_scores["quality"])
            if llm_analysis.get("quality_score") is not None:
                scores_list.append(llm_analysis["quality_score"])
            if embedding_matches:
                embed_scores = [m.get("similarity", 0.5) for m in embedding_matches[:3] if m.get("similarity") is not None]
                if embed_scores:
                    scores_list.append(sum(embed_scores) / len(embed_scores))
            
            if not scores_list:
                scores_list = [0.5]
            
            overall_quality = sum(scores_list) / len(scores_list)
            
            return PerceptionResult(
                llm_analysis=llm_analysis,
                embedding_matches=embedding_matches[:10],
                perception_scores=perception_scores,
                concepts_extracted=concepts_extracted,
                issues_detected=list(set(issues_detected)),
                strengths_found=list(set(strengths_found)),
                overall_quality=overall_quality,
            )
    
    async def evaluate(
        self,
        content: dict,
        perception: PerceptionResult,
        domain: str = "music"
    ) -> EvaluationResult:
        """Deep evaluation using LLM reasoning + LCM concepts.
        
        Uses:
        1. LLM to reason about strengths/weaknesses
        2. LCM to encode concept sequences
        3. Pattern matching against master works
        4. Quality breakdown by aspect
        
        Args:
            content: The content to evaluate
            perception: Previous perception result
            domain: Domain type
            
        Returns:
            EvaluationResult with deep analysis
        """
        with self._lock:
            llm_reasoning = ""
            concept_sequence = []
            quality_breakdown = {}
            master_comparison = {}
            best_practices = []
            
            # 1. LLM reasoning about the content
            llm_reasoning = await self._llm_reason(content, perception, domain)
            
            # 2. LCM concept sequence encoding
            concept_sequence = await self._encode_concept_sequence(content, domain)
            
            # 3. Quality breakdown by aspect
            quality_breakdown = {
                "overall": perception.overall_quality,
                "llm_score": perception.llm_analysis.get("quality_score", 0.5),
                "perception_score": perception.perception_scores.get("quality", 0.5),
                "concept_alignment": self._calculate_concept_alignment(concept_sequence),
                "embedding_similarity": perception.embedding_matches[0].get("similarity", 0.5)
                if perception.embedding_matches else 0.5,
            }
            
            # 4. Compare to master works
            master_comparison = await self._compare_to_masters(content, domain)
            
            # 5. Get relevant best practices
            best_practices = await self._get_best_practices(domain, quality_breakdown)
            
            # Calculate improvement potential
            improvement_potential = max(0, 1.0 - quality_breakdown["overall"])
            
            return EvaluationResult(
                llm_reasoning=llm_reasoning,
                concept_sequence=concept_sequence,
                quality_breakdown=quality_breakdown,
                improvement_potential=improvement_potential,
                master_comparison=master_comparison,
                best_practices=best_practices,
            )
    
    async def refine(
        self,
        content: dict,
        evaluation: EvaluationResult,
        domain: str = "music"
    ) -> RefinementPlan:
        """Create refinement plan using multi-model intelligence.
        
        Uses:
        1. LLM to generate specific suggestions
        2. LCM to guide concept sequence
        3. Embedding lookup for successful patterns
        4. Technique prioritization
        
        Args:
            content: Content to refine
            evaluation: Previous evaluation
            domain: Domain type
            
        Returns:
            RefinementPlan with actionable suggestions
        """
        with self._lock:
            llm_suggestions = []
            lcm_guidance = {}
            embedding_patterns = []
            techniques_to_apply = []
            priority_order = []
            
            # 1. Get LLM suggestions
            llm_suggestions = await self._llm_suggest(content, evaluation, domain)
            
            # 2. Get LCM concept guidance
            lcm_guidance = await self._get_lcm_guidance(evaluation.concept_sequence, domain)
            
            # 3. Find similar successful patterns from embeddings
            embedding_patterns = await self._find_successful_patterns(content, domain)
            
            # 4. Compile techniques to apply
            techniques_to_apply = self._compile_techniques(
                llm_suggestions, lcm_guidance, embedding_patterns, domain
            )
            
            # 5. Prioritize by expected impact
            priority_order = self._prioritize_techniques(
                techniques_to_apply, evaluation.quality_breakdown
            )
            
            # Calculate expected improvement
            expected_improvement = 0.0
            for t in techniques_to_apply[:3]:
                if isinstance(t, dict):
                    expected_improvement += min(0.2, t.get("expected_impact", 0.05))
                else:
                    expected_improvement += 0.05  # Default for string techniques
            expected_improvement = min(0.5, expected_improvement)
            
            return RefinementPlan(
                llm_suggestions=llm_suggestions,
                lcm_guidance=lcm_guidance,
                embedding_patterns=embedding_patterns,
                techniques_to_apply=techniques_to_apply,
                priority_order=priority_order,
                expected_improvement=expected_improvement,
            )
    
    async def remember(
        self,
        content: dict,
        perception: PerceptionResult,
        evaluation: EvaluationResult,
        refinement: RefinementPlan,
        outcome_score: float,
        domain: str = "music"
    ) -> None:
        """Store learning in multi-system memory.
        
        Uses:
        1. LCM concept training
        2. Embedding storage
        3. Knowledge graph linking
        4. Technique effectiveness tracking
        
        Args:
            content: Original content
            perception: Perception result
            evaluation: Evaluation result
            refinement: Refinement plan applied
            outcome_score: Final quality score achieved
            domain: Domain type
        """
        with self._lock:
            # 1. Store in memory
            embedding = await self._create_embedding(content, domain)
            
            entry = MemoryEntry(
                concept_sequence=evaluation.concept_sequence,
                embedding=embedding,
                quality_score=outcome_score,
                technique_results={
                    t: outcome_score for t in refinement.techniques_to_apply
                },
                llm_insights=perception.llm_analysis.get("insights", []),
            )
            self.memory.append(entry)
            if len(self.memory) > self.MAX_MEMORY:
                self.memory = self.memory[-self.MAX_MEMORY:]
            
            # 2. Update technique effectiveness
            for technique in refinement.techniques_to_apply:
                tech_name = technique if isinstance(technique, str) else technique.get("name", technique)
                self.technique_effectiveness[tech_name]["applications"] += 1
                if outcome_score > 0.6:
                    self.technique_effectiveness[tech_name]["successes"] += 1
                self.technique_effectiveness[tech_name]["avg_improvement"] = (
                    0.9 * self.technique_effectiveness[tech_name]["avg_improvement"] +
                    0.1 * outcome_score
                )
            
            # 3. Store in embedding space
            if embedding is not None:
                if outcome_score > 0.6:
                    self.successful_embeddings.append((embedding, {
                        "domain": domain,
                        "score": outcome_score,
                        "concepts": [c.get("concept_type", "") for c in evaluation.concept_sequence],
                    }))
                else:
                    self.failed_embeddings.append((embedding, {
                        "domain": domain,
                        "score": outcome_score,
                    }))
                
                # Limit stored embeddings
                if len(self.successful_embeddings) > 500:
                    self.successful_embeddings = self.successful_embeddings[-500:]
                if len(self.failed_embeddings) > 200:
                    self.failed_embeddings = self.failed_embeddings[-200:]
            
            # 4. Train LCM concepts
            await self._train_lcm_concepts(evaluation.concept_sequence, outcome_score, domain)
            
            # 5. Link in knowledge graph
            await self._link_knowledge_graph(content, evaluation, outcome_score, domain)
            
            # 6. Save state
            self._save_state()
            
            logger.info(f"[OrchestratedLearning] Remembered: score={outcome_score:.2f}, "
                       f"techniques={len(refinement.techniques_to_apply)}, "
                       f"concepts={len(evaluation.concept_sequence)}")
    
    # ==================== LLM Methods ====================
    
    async def _llm_analyze(self, content: dict, domain: str, phase: str) -> dict:
        """Use BOTH LLMs for multi-perspective analysis.
        
        Uses:
        1. Primary LLM (Kimi K2.5) for semantic understanding
        2. Backend LLM (GLM-5) for analytical reasoning
        
        Returns combined results from both perspectives.
        """
        results = {
            "quality_score": 0.5,
            "issues": [],
            "strengths": [],
            "suggestions": [],
            "insights": [],
            "primary_perspective": None,
            "backend_perspective": None,
        }
        
        domain_prompts = {
            "music": f"""Analyze this musical composition for quality issues and strengths.

Composition: {content.get('title', 'Untitled')}
Style: {content.get('style', 'classical')}
Key: {content.get('key', 'C')}
Form: {content.get('form', 'ternary')}

Sections: {len(content.get('sections', []))} sections
Measures: {len(content.get('measures', []))} measures

Analyze and return JSON:
{{
    "quality_score": 0.0-1.0,
    "issues": ["list of problems"],
    "strengths": ["list of what works"],
    "suggestions": ["list of improvements"],
    "insights": ["key insights"]
}}""",
            "code": """Analyze this code for quality issues.

Return JSON with quality_score, issues, strengths, suggestions, insights.""",
            "insight": """Analyze this insight for quality and novelty.

Return JSON with quality_score, issues, strengths, suggestions, insights.""",
            "general": """Analyze this content for quality.

Return JSON with quality_score, issues, strengths, suggestions, insights.""",
        }
        
        prompt = domain_prompts.get(domain, domain_prompts["general"])
        
        # 1. Primary LLM (Kimi K2.5) - Semantic understanding
        primary_llm = self._get_primary_llm()
        if primary_llm is not None:
            try:
                if asyncio.iscoroutinefunction(primary_llm.generate):
                    primary_response = await primary_llm.generate(prompt)
                else:
                    primary_response = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: primary_llm.generate(prompt)
                    )
                
                if primary_response:
                    import json
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', primary_response)
                    if json_match:
                        primary_data = json.loads(json_match.group())
                        results["primary_perspective"] = primary_data.get("perspective", "semantic")
                        results["strengths"].extend(primary_data.get("strengths", [])[:2])
                        if primary_data.get("quality_score"):
                            results["quality_score"] = primary_data["quality_score"]
            except Exception as e:
                logger.debug(f"Primary LLM analysis failed: {e}")
        
        # 2. Backend LLM (GLM-5) - Analytical reasoning
        backend_llm = self._get_backend_llm()
        if backend_llm is not None:
            try:
                if asyncio.iscoroutinefunction(backend_llm):
                    backend_response = await backend_llm(prompt, system_prompt="You are an expert analyst. Return only valid JSON.")
                else:
                    backend_response = await asyncio.get_running_loop().run_in_executor(
                        None, backend_llm, prompt
                    )
                
                if backend_response:
                    import json
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', backend_response)
                    if json_match:
                        backend_data = json.loads(json_match.group())
                        results["backend_perspective"] = backend_data.get("perspective", "analytical")
                        results["issues"].extend(backend_data.get("issues", [])[:3])
                        results["suggestions"].extend(backend_data.get("suggestions", [])[:2])
                        results["insights"].extend(backend_data.get("insights", [])[:2])
                        # Combine quality scores
                        if backend_data.get("quality_score") and results["quality_score"] == 0.5:
                            results["quality_score"] = backend_data["quality_score"]
                        elif backend_data.get("quality_score"):
                            results["quality_score"] = (results["quality_score"] + backend_data["quality_score"]) / 2
            except Exception as e:
                logger.debug(f"Backend LLM analysis failed: {e}")
        
        return results
    
    async def _llm_reason(self, content: dict, perception: PerceptionResult, domain: str) -> str:
        """Use BOTH LLMs for multi-perspective reasoning.
        
        Primary LLM provides semantic reasoning.
        Backend LLM provides analytical reasoning.
        """
        reasoning_parts = []
        
        # Primary LLM reasoning (semantic)
        primary_llm = self._get_primary_llm()
        if primary_llm is not None:
            try:
                prompt = f"""From a semantic understanding perspective, explain why this {domain} content 
has these characteristics:

Quality: {perception.overall_quality:.2f}
Strengths: {perception.strengths_found[:3]}

Provide 1-2 sentences on what makes this content effective or ineffective."""
                
                if asyncio.iscoroutinefunction(primary_llm.generate):
                    primary_reason = await primary_llm.generate(prompt)
                else:
                    primary_reason = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: primary_llm.generate(prompt)
                    )
                
                if primary_reason:
                    reasoning_parts.append(f"[Semantic] {primary_reason[:200]}")
            except Exception as e:
                logger.debug(f"Primary LLM reasoning failed: {e}")
        
        # Backend LLM reasoning (analytical)
        backend_llm = self._get_backend_llm()
        if backend_llm is not None:
            try:
                prompt = f"""Reason about this {domain} content's quality and potential improvements.

Quality Score: {perception.overall_quality:.2f}
Issues Found: {perception.issues_detected}
Strengths: {perception.strengths_found}
Concepts: {perception.concepts_extracted}

Provide detailed reasoning (2-3 sentences) about why this content has these issues/strengths
and what fundamental principles could improve it."""
                
                if asyncio.iscoroutinefunction(backend_llm):
                    backend_reason = await backend_llm(prompt)
                else:
                    backend_reason = await asyncio.get_running_loop().run_in_executor(
                        None, backend_llm, prompt
                    )
                
                if backend_reason:
                    reasoning_parts.append(f"[Analytical] {backend_reason[:300]}")
            except Exception as e:
                logger.debug(f"Backend LLM reasoning failed: {e}")
        
        return " | ".join(reasoning_parts) if reasoning_parts else ""
    
    async def _llm_suggest(self, content: dict, evaluation: EvaluationResult, domain: str) -> list[dict]:
        """Use BOTH LLMs to generate specific suggestions.
        
        Primary LLM focuses on creative/artistic suggestions.
        Backend LLM focuses on technical/improvement suggestions.
        """
        suggestions = []
        
        # Primary LLM suggestions (creative/artistic)
        primary_llm = self._get_primary_llm()
        if primary_llm is not None:
            try:
                prompt = f"""From a creative perspective, suggest artistic improvements for this {domain} content.

Current Quality: {evaluation.quality_breakdown.get('overall', 0.5):.2f}
Strengths: {evaluation.best_practices[:2]}

Generate 2-3 creative, artistic suggestions as JSON list:
[{{"suggestion": "...", "technique": "...", "perspective": "creative"}}]"""
                
                if asyncio.iscoroutinefunction(primary_llm.generate):
                    primary_response = await primary_llm.generate(prompt)
                else:
                    primary_response = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: primary_llm.generate(prompt)
                    )
                
                if primary_response:
                    import json
                    import re
                    json_match = re.search(r'\[[\s\S]*\]', primary_response)
                    if json_match:
                        primary_suggestions = json.loads(json_match.group())
                        for s in primary_suggestions[:2]:
                            s["source"] = "primary_llm"
                        suggestions.extend(primary_suggestions[:2])
            except Exception as e:
                logger.debug(f"Primary LLM suggestions failed: {e}")
        
        # Backend LLM suggestions (technical/improvement)
        backend_llm = self._get_backend_llm()
        if backend_llm is not None:
            try:
                prompt = f"""Suggest specific improvements for this {domain} content.

Current Quality: {evaluation.quality_breakdown.get('overall', 0.5):.2f}
Issues: {evaluation.quality_breakdown}
Best Practices: {evaluation.best_practices}

Generate 3-5 specific, actionable suggestions as JSON list:
[{{"suggestion": "...", "technique": "...", "expected_impact": 0.0-1.0}}]"""
                
                if asyncio.iscoroutinefunction(backend_llm):
                    backend_response = await backend_llm(prompt, system_prompt="Return only valid JSON.")
                else:
                    backend_response = await asyncio.get_running_loop().run_in_executor(
                        None, backend_llm, prompt
                    )
                
                if backend_response:
                    import json
                    import re
                    json_match = re.search(r'\[[\s\S]*\]', backend_response)
                    if json_match:
                        backend_suggestions = json.loads(json_match.group())
                        for s in backend_suggestions[:3]:
                            s["source"] = "backend_llm"
                        suggestions.extend(backend_suggestions[:3])
            except Exception as e:
                logger.debug(f"Backend LLM suggestions failed: {e}")
        
        return suggestions
    
    # ==================== Perception Methods ====================
    
    async def _perception_analyze(self, content: dict, domain: str) -> dict:
        """Use domain-specific perception engine."""
        try:
            if domain == "music":
                from core.music_perception import get_music_perception
                perception = get_music_perception()
                
                # Get audio if available
                audio = content.get("audio_data")
                if audio is not None:
                    features = perception.analyze_composition(
                        audio=audio,
                        sample_rate=content.get("sample_rate", 32000),
                        expected_form=content.get("form", "ternary"),
                        expected_key=content.get("key", "C"),
                        expected_tempo=content.get("tempo", 120),
                    )
                    
                    feedback = perception.generate_feedback(
                        features=features,
                        expected_form=content.get("form", "ternary"),
                        expected_key=content.get("key", "C"),
                        expected_tempo=content.get("tempo", 120),
                    )
                    
                    return {
                        "scores": {
                            "quality": feedback.overall_score,
                            "coherence": features.coherence_score,
                            "development": features.development_score,
                            "resolution": features.resolution_score,
                            "structure": features.structure_score,
                        },
                        "issues": feedback.weaknesses,
                        "strengths": feedback.strengths,
                    }
            
            elif domain == "code":
                # Code perception
                from core.insight_bridge import get_insight_bridge
                get_insight_bridge()
                # Analyze code quality
                return {
                    "scores": {"quality": 0.5},
                    "issues": [],
                    "strengths": [],
                }
            
        except Exception as e:
            logger.debug(f"Perception analysis failed: {e}")
        
        return {"scores": {"quality": 0.5}, "issues": [], "strengths": []}
    
    # ==================== Embedding Methods ====================
    
    async def _embedding_lookup(self, content: dict, domain: str) -> list[dict]:
        """Find similar patterns in embedding space."""
        try:
            embeddings = self._get_embeddings()
            if embeddings is None:
                return []
            
            # Create content embedding
            content_text = self._content_to_text(content, domain)
            content_embedding = embeddings.embed(content_text)
            
            matches = []
            
            # Search successful embeddings
            for stored_embedding, metadata in self.successful_embeddings[-100:]:
                if metadata.get("domain") != domain:
                    continue
                
                similarity = self._cosine_similarity(content_embedding, stored_embedding)
                if similarity > 0.5:
                    matches.append({
                        "similarity": similarity,
                        "concepts": metadata.get("concepts", []),
                        "score": metadata.get("score", 0.5),
                        "source": "successful_patterns",
                    })
            
            # Sort by similarity
            matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            return matches[:10]
            
        except Exception as e:
            logger.debug(f"Embedding lookup failed: {e}")
            return []
    
    async def _create_embedding(self, content: dict, domain: str) -> np.ndarray | None:
        """Create embedding for content."""
        try:
            embeddings = self._get_embeddings()
            if embeddings is None:
                return None
            
            content_text = self._content_to_text(content, domain)
            return embeddings.embed(content_text)
            
        except Exception as e:
            logger.debug(f"Embedding creation failed: {e}")
            return None
    
    async def _find_successful_patterns(self, content: dict, domain: str) -> list[dict]:
        """Find successful patterns similar to content."""
        return await self._embedding_lookup(content, domain)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        try:
            if a is None or b is None:
                return 0.0
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(dot / (norm_a * norm_b))
        except Exception:
            return 0.0
    
    def _content_to_text(self, content: dict, domain: str) -> str:
        """Convert content dict to text for embedding."""
        parts = []
        
        if domain == "music":
            parts.append(f"Style: {content.get('style', 'unknown')}")
            parts.append(f"Form: {content.get('form', 'unknown')}")
            parts.append(f"Key: {content.get('key', 'unknown')}")
            parts.append(f"Tempo: {content.get('tempo', 120)}")
            if content.get("reasoning"):
                parts.append(f"Reasoning: {content.get('reasoning', {}).get('artistic_intent', '')}")
        else:
            parts.append(str(content)[:500])
        
        return " | ".join(parts)
    
    # ==================== LCM Methods ====================
    
    async def _extract_concepts(self, content: dict, domain: str) -> list[str]:
        """Extract LCM concepts from content."""
        try:
            if domain == "music":
                from core.music_lcm import get_music_lcm
                music_lcm = get_music_lcm()
                
                concepts = music_lcm.encode_composition(
                    composition_data=content,
                    style=content.get("style", "classical"),
                    form=content.get("form", "ternary"),
                )
                
                return [c.concept_type for c in concepts]
            
            else:
                # General concept extraction via PFC
                pfc = self._get_pfc()
                if pfc and pfc._running:
                    from core.prefrontal_cortex import ConceptType
                    
                    content_text = self._content_to_text(content, domain)
                    concept = pfc.observe(
                        content=content_text,
                        concept_type=ConceptType.OBSERVATION,
                        source=f"{domain}_orchestrated_learning",
                    )
                    return [content_text[:100]] if concept else []
            
        except Exception as e:
            logger.debug(f"Concept extraction failed: {e}")
        
        return []
    
    async def _encode_concept_sequence(self, content: dict, domain: str) -> list[dict]:
        """Encode content as LCM concept sequence."""
        try:
            if domain == "music":
                from core.music_lcm import get_music_lcm
                music_lcm = get_music_lcm()
                
                concepts = music_lcm.encode_composition(
                    composition_data=content,
                    style=content.get("style", "classical"),
                    form=content.get("form", "ternary"),
                )
                
                return [
                    {
                        "id": c.id,
                        "type": c.concept_type,
                        "position": c.section_position,
                        "tension": c.tension_level,
                    }
                    for c in concepts
                ]
            
        except Exception as e:
            logger.debug(f"Concept sequence encoding failed: {e}")
        
        return []
    
    async def _get_lcm_guidance(self, concept_sequence: list[dict], domain: str) -> dict:
        """Get LCM guidance based on concept sequence."""
        try:
            if domain == "music":
                from core.music_lcm import get_music_lcm, MusicalConcept
                music_lcm = get_music_lcm()
                
                # Convert back to MusicalConcept
                concepts = []
                for c in concept_sequence[-5:]:
                    concepts.append(MusicalConcept(
                        id=c.get("id", "concept"),
                        concept_type=c.get("type", "theme_statement"),
                        section_position=c.get("position", 0.5),
                        tension_level=c.get("tension", 0.5),
                    ))
                
                # Get next concept suggestion
                guidance = music_lcm.get_next_concept_suggestion(
                    current_position=0.5,
                    form="ternary",
                    style="classical",
                    current_concepts=concepts,
                )
                
                return guidance
            
        except Exception as e:
            logger.debug(f"LCM guidance failed: {e}")
        
        return {}
    
    async def _train_lcm_concepts(self, concept_sequence: list[dict], score: float, domain: str) -> None:
        """Train LCM on successful concept sequences."""
        try:
            if domain == "music":
                from core.music_lcm import get_music_lcm, MusicalConcept
                music_lcm = get_music_lcm()
                
                # Convert to MusicalConcept
                concepts = [
                    MusicalConcept(
                        id=c.get("id", "concept"),
                        concept_type=c.get("type", "theme_statement"),
                        section_position=c.get("position", 0.5),
                        effectiveness=score,
                    )
                    for c in concept_sequence
                ]
                
                music_lcm.learn_from_composition(
                    concepts=concepts,
                    quality_score=score,
                    style="classical",
                )
            
        except Exception as e:
            logger.debug(f"LCM training failed: {e}")
    
    # ==================== Knowledge Graph Methods ====================
    
    async def _compare_to_masters(self, content: dict, domain: str) -> dict:
        """Compare content to master works in knowledge graph."""
        try:
            kg = self._get_knowledge_graph()
            if kg is None:
                return {}
            
            # Query for similar master works
            # Implementation depends on knowledge graph structure
            return {
                "master_similarity": 0.5,
                "master_techniques": [],
            }
            
        except Exception as e:
            logger.debug(f"Master comparison failed: {e}")
            return {}
    
    async def _get_best_practices(self, domain: str, quality_breakdown: dict) -> list[str]:
        """Get relevant best practices from knowledge."""
        practices = {
            "music": [
                "Use stepwise motion for 65% of melody",
                "Resolve dominant to tonic at phrase endings",
                "Include contrast in dynamics and texture",
                "Develop motifs through sequence and transformation",
                "Balance tension and resolution across form",
            ],
            "code": [
                "Follow DRY principle",
                "Use meaningful variable names",
                "Add documentation for complex logic",
                "Write unit tests for core functions",
            ],
        }
        
        return practices.get(domain, [])
    
    async def _link_knowledge_graph(
        self,
        content: dict,
        evaluation: EvaluationResult,
        score: float,
        domain: str
    ) -> None:
        """Link learning to knowledge graph."""
        try:
            kg = self._get_knowledge_graph()
            if kg is None:
                return
            
            # Add node for this learning
            # Implementation depends on knowledge graph structure
            
        except Exception as e:
            logger.debug(f"Knowledge graph linking failed: {e}")
    
    # ==================== Utility Methods ====================
    
    def _calculate_concept_alignment(self, concept_sequence: list[dict]) -> float:
        """Calculate how well concepts align with expected patterns."""
        if not concept_sequence:
            return 0.5
        
        # Simple heuristic: check for expected concept progression
        expected_order = {
            "exposition_start": 0,
            "theme_statement": 1,
            "development_open": 2,
            "development_climax": 3,
            "recapitulation_return": 4,
            "peaceful_resolution": 5,
        }
        
        alignment_scores = []
        for c in concept_sequence:
            concept_type = c.get("type", "")
            position = c.get("position", 0.5)
            expected_pos = expected_order.get(concept_type, -1)
            if expected_pos >= 0:
                # Check if position matches expected
                expected_position = expected_pos / max(1, len(expected_order) - 1)
                alignment = 1.0 - abs(position - expected_position)
                alignment_scores.append(alignment)
        
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5
    
    def _compile_techniques(
        self,
        llm_suggestions: list[dict],
        lcm_guidance: dict,
        embedding_patterns: list[dict],
        domain: str
    ) -> list[str]:
        """Compile techniques from multiple sources."""
        techniques = []
        
        # From LLM suggestions
        for s in llm_suggestions[:5]:
            tech = s.get("technique", "")
            if tech:
                techniques.append(tech)
        
        # From LCM guidance
        guidance_concepts = lcm_guidance.get("alternative_concepts", [])
        techniques.extend(guidance_concepts[:3])
        
        # From embedding patterns (learned patterns)
        for p in embedding_patterns[:3]:
            concepts = p.get("concepts", [])
            techniques.extend(concepts[:1])
        
        return list(set(techniques))[:10]
    
    def _prioritize_techniques(
        self,
        techniques: list[str],
        quality_breakdown: dict
    ) -> list[str]:
        """Prioritize techniques by expected impact."""
        # Score each technique by relevance to quality deficits
        scored = []
        
        overall_quality = quality_breakdown.get("overall", 0.5)
        
        for tech in techniques:
            # Get tracked effectiveness
            effectiveness = self.technique_effectiveness.get(tech, {})
            success_rate = (
                effectiveness.get("successes", 0) / 
                max(1, effectiveness.get("applications", 1))
            )
            avg_improvement = effectiveness.get("avg_improvement", 0.05)
            
            # Calculate priority score
            priority = (
                success_rate * 0.4 +
                avg_improvement * 0.3 +
                (1.0 - overall_quality) * 0.3  # Lower quality = higher priority
            )
            
            scored.append((tech, priority))
        
        # Sort by priority
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [t[0] for t in scored]
    
    # ==================== State Management ====================
    
    def _load_state(self) -> None:
        """Load saved state."""
        try:
            state_file = self.storage_path / "orchestrated_learning_state.json"
            if state_file.exists():
                state = read_json_with_integrity(state_file)
                
                # Restore technique effectiveness
                for tech, data in state.get("technique_effectiveness", {}).items():
                    self.technique_effectiveness[tech] = data
                
                # Restore concept effectiveness
                for concept, score in state.get("concept_effectiveness", {}).items():
                    self.concept_effectiveness[concept] = score
                
                logger.info(f"[OrchestratedLearning] Loaded state: {len(self.technique_effectiveness)} techniques")
                
        except Exception as e:
            logger.debug(f"Could not load state: {e}")
    
    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            state = {
                "technique_effectiveness": dict(self.technique_effectiveness),
                "concept_effectiveness": dict(self.concept_effectiveness),
                "saved_at": datetime.now().isoformat(),
            }
            
            state_file = self.storage_path / "orchestrated_learning_state.json"
            atomic_write_json(state_file, state)
                
        except Exception as e:
            logger.debug(f"Could not save state: {e}")
    
    def get_learning_stats(self) -> dict:
        """Get learning statistics."""
        return {
            "memory_size": len(self.memory),
            "successful_embeddings": len(self.successful_embeddings),
            "failed_embeddings": len(self.failed_embeddings),
            "techniques_tracked": len(self.technique_effectiveness),
            "concepts_tracked": len(self.concept_effectiveness),
            "top_techniques": sorted(
                [
                    {"technique": k, "effectiveness": v["avg_improvement"]}
                    for k, v in self.technique_effectiveness.items()
                ],
                key=lambda x: x["effectiveness"],
                reverse=True
            )[:10],
        }


# Singleton getter
def get_orchestrated_learning() -> OrchestratedSelfLearning:
    """Get singleton orchestrated learning instance."""
    global _orchestrated_learning
    if _orchestrated_learning is None:
        with _orchestration_lock:
            if _orchestrated_learning is None:
                _orchestrated_learning = OrchestratedSelfLearning()
    return _orchestrated_learning


class InsightOrchestratedLearning:
    """Orchestrated learning for general insights and discoveries.
    
    Uses the same multi-model approach (LLM + LCM + Embeddings) but
    generalized for Erdos engine insights, code analysis, and general learning.
    """
    
    def __init__(self):
        self._orchestrated = None
        
        # Domain-specific knowledge
        self.domain_knowledge = {
            "mathematics": {
                "patterns": ["ramsey", "prime", "graph", "combinatorics", "probability", "number_theory"],
                "quality_factors": ["novelty", "depth", "generality", "unexpectedness"],
                "evolution_domains": ["reasoning", "learning", "adaptation", "efficiency"],
            },
            "music": {
                "patterns": ["harmony", "melody", "rhythm", "form", "texture"],
                "quality_factors": ["coherence", "development", "resolution", "structure"],
                "evolution_domains": ["creativity", "pattern_recognition", "aesthetic"],
            },
            "code": {
                "patterns": ["architecture", "algorithm", "pattern", "optimization", "security"],
                "quality_factors": ["correctness", "efficiency", "maintainability", "security"],
                "evolution_domains": ["reasoning", "efficiency", "adaptation"],
            },
            "general": {
                "patterns": ["observation", "inference", "synthesis", "application"],
                "quality_factors": ["relevance", "novelty", "utility", "clarity"],
                "evolution_domains": ["reasoning", "learning", "creativity"],
            },
        }
        
        self.insight_history: list[dict] = []
        self.MAX_HISTORY = 500
        self.pattern_effectiveness: dict[str, float] = defaultdict(float)
    
    def _get_orchestrated(self):
        if self._orchestrated is None:
            self._orchestrated = get_orchestrated_learning()
        return self._orchestrated
    
    async def perceive_insight(self, insight: dict, domain: str = "general") -> dict:
        orchestrated = self._get_orchestrated()
        perception = await orchestrated.perceive(
            content=insight,
            domain=domain,
            context={"insight_type": insight.get("type", "observation")}
        )
        domain_config = self.domain_knowledge.get(domain, self.domain_knowledge["general"])
        patterns_found = []
        content_lower = insight.get("content", "").lower()
        for pattern in domain_config["patterns"]:
            if pattern in content_lower:
                patterns_found.append(pattern)
        return {
            "perception": perception,
            "domain": domain,
            "patterns_found": patterns_found,
            "quality_factors": domain_config["quality_factors"],
        }
    
    async def evaluate_insight(self, insight: dict, perception: dict) -> dict:
        orchestrated = self._get_orchestrated()
        eval_result = await orchestrated.evaluate(
            content=insight,
            perception=perception["perception"],
            domain=perception["domain"]
        )
        return {
            "evaluation": eval_result,
            "novelty_score": eval_result.improvement_potential,
            "impact_score": len(perception.get("patterns_found", [])) / max(1, len(perception.get("quality_factors", []))),
            "recommended_applications": self._suggest_applications(perception.get("patterns_found", []), perception["domain"]),
        }
    
    def _suggest_applications(self, patterns: list[str], domain: str) -> list[str]:
        applications = []
        pattern_apps = {
            "ramsey": ["Graph theory", "Network analysis"],
            "prime": ["Cryptography", "Number theory"],
            "graph": ["Network optimization", "Path finding"],
            "harmony": ["Composition", "Arranging"],
            "melody": ["Theme development", "Variation writing"],
        }
        for pattern in patterns:
            if pattern in pattern_apps:
                applications.extend(pattern_apps[pattern])
        return list(set(applications))[:5]


_insight_orchestrator = None
_insight_lock = threading.Lock()


def get_insight_orchestrator() -> "InsightOrchestratedLearning":
    global _insight_orchestrator
    if _insight_orchestrator is None:
        with _insight_lock:
            if _insight_orchestrator is None:
                _insight_orchestrator = InsightOrchestratedLearning()
    return _insight_orchestrator