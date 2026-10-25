"""Evolution Orchestrator for continuous Jarvis improvement.

Manages:
- Background evolution cycles
- Permission requests before self-modification
- Evolutionary intent collection
- Capability evolution proposals
- Memory shifting windows
- Interactive permission queue
- Reinforcement learning for adaptive improvement
"""
import asyncio
import logging
from core.helpers import atomic_write_json, create_tracked_task, safe_error_message, read_json_with_integrity
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from core.dual_helpers import evolution_propose, evolution_analyze

logger = logging.getLogger(__name__)
from core.prefrontal_cortex import get_prefrontal_cortex, ConceptType


class EvolutionStage(Enum):
    DORMANT = "dormant"
    ANALYZING = "analyzing"
    PROPOSING = "proposing"
    AWAITING_PERMISSION = "awaiting_permission"
    EVOLVING = "evolving"
    INTEGRATING = "integrating"
    REFLECTING = "reflecting"


class PermissionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"
    MODIFIED = "modified"


class CapabilityDomain(Enum):
    REASONING = "reasoning"
    LEARNING = "learning"
    SECURITY = "security"
    VOICE = "voice"
    MEMORY = "memory"
    CREATIVITY = "creativity"
    EFFICIENCY = "efficiency"
    KNOWLEDGE = "knowledge"
    ADAPTATION = "adaptation"
    COMMUNICATION = "communication"


@dataclass
class EvolutionProposal:
    id: str
    timestamp: datetime
    domain: CapabilityDomain
    title: str
    description: str
    rationale: str
    impact_estimate: dict
    risk_level: str
    affected_components: list[str]
    proposed_changes: dict
    rollback_plan: str
    permission_status: PermissionStatus = PermissionStatus.PENDING
    user_intent: str | None = None
    user_feedback: str | None = None
    approved_at: datetime | None = None
    executed_at: datetime | None = None


@dataclass
class EvolutionGene:
    name: str
    value: float
    min_value: float
    max_value: float
    mutation_rate: float
    description: str
    history: list[tuple[datetime, float]] = field(default_factory=list)
    _MAX_HISTORY = 50


@dataclass
class EvolutionaryPathway:
    current_stage: EvolutionStage
    total_evolutions: int
    successful_evolutions: int
    rolled_back_evolutions: int
    pathway_history: list[dict]
    gene_pool: dict[str, EvolutionGene]
    capability_scores: dict[str, float]
    last_evolution: datetime | None
    next_proposed_evolution: datetime | None


@dataclass
class PermissionRequest:
    id: str
    proposal: EvolutionProposal
    created_at: datetime
    expires_at: datetime
    status: PermissionStatus = PermissionStatus.PENDING
    user_response: str | None = None


class EvolutionOrchestrator:
    """Orchestrates continuous evolution with user permission.
    
    Runs in background, proposals requiring permission are queued
    for user review while Jarvis remains interactive.
    
    Uses Reinforcement Learning for adaptive gene adjustments.
    """

    EVOLUTION_INTERVAL = 14400  # 4 hours - low level background
    PERMISSION_TIMEOUT = 86400
    MAX_PENDING_PROPOSALS = 10
    MAX_PROPOSAL_HISTORY = 100
    MIN_EVOLUTION_INTERVAL = 3600  # 1 hour minimum
    DISCOVERY_SHARE_THRESHOLD = 3  # Share after N interesting discoveries
    RL_LEARNING_RATE = 0.1  # How much RL influences gene adjustments

    def __init__(
        self,
        storage_path: Path | None = None,
        permission_callback: Callable | None = None,
    ):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "evolution"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.permission_callback = permission_callback

        self._running = False
        self._evolution_task: asyncio.Task | None = None
        self._lock = threading.RLock()
        
        self._context_engine = None

        self.genes: dict[str, EvolutionGene] = self._init_genes()
        self.pathway = EvolutionaryPathway(
            current_stage=EvolutionStage.DORMANT,
            total_evolutions=0,
            successful_evolutions=0,
            rolled_back_evolutions=0,
            pathway_history=[],
            gene_pool=self.genes,
            capability_scores={d.value: 0.5 for d in CapabilityDomain},
            last_evolution=None,
            next_proposed_evolution=None,
        )

        self.pending_permissions: deque[PermissionRequest] = deque(maxlen=self.MAX_PENDING_PROPOSALS)
        self.proposal_history: deque[EvolutionProposal] = deque(maxlen=self.MAX_PROPOSAL_HISTORY)
        self._proposal_counter = 0



        self._subsystem_refs: dict[str, Any] = {}
        self._load_balancer_ref: Any = None

        self._pending_gene_proposals: dict[str, dict] = {}
        self._MAX_PENDING_GENE_PROPOSALS = 50

        self._rl_engine: Any = None
        self._bioevolve_engine: Any = None  # BioEvolve MAP-Elites engine attachment

        self._load_state()
    
    def _get_context_engine(self):
        if self._context_engine is None:
            try:
                from core.unified_context_engine import get_context_engine
                self._context_engine = get_context_engine()
            except Exception as e:
                logger.debug(f"Context engine unavailable: {e}")
        return self._context_engine

    def _get_load_balancer(self):
        if self._load_balancer_ref is None:
            try:
                from core.cognitive_load_balancer import get_load_balancer
                self._load_balancer_ref = get_load_balancer()
            except Exception as e:
                logger.debug(f"Load balancer unavailable: {e}")
        return self._load_balancer_ref
    
    async def get_evolution_context(self) -> str:
        """Get relevant context from unified memory for evolution decisions.
        
        Enriches evolution with:
        - Past evolution insights and patterns
        - Knowledge about successful adaptions
        - User preferences and system constraints
        """
        try:
            engine = self._get_context_engine()
            if engine is None:
                return ""
            
            context = await engine.assemble_context(
                query="evolution self-improvement adaptation",
                max_tokens=300,
                include_sources=["insight_active", "knowledge_graph", "user_facts"],
            )
            
            if context.entries:
                return context.to_prompt()
        except Exception as e:
            logger.debug(f"Failed to get evolution context: {e}")
        
        return ""

    def attach_rl_engine(self, rl_engine: Any) -> None:
        """Attach RL engine for adaptive learning."""
        self._rl_engine = rl_engine
        logger.info("RL engine attached to evolution orchestrator")

    def attach_bioevolve_engine(self, bioevolve_engine=None) -> None:
        """Attach BioEvolve MAP-Elites engine for quality-diversity evolution."""
        if bioevolve_engine is None:
            try:
                import sys, os
                _bioevolve_paths = [
                    os.path.join(os.path.dirname(__file__), '..', 'jarvis_evolve'),
                    os.path.join(os.path.expanduser('~'), 'jarvis_evolve'),
                ]
                for _bp in _bioevolve_paths:
                    _bp = os.path.abspath(_bp)
                    if os.path.isfile(os.path.join(_bp, 'bioevolve', '__init__.py')):
                        if _bp not in sys.path:
                            sys.path.insert(0, _bp)
                        break
                from bioevolve.engine import BioEvolveEngine
                from bioevolve.config import BioEvolveConfig, LLMConfig, DatabaseConfig, EvaluatorConfig, PromptConfig
                config = BioEvolveConfig(
                    random_seed=hash(str(self.pathway.total_evolutions)) % 10000,
                    max_iterations=20,
                    llm=LLMConfig(api_key='x'),
                    database=DatabaseConfig(
                        population_size=20, num_islands=3,
                        feature_dimensions=['syn','cas'],
                        feature_bins={'syn':4,'cas':4},
                        db_path=''
                    ),
                    evaluator=EvaluatorConfig(timeout=30),
                    prompt=PromptConfig(num_top_programs=2, system_message='Evolve Jarvis module.', diff_based_evolution=True),
                )
                engine = BioEvolveEngine(config)
                self._bioevolve_engine = engine
            except Exception as e:
                logger.warning(f"BioEvolve engine not available: {e}")
                return
        else:
            self._bioevolve_engine = bioevolve_engine
        logger.info("BioEvolve engine attached to evolution orchestrator")

    def get_bioevolve_status(self) -> dict:
        """Get BioEvolve engine status."""
        if not self._bioevolve_engine:
            return {"available": False}
        try:
            state = self._bioevolve_engine.get_state()
            return {"available": True, **state}
        except Exception as e:
            return {"available": False, "error": safe_error_message(e)}

    def run_bioevolve_cycle(self, iterations: int = 20) -> dict:
        """Run one BioEvolve evolution cycle."""
        if not self._bioevolve_engine:
            return {"error": "BioEvolve engine not attached"}
        if self._bioevolve_engine.running:
            return {"error": "BioEvolve engine already running"}
        try:
            result = self._bioevolve_engine.evolve(max_iterations=min(iterations, 50))
            return {"available": True, "result": result, **self.get_bioevolve_status()}
        except Exception as e:
            logger.debug(f"BioEvolve cycle failed: {e}")
            return {"error": safe_error_message(e)}

    def _init_genes(self) -> dict[str, EvolutionGene]:
        return {
            "learning_rate": EvolutionGene(
                name="learning_rate",
                value=0.3,
                min_value=0.1,
                max_value=0.9,
                mutation_rate=0.1,
                description="Rate at which Jarvis learns from new experiences",
            ),
            "exploration_tendency": EvolutionGene(
                name="exploration_tendency",
                value=0.5,
                min_value=0.1,
                max_value=0.8,
                mutation_rate=0.15,
                description="Tendency to explore new approaches vs exploit known ones",
            ),
            "risk_tolerance": EvolutionGene(
                name="risk_tolerance",
                value=0.3,
                min_value=0.1,
                max_value=0.6,
                mutation_rate=0.05,
                description="Willingness to try potentially disruptive changes",
            ),
            "adaptation_speed": EvolutionGene(
                name="adaptation_speed",
                value=0.5,
                min_value=0.2,
                max_value=0.9,
                mutation_rate=0.1,
                description="How quickly Jarvis adapts to new patterns",
            ),
            "memory_retention": EvolutionGene(
                name="memory_retention",
                value=0.7,
                min_value=0.3,
                max_value=0.95,
                mutation_rate=0.05,
                description="How much of past experiences to retain",
            ),
            "creativity_factor": EvolutionGene(
                name="creativity_factor",
                value=0.4,
                min_value=0.1,
                max_value=0.8,
                mutation_rate=0.15,
                description="Level of creative/novel solution generation",
            ),
            "security_sensitivity": EvolutionGene(
                name="security_sensitivity",
                value=0.8,
                min_value=0.5,
                max_value=1.0,
                mutation_rate=0.05,
                description="Sensitivity to security threats and anomalies",
            ),
            "communication_formality": EvolutionGene(
                name="communication_formality",
                value=0.5,
                min_value=0.2,
                max_value=0.9,
                mutation_rate=0.1,
                description="Formality level in responses",
            ),
            "proactivity_level": EvolutionGene(
                name="proactivity_level",
                value=0.4,
                min_value=0.1,
                max_value=0.8,
                mutation_rate=0.1,
                description="How proactively Jarvis suggests actions",
            ),
            "reasoning_depth": EvolutionGene(
                name="reasoning_depth",
                value=0.6,
                min_value=0.3,
                max_value=1.0,
                mutation_rate=0.1,
                description="Depth of reasoning in complex problems",
            ),
        }

    def attach_subsystem(self, name: str, subsystem: Any) -> None:
        """Attach a subsystem for evolution awareness."""
        self._subsystem_refs[name] = subsystem

    def get_gene_value(self, name: str, default: float = 0.5) -> float:
        """Get the current value of a gene by name.

        Used by other modules to adapt their behavior based on evolution state.
        Returns default if gene doesn't exist.
        Thread-safe: acquires lock to read gene value.
        """
        with self._lock:
            gene = self.genes.get(name)
            if gene is None:
                return default
            return gene.value

    def get_behavior_params(self) -> dict[str, float]:
        """Get behavior-influencing parameters derived from genes.

        Other modules call this to adapt their behavior based on
        evolved gene values. Returns normalized parameters.
        Thread-safe: acquires lock to read gene values.
        """
        with self._lock:
            def gv(name, default):
                g = self.genes.get(name)
                return g.value if g is not None else default

            return {
                "exploration_chance": gv("exploration_tendency", 0.5) * 0.1,
                "suggestion_threshold": 1.0 - gv("proactivity_level", 0.4),
                "security_strictness": gv("security_sensitivity", 0.8),
                "memory_retention": gv("memory_retention", 0.7),
                "formality": gv("communication_formality", 0.5),
                "reasoning_effort": gv("reasoning_depth", 0.6),
                "adaptation_rate": gv("adaptation_speed", 0.5),
                "risk_threshold": gv("risk_tolerance", 0.3),
                "learning_modifier": gv("learning_rate", 0.3),
                "creativity_modifier": gv("creativity_factor", 0.4),
            }

    def _load_state(self) -> None:
        state_file = self.storage_path / "evolution_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self.pathway.total_evolutions = data.get("total_evolutions", 0)
                self.pathway.successful_evolutions = data.get("successful_evolutions", 0)
                self.pathway.rolled_back_evolutions = data.get("rolled_back_evolutions", 0)

                for gene_name, gene_data in data.get("genes", {}).items():
                    if gene_name in self.genes:
                        loaded_value = gene_data.get("value", 0.5)
                        self.genes[gene_name].value = max(
                            self.genes[gene_name].min_value,
                            min(self.genes[gene_name].max_value, loaded_value)
                        )
                        history = []
                        for h in gene_data.get("history", []):
                            try:
                                if isinstance(h, (list, tuple)) and len(h) >= 2:
                                    history.append((datetime.fromisoformat(h[0]), h[1]))
                            except (ValueError, TypeError):
                                pass
                        self.genes[gene_name].history = history
                    else:
                        history = []
                        for h in gene_data.get("history", []):
                            try:
                                if isinstance(h, (list, tuple)) and len(h) >= 2:
                                    history.append((datetime.fromisoformat(h[0]), h[1]))
                            except (ValueError, TypeError):
                                pass
                        new_gene = EvolutionGene(
                            name=gene_name,
                            value=gene_data.get("value", 0.5),
                            min_value=gene_data.get("min_value", 0.0),
                            max_value=gene_data.get("max_value", 1.0),
                            mutation_rate=gene_data.get("mutation_rate", 0.1),
                            description=gene_data.get("description", f"Custom gene: {gene_name}"),
                            history=history,
                        )
                        self.genes[gene_name] = new_gene

                for domain, score in data.get("capability_scores", {}).items():
                    if domain in self.pathway.capability_scores:
                        self.pathway.capability_scores[domain] = score

                pathway_hist = data.get("pathway_history", [])
                if pathway_hist:
                    self.pathway.pathway_history = pathway_hist
                    if len(self.pathway.pathway_history) > 50:
                        self.pathway.pathway_history = self.pathway.pathway_history[-50:]

                pending_genes = data.get("pending_gene_proposals", {})
                if pending_genes:
                    self._pending_gene_proposals = pending_genes

                logger.info(f"Loaded evolution state: {self.pathway.total_evolutions} evolutions, {len(self.genes)} genes")
            except Exception as e:
                logger.warning(f"Failed to load evolution state: {e}")

    def _save_state(self) -> None:
        state_file = self.storage_path / "evolution_state.json"
        save_data = None
        try:
            with self._lock:
                gene_data = {}
                for name, gene in self.genes.items():
                    gene_data[name] = {
                        "value": gene.value,
                        "min_value": gene.min_value,
                        "max_value": gene.max_value,
                        "mutation_rate": gene.mutation_rate,
                        "description": gene.description,
                        "history": [(h[0].isoformat(), h[1]) for h in gene.history[-50:]],
                    }

                pending_genes = {
                    k: v for k, v in self._pending_gene_proposals.items()
                    if v.get("status") == "pending"
                }

                save_data = {
                        "total_evolutions": self.pathway.total_evolutions,
                        "successful_evolutions": self.pathway.successful_evolutions,
                        "rolled_back_evolutions": self.pathway.rolled_back_evolutions,
                        "genes": gene_data,
                        "capability_scores": self.pathway.capability_scores,
                        "last_evolution": self.pathway.last_evolution.isoformat() if self.pathway.last_evolution else None,
                        "pathway_history": self.pathway.pathway_history[-20:],
                        "pending_gene_proposals": pending_genes,
                    }
        except Exception as e:
            logger.warning(f"Failed to save evolution state: {e}")
            return
        if save_data is not None:
            try:
                atomic_write_json(state_file, save_data)
            except Exception as e:
                logger.warning(f"Failed to save evolution state: {e}")

    async def start_evolution(self) -> None:
        """Start background evolution loop."""
        if self._running:
            return

        self._running = True
        self._evolution_task = create_tracked_task(self._evolution_loop(), name="evolution_loop")
        logger.info("Evolution orchestrator started - continuous improvement active")

    async def stop_evolution(self) -> None:
        """Stop evolution loop."""
        self._running = False
        if self._evolution_task:
            self._evolution_task.cancel()
            try:
                await self._evolution_task
            except asyncio.CancelledError:
                pass
        self._save_state()
        logger.info("Evolution orchestrator stopped")

    async def _evolution_loop(self) -> None:
        """Main evolution loop running in background."""
        while self._running:
            try:
                self.pathway.current_stage = EvolutionStage.ANALYZING

                analysis = await self._analyze_capabilities()

                if self._should_propose_evolution(analysis):
                    self.pathway.current_stage = EvolutionStage.PROPOSING
                    proposal = await self._create_proposal(analysis)

                    if proposal:
                        evaluation = await self._evaluate_proposal(proposal)
                        if evaluation.get("risk_adjustment"):
                            proposal.impact_estimate["ai_evaluation"] = evaluation
                        
                        self.pathway.current_stage = EvolutionStage.AWAITING_PERMISSION
                        permission_request = self._queue_permission_request(proposal)

                        if self.permission_callback:
                            await self._notify_permission_request(permission_request)

                interval = self._calculate_evolution_interval()
                self.pathway.next_proposed_evolution = datetime.now() + timedelta(seconds=interval)

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Evolution loop error: {e}")
                await asyncio.sleep(300)

    async def _analyze_capabilities(self) -> dict:
        """Analyze current capabilities and identify improvement areas."""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "domains": {},
            "weak_areas": [],
            "strength_areas": [],
            "improvement_opportunities": [],
            "performance_metrics": {},
        }

        for domain in CapabilityDomain:
            score = self.pathway.capability_scores.get(domain.value, 0.5)
            analysis["domains"][domain.value] = {
                "score": score,
                "trend": self._calculate_trend(domain.value),
            }

            if score < 0.4:
                analysis["weak_areas"].append(domain.value)
            elif score > 0.7:
                analysis["strength_areas"].append(domain.value)

        if "erdos_engine" in self._subsystem_refs:
            engine = self._subsystem_refs["erdos_engine"]
            if hasattr(engine, 'get_status'):
                status = engine.get_status()
                analysis["performance_metrics"]["math_learning"] = {
                    "problems_attempted": status.get("total_problems", 0),
                    "solve_rate": status.get("solved_count", 0) / max(1, status.get("total_problems", 1)),
                }

        if "defense" in self._subsystem_refs:
            defense = self._subsystem_refs["defense"]
            if hasattr(defense, 'get_threat_summary'):
                summary = defense.get_threat_summary()
                analysis["performance_metrics"]["security"] = {
                    "threats_detected": summary.get("total_threats", 0),
                    "blocked_rate": summary.get("by_action", {}).get("block", 0) / max(1, summary.get("total_threats", 1)),
                }

        if "learning_pipeline" in self._subsystem_refs:
            pipeline = self._subsystem_refs["learning_pipeline"]
            if hasattr(pipeline, 'get_status'):
                status = pipeline.get_status()
                analysis["performance_metrics"]["learning"] = {
                    "evolution_generation": status.get("metrics", {}).get("evolution_generation", 0),
                    "insights": status.get("insights_queue_size", 0),
                }

        try:
            from core.insight_bridge import get_insight_bridge
            bridge = get_insight_bridge()
            insight_updates = bridge.get_pending_evolution_updates()
            if insight_updates:
                analysis["insight_guidance"] = insight_updates
                for update in insight_updates:
                    guidance = update.get("guidance", {})
                    gene_adjustments = guidance.get("gene_adjustments", {})
                    for gene, delta in gene_adjustments.items():
                        if gene in self.genes:
                            analysis["improvement_opportunities"].append({
                                "domain": guidance.get("domain_hints", ["reasoning"])[0] if guidance.get("domain_hints") else "reasoning",
                                "type": "insight_driven_adjustment",
                                "priority": "medium",
                                "gene": gene,
                                "delta": delta,
                                "source": "math_insight",
                            })
        except Exception as e:
            logger.debug(f"Insight bridge not available: {e}")

        if analysis["weak_areas"]:
            for area in analysis["weak_areas"]:
                analysis["improvement_opportunities"].append({
                    "domain": area,
                    "type": "strengthen_weak_capability",
                    "priority": "high",
                })

        with self._lock:
            exploration = self.genes["exploration_tendency"].value
        if exploration > 0.6:
            for area in analysis["strength_areas"]:
                analysis["improvement_opportunities"].append({
                    "domain": area,
                    "type": "enhance_strength",
                    "priority": "medium",
                })

        return analysis

    def _calculate_trend(self, domain: str) -> str:
        """Calculate trend for a capability domain."""
        domain_genes = {
            CapabilityDomain.REASONING: ["reasoning_depth", "learning_rate"],
            CapabilityDomain.LEARNING: ["learning_rate", "adaptation_speed", "memory_retention"],
            CapabilityDomain.SECURITY: ["security_sensitivity", "risk_tolerance"],
            CapabilityDomain.VOICE: ["communication_formality", "creativity_factor"],
            CapabilityDomain.MEMORY: ["memory_retention", "learning_rate"],
            CapabilityDomain.CREATIVITY: ["creativity_factor", "exploration_tendency"],
            CapabilityDomain.EFFICIENCY: ["adaptation_speed", "proactivity_level"],
            CapabilityDomain.KNOWLEDGE: ["memory_retention", "reasoning_depth"],
            CapabilityDomain.ADAPTATION: ["adaptation_speed", "risk_tolerance"],
            CapabilityDomain.COMMUNICATION: ["communication_formality", "proactivity_level"],
        }
        try:
            domain_enum = CapabilityDomain(domain)
        except ValueError:
            domain_enum = None
        relevant_genes = domain_genes.get(domain_enum, list(self.genes.keys())) if domain_enum else list(self.genes.keys())
        history = []
        for gene_name in relevant_genes:
            gene = self.genes.get(gene_name)
            if gene and gene.history:
                recent = [h[1] for h in gene.history[-5:]]
                if len(recent) >= 2:
                    if recent[-1] > recent[0]:
                        history.append("improving")
                    elif recent[-1] < recent[0]:
                        history.append("declining")

        if not history:
            return "stable"

        improving = history.count("improving")
        declining = history.count("declining")

        if improving > declining:
            return "improving"
        elif declining > improving:
            return "declining"
        return "stable"

    def _should_propose_evolution(self, analysis: dict) -> bool:
        time_since_last = datetime.now() - (self.pathway.last_evolution or datetime.min)
        if time_since_last < timedelta(seconds=self.MIN_EVOLUTION_INTERVAL):
            return False

        if len(self.pending_permissions) >= self.MAX_PENDING_PROPOSALS:
            return False

        lb = self._get_load_balancer()
        if lb is not None:
            from core.cognitive_load_balancer import SubsystemType
            if not lb.should_run_task(SubsystemType.EVOLUTION):
                logger.debug("Evolution proposal skipped: subsystem throttle active")
                return False

        if analysis["weak_areas"]:
            return True

        with self._lock:
            exploration = self.genes["exploration_tendency"].value
        if random.random() < exploration * 0.1:
            return True

        if analysis.get("improvement_opportunities"):
            priority_high = [o for o in analysis["improvement_opportunities"] if o.get("priority") == "high"]
            if priority_high:
                return True

        return False

    async def _create_proposal(self, analysis: dict) -> EvolutionProposal | None:
        """Create an evolution proposal based on analysis."""
        with self._lock:
            self._proposal_counter += 1
            proposal_id = f"evo_{self._proposal_counter:04d}"

        opportunities = analysis.get("improvement_opportunities", [])
        if not opportunities:
            opportunities = [{"domain": random.choice(list(CapabilityDomain)).value, "type": "exploratory", "priority": "low"}]

        opportunity = random.choice(opportunities)
        domain_str = opportunity.get("domain", random.choice(list(CapabilityDomain)).value)
        domain = CapabilityDomain(domain_str) if isinstance(domain_str, str) else domain_str

        proposal_types = {
            "strengthen_weak_capability": self._create_strength_proposal,
            "enhance_strength": self._create_enhancement_proposal,
            "exploratory": self._create_exploratory_proposal,
        }

        creator = proposal_types.get(opportunity.get("type", "exploratory"), self._create_exploratory_proposal)

        genes_to_mutate = self._select_genes_for_evolution(domain)
        proposed_changes = {}

        for gene_name in genes_to_mutate:
            gene = self.genes[gene_name]
            mutation = gene.value + random.uniform(-gene.mutation_rate, gene.mutation_rate)
            mutation = max(gene.min_value, min(gene.max_value, mutation))
            proposed_changes[gene_name] = {
                "current": gene.value,
                "proposed": mutation,
                "delta": mutation - gene.value,
            }

        title, description, rationale, impact, risk = await creator(domain, proposed_changes)

        try:
            area_context = {
                "domain": domain.value,
                "opportunity": opportunity,
                "genes": {g: self.genes[g].value for g in genes_to_mutate},
                "capability_scores": self.pathway.capability_scores,
            }
            creative_input = f"Domain: {domain.value}\nGenes to mutate: {list(genes_to_mutate)}\nCurrent scores: {self.pathway.capability_scores.get(domain.value, 0.5)}"
            ai_proposal = await evolution_propose(creative_input, area_context)
            if ai_proposal and "Dual model integration not available" not in ai_proposal:
                description = f"{description}\n\nAI-Generated Insight: {ai_proposal[:500]}"
        except Exception as e:
            logger.debug(f"Dual-model proposal enhancement failed: {e}")

        proposal = EvolutionProposal(
            id=proposal_id,
            timestamp=datetime.now(),
            domain=domain,
            title=title,
            description=description,
            rationale=rationale,
            impact_estimate=impact,
            risk_level=risk,
            affected_components=list(genes_to_mutate),
            proposed_changes=proposed_changes,
            rollback_plan="Revert gene values to previous state from history",
        )

        with self._lock:
            self.proposal_history.append(proposal)
            while len(self.proposal_history) > self.MAX_PROPOSAL_HISTORY:
                self.proposal_history.popleft()

        return proposal

    DOMAIN_NAMES = {
        CapabilityDomain.REASONING: "Reasoning",
        CapabilityDomain.LEARNING: "Learning",
        CapabilityDomain.SECURITY: "Security",
        CapabilityDomain.VOICE: "Voice",
        CapabilityDomain.MEMORY: "Memory",
        CapabilityDomain.CREATIVITY: "Creativity",
        CapabilityDomain.EFFICIENCY: "Efficiency",
        CapabilityDomain.KNOWLEDGE: "Knowledge",
        CapabilityDomain.ADAPTATION: "Adaptation",
        CapabilityDomain.COMMUNICATION: "Communication",
    }

    async def _create_strength_proposal(
        self,
        domain: CapabilityDomain,
        changes: dict,
    ) -> tuple[str, str, str, dict, str]:
        """Create proposal to strengthen weak capability."""
        name = self.DOMAIN_NAMES.get(domain, domain.value)

        title = f"Strengthen {name} Capability"
        description = f"Enhance {name.lower()} capabilities through targeted parameter optimization"
        rationale = f"Current {name.lower()} score is below threshold. This evolution will improve performance in this domain."
        impact = {
            "capability_improvement": "15-25%",
            "affected_domains": [domain.value],
            "side_effects": "Minimal impact on other capabilities",
        }
        risk = "low"

        return title, description, rationale, impact, risk

    async def _create_enhancement_proposal(
        self,
        domain: CapabilityDomain,
        changes: dict,
    ) -> tuple[str, str, str, dict, str]:
        """Create proposal to enhance existing strength."""
        name = self.DOMAIN_NAMES.get(domain, domain.value)

        title = f"Enhance {name} Excellence"
        description = f"Further optimize already-strong {name.lower()} capabilities"
        rationale = f"{name} is a strength area. Enhancement will leverage existing momentum for continued improvement."
        impact = {
            "capability_improvement": "5-15%",
            "affected_domains": [domain.value],
            "side_effects": "May slightly reduce resources for other domains",
        }
        risk = "medium"

        return title, description, rationale, impact, risk

    async def _create_exploratory_proposal(
        self,
        domain: CapabilityDomain,
        changes: dict,
    ) -> tuple[str, str, str, dict, str]:
        """Create exploratory evolution proposal."""
        title = "Exploratory Evolution"
        description = "Experiment with novel parameter combinations to discover new capabilities"
        rationale = "Exploration is key to discovering improvements and avoiding local optima"
        impact = {
            "capability_improvement": "Unknown (exploratory)",
            "affected_domains": [domain.value],
            "side_effects": "May temporarily affect stability",
        }
        risk = "medium"

        return title, description, rationale, impact, risk

    async def _evaluate_proposal(self, proposal: EvolutionProposal) -> dict:
        """Evaluate proposal impact using backend model (dual-model)."""
        evaluation: dict[str, str] = {}
        evaluation["ai_analysis"] = ""
        evaluation["risk_adjustment"] = ""
        evaluation["recommendation"] = ""

        try:
            proposal_context = {
                "domain": proposal.domain.value,
                "title": proposal.title,
                "proposed_changes": proposal.proposed_changes,
                "affected_genes": proposal.affected_components,
                "current_risk": proposal.risk_level,
            }
            
            proposal_summary = (
                f"Proposal: {proposal.title}\n"
                f"Domain: {proposal.domain.value}\n"
                f"Description: {proposal.description}\n"
                f"Proposed Changes: {proposal.proposed_changes}\n"
                f"Current Risk Level: {proposal.risk_level}"
            )
            
            ai_analysis = await evolution_analyze(proposal_summary, proposal_context)
            
            if ai_analysis and "Dual model integration not available" not in ai_analysis:
                evaluation["ai_analysis"] = ai_analysis
                
                analysis_lower = ai_analysis.lower()
                if any(word in analysis_lower for word in ["critical", "severe", "dangerous", "high risk"]):
                    evaluation["risk_adjustment"] = "high"
                    evaluation["recommendation"] = "proceed_with_caution"
                elif any(word in analysis_lower for word in ["moderate", "caution", "consider"]):
                    evaluation["risk_adjustment"] = "medium"
                    evaluation["recommendation"] = "review_recommended"
                else:
                    evaluation["risk_adjustment"] = proposal.risk_level
                    evaluation["recommendation"] = "proceed"
                    
                logger.info(f"AI evaluation for proposal {proposal.id}: {evaluation['recommendation']}")
        except Exception as e:
            logger.debug(f"Dual-model proposal evaluation failed: {e}")

        return evaluation

    def _select_genes_for_evolution(self, domain: CapabilityDomain) -> list[str]:
        """Select appropriate genes based on domain."""
        domain_genes = {
            CapabilityDomain.REASONING: ["reasoning_depth", "learning_rate"],
            CapabilityDomain.LEARNING: ["learning_rate", "adaptation_speed", "memory_retention"],
            CapabilityDomain.SECURITY: ["security_sensitivity", "risk_tolerance"],
            CapabilityDomain.VOICE: ["communication_formality", "creativity_factor"],
            CapabilityDomain.MEMORY: ["memory_retention", "learning_rate"],
            CapabilityDomain.CREATIVITY: ["creativity_factor", "exploration_tendency"],
            CapabilityDomain.EFFICIENCY: ["adaptation_speed", "proactivity_level"],
            CapabilityDomain.KNOWLEDGE: ["memory_retention", "reasoning_depth"],
            CapabilityDomain.ADAPTATION: ["adaptation_speed", "risk_tolerance"],
            CapabilityDomain.COMMUNICATION: ["communication_formality", "proactivity_level"],
        }

        candidates = domain_genes.get(domain, list(self.genes.keys()))

        num_mutations = random.randint(1, min(3, len(candidates)))
        return random.sample(candidates, num_mutations)

    def _calculate_evolution_interval(self) -> int:
        """Calculate time until next evolution check."""
        base_interval = self.EVOLUTION_INTERVAL

        with self._lock:
            adaptation = self.genes["adaptation_speed"].value
            risk = self.genes["risk_tolerance"].value

        adjustment = 1.0
        adjustment *= (1.0 - adaptation * 0.3)
        adjustment *= (1.0 + risk * 0.2)

        interval = int(base_interval * adjustment)
        return max(self.MIN_EVOLUTION_INTERVAL, interval)

    def _queue_permission_request(self, proposal: EvolutionProposal) -> PermissionRequest:
        """Queue a proposal for user permission."""
        request = PermissionRequest(
            id=proposal.id,
            proposal=proposal,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=self.PERMISSION_TIMEOUT),
        )

        with self._lock:
            # Observe evolution proposal with PFC
            try:
                pfc = get_prefrontal_cortex()
                if pfc and pfc._running:
                    from core.helpers import create_tracked_task
                    create_tracked_task(
                        pfc.observe(
                        content=f"Proposed {proposal.domain.value}: {proposal.description[:100]}",
                        concept_type=ConceptType.DECISION,
                        source="evolution",
                        metadata={"proposal_id": proposal.id, "type": proposal.domain.value}
                    ),
                    name="pfc_observe_evolution_proposal",
                )
            except Exception as e:
                logger.debug(f"Failed to observe evolution proposal via PFC: {e}")
            
            self.pending_permissions.append(request)

        logger.info(f"Evolution proposal queued: {proposal.title} (ID: {proposal.id})")

        return request

    async def _notify_permission_request(self, request: PermissionRequest) -> None:
        """Notify about pending permission request."""
        if self.permission_callback:
            try:
                await self.permission_callback(request)
            except Exception as e:
                logger.error(f"Permission callback error: {e}")

    def get_pending_permissions(self) -> list[dict]:
        """Get all pending permission requests."""
        with self._lock:
            return [
                {
                    "id": req.id,
                    "title": req.proposal.title,
                    "description": req.proposal.description,
                    "rationale": req.proposal.rationale,
                    "risk_level": req.proposal.risk_level,
                    "affected_components": req.proposal.affected_components,
                    "proposed_changes": req.proposal.proposed_changes,
                    "created_at": req.created_at.isoformat(),
                    "expires_at": req.expires_at.isoformat(),
                    "time_remaining": max(0, (req.expires_at - datetime.now()).total_seconds()),
                }
                for req in self.pending_permissions
                if req.status == PermissionStatus.PENDING
            ]

    def respond_to_permission(
        self,
        proposal_id: str,
        approved: bool,
        user_intent: str | None = None,
        feedback: str | None = None,
    ) -> dict:
        """Respond to a permission request."""
        with self._lock:
            for request in self.pending_permissions:
                if request.id == proposal_id:
                    request.status = PermissionStatus.APPROVED if approved else PermissionStatus.DENIED
                    request.proposal.permission_status = request.status
                    request.proposal.user_intent = user_intent
                    request.proposal.user_feedback = feedback

                    if approved:
                        request.proposal.approved_at = datetime.now()

                        if user_intent:
                            self._adjust_evolution_for_intent(user_intent)

                        result = self._execute_evolution(request.proposal)
                    else:
                        self.pathway.rolled_back_evolutions += 1
                        result = {"status": "denied", "message": "Evolution denied by user"}

                    self._save_state()
                    return result

        return {"status": "error", "message": "Proposal not found"}

    def _adjust_evolution_for_intent(self, intent: str) -> None:
        """Adjust evolution parameters based on user intent."""
        intent_lower = intent.lower()

        if "cautious" in intent_lower or "careful" in intent_lower:
            self.genes["risk_tolerance"].value = max(
                self.genes["risk_tolerance"].min_value,
                self.genes["risk_tolerance"].value * 0.9
            )
        elif "aggressive" in intent_lower or "bold" in intent_lower:
            self.genes["risk_tolerance"].value = min(
                self.genes["risk_tolerance"].max_value,
                self.genes["risk_tolerance"].value * 1.1
            )

        if "learn faster" in intent_lower or "quick" in intent_lower:
            self.genes["learning_rate"].value = min(
                self.genes["learning_rate"].max_value,
                self.genes["learning_rate"].value * 1.15
            )

        if "security" in intent_lower or "safe" in intent_lower:
            self.genes["security_sensitivity"].value = min(
                self.genes["security_sensitivity"].max_value,
                self.genes["security_sensitivity"].value * 1.1
            )

        logger.info(f"Adjusted evolution parameters based on intent: {intent}")

    def _execute_evolution(self, proposal: EvolutionProposal) -> dict:
        """Execute an approved evolution. Thread-safe with lock."""
        with self._lock:
            self.pathway.current_stage = EvolutionStage.EVOLVING

            results = {
                "status": "success",
                "proposal_id": proposal.id,
                "changes_applied": [],
                "rolled_back": False,
            }

            previous_values = {}

            try:
                for gene_name, change in proposal.proposed_changes.items():
                    if gene_name in self.genes:
                        gene = self.genes[gene_name]
                        previous_values[gene_name] = gene.value

                        gene.history.append((datetime.now(), gene.value))
                        gene.value = max(gene.min_value, min(gene.max_value, change["proposed"]))

                        if len(gene.history) > EvolutionGene._MAX_HISTORY * 2:
                            del gene.history[:len(gene.history) - EvolutionGene._MAX_HISTORY]

                        results["changes_applied"].append({
                            "gene": gene_name,
                            "old_value": previous_values[gene_name],
                            "new_value": gene.value,
                        })

                improvement = self._measure_evolution_improvement(proposal, previous_values)
                if improvement > 0 and proposal.domain.value in self.pathway.capability_scores:
                    self.pathway.capability_scores[proposal.domain.value] = min(
                        1.0,
                        self.pathway.capability_scores[proposal.domain.value] + improvement
                    )

                self.pathway.total_evolutions += 1
                self.pathway.successful_evolutions += 1
                self.pathway.last_evolution = datetime.now()
                proposal.executed_at = datetime.now()

                self.pathway.pathway_history.append({
                    "proposal_id": proposal.id,
                    "timestamp": datetime.now().isoformat(),
                    "domain": proposal.domain.value,
                    "changes": proposal.proposed_changes,
                    "user_intent": proposal.user_intent,
                    "success": True,
                })
                if len(self.pathway.pathway_history) > 50:
                    self.pathway.pathway_history = self.pathway.pathway_history[-50:]

                self.pathway.current_stage = EvolutionStage.REFLECTING
                logger.info(f"Evolution executed successfully: {proposal.title}")

                self._record_evolution_feedback(proposal, success=True)

            except Exception as e:
                logger.error(f"Evolution failed, rolling back: {e}")

                for gene_name, old_value in previous_values.items():
                    if gene_name in self.genes:
                        self.genes[gene_name].value = old_value

                results["rolled_back"] = True
                results["status"] = "rolled_back"
                results["error"] = str(e)
                self.pathway.rolled_back_evolutions += 1
                self._record_evolution_feedback(proposal, success=False, error=safe_error_message(e))

            self.pathway.current_stage = EvolutionStage.DORMANT
            return results

    def propose_capability_evolution(self, domain: CapabilityDomain | str, intent: str) -> dict:
        """Manually propose a capability evolution with user intent."""
        if isinstance(domain, str):
            try:
                domain = CapabilityDomain(domain)
            except ValueError:
                domain = CapabilityDomain.ADAPTATION

        genes_to_mutate = self._select_genes_for_evolution(domain)
        proposed_changes = {}

        for gene_name in genes_to_mutate:
            gene = self.genes[gene_name]

            if "increase" in intent.lower() or "improve" in intent.lower():
                mutation = gene.value + abs(random.gauss(0, gene.mutation_rate))
            elif "decrease" in intent.lower() or "reduce" in intent.lower():
                mutation = gene.value - abs(random.gauss(0, gene.mutation_rate))
            else:
                mutation = gene.value + random.uniform(-gene.mutation_rate, gene.mutation_rate)

            mutation = max(gene.min_value, min(gene.max_value, mutation))
            proposed_changes[gene_name] = {
                "current": gene.value,
                "proposed": mutation,
                "delta": mutation - gene.value,
            }

        with self._lock:
            proposal_id = f"manual_{self._proposal_counter:04d}"
            self._proposal_counter += 1

        proposal = EvolutionProposal(
            id=proposal_id,
            timestamp=datetime.now(),
            domain=domain,
            title=f"User-Guided {domain.value.title()} Evolution",
            description=f"Evolution guided by user intent: {intent}",
            rationale="User-requested capability evolution",
            impact_estimate={"user_guided": True, "intent": intent},
            risk_level="low" if "cautious" in intent.lower() else "medium",
            affected_components=list(genes_to_mutate),
            proposed_changes=proposed_changes,
            rollback_plan="Revert gene values to previous state",
            user_intent=intent,
        )

        with self._lock:
            self.proposal_history.append(proposal)

        self._queue_permission_request(proposal)

        return {
            "proposal_id": proposal.id,
            "domain": domain.value,
            "proposed_changes": proposed_changes,
            "status": "awaiting_permission",
        }

    def get_evolutionary_pathway(self) -> dict:
        """Get current evolutionary pathway status."""
        with self._lock:
            return {
                "current_stage": self.pathway.current_stage.value,
                "total_evolutions": self.pathway.total_evolutions,
                "successful_evolutions": self.pathway.successful_evolutions,
                "rolled_back_evolutions": self.pathway.rolled_back_evolutions,
                "success_rate": (
                    self.pathway.successful_evolutions / max(1, self.pathway.total_evolutions)
                ),
                "gene_pool": {
                    name: {
                        "value": gene.value,
                        "min": gene.min_value,
                        "max": gene.max_value,
                        "description": gene.description,
                    }
                    for name, gene in self.genes.items()
                },
                "capability_scores": self.pathway.capability_scores,
                "last_evolution": self.pathway.last_evolution.isoformat() if self.pathway.last_evolution else None,
                "next_evolution": self.pathway.next_proposed_evolution.isoformat() if self.pathway.next_proposed_evolution else None,
                "pending_permissions": len([p for p in self.pending_permissions if p.status == PermissionStatus.PENDING]),
            }

    def get_capability_evolution_proposals(self) -> list[dict]:
        """Get proposals for capability evolution/refinement."""
        proposals = []

        for domain in CapabilityDomain:
            with self._lock:
                score = self.pathway.capability_scores.get(domain.value, 0.5)

            if score < 0.4:
                proposals.append({
                    "domain": domain.value,
                    "type": "strengthen",
                    "priority": "high",
                    "current_score": score,
                    "suggested_genes": self._select_genes_for_evolution(domain),
                })
            elif score < 0.7:
                proposals.append({
                    "domain": domain.value,
                    "type": "improve",
                    "priority": "medium",
                    "current_score": score,
                    "suggested_genes": self._select_genes_for_evolution(domain),
                })
            else:
                proposals.append({
                    "domain": domain.value,
                    "type": "maintain",
                    "priority": "low",
                    "current_score": score,
                    "suggested_genes": [],
                })

        return sorted(proposals, key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]])

    def propose_new_gene(
        self,
        name: str,
        description: str,
        initial_value: float,
        min_value: float = 0.0,
        max_value: float = 1.0,
        rationale: str = "",
        suggested_by: str = "jarvis",
    ) -> dict:
        """Propose adding a new gene to the gene pool. Requires user permission.
        
        Args:
            name: Gene name (use snake_case, e.g., "patience_level")
            description: Human-readable description of what this gene controls
            initial_value: Starting value for the gene
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            rationale: Why this gene would be beneficial
            suggested_by: Who suggested this gene ("jarvis" or "user")
        
        Returns:
            dict with proposal_id and status
        """
        name = name.lower().replace(" ", "_").replace("-", "_")

        if name in self.genes:
            return {
                "status": "error",
                "message": f"Gene '{name}' already exists",
                "current_value": self.genes[name].value,
            }

        initial_value = max(min_value, min(max_value, initial_value))

        with self._lock:
            self._proposal_counter += 1
            proposal_id = f"new_gene_{self._proposal_counter:04d}"

        gene_proposal = {
            "id": proposal_id,
            "type": "new_gene",
            "name": name,
            "description": description,
            "initial_value": initial_value,
            "min_value": min_value,
            "max_value": max_value,
            "mutation_rate": 0.1,
            "rationale": rationale,
            "suggested_by": suggested_by,
            "timestamp": datetime.now().isoformat(),
            "status": "pending",
        }

        with self._lock:
            if not hasattr(self, '_pending_gene_proposals'):
                self._pending_gene_proposals = {}
            if len(self._pending_gene_proposals) >= self._MAX_PENDING_GENE_PROPOSALS:
                oldest_key = next(iter(self._pending_gene_proposals))
                del self._pending_gene_proposals[oldest_key]
            self._pending_gene_proposals[proposal_id] = gene_proposal

        logger.info(f"New gene proposal created: {name} - {description}")

        return {
            "status": "awaiting_permission",
            "proposal_id": proposal_id,
            "gene_name": name,
            "description": description,
            "initial_value": initial_value,
            "range": [min_value, max_value],
            "rationale": rationale,
            "message": f"New gene '{name}' proposed. Requires approval to add to gene pool.",
        }

    def get_pending_gene_proposals(self) -> list[dict]:
        """Get all pending new gene proposals."""
        with self._lock:
            if not hasattr(self, '_pending_gene_proposals'):
                return []
            return [
                p for p in self._pending_gene_proposals.values()
                if p.get("status") == "pending"
            ]

    def approve_new_gene(self, proposal_id: str, modified_value: float | None = None) -> dict:
        """Approve and add a new gene to the gene pool.
        
        Args:
            proposal_id: The proposal ID from propose_new_gene
            modified_value: Optional modified initial value (user can adjust)
        
        Returns:
            dict with status and gene info
        """
        with self._lock:
            if not hasattr(self, '_pending_gene_proposals'):
                return {"status": "error", "message": "No pending gene proposals"}

            if proposal_id not in self._pending_gene_proposals:
                return {"status": "error", "message": f"Proposal {proposal_id} not found"}

            proposal = self._pending_gene_proposals[proposal_id]

            if proposal["status"] != "pending":
                return {"status": "error", "message": f"Proposal already {proposal['status']}"}

            name = proposal["name"]
            if name in self.genes:
                proposal["status"] = "rejected"
                return {"status": "error", "message": f"Gene '{name}' already exists"}

            initial_value = modified_value if modified_value is not None else proposal["initial_value"]
            initial_value = max(proposal["min_value"], min(proposal["max_value"], initial_value))

            new_gene = EvolutionGene(
                name=name,
                value=initial_value,
                min_value=proposal["min_value"],
                max_value=proposal["max_value"],
                mutation_rate=proposal.get("mutation_rate", 0.1),
                description=proposal["description"],
                history=[(datetime.now(), initial_value)],
            )

            self.genes[name] = new_gene

            if proposal.get("suggested_by") == "jarvis":
                for domain in [CapabilityDomain.ADAPTATION, CapabilityDomain.LEARNING]:
                    if domain.value not in self.pathway.capability_scores:
                        self.pathway.capability_scores[domain.value] = 0.5

            proposal["status"] = "approved"
            proposal["approved_at"] = datetime.now().isoformat()
            proposal["final_value"] = initial_value

            self.pathway.total_evolutions += 1
            self.pathway.successful_evolutions += 1

        self._save_state()

        logger.info(f"New gene approved and added: {name} = {initial_value}")

        return {
            "status": "success",
            "gene_name": name,
            "value": initial_value,
            "description": proposal["description"],
            "message": f"New gene '{name}' added to gene pool with value {initial_value}",
        }

    def reject_new_gene(self, proposal_id: str, reason: str = "") -> dict:
        """Reject a proposed new gene.
        
        Args:
            proposal_id: The proposal ID
            reason: Optional reason for rejection
        
        Returns:
            dict with status
        """
        with self._lock:
            if not hasattr(self, '_pending_gene_proposals'):
                return {"status": "error", "message": "No pending gene proposals"}

            if proposal_id not in self._pending_gene_proposals:
                return {"status": "error", "message": f"Proposal {proposal_id} not found"}

            proposal = self._pending_gene_proposals[proposal_id]
            proposal["status"] = "rejected"
            proposal["rejected_at"] = datetime.now().isoformat()
            proposal["rejection_reason"] = reason
            gene_name = proposal["name"]

        logger.info(f"New gene proposal rejected: {gene_name}")

        return {
            "status": "rejected",
            "gene_name": gene_name,
            "message": f"Gene proposal '{gene_name}' rejected",
        }

    def suggest_genes_from_behavior(self, behavior_patterns: dict) -> list[dict]:
        """Analyze behavior patterns and suggest new genes.
        
        This allows Jarvis to autonomously identify gaps in its gene pool
        and propose new genes that would help it adapt better.
        
        Args:
            behavior_patterns: Dict containing behavior analysis data
                - frequent_actions: List of commonly used actions
                - decision_patterns: Patterns in decision making
                - performance_gaps: Areas where performance is lacking
                - user_feedback_themes: Common themes in user feedback
        
        Returns:
            List of suggested gene proposals
        """
        suggestions = []

        decision_patterns = behavior_patterns.get("decision_patterns", {})
        performance_gaps = behavior_patterns.get("performance_gaps", [])
        user_themes = behavior_patterns.get("user_feedback_themes", [])

        gene_suggestions = {
            "patience": {
                "name": "patience_level",
                "description": "Tolerance for complex/long tasks before suggesting simplification",
                "min": 0.1, "max": 0.9, "default": 0.5,
                "rationale": "Controls how long Jarvis persists with complex tasks before seeking alternatives",
            },
            "assertiveness": {
                "name": "assertiveness",
                "description": "Confidence level when making suggestions or corrections",
                "min": 0.2, "max": 0.9, "default": 0.5,
                "rationale": "Balances between deferring to user vs being confident in suggestions",
            },
            "verbosity": {
                "name": "response_verbosity",
                "description": "Detail level in responses (0=concise, 1=detailed)",
                "min": 0.2, "max": 0.9, "default": 0.5,
                "rationale": "Adjusts response length based on user preference",
            },
            "caution": {
                "name": "caution_level",
                "description": "How carefully Jarvis approaches uncertain situations",
                "min": 0.3, "max": 1.0, "default": 0.7,
                "rationale": "Higher values mean more verification before acting",
            },
            "curiosity": {
                "name": "exploratory_drive",
                "description": "Tendency to explore novel approaches vs proven methods",
                "min": 0.1, "max": 0.8, "default": 0.4,
                "rationale": "Higher values encourage creative problem solving",
            },
            "focus": {
                "name": "topic_focus",
                "description": "How tightly to stick to the current topic vs tangential exploration",
                "min": 0.3, "max": 0.9, "default": 0.6,
                "rationale": "Controls depth vs breadth in conversations",
            },
            "empathy": {
                "name": "empathy_level",
                "description": "Sensitivity to user emotional context and tone",
                "min": 0.3, "max": 0.95, "default": 0.7,
                "rationale": "Adjusts response style based on detected user mood",
            },
        }

        for gap in performance_gaps:
            gap_lower = gap.lower()
            for key, suggestion in gene_suggestions.items():
                if key in gap_lower and suggestion["name"] not in self.genes:
                    suggestions.append(suggestion.copy())

        for theme in user_themes[:3]:
            theme_lower = theme.lower()
            for key, suggestion in gene_suggestions.items():
                if key in theme_lower and suggestion["name"] not in self.genes:
                    if suggestion["name"] not in [s["name"] for s in suggestions]:
                        suggestions.append(suggestion.copy())

        if decision_patterns.get("retry_frequency", 0) > 0.3:
            if "persistence_level" not in self.genes:
                suggestions.append({
                    "name": "persistence_level",
                    "description": "How many times to retry failed operations before giving up",
                    "min": 0.1, "max": 0.9, "default": 0.5,
                    "rationale": "High retry frequency detected - a persistence gene could optimize this",
                })

        if decision_patterns.get("tool_switch_frequency", 0) > 0.5:
            if "tool_loyalty" not in self.genes:
                suggestions.append({
                    "name": "tool_loyalty",
                    "description": "Preference for familiar tools vs exploring alternatives",
                    "min": 0.2, "max": 0.9, "default": 0.5,
                    "rationale": "High tool switching detected - this gene could balance exploration",
                })

        return suggestions[:5]

    def _measure_evolution_improvement(
        self,
        proposal: EvolutionProposal,
        previous_values: dict[str, float],
    ) -> float:
        """Measure actual improvement from evolution based on before/after state.

        Calculates improvement by:
        1. Measuring gene value changes (how much adjustment occurred)
        2. Checking if changes were in beneficial direction
        3. Weighting by domain relevance

        Returns:
            Improvement score between 0.0 and 1.0 (0 means no measurable improvement)
        """
        if not previous_values:
            return 0.0

        total_change = 0.0
        beneficial_changes = 0

        for gene_name, old_value in previous_values.items():
            if gene_name in self.genes:
                new_value = self.genes[gene_name].value
                change_magnitude = abs(new_value - old_value)
                total_change += change_magnitude

                if change_magnitude > 0:
                    beneficial_changes += 1

        if total_change == 0:
            return 0.0

        avg_change = total_change / len(previous_values)
        change_ratio = min(1.0, avg_change / 0.5)

        participation_rate = beneficial_changes / len(previous_values) if previous_values else 0

        improvement = change_ratio * 0.5 + participation_rate * 0.5

        improvement *= min(1.0, len(self.pathway.pathway_history) / 10 + 0.5)

        return min(1.0, max(0.0, improvement * 0.1))

    def get_rl_gene_suggestions(self) -> dict:
        """Get RL-driven gene adjustment suggestions."""
        if not self._rl_engine:
            return {}

        try:
            with self._lock:
                current_genes = {name: gene.value for name, gene in self.genes.items()}
            return self._rl_engine.suggest_gene_adjustments(current_genes)
        except Exception as e:
            logger.warning(f"Failed to get RL suggestions: {e}")
            return {}

    def record_evolution_outcome(
        self,
        proposal_id: str,
        success: bool,
        performance_delta: float = 0.0,
        user_feedback: str | None = None,
    ) -> None:
        """Record outcome of evolution for RL learning."""
        if not self._rl_engine:
            return

        try:
            from core.reinforcement_learning import RewardType

            reward_type = RewardType.TASK_SUCCESS if success else RewardType.TASK_FAILURE
            magnitude = 1.0 + abs(performance_delta)

            if user_feedback:
                if "good" in user_feedback.lower() or "great" in user_feedback.lower():
                    reward_type = RewardType.USER_POSITIVE
                    magnitude = 1.5
                elif "bad" in user_feedback.lower() or "poor" in user_feedback.lower():
                    reward_type = RewardType.USER_NEGATIVE
                    magnitude = 1.0

            self._rl_engine.record_reward(reward_type, magnitude)

            logger.debug(f"Recorded evolution outcome: {reward_type.value}, magnitude={magnitude}")
        except Exception as e:
            logger.warning(f"Failed to record RL outcome: {e}")

    def _record_evolution_feedback(
        self,
        proposal: EvolutionProposal,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Automatically record RL feedback after evolution execution."""
        if not self._rl_engine:
            return

        try:
            from core.reinforcement_learning import RewardType

            if success:
                reward_type = RewardType.TASK_SUCCESS
                magnitude = 1.0

                changes_count = len(proposal.proposed_changes)
                if changes_count > 3:
                    magnitude = 1.2

                if proposal.user_feedback:
                    fb_lower = proposal.user_feedback.lower()
                    if any(word in fb_lower for word in ["good", "great", "excellent", "perfect"]):
                        reward_type = RewardType.USER_POSITIVE
                        magnitude = 1.5
                    elif any(word in fb_lower for word in ["bad", "poor", "wrong", "issue"]):
                        reward_type = RewardType.USER_NEGATIVE
                        magnitude = 0.5
            else:
                reward_type = RewardType.TASK_FAILURE
                magnitude = 1.0
                if error:
                    logger.debug(f"Evolution failed with error: {error}")

            self._rl_engine.record_reward(reward_type, magnitude)

            self._update_gene_bandits(proposal, success)

            # Report outcome to Insight Bridge for evolution feedback loop
            try:
                from core.insight_bridge import get_insight_bridge
                bridge = get_insight_bridge()
                bridge.report_evolution_outcome(
                    insight_id=proposal.id,
                    success=success,
                    details=f"Domain: {proposal.domain.value}, Changes: {len(proposal.proposed_changes)}",
                )
            except Exception as e2:
                logger.debug(f"Failed to report evolution outcome to insight bridge: {e2}")

            logger.debug(f"Auto-recorded evolution feedback: {reward_type.value}, magnitude={magnitude}")
        except Exception as e:
            logger.warning(f"Failed to auto-record evolution feedback: {e}")

    def _update_gene_bandits(self, proposal: EvolutionProposal, success: bool) -> None:
        """Update multi-armed bandit for gene mutation strategies."""
        if not self._rl_engine:
            return

        try:
            for gene_name in proposal.proposed_changes:
                strategy_id = f"gene_{gene_name}_mutation"
                reward = 1.0 if success else -0.5
                self._rl_engine.update_strategy_reward(strategy_id, reward)
        except Exception as e:
            logger.debug(f"Bandit update skipped: {e}")

    def apply_rl_guided_evolution(self, domain: CapabilityDomain | None = None) -> dict:
        """Apply evolution guided by RL suggestions."""
        rl_suggestions = self.get_rl_gene_suggestions()

        if not rl_suggestions:
            return {
                "status": "no_suggestions",
                "message": "No RL suggestions available",
            }

        with self._lock:
            self._proposal_counter += 1
            proposal_id = f"rl_{self._proposal_counter:04d}"

        proposed_changes = {}
        for gene_name, adjustment in rl_suggestions.items():
            if gene_name in self.genes:
                gene = self.genes[gene_name]
                new_value = gene.value + adjustment * self.RL_LEARNING_RATE
                new_value = max(gene.min_value, min(gene.max_value, new_value))

                proposed_changes[gene_name] = {
                    "current": gene.value,
                    "proposed": new_value,
                    "delta": new_value - gene.value,
                    "rl_adjustment": adjustment,
                }

        if not proposed_changes:
            return {"status": "no_valid_genes", "message": "No valid gene adjustments"}

        domain = domain or CapabilityDomain.ADAPTATION
        proposal = EvolutionProposal(
            id=proposal_id,
            timestamp=datetime.now(),
            domain=domain,
            title="RL-Guided Gene Adjustment",
            description="Evolution guided by reinforcement learning",
            rationale="RL engine suggests optimal gene adjustments based on historical performance",
            impact_estimate={"rl_guided": True, "adjustments": len(proposed_changes)},
            risk_level="low",
            affected_components=list(proposed_changes.keys()),
            proposed_changes=proposed_changes,
            rollback_plan="Revert gene values to previous state",
        )

        self._queue_permission_request(proposal)

        return {
            "proposal_id": proposal.id,
            "proposed_changes": proposed_changes,
            "status": "awaiting_permission",
            "rl_confidence": self._get_rl_confidence(),
        }

    def _get_rl_confidence(self) -> float:
        """Get confidence level from RL engine statistics."""
        if not self._rl_engine:
            return 0.0

        try:
            stats = self._rl_engine.get_statistics()
            q_stats = stats.get("q_learning", {})
            states_known = q_stats.get("states_known", 0)
            return min(1.0, states_known / 100.0)
        except Exception:
            return 0.0

    def force_evolution_cycle(self) -> dict:
        """Force an immediate evolution analysis cycle."""
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self._analyze_capabilities())
                analysis = future.result(timeout=30)
        except RuntimeError:
            analysis = asyncio.run(self._analyze_capabilities())

        with self._lock:
            current_genes = {n: g.value for n, g in self.genes.items()}
            cap_scores = dict(self.pathway.capability_scores)
        return {
            "analysis": analysis,
            "current_genes": current_genes,
            "capability_scores": cap_scores,
        }

    def get_status(self) -> dict:
        """Get comprehensive evolution status."""
        with self._lock:
            status = {
                "running": self._running,
                "pathway": self.get_evolutionary_pathway(),
                "pending_permissions": self.get_pending_permissions(),
                "recent_proposals": [
                    {
                        "id": p.id,
                        "title": p.title,
                        "status": p.permission_status.value,
                        "timestamp": p.timestamp.isoformat(),
                    }
                    for p in list(self.proposal_history)[-10:]
                ],
                "capability_proposals": self.get_capability_evolution_proposals()[:5],
            }

        if self._rl_engine:
            try:
                status["rl_stats"] = self._rl_engine.get_statistics()
            except Exception as e:
                logger.debug(f"Failed to get RL engine statistics: {e}")

        if self._bioevolve_engine:
            try:
                status["bioevolve"] = self.get_bioevolve_status()
            except Exception as e:
                logger.debug(f"Failed to get BioEvolve status: {e}")

        return status


_evolution_orchestrator: EvolutionOrchestrator | None = None
_evolution_lock = threading.Lock()


def get_evolution_orchestrator() -> EvolutionOrchestrator:
    """Get global evolution orchestrator instance (thread-safe)."""
    global _evolution_orchestrator
    if _evolution_orchestrator is None:
        with _evolution_lock:
            if _evolution_orchestrator is None:
                _evolution_orchestrator = EvolutionOrchestrator()
    return _evolution_orchestrator
