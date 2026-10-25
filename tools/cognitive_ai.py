"""Cognitive-AI Tools for Jarvis.

Exposes cognitive psychology, machine alignment, and neuroscience
capabilities as Jarvis tools for use by the LLM agent.
"""
import asyncio
import logging
from typing import Any

from .base import Tool
from core.helpers import safe_error_message

logger = logging.getLogger(__name__)


class CognitivePsychologyTool(Tool):
    """Cognitive bias detection, dual-process analysis, and debiasing."""

    name = "cognitive_psychology"
    description = "Analyze cognitive biases, dual-process thinking (System 1/2), and get debiasing strategies. Supports: detect_biases, dual_process, debias, bias_awareness, profile"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["detect_biases", "dual_process", "debias", "bias_awareness", "profile"],
                "description": "Cognitive psychology action to perform",
            },
            "text": {
                "type": "string",
                "description": "Text to analyze for biases or process through dual-process model",
            },
            "context": {
                "type": "string",
                "description": "Additional context for analysis",
            },
            "bias": {
                "type": "string",
                "description": "Specific bias type for debiasing (e.g., 'confirmation_bias', 'overconfidence')",
            },
            "complexity": {
                "type": "number",
                "description": "Task complexity 0-1 for dual-process (default 0.5)",
            },
            "novelty": {
                "type": "number",
                "description": "Task novelty 0-1 for dual-process (default 0.5)",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs) -> dict[str, Any]:
        action = kwargs.get("action", "detect_biases")
        try:
            from core.cognitive_psychology import CognitivePsychologyEngine, CognitiveBias

            engine = CognitivePsychologyEngine.get_instance()

            if action == "detect_biases":
                text = kwargs.get("text", "")
                context = kwargs.get("context", "")
                if not text:
                    return {"error": "text parameter required for detect_biases"}
                detections = engine.detect_biases(text, context)
                return {
                    "biases_detected": [d.to_dict() for d in detections],
                    "count": len(detections),
                    "summary": f"Detected {len(detections)} cognitive bias(es)",
                }

            elif action == "dual_process":
                text = kwargs.get("text", "")
                if not text:
                    return {"error": "text parameter required for dual_process"}
                complexity = float(kwargs.get("complexity", 0.5))
                novelty = float(kwargs.get("novelty", 0.5))
                state = engine.apply_dual_process(text, complexity, novelty)
                return state.to_dict()

            elif action == "debias":
                bias_name = kwargs.get("bias", "confirmation_bias")
                context = kwargs.get("context", "")
                try:
                    bias = CognitiveBias(bias_name)
                except ValueError:
                    return {"error": f"Unknown bias: {bias_name}"}
                prompt = engine.get_debiasing_prompt(bias, context)
                return {"bias": bias_name, "debiasing_prompt": prompt}

            elif action == "bias_awareness":
                text = kwargs.get("text", "")
                context = kwargs.get("context", "")
                if not text:
                    return {"error": "text parameter required for bias_awareness"}
                prompt = engine.generate_bias_awareness_prompt(text, context)
                return {"bias_awareness_prompt": prompt, "biases_found": bool(prompt)}

            elif action == "profile":
                return engine.get_bias_profile()

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"CognitivePsychology tool error: {safe_error_message(e)}")
            return {"error": safe_error_message(e)}


class MachineAlignmentTool(Tool):
    """Machine alignment checks, corrigibility tracking, and preference inference."""

    name = "machine_alignment"
    description = "Check AI alignment against constitutional principles, detect deceptive alignment, track corrigibility, infer user preferences. Supports: constitutional_check, detect_deception, corrigibility, infer_preference, cooperative_inference, status"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["constitutional_check", "detect_deception", "corrigibility", "infer_preference", "cooperative_inference", "status"],
                "description": "Alignment action to perform",
            },
            "action_text": {
                "type": "string",
                "description": "Proposed action text for constitutional check or deception detection",
            },
            "context": {
                "type": "string",
                "description": "Additional context",
            },
            "override_type": {
                "type": "string",
                "description": "Override type for corrigibility tracking",
            },
            "accepted": {
                "type": "boolean",
                "description": "Whether the override was accepted (for corrigibility)",
            },
            "user_actions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "User actions for preference inference",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs) -> dict[str, Any]:
        action = kwargs.get("action", "status")
        try:
            from core.machine_alignment import MachineAlignmentEngine

            engine = MachineAlignmentEngine.get_instance()

            if action == "constitutional_check":
                action_text = kwargs.get("action_text", "")
                context = kwargs.get("context", "")
                if not action_text:
                    return {"error": "action_text parameter required"}
                return engine.constitutional_check(action_text, context)

            elif action == "detect_deception":
                text = kwargs.get("action_text", "")
                context = kwargs.get("context", "")
                if not text:
                    return {"error": "action_text parameter required"}
                return engine.detect_deceptive_alignment(text, context)

            elif action == "corrigibility":
                override_type = kwargs.get("override_type", "correction")
                accepted = kwargs.get("accepted", True)
                return engine.check_corrigibility(override_type, accepted)

            elif action == "infer_preference":
                action_text = kwargs.get("action_text", "")
                context = kwargs.get("context", "")
                if not action_text:
                    return {"error": "action_text parameter required"}
                obs = engine.infer_preference(action_text, context)
                return obs.to_dict()

            elif action == "cooperative_inference":
                actions = kwargs.get("user_actions", [])
                context = kwargs.get("context", "")
                if not actions:
                    return {"error": "user_actions parameter required"}
                return engine.cooperative_inference(actions, context)

            elif action == "status":
                return engine.get_alignment_status()

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"MachineAlignment tool error: {safe_error_message(e)}")
            return {"error": safe_error_message(e)}


class NeuroscienceTool(Tool):
    """Neuroscience cognitive architecture: neural processing, neurotransmitters, decisions."""

    name = "neuroscience"
    description = "Process input through neuroscience cognitive architecture, track neurotransmitters, make decisions via basal ganglia, regulate emotions. Supports: process, state, reward, decide, regulate_emotion"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["process", "state", "reward", "decide", "regulate_emotion"],
                "description": "Neuroscience action to perform",
            },
            "text": {
                "type": "string",
                "description": "Text input for neural processing",
            },
            "emotional_valence": {
                "type": "number",
                "description": "Emotional valence -1 to 1 (default 0)",
            },
            "importance": {
                "type": "number",
                "description": "Importance 0-1 (default 0.5)",
            },
            "novelty": {
                "type": "number",
                "description": "Novelty 0-1 (default 0.5)",
            },
            "predicted_reward": {
                "type": "number",
                "description": "Predicted reward for RPE calculation",
            },
            "actual_reward": {
                "type": "number",
                "description": "Actual reward for RPE calculation",
            },
            "evidence": {
                "type": "object",
                "description": "Evidence dict for decision circuit (e.g., {\"safety\": 0.8})",
            },
            "urgency": {
                "type": "number",
                "description": "Urgency 0-1 for decision (default 0.5)",
            },
            "strategy": {
                "type": "string",
                "enum": ["reappraisal", "suppression", "mindfulness"],
                "description": "Emotion regulation strategy",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs) -> dict[str, Any]:
        action = kwargs.get("action", "state")
        try:
            from core.neuroscience import NeuroscienceCognitiveArchitecture

            arch = NeuroscienceCognitiveArchitecture.get_instance()

            if action == "process":
                text = kwargs.get("text", "")
                if not text:
                    return {"error": "text parameter required for process"}
                return arch.process_input(
                    text,
                    emotional_valence=float(kwargs.get("emotional_valence", 0.0)),
                    importance=float(kwargs.get("importance", 0.5)),
                    novelty=float(kwargs.get("novelty", 0.5)),
                )

            elif action == "state":
                return arch.get_neural_state()

            elif action == "reward":
                predicted = float(kwargs.get("predicted_reward", 0.5))
                actual = float(kwargs.get("actual_reward", 0.5))
                return arch.predict_reward(predicted, actual)

            elif action == "decide":
                evidence = kwargs.get("evidence", {})
                if not evidence or not isinstance(evidence, dict):
                    return {"error": "evidence must be a dict of {factor: value}"}
                urgency = float(kwargs.get("urgency", 0.5))
                return arch.make_decision(evidence, urgency)

            elif action == "regulate_emotion":
                strategy = kwargs.get("strategy", "reappraisal")
                if strategy not in ("reappraisal", "suppression", "mindfulness"):
                    return {"error": "strategy must be reappraisal, suppression, or mindfulness"}
                return arch.regulate_emotion(strategy)

            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as e:
            logger.error(f"Neuroscience tool error: {safe_error_message(e)}")
            return {"error": safe_error_message(e)}


def register_cognitive_tools(registry) -> None:
    """Register cognitive-AI tools with the tool registry."""
    registry.register(CognitivePsychologyTool())
    registry.register(MachineAlignmentTool())
    registry.register(NeuroscienceTool())