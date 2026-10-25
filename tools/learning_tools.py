"""Learning and Defense tools for Jarvis.

Provides tools for:
- Erdős math problem solving
- Defense monitoring and management
- Learning pipeline control
- Insight retrieval
"""
import json

from .base import Tool


class ErdosMathTool(Tool):
    name = "erdos_math"
    description = "Generate and attempt Erdős-style mathematical problems for continuous learning"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["generate", "attempt", "status", "solved", "active", "insights"],
                "description": "Action to perform",
            },
            "category": {
                "type": "string",
                "description": "Problem category: graph_theory, number_theory, combinatorics, probability, ramsey_theory",
            },
        },
        "required": ["action"],
    }
    requires_permission = False

    async def execute(self, action: str, category: str | None = None, **params) -> str:
        from core.erdos_engine import get_erdos_engine, ProblemCategory

        engine = get_erdos_engine()

        if action == "generate":
            cat = None
            if category:
                try:
                    cat = ProblemCategory(category)
                except ValueError:
                    pass
            problem = engine.generator.generate_problem(category=cat)
            return f"Generated problem: {problem.id}\n{problem.title}\n{problem.statement}"

        elif action == "attempt":
            result = await engine.force_attempt()
            status = "✓ SOLVED" if result.get("success") else "○ In progress"
            return f"{status}\nProblem: {result.get('title', 'unknown')}\nStrategy: {result.get('strategy', 'none')}\nInsight: {result.get('insight', 'none')}"

        elif action == "status":
            status = engine.get_status()
            return f"Erdős Engine Status:\n" \
                   f"Total problems: {status['total_problems']}\n" \
                   f"Current problems: {status['current_problems']}\n" \
                   f"Solved: {status['solved_count']}\n" \
                   f"Running: {status['running']}\n" \
                   f"Best strategies: {status['best_strategies'][:3]}"

        elif action == "solved":
            solved = engine.get_solved_problems()
            if not solved:
                return "No problems solved yet."
            lines = ["Solved problems:"]
            for p in solved[:10]:
                lines.append(f"  {p['id']}: {p['title']} ({p['attempts']} attempts)")
            return "\n".join(lines)

        elif action == "active":
            active = engine.get_active_problems()
            if not active:
                return "No active problems."
            lines = ["Active problems:"]
            for p in active[:10]:
                lines.append(f"  {p['id']}: {p['title']} ({p['attempts']} attempts)")
            return "\n".join(lines)

        elif action == "insights":
            insights = engine.get_insights()
            if not insights:
                return "No insights accumulated yet."
            return "Recent mathematical insights:\n" + "\n".join(f"  • {i}" for i in insights[:10])

        return f"Unknown action: {action}"


class DefenseTool(Tool):
    name = "defense"
    description = "Monitor and manage data defense operations - threat detection, integrity, and security"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "scan", "protect", "quarantine", "events", "summary", "block", "unblock"],
                "description": "Defense action to perform",
            },
            "target": {
                "type": "string",
                "description": "File path or entity to target",
            },
        },
        "required": ["action"],
    }
    requires_permission = False  # Defense operations allowed for autonomous security

    async def execute(self, action: str, target: str | None = None, **params) -> str:
        from core.defense import get_defense_agent
        from pathlib import Path

        agent = get_defense_agent()

        if action == "status":
            status = agent.get_defense_status()
            stats = status["stats"]
            return f"Defense Agent Status:\n" \
                   f"Running: {status['running']}\n" \
                   f"Protected files: {status['protected_files']}\n" \
                   f"Blocked entities: {status['blocked_entities']}\n" \
                   f"Quarantined files: {status['quarantined_files']}\n" \
                   f"Threats detected: {stats['threats_detected']}\n" \
                   f"Threats blocked: {stats['threats_blocked']}\n" \
                   f"Auto resolved: {stats['auto_resolved']}"

        elif action == "scan":
            if not target:
                return "Error: target file path required for scan"
            result = agent.scan_file(Path(target))
            return f"Scan result for {result['path']}:\n" \
                   f"Status: {result['status']}\n" \
                   f"Risk score: {result['risk_score']}\n" \
                   f"Threats: {len(result.get('threats', []))}"

        elif action == "protect":
            if not target:
                result = agent.protect_jarvis_core()
                return f"Protected {len(result['protected'])} core files\n" \
                       f"Failed: {len(result['failed'])}\n" \
                       f"Skipped: {len(result['skipped'])}"
            success = agent.protect_file(Path(target))
            return f"File protection: {'success' if success else 'failed'}"

        elif action == "quarantine":
            if not target:
                return "Error: target file path required for quarantine"
            success = agent.quarantine_file(Path(target))
            return f"Quarantine: {'success' if success else 'failed'}"

        elif action == "events":
            events = agent.get_recent_events(limit=10)
            if not events:
                return "No recent threat events."
            lines = ["Recent threat events:"]
            for e in events:
                lines.append(f"  [{e['level']}] {e['threat_type']}: {e['description'][:60]}...")
            return "\n".join(lines)

        elif action == "summary":
            summary = agent.get_threat_summary()
            return f"Threat summary (24h):\n" \
                   f"Total: {summary['total_threats']}\n" \
                   f"Critical: {summary['critical_count']}\n" \
                   f"High: {summary['high_count']}\n" \
                   f"Resolved: {summary['resolved_count']}\n" \
                   f"By type: {summary['by_type']}"

        elif action == "block":
            if not target:
                return "Error: entity required for blocking"
            agent.block_entity(target)
            return f"Blocked entity: {target}"

        elif action == "unblock":
            if not target:
                return "Error: entity required for unblocking"
            success = agent.unblock_entity(target)
            return f"Unblock: {'success' if success else 'not found'}"

        return f"Unknown action: {action}"


class LearningTool(Tool):
    name = "learning"
    description = "Control and monitor continuous learning pipeline - math, defense, dreams, evolution"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "start", "stop", "insights", "sessions", "metrics"],
                "description": "Learning action to perform",
            },
        },
        "required": ["action"],
    }
    requires_permission = False

    async def execute(self, action: str, **params) -> str:
        from core.learning_pipeline import get_learning_pipeline

        pipeline = get_learning_pipeline()

        if action == "status":
            status = pipeline.get_status()
            metrics = status["metrics"]
            return f"Learning Pipeline Status:\n" \
                   f"Running: {status['running']}\n" \
                   f"Phase: {status['current_phase']}\n" \
                   f"Problems attempted/solved: {metrics['problems_attempted']}/{metrics['problems_solved']}\n" \
                   f"Solve rate: {metrics['solve_rate']:.1%}\n" \
                   f"Threats analyzed: {metrics['threats_analyzed']}\n" \
                   f"Defense accuracy: {metrics['defense_accuracy']:.1%}\n" \
                   f"Dream insights: {metrics['dream_insights']}\n" \
                   f"Evolution generation: {metrics['evolution_generation']}\n" \
                   f"Learning time: {metrics['total_learning_time_hours']:.1f}h"

        elif action == "start":
            from core.erdos_engine import get_erdos_engine
            from core.defense import get_defense_agent
            from core.memory_manager import get_memory_manager

            erdos = get_erdos_engine()
            defense = get_defense_agent()
            memory = get_memory_manager()

            pipeline.attach_erdos_engine(erdos)
            pipeline.attach_defense_agent(defense)
            pipeline.attach_memory_manager(memory)

            await defense.start_monitoring()
            await erdos.start_background_learning()
            await pipeline.start_pipeline()

            return "Learning pipeline started with all subsystems attached"

        elif action == "stop":
            await pipeline.stop_pipeline()
            return "Learning pipeline stopped"

        elif action == "insights":
            insights = pipeline.get_recent_insights()
            if not insights:
                return "No insights accumulated yet."
            lines = ["Recent learning insights:"]
            for i in insights:
                lines.append(f"  [{i['type']}] {i['content'][:80]}")
            return "\n".join(lines)

        elif action == "sessions":
            sessions = pipeline.get_session_history()
            if not sessions:
                return "No learning sessions recorded."
            lines = ["Recent learning sessions:"]
            for s in sessions:
                lines.append(f"  {s['id']}: {s['phase']} ({s['duration']:.1f}s, {s['insights_count']} insights)")
            return "\n".join(lines)

        elif action == "metrics":
            status = pipeline.get_status()
            metrics = status["metrics"]
            return json.dumps(metrics, indent=2)

        return f"Unknown action: {action}"
