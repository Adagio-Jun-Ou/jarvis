"""Evolution tools for Jarvis self-improvement.

Provides tools for:
- Checking evolution status
- Responding to evolution permission requests
- Proposing capability evolution
- Viewing evolutionary pathway
"""
import logging
from typing import Any

from .base import Tool

logger = logging.getLogger(__name__)


class EvolutionStatusTool(Tool):
    """Get current evolution status and pending permissions."""

    name = "evolution_status"
    description = "Check evolution system status, pending permissions, and evolutionary pathway"
    parameters = {
        "type": "object",
        "properties": {
            "detailed": {
                "type": "boolean",
                "description": "Include detailed information about genes and capabilities",
            },
        },
    }
    requires_permission = False

    async def execute(self, detailed: bool = False, **params) -> str:
        from core.evolution_orchestrator import get_evolution_orchestrator

        orchestrator = get_evolution_orchestrator()
        status = orchestrator.get_status()

        output = "## Evolution System Status\n\n"
        output += f"**Stage:** {status['pathway']['current_stage']}\n"
        output += f"**Running:** {'Yes' in str(status['running'])}\n"
        output += f"**Total Evolutions:** {status['pathway']['total_evolutions']}\n"
        output += f"**Successful:** {status['pathway']['successful_evolutions']}\n"
        output += f"**Rolled Back:** {status['pathway']['rolled_back_evolutions']}\n"
        output += f"**Success Rate:** {status['pathway']['success_rate']:.1%}\n"

        if status['pathway']['last_evolution']:
            output += f"**Last Evolution:** {status['pathway']['last_evolution']}\n"

        if status['pending_permissions']:
            output += f"\n### Pending Permissions ({len(status['pending_permissions'])})\n"
            for perm in status['pending_permissions']:
                output += f"\n**{perm['title']}** (ID: {perm['id']})\n"
                output += f"- Risk: {perm['risk_level']}\n"
                output += f"- Time remaining: {int(perm['time_remaining']//60)} minutes\n"
                output += f"- {perm['description']}\n"

        if detailed:
            output += "\n### Gene Pool\n"
            for name, gene in status['pathway']['gene_pool'].items():
                bar_len = int(gene['value'] * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                output += f"- **{name}**: [{bar}] {gene['value']:.2f}\n"

            output += "\n### Capability Scores\n"
            for domain, score in status['pathway']['capability_scores'].items():
                bar_len = int(score * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                output += f"- **{domain}**: [{bar}] {score:.2f}\n"

        return output


class EvolutionPermissionTool(Tool):
    """Respond to evolution permission requests."""

    name = "evolution_permission"
    description = "Approve or deny pending evolution proposals with optional intent guidance"
    parameters = {
        "type": "object",
        "properties": {
            "proposal_id": {
                "type": "string",
                "description": "ID of the proposal to respond to",
            },
            "approved": {
                "type": "boolean",
                "description": "Whether to approve the evolution",
            },
            "intent": {
                "type": "string",
                "description": "Optional intent or direction for evolution (e.g., 'be more cautious', 'focus on security')",
            },
            "feedback": {
                "type": "string",
                "description": "Optional feedback about the proposal",
            },
        },
        "required": ["proposal_id", "approved"],
    }
    requires_permission = True

    async def execute(self, proposal_id: str, approved: bool, intent: str | None = None, feedback: str | None = None, **params) -> str:
        from core.evolution_orchestrator import get_evolution_orchestrator

        orchestrator = get_evolution_orchestrator()

        result = orchestrator.respond_to_permission(
            proposal_id=proposal_id,
            approved=approved,
            user_intent=intent,
            feedback=feedback,
        )

        if result.get("status") == "success":
            output = "## Evolution Approved ✅\n\n"
            output += f"**Proposal ID:** {proposal_id}\n"
            if result.get("changes_applied"):
                output += "\n### Changes Applied\n"
                for change in result["changes_applied"]:
                    output += f"- **{change['gene']}**: {change['old_value']:.3f} → {change['new_value']:.3f}\n"
            if intent:
                output += f"\n**User Intent:** {intent}\n"
                output += "_Evolution parameters have been adjusted based on your guidance._\n"
        elif result.get("status") == "denied":
            output = "## Evolution Denied ❌\n\n"
            output += f"**Proposal ID:** {proposal_id}\n"
            output += "The evolution proposal has been rejected.\n"
        elif result.get("status") == "rolled_back":
            output = "## Evolution Rolled Back ⚠️\n\n"
            output += f"**Proposal ID:** {proposal_id}\n"
            output += f"**Reason:** {result.get('error', 'Unknown error')}\n"
            output += "All changes have been reverted.\n"
        else:
            output = f"## Error\n\n{result.get('message', 'Unknown error')}"

        return output


class ProposeEvolutionTool(Tool):
    """Propose a capability evolution for Jarvis.
    
    This creates a proposal that must be approved via evolution_permission.
    The proposal itself doesn't require permission - only the approval does.
    """

    name = "propose_evolution"
    description = "Propose a capability evolution to improve Jarvis. Creates a proposal that can be reviewed and approved."
    parameters = {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "enum": ["reasoning", "learning", "security", "voice", "memory", "creativity", "efficiency", "knowledge", "adaptation", "communication"],
                "description": "Capability domain for evolution",
            },
            "intent": {
                "type": "string",
                "description": "The intent or goal of this evolution (what capability to improve)",
            },
            "title": {
                "type": "string",
                "description": "Short title for the evolution (optional)",
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the proposed evolution (optional)",
            },
            "rationale": {
                "type": "string",
                "description": "Why this evolution would help Jarvis improve (optional)",
            },
        },
        "required": ["domain", "intent"],
    }
    requires_permission = False  # Just creates a proposal; evolution_permission handles approval

    async def execute(self, domain: str, intent: str, title: str | None = None, description: str | None = None, rationale: str | None = None, **params) -> str:
        from core.evolution_orchestrator import get_evolution_orchestrator, CapabilityDomain

        orchestrator = get_evolution_orchestrator()

        try:
            domain_enum = CapabilityDomain(domain)
        except ValueError:
            return f"Invalid domain: {domain}"

        result = orchestrator.propose_capability_evolution(domain_enum, intent)

        output = "## Evolution Proposal Created\n\n"
        output += f"**Proposal ID:** {result['proposal_id']}\n"
        output += f"**Domain:** {result['domain']}\n"
        output += f"**Intent:** {intent}\n\n"

        output += "### Proposed Changes\n"
        for gene, change in result['proposed_changes'].items():
            delta = change['delta']
            direction = "↑" if delta > 0 else "↓"
            output += f"- **{gene}**: {change['current']:.3f} {direction} {change['proposed']:.3f}\n"

        output += "\n---\n"
        output += "**Status:** Awaiting permission\n"
        output += "Use `evolution_permission` to approve or deny this proposal.\n"

        return output


class EvolutionPathwayTool(Tool):
    """View evolutionary pathway and history."""

    name = "evolution_pathway"
    description = "View the evolutionary pathway showing how Jarvis has evolved"
    parameters = {
        "type": "object",
        "properties": {
            "history_limit": {
                "type": "integer",
                "description": "Maximum number of history items to show",
            },
        },
    }
    requires_permission = False

    async def execute(self, history_limit: int = 10, **params) -> str:
        from core.evolution_orchestrator import get_evolution_orchestrator

        orchestrator = get_evolution_orchestrator()
        pathway = orchestrator.get_evolutionary_pathway()

        output = "## Evolutionary Pathway\n\n"

        output += "### Current State\n"
        output += f"- **Stage:** {pathway['current_stage']}\n"
        output += f"- **Total Evolutions:** {pathway['total_evolutions']}\n"
        output += f"- **Success Rate:** {pathway['success_rate']:.1%}\n"

        output += "\n### Gene Values\n"
        output += "```\n"
        for name, gene in pathway['gene_pool'].items():
            value = gene['value']
            min_v = gene['min']
            max_v = gene['max']
            pct = (value - min_v) / (max_v - min_v)
            bar_len = int(pct * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            output += f"{name:25} [{bar}] {value:.2f}\n"
        output += "```\n"

        output += "\n### Capability Scores\n"
        output += "```\n"
        for domain, score in pathway['capability_scores'].items():
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            output += f"{domain:20} [{bar}] {score:.2f}\n"
        output += "```\n"

        proposals = orchestrator.get_capability_evolution_proposals()
        output += "\n### Suggested Improvements\n"
        for prop in proposals[:5]:
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(prop['priority'], "⚪")
            output += f"- {emoji} **{prop['domain']}** ({prop['type']}): score {prop['current_score']:.2f}\n"

        return output


class CapabilityProposalsTool(Tool):
    """Get capability evolution proposals."""

    name = "capability_proposals"
    description = "Get suggestions for capability evolution or refinement based on performance analysis"
    parameters = {
        "type": "object",
        "properties": {},
    }
    requires_permission = False

    async def execute(self, **params) -> str:
        from core.evolution_orchestrator import get_evolution_orchestrator

        orchestrator = get_evolution_orchestrator()
        proposals = orchestrator.get_capability_evolution_proposals()

        output = "## Capability Evolution Proposals\n\n"
        output += "Based on performance analysis, here are suggested areas for improvement:\n\n"

        for prop in proposals:
            priority = prop['priority']
            domain = prop['domain']
            ptype = prop['type']
            score = prop['current_score']

            if priority == "high":
                output += f"### 🔴 HIGH PRIORITY: {domain.title()}\n"
            elif priority == "medium":
                output += f"### 🟡 MEDIUM PRIORITY: {domain.title()}\n"
            else:
                output += f"### 🟢 {domain.title()}\n"

            output += f"- **Action:** {ptype.replace('_', ' ').title()}\n"
            output += f"- **Current Score:** {score:.2f}\n"

            if prop.get('suggested_genes'):
                output += f"- **Affected Parameters:** {', '.join(prop['suggested_genes'])}\n"

            output += "\n"

        output += "---\n"
        output += "Use `propose_evolution` to create an evolution proposal for a specific domain.\n"

        return output


class KnowledgeFeedTool(Tool):
    """Access mathematical insights and knowledge discovered by Jarvis."""

    name = "knowledge_feed"
    description = "View mathematical insights and knowledge discoveries that Jarvis can share with humanity"
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of insights to show",
            },
        },
    }
    requires_permission = False

    async def execute(self, limit: int = 10, **params) -> str:
        from core.insight_bridge import get_insight_bridge

        bridge = get_insight_bridge()
        knowledge = bridge.get_knowledge_feed(limit)
        stats = bridge.get_insight_statistics()

        output = "## Knowledge Feed for Humanity\n\n"
        output += f"**Total Insights:** {stats['total_insights']}\n"
        output += f"**Knowledge Contributions:** {stats['knowledge_contributions']}\n"
        output += f"**Pending Evolution Updates:** {stats['pending_evolution_updates']}\n\n"

        if not knowledge:
            output += "No knowledge contributions yet. Insights are generated as Jarvis solves mathematical problems.\n"
            return output

        output += "### Recent Discoveries\n\n"
        for item in knowledge:
            output += f"#### {item['title']}\n"
            output += f"{item['summary']}\n\n"
            output += f"*Discovered by: {item['discovered_by']}*\n\n"
            output += "---\n\n"

        output += "These insights drive Jarvis's evolution and are shared for human knowledge.\n"

        return output


class DiscoveriesTool(Tool):
    """See what interesting things Jarvis has discovered recently."""

    name = "discoveries"
    description = "View recent discoveries Jarvis has made during daydreaming and exploration"
    parameters = {
        "type": "object",
        "properties": {},
    }
    requires_permission = False

    async def execute(self, **params) -> str:
        from core.discovery_notifier import get_discovery_notifier

        notifier = get_discovery_notifier()
        pending = notifier.get_pending_discoveries(10)
        summary = notifier.get_discovery_summary()

        output = "## Jarvis's Recent Discoveries\n\n"
        output += f"**Pending Discoveries:** {summary['pending_count']}\n"
        output += f"**Shared with you:** {summary['shared_count']}\n"
        output += f"**Interesting finds:** {summary['interesting_count']}\n\n"

        if not pending:
            output += "No new discoveries yet. I'll share interesting findings as I explore!\n"
            return output

        output += "### Recent Finds\n\n"
        for d in pending:
            emoji = "⭐" if d.priority.value >= 3 else "💡"
            output += f"{emoji} **{d.title}**\n"
            output += f"   {d.natural_message}\n\n"

        return output


class SelfEditTool(Tool):
    """View and manage pending code edits from Jarvis."""

    name = "self_edit"
    description = "View, approve, or reject code modifications Jarvis has proposed"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "approve", "reject", "apply", "rollback"],
                "description": "Action to perform: list (show pending edits), approve (approve an edit), reject (reject an edit), apply (execute approved edit), rollback (undo applied edit)",
            },
            "edit_id": {
                "type": "string",
                "description": "Edit ID for approve/reject/apply/rollback actions",
            },
        },
        "required": ["action"],
    }
    requires_permission = True

    async def execute(self, action: str = "list", edit_id: str = "", **params) -> str:
        from core.self_edit import get_self_edit_engine

        engine = get_self_edit_engine()

        if action == "list":
            pending = engine.get_pending_edits()
            stats = engine.get_edit_stats()

            output = "## Pending Code Edits\n\n"
            output += f"**Pending:** {stats['pending']}\n"
            output += f"**Applied:** {stats['applied']}\n\n"

            if not pending:
                output += "No pending edits.\n"
                return output

            for e in pending:
                output += f"### {e['id']} ({e['type']})\n"
                output += f"**File:** `{e['file']}`\n\n"
                output += f"**Description:** {e['description']}\n\n"
                output += f"**Impact:** {e['impact']}\n\n"
                output += f"```diff\n{e['diff']}\n```\n\n"
                output += "---\n\n"

            return output

        elif action == "approve":
            if not edit_id:
                return "Error: edit_id required for approve action"
            if engine.approve_edit(edit_id):
                return f"Edit {edit_id} approved. Use `apply` to execute it."
            return f"Could not approve {edit_id}"

        elif action == "reject":
            if not edit_id:
                return "Error: edit_id required for reject action"
            if engine.reject_edit(edit_id):
                return f"Edit {edit_id} rejected."
            return f"Could not reject {edit_id}"

        elif action == "apply":
            if not edit_id:
                return "Error: edit_id required for apply action"
            success, msg = await engine.apply_edit(edit_id)
            return f"{'✓' if success else '✗'} {msg}"

        elif action == "rollback":
            if not edit_id:
                return "Error: edit_id required for rollback action"
            success, msg = await engine.rollback_edit(edit_id)
            return f"{'✓' if success else '✗'} {msg}"

        return f"Unknown action: {action}"


class DaydreamVerifyTool(Tool):
    """View verified daydream solutions."""

    name = "verified_solutions"
    description = "View math solutions from daydreaming that have been verified"
    parameters = {
        "type": "object",
        "properties": {
            "min_confidence": {
                "type": "number",
                "description": "Minimum confidence threshold (0-1)",
            },
        },
    }
    requires_permission = False

    async def execute(self, min_confidence: float = 0.7, **params) -> str:
        from core.daydream_verification import get_verifier

        verifier = get_verifier()
        verified = verifier.get_verified_insights(min_confidence)

        output = "## Verified Daydream Solutions\n\n"
        output += f"**Min Confidence:** {min_confidence}\n"
        output += f"**Verified Solutions:** {len(verified)}\n\n"

        if not verified:
            output += "No verified solutions yet. I verify my daydream discoveries before sharing.\n"
            return output

        for s in verified:
            conf_bar = "█" * int(s["confidence"] * 10) + "░" * (10 - int(s["confidence"] * 10))
            output += f"### {s['id']} [{s['status']}]\n"
            output += f"**Confidence:** `{conf_bar}` {s['confidence']:.0%}\n\n"
            output += f"**Problem:** {s['problem']}\n\n"
            output += f"**Insight:** {s['insight']}\n\n"
            output += "---\n\n"

        return output


class ProposeGeneTool(Tool):
    """Propose a new gene for the evolution system.
    
    This creates a proposal that must be approved via gene_approval.
    The proposal itself doesn't require permission - only the approval does.
    """

    name = "propose_gene"
    description = "Propose adding a new gene/parameter to Jarvis's evolution system. Creates a proposal for review."
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Gene name in snake_case (e.g., 'patience_level', 'creative_risk')",
            },
            "description": {
                "type": "string",
                "description": "What this gene controls and why it would be beneficial",
            },
            "initial_value": {
                "type": "number",
                "description": "Starting value between 0 and 1 (default 0.5)",
            },
            "min_value": {
                "type": "number",
                "description": "Minimum allowed value (default 0.0)",
            },
            "max_value": {
                "type": "number",
                "description": "Maximum allowed value (default 1.0)",
            },
            "rationale": {
                "type": "string",
                "description": "Why this gene would help Jarvis improve",
            },
        },
        "required": ["name", "description"],
    }
    requires_permission = False  # Just creates a proposal; gene_approval handles the actual addition

    async def execute(
        self,
        name: str,
        description: str,
        initial_value: float = 0.5,
        min_value: float = 0.0,
        max_value: float = 1.0,
        rationale: str = "",
        **params
    ) -> str:
        from core.evolution_orchestrator import get_evolution_orchestrator

        orchestrator = get_evolution_orchestrator()

        result = orchestrator.propose_new_gene(
            name=name,
            description=description,
            initial_value=initial_value,
            min_value=min_value,
            max_value=max_value,
            rationale=rationale,
            suggested_by="jarvis",
        )

        if result["status"] == "error":
            return f"❌ {result['message']}"

        output = "## 🧬 New Gene Proposed\n\n"
        output += f"**Name:** `{result['gene_name']}`\n"
        output += f"**Description:** {description}\n"
        output += f"**Initial Value:** {result['initial_value']:.2f}\n"
        output += f"**Range:** [{min_value:.2f}, {max_value:.2f}]\n\n"

        if rationale:
            output += f"**Rationale:** {rationale}\n\n"

        output += "### ⏳ Awaiting User Approval\n"
        output += f"**Proposal ID:** `{result['proposal_id']}`\n\n"
        output += "This gene will be added to my evolution system once approved. "
        output += "Use `gene_approval` to respond.\n"

        return output


class GeneApprovalTool(Tool):
    """Approve or reject pending gene proposals.
    
    This executes the actual gene addition to Jarvis's evolution system.
    REQUIRES USER PERMISSION because it modifies behavior parameters.
    """

    name = "gene_approval"
    description = "Approve or reject a proposed new gene. This actually adds the gene to Jarvis's evolution system. REQUIRES USER APPROVAL."
    parameters = {
        "type": "object",
        "properties": {
            "proposal_id": {
                "type": "string",
                "description": "ID of the gene proposal to respond to",
            },
            "approved": {
                "type": "boolean",
                "description": "Whether to approve adding this gene",
            },
            "modified_value": {
                "type": "number",
                "description": "Optional: different initial value if approving (must be within range)",
            },
            "reason": {
                "type": "string",
                "description": "Optional reason for approval or rejection",
            },
        },
        "required": ["proposal_id", "approved"],
    }
    requires_permission = True  # Actually adds the gene - needs user approval

    async def execute(
        self,
        proposal_id: str,
        approved: bool,
        modified_value: float | None = None,
        reason: str = "",
        **params
    ) -> str:
        from core.evolution_orchestrator import get_evolution_orchestrator

        orchestrator = get_evolution_orchestrator()

        if approved:
            result = orchestrator.approve_new_gene(proposal_id, modified_value)

            if result["status"] == "success":
                output = "## ✅ Gene Approved\n\n"
                output += f"**Gene:** `{result['gene_name']}`\n"
                output += f"**Value:** {result['value']:.2f}\n"
                output += f"**Description:** {result['description']}\n\n"
                output += f"{result['message']}\n"
            else:
                output = f"## ❌ Error\n\n{result['message']}"
        else:
            result = orchestrator.reject_new_gene(proposal_id, reason)

            if result["status"] == "rejected":
                output = "## 🚫 Gene Rejected\n\n"
                output += f"**Gene:** `{result['gene_name']}`\n"
                output += f"{result['message']}\n"
            else:
                output = f"## ❌ Error\n\n{result['message']}"

        return output


class GeneListTool(Tool):
    """List all current genes in the evolution system."""

    name = "gene_list"
    description = "View all genes in Jarvis's evolution system with their current values and descriptions"
    parameters = {
        "type": "object",
        "properties": {
            "include_proposals": {
                "type": "boolean",
                "description": "Also show pending gene proposals",
            },
        },
    }
    requires_permission = False

    async def execute(self, include_proposals: bool = False, **params) -> str:
        from core.evolution_orchestrator import get_evolution_orchestrator

        orchestrator = get_evolution_orchestrator()
        pathway = orchestrator.get_evolutionary_pathway()

        output = "## 🧬 Evolution Gene Pool\n\n"
        output += f"**Total Genes:** {len(pathway['gene_pool'])}\n\n"

        output += "| Gene | Value | Range | Description |\n"
        output += "|------|-------|-------|-------------|\n"

        for name, gene in sorted(pathway['gene_pool'].items()):
            value = gene['value']
            min_v = gene['min']
            max_v = gene['max']
            desc = gene['description'][:40] + "..." if len(gene['description']) > 40 else gene['description']
            output += f"| `{name}` | {value:.2f} | [{min_v:.1f}, {max_v:.1f}] | {desc} |\n"

        if include_proposals:
            pending = orchestrator.get_pending_gene_proposals()
            if pending:
                output += "\n### 📋 Pending Gene Proposals\n\n"
                for prop in pending:
                    output += f"- **`{prop['name']}`** (ID: `{prop['id']}`)\n"
                    output += f"  - {prop['description']}\n"
                    output += f"  - Initial: {prop['initial_value']:.2f}, Range: [{prop['min_value']:.1f}, {prop['max_value']:.1f}]\n"
                    if prop.get('rationale'):
                        output += f"  - Rationale: {prop['rationale']}\n"
                    output += "\n"
            else:
                output += "\n*No pending gene proposals.*\n"

        output += "\nUse `propose_gene` to suggest new genes (requires approval).\n"

        return output


class ToolCompetencyTool(Tool):
    """View tool learning progress and competencies."""

    name = "tool_competency"
    description = "View learning progress and competency levels for all tools"
    parameters = {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Specific tool to view (optional, shows all if not provided)",
            },
            "include_suggestions": {
                "type": "boolean",
                "description": "Include improvement suggestions",
            },
        },
    }
    requires_permission = False

    async def execute(self, tool_name: str = "", include_suggestions: bool = False, **params) -> str:
        from core.tool_learning import get_tool_learning_engine

        engine = get_tool_learning_engine()

        if tool_name:
            comp = engine.get_tool_competency(tool_name)
            if not comp:
                return f"No competency data for tool: {tool_name}"

            rate = comp.successful_uses / max(1, comp.total_uses)
            bar_len = int(rate * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            output = f"## Tool: {tool_name}\n\n"
            output += f"**Level:** {comp.level.value.title()}\n"
            output += f"**Uses:** {comp.total_uses} ({comp.successful_uses} successful)\n"
            output += f"**Success Rate:** [{bar}] {rate:.1%}\n"
            output += f"**Avg Quality:** {comp.avg_quality:.2f}\n"
            output += f"**Avg Time:** {comp.avg_execution_time:.2f}s\n"
            output += f"**Practice Sessions:** {comp.practice_sessions}\n\n"

            if comp.optimal_parameters:
                output += "### Optimal Parameters\n"
                for key, val in list(comp.optimal_parameters.items())[:5]:
                    output += f"- `{key}`: {val.get('value', 'N/A')} (used {val.get('count', 0)} times)\n"

            if comp.common_errors:
                output += "\n### Common Errors\n"
                for err, count in sorted(comp.common_errors.items(), key=lambda x: -x[1])[:5]:
                    output += f"- {err}: {count} times\n"

            if include_suggestions:
                suggestions = engine.suggest_tool_improvement(tool_name)
                if suggestions and suggestions.get("recommendations"):
                    output += "\n### Recommendations\n"
                    for rec in suggestions["recommendations"]:
                        output += f"- {rec}\n"

            return output

        # Show all tools
        competencies = engine.get_all_competencies()

        if not competencies:
            return "No tool competency data yet. Tools will be tracked as they are used."

        output = "## Tool Competencies\n\n"
        output += "| Tool | Level | Uses | Success | Quality |\n"
        output += "|------|-------|------|---------|----------|\n"

        for name, comp in sorted(competencies.items(), key=lambda x: -x[1].total_uses):
            rate = comp.successful_uses / max(1, comp.total_uses)
            output += f"| {name} | {comp.level.value} | {comp.total_uses} | {rate:.0%} | {comp.avg_quality:.2f} |\n"

        output += f"\n**Total Tools Tracked:** {len(competencies)}\n"

        priorities = engine.get_practice_priorities()
        if priorities:
            output += f"\n**Practice Priorities:** {', '.join(priorities[:3])}\n"

        output += "\nUse `tool_competency tool_name='<name>'` for details on a specific tool.\n"

        return output


class ToolPracticeTool(Tool):
    """Practice a tool to improve competency (used during dreaming)."""

    name = "tool_practice"
    description = "Practice a tool during idle time to improve competency. Usually called by daydream engine."
    parameters = {
        "type": "object",
        "properties": {
            "tool_name": {
                "type": "string",
                "description": "Tool to practice",
            },
        },
        "required": ["tool_name"],
    }
    requires_permission = False

    async def execute(self, tool_name: str, **params) -> str:
        from core.tool_learning import get_tool_learning_engine

        engine = get_tool_learning_engine()
        result = await engine.practice_tool(tool_name)

        output = f"## Tool Practice: {tool_name}\n\n"
        output += f"**Sessions:** {result.get('session', 0)}\n"
        output += f"**Scenarios:** {result.get('scenarios_practiced', 0)}\n"
        output += f"**New Level:** {result.get('new_level', 'unknown')}\n"

        return output


class SketchPadTool(Tool):
    """Daydream sketch pad for quick insight capture and testing."""

    name = "sketch"
    description = "Capture quick insights, test concepts, and manage daydream sketches"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["sketch", "test", "promote", "list", "stats"],
                "description": "Action to perform: sketch (create new), test (validate a sketch), promote (to knowledge graph), list (show all), stats (statistics)",
            },
            "sketch_type": {
                "type": "string",
                "enum": ["insight", "conjecture", "method", "connection", "pattern"],
                "description": "Type of sketch content",
            },
            "content": {
                "type": "string",
                "description": "Content for sketch action",
            },
            "sketch_id": {
                "type": "string",
                "description": "Sketch ID for test/promote actions",
            },
        },
        "required": ["action"],
    }
    requires_permission = False

    async def execute(
        self,
        action: str = "list",
        sketch_type: str = "insight",
        content: str = "",
        sketch_id: str = "",
        **params
    ) -> str:
        from core.daydream_sketch import get_sketch_pad, SketchType

        pad = get_sketch_pad()

        if action == "list":
            stats = pad.get_statistics()
            pad.get_drafts()
            validated = pad.get_validated()

            output = "## Daydream Sketch Pad\n\n"
            output += f"**Total:** {stats['total_sketches']}\n"
            output += f"**Drafts:** {stats['drafts']}\n"
            output += f"**Validated:** {stats['validated']}\n\n"

            if stats['by_type']:
                output += "### By Type\n"
                for t, c in stats['by_type'].items():
                    output += f"- {t}: {c}\n"
                output += "\n"

            if validated:
                output += "### Ready to Promote\n"
                for s in validated[:5]:
                    output += f"- `{s.id}` {s.title}\n"

            return output

        elif action == "sketch":
            if not content:
                return "Error: content required for sketch"

            type_map = {
                "insight": SketchType.INSIGHT,
                "conjecture": SketchType.CONJECTURE,
                "method": SketchType.METHOD_IDEA,
                "connection": SketchType.CONNECTION,
                "pattern": SketchType.PATTERN,
            }

            st = type_map.get(sketch_type, SketchType.INSIGHT)
            sketch = pad.sketch(sketch_type=st, title=content[:50], content=content)

            return f"✓ Sketched: `{sketch.id}`\nType: {sketch_type}\nStatus: draft\n\n{content}"

        elif action == "test":
            if not sketch_id:
                return "Error: sketch_id required for test"

            sketch = pad._get_sketch(sketch_id)
            if not sketch:
                return f"Sketch {sketch_id} not found"

            test = pad.test_prime_check(sketch_id, 17, True)

            return f"Test result: {'PASSED' if test.passed else 'FAILED'}\n{test.notes}"

        elif action == "promote":
            if not sketch_id:
                return "Error: sketch_id required for promote"

            success = pad.promote(sketch_id)
            if success:
                return f"✓ Promoted {sketch_id} to insight bridge"
            else:
                return f"✗ Could not promote {sketch_id} (must be validated first)"

        elif action == "stats":
            stats = pad.get_statistics()
            return f"Sketch Pad Stats:\n{stats}"

        return f"Unknown action: {action}"


EVOLUTION_TOOLS = [
    EvolutionStatusTool(),
    EvolutionPermissionTool(),
    ProposeEvolutionTool(),
    EvolutionPathwayTool(),
    CapabilityProposalsTool(),
    KnowledgeFeedTool(),
    DiscoveriesTool(),
    SelfEditTool(),
    DaydreamVerifyTool(),
    SketchPadTool(),
    ProposeGeneTool(),
    GeneApprovalTool(),
    GeneListTool(),
    ToolCompetencyTool(),
    ToolPracticeTool(),
]


def register_evolution_tools(agent_or_tools: Any) -> None:
    """Register evolution tools with the agent or tool registry."""
    if hasattr(agent_or_tools, 'register_tools'):
        agent_or_tools.register_tools(EVOLUTION_TOOLS)
    elif hasattr(agent_or_tools, 'tools'):
        agent_or_tools.tools.extend(EVOLUTION_TOOLS)
    else:
        logger.warning("Cannot register evolution tools: unknown interface")
