"""Insight Search Tool - Search and retrieve insights from Jarvis's knowledge base."""

import logging
from typing import Optional
from .base import Tool

logger = logging.getLogger(__name__)


class InsightSearchTool(Tool):
    """Search through Jarvis's accumulated insights.
    
    This tool searches insights from:
    - Erdős Engine (mathematical discoveries)
    - Daydream Engine (exploration insights)
    - Insight Bridge (cross-domain insights)
    - Evolution Orchestrator (learned patterns)
    
    Use this to find relevant insights, patterns, or knowledge.
    """
    
    name = "insight_search"
    description = """Search through Jarvis's accumulated insights and knowledge.

Searches insights from mathematical discoveries, exploration, cross-domain patterns,
and learned behaviors. Returns relevant insights matching your query.

Use this to:
- Find insights about a topic or concept
- Retrieve learned patterns or knowledge
- Search for mathematical discoveries
- Look up cross-domain insights
- Find relevant past experiences

Examples:
- "insight_search: pattern discovery" - Find insights about patterns
- "insight_search: math graph theory" - Find math insights about graphs
- "insight_search: learning optimization" - Find insights about learning
"""
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant insights",
            },
            "source": {
                "type": "string",
                "description": "Filter by source (insight_bridge, erdos, daydream, all)",
                "default": "all",
            },
            "type": {
                "type": "string", 
                "description": "Filter by type (pattern_discovery, strategy_success, cross_domain, etc.)",
                "default": None,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (default: 10)",
                "default": 10,
            },
        },
        "required": ["query"],
    }
    requires_permission = False
    
    async def execute(
        self,
        query: str,
        source: str = "all",
        type: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Execute insight search."""
        results = []
        
        try:
            # Try insight_review first (indexed search)
            from core.insight_review import get_insight_review
            review = get_insight_review()
            
            filters = {}
            if source and source != "all":
                filters["source"] = source
            if type:
                filters["type"] = type
            
            search_results = review.search(query, filters if filters else None)
            
            for insight in search_results[:limit]:
                results.append({
                    "id": insight.get("id", "unknown"),
                    "source": insight.get("source", "unknown"),
                    "type": insight.get("type", "unknown"),
                    "content": insight.get("content", "")[:500],
                    "timestamp": insight.get("timestamp", ""),
                })
            
            if results:
                return self._format_results(results, query)
            
        except Exception as e:
            logger.warning(f"Failed to search insights via review: {e}")
        
        # Fallback: Try unified_memory search
        try:
            from core.unified_memory import get_unified_memory
            memory = get_unified_memory()
            
            search_results = memory.search_insights(query, limit=limit)
            
            for insight in search_results:
                if isinstance(insight, dict):
                    results.append({
                        "id": insight.get("id", "unknown"),
                        "source": insight.get("source", "unknown"),
                        "content": insight.get("content", "")[:500],
                        "relevance": insight.get("relevance", 0),
                    })
            
            if results:
                return self._format_results(results, query)
            
        except Exception as e:
            logger.warning(f"Failed to search insights via unified_memory: {e}")
        
        # Fallback: Try insight_bridge directly
        try:
            from core.insight_bridge import get_insight_bridge
            bridge = get_insight_bridge()
            
            # Get recent insights if search fails
            recent = list(getattr(bridge, 'processed_insights', []))[-limit:]
            
            for insight in recent:
                raw = getattr(insight, 'raw_insight', '') or ''
                if query.lower() in raw.lower():
                    results.append({
                        "id": getattr(insight, 'id', 'unknown'),
                        "source": "insight_bridge",
                        "type": str(getattr(insight, 'insight_type', 'unknown')),
                        "content": raw[:500],
                    })
            
            if results:
                return self._format_results(results, query)
            
        except Exception as e:
            logger.warning(f"Failed to search insights via insight_bridge: {e}")
        
        if not results:
            return f"No insights found for query: '{query}'\n\nTry a different search term or check if the insight database has been populated."
        
        return self._format_results(results, query)
    
    def _format_results(self, results: list, query: str) -> str:
        """Format search results for display."""
        lines = [f"# Insight Search Results for '{query}'\n"]
        lines.append(f"Found {len(results)} matching insights:\n")
        
        for i, r in enumerate(results, 1):
            lines.append(f"## Insight {i}")
            lines.append(f"**Source:** {r.get('source', 'unknown')}")
            if r.get('type'):
                lines.append(f"**Type:** {r.get('type')}")
            if r.get('timestamp'):
                lines.append(f"**Time:** {r.get('timestamp')}")
            if r.get('relevance'):
                lines.append(f"**Relevance:** {r.get('relevance'):.2f}")
            lines.append(f"\n{r.get('content', 'No content')}\n")
            lines.append("---\n")
        
        return "\n".join(lines)


class InsightHeatmapTool(Tool):
    """Get word frequency heatmap from insights.
    
    Shows the most common words appearing in insights.
    """
    
    name = "insight_heatmap"
    description = """Get the word frequency heatmap from accumulated insights.

Returns the most common words appearing across all insights, useful for
understanding what topics and concepts Jarvis has learned about.

Use this to:
- See what topics are most common in insights
- Understand knowledge distribution
- Identify patterns in learning
"""
    parameters = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Number of top words to return (default: 50)",
                "default": 50,
            },
        },
    }
    requires_permission = False
    
    async def execute(self, limit: int = 50) -> str:
        """Execute heatmap request."""
        try:
            from core.insight_review import get_insight_review
            review = get_insight_review()
            
            heatmap = review.get_word_heatmap(limit)
            
            lines = ["# Insight Word Heatmap\n"]
            lines.append("Most common words in accumulated insights:\n")
            lines.append("| Word | Count |")
            lines.append("|------|-------|")
            
            for word, count in heatmap:
                lines.append(f"| {word} | {count} |")
            
            stats = review.get_stats()
            lines.append(f"\n**Total Insights:** {stats.get('total_insights', 0)}")
            lines.append(f"**Unique Words:** {stats.get('unique_words', 0)}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error getting insight heatmap: {e}"


class InsightStatsTool(Tool):
    """Get statistics about accumulated insights."""
    
    name = "insight_stats"
    description = """Get statistics about Jarvis's accumulated insights.

Returns counts, sources, types, and other metadata about the insight database.

Use this to:
- See how many insights have been accumulated
- Check insight sources and types
- Monitor knowledge growth
"""
    parameters = {
        "type": "object",
        "properties": {},
    }
    requires_permission = False
    
    async def execute(self) -> str:
        """Execute stats request."""
        try:
            from core.insight_review import get_insight_review
            review = get_insight_review()
            
            stats = review.get_stats()
            
            lines = ["# Insight Statistics\n"]
            lines.append(f"**Total Insights:** {stats.get('total_insights', 0)}")
            lines.append(f"**Unique Hashes:** {stats.get('total_hashes', 0)}")
            lines.append(f"**Unique Words:** {stats.get('unique_words', 0)}")
            
            lines.append("\n## By Source")
            for source, count in stats.get('sources', {}).items():
                lines.append(f"- **{source}:** {count}")
            
            lines.append("\n## By Type")
            for itype, count in stats.get('types', {}).items():
                lines.append(f"- **{itype}:** {count}")
            
            lines.append("\n## Top Words")
            for word, count in stats.get('top_words', [])[:10]:
                lines.append(f"- **{word}:** {count}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error getting insight stats: {e}"


# Export tools
TOOLS = [InsightSearchTool(), InsightHeatmapTool(), InsightStatsTool()]