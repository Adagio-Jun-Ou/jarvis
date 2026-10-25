"""Knowledge Graph - Manages insight relationships with PageRank and community detection.

Builds a graph of insights across domains, tracks their relationships,
and identifies important knowledge clusters using graph algorithms.
Supports concept extraction, semantic search, and cross-domain linking.
"""
import hashlib
import logging
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class NodeType(Enum):
    INSIGHT = "insight"
    PROBLEM = "problem"
    STRATEGY = "strategy"
    CATEGORY = "category"
    CONCEPT = "concept"
    MUSICAL_INSIGHT = "musical_insight"
    ENTITY = "entity"
    DOMAIN = "domain"
    TECHNIQUE = "technique"


class EdgeType(Enum):
    DERIVED_FROM = "derived_from"
    RELATED_TO = "related_to"
    SOLVES = "solves"
    USES_STRATEGY = "uses_strategy"
    IN_CATEGORY = "in_category"
    REFERENCES_CONCEPT = "references_concept"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    IN_DOMAIN = "in_domain"
    USES_TECHNIQUE = "uses_technique"
    MENTIONS_ENTITY = "mentions_entity"


@dataclass
class GraphNode:
    id: str
    node_type: NodeType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    pagerank: float = 0.0
    community_id: int = -1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "pagerank": round(self.pagerank, 6),
            "community_id": self.community_id,
        }


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "weight": self.weight,
        }


class KnowledgeGraph:
    """Graph-based knowledge management with PageRank and community detection."""

    MAX_NODES = 10000
    MAX_EDGES = 50000
    MAX_COMMUNITIES = 100

    PAGERANK_DAMPING = 0.85
    PAGERANK_ITERATIONS = 20
    COMMUNITY_RESOLUTION = 1.0

    SIMILARITY_THRESHOLD = 0.8

    CONCEPT_KEYWORDS = {
        "ramsey": ["ramsey", "coloring", " clique", "complete graph"],
        "prime": ["prime", "divisor", "factor", "gcd"],
        "graph_theory": ["graph", "vertex", "edge", "degree", "path"],
        "combinatorics": ["combination", "permutation", "counting", "arrangement"],
        "probability": ["probability", "random", "expected", "distribution"],
        "number_theory": ["integer", "modulo", "divisibility", "congruence"],
        "optimization": ["optimal", "minimize", "maximize", "bound"],
        "symmetry": ["symmetric", "invariant", "transformation"],
        "induction": ["induction", "base case", "inductive step"],
        "extremal": ["extremal", "maximum", "minimum", "best"],
        "machine_learning": ["neural network", "deep learning", "training", "inference",
                              "gradient", "backpropagation", "loss function", "epoch",
                              "batch", "model", "embedding", "transformer", "attention",
                              "fine-tuning", "overfitting", "regularization"],
        "nlp": ["natural language", "tokenization", "embedding", "sentiment",
                "named entity", "language model", "parse", "semantic"],
        "computer_vision": ["image recognition", "object detection", "segmentation",
                            "convolutional", "feature extraction", "visual"],
        "cybersecurity": ["encryption", "authentication", "vulnerability", "exploit",
                          "firewall", "intrusion", "malware", "phishing", "zero-day"],
        "distributed_systems": ["consensus", "replication", "partition", "consistency",
                                "availability", "byzantine", "fault tolerance", "sharding"],
        "databases": ["query", "index", "transaction", "normalization", "join",
                       "relational", "nosql", "acid", "migration"],
        "web_development": ["rest api", "websocket", "frontend", "backend", "middleware",
                            "cors", "oauth", "session", "cookie", "http"],
        "devops": ["container", "orchestration", "ci/cd", "deployment", "monitoring",
                   "microservice", "kubernetes", "docker", "pipeline"],
        "algorithms": ["sorting", "searching", "dynamic programming", "greedy",
                       "backtracking", "divide and conquer", "hash", "tree traversal"],
        "data_science": ["regression", "classification", "clustering", "feature",
                         "dataset", "correlation", "variance", "hypothesis", "p-value"],
        "physics": ["entropy", "quantum", "relativity", "thermodynamics",
                    "electromagnetic", "oscillation", "wave", "momentum"],
        "economics": ["inflation", "gdp", "fiscal", "monetary", "supply demand",
                      "market", "interest rate", "debt", "recession"],
        "biology": ["protein", "genome", "cell", "dna", "rna", "enzyme",
                    "mutation", "evolution", "selection", "expression"],
        "psychology": ["cognitive", "bias", "perception", "memory", "emotion",
                       "motivation", "behavior", "learning", "attention"],
        "music_theory": ["harmony", "melody", "rhythm", "chord", "scale",
                         "interval", "tonality", "cadence", "counterpoint"],
    }

    ENTITY_PATTERNS = [
        (re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'), "proper_noun"),
        (re.compile(r'"([^"]+)"'), "quoted_term"),
        (re.compile(r'`([^`]+)`'), "code_term"),
    ]

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "knowledge_graph"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, list[str]] = defaultdict(list)
        self.reverse_edges: dict[str, list[str]] = defaultdict(list)
        self.edge_details: dict[str, GraphEdge] = {}

        self._node_counter = 0
        self._lock = threading.RLock()
        self._MAX_EDGES_PER_NODE = 50

        self._pagerank_valid = False
        self._communities_valid = False

        self._keyword_index: dict[str, set[str]] = defaultdict(set)
        self._MAX_KEYWORD_INDEX = 10000  # Bound on keyword index
        
        self._interconnect = None
        
        self._load_state()
    
    def _get_interconnect(self):
        """Get shared embedding interconnect."""
        if self._interconnect is None:
            try:
                from core.embedding_interconnect import get_embedding_interconnect
                self._interconnect = get_embedding_interconnect()
            except Exception as e:
                logger.debug(f"Embedding interconnect unavailable: {e}")
        return self._interconnect

    def _load_state(self) -> None:
        state_file = self.storage_path / "graph_state.json"
        if state_file.exists():
            try:
                import json
                with open(state_file, encoding='utf-8') as f:
                    data = json.load(f)
                    self._node_counter = data.get("counter", 0)
                    for node_id, node_data in data.get("nodes_data", {}).items():
                        try:
                            self.nodes[node_id] = GraphNode(
                                id=node_id,
                                node_type=NodeType(node_data["type"]),
                                content=node_data["content"],
                                metadata=node_data.get("metadata", {}),
                                pagerank=node_data.get("pagerank", 0.0),
                                created_at=datetime.fromisoformat(node_data["created_at"]) if node_data.get("created_at") else datetime.now(),
                            )
                            self._index_keywords(node_id, node_data["content"])
                        except Exception as e:
                            logger.debug(f"Could not load node {node_id}: {e}")
                    for edge_key, edge_data in data.get("edges_data", {}).items():
                        try:
                            self.edge_details[edge_key] = GraphEdge(
                                source_id=edge_data["source_id"] if "source_id" in edge_data else edge_data.get("source", ""),
                                target_id=edge_data["target_id"] if "target_id" in edge_data else edge_data.get("target", ""),
                                edge_type=EdgeType(edge_data["edge_type"]),
                                weight=edge_data.get("weight", 1.0),
                                metadata=edge_data.get("metadata", {}),
                            )
                            # Rebuild edge indices
                            source = self.edge_details[edge_key].source_id
                            target = self.edge_details[edge_key].target_id
                            if source not in self.edges:
                                self.edges[source] = []
                            self.edges[source].append(edge_key)
                            if target not in self.reverse_edges:
                                self.reverse_edges[target] = []
                            self.reverse_edges[target].append(edge_key)
                        except Exception as e:
                            logger.debug(f"Could not load edge {edge_key}: {e}")
                    self._pagerank_valid = False
                    self._communities_valid = False
                    logger.info(f"Loaded knowledge graph: {len(self.nodes)} nodes, {len(self.edge_details)} edges")
            except Exception as e:
                logger.warning(f"Failed to load graph state: {e}")

    def _save_state(self) -> None:
        """Save graph state to disk (acquires lock for data snapshot, I/O outside)."""
        state_file = self.storage_path / "graph_state.json"
        try:
            with self._lock:
                import json
                nodes_data = {}
                for node_id, node in self.nodes.items():
                    nodes_data[node_id] = {
                        "type": node.node_type.value,
                        "content": node.content,
                        "metadata": dict(node.metadata),
                        "pagerank": node.pagerank,
                        "created_at": node.created_at.isoformat() if node.created_at else None,
                    }
                edges_data = {}
                for edge_key, edge in self.edge_details.items():
                    edges_data[edge_key] = {
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "edge_type": edge.edge_type.value,
                        "weight": edge.weight,
                        "metadata": dict(edge.metadata),
                    }
                save_data = {
                    "counter": self._node_counter,
                    "nodes": len(self.nodes),
                    "edges": len(self.edge_details),
                    "nodes_data": nodes_data,
                    "edges_data": edges_data,
                    "last_updated": datetime.now().isoformat(),
                }
            tmp = state_file.with_suffix(".tmp")
            with open(tmp, "w", encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)
            tmp.replace(state_file)
        except Exception as e:
            logger.warning(f"Failed to save graph state: {e}")

    def add_node(
        self,
        content: str,
        node_type: NodeType,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        should_save = False
        with self._lock:
            if len(self.nodes) >= self.MAX_NODES:
                self._evict_low_pagerank_nodes(100)

            existing = self._find_similar_node(content, node_type)
            if existing:
                return existing

            self._node_counter += 1
            node_id = f"{node_type.value}_{self._node_counter:06d}"

            node = GraphNode(
                id=node_id,
                node_type=node_type,
                content=content,
                metadata=metadata or {},
            )

            self.nodes[node_id] = node
            self._index_keywords(node_id, content)
            self._enforce_keyword_bounds()
            
            interconnect = self._get_interconnect()
            if interconnect:
                try:
                    interconnect.store_knowledge_node(node_id, content)
                except Exception as e:
                    logger.debug(f"Failed to store node embedding: {e}")

            self._pagerank_valid = False
            self._communities_valid = False
            should_save = True

        if should_save:
            self._save_state()

        return node_id

    def _find_similar_node(self, content: str, node_type: NodeType) -> str | None:
        content_lower = content.lower()
        for node_id, node in self.nodes.items():
            if node.node_type != node_type:
                continue
            similarity = self._compute_similarity(content_lower, node.content.lower())
            if similarity >= self.SIMILARITY_THRESHOLD:
                return node_id
        return None

    def _compute_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _index_keywords(self, node_id: str, content: str) -> None:
        words = content.lower().split()
        for word in words:
            self._keyword_index[word].add(node_id)


    def _enforce_keyword_bounds(self) -> None:
        """Enforce bounds on keyword index."""
        if len(self._keyword_index) > self._MAX_KEYWORD_INDEX:
            # Remove words that reference fewest nodes
            word_counts = {w: len(nodes) for w, nodes in self._keyword_index.items()}
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1])
            words_to_remove = len(self._keyword_index) - self._MAX_KEYWORD_INDEX
            for word, _ in sorted_words[:words_to_remove]:
                del self._keyword_index[word]

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        should_save = False
        with self._lock:
            if source_id not in self.nodes or target_id not in self.nodes:
                return False

            edge_key = f"{source_id}:{target_id}:{edge_type.value}"

            if edge_key in self.edge_details:
                self.edge_details[edge_key].weight = max(
                    self.edge_details[edge_key].weight, weight
                )
                should_save = True
            else:
                if len(self.edge_details) >= self.MAX_EDGES:
                    self._evict_low_weight_edges(500)

                edge = GraphEdge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type,
                    weight=weight,
                    metadata=metadata or {},
                )

                self.edge_details[edge_key] = edge
                self.edges[source_id].append(edge_key)
                self.reverse_edges[target_id].append(edge_key)

                if len(self.edges[source_id]) > self._MAX_EDGES_PER_NODE:
                    self.edges[source_id] = self._top_weighted_edges(source_id, self.edges[source_id])
                if len(self.reverse_edges[target_id]) > self._MAX_EDGES_PER_NODE:
                    self.reverse_edges[target_id] = self._top_weighted_edges(target_id, self.reverse_edges[target_id])

                self._pagerank_valid = False
                self._communities_valid = False
                should_save = True

        if should_save:
            self._save_state()
        return True

    def compute_pagerank(self, force: bool = False) -> dict[str, float]:
        with self._lock:
            if self._pagerank_valid and not force:
                return {nid: n.pagerank for nid, n in self.nodes.items()}

            if not self.nodes:
                return {}

            n = len(self.nodes)
            node_ids = list(self.nodes.keys())
            node_idx = {nid: i for i, nid in enumerate(node_ids)}

            ranks = [1.0 / n] * n

            out_degree = [0] * n
            for nid in node_ids:
                out_degree[node_idx[nid]] = len([
                    e for e in self.edges[nid]
                    if self.edge_details[e].target_id in node_idx
                ])

            for iteration in range(self.PAGERANK_ITERATIONS):
                new_ranks = [(1 - self.PAGERANK_DAMPING) / n] * n

                for target_id in node_ids:
                    target_idx = node_idx[target_id]

                    incoming_sum = 0.0
                    for edge_key in self.reverse_edges[target_id]:
                        edge = self.edge_details[edge_key]
                        source_id = edge.source_id

                        if source_id not in node_idx:
                            continue

                        source_idx = node_idx[source_id]
                        deg = out_degree[source_idx]

                        if deg > 0:
                            incoming_sum += ranks[source_idx] / deg * edge.weight

                    new_ranks[target_idx] += self.PAGERANK_DAMPING * incoming_sum

                total = sum(new_ranks)
                if total > 0:
                    ranks = [r / total for r in new_ranks]
                else:
                    ranks = new_ranks

            for nid, rank in zip(node_ids, ranks):
                self.nodes[nid].pagerank = rank

            self._pagerank_valid = True
            logger.info(f"Computed PageRank for {n} nodes")

            return {nid: n.pagerank for nid, n in self.nodes.items()}

    def detect_communities(self, force: bool = False) -> dict[str, int]:
        with self._lock:
            if self._communities_valid and not force:
                return {nid: n.community_id for nid, n in self.nodes.items()}

            if not self.nodes:
                return {}

            communities = {nid: i for i, nid in enumerate(self.nodes.keys())}

            for _ in range(10):
                changed = False
                node_ids = list(self.nodes.keys())

                for nid in node_ids:
                    neighbor_communities: dict[int, float] = defaultdict(float)

                    for edge_key in self.edges[nid]:
                        edge = self.edge_details[edge_key]
                        if edge.target_id in communities:
                            neighbor_communities[communities[edge.target_id]] += edge.weight

                    for edge_key in self.reverse_edges[nid]:
                        edge = self.edge_details[edge_key]
                        if edge.source_id in communities:
                            neighbor_communities[communities[edge.source_id]] += edge.weight

                    if not neighbor_communities:
                        continue

                    best_community = max(neighbor_communities.keys(), key=lambda c: neighbor_communities[c])

                    if communities[nid] != best_community:
                        communities[nid] = best_community
                        changed = True

                if not changed:
                    break

            unique_communities = sorted(set(communities.values()))
            community_map = {old: new for new, old in enumerate(unique_communities)}

            for nid, comm in communities.items():
                self.nodes[nid].community_id = community_map[comm]

            self._communities_valid = True
            logger.info(f"Detected {len(unique_communities)} communities")

            return {nid: n.community_id for nid, n in self.nodes.items()}

    def _evict_low_pagerank_nodes(self, count: int) -> None:
        if not self.nodes:
            return

        self.compute_pagerank()

        sorted_nodes = sorted(
            self.nodes.items(),
            key=lambda x: x[1].pagerank
        )

        evicted_keys = set()
        for nid, _ in sorted_nodes[:count]:
            evicted_keys.update(self.edges.get(nid, []))
            evicted_keys.update(self.reverse_edges.get(nid, []))
            del self.nodes[nid]
            del self.edges[nid]
            del self.reverse_edges[nid]

            for word, node_set in list(self._keyword_index.items()):
                node_set.discard(nid)
                if not node_set:
                    del self._keyword_index[word]

        for edge_key in evicted_keys:
            self.edge_details.pop(edge_key, None)

        for nid in list(self.edges):
            self.edges[nid] = [e for e in self.edges[nid] if e in self.edge_details]
        for nid in list(self.reverse_edges):
            self.reverse_edges[nid] = [e for e in self.reverse_edges[nid] if e in self.edge_details]

        logger.info(f"Evicted {min(count, len(sorted_nodes))} low PageRank nodes")

    def _top_weighted_edges(self, node_id: str, edge_keys: list[str]) -> list[str]:
        """Keep only the top-weighted edges for a node."""
        weighted = []
        for ek in edge_keys:
            if ek in self.edge_details:
                weighted.append((ek, self.edge_details[ek].weight))
        weighted.sort(key=lambda x: x[1], reverse=True)
        return [ek for ek, _ in weighted[:self._MAX_EDGES_PER_NODE]]

    def _evict_low_weight_edges(self, count: int) -> None:
        if not self.edge_details:
            return

        sorted_edges = sorted(
            self.edge_details.items(),
            key=lambda x: x[1].weight
        )

        for edge_key, _ in sorted_edges[:count]:
            edge = self.edge_details[edge_key]
            del self.edge_details[edge_key]

            if edge.source_id in self.edges:
                self.edges[edge.source_id] = [
                    e for e in self.edges[edge.source_id] if e != edge_key
                ]

            if edge.target_id in self.reverse_edges:
                self.reverse_edges[edge.target_id] = [
                    e for e in self.reverse_edges[edge.target_id] if e != edge_key
                ]

        logger.info(f"Evicted {count} low weight edges")

    def add_insight(
        self,
        insight_id: str,
        content: str,
        problem_id: str | None = None,
        strategy: str | None = None,
        category: str | None = None,
        related_insights: list[str] | None = None,
        domain: str | None = None,
        technique: str | None = None,
    ) -> str:
        node_id = self.add_node(content, NodeType.INSIGHT, {"insight_id": insight_id})

        if problem_id:
            problem_node = self.add_node(problem_id, NodeType.PROBLEM)
            self.add_edge(node_id, problem_node, EdgeType.SOLVES)

        if strategy:
            strategy_node = self.add_node(strategy, NodeType.STRATEGY)
            self.add_edge(node_id, strategy_node, EdgeType.USES_STRATEGY)

        if category:
            category_node = self.add_node(category, NodeType.CATEGORY)
            self.add_edge(node_id, category_node, EdgeType.IN_CATEGORY)

        if domain:
            domain_node = self.add_node(domain, NodeType.DOMAIN)
            self.add_edge(node_id, domain_node, EdgeType.IN_DOMAIN)

        if technique:
            technique_node = self.add_node(technique, NodeType.TECHNIQUE)
            self.add_edge(node_id, technique_node, EdgeType.USES_TECHNIQUE)

        concepts = self._extract_concepts(content)
        for concept in concepts:
            concept_node = self.add_node(concept, NodeType.CONCEPT)
            self.add_edge(node_id, concept_node, EdgeType.REFERENCES_CONCEPT, 0.5)

        entities = self._extract_entities(content)
        for entity, etype in entities:
            entity_node = self.add_node(entity, NodeType.ENTITY, {"entity_type": etype})
            self.add_edge(node_id, entity_node, EdgeType.MENTIONS_ENTITY, 0.3)

        if related_insights:
            for related in related_insights:
                related_node = self._find_node_by_metadata("insight_id", related)
                if related_node:
                    self.add_edge(node_id, related_node, EdgeType.RELATED_TO, 0.7)

        return node_id

    def _extract_concepts(self, content: str) -> list[str]:
        content_lower = content.lower()
        found_concepts = []

        for concept, keywords in self.CONCEPT_KEYWORDS.items():
            if any(kw in content_lower for kw in keywords):
                found_concepts.append(concept)

        entities = self._extract_entities(content)
        for entity, _etype in entities:
            if entity.lower() not in [c.lower() for c in found_concepts]:
                found_concepts.append(entity.lower())

        return found_concepts[:10]

    def _extract_entities(self, content: str) -> list[tuple[str, str]]:
        entities = []
        seen = set()
        for pattern, etype in self.ENTITY_PATTERNS:
            for match in pattern.finditer(content):
                entity = match.group(1).strip()
                if len(entity) > 2 and entity.lower() not in seen:
                    seen.add(entity.lower())
                    entities.append((entity, etype))
        return entities[:20]

    def _find_node_by_metadata(self, key: str, value: str) -> str | None:
        for nid, node in self.nodes.items():
            if node.metadata.get(key) == value:
                return nid
        return None

    def get_important_insights(self, limit: int = 20) -> list[dict]:
        with self._lock:
            self.compute_pagerank()

            insight_nodes = [
                (nid, n) for nid, n in self.nodes.items()
                if n.node_type == NodeType.INSIGHT
            ]

            sorted_insights = sorted(
                insight_nodes,
                key=lambda x: x[1].pagerank,
                reverse=True
            )[:limit]

            return [n.to_dict() for nid, n in sorted_insights]

    def get_communities(self) -> dict[int, list[dict]]:
        with self._lock:
            self.detect_communities()

            communities: dict[int, list[dict]] = defaultdict(list)

            for node in self.nodes.values():
                if node.community_id >= 0:
                    communities[node.community_id].append(node.to_dict())

            return dict(communities)

    def get_related_insights(self, insight_id: str, max_depth: int = 2) -> list[dict]:
        with self._lock:
            node_id = self._find_node_by_metadata("insight_id", insight_id)
            if not node_id:
                return []

            visited = {node_id}
            frontier = {node_id}
            related = []

            for depth in range(max_depth):
                next_frontier = set()

                for nid in frontier:
                    for edge_key in self.edges[nid]:
                        edge = self.edge_details[edge_key]
                        if edge.target_id not in visited:
                            visited.add(edge.target_id)
                            next_frontier.add(edge.target_id)
                            if self.nodes[edge.target_id].node_type == NodeType.INSIGHT:
                                related.append({
                                    **self.nodes[edge.target_id].to_dict(),
                                    "distance": depth + 1,
                                    "relationship": edge.edge_type.value,
                                })

                    for edge_key in self.reverse_edges[nid]:
                        edge = self.edge_details[edge_key]
                        if edge.source_id not in visited:
                            visited.add(edge.source_id)
                            next_frontier.add(edge.source_id)
                            if self.nodes[edge.source_id].node_type == NodeType.INSIGHT:
                                related.append({
                                    **self.nodes[edge.source_id].to_dict(),
                                    "distance": depth + 1,
                                    "relationship": edge.edge_type.value,
                                })

                frontier = next_frontier

            return related[:50]

    def search_by_concept(self, concept: str, limit: int = 20) -> list[dict]:
        with self._lock:
            concept_lower = concept.lower()
            matching_nodes = []

            for node in self.nodes.values():
                if node.node_type not in (NodeType.INSIGHT, NodeType.CONCEPT):
                    continue

                node_concepts = self._extract_concepts(node.content)
                if concept_lower in node_concepts or concept_lower in node.content.lower():
                    matching_nodes.append(node)

            self.compute_pagerank()
            matching_nodes.sort(key=lambda n: n.pagerank, reverse=True)

            return [n.to_dict() for n in matching_nodes[:limit]]
    
    def search_semantic(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Search knowledge graph using semantic similarity via EmbeddingInterconnect."""
        interconnect = self._get_interconnect()
        if not interconnect:
            return []
        
        try:
            results = interconnect.search_knowledge(query, top_k=limit)
            return results
        except Exception as e:
            logger.debug(f"Semantic search failed: {e}")
            return []

    def get_graph_statistics(self) -> dict[str, Any]:
        with self._lock:
            self.compute_pagerank()
            self.detect_communities()

            node_types: dict[str, int] = defaultdict(int)
            for node in self.nodes.values():
                node_types[node.node_type.value] += 1

            edge_types: dict[str, int] = defaultdict(int)
            for edge in self.edge_details.values():
                edge_types[edge.edge_type.value] += 1

            communities = self.get_communities()

            pageranks = [n.pagerank for n in self.nodes.values()]
            avg_pagerank = sum(pageranks) / len(pageranks) if pageranks else 0

            return {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edge_details),
                "node_types": dict(node_types),
                "edge_types": dict(edge_types),
                "communities": len(communities),
                "avg_pagerank": round(avg_pagerank, 6),
                "max_pagerank": round(max(pageranks), 6) if pageranks else 0,
            }

    def get_pagerank_insights(self, limit: int = 20) -> list[dict]:
        with self._lock:
            self.compute_pagerank()
            all_nodes = sorted(
                self.nodes.values(),
                key=lambda n: n.pagerank,
                reverse=True,
            )[:limit]
            return [
                {
                    "id": n.id,
                    "type": n.node_type.value,
                    "content": n.content[:200],
                    "pagerank": round(n.pagerank, 6),
                    "community_id": n.community_id,
                }
                for n in all_nodes
            ]

    def search_concepts(self, query: str, limit: int = 20) -> list[dict]:
        with self._lock:
            query_lower = query.lower()
            results = []
            for node in self.nodes.values():
                if node.node_type == NodeType.CONCEPT:
                    if query_lower in node.content.lower() or node.content.lower() in query_lower:
                        connected = self._get_connected_insights(node.id, limit=3)
                        results.append({
                            "concept": node.content,
                            "concept_id": node.id,
                            "pagerank": round(node.pagerank, 6),
                            "connected_insights": connected,
                        })
            results.sort(key=lambda r: r["pagerank"], reverse=True)
            return results[:limit]

    def _get_connected_insights(self, node_id: str, limit: int = 5) -> list[dict]:
        insights = []
        for edge_key in self.reverse_edges.get(node_id, []):
            edge = self.edge_details.get(edge_key)
            if edge and edge.source_id in self.nodes:
                source = self.nodes[edge.source_id]
                if source.node_type == NodeType.INSIGHT:
                    insights.append({
                        "id": source.id,
                        "content": source.content[:200],
                        "relationship": edge.edge_type.value,
                    })
                    if len(insights) >= limit:
                        break
        for edge_key in self.edges.get(node_id, []):
            edge = self.edge_details.get(edge_key)
            if edge and edge.target_id in self.nodes:
                target = self.nodes[edge.target_id]
                if target.node_type == NodeType.INSIGHT:
                    insights.append({
                        "id": target.id,
                        "content": target.content[:200],
                        "relationship": edge.edge_type.value,
                    })
                    if len(insights) >= limit:
                        break
        return insights

    def get_neighborhood(self, node_id: str, depth: int = 1) -> dict[str, Any]:
        with self._lock:
            if node_id not in self.nodes:
                return {"error": "Node not found", "nodes": [], "edges": []}
            visited_nodes = {}
            visited_edges = {}
            frontier = {node_id}
            for d in range(depth + 1):
                next_frontier = set()
                for nid in frontier:
                    if nid in visited_nodes:
                        continue
                    if nid in self.nodes:
                        visited_nodes[nid] = self.nodes[nid]
                    for ek in self.edges.get(nid, []):
                        if ek in self.edge_details and ek not in visited_edges:
                            visited_edges[ek] = self.edge_details[ek]
                            target = self.edge_details[ek].target_id
                            if target not in visited_nodes:
                                next_frontier.add(target)
                    for ek in self.reverse_edges.get(nid, []):
                        if ek in self.edge_details and ek not in visited_edges:
                            visited_edges[ek] = self.edge_details[ek]
                            source = self.edge_details[ek].source_id
                            if source not in visited_nodes:
                                next_frontier.add(source)
                frontier = next_frontier
            return {
                "center": node_id,
                "depth": depth,
                "nodes": [n.to_dict() for n in visited_nodes.values()],
                "edges": [e.to_dict() for e in visited_edges.values()],
            }

    def shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        with self._lock:
            if source_id not in self.nodes or target_id not in self.nodes:
                return None
            visited = {source_id}
            queue = deque([(source_id, [source_id])])
            while queue:
                current, path = queue.popleft()
                if current == target_id:
                    return path
                neighbors = set()
                for ek in self.edges.get(current, []):
                    if ek in self.edge_details:
                        neighbors.add(self.edge_details[ek].target_id)
                for ek in self.reverse_edges.get(current, []):
                    if ek in self.edge_details:
                        neighbors.add(self.edge_details[ek].source_id)
                for neighbor in neighbors:
                    if neighbor not in visited and neighbor in self.nodes:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            return None

    def get_domain_map(self) -> dict[str, list[dict]]:
        with self._lock:
            domain_map: dict[str, list[dict]] = defaultdict(list)
            for node in self.nodes.values():
                if node.node_type == NodeType.DOMAIN:
                    for ek in self.reverse_edges.get(node.id, []):
                        edge = self.edge_details.get(ek)
                        if edge and edge.source_id in self.nodes:
                            source = self.nodes[edge.source_id]
                            domain_map[node.content].append({
                                "id": source.id,
                                "type": source.node_type.value,
                                "content": source.content[:100],
                            })
            return dict(domain_map)

    def auto_link_similar(self, threshold: float | None = None) -> int:
        with self._lock:
            if threshold is None:
                threshold = self.SIMILARITY_THRESHOLD
            links_added = 0
            insight_nodes = [
                (nid, n) for nid, n in self.nodes.items()
                if n.node_type == NodeType.INSIGHT
            ]
            for i, (nid_a, node_a) in enumerate(insight_nodes):
                for nid_b, node_b in insight_nodes[i + 1:]:
                    existing = f"{nid_a}:{nid_b}:{EdgeType.SIMILAR_TO.value}"
                    existing_rev = f"{nid_b}:{nid_a}:{EdgeType.SIMILAR_TO.value}"
                    if existing in self.edge_details or existing_rev in self.edge_details:
                        continue
                    sim = self._compute_similarity(node_a.content, node_b.content)
                    if sim >= threshold:
                        self.add_edge(nid_a, nid_b, EdgeType.SIMILAR_TO, sim)
                        links_added += 1
                        if links_added >= 100:
                            return links_added
            return links_added

    def fuzzy_search_concepts(self, query: str, limit: int = 20, min_score: float = 0.3) -> list[dict]:
        """Fuzzy concept search with scoring based on keyword overlap and pagerank.

        Unlike search_concepts which requires exact substring match, this
        method scores concepts by token overlap with the query.
        """
        with self._lock:
            query_tokens = set(query.lower().split())
            if not query_tokens:
                return []

            results = []
            for node in self.nodes.values():
                if node.node_type != NodeType.CONCEPT:
                    continue
                content_tokens = set(node.content.lower().split())
                overlap = len(query_tokens & content_tokens)
                if overlap == 0:
                    query_chars = set(query.lower())
                    content_chars = set(node.content.lower())
                    char_overlap = len(query_chars & content_chars) / max(len(query_chars | content_chars), 1)
                    if char_overlap < min_score:
                        continue
                    score = char_overlap * 0.5 + node.pagerank * 0.5
                else:
                    precision = overlap / max(len(query_tokens), 1)
                    recall = overlap / max(len(content_tokens), 1)
                    f1 = 2 * precision * recall / max(precision + recall, 0.001)
                    score = f1 * 0.6 + min(node.pagerank, 1.0) * 0.4

                if score >= min_score:
                    connected = self._get_connected_insights(node.id, limit=3)
                    results.append({
                        "concept": node.content,
                        "concept_id": node.id,
                        "score": round(score, 4),
                        "pagerank": round(node.pagerank, 6),
                        "connected_insights": connected,
                    })

            results.sort(key=lambda r: r["score"], reverse=True)
            return results[:limit]

    def get_temporal_insights(self, days: int = 30, limit: int = 20) -> list[dict]:
        """Get recent insights with temporal decay scoring.
        
        Newer insights get boosted scores, useful for identifying
        evolving trends and recent discoveries.
        """
        import time as _time_mod
        with self._lock:
            now = _time_mod.time()
            cutoff = now - (days * 86400)

            recent = []
            for node in self.nodes.values():
                created_ts = node.created_at.timestamp() if hasattr(node.created_at, 'timestamp') else 0
                if created_ts < cutoff:
                    continue

                age_days = max((now - created_ts) / 86400, 0.1)
                recency_score = 1.0 / (1.0 + age_days / 7.0)
                combined_score = recency_score * 0.4 + node.pagerank * 0.6

                recent.append({
                    "id": node.id,
                    "type": node.node_type.value,
                    "content": node.content[:200],
                    "created_at": node.created_at.isoformat() if hasattr(node.created_at, 'isoformat') else str(node.created_at),
                    "age_days": round(age_days, 1),
                    "recency_score": round(recency_score, 4),
                    "pagerank": round(node.pagerank, 6),
                    "combined_score": round(combined_score, 4),
                })

            recent.sort(key=lambda r: r["combined_score"], reverse=True)
            return recent[:limit]

    def find_knowledge_paths(self, source_query: str, target_query: str, max_hops: int = 5) -> list[dict]:
        """Find knowledge paths between two concepts/topics.
        
        Discovers how two topics connect through the knowledge graph,
        useful for cross-domain insight discovery.
        """
        with self._lock:
            source_nodes = self._find_matching_nodes(source_query)
            target_nodes = self._find_matching_nodes(target_query)

            if not source_nodes or not target_nodes:
                return []

            source_ids = {n["id"] for n in source_nodes}
            target_ids = {n["id"] for n in target_nodes}

            paths = []
            for sid in source_ids:
                for tid in target_ids:
                    if sid == tid:
                        continue
                    path = self.shortest_path(sid, tid)
                    if path and len(path) <= max_hops + 1:
                        path_data = {
                            "source_concept": source_query,
                            "target_concept": target_query,
                            "path_length": len(path) - 1,
                            "nodes": [],
                        }
                        for nid in path:
                            if nid in self.nodes:
                                n = self.nodes[nid]
                                path_data["nodes"].append({
                                    "id": n.id,
                                    "type": n.node_type.value,
                                    "content": n.content[:100],
                                    "pagerank": round(n.pagerank, 6),
                                })
                        paths.append(path_data)

            paths.sort(key=lambda p: p["path_length"])
            return paths[:10]

    def _find_matching_nodes(self, query: str, limit: int = 5) -> list[dict]:
        """Find nodes matching a query string."""
        query_lower = query.lower()
        matches = []
        for node in self.nodes.values():
            if query_lower in node.content.lower() or node.content.lower() in query_lower:
                matches.append({
                    "id": node.id,
                    "type": node.node_type.value,
                    "content": node.content[:100],
                    "score": 1.0,
                })
            elif any(w in node.content.lower() for w in query_lower.split() if len(w) > 2):
                matches.append({
                    "id": node.id,
                    "type": node.node_type.value,
                    "content": node.content[:100],
                    "score": 0.5,
                })

        matches.sort(key=lambda m: m["score"], reverse=True)
        return matches[:limit]

    def get_cross_domain_connections(self, domain_a: str, domain_b: str) -> list[dict]:
        """Find connections between two domains in the knowledge graph.
        
        Discovers bridging concepts and insights that connect different
        knowledge domains.
        """
        with self._lock:
            nodes_a = set()
            nodes_b = set()

            for node in self.nodes.values():
                if node.node_type == NodeType.DOMAIN:
                    if domain_a.lower() in node.content.lower():
                        for ek in self.edges.get(node.id, []) + self.reverse_edges.get(node.id, []):
                            edge = self.edge_details.get(ek)
                            if edge:
                                other_id = edge.target_id if edge.source_id == node.id else edge.source_id
                                if other_id in self.nodes and self.nodes[other_id].node_type != NodeType.DOMAIN:
                                    nodes_a.add(other_id)
                    if domain_b.lower() in node.content.lower():
                        for ek in self.edges.get(node.id, []) + self.reverse_edges.get(node.id, []):
                            edge = self.edge_details.get(ek)
                            if edge:
                                other_id = edge.target_id if edge.source_id == node.id else edge.source_id
                                if other_id in self.nodes and self.nodes[other_id].node_type != NodeType.DOMAIN:
                                    nodes_b.add(other_id)

                if node.node_type == NodeType.INSIGHT:
                    content_lower = node.content.lower()
                    if domain_a.lower() in content_lower:
                        nodes_a.add(node.id)
                    if domain_b.lower() in content_lower:
                        nodes_b.add(node.id)

            bridge_nodes = nodes_a & nodes_b
            connections = []
            for nid in bridge_nodes:
                if nid in self.nodes:
                    n = self.nodes[nid]
                    connections.append({
                        "id": n.id,
                        "type": n.node_type.value,
                        "content": n.content[:200],
                        "pagerank": round(n.pagerank, 6),
                        "bridge": True,
                    })

            for nid in nodes_a - bridge_nodes:
                for nid2 in nodes_b - bridge_nodes:
                    edge_key = f"{nid}:{nid2}:related_to"
                    if edge_key in self.edge_details:
                        if nid in self.nodes:
                            connections.append({
                                "id": nid,
                                "type": self.nodes[nid].node_type.value,
                                "content": self.nodes[nid].content[:200],
                                "pagerank": round(self.nodes[nid].pagerank, 6),
                                "bridge": False,
                                "connected_to": nid2,
                            })

            connections.sort(key=lambda c: c.get("pagerank", 0), reverse=True)
            return connections[:20]


_knowledge_graph: KnowledgeGraph | None = None
_graph_lock = threading.Lock()


def get_knowledge_graph() -> KnowledgeGraph:
    global _knowledge_graph
    if _knowledge_graph is None:
        with _graph_lock:
            if _knowledge_graph is None:
                _knowledge_graph = KnowledgeGraph()
    return _knowledge_graph


class KnowledgeGraphContext:
    """Context layer integration for knowledge graph.
    
    Provides context-aware retrieval optimized for LLM context assembly.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph | None = None):
        self._kg = knowledge_graph
        self._context_cache: dict[str, list[dict]] = {}
        self._cache_lock = threading.RLock()
        self._cache_ttl = 600  # 10 minutes - graph queries stable
        self._last_cache_update: dict[str, float] = {}
        self._max_cache_size = 500  # Allow rich context caching
        self._cache_order: deque[str] = deque()
    
    def _evict_cache(self) -> None:
        """Evict expired and excess cache entries."""
        now = time.time()
        expired_keys = [
            k for k, t in self._last_cache_update.items()
            if now - t > self._cache_ttl
        ]
        for key in expired_keys:
            self._context_cache.pop(key, None)
            self._last_cache_update.pop(key, None)
            if key in self._cache_order:
                self._cache_order.remove(key)
        
        while len(self._context_cache) > self._max_cache_size:
            if self._cache_order:
                oldest = self._cache_order.popleft()
                self._context_cache.pop(oldest, None)
                self._last_cache_update.pop(oldest, None)
            else:
                break
    
    def _get_kg(self) -> KnowledgeGraph:
        if self._kg is None:
            self._kg = get_knowledge_graph()
        return self._kg
    
    def get_context_for_query(
        self,
        query: str,
        max_items: int = 10,
        max_tokens: int = 2000,
    ) -> list[dict]:
        """Get relevant context from knowledge graph for a query."""
        cache_key = hashlib.sha256(f"{query}:{max_items}:{max_tokens}".encode()).hexdigest()
        
        with self._cache_lock:
            if cache_key in self._context_cache:
                if time.time() - self._last_cache_update.get(cache_key, 0) < self._cache_ttl:
                    return self._context_cache[cache_key]
        
        kg = self._get_kg()
        context_items = []
        total_tokens = 0
        
        concepts = kg._extract_concepts(query)
        for concept in concepts[:3]:
            results = kg.search_by_concept(concept, limit=5)
            for item in results:
                tokens = len(item.get("content", "").split()) * 2
                if total_tokens + tokens <= max_tokens:
                    context_items.append({
                        "role": "system",
                        "content": f"[Knowledge: {concept}] {item.get('content', '')}",
                        "source": "knowledge_graph",
                        "pagerank": item.get("pagerank", 0),
                    })
                    total_tokens += tokens
        
        insights = kg.get_important_insights(limit=5)
        for insight in insights:
            tokens = len(insight.get("content", "").split()) * 2
            if total_tokens + tokens <= max_tokens and len(context_items) < max_items:
                content = insight.get("content", "")
                if content not in [c.get("content", "") for c in context_items]:
                    context_items.append({
                        "role": "system",
                        "content": f"[Important Insight] {content}",
                        "source": "knowledge_graph",
                        "pagerank": insight.get("pagerank", 0),
                    })
                    total_tokens += tokens
        
        context_items.sort(key=lambda x: x.get("pagerank", 0), reverse=True)
        result = context_items[:max_items]
        
        with self._cache_lock:
            self._evict_cache()
            self._context_cache[cache_key] = result
            self._last_cache_update[cache_key] = time.time()
            if cache_key not in self._cache_order:
                self._cache_order.append(cache_key)
        
        return result
    
    def get_related_context(
        self,
        insight_id: str,
        max_depth: int = 2,
        max_items: int = 10,
    ) -> list[dict]:
        """Get context related to a specific insight."""
        kg = self._get_kg()
        related = kg.get_related_insights(insight_id, max_depth=max_depth)
        
        context_items = []
        for item in related[:max_items]:
            context_items.append({
                "role": "system",
                "content": f"[Related: {item.get('relationship', 'unknown')}] {item.get('content', '')}",
                "source": "knowledge_graph",
                "distance": item.get("distance", 1),
            })
        
        return context_items
    
    def clear_cache(self) -> None:
        with self._cache_lock:
            self._context_cache.clear()
            self._last_cache_update.clear()


_kg_context: KnowledgeGraphContext | None = None
_context_lock = threading.Lock()


def get_knowledge_graph_context() -> KnowledgeGraphContext:
    global _kg_context
    if _kg_context is None:
        with _context_lock:
            if _kg_context is None:
                _kg_context = KnowledgeGraphContext()
    return _kg_context



