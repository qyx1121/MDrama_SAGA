"""
Matchers module: Node matching, event neighborhood fingerprints, visualization.
"""

import re
import logging
from typing import List, Dict, Set
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import linear_sum_assignment

from config import NodeMatchConfig, GraphMatchConfig

logger = logging.getLogger(__name__)


def visualize_bipartite_matching(
    gt_triplets, pred_triplets, matches,
    title="Triplet Matching", save_path=None
):
    """
    matches: list of (gt_idx, pred_idx, score)
    triplets: list of ((type, text), relation, (type, text))
    """
    G = nx.Graph()

    def triplet_label(t):
        return f"{t[0][1]}—[{t[1]}]—{t[2][1]}"

    gt_nodes = [f"GT_{i}: {triplet_label(t)}" for i, t in enumerate(gt_triplets)]
    pred_nodes = [f"Pred_{i}: {triplet_label(t)}" for i, t in enumerate(pred_triplets)]

    G.add_nodes_from(gt_nodes, bipartite=0)
    G.add_nodes_from(pred_nodes, bipartite=1)

    edges = []
    colors = []
    for gt_idx, pred_idx, score in matches:
        if score > 0:
            u = gt_nodes[gt_idx]
            v = pred_nodes[pred_idx]
            G.add_edge(u, v, weight=score)
            edges.append((u, v))
            colors.append(score)

    pos = {}
    for i, node in enumerate(gt_nodes):
        pos[node] = (-1, -i)
    for i, node in enumerate(pred_nodes):
        pos[node] = (1, -i)

    plt.figure(figsize=(16, max(6, len(gt_nodes) * 0.6 + len(pred_nodes) * 0.6)))

    nx.draw_networkx_nodes(G, pos, nodelist=gt_nodes, node_color='lightblue', node_shape='s', node_size=300)
    nx.draw_networkx_nodes(G, pos, nodelist=pred_nodes, node_color='lightgreen', node_shape='s', node_size=300)
    nx.draw_networkx_labels(G, pos, font_size=7, horizontalalignment='center')

    if edges:
        drawn_edges = nx.draw_networkx_edges(
            G, pos, edgelist=edges, edge_color=colors,
            edge_cmap=plt.cm.RdYlGn, width=2, edge_vmin=0.5, edge_vmax=1.0
        )
        plt.colorbar(drawn_edges, label="Match Score")

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


class NodeMatcher:
    # "[Role] ([Name])" format
    CHAR_FORMAT_PATTERN = re.compile(
        r'^(?P<role>[^(]+?)\s*\((?P<name>[^)]+)\)\s*$'
    )

    # Possessive pattern: "X's Y"
    POSSESSIVE_PATTERN = re.compile(
        r"^(?P<base>.+?)(?:'s?\s+)(?P<modifier>.+)$", re.IGNORECASE
    )

    # Words indicating a "relation/identity" → implies a different entity
    RELATIONAL_WORDS: Set[str] = {
        # Family relations
        "mother", "mom", "father", "dad", "parent",
        "sister", "brother", "sibling",
        "son", "daughter", "child", "children", "kid",
        "wife", "husband", "spouse", "fiancé", "fiancee", "fiance",
        "uncle", "aunt", "cousin", "nephew", "niece",
        "grandfather", "grandmother", "grandpa", "grandma",
        "grandson", "granddaughter",
        "mother-in-law", "father-in-law",
        "stepmother", "stepfather", "stepsister", "stepbrother",
        # Occupational/social relations
        "boss", "assistant", "secretary", "driver", "bodyguard",
        "lawyer", "doctor", "friend", "enemy", "rival",
        "colleague", "partner", "subordinate", "servant", "maid",
        "teacher", "student", "master", "disciple",
        # Possible English translations of Chinese kinship terms
        "ex", "ex-wife", "ex-husband", "ex-girlfriend", "ex-boyfriend",
        "lover", "mistress",
    }

    @classmethod
    def parse_character_name(cls, text: str) -> Dict[str, str]:
        """
        "Male Lead (Gu Han)" → {role: "Male Lead", name: "Gu Han"}
        "Female Lead"        → {role: "Female Lead", name: ""}
        "Chloe's mother"     → {role: "Chloe's mother", name: ""}
        """
        text = text.strip()
        match = cls.CHAR_FORMAT_PATTERN.match(text)
        if match:
            return {
                "role": match.group("role").strip(),
                "name": match.group("name").strip(),
                "raw": text
            }
        return {"role": text, "name": "", "raw": text}

    @classmethod
    def is_possessive_relation(cls, shorter: str, longer: str) -> bool:
        """
        ("Chloe", "Chloe's mother")   → True   different person
        ("Gu Han", "Gu Han's father") → True   different person
        ("Peter", "Peter's company")  → False  company is not a relational word
        ("Peter", "Peter Anderson")   → False  no possessive pattern
        """
        match = cls.POSSESSIVE_PATTERN.match(longer)
        if not match:
            return False

        base = match.group("base").strip().lower()
        modifier = match.group("modifier").strip().lower()

        if base != shorter.lower().strip():
            return False

        first_word = modifier.split()[0] if modifier else ""
        # also match the whole modifier, e.g. "mother-in-law"
        return (first_word in cls.RELATIONAL_WORDS or
                modifier in cls.RELATIONAL_WORDS)

    @classmethod
    def is_name_component_match(cls, shorter: str, longer: str) -> bool:
        """
        ("Peter", "Peter Anderson")    → True   same person
        ("Anderson", "Peter Anderson") → True   same person
        ("Gu Han", "Gu Han Chen")      → True   same person
        ("Han", "Gu Han")              → True   same person
        ("Peter", "Peterson")          → False  not a word-level match
        ("Chloe", "Chloe's mother")    → False  blocked by the possessive check
        """
        shorter_words = shorter.lower().strip().split()
        longer_words = longer.lower().strip().split()

        if not shorter_words or not longer_words:
            return False

        longer_lower = " ".join(longer_words)
        shorter_lower = " ".join(shorter_words)

        # every word of shorter appears in longer (word-level, not substring)
        if all(w in longer_words for w in shorter_words):
            return True

        if (longer_lower.startswith(shorter_lower + " ") or
            longer_lower.endswith(" " + shorter_lower)):
            return True

        return False

    @classmethod
    def classify_containment(
        cls, text_a: str, text_b: str
    ) -> str:
        """
        Classify the relationship between two texts.

        Returns:
            "IDENTICAL"     - exactly the same
            "POSSESSIVE"    - possessive kinship relation (different entities)
            "NAME_COMPONENT"- name component relation (same entity)
            "GENERIC_CONTAIN" - generic substring containment (needs extra judgment)
            "NO_CONTAIN"    - no containment relation
        """
        a = text_a.strip()
        b = text_b.strip()

        if a.lower() == b.lower():
            return "IDENTICAL"

        if len(a) <= len(b):
            shorter, longer = a, b
        else:
            shorter, longer = b, a

        # must run before the name component check
        if cls.is_possessive_relation(shorter, longer):
            return "POSSESSIVE"

        if cls.is_name_component_match(shorter, longer):
            return "NAME_COMPONENT"

        if shorter.lower() in longer.lower():
            if len(shorter) / len(longer) > 0.85:
                return "IDENTICAL"  # e.g. singular vs plural
            return "GENERIC_CONTAIN"

        return "NO_CONTAIN"

    @classmethod
    def get_containment_multiplier(cls, classification: str) -> float:
        return {
            "IDENTICAL": 1.0,
            "NAME_COMPONENT": NodeMatchConfig.NAME_COMPONENT_BOOST,
            "POSSESSIVE": NodeMatchConfig.POSSESSIVE_PENALTY,
            "GENERIC_CONTAIN": NodeMatchConfig.GENERIC_CONTAINMENT_PENALTY,
            "NO_CONTAIN": 1.0,  # left to the embedding to judge
        }.get(classification, 1.0)

    @classmethod
    def character_structural_similarity(
        cls, char_a: str, char_b: str, embedding_sim: float
    ) -> float:
        """Compare the Name part when the [Role] ([Name]) format is present,
        otherwise fall back to containment classification + embedding."""
        parsed_a = cls.parse_character_name(char_a)
        parsed_b = cls.parse_character_name(char_b)

        name_a = parsed_a["name"]
        name_b = parsed_b["name"]
        role_a = parsed_a["role"]
        role_b = parsed_b["role"]

        if name_a and name_b:
            name_rel = cls.classify_containment(name_a, name_b)

            if name_rel == "IDENTICAL":
                return max(embedding_sim, 0.95)

            elif name_rel == "NAME_COMPONENT":
                return max(embedding_sim, 0.90)

            elif name_rel == "POSSESSIVE":
                return 0.0

            else:
                if NodeMatchConfig.CHAR_NAME_MISMATCH_KILL:
                    return 0.0
                return embedding_sim * 0.2

        # one side has a Name, the other does not, e.g. "Male Lead (Gu Han)" vs "Gu Han"
        if name_a and not name_b:
            rel = cls.classify_containment(role_b, name_a)
            if rel in ("IDENTICAL", "NAME_COMPONENT"):
                return max(embedding_sim, 0.90)

            rel_role = cls.classify_containment(role_b, role_a)
            if rel_role == "IDENTICAL":
                return max(embedding_sim, 0.85)

            containment = cls.classify_containment(char_a, char_b)
            multiplier = cls.get_containment_multiplier(containment)
            return embedding_sim * multiplier

        if name_b and not name_a:
            return cls.character_structural_similarity(char_b, char_a, embedding_sim)

        # e.g. "Chloe" vs "Chloe's mother", "Peter" vs "Peter Anderson"
        containment = cls.classify_containment(char_a, char_b)
        multiplier = cls.get_containment_multiplier(containment)
        return embedding_sim * multiplier

    @classmethod
    def compute_adjusted_similarity(
        cls, node_type: str, text_a: str, text_b: str, embedding_sim: float
    ) -> float:
        if node_type == "Character":
            return cls.character_structural_similarity(text_a, text_b, embedding_sim)
        else:
            # Prop / Scene
            containment = cls.classify_containment(text_a, text_b)
            multiplier = cls.get_containment_multiplier(containment)
            return embedding_sim * multiplier


class EventNeighborhood:
    @staticmethod
    def extract_event_neighborhoods(
        graph_dict: Dict,
    ) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """
        Returns:
            {
                event_node_id: {
                    "text": "Estelle declares ...",
                    "type": "Event",
                    "neighbors": {
                        "AGENT_OF": [{"type": "Character", "text": "Estelle"}],
                        "TARGET_OF": [{"type": "Character", "text": "Gabriel"}],
                        "LOCATED_IN": [{"type": "Scene", "text": "Banquet Hall"}],
                        "HAS_PROP": [],
                    }
                }
            }
        """
        # Lazy import to avoid circular dependencies
        from evaluator import GraphEvaluator

        nodes = GraphEvaluator.extract_nodes(graph_dict)

        event_neighborhoods = {}
        for nid, info in nodes.items():
            if info['type'] == 'Event':
                event_neighborhoods[nid] = {
                    "text": info['text'],
                    "type": info['type'],
                    "neighbors": defaultdict(list)
                }

        for edge in graph_dict.get('edges', []):
            src_id = str(edge.get('source'))
            tgt_id = str(edge.get('target'))
            rel = edge.get('relation', '').strip().upper()

            if rel in GraphMatchConfig.STRUCTURAL_RELATIONS:
                continue  # skip NEXT_EVENT

            if src_id not in nodes or tgt_id not in nodes:
                continue

            # AGENT_OF / TARGET_OF: Character → Event, LOCATED_IN: Event → Scene;
            # HAS_PROP direction depends on how the graph was built
            if tgt_id in event_neighborhoods and nodes[src_id]['type'] != 'Event':
                # src (non-Event) --[rel]--> tgt (Event)
                event_neighborhoods[tgt_id]["neighbors"][rel].append({
                    "type": nodes[src_id]['type'],
                    "text": nodes[src_id]['text']
                })
            elif src_id in event_neighborhoods and nodes[tgt_id]['type'] != 'Event':
                # src (Event) --[rel]--> tgt (non-Event)
                event_neighborhoods[src_id]["neighbors"][rel].append({
                    "type": nodes[tgt_id]['type'],
                    "text": nodes[tgt_id]['text']
                })

        return event_neighborhoods

    @staticmethod
    def build_event_text_to_neighborhood(
        neighborhoods: Dict[str, Dict]
    ) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """Re-index by event text, since triplets are keyed by text."""
        result = {}
        for nid, info in neighborhoods.items():
            event_text = info['text']
            result[event_text] = dict(info['neighbors'])
        return result

    @classmethod
    def compute_neighbor_similarity(
        cls,
        gt_neighbors: Dict[str, List[Dict[str, str]]],
        pred_neighbors: Dict[str, List[Dict[str, str]]],
        emb_dict: Dict[str, np.ndarray],
        config: GraphMatchConfig = GraphMatchConfig(),
    ) -> float:
        """Weighted average of per-relation matching scores, AGENT_OF weighing the most."""
        all_relations = set(list(gt_neighbors.keys()) + list(pred_neighbors.keys()))

        if not all_relations:
            return 0.5  # neutral, let text similarity dominate

        weighted_score = 0.0
        total_weight = 0.0

        for rel in GraphMatchConfig.SEMANTIC_RELATIONS:
            gt_items = gt_neighbors.get(rel, [])
            pred_items = pred_neighbors.get(rel, [])

            rel_weight = config.NEIGHBOR_ROLE_WEIGHTS.get(rel, 1.0)

            if not gt_items and not pred_items:
                continue

            total_weight += rel_weight

            if not gt_items or not pred_items:
                continue  # one side missing → no credit for this relation

            rel_score = cls._match_neighbor_group(
                gt_items, pred_items, emb_dict
            )
            weighted_score += rel_weight * rel_score

        if total_weight == 0:
            return 0.5  # neutral

        return weighted_score / total_weight

    @classmethod
    def _match_neighbor_group(
        cls,
        gt_items: List[Dict[str, str]],
        pred_items: List[Dict[str, str]],
        emb_dict: Dict[str, np.ndarray],
    ) -> float:
        """Optimal matching within one relation group, e.g. GT=[Estelle, Gabriel]
        vs Pred=[Blonde Woman, Gabriel]."""
        M, N = len(gt_items), len(pred_items)

        valid_gt = [(item, emb_dict[item['text']])
                    for item in gt_items if item['text'] in emb_dict]
        valid_pred = [(item, emb_dict[item['text']])
                      for item in pred_items if item['text'] in emb_dict]

        if not valid_gt or not valid_pred:
            return 0.0

        M, N = len(valid_gt), len(valid_pred)

        gt_vecs = np.array([v[1] for v in valid_gt])
        pred_vecs = np.array([v[1] for v in valid_pred])
        raw_sim = gt_vecs @ pred_vecs.T

        adjusted_sim = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                gt_type = valid_gt[i][0]['type']
                pred_type = valid_pred[j][0]['type']

                if gt_type != pred_type:
                    adjusted_sim[i, j] = 0.0
                else:
                    adjusted_sim[i, j] = NodeMatcher.compute_adjusted_similarity(
                        gt_type,
                        valid_gt[i][0]['text'],
                        valid_pred[j][0]['text'],
                        float(raw_sim[i, j])
                    )

        # Hungarian matching
        cost_matrix = 1.0 - adjusted_sim
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_scores = []
        for r, c in zip(row_ind, col_ind):
            matched_scores.append(adjusted_sim[r, c])

        # max rather than min, so extra unmatched items lower the score
        return sum(matched_scores) / max(M, N)

    @classmethod
    def compute_event_similarity(
        cls,
        gt_text: str,
        pred_text: str,
        raw_embedding_sim: float,
        gt_event_neighbors: Dict[str, Dict[str, List[Dict[str, str]]]],
        pred_event_neighbors: Dict[str, Dict[str, List[Dict[str, str]]]],
        emb_dict: Dict[str, np.ndarray],
        config: GraphMatchConfig = GraphMatchConfig(),
    ) -> float:
        """Score = α * text_sim + β * neighbor_sim, falling back to pure text
        similarity when neither side has neighborhood info."""
        gt_nb = gt_event_neighbors.get(gt_text, {})
        pred_nb = pred_event_neighbors.get(pred_text, {})

        if not gt_nb and not pred_nb:
            return raw_embedding_sim
        
        neighbor_sim = cls.compute_neighbor_similarity(
            gt_nb, pred_nb, emb_dict, config
        )
        
        combined = (config.EVENT_TEXT_WEIGHT * raw_embedding_sim + 
                    config.EVENT_NEIGHBOR_WEIGHT * neighbor_sim)
        
        return combined