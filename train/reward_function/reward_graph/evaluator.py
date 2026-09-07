"""
Evaluator module: Graph evaluator and response parser.
"""
import re
import asyncio
import logging
from typing import Any, List, Dict, Union, Tuple
from collections import defaultdict

import numpy as np
import json_repair
import aiohttp
from scipy.optimize import linear_sum_assignment

from config import GraphMatchConfig, Patterns, PROMPTS
from matchers import NodeMatcher, EventNeighborhood, visualize_bipartite_matching

logger = logging.getLogger(__name__)


class ResponseParser:
    @staticmethod
    def check_format_reward(response: str) -> float:
        if not response:
            return 0.0
        return 1.0 if Patterns.FORMAT_CHECK.fullmatch(response) else 0.0

    @staticmethod
    def extract_tag_content(text: str, tag: str, default: str = "") -> str:
        if not text:
            return default
        pattern = Patterns.TAG_CONTENT.get(tag)
        if pattern:
            match = pattern.search(text)
            return match.group(1).strip() if match else default

        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else default

    @staticmethod
    def parse_score_from_tag(content: str, scale: float = 1.0) -> float:
        try:
            val_str = ResponseParser.extract_tag_content(content, "answer", default="0")
            val_clean = "".join(c for c in val_str if c.isdigit() or c == '.')
            if val_clean.count('.') > 1:
                val_clean = val_clean.rsplit('.', 1)[0]
            if not val_clean:
                return 0.0
            return round(float(val_clean) / scale, 2)
        except ValueError:
            return 0.0

    @staticmethod
    def parse_llm_score(content: str, task_type: Union[int, str]) -> Union[float, List[float]]:
        if not content:
            return 0.0
        content_lower = content.lower()
        if task_type == "OE":
            return 1.0 if "yes" in content_lower else 0.0
        elif task_type == "SUMMARY":
            return ResponseParser.parse_score_from_tag(content, scale=10.0)
        return 0.0


class GraphEvaluator:

    _vllm_model_name: str = ""
    _vllm_model_lock: asyncio.Lock = None

    @classmethod
    def _match_events_with_neighborhood(
        cls,
        gt_events: List[tuple],
        pred_events: List[tuple],
        emb_dict: Dict[str, np.ndarray],
        gt_event_neighbors: Dict[str, Dict],
        pred_event_neighbors: Dict[str, Dict],
        config: GraphMatchConfig = GraphMatchConfig(),
    ) -> Dict[tuple, tuple]:
        """Score(E_gt, E_pred) = α * text_sim + (1-α) * neighbor_sim.

        gt_event_neighbors / pred_event_neighbors: {event_text: {rel: [neighbors]}}
        Returns {gt_event: pred_event}.
        """
        if not gt_events or not pred_events:
            return {}

        M, N = len(gt_events), len(pred_events)

        gt_vecs = np.array([emb_dict[e[1]] for e in gt_events])
        pred_vecs = np.array([emb_dict[e[1]] for e in pred_events])
        text_sim_matrix = gt_vecs @ pred_vecs.T

        combined_matrix = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                combined_matrix[i, j] = cls._compute_node_similarity(
                    "Event",                    # both are Event
                    gt_events[i][1],            # gt text
                    pred_events[j][1],          # pred text
                    float(text_sim_matrix[i, j]),
                    gt_event_neighbors,
                    pred_event_neighbors,
                    emb_dict, config
                )

        INVALID_COST = 1e5
        cost_matrix = np.full((M, N), INVALID_COST)
        for i in range(M):
            for j in range(N):
                if (text_sim_matrix[i, j] >= config.EVENT_TEXT_MIN_THRESHOLD and
                    combined_matrix[i, j] >= config.EVENT_MATCH_THRESHOLD):
                    cost_matrix[i, j] = 1.0 - combined_matrix[i, j]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        gt_to_pred = {}
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < INVALID_COST - 1:
                gt_to_pred[gt_events[r]] = pred_events[c]
                logger.debug(
                    f"  [Event Match] '{gt_events[r][1][:50]}' ↔ "
                    f"'{pred_events[c][1][:50]}'\n"
                    f"    text={text_sim_matrix[r,c]:.3f}, "
                    f"combined={combined_matrix[r,c]:.3f}"
                )
        return gt_to_pred

    @staticmethod
    def parse_graph_input(graph_input: Any) -> Dict:
        """Accepts a dict, or a JSON string possibly wrapped in a code fence."""
        if isinstance(graph_input, dict):
            return graph_input
        if isinstance(graph_input, str):
            try:
                clean_str = graph_input.strip()
                if "```" in clean_str:
                    match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_str, re.DOTALL)
                    if match:
                        clean_str = match.group(1)
                return json_repair.loads(clean_str)
            except Exception:
                return {}
        return {}

    @staticmethod
    def check_graph_format(graph_input: Any) -> bool:
        """Validate the basic format correctness of the graph"""
        if not isinstance(graph_input, dict):
            return False

        nodes = graph_input.get("nodes")
        edges = graph_input.get("edges")

        if not isinstance(nodes, list) or not isinstance(edges, list):
            return False

        if len(nodes) == 0:
            return False

        try:
            valid_node_ids = set()
            for n in nodes:
                if not isinstance(n, dict) or "id" not in n or "type" not in n:
                    return False
                nid = str(n["id"])
                if nid in valid_node_ids:
                    return False  # duplicate id
                valid_node_ids.add(nid)

            required_keys = {"source", "relation", "target"}
            valid_relations = GraphMatchConfig.SEMANTIC_RELATIONS | GraphMatchConfig.STRUCTURAL_RELATIONS

            for e in edges:
                if not isinstance(e, dict):
                    return False
                if not required_keys.issubset(e.keys()):
                    return False
                if str(e["source"]) not in valid_node_ids or str(e["target"]) not in valid_node_ids:
                    return False
                # Validate relation legality (lenient: warn only, do not reject)
                rel = e.get("relation", "").strip().upper()
                if rel not in valid_relations:
                    logger.debug(f"Unknown relation type: {rel}")

            return True
        except Exception as e:
            logger.info(f"Graph format check exception: {e}")
            return False

    @staticmethod
    def extract_nodes(graph_dict: Dict) -> Dict[str, Dict]:
        """Returns {node_id: {type, text}}; Event nodes prefer semantic_context."""
        nodes = {}
        for n in graph_dict.get('nodes', []):
            n_id = str(n.get('id'))
            n_type = n.get('type', 'Unknown').strip()

            if n_type == 'Event' and n.get('semantic_context'):
                n_text = n.get('semantic_context').strip()
            else:
                n_text = n.get('name', '').strip()

            if n_text:
                nodes[n_id] = {'text': n_text, 'type': n_type}
        return nodes

    @staticmethod
    def extract_triplets_by_category(graph_dict: Dict) -> Tuple[List[tuple], List[tuple]]:
        """Returns (semantic_triplets, structural_triplets), i.e. AGENT_OF/TARGET_OF/
        HAS_PROP/LOCATED_IN vs NEXT_EVENT."""
        if not isinstance(graph_dict, dict):
            return [], []

        nodes = GraphEvaluator.extract_nodes(graph_dict)

        semantic_triplets = []
        structural_triplets = []

        for edge in graph_dict.get('edges', []):
            src_id = str(edge.get('source'))
            tgt_id = str(edge.get('target'))
            rel = edge.get('relation', "").strip().upper()

            if src_id in nodes and tgt_id in nodes and rel:
                src_node = nodes[src_id]
                tgt_node = nodes[tgt_id]
                triplet = (
                    (src_node['type'], src_node['text']),
                    rel,
                    (tgt_node['type'], tgt_node['text'])
                )

                if rel in GraphMatchConfig.STRUCTURAL_RELATIONS:
                    structural_triplets.append(triplet)
                else:
                    semantic_triplets.append(triplet)

        return semantic_triplets, structural_triplets

    @staticmethod
    def extract_typed_nodes(graph_dict: Dict) -> Dict[str, List[str]]:
        """Node texts grouped by type: {"Character": ["Male Lead (Gu Han)", ...], ...}"""
        nodes = GraphEvaluator.extract_nodes(graph_dict)
        typed = defaultdict(list)
        for nid, info in nodes.items():
            typed[info['type']].append(info['text'])
        return dict(typed)

    @classmethod
    async def _get_vllm_model_name(
        cls, session: aiohttp.ClientSession, base_url: str
    ) -> str:
        """First model name from the vLLM /v1/models endpoint, cached on the class."""
        if cls._vllm_model_name:
            return cls._vllm_model_name

        # asyncio.Lock must be created inside a running event loop
        if cls._vllm_model_lock is None:
            cls._vllm_model_lock = asyncio.Lock()

        async with cls._vllm_model_lock:
            # another coroutine may have populated it while we waited for the lock
            if cls._vllm_model_name:
                return cls._vllm_model_name
            try:
                async with session.get(f"{base_url}/v1/models", timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("data", [])
                        if models:
                            cls._vllm_model_name = models[0]["id"]
                            logger.info(f"[vLLM] Auto-detected model: {cls._vllm_model_name}")
                            return cls._vllm_model_name
                    logger.error(f"[vLLM] /v1/models returned status {resp.status}: {await resp.text()}")
            except Exception as e:
                logger.error(f"[vLLM] Failed to fetch model name: {e}")
            return ""

    @classmethod
    async def _cal_embedding_vllm(
        cls, session: aiohttp.ClientSession, texts: List[str],
        embedding_ip: str, embedding_port: str
    ) -> np.ndarray:
        """Embeddings via the vLLM OpenAI-compatible endpoint."""
        base_url = f"http://{embedding_ip}:{embedding_port}"
        url = f"{base_url}/v1/embeddings"
        headers = {'Content-Type': 'application/json'}
        model = await cls._get_vllm_model_name(session, base_url)
        if not model:
            logger.error("[vLLM] Cannot get model name, aborting embedding request.")
            return np.empty((0, 0))
        payload = {'model': model, 'input': texts}

        for attempt in range(5):
            try:
                async with session.post(url, headers=headers, json=payload, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"[vLLM] Embedding API Error (attempt {attempt}): {await response.text()}")
                        continue

                    data = await response.json()
                    embeddings = [item['embedding'] for item in data.get('data', [])]
                    if len(embeddings) != len(texts):
                        logger.error(f"[vLLM] Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}")
                        continue
                    return np.array(embeddings, dtype=np.float32)
            except asyncio.TimeoutError:
                logger.warning(f"[vLLM] Embedding timeout (attempt {attempt})")
            except Exception as e:
                logger.error(f"[vLLM] Embedding Request Failed (attempt {attempt}): {e}")

            if attempt < 4:
                await asyncio.sleep(min(0.5 * (2 ** attempt), 5))

        return np.empty((0, 0))

    @classmethod
    async def cal_embedding(
        cls, session: aiohttp.ClientSession, texts: List[str],
        embedding_ip: str = "127.0.0.1", embedding_port: str = "8000"
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0))

        return await cls._cal_embedding_vllm(session, texts, embedding_ip, embedding_port)

    @classmethod
    def _build_embedding_dict(cls, unique_texts: List[str], embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """{text: L2-normalized vector}"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = embeddings / norms
        return {text: vec for text, vec in zip(unique_texts, normed)}

    @classmethod
    def _compute_semantic_triplet_f1(
        cls,
        gt_triplets: List[tuple],
        pred_triplets: List[tuple],
        emb_dict: Dict[str, np.ndarray],
        gt_event_neighbors: Dict[str, Dict],
        pred_event_neighbors: Dict[str, Dict], 
        config: GraphMatchConfig = GraphMatchConfig(),
    ) -> Tuple[float, List[Tuple[int, int, float]]]:
        """Head/tail similarities are NodeMatcher-adjusted and thresholded per node type.

        Returns f1_score, matched_pairs [(gt_idx, pred_idx, score), ...].
        """
        M, N = len(gt_triplets), len(pred_triplets)
        if M == 0 and N == 0:
            return 1.0, []
        if M == 0 or N == 0:
            return 0.0, []

        gt_heads = [t[0][1] for t in gt_triplets]
        gt_tails = [t[2][1] for t in gt_triplets]
        pred_heads = [t[0][1] for t in pred_triplets]
        pred_tails = [t[2][1] for t in pred_triplets]

        gt_head_vecs = np.array([emb_dict[h] for h in gt_heads])
        pred_head_vecs = np.array([emb_dict[h] for h in pred_heads])
        head_raw_sim = gt_head_vecs @ pred_head_vecs.T

        gt_tail_vecs = np.array([emb_dict[t] for t in gt_tails])
        pred_tail_vecs = np.array([emb_dict[t] for t in pred_tails])
        tail_raw_sim = gt_tail_vecs @ pred_tail_vecs.T

        INVALID_COST = 1e5
        cost_matrix = np.full((M, N), INVALID_COST)
        score_matrix = np.zeros((M, N))

        for i in range(M):
            h_gt_type, h_gt_text = gt_triplets[i][0]
            r_gt = gt_triplets[i][1]
            t_gt_type, t_gt_text = gt_triplets[i][2]

            for j in range(N):
                h_pred_type, h_pred_text = pred_triplets[j][0]
                r_pred = pred_triplets[j][1]
                t_pred_type, t_pred_text = pred_triplets[j][2]

                if r_gt != r_pred:
                    continue
                if h_gt_type != h_pred_type or t_gt_type != t_pred_type:
                    continue

                adj_sim_h = cls._compute_node_similarity(
                    h_gt_type, h_gt_text, h_pred_text,
                    float(head_raw_sim[i, j]),
                    gt_event_neighbors, pred_event_neighbors,
                    emb_dict, config
                )
                adj_sim_t = cls._compute_node_similarity(
                    t_gt_type, t_gt_text, t_pred_text,
                    float(tail_raw_sim[i, j]),
                    gt_event_neighbors, pred_event_neighbors,
                    emb_dict, config
                )

                h_threshold = config.get_type_threshold(h_gt_type)
                t_threshold = config.get_type_threshold(t_gt_type)

                if adj_sim_h >= h_threshold and adj_sim_t >= t_threshold:
                    score = (0.5 * adj_sim_h + 0.5 * adj_sim_t) if config.USE_SOFT_F1 else 1.0
                    score_matrix[i, j] = score
                    cost_matrix[i, j] = 1.0 - score

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs = []
        total_matched_score = 0.0

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < INVALID_COST - 1:
                s = score_matrix[r, c]
                matched_pairs.append((r, c, s))
                total_matched_score += s

        if config.USE_SOFT_F1:
            precision = total_matched_score / N if N > 0 else 0.0
            recall = total_matched_score / M if M > 0 else 0.0
        else:
            precision = len(matched_pairs) / N if N > 0 else 0.0
            recall = len(matched_pairs) / M if M > 0 else 0.0

        f1 = 0.0
        if precision + recall > 0:
            f1 = round(2 * precision * recall / (precision + recall), 4)

        return f1, matched_pairs

    @classmethod
    def _extract_ordering_pairs(
        cls, structural_triplets: List[tuple]
    ) -> List[Tuple[tuple, str, tuple]]:
        """Transitive closure of NEXT_EVENT edges, as all implied (A, BEFORE, B) pairs.

        A→B→C→D yields (A,B), (A,C), (A,D), (B,C), (B,D), (C,D).
        """
        if not structural_triplets:
            return []

        adj = defaultdict(set)
        all_nodes = set()

        for (h_type, h_text), rel, (t_type, t_text) in structural_triplets:
            h_node = (h_type, h_text)
            t_node = (t_type, t_text)
            adj[h_node].add(t_node)
            all_nodes.add(h_node)
            all_nodes.add(t_node)

        ordering_pairs = []
        for start_node in all_nodes:
            visited = {start_node}
            queue = list(adj.get(start_node, set()))
            reachable = set()

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                reachable.add(current)
                for neighbor in adj.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)

            for end_node in reachable:
                ordering_pairs.append((start_node, "BEFORE", end_node))

        return ordering_pairs

    @classmethod
    def _match_events_across_graphs(
        cls,
        gt_events: List[tuple],
        pred_events: List[tuple],
        emb_dict: Dict[str, np.ndarray],
        sim_threshold: float = 0.60
    ) -> Dict[tuple, tuple]:
        """Returns {gt_event_node: pred_event_node} for the given (type, text) events."""
        if not gt_events or not pred_events:
            return {}

        M, N = len(gt_events), len(pred_events)

        gt_vecs = np.array([emb_dict[e[1]] for e in gt_events])
        pred_vecs = np.array([emb_dict[e[1]] for e in pred_events])
        sim_matrix = gt_vecs @ pred_vecs.T

        adjusted_sim = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                adjusted_sim[i, j] = NodeMatcher.compute_adjusted_similarity(
                    gt_events[i][0],  # type
                    gt_events[i][1],  # text
                    pred_events[j][1],  # text
                    float(sim_matrix[i, j])
                )

        INVALID_COST = 1e5
        cost_matrix = np.where(
            adjusted_sim >= sim_threshold,
            1.0 - adjusted_sim,
            INVALID_COST
        )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        gt_to_pred = {}
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < INVALID_COST - 1:
                gt_to_pred[gt_events[r]] = pred_events[c]
                logger.debug(
                    f"  [Event Match] '{gt_events[r][1][:40]}' ↔ "
                    f"'{pred_events[c][1][:40]}' (sim={adjusted_sim[r,c]:.3f})"
                )

        return gt_to_pred

    @classmethod
    def _compute_structural_f1_transitive(
        cls,
        gt_struct_triplets: List[tuple],
        pred_struct_triplets: List[tuple],
        emb_dict: Dict[str, np.ndarray],
        gt_event_neighbors: Dict[str, Dict],
        pred_event_neighbors: Dict[str, Dict],
        config: GraphMatchConfig = GraphMatchConfig(),
    ) -> Tuple[float, Dict]:
        """
        Structural scoring over transitive ordering pairs + neighborhood fingerprints.

        Returns:
            f1_score, debug_info dict
        """
        gt_ordering = cls._extract_ordering_pairs(gt_struct_triplets)
        pred_ordering = cls._extract_ordering_pairs(pred_struct_triplets)

        n_gt_pairs = len(gt_ordering)
        n_pred_pairs = len(pred_ordering)

        if n_gt_pairs == 0 and n_pred_pairs == 0:
            return 1.0, {"gt_pairs": 0, "pred_pairs": 0, "matched": 0}
        if n_gt_pairs == 0 or n_pred_pairs == 0:
            return 0.0, {"gt_pairs": n_gt_pairs, "pred_pairs": n_pred_pairs, "matched": 0}

        gt_event_set = set()
        pred_event_set = set()
        for (h, rel, t) in gt_ordering:
            gt_event_set.add(h)
            gt_event_set.add(t)
        for (h, rel, t) in pred_ordering:
            pred_event_set.add(h)
            pred_event_set.add(t)

        gt_events = list(gt_event_set)
        pred_events = list(pred_event_set)

        gt_to_pred = cls._match_events_with_neighborhood(
            gt_events, pred_events, emb_dict,
            gt_event_neighbors, pred_event_neighbors,
            config=config
        )

        if not gt_to_pred:
            return 0.0, {
                "gt_pairs": n_gt_pairs, "pred_pairs": n_pred_pairs,
                "matched": 0, "event_matches": 0
            }

        pred_to_gt = {v: k for k, v in gt_to_pred.items()}

        pred_ordering_set = set()
        for (h, rel, t) in pred_ordering:
            pred_ordering_set.add((h, t))

        gt_ordering_set = set()
        for (h, rel, t) in gt_ordering:
            gt_ordering_set.add((h, t))

        # recall: how many GT ordering pairs survive in Pred
        gt_matched = 0
        gt_evaluable = 0  # GT pairs where both events have a match

        for (h_gt, rel, t_gt) in gt_ordering:
            if h_gt in gt_to_pred and t_gt in gt_to_pred:
                gt_evaluable += 1
                h_pred = gt_to_pred[h_gt]
                t_pred = gt_to_pred[t_gt]
                if (h_pred, t_pred) in pred_ordering_set:
                    gt_matched += 1

        # precision: how many Pred ordering pairs are consistent with GT
        pred_matched = 0
        pred_evaluable = 0

        for (h_pred, rel, t_pred) in pred_ordering:
            if h_pred in pred_to_gt and t_pred in pred_to_gt:
                pred_evaluable += 1
                h_gt = pred_to_gt[h_pred]
                t_gt = pred_to_gt[t_pred]
                if (h_gt, t_gt) in gt_ordering_set:
                    pred_matched += 1

        recall = gt_matched / n_gt_pairs if n_gt_pairs > 0 else 0.0
        precision = pred_matched / n_pred_pairs if n_pred_pairs > 0 else 0.0

        f1 = 0.0
        if precision + recall > 0:
            f1 = round(2 * precision * recall / (precision + recall), 4)

        debug_info = {
            "gt_pairs": n_gt_pairs,
            "pred_pairs": n_pred_pairs,
            "gt_evaluable": gt_evaluable,
            "pred_evaluable": pred_evaluable,
            "gt_matched": gt_matched,
            "pred_matched": pred_matched,
            "event_matches": len(gt_to_pred),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": f1
        }

        logger.debug(
            f"  [Structural] Event matches: {len(gt_to_pred)}, "
            f"GT pairs: {n_gt_pairs} (evaluable: {gt_evaluable}, matched: {gt_matched}), "
            f"Pred pairs: {n_pred_pairs} (evaluable: {pred_evaluable}, matched: {pred_matched}), "
            f"P={precision:.3f}, R={recall:.3f}, F1={f1:.4f}"
        )

        return f1, debug_info

    @classmethod
    def _compute_node_similarity(
        cls,
        node_type: str,
        gt_text: str,
        pred_text: str,
        raw_embedding_sim: float,
        gt_event_neighbors: Dict[str, Dict],
        pred_event_neighbors: Dict[str, Dict],
        emb_dict: Dict[str, np.ndarray],
        config: GraphMatchConfig = GraphMatchConfig(),
    ) -> float:
        """Event: text + neighborhood fingerprint. Others: NodeMatcher text matching."""
        if node_type == "Event":
            return EventNeighborhood.compute_event_similarity(
                gt_text, pred_text, raw_embedding_sim,
                gt_event_neighbors, pred_event_neighbors,
                emb_dict, config
            )
        else:
            return NodeMatcher.compute_adjusted_similarity(
                node_type, gt_text, pred_text, raw_embedding_sim
            )

    @classmethod
    def _compute_triplet_f1(
        cls,
        gt_triplets: List[tuple],
        pred_triplets: List[tuple],
        emb_dict: Dict[str, np.ndarray],
        sim_threshold: float,
        use_soft: bool = True
    ) -> Tuple[float, List[Tuple[int, int, float]]]:
        """Edge-by-edge matching, used for structural edges when USE_TRANSITIVE_ORDERING
        is off. Semantic edges use _compute_semantic_triplet_f1 instead."""
        M, N = len(gt_triplets), len(pred_triplets)
        if M == 0 and N == 0:
            return 1.0, []
        if M == 0 or N == 0:
            return 0.0, []

        gt_heads = [t[0][1] for t in gt_triplets]
        gt_tails = [t[2][1] for t in gt_triplets]
        pred_heads = [t[0][1] for t in pred_triplets]
        pred_tails = [t[2][1] for t in pred_triplets]

        gt_head_vecs = np.array([emb_dict[h] for h in gt_heads])
        pred_head_vecs = np.array([emb_dict[h] for h in pred_heads])
        head_sim_matrix = gt_head_vecs @ pred_head_vecs.T

        gt_tail_vecs = np.array([emb_dict[t] for t in gt_tails])
        pred_tail_vecs = np.array([emb_dict[t] for t in pred_tails])
        tail_sim_matrix = gt_tail_vecs @ pred_tail_vecs.T

        INVALID_COST = 1e5
        cost_matrix = np.full((M, N), INVALID_COST)
        score_matrix = np.zeros((M, N))

        for i in range(M):
            r_gt = gt_triplets[i][1]
            h_gt_type = gt_triplets[i][0][0]
            t_gt_type = gt_triplets[i][2][0]
            for j in range(N):
                r_pred = pred_triplets[j][1]
                h_pred_type = pred_triplets[j][0][0]
                t_pred_type = pred_triplets[j][2][0]

                if r_gt != r_pred:
                    continue
                if h_gt_type != h_pred_type or t_gt_type != t_pred_type:
                    continue

                sim_h = head_sim_matrix[i, j]
                sim_t = tail_sim_matrix[i, j]

                if sim_h >= sim_threshold and sim_t >= sim_threshold:
                    score = 0.5 * sim_h + 0.5 * sim_t if use_soft else 1.0
                    score_matrix[i, j] = score
                    cost_matrix[i, j] = 1.0 - score

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs = []
        total_matched_score = 0.0
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < INVALID_COST - 1:
                s = score_matrix[r, c]
                matched_pairs.append((r, c, s))
                total_matched_score += s

        if use_soft:
            precision = total_matched_score / N if N > 0 else 0.0
            recall = total_matched_score / M if M > 0 else 0.0
        else:
            precision = len(matched_pairs) / N if N > 0 else 0.0
            recall = len(matched_pairs) / M if M > 0 else 0.0

        f1 = 0.0
        if precision + recall > 0:
            f1 = round(2 * precision * recall / (precision + recall), 4)

        return f1, matched_pairs

    @classmethod
    async def calculate_graph_score_async(
        cls,
        session: aiohttp.ClientSession,
        gt_graph: Dict,
        pred_graph: Dict,
        embedding_ip: str,
        embedding_port: str,
        config: GraphMatchConfig = GraphMatchConfig(),
        idx: int = -1,
        visualize: bool = False
    ) -> float:
        """Weighted sum of semantic triplet F1 and structural ordering F1."""
        gt_sem, gt_struct = cls.extract_triplets_by_category(gt_graph)
        pred_sem, pred_struct = cls.extract_triplets_by_category(pred_graph)
        gt_typed_nodes = cls.extract_typed_nodes(gt_graph)
        pred_typed_nodes = cls.extract_typed_nodes(pred_graph)

        logger.debug(f"\n{'='*20} ID: {idx} {'='*20}")
        logger.debug(f"[GT] Semantic: {len(gt_sem)}, Structural: {len(gt_struct)}")
        logger.debug(f"[Pred] Semantic: {len(pred_sem)}, Structural: {len(pred_struct)}")

        all_gt = gt_sem + gt_struct
        all_pred = pred_sem + pred_struct
        if not all_gt and not all_pred:
            return 1.0
        if not all_gt or not all_pred:
            return 0.0

        gt_neighborhoods = EventNeighborhood.extract_event_neighborhoods(gt_graph)
        pred_neighborhoods = EventNeighborhood.extract_event_neighborhoods(pred_graph)
        gt_event_neighbors = EventNeighborhood.build_event_text_to_neighborhood(gt_neighborhoods)
        pred_event_neighbors = EventNeighborhood.build_event_text_to_neighborhood(pred_neighborhoods)

        unique_texts = set()
        for triplets in [gt_sem, gt_struct, pred_sem, pred_struct]:
            for (h_type, h_text), rel, (t_type, t_text) in triplets:
                unique_texts.add(h_text)
                unique_texts.add(t_text)
        for ntype_texts in list(gt_typed_nodes.values()) + list(pred_typed_nodes.values()):
            unique_texts.update(ntype_texts)

        for neighbors_map in [gt_event_neighbors, pred_event_neighbors]:
            for event_text, rel_neighbors in neighbors_map.items():
                unique_texts.add(event_text)
                for rel, neighbor_list in rel_neighbors.items():
                    for nb in neighbor_list:
                        unique_texts.add(nb['text'])

        unique_texts = list(unique_texts)
        if not unique_texts:
            return 0.0

        embeddings = await cls.cal_embedding(
            session, unique_texts, embedding_ip, embedding_port
        )
        if embeddings.size == 0 or len(embeddings) != len(unique_texts):
            logger.warning(f"[ID:{idx}] Embedding failed.")
            return 0.0

        emb_dict = cls._build_embedding_dict(unique_texts, embeddings)

        def compute_all_scores():
            sem_f1, sem_matches = cls._compute_semantic_triplet_f1(
                gt_sem, pred_sem, emb_dict,
                gt_event_neighbors,
                pred_event_neighbors,
                config=config
            )

            struct_debug = {}
            if config.USE_TRANSITIVE_ORDERING:
                struct_f1, struct_debug = cls._compute_structural_f1_transitive(
                    gt_struct, pred_struct, emb_dict,
                    gt_event_neighbors,
                    pred_event_neighbors,
                    config=config
                )
                struct_matches = []
            else:
                struct_f1, struct_matches = cls._compute_triplet_f1(
                    gt_struct, pred_struct, emb_dict,
                    sim_threshold=config.STRUCTURAL_SIM_THRESHOLD,
                    use_soft=config.USE_SOFT_F1
                )

            return (sem_f1, sem_matches, struct_f1, struct_matches,
                    struct_debug)

        (sem_f1, sem_matches, struct_f1, struct_matches,
         struct_debug) = await asyncio.to_thread(
            compute_all_scores
        )

        w_sem = config.SEMANTIC_TRIPLET_WEIGHT
        w_struct = config.STRUCTURAL_TRIPLET_WEIGHT

        if not gt_struct and not pred_struct:
            # no structural edges on either side, so put all weight on semantic
            w_sem += w_struct
            w_struct = 0.0

        final_score = round(
            w_sem * sem_f1 + w_struct * struct_f1, 4
        )

        logger.debug(
            f"[ID:{idx}] SemF1={sem_f1:.3f}(w={w_sem:.2f}), "
            f"StructF1={struct_f1:.3f}(w={w_struct:.2f}), "
            f"Final={final_score:.4f}"
        )

        if visualize:
            try:
                all_matches = sem_matches + [
                    (m[0] + len(gt_sem), m[1] + len(pred_sem), m[2])
                    for m in struct_matches
                ]
                visualize_bipartite_matching(
                    gt_sem + gt_struct, pred_sem + pred_struct, all_matches,
                    title=(
                        f"ID:{idx} | Sem={sem_f1:.3f} Struct={struct_f1:.3f} "
                        f"| Total={final_score:.4f}"
                    )
                )
            except Exception as e:
                logger.error(f"Visualization failed for ID {idx}: {e}")

        return final_score
