import re
import logging
from typing import Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class NodeMatchConfig:
    BASE_SIM_THRESHOLD: float = 0.70

    POSSESSIVE_PENALTY: float = 0.3  # different entity, e.g. "Chloe" vs "Chloe's mother"
    GENERIC_CONTAINMENT_PENALTY: float = 0.5  # plain substring containment
    NAME_COMPONENT_BOOST: float = 1.0  # same entity, e.g. "Peter" vs "Peter Anderson"

    CHAR_NAME_MISMATCH_KILL: bool = True
    CHAR_NAME_COMPONENT_THRESHOLD: float = 0.85


class GraphMatchConfig:

    TYPE_SIM_THRESHOLDS: Dict[str, float] = {
        "Character": 0.8,
        "Event": 0.65,
        "Prop": 0.7,
        "Scene": 0.7,
    }

    STRUCTURAL_SIM_THRESHOLD: float = 0.60
    USE_TRANSITIVE_ORDERING: bool = True

    EVENT_TEXT_WEIGHT: float = 0.65
    EVENT_NEIGHBOR_WEIGHT: float = 0.35
    EVENT_MATCH_THRESHOLD: float = 0.75
    EVENT_TEXT_MIN_THRESHOLD: float = 0.65
    NEIGHBOR_ROLE_WEIGHTS: Dict[str, float] = {
        "AGENT_OF": 1.5,
        "TARGET_OF": 1.2,
        "HAS_PROP": 0.5,
        "LOCATED_IN": 0.8,
    }
    
    SEMANTIC_TRIPLET_WEIGHT: float = 0.725
    STRUCTURAL_TRIPLET_WEIGHT: float = 0.275

    USE_SOFT_F1: bool = True

    STRUCTURAL_RELATIONS: set = {"NEXT_EVENT"}
    SEMANTIC_RELATIONS: set = {"AGENT_OF", "TARGET_OF", "HAS_PROP", "LOCATED_IN"}

    MAX_GRAPH_BUILD_RETRIES: int = 3

    def get_type_threshold(self, node_type: str) -> float:
        return self.TYPE_SIM_THRESHOLDS.get(node_type)


class Patterns:
    TAG_CONTENT = {
        "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE),
        "caption": re.compile(r"<caption>(.*?)</caption>", re.DOTALL | re.IGNORECASE)
    }
    FORMAT_CHECK = re.compile(
        r"<caption>.*?</caption>\s*<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL
    )


class PromptConfig:
    # Anchor to <repo>/train/system_prompts so prompts resolve regardless of CWD
    PROMPT_DIR = Path(__file__).resolve().parents[2] / "system_prompts"

    @classmethod
    def load(cls, filename: str) -> str:
        path = cls.PROMPT_DIR / filename
        try:
            return path.read_text(encoding='utf-8').strip()
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {filename}")
            return ""


PROMPTS = {
    "SUMMARY": PromptConfig.load("summary_judge_prompt.txt"),
    "GRAPH": PromptConfig.load("graph_build_prompt.txt")
}
