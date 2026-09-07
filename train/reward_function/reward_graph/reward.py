"""
Main entry module: Async LLM client, scoring pipeline, command-line entry.
"""

import sys
from pathlib import Path

# The training framework loads this file via importlib.spec_from_file_location,
# which has no package context, so sibling modules must be resolved via sys.path.
sys.path.append(str(Path(__file__).resolve().parent))

import asyncio
import random
import logging
from typing import Any, List, Dict, Optional, Tuple

import aiohttp
from mathruler.grader import grade_answer

from config import GraphMatchConfig, PROMPTS
from evaluator import GraphEvaluator, ResponseParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AsyncLLMClient:
    def __init__(
        self,
        llm_judge_ip: str,
        llm_judge_ports: str,
        concurrency_limit: int = 512,
        per_endpoint_sems: dict = None,
        max_per_endpoint: int = 16,
    ):
        self.llm_judge_ip = [ip.strip() for ip in llm_judge_ip.split(",")]
        port_groups = [
            [p.strip() for p in group.split(",")]
            for group in str(llm_judge_ports).split(";")
        ]

        self.endpoints = []
        if len(port_groups) == 1:
            for ip in self.llm_judge_ip:
                for port in port_groups[0]:
                    self.endpoints.append((ip, port))
        else:
            for i, ip in enumerate(self.llm_judge_ip):
                if i < len(port_groups):
                    for port in port_groups[i]:
                        self.endpoints.append((ip, port))

        self.url_template = "http://{}:{}/v1/chat/completions"

        self.sem = asyncio.Semaphore(concurrency_limit)

        if per_endpoint_sems is not None:
            self.per_endpoint_sem = per_endpoint_sems
        else:
            self.per_endpoint_sem = {
                f"{ip}:{port}": asyncio.Semaphore(max_per_endpoint)
                for ip, port in self.endpoints
            }

    async def request(
        self, session: aiohttp.ClientSession, prompt: str,
        max_tokens: int = 8192, top_p: float = 0.95, temperature: float = 0.2,
        think_mode: bool = False, json_mode: bool = False,
        max_retries: int = 3
    ) -> Optional[str]:
        async with self.sem:
            # at least one attempt per endpoint
            effective_retries = max(max_retries, len(self.endpoints) * 2)
            failed_endpoints = set()

            for attempt in range(effective_retries):
                healthy = [ep for ep in self.endpoints if f"{ep[0]}:{ep[1]}" not in failed_endpoints]
                if not healthy:
                    healthy = self.endpoints
                ip, port = random.choice(healthy)
                endpoint_key = f"{ip}:{port}"

                sleep_time = 0
                async with self.per_endpoint_sem[endpoint_key]:
                    url = self.url_template.format(ip, port)
                    payload = {
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "chat_template_kwargs": {"enable_thinking": think_mode}
                    }
                    if json_mode:
                        payload["response_format"] = {"type": "json_object"}

                    try:
                        async with session.post(
                            url, json=payload,
                            timeout=aiohttp.ClientTimeout(total=180)
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                return data['choices'][0]['message']['content']
                            elif response.status == 503:
                                sleep_time = min(2 * (2 ** attempt), 30)
                                failed_endpoints.add(endpoint_key)
                            else:
                                sleep_time = 1
                                failed_endpoints.add(endpoint_key)
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout on {ip}:{port}, retry {attempt}")
                        sleep_time = min(1 * (2 ** attempt), 15)
                        failed_endpoints.add(endpoint_key)
                    except Exception as e:
                        if attempt >= 2:
                            logger.error(f"Connection error on {ip}:{port}: {e}")
                        sleep_time = min(0.5 * (2 ** attempt), 10)
                        failed_endpoints.add(endpoint_key)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            return None


def build_judge_prompt(pred: str, gt: str, task_type: str) -> Optional[str]:
    if task_type == "OE":
        return (
            f"Please determine whether the two answers convey the same meaning. "
            f"You only need to output 'Yes' or 'No', without any other output. "
            f"\nPrediction: {pred} \nGround Truth: {gt}"
        )
    elif task_type == "SUMMARY":
        return f"{PROMPTS['SUMMARY']}\n**Ground Truth Description**\n{gt}\n**Model-Generated Summary**\n{pred}"
    elif task_type == "GRAPH":
        return f"{PROMPTS['GRAPH']}\n**Input Caption**\n{pred}"
    return None


async def process_single_item(
    idx: int,
    inp: Dict,
    session: aiohttp.ClientSession,
    judge_client: AsyncLLMClient,
    graph_client: AsyncLLMClient,
    format_weight: float,
    embedding_ip: str,
    embedding_port: str,
    result_container: List[Dict],
    graph_match_config: GraphMatchConfig = GraphMatchConfig(),
    visualize_graph: bool = False
):
    try:
        response = inp.get("response", "")
        gt_raw = inp.get("ground_truth", "")
        task_type = inp.get("task_type", "MC")

        gt_graph = GraphEvaluator.parse_graph_input(inp.get("gt_graph", "{}"))

        fmt_score = ResponseParser.check_format_reward(response)
        pred_answer = ResponseParser.extract_tag_content(response, "answer")
        gt_answer = ResponseParser.extract_tag_content(gt_raw, "answer")

        async def calc_accuracy_task() -> Tuple[float, List[float]]:
            local_acc = 0.0
            local_llm_scores = []
            try:
                if task_type == "MC":
                    local_acc = 1.0 if grade_answer(pred_answer, gt_answer) else 0.0
                elif task_type in ["OE", "SUMMARY"]:
                    prompt = build_judge_prompt(pred_answer, gt_answer, task_type)
                    if prompt:
                        content = await judge_client.request(session, prompt, 1024)
                        if content:
                            score = ResponseParser.parse_llm_score(content, task_type)
                            if isinstance(score, list):
                                local_acc = sum(score) / len(score) if score else 0.0
                                local_llm_scores.extend(score)
                            else:
                                local_acc = score
                                local_llm_scores.append(score)
            except Exception as e:
                logger.error(f"Accuracy calc failed for idx {idx}: {e}")
            return local_acc, local_llm_scores

        async def calc_graph_task() -> Tuple[float, Dict]:
            local_cap_score = 0.0
            local_pred_graph = {}

            if not PROMPTS.get('GRAPH'):
                return local_cap_score, local_pred_graph

            pred_caption = ResponseParser.extract_tag_content(response, "caption", default=response)
            prompt_pred = build_judge_prompt(pred_caption, None, "GRAPH")

            if not prompt_pred:
                return local_cap_score, local_pred_graph

            max_retries = graph_match_config.MAX_GRAPH_BUILD_RETRIES
            for retry in range(max_retries):
                try:
                    pred_json_str = await graph_client.request(
                        session, prompt_pred,
                        max_tokens=6000, top_p=0.7, temperature=0.2
                    )
                    if not pred_json_str:
                        logger.warning(f"[ID:{idx}] Graph build returned empty (retry {retry}/{max_retries})")
                        await asyncio.sleep(min(1 * (2 ** retry), 10))
                        continue

                    local_pred_graph = GraphEvaluator.parse_graph_input(pred_json_str)

                    if not GraphEvaluator.check_graph_format(local_pred_graph):
                        logger.warning(f"[ID:{idx}] Invalid graph format (retry {retry}/{max_retries})")
                        local_pred_graph = {}
                        await asyncio.sleep(min(1 * (2 ** retry), 10))
                        continue

                    local_cap_score = await GraphEvaluator.calculate_graph_score_async(
                        session, gt_graph, local_pred_graph,
                        embedding_ip, embedding_port,
                        config=graph_match_config,
                        idx=idx, visualize=visualize_graph
                    )
                    break

                except Exception as e:
                    logger.warning(f"[ID:{idx}] Graph eval error (retry {retry}/{max_retries}): {e}")
                    if retry < max_retries - 1:
                        await asyncio.sleep(min(1 * (2 ** retry), 10))

            return local_cap_score, local_pred_graph

        (acc, llm_scores), (cap_score, pred_graph) = await asyncio.gather(
            calc_accuracy_task(),
            calc_graph_task()
        )

        qa_logs = {
            "prediction": response,
            "ground_truth": gt_raw,
            "llm_scores": llm_scores,
            "pred_graph": pred_graph,
            "gt_graph": gt_graph
        }

        result_container[idx]["qa_logs"] = qa_logs
        result_container[idx]["format"] = fmt_score
        result_container[idx]["accuracy"] = round(acc, 3)
        result_container[idx][f"accuracy_{task_type}"] = round(acc, 3)
        result_container[idx]["global_caption_rewards"] = cap_score
        result_container[idx]["overall"] = round(
            (1 - format_weight) * acc + format_weight * fmt_score, 3
        )

    except Exception as e:
        logger.error(f"Critical error processing item {idx}: {e}", exc_info=True)
        result_container[idx]["error"] = str(e)
        result_container[idx]["overall"] = 0.0


async def compute_score_async(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    llm_judge_ip: str = "localhost",
    llm_judge_port: str = "8000",
    graph_ip: str = "localhost",
    graph_port: str = "8000",
    embedding_ip: str = "127.0.0.1",
    embedding_port: str = "8080",
    concurrency_limit: int = 128,
    max_num_seqs: int = 16,
    visualize_graph: bool = False,
    graph_match_config: GraphMatchConfig = None,
    **kwargs
) -> List[Dict[str, float]]:

    if graph_match_config is None:
        graph_match_config = GraphMatchConfig()

    def _parse_endpoints(ip_str: str, port_str: str):
        ips = [ip.strip() for ip in ip_str.split(",")]
        port_groups = [
            [p.strip() for p in group.split(",")]
            for group in str(port_str).split(";")
        ]
        eps = []
        if len(port_groups) == 1:
            for ip in ips:
                for port in port_groups[0]:
                    eps.append((ip, port))
        else:
            for i, ip in enumerate(ips):
                if i < len(port_groups):
                    for port in port_groups[i]:
                        eps.append((ip, port))
        return eps

    judge_endpoints = _parse_endpoints(llm_judge_ip, str(llm_judge_port))
    graph_endpoints = _parse_endpoints(graph_ip, str(graph_port))
    all_endpoints = set(judge_endpoints) | set(graph_endpoints)

    per_endpoint_limit = max_num_seqs
    shared_sems = {
        f"{ip}:{port}": asyncio.Semaphore(per_endpoint_limit)
        for ip, port in all_endpoints
    }
    num_endpoints = len(all_endpoints)
    total_concurrency = per_endpoint_limit * num_endpoints
    logger.info(
        f"Endpoints: {num_endpoints} "
        f"(judge={len(judge_endpoints)}, graph={len(graph_endpoints)}), "
        f"per_endpoint={per_endpoint_limit}, total_concurrency={total_concurrency}"
    )

    judge_client = AsyncLLMClient(
        llm_judge_ip, llm_judge_port,
        concurrency_limit=total_concurrency,
        per_endpoint_sems=shared_sems,
    )
    graph_client = AsyncLLMClient(
        graph_ip, graph_port,
        concurrency_limit=total_concurrency,
        per_endpoint_sems=shared_sems,
    )

    scores = [{} for _ in reward_inputs]

    connector = aiohttp.TCPConnector(
        limit=total_concurrency + 50, ttl_dns_cache=300
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_single_item(
                i, inp, session, judge_client, graph_client,
                format_weight, embedding_ip, embedding_port, scores,
                graph_match_config=graph_match_config,
                visualize_graph=visualize_graph
            )
            for i, inp in enumerate(reward_inputs)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    return scores


def compute_score(reward_inputs, **kwargs):
    return asyncio.run(compute_score_async(reward_inputs, **kwargs))