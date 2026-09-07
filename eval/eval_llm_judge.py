"""
Unified LLM-as-a-Judge script for drama evaluation, supporting both QA and
Summary tasks.

- QA task: multiple-choice questions are scored by rule-based choice extraction;
  open-ended questions are judged by an LLM (judge prompt concatenated into the
  user message).
- Summary task: the model-generated summary is compared against the ground-truth
  caption by an LLM (judge prompt placed in the system message).

The script talks to any OpenAI-compatible chat API (OpenAI, or self-hosted
gateways via --base_url / OPENAI_BASE_URL). Credentials are read from
OPENAI_API_KEY (or passed via --api_key).

Usage examples:
    # QA judge
    python eval/eval_llm_judge.py --task qa \
        --result_path results/qa/model.json \
        --output_path judge/qa/model_oe.json \
        --caption_csv eval/data/test_qa_captions.csv \
        --task_type OE

    # Summary judge
    python eval/eval_llm_judge.py --task summary \
        --result_path results/summary/model.json \
        --output_path judge/summary/model.json \
        --model gpt-5-mini
"""
import os
import os.path as osp
import json
import re
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import pandas as pd
from openai import OpenAI

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
PROMPT_DIR = osp.join(SCRIPT_DIR, "prompts")

# Marker written to the output file for samples whose judge request failed
FAILED_MARKER = "FAILED"

ans_pat = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
cap_pat = re.compile(r"<caption>(.*?)</caption>", re.DOTALL | re.IGNORECASE)

# Judge prompt file for each task
TASK_PROMPT_FILE = {
    "qa": "oe_judge_prompt.txt",
    "summary": "summary_judge_prompt.txt",
}

# Pricing table for cost estimation, in dollars per 1M tokens.
# Unknown models are estimated as free; edit this table for your own models.
model2cost = {
    "deepseek-v3.2": {
        "in": 2,
        "out": 3,
    },
    "gpt-5-mini": {
        "in": 1.75,
        "out": 14,
    },
    "gemini-2.5-flash": {
        "in": 2.1,
        "out": 17.5,
    },
}

# Runtime globals, set by main()
SYSTEM_PROMPT = None
CAPTION_DATA = None  # only used by the QA task: DataFrame with qid/caption columns


def load_prompt(task):
    path = osp.join(PROMPT_DIR, TASK_PROMPT_FILE[task])
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"Judge prompt file not found: {path}. "
            f"Expected one of the prompt files under {PROMPT_DIR}."
        )
    with open(path, encoding='utf-8') as f:
        return f.read()


def extract_choice(pred_text, valid_choices=None):
    """
    Extract the choice letter (A/B/C/D) from a model prediction.
    Supports multiple formats: bare letters, \\boxed{X}, <answer>X</answer>,
    "answer is X", single-letter endings after long reasoning, etc.
    """
    if valid_choices is None:
        valid_choices = ['A', 'B', 'C', 'D']

    if not pred_text or not pred_text.strip():
        return None

    pred_text = pred_text.strip()

    # 1. Strip the thinking block (keep content after </think>)
    if '</think>' in pred_text:
        pred_text = pred_text.split('</think>')[-1].strip()

    # 2. Strip the <analysis> block
    if '</analysis>' in pred_text:
        pred_text = pred_text.split('</analysis>')[-1].strip()

    # 3. \boxed{X} format
    boxed_match = re.search(r'\\boxed\{\s*([A-Da-d])\s*\}', pred_text)
    if boxed_match:
        choice = boxed_match.group(1).upper()
        if choice in valid_choices:
            return choice

    # 4. <answer>X</answer> format
    answer_tag_match = re.search(r'<answer>\s*([A-Da-d])\s*</answer>', pred_text, re.IGNORECASE)
    if answer_tag_match:
        choice = answer_tag_match.group(1).upper()
        if choice in valid_choices:
            return choice

    # 5. "The answer is X" / "X is correct" patterns
    answer_match = re.search(r'(?:The\s+)?(?:correct\s+)?answer\s+is[:\s]*\[?([A-Da-d])\]?', pred_text, re.IGNORECASE)
    if answer_match:
        choice = answer_match.group(1).upper()
        if choice in valid_choices:
            return choice

    # 5.5 Chinese answer patterns, e.g. "answer:B" or "the answer is B" in Chinese
    cn_answer_match = re.search(r'答案[是:：]\s*\[?([A-Da-d])\]?', pred_text)
    if cn_answer_match:
        choice = cn_answer_match.group(1).upper()
        if choice in valid_choices:
            return choice

    # 6. Bracket format: [A. xxx]
    bracket_match = re.match(r'^\[([A-Da-d])[.\s]', pred_text)
    if bracket_match:
        choice = bracket_match.group(1).upper()
        if choice in valid_choices:
            return choice

    # 7. Starts with a choice letter: "A", "A.", "A. xxx", "A," (also matches CJK separators)
    start_match = re.match(r'^([A-Da-d])\s*[.．、：:)\]]\s*', pred_text)
    if start_match:
        choice = start_match.group(1).upper()
        if choice in valid_choices:
            return choice

    # 8. Bare single letter
    if pred_text.upper() in valid_choices:
        return pred_text.upper()

    # 9. Thinking models: scan lines backwards for a letter-leading or single-letter line
    lines = pred_text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        line_match = re.match(r'^([A-Da-d])\s*[.．、：:)\]]\s*', line)
        if line_match:
            choice = line_match.group(1).upper()
            if choice in valid_choices:
                return choice
        if line.upper() in valid_choices:
            return line.upper()

    # 10. Last resort: "choose A" "is A" "option A" (also matches the Chinese verb for "choose")
    last_resort_match = re.search(r'(?:选|choose|is|option|answer)\s*([A-Da-d])', pred_text, re.IGNORECASE)
    if last_resort_match:
        choice = last_resort_match.group(1).upper()
        if choice in valid_choices:
            return choice

    return None


def chat(client, model, user_prompt, task):
    """
    Call an OpenAI-compatible chat completion API.
    - qa task: the judge prompt is concatenated into the user message
    - summary task: the judge prompt is placed in the system message
    Returns (response_text, cost).
    """
    if task == "qa":
        messages = [
            {
                "role": "user",
                "content": SYSTEM_PROMPT + "\n\n" + user_prompt,
            },
        ]
    else:  # summary
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

    ret = None
    for i in range(5):
        try:
            ret = client.chat.completions.create(model=model, messages=messages)
        except Exception as e:
            print(f"Request failed, retry {i + 1}/5: {e}")
            time.sleep(5)
            continue
        pricing = model2cost.get(model, {"in": 0, "out": 0})
        cost = (ret.usage.completion_tokens * pricing['out'] + ret.usage.prompt_tokens * pricing['in']) / 1e6
        return ret.choices[0].message.content, cost

    raise Exception("Request still failing after 5 retries" + (f": {ret}" if ret else ""))


def process_item_qa(item, client, model):
    """
    QA task: process a single question. MC questions are scored by rules,
    OE questions are judged by the LLM.
    """
    qid = item['qid']
    data = CAPTION_DATA[CAPTION_DATA['qid'] == qid]
    gt_answer = item['gt']
    caption = data['caption'].item()
    prediction = item['pred']
    question = item['question']
    match = ans_pat.search(prediction)
    try:
        pred_ans = match.group(1).strip()
    except Exception:
        pred_ans = prediction
    q_type = item['q_type']

    if q_type == "MC":
        # Extract the choice letter from the full prediction with extract_choice
        # (does not rely on the <answer> tag; supports single-letter endings
        # after long reasoning, etc.)
        extracted = extract_choice(prediction)
        score = 1.0 if extracted == gt_answer else 0.0
        return score, 0
    elif q_type == "OE":
        gt_cap = cap_pat.search(caption).group(1).strip()
        if gt_answer == pred_ans:
            return 1.0, 0
        user_prompt = f"**Video Caption**\n{gt_cap}\n\n**Question**\n{question}\n\n**Ground Truth**\n{gt_answer}\n\n**Prediction**\n{pred_ans}"
        try:
            response_text, cost = chat(client, model, user_prompt, "qa")
            return response_text, cost
        except Exception as e:
            print(f"Error while processing question {qid}: {e}")
            return None, 0.0


def process_item_summary(item, client, model):
    """
    Summary task: process a single video, sending the model summary and the GT
    caption to the LLM judge.
    """
    vid = item['vid']
    gt_answer = item['gt']
    prediction = item['pred']
    cap_match = cap_pat.search(gt_answer)
    if not cap_match:
        print(f"Skipping {vid}: gt is missing the <caption> tag")
        return None, 0.0
    # Fall back to the full pred text when the <answer> tag is absent
    # (summary outputs are usually plain-text paragraphs)
    ans_match = ans_pat.search(prediction)
    pred_ans = ans_match.group(1).strip() if ans_match else prediction.strip()
    gt_cap = cap_match.group(1).strip()

    user_prompt = f"**Video Caption**\n{gt_cap}\n\n**Model-Generated Summary**\n{pred_ans}"
    try:
        response_text, cost = chat(client, model, user_prompt, "summary")
        return response_text, cost
    except Exception as e:
        print(f"Error while processing video {vid}: {e}")
        return None, 0.0


PROCESS_ITEM = {
    "qa": process_item_qa,
    "summary": process_item_summary,
}


def get_output_path(output_path, model):
    """Generate a dedicated output file per judge model: xxx.json -> xxx_{model}.json"""
    base, ext = osp.splitext(output_path)
    return f"{base}_{model}{ext or '.json'}"


def run_task(args, client, model):
    process_item = PROCESS_ITEM[args.task]
    result_file = json.load(open(args.result_path))
    if isinstance(result_file, dict):
        result_file = result_file['details']

    output_path = get_output_path(args.output_path, model)

    # Resume support: skip already completed samples
    exists_results = []
    if osp.exists(output_path):
        exists_results = json.load(open(output_path))
        exists_results = [i for i in exists_results if i['response'] != FAILED_MARKER]
        if args.task == "qa":
            done_keys = [i['qid'] + i['pred'] for i in exists_results]
            result_file = [i for i in result_file if i['qid'] + i['pred'] not in done_keys]
        else:
            done_keys = [i['vid'] + i['pred'] for i in exists_results]
            result_file = [i for i in result_file if i['vid'] + i['pred'] not in done_keys]

    # The QA task can filter by question type
    if args.task == "qa":
        results = []
        if "OE" in args.task_type:
            results += [i for i in result_file if i['q_type'] == "OE"]
        if "MC" in args.task_type:
            results += [i for i in result_file if i['q_type'] == "MC"]
    else:
        results = list(result_file)
    random.shuffle(results)

    new_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        all_future = {executor.submit(process_item, it, client, model): it for it in results}
        for future in tqdm(as_completed(all_future), total=len(results), desc=f"LLM judging ({model})"):
            try:
                score, cost = future.result()
                item = all_future[future]
                if score is not None:
                    item['response'] = score
                else:
                    item['response'] = FAILED_MARKER
                item['cost'] = cost
                item['reviewer'] = model
                new_results.append(item)

                # Periodically flush results to disk to support resuming
                if len(new_results) % 10 == 0:
                    with open(output_path, "w", encoding='utf-8') as f:
                        json.dump(new_results + exists_results, f, ensure_ascii=False, indent=4)
            except Exception as exc:
                item = all_future[future]
                key = item.get('qid', item.get('vid'))
                print(f"Severe error while processing {key}: {exc}")

    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(new_results + exists_results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to: {output_path}")

    return sum(i['cost'] for i in new_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified LLM-as-a-Judge script for drama QA / Summary evaluation")
    parser.add_argument('--task', type=str, required=True, choices=['qa', 'summary'],
                        help="Evaluation task type: qa or summary")
    parser.add_argument('--result_path', type=str, default="results/qa/result.json",
                        help="Inference output of eval_drama.py (a list or {'details': [...]})")
    parser.add_argument('--output_path', type=str, default="judge/qa/result.json",
                        help="Judge output path (a _{model} suffix is appended, one file per judge model)")
    parser.add_argument('--model', type=str, nargs='+', default=["gpt-5-mini"],
                        help="One or more judge model names (OpenAI-compatible)")
    parser.add_argument('--task_type', type=str, default="OE",
                        help="QA task: question types to judge (OE/MC, combinable like 'OEMC'); ignored by summary")
    parser.add_argument('--caption_csv', type=str, default="eval/data/test_qa_captions.csv",
                        help="QA task: CSV with qid/caption columns used as Video Caption for the OE judge "
                             "(produced by eval/prepare_eval_data.py); ignored by summary")
    parser.add_argument('--max_workers', type=int, default=10,
                        help="Maximum number of concurrent threads, tune to your API rate limits")
    parser.add_argument('--base_url', type=str, default=None,
                        help="Base URL of an OpenAI-compatible API (defaults to OPENAI_BASE_URL env var)")
    parser.add_argument('--api_key', type=str, default=None,
                        help="API key (defaults to OPENAI_API_KEY env var)")
    args = parser.parse_args()

    SYSTEM_PROMPT = load_prompt(args.task)
    if args.task == "qa":
        CAPTION_DATA = pd.read_csv(args.caption_csv)

    # The OpenAI client is thread-safe and can be shared across the thread pool
    client = OpenAI(
        base_url=args.base_url or os.environ.get("OPENAI_BASE_URL"),
        api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
    )

    total_cost = 0
    for model in args.model:
        total_cost += run_task(args, client, model)
        print(f"[{model}] All tasks finished.")

    print(f"Total estimated cost: ${total_cost:.6f}")
