"""Build the SAGA RL training parquet from the released M-Drama dataset.

M-Drama (https://huggingface.co/datasets/yixin1121/M-Drama) ships annotations
only: 32,361 QA pairs over 8,102 short-drama clips, of which the 14,135 rows
tagged ``stage == "SFT+RL"`` are the ones used for RL. The videos themselves
are not distributed -- each clip is defined by ``(url, start_time, end_time)``
pointing into a YouTube video, so you have to fetch them yourself (see
``sft/data/download_videos.py``).

This script turns those annotations into the parquet consumed by the SAGA
reward function (``train/reward_function/reward_graph``), with the same schema
as EasyR1's RL data:

    videos     list[str]  absolute path of the clip
    system     str        system prompt (prefix of train/format_prompt/drama.jinja)
    problem    str        "<video>\\n{question}" (+ "\\nOptions: [...]" for MC)
    answer     str        <caption>...</caption><think>...</think><answer>...</answer>
    task_type  str        MC | OE | SUMMARY
    gt_graph   str        JSON scene graph built from the ground-truth caption

Pipeline
--------
    1. download    HF yixin1121/M-Drama  ->  data/rl/mdrama_annotations.jsonl
    2. video-map   resolve video_id -> local .mp4 (only needed if your clips are
                   not already named {video_id}.mp4)
    3. build       jsonl + videos + scene-graph cache  ->  .../train.parquet

Usage
-----
    # 1. annotations (the repo is gated: huggingface-cli login first)
    python data/rl/prepare_rl_data.py download

    # 2. only if your clips are not named {video_id}.mp4
    python data/rl/prepare_rl_data.py video-map \
        --video_dir /path/to/clips --mode timestamp

    # 3. parquet (--cap2graph accepts the dataset's video_id-keyed
    #    gt_graphs.json directly, or a caption-keyed cache)
    python data/rl/prepare_rl_data.py build \
        --video_dir /path/to/clips \
        --cap2graph gt_graphs.json

    # all three at once
    bash data/rl/run.sh --video_dir /path/to/clips
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RL_DIR = os.path.join(REPO_ROOT, "data", "rl")

DEFAULT_DATASET = "yixin1121/M-Drama"
DEFAULT_ANNOTATIONS = os.path.join(RL_DIR, "mdrama_annotations.jsonl")
DEFAULT_VIDEO_MAP = os.path.join(RL_DIR, "video_map.json")
DEFAULT_OUTPUT = os.path.join(RL_DIR, "mdrama_rl_graph", "train.parquet")

# Only these annotations carry the RL split.
RL_STAGE = "SFT+RL"

ASSISTANT_TEMPLATE = (
    "<caption>\n{caption}\n</caption>\n<think>\n{thinking}\n</think>\n<answer>\n{answer}\n</answer>"
)

# The canned question that marks a caption/summary item (M-Drama tags it q_type=summary).
SUMMARY_QUESTION = "Please describe the video and analyze the plot in detail."

# "{title}_{HH:MM:SS}-{HH:MM:SS}.mp4", the naming used by the in-house clip library.
TIMESTAMP_RE = re.compile(r"^.*_(\d{2}:\d{2}:\d{2})-(\d{2}:\d{2}:\d{2})\.mp4$", re.IGNORECASE)

CAPTION_RE = re.compile(r"<caption>(.*?)</caption>", re.DOTALL)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_annotations(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_problem(row):
    """`<video>` + question, with MC options appended in a single bracketed list."""
    text = "<video>\n" + row["question"]
    candidates = row.get("candidates") or []
    if len(candidates) > 0:
        text += "\nOptions: [" + "; ".join(str(c) for c in candidates) + "]"
    return text


def build_answer(row):
    return ASSISTANT_TEMPLATE.format(
        caption=row["caption"], thinking=row["thinking"], answer=row["answer"]
    )


def build_task_type(row):
    if str(row["answer"]).strip() in ("A", "B", "C", "D"):
        return "MC"
    if row["question"].strip() == SUMMARY_QUESTION:
        return "SUMMARY"
    return "OE"


def load_system_prompt():
    """Reuse the training-side chat template so data and rollout cannot drift."""
    jinja_path = os.path.join(REPO_ROOT, "train", "format_prompt", "drama.jinja")
    with open(jinja_path, encoding="utf-8") as f:
        template = f.read()
    # The template is "<system prompt>{{ content | trim }}"; keep the prompt only.
    # drama.jinja keeps the whole prompt on one line with literal "\n" escapes,
    # so restore the real newlines to match sft/data/sft_system_prompt.txt.
    return template.split("{{")[0].strip().replace("\\n", "\n")


# --------------------------------------------------------------------------- #
# 1. download
# --------------------------------------------------------------------------- #
def cmd_download(args):
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(ds)} annotations from {args.dataset} [{args.split}]")

    kept, total = [], 0
    for row in ds:
        total += 1
        if args.stage != "all" and row["stage"] != args.stage:
            continue
        kept.append(
            {
                "qid": row["qid"],
                "video_id": row["video_id"],
                "question": row["question"],
                "candidates": list(row["candidates"]) if row["candidates"] is not None else [],
                "answer": row["answer"],
                "thinking": row["thinking"],
                "caption": row["caption"],
                "task_type": row["task_type"],
                "q_type": row["q_type"],
                "url": row["url"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "stage": row["stage"],
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(kept)}/{total} annotations (stage={args.stage}) to {args.output}")


# --------------------------------------------------------------------------- #
# 2. video-map
# --------------------------------------------------------------------------- #
def _unique_clips(annotations):
    """Iterate (video_id, first annotation row) once per clip."""
    seen = set()
    for row in annotations:
        if row["video_id"] not in seen:
            seen.add(row["video_id"])
            yield row["video_id"], row


def map_by_timestamp(annotations, video_dir):
    """Index the clip library by the (start, end) timestamps embedded in filenames.

    Works with the in-house naming ``{title}_{start}-{end}.mp4``. Ambiguous keys
    (two different source videos cut at the same offsets) are reported as
    unresolved -- fix those by hand or use ``--mode caption``.
    """
    index = defaultdict(list)
    for name in os.listdir(video_dir):
        match = TIMESTAMP_RE.match(name)
        if match:
            index[(match.group(1), match.group(2))].append(name)
    print(f"Indexed {sum(len(v) for v in index.values())} clips under {video_dir}")

    video_map, unresolved = {}, []
    for vid, row in _unique_clips(annotations):
        hits = index.get((row["start_time"], row["end_time"]), [])
        if len(hits) == 1:
            video_map[vid] = os.path.join(video_dir, hits[0])
        else:
            unresolved.append((vid, row["start_time"], row["end_time"], len(hits)))
    return video_map, unresolved


def map_by_caption(annotations, sft_json):
    """Recover video_id -> path from an SFT json whose assistant turn holds the caption.

    Any json produced by ``sft/data/make_sft_data.py`` works; the caption text is
    the join key, so this stays exact where timestamp matching is ambiguous.
    """
    with open(sft_json, encoding="utf-8") as f:
        sft_data = json.load(f)

    caption_to_path = {}
    for item in sft_data:
        video_path, caption = None, None
        for message in item["messages"]:
            for chunk in message["content"]:
                if chunk.get("type") == "video":
                    video_path = chunk["video"]
                elif chunk.get("type") == "text":
                    match = CAPTION_RE.search(chunk["text"])
                    if match:
                        caption = match.group(1).strip()
        if video_path and caption:
            caption_to_path[caption] = video_path
    print(f"Indexed {len(caption_to_path)} caption -> path entries from {sft_json}")

    # M-Drama contains a handful of clips whose captions are byte-identical to
    # another clip's, so the caption alone cannot tell them apart.
    caption_counts = defaultdict(int)
    for vid, row in _unique_clips(annotations):
        caption_counts[row["caption"].strip()] += 1

    video_map, unresolved = {}, []
    for vid, row in _unique_clips(annotations):
        caption = row["caption"].strip()
        path = caption_to_path.get(caption)
        if path and caption_counts[caption] == 1:
            video_map[vid] = path
        else:
            unresolved.append((vid, row["start_time"], row["end_time"], caption_counts[caption]))
    return video_map, unresolved


def cmd_video_map(args):
    annotations = load_annotations(args.annotations)
    args.video_dir = os.path.abspath(args.video_dir)

    if args.mode == "timestamp":
        video_map, unresolved = map_by_timestamp(annotations, args.video_dir)
    else:
        if not args.sft_json:
            raise SystemExit("--mode caption requires --sft_json")
        video_map, unresolved = map_by_caption(annotations, args.sft_json)

    clips = len({row["video_id"] for row in annotations})
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(video_map, f, ensure_ascii=False, indent=2)

    print(f"Resolved {len(video_map)}/{clips} clips -> {args.output}")
    if unresolved:
        report = args.output + ".unresolved.json"
        with open(report, "w", encoding="utf-8") as f:
            json.dump(
                [{"video_id": v, "start_time": s, "end_time": e, "candidates": c}
                 for v, s, e, c in unresolved],
                f, ensure_ascii=False, indent=2,
            )
        print(f"{len(unresolved)} clips unresolved, written to {report}")


# --------------------------------------------------------------------------- #
# 3. build
# --------------------------------------------------------------------------- #
def resolve_video_path(row, video_dir, video_map):
    if video_map and row["video_id"] in video_map:
        return video_map[row["video_id"]]
    return os.path.join(video_dir, row["video_id"] + ".mp4")


def cmd_build(args):
    annotations = load_annotations(args.annotations)
    print(f"Loaded {len(annotations)} annotations from {args.annotations}")

    video_map = {}
    if args.video_map:
        with open(args.video_map, encoding="utf-8") as f:
            video_map = json.load(f)
        print(f"Loaded {len(video_map)} entries from {args.video_map}")

    cap2graph = {}
    if args.cap2graph:
        with open(args.cap2graph, encoding="utf-8") as f:
            cap2graph = json.load(f)
        print(f"Loaded {len(cap2graph)} graph entries from {args.cap2graph}")
    else:
        print("WARNING: no --cap2graph given, gt_graph will be empty and the "
              "graph-alignment reward will stay at 0.")

    system_prompt = load_system_prompt()

    records, missing_videos, missing_graphs = [], [], []
    for row in annotations:
        video_path = resolve_video_path(row, args.video_dir, video_map)
        if not os.path.exists(video_path):
            missing_videos.append(video_path)
            continue

        caption_key = row["caption"].strip()
        gt_graph = cap2graph.get(caption_key)
        if gt_graph is None:
            # also accept the dataset's video_id-keyed gt_graphs.json
            gt_graph = cap2graph.get(row["video_id"])
        if gt_graph is None:
            missing_graphs.append(row["qid"])
            gt_graph = {}

        records.append(
            {
                "videos": [video_path],
                "system": system_prompt,
                "problem": build_problem(row),
                "answer": build_answer(row),
                "task_type": build_task_type(row),
                "gt_graph": json.dumps(gt_graph, ensure_ascii=False),
            }
        )

    if missing_videos:
        uniq = sorted(set(missing_videos))
        print(f"WARNING: {len(missing_videos)} rows reference {len(uniq)} missing clips, "
              f"first: {uniq[0]}")
        if args.strict:
            raise SystemExit("aborting because --strict was set")
    if missing_graphs:
        print(f"WARNING: {len(missing_graphs)} rows have no cached gt_graph "
              f"(first qid={missing_graphs[0]}); their graph reward will be 0.")

    if not records:
        raise SystemExit("no rows left to write - check --video_dir")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"Wrote {len(df)} rows to {args.output}")
    print("task_type distribution:", df["task_type"].value_counts().to_dict())


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("download", help="pull annotations from HuggingFace")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default="train")
    p.add_argument("--stage", default=RL_STAGE,
                   help=f"keep only this stage (default: {RL_STAGE}, use 'all' to keep everything)")
    p.add_argument("--output", default=DEFAULT_ANNOTATIONS)
    p.set_defaults(func=cmd_download)

    p = subparsers.add_parser("video-map", help="resolve video_id -> local .mp4")
    p.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--mode", choices=["timestamp", "caption"], default="timestamp")
    p.add_argument("--sft_json", help="SFT json, required by --mode caption")
    p.add_argument("--output", default=DEFAULT_VIDEO_MAP)
    p.set_defaults(func=cmd_video_map)

    p = subparsers.add_parser("build", help="assemble the RL parquet")
    p.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    p.add_argument("--video_dir", required=True)
    p.add_argument("--video_map", help=f"video_id -> path json (default: {DEFAULT_VIDEO_MAP} if present)")
    p.add_argument("--cap2graph", help="scene-graph cache json: caption-keyed, or "
                                       "video_id-keyed (the dataset's gt_graphs.json works as-is)")
    p.add_argument("--strict", action="store_true",
                   help="fail instead of dropping rows with missing clips")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.set_defaults(func=cmd_build)

    args = parser.parse_args()

    if args.command == "build":
        args.video_dir = os.path.abspath(args.video_dir)
        if not args.video_map and os.path.exists(DEFAULT_VIDEO_MAP):
            args.video_map = DEFAULT_VIDEO_MAP

    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
