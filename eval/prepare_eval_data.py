"""Turn the released M-Drama annotations into the CSVs consumed by eval_drama.py.

M-Drama (https://huggingface.co/datasets/yixin1121/M-Drama) ships annotations
only, in two splits:

    train   32,361 rows over 8,102 clips   (stage: SFT / SFT+RL)  -> training
    test     2,075 rows over 1,036 clips   (stage: test)          -> evaluation

The test split holds 1,562 multiple-choice + 513 open-ended questions and no
caption/summary annotation of its own; the reference for the summary task is the
per-clip ``caption`` field, which already contains the ``<caption> /
<highlights> / <summary>`` blocks. Videos are not distributed -- each clip is
``(url, start_time, end_time)`` into a YouTube video, so fetch them with
``sft/data/download_videos.py --split test`` first.

Outputs (written to --output_dir, default ``eval/data``):

    test_qa.csv            one row per QA annotation (MC + OE)
    test_summary.csv       one row per clip (reference caption)
    test_qa_captions.csv   qid -> caption, the context the open-ended judge needs
    test_video_map.json    {video_id: /abs/path.mp4}, for eval_drama.py --video_map

The CSVs carry a ``drama_name`` column because that is the clip id eval_drama.py
looks up; this script fills it with M-Drama's ``video_id``.

Usage:
    # 1. clips (gated dataset: huggingface-cli login first)
    python sft/data/download_videos.py --split test --output_dir data/videos

    # 2. CSVs + video map
    python eval/prepare_eval_data.py --video_dir data/videos

See eval/README.md for the full pipeline.
"""
import argparse
import json
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "eval", "data")

DEFAULT_DATASET = "yixin1121/M-Drama"
DEFAULT_SPLIT = "test"

# q_type values that make up the QA set; "summary" rows are handled separately.
QA_TYPES = ("MC", "OE")

CAPTION_TAG = "<caption>"


def load_annotations(dataset, split):
    """Pull one split of M-Drama and normalize it into plain python rows."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    rows = []
    for row in ds:
        rows.append(
            {
                "qid": row["qid"],
                "video_id": row["video_id"],
                "question": row["question"],
                "candidates": list(row["candidates"]) if row["candidates"] is not None else [],
                "answer": row["answer"],
                "caption": row["caption"],
                "q_type": row["q_type"],
                "url": row["url"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
            }
        )
    print(f"Loaded {len(rows)} annotations from {dataset} [{split}]")
    return rows


def build_qa_frame(rows):
    """One row per QA annotation. `candidates` is stored as a JSON list because
    eval_drama.py reads it back with `eval()`."""
    records = []
    for row in rows:
        if row["q_type"] not in QA_TYPES:
            continue
        records.append(
            {
                "qid": row["qid"],
                "drama_name": row["video_id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "question": row["question"],
                "q_type": row["q_type"],
                "candidates": json.dumps(row["candidates"], ensure_ascii=False),
                "answer": row["answer"],
            }
        )
    return pd.DataFrame(records)


def build_summary_frame(rows):
    """One row per clip: the reference caption of the summary task."""
    records, seen = [], set()
    for row in rows:
        if row["video_id"] in seen:
            continue
        seen.add(row["video_id"])
        records.append(
            {
                "drama_name": row["video_id"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "caption": row["caption"],
            }
        )
    return pd.DataFrame(records)


def build_caption_frame(rows):
    """qid -> reference caption, the `Video Caption` context the open-ended judge
    is prompted with (eval_llm_judge.py --caption_csv)."""
    records = [
        {"qid": row["qid"], "caption": row["caption"]}
        for row in rows
        if row["q_type"] in QA_TYPES
    ]
    return pd.DataFrame(records)


def build_video_map(rows, video_dir):
    """Map video_id -> {video_dir}/{video_id}.mp4, the layout written by
    sft/data/download_videos.py. Clips that are not on disk are reported."""
    video_dir = os.path.abspath(video_dir)
    video_map, missing = {}, []
    for video_id in sorted({row["video_id"] for row in rows}):
        path = os.path.join(video_dir, video_id + ".mp4")
        if os.path.exists(path):
            video_map[video_id] = path
        else:
            missing.append(video_id)
    return video_map, missing


def check_summary_captions(summary_df):
    """The summary judge reads the reference out of a <caption> tag; warn when the
    annotations do not have one, otherwise every sample is skipped."""
    if summary_df.empty:
        return
    tagged = summary_df["caption"].astype(str).str.contains(CAPTION_TAG, regex=False).sum()
    if tagged < len(summary_df):
        print(f"WARNING: {len(summary_df) - tagged}/{len(summary_df)} summary rows have no "
              f"<caption> block; eval_llm_judge.py skips those samples.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        help=f"M-Drama split to convert (default: {DEFAULT_SPLIT})")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where the CSVs and the video map are written")
    parser.add_argument("--video_dir", default=None,
                        help="Directory holding {video_id}.mp4 clips; when given, "
                             "{output_dir}/{split}_video_map.json is written as well")
    parser.add_argument("--strict", action="store_true",
                        help="fail instead of warning when clips are missing")
    args = parser.parse_args()

    rows = load_annotations(args.dataset, args.split)

    qa_df = build_qa_frame(rows)
    summary_df = build_summary_frame(rows)
    caption_df = build_caption_frame(rows)

    os.makedirs(args.output_dir, exist_ok=True)
    qa_path = os.path.join(args.output_dir, f"{args.split}_qa.csv")
    summary_path = os.path.join(args.output_dir, f"{args.split}_summary.csv")
    caption_path = os.path.join(args.output_dir, f"{args.split}_qa_captions.csv")
    qa_df.to_csv(qa_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    caption_df.to_csv(caption_path, index=False)
    print(f"Wrote {len(qa_df)} QA rows      -> {qa_path}")
    print(f"Wrote {len(summary_df)} summary rows -> {summary_path}")
    print(f"Wrote {len(caption_df)} caption rows -> {caption_path}")
    print("q_type distribution:", qa_df["q_type"].value_counts().to_dict())

    check_summary_captions(summary_df)

    if args.video_dir:
        video_map, missing = build_video_map(rows, args.video_dir)
        map_path = os.path.join(args.output_dir, f"{args.split}_video_map.json")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(video_map, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(video_map)} clip paths -> {map_path}")
        if missing:
            print(f"WARNING: {len(missing)}/{len(video_map) + len(missing)} clips missing under "
                  f"{args.video_dir}, first: {missing[0]}")
            if args.strict:
                raise SystemExit("aborting because --strict was set")


if __name__ == "__main__":
    sys.exit(main())
