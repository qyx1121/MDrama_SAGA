"""Build the cold-start SFT training json from the released M-Drama dataset.

Pipeline:
    1. Load annotations from HuggingFace: yixin1121/M-Drama (split="train").
       The repo is gated — request access on the dataset page and pass a token
       via `huggingface-cli login` or the HF_TOKEN env var.
    2. Convert each annotation to the Qwen-VL chat ("messages") format consumed
       by finetune/finetune_main.py:
         system    : sft_system_prompt.txt
         user      : <video> + question (+ "Options: [...]" for MC)
         assistant : <caption>...<think>...<answer>...
    3. Video clips are NOT shipped with the dataset. Run download_videos.py
       first; each clip is expected at {video_dir}/{video_id}.mp4.

Usage:
    python data/make_sft_data.py \
        --video_dir data/videos \
        --output data/mdrama_sft.json
"""
import argparse
import json
import os

from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Video sampling config used for the paper's cold-start SFT (192 frames @ fps=2)
VIDEO_KWARGS = {
    "max_pixels": 100352,   # 128 * 28 * 28
    "min_pixels": 50176,    # 64 * 28 * 28
    "max_frames": 192,
    "video_fps": 2.0,
}

ASSISTANT_TEMPLATE = "<caption>\n{caption}\n</caption>\n<think>\n{thinking}\n</think>\n<answer>\n{answer}\n</answer>"


def build_user_text(row):
    text = "<video>\n" + row["question"]
    if row["q_type"] == "MC" and row["candidates"] is not None:
        options = "; ".join(str(o) for o in row["candidates"])
        text += f"\nOptions: [{options}]"
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yixin1121/M-Drama")
    parser.add_argument("--split", default="train")
    parser.add_argument("--video_dir", default="data/videos",
                        help="Directory of clips downloaded by download_videos.py")
    parser.add_argument("--system_prompt", default=os.path.join(SCRIPT_DIR, "sft_system_prompt.txt"))
    parser.add_argument("--output", default="data/mdrama_sft.json")
    args = parser.parse_args()

    system_prompt = open(args.system_prompt).read()
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(ds)} annotations from {args.dataset} ({args.split})")

    data = []
    for row in ds:
        video_path = os.path.join(args.video_dir, row["video_id"] + ".mp4")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "video", "video": video_path, **VIDEO_KWARGS},
                {"type": "text", "text": build_user_text(row)},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": ASSISTANT_TEMPLATE.format(
                    caption=row["caption"], thinking=row["thinking"], answer=row["answer"])},
            ]},
        ]
        data.append({"messages": messages})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Wrote {len(data)} examples to {args.output}")


if __name__ == "__main__":
    main()
