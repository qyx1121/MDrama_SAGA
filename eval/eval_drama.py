"""
Unified inference script for drama video evaluation, supporting both QA and
Summary tasks.

Usage examples:
    # QA inference (multi-GPU DDP, pass@k)
    python eval/eval_drama.py --task qa \
        --model_path ckpt/qwen3_vl_8b_mdrama_saga/huggingface \
        --processor_path Qwen/Qwen3-VL-8B-Instruct \
        --data_path eval/data/test_qa.csv \
        --output_path results/qa/saga.json \
        --video_dir data/videos \
        --video_map eval/data/test_video_map.json \
        --passk 3

    # Summary inference
    python eval/eval_drama.py --task summary \
        --model_path ckpt/qwen3_vl_8b_mdrama_saga/huggingface \
        --processor_path Qwen/Qwen3-VL-8B-Instruct \
        --data_path eval/data/test_summary.csv \
        --output_path results/summary/saga.json \
        --video_dir data/videos \
        --video_map eval/data/test_video_map.json

See eval/README.md for the end-to-end pipeline (build the CSVs from M-Drama,
run inference, then score with eval_llm_judge.py).
"""
import os
import json
import argparse
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial

from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import pandas as pd
from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SUMMARY_USER_PROMPT = "<video>\nPlease describe the video and analyze the plot in detail."

# Per-task default sampling parameters for pass@k (can be overridden via CLI)
TASK_SAMPLING_DEFAULTS = {
    "qa": {"temperature": 1.0, "top_p": 0.95},
    "summary": {"temperature": 0.7, "top_p": 0.8},
}


# =====================================================================================
# 1. Helpers for distributed environment setup
# =====================================================================================
def setup(rank, world_size):
    """Initialize the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # use a free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Destroy the distributed process group."""
    dist.destroy_process_group()


def load_system_prompt(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"System prompt file not found: {path}. "
            f"Expected one of the prompt files under {os.path.join(SCRIPT_DIR, 'prompts')}."
        )
    with open(path, encoding='utf-8') as f:
        return f.read()


def load_video_map(path):
    """Load an optional {clip_id: /abs/path/to/clip.mp4} json.

    Produced by `data/rl/prepare_rl_data.py video-map` or
    `eval/prepare_eval_data.py --video_dir ...`; pass it when your clips are not
    named `{drama_name}_{start_time}-{end_time}.mp4`.
    """
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def resolve_video_path(video_dir, video_map, clip_id, fallback_name):
    """An explicit video map wins; otherwise fall back to the
    `{clip_id}_{start_time}-{end_time}.mp4` naming convention inside video_dir."""
    if clip_id in video_map:
        return video_map[clip_id]
    return os.path.join(video_dir, fallback_name + ".mp4")


# =====================================================================================
# 2. Custom datasets
# =====================================================================================
class VideoQADataset(Dataset):
    """QA dataset.

    The CSV file must contain the columns:
    drama_name / start_time / end_time / question / q_type / candidates / answer / qid
    """

    def __init__(self, csv_file, video_dir, video_map=None):
        self.data = pd.read_csv(csv_file, on_bad_lines="skip")
        self.video_dir = video_dir
        self.video_map = video_map or {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item["drama_name"] + "_" + item['start_time'] + "-" + item['end_time']
        question = item["question"]

        if item['q_type'] == "MC":
            candidates = eval(item["candidates"])
            user_prompt = question + "\nOptions: [" + "; ".join(candidates) + "]"
        else:
            user_prompt = question
        gt = item['answer']
        video_path = resolve_video_path(self.video_dir, self.video_map, item["drama_name"], vid)

        return {
            "vid": vid,
            "qid": item["qid"],
            "q_type": item['q_type'],
            "video_path": video_path,
            "user_prompt": user_prompt,
            "question": question,
            "gt": gt,
        }


class VideoSummaryDataset(Dataset):
    """Summary dataset.

    The CSV file must contain the columns:
    drama_name / start_time / end_time / caption
    Samples are de-duplicated by video id.
    """

    def __init__(self, csv_file, video_dir, video_map=None):
        data = pd.read_csv(csv_file)
        self.data = {}
        self.clip_ids = {}
        for idx, item in data.iterrows():
            key = item["drama_name"] + "_" + item['start_time'] + "-" + item['end_time']
            self.data[key] = item['caption']
            self.clip_ids[key] = item["drama_name"]
        self.vids = list(self.data.keys())
        self.video_dir = video_dir
        self.video_map = video_map or {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vid = self.vids[idx]
        gt_cap = self.data[vid]
        video_path = resolve_video_path(self.video_dir, self.video_map, self.clip_ids[vid], vid)

        return {
            "vid": vid,
            "video_path": video_path,
            "gt": gt_cap,
        }


# =====================================================================================
# 3. Custom collate function
# =====================================================================================
def collate_fn(batch, processor, task, system_prompt):
    """
    Turn a list of dataset samples into the batch tensors required by the model.
    Tokenization is performed once for the whole batch for efficiency.
    """
    vids = [item['vid'] for item in batch]
    gts = [item['gt'] for item in batch]

    batch_messages = []
    for item in batch:
        if task == "qa":
            video_content = {
                "type": "video",
                "video": item['video_path'],
                "fps": 2.0,
                "max_frames": 192,
                "max_pixels": 128 * 28 * 28,
                "min_pixels": 64 * 28 * 28,
            }
            text_content = {"type": "text", "text": item['user_prompt']}
        else:  # summary
            video_content = {
                "type": "video",
                "video": item['video_path'],
                "fps": 2.0,
                "max_frames": 192,
                "max_pixels": 128 * 28 * 28,
                "min_pixels": 64 * 28 * 28
            }
            text_content = {"type": "text", "text": SUMMARY_USER_PROMPT}

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [video_content, text_content],
            },
        ]
        batch_messages.append(messages)

    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]

    image_inputs, video_inputs = process_vision_info(batch_messages, return_video_metadata=True)

    videos = [i[0] for i in video_inputs]
    video_metadata = [i[1] for i in video_inputs]

    inputs = processor(
        text=texts,
        images=None,
        videos=videos,
        video_metadata=video_metadata,
        padding=True,
        return_tensors="pt",
        do_sample_frames=False,
    )

    result = {"inputs": inputs, "vids": vids, "gts": gts}
    if task == "qa":
        result["questions"] = [item['question'] for item in batch]
        result["q_types"] = [item['q_type'] for item in batch]
        result["qids"] = [item['qid'] for item in batch]
    return result


# =====================================================================================
# 4. Main inference worker (executed on each GPU process)
# =====================================================================================
def main_worker(rank, world_size, args):
    print(f"Running DDP inference on rank {rank} (task={args.task}).")
    setup(rank, world_size)

    system_prompt = load_system_prompt(args.system_prompt)
    video_map = load_video_map(args.video_map) if args.video_map else {}
    if video_map:
        print(f"Loaded {len(video_map)} clip paths from {args.video_map}")

    # Load the model in each process and move it to the designated GPU
    # (device_map="auto" is intentionally not used)
    if "qwen2.5" in args.processor_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(rank)
    elif "qwen3" in args.processor_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(rank)
    else:
        raise ValueError(f"Unsupported processor_path: {args.processor_path} (must contain 'qwen2.5' or 'qwen3')")

    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank])

    processor = AutoProcessor.from_pretrained(args.processor_path)
    processor.video_processor.do_sample_frames = False
    processor.tokenizer.padding_side = "left"

    # Build dataset and distributed sampler
    if args.task == "qa":
        dataset = VideoQADataset(args.data_path, args.video_dir, video_map)
    else:
        dataset = VideoSummaryDataset(args.data_path, args.video_dir, video_map)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    custom_collate_fn = partial(collate_fn, processor=processor, task=args.task, system_prompt=system_prompt)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    # Sampling parameters: CLI values take precedence, otherwise task defaults
    temperature = args.temperature if args.temperature is not None else TASK_SAMPLING_DEFAULTS[args.task]["temperature"]
    top_p = args.top_p if args.top_p is not None else TASK_SAMPLING_DEFAULTS[args.task]["top_p"]

    # Start inference
    results_list = []
    progress_bar = tqdm(dataloader, disable=(rank != 0))

    for batch in progress_bar:
        inputs = batch['inputs']
        # Move the batch to the current GPU
        inputs = {k: v.to(rank) for k, v in inputs.items()}

        # Inference: use model.module to access the underlying model under DDP
        with torch.no_grad():
            if args.passk == 1:
                generated_ids = model.module.generate(**inputs, max_new_tokens=args.max_new_tokens)
            else:
                generated_ids = model.module.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=args.passk,
                    max_new_tokens=args.max_new_tokens,
                )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(
                inputs['input_ids'].repeat_interleave(args.passk, dim=0),
                generated_ids,
            )
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for i, pred_text in enumerate(output_texts):
            batch_idx = i // args.passk
            item = {
                "vid": batch['vids'][batch_idx],
                "pred": pred_text,
                "gt": batch['gts'][batch_idx],
            }
            if args.task == "qa":
                item["qid"] = batch['qids'][batch_idx]
                item["question"] = batch['questions'][batch_idx]
                item["q_type"] = batch['q_types'][batch_idx]
            results_list.append(item)

    # =================================================================================
    # 5. Gather results from all processes to the master process
    # =================================================================================
    all_results = [None] * world_size
    dist.all_gather_object(all_results, results_list)

    # Only the master process (rank 0) performs post-processing and saving
    if rank == 0:
        print("Gathering results from all processes.")
        final_results = []
        for res_list in all_results:
            final_results.extend(res_list)

        if args.task == "qa":
            # Compute overall accuracy (extract <answer> tag and compare with gt)
            correct_predictions = 0
            for item in final_results:
                soup = BeautifulSoup(item['pred'], "html.parser")
                pred = soup.find("answer")
                pred = pred.get_text(strip=True) if pred else ""
                correct_predictions += (pred == item['gt'])

            total_predictions = len(final_results)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

            print(f"Total samples: {total_predictions}")
            print(f"Accuracy: {accuracy:.4f}")

            res = {'score': accuracy, 'details': final_results}
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=4)
        else:
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=4)

        print(f"Results saved to {args.output_path}")

    cleanup()


# =====================================================================================
# 6. Main entry point and launcher
# =====================================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified inference script for drama QA / Summary evaluation")
    parser.add_argument('--task', type=str, required=True, choices=['qa', 'summary'],
                        help="Evaluation task type: qa or summary")
    parser.add_argument('--model_path', type=str, default="ckpt/qwen3_vl_8b_mdrama_saga/huggingface")
    parser.add_argument('--processor_path', type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument('--data_path', type=str, default="eval/data/test_qa.csv",
                        help="QA: question CSV; Summary: CSV with a caption column")
    parser.add_argument('--output_path', type=str, default="results/qa/result.json")
    parser.add_argument('--video_dir', type=str, default="data/videos")
    parser.add_argument('--video_map', type=str, default=None,
                        help="Optional {clip_id: /abs/path.mp4} json. Use it when your clips are not "
                             "named '{drama_name}_{start_time}-{end_time}.mp4' — e.g. the "
                             "{video_id}.mp4 layout produced by sft/data/download_videos.py "
                             "(see eval/prepare_eval_data.py, which writes this file)")
    parser.add_argument('--system_prompt', type=str,
                        default=os.path.join(SCRIPT_DIR, "prompts", "system_prompt.txt"))
    parser.add_argument('--nproc_per_node', type=int, default=torch.cuda.device_count(), help="Number of GPUs to use.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size per GPU.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader per GPU.")
    parser.add_argument('--passk', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=None,
                        help="Sampling temperature for pass@k. Default: qa=1.0, summary=0.7")
    parser.add_argument('--top_p', type=float, default=None,
                        help="Sampling top_p for pass@k. Default: qa=0.95, summary=0.8")
    args = parser.parse_args()

    world_size = args.nproc_per_node

    # Launch distributed inference via mp.spawn
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    main()
