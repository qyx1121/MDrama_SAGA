# MDrama-SAGA

Code for **"Beyond Sparse Rewards: A New Benchmark and Structure-Aware Graph Alignment for Micro-Drama Understanding"** (EMNLP 2026).

- M-Drama Data: [yixin1121/M-Drama](https://huggingface.co/datasets/yixin1121/M-Drama)
- SAGA Weights: [yixin1121/SAGA_Qwen3-8B](https://huggingface.co/yixin1121/SAGA_Qwen3-8B)

## Results

M-Drama test split. MC/OE in % (pass@1 / pass@3); Summary on a 10-point scale, judged by DeepSeek-V3.2 and GPT-5-mini.

| Model | Think | MC@1 | MC@3 | OE@1 | OE@3 | Sum. (DS-V3.2) | Sum. (GPT-5-mini) |
|---|:---:|---:|---:|---:|---:|---:|---:|
| *Closed-source flagships* | | | | | | | |
| Gemini-2.5-Flash | ✓ | 76.7 | — | 50.1 | — | 5.78 | 6.11 |
| Gemini-3-Flash | ✓ | 85.3 | — | 56.0 | — | 6.21 | 6.53 |
| GPT-5 | ✓ | 83.7 | — | 60.0 | — | 7.54 | 7.09 |
| *Open-source VLMs* | | | | | | | |
| Qwen2.5-VL-7B | ✗ | 38.6 | 53.2 | 24.0 | 36.8 | 3.67 | 4.54 |
| Qwen3-VL-8B | ✓ | 56.5 | 67.2 | 33.3 | 47.2 | 4.18 | 4.81 |
| InternVL3.5-8B | ✓ | 52.9 | 59.2 | 25.6 | 36.3 | 3.96 | 4.77 |
| Keye-VL 1.5-8B | ✓ | 50.3 | 66.8 | 23.2 | 37.4 | 3.54 | 4.30 |
| MiMo-VL-7B | ✓ | 60.5 | 71.6 | 25.7 | 41.3 | 4.16 | 5.65 |
| Qwen2.5-VL-72B | ✗ | 54.6 | 65.2 | 31.8 | 48.7 | 5.03 | 5.77 |
| InternVL3.5-38B | ✓ | 59.1 | 63.9 | 25.2 | 37.8 | 4.46 | 5.28 |
| Qwen3-VL-32B | ✓ | 66.5 | 74.1 | 38.6 | 55.4 | 4.93 | 5.50 |
| **SAGA (ours)** | ✓ | 69.3 | 78.7 | 40.0 | 56.3 | 5.74 | 6.41 |

## Layout

```
verl/            EasyR1 (verl) + SAGA changes (grpo_caption, caption-level loss, reward wiring)
sft/             stage 1: cold-start SFT -> RL init checkpoint          (sft/README.md)
train/           stage 2: RL
  qwen3_vl_8b_drama_dapo_graph.sh   main run
  config.yaml                       base hydra config
  format_prompt/drama.jinja         chat template: <caption>/<think>/<answer>
  system_prompts/                   prompts the reward loads (graph build, summary judge)
  reward_function/reward_graph/     R_graph = 0.725*F1_semantic + 0.275*F1_structural
data/rl/         M-Drama annotations -> RL parquet (14,135 rows)        (data/rl/README.md)
eval/            test-set prep -> inference -> LLM judge                (eval/README.md)
scripts/         model_merger.py: FSDP shards -> HF checkpoint
```

## Main run

`train/qwen3_vl_8b_drama_dapo_graph.sh`:

| | |
|---|---|
| init | stage-1 SFT checkpoint (`sft/`), 192 frames @ fps=2, vision tower frozen |
| reward | `reward_graph`: `0.725*F1_semantic + 0.275*F1_structural` |
| algo | `grpo_caption`, DAPO asymmetric clip (0.2/0.28), no KL, `caption_loss_weight=0.75` |
| rollout | n=8, TP=1 |

The reward calls three external OpenAI-compatible services — graph construction, LLM judge, embedding — set via `worker.reward.{graph,llm_judge,embedding}_{ip,port}`.

## Usage

Pipeline: download clips -> SFT (stage 1) -> RL data -> RL (stage 2) -> merge -> eval.
To only reproduce the main result, skip stage 1 and use the released weights.

```bash
pip install -e .

# 0. clips — M-Drama ships annotations only, fetch the videos yourself
#    (gated dataset: huggingface-cli login first; ~52 GB, yt-dlp + ffmpeg)
python sft/data/download_videos.py --output_dir data/videos

# 1. stage 1: cold-start SFT -> the RL init checkpoint (see sft/README.md)
python sft/data/make_sft_data.py --video_dir data/videos --output sft/data/mdrama_sft.json
cd sft && bash finetune/finetune_main.sh && cd ..

# 2. RL data — ground-truth scene graphs ship with the dataset; without
#    --cap2graph gt_graph is empty and the graph reward is 0
GT_GRAPHS=$(python -c "from huggingface_hub import hf_hub_download; \
  print(hf_hub_download('yixin1121/M-Drama', 'gt_graphs.json', repo_type='dataset'))")
bash data/rl/run.sh --video_dir data/videos --cap2graph "${GT_GRAPHS}"

# 3. stage 2: RL (set MODEL_PATH to the stage-1 checkpoint + service addresses first)
bash train/qwen3_vl_8b_drama_dapo_graph.sh

# 4. merge FSDP shards -> <ckpt_dir>/huggingface  (add --hf_upload_path <repo> to push)
python scripts/model_merger.py --local_dir <ckpt_dir>

# 5. eval (see eval/README.md) — needs the test clips:
python sft/data/download_videos.py --split test --output_dir data/videos
python eval/prepare_eval_data.py --video_dir data/videos
```

## License

Apache 2.0 ([LICENSE](LICENSE)). Built on [EasyR1](https://github.com/hiyouga/EasyR1) and [verl](https://github.com/volcengine/verl) (both Apache 2.0, notices kept in-tree). M-Drama is CC BY-NC 4.0, research use only.
