# Stage 1: Cold-Start SFT

Cold-start supervised fine-tuning of **Qwen3-VL-8B-Instruct** before the SAGA RL stage (see the repository root README). This stage teaches the model the `<caption> / <think> / <answer>` output format on micro-drama data.

## Structure

```
sft/
├── finetune/
│   ├── finetune_main.sh      # ★ Entry point (torchrun + DeepSpeed ZeRO-3)
│   ├── finetune_main.py      # Training script (HF Trainer, Qwen3-VL full-parameter SFT)
│   └── ds_config_zero3.json  # DeepSpeed ZeRO-3 config used by the paper run
└── data/
    ├── make_sft_data.py       # ★ M-Drama (HF) annotations → SFT training json (messages format)
    ├── download_videos.py     # Fetch the YouTube clips (url/start_time/end_time) via yt-dlp
    └── sft_system_prompt.txt  # System prompt enforcing <caption>/<think>/<answer>
```

## Data Preparation

The SFT data is rebuilt from the released [M-Drama](https://huggingface.co/datasets/yixin1121/M-Drama)
dataset (32,361 annotations over 8,102 clips). The HF repo is **gated** — request
access first, then `huggingface-cli login`.

```bash
# 1. Download the 8,102 clips from YouTube (~52 GB, requires yt-dlp + ffmpeg)
python data/download_videos.py --output_dir data/videos --workers 8

# 2. Convert annotations to the SFT messages json (verified to reproduce the
#    paper's training file exactly, modulo video paths)
python data/make_sft_data.py --video_dir data/videos --output data/mdrama_sft.json
```

## Paper Configuration

- Base model: `Qwen3-VL-8B-Instruct`
- Data: micro-drama SFT set at **192 frames @ fps=2** (`data/mdrama_sft.json`, produced by `data/make_sft_data.py` from M-Drama)
- Vision tower frozen (`--tune_vision False`), LLM + MLP + LM head tuned
- Full-parameter SFT, bf16, lr 1e-5, cosine schedule, 2 epochs, global batch 64, max_length 16384
- The resulting checkpoint (paper: `checkpoint-506`) is the SFT init for the RL stage.

## Run

```bash
cd sft
# 1. Prepare data (see "Data Preparation" above).
# 2. Edit finetune/finetune_main.sh: set MODEL to your Qwen3-VL-8B-Instruct path,
#    and NPROC_PER_NODE to your GPU count.
bash finetune/finetune_main.sh
```

## Notes

- `--data_path` takes a single training json directly (a list of `{"messages": [...]}` examples).
- Data prep requires `datasets`; video download requires `yt-dlp` + `ffmpeg`.
- Requires `qwen_vl_utils`, `deepspeed`, `peft` and `flash-attn`. The video backend is `av`,
  pulled in by `qwen-vl-utils` (`decord` is supported as an alternative, install it if you
  prefer that reader).
- All of the above are in the root `requirements.txt`.
