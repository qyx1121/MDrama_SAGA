#!/bin/bash
# SAGA RL training (Stage 2): Qwen3-VL-8B on M-Drama with the graph-alignment
# reward. Fill in MODEL_PATH / OUTPUT_DIR and the three reward-service
# addresses below before running. CWD must be the repository root.

set -x

# Set these to your cluster's NIC if needed
# export NCCL_SOCKET_IFNAME=eth0
# export GLOO_SOCKET_IFNAME=eth0

export WANDB_MODE="offline"  # use "online" with WANDB_API_KEY to sync

MODEL_PATH="/path/to/sft/checkpoint"  # SFT init, e.g. from sft/finetune/finetune_main.sh
OUTPUT_DIR="ckpt/qwen3_vl_8b_mdrama_saga"

# The reward function calls three OpenAI-compatible services asynchronously:
#   judge     — LLM judge for OE / SUMMARY accuracy
#   graph     — caption -> scene-graph construction (train/system_prompts/graph_build_prompt.txt)
#   embedding — text embeddings for graph matching (e.g. a vLLM embedding server)
# Multiple ports on one IP are comma-separated; multiple IPs are comma-separated
# and per-IP port groups are separated by ";".
JUDGE_IP="127.0.0.1"
JUDGE_PORTS="8000"
GRAPH_IP="127.0.0.1"
GRAPH_PORTS="8000"
EMBED_IP="127.0.0.1"
EMBED_PORT="8000"

python3 -m verl.trainer.main \
    config=train/config.yaml \
    data.train_files=data/rl/mdrama_rl_graph \
    data.val_files=data/rl/mdrama_rl_graph \
    data.mini_rollout_batch_size=64 \
    data.rollout_batch_size=64 \
    data.format_prompt=./train/format_prompt/drama.jinja \
    data.video_fps=2.0 \
    data.max_frames=192 \
    data.max_pixels=100352 \
    data.min_pixels=50176 \
    data.filter_overlong_prompts=False \
    data.shuffle=True \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.global_batch_size=64 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.freeze_vision_tower=True \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.rollout.max_num_batched_tokens=20000 \
    worker.reward.reward_function=train/reward_function/reward_graph/reward.py:compute_score \
    worker.reward.llm_judge_ip=${JUDGE_IP} \
    worker.reward.llm_judge_port=${JUDGE_PORTS} \
    worker.reward.graph_ip=${GRAPH_IP} \
    worker.reward.graph_port=${GRAPH_PORTS} \
    worker.reward.embedding_ip=${EMBED_IP} \
    worker.reward.embedding_port=${EMBED_PORT} \
    worker.rollout.n=8 \
    worker.rollout.tensor_parallel_size=1 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=False \
    trainer.total_epochs=1 \
    trainer.logger=['console','tensorboard','wandb'] \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=5 \
    trainer.save_limit=10 \
    trainer.val_freq=-1 \
    trainer.experiment_name=qwen3_vl_8b_mdrama_saga \
    trainer.save_checkpoint_path=${OUTPUT_DIR} \
    worker.reward.caption_reward_type=['global'] \
    worker.actor.caption_loss_weight=0.75 \
    worker.actor.reward_type=['global'] \
    algorithm.adv_estimator=grpo_caption \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
