#!/bin/bash
# ============================================================
# Stage 1: Cold-start SFT for SAGA (Qwen3-VL-8B-Instruct)
# Trains the base model on caption/<think>/<answer> formatted
# micro-drama data before the RL stage (see ../README.md).
# Run from this directory's parent (sft/):
#   cd sft && bash finetune/finetune_main.sh
# ============================================================

export FFMPEG_LOG_LEVEL="quiet"
export AV_LOG_LEVEL="quiet"

# TODO: point to your local Qwen3-VL-8B-Instruct checkpoint
export MODEL="/path/to/Qwen3-VL-8B-Instruct/"
# Path to the SFT training json, produced by data/make_sft_data.py
export DATA="data/mdrama_sft.json"

set -x

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='output/sft_qwen3vl_8b_drama_192fr'
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

NPROC_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6001

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPUS=$((NNODES * NPROC_PER_NODE))
BATCH_SIZE=${BATCH_SIZE:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

torchrun $DISTRIBUTED_ARGS \
  finetune/finetune_main.py \
  --model_name_or_path $MODEL \
  --output_dir ${OUTPUT_DIR} \
  --data_path ${DATA} \
  --overwrite_output_dir True \
  --tune_vision False \
  --tune_llm True \
  --tune_mlp True \
  --tune_llm_head True \
  --use_lora False \
  --dataloader_num_workers 2 \
  --bf16 True \
  --num_train_epochs 2 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_only_model True \
  --save_steps 20 \
  --attn_implementation "flash_attention_2" \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_length 16384 \
  --do_train True \
  --deepspeed "finetune/ds_config_zero3.json" \
  --gradient_checkpointing True \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
