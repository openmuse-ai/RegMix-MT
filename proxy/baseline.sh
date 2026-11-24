#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <DATASET_KEY> [--learning_rate LR] [--output_dir DIR]" >&2
}

DATASET_KEY=${1:-}
if [[ -z "$DATASET_KEY" ]]; then
  usage
  exit 1
fi
shift

# Parse optional arguments
LEARNING_RATE=""
CUSTOM_OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --output_dir)
      CUSTOM_OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

#MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"/mnt/data1/models/Qwen2.5-0.5B/"}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"/mnt/data1/models/SmolLM2-135M/"}
DATASET_DIR=${DATASET_DIR:-"/mnt/data3/gaowg/cpt-tech-rule/exp/data/mixed"}
EVAL_DATASET=${EVAL_DATASET:-"mathinstruct-sample"}
OUTPUT_DIR=${CUSTOM_OUTPUT_DIR:-${OUTPUT_DIR:-"saves/${DATASET_KEY}"}}
LEARNING_RATE=${LEARNING_RATE:-"1.0e-5"}
LLAMAFACTORY_CLI=${LLAMAFACTORY_CLI:-"llamafactory-cli"}

if ! command -v "$LLAMAFACTORY_CLI" >/dev/null 2>&1; then
  echo "Error: 找不到命令 $LLAMAFACTORY_CLI" >&2
  exit 127
fi

if [[ -z "$MODEL_NAME_OR_PATH" || -z "$DATASET_DIR" ]]; then
  echo "Error: MODEL_NAME_OR_PATH 或 DATASET_DIR 未设置" >&2
  exit 1
fi

cmd=(
  "$LLAMAFACTORY_CLI" train
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --trust_remote_code true
  --finetuning_type full
  --flash_attn fa2
  --stage pt
  --do_train true
  --dataset_dir "$DATASET_DIR"
  --dataset "$DATASET_KEY"
  --seed 42
  --data_seed 42
  --cutoff_len 2048
  --packing true
  --max_steps 125
  --overwrite_cache true
  --preprocessing_num_workers 2
  --dataloader_num_workers 2
  --output_dir "$OUTPUT_DIR"
  --log_level info
  --log_level_replica info
  --logging_first_step true
  --logging_steps 1
  --save_strategy steps
  --save_steps 10
  --save_total_limit 1
  --plot_loss true
  --overwrite_output_dir true
  --save_only_model false
  --report_to tensorboard
  --include_num_input_tokens_seen true
  --per_device_train_batch_size 64
  --gradient_accumulation_steps 32
  --learning_rate "$LEARNING_RATE"
  --num_train_epochs 1.0
  --lr_scheduler_type linear
  --warmup_steps 0
  --warmup_ratio 0.0
  --bf16 true
  --eval_dataset "$EVAL_DATASET"
  --per_device_eval_batch_size 1
  --eval_strategy steps
  --eval_steps 50
  --ddp_timeout 180000000
)

printf 'Running: %s\n' "${cmd[*]}"
"${cmd[@]}"

shopt -s nullglob
checkpoints=("$OUTPUT_DIR"/checkpoint*)
if (( ${#checkpoints[@]} )); then
  echo "Cleaning checkpoints under $OUTPUT_DIR"
  rm -rf "${checkpoints[@]}"
fi
shopt -u nullglob

