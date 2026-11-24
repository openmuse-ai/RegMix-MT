#!/bin/bash

export DATASET_DIR="${DATASET_DIR:-data/mixed}"
CUDA_VISIBLE_DEVICES=0 python cpt4math/run_baseline_jobs.py \
  --max-workers 1 \
  --dataset-info "$DATASET_DIR/dataset_info.json" \
  "$@"
