# Beyond Manual Tuning: Proxy-Based Data Mixture Optimization for Mid-Training

## Abstract

Mid-training has become the standard approach for injecting domain expertise into foundation models; however, determining optimal data mixtures remains an open problem.

Mixture selection is difficult, as mathematical reasoning and general language modeling are anti-correlated across 512 controlled experiments, exhibiting a multi-objective conflict ($\rho = -0.594$, $p < 0.0001$ via permutation test), so no single mixture simultaneously minimizes both.

We present RegMix-MT (RegMix for Mid-Training), a two-stage framework for mid-training mixture selection. It adopts negative log-likelihood (NLL) on expert step-by-step solution traces—rather than accuracy metrics—as a continuous proxy that ranks mixtures at 135M scale, then navigates the conflict via priority-based selection—selecting a high-reasoning mixture that best preserves general capabilities under a minimum general-data constraint.

Across five models (360M–8B) spanning two architecture families, a single ratio discovered by a 135M proxy consistently improves math reasoning (GSM8K: +6pp on SmolLM2-1.7B, +14.5pp on Llama-3.1-8B) while maintaining general and code ability—at 8.1% of the FLOPs of a target-scale search.

## Datasets

The source data comes from the OLMo 2 DOLMino mix released by the Allen Institute for AI (Ai2). [Ai2 DOLMino Mix](https://huggingface.co/datasets/allenai/dolmino-mix-1124)

- **Source data (`sources/`).** The source data covers six domains. DCLM is split into 18 gzip shards. [Source files](https://huggingface.co/datasets/openmuse/RegMix-MT-Datasets/tree/main/sources)
- **Prepared mixtures (`dataset_bin/`).** This directory contains training data for selected mixture configurations. Each dataset follows the domain proportions of its configuration. [Prepared files](https://huggingface.co/datasets/openmuse/RegMix-MT-Datasets/tree/main/dataset_bin)
- **Dataset manifest (`dataset_info.json`).** Steps 2 and 4 generate a manifest that maps dataset names to their training and evaluation files.

To download and train on the prepared `r2006` mixture, go directly to [Step 4](#4-train-the-360m-1b-17b-and-8b-models).

## How to run

Run the commands from the repository root. Steps 2 and 4 require the `hf` CLI and a local copy of `mathinstruct-sample.jsonl` for training-time evaluation. Set `EVAL_FILE` to its absolute path. [Training setup](TRAINING.md)

### 1. Generate 512 data mixtures

```bash
python -m pip install -r requirements-training.txt
python proxy/cpt4math/synthesize_mixture.py \
  --output-manifest output/proxy_dataset_info.json
```

### 2. Train 512 SmolLM2-135M models

```bash
(
  set -eu
  export DATASET_DIR="$PWD/output/proxy-data"
  export EVAL_DATASET=mathinstruct-sample
  mkdir -p "$DATASET_DIR"
  cp "$EVAL_FILE" "$DATASET_DIR/mathinstruct-sample.jsonl"

  # Download and unpack the six source domains.
  hf download openmuse/RegMix-MT-Datasets --repo-type dataset \
    --include 'sources/*.jsonl.gz' --local-dir "$DATASET_DIR"
  gzip -dc "$DATASET_DIR"/sources/dclm-all-*.jsonl.gz > "$DATASET_DIR/sources/dclm.jsonl"
  for domain in flan math pes2o stackexchange wiki; do
    gzip -dc "$DATASET_DIR/sources/$domain.jsonl.gz" > "$DATASET_DIR/sources/$domain.jsonl"
  done

  # Build the 512 training datasets and register the evaluation file.
  python proxy/cpt4math/generate_data_by_dataset_info.py \
    "$DATASET_DIR/sources" "$DATASET_DIR" output/proxy_dataset_info.json \
    --total_count 50000 --seed 41
  python - output/proxy_dataset_info.json > "$DATASET_DIR/dataset_info.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
manifest["mathinstruct-sample"] = {
    "file_name": "mathinstruct-sample.jsonl", "columns": {"prompt": "text"}
}
print(json.dumps(manifest, indent=2))
PY

  # Train the 512 SmolLM2-135M proxy models.
  cd proxy
  MODEL_NAME_OR_PATH=HuggingFaceTB/SmolLM2-135M bash train_batch.sh \
    --keys sample{1..512} --learning-rates 2e-5
)
```

### 3. Fit the regressors and select a mixture

Run this command to test regression fitting and mixture selection on the synthetic data in `data/demo/`. Use measured proxy metrics for experiment reproduction.

```bash
python proxy/train/regmix_train.py \
  --nll_path data/demo/nll.csv \
  --loss_path data/demo/loss.csv \
  --dataset_info data/demo/dataset_info.json \
  --output_dir output/regmix-demo \
  --single_pool
```

### 4. Train the 360M, 1B, 1.7B, and 8B models

Download the prepared `r2006` mixture and run the target model launchers.

```bash
(
  set -eu
  export DATASET_DIR="$PWD/output/target-data"
  export EVAL_DATASET=mathinstruct-sample
  mkdir -p "$DATASET_DIR"
  cp "$EVAL_FILE" "$DATASET_DIR/mathinstruct-sample.jsonl"
  hf download openmuse/RegMix-MT-Datasets dataset_bin/sample_r2006/optimal.jsonl \
    --repo-type dataset --local-dir "$DATASET_DIR"
  cat > "$DATASET_DIR/dataset_info.json" <<'JSON'
{
  "sample_r2006": {"file_name": "dataset_bin/sample_r2006/optimal.jsonl", "columns": {"prompt": "text"}},
  "mathinstruct-sample": {"file_name": "mathinstruct-sample.jsonl", "columns": {"prompt": "text"}}
}
JSON

  cd targets
  MODEL_NAME_OR_PATH=HuggingFaceTB/SmolLM2-360M bash ./baseline_8gpu_smolLM2_360M.sh sample_r2006 --output_dir ../runs/SmolLM2-360M/sample_r2006_seed42
  MODEL_NAME_OR_PATH=meta-llama/Llama-3.2-1B bash ./baseline_8gpu_llama.sh sample_r2006 --output_dir ../runs/Llama-3.2-1B/sample_r2006_seed42
  MODEL_NAME_OR_PATH=HuggingFaceTB/SmolLM2-1.7B bash ./baseline_8gpu_smolLM2.sh sample_r2006 --output_dir ../runs/SmolLM2-1.7B/sample_r2006_seed42
  MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B bash ./baseline_8gpu_llama_8B.sh sample_r2006 --output_dir ../runs/Llama-3.1-8B/sample_r2006_seed42
)
```

## Evaluation

Run from the repository root with the local benchmark snapshots described in [TRAINING.md](TRAINING.md).

```bash
export REGMIX_EVAL_DATA_DIR=/path/to/evaluation-data
PYTHONPATH="$PWD" python proxy/cpt4math/run_lighteval.py \
  --models runs/SmolLM2-1.7B/sample_r2006_seed42 \
  --tasks math general --tp 1 \
  --custom-tasks proxy.cpt4math.tasks
```

Results are saved under the model's `evaluation_results/lighteval` directory. The wrapper skips completed tasks; the archived `arc` alias may rerun because Lighteval records `arc:challenge`.

## Checkpoints

Download the released `r2006` checkpoints from [Hugging Face](https://huggingface.co/models?author=openmuse&search=r2006).
