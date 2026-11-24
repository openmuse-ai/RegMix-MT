# Training and regression code

This package contains proxy training and regression scripts, plus launchers for SmolLM2-360M, SmolLM2-1.7B, Llama-3.2-1B, and Llama-3.1-8B. The proxy and target scripts follow Gitee commit `0f1e7403e748f4bf6fe031320ab9d32f2b37ef2d`. `proxy/train_batch.sh` forwards arguments to the original runner. The ratio sampler preserves the archived generation algorithm. The regression demo uses synthetic metrics in `data/demo/`. Exporting new metric CSVs requires a separate process that collects results from proxy training runs. Raw corpus JSONL, the full LlamaFactory manifest, and model checkpoints are external inputs.

## Inputs and environment

Prepare LlamaFactory-compatible data separately. In the canonical Gitee manifest, unpadded keys `sample1` through `sample512` map to `sample_001.jsonl` through `sample_512.jsonl`. Each JSONL record uses the `text` field. The evaluation key `mathinstruct-sample` maps to `mathinstruct-sample.jsonl`. The target recipe key `sample_r2006` maps to `sample_r2006/optimal.jsonl`. Its reported shares are DCLM/FLAN/StackExchange/PES2O/Wiki/Math = 33.1/25.1/1.0/1.0/3.0/36.9%. Place `dataset_info.json` and the JSONL files referenced by the selected training and evaluation keys in `DATASET_DIR`. A 512-proxy run needs the 512 `sample` files and the selected evaluation file. A target run needs `sample_r2006/optimal.jsonl` and the selected evaluation file. Set `MODEL_NAME_OR_PATH` to the local checkpoint and configure LlamaFactory, PyTorch, DeepSpeed, FlashAttention 2, CUDA, and the required GPUs before training.

The original launchers retain provider-specific fallback paths for `MODEL_NAME_OR_PATH` and `DATASET_DIR`. Override them with your own paths. `LLAMAFACTORY_CLI` defaults to `llamafactory-cli`. The target launchers use `CUDA_DEVICES` (default `0,1,2,3`).

For the CPU regression code, install `requirements-training.txt` with Python 3.10 or later. The dependencies are NumPy, pandas, SciPy, scikit-learn, LightGBM, and joblib.

## Generate 512 numeric mixture ratios

The optional `--output-manifest PATH` flag writes 512 recipe entries to JSON. A call without arguments retains the original 10-example demonstration. Use the original caller's seed, prior, domain order, and sampler arguments below to generate the same 512 six-domain ratios as a CSV.

```bash
mkdir -p output
PYTHONPATH=proxy/cpt4math python - <<'PY'
import numpy as np
from synthesize_mixture import DataProportionGenerator

prior = np.array([40.2, 16.6, 2.45, 5.85, 7.11, 27.8])
prior /= prior.sum()
domains = ['dclm', 'flan', 'stackexchange', 'pes2o', 'wiki', 'math']
weights = DataProportionGenerator(seed=42, verbose=False).generate_weights_dirichlet(
    prior, domains, minimum_number=2e-4, num_samples=512,
    enable_bound=True, temperature=0.5
)
np.savetxt(
    'output/proxy_ratios.csv', weights, fmt='%.6f', delimiter=',',
    header=','.join(domains), comments=''
)
PY
```

The sampler normalizes the nonuniform prior, uses temperature 0.5 and bounded Dirichlet draws with its default strength range, zeroes shares below `2e-4`, then renormalizes and quantizes to `2e-4` increments. Row 1 is `sample1`, continuing in order through row 512 as `sample512`. The command writes a six-column CSV using NumPy. The regression CLI takes the dataset manifest described below. The six-decimal ratios match the canonical Gitee manifest at commit `0f1e7403e748f4bf6fe031320ab9d32f2b37ef2d`.

## Run proxy mixtures

`proxy/train_batch.sh` keeps the original batch command structure and calls `proxy/cpt4math/run_baseline_jobs.py`. Run it from `proxy`. It defaults `DATASET_DIR` to `data/mixed` relative to that working directory. Export `DATASET_DIR` to choose another prepared directory. The directory needs `dataset_info.json` and the JSONL files referenced by the selected training and evaluation keys.

With no `--keys` argument, the runner selects every enabled entry in `dataset_info.json` (entries whose `todo` field is `false` are skipped). Pass `--keys` to select one or more mixtures. The upstream source manifest has more than 512 enabled entries, so use the explicit 512 keys below for the paper's matched proxy set. The original command uses GPU 0 and one worker. A single run without `--learning-rates` uses the baseline's historical learning rate of `1e-5`. The paper's 512-run configuration uses `2e-5`.

To run all enabled entries from the default `data/mixed` directory, or one selected mixture:

```bash
cd proxy
export DATASET_DIR=data/mixed
export MODEL_NAME_OR_PATH=/path/to/SmolLM2-135M
export EVAL_DATASET=mathinstruct-sample

bash train_batch.sh
bash train_batch.sh --keys sample274
```

To run the 512 matched mixtures at the paper's learning rate, use Bash so `sample{1..512}` expands to the explicit keys:

```bash
cd proxy
export DATASET_DIR=/path/to/prepared/mixed-data
export MODEL_NAME_OR_PATH=/path/to/SmolLM2-135M
export EVAL_DATASET=mathinstruct-sample
bash train_batch.sh --keys sample{1..512} --learning-rates 2e-5
```

Add `--dry-run` to print commands without starting training. The original scheduler assigns GPUs round-robin when jobs start. This entry point runs one worker on GPU 0. Logs and default outputs are under `proxy/logs/` and `proxy/saves/` when invoked from the documented working directory. The proxy baseline uses `--overwrite_output_dir true` and removes `checkpoint*` after a successful run, so reruns of a key can replace its previous output.

The copied 135M proxy baseline uses seed and data seed 42, 125 steps, per-device batch size 64, gradient accumulation 32, and cutoff length 2048. Its single-GPU run has a nominal `125 × 64 × 32 × 2048 = 524,288,000` input-token slots before packing and partial-batch effects. The paper reports four A100 GPUs and a 1B-token budget. The release preserves the historical 125-step script settings.

## Re-run regression, Stage 1, and Stage 2

To rerun the historical `--two_stage` analysis, provide the original Math NLL and domain-loss CSVs from the 512 matched proxy runs. Use the Gitee `data/mixed/dataset_info.json` from commit `0f1e7403e748f4bf6fe031320ab9d32f2b37ef2d`, or an equivalent manifest with the matched IDs and six-domain `mixture` values in DCLM/FLAN/StackExchange/PES2O/Wiki/Math order. A reduced analysis manifest can retain each key's `mixture` and `train_groups` fields. The regression loader reads `mixture`. LlamaFactory training uses the full `file_name` mappings described above. The canonical manifest is `data/mixed/dataset_info.json` in the pinned source. The rank-2006 recipe is in its `output/top_mixtures.json`. The archived `two_stage_mixture.json` came from a separate archive and is only needed for comparison.

```bash
python proxy/train/regmix_train.py \
  --nll_path /path/to/proxy-math-nll.csv \
  --loss_path /path/to/proxy-domain-loss.csv \
  --dataset_info /path/to/prepared/mixed-data/dataset_info.json \
  --output_dir /path/to/new-regmix-output \
  --two_stage
```

The CLI defaults to seed 42, 1,000 LightGBM estimators, and five-fold cross-validation. It draws 100,000 fresh Dirichlet candidates in each stage. Stage 1 sets the DCLM interval from the top 128 NLL candidates. Stage 2 adds two reference recipes and 1,000 perturbations around each, then applies that interval. The CLI writes the fitted model, statistics, and `two_stage_mixture.json`. A prior CPU verification used archived metric CSVs with the canonical Gitee manifest. It matched 512 mixture IDs and the archived selection, corresponding to rank 2006 when rounded to one decimal percentage. New metric export and language-model training require separate runs with the launchers and external inputs.

A separate call to the existing optimizer API used `n_simulations=20_000` on the same fitted regressor with seed 42. It selected DCLM/FLAN/StackExchange/PES2O/Wiki/Math = 31.39/29.61/1.00/1.00/1.00/36.00%. This counterfactual sensitivity check is separate from the paper's 100,000-draw CLI recipe.

## Run the 12 target-seed jobs

Prepare the full target-training data and models, then set the paths and run these commands from the repository root. Each invocation uses the existing launcher's supported dataset key, learning-rate, seed, and output-directory arguments. The 1.7B and other target launchers use `./ds_zero2_config.json`, so run them from `targets` as shown. The per-model output paths are unique and resolve under the repository's ignored `runs/` directory.

```bash
export DATASET_DIR=/path/to/prepared/mixed-data
export EVAL_DATASET=mathinstruct-sample
export LLAMAFACTORY_CLI=llamafactory-cli
export CUDA_DEVICES=0,1,2,3
cd targets

MODEL_NAME_OR_PATH=/path/to/SmolLM2-360M bash ./baseline_8gpu_smolLM2_360M.sh sample_r2006 --learning_rate 2e-5 --seed 42 --output_dir ../runs/SmolLM2-360M/sample_r2006_seed42
MODEL_NAME_OR_PATH=/path/to/SmolLM2-360M bash ./baseline_8gpu_smolLM2_360M.sh sample_r2006 --learning_rate 2e-5 --seed 123 --output_dir ../runs/SmolLM2-360M/sample_r2006_seed123
MODEL_NAME_OR_PATH=/path/to/SmolLM2-360M bash ./baseline_8gpu_smolLM2_360M.sh sample_r2006 --learning_rate 2e-5 --seed 456 --output_dir ../runs/SmolLM2-360M/sample_r2006_seed456

MODEL_NAME_OR_PATH=/path/to/SmolLM2-1.7B bash ./baseline_8gpu_smolLM2.sh sample_r2006 --learning_rate 2e-5 --seed 42 --output_dir ../runs/SmolLM2-1.7B/sample_r2006_seed42
MODEL_NAME_OR_PATH=/path/to/SmolLM2-1.7B bash ./baseline_8gpu_smolLM2.sh sample_r2006 --learning_rate 2e-5 --seed 123 --output_dir ../runs/SmolLM2-1.7B/sample_r2006_seed123
MODEL_NAME_OR_PATH=/path/to/SmolLM2-1.7B bash ./baseline_8gpu_smolLM2.sh sample_r2006 --learning_rate 2e-5 --seed 456 --output_dir ../runs/SmolLM2-1.7B/sample_r2006_seed456

MODEL_NAME_OR_PATH=/path/to/Llama-3.2-1B bash ./baseline_8gpu_llama.sh sample_r2006 --learning_rate 2e-5 --seed 42 --output_dir ../runs/Llama-3.2-1B/sample_r2006_seed42
MODEL_NAME_OR_PATH=/path/to/Llama-3.2-1B bash ./baseline_8gpu_llama.sh sample_r2006 --learning_rate 2e-5 --seed 123 --output_dir ../runs/Llama-3.2-1B/sample_r2006_seed123
MODEL_NAME_OR_PATH=/path/to/Llama-3.2-1B bash ./baseline_8gpu_llama.sh sample_r2006 --learning_rate 2e-5 --seed 456 --output_dir ../runs/Llama-3.2-1B/sample_r2006_seed456

MODEL_NAME_OR_PATH=/path/to/Llama-3.1-8B bash ./baseline_8gpu_llama_8B.sh sample_r2006 --learning_rate 2e-5 --seed 42 --output_dir ../runs/Llama-3.1-8B/sample_r2006_seed42
MODEL_NAME_OR_PATH=/path/to/Llama-3.1-8B bash ./baseline_8gpu_llama_8B.sh sample_r2006 --learning_rate 2e-5 --seed 123 --output_dir ../runs/Llama-3.1-8B/sample_r2006_seed123
MODEL_NAME_OR_PATH=/path/to/Llama-3.1-8B bash ./baseline_8gpu_llama_8B.sh sample_r2006 --learning_rate 2e-5 --seed 456 --output_dir ../runs/Llama-3.1-8B/sample_r2006_seed456
```

The original target launchers pass `--overwrite_output_dir true` to LlamaFactory. After a successful run, they remove `checkpoint*` subdirectories under the selected output directory. Use a new output path for each run. Target runs require external checkpoints, the JSONL files referenced by the selected training and evaluation keys, LlamaFactory, DeepSpeed, CUDA, and GPUs.

## Lighteval evaluation

Use a separate Python/CUDA environment for the archived Lighteval evaluator. Install the pinned evaluator, its acceleration/task extras, the math parser dependency, and a legacy-compatible Datasets version.

```bash
python -m pip install \
  'lighteval[accelerate,extended_tasks,vllm] @ git+https://github.com/huggingface/lighteval.git@ea46419a93fb390e8f694f7c6c64c1e684487c9d' \
  word2number 'datasets<4'
```

Set `REGMIX_EVAL_DATA_DIR` to a directory containing the prepared `math_tasks_fixed`, `piqa_parquet`, and `gpqa_diamond` benchmark snapshots. The evaluation tasks file uses those exact folder names and preserves its original `/root/autodl-tmp/data` default when the variable is unset. Provide any other task datasets through the usual Lighteval cache or hub access.

The README command runs the `math` and `general` task groups. You can select `bbh` as an additional group. The archived task presets use 0-shot MMLU STEM and 0-shot HellaSwag, while the manuscript reports 5-shot MMLU STEM and 10-shot HellaSwag. HumanEval and MBPP used a separate BigCode evaluation pipeline and are not part of this wrapper.

The archived `arc` group may rerun after completion because its result key `arc:challenge` differs from the task alias `arc`.
