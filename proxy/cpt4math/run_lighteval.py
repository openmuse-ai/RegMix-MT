#!/usr/bin/env python3
"""Batch run lighteval for all benchmarks.

Runs each task (or task group) as a separate lighteval invocation so that
one failure does not lose results from other tasks.  Already-completed
tasks are skipped automatically (use --force to re-run).
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

BBH_SUBTASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation",
    "multistep_arithmetic_two", "navigate", "object_counting",
    "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
]

# Task specs: "suite|task|num_fewshot|truncate_fewshot"
TASK_PRESETS = {
    # Math
    "gsm8k": "custom|gsm8k|5|1",
    "math": "custom|math|4|1",
    # General
    "mmlu": "custom|mmlu|5|1",
    "mmlu_pro": "custom|mmlu_pro|5|1",
    "mmlu_stem": "custom|mmlu_stem|0|1",
    "hellaswag": "custom|hellaswag|0|1",
    "arc": "custom|arc:challenge|0|1",
    "winogrande": "custom|winogrande|0|1",
    "piqa": "custom|piqa|0|1",
    "openbook_qa": "custom|openbook_qa|0|1",
    "commonsense_qa": "custom|commonsense_qa|0|1",
    "trivia_qa": "custom|trivia_qa|0|1",
    "gpqa": "custom|gpqa|0|1",
}

# Add BBH subtasks
for _s in BBH_SUBTASKS:
    TASK_PRESETS[f"bbh:{_s}"] = f"custom|bbh:{_s}|3|1"

TASK_GROUPS = {
    "math": ["gsm8k", "math", "mmlu_stem"],
    "general": [
        "mmlu", "hellaswag",
        "arc", "winogrande",
        "piqa", "openbook_qa", "commonsense_qa", "trivia_qa",
        "gpqa",
    ],
    "bbh": [f"bbh:{s}" for s in BBH_SUBTASKS],
}
TASK_GROUPS["all"] = (
    TASK_GROUPS["math"] + TASK_GROUPS["general"] + TASK_GROUPS["bbh"]
)

# Tasks that should be batched together (same request type, same format)
BATCH_GROUPS = {
    "bbh": [f"bbh:{s}" for s in BBH_SUBTASKS],
}


def find_completed_tasks(output_dir):
    """Scan existing result files and return set of completed task names."""
    pattern = os.path.join(output_dir, "**/results*.json")
    files = glob.glob(pattern, recursive=True)
    completed = set()
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        for task_key in data.get("results", {}):
            parts = task_key.split("|")
            if len(parts) >= 2:
                completed.add(parts[1])
    return completed


def build_cmd(model_path, task_string, output_dir, args):
    """Build the lighteval command string."""
    parts = [
        f'pretrained={model_path}',
        'dtype=bfloat16',
        'vllm',
        f'gpu_memory_utilisation={args.gpu_memory}',
        f'max_model_length={args.max_model_length}',
        f'tensor_parallel_size={args.tp}',
    ]
    if args.enforce_eager:
        parts.append('enforce_eager=True')
    if args.max_num_seqs > 0:
        parts.append(f'max_num_seqs={args.max_num_seqs}')
    model_args = ','.join(parts)
    cmd = (
        f'lighteval accelerate'
        f' --model_args "{model_args}"'
        f' --custom_tasks "{args.custom_tasks}"'
        f' --tasks "{task_string}"'
        f' --output_dir "{output_dir}"'
        f' --save_details'
    )
    if args.batch_size > 0:
        cmd += f' --override_batch_size {args.batch_size}'
    return cmd


def group_tasks_for_execution(task_names):
    """Group tasks into execution batches.

    BBH subtasks are batched together (same format, low risk of failure).
    Other tasks run individually for isolation.

    Returns list of (label, [task_names]) tuples.
    """
    batches = []
    remaining = list(task_names)

    for group_label, group_tasks in BATCH_GROUPS.items():
        batch = [t for t in remaining if t in group_tasks]
        if batch:
            batches.append((group_label, batch))
            remaining = [t for t in remaining if t not in group_tasks]

    for t in remaining:
        batches.append((t, [t]))

    return batches


def _expand_model_paths(paths):
    """Expand directory paths: if a path is a parent dir (no config.json),
    use all its immediate subdirectories as model paths."""
    result = []
    for p in paths:
        p = os.path.abspath(p)
        if (os.path.isdir(p)
                and not os.path.isfile(os.path.join(p, "config.json"))):
            subs = sorted(
                d for d in (os.path.join(p, x) for x in os.listdir(p))
                if os.path.isdir(d)
            )
            if not subs:
                print(f"[WARN] No subdirectories found in {p}",
                      file=sys.stderr)
            result.extend(subs)
        else:
            result.append(p)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run lighteval benchmarks per-task with auto-resume")
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model paths or parent dirs (auto-expands to subdirectories)"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["all"],
        help="Task names or group names (all/math/general/bbh)")
    parser.add_argument("--custom-tasks", default="cpt4math/tasks.py")
    parser.add_argument("--tp", type=int, default=4,
                        help="tensor_parallel_size")
    parser.add_argument("--gpu-memory", type=float, default=0.8)
    parser.add_argument("--max-model-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=-1,
                        help="override_batch_size (-1 = auto)")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs to save memory")
    parser.add_argument("--max-num-seqs", type=int, default=-1,
                        help="Max sequences per vLLM iteration (-1 = default)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run tasks even if results exist")
    parser.add_argument("--gpu", default=None,
                        help="GPU IDs (comma-separated) for multi-GPU parallel eval")
    args = parser.parse_args()

    # Resolve task names from groups
    task_names = []
    for t in args.tasks:
        if t in TASK_GROUPS:
            task_names.extend(TASK_GROUPS[t])
        elif t in TASK_PRESETS:
            task_names.append(t)
        else:
            print(f"Unknown task/group: {t}", file=sys.stderr)
            sys.exit(1)
    task_names = list(dict.fromkeys(task_names))

    print(f"Tasks ({len(task_names)}): {', '.join(task_names)}")

    models = _expand_model_paths(args.models)

    if args.gpu:
        _run_parallel(models, task_names, args)
    else:
        _run_serial(models, task_names, args)


def _run_parallel(models, task_names, args):
    """Dispatch each model to a separate GPU via gpu_batch_runner."""
    from gpu_batch_runner import prepare_gpu_pool, run_jobs

    gpu_ids, max_workers = prepare_gpu_pool(args.gpu, None)
    print(f"[INFO] Parallel eval: {len(models)} models, "
          f"{max_workers} GPUs ({', '.join(gpu_ids)})", file=sys.stderr)

    # Build the base command args (without --models and --gpu)
    base_args = [
        sys.executable, __file__,
        "--custom-tasks", args.custom_tasks,
        "--tasks", *args.tasks,
        "--tp", str(args.tp),
        "--gpu-memory", str(args.gpu_memory),
        "--max-model-length", str(args.max_model_length),
        "--batch-size", str(args.batch_size),
        "--max-num-seqs", str(args.max_num_seqs),
    ]
    if args.enforce_eager:
        base_args.append("--enforce-eager")
    if args.force:
        base_args.append("--force")

    log_dir = Path("logs/lighteval_parallel")
    log_dir.mkdir(parents=True, exist_ok=True)

    failures = run_jobs(
        models,
        build_command=lambda m: base_args + ["--models", m],
        log_path=lambda m: log_dir / f"{os.path.basename(m)}.log",
        label=lambda m: os.path.basename(m),
        active_gpu_ids=gpu_ids,
        max_workers=max_workers,
    )

    total = len(models)
    print(f"\nModels completed: {total - len(failures)}/{total}")
    if failures:
        print(f"Models with failures: "
              f"{', '.join(os.path.basename(f) for f in failures)}")
        sys.exit(1)


def _run_serial(models, task_names, args):
    """Run models one by one (original behavior)."""
    total_models = len(models)
    model_failures = []

    for i, model_path in enumerate(models):
        model_path = os.path.abspath(model_path)
        model_name = os.path.basename(model_path)
        output_dir = os.path.join(
            model_path, "evaluation_results", "lighteval")

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total_models}] {model_name}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")

        # Check which tasks are already done
        if not args.force:
            completed = find_completed_tasks(output_dir)
        else:
            completed = set()

        # Build execution batches
        batches = group_tasks_for_execution(task_names)

        succeeded = []
        failed = []
        skipped = []

        for batch_label, batch_tasks in batches:
            # Filter out completed tasks
            todo = [t for t in batch_tasks
                    if t not in completed]
            if not todo:
                skipped.extend(batch_tasks)
                print(f"  SKIP  {batch_label} "
                      f"({len(batch_tasks)} tasks, already done)")
                continue

            task_string = ",".join(TASK_PRESETS[t] for t in todo)
            cmd = build_cmd(model_path, task_string, output_dir, args)

            print(f"\n  RUN   {batch_label} ({len(todo)} tasks)")
            print(f"  CMD:  {cmd}\n")

            rc = os.system(cmd)
            if rc != 0:
                failed.extend(todo)
                print(f"  FAIL  {batch_label}")
            else:
                succeeded.extend(todo)
                print(f"  OK    {batch_label}")

        # Summary for this model
        print(f"\n--- {model_name} ---")
        print(f"  Succeeded: {len(succeeded)}  "
              f"Skipped: {len(skipped)}  "
              f"Failed: {len(failed)}")
        if failed:
            print(f"  Failed tasks: {', '.join(failed)}")
            model_failures.append(model_name)

    print(f"\nModels completed: "
          f"{total_models - len(model_failures)}/{total_models}")
    if model_failures:
        print(f"Models with failures: {', '.join(model_failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
