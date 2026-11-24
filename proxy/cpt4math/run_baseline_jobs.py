#!/usr/bin/env python3
"""Launch baseline.sh for each dataset described in dataset_info.json.

Supports two modes:
1. Multi-dataset mode: --keys sample274 sample275 ...
2. LR experiment mode: --keys sample274 --learning-rates 2e-5 5e-5 1e-4 2e-4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Union

from gpu_batch_runner import prepare_gpu_pool, run_jobs

JobType = Union[str, Tuple[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run baseline.sh for multiple datasets")
    parser.add_argument(
        "--dataset-info",
        default="dataset_info.json",
        help="Path to dataset_info.json",
    )
    parser.add_argument(
        "--baseline-script",
        default="baseline.sh",
        help="Path to baseline.sh script",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of concurrent runs (defaults to available GPU count)",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Comma separated GPU ids; defaults to CUDA_VISIBLE_DEVICES or detected GPUs",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        help="Optional subset of dataset keys to run",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to test (enables LR experiment mode)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory to store per-dataset log files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands without executing",
    )
    return parser.parse_args()


def format_lr(lr: float) -> str:
    """Format learning rate for display and filenames."""
    return f"{lr:.0e}".replace("-0", "-").replace("+0", "")


def read_dataset_keys(info_path: str, selected: Iterable[str] | None) -> List[str]:
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    if not isinstance(info, dict):
        raise ValueError("dataset_info.json 顶层必须是对象")

    all_keys = list(info.keys())
    if selected:
        selected_set = set(selected)
        missing = sorted(selected_set - set(all_keys))
        if missing:
            raise KeyError(f"在 {info_path} 中找不到这些 key: {', '.join(missing)}")
        keys = [
            key
            for key in selected
            if not (isinstance(info.get(key), dict) and info[key].get("todo") is False)
        ]
    else:
        keys = [
            key
            for key in all_keys
            if not (isinstance(info.get(key), dict) and info[key].get("todo") is False)
        ]

    if not keys:
        raise ValueError("没有可用的 dataset key")

    print(f"筛选后剩余 {len(keys)} 个 dataset key", flush=True)

    return keys


def main() -> None:
    args = parse_args()

    info_path = os.path.abspath(args.dataset_info)
    script_path = os.path.abspath(args.baseline_script)
    log_dir = os.path.abspath(args.log_dir)

    if not os.path.isfile(info_path):
        raise FileNotFoundError(f"找不到 dataset_info.json: {info_path}")
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"找不到 baseline.sh: {script_path}")

    os.makedirs(log_dir, exist_ok=True)

    keys = read_dataset_keys(info_path, args.keys)

    active_gpu_ids, max_workers = prepare_gpu_pool(args.gpu, args.max_workers)

    print(
        f"检测到 {len(active_gpu_ids)} 张可用 GPU，将使用 {max_workers} 个并发任务: {', '.join(active_gpu_ids)}",
        flush=True,
    )

    # LR experiment mode
    if args.learning_rates:
        learning_rates = args.learning_rates
        jobs: List[JobType] = [
            (key, lr) for key in keys for lr in learning_rates
        ]
        lr_strs = [format_lr(lr) for lr in learning_rates]
        print(f"LR 实验模式: {len(keys)} 个数据集 x {len(learning_rates)} 个学习率", flush=True)
        print(f"数据集: {', '.join(keys)}", flush=True)
        print(f"学习率: {', '.join(lr_strs)}", flush=True)
        print(f"总任务数: {len(jobs)}", flush=True)

        def build_command_lr(job: JobType) -> List[str]:
            key, lr = job  # type: ignore
            lr_str = format_lr(lr)
            output_dir = f"saves/{key}_lr{lr_str}"
            return ["bash", script_path, key, "--learning_rate", str(lr), "--output_dir", output_dir]

        def log_path_lr(job: JobType) -> Path:
            key, lr = job  # type: ignore
            return Path(log_dir) / f"{key}_lr{format_lr(lr)}.log"

        def label_lr(job: JobType) -> str:
            key, lr = job  # type: ignore
            return f"{key}_lr{format_lr(lr)}"

        failures = run_jobs(
            jobs,
            build_command=build_command_lr,
            log_path=log_path_lr,
            label=label_lr,
            active_gpu_ids=active_gpu_ids,
            dry_run=args.dry_run,
            max_workers=max_workers,
        )

        if args.dry_run:
            return

        if failures:
            failed_labels = [label_lr(f) for f in failures]
            print(f"以下任务失败: {', '.join(failed_labels)}", flush=True)
            sys.exit(1)

    # Multi-dataset mode (original behavior)
    else:
        print(f"发现 {len(keys)} 个数据集: {', '.join(keys)}", flush=True)

        failures = run_jobs(
            keys,
            build_command=lambda key: ["bash", script_path, key],
            log_path=lambda key: Path(log_dir) / f"{key}.log",
            label=lambda key: key,
            active_gpu_ids=active_gpu_ids,
            dry_run=args.dry_run,
            max_workers=max_workers,
        )

        if args.dry_run:
            return

        if failures:
            print(f"以下任务失败: {', '.join(failures)}", flush=True)
            sys.exit(1)


if __name__ == "__main__":
    main()

