"""Shared helpers for running shell jobs across multiple GPUs."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from pathlib import Path
from threading import Lock
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


def _detect_gpu_ids_from_system() -> List[str]:
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], encoding="utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    lines = [line for line in output.strip().splitlines() if line.strip()]
    return [str(idx) for idx, _ in enumerate(lines)]


def parse_gpu_list(arg: Optional[str]) -> List[str]:
    if arg:
        ids = [gpu.strip() for gpu in arg.split(",") if gpu.strip()]
        if not ids:
            raise SystemExit("--gpu 未提供有效的 GPU ID")
        return ids

    env_setting = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_setting:
        ids = [gpu.strip() for gpu in env_setting.split(",") if gpu.strip()]
        if ids:
            return ids

    return _detect_gpu_ids_from_system()


def prepare_gpu_pool(gpu_arg: Optional[str], max_workers_arg: Optional[int]) -> Tuple[List[str], int]:
    gpu_ids = parse_gpu_list(gpu_arg)
    if not gpu_ids:
        raise SystemExit("未检测到可用 GPU，请通过 --gpu 或 CUDA_VISIBLE_DEVICES 指定")

    max_workers = len(gpu_ids) if max_workers_arg is None else max(1, max_workers_arg)

    return gpu_ids, max_workers


def run_jobs(
    jobs: Iterable[T],
    *,
    build_command: Callable[[T], Sequence[str]],
    log_path: Callable[[T], Path],
    label: Callable[[T], str],
    active_gpu_ids: Sequence[str],
    dry_run: bool = False,
    base_env: Optional[Mapping[str, str]] = None,
    max_workers: Optional[int] = None,
) -> List[T]:
    job_list = list(jobs)
    if not job_list:
        return []

    gpu_count = len(active_gpu_ids)
    if gpu_count == 0:
        raise SystemExit("active_gpu_ids 不能为空")

    worker_limit = max_workers if max_workers is not None else gpu_count
    if worker_limit < 1:
        raise SystemExit("max_workers 必须大于 0")

    if dry_run:
        for idx, job in enumerate(job_list):
            gpu = active_gpu_ids[idx % gpu_count]
            cmd_preview = " ".join(str(part) for part in build_command(job))
            print(
                f"DRY RUN: CUDA_VISIBLE_DEVICES={gpu} {cmd_preview}",
                file=sys.stderr,
            )
        return []

    failures: List[T] = []
    gpu_cycle = cycle(active_gpu_ids)
    gpu_lock = Lock()

    def worker(job: T) -> Tuple[T, str, int, str]:
        with gpu_lock:
            gpu_id = next(gpu_cycle)
        cmd = [str(part) for part in build_command(job)]
        env = os.environ.copy()
        if base_env:
            env.update(base_env)
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        job_label = label(job)
        print(f"[INFO] 开始 {job_label} -> GPU {gpu_id}", file=sys.stderr)
        print(f"[DEBUG] 命令: {' '.join(cmd)}", file=sys.stderr)
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        output = result.stdout or ""
        return job, gpu_id, result.returncode, output

    with ThreadPoolExecutor(max_workers=worker_limit) as executor:
        futures = {executor.submit(worker, job): job for job in job_list}
        for future in as_completed(futures):
            job = futures[future]
            job_label = label(job)
            try:
                _, gpu_id, returncode, output = future.result()
            except Exception as exc:  # pragma: no cover
                print(f"[ERROR] 任务 {job_label} 失败: {exc}", file=sys.stderr)
                failures.append(job)
                continue

            status = "成功" if returncode == 0 else f"失败 (exit={returncode})"
            print(f"[INFO] 完成 {job_label} @ GPU {gpu_id}: {status}", file=sys.stderr)

            log_file = log_path(job)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text(output, encoding="utf-8")
            print(f"[DEBUG] 日志已写入 {log_file}", file=sys.stderr)

            if returncode != 0:
                failures.append(job)

    return failures

