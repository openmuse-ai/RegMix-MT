
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按数据比例抽样的类
支持按source比例进行数据抽样
"""
import json
import os
import random
from typing import Dict, Optional


class MixSampler:
    """按数据比例抽样的类"""
    
    def __init__(self, global_rate: float = 1.0, source_factors: Optional[Dict[str, float]] = None):
        """
        初始化抽样器
        
        Args:
            global_rate: 全局抽样比例，默认1.0（100%）
            source_factors: 各源系数字典，key为source前缀，value为系数
        """
        self.global_rate = global_rate
        factors = source_factors or {}
        total = sum([v for v in factors.values() if v > 0])
        if total > 0:
            # 归一化到和为1（只保留正数）
            self.source_factors = {k: (v / total if v > 0 else 0.0) for k, v in factors.items()}
        else:
            # 全为0或者全部非正
            self.source_factors = {k: 0.0 for k in factors}
        self.seen = 0
        self.selected = 0
    
    def pick_prefix(self, src: str) -> Optional[str]:
        """提取source的前缀"""
        if not isinstance(src, str) or not src:
            return None
        return src.split("/", 1)[0].lower()
    
    def extract_text(self, obj: dict) -> str:
        """提取text内容"""
        # 外层 text
        t = obj.get("text", "")
        if not isinstance(t, str):
            return ""
        s = t.lstrip()
        # 若外层 text 是 JSON 串，优先取内层的 text
        if s.startswith("{"):
            try:
                inner = json.loads(t)
                if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                    return inner["text"]
            except json.JSONDecodeError:
                pass
        return t
    
    def should_sample(self, prefix: str) -> bool:
        """
        判断是否应该抽样
        
        Args:
            prefix: source前缀
            
        Returns:
            是否应该抽样
        """
        if not prefix:
            return False
        
        # 最终抽样概率 = 全局比例 × 源系数；未知前缀默认不抽
        factor = self.source_factors.get(prefix, 0.0)
        if factor <= 0.0:
            return False
        
        p = self.global_rate * factor
        return random.random() < p
    
    def process_file(self, input_dir, output_path, total_count=None, verbose=True, seed=42):
        """
        基于已打乱的source文件，按比例直接截取前N条样本。
        :param input_dir: 目录/多源jsonl每文件为一个source
        :param output_path: 输出的jsonl文件路径
        :param total_count: 总采样数，必须指定
        :param verbose: 是否输出记录信息
        :param seed: 随机种子，用于最终样本顺序
        """
        import numpy as np

        if total_count is None:
            raise ValueError("process_file需要明确指定total_count（即目标采样总数）")

        all_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jsonl")])
        if not all_files:
            raise ValueError(f"在目录 {input_dir} 中未找到任何 .jsonl 文件")

        all_sources = [os.path.splitext(f)[0].lower() for f in all_files]
        weights = np.array([self.source_factors.get(s, 0.0) for s in all_sources], dtype=float)
        wsum = weights.sum()
        if wsum == 0:
            raise ValueError("All source_factors are zero; check source_weights.")
        weights = weights / wsum

        ideal = weights * total_count
        counts = np.floor(ideal).astype(int)
        remainder = int(total_count - counts.sum())
        if remainder > 0:
            fractional = ideal - counts
            order = np.argsort(-fractional)
            for idx in order[:remainder]:
                counts[idx] += 1

        sampler_rng = random.Random(seed)
        samples = []
        actual_counts = {}

        for filename, source, target in zip(all_files, all_sources, counts.tolist()):
            if target <= 0:
                actual_counts[source] = 0
                continue

            input_path = os.path.join(input_dir, filename)
            selected = 0
            try:
                with open(input_path, "r", encoding="utf-8") as fin:
                    for line in fin:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = self.extract_text(obj)
                        if not text:
                            continue
                        samples.append({"text": text, "source": source})
                        selected += 1
                        if selected >= target:
                            break
            except FileNotFoundError:
                if verbose:
                    print(f"[WARN] 未找到文件 {input_path}，跳过")
                selected = 0

            actual_counts[source] = selected
            if verbose and selected < target:
                print(f"  [WARN] source={source} 仅取得 {selected}/{target} 条记录")

        sampler_rng.shuffle(samples)

        with open(output_path, "w", encoding="utf-8") as fout:
            for obj in samples:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

        total_actual = len(samples)
        target_counts_map = {src: int(cnt) for src, cnt in zip(all_sources, counts.tolist())}
        actual_ratios = {src: (actual_counts.get(src, 0) / total_actual if total_actual else 0.0)
                         for src in all_sources}

        if verbose:
            print(f"[INFO] 输出样本 {total_actual} 条 --> {output_path}", flush=True)

        return {
            "output_path": output_path,
            "target_counts": target_counts_map,
            "actual_total": total_actual,
            "actual_counts": actual_counts,
            "actual_ratios": actual_ratios
        }
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return {
            "seen": self.seen,
            "selected": self.selected
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.seen = 0
        self.selected = 0


# 命令行接口
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="多源按比例采样工具（基于已打乱的源文件）")
    parser.add_argument("input_dir", help="输入目录（包含已shuffle的jsonl文件）")
    parser.add_argument("output_path", help="输出的jsonl文件路径")
    parser.add_argument("--total-count", "-n", type=int, required=True, help="采样的总条数")
    parser.add_argument("--global-rate", type=float, default=1.0, help="全局抽样比例，默认1.0")
    parser.add_argument("--source-factors", type=str, help="source权重映射，JSON字符串")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认42")
    parser.add_argument("--quiet", action="store_true", help="关闭详细日志输出")
    args = parser.parse_args()

    if args.source_factors:
        try:
            source_factors = json.loads(args.source_factors)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"source_factors JSON 解析失败: {exc}")
    else:
        source_factors = {}

    sampler = MixSampler(global_rate=args.global_rate, source_factors=source_factors)
    result = sampler.process_file(
        args.input_dir,
        args.output_path,
        total_count=args.total_count,
        verbose=not args.quiet,
        seed=args.seed,
    )

    print(
        "done. actual_total={actual_total}, target_counts={target_counts}, out={output_path}".format(
            actual_total=result["actual_total"],
            target_counts=result["target_counts"],
            output_path=result["output_path"],
        ),
        flush=True,
    )
