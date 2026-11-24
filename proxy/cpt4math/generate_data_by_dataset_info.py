import os, json
from pathlib import Path
import numpy as np
from mix_sample import MixSampler

def main():
    import argparse
    parser = argparse.ArgumentParser(description="根据dataset_info.json生成混合数据集")
    parser.add_argument('input_dir', help='输入目录(含各source已打乱的jsonl)')
    parser.add_argument('output_dir', help='输出目录')
    parser.add_argument('dataset_info', help='数据集配置信息 (dataset_info.json)')
    parser.add_argument('--total_count', type=int, default=50000, help='目标总样本数')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    info_file = args.dataset_info
    total_count = args.total_count
    seed = args.seed
    verbose = not args.quiet
    os.makedirs(output_dir, exist_ok=True)

    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)

    emitted = 0
    samples_config = {}
    for i, (key, conf) in enumerate(info.items(), 1):
        if "mixture" not in conf or "file_name" not in conf or "train_groups" not in conf:
            if verbose:
                print(f"跳过无效配置: {key}")
            continue
        mixture = [float(x) for x in conf["mixture"].split(",")]
        train_groups = conf["train_groups"]
        filename = conf["file_name"]
        print(f"mixture:{mixture}")
        print(f"filename:{filename}")
        if len(mixture) != len(train_groups):
            if verbose:
                print(f"{key}: mixture和train_groups数量不等，跳过")
            continue
        output_path = os.path.join(output_dir, filename)
        sampler = MixSampler(global_rate=1.0, source_factors={g: w for g, w in zip(train_groups, mixture)})
        result = sampler.process_file(
            input_dir,
            output_path,
            total_count=total_count,
            verbose=verbose,
            seed=seed+i,
        )
        if verbose:
            print(f"{key}: 写入{result['actual_total']}条到 {output_path}")
        if result['actual_total'] > 0:
            emitted += 1

    print(f"完成，生成{emitted}个文件，配置信息写入")

if __name__ == "__main__":
    main()
