import argparse
import json
import random
from pathlib import Path

import numpy as np


class DataProportionGenerator:
    """
    生成数据比例的类，完全复制自 generate_weights_dirichlet 函数。
    """
    
    def __init__(self,
                 temp=0.5,
                 min_strength=0.1,
                 max_strength=5.0,
                 sample_multiplier=100,
                 maximum_usage=2,
                 minimum=2e-4,
                 seed=42,
                 verbose=True):
        """
        初始化参数。
        
        Args:
            temp (float): Temperature for the prior distribution, if your distribution is too skewed, 
                        you can use a temperature to smooth it.
            min_strength (float): The minimum strength for the dirichlet distribution.
                                With a small value, the distribution will be more concentrated.
            max_strength (float): The maximum strength for the dirichlet distribution.
                                With a large value, the distribution will be more uniform.
            sample_multiplier (int): We first sample this times more samples than randomly select some of them.
            maximum_usage (float): How many epochs are allowed for each domain for the large-scale model training.
            minimum (float): Minimum sampling rate to ensure statistical significance.
            seed (int): Random seed.
            verbose (bool): Whether to print progress information.
        """
        self.TEMP = temp
        self.MIN_STRENGH = min_strength
        self.MAX_STRENGH = max_strength
        self.SAMPLE_MULTIPLIER = sample_multiplier
        self.MAXIMUM_USAGE = maximum_usage
        self.MINIMUM = minimum
        self.verbose = verbose
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
    
    def sort_and_deduplicate(self, data, threshold=1e-5):
        """
        Remove identify configs to avoid duplicated training.
        """
        arr = np.array(data)
        sorted_indices = np.lexsort(arr.T)
        sorted_arr = arr[sorted_indices]
        result = [sorted_arr[0]]
        
        for i in range(1, len(sorted_arr)):
            diff = np.sum(np.abs(sorted_arr[i] - result[-1]))
            if diff > threshold:
                result.append(sorted_arr[i])
        
        return result
    
    def generate_weights_dirichlet(self, prior_dist,
                                   train_groups,
                                   minimum_number,
                                   num_samples=128, 
                                   enable_bound=True,
                                   temperature=1.0):

        final_samples = []
        
        if enable_bound:
            # generate the bound for reject sampling
            number_bound = []
            for i in range(len(prior_dist)):
                # the token cannot be used more than 4 times
                number_bound.append([0.0,
                                 min(prior_dist[i] * self.MAXIMUM_USAGE, 1.0)])
        else:
            number_bound = None
            
        # apply temperature
        if temperature < 1.0:
            prior_dist = prior_dist ** self.TEMP
            prior_dist = prior_dist / np.sum(prior_dist)
            print("\n\nWith temperature: ", prior_dist)

        print("\n\nThe domain usage bound (maximum domain weight): ")
        # print the bound for each group
        for i in range(len(prior_dist)):
            print(f"{train_groups[i]}: {number_bound[i][1]}")

        # combine reject sampling with dirichlet distribution
        for i in range(num_samples * self.SAMPLE_MULTIPLIER):
            if self.MIN_STRENGH == self.MAX_STRENGH:
                samples = np.random.dirichlet(prior_dist * self.MIN_STRENGH, 1)
            else:
                samples = []
                min_strength_log = np.log10(self.MIN_STRENGH)
                max_strength_log = np.log10(self.MAX_STRENGH)
                for strength in np.logspace(min_strength_log, max_strength_log, 15):
                    # add a noise to the strength
                    samples_per_strength = np.random.dirichlet(prior_dist * strength, 1)
                    samples.append(samples_per_strength)
                # random sample one
                samples = random.choice(samples)
            # if there is a bound, the bound is a list of tuples indicating the lower and upper bound of each group
            ensure_flag = True
            if number_bound is not None:
                for j in range(len(samples[0])):
                    if samples[0][j] < number_bound[j][0] or samples[0][j] > number_bound[j][1]:
                        ensure_flag = False
                        break
            if ensure_flag is False:
                continue
            # post normalization, set zero for the number less than minimum_number
            samples = np.where(samples < minimum_number, 0.0, samples)
            # round samples into the same scale of minimum_number
            samples = samples / np.sum(samples, axis=1).reshape(-1, 1)
            samples = np.round(samples / minimum_number) * minimum_number
            # add the samples to the final_samples
            final_samples.append(samples[0])

        # remove the samples with the nearly same values
        print("\nThe number of avaiable samples: ", len(final_samples))
        # deduplicate the samples
        final_samples = self.sort_and_deduplicate(np.array(final_samples))
        # remove the samples with the nearly same values
        print("The number of deduplicated samples: ", len(final_samples))
        selected_samples = random.sample(final_samples, num_samples)
        print("The number of selected samples: ", len(selected_samples))
        selected_samples = np.stack(selected_samples, axis=0)
        return selected_samples


def write_proxy_manifest(output_path):
    train_groups = ["dclm", "flan", "stackexchange", "pes2o", "wiki", "math"]
    prior_dist = np.array([40.2, 16.6, 2.45, 5.85, 7.11, 27.8])
    prior_dist = prior_dist / np.sum(prior_dist)
    weights = DataProportionGenerator(seed=42, verbose=False).generate_weights_dirichlet(
        prior_dist,
        train_groups,
        minimum_number=2e-4,
        num_samples=512,
        enable_bound=True,
        temperature=0.5,
    )

    manifest = {
        f"sample{i}": {
            "mixture": ",".join(f"{value:.6f}" for value in row),
            "train_groups": train_groups,
            "file_name": f"sample_{i:03d}.jsonl",
            "columns": {"prompt": "text"},
        }
        for i, row in enumerate(weights, start=1)
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} mixture entries to {output_path}")


def main(argv=None):
    """
    使用示例：展示如何使用 DataProportionGenerator 类生成数据比例。
    """
    parser = argparse.ArgumentParser(description="Generate example or proxy mixture weights.")
    parser.add_argument(
        "--output-manifest",
        metavar="PATH",
        help="write the canonical 512-mixture proxy manifest to PATH",
    )
    args = parser.parse_args(argv)
    if args.output_manifest:
        write_proxy_manifest(args.output_manifest)
        return
    # 示例：创建一个生成器实例（使用默认参数，与原代码保持一致）
    generator = DataProportionGenerator(
        temp=0.5,
        min_strength=0.1,
        max_strength=5.0,
        sample_multiplier=100,
        maximum_usage=15,
        minimum=2e-4,
        seed=42,
        verbose=True
    )
    
    # 示例：准备先验分布和训练组名称
    # 这里使用一个简化的示例，实际使用时应该使用真实的数据分布
    # dclm, flan, stackexchange, pes2o, wiki, math, 
    train_groups = [
        "dclm",
        "flan",
        "stackexchange",
        "pes2o",
        "wiki",
        "math"
    ]
    
    # 示例：先验分布（各域的比例）
    prior_dist = np.array([
        47.2,
        16.6,
        2.45,
        5.85,
        7.11,
        20.8
    ])
    
    # 归一化先验分布
    prior_dist = prior_dist / np.sum(prior_dist)
    print("Prior distribution after normalization:", prior_dist)
    print(f"Train groups: {train_groups}\n")
    
    # 生成权重
    train_weights = generator.generate_weights_dirichlet(
        prior_dist=prior_dist,
        train_groups=train_groups,
        minimum_number=2e-4,
        num_samples=10,  # 示例只生成5个样本
        enable_bound=True,
        temperature=0.5
    )
    
    # 打印生成的权重
    print("\n生成的权重样本：")
    for i, weights in enumerate(train_weights, 1):
        print(f"\n样本 {i}:")
        for group, weight in zip(train_groups, weights):
            print(f"  {group}: {weight:.6f}")
        # 验证权重和为1.0
        print(f"  权重和: {np.sum(weights):.10f}")


if __name__ == "__main__":
    main()
