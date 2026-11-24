"""
RegMix Training Module.

Usage:
    python train/regmix_train.py [--output_dir output/models]

Functions:
    1. Load 125M training data (NLL + Loss + Mixture)
    2. Train NLL and Loss regression models
    3. 5-fold CV validation
    4. Save models to specified directory
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from regmix_common import Config, DataLoader, MixtureRegressor, TwoStageOptimizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class Trainer:
    """Handles model training workflow."""

    def __init__(self, config: Config):
        self.config = config
        self.loader = DataLoader(config)
        self.regressor = MixtureRegressor(config)
        self.X = None
        self.df = None
        self.X_dict = None

    def train(self) -> Dict:
        """Load data and train models."""
        logger.info("=" * 60)
        logger.info("REGMIX TRAINING")
        logger.info("=" * 60)

        # Load data
        logger.info("\n[Step 1] Loading training data...")
        self.X, self.df, self.X_dict = self.loader.load_training_data()
        logger.info(f"  Training samples: {len(self.X)}")
        logger.info(f"  Features: {self.config.feature_names}")

        # Train with CV
        logger.info("\n[Step 2] Training models with cross-validation...")
        cv_results = self.regressor.fit(self.X, self.df, run_cv=True)

        # Save model
        model_path = Path(self.config.model_dir) / "regmix_model.pkl"
        logger.info(f"\n[Step 3] Saving model to {model_path}...")
        self.regressor.save(str(model_path))

        # Save training stats
        stats = self._generate_stats(cv_results)
        stats_path = Path(self.config.model_dir) / "train_stats.json"
        self._save_stats(stats, str(stats_path))

        return stats

    def run_two_stage(self) -> Dict:
        """Run two-stage optimization after training."""
        if self.X is None:
            raise RuntimeError("Must call train() first")

        logger.info("\n[Step 4] Running two-stage optimization...")

        prior = self.X.mean(axis=0)
        prior = prior / prior.sum()

        np.random.seed(self.config.random_seed)
        two_stage = TwoStageOptimizer(self.config)
        result = two_stage.optimize(self.regressor, prior)

        output_path = Path(self.config.model_dir) / "two_stage_mixture.json"
        two_stage.save_result(result, str(output_path))

        return result

    def _generate_stats(self, cv_results: Dict) -> Dict:
        """Generate training statistics."""
        return {
            "n_samples": len(self.X),
            "n_features": len(self.config.feature_names),
            "feature_names": self.config.feature_names,
            "cv_results": cv_results,
            "model_config": {
                "n_estimators": self.config.n_estimators,
                "learning_rate": self.config.learning_rate,
                "cv_folds": self.config.cv_folds,
            },
            "nll_stats": {
                "mean": self.regressor.nll_stats[0],
                "std": self.regressor.nll_stats[1],
            },
            "loss_stats": {
                "mean": self.regressor.loss_stats[0],
                "std": self.regressor.loss_stats[1],
            },
        }

    def _save_stats(self, stats: Dict, path: str) -> None:
        """Save training statistics to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Training stats saved to {path}")


def run_single_pool(trainer: Trainer) -> Dict:
    """Select a mixture from one seeded pool using the fitted regressors."""
    if trainer.X is None or trainer.df is None:
        raise RuntimeError("Must call train() before single-pool selection")

    X = np.asarray(trainer.X, dtype=float)
    feature_names = trainer.config.feature_names
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != len(feature_names):
        raise ValueError("Training features do not match the configured domains")

    pool_size = trainer.config.num_simulations
    seed = trainer.config.random_seed
    if pool_size <= 0:
        raise ValueError("The single-pool sample count must be positive")
    rng = np.random.RandomState(seed)
    pool = rng.dirichlet(np.ones(len(feature_names)), size=pool_size)
    pred_nll, pred_general_loss = trainer.regressor.predict(pool)
    pred_nll = np.asarray(pred_nll)
    pred_general_loss = np.asarray(pred_general_loss)
    if (pred_nll.shape != (pool_size,) or pred_general_loss.shape != (pool_size,)
            or not np.isfinite(pred_nll).all()
            or not np.isfinite(pred_general_loss).all()):
        raise ValueError("Regressor predictions must be finite vectors for the pool")

    try:
        dclm_idx = feature_names.index("dclm")
    except ValueError as exc:
        raise ValueError("Configured domains must include dclm") from exc
    feasible_indices = np.flatnonzero(pool[:, dclm_idx] >= 0.30)
    if len(feasible_indices) == 0:
        raise ValueError("No Dirichlet candidates satisfy DCLM >= 0.30")

    stage1_count = max(1, int(len(feasible_indices) * 0.10))
    nll_order = np.argsort(pred_nll[feasible_indices], kind="stable")
    stage1_indices = feasible_indices[nll_order[:stage1_count]]
    selected_index = int(stage1_indices[
        np.argmin(pred_general_loss[stage1_indices])
    ])
    mixture = {
        name: float(value)
        for name, value in zip(feature_names, pool[selected_index])
    }
    result = {
        "method": "single_pool",
        "seed": seed,
        "pool_size": pool_size,
        "dclm_min_ratio": 0.30,
        "stage1_top_fraction": 0.10,
        "n_proxy_runs_matched": int(len(X)),
        "n_dclm_feasible": int(len(feasible_indices)),
        "n_stage1_candidates": int(len(stage1_indices)),
        "selected_pool_index": selected_index,
        "final_mixture": mixture,
        "predicted_math_nll_z": float(pred_nll[selected_index]),
        "predicted_general_loss_z": float(pred_general_loss[selected_index]),
    }
    output_path = Path(trainer.config.model_dir) / "single_pool_mixture.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Matched proxy runs: {len(X)}")
    logger.info(f"DCLM-feasible pool candidates: {len(feasible_indices)}")
    logger.info(f"Stage 1 candidates: {len(stage1_indices)}")
    logger.info(f"Selected mixture: {mixture}")
    logger.info(f"Single-pool result saved to {output_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RegMix Training")
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to save trained models and mixture results"
    )
    parser.add_argument(
        "--nll_path", default="output/train/smolLM2_nll.csv",
        help="Path to NLL data"
    )
    parser.add_argument(
        "--loss_path", default="output/train/smolLM2_output.csv",
        help="Path to loss data"
    )
    parser.add_argument(
        "--dataset_info", default="dataset_info.json",
        help="Path to dataset info"
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--two_stage", action="store_true",
        help="Run two-stage optimization after training"
    )
    mode_group.add_argument(
        "--single_pool", action="store_true",
        help="Select from one seeded Dirichlet pool using the fitted regressors"
    )
    parser.add_argument(
        "--n_estimators", type=int, default=1000,
        help="Number of estimators for LightGBM"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01,
        help="Learning rate for LightGBM"
    )
    parser.add_argument(
        "--cv_folds", type=int, default=5,
        help="Number of cross-validation folds"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "output/single_pool" if args.single_pool else "output/models"
    config = Config(
        nll_path=args.nll_path,
        loss_path=args.loss_path,
        dataset_info_path=args.dataset_info,
        model_dir=output_dir,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        cv_folds=args.cv_folds,
    )

    trainer = Trainer(config)
    stats = trainer.train()
    if args.single_pool:
        run_single_pool(trainer)
        return

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Samples: {stats['n_samples']}")
    logger.info(f"  NLL CV: {stats['cv_results']['nll']['mean']:.3f} "
                f"+/- {stats['cv_results']['nll']['std']:.3f}")
    logger.info(f"  Loss CV: {stats['cv_results']['loss']['mean']:.3f} "
                f"+/- {stats['cv_results']['loss']['std']:.3f}")
    logger.info(f"  Model saved to: {config.model_dir}/regmix_model.pkl")

    if args.two_stage:
        trainer.run_two_stage()

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
