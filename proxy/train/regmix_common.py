"""
Common module for RegMix training and evaluation.
Contains shared classes: Config, DataLoader, MixtureRegressor.
"""
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", message="X does not have valid feature names")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for RegMix training and evaluation."""
    # File paths
    nll_path: str = "output/train/smolLM2_nll.csv"
    loss_path: str = "output/train/smolLM2_output.csv"
    dataset_info_path: str = "dataset_info.json"
    result_125m_path: str = "output/125M/smolLM2.eval.125M.result.csv"
    result_1p7b_path: str = "output/1.7B/smolLM2.eval.1.7B.result.csv"
    output_dir: str = "output"
    model_dir: str = "output/models"
    eval_dir: str = "output/eval"

    # Feature configuration
    feature_names: List[str] = field(default_factory=lambda: [
        "dclm", "flan", "stackexchange", "pes2o", "wiki", "math"
    ])
    loss_cols: Dict[str, str] = field(default_factory=lambda: {
        "pes2o": "pes2o-val-log.eval_loss",
        "wiki": "wiki-val-log.eval_loss",
        "math": "math-val-log.eval_loss",
        "flan": "flan-val-log.eval_loss",
        "stackexchange": "stackexchange-val-log.eval_loss",
        "dclm": "dclm-val-log.eval_loss"
    })
    nonmath_loss_keys: List[str] = field(
        default_factory=lambda: ["pes2o", "wiki", "flan", "stackexchange", "dclm"]
    )

    # Column names
    col_nll: str = "nll"
    col_loss: str = "avg_loss_no_math"
    col_sample_name: str = "sample_name"
    col_math_avg: str = "math_avg"
    col_nonmath_avg: str = "nonmath_avg"
    col_math_norm: str = "math_norm"
    col_nonmath_norm: str = "nonmath_norm"

    # Model hyperparameters
    n_estimators: int = 1000
    learning_rate: float = 0.01
    cv_folds: int = 5

    # Simulation parameters
    num_simulations: int = 100000
    random_seed: int = 42
    top_k: int = 128

    # Weight
    fixed_weight: float = 0.5

    # Evaluation metrics
    math_metrics: List[str] = field(default_factory=lambda: [
        "math_avg", "Math500", "gsm8k 4-shot"
    ])
    nonmath_metrics: List[str] = field(default_factory=lambda: [
        "nonmath_avg", "GPQA 0-shot", "BBH 0-shot", "HellaSwag 10-shot 4096"
    ])
    exclude_pattern: str = "SmolLM2"


class DataLoader:
    """Handles loading and preprocessing of data."""

    def __init__(self, config: Config):
        self.config = config

    def load_training_data(self) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
        """Load NLL, loss data, and mixture information."""
        nll_dict = self._load_nll_data()
        df = self._load_loss_data(nll_dict)
        X_dict = self._load_mixture_info()

        col_name = self.config.col_sample_name
        common = sorted(set(X_dict.keys()) & set(df[col_name]))
        if not common:
            raise ValueError("No common samples found between NLL and mixture data")

        X = np.array([X_dict[n] for n in common])
        df = df.set_index(col_name).loc[common]

        return X, df, X_dict

    def load_validation_data(self, X_dict: Dict, result_path: str) -> pd.DataFrame:
        """Load validation data with normalized scores."""
        path = Path(result_path)
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")

        df = pd.read_csv(path)
        df = df[~df["dir"].str.contains(self.config.exclude_pattern)].copy()
        df[self.config.col_sample_name] = df["dir"].str.replace(
            "_lr2e-5", "", regex=False
        )
        df = df[df[self.config.col_sample_name].isin(X_dict.keys())]

        if df.empty:
            raise ValueError("No valid validation samples found")

        # Compute averages (mmlu excluded due to threshold effect)
        df[self.config.col_math_avg] = df[["Math500", "gsm8k 4-shot"]].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
        df[self.config.col_nonmath_avg] = df[
            ["GPQA 0-shot", "BBH 0-shot", "HellaSwag 10-shot 4096"]
        ].apply(pd.to_numeric, errors="coerce").mean(axis=1)

        # Normalize
        math_col = self.config.col_math_avg
        nonmath_col = self.config.col_nonmath_avg
        df[self.config.col_math_norm] = (df[math_col] - df[math_col].mean()) / df[math_col].std()
        df[self.config.col_nonmath_norm] = (
            (df[nonmath_col] - df[nonmath_col].mean()) / df[nonmath_col].std()
        )

        return df

    def _load_nll_data(self) -> Dict[str, float]:
        path = Path(self.config.nll_path)
        if not path.exists():
            raise FileNotFoundError(f"NLL file not found: {path}")

        nll_dict = {}
        with open(path) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    name = parts[0].replace("_lr2e-5", "")
                    try:
                        nll_dict[name] = float(parts[2])
                    except ValueError:
                        logger.warning(f"Invalid NLL value for {name}")
        return nll_dict

    def _load_loss_data(self, nll_dict: Dict[str, float]) -> pd.DataFrame:
        path = Path(self.config.loss_path)
        if not path.exists():
            raise FileNotFoundError(f"Loss file not found: {path}")

        df = pd.read_csv(path)
        df[self.config.col_sample_name] = df["log"].apply(lambda x: x.split("_")[0])
        df[self.config.col_nll] = df[self.config.col_sample_name].map(nll_dict)

        nonmath_loss_cols = [
            self.config.loss_cols[k] for k in self.config.nonmath_loss_keys
        ]
        df[self.config.col_loss] = df[nonmath_loss_cols].mean(axis=1)
        df = df[df[self.config.col_nll].notna()].copy()

        return df

    def _load_mixture_info(self) -> Dict[str, List[float]]:
        path = Path(self.config.dataset_info_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset info file not found: {path}")

        with open(path) as f:
            dataset_info = json.load(f)

        return {
            n: [float(x) for x in info["mixture"].split(",")]
            for n, info in dataset_info.items()
            if "mixture" in info
        }


class MixtureRegressor:
    """Trains and manages NLL and Loss regression models."""

    def __init__(self, config: Config):
        self.config = config
        self.model_nll: Optional[lgb.LGBMRegressor] = None
        self.model_loss: Optional[lgb.LGBMRegressor] = None
        self.nll_stats: Optional[Tuple[float, float]] = None
        self.loss_stats: Optional[Tuple[float, float]] = None
        self.cv_results: Optional[Dict] = None

    def fit(
        self, X: np.ndarray, df: pd.DataFrame, run_cv: bool = True
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Train both models, optionally with cross-validation."""
        y_nll = df[self.config.col_nll].values
        y_loss = df[self.config.col_loss].values

        cv_results = None
        if run_cv:
            cv_results = {}
            logger.info("Cross-validating NLL model...")
            cv_results["nll"] = self._cross_validate(X, y_nll)
            logger.info(f"  CV Spearman: {cv_results['nll']['mean']:.3f} "
                        f"+/- {cv_results['nll']['std']:.3f}")

            logger.info("Cross-validating Loss model...")
            cv_results["loss"] = self._cross_validate(X, y_loss)
            logger.info(f"  CV Spearman: {cv_results['loss']['mean']:.3f} "
                        f"+/- {cv_results['loss']['std']:.3f}")

            self.cv_results = cv_results

        self._fit_models(X, y_nll, y_loss)

        if run_cv:
            logger.info("Model NLL trained")
            logger.info("Model Loss trained")

        return cv_results

    def _fit_models(
        self, X: np.ndarray, y_nll: np.ndarray, y_loss: np.ndarray
    ) -> None:
        """Internal method to fit both models."""
        self.nll_stats = (float(y_nll.mean()), float(y_nll.std()))
        self.loss_stats = (float(y_loss.mean()), float(y_loss.std()))

        nll_norm = (y_nll - self.nll_stats[0]) / self.nll_stats[1]
        loss_norm = (y_loss - self.loss_stats[0]) / self.loss_stats[1]

        self.model_nll = self._train_model(X, nll_norm)
        self.model_loss = self._train_model(X, loss_norm)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict NLL and Loss (normalized)."""
        if self.model_nll is None or self.model_loss is None:
            raise RuntimeError("Models not trained. Call fit() or load() first.")
        return self.model_nll.predict(X), self.model_loss.predict(X)

    def save(self, path: str) -> None:
        """Save trained models to file."""
        if self.model_nll is None or self.model_loss is None:
            raise RuntimeError("Models not trained. Call fit() first.")

        state = {
            "model_nll": self.model_nll,
            "model_loss": self.model_loss,
            "nll_stats": self.nll_stats,
            "loss_stats": self.loss_stats,
            "cv_results": self.cv_results,
            "config": {
                "n_estimators": self.config.n_estimators,
                "learning_rate": self.config.learning_rate,
                "feature_names": self.config.feature_names,
            }
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load trained models from file."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state = joblib.load(path)
        self.model_nll = state["model_nll"]
        self.model_loss = state["model_loss"]
        self.nll_stats = state["nll_stats"]
        self.loss_stats = state["loss_stats"]
        self.cv_results = state.get("cv_results")

        logger.info(f"Model loaded from {path}")

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> lgb.LGBMRegressor:
        model = lgb.LGBMRegressor(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            verbosity=-1
        )
        model.fit(X, y)
        return model

    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        kf = KFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_seed
        )
        scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            mean, std = y_train.mean(), y_train.std()
            y_train_norm = (y_train - mean) / std

            model = self._train_model(X_train, y_train_norm)
            pred = model.predict(X_val)
            pred_denorm = pred * std + mean

            r, _ = spearmanr(pred_denorm, y_val)
            scores.append(r)

        return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}


class TwoStageOptimizer:
    """
    Two-stage optimization using Pareto front.

    Stage 1: Optimize NLL only to find dclm range from top performers
    Stage 2: Find Pareto front (no weighted sum), output multiple candidates
    """

    REFERENCE_MIXTURES = {
        "olmo2_dolmino": {
            "dclm": 0.472, "flan": 0.166, "stackexchange": 0.025,
            "pes2o": 0.059, "wiki": 0.071, "math": 0.208
        },
        "math_heavy": {
            "dclm": 0.30, "flan": 0.10, "stackexchange": 0.05,
            "pes2o": 0.05, "wiki": 0.10, "math": 0.40
        },
    }

    MIN_DOMAIN_RATIO = {
        "dclm": 0.10,
        "flan": 0.05,
        "stackexchange": 0.01,
        "pes2o": 0.01,
        "wiki": 0.01,
        "math": 0.10,
    }

    def __init__(self, config: Config):
        self.config = config
        self.dclm_idx = config.feature_names.index("dclm")
        self.math_idx = config.feature_names.index("math")

    def _find_pareto_front(self, nll: np.ndarray, loss: np.ndarray) -> np.ndarray:
        """Find Pareto optimal indices (minimize both NLL and Loss)."""
        n = len(nll)
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            # Check if point i is dominated by any other point
            for j in range(n):
                if i != j:
                    # j dominates i if j <= on both and < on at least one
                    if (nll[j] <= nll[i] and loss[j] <= loss[i] and
                        (nll[j] < nll[i] or loss[j] < loss[i])):
                        is_pareto[i] = False
                        break
        return np.where(is_pareto)[0]

    def optimize(
        self,
        regressor: MixtureRegressor,
        prior: np.ndarray,
        n_simulations: int = 100000
    ) -> Dict:
        """Run two-stage optimization with Pareto front."""
        logger.info("=" * 60)
        logger.info("PARETO FRONT OPTIMIZATION")
        logger.info("=" * 60)

        stage1_result = self._stage1_nll_optimization(regressor, prior, n_simulations)

        # Use Stage 1 dclm range as constraint
        dclm_range = (stage1_result["dclm_lower_bound"], stage1_result["dclm_upper_bound"])
        stage2_result = self._stage2_pareto_optimization(
            regressor, prior, n_simulations, dclm_range=dclm_range
        )

        # Use recommended mixture as final, fallback to first representative
        final_mixture = None
        if stage2_result.get("recommended"):
            final_mixture = stage2_result["recommended"]["mixture"]
        elif stage2_result.get("representatives"):
            final_mixture = stage2_result["representatives"][0]["mixture"]

        return {
            "stage1": stage1_result,
            "stage2": stage2_result,
            "final_mixture": final_mixture,
            "pareto_front": stage2_result.get("pareto_front", []),
        }

    def _stage1_nll_optimization(
        self,
        regressor: MixtureRegressor,
        prior: np.ndarray,
        n_simulations: int
    ) -> Dict:
        """Stage 1: Optimize NLL only to find optimal dclm range."""
        logger.info("\n[Stage 1] NLL-only optimization (w=1.0)")
        logger.info("-" * 50)

        samples = np.random.dirichlet(prior * 1, n_simulations)
        pred_nll, _ = regressor.predict(samples)

        top_k = self.config.top_k
        top_indices = np.argsort(pred_nll)[:top_k]
        top_samples = samples[top_indices]

        dclm_values = top_samples[:, self.dclm_idx]
        dclm_min = float(np.min(dclm_values))
        dclm_max = float(np.max(dclm_values))
        dclm_mean = float(np.mean(dclm_values))
        dclm_std = float(np.std(dclm_values))

        dclm_lower = max(0.1, dclm_mean - 1.5 * dclm_std)
        dclm_upper = min(0.9, dclm_mean + 1.5 * dclm_std)

        logger.info(f"Top-{top_k} NLL samples dclm distribution:")
        logger.info(f"  min: {dclm_min:.3f}, max: {dclm_max:.3f}")
        logger.info(f"  mean: {dclm_mean:.3f}, std: {dclm_std:.3f}")
        logger.info(f"  Recommended dclm range: [{dclm_lower:.3f}, {dclm_upper:.3f}]")

        avg_mixture = top_samples.mean(axis=0)
        logger.info(f"\nStage 1 optimal mixture (NLL-only):")
        for name, val in zip(self.config.feature_names, avg_mixture):
            logger.info(f"  {name}: {val:.3f}")

        return {
            "dclm_min": dclm_min,
            "dclm_max": dclm_max,
            "dclm_mean": dclm_mean,
            "dclm_std": dclm_std,
            "dclm_lower_bound": dclm_lower,
            "dclm_upper_bound": dclm_upper,
            "optimal_mixture": dict(zip(self.config.feature_names, avg_mixture.tolist())),
            "top_nll": float(pred_nll[top_indices[0]])
        }

    def _stage2_pareto_optimization(
        self,
        regressor: MixtureRegressor,
        prior: np.ndarray,
        n_simulations: int,
        dclm_range: tuple = None
    ) -> Dict:
        """Stage 2: Pareto front optimization (no weighted sum)."""
        logger.info("\n[Stage 2] Pareto Front Optimization")
        logger.info("-" * 50)

        if dclm_range:
            logger.info(f"Constraint: dclm in [{dclm_range[0]:.3f}, {dclm_range[1]:.3f}]")

        # Generate samples that respect minimum domain ratios
        min_ratios = np.array([self.MIN_DOMAIN_RATIO[f] for f in self.config.feature_names])
        remaining = 1.0 - min_ratios.sum()
        if remaining < 0:
            raise ValueError(f"MIN_DOMAIN_RATIO sum > 1: {min_ratios.sum()}")

        # Dirichlet on remaining budget, then add min ratios
        raw_samples = np.random.dirichlet(prior * 1, n_simulations)
        dirichlet_samples = min_ratios + raw_samples * remaining

        # Add reference mixtures
        ref_samples = []
        for mix in self.REFERENCE_MIXTURES.values():
            ref_samples.append([mix[f] for f in self.config.feature_names])
        ref_samples = np.array(ref_samples)

        # Add grid samples around reference mixtures (perturbations)
        grid_samples = []
        for ref in ref_samples:
            for _ in range(1000):
                # Small perturbation
                noise = np.random.dirichlet(np.ones(len(ref)) * 10) * 0.1
                perturbed = ref * 0.9 + noise
                perturbed = perturbed / perturbed.sum()
                grid_samples.append(perturbed)
        grid_samples = np.array(grid_samples) if grid_samples else np.empty((0, len(prior)))

        # Combine all samples
        samples = np.vstack([dirichlet_samples, ref_samples, grid_samples])

        # Apply dclm constraint if specified
        if dclm_range:
            dclm_vals = samples[:, self.dclm_idx]
            mask = (dclm_vals >= dclm_range[0]) & (dclm_vals <= dclm_range[1])
            samples = samples[mask]
            logger.info(f"Samples after dclm constraint: {len(samples)}")

        # All samples already satisfy min ratio constraints from generation
        logger.info(f"Min ratios enforced: {dict(zip(self.config.feature_names, min_ratios))}")

        pred_nll, pred_loss = regressor.predict(samples)

        logger.info(f"Total samples: {len(samples)}")

        # Find Pareto front
        pareto_idx = self._find_pareto_front(pred_nll, pred_loss)
        logger.info(f"Pareto front size: {len(pareto_idx)}")

        pareto_samples = samples[pareto_idx]
        pareto_nll = pred_nll[pareto_idx]
        pareto_loss = pred_loss[pareto_idx]

        # Sort by NLL (best math first)
        sort_idx = np.argsort(pareto_nll)
        pareto_samples = pareto_samples[sort_idx]
        pareto_nll = pareto_nll[sort_idx]
        pareto_loss = pareto_loss[sort_idx]

        # Build Pareto front list
        pareto_front = []
        for i in range(len(pareto_samples)):
            mixture = dict(zip(self.config.feature_names, pareto_samples[i].tolist()))
            pareto_front.append({
                "nll": float(pareto_nll[i]),
                "loss": float(pareto_loss[i]),
                "mixture": mixture,
                "math_ratio": float(pareto_samples[i, self.math_idx]),
                "dclm_ratio": float(pareto_samples[i, self.dclm_idx]),
            })

        # Select representative points (5 evenly spaced)
        n_representatives = min(5, len(pareto_front))
        step = max(1, len(pareto_front) // n_representatives)
        representatives = [pareto_front[i * step] for i in range(n_representatives)]

        logger.info("\nPareto Front Representatives:")
        logger.info(f"{'#':<3} {'NLL':>8} {'Loss':>8} {'math':>6} {'dclm':>6}")
        logger.info("-" * 40)
        for i, rep in enumerate(representatives):
            logger.info(
                f"{i+1:<3} {rep['nll']:>8.3f} {rep['loss']:>8.3f} "
                f"{rep['math_ratio']:>6.3f} {rep['dclm_ratio']:>6.3f}"
            )

        # Constrained optimization: filter by math target, select best Loss
        math_target = (0.35, 0.45)
        math_ratios = pareto_samples[:, self.math_idx]
        target_mask = (math_ratios >= math_target[0]) & (math_ratios <= math_target[1])

        if target_mask.sum() > 0:
            # Select point with best Loss within math target range
            target_loss = pareto_loss.copy()
            target_loss[~target_mask] = np.inf
            best_idx = np.argmin(target_loss)
            selection_method = f"best Loss in math [{math_target[0]}, {math_target[1]}]"
        else:
            # Fallback: select point closest to math target center
            target_center = (math_target[0] + math_target[1]) / 2
            distances = np.abs(math_ratios - target_center)
            best_idx = np.argmin(distances)
            selection_method = f"closest to math={target_center:.2f}"

        best_mixture_arr = pareto_samples[best_idx]
        best_mixture = dict(zip(self.config.feature_names, best_mixture_arr.tolist()))

        recommended = {
            "nll": float(pareto_nll[best_idx]),
            "loss": float(pareto_loss[best_idx]),
            "mixture": best_mixture,
            "math_ratio": float(best_mixture_arr[self.math_idx]),
            "dclm_ratio": float(best_mixture_arr[self.dclm_idx]),
        }

        logger.info(f"\nRecommended ({selection_method}):")
        logger.info(f"  NLL: {recommended['nll']:.3f}, Loss: {recommended['loss']:.3f}")
        for name, val in best_mixture.items():
            logger.info(f"  {name}: {val:.3f}")

        # Compare with reference mixtures
        logger.info("\nReference Baselines:")
        ref_results = {}
        for name, mix in self.REFERENCE_MIXTURES.items():
            mix_arr = np.array([mix[f] for f in self.config.feature_names]).reshape(1, -1)
            ref_nll, ref_loss = regressor.predict(mix_arr)
            ref_results[name] = {"nll": float(ref_nll[0]), "loss": float(ref_loss[0])}
            logger.info(
                f"  {name}: NLL={ref_nll[0]:.3f}, Loss={ref_loss[0]:.3f}, "
                f"math={mix['math']:.3f}"
            )

        return {
            "pareto_front": pareto_front,
            "representatives": representatives,
            "recommended": recommended,
            "reference_baselines": ref_results,
            "n_pareto_points": len(pareto_front),
        }

    def save_result(self, result: Dict, output_path: str) -> None:
        """Save two-stage optimization result."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"\nSaved two-stage result to {output_path}")
