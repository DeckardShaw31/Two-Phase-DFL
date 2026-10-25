"""
Statistical Inference Suite for PIRP Decision-Focused Learning Benchmark.

Implements rigorous hypothesis testing comparing Two-Phase DFL (Ours) against
all 13 baseline methods across K_test = 30 out-of-sample test instances:
  1. Primary unit of statistical independence: n = 30 seed-averaged test instances.
     Performance is averaged across the 5 independent random seeds per test instance
     prior to paired difference calculation, resolving pseudo-replication.
  2. Non-parametric Paired Wilcoxon Signed-Rank Test (one-sided & two-sided).
  3. Parametric Paired Student's t-Test (one-sided & two-sided).
  4. Step-down Holm-Bonferroni multiplicity correction across all 13 baseline comparisons.
  5. Standardized effect size: Cohen's d_z (paired standardized mean difference).
  6. 95% Confidence Intervals: Parametric Student's t and Non-parametric Bootstrap (10,000 resamples).
  7. Clear differentiation between cross-seed standard deviation (SD_seed)
     and cross-instance standard deviation (SD_inst).

Outputs:
  - results/statistical_tests.csv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def holm_bonferroni_adjustment(p_values: np.ndarray | list[float]) -> np.ndarray:
    """
    Computes step-down Holm-Bonferroni adjusted p-values.

    Given M sorted raw p-values p_(1) <= p_(2) <= ... <= p_(M):
        p_adj,(k) = min(1.0, max_{j <= k} (M - j + 1) * p_(j))
    """
    raw_p = np.asarray(p_values, dtype=np.float64)
    m = len(raw_p)
    if m == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(raw_p)
    adjusted_p = np.empty(m, dtype=np.float64)

    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        step_multiplier = m - rank_idx
        adj_val = min(1.0, float(step_multiplier * raw_p[orig_idx]))
        running_max = max(running_max, adj_val)
        adjusted_p[orig_idx] = min(1.0, running_max)

    return adjusted_p


def compute_bootstrap_ci(
    diffs: np.ndarray,
    alpha: float = 0.05,
    num_resamples: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Computes two-sided bootstrap confidence interval for paired mean difference."""
    n = len(diffs)
    if n <= 1:
        return float(diffs[0]), float(diffs[0])

    rng = np.random.default_rng(seed)
    boot_samples = rng.choice(diffs, size=(num_resamples, n), replace=True)
    boot_means = np.mean(boot_samples, axis=1)

    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return ci_lower, ci_upper


def run_statistical_tests(
    instance_metrics_df: pd.DataFrame,
    target_model: str = "Two-Phase_DFL (Ours)",
    alpha: float = 0.05,
    num_bootstrap: int = 10000,
    output_path: str = "results/statistical_tests.csv",
    save_results: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Computes complete statistical inference table comparing target_model against all baselines.

    Args:
        instance_metrics_df: Relational DataFrame containing instance_id, seed, model, normalized_regret_pct.
        target_model: Name of the reference model (Two-Phase_DFL (Ours)).
        alpha: Nominal significance level (0.05).
        num_bootstrap: Number of bootstrap resamples for CI estimation.
        output_path: Path to save the resulting CSV.
        save_results: Whether to write output CSV to disk.
        seed: Random seed for reproducible bootstrap resampling.

    Returns:
        DataFrame containing complete statistical test results.
    """
    df = instance_metrics_df.copy()

    # Standardize column names
    col_map = {
        "Model": "model",
        "NormalizedRegret_pct": "normalized_regret_pct",
        "Instance_ID": "instance_id",
        "Instance": "instance_id",
        "Seed": "seed",
    }
    for old_col, new_col in col_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    required_cols = ["instance_id", "model", "normalized_regret_pct"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column `{c}` in instance metrics dataframe.")

    has_seed = "seed" in df.columns
    unique_models = [m for m in df["model"].unique() if m != target_model]
    unique_models.sort()

    if target_model not in df["model"].unique():
        raise ValueError(f"Target model `{target_model}` not found in metrics dataframe.")

    # Primary Statistical Unit: Seed-Averaged Per-Instance Performance (n = 30)
    # Shape per model: (30,)
    df_inst_means = (
        df.groupby(["model", "instance_id"])["normalized_regret_pct"]
        .mean()
        .reset_index()
    )

    # Uncertainty Quantification:
    # 1. Cross-Seed SD (SD_seed): Standard deviation of seed-level mean performance across seeds
    sd_seed_dict: dict[str, float] = {}
    if has_seed and df["seed"].nunique() > 1:
        df_seed_means = (
            df.groupby(["model", "seed"])["normalized_regret_pct"]
            .mean()
            .reset_index()
        )
        for m in df["model"].unique():
            vals = df_seed_means[df_seed_means["model"] == m]["normalized_regret_pct"].values
            sd_seed_dict[m] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    else:
        for m in df["model"].unique():
            sd_seed_dict[m] = 0.0

    # 2. Cross-Instance SD (SD_inst): Standard deviation of seed-averaged performance across instances
    sd_inst_dict: dict[str, float] = {}
    for m in df["model"].unique():
        vals = df_inst_means[df_inst_means["model"] == m]["normalized_regret_pct"].values
        sd_inst_dict[m] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    # Reference model vectors
    ours_df = df_inst_means[df_inst_means["model"] == target_model].sort_values("instance_id")
    ours_inst = ours_df["normalized_regret_pct"].values
    n_instances = len(ours_inst)

    raw_results: list[dict[str, Any]] = []

    for m_name in unique_models:
        base_df = df_inst_means[df_inst_means["model"] == m_name].sort_values("instance_id")
        base_inst = base_df["normalized_regret_pct"].values

        if len(base_inst) != n_instances:
            raise ValueError(
                f"Mismatched instance count between `{target_model}` ({n_instances}) and `{m_name}` ({len(base_inst)})."
            )

        diffs = base_inst - ours_inst  # H_1: Baseline regret > Ours regret => diffs > 0
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if n_instances > 1 else 0.0
        se_diff = std_diff / np.sqrt(n_instances) if n_instances > 0 else 0.0

        # Effect Size (Cohen's d_z for paired designs)
        cohen_dz = float(mean_diff / std_diff) if std_diff > 1e-9 else 0.0

        # Parametric Paired Student's t-test
        if se_diff > 1e-9:
            t_stat = float(mean_diff / se_diff)
            df_t = n_instances - 1
            t_p_one = float(stats.t.sf(t_stat, df=df_t))
            t_p_two = float(2.0 * stats.t.sf(abs(t_stat), df=df_t))
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_t))
            ci_lower = float(mean_diff - t_crit * se_diff)
            ci_upper = float(mean_diff + t_crit * se_diff)
        else:
            t_stat = 0.0
            t_p_one = 1.0
            t_p_two = 1.0
            ci_lower = mean_diff
            ci_upper = mean_diff

        # Bootstrap 95% Confidence Interval
        boot_ci_lower, boot_ci_upper = compute_bootstrap_ci(
            diffs=diffs,
            alpha=alpha,
            num_resamples=num_bootstrap,
            seed=seed,
        )

        # Non-parametric Paired Wilcoxon Signed-Rank Test
        nonzero_diffs = diffs[~np.isclose(diffs, 0.0, atol=1e-8)]
        if len(nonzero_diffs) >= 1:
            try:
                w_one = stats.wilcoxon(base_inst, ours_inst, alternative="greater", zero_method="wilcox")
                w_two = stats.wilcoxon(base_inst, ours_inst, alternative="two-sided", zero_method="wilcox")
                w_stat = float(w_one.statistic)
                w_p_one = float(w_one.pvalue)
                w_p_two = float(w_two.pvalue)
            except (ValueError, ZeroDivisionError):
                w_stat = 0.0
                w_p_one = 1.0
                w_p_two = 1.0
        else:
            w_stat = 0.0
            w_p_one = 1.0
            w_p_two = 1.0

        raw_results.append({
            "Baseline_Model": m_name,
            "Ours_Model": target_model,
            "N_Instances": n_instances,
            "Mean_Regret_Baseline_pct": float(np.mean(base_inst)),
            "Mean_Regret_Ours_pct": float(np.mean(ours_inst)),
            "Mean_Difference_pct": mean_diff,
            "CI_95_Lower": ci_lower,
            "CI_95_Upper": ci_upper,
            "Bootstrap_CI_95_Lower": boot_ci_lower,
            "Bootstrap_CI_95_Upper": boot_ci_upper,
            "Cohen_dz": cohen_dz,
            "Wilcoxon_Stat": w_stat,
            "Wilcoxon_p_raw": w_p_one,
            "Wilcoxon_p_two_sided": w_p_two,
            "T_Stat": t_stat,
            "T_p_raw": t_p_one,
            "T_p_two_sided": t_p_two,
            "SD_Seed_Baseline": sd_seed_dict.get(m_name, 0.0),
            "SD_Instance_Baseline": sd_inst_dict.get(m_name, 0.0),
            "SD_Seed_Ours": sd_seed_dict.get(target_model, 0.0),
            "SD_Instance_Ours": sd_inst_dict.get(target_model, 0.0),
        })

    # Apply Multiplicity Adjustments (Step-down Holm-Bonferroni across all 13 baselines)
    w_p_raw = [r["Wilcoxon_p_raw"] for r in raw_results]
    t_p_raw = [r["T_p_raw"] for r in raw_results]

    w_p_holm = holm_bonferroni_adjustment(w_p_raw)
    t_p_holm = holm_bonferroni_adjustment(t_p_raw)

    for idx, r in enumerate(raw_results):
        r["Wilcoxon_p_Holm"] = float(w_p_holm[idx])
        r["T_p_Holm"] = float(t_p_holm[idx])
        r["Significant_Holm_05"] = bool(w_p_holm[idx] < 0.05 and t_p_holm[idx] < 0.05)
        r["Significant_Holm_01"] = bool(w_p_holm[idx] < 0.01 and t_p_holm[idx] < 0.01)

    result_df = pd.DataFrame(raw_results)

    # Reorder columns logically
    ordered_cols = [
        "Baseline_Model",
        "Ours_Model",
        "N_Instances",
        "Mean_Regret_Baseline_pct",
        "Mean_Regret_Ours_pct",
        "Mean_Difference_pct",
        "CI_95_Lower",
        "CI_95_Upper",
        "Bootstrap_CI_95_Lower",
        "Bootstrap_CI_95_Upper",
        "Cohen_dz",
        "Wilcoxon_Stat",
        "Wilcoxon_p_raw",
        "Wilcoxon_p_Holm",
        "T_Stat",
        "T_p_raw",
        "T_p_Holm",
        "SD_Seed_Baseline",
        "SD_Instance_Baseline",
        "SD_Seed_Ours",
        "SD_Instance_Ours",
        "Significant_Holm_05",
        "Significant_Holm_01",
    ]
    result_df = result_df[ordered_cols]

    if save_results:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.to_csv(output_path, index=False)
        print(f"\nSaved statistical test results to: {output_path}")

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIRP Benchmark Statistical Testing Suite")
    parser.add_argument("--input", type=str, default="results/instance_level_metrics.csv")
    parser.add_argument("--output", type=str, default="results/statistical_tests.csv")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file `{args.input}` not found.", file=sys.stderr)
        sys.exit(1)

    df_metrics = pd.read_csv(args.input)
    res = run_statistical_tests(
        instance_metrics_df=df_metrics,
        alpha=args.alpha,
        num_bootstrap=args.bootstrap,
        output_path=args.output,
        seed=args.seed,
    )
    print("\n=== STATISTICAL INFERENCE SUMMARY (HOLM-BONFERRONI ADJUSTED) ===")
    print(res[["Baseline_Model", "Mean_Difference_pct", "Cohen_dz", "Wilcoxon_p_Holm", "T_p_Holm", "Significant_Holm_05"]].to_string(index=False))
