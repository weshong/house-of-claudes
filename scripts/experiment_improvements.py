"""Experiment 6: Test tournament weighting, shrinkage, new features."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marchmadness.config import CV_SEASONS, PREDICTION_CLIP
from marchmadness.data_loader import load_all
from marchmadness.features.builder import (
    build_training_data, get_feature_columns
)
from marchmadness.evaluation.metrics import compute_all_metrics
from marchmadness.evaluation.cross_val import leave_season_out_cv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def run_cv_config(X, y, seasons, weights=None, C=1.0, shrinkage=0.0, label=""):
    """Run CV with given config and return metrics."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
    ])
    result = leave_season_out_cv(
        model, X, y, seasons, CV_SEASONS,
        sample_weights=weights,
        shrinkage_alpha=shrinkage
    )
    metrics = result["overall"]
    print(f"  {label}: Brier={metrics['brier_score']:.4f}, "
          f"Acc={metrics['accuracy']:.3f}, ECE={metrics.get('ece', 0):.4f}")
    return metrics, result


def main():
    print("Loading data...")
    data = load_all()

    results_table = []

    for gender in ["M", "W"]:
        print(f"\n{'='*60}")
        print(f"  {gender} - Testing improvements")
        print(f"{'='*60}")

        # --- Baseline: seeds_only, no weighting, no shrinkage ---
        print(f"\n--- Baseline (seeds_only) ---")
        df = build_training_data(data, list(range(1985 if gender == "M" else 1998, 2026)),
                                 gender, "seeds_only")
        feat_cols = get_feature_columns(df)
        X = df[feat_cols].values.copy()
        y = df["Label"].values.copy()
        seasons = df["Season"].values.copy()
        weights = df["SampleWeight"].values.copy() if "SampleWeight" in df.columns else None

        col_medians = np.nanmedian(X, axis=0)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0

        m, _ = run_cv_config(X, y, seasons, label="seeds_only LR C=1.0")
        results_table.append((gender, "seeds_only LR C=1.0", m["brier_score"]))

        # --- Test shrinkage values ---
        print(f"\n--- Shrinkage sweep ---")
        for alpha in [0.05, 0.10, 0.15, 0.20]:
            m, _ = run_cv_config(X, y, seasons, shrinkage=alpha,
                                 label=f"seeds_only + shrink={alpha}")
            results_table.append((gender, f"seeds_only + shrink={alpha}", m["brier_score"]))

        # --- Test C values with shrinkage ---
        print(f"\n--- C sweep with shrinkage ---")
        for C in [0.01, 0.1, 1.0]:
            m, _ = run_cv_config(X, y, seasons, C=C, shrinkage=0.10,
                                 label=f"seeds_only C={C} + shrink=0.10")
            results_table.append((gender, f"seeds_only C={C} + shrink=0.10", m["brier_score"]))

        # --- With regular season games + tournament weighting ---
        print(f"\n--- With regular season (tournament 6x weight) ---")
        df_rs = build_training_data(data, list(range(1985 if gender == "M" else 1998, 2026)),
                                    gender, "seeds_only", include_regular_season=True)
        feat_cols_rs = get_feature_columns(df_rs)
        X_rs = df_rs[feat_cols_rs].values.copy()
        y_rs = df_rs["Label"].values.copy()
        seasons_rs = df_rs["Season"].values.copy()
        weights_rs = df_rs["SampleWeight"].values.copy() if "SampleWeight" in df_rs.columns else None

        col_medians = np.nanmedian(X_rs, axis=0)
        for i in range(X_rs.shape[1]):
            mask = np.isnan(X_rs[:, i])
            X_rs[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0

        print(f"  Training data: {len(y_rs)} games ({(weights_rs == 6.0).sum()} tourney, {(weights_rs == 1.0).sum()} regular)")

        m, _ = run_cv_config(X_rs, y_rs, seasons_rs, weights=weights_rs,
                             label="seeds_only + RS (6x weight)")
        results_table.append((gender, "seeds_only + RS (6x weight)", m["brier_score"]))

        m, _ = run_cv_config(X_rs, y_rs, seasons_rs, weights=weights_rs, shrinkage=0.10,
                             label="seeds_only + RS (6x) + shrink=0.10")
        results_table.append((gender, "seeds_only + RS (6x) + shrink=0.10", m["brier_score"]))

        # --- tier1 (seeds + ordinals with new features) ---
        print(f"\n--- tier1 (seeds + improved ordinals) ---")
        df_t1 = build_training_data(data, list(range(1985 if gender == "M" else 1998, 2026)),
                                    gender, "tier1")
        feat_cols_t1 = get_feature_columns(df_t1)
        X_t1 = df_t1[feat_cols_t1].values.copy()
        y_t1 = df_t1["Label"].values.copy()
        seasons_t1 = df_t1["Season"].values.copy()
        weights_t1 = df_t1["SampleWeight"].values.copy() if "SampleWeight" in df_t1.columns else None

        col_medians = np.nanmedian(X_t1, axis=0)
        for i in range(X_t1.shape[1]):
            mask = np.isnan(X_t1[:, i])
            X_t1[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0

        print(f"  Features: {feat_cols_t1}")

        for C in [0.01, 0.1, 1.0]:
            for alpha in [0.0, 0.10]:
                m, _ = run_cv_config(X_t1, y_t1, seasons_t1, C=C, shrinkage=alpha,
                                     label=f"tier1 C={C} shrink={alpha}")
                results_table.append((gender, f"tier1 C={C} shrink={alpha}", m["brier_score"]))

        # --- tier1 + regular season with weighting ---
        print(f"\n--- tier1 + RS (6x weight) ---")
        df_t1_rs = build_training_data(data, list(range(1985 if gender == "M" else 1998, 2026)),
                                       gender, "tier1", include_regular_season=True)
        feat_cols_t1_rs = get_feature_columns(df_t1_rs)
        X_t1_rs = df_t1_rs[feat_cols_t1_rs].values.copy()
        y_t1_rs = df_t1_rs["Label"].values.copy()
        seasons_t1_rs = df_t1_rs["Season"].values.copy()
        weights_t1_rs = df_t1_rs["SampleWeight"].values.copy() if "SampleWeight" in df_t1_rs.columns else None

        col_medians = np.nanmedian(X_t1_rs, axis=0)
        for i in range(X_t1_rs.shape[1]):
            mask = np.isnan(X_t1_rs[:, i])
            X_t1_rs[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0

        print(f"  Training data: {len(y_t1_rs)} games")

        m, _ = run_cv_config(X_t1_rs, y_t1_rs, seasons_t1_rs, weights=weights_t1_rs,
                             C=0.01, shrinkage=0.10,
                             label="tier1 + RS (6x) C=0.01 shrink=0.10")
        results_table.append((gender, "tier1 + RS (6x) C=0.01 shrink=0.10", m["brier_score"]))

        # --- all features + regular season + weighting ---
        print(f"\n--- all features + RS (6x weight) ---")
        df_all_rs = build_training_data(data, list(range(2003, 2026)),
                                        gender, "all", include_regular_season=True)
        feat_cols_all = get_feature_columns(df_all_rs)
        X_all = df_all_rs[feat_cols_all].values.copy()
        y_all = df_all_rs["Label"].values.copy()
        seasons_all = df_all_rs["Season"].values.copy()
        weights_all = df_all_rs["SampleWeight"].values.copy() if "SampleWeight" in df_all_rs.columns else None

        col_medians = np.nanmedian(X_all, axis=0)
        for i in range(X_all.shape[1]):
            mask = np.isnan(X_all[:, i])
            X_all[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0

        print(f"  Training data: {len(y_all)} games, {len(feat_cols_all)} features")

        for C in [0.01, 0.1]:
            m, _ = run_cv_config(X_all, y_all, seasons_all, weights=weights_all,
                                 C=C, shrinkage=0.10,
                                 label=f"all + RS (6x) C={C} shrink=0.10")
            results_table.append((gender, f"all + RS (6x) C={C} shrink=0.10", m["brier_score"]))

    # --- Summary ---
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Gender':<3} {'Config':<45} {'Brier':>8}")
    print("-" * 60)
    for gender, config, brier in sorted(results_table, key=lambda x: (x[0], x[2])):
        marker = " ***" if brier == min(b for g, _, b in results_table if g == gender) else ""
        print(f"{gender:<3} {config:<45} {brier:>8.4f}{marker}")


if __name__ == "__main__":
    main()
