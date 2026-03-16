"""Experiment 9B: Test shallow LightGBM on women's features.

Current women's best: tier2 LR C=0.1 at Brier 0.1390.
Test various LGB configs to see if tree models can beat LR on women's data.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marchmadness.data_loader import load_all
from marchmadness.features.builder import build_training_data, get_feature_columns
from marchmadness.evaluation.cross_val import leave_season_out_cv
from marchmadness.config import CV_SEASONS
from lightgbm import LGBMClassifier


def prep_data(df):
    """Prepare X, y, seasons from training DataFrame."""
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].values.copy()
    y = df["Label"].values.copy()
    seasons = df["Season"].values.copy()
    weights = df["SampleWeight"].values.copy() if "SampleWeight" in df.columns else None
    # Impute NaN with column medians
    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0
    return X, y, seasons, feat_cols, weights


def run_config(data, gender, feature_set, start_year, n_estimators, max_depth, learning_rate, label):
    """Build data and run CV for a single config."""
    train_seasons = list(range(start_year, 2026))
    df = build_training_data(data, train_seasons, gender, feature_set)
    if df.empty:
        print(f"  {label}: NO DATA")
        return None

    X, y, seasons, feat_cols, weights = prep_data(df)
    print(f"  {label}: {len(feat_cols)} features, {len(y)} samples")

    model = LGBMClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )

    result = leave_season_out_cv(model, X, y, seasons, CV_SEASONS, sample_weights=weights)
    brier = result["overall"]["brier_score"]
    logloss = result["overall"]["log_loss"]
    print(f"    Brier={brier:.4f}  LogLoss={logloss:.4f}")
    for s in CV_SEASONS:
        if s in result["per_fold"]:
            fb = result["per_fold"][s]["brier_score"]
            print(f"      {s}: Brier={fb:.4f}")
    return result


def main():
    print("Loading data...")
    data = load_all()

    gender = "W"
    print(f"\n{'='*60}")
    print(f"  Women's LightGBM Experiments")
    print(f"  Current best: tier2 LR C=0.1 Brier=0.1390")
    print(f"{'='*60}")

    # Config 1: tier2 LGB d=3 n=300 lr=0.03 (start_year=1998)
    print("\n--- Config 1: tier2 LGB d=3 n=300 lr=0.03 (start=1998) ---")
    run_config(data, gender, "tier2", 1998, n_estimators=300, max_depth=3, learning_rate=0.03,
               label="tier2 LGB d3 n300 lr0.03")

    # Config 2: tier2 LGB d=3 n=400 lr=0.02 (start_year=1998)
    print("\n--- Config 2: tier2 LGB d=3 n=400 lr=0.02 (start=1998) ---")
    run_config(data, gender, "tier2", 1998, n_estimators=400, max_depth=3, learning_rate=0.02,
               label="tier2 LGB d3 n400 lr0.02")

    # Config 3: all LGB d=3 n=300 lr=0.03 (start_year=2003)
    print("\n--- Config 3: all LGB d=3 n=300 lr=0.03 (start=2003) ---")
    run_config(data, gender, "all", 2003, n_estimators=300, max_depth=3, learning_rate=0.03,
               label="all LGB d3 n300 lr0.03")

    # Config 4: iter_eff LGB d=3 n=300 lr=0.03 (start_year=2003)
    print("\n--- Config 4: iter_eff LGB d=3 n=300 lr=0.03 (start=2003) ---")
    run_config(data, gender, "iter_eff", 2003, n_estimators=300, max_depth=3, learning_rate=0.03,
               label="iter_eff LGB d3 n300 lr0.03")

    print("\nDone!")


if __name__ == "__main__":
    main()
