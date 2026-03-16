"""Experiment 7: Test Torvik external ratings."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marchmadness.config import CV_SEASONS
from marchmadness.data_loader import load_all
from marchmadness.features.builder import build_training_data, get_feature_columns
from marchmadness.evaluation.metrics import compute_all_metrics
from marchmadness.evaluation.cross_val import leave_season_out_cv
from marchmadness.models.ensemble import optimize_weights, ensemble_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from itertools import combinations


def prep_and_cv(data, gender, feature_set, C=1.0, start_year=None):
    if start_year is None:
        start_year = 2015  # Torvik data starts 2015
    df = build_training_data(data, list(range(start_year, 2026)), gender, feature_set)
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].values.copy()
    y = df["Label"].values.copy()
    seasons = df["Season"].values.copy()
    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
    ])
    result = leave_season_out_cv(model, X, y, seasons, CV_SEASONS)
    return result, feat_cols


def prep_and_cv_lgb(data, gender, feature_set, start_year=None):
    if start_year is None:
        start_year = 2015
    df = build_training_data(data, list(range(start_year, 2026)), gender, feature_set)
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].values.copy()
    y = df["Label"].values.copy()
    seasons = df["Season"].values.copy()
    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    result = leave_season_out_cv(model, X, y, seasons, CV_SEASONS)
    return result, feat_cols


def main():
    print("Loading data...")
    data = load_all()

    gender = "M"
    print(f"\n{'='*60}")
    print(f"  Men's - Testing Torvik features")
    print(f"{'='*60}")

    models = {}

    # Baseline: seeds_only (from 2015 to match Torvik availability)
    print("\n--- seeds_only (2015+) ---")
    r, fc = prep_and_cv(data, gender, "seeds_only", C=1.0, start_year=2015)
    print(f"  LR C=1.0: Brier={r['overall']['brier_score']:.4f} ({len(fc)} features)")
    models["seeds_LR"] = r

    r, _ = prep_and_cv_lgb(data, gender, "seeds_only", start_year=2015)
    print(f"  LGB: Brier={r['overall']['brier_score']:.4f}")
    models["seeds_LGB"] = r

    # tier1 (seeds + ordinals)
    print("\n--- tier1 (seeds + ordinals, 2015+) ---")
    r, fc = prep_and_cv(data, gender, "tier1", C=0.01, start_year=2015)
    print(f"  LR C=0.01: Brier={r['overall']['brier_score']:.4f} ({len(fc)} features: {fc})")
    models["tier1_LR"] = r

    # torvik feature set (seeds + ordinals + torvik)
    print("\n--- torvik (seeds + ordinals + torvik) ---")
    r, fc = prep_and_cv(data, gender, "torvik", C=0.01, start_year=2015)
    print(f"  LR C=0.01: Brier={r['overall']['brier_score']:.4f} ({len(fc)} features: {fc})")
    models["torvik_LR_001"] = r

    r, fc = prep_and_cv(data, gender, "torvik", C=0.1, start_year=2015)
    print(f"  LR C=0.1: Brier={r['overall']['brier_score']:.4f}")
    models["torvik_LR_01"] = r

    r, fc = prep_and_cv(data, gender, "torvik", C=1.0, start_year=2015)
    print(f"  LR C=1.0: Brier={r['overall']['brier_score']:.4f}")
    models["torvik_LR_1"] = r

    r, _ = prep_and_cv_lgb(data, gender, "torvik", start_year=2015)
    print(f"  LGB: Brier={r['overall']['brier_score']:.4f}")
    models["torvik_LGB"] = r

    # tier2 (seeds + ordinals + elo + stats)
    print("\n--- tier2 (seeds + ordinals + elo + stats, 2015+) ---")
    r, fc = prep_and_cv(data, gender, "tier2", C=0.01, start_year=2015)
    print(f"  LR C=0.01: Brier={r['overall']['brier_score']:.4f} ({len(fc)} features)")
    models["tier2_LR"] = r

    # all (everything including torvik)
    print("\n--- all (everything + torvik, 2015+) ---")
    r, fc = prep_and_cv(data, gender, "all", C=0.01, start_year=2015)
    print(f"  LR C=0.01: Brier={r['overall']['brier_score']:.4f} ({len(fc)} features)")
    models["all_LR"] = r

    # --- Ensembles ---
    y_true = models["seeds_LR"]["oof_labels"]

    print(f"\n{'='*60}")
    print("  Pairwise ensembles")
    print(f"{'='*60}")
    model_names = list(models.keys())
    best_brier = 1.0
    best_config = None
    for i, j in combinations(range(len(model_names)), 2):
        n1, n2 = model_names[i], model_names[j]
        p1 = models[n1]["oof_predictions"]
        p2 = models[n2]["oof_predictions"]
        w = optimize_weights([p1, p2], y_true)
        ens = np.clip(ensemble_predict([p1, p2], w), 0.025, 0.975)
        m = compute_all_metrics(y_true, ens)
        brier = m["brier_score"]
        if brier < best_brier:
            best_brier = brier
            best_config = f"{n1}({w[0]:.0%}) + {n2}({w[1]:.0%})"
        print(f"  {n1}({w[0]:.0%}) + {n2}({w[1]:.0%}): {brier:.4f}")

    print(f"\n  BEST PAIR: {best_config} = {best_brier:.4f}")

    # All-model ensemble
    print(f"\n--- All {len(model_names)} models ---")
    all_preds = [models[n]["oof_predictions"] for n in model_names]
    w_all = optimize_weights(all_preds, y_true)
    ens_all = np.clip(ensemble_predict(all_preds, w_all), 0.025, 0.975)
    m_all = compute_all_metrics(y_true, ens_all)
    print(f"  All-model: Brier={m_all['brier_score']:.4f}")
    for n, w in zip(model_names, w_all):
        if w > 0.01:
            print(f"    {n}: {w:.1%}")


if __name__ == "__main__":
    main()
