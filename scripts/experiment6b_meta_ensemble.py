"""Experiment 6b: Meta-ensemble blending predictions from different feature sets."""

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


def prep_and_cv(data, gender, feature_set, C=1.0, start_year=None):
    if start_year is None:
        start_year = 1985 if gender == "M" else 1998
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
    return result


def prep_and_cv_lgb(data, gender, feature_set, start_year=None):
    if start_year is None:
        start_year = 1985 if gender == "M" else 1998
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
    return result


def main():
    print("Loading data...")
    data = load_all()

    for gender in ["M", "W"]:
        print(f"\n{'='*60}")
        print(f"  {gender} - Meta-ensemble")
        print(f"{'='*60}")

        # Collect OOF predictions from many models
        models = {}

        print("  Training seeds_only LR...")
        r = prep_and_cv(data, gender, "seeds_only", C=1.0)
        models["seeds_LR"] = r
        print(f"    Brier={r['overall']['brier_score']:.4f}")

        print("  Training seeds_only LGB...")
        r = prep_and_cv_lgb(data, gender, "seeds_only")
        models["seeds_LGB"] = r
        print(f"    Brier={r['overall']['brier_score']:.4f}")

        print("  Training tier1 LR C=0.01...")
        r = prep_and_cv(data, gender, "tier1", C=0.01)
        models["tier1_LR"] = r
        print(f"    Brier={r['overall']['brier_score']:.4f}")

        print("  Training tier2 LR C=0.01...")
        r = prep_and_cv(data, gender, "tier2", C=0.01)
        models["tier2_LR"] = r
        print(f"    Brier={r['overall']['brier_score']:.4f}")

        print("  Training tier2 LR C=0.1...")
        r = prep_and_cv(data, gender, "tier2", C=0.1)
        models["tier2_LR_01"] = r
        print(f"    Brier={r['overall']['brier_score']:.4f}")

        print("  Training all LR C=0.01...")
        r = prep_and_cv(data, gender, "all", C=0.01)
        models["all_LR"] = r
        print(f"    Brier={r['overall']['brier_score']:.4f}")

        # Try all pairwise and triple ensembles
        model_names = list(models.keys())
        y_true = models[model_names[0]]["oof_labels"]

        print(f"\n  --- Pairwise ensembles ---")
        best_pair = None
        best_pair_brier = 1.0
        from itertools import combinations
        for i, j in combinations(range(len(model_names)), 2):
            n1, n2 = model_names[i], model_names[j]
            p1 = models[n1]["oof_predictions"]
            p2 = models[n2]["oof_predictions"]
            w = optimize_weights([p1, p2], y_true)
            ens = ensemble_predict([p1, p2], w)
            ens = np.clip(ens, 0.025, 0.975)
            m = compute_all_metrics(y_true, ens)
            brier = m["brier_score"]
            if brier < best_pair_brier:
                best_pair = (n1, n2, w, brier)
                best_pair_brier = brier
            print(f"    {n1} ({w[0]:.0%}) + {n2} ({w[1]:.0%}): {brier:.4f}")

        print(f"\n  Best pair: {best_pair[0]} + {best_pair[1]} = {best_pair[3]:.4f}")

        # Try 3-model ensembles with best models
        print(f"\n  --- Triple ensembles (top combos) ---")
        best_triple = None
        best_triple_brier = 1.0
        for combo in combinations(range(len(model_names)), 3):
            names = [model_names[i] for i in combo]
            preds = [models[n]["oof_predictions"] for n in names]
            w = optimize_weights(preds, y_true)
            ens = np.clip(ensemble_predict(preds, w), 0.025, 0.975)
            m = compute_all_metrics(y_true, ens)
            brier = m["brier_score"]
            if brier < best_triple_brier:
                best_triple = (names, w, brier)
                best_triple_brier = brier
            wstr = " + ".join(f"{n}({wi:.0%})" for n, wi in zip(names, w))
            print(f"    {wstr}: {brier:.4f}")

        print(f"\n  Best triple: {best_triple[2]:.4f}")
        for n, w in zip(best_triple[0], best_triple[1]):
            print(f"    {n}: {w:.1%}")

        # Try all models
        print(f"\n  --- All {len(model_names)} models ---")
        all_preds = [models[n]["oof_predictions"] for n in model_names]
        w_all = optimize_weights(all_preds, y_true)
        ens_all = np.clip(ensemble_predict(all_preds, w_all), 0.025, 0.975)
        m_all = compute_all_metrics(y_true, ens_all)
        print(f"  All-model ensemble: Brier={m_all['brier_score']:.4f}")
        for n, w in zip(model_names, w_all):
            print(f"    {n}: {w:.1%}")

        # Try shrinkage on best ensemble
        print(f"\n  --- Shrinkage on best ensemble ---")
        best_ens = ens_all
        for alpha in [0.02, 0.05, 0.10, 0.15]:
            shrunk = best_ens * (1 - alpha) + 0.5 * alpha
            m_s = compute_all_metrics(y_true, np.clip(shrunk, 0.025, 0.975))
            print(f"    shrink={alpha}: Brier={m_s['brier_score']:.4f}")


if __name__ == "__main__":
    main()
