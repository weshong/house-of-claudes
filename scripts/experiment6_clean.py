"""Experiment 6 (clean): Test tier1 improvements + LightGBM ensemble."""

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


def prep_data(data, gender, feature_set, start_year=None):
    if start_year is None:
        start_year = 1985 if gender == "M" else 1998
    df = build_training_data(data, list(range(start_year, 2026)), gender, feature_set)
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].values.copy()
    y = df["Label"].values.copy()
    seasons = df["Season"].values.copy()
    # Handle NaN
    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0
    return X, y, seasons, feat_cols


def run_single(X, y, seasons, C=1.0, shrinkage=0.0, label=""):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
    ])
    result = leave_season_out_cv(model, X, y, seasons, CV_SEASONS, shrinkage_alpha=shrinkage)
    m = result["overall"]
    print(f"  {label}: Brier={m['brier_score']:.4f}, Acc={m['accuracy']:.3f}, ECE={m.get('ece', 0):.4f}")
    return m, result


def run_lgb(X, y, seasons, shrinkage=0.0, label=""):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    result = leave_season_out_cv(model, X, y, seasons, CV_SEASONS, shrinkage_alpha=shrinkage)
    m = result["overall"]
    print(f"  {label}: Brier={m['brier_score']:.4f}, Acc={m['accuracy']:.3f}, ECE={m.get('ece', 0):.4f}")
    return m, result


def main():
    print("Loading data...")
    data = load_all()

    results = []

    for gender in ["M", "W"]:
        print(f"\n{'='*60}")
        print(f"  {gender}")
        print(f"{'='*60}")

        # seeds_only
        X_s, y_s, seas_s, _ = prep_data(data, gender, "seeds_only")
        print(f"\n--- seeds_only ({len(y_s)} games, {X_s.shape[1]} features) ---")
        m, r_lr_s = run_single(X_s, y_s, seas_s, C=1.0, label="LR C=1.0")
        results.append((gender, "seeds_only LR", m["brier_score"]))

        m, r_lgb_s = run_lgb(X_s, y_s, seas_s, label="LGB")
        results.append((gender, "seeds_only LGB", m["brier_score"]))

        # Ensemble seeds_only
        oof_lr = r_lr_s["oof_predictions"]
        oof_lgb = r_lgb_s["oof_predictions"]
        oof_y = r_lr_s["oof_labels"]
        weights = optimize_weights([oof_lr, oof_lgb], oof_y)
        ens_pred = ensemble_predict([oof_lr, oof_lgb], weights)
        ens_m = compute_all_metrics(oof_y, np.clip(ens_pred, 0.025, 0.975))
        print(f"  Ensemble (LR {weights[0]:.0%} + LGB {weights[1]:.0%}): Brier={ens_m['brier_score']:.4f}")
        results.append((gender, f"seeds_only ens (LR{weights[0]:.0%}+LGB{weights[1]:.0%})", ens_m["brier_score"]))

        # tier1 (seeds + ordinals with new features)
        X_t1, y_t1, seas_t1, feat_cols_t1 = prep_data(data, gender, "tier1")
        print(f"\n--- tier1 ({len(y_t1)} games, {X_t1.shape[1]} features) ---")
        print(f"  Features: {feat_cols_t1}")

        m, r_lr_t1 = run_single(X_t1, y_t1, seas_t1, C=0.01, label="LR C=0.01")
        results.append((gender, "tier1 LR C=0.01", m["brier_score"]))

        m, r_lr_t1_s = run_single(X_t1, y_t1, seas_t1, C=0.01, shrinkage=0.10, label="LR C=0.01 shrink=0.10")
        results.append((gender, "tier1 LR C=0.01 shrink=0.10", m["brier_score"]))

        m, r_lgb_t1 = run_lgb(X_t1, y_t1, seas_t1, label="LGB")
        results.append((gender, "tier1 LGB", m["brier_score"]))

        # Ensemble tier1
        oof_lr_t1 = r_lr_t1["oof_predictions"]
        oof_lgb_t1 = r_lgb_t1["oof_predictions"]
        oof_y_t1 = r_lr_t1["oof_labels"]
        weights_t1 = optimize_weights([oof_lr_t1, oof_lgb_t1], oof_y_t1)
        ens_pred_t1 = ensemble_predict([oof_lr_t1, oof_lgb_t1], weights_t1)
        ens_m_t1 = compute_all_metrics(oof_y_t1, np.clip(ens_pred_t1, 0.025, 0.975))
        print(f"  Ensemble (LR {weights_t1[0]:.0%} + LGB {weights_t1[1]:.0%}): Brier={ens_m_t1['brier_score']:.4f}")
        results.append((gender, f"tier1 ens (LR{weights_t1[0]:.0%}+LGB{weights_t1[1]:.0%})", ens_m_t1["brier_score"]))

        # tier2 (seeds + ordinals + elo + stats)
        X_t2, y_t2, seas_t2, feat_cols_t2 = prep_data(data, gender, "tier2")
        print(f"\n--- tier2 ({len(y_t2)} games, {X_t2.shape[1]} features) ---")
        m, r_lr_t2 = run_single(X_t2, y_t2, seas_t2, C=0.01, label="LR C=0.01")
        results.append((gender, "tier2 LR C=0.01", m["brier_score"]))

        m, r_lgb_t2 = run_lgb(X_t2, y_t2, seas_t2, label="LGB")
        results.append((gender, "tier2 LGB", m["brier_score"]))

        # Ensemble tier2
        oof_lr_t2 = r_lr_t2["oof_predictions"]
        oof_lgb_t2 = r_lgb_t2["oof_predictions"]
        oof_y_t2 = r_lr_t2["oof_labels"]
        weights_t2 = optimize_weights([oof_lr_t2, oof_lgb_t2], oof_y_t2)
        ens_pred_t2 = ensemble_predict([oof_lr_t2, oof_lgb_t2], weights_t2)
        ens_m_t2 = compute_all_metrics(oof_y_t2, np.clip(ens_pred_t2, 0.025, 0.975))
        print(f"  Ensemble (LR {weights_t2[0]:.0%} + LGB {weights_t2[1]:.0%}): Brier={ens_m_t2['brier_score']:.4f}")
        results.append((gender, f"tier2 ens (LR{weights_t2[0]:.0%}+LGB{weights_t2[1]:.0%})", ens_m_t2["brier_score"]))

        # 3-model ensemble: seeds LR + tier1 LR + tier1 LGB
        # Only for men (women don't have ordinals)
        if gender == "M" and len(oof_lr) == len(oof_lr_t1):
            print(f"\n--- 3-model cross-feature ensemble ---")
            weights_3 = optimize_weights([oof_lr, oof_lr_t1, oof_lgb_t1], oof_y)
            ens_3 = ensemble_predict([oof_lr, oof_lr_t1, oof_lgb_t1], weights_3)
            ens_3_m = compute_all_metrics(oof_y, np.clip(ens_3, 0.025, 0.975))
            print(f"  3-model ens: Brier={ens_3_m['brier_score']:.4f} "
                  f"(seeds_LR {weights_3[0]:.0%} + t1_LR {weights_3[1]:.0%} + t1_LGB {weights_3[2]:.0%})")
            results.append((gender, "3-model ens", ens_3_m["brier_score"]))

    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    for gender in ["M", "W"]:
        print(f"\n{gender}:")
        g_results = [(c, b) for g, c, b in results if g == gender]
        best = min(b for _, b in g_results)
        for config, brier in sorted(g_results, key=lambda x: x[1]):
            marker = " <-- BEST" if brier == best else ""
            print(f"  {brier:.4f}  {config}{marker}")


if __name__ == "__main__":
    main()
