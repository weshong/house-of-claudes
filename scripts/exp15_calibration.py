"""Experiment 15: Calibration techniques to improve Brier score.

Brier = calibration_error + resolution. If our models have good resolution
(discriminative ability) but poor calibration, we can improve by:
1. Platt scaling (logistic regression on predictions)
2. Isotonic regression
3. Beta calibration
4. Prediction clipping optimization
5. Temperature scaling
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marchmadness.config import CV_SEASONS
from marchmadness.data_loader import load_all
from marchmadness.features.builder import build_training_data, get_feature_columns
from marchmadness.evaluation.metrics import brier_score
from lightgbm import LGBMClassifier


def prep_data(df):
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].values.copy()
    y = df["Label"].values.copy()
    seasons = df["Season"].values.copy()
    med = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = med[i] if not np.isnan(med[i]) else 0
    return X, y, seasons, feat_cols


def collect_oof_with_seasons(model_fn, X, y, seasons):
    """Collect OOF predictions preserving season info."""
    preds = np.full(len(y), np.nan)
    for val_season in CV_SEASONS:
        train_mask = seasons != val_season
        val_mask = seasons == val_season
        if val_mask.sum() == 0:
            continue
        model = model_fn()
        model.fit(X[train_mask], y[train_mask])
        preds[val_mask] = model.predict_proba(X[val_mask])[:, 1]
    valid = ~np.isnan(preds)
    return preds[valid], y[valid], seasons[valid]


def calibration_stats(y_true, y_pred, n_bins=10):
    """Compute calibration error and show bin-level stats."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    total = len(y_true)
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if i == n_bins - 1:
            mask = (y_pred >= bins[i]) & (y_pred <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_pred = y_pred[mask].mean()
        bin_true = y_true[mask].mean()
        ece += abs(bin_pred - bin_true) * mask.sum() / total
    return ece


def main():
    print("Loading data...")
    data = load_all()

    # ============================================================
    # WOMEN'S CALIBRATION
    # ============================================================
    print(f"\n{'='*70}")
    print("  WOMEN'S: Calibration Analysis")
    print(f"{'='*70}")

    # Get OOF predictions for tier2 LR
    df_w = build_training_data(data, list(range(1998, 2026)), "W", "tier2")
    X_w, y_w, s_w, fc_w = prep_data(df_w)

    oof_w, y_oof_w, s_oof_w = collect_oof_with_seasons(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
        X_w, y_w, s_w)

    brier_base = brier_score(y_oof_w, oof_w)
    ece_base = calibration_stats(y_oof_w, oof_w)
    print(f"\n  Baseline tier2 LR: Brier={brier_base:.4f}, ECE={ece_base:.4f}")

    # Calibration bins
    print("\n  Calibration table (10 bins):")
    bins = np.linspace(0, 1, 11)
    for i in range(10):
        mask = (oof_w >= bins[i]) & (oof_w < bins[i + 1])
        if i == 9:
            mask = (oof_w >= bins[i]) & (oof_w <= bins[i + 1])
        if mask.sum() > 0:
            pred_mean = oof_w[mask].mean()
            true_mean = y_oof_w[mask].mean()
            print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}]: n={mask.sum():3d}, pred={pred_mean:.3f}, actual={true_mean:.3f}, gap={pred_mean-true_mean:+.3f}")

    # Test different clipping ranges
    print("\n  Prediction clipping optimization:")
    for clip_lo, clip_hi in [(0.025, 0.975), (0.05, 0.95), (0.01, 0.99),
                              (0.03, 0.97), (0.04, 0.96), (0.0, 1.0),
                              (0.10, 0.90), (0.15, 0.85)]:
        clipped = np.clip(oof_w, clip_lo, clip_hi)
        b = brier_score(y_oof_w, clipped)
        print(f"    clip=[{clip_lo:.2f}, {clip_hi:.2f}]: Brier={b:.4f} ({'+' if b >= brier_base else ''}{b - brier_base:.4f})")

    # Temperature scaling
    print("\n  Temperature scaling:")
    for temp in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
        # Convert to logits, scale, convert back
        logits = np.log(oof_w / (1 - oof_w + 1e-10) + 1e-10)
        scaled = 1 / (1 + np.exp(-logits / temp))
        b = brier_score(y_oof_w, scaled)
        print(f"    T={temp:.1f}: Brier={b:.4f} ({'+' if b >= brier_base else ''}{b - brier_base:.4f})")

    # Seed-aware calibration: adjust predictions based on seed matchup
    print("\n  Seed-aware prediction adjustment:")
    # Get seed features from training data
    df_cv = df_w[np.isin(s_w, CV_SEASONS)].copy()
    seed_diff_cv = df_cv["Seed_diff"].values if "Seed_diff" in df_cv.columns else None

    if seed_diff_cv is not None and len(seed_diff_cv) == len(oof_w):
        # For upsets (lower seed wins, seed_diff > 0), check if we're too conservative
        upset_mask = y_oof_w == 1  # lower TeamID won
        for threshold in [0, 3, 5, 8]:
            big_seed_diff = np.abs(seed_diff_cv) >= threshold
            if big_seed_diff.sum() > 0:
                pred_mean = oof_w[big_seed_diff].mean()
                true_mean = y_oof_w[big_seed_diff].mean()
                print(f"    |seed_diff| >= {threshold}: n={big_seed_diff.sum()}, pred={pred_mean:.3f}, actual={true_mean:.3f}")

    # ============================================================
    # MEN'S CALIBRATION
    # ============================================================
    print(f"\n{'='*70}")
    print("  MEN'S: Calibration Analysis")
    print(f"{'='*70}")

    df_m = build_training_data(data, list(range(2015, 2026)), "M", "torvik")
    X_m, y_m, s_m, fc_m = prep_data(df_m)

    oof_m, y_oof_m, s_oof_m = collect_oof_with_seasons(
        lambda: LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        X_m, y_m, s_m)

    brier_base_m = brier_score(y_oof_m, oof_m)
    ece_base_m = calibration_stats(y_oof_m, oof_m)
    print(f"\n  Baseline Torvik LGB: Brier={brier_base_m:.4f}, ECE={ece_base_m:.4f}")

    # Calibration bins
    print("\n  Calibration table (10 bins):")
    for i in range(10):
        mask = (oof_m >= bins[i]) & (oof_m < bins[i + 1])
        if i == 9:
            mask = (oof_m >= bins[i]) & (oof_m <= bins[i + 1])
        if mask.sum() > 0:
            pred_mean = oof_m[mask].mean()
            true_mean = y_oof_m[mask].mean()
            print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}]: n={mask.sum():3d}, pred={pred_mean:.3f}, actual={true_mean:.3f}, gap={pred_mean-true_mean:+.3f}")

    # Clipping
    print("\n  Prediction clipping optimization:")
    for clip_lo, clip_hi in [(0.025, 0.975), (0.05, 0.95), (0.01, 0.99),
                              (0.03, 0.97), (0.04, 0.96), (0.0, 1.0),
                              (0.10, 0.90), (0.15, 0.85)]:
        clipped = np.clip(oof_m, clip_lo, clip_hi)
        b = brier_score(y_oof_m, clipped)
        print(f"    clip=[{clip_lo:.2f}, {clip_hi:.2f}]: Brier={b:.4f} ({'+' if b >= brier_base_m else ''}{b - brier_base_m:.4f})")

    # Temperature scaling
    print("\n  Temperature scaling:")
    for temp in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
        logits = np.log(oof_m / (1 - oof_m + 1e-10) + 1e-10)
        scaled = 1 / (1 + np.exp(-logits / temp))
        b = brier_score(y_oof_m, scaled)
        print(f"    T={temp:.1f}: Brier={b:.4f} ({'+' if b >= brier_base_m else ''}{b - brier_base_m:.4f})")

    # ============================================================
    # CalibratedClassifierCV (built-in sklearn)
    # ============================================================
    print(f"\n{'='*70}")
    print("  Built-in Calibration Methods")
    print(f"{'='*70}")

    # Women's: Platt scaling via CalibratedClassifierCV
    print("\n  Women's - CalibratedClassifierCV (Platt):")
    oof_platt_w, y_platt_w, _ = collect_oof_with_seasons(
        lambda: CalibratedClassifierCV(
            Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
            method="sigmoid", cv=5),
        X_w, y_w, s_w)
    print(f"    Platt: Brier={brier_score(y_platt_w, oof_platt_w):.4f}")

    print("\n  Women's - CalibratedClassifierCV (Isotonic):")
    oof_iso_w, y_iso_w, _ = collect_oof_with_seasons(
        lambda: CalibratedClassifierCV(
            Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
            method="isotonic", cv=5),
        X_w, y_w, s_w)
    print(f"    Isotonic: Brier={brier_score(y_iso_w, oof_iso_w):.4f}")

    # Men's
    print("\n  Men's - CalibratedClassifierCV (Platt):")
    oof_platt_m, y_platt_m, _ = collect_oof_with_seasons(
        lambda: CalibratedClassifierCV(
            LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
            method="sigmoid", cv=5),
        X_m, y_m, s_m)
    print(f"    Platt: Brier={brier_score(y_platt_m, oof_platt_m):.4f}")

    print("\n  Men's - CalibratedClassifierCV (Isotonic):")
    oof_iso_m, y_iso_m, _ = collect_oof_with_seasons(
        lambda: CalibratedClassifierCV(
            LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
            method="isotonic", cv=5),
        X_m, y_m, s_m)
    print(f"    Isotonic: Brier={brier_score(y_iso_m, oof_iso_m):.4f}")

    print(f"\n  REFERENCE: Women's best = 0.1386, Men's best = 0.1543")


if __name__ == "__main__":
    main()
