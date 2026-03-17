"""Experiment 10B: Proper ensemble optimization with OOF predictions.

Fix the OOF collection bug from exp10 and test diverse ensembles.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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


def collect_oof(model_fn, X, y, seasons):
    """Collect OOF predictions only for CV seasons, return (preds, mask)."""
    cv_mask = np.isin(seasons, CV_SEASONS)
    preds = np.full(len(y), np.nan)

    for val_season in CV_SEASONS:
        train_mask = (seasons != val_season) & cv_mask  # Train on other CV seasons only
        # Actually train on ALL non-val data (including non-CV seasons)
        train_mask = seasons != val_season
        val_mask = seasons == val_season
        if val_mask.sum() == 0:
            continue

        model = model_fn()
        model.fit(X[train_mask], y[train_mask])
        if hasattr(model, 'predict_proba'):
            preds[val_mask] = model.predict_proba(X[val_mask])[:, 1]
        else:
            preds[val_mask] = model.predict(X[val_mask])

    # Return only the CV predictions
    valid = ~np.isnan(preds)
    return preds[valid], y[valid], seasons[valid]


def compute_seed_priors(data, gender="M"):
    """Compute historical P(lower seed wins) for each seed matchup."""
    tourney = data[f"{gender}NCAATourneyCompactResults"]
    seeds_df = data[f"{gender}NCAATourneySeeds"]

    def parse_seed(s):
        return int(s[1:3])

    priors = {}
    counts = {}

    for season in tourney["Season"].unique():
        season_seeds = seeds_df[seeds_df["Season"] == season].copy()
        season_seeds["SeedNum"] = season_seeds["Seed"].apply(parse_seed)
        seed_map = dict(zip(season_seeds["TeamID"], season_seeds["SeedNum"]))

        season_games = tourney[tourney["Season"] == season]
        for _, game in season_games.iterrows():
            w_seed = seed_map.get(game["WTeamID"])
            l_seed = seed_map.get(game["LTeamID"])
            if w_seed is None or l_seed is None:
                continue

            sa, sb = min(w_seed, l_seed), max(w_seed, l_seed)
            key = (sa, sb)
            if key not in priors:
                priors[key] = 0
                counts[key] = 0
            if w_seed <= l_seed:
                priors[key] += 1
            counts[key] += 1

    for key in priors:
        priors[key] = priors[key] / counts[key] if counts[key] > 0 else 0.5

    return priors, counts


def get_seed_prior_preds(df, priors):
    """Get seed-prior predictions for each row in training data."""
    prior_vals = []
    for _, row in df.iterrows():
        sa = row.get("seed_a", np.nan)
        sb = row.get("seed_b", np.nan)
        if pd.isna(sa) or pd.isna(sb):
            prior_vals.append(0.5)
        else:
            sa, sb = int(sa), int(sb)
            key = (min(sa, sb), max(sa, sb))
            prior_vals.append(priors.get(key, 0.5))
    return np.array(prior_vals)


def main():
    print("Loading data...")
    data = load_all()
    train_seasons = list(range(2015, 2026))

    # Build datasets
    print("\nBuilding datasets...")
    df_torvik = build_training_data(data, train_seasons, "M", "torvik")
    X_torvik, y_torvik, s_torvik, fc_torvik = prep_data(df_torvik)

    df_seeds = build_training_data(data, train_seasons, "M", "seeds_only")
    X_seeds, y_seeds, s_seeds, fc_seeds = prep_data(df_seeds)

    df_tier2 = build_training_data(data, train_seasons, "M", "tier2")
    X_tier2, y_tier2, s_tier2, fc_tier2 = prep_data(df_tier2)

    # Seed priors
    priors_m, _ = compute_seed_priors(data, "M")

    print(f"\n{'='*70}")
    print(f"  ENSEMBLE OPTIMIZATION (Men's)")
    print(f"{'='*70}")

    # Collect OOF predictions from diverse models
    models = {}

    print("\n--- Model 1: Torvik LGB d=3 n=400 lr=0.02 ---")
    p1, y1, s1 = collect_oof(
        lambda: LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        X_torvik, y_torvik, s_torvik)
    print(f"  Brier: {brier_score(y1, p1):.4f}")
    models["torvik_lgb_d3"] = p1

    print("\n--- Model 2: Torvik LGB d=2 n=500 lr=0.02 ---")
    p2, y2, s2 = collect_oof(
        lambda: LGBMClassifier(n_estimators=500, max_depth=2, learning_rate=0.02,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        X_torvik, y_torvik, s_torvik)
    print(f"  Brier: {brier_score(y2, p2):.4f}")
    models["torvik_lgb_d2"] = p2

    print("\n--- Model 3: Torvik LR C=1.0 ---")
    p3, y3, s3 = collect_oof(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=1000))]),
        X_torvik, y_torvik, s_torvik)
    print(f"  Brier: {brier_score(y3, p3):.4f}")
    models["torvik_lr_c1"] = p3

    print("\n--- Model 4: Torvik LR C=0.1 ---")
    p4, y4, s4 = collect_oof(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
        X_torvik, y_torvik, s_torvik)
    print(f"  Brier: {brier_score(y4, p4):.4f}")
    models["torvik_lr_c01"] = p4

    print("\n--- Model 5: Seeds LR C=1.0 ---")
    p5, y5, s5 = collect_oof(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=1000))]),
        X_seeds, y_seeds, s_seeds)
    print(f"  Brier: {brier_score(y5, p5):.4f}")
    models["seeds_lr"] = p5

    print("\n--- Model 6: Tier2 LR C=0.1 ---")
    p6, y6, s6 = collect_oof(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
        X_tier2, y_tier2, s_tier2)
    print(f"  Brier: {brier_score(y6, p6):.4f}")
    models["tier2_lr"] = p6

    print("\n--- Model 7: Seed matchup prior (no ML) ---")
    # Get prior for CV games only
    cv_mask = np.isin(s_torvik, CV_SEASONS)
    df_cv = df_torvik[cv_mask].copy()
    prior_preds = get_seed_prior_preds(df_cv, priors_m)
    y_cv = y_torvik[cv_mask]
    print(f"  Brier: {brier_score(y_cv, prior_preds):.4f}")

    # All models should have same games in same order (they all use same seasons)
    # Verify
    assert len(y1) == len(y2) == len(y3) == len(y4), "Model predictions have different lengths"
    assert np.array_equal(y1, y2), "Labels don't match between models"

    y_ref = y1

    # ---- Pairwise ensembles ----
    print(f"\n{'='*70}")
    print("  PAIRWISE ENSEMBLES")
    print(f"{'='*70}")

    # All torvik-based models have same length
    torvik_models = {k: v for k, v in models.items() if k.startswith("torvik")}
    all_model_names = list(torvik_models.keys())
    all_model_preds = [torvik_models[k] for k in all_model_names]

    for i in range(len(all_model_names)):
        for j in range(i + 1, len(all_model_names)):
            best_w = 0
            best_b = float('inf')
            for w in np.arange(0.1, 0.95, 0.05):
                blend = w * all_model_preds[i] + (1 - w) * all_model_preds[j]
                b = brier_score(y_ref, blend)
                if b < best_b:
                    best_b = b
                    best_w = w
            print(f"  {all_model_names[i]}({best_w:.0%}) + {all_model_names[j]}({1-best_w:.0%}): Brier={best_b:.4f}")

    # ---- 3-model ensembles ----
    print(f"\n{'='*70}")
    print("  3-MODEL ENSEMBLES (grid search)")
    print(f"{'='*70}")

    best_3 = float('inf')
    best_3_config = None
    for i in range(len(all_model_names)):
        for j in range(i + 1, len(all_model_names)):
            for k in range(j + 1, len(all_model_names)):
                for wi in np.arange(0.1, 0.8, 0.1):
                    for wj in np.arange(0.1, 0.8 - wi, 0.1):
                        wk = 1.0 - wi - wj
                        if wk < 0.05:
                            continue
                        blend = wi * all_model_preds[i] + wj * all_model_preds[j] + wk * all_model_preds[k]
                        b = brier_score(y_ref, blend)
                        if b < best_3:
                            best_3 = b
                            best_3_config = (all_model_names[i], wi,
                                             all_model_names[j], wj,
                                             all_model_names[k], wk)

    if best_3_config:
        print(f"  Best 3-model: Brier={best_3:.4f}")
        print(f"    {best_3_config[0]}: {best_3_config[1]:.0%}")
        print(f"    {best_3_config[2]}: {best_3_config[3]:.0%}")
        print(f"    {best_3_config[4]}: {best_3_config[5]:.0%}")

    # ---- All 4 torvik models ----
    print(f"\n{'='*70}")
    print("  4-MODEL ENSEMBLE (fine grid)")
    print(f"{'='*70}")

    best_4 = float('inf')
    best_4_w = None
    step = 0.05
    for w0 in np.arange(0.1, 0.8, step):
        for w1 in np.arange(0.05, 0.8 - w0, step):
            for w2 in np.arange(0.05, 0.8 - w0 - w1, step):
                w3 = 1.0 - w0 - w1 - w2
                if w3 < 0.05:
                    continue
                blend = (w0 * all_model_preds[0] + w1 * all_model_preds[1] +
                         w2 * all_model_preds[2] + w3 * all_model_preds[3])
                b = brier_score(y_ref, blend)
                if b < best_4:
                    best_4 = b
                    best_4_w = [w0, w1, w2, w3]

    print(f"  Best 4-model: Brier={best_4:.4f}")
    for name, w in zip(all_model_names, best_4_w):
        print(f"    {name}: {w:.0%}")

    # ---- Blend with seed prior ----
    print(f"\n{'='*70}")
    print("  BLEND BEST ENSEMBLE WITH SEED PRIOR")
    print(f"{'='*70}")

    best_ensemble = sum(w * p for w, p in zip(best_4_w, all_model_preds))
    for alpha in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
        blended = (1 - alpha) * best_ensemble + alpha * prior_preds
        b = brier_score(y_ref, blended)
        print(f"  alpha={alpha:.2f}: Brier={b:.4f}")

    # ---- Log-odds averaging ----
    print(f"\n{'='*70}")
    print("  LOG-ODDS AVERAGING")
    print(f"{'='*70}")

    eps = 1e-6
    def to_lo(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))
    def from_lo(lo):
        return 1 / (1 + np.exp(-lo))

    lo_preds = [to_lo(p) for p in all_model_preds]
    best_lo = float('inf')
    best_lo_w = None
    for w0 in np.arange(0.1, 0.8, step):
        for w1 in np.arange(0.05, 0.8 - w0, step):
            for w2 in np.arange(0.05, 0.8 - w0 - w1, step):
                w3 = 1.0 - w0 - w1 - w2
                if w3 < 0.05:
                    continue
                blend_lo = w0*lo_preds[0] + w1*lo_preds[1] + w2*lo_preds[2] + w3*lo_preds[3]
                blend_p = from_lo(blend_lo)
                b = brier_score(y_ref, blend_p)
                if b < best_lo:
                    best_lo = b
                    best_lo_w = [w0, w1, w2, w3]

    print(f"  Best log-odds 4-model: Brier={best_lo:.4f}")
    for name, w in zip(all_model_names, best_lo_w):
        print(f"    {name}: {w:.0%}")

    # ---- Per-season breakdown of best ensemble ----
    print(f"\n{'='*70}")
    print("  PER-SEASON BREAKDOWN")
    print(f"{'='*70}")

    best_blend = sum(w * p for w, p in zip(best_4_w, all_model_preds))
    for season in CV_SEASONS:
        mask = s1 == season
        b_ens = brier_score(y_ref[mask], best_blend[mask])
        b_single = brier_score(y_ref[mask], all_model_preds[0][mask])
        print(f"  {season}: ensemble={b_ens:.4f}  single_lgb_d3={b_single:.4f}  delta={b_ens-b_single:+.4f}")

    print(f"\n  SUMMARY: Previous best=0.1543, Best ensemble={best_4:.4f}, Log-odds={best_lo:.4f}")


if __name__ == "__main__":
    main()
