"""Experiment 10: Enhanced features for closing the gap to ~0.10 Brier.

Three categories of enhancements:
1. Torvik-derived features (AdjEM, off/def balance, percentile ranks, matchup interactions)
2. Seed-matchup historical priors (calibration anchor)
3. Ensemble with diverse error patterns (LGB + LR + prior blend)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marchmadness.config import CV_SEASONS, PROJECT_ROOT
from marchmadness.data_loader import load_all
from marchmadness.features.builder import (
    build_training_data, build_team_features, build_matchup_features, get_feature_columns
)
from marchmadness.evaluation.cross_val import leave_season_out_cv
from marchmadness.evaluation.metrics import brier_score
from lightgbm import LGBMClassifier


# ============================================================
# 1. SEED-MATCHUP HISTORICAL PRIORS
# ============================================================

def compute_seed_priors(data, gender="M"):
    """Compute historical P(lower seed wins) for each seed matchup.

    Returns dict: (seed_a, seed_b) -> historical_win_rate for lower seed.
    seed_a < seed_b always.
    """
    tourney = data[f"{gender}NCAATourneyCompactResults"]
    seeds_df = data[f"{gender}NCAATourneySeeds"]

    # Parse seeds
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

            # Did the lower-seeded team (better seed = lower number) win?
            if w_seed <= l_seed:
                priors[key] += 1
            counts[key] += 1

    # Convert to rates
    for key in priors:
        priors[key] = priors[key] / counts[key] if counts[key] > 0 else 0.5

    return priors, counts


def add_seed_prior_feature(df, priors):
    """Add historical seed matchup prior as a feature column."""
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
    df["seed_matchup_prior"] = prior_vals
    return df


# ============================================================
# 2. ENHANCED TORVIK-DERIVED FEATURES
# ============================================================

def add_torvik_derived_features(team_features):
    """Add derived features computed from raw Torvik columns.

    These go beyond simple differences to capture matchup dynamics.
    """
    tf = team_features.copy()

    if "Torvik_adjoe" in tf.columns and "Torvik_adjde" in tf.columns:
        # AdjEM = AdjOE - AdjDE (efficiency margin, most predictive single stat)
        tf["Torvik_adjem"] = tf["Torvik_adjoe"] - tf["Torvik_adjde"]

        # Offensive vs defensive balance: high = offense-oriented, low = defense-oriented
        # Useful because defensive teams may perform differently in tournament
        total = tf["Torvik_adjoe"] + tf["Torvik_adjde"]
        tf["Torvik_off_pct"] = tf["Torvik_adjoe"] / total.where(total > 0, 1)

    if "Torvik_barthag" in tf.columns:
        # Percentile rank within season (normalizes across years)
        tf["Torvik_barthag_pctile"] = tf["Torvik_barthag"].rank(pct=True)

    if "Torvik_adjem" in tf.columns:
        tf["Torvik_adjem_pctile"] = tf["Torvik_adjem"].rank(pct=True)

    return tf


def build_enhanced_matchup_features(team_features, team_a, team_b):
    """Build matchup features including cross-team interactions.

    Beyond simple diffs, adds offense-vs-defense matchup features.
    """
    a = team_features[team_features["TeamID"] == team_a]
    b = team_features[team_features["TeamID"] == team_b]

    if a.empty or b.empty:
        return {}

    a = a.iloc[0]
    b = b.iloc[0]

    # Start with standard diff features
    features = {}
    skip_cols = {"TeamID"}
    for col in team_features.columns:
        if col in skip_cols:
            continue
        val_a, val_b = a[col], b[col]
        if pd.isna(val_a) or pd.isna(val_b):
            features[f"{col}_diff"] = np.nan
        else:
            features[f"{col}_diff"] = val_a - val_b

        if col == "Seed":
            features["seed_a"] = val_a if not pd.isna(val_a) else np.nan
            features["seed_b"] = val_b if not pd.isna(val_b) else np.nan

    # Interaction features
    if "Seed_diff" in features and "OrdinalMean_diff" in features:
        seed_d = features["Seed_diff"]
        ord_d = features["OrdinalMean_diff"]
        if not (pd.isna(seed_d) or pd.isna(ord_d)):
            features["seed_ordinal_agreement"] = seed_d * ord_d
            features["seed_ordinal_mismatch"] = abs(seed_d * 20 - ord_d)

    # Cross-team Torvik interactions (offense vs defense matchup)
    if "Torvik_adjoe" in team_features.columns and "Torvik_adjde" in team_features.columns:
        adjoe_a = a.get("Torvik_adjoe", np.nan)
        adjoe_b = b.get("Torvik_adjoe", np.nan)
        adjde_a = a.get("Torvik_adjde", np.nan)
        adjde_b = b.get("Torvik_adjde", np.nan)

        if not any(pd.isna(x) for x in [adjoe_a, adjoe_b, adjde_a, adjde_b]):
            # Team A's offense vs Team B's defense (and vice versa)
            # Higher = team A has advantage in that matchup
            features["torvik_a_off_vs_b_def"] = adjoe_a - adjde_b
            features["torvik_b_off_vs_a_def"] = adjoe_b - adjde_a
            # Net matchup advantage for team A
            features["torvik_matchup_edge"] = (adjoe_a - adjde_b) - (adjoe_b - adjde_a)

    # Tempo mismatch (absolute difference in pace)
    if "Torvik_adjt" in team_features.columns:
        adjt_a = a.get("Torvik_adjt", np.nan)
        adjt_b = b.get("Torvik_adjt", np.nan)
        if not (pd.isna(adjt_a) or pd.isna(adjt_b)):
            features["torvik_tempo_mismatch"] = abs(adjt_a - adjt_b)

    return features


def build_enhanced_training_data(data, seasons, gender="M", feature_set="torvik"):
    """Build training data with enhanced Torvik features."""
    tourney_key = f"{gender}NCAATourneyCompactResults"
    tourney = data[tourney_key]

    all_rows = []
    for season in seasons:
        team_features = build_team_features(data, season, gender, feature_set)
        if team_features.empty:
            continue

        # Add derived Torvik features at team level
        team_features = add_torvik_derived_features(team_features)

        season_tourney = tourney[tourney["Season"] == season]
        for _, game in season_tourney.iterrows():
            w_id, l_id = game["WTeamID"], game["LTeamID"]
            team_a, team_b = min(w_id, l_id), max(w_id, l_id)
            label = 1 if w_id == team_a else 0

            matchup = build_enhanced_matchup_features(team_features, team_a, team_b)
            if not matchup:
                continue

            matchup["Season"] = season
            matchup["Label"] = label
            matchup["DayNum"] = game["DayNum"]
            matchup["SampleWeight"] = 6.0
            all_rows.append(matchup)

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


# ============================================================
# 3. HELPERS
# ============================================================

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


def run_cv(model, X, y, seasons, label):
    result = leave_season_out_cv(model, X, y, seasons, CV_SEASONS)
    brier = result["overall"]["brier_score"]
    ll = result["overall"]["log_loss"]
    print(f"  {label}: Brier={brier:.4f}  LogLoss={ll:.4f}")
    for s in CV_SEASONS:
        if s in result["per_fold"]:
            print(f"    {s}: Brier={result['per_fold'][s]['brier_score']:.4f}")
    return result


def collect_oof_predictions(model, X, y, seasons):
    """Collect out-of-fold predictions for ensemble optimization."""
    preds = np.zeros(len(y))
    for val_season in CV_SEASONS:
        train_mask = seasons != val_season
        val_mask = seasons == val_season
        if val_mask.sum() == 0:
            continue
        m = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
        m.fit(X[train_mask], y[train_mask])
        if hasattr(m, 'predict_proba'):
            preds[val_mask] = m.predict_proba(X[val_mask])[:, 1]
        else:
            preds[val_mask] = m.predict(X[val_mask])
    return preds


def collect_oof_pipeline(pipeline_fn, X, y, seasons):
    """Collect OOF predictions for sklearn Pipeline."""
    preds = np.zeros(len(y))
    for val_season in CV_SEASONS:
        train_mask = seasons != val_season
        val_mask = seasons == val_season
        if val_mask.sum() == 0:
            continue
        p = pipeline_fn()
        p.fit(X[train_mask], y[train_mask])
        preds[val_mask] = p.predict_proba(X[val_mask])[:, 1]
    return preds


# ============================================================
# 4. MAIN EXPERIMENTS
# ============================================================

def main():
    print("Loading data...")
    data = load_all()
    train_seasons = list(range(2015, 2026))

    # --- Part A: Enhanced Torvik features ---
    print("\n" + "=" * 70)
    print("  PART A: Enhanced Torvik-derived features (Men's)")
    print("=" * 70)

    # Baseline: standard torvik features
    print("\n--- Baseline: standard torvik, LGB d=3 ---")
    df_base = build_training_data(data, train_seasons, "M", "torvik")
    X_base, y_base, s_base, fc_base = prep_data(df_base)
    print(f"  Features ({len(fc_base)}): {fc_base}")
    lgb = LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    run_cv(lgb, X_base, y_base, s_base, "Torvik baseline LGB")

    # Enhanced: with derived features
    print("\n--- Enhanced torvik features, LGB d=3 ---")
    df_enh = build_enhanced_training_data(data, train_seasons, "M", "torvik")
    X_enh, y_enh, s_enh, fc_enh = prep_data(df_enh)
    new_feats = [f for f in fc_enh if f not in fc_base]
    print(f"  Features ({len(fc_enh)}): +{len(new_feats)} new: {new_feats}")
    lgb2 = LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    run_cv(lgb2, X_enh, y_enh, s_enh, "Enhanced torvik LGB")

    # Enhanced with LR
    print("\n--- Enhanced torvik features, LR C=1.0 ---")
    lr_pipe = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=1000))])
    run_cv(lr_pipe, X_enh, y_enh, s_enh, "Enhanced torvik LR C=1")

    print("\n--- Enhanced torvik features, LR C=0.1 ---")
    lr_pipe2 = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))])
    run_cv(lr_pipe2, X_enh, y_enh, s_enh, "Enhanced torvik LR C=0.1")

    # --- Part B: Seed matchup priors ---
    print("\n" + "=" * 70)
    print("  PART B: Seed-matchup historical priors")
    print("=" * 70)

    priors_m, counts_m = compute_seed_priors(data, "M")
    print("\n  Historical seed matchup priors (Men's):")
    for key in sorted(priors_m.keys()):
        if counts_m[key] >= 10:
            print(f"    {key[0]}v{key[1]}: {priors_m[key]:.3f} ({counts_m[key]} games)")

    # Add prior as feature
    print("\n--- Enhanced torvik + seed prior feature, LGB d=3 ---")
    df_prior = df_enh.copy()
    df_prior = add_seed_prior_feature(df_prior, priors_m)
    X_prior, y_prior, s_prior, fc_prior = prep_data(df_prior)
    print(f"  Features ({len(fc_prior)})")
    lgb3 = LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    run_cv(lgb3, X_prior, y_prior, s_prior, "Enhanced + prior LGB")

    # --- Part C: Ensemble optimization ---
    print("\n" + "=" * 70)
    print("  PART C: Ensemble with diverse models")
    print("=" * 70)

    # Collect OOF predictions from diverse models
    print("\n  Collecting OOF predictions...")

    # Model 1: Enhanced Torvik LGB d=3
    lgb_oof = collect_oof_predictions(
        LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        X_enh, y_enh, s_enh
    )
    print(f"  LGB OOF Brier: {brier_score(y_enh, lgb_oof):.4f}")

    # Model 2: Enhanced Torvik LR C=1.0
    lr_oof = collect_oof_pipeline(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=1000))]),
        X_enh, y_enh, s_enh
    )
    print(f"  LR C=1 OOF Brier: {brier_score(y_enh, lr_oof):.4f}")

    # Model 3: Enhanced Torvik LR C=0.1
    lr01_oof = collect_oof_pipeline(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
        X_enh, y_enh, s_enh
    )
    print(f"  LR C=0.1 OOF Brier: {brier_score(y_enh, lr01_oof):.4f}")

    # Model 4: Seed-only LR (maximally different error pattern)
    df_seeds = build_training_data(data, train_seasons, "M", "seeds_only")
    X_seeds, y_seeds, s_seeds, _ = prep_data(df_seeds)
    seeds_oof = collect_oof_pipeline(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=1000))]),
        X_seeds, y_seeds, s_seeds
    )
    print(f"  Seeds LR OOF Brier: {brier_score(y_seeds, seeds_oof):.4f}")

    # Model 5: Seed prior (no ML, just historical rates)
    df_seeds_with_prior = df_seeds.copy()
    df_seeds_with_prior = add_seed_prior_feature(df_seeds_with_prior, priors_m)
    prior_oof = df_seeds_with_prior["seed_matchup_prior"].values
    print(f"  Seed prior OOF Brier: {brier_score(y_seeds, prior_oof):.4f}")

    # Optimize ensemble weights
    print("\n  Optimizing ensemble weights...")
    # All models should have same y, but seeds model may have different ordering
    # Use enhanced data as base (has all games that seeds data has since seeds_only is a subset)

    model_preds = [lgb_oof, lr_oof, lr01_oof]
    model_names = ["LGB_enh", "LR_C1", "LR_C01"]

    def ensemble_brier(weights):
        w = np.array(weights)
        w = w / w.sum()
        blend = sum(w[i] * model_preds[i] for i in range(len(model_preds)))
        return brier_score(y_enh, blend)

    best_brier = float('inf')
    best_weights = None
    # Grid search for interpretability
    for w0 in np.arange(0.1, 0.9, 0.1):
        for w1 in np.arange(0.1, 0.9 - w0 + 0.05, 0.1):
            w2 = 1.0 - w0 - w1
            if w2 < 0.05:
                continue
            b = ensemble_brier([w0, w1, w2])
            if b < best_brier:
                best_brier = b
                best_weights = [w0, w1, w2]

    print(f"\n  Best 3-model ensemble: Brier={best_brier:.4f}")
    for name, w in zip(model_names, best_weights):
        print(f"    {name}: {w:.1%}")

    # Also try scipy optimization
    result = minimize(ensemble_brier, [0.4, 0.3, 0.3],
                      method="Nelder-Mead",
                      bounds=[(0.05, 0.95)] * 3)
    opt_w = result.x / result.x.sum()
    opt_brier = ensemble_brier(opt_w)
    print(f"\n  Scipy-optimized ensemble: Brier={opt_brier:.4f}")
    for name, w in zip(model_names, opt_w):
        print(f"    {name}: {w:.1%}")

    # Try blending with seed prior
    print("\n  Testing blend with seed prior anchor...")
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        best_blend = sum(best_weights[i] * model_preds[i] for i in range(len(model_preds)))
        # Blend model ensemble with historical prior
        # Need to align prior_oof with enhanced data
        df_enh_with_prior = df_enh.copy()
        df_enh_with_prior = add_seed_prior_feature(df_enh_with_prior, priors_m)
        prior_aligned = df_enh_with_prior["seed_matchup_prior"].values
        blended = (1 - alpha) * best_blend + alpha * prior_aligned
        b = brier_score(y_enh, blended)
        print(f"    alpha={alpha:.2f}: Brier={b:.4f}")

    # --- Part D: Log-odds averaging ---
    print("\n" + "=" * 70)
    print("  PART D: Log-odds averaging (instead of probability averaging)")
    print("=" * 70)

    eps = 1e-6
    def to_logodds(p):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    def from_logodds(lo):
        return 1 / (1 + np.exp(-lo))

    lo_preds = [to_logodds(p) for p in model_preds]

    def logodds_ensemble_brier(weights):
        w = np.array(weights)
        w = w / w.sum()
        blend_lo = sum(w[i] * lo_preds[i] for i in range(len(lo_preds)))
        blend_p = from_logodds(blend_lo)
        return brier_score(y_enh, blend_p)

    best_lo_brier = float('inf')
    best_lo_weights = None
    for w0 in np.arange(0.1, 0.9, 0.1):
        for w1 in np.arange(0.1, 0.9 - w0 + 0.05, 0.1):
            w2 = 1.0 - w0 - w1
            if w2 < 0.05:
                continue
            b = logodds_ensemble_brier([w0, w1, w2])
            if b < best_lo_brier:
                best_lo_brier = b
                best_lo_weights = [w0, w1, w2]

    print(f"\n  Best log-odds ensemble: Brier={best_lo_brier:.4f}")
    for name, w in zip(model_names, best_lo_weights):
        print(f"    {name}: {w:.1%}")

    # --- Part E: Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Previous best (Torvik LGB d=3): 0.1543")
    print(f"  Enhanced Torvik LGB: see above")
    print(f"  Best probability ensemble: {best_brier:.4f}")
    print(f"  Best log-odds ensemble: {best_lo_brier:.4f}")
    print(f"  Scipy-optimized: {opt_brier:.4f}")


if __name__ == "__main__":
    main()
