"""Experiment 14: T-Rank features combined with tier2 + 3-way blend.

Two approaches:
1. Add T-Rank columns as features alongside tier2 features
2. 3-way prediction blend: tier2 + iter_eff + T-Rank
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
from marchmadness.features.builder import (
    build_team_features, build_matchup_features, build_training_data, get_feature_columns
)
from marchmadness.features import seeds, ordinals
from marchmadness.features.trank_clone import compute as compute_trank
from marchmadness.evaluation.cross_val import leave_season_out_cv
from marchmadness.evaluation.metrics import brier_score


# Best Optuna params from exp13
BEST_TRANK_KWARGS = {
    "n_iterations": 10,
    "hca_factor": 0.01766,
    "recency_full_window": 31,
    "recency_decay_rate": 0.01139,
    "recency_floor": 0.431,
    "blowout_threshold": 9,
    "barthag_exponent": 12.958,
    "preseason_games": 5,
}


def build_tier2_plus_trank(data, seasons, gender, trank_kwargs=None):
    """Build tier2 features + T-Rank features combined."""
    tourney_key = f"{gender}NCAATourneyCompactResults"
    tourney = data[tourney_key]

    all_rows = []
    for season in seasons:
        team_features = build_team_features(data, season, gender, "tier2")
        if team_features.empty:
            continue

        kwargs = trank_kwargs or {}
        trank_df = compute_trank(data, season, gender, **kwargs)
        if not trank_df.empty:
            team_features = team_features.merge(trank_df, on="TeamID", how="left")

        season_tourney = tourney[tourney["Season"] == season]
        for _, game in season_tourney.iterrows():
            w_id, l_id = game["WTeamID"], game["LTeamID"]
            team_a, team_b = min(w_id, l_id), max(w_id, l_id)
            label = 1 if w_id == team_a else 0

            matchup = build_matchup_features(team_features, team_a, team_b)
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
    return preds[valid], y[valid]


def main():
    print("Loading data...")
    data = load_all()
    w_seasons = list(range(2010, 2026))

    # ============================================================
    # Part 1: tier2 + T-Rank combined features
    # ============================================================
    print(f"\n{'='*70}")
    print("  PART 1: tier2 + T-Rank Combined Features (Women's)")
    print(f"{'='*70}")

    df_combined = build_tier2_plus_trank(data, w_seasons, "W", BEST_TRANK_KWARGS)
    X_c, y_c, s_c, fc_c = prep_data(df_combined)
    print(f"  Features ({len(fc_c)}): {fc_c}")

    for c_val in [0.01, 0.05, 0.1, 0.5]:
        model = Pipeline([("s", StandardScaler()),
                          ("lr", LogisticRegression(C=c_val, max_iter=1000))])
        result = leave_season_out_cv(model, X_c, y_c, s_c, CV_SEASONS)
        brier = result["overall"]["brier_score"]
        print(f"  tier2+TRank LR C={c_val}: Brier={brier:.4f}")

    # ============================================================
    # Part 2: 3-way prediction blend
    # ============================================================
    print(f"\n{'='*70}")
    print("  PART 2: 3-Way Prediction Blend (Women's)")
    print(f"{'='*70}")

    # Collect OOF from each model
    # tier2
    df_t2 = build_training_data(data, list(range(1998, 2026)), "W", "tier2")
    X_t2, y_t2, s_t2, _ = prep_data(df_t2)
    oof_t2, y_t2_cv = collect_oof(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
        X_t2, y_t2, s_t2)

    # iter_eff
    df_ie = build_training_data(data, list(range(2003, 2026)), "W", "iter_eff")
    X_ie, y_ie, s_ie, _ = prep_data(df_ie)
    oof_ie, y_ie_cv = collect_oof(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
        X_ie, y_ie, s_ie)

    # T-Rank (optimized) - reuse build function from exp13
    df_tr = build_tier2_plus_trank(data, w_seasons, "W", BEST_TRANK_KWARGS)
    # Use only TRank features (filter out tier2)
    trank_feat_cols = [c for c in get_feature_columns(df_tr) if "TRank" in c]
    # Actually just use the combined data for T-Rank-only predictions
    from marchmadness.features.trank_clone import compute as compute_trank_fn
    # Build T-Rank only training data
    from marchmadness.features.builder import build_matchup_features as bmf
    tourney = data["WNCAATourneyCompactResults"]
    tr_rows = []
    for season in w_seasons:
        tf = seeds.compute(data, season, "W")
        if tf.empty:
            continue
        trank_df = compute_trank_fn(data, season, "W", **BEST_TRANK_KWARGS)
        if not trank_df.empty:
            tf = tf.merge(trank_df, on="TeamID", how="left")
        st = tourney[tourney["Season"] == season]
        for _, game in st.iterrows():
            w_id, l_id = game["WTeamID"], game["LTeamID"]
            ta, tb = min(w_id, l_id), max(w_id, l_id)
            label = 1 if w_id == ta else 0
            m = bmf(tf, ta, tb)
            if not m:
                continue
            m["Season"] = season
            m["Label"] = label
            m["DayNum"] = game["DayNum"]
            m["SampleWeight"] = 6.0
            tr_rows.append(m)
    df_tr = pd.DataFrame(tr_rows)
    X_tr, y_tr, s_tr, _ = prep_data(df_tr)
    oof_tr, y_tr_cv = collect_oof(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.05, max_iter=1000))]),
        X_tr, y_tr, s_tr)

    # tier2+TRank combined
    oof_ct, y_ct_cv = collect_oof(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.05, max_iter=1000))]),
        X_c, y_c, s_c)

    print(f"  tier2 OOF: {brier_score(y_t2_cv, oof_t2):.4f}")
    print(f"  iter_eff OOF: {brier_score(y_ie_cv, oof_ie):.4f}")
    print(f"  T-Rank OOF: {brier_score(y_tr_cv, oof_tr):.4f}")
    print(f"  tier2+TRank OOF: {brier_score(y_ct_cv, oof_ct):.4f}")

    # Check all have same number of CV games
    n_t2, n_ie, n_tr = len(y_t2_cv), len(y_ie_cv), len(y_tr_cv)
    print(f"\n  CV games: tier2={n_t2}, iter_eff={n_ie}, T-Rank={n_tr}, combined={len(y_ct_cv)}")

    if n_t2 == n_tr == len(y_ct_cv):
        print("\n  3-way blend: tier2 + T-Rank + combined")
        best_b = float('inf')
        best_w = None
        for w1 in np.arange(0, 1.05, 0.1):
            for w2 in np.arange(0, 1.05 - w1, 0.1):
                w3 = 1.0 - w1 - w2
                if w3 < -0.01:
                    continue
                blend = w1 * oof_t2 + w2 * oof_tr + w3 * oof_ct
                b = brier_score(y_t2_cv, blend)
                if b < best_b:
                    best_b = b
                    best_w = (w1, w2, w3)
        print(f"    Best blend: tier2={best_w[0]:.0%} + TRank={best_w[1]:.0%} + combined={best_w[2]:.0%}: Brier={best_b:.4f}")

    if n_t2 == n_ie:
        print("\n  2-way blend: tier2 + iter_eff (current best)")
        for w1 in np.arange(0, 1.05, 0.1):
            w2 = 1.0 - w1
            blend = w1 * oof_t2 + w2 * oof_ie
            b = brier_score(y_t2_cv, blend)
            print(f"    tier2={w1:.0%} + iter_eff={w2:.0%}: Brier={b:.4f}")

    if n_t2 == n_ie == n_tr:
        print("\n  3-way blend: tier2 + iter_eff + T-Rank")
        best_b = float('inf')
        best_w = None
        for w1 in np.arange(0, 1.05, 0.1):
            for w2 in np.arange(0, 1.05 - w1, 0.1):
                w3 = 1.0 - w1 - w2
                if w3 < -0.01:
                    continue
                blend = w1 * oof_t2 + w2 * oof_ie + w3 * oof_tr
                b = brier_score(y_t2_cv, blend)
                if b < best_b:
                    best_b = b
                    best_w = (w1, w2, w3)
        print(f"    Best blend: tier2={best_w[0]:.0%} + iter_eff={best_w[1]:.0%} + TRank={best_w[2]:.0%}: Brier={best_b:.4f}")

    print(f"\n  REFERENCE: Women's current best = 0.1387")


if __name__ == "__main__":
    main()
