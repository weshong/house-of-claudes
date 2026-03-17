"""Experiment 11: Enhanced iterative efficiency (v2) for men's and women's.

Tests the improved KenPom clone with:
- Recency weighting
- Home court adjustment
- Possession-weighted averaging
- Preseason regression prior
- Derived SOS from adjusted ratings

Focus: Women's (where we don't have Torvik) and comparing v1 vs v2.
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
from marchmadness.features import adj_efficiency, seeds, ordinals, elo, season_stats
from marchmadness.features.adj_efficiency_v2 import compute as compute_v2
from marchmadness.evaluation.cross_val import leave_season_out_cv
from marchmadness.evaluation.metrics import brier_score
from lightgbm import LGBMClassifier


def build_custom_team_features(data, season, gender, modules):
    """Build team features from a custom list of feature modules."""
    team_df = seeds.compute(data, season, gender)
    if team_df.empty:
        return team_df

    ord_df = ordinals.compute(data, season, gender)
    if not ord_df.empty:
        team_df = team_df.merge(ord_df, on="TeamID", how="left")

    for mod_name, mod_fn in modules:
        mod_df = mod_fn(data, season, gender)
        if not mod_df.empty:
            team_df = team_df.merge(mod_df, on="TeamID", how="left")

    return team_df


def build_custom_training_data(data, seasons, gender, modules):
    """Build training data with custom feature modules."""
    tourney_key = f"{gender}NCAATourneyCompactResults"
    tourney = data[tourney_key]

    all_rows = []
    for season in seasons:
        team_features = build_custom_team_features(data, season, gender, modules)
        if team_features.empty:
            continue

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


def run_cv(model, X, y, seasons, label):
    result = leave_season_out_cv(model, X, y, seasons, CV_SEASONS)
    brier = result["overall"]["brier_score"]
    print(f"  {label}: Brier={brier:.4f}")
    for s in CV_SEASONS:
        if s in result["per_fold"]:
            print(f"    {s}: Brier={result['per_fold'][s]['brier_score']:.4f}")
    return result


def main():
    print("Loading data...")
    data = load_all()

    # ============================================================
    # WOMEN'S: Compare v1 vs v2 iterative efficiency
    # ============================================================
    print(f"\n{'='*70}")
    print("  WOMEN'S: Iterative Efficiency v1 vs v2")
    print(f"{'='*70}")

    w_start = 2003  # Detailed results start year for women: 2010, but seeds go back further

    # Current best: tier2 LR(72%) + iter_eff LR(28%) = 0.1387
    # Let's test v2 iter_eff

    # v1: Current iter_eff
    print("\n--- Women's iter_eff v1 LR C=0.1 ---")
    df_v1 = build_training_data(data, list(range(2003, 2026)), "W", "iter_eff")
    if not df_v1.empty:
        X_v1, y_v1, s_v1, fc_v1 = prep_data(df_v1)
        print(f"  Features ({len(fc_v1)}): {fc_v1[:10]}...")
        lr = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))])
        run_cv(lr, X_v1, y_v1, s_v1, "iter_eff_v1 LR C=0.1")

    # v2: Enhanced iter_eff
    print("\n--- Women's iter_eff v2 LR C=0.1 ---")
    modules_v2 = [("iter_eff_v2", compute_v2)]
    df_v2 = build_custom_training_data(data, list(range(2010, 2026)), "W", modules_v2)
    if not df_v2.empty:
        X_v2, y_v2, s_v2, fc_v2 = prep_data(df_v2)
        print(f"  Features ({len(fc_v2)}): {fc_v2}")
        lr = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))])
        run_cv(lr, X_v2, y_v2, s_v2, "iter_eff_v2 LR C=0.1")

    # v2 with more C values
    for c_val in [0.01, 0.5, 1.0]:
        print(f"\n--- Women's iter_eff v2 LR C={c_val} ---")
        lr = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=c_val, max_iter=1000))])
        run_cv(lr, X_v2, y_v2, s_v2, f"iter_eff_v2 LR C={c_val}")

    # v2 with LGB
    print("\n--- Women's iter_eff v2 LGB d=3 ---")
    lgb = LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    run_cv(lgb, X_v2, y_v2, s_v2, "iter_eff_v2 LGB d=3")

    # Combined: tier2 + v2 iter_eff
    print("\n--- Women's tier2 + iter_eff v2 ---")
    modules_combined = [
        ("elo", elo.compute),
        ("stats", season_stats.compute),
        ("iter_eff_v2", compute_v2),
    ]
    df_combined = build_custom_training_data(data, list(range(2010, 2026)), "W", modules_combined)
    if not df_combined.empty:
        X_comb, y_comb, s_comb, fc_comb = prep_data(df_combined)
        print(f"  Features ({len(fc_comb)})")

        for c_val in [0.01, 0.1, 0.5]:
            lr = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=c_val, max_iter=1000))])
            run_cv(lr, X_comb, y_comb, s_comb, f"tier2+v2 LR C={c_val}")

        lgb = LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        run_cv(lgb, X_comb, y_comb, s_comb, "tier2+v2 LGB d=3")

    # ============================================================
    # WOMEN'S: Blend optimization
    # ============================================================
    print(f"\n{'='*70}")
    print("  WOMEN'S: OOF Blend Optimization")
    print(f"{'='*70}")

    # Collect OOF from best configs
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

    # tier2 LR C=0.1 (current best component)
    df_tier2 = build_training_data(data, list(range(1998, 2026)), "W", "tier2")
    X_t2, y_t2, s_t2, _ = prep_data(df_tier2)
    oof_tier2, y_oof = collect_oof(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
        X_t2, y_t2, s_t2)
    print(f"  tier2 LR C=0.1: {brier_score(y_oof, oof_tier2):.4f}")

    # iter_eff v1 LR C=0.1
    if not df_v1.empty:
        oof_v1, _ = collect_oof(
            lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
            X_v1, y_v1, s_v1)
        print(f"  iter_eff_v1 LR C=0.1: {brier_score(y_oof[:len(oof_v1)], oof_v1):.4f}")

    # iter_eff v2 LR C=0.1 (need matching data length)
    if not df_v2.empty:
        oof_v2, y_oof_v2 = collect_oof(
            lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
            X_v2, y_v2, s_v2)
        print(f"  iter_eff_v2 LR C=0.1: {brier_score(y_oof_v2, oof_v2):.4f}")

    # tier2+v2 combined LR
    if not df_combined.empty:
        oof_comb, y_oof_comb = collect_oof(
            lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
            X_comb, y_comb, s_comb)
        print(f"  tier2+v2 LR C=0.1: {brier_score(y_oof_comb, oof_comb):.4f}")

    # ============================================================
    # MEN'S: Does v2 improve on v1? (Torvik still likely better)
    # ============================================================
    print(f"\n{'='*70}")
    print("  MEN'S: iter_eff v2 (comparison with Torvik)")
    print(f"{'='*70}")

    modules_m_v2 = [("iter_eff_v2", compute_v2)]
    df_m_v2 = build_custom_training_data(data, list(range(2003, 2026)), "M", modules_m_v2)
    if not df_m_v2.empty:
        X_m, y_m, s_m, fc_m = prep_data(df_m_v2)
        print(f"  Features ({len(fc_m)})")

        lr = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))])
        run_cv(lr, X_m, y_m, s_m, "M iter_eff_v2 LR C=0.1")

        lgb = LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        run_cv(lgb, X_m, y_m, s_m, "M iter_eff_v2 LGB d=3")

    print("\n  REFERENCE: Men's Torvik LGB d=3 = 0.1543")
    print("  REFERENCE: Women's tier2(72%)+iter_eff_v1(28%) = 0.1387")
    print("\nDone!")


if __name__ == "__main__":
    main()
