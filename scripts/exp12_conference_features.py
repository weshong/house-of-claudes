"""Experiment 12: Conference strength features for women's.

Since we don't have women's Massey Ordinals or Torvik, conference strength
derived from our own ratings could add signal. Teams from strong conferences
(SEC, Big Ten) are underrated by seed alone.
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
from marchmadness.evaluation.cross_val import leave_season_out_cv
from lightgbm import LGBMClassifier


def compute_conference_strength(data, season, gender, base_features_df):
    """Compute conference-level features from team-level features.

    For each team, adds:
    - ConfStrength: average Elo/rating of conference peers
    - ConfRank: team's rank within its conference
    - ConfSize: number of teams in conference
    """
    conf_key = f"{gender}TeamConferences"
    if conf_key not in data:
        return pd.DataFrame(columns=["TeamID"])

    confs = data[conf_key]
    season_confs = confs[confs["Season"] == season]
    if season_confs.empty:
        return pd.DataFrame(columns=["TeamID"])

    # Get a strength proxy from base features
    merged = base_features_df.merge(season_confs[["TeamID", "ConfAbbrev"]], on="TeamID", how="inner")

    # Use Seed as primary strength proxy (lower = better)
    # Also use Elo if available
    strength_col = None
    if "Elo" in merged.columns:
        strength_col = "Elo"
    elif "WinPct" in merged.columns:
        strength_col = "WinPct"

    if strength_col is None and "Seed" in merged.columns:
        # Use inverted seed as proxy
        merged["_strength"] = 17 - merged["Seed"]  # 1-seed -> 16, 16-seed -> 1
        strength_col = "_strength"

    if strength_col is None:
        return pd.DataFrame(columns=["TeamID"])

    # Compute conference averages (including non-tournament teams from the conference)
    # Use ALL teams in conference from the conference table
    all_conf_teams = season_confs[["TeamID", "ConfAbbrev"]].copy()

    # Get all teams' strength (not just tournament teams)
    # For non-seeded teams, we need a proxy. Use overall team pool stats
    rs_key = f"{gender}RegularSeasonCompactResults"
    rs = data[rs_key]
    season_rs = rs[rs["Season"] == season]

    # Compute simple win percentage for ALL teams
    team_wins = {}
    team_games = {}
    team_ppg = {}
    team_ppg_opp = {}
    for _, g in season_rs.iterrows():
        for tid, opp_tid, score, opp_score in [
            (g["WTeamID"], g["LTeamID"], g["WScore"], g["LScore"]),
            (g["LTeamID"], g["WTeamID"], g["LScore"], g["WScore"]),
        ]:
            team_games[tid] = team_games.get(tid, 0) + 1
            team_ppg[tid] = team_ppg.get(tid, 0) + score
            team_ppg_opp[tid] = team_ppg_opp.get(tid, 0) + opp_score
        team_wins[g["WTeamID"]] = team_wins.get(g["WTeamID"], 0) + 1
        team_wins.setdefault(g["LTeamID"], 0)

    all_strength = {}
    for tid in all_conf_teams["TeamID"]:
        if tid in team_games and team_games[tid] > 0:
            wp = team_wins.get(tid, 0) / team_games[tid]
            pd_val = (team_ppg.get(tid, 0) - team_ppg_opp.get(tid, 0)) / team_games[tid]
            all_strength[tid] = wp * 100 + pd_val  # Combined metric
        else:
            all_strength[tid] = 50  # Default

    all_conf_teams["_all_strength"] = all_conf_teams["TeamID"].map(all_strength)

    # Conference-level aggregates
    conf_stats = all_conf_teams.groupby("ConfAbbrev").agg(
        conf_mean_strength=("_all_strength", "mean"),
        conf_std_strength=("_all_strength", "std"),
        conf_max_strength=("_all_strength", "max"),
        conf_size=("_all_strength", "count"),
    ).fillna(0)

    # Merge back to tournament teams
    result = merged[["TeamID", "ConfAbbrev"]].merge(conf_stats, on="ConfAbbrev", how="left")

    # Team's rank within conference
    all_conf_teams_sorted = all_conf_teams.sort_values("_all_strength", ascending=False)
    conf_ranks = {}
    for conf, group in all_conf_teams_sorted.groupby("ConfAbbrev"):
        for rank, (_, row) in enumerate(group.iterrows(), 1):
            conf_ranks[row["TeamID"]] = rank

    result["conf_rank"] = result["TeamID"].map(conf_ranks).fillna(result["conf_size"] / 2)

    # Team's strength relative to conference average
    team_strength_map = {row["TeamID"]: all_strength.get(row["TeamID"], 50)
                         for _, row in all_conf_teams.iterrows()}
    result["team_strength_vs_conf"] = (
        result["TeamID"].map(team_strength_map).fillna(50) - result["conf_mean_strength"]
    )

    return result[["TeamID", "conf_mean_strength", "conf_std_strength",
                    "conf_max_strength", "conf_size", "conf_rank",
                    "team_strength_vs_conf"]].copy()


def build_conf_training_data(data, seasons, gender, feature_set):
    """Build training data with conference features added."""
    tourney_key = f"{gender}NCAATourneyCompactResults"
    tourney = data[tourney_key]

    all_rows = []
    for season in seasons:
        team_features = build_team_features(data, season, gender, feature_set)
        if team_features.empty:
            continue

        # Add conference features
        conf_df = compute_conference_strength(data, season, gender, team_features)
        if not conf_df.empty:
            team_features = team_features.merge(conf_df, on="TeamID", how="left")

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
    # WOMEN'S: Conference strength features
    # ============================================================
    print(f"\n{'='*70}")
    print("  WOMEN'S: Conference Strength Features")
    print(f"{'='*70}")

    # Baseline: tier2 LR C=0.1
    print("\n--- Baseline: tier2 LR C=0.1 ---")
    df_base = build_training_data(data, list(range(1998, 2026)), "W", "tier2")
    X_base, y_base, s_base, fc_base = prep_data(df_base)
    print(f"  Features ({len(fc_base)})")
    lr = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))])
    run_cv(lr, X_base, y_base, s_base, "tier2 LR C=0.1 baseline")

    # With conference features
    print("\n--- tier2 + conference features, LR C=0.1 ---")
    df_conf = build_conf_training_data(data, list(range(1998, 2026)), "W", "tier2")
    X_conf, y_conf, s_conf, fc_conf = prep_data(df_conf)
    new_feats = [f for f in fc_conf if f not in fc_base]
    print(f"  Features ({len(fc_conf)}), +{len(new_feats)} new: {new_feats}")
    lr2 = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))])
    run_cv(lr2, X_conf, y_conf, s_conf, "tier2+conf LR C=0.1")

    # With different C values
    for c in [0.01, 0.5]:
        lr3 = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=c, max_iter=1000))])
        run_cv(lr3, X_conf, y_conf, s_conf, f"tier2+conf LR C={c}")

    # LGB
    print("\n--- tier2 + conference features, LGB d=3 ---")
    lgb = LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    run_cv(lgb, X_conf, y_conf, s_conf, "tier2+conf LGB d=3")

    # ============================================================
    # MEN'S: Conference features on top of Torvik
    # ============================================================
    print(f"\n{'='*70}")
    print("  MEN'S: Torvik + Conference Strength")
    print(f"{'='*70}")

    print("\n--- Baseline: Torvik LGB d=3 ---")
    df_m_base = build_training_data(data, list(range(2015, 2026)), "M", "torvik")
    X_m_base, y_m_base, s_m_base, fc_m_base = prep_data(df_m_base)
    lgb_m = LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    run_cv(lgb_m, X_m_base, y_m_base, s_m_base, "Torvik LGB d=3 baseline")

    print("\n--- Torvik + conference, LGB d=3 ---")
    df_m_conf = build_conf_training_data(data, list(range(2015, 2026)), "M", "torvik")
    X_m_conf, y_m_conf, s_m_conf, fc_m_conf = prep_data(df_m_conf)
    new_m_feats = [f for f in fc_m_conf if f not in fc_m_base]
    print(f"  +{len(new_m_feats)} new: {new_m_feats}")
    lgb_m2 = LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    run_cv(lgb_m2, X_m_conf, y_m_conf, s_m_conf, "Torvik+conf LGB d=3")

    # ============================================================
    # WOMEN'S: Different training windows
    # ============================================================
    print(f"\n{'='*70}")
    print("  WOMEN'S: Training Window Sensitivity")
    print(f"{'='*70}")

    for start_year in [1998, 2003, 2008, 2012, 2015]:
        df_w = build_training_data(data, list(range(start_year, 2026)), "W", "tier2")
        if df_w.empty:
            continue
        X_w, y_w, s_w, _ = prep_data(df_w)
        lr_w = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))])
        result = leave_season_out_cv(lr_w, X_w, y_w, s_w, CV_SEASONS)
        print(f"  start={start_year}: Brier={result['overall']['brier_score']:.4f} ({len(y_w)} samples)")

    print("\n  REFERENCE: Women's best = 0.1387 (tier2 72% + iter_eff 28%)")
    print("  REFERENCE: Men's best = 0.1543 (Torvik LGB d=3)")


if __name__ == "__main__":
    main()
