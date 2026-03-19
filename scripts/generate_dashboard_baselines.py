"""Generate cached prediction baselines for the tournament dashboard.

Creates two files in outputs/submissions/:
  - baseline_seed_2026.csv  (seed-only logistic model)
  - baseline_rf_2026.csv    (R-model-style Random Forest)

Both contain predictions for all tournament-team matchups in the same
format as the v7 submission (ID, Pred where Pred = P(lower TeamID wins)).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from marchmadness.data_loader import load_all
from marchmadness.features.seeds import parse_seed
from exp_r_model_port import (
    compute_r_model_team_features,
    build_r_style_matchup_features,
)

CLIP = (0.025, 0.975)
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'submissions')


def generate_seed_baseline(m_seed_map, w_seed_map, m_teams, w_teams):
    """Seed-only baseline: logistic function of seed difference."""
    rows = []
    for teams, smap in [(m_teams, m_seed_map), (w_teams, w_seed_map)]:
        for i, t1 in enumerate(teams):
            for t2 in teams[i + 1:]:
                s1, s2 = smap.get(t1, 8), smap.get(t2, 8)
                diff = s2 - s1  # positive = t1 has better (lower) seed
                p = 1.0 / (1.0 + 10 ** (-0.175 * diff))
                rows.append({'ID': f"2026_{t1}_{t2}",
                             'Pred': np.clip(p, *CLIP)})
    return pd.DataFrame(rows)


def generate_rf_baseline(data, m_seed_map, w_seed_map, m_teams, w_teams):
    """R-model-style Random Forest (1000 trees) trained on all historical data."""
    # Build training data from 2003-2025 tournament games
    train_seasons = list(range(2003, 2026))
    all_X, all_y = [], []
    feat_cols = None

    for gender, smap in [('M', m_seed_map), ('W', w_seed_map)]:
        seeds_key = f"{gender}NCAATourneySeeds"
        tourney_key = f"{gender}NCAATourneyCompactResults"
        seeds_df = data[seeds_key]
        tourney = data[tourney_key]

        for season in train_seasons:
            team_feats = compute_r_model_team_features(data, season, gender)
            if team_feats.empty:
                continue

            season_seeds = seeds_df[seeds_df['Season'] == season].copy()
            season_seeds['SeedNum'] = season_seeds['Seed'].apply(parse_seed)
            sm = dict(zip(season_seeds['TeamID'], season_seeds['SeedNum']))

            for _, game in tourney[tourney['Season'] == season].iterrows():
                w_id, l_id = game['WTeamID'], game['LTeamID']
                t_low, t_high = min(w_id, l_id), max(w_id, l_id)
                label = 1 if w_id == t_low else 0

                feats = build_r_style_matchup_features(
                    team_feats, t_low, t_high,
                    sm.get(t_low, 99), sm.get(t_high, 99))
                if not feats:
                    continue

                if feat_cols is None:
                    feat_cols = sorted(feats.keys())
                row = [feats.get(c, np.nan) for c in feat_cols]
                all_X.append(row)
                all_y.append(label)

    X = np.array(all_X)
    y = np.array(all_y)
    print(f"  RF training data: {X.shape[0]} games, {X.shape[1]} features")

    # Impute NaN with column medians
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_medians[j]

    # Train
    rf = RandomForestClassifier(n_estimators=1000, random_state=313, n_jobs=-1)
    rf.fit(X, y)
    print(f"  RF trained on {len(y)} games")

    # Predict 2026 matchups
    rows = []
    for gender, teams, smap in [('M', m_teams, m_seed_map), ('W', w_teams, w_seed_map)]:
        team_feats = compute_r_model_team_features(data, 2026, gender)
        if team_feats.empty:
            print(f"  WARNING: No team features for {gender} 2026")
            continue

        for i, t1 in enumerate(teams):
            for t2 in teams[i + 1:]:
                feats = build_r_style_matchup_features(
                    team_feats, t1, t2,
                    smap.get(t1, 99), smap.get(t2, 99))
                if not feats:
                    p = 0.5
                else:
                    x = np.array([[feats.get(c, np.nan) for c in feat_cols]])
                    for j in range(x.shape[1]):
                        if np.isnan(x[0, j]):
                            x[0, j] = col_medians[j]
                    p = rf.predict_proba(x)[0, 1]

                rows.append({'ID': f"2026_{t1}_{t2}",
                             'Pred': np.clip(p, *CLIP)})

    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    data = load_all()

    # Get 2026 tournament seeds
    seeds_m = data['MNCAATourneySeeds']
    seeds_w = data['WNCAATourneySeeds']
    s2026_m = seeds_m[seeds_m['Season'] == 2026].copy()
    s2026_w = seeds_w[seeds_w['Season'] == 2026].copy()
    s2026_m['SeedNum'] = s2026_m['Seed'].apply(parse_seed)
    s2026_w['SeedNum'] = s2026_w['Seed'].apply(parse_seed)

    m_seed_map = dict(zip(s2026_m['TeamID'], s2026_m['SeedNum']))
    w_seed_map = dict(zip(s2026_w['TeamID'], s2026_w['SeedNum']))
    m_teams = sorted(s2026_m['TeamID'].tolist())
    w_teams = sorted(s2026_w['TeamID'].tolist())

    print(f"Tournament teams: {len(m_teams)} men's, {len(w_teams)} women's")

    # Seed baseline
    print("\nGenerating seed baseline...")
    seed_df = generate_seed_baseline(m_seed_map, w_seed_map, m_teams, w_teams)
    seed_path = os.path.join(OUT_DIR, 'baseline_seed_2026.csv')
    seed_df.to_csv(seed_path, index=False)
    print(f"  Saved: {seed_path} ({len(seed_df)} matchups)")

    # RF baseline
    print("\nGenerating RF baseline...")
    rf_df = generate_rf_baseline(data, m_seed_map, w_seed_map, m_teams, w_teams)
    rf_path = os.path.join(OUT_DIR, 'baseline_rf_2026.csv')
    rf_df.to_csv(rf_path, index=False)
    print(f"  Saved: {rf_path} ({len(rf_df)} matchups)")

    print("\nDone! All baselines cached.")


if __name__ == '__main__':
    main()
