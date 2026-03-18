"""Autoresearch experiment file. THIS IS THE FILE THE AGENT MODIFIES.

Current best scores:
  Men's:   0.1307  (55% LGB d=3 + 45% L2 LR C=100, ordinal aggs + Torvik + TRank + gaps)
  Women's: 0.1365  (seeds + Elo + TRank clone, L1 LR C=0.15)

The agent modifies the functions below, then runs this file to evaluate.
If Brier improves, keep the change. If not, revert.

Available data sources:
  - Seeds (M+W): parsed tournament seed numbers
  - Massey ordinals (M only): 197 ranking systems, top 10 used
  - Elo ratings (M+W): computed from game results
  - Season stats (M+W): win%, PPG, point diff, SOS
  - Efficiency (M+W, 2003+): KenPom-style AdjO/AdjD/AdjEM/Tempo from detailed results
  - Four factors (M+W, 2003+): eFG%, TO rate, OR%, FT rate
  - Iterative adj efficiency (M+W, 2003+): opponent-adjusted, 15 iterations
  - T-Rank clone (M+W): Torvik-style ratings from box scores (vectorized)
  - Torvik external (M only, 2015+): real Bart Torvik T-Rank ratings
  - Women's Torvik (W, 2025/2026 only): scraped from barttorvik.com

Available model libraries: sklearn, lightgbm, numpy, scipy
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).parent / "src"))

from marchmadness.data_loader import load_all
from marchmadness.features.builder import (
    build_team_features, build_matchup_features, build_training_data, get_feature_columns
)
from marchmadness.features import seeds, ordinals, elo, season_stats, efficiency, four_factors, adj_efficiency
from marchmadness.features.trank_clone import compute as compute_trank

# ============================================================
# CONFIGURATION — modify these to experiment
# ============================================================

# Men's config — blend LGB + LR
M_CONFIG = {
    "blend": True,
    "models": [
        {
            "feature_set": "custom",
            "start_year": 2015,
            "model": lambda: LGBMClassifier(
                n_estimators=250, max_depth=3, learning_rate=0.025,
                subsample=0.8, colsample_bytree=0.6, min_split_gain=0.005,
                random_state=42, verbose=-1
            ),
            "weight": 0.55,
        },
        {
            "feature_set": "custom",
            "start_year": 2015,
            "model": lambda: Pipeline([
                ("s", StandardScaler()),
                ("lr", LogisticRegression(C=100.0, penalty='l2', solver='lbfgs', max_iter=2000))
            ]),
            "weight": 0.45,
        },
    ],
}

# Women's config — blend of two models
W_CONFIG = {
    "feature_set": "custom",
    "start_year": 2003,
    "model": lambda: Pipeline([
        ("s", StandardScaler()),
        ("lr", LogisticRegression(C=0.15, penalty='l1', solver='liblinear', max_iter=2000))
    ]),
}


# ============================================================
# CUSTOM FEATURE BUILDING — modify to add/change features
# ============================================================

def build_custom_team_features(data, season, gender):
    """Build custom team-level features. Override this to add new features.

    Return a DataFrame with TeamID column + feature columns.
    Called when feature_set="custom" in config.
    """
    from marchmadness.features import torvik

    if gender == "W":
        # For women's: seeds + Elo + T-Rank clone + PointDiff
        team_df = seeds.compute(data, season, gender)
        if team_df.empty:
            return team_df

        elo_df = elo.compute(data, season, gender)
        if not elo_df.empty:
            team_df = team_df.merge(elo_df, on="TeamID", how="left")

        trank_df = compute_trank(data, season, gender)
        if not trank_df.empty:
            team_df = team_df.merge(trank_df[["TeamID", "TRank_barthag", "TRank_adjoe", "TRank_adjde", "TRank_adjem"]], on="TeamID", how="left")

        stats_df = season_stats.compute(data, season, gender)
        if not stats_df.empty:
            team_df = team_df.merge(stats_df[["TeamID", "PointDiff"]], on="TeamID", how="left")

        return team_df

    # Men's: custom build with seeds, ordinals, Torvik, Elo, T-Rank clone + disagreement
    team_df = seeds.compute(data, season, gender)
    if team_df.empty:
        return team_df

    # Add ordinals (PCA only — PCA replaces mean/std)
    # OrdinalMean and OrdinalStd removed — PCA captures these better

    # PCA on all ordinal systems
    ord_key = f"{gender}MasseyOrdinals"
    if ord_key in data:
        from sklearn.decomposition import PCA
        all_ord = data[ord_key]
        season_ord = all_ord[
            (all_ord["Season"] == season) &
            (all_ord["RankingDayNum"] >= 128)
        ]
        if not season_ord.empty:
            # Get all systems with enough coverage
            sys_counts = season_ord.groupby("SystemName")["TeamID"].nunique()
            good_sys = sys_counts[sys_counts >= 50].index.tolist()
            # Pivot to team x system matrix
            latest = season_ord[season_ord["SystemName"].isin(good_sys)]
            latest = latest.loc[latest.groupby(["TeamID", "SystemName"])["RankingDayNum"].idxmax()]
            pivot = latest.pivot(index="TeamID", columns="SystemName", values="OrdinalRank")
            pivot = pivot.fillna(pivot.median())
            if pivot.shape[1] >= 5:
                n_components = min(2, pivot.shape[1])
                pca = PCA(n_components=n_components)
                comps = pca.fit_transform(pivot.values)
                pca_df = pd.DataFrame({"TeamID": pivot.index})
                for i in range(n_components):
                    pca_df[f"OrdPCA_{i}"] = comps[:, i]
                team_df = team_df.merge(pca_df, on="TeamID", how="left")

    # Add Torvik external ratings
    torvik_df = torvik.compute(data, season, gender)
    if not torvik_df.empty:
        team_df = team_df.merge(torvik_df, on="TeamID", how="left")

    # Add Elo
    elo_df = elo.compute(data, season, gender)
    if not elo_df.empty:
        team_df = team_df.merge(elo_df, on="TeamID", how="left")

    # Add T-Rank clone features (complementary to external Torvik)
    trank_df = compute_trank(data, season, gender)
    if not trank_df.empty:
        team_df = team_df.merge(trank_df[["TeamID", "TRank_barthag", "TRank_adjoe", "TRank_adjde", "TRank_adjem", "TRank_sos", "TRank_tempo"]], on="TeamID", how="left")

    # Disagreement between external Torvik and our clone
    if "Torvik_barthag" in team_df.columns and "TRank_barthag" in team_df.columns:
        team_df["barthag_gap"] = team_df["Torvik_barthag"] - team_df["TRank_barthag"]
    if "Torvik_adjoe" in team_df.columns and "TRank_adjoe" in team_df.columns:
        team_df["adjoe_gap"] = team_df["Torvik_adjoe"] - team_df["TRank_adjoe"]
    if "Torvik_adjde" in team_df.columns and "TRank_adjde" in team_df.columns:
        team_df["adjde_gap"] = team_df["Torvik_adjde"] - team_df["TRank_adjde"]
    if "Torvik_sos" in team_df.columns and "TRank_sos" in team_df.columns:
        team_df["sos_gap"] = team_df["Torvik_sos"] - team_df["TRank_sos"]
    if "Torvik_adjt" in team_df.columns and "TRank_tempo" in team_df.columns:
        team_df["tempo_gap"] = team_df["Torvik_adjt"] - team_df["TRank_tempo"]

    # Elo-Torvik disagreement: normalize Elo to similar scale as barthag
    if "Torvik_barthag" in team_df.columns and "Elo" in team_df.columns:
        # Convert Elo to win probability scale (~0-1) to compare with barthag
        elo_norm = 1 / (1 + 10 ** ((1500 - team_df["Elo"]) / 400))
        team_df["elo_barthag_gap"] = elo_norm - team_df["Torvik_barthag"]

    return team_df


def build_custom_training_data(data, seasons, gender):
    """Build training data using custom features."""
    tourney_key = f"{gender}NCAATourneyCompactResults"
    tourney = data[tourney_key]

    all_rows = []
    for season in seasons:
        team_features = build_custom_team_features(data, season, gender)
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


# ============================================================
# EVALUATION — runs CV and reports results
# ============================================================

def build_data_fn(data, gender, feature_set, start_year):
    """Build training data using the specified feature set."""
    seasons = list(range(start_year, 2026))
    if feature_set == "custom":
        return build_custom_training_data(data, seasons, gender)
    else:
        return build_training_data(data, seasons, gender, feature_set)


def run_evaluation():
    """Run full evaluation. Prints results and returns Brier scores."""
    from evaluate import evaluate_single_model, evaluate_blend, load_data

    data = load_data()
    results = {}

    # === MEN'S ===
    print("=" * 60)
    print("  MEN'S EVALUATION")
    print("=" * 60)

    if M_CONFIG.get("blend"):
        m_results = []
        for i, mc in enumerate(M_CONFIG["models"]):
            m_df = build_data_fn(data, "M", mc["feature_set"], mc["start_year"])
            fc = get_feature_columns(m_df)

            def m_build(data, gender, _df=m_df):
                return _df

            r = evaluate_single_model(m_build, mc["model"], data, "M",
                                      f"M model {i} ({mc['feature_set']})")
            m_results.append(r)
            print(f"  Model {i} ({mc['feature_set']}): {r['brier']:.6f} ({len(fc)} features)")

        weights = [mc["weight"] for mc in M_CONFIG["models"]]
        n_games = [len(r["oof_preds"]) for r in m_results]
        if len(set(n_games)) == 1:
            blend_result = evaluate_blend(m_results, weights)
            m_brier = blend_result["brier"]
            wt_str = " + ".join(f"{mc['feature_set']}({mc['weight']:.0%})" for mc in M_CONFIG["models"])
            print(f"  Blend [{wt_str}]: {m_brier:.6f}")
        else:
            print(f"  WARNING: CV game count mismatch: {n_games}")
            m_brier = min(r["brier"] for r in m_results)
    else:
        m_df = build_data_fn(data, "M", M_CONFIG["feature_set"], M_CONFIG["start_year"])
        feat_cols = get_feature_columns(m_df)

        def m_build(data, gender):
            return m_df

        m_result = evaluate_single_model(m_build, M_CONFIG["model"], data, "M", "Men's")
        m_brier = m_result["brier"]
        print(f"  Men's Brier: {m_brier:.6f}  ({len(feat_cols)} features, {m_result.get('n_samples', 0)} samples)")
        for s, b in sorted(m_result.get("per_fold", {}).items()):
            print(f"    {s}: {b:.4f}")
    results["M"] = m_brier

    # === WOMEN'S ===
    print(f"\n{'=' * 60}")
    print("  WOMEN'S EVALUATION")
    print("=" * 60)

    if W_CONFIG.get("blend"):
        w_results = []
        for i, mc in enumerate(W_CONFIG["models"]):
            w_df = build_data_fn(data, "W", mc["feature_set"], mc["start_year"])
            fc = get_feature_columns(w_df)

            def w_build(data, gender, _df=w_df):
                return _df

            r = evaluate_single_model(w_build, mc["model"], data, "W",
                                      f"W model {i} ({mc['feature_set']})")
            w_results.append(r)
            print(f"  Model {i} ({mc['feature_set']}): {r['brier']:.6f} ({len(fc)} features)")

        weights = [mc["weight"] for mc in W_CONFIG["models"]]

        # Check all have same CV games
        n_games = [len(r["oof_preds"]) for r in w_results]
        if len(set(n_games)) == 1:
            blend_result = evaluate_blend(w_results, weights)
            w_brier = blend_result["brier"]
            wt_str = " + ".join(f"{mc['feature_set']}({mc['weight']:.0%})" for mc in W_CONFIG["models"])
            print(f"  Blend [{wt_str}]: {w_brier:.6f}")
        else:
            print(f"  WARNING: CV game count mismatch: {n_games}")
            w_brier = min(r["brier"] for r in w_results)
    else:
        w_df = build_data_fn(data, "W", W_CONFIG["feature_set"], W_CONFIG["start_year"])
        fc = get_feature_columns(w_df)

        def w_build(data, gender):
            return w_df

        w_result = evaluate_single_model(w_build, W_CONFIG["model"], data, "W", "Women's")
        w_brier = w_result["brier"]
        print(f"  Women's Brier: {w_brier:.6f}  ({len(fc)} features)")
        for s, b in sorted(w_result.get("per_fold", {}).items()):
            print(f"    {s}: {b:.4f}")

    results["W"] = w_brier

    # === SUMMARY ===
    combined = (m_brier + w_brier) / 2
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Men's:    {m_brier:.6f}  (baseline: 0.1543)")
    print(f"  Women's:  {w_brier:.6f}  (baseline: 0.1386)")
    print(f"  Combined: {combined:.6f}  (baseline: 0.1465)")
    print()

    if m_brier < 0.1543:
        print(f"  *** MEN'S IMPROVED by {0.1543 - m_brier:.6f} ***")
    if w_brier < 0.1386:
        print(f"  *** WOMEN'S IMPROVED by {0.1386 - w_brier:.6f} ***")
    if m_brier >= 0.1543 and w_brier >= 0.1386:
        print(f"  No improvement over baseline.")

    return results


if __name__ == "__main__":
    run_evaluation()
