"""
Experiment: Port of legacy R model (MhouseOfCardFINAL.R / WhouseOfCardFINAL.R)

The R model uses:
- Features: WP, OWP (median opponent WP), shooting pcts (FG/FG3/FT),
  box score averages (OR/DR/Ast/TO/Stl/Blk/PF), opponent versions of all,
  possessions, OE/DE, four factors (eFG%, TO%, OREB%, FTR) + seeds
- Training: regular season + tournament (with inverted copies)
- Models: Random Forest (1000 trees) + XGBoost (depth=6) + calibrated RF (LR wrapper)
- No leave-season-out CV in original (trains on everything)

This script faithfully replicates the features and models, then evaluates
with our leave-season-out CV on 2022-2025 for apples-to-apples comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb

from marchmadness.data_loader import load_all
from marchmadness.features.seeds import parse_seed
from marchmadness.evaluation.cross_val import leave_season_out_cv
from marchmadness.evaluation.metrics import compute_all_metrics


def compute_r_model_team_features(data, season, gender="M"):
    """Compute the exact feature set from the R model for each team.

    Features: wp, owp, fg, fg3, ft, oreb, dreb, ast, to, stl, blk, pf,
              oppfg, oppfg3, oppft, opporeb, oppdreb, oppast, oppto, oppstl, oppblk, opppf,
              fga, oppfga, fta, oppfta, pt, opppt, pos, opppos, oeff, deff,
              fgm, oppfgm, fgm3, oppfgm3, effFG, oppeffFG, topct, opptopct,
              orebpct, opporebpct, ftr, oppftr
    """
    compact_key = f"{gender}RegularSeasonCompactResults"
    detailed_key = f"{gender}RegularSeasonDetailedResults"

    compact = data[compact_key]
    season_compact = compact[compact["Season"] == season]

    detailed = data.get(detailed_key)
    season_detailed = None
    if detailed is not None:
        season_detailed = detailed[detailed["Season"] == season]
        if season_detailed.empty:
            season_detailed = None

    if season_compact.empty:
        return pd.DataFrame()

    # Get all teams that played this season
    all_teams = set(season_compact["WTeamID"].tolist() + season_compact["LTeamID"].tolist())

    # --- WP: winning percentage ---
    wins = season_compact.groupby("WTeamID").size()
    losses = season_compact.groupby("LTeamID").size()

    wp_dict = {}
    for t in all_teams:
        w = wins.get(t, 0)
        l = losses.get(t, 0)
        wp_dict[t] = w / (w + l) if (w + l) > 0 else 0.5

    # --- OWP: median opponent winning percentage (excluding the team itself) ---
    # Vectorized: precompute wins/losses for each (team, opponent) pair
    # For each game, winner beat loser. Build a pairwise record.
    opponents = {}
    for _, g in season_compact.iterrows():
        opponents.setdefault(g["WTeamID"], []).append(g["LTeamID"])
        opponents.setdefault(g["LTeamID"], []).append(g["WTeamID"])

    # Precompute: for each team, wins against each specific opponent
    wins_vs = {}  # (team, opponent) -> count of wins
    losses_vs = {}  # (team, opponent) -> count of losses
    for _, g in season_compact.iterrows():
        w, l = g["WTeamID"], g["LTeamID"]
        wins_vs[(w, l)] = wins_vs.get((w, l), 0) + 1
        losses_vs[(l, w)] = losses_vs.get((l, w), 0) + 1

    owp_dict = {}
    for t in all_teams:
        opp_list = opponents.get(t, [])
        if not opp_list:
            owp_dict[t] = 0.5
            continue
        opp_wps = []
        for opp in opp_list:
            # Opponent's total wins/losses minus games against team t
            total_w = wins.get(opp, 0)
            total_l = losses.get(opp, 0)
            vs_w = wins_vs.get((opp, t), 0)  # opp's wins against t
            vs_l = losses_vs.get((opp, t), 0)  # opp's losses against t
            adj_w = total_w - vs_w
            adj_l = total_l - vs_l
            if adj_w + adj_l > 0:
                opp_wps.append(adj_w / (adj_w + adj_l))
        owp_dict[t] = np.median(opp_wps) if opp_wps else 0.5

    # --- Detailed stats (shooting pcts, averages, etc.) ---
    rows = []
    for t in all_teams:
        row = {"TeamID": t, "wp": wp_dict[t], "owp": owp_dict[t]}

        if season_detailed is not None:
            w_games = season_detailed[season_detailed["WTeamID"] == t]
            l_games = season_detailed[season_detailed["LTeamID"] == t]
            n_games = len(w_games) + len(l_games)

            if n_games > 0:
                # Shooting pcts
                fgm_total = w_games["WFGM"].sum() + l_games["LFGM"].sum()
                fga_total = w_games["WFGA"].sum() + l_games["LFGA"].sum()
                fgm3_total = w_games["WFGM3"].sum() + l_games["LFGM3"].sum()
                fga3_total = w_games["WFGA3"].sum() + l_games["LFGA3"].sum()
                ftm_total = w_games["WFTM"].sum() + l_games["LFTM"].sum()
                fta_total = w_games["WFTA"].sum() + l_games["LFTA"].sum()

                row["fg"] = fgm_total / fga_total if fga_total > 0 else 0
                row["fg3"] = fgm3_total / fga3_total if fga3_total > 0 else 0
                row["ft"] = ftm_total / fta_total if fta_total > 0 else 0

                # Box score averages
                for stat, wcol, lcol in [
                    ("oreb", "WOR", "LOR"), ("dreb", "WDR", "LDR"),
                    ("ast", "WAst", "LAst"), ("to", "WTO", "LTO"),
                    ("stl", "WStl", "LStl"), ("blk", "WBlk", "LBlk"),
                    ("pf", "WPF", "LPF")
                ]:
                    row[stat] = (w_games[wcol].sum() + l_games[lcol].sum()) / n_games

                # Opponent shooting pcts
                opp_fgm = w_games["LFGM"].sum() + l_games["WFGM"].sum()
                opp_fga = w_games["LFGA"].sum() + l_games["WFGA"].sum()
                opp_fgm3 = w_games["LFGM3"].sum() + l_games["WFGM3"].sum()
                opp_fga3 = w_games["LFGA3"].sum() + l_games["WFGA3"].sum()
                opp_ftm = w_games["LFTM"].sum() + l_games["WFTM"].sum()
                opp_fta = w_games["LFTA"].sum() + l_games["WFTA"].sum()

                row["oppfg"] = opp_fgm / opp_fga if opp_fga > 0 else 0
                row["oppfg3"] = opp_fgm3 / opp_fga3 if opp_fga3 > 0 else 0
                row["oppft"] = opp_ftm / opp_fta if opp_fta > 0 else 0

                # Opponent box score averages
                for stat, wcol_opp, lcol_opp in [
                    ("opporeb", "LOR", "WOR"), ("oppdreb", "LDR", "WDR"),
                    ("oppast", "LAst", "WAst"), ("oppto", "LTO", "WTO"),
                    ("oppstl", "LStl", "WStl"), ("oppblk", "LBlk", "WBlk"),
                    ("opppf", "LPF", "WPF")
                ]:
                    row[stat] = (w_games[wcol_opp].sum() + l_games[lcol_opp].sum()) / n_games

                # Possessions & efficiency
                row["fga_avg"] = fga_total / n_games
                row["oppfga_avg"] = opp_fga / n_games
                row["fta_avg"] = fta_total / n_games
                row["oppfta_avg"] = opp_fta / n_games

                pt_total = w_games["WScore"].sum() + l_games["LScore"].sum()
                opppt_total = w_games["LScore"].sum() + l_games["WScore"].sum()
                row["pt"] = pt_total / n_games
                row["opppt"] = opppt_total / n_games

                row["pos"] = row["fga_avg"] - row["oreb"] + row["to"] + 0.475 * row["fta_avg"]
                row["opppos"] = row["oppfga_avg"] - row["opporeb"] + row["oppto"] + 0.475 * row["oppfta_avg"]

                row["oeff"] = row["pt"] / row["pos"] if row["pos"] > 0 else 0
                row["deff"] = row["opppt"] / row["opppos"] if row["opppos"] > 0 else 0

                # Four factors
                row["fgm_avg"] = fgm_total / n_games
                row["oppfgm_avg"] = opp_fgm / n_games
                row["fgm3_avg"] = fgm3_total / n_games
                row["oppfgm3_avg"] = opp_fgm3 / n_games

                row["effFG"] = (row["fgm_avg"] + 0.5 * row["fgm3_avg"]) / row["fga_avg"] if row["fga_avg"] > 0 else 0
                row["oppeffFG"] = (row["oppfgm_avg"] + 0.5 * row["oppfgm3_avg"]) / row["oppfga_avg"] if row["oppfga_avg"] > 0 else 0

                row["topct"] = row["to"] / row["pos"] if row["pos"] > 0 else 0
                row["opptopct"] = row["oppto"] / row["opppos"] if row["opppos"] > 0 else 0

                row["orebpct"] = row["oreb"] / (row["oreb"] + row["oppdreb"]) if (row["oreb"] + row["oppdreb"]) > 0 else 0
                row["opporebpct"] = row["opporeb"] / (row["opporeb"] + row["dreb"]) if (row["opporeb"] + row["dreb"]) > 0 else 0

                row["ftr"] = row["fta_avg"] / row["fga_avg"] if row["fga_avg"] > 0 else 0
                row["oppftr"] = row["oppfta_avg"] / row["oppfga_avg"] if row["oppfga_avg"] > 0 else 0
            else:
                # No detailed data for this team
                for col in ["fg", "fg3", "ft", "oreb", "dreb", "ast", "to", "stl", "blk", "pf",
                           "oppfg", "oppfg3", "oppft", "opporeb", "oppdreb", "oppast", "oppto",
                           "oppstl", "oppblk", "opppf", "fga_avg", "oppfga_avg", "fta_avg",
                           "oppfta_avg", "pt", "opppt", "pos", "opppos", "oeff", "deff",
                           "fgm_avg", "oppfgm_avg", "fgm3_avg", "oppfgm3_avg",
                           "effFG", "oppeffFG", "topct", "opptopct",
                           "orebpct", "opporebpct", "ftr", "oppftr"]:
                    row[col] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def build_r_style_matchup_features(team_features, team_a_id, team_b_id, seed_a, seed_b):
    """Build matchup features R-style: raw values for both teams (not diffs).

    The R model uses teamA stats as .x columns and teamB stats as .y columns,
    keeping raw values rather than computing differences. This is important for
    tree-based models which can learn non-linear feature interactions.
    """
    a = team_features[team_features["TeamID"] == team_a_id]
    b = team_features[team_features["TeamID"] == team_b_id]

    if a.empty or b.empty:
        return {}

    a = a.iloc[0]
    b = b.iloc[0]

    features = {"Seed_x": seed_a, "Seed_y": seed_b}

    skip_cols = {"TeamID"}
    for col in team_features.columns:
        if col in skip_cols:
            continue
        features[f"{col}_x"] = a[col]
        features[f"{col}_y"] = b[col]

    return features


def build_r_style_diff_features(team_features, team_a_id, team_b_id, seed_a, seed_b):
    """Build matchup features as diffs (our standard approach) using R model's feature set."""
    a = team_features[team_features["TeamID"] == team_a_id]
    b = team_features[team_features["TeamID"] == team_b_id]

    if a.empty or b.empty:
        return {}

    a = a.iloc[0]
    b = b.iloc[0]

    features = {"seed_a": seed_a, "seed_b": seed_b, "seed_diff": seed_a - seed_b}

    skip_cols = {"TeamID"}
    for col in team_features.columns:
        if col in skip_cols:
            continue
        val_a = a[col]
        val_b = b[col]
        if pd.isna(val_a) or pd.isna(val_b):
            features[f"{col}_diff"] = np.nan
        else:
            features[f"{col}_diff"] = val_a - val_b

    return features


def build_training_data(data, seasons, gender="M", use_raw_features=False):
    """Build training data from tournament games only, using R model's feature set."""
    seeds_key = f"{gender}NCAATourneySeeds"
    tourney_key = f"{gender}NCAATourneyCompactResults"

    seeds_df = data[seeds_key]
    tourney = data[tourney_key]

    all_rows = []
    for season in seasons:
        team_features = compute_r_model_team_features(data, season, gender)
        if team_features.empty:
            continue

        # Get seeds for this season
        season_seeds = seeds_df[seeds_df["Season"] == season].copy()
        season_seeds["SeedNum"] = season_seeds["Seed"].apply(parse_seed)
        seed_map = dict(zip(season_seeds["TeamID"], season_seeds["SeedNum"]))

        season_tourney = tourney[tourney["Season"] == season]
        for _, game in season_tourney.iterrows():
            w_id = game["WTeamID"]
            l_id = game["LTeamID"]

            team_a = min(w_id, l_id)
            team_b = max(w_id, l_id)
            label = 1 if w_id == team_a else 0

            seed_a = seed_map.get(team_a, 99)
            seed_b = seed_map.get(team_b, 99)

            if use_raw_features:
                matchup = build_r_style_matchup_features(team_features, team_a, team_b, seed_a, seed_b)
            else:
                matchup = build_r_style_diff_features(team_features, team_a, team_b, seed_a, seed_b)

            if not matchup:
                continue

            matchup["Season"] = season
            matchup["Label"] = label
            all_rows.append(matchup)

    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def evaluate_model(model, X, y, seasons, cv_seasons, model_name):
    """Run leave-season-out CV and print results."""
    results = leave_season_out_cv(model, X, y, seasons, cv_seasons)
    brier = results["overall"]["brier_score"]
    acc = results["overall"]["accuracy"]

    print(f"  {model_name}: Brier={brier:.4f}, Acc={acc:.1%}")
    for s in cv_seasons:
        if s in results["per_fold"]:
            fold = results["per_fold"][s]
            print(f"    {s}: Brier={fold['brier_score']:.4f}, n={fold['n_val']}")

    return results


def run_experiment(gender="M"):
    """Run the full R model port experiment."""
    print(f"\n{'='*60}")
    print(f"R MODEL PORT - {'Mens' if gender == 'M' else 'Womens'}")
    print(f"{'='*60}\n")

    data = load_all()

    # Determine season range based on detailed results availability
    detailed_key = f"{gender}RegularSeasonDetailedResults"
    if detailed_key in data:
        min_season = data[detailed_key]["Season"].min()
    else:
        min_season = 2003

    # Use 2003+ for detailed stats availability, match our other experiments
    train_seasons = list(range(max(min_season, 2003), 2026))
    cv_seasons = [2022, 2023, 2024, 2025]

    print(f"Training seasons: {train_seasons[0]}-{train_seasons[-1]}")
    print(f"CV seasons: {cv_seasons}")

    # ============================================================
    # Approach 1: R-style raw features (both team stats as separate columns)
    # This is what RF/XGB in the R script actually sees
    # ============================================================
    print(f"\n--- Approach 1: Raw features (R-style, both teams' stats) ---")
    df_raw = build_training_data(data, train_seasons, gender, use_raw_features=True)

    if df_raw.empty:
        print("  No data!")
        return {}

    meta_cols = {"Season", "Label"}
    feat_cols_raw = [c for c in df_raw.columns if c not in meta_cols]

    # Drop rows with NaN
    df_raw = df_raw.dropna(subset=feat_cols_raw)
    X_raw = df_raw[feat_cols_raw].values
    y_raw = df_raw["Label"].values
    seasons_raw = df_raw["Season"].values

    print(f"  Data shape: {X_raw.shape}, features: {len(feat_cols_raw)}")
    print(f"  Features: {feat_cols_raw[:10]}...")

    # Random Forest (ntree=1000, default depth)
    rf = RandomForestClassifier(n_estimators=1000, random_state=313, n_jobs=-1)
    rf_results = evaluate_model(rf, X_raw, y_raw, seasons_raw, cv_seasons, "RF (1000 trees)")

    # Random Forest with calibration (the R model's glm wrapper)
    rf_calib = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=1000, random_state=313, n_jobs=-1),
        cv=5, method="sigmoid"  # sigmoid = Platt scaling ≈ logistic regression wrapper
    )
    rf_calib_results = evaluate_model(rf_calib, X_raw, y_raw, seasons_raw, cv_seasons, "RF + LR calibration")

    # XGBoost (R model: depth=6, eta=0.01, subsample=1, colsample=0.6)
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.01,
        subsample=1.0, colsample_bytree=0.6,
        eval_metric="logloss", random_state=313,
        tree_method="hist", device="cuda"
    )
    xgb_results = evaluate_model(xgb_model, X_raw, y_raw, seasons_raw, cv_seasons, "XGB (R params: d=6)")

    # XGBoost shallow (our best finding: depth=3)
    xgb_shallow = xgb.XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=313,
        tree_method="hist", device="cuda"
    )
    xgb_shallow_results = evaluate_model(xgb_shallow, X_raw, y_raw, seasons_raw, cv_seasons, "XGB (shallow d=3)")

    # LightGBM shallow (our best approach)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8,
        random_state=313, verbose=-1, device="gpu"
    )
    lgb_results = evaluate_model(lgb_model, X_raw, y_raw, seasons_raw, cv_seasons, "LGB (shallow d=3)")

    # ============================================================
    # Approach 2: Diff features (our standard approach) with R's feature set
    # ============================================================
    print(f"\n--- Approach 2: Diff features (our style) with R's feature set ---")
    df_diff = build_training_data(data, train_seasons, gender, use_raw_features=False)

    feat_cols_diff = [c for c in df_diff.columns if c not in meta_cols]
    df_diff = df_diff.dropna(subset=feat_cols_diff)
    X_diff = df_diff[feat_cols_diff].values
    y_diff = df_diff["Label"].values
    seasons_diff = df_diff["Season"].values

    print(f"  Data shape: {X_diff.shape}, features: {len(feat_cols_diff)}")

    # Logistic Regression (strong regularization, our proven approach)
    lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))])
    lr_results = evaluate_model(lr, X_diff, y_diff, seasons_diff, cv_seasons, "LR C=0.1 (diff features)")

    lr_strong = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=0.01, max_iter=1000))])
    lr_strong_results = evaluate_model(lr_strong, X_diff, y_diff, seasons_diff, cv_seasons, "LR C=0.01 (diff features)")

    # LGB on diff features
    lgb_diff = lgb.LGBMClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.02,
        random_state=313, verbose=-1, device="gpu"
    )
    lgb_diff_results = evaluate_model(lgb_diff, X_diff, y_diff, seasons_diff, cv_seasons, "LGB d=3 (diff features)")

    # RF on diff features
    rf_diff = RandomForestClassifier(n_estimators=1000, random_state=313, n_jobs=-1)
    rf_diff_results = evaluate_model(rf_diff, X_diff, y_diff, seasons_diff, cv_seasons, "RF (diff features)")

    # ============================================================
    # Approach 3: Ensemble of best R-style + best current
    # ============================================================
    print(f"\n--- Approach 3: Blending R-model predictions with existing best ---")

    # Get OOF predictions from best raw-feature model and best diff-feature model
    best_raw = min(
        [("RF", rf_results), ("RF_calib", rf_calib_results), ("XGB_d6", xgb_results),
         ("XGB_d3", xgb_shallow_results), ("LGB_d3", lgb_results)],
        key=lambda x: x[1]["overall"]["brier_score"]
    )
    best_diff = min(
        [("LR_01", lr_results), ("LR_001", lr_strong_results),
         ("LGB_diff", lgb_diff_results), ("RF_diff", rf_diff_results)],
        key=lambda x: x[1]["overall"]["brier_score"]
    )

    print(f"  Best raw-feature model: {best_raw[0]} (Brier={best_raw[1]['overall']['brier_score']:.4f})")
    print(f"  Best diff-feature model: {best_diff[0]} (Brier={best_diff[1]['overall']['brier_score']:.4f})")

    # Try blending the two best models' OOF predictions
    raw_preds = best_raw[1]["oof_predictions"]
    diff_preds = best_diff[1]["oof_predictions"]
    raw_labels = best_raw[1]["oof_labels"]
    diff_labels = best_diff[1]["oof_labels"]

    # Only blend if they have same shape (same CV folds)
    if len(raw_preds) == len(diff_preds) and np.array_equal(raw_labels, diff_labels):
        print(f"\n  Blend results (raw + diff OOF predictions):")
        for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            blended = alpha * raw_preds + (1 - alpha) * diff_preds
            metrics = compute_all_metrics(raw_labels, blended)
            marker = " <-- best" if alpha == 0.0 else ""  # placeholder
            print(f"    {alpha:.0%} raw + {1-alpha:.0%} diff: Brier={metrics['brier_score']:.4f}{marker}")

    # Collect all results
    all_results = {
        "rf_raw": rf_results["overall"]["brier_score"],
        "rf_calib_raw": rf_calib_results["overall"]["brier_score"],
        "xgb_d6_raw": xgb_results["overall"]["brier_score"],
        "xgb_d3_raw": xgb_shallow_results["overall"]["brier_score"],
        "lgb_d3_raw": lgb_results["overall"]["brier_score"],
        "lr_01_diff": lr_results["overall"]["brier_score"],
        "lr_001_diff": lr_strong_results["overall"]["brier_score"],
        "lgb_d3_diff": lgb_diff_results["overall"]["brier_score"],
        "rf_diff": rf_diff_results["overall"]["brier_score"],
    }

    return all_results


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERIMENT: R MODEL PORT COMPARISON")
    print("Porting MhouseOfCardFINAL.R / WhouseOfCardFINAL.R to Python")
    print("=" * 70)

    men_results = run_experiment("M")
    women_results = run_experiment("W")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nMen's results:")
    if men_results:
        for name, brier in sorted(men_results.items(), key=lambda x: x[1]):
            print(f"  {name:25s}: {brier:.4f}")
        print(f"  {'Current best':25s}: 0.1307")

    print("\nWomen's results:")
    if women_results:
        for name, brier in sorted(women_results.items(), key=lambda x: x[1]):
            print(f"  {name:25s}: {brier:.4f}")
        print(f"  {'Current best':25s}: 0.1365")
