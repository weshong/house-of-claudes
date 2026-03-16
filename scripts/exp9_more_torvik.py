"""Experiment 9A: Test if adding more Torvik features helps.

The existing torvik module uses 5 features: adjoe, adjde, barthag, adjt, sos.
The CSV has ~45 columns. We test adding: WAB, elite SOS, Qual Barthag, ncsos, Qual O, Qual D.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marchmadness.config import CV_SEASONS, PROJECT_ROOT
from marchmadness.data_loader import load_all
from marchmadness.features.builder import (
    build_training_data, get_feature_columns,
    build_team_features, build_matchup_features,
)
from marchmadness.features.torvik import TORVIK_DIR, NAME_OVERRIDES, _build_name_map
from marchmadness.evaluation.cross_val import leave_season_out_cv
from lightgbm import LGBMClassifier


# Extra Torvik columns to try (beyond the 5 already used)
EXTRA_TORVIK_COLS_BY_HEADER = {
    "WAB": "WAB",
    "elite SOS": "elite_SOS",
    "Qual Barthag": "Qual_Barthag",
    "ncsos": "ncsos",
    "Qual O": "Qual_O",
    "Qual D": "Qual_D",
    "Conf Win%": "Conf_WinPct",
    "FUN": "FUN",
}


def load_extra_torvik(data, season, gender="M"):
    """Load extra Torvik columns not in the standard feature set."""
    if gender != "M":
        return pd.DataFrame(columns=["TeamID"])

    torvik_path = TORVIK_DIR / f"torvik_{season}.csv"
    if not torvik_path.exists():
        return pd.DataFrame(columns=["TeamID"])

    try:
        # Read with headers this time to get named columns
        torvik_df = pd.read_csv(torvik_path)
        cols = torvik_df.columns.tolist()

        # The last column may be "Fun Rk, adjt" (quoted) in older files
        # or separate "Fun Rk" and "adjt" in newer files.
        # We only care about the named extra columns.

        # Check which extra columns exist
        available_extras = {}
        for csv_col, feat_name in EXTRA_TORVIK_COLS_BY_HEADER.items():
            if csv_col in cols:
                available_extras[csv_col] = feat_name

        if not available_extras:
            return pd.DataFrame(columns=["TeamID"])

        # Get team name column
        team_col = "team"
        if team_col not in cols:
            # Try positional fallback
            return pd.DataFrame(columns=["TeamID"])

        # Get Kaggle teams
        teams_key = f"{gender}Teams"
        if teams_key not in data:
            return pd.DataFrame(columns=["TeamID"])
        kaggle_teams = data[teams_key][["TeamID", "TeamName"]].copy()

        # Build name mapping
        name_map = _build_name_map(
            kaggle_teams["TeamName"].tolist(),
            torvik_df[team_col].astype(str).tolist()
        )

        rows = []
        for _, team in kaggle_teams.iterrows():
            kname = team["TeamName"]
            if kname not in name_map:
                continue

            tname = name_map[kname]
            torvik_row = torvik_df[torvik_df[team_col] == tname]
            if torvik_row.empty:
                continue

            torvik_row = torvik_row.iloc[0]
            row = {"TeamID": team["TeamID"]}
            for csv_col, feat_name in available_extras.items():
                val = torvik_row[csv_col]
                row[f"TorEx_{feat_name}"] = float(val) if pd.notna(val) else np.nan
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["TeamID"])

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"  Extra Torvik load error for {season}: {e}")
        return pd.DataFrame(columns=["TeamID"])


def build_training_with_extra_torvik(data, seasons, gender="M", feature_set="all"):
    """Build training data using existing features + extra Torvik columns."""
    tourney_key = f"{gender}NCAATourneyCompactResults"
    tourney = data[tourney_key]

    all_rows = []
    for season in seasons:
        # Get standard team features
        team_features = build_team_features(data, season, gender, feature_set)
        if team_features.empty:
            continue

        # Merge extra Torvik features
        extra_df = load_extra_torvik(data, season, gender)
        if not extra_df.empty:
            team_features = team_features.merge(extra_df, on="TeamID", how="left")

        # Build matchups from tournament games
        season_tourney = tourney[tourney["Season"] == season]
        for _, game in season_tourney.iterrows():
            w_id = game["WTeamID"]
            l_id = game["LTeamID"]
            team_a = min(w_id, l_id)
            team_b = max(w_id, l_id)
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


def run_cv(X, y, seasons, label, n_estimators=400, max_depth=3, learning_rate=0.02):
    """Run CV with LGBMClassifier."""
    model = LGBMClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    result = leave_season_out_cv(model, X, y, seasons, CV_SEASONS)
    brier = result["overall"]["brier_score"]
    logloss = result["overall"]["log_loss"]
    print(f"  {label}: Brier={brier:.4f}  LogLoss={logloss:.4f}")
    for s in CV_SEASONS:
        if s in result["per_fold"]:
            fb = result["per_fold"][s]["brier_score"]
            print(f"    {s}: Brier={fb:.4f}")
    return result


def prep_data(df):
    """Prepare X, y, seasons from training DataFrame."""
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].values.copy()
    y = df["Label"].values.copy()
    seasons = df["Season"].values.copy()
    # Impute NaN with column medians
    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0
    return X, y, seasons, feat_cols


def main():
    print("Loading data...")
    data = load_all()

    train_seasons = list(range(2015, 2026))

    # ---- Baseline: standard "all" feature set with LGB d=3 n=400 lr=0.02 ----
    print("\n" + "=" * 60)
    print("  Baseline: standard 'all' features, LGB d=3 n=400 lr=0.02")
    print("=" * 60)
    df_base = build_training_data(data, train_seasons, "M", "all")
    X_base, y_base, s_base, fc_base = prep_data(df_base)
    print(f"  Features ({len(fc_base)}): {fc_base}")
    run_cv(X_base, y_base, s_base, "Baseline all")

    # ---- Test: standard "all" + extra Torvik columns ----
    print("\n" + "=" * 60)
    print("  Test: 'all' features + extra Torvik columns")
    print("=" * 60)
    df_extra = build_training_with_extra_torvik(data, train_seasons, "M", "all")
    X_extra, y_extra, s_extra, fc_extra = prep_data(df_extra)
    new_cols = [c for c in fc_extra if c not in fc_base]
    print(f"  Features ({len(fc_extra)}): added {len(new_cols)} new: {new_cols}")
    run_cv(X_extra, y_extra, s_extra, "All + extra Torvik")

    # ---- Test: "torvik" feature set + extra Torvik (smaller, less noise) ----
    print("\n" + "=" * 60)
    print("  Test: 'torvik' features + extra Torvik columns")
    print("=" * 60)
    df_torvik_extra = build_training_with_extra_torvik(data, train_seasons, "M", "torvik")
    X_te, y_te, s_te, fc_te = prep_data(df_torvik_extra)
    print(f"  Features ({len(fc_te)}): {fc_te}")
    run_cv(X_te, y_te, s_te, "Torvik + extra Torvik")

    # ---- Test: "torvik" baseline (no extras) for comparison ----
    print("\n" + "=" * 60)
    print("  Baseline: 'torvik' features only, LGB d=3 n=400 lr=0.02")
    print("=" * 60)
    df_torvik_base = build_training_data(data, train_seasons, "M", "torvik")
    X_tb, y_tb, s_tb, fc_tb = prep_data(df_torvik_base)
    print(f"  Features ({len(fc_tb)}): {fc_tb}")
    run_cv(X_tb, y_tb, s_tb, "Torvik baseline")

    print("\nDone!")


if __name__ == "__main__":
    main()
