"""Orchestrate all feature modules into matchup-level DataFrames."""

import pandas as pd
import numpy as np
from itertools import combinations

from marchmadness.data_loader import load_all
from marchmadness.features import seeds, ordinals, elo, season_stats, efficiency, four_factors, torvik


def build_team_features(data: dict[str, pd.DataFrame], season: int, gender: str = "M",
                        feature_set: str = "all") -> pd.DataFrame:
    """Build team-level features for a given season.

    Args:
        data: Dict of loaded DataFrames
        season: Target season
        gender: "M" or "W"
        feature_set: "seeds_only", "tier1", "tier2", or "all"

    Returns:
        DataFrame with TeamID and all feature columns
    """
    # Always start with seeds
    team_df = seeds.compute(data, season, gender)

    if feature_set == "seeds_only":
        return team_df

    if team_df.empty:
        return team_df

    # Tier 1: Massey ordinals
    ord_df = ordinals.compute(data, season, gender)
    if not ord_df.empty:
        team_df = team_df.merge(ord_df, on="TeamID", how="left")

    if feature_set == "tier1":
        return team_df

    # Tier 1.5: Seeds + ordinals + Torvik external ratings
    if feature_set == "torvik":
        torvik_df = torvik.compute(data, season, gender)
        if not torvik_df.empty:
            team_df = team_df.merge(torvik_df, on="TeamID", how="left")
        return team_df

    # Tier 2: Elo + season stats
    elo_df = elo.compute(data, season, gender)
    if not elo_df.empty:
        team_df = team_df.merge(elo_df, on="TeamID", how="left")

    stats_df = season_stats.compute(data, season, gender)
    if not stats_df.empty:
        team_df = team_df.merge(stats_df, on="TeamID", how="left")

    if feature_set == "tier2":
        return team_df

    # Tier 3: Efficiency + four factors (needs detailed results) + Torvik
    eff_df = efficiency.compute(data, season, gender)
    if not eff_df.empty:
        team_df = team_df.merge(eff_df, on="TeamID", how="left")

    ff_df = four_factors.compute(data, season, gender)
    if not ff_df.empty:
        team_df = team_df.merge(ff_df, on="TeamID", how="left")

    # External: Bart Torvik T-Rank (men's only, 2015+)
    torvik_df = torvik.compute(data, season, gender)
    if not torvik_df.empty:
        team_df = team_df.merge(torvik_df, on="TeamID", how="left")

    return team_df


def build_matchup_features(team_features: pd.DataFrame, team_a_id: int, team_b_id: int) -> dict:
    """Build features for a single matchup (team_a_id < team_b_id).

    Returns dict of feature name -> value.
    """
    a = team_features[team_features["TeamID"] == team_a_id]
    b = team_features[team_features["TeamID"] == team_b_id]

    if a.empty or b.empty:
        return {}

    a = a.iloc[0]
    b = b.iloc[0]

    features = {}
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

        # Keep individual seed values (useful for tree models)
        if col == "Seed":
            features["seed_a"] = val_a if not pd.isna(val_a) else np.nan
            features["seed_b"] = val_b if not pd.isna(val_b) else np.nan

    # Interaction features (only if both seed and ordinal features exist)
    if "Seed_diff" in features and "OrdinalMean_diff" in features:
        seed_d = features["Seed_diff"]
        ord_d = features["OrdinalMean_diff"]
        if not (pd.isna(seed_d) or pd.isna(ord_d)):
            # Seed-ranking agreement: do seeds and rankings agree on who's better?
            features["seed_ordinal_agreement"] = seed_d * ord_d
            # Seed-ranking mismatch: larger = seeds and rankings disagree more
            # Normalize seed_diff to ~same scale as ordinal_diff
            features["seed_ordinal_mismatch"] = abs(seed_d * 20 - ord_d)

    if "Elo_diff" in features and "OrdinalMean_diff" in features:
        elo_d = features.get("Elo_diff", np.nan)
        ord_d = features["OrdinalMean_diff"]
        if not (pd.isna(elo_d) or pd.isna(ord_d)):
            features["elo_ordinal_agreement"] = np.sign(elo_d) * np.sign(ord_d)

    return features


def build_training_data(data: dict[str, pd.DataFrame], seasons: list[int],
                        gender: str = "M", feature_set: str = "all",
                        include_regular_season: bool = False) -> pd.DataFrame:
    """Build matchup features + labels for tournament games across multiple seasons.

    Args:
        include_regular_season: If True, include regular season games (weighted 1x)
            alongside tournament games (weighted 6x). This increases training data.

    Returns DataFrame with feature columns + 'Label' (1 if lower TeamID won, 0 otherwise)
    + 'Season' + 'SampleWeight' columns.
    """
    tourney_key = f"{gender}NCAATourneyCompactResults"
    tourney = data[tourney_key]

    all_rows = []
    for season in seasons:
        team_features = build_team_features(data, season, gender, feature_set)
        if team_features.empty:
            continue

        # Tournament games (weight = 6.0)
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

        # Optional: regular season games between tournament teams (weight = 1.0)
        if include_regular_season:
            rs_key = f"{gender}RegularSeasonCompactResults"
            rs = data[rs_key]
            season_rs = rs[rs["Season"] == season]
            # Only include games between teams that have seeds (tournament teams)
            tourney_teams = set(
                season_tourney["WTeamID"].tolist() + season_tourney["LTeamID"].tolist()
            )
            if not tourney_teams:
                seeds_key = f"{gender}NCAATourneySeeds"
                season_seeds = data[seeds_key]
                tourney_teams = set(
                    season_seeds[season_seeds["Season"] == season]["TeamID"].tolist()
                )

            for _, game in season_rs.iterrows():
                w_id = game["WTeamID"]
                l_id = game["LTeamID"]
                if w_id not in tourney_teams or l_id not in tourney_teams:
                    continue

                team_a = min(w_id, l_id)
                team_b = max(w_id, l_id)
                label = 1 if w_id == team_a else 0

                matchup = build_matchup_features(team_features, team_a, team_b)
                if not matchup:
                    continue

                matchup["Season"] = season
                matchup["Label"] = label
                matchup["DayNum"] = game["DayNum"]
                matchup["SampleWeight"] = 1.0
                all_rows.append(matchup)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def build_prediction_matchups(data: dict[str, pd.DataFrame], season: int,
                              gender: str = "M", feature_set: str = "all") -> pd.DataFrame:
    """Build features for ALL possible matchups in a season (for submission).

    Returns DataFrame with feature columns + 'TeamA', 'TeamB' columns.
    """
    seeds_key = f"{gender}NCAATourneySeeds"
    season_seeds = data[seeds_key]
    season_seeds = season_seeds[season_seeds["Season"] == season]

    if season_seeds.empty:
        return pd.DataFrame()

    team_ids = sorted(season_seeds["TeamID"].unique())
    team_features = build_team_features(data, season, gender, feature_set)

    all_rows = []
    for team_a, team_b in combinations(team_ids, 2):
        matchup = build_matchup_features(team_features, team_a, team_b)
        if not matchup:
            continue
        matchup["TeamA"] = team_a
        matchup["TeamB"] = team_b
        all_rows.append(matchup)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature column names (excluding metadata and label columns)."""
    exclude = {"Season", "Label", "DayNum", "TeamA", "TeamB", "SampleWeight"}
    return [c for c in df.columns if c not in exclude]
