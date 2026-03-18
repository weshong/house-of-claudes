"""Massey ordinal ranking features."""

import pandas as pd
import numpy as np
from marchmadness.config import ORDINALS_MIN_DAY, TOP_RANKING_SYSTEMS


def compute(data: dict[str, pd.DataFrame], season: int, gender: str = "M") -> pd.DataFrame:
    """Compute ordinal-based features for each team in the given season.

    Uses late-season rankings (DayNum >= 128) from top ranking systems.
    Returns DataFrame with columns: [TeamID, OrdinalMean, OrdinalStd, Ordinal_{system}, ...]
    """
    key = f"{gender}MasseyOrdinals"
    if key not in data:
        return pd.DataFrame(columns=["TeamID"])

    ordinals = data[key]
    season_ord = ordinals[
        (ordinals["Season"] == season) &
        (ordinals["RankingDayNum"] >= ORDINALS_MIN_DAY)
    ]

    if season_ord.empty:
        return pd.DataFrame(columns=["TeamID"])

    # Find which top systems are available this season
    available_systems = [s for s in TOP_RANKING_SYSTEMS if s in season_ord["SystemName"].values]

    if not available_systems:
        # Fall back to whatever systems are available
        available_systems = season_ord["SystemName"].unique().tolist()[:10]

    # Filter out systems with very low coverage (e.g. polls that only rank top 25)
    min_teams = 50
    available_systems = [
        s for s in available_systems
        if len(season_ord[season_ord["SystemName"] == s]["TeamID"].unique()) >= min_teams
    ]

    if not available_systems:
        return pd.DataFrame(columns=["TeamID"])

    # For each team, get mean rank from each system (using latest available day)
    results = []
    for system in available_systems:
        sys_data = season_ord[season_ord["SystemName"] == system]
        # Use the latest ranking day for each team
        latest = sys_data.loc[sys_data.groupby("TeamID")["RankingDayNum"].idxmax()]
        latest = latest[["TeamID", "OrdinalRank"]].rename(
            columns={"OrdinalRank": f"Ordinal_{system}"}
        )
        results.append(latest)

    if not results:
        return pd.DataFrame(columns=["TeamID"])

    # Merge all systems
    merged = results[0]
    for r in results[1:]:
        merged = merged.merge(r, on="TeamID", how="outer")

    # Compute aggregate features
    ordinal_cols = [c for c in merged.columns if c.startswith("Ordinal_")]
    merged["OrdinalMean"] = merged[ordinal_cols].mean(axis=1)
    merged["OrdinalMedian"] = merged[ordinal_cols].median(axis=1)
    merged["OrdinalStd"] = merged[ordinal_cols].std(axis=1).fillna(0)
    merged["OrdinalMin"] = merged[ordinal_cols].min(axis=1)  # best ranking
    merged["OrdinalMax"] = merged[ordinal_cols].max(axis=1)  # worst ranking

    return merged.reset_index(drop=True)
