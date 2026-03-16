"""Season-level aggregate statistics."""

import pandas as pd
import numpy as np


def compute(data: dict[str, pd.DataFrame], season: int, gender: str = "M") -> pd.DataFrame:
    """Compute season stats for each team.

    Returns DataFrame with columns:
    [TeamID, WinPct, PPG, PPGAllowed, PointDiff, GamesPlayed, SOS]
    """
    results_key = f"{gender}RegularSeasonCompactResults"
    results = data[results_key]
    season_games = results[results["Season"] == season]

    if season_games.empty:
        return pd.DataFrame(columns=["TeamID"])

    # Wins
    wins = season_games.groupby("WTeamID").agg(
        Wins=("WScore", "count"),
        PointsFor_W=("WScore", "sum"),
        PointsAgainst_W=("LScore", "sum"),
    ).rename_axis("TeamID")

    # Losses
    losses = season_games.groupby("LTeamID").agg(
        Losses=("LScore", "count"),
        PointsFor_L=("LScore", "sum"),
        PointsAgainst_L=("WScore", "sum"),
    ).rename_axis("TeamID")

    # Combine
    stats = wins.join(losses, how="outer").fillna(0)
    stats["GamesPlayed"] = stats["Wins"] + stats["Losses"]
    stats["WinPct"] = stats["Wins"] / stats["GamesPlayed"]
    stats["PPG"] = (stats["PointsFor_W"] + stats["PointsFor_L"]) / stats["GamesPlayed"]
    stats["PPGAllowed"] = (stats["PointsAgainst_W"] + stats["PointsAgainst_L"]) / stats["GamesPlayed"]
    stats["PointDiff"] = stats["PPG"] - stats["PPGAllowed"]

    # Strength of Schedule (SOS): mean win% of opponents
    opponents = {}
    for _, game in season_games.iterrows():
        opponents.setdefault(game["WTeamID"], []).append(game["LTeamID"])
        opponents.setdefault(game["LTeamID"], []).append(game["WTeamID"])

    win_pct_map = stats["WinPct"].to_dict()
    sos = {}
    for team_id, opp_list in opponents.items():
        opp_win_pcts = [win_pct_map.get(opp, 0.5) for opp in opp_list]
        sos[team_id] = np.mean(opp_win_pcts) if opp_win_pcts else 0.5

    stats["SOS"] = pd.Series(sos)
    stats["SOS"] = stats["SOS"].fillna(0.5)

    result = stats[["WinPct", "PPG", "PPGAllowed", "PointDiff", "GamesPlayed", "SOS"]]
    return result.reset_index()
