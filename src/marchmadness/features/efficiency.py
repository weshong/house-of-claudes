"""KenPom-style efficiency metrics. Requires detailed results (2003+)."""

import pandas as pd
import numpy as np


def estimate_possessions(fga, ora, to, fta):
    """Estimate possessions from box score stats.

    Possessions ≈ FGA - OR + TO + 0.475 * FTA
    """
    return fga - ora + to + 0.475 * fta


def compute(data: dict[str, pd.DataFrame], season: int, gender: str = "M") -> pd.DataFrame:
    """Compute efficiency metrics for each team.

    Returns DataFrame with columns:
    [TeamID, AdjO, AdjD, AdjEM, Tempo, Possessions]
    """
    key = f"{gender}RegularSeasonDetailedResults"
    if key not in data:
        return pd.DataFrame(columns=["TeamID"])

    detailed = data[key]
    season_games = detailed[detailed["Season"] == season]

    if season_games.empty:
        return pd.DataFrame(columns=["TeamID"])

    team_stats = {}

    for _, game in season_games.iterrows():
        w_id = game["WTeamID"]
        l_id = game["LTeamID"]

        # Winner possessions
        w_poss = estimate_possessions(game["WFGA"], game["WOR"], game["WTO"], game["WFTA"])
        # Loser possessions
        l_poss = estimate_possessions(game["LFGA"], game["LOR"], game["LTO"], game["LFTA"])
        # Average possessions (should be roughly equal)
        avg_poss = (w_poss + l_poss) / 2

        if avg_poss <= 0:
            continue

        # Offensive/Defensive efficiency (points per 100 possessions)
        w_off_eff = (game["WScore"] / avg_poss) * 100
        w_def_eff = (game["LScore"] / avg_poss) * 100
        l_off_eff = (game["LScore"] / avg_poss) * 100
        l_def_eff = (game["WScore"] / avg_poss) * 100

        team_stats.setdefault(w_id, {"off": [], "def": [], "poss": []})
        team_stats[w_id]["off"].append(w_off_eff)
        team_stats[w_id]["def"].append(w_def_eff)
        team_stats[w_id]["poss"].append(avg_poss)

        team_stats.setdefault(l_id, {"off": [], "def": [], "poss": []})
        team_stats[l_id]["off"].append(l_off_eff)
        team_stats[l_id]["def"].append(l_def_eff)
        team_stats[l_id]["poss"].append(avg_poss)

    rows = []
    for team_id, stats in team_stats.items():
        adj_o = np.mean(stats["off"])
        adj_d = np.mean(stats["def"])
        rows.append({
            "TeamID": team_id,
            "AdjO": adj_o,
            "AdjD": adj_d,
            "AdjEM": adj_o - adj_d,
            "Tempo": np.mean(stats["poss"]),
        })

    return pd.DataFrame(rows)
