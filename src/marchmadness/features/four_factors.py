"""Dean Oliver's Four Factors of basketball success. Requires detailed results."""

import pandas as pd
import numpy as np


def compute(data: dict[str, pd.DataFrame], season: int, gender: str = "M") -> pd.DataFrame:
    """Compute Four Factors for each team.

    Four Factors (offense):
    - eFG%: (FGM + 0.5 * FGM3) / FGA
    - TO_rate: TO / Possessions
    - OR%: OR / (OR + opponent DR)
    - FT_rate: FTM / FGA

    Returns DataFrame with columns:
    [TeamID, Off_eFGPct, Off_TORate, Off_ORPct, Off_FTRate,
     Def_eFGPct, Def_TORate, Def_ORPct, Def_FTRate]
    """
    key = f"{gender}RegularSeasonDetailedResults"
    if key not in data:
        return pd.DataFrame(columns=["TeamID"])

    detailed = data[key]
    season_games = detailed[detailed["Season"] == season]

    if season_games.empty:
        return pd.DataFrame(columns=["TeamID"])

    team_stats: dict[int, dict[str, list]] = {}

    for _, g in season_games.iterrows():
        w_id, l_id = g["WTeamID"], g["LTeamID"]

        # Winner offensive four factors
        w_efg = (g["WFGM"] + 0.5 * g["WFGM3"]) / max(g["WFGA"], 1)
        w_poss = g["WFGA"] - g["WOR"] + g["WTO"] + 0.475 * g["WFTA"]
        w_to_rate = g["WTO"] / max(w_poss, 1)
        w_or_pct = g["WOR"] / max(g["WOR"] + g["LDR"], 1)
        w_ft_rate = g["WFTM"] / max(g["WFGA"], 1)

        # Loser offensive four factors
        l_efg = (g["LFGM"] + 0.5 * g["LFGM3"]) / max(g["LFGA"], 1)
        l_poss = g["LFGA"] - g["LOR"] + g["LTO"] + 0.475 * g["LFTA"]
        l_to_rate = g["LTO"] / max(l_poss, 1)
        l_or_pct = g["LOR"] / max(g["LOR"] + g["WDR"], 1)
        l_ft_rate = g["LFTM"] / max(g["LFGA"], 1)

        # Store: winner's offense = their offensive stats, winner's defense = loser's offensive stats
        team_stats.setdefault(w_id, {k: [] for k in [
            "off_efg", "off_to", "off_or", "off_ft",
            "def_efg", "def_to", "def_or", "def_ft"
        ]})
        team_stats[w_id]["off_efg"].append(w_efg)
        team_stats[w_id]["off_to"].append(w_to_rate)
        team_stats[w_id]["off_or"].append(w_or_pct)
        team_stats[w_id]["off_ft"].append(w_ft_rate)
        team_stats[w_id]["def_efg"].append(l_efg)
        team_stats[w_id]["def_to"].append(l_to_rate)
        team_stats[w_id]["def_or"].append(l_or_pct)
        team_stats[w_id]["def_ft"].append(l_ft_rate)

        team_stats.setdefault(l_id, {k: [] for k in [
            "off_efg", "off_to", "off_or", "off_ft",
            "def_efg", "def_to", "def_or", "def_ft"
        ]})
        team_stats[l_id]["off_efg"].append(l_efg)
        team_stats[l_id]["off_to"].append(l_to_rate)
        team_stats[l_id]["off_or"].append(l_or_pct)
        team_stats[l_id]["off_ft"].append(l_ft_rate)
        team_stats[l_id]["def_efg"].append(w_efg)
        team_stats[l_id]["def_to"].append(w_to_rate)
        team_stats[l_id]["def_or"].append(w_or_pct)
        team_stats[l_id]["def_ft"].append(w_ft_rate)

    rows = []
    for team_id, stats in team_stats.items():
        rows.append({
            "TeamID": team_id,
            "Off_eFGPct": np.mean(stats["off_efg"]),
            "Off_TORate": np.mean(stats["off_to"]),
            "Off_ORPct": np.mean(stats["off_or"]),
            "Off_FTRate": np.mean(stats["off_ft"]),
            "Def_eFGPct": np.mean(stats["def_efg"]),
            "Def_TORate": np.mean(stats["def_to"]),
            "Def_ORPct": np.mean(stats["def_or"]),
            "Def_FTRate": np.mean(stats["def_ft"]),
        })

    return pd.DataFrame(rows)
