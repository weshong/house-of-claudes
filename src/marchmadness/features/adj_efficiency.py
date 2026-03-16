"""Iterative opponent-adjusted efficiency metrics (KenPom/Torvik-style).

Instead of just averaging raw efficiency, we iteratively adjust each team's
offensive/defensive efficiency based on the quality of opponents they faced.
This converges to a rating where strong teams playing weak opponents get
properly credited, and vice versa.

This works for both men's and women's (unlike external Torvik data).
"""

import pandas as pd
import numpy as np


def estimate_possessions(fga, ora, to, fta):
    """Estimate possessions: FGA - OR + TO + 0.475 * FTA"""
    return fga - ora + to + 0.475 * fta


def compute(data: dict[str, pd.DataFrame], season: int, gender: str = "M",
            n_iterations: int = 15) -> pd.DataFrame:
    """Compute iterative opponent-adjusted efficiency for each team.

    The algorithm:
    1. Compute raw offensive/defensive efficiency for each game
    2. Initialize each team's AdjOE/AdjDE as their raw mean
    3. Iteratively re-weight: AdjOE_i = mean(raw_OE * (opp_AdjDE / league_avg_DE))
       This means scoring 100 pts/100poss against a team with AdjDE of 110
       (bad defense) is worth less than against AdjDE of 90 (good defense).
    4. Converge after ~10-15 iterations.

    Returns DataFrame with:
    [TeamID, IterAdjO, IterAdjD, IterAdjEM, IterTempo]
    """
    key = f"{gender}RegularSeasonDetailedResults"
    if key not in data:
        return pd.DataFrame(columns=["TeamID"])

    detailed = data[key]
    season_games = detailed[detailed["Season"] == season]

    if season_games.empty:
        return pd.DataFrame(columns=["TeamID"])

    # Step 1: Compute raw efficiency for each game
    games = []
    for _, game in season_games.iterrows():
        w_id = game["WTeamID"]
        l_id = game["LTeamID"]

        w_poss = estimate_possessions(game["WFGA"], game["WOR"], game["WTO"], game["WFTA"])
        l_poss = estimate_possessions(game["LFGA"], game["LOR"], game["LTO"], game["LFTA"])
        avg_poss = (w_poss + l_poss) / 2

        if avg_poss <= 0:
            continue

        w_oe = (game["WScore"] / avg_poss) * 100
        w_de = (game["LScore"] / avg_poss) * 100
        l_oe = (game["LScore"] / avg_poss) * 100
        l_de = (game["WScore"] / avg_poss) * 100

        games.append({
            "team": w_id, "opp": l_id,
            "raw_oe": w_oe, "raw_de": w_de, "poss": avg_poss
        })
        games.append({
            "team": l_id, "opp": w_id,
            "raw_oe": l_oe, "raw_de": l_de, "poss": avg_poss
        })

    if not games:
        return pd.DataFrame(columns=["TeamID"])

    games_df = pd.DataFrame(games)
    teams = sorted(games_df["team"].unique())

    # Step 2: Initialize with raw means
    team_adj_oe = {}
    team_adj_de = {}
    for team in teams:
        team_games = games_df[games_df["team"] == team]
        team_adj_oe[team] = team_games["raw_oe"].mean()
        team_adj_de[team] = team_games["raw_de"].mean()

    league_avg_oe = np.mean(list(team_adj_oe.values()))
    league_avg_de = np.mean(list(team_adj_de.values()))

    # Step 3: Iterate
    for iteration in range(n_iterations):
        new_adj_oe = {}
        new_adj_de = {}

        for team in teams:
            team_games = games_df[games_df["team"] == team]

            # Adjust offensive efficiency based on opponent defensive quality
            adj_oe_vals = []
            adj_de_vals = []
            for _, g in team_games.iterrows():
                opp = g["opp"]
                opp_de = team_adj_de.get(opp, league_avg_de)
                opp_oe = team_adj_oe.get(opp, league_avg_oe)

                # If opponent has below-average defense (high AdjDE),
                # discount the offensive efficiency
                oe_adjustment = league_avg_de / opp_de if opp_de > 0 else 1.0
                de_adjustment = league_avg_oe / opp_oe if opp_oe > 0 else 1.0

                adj_oe_vals.append(g["raw_oe"] * oe_adjustment)
                adj_de_vals.append(g["raw_de"] * de_adjustment)

            new_adj_oe[team] = np.mean(adj_oe_vals) if adj_oe_vals else league_avg_oe
            new_adj_de[team] = np.mean(adj_de_vals) if adj_de_vals else league_avg_de

        # Re-center to preserve league average
        oe_mean = np.mean(list(new_adj_oe.values()))
        de_mean = np.mean(list(new_adj_de.values()))
        for team in teams:
            new_adj_oe[team] = new_adj_oe[team] * (league_avg_oe / oe_mean)
            new_adj_de[team] = new_adj_de[team] * (league_avg_de / de_mean)

        team_adj_oe = new_adj_oe
        team_adj_de = new_adj_de

    # Step 4: Build output
    rows = []
    for team in teams:
        team_games = games_df[games_df["team"] == team]
        rows.append({
            "TeamID": team,
            "IterAdjO": team_adj_oe[team],
            "IterAdjD": team_adj_de[team],
            "IterAdjEM": team_adj_oe[team] - team_adj_de[team],
            "IterTempo": team_games["poss"].mean(),
        })

    return pd.DataFrame(rows)
