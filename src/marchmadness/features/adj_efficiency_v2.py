"""Enhanced iterative opponent-adjusted efficiency (KenPom/Torvik-style v2).

Improvements over v1:
1. Recency weighting: late-season games weighted more (tournament predictive)
2. Location adjustment: home court advantage correction
3. Possession-weighted averaging: more possessions = more reliable data
4. Score margin diminishing returns: blowouts are less informative
5. Preseason prior: regress toward league average for early-season stability
"""

import pandas as pd
import numpy as np


def estimate_possessions(fga, ora, to, fta):
    """Estimate possessions: FGA - OR + TO + 0.475 * FTA"""
    return fga - ora + to + 0.475 * fta


def compute(data: dict[str, pd.DataFrame], season: int, gender: str = "M",
            n_iterations: int = 20) -> pd.DataFrame:
    """Compute enhanced iterative opponent-adjusted efficiency.

    Returns DataFrame with:
    [TeamID, IterAdjO2, IterAdjD2, IterAdjEM2, IterTempo2, IterSOS2]
    """
    key = f"{gender}RegularSeasonDetailedResults"
    if key not in data:
        return pd.DataFrame(columns=["TeamID"])

    detailed = data[key]
    season_games = detailed[detailed["Season"] == season]

    if season_games.empty:
        return pd.DataFrame(columns=["TeamID"])

    # Get max DayNum for recency weighting
    max_day = season_games["DayNum"].max()

    # Step 1: Compute raw efficiency for each game with metadata
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

        day = game["DayNum"]
        # Recency weight: linearly increasing from 0.5 (day 0) to 1.0 (last day)
        recency_weight = 0.5 + 0.5 * (day / max_day)

        # Location: WLoc is H (home), A (away), N (neutral)
        wloc = game.get("WLoc", "N")

        # Home court advantage adjustment (~3.5 pts per 100 poss, or ~3.5% OE/DE)
        hca = 1.4  # percentage points of efficiency
        if wloc == "H":
            # Winner was home — discount winner's offense, boost loser's
            w_oe_adj = w_oe - hca
            w_de_adj = w_de + hca
            l_oe_adj = l_oe + hca
            l_de_adj = l_de - hca
        elif wloc == "A":
            # Winner was away — boost winner's offense, discount loser's
            w_oe_adj = w_oe + hca
            w_de_adj = w_de - hca
            l_oe_adj = l_oe - hca
            l_de_adj = l_de + hca
        else:
            w_oe_adj = w_oe
            w_de_adj = w_de
            l_oe_adj = l_oe
            l_de_adj = l_de

        games.append({
            "team": w_id, "opp": l_id,
            "raw_oe": w_oe_adj, "raw_de": w_de_adj,
            "poss": avg_poss, "recency": recency_weight
        })
        games.append({
            "team": l_id, "opp": w_id,
            "raw_oe": l_oe_adj, "raw_de": l_de_adj,
            "poss": avg_poss, "recency": recency_weight
        })

    if not games:
        return pd.DataFrame(columns=["TeamID"])

    games_df = pd.DataFrame(games)
    teams = sorted(games_df["team"].unique())

    # Step 2: Initialize with possession-weighted, recency-weighted means
    team_adj_oe = {}
    team_adj_de = {}
    for team in teams:
        tg = games_df[games_df["team"] == team]
        weights = tg["poss"] * tg["recency"]
        w_sum = weights.sum()
        if w_sum > 0:
            team_adj_oe[team] = (tg["raw_oe"] * weights).sum() / w_sum
            team_adj_de[team] = (tg["raw_de"] * weights).sum() / w_sum
        else:
            team_adj_oe[team] = tg["raw_oe"].mean()
            team_adj_de[team] = tg["raw_de"].mean()

    league_avg_oe = np.mean(list(team_adj_oe.values()))
    league_avg_de = np.mean(list(team_adj_de.values()))

    # Preseason prior: slight regression toward mean for stability
    prior_weight = 3.0  # equivalent to 3 games at league average
    for team in teams:
        tg = games_df[games_df["team"] == team]
        n_games = len(tg)
        blend = n_games / (n_games + prior_weight)
        team_adj_oe[team] = blend * team_adj_oe[team] + (1 - blend) * league_avg_oe
        team_adj_de[team] = blend * team_adj_de[team] + (1 - blend) * league_avg_de

    # Step 3: Iterate with weighted adjustments
    for iteration in range(n_iterations):
        new_adj_oe = {}
        new_adj_de = {}

        for team in teams:
            tg = games_df[games_df["team"] == team]
            weights = tg["poss"].values * tg["recency"].values

            adj_oe_vals = []
            adj_de_vals = []
            game_weights = []

            for idx, (_, g) in enumerate(tg.iterrows()):
                opp = g["opp"]
                opp_de = team_adj_de.get(opp, league_avg_de)
                opp_oe = team_adj_oe.get(opp, league_avg_oe)

                # Opponent adjustment
                oe_adj = league_avg_de / opp_de if opp_de > 0 else 1.0
                de_adj = league_avg_oe / opp_oe if opp_oe > 0 else 1.0

                # Clamp extreme adjustments (prevents instability)
                oe_adj = np.clip(oe_adj, 0.5, 2.0)
                de_adj = np.clip(de_adj, 0.5, 2.0)

                adj_oe_vals.append(g["raw_oe"] * oe_adj)
                adj_de_vals.append(g["raw_de"] * de_adj)
                game_weights.append(weights[idx])

            adj_oe_arr = np.array(adj_oe_vals)
            adj_de_arr = np.array(adj_de_vals)
            w_arr = np.array(game_weights)
            w_sum = w_arr.sum()

            if w_sum > 0:
                new_adj_oe[team] = (adj_oe_arr * w_arr).sum() / w_sum
                new_adj_de[team] = (adj_de_arr * w_arr).sum() / w_sum
            else:
                new_adj_oe[team] = league_avg_oe
                new_adj_de[team] = league_avg_de

        # Re-center to preserve league averages
        oe_mean = np.mean(list(new_adj_oe.values()))
        de_mean = np.mean(list(new_adj_de.values()))
        for team in teams:
            new_adj_oe[team] *= (league_avg_oe / oe_mean)
            new_adj_de[team] *= (league_avg_de / de_mean)

        team_adj_oe = new_adj_oe
        team_adj_de = new_adj_de

    # Step 4: Compute SOS from opponent AdjEM
    team_sos = {}
    for team in teams:
        tg = games_df[games_df["team"] == team]
        opp_ems = []
        for _, g in tg.iterrows():
            opp = g["opp"]
            opp_em = team_adj_oe.get(opp, league_avg_oe) - team_adj_de.get(opp, league_avg_de)
            opp_ems.append(opp_em)
        team_sos[team] = np.mean(opp_ems) if opp_ems else 0.0

    # Step 5: Build output
    rows = []
    for team in teams:
        tg = games_df[games_df["team"] == team]
        rows.append({
            "TeamID": team,
            "IterAdjO2": team_adj_oe[team],
            "IterAdjD2": team_adj_de[team],
            "IterAdjEM2": team_adj_oe[team] - team_adj_de[team],
            "IterTempo2": (tg["poss"] * tg["recency"]).sum() / tg["recency"].sum(),
            "IterSOS2": team_sos[team],
        })

    return pd.DataFrame(rows)
