"""T-Rank clone: Torvik-style adjusted efficiency ratings (vectorized).

Implements the key components described by Adam Cwik:
1. Per-game offensive/defensive efficiency (PPP)
2. Opponent adjustment: Game_AdjOE = PPPo / (Opp_AdjDE / League_Avg)
3. Home court adjustment (1.4% multiplicative)
4. Recency weighting: 100% within 40 days, 1%/day decay, 60% floor
5. Blowout discount for 10+ point margins
6. Iterative convergence
7. Barthag via pythagorean expectation (exponent 11.5)
8. Log5 for matchup predictions

Works for both men's and women's. Fully vectorized for speed.
"""

import pandas as pd
import numpy as np


def compute(data: dict[str, pd.DataFrame], season: int, gender: str = "M",
            n_iterations: int = 25,
            hca_factor: float = 0.014,
            recency_full_window: int = 40,
            recency_decay_rate: float = 0.01,
            recency_floor: float = 0.60,
            blowout_threshold: int = 10,
            barthag_exponent: float = 11.5,
            preseason_games: int = 5) -> pd.DataFrame:
    """Compute T-Rank-style adjusted efficiency ratings (vectorized).

    Returns DataFrame with:
    [TeamID, TRank_adjoe, TRank_adjde, TRank_adjem, TRank_barthag,
     TRank_tempo, TRank_sos, TRank_luck]
    """
    key = f"{gender}RegularSeasonDetailedResults"
    if key not in data:
        return pd.DataFrame(columns=["TeamID"])

    detailed = data[key]
    sg = detailed[detailed["Season"] == season]

    if sg.empty:
        return pd.DataFrame(columns=["TeamID"])

    max_day = sg["DayNum"].max()

    # Step 1: Vectorized per-game raw efficiency
    w_poss = sg["WFGA"].values - sg["WOR"].values + sg["WTO"].values + 0.475 * sg["WFTA"].values
    l_poss = sg["LFGA"].values - sg["LOR"].values + sg["LTO"].values + 0.475 * sg["LFTA"].values
    avg_poss = (w_poss + l_poss) / 2.0

    valid = avg_poss > 0
    sg = sg[valid].copy()
    avg_poss = avg_poss[valid]
    n_games = len(sg)

    if n_games == 0:
        return pd.DataFrame(columns=["TeamID"])

    w_ppp = sg["WScore"].values / avg_poss
    l_ppp = sg["LScore"].values / avg_poss
    days_from_end = max_day - sg["DayNum"].values
    margin = sg["WScore"].values - sg["LScore"].values
    wloc = sg["WLoc"].values if "WLoc" in sg.columns else np.full(n_games, "N")

    # Recency weights
    recency = np.where(
        days_from_end <= recency_full_window,
        1.0,
        np.maximum(recency_floor, 1.0 - recency_decay_rate * (days_from_end - recency_full_window))
    )

    # HCA multipliers
    w_hca_oe = np.where(wloc == "H", 1 - hca_factor,
               np.where(wloc == "A", 1 + hca_factor, 1.0))
    w_hca_de = np.where(wloc == "H", 1 + hca_factor,
               np.where(wloc == "A", 1 - hca_factor, 1.0))
    l_hca_oe = np.where(wloc == "H", 1 + hca_factor,
               np.where(wloc == "A", 1 - hca_factor, 1.0))
    l_hca_de = np.where(wloc == "H", 1 - hca_factor,
               np.where(wloc == "A", 1 + hca_factor, 1.0))

    # Build game-level arrays (2 rows per game: winner perspective + loser perspective)
    w_ids = sg["WTeamID"].values
    l_ids = sg["LTeamID"].values

    # Stack: [winner rows, loser rows]
    team_arr = np.concatenate([w_ids, l_ids])
    opp_arr = np.concatenate([l_ids, w_ids])
    raw_oe = np.concatenate([w_ppp * w_hca_oe, l_ppp * l_hca_oe])
    raw_de = np.concatenate([l_ppp * w_hca_de, w_ppp * l_hca_de])
    poss_arr = np.concatenate([avg_poss, avg_poss])
    recency_arr = np.concatenate([recency, recency])
    margin_arr = np.concatenate([margin, -margin])
    won_arr = np.concatenate([np.ones(n_games, dtype=bool), np.zeros(n_games, dtype=bool)])

    # Map team IDs to contiguous indices
    unique_teams = np.unique(team_arr)
    n_teams = len(unique_teams)
    team_to_idx = {t: i for i, t in enumerate(unique_teams)}

    team_idx = np.array([team_to_idx[t] for t in team_arr])
    opp_idx = np.array([team_to_idx[t] for t in opp_arr])

    n_rows = len(team_arr)

    # Step 2: Initialize with weighted averages + preseason prior
    # Weighted sum per team
    w_raw_oe = raw_oe * recency_arr
    w_raw_de = raw_de * recency_arr

    adj_oe = np.zeros(n_teams)
    adj_de = np.zeros(n_teams)
    w_sum_per_team = np.zeros(n_teams)
    game_count = np.zeros(n_teams)

    np.add.at(adj_oe, team_idx, w_raw_oe)
    np.add.at(adj_de, team_idx, w_raw_de)
    np.add.at(w_sum_per_team, team_idx, recency_arr)
    np.add.at(game_count, team_idx, 1)

    nonzero = w_sum_per_team > 0
    adj_oe[nonzero] /= w_sum_per_team[nonzero]
    adj_de[nonzero] /= w_sum_per_team[nonzero]
    # fallback for zero-weight teams (shouldn't happen)
    adj_oe[~nonzero] = raw_oe.mean()
    adj_de[~nonzero] = raw_de.mean()

    league_avg = adj_oe.mean()

    # Preseason regression
    blend = game_count / (game_count + preseason_games)
    adj_oe = blend * adj_oe + (1 - blend) * league_avg
    adj_de = blend * adj_de + (1 - blend) * league_avg

    # Step 3: Iterative convergence (fully vectorized)
    abs_margin = np.abs(margin_arr)

    for iteration in range(n_iterations):
        # Look up opponent ratings for all games at once
        opp_de_vals = adj_de[opp_idx]
        opp_oe_vals = adj_oe[opp_idx]

        # Opponent adjustment
        safe_opp_de = np.where(opp_de_vals > 0, opp_de_vals, 1.0)
        safe_opp_oe = np.where(opp_oe_vals > 0, opp_oe_vals, 1.0)
        game_adj_oe = raw_oe / (safe_opp_de / league_avg)
        game_adj_de = raw_de / (safe_opp_oe / league_avg)

        # Blowout discount (vectorized)
        team_oe_vals = adj_oe[team_idx]
        opp_oe_for_barthag = adj_oe[opp_idx]
        barthag_diff = np.abs(team_oe_vals - opp_oe_for_barthag) / league_avg

        discount = np.ones(n_rows)
        # Case 1: margin >= 20 and barthag_diff > 0.5
        mask1 = (abs_margin >= 20) & (barthag_diff > 0.5) & (abs_margin > blowout_threshold)
        discount[mask1] = np.maximum(0.3, 1 - (barthag_diff[mask1] - 0.5) * 2)
        # Case 2: margin >= 20 and barthag_diff <= 0.5
        mask2 = (abs_margin >= 20) & (barthag_diff <= 0.5) & (abs_margin > blowout_threshold) & ~mask1
        discount[mask2] = 0.85
        # Case 3: blowout_threshold < margin < 20
        mask3 = (abs_margin > blowout_threshold) & (abs_margin < 20) & ~mask1 & ~mask2
        discount[mask3] = np.maximum(0.7, 1.0 - (abs_margin[mask3] - blowout_threshold) * 0.005)

        game_weight = recency_arr * discount

        # Weighted aggregation per team using np.add.at
        new_adj_oe = np.zeros(n_teams)
        new_adj_de = np.zeros(n_teams)
        w_sum = np.zeros(n_teams)

        weighted_oe = game_adj_oe * game_weight
        weighted_de = game_adj_de * game_weight

        np.add.at(new_adj_oe, team_idx, weighted_oe)
        np.add.at(new_adj_de, team_idx, weighted_de)
        np.add.at(w_sum, team_idx, game_weight)

        nonzero = w_sum > 0
        new_adj_oe[nonzero] /= w_sum[nonzero]
        new_adj_de[nonzero] /= w_sum[nonzero]
        new_adj_oe[~nonzero] = league_avg
        new_adj_de[~nonzero] = league_avg

        # Re-center
        oe_mean = new_adj_oe.mean()
        de_mean = new_adj_de.mean()
        new_adj_oe *= (league_avg / oe_mean)
        new_adj_de *= (league_avg / de_mean)

        adj_oe = new_adj_oe
        adj_de = new_adj_de

    # Step 4: Compute derived metrics (vectorized)
    em = adj_oe - adj_de

    # Barthag
    oe_exp = adj_oe ** barthag_exponent
    de_exp = adj_de ** barthag_exponent
    denom = oe_exp + de_exp
    barthag = np.where(denom > 0, oe_exp / denom, 0.5)

    # Tempo: mean possessions per team
    tempo = np.zeros(n_teams)
    poss_sum = np.zeros(n_teams)
    np.add.at(poss_sum, team_idx, poss_arr)
    tempo = poss_sum / np.maximum(game_count, 1)

    # SOS: average opponent AdjEM
    opp_em = adj_oe[opp_idx] - adj_de[opp_idx]
    sos_sum = np.zeros(n_teams)
    np.add.at(sos_sum, team_idx, opp_em)
    sos = sos_sum / np.maximum(game_count, 1)

    # Luck: actual win% - barthag
    wins = np.zeros(n_teams)
    np.add.at(wins, team_idx, won_arr.astype(float))
    actual_wp = wins / np.maximum(game_count, 1)
    luck = actual_wp - barthag

    return pd.DataFrame({
        "TeamID": unique_teams,
        "TRank_adjoe": adj_oe * 100,
        "TRank_adjde": adj_de * 100,
        "TRank_adjem": em * 100,
        "TRank_barthag": barthag,
        "TRank_tempo": tempo,
        "TRank_sos": sos * 100,
        "TRank_luck": luck,
    })
