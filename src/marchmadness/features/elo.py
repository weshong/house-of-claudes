"""Elo rating system with margin-of-victory adjustment."""

import pandas as pd
import numpy as np
from marchmadness.config import ELO_K, ELO_HOME_ADV, ELO_INITIAL, ELO_SEASON_REGRESSION


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A given ratings."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def mov_multiplier(margin: int) -> float:
    """Margin of victory multiplier. Diminishing returns for blowouts."""
    return np.log(abs(margin) + 1) * (2.2 / (2.2 + 0.001 * abs(margin)))


def compute(data: dict[str, pd.DataFrame], season: int, gender: str = "M") -> pd.DataFrame:
    """Compute end-of-regular-season Elo ratings for each team.

    Processes all games from earliest available season up to the target season's
    regular season (no tournament games for the target season).

    Returns DataFrame with columns: [TeamID, Elo]
    """
    results_key = f"{gender}RegularSeasonCompactResults"
    tourney_key = f"{gender}NCAATourneyCompactResults"

    regular = data[results_key]
    tourney = data[tourney_key]

    # All seasons up to and including target
    all_seasons = sorted(regular["Season"].unique())
    all_seasons = [s for s in all_seasons if s <= season]

    ratings: dict[int, float] = {}

    for s in all_seasons:
        # Regress ratings toward mean at start of each season
        for team_id in ratings:
            ratings[team_id] = (
                ELO_SEASON_REGRESSION * ratings[team_id]
                + (1 - ELO_SEASON_REGRESSION) * ELO_INITIAL
            )

        # Process regular season games in order
        season_games = regular[regular["Season"] == s].sort_values("DayNum")
        for _, game in season_games.iterrows():
            w_id = game["WTeamID"]
            l_id = game["LTeamID"]
            w_loc = game.get("WLoc", "N")

            # Initialize if new team
            ratings.setdefault(w_id, ELO_INITIAL)
            ratings.setdefault(l_id, ELO_INITIAL)

            # Home advantage adjustment
            w_rating = ratings[w_id]
            l_rating = ratings[l_id]
            if w_loc == "H":
                w_rating += ELO_HOME_ADV
            elif w_loc == "A":
                l_rating += ELO_HOME_ADV

            # Update
            w_expected = expected_score(w_rating, l_rating)
            margin = game["WScore"] - game["LScore"]
            mult = mov_multiplier(margin)
            k = ELO_K * mult

            ratings[w_id] += k * (1 - w_expected)
            ratings[l_id] -= k * (1 - w_expected)

        # Also process tournament games for past seasons (not target season)
        if s < season:
            tourney_games = tourney[tourney["Season"] == s].sort_values("DayNum")
            for _, game in tourney_games.iterrows():
                w_id = game["WTeamID"]
                l_id = game["LTeamID"]

                ratings.setdefault(w_id, ELO_INITIAL)
                ratings.setdefault(l_id, ELO_INITIAL)

                w_expected = expected_score(ratings[w_id], ratings[l_id])
                margin = game["WScore"] - game["LScore"]
                mult = mov_multiplier(margin)
                k = ELO_K * mult

                ratings[w_id] += k * (1 - w_expected)
                ratings[l_id] -= k * (1 - w_expected)

    # Return ratings for teams that played in the target season
    season_teams = set(
        regular[regular["Season"] == season]["WTeamID"].tolist()
        + regular[regular["Season"] == season]["LTeamID"].tolist()
    )

    result = pd.DataFrame([
        {"TeamID": tid, "Elo": ratings.get(tid, ELO_INITIAL)}
        for tid in season_teams
    ])
    return result
