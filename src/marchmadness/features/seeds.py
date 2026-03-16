"""Seed parsing and seed-based features."""

import pandas as pd
import numpy as np
from marchmadness.data_loader import load


def parse_seed(seed_str: str) -> int:
    """Parse seed string like 'W01', 'X16a' to integer seed number."""
    # Strip region letter (first char) and any play-in suffix (a/b)
    num_str = seed_str[1:].rstrip("ab")
    return int(num_str)


def compute(data: dict[str, pd.DataFrame], season: int, gender: str = "M") -> pd.DataFrame:
    """Compute seed features for each team in the given season.

    Returns DataFrame with columns: [TeamID, Seed]
    """
    seeds_key = f"{gender}NCAATourneySeeds"
    seeds_df = data[seeds_key]
    season_seeds = seeds_df[seeds_df["Season"] == season].copy()

    if season_seeds.empty:
        return pd.DataFrame(columns=["TeamID", "Seed"])

    season_seeds["SeedNum"] = season_seeds["Seed"].apply(parse_seed)
    result = season_seeds[["TeamID", "SeedNum"]].rename(columns={"SeedNum": "Seed"})
    return result.reset_index(drop=True)


def compute_matchup_features(team_a_seed: int, team_b_seed: int) -> dict:
    """Compute matchup-level seed features."""
    return {
        "seed_a": team_a_seed,
        "seed_b": team_b_seed,
        "seed_diff": team_a_seed - team_b_seed,
    }


# Historical seed matchup win rates (approximate, for priors)
HISTORICAL_SEED_WIN_RATES = {
    (1, 16): 0.985, (2, 15): 0.94, (3, 14): 0.85, (4, 13): 0.79,
    (5, 12): 0.65, (6, 11): 0.63, (7, 10): 0.61, (8, 9): 0.51,
    (1, 8): 0.80, (1, 9): 0.85, (2, 7): 0.65, (2, 10): 0.72,
    (3, 6): 0.58, (3, 11): 0.70, (4, 5): 0.55, (4, 12): 0.79,
}
