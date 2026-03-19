"""Compute the theoretical Brier score floor and explore gaming strategies.

For a perfectly calibrated model, E[Brier for game i] = p_i * (1 - p_i)
where p_i is the TRUE win probability. This is the irreducible uncertainty.

Key questions:
1. What's the Brier floor given tournament structure?
2. Can we game the metric by deviating from true probabilities?
3. How does bracket structure affect optimal strategy?
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from marchmadness.data_loader import load_all
from marchmadness.features.seeds import parse_seed


def analyze_brier_floor():
    data = load_all()

    print("=" * 70)
    print("THEORETICAL BRIER SCORE FLOOR ANALYSIS")
    print("=" * 70)

    for gender, label in [("M", "Men's"), ("W", "Women's")]:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

        tourney = data[f"{gender}NCAATourneyCompactResults"]
        seeds_df = data[f"{gender}NCAATourneySeeds"]

        # Build historical win rates by seed matchup
        # Use ALL available history, not just CV seasons
        matchup_wins = {}  # (seed_low, seed_high) -> [outcomes]
        game_details = []  # for per-round analysis

        for season in tourney["Season"].unique():
            season_seeds = seeds_df[seeds_df["Season"] == season]
            seed_map = {}
            for _, row in season_seeds.iterrows():
                seed_map[row["TeamID"]] = parse_seed(row["Seed"])

            season_games = tourney[tourney["Season"] == season]
            for _, game in season_games.iterrows():
                w_seed = seed_map.get(game["WTeamID"])
                l_seed = seed_map.get(game["LTeamID"])
                if w_seed is None or l_seed is None:
                    continue

                s_low, s_high = min(w_seed, l_seed), max(w_seed, l_seed)
                fav_won = (w_seed == s_low)  # lower seed (favorite) won

                key = (s_low, s_high)
                matchup_wins.setdefault(key, []).append(1 if fav_won else 0)

                game_details.append({
                    "season": season,
                    "day": game["DayNum"],
                    "seed_low": s_low,
                    "seed_high": s_high,
                    "fav_won": fav_won,
                    "seed_diff": s_high - s_low,
                })

        df = pd.DataFrame(game_details)

        # ============================================================
        # 1. Historical base rates by seed matchup
        # ============================================================
        print(f"\n--- Historical base rates (all seasons) ---")
        print(f"{'Matchup':>8} {'N':>5} {'Fav Win%':>9} {'p(1-p)':>8}  <- Brier floor per game")
        print("-" * 45)

        first_round = [(1,16),(2,15),(3,14),(4,13),(5,12),(6,11),(7,10),(8,9)]
        for s_low, s_high in first_round:
            outcomes = matchup_wins.get((s_low, s_high), [])
            if not outcomes:
                continue
            p = np.mean(outcomes)
            floor = p * (1 - p)
            print(f"  {s_low}v{s_high}:   {len(outcomes):4d}   {p:7.1%}    {floor:.4f}")

        # ============================================================
        # 2. Per-round analysis
        # ============================================================
        print(f"\n--- Brier floor by tournament round ---")

        rounds = [
            ("R64", 134, 135),
            ("R32", 136, 137),
            ("S16", 138, 139),
            ("E8", 143, 144),
            ("F4", 152, 152),
            ("Champ", 154, 154),
        ]

        round_floors = []
        for rname, d_min, d_max in rounds:
            mask = (df["day"] >= d_min) & (df["day"] <= d_max)
            rdf = df[mask]
            if rdf.empty:
                continue

            # Group by seed matchup within this round
            matchups = rdf.groupby(["seed_low", "seed_high"])["fav_won"].agg(["mean", "count"])
            # Weighted average Brier floor
            total_games = matchups["count"].sum()
            weighted_floor = sum(
                row["mean"] * (1 - row["mean"]) * row["count"]
                for _, row in matchups.iterrows()
            ) / total_games

            # Average seed diff (proxy for predictability)
            avg_diff = rdf["seed_diff"].mean()
            avg_p = rdf["fav_won"].mean()

            round_floors.append({
                "round": rname,
                "n_games_per_tourney": total_games / len(rdf["season"].unique()),
                "avg_seed_diff": avg_diff,
                "avg_fav_win": avg_p,
                "brier_floor": weighted_floor,
            })

            print(f"  {rname:>6}: {total_games/len(rdf['season'].unique()):4.0f} games/yr, "
                  f"avg seed diff={avg_diff:.1f}, fav wins {avg_p:.1%}, "
                  f"Brier floor={weighted_floor:.4f}")

        # ============================================================
        # 3. Overall Brier floor for a full tournament
        # ============================================================
        print(f"\n--- Overall Brier floor (weighted by games per round) ---")

        # Use CV seasons (2022-2025) for apples-to-apples comparison
        cv_seasons = [2022, 2023, 2024, 2025]
        cv_df = df[df["season"].isin(cv_seasons)]

        # Compute per-game Brier floor using historical base rates
        per_game_floors = []
        for _, game in cv_df.iterrows():
            key = (game["seed_low"], game["seed_high"])
            outcomes = matchup_wins.get(key, [])
            if outcomes:
                p = np.mean(outcomes)
            else:
                # Fall back to seed-diff-based estimate
                p = 0.5 + 0.03 * game["seed_diff"]
                p = min(p, 0.99)
            per_game_floors.append(p * (1 - p))

        overall_floor = np.mean(per_game_floors)
        n_games = len(cv_df) / len(cv_seasons)
        print(f"  Games per tournament: {n_games:.0f}")
        print(f"  Brier floor (perfect calibration): {overall_floor:.4f}")
        print(f"  Our v7 Brier:                      {0.1266 if gender == 'M' else 0.1335:.4f}")
        print(f"  Gap to floor:                      {(0.1266 if gender == 'M' else 0.1335) - overall_floor:.4f}")
        print(f"  % of reducible error remaining:    {((0.1266 if gender == 'M' else 0.1335) - overall_floor) / (0.25 - overall_floor):.1%}")

        # ============================================================
        # 4. Brier floor decomposition: where do we lose points?
        # ============================================================
        print(f"\n--- Where Brier points are lost (floor decomposition) ---")
        print(f"{'Round':>6} {'Games':>5} {'Floor':>7} {'Contribution':>13}  {'% of total':>10}")
        print("-" * 50)

        total_floor_pts = 0
        for rf in round_floors:
            contrib = rf["brier_floor"] * rf["n_games_per_tourney"] / n_games
            total_floor_pts += contrib

        for rf in round_floors:
            contrib = rf["brier_floor"] * rf["n_games_per_tourney"] / n_games
            pct = contrib / total_floor_pts if total_floor_pts > 0 else 0
            print(f"  {rf['round']:>6} {rf['n_games_per_tourney']:5.0f}   {rf['brier_floor']:.4f}   "
                  f"{contrib:.4f}         {pct:.1%}")
        print(f"  {'Total':>6} {n_games:5.0f}   {overall_floor:.4f}")

        # ============================================================
        # 5. Can we game the Brier score?
        # ============================================================
        print(f"\n--- Gaming analysis ---")
        print()
        print("  Brier is a PROPER scoring rule: optimal prediction = true probability p.")
        print("  Deviating from p always increases expected Brier. HOWEVER:")
        print()

        # Scenario 1: What if we knew exact base rates but not team-level?
        # Using seed matchup base rates as predictions
        seed_only_brier_parts = []
        for _, game in cv_df.iterrows():
            key = (game["seed_low"], game["seed_high"])
            outcomes = matchup_wins.get(key, [])
            if outcomes:
                p = np.mean(outcomes)
            else:
                p = 0.5
            # Actual outcome
            actual = 1 if game["fav_won"] else 0
            seed_only_brier_parts.append((p - actual) ** 2)

        seed_only_brier = np.mean(seed_only_brier_parts)
        print(f"  Scenario: predict historical base rate for each seed matchup")
        print(f"  (no team-level info, just 1v16 -> 0.99, 5v12 -> 0.65, etc.)")
        print(f"  Expected Brier: {seed_only_brier:.4f}")
        print()

        # Scenario 2: Effect of prediction clipping
        print(f"  --- Clipping analysis ---")
        print(f"  If true p=0.99 (1v16), clipping to 0.975 costs:")
        p_true = 0.99
        p_clip = 0.975
        cost_clip = p_true * (p_clip - 1)**2 + (1-p_true) * p_clip**2 - p_true*(p_true-1)**2 - (1-p_true)*p_true**2
        print(f"    E[Brier(0.975)] - E[Brier(0.99)] = {cost_clip:.6f} per game")
        print(f"    But if true p=0.95 and we predict 0.975:")
        p_true = 0.95
        cost_over = p_true * (p_clip - 1)**2 + (1-p_true) * p_clip**2 - p_true*(p_true-1)**2 - (1-p_true)*p_true**2
        print(f"    E[Brier(0.975)] - E[Brier(0.95)] = {cost_over:.6f} per game (overconfidence hurts more)")
        print()

        # Scenario 3: Asymmetric payoffs from confidence
        print(f"  --- Confidence asymmetry on 5v12 ---")
        # If true p = 0.65 (historical 5v12 rate)
        p_true = 0.65
        print(f"  If true P(5-seed wins) = {p_true:.2f}:")
        for pred in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            expected_brier = p_true * (pred - 1)**2 + (1 - p_true) * pred**2
            delta = expected_brier - p_true * (1 - p_true)
            print(f"    Predict {pred:.2f}: E[Brier]={expected_brier:.4f} (excess={delta:+.4f})")
        print()

        # Scenario 4: Two-submission strategy
        print(f"  --- Two-submission strategy ---")
        print(f"  Competition allows 2 submissions. Key insight: Brier rewards")
        print(f"  the BETTER of two correlated predictions differently than two")
        print(f"  independent ones.")
        print()
        print(f"  Strategy 1: Conservative (shrink toward seed base rates)")
        print(f"  Strategy 2: Aggressive (trust model fully, wider prediction spread)")
        print()
        print(f"  In a 'normal' tournament (few upsets), the aggressive model wins")
        print(f"  because it correctly backs favorites with high confidence.")
        print(f"  In a 'chaos' tournament (many upsets), the conservative model wins")
        print(f"  because it didn't overcommit to favorites.")
        print()
        print(f"  Expected Brier by scenario (5v12 with p=0.65):")
        n_games_512 = 4  # per tournament
        for scenario, outcome_rate in [("Normal (75% fav wins)", 0.75),
                                         ("Average (65% fav wins)", 0.65),
                                         ("Chaos (50% fav wins)", 0.50)]:
            # Conservative: predict 0.65 (base rate)
            conservative = outcome_rate * (0.65 - 1)**2 + (1 - outcome_rate) * 0.65**2
            # Aggressive: predict 0.80 (model's current)
            aggressive = outcome_rate * (0.80 - 1)**2 + (1 - outcome_rate) * 0.80**2
            # Team-aware: varies, but average 0.75
            print(f"    {scenario:30s}: conservative={conservative:.4f}, aggressive={aggressive:.4f}, "
                  f"better={('conserv' if conservative < aggressive else 'aggress')}")

        # ============================================================
        # 6. Bracket structure exploitation
        # ============================================================
        print(f"\n--- Bracket structure ---")
        print(f"  Tournament has {n_games:.0f} games total:")
        for rf in round_floors:
            print(f"    {rf['round']}: {rf['n_games_per_tourney']:.0f} games ({rf['n_games_per_tourney']/n_games:.0%} of total)")
        print()
        print(f"  R64 = {round_floors[0]['n_games_per_tourney']/n_games:.0%} of games but only "
              f"{round_floors[0]['brier_floor']*round_floors[0]['n_games_per_tourney']/(overall_floor*n_games):.0%} of Brier floor")
        later_rounds = sum(rf["brier_floor"] * rf["n_games_per_tourney"] for rf in round_floors[1:])
        r64_pts = round_floors[0]["brier_floor"] * round_floors[0]["n_games_per_tourney"]
        print(f"  R32+ = {(n_games-round_floors[0]['n_games_per_tourney'])/n_games:.0%} of games but "
              f"{later_rounds/(overall_floor*n_games):.0%} of Brier floor")
        print()
        print(f"  Implication: Later rounds are where Brier is won or lost.")
        print(f"  R64 favorites are so dominant that even bad models score well there.")
        print(f"  The competitive edge is in R32+ where matchups are less predictable.")
        print(f"  A model that's excellent at S16/E8/F4 but mediocre at R64 will beat")
        print(f"  a model that's great at R64 but poor at later rounds.")


if __name__ == "__main__":
    analyze_brier_floor()
