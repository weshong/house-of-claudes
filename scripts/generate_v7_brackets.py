"""Generate predicted brackets for v7 report."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from marchmadness.config import PREDICTION_CLIP
from marchmadness.data_loader import load_all
from marchmadness.features.builder import build_matchup_features, get_feature_columns
from marchmadness.features.seeds import parse_seed
from research import build_custom_team_features, build_custom_training_data


def train_models(data, gender):
    """Train v7 models on all data (for 2026 predictions)."""
    if gender == "M":
        seasons = list(range(2015, 2026))
        df = build_custom_training_data(data, seasons, gender)
        fc = get_feature_columns(df)
        X = df[fc].values.copy()
        y = df["Label"].values.copy()
        med = np.nanmedian(X, axis=0)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if mask.any():
                X[mask, i] = med[i] if not np.isnan(med[i]) else 0

        lgb = LGBMClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.025,
            subsample=0.8, colsample_bytree=0.6, min_split_gain=0.005,
            random_state=42, verbose=-1
        )
        lr = Pipeline([
            ("s", StandardScaler()),
            ("lr", LogisticRegression(C=100.0, penalty='l2', solver='lbfgs', max_iter=2000))
        ])
        lgb.fit(X, y)
        lr.fit(X, y)
        return {"lgb": lgb, "lr": lr, "fc": fc, "med": med}
    else:
        seasons = list(range(2003, 2026))
        df = build_custom_training_data(data, seasons, gender)
        fc = get_feature_columns(df)
        X = df[fc].values.copy()
        y = df["Label"].values.copy()
        med = np.nanmedian(X, axis=0)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if mask.any():
                X[mask, i] = med[i] if not np.isnan(med[i]) else 0

        model = Pipeline([
            ("q", QuantileTransformer(output_distribution='normal', n_quantiles=50)),
            ("lr", LogisticRegression(C=0.20, penalty='l1', solver='liblinear', max_iter=2000))
        ])
        model.fit(X, y)
        return {"model": model, "fc": fc, "med": med}


def predict(tf, team_a, team_b, models, gender):
    """Predict P(team_a wins)."""
    fc = models["fc"]
    med = models["med"]
    matchup = build_matchup_features(tf, team_a, team_b)
    if not matchup:
        return 0.5
    fv = np.array([[matchup.get(c, np.nan) for c in fc]])
    for i in range(fv.shape[1]):
        if np.isnan(fv[0, i]):
            fv[0, i] = med[i] if not np.isnan(med[i]) else 0

    if gender == "M":
        p_lgb = float(models["lgb"].predict_proba(fv)[:, 1][0])
        p_lr = float(models["lr"].predict_proba(fv)[:, 1][0])
        p = 0.40 * p_lgb + 0.60 * p_lr
    else:
        p = float(models["model"].predict_proba(fv)[:, 1][0])
    return float(np.clip(p, *PREDICTION_CLIP))


def get_bracket_structure(data, gender):
    """Get 2026 bracket from seeds."""
    seeds_df = data[f"{gender}NCAATourneySeeds"]
    seeds_2026 = seeds_df[seeds_df["Season"] == 2026]
    teams = data[f"{gender}Teams"]
    name_map = dict(zip(teams["TeamID"], teams["TeamName"]))

    bracket = {}  # region -> list of (seed, team_id, team_name)
    for _, row in seeds_2026.iterrows():
        region = row["Seed"][0]
        seed_num = parse_seed(row["Seed"])
        tid = row["TeamID"]
        name = name_map.get(tid, str(tid))
        bracket.setdefault(region, []).append((seed_num, tid, name))

    for region in bracket:
        bracket[region].sort(key=lambda x: x[0])

    return bracket


def simulate_bracket(bracket, tf, models, gender):
    """Simulate entire tournament, return results."""
    # Standard bracket matchups for R64
    r64_matchups = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]

    results = {}

    for region, teams in bracket.items():
        team_map = {seed: (tid, name) for seed, tid, name in teams}
        results[region] = {"rounds": []}

        # R64
        r64_winners = []
        r64_results = []
        for s1, s2 in r64_matchups:
            if s1 not in team_map or s2 not in team_map:
                continue
            t1_id, t1_name = team_map[s1]
            t2_id, t2_name = team_map[s2]
            ta, tb = min(t1_id, t2_id), max(t1_id, t2_id)
            p = predict(tf, ta, tb, models, gender)
            # p = P(ta wins), adjust for who is the favorite
            if ta == t1_id:
                p_fav = p  # P(seed s1 wins)
            else:
                p_fav = 1 - p  # P(seed s1 wins)

            if p_fav >= 0.5:
                winner = (s1, t1_id, t1_name)
                win_pct = p_fav
            else:
                winner = (s2, t2_id, t2_name)
                win_pct = 1 - p_fav

            r64_results.append({
                "fav": (s1, t1_name), "dog": (s2, t2_name),
                "winner": winner, "pct": win_pct,
                "upset": winner[0] == s2
            })
            r64_winners.append(winner)

        results[region]["R64"] = r64_results

        # R32: winners play in order (1/16 winner vs 8/9 winner, etc.)
        def sim_round(teams_in, round_name):
            round_results = []
            winners = []
            for i in range(0, len(teams_in), 2):
                if i + 1 >= len(teams_in):
                    winners.append(teams_in[i])
                    continue
                s1, t1_id, t1_name = teams_in[i]
                s2, t2_id, t2_name = teams_in[i + 1]
                ta, tb = min(t1_id, t2_id), max(t1_id, t2_id)
                p = predict(tf, ta, tb, models, gender)
                if ta == t1_id:
                    p1 = p
                else:
                    p1 = 1 - p

                if p1 >= 0.5:
                    winner = (s1, t1_id, t1_name)
                    win_pct = p1
                else:
                    winner = (s2, t2_id, t2_name)
                    win_pct = 1 - p1

                round_results.append({
                    "team1": (s1, t1_name), "team2": (s2, t2_name),
                    "winner": winner, "pct": win_pct,
                    "upset": s1 < s2 and winner[0] == s2 or s2 < s1 and winner[0] == s1
                })
                winners.append(winner)

            results[region][round_name] = round_results
            return winners

        r32_winners = sim_round(r64_winners, "R32")
        s16_winners = sim_round(r32_winners, "S16")
        e8_winners = sim_round(s16_winners, "E8")
        results[region]["winner"] = e8_winners[0] if e8_winners else None

    return results


def format_bracket(results, gender_label):
    """Format bracket as ASCII art similar to v5 report."""
    lines = []
    lines.append(f"### {gender_label} Tournament")
    lines.append("")
    lines.append("```")

    region_names = {"W": "EAST", "X": "SOUTH", "Y": "MIDWEST", "Z": "WEST"}
    ff_teams = []

    for region in ["W", "X", "Y", "Z"]:
        if region not in results:
            continue
        r = results[region]
        rname = region_names.get(region, region)
        lines.append(f"REGION {region} ({rname})")
        lines.append(f"{'':20s}R64{'':17s}R32{'':14s}S16{'':13s}E8")
        lines.append(f"{'':20s}{'─'*17}   {'─'*14}   {'─'*13}   {'─'*14}")

        r64 = r.get("R64", [])
        r32 = r.get("R32", [])
        s16 = r.get("S16", [])
        e8 = r.get("E8", [])

        # Top half of bracket (first 4 R64 matchups)
        for i in range(4):
            if i >= len(r64):
                break
            m = r64[i]
            fav_s, fav_n = m["fav"]
            dog_s, dog_n = m["dog"]
            w = m["winner"]
            pct = m["pct"]
            upset = m["upset"]

            fav_short = fav_n[:12]
            dog_short = dog_n[:12]
            w_short = w[2][:12]

            lines.append(f" ({fav_s:>2d}) {fav_short:<12s}──┐")
            r32_part = ""
            if i % 2 == 0 and i // 2 < len(r32):
                r32m = r32[i // 2]
                r32w = r32m["winner"]
                r32pct = r32m["pct"]
                r32_short = r32w[2][:12]

                s16_part = ""
                if i == 0 and len(s16) > 0:
                    s16m = s16[0]
                    s16w = s16m["winner"]
                    s16pct = s16m["pct"]
                    s16_short = s16w[2][:12]

                    e8_part = ""
                    if len(e8) > 0:
                        e8m = e8[0]
                        e8w = e8m["winner"]
                        e8pct = e8m["pct"]

                    lines.append(f"        {pct:5.1%}{'':3s}├── ({w[0]:>2d}) {w_short:<11s}──┐")
                    lines.append(f"({dog_s:>2d}) {dog_short:<12s}──┘{'':10s}{r32pct:5.1%}{'':3s}├── ({r32w[0]:>2d}) {r32_short:<11s}──┐")
                else:
                    lines.append(f"        {pct:5.1%}{'':3s}├── ({w[0]:>2d}) {w_short:<11s}──┐")
                    lines.append(f"({dog_s:>2d}) {dog_short:<12s}──┘{'':10s}{r32pct:5.1%}{'':3s}├── ({r32w[0]:>2d}) {r32_short:<11s}──┘")
            else:
                lines.append(f"        {pct:5.1%}{'':3s}├── ({w[0]:>2d}) {w_short:<11s}──┘")
                lines.append(f"({dog_s:>2d}) {dog_short:<12s}──┘")

            if i == 1:
                # After second R64 game, show S16 winner and E8
                if len(s16) > 0 and len(e8) > 0:
                    e8m = e8[0]
                    e8w = e8m["winner"]
                    e8pct = e8m["pct"]
                    lines.append(f"{'':64s}├── ({e8w[0]:>2d}) {e8w[2][:12]}")
                    lines.append(f"{'':64s}│{'':7s}{e8pct:.1%}")

        lines.append("")

        # Bottom half (next 4 R64 matchups)
        for i in range(4, 8):
            if i >= len(r64):
                break
            m = r64[i]
            fav_s, fav_n = m["fav"]
            dog_s, dog_n = m["dog"]
            w = m["winner"]
            pct = m["pct"]
            upset_str = "  ** UPSET **" if m["upset"] else ""

            fav_short = fav_n[:12]
            dog_short = dog_n[:12]
            w_short = w[2][:12]

            lines.append(f" ({fav_s:>2d}) {fav_short:<12s}──┐")
            if (i - 4) % 2 == 0 and (i - 4) // 2 < len(r32) - 2:
                r32m = r32[2 + (i - 4) // 2]
                r32w = r32m["winner"]
                r32pct = r32m["pct"]
                r32_short = r32w[2][:12]

                lines.append(f"        {pct:5.1%}{'':3s}├── ({w[0]:>2d}) {w_short:<11s}──┐{upset_str}")
                lines.append(f"({dog_s:>2d}) {dog_short:<12s}──┘{'':10s}{r32pct:5.1%}{'':3s}├── ({r32w[0]:>2d}) {r32_short:<11s}──┐")
            else:
                if (i - 4) == 1 and len(s16) > 1:
                    s16m = s16[1]
                    s16w = s16m["winner"]
                    s16pct = s16m["pct"]
                    lines.append(f"        {pct:5.1%}{'':3s}├── ({w[0]:>2d}) {w_short:<11s}──┐{upset_str}")
                    lines.append(f"({dog_s:>2d}) {dog_short:<12s}──┘{'':10s}{r32pct:5.1%}{'':3s}├── ({r32w[0]:>2d}) {r32_short:<11s}──┘")
                else:
                    lines.append(f"        {pct:5.1%}{'':3s}├── ({w[0]:>2d}) {w_short:<11s}──┘{upset_str}")
                    lines.append(f"({dog_s:>2d}) {dog_short:<12s}──┘")

        winner = r.get("winner")
        if winner:
            ff_teams.append((region, winner))
            lines.append(f"\n>>> REGION {region} WINNER: ({winner[0]}) {winner[2]}")
        lines.append("")
        lines.append("")

    # Final Four
    if len(ff_teams) >= 4:
        lines.append("FINAL FOUR")
        lines.append("═" * 67)
        lines.append("")

        # Semi 1: W vs X
        _, t1 = ff_teams[0]
        _, t2 = ff_teams[1]
        ta, tb = min(t1[1], t2[1]), max(t1[1], t2[1])
        # We need tf and models but don't have them here
        # We'll compute this in the caller
        lines.append(f"  Semifinal 1:  ({t1[0]}) {t1[2]} [{ff_teams[0][0]}]  vs  ({t2[0]}) {t2[2]} [{ff_teams[1][0]}]")
        lines.append(f"  Semifinal 2:  ({ff_teams[2][1][0]}) {ff_teams[2][1][2]} [{ff_teams[2][0]}]  vs  ({ff_teams[3][1][0]}) {ff_teams[3][1][2]} [{ff_teams[3][0]}]")

    lines.append("```")

    return "\n".join(lines), ff_teams


def main():
    print("Loading data...")
    data = load_all()

    for gender, label in [("M", "Men's"), ("W", "Women's")]:
        print(f"\n=== {label} ===")
        print("Training models...")
        models = train_models(data, gender)
        print(f"  Features: {len(models['fc'])}")

        print("Getting bracket structure...")
        bracket = get_bracket_structure(data, gender)
        for region, teams in sorted(bracket.items()):
            print(f"  Region {region}: {len(teams)} teams")
            for seed, tid, name in teams:
                print(f"    ({seed}) {name} [{tid}]")

        print("Building team features...")
        tf = build_custom_team_features(data, 2026, gender)
        print(f"  {len(tf)} teams with features")

        print("Simulating bracket...")
        results = simulate_bracket(bracket, tf, models, gender)

        # Print results
        for region in ["W", "X", "Y", "Z"]:
            if region not in results:
                continue
            r = results[region]
            winner = r.get("winner")
            print(f"\n  Region {region}:")
            for rnd_name in ["R64", "R32", "S16", "E8"]:
                rnd = r.get(rnd_name, [])
                for m in rnd:
                    w = m["winner"]
                    pct = m["pct"]
                    upset = " ** UPSET **" if m.get("upset") else ""
                    if "fav" in m:
                        print(f"    {rnd_name}: ({m['fav'][0]}) {m['fav'][1]} vs ({m['dog'][0]}) {m['dog'][1]} -> ({w[0]}) {w[2]} {pct:.1%}{upset}")
                    else:
                        print(f"    {rnd_name}: ({m['team1'][0]}) {m['team1'][1]} vs ({m['team2'][0]}) {m['team2'][1]} -> ({w[0]}) {w[2]} {pct:.1%}{upset}")
            if winner:
                print(f"    Winner: ({winner[0]}) {winner[2]}")

        # Final Four
        ff = []
        for region in ["W", "X", "Y", "Z"]:
            if region in results and results[region].get("winner"):
                ff.append((region, results[region]["winner"]))

        if len(ff) >= 4:
            print(f"\n  FINAL FOUR:")
            # Semi 1: W vs X
            t1 = ff[0][1]
            t2 = ff[1][1]
            ta, tb = min(t1[1], t2[1]), max(t1[1], t2[1])
            p = predict(tf, ta, tb, models, gender)
            p1 = p if ta == t1[1] else 1 - p
            semi1_winner = t1 if p1 >= 0.5 else t2
            semi1_pct = p1 if p1 >= 0.5 else 1 - p1
            print(f"    Semi 1: ({t1[0]}) {t1[2]} vs ({t2[0]}) {t2[2]} -> ({semi1_winner[0]}) {semi1_winner[2]} {semi1_pct:.1%}")

            t3 = ff[2][1]
            t4 = ff[3][1]
            ta, tb = min(t3[1], t4[1]), max(t3[1], t4[1])
            p = predict(tf, ta, tb, models, gender)
            p3 = p if ta == t3[1] else 1 - p
            semi2_winner = t3 if p3 >= 0.5 else t4
            semi2_pct = p3 if p3 >= 0.5 else 1 - p3
            print(f"    Semi 2: ({t3[0]}) {t3[2]} vs ({t4[0]}) {t4[2]} -> ({semi2_winner[0]}) {semi2_winner[2]} {semi2_pct:.1%}")

            ta, tb = min(semi1_winner[1], semi2_winner[1]), max(semi1_winner[1], semi2_winner[1])
            p = predict(tf, ta, tb, models, gender)
            p_s1 = p if ta == semi1_winner[1] else 1 - p
            champ = semi1_winner if p_s1 >= 0.5 else semi2_winner
            champ_pct = p_s1 if p_s1 >= 0.5 else 1 - p_s1
            print(f"    CHAMPION: ({champ[0]}) {champ[2]} {champ_pct:.1%}")


if __name__ == "__main__":
    main()
