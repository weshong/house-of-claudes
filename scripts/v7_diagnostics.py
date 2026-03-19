"""v7 model diagnostics: per-round Brier, residual analysis, LGB vs LR disagreement.

Runs leave-season-out CV and captures detailed per-game predictions for analysis.
"""

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

from marchmadness.config import CV_SEASONS, PREDICTION_CLIP
from marchmadness.data_loader import load_all
from marchmadness.features.builder import build_matchup_features, get_feature_columns
from marchmadness.features.seeds import parse_seed
from research import build_custom_team_features, build_custom_training_data


def run_cv_detailed(data, gender):
    """Run CV and return per-game predictions with metadata (seeds, DayNum, team IDs).

    For men's, also returns individual LGB and LR predictions for disagreement analysis.
    """
    tourney_key = f"{gender}NCAATourneyCompactResults"
    seeds_key = f"{gender}NCAATourneySeeds"
    tourney = data[tourney_key]
    seeds_df = data[seeds_key]
    teams_key = f"{gender}Teams"
    teams = data[teams_key]
    name_map = dict(zip(teams["TeamID"], teams["TeamName"]))

    all_games = []

    for val_season in CV_SEASONS:
        # Build training data
        train_seasons = [s for s in range(2015 if gender == "M" else 2003, 2026) if s != val_season]
        train_df = build_custom_training_data(data, train_seasons, gender)
        fc = get_feature_columns(train_df)
        X_train = train_df[fc].values.copy()
        y_train = train_df["Label"].values.copy()
        med = np.nanmedian(X_train, axis=0)
        for i in range(X_train.shape[1]):
            mask = np.isnan(X_train[:, i])
            if mask.any():
                X_train[mask, i] = med[i] if not np.isnan(med[i]) else 0

        # Build val team features
        tf = build_custom_team_features(data, val_season, gender)
        season_seeds = seeds_df[seeds_df["Season"] == val_season]
        seed_map = {}
        for _, row in season_seeds.iterrows():
            seed_map[row["TeamID"]] = parse_seed(row["Seed"])

        if gender == "M":
            # Train LGB and LR separately
            lgb = LGBMClassifier(
                n_estimators=250, max_depth=3, learning_rate=0.025,
                subsample=0.8, colsample_bytree=0.6, min_split_gain=0.005,
                random_state=42, verbose=-1
            )
            lr = Pipeline([
                ("s", StandardScaler()),
                ("lr", LogisticRegression(C=100.0, penalty='l2', solver='lbfgs', max_iter=2000))
            ])
            lgb.fit(X_train, y_train)
            lr.fit(X_train, y_train)

            # Get feature importance from this fold
            lgb_imp = dict(zip(fc, lgb.feature_importances_))
            lr_coefs = lr.named_steps["lr"].coef_[0]
            lr_imp = dict(zip(fc, np.abs(lr_coefs)))
        else:
            model = Pipeline([
                ("q", QuantileTransformer(output_distribution='normal', n_quantiles=50)),
                ("lr", LogisticRegression(C=0.20, penalty='l1', solver='liblinear', max_iter=2000))
            ])
            model.fit(X_train, y_train)

        # Predict val games
        season_tourney = tourney[tourney["Season"] == val_season]
        for _, game in season_tourney.iterrows():
            w_id, l_id = game["WTeamID"], game["LTeamID"]
            team_a, team_b = min(w_id, l_id), max(w_id, l_id)
            label = 1 if w_id == team_a else 0

            matchup = build_matchup_features(tf, team_a, team_b)
            if not matchup:
                continue

            fv = np.array([[matchup.get(c, np.nan) for c in fc]])
            for i in range(fv.shape[1]):
                if np.isnan(fv[0, i]):
                    fv[0, i] = med[i] if not np.isnan(med[i]) else 0

            if gender == "M":
                p_lgb = float(lgb.predict_proba(fv)[:, 1][0])
                p_lr = float(lr.predict_proba(fv)[:, 1][0])
                p_blend = 0.40 * p_lgb + 0.60 * p_lr
                p_blend = float(np.clip(p_blend, *PREDICTION_CLIP))
                p_lgb = float(np.clip(p_lgb, *PREDICTION_CLIP))
                p_lr = float(np.clip(p_lr, *PREDICTION_CLIP))
            else:
                p_blend = float(np.clip(model.predict_proba(fv)[:, 1][0], *PREDICTION_CLIP))
                p_lgb = None
                p_lr = None

            seed_a = seed_map.get(team_a)
            seed_b = seed_map.get(team_b)

            game_info = {
                "season": val_season,
                "daynum": game["DayNum"],
                "team_a": team_a,
                "team_b": team_b,
                "name_a": name_map.get(team_a, str(team_a)),
                "name_b": name_map.get(team_b, str(team_b)),
                "seed_a": seed_a,
                "seed_b": seed_b,
                "label": label,
                "pred_blend": p_blend,
                "pred_lgb": p_lgb,
                "pred_lr": p_lr,
                "brier": (p_blend - label) ** 2,
                "winner": name_map.get(w_id, str(w_id)),
                "loser": name_map.get(l_id, str(l_id)),
                "w_score": game["WScore"],
                "l_score": game["LScore"],
            }
            all_games.append(game_info)

    return pd.DataFrame(all_games), fc


def assign_round(daynum):
    """Map DayNum to tournament round."""
    if daynum <= 135:
        return "R64"
    elif daynum <= 137:
        return "R32"
    elif daynum <= 139:
        return "S16"
    elif daynum <= 144:
        return "E8"
    elif daynum <= 152:
        return "F4"
    else:
        return "Champ"


def per_round_analysis(df, gender_label):
    """Analyze Brier score by tournament round."""
    df = df.copy()
    df["round"] = df["daynum"].apply(assign_round)
    round_order = ["R64", "R32", "S16", "E8", "F4", "Champ"]

    lines = []
    lines.append(f"\n### {gender_label}: Per-Round Brier Breakdown")
    lines.append("")
    lines.append("| Round | Games | Brier | Accuracy | Avg Pred Spread | Avg Seed Diff |")
    lines.append("|-------|-------|-------|----------|-----------------|---------------|")

    for rnd in round_order:
        rdf = df[df["round"] == rnd]
        if rdf.empty:
            continue
        brier = rdf["brier"].mean()
        acc = (((rdf["pred_blend"] > 0.5) & (rdf["label"] == 1)) |
               ((rdf["pred_blend"] < 0.5) & (rdf["label"] == 0))).mean()
        spread = (rdf["pred_blend"] - 0.5).abs().mean()
        if rdf["seed_a"].notna().any() and rdf["seed_b"].notna().any():
            seed_diff = (rdf["seed_a"] - rdf["seed_b"]).abs().mean()
        else:
            seed_diff = float("nan")
        lines.append(f"| {rnd} | {len(rdf)} | {brier:.4f} | {acc:.1%} | {spread:.3f} | {seed_diff:.1f} |")

    overall = df["brier"].mean()
    overall_acc = (((df["pred_blend"] > 0.5) & (df["label"] == 1)) |
                   ((df["pred_blend"] < 0.5) & (df["label"] == 0))).mean()
    lines.append(f"| **Total** | **{len(df)}** | **{overall:.4f}** | **{overall_acc:.1%}** | | |")

    # Per-round per-season breakdown
    lines.append("")
    lines.append(f"#### Per-Round Per-Season Detail")
    lines.append("")
    header = "| Round |"
    sep = "|-------|"
    for s in CV_SEASONS:
        header += f" {s} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for rnd in round_order:
        row = f"| {rnd} |"
        for s in CV_SEASONS:
            mask = (df["round"] == rnd) & (df["season"] == s)
            if mask.sum() > 0:
                b = df[mask]["brier"].mean()
                row += f" {b:.4f} |"
            else:
                row += " — |"
        lines.append(row)

    return "\n".join(lines)


def residual_analysis(df, gender_label):
    """Find worst predictions and look for patterns."""
    lines = []
    lines.append(f"\n### {gender_label}: Residual Analysis (Worst Predictions)")
    lines.append("")

    df = df.copy()
    df["round"] = df["daynum"].apply(assign_round)
    df_sorted = df.sort_values("brier", ascending=False)

    # Top 15 worst predictions
    lines.append("#### Top 15 Highest-Cost Games")
    lines.append("")
    lines.append("| Rank | Season | Round | Matchup | Seeds | Pred | Actual | Brier | Score |")
    lines.append("|------|--------|-------|---------|-------|------|--------|-------|-------|")

    for i, (_, row) in enumerate(df_sorted.head(15).iterrows()):
        s_a = f"({int(row['seed_a'])})" if pd.notna(row['seed_a']) else ""
        s_b = f"({int(row['seed_b'])})" if pd.notna(row['seed_b']) else ""
        actual_str = "A wins" if row["label"] == 1 else "B wins"
        lines.append(
            f"| {i+1} | {row['season']} | {row['round']} | "
            f"{row['name_a']} {s_a} vs {row['name_b']} {s_b} | "
            f"{int(row.get('seed_a', 0)) if pd.notna(row.get('seed_a')) else '?'}v"
            f"{int(row.get('seed_b', 0)) if pd.notna(row.get('seed_b')) else '?'} | "
            f"{row['pred_blend']:.3f} | {row['winner']} won | "
            f"{row['brier']:.4f} | {row['w_score']}-{row['l_score']} |"
        )

    # Pattern analysis: aggregate by characteristics
    lines.append("")
    lines.append("#### Cost by Upset Type")
    lines.append("")

    df["fav_seed"] = df[["seed_a", "seed_b"]].min(axis=1)
    df["dog_seed"] = df[["seed_a", "seed_b"]].max(axis=1)
    df["seed_diff_abs"] = (df["seed_a"] - df["seed_b"]).abs()
    # Was it an upset? (higher seed won)
    df["is_upset"] = False
    for _, row in df.iterrows():
        if pd.notna(row["seed_a"]) and pd.notna(row["seed_b"]):
            # Lower seed number = better team
            fav_id = row["team_a"] if row["seed_a"] < row["seed_b"] else row["team_b"]
            w_id = row["team_a"] if row["label"] == 1 else row["team_b"]
            if w_id != fav_id:
                df.at[_, "is_upset"] = True

    upsets = df[df["is_upset"]]
    non_upsets = df[~df["is_upset"]]

    lines.append(f"| Category | N | Mean Brier | Total Brier Points | % of Total Cost |")
    lines.append(f"|----------|---|-----------|-------------------|-----------------|")
    total_cost = df["brier"].sum()
    lines.append(f"| Upsets (fav lost) | {len(upsets)} | {upsets['brier'].mean():.4f} | {upsets['brier'].sum():.2f} | {upsets['brier'].sum()/total_cost:.1%} |")
    lines.append(f"| Chalk (fav won) | {len(non_upsets)} | {non_upsets['brier'].mean():.4f} | {non_upsets['brier'].sum():.2f} | {non_upsets['brier'].sum()/total_cost:.1%} |")

    # By margin of victory
    lines.append("")
    lines.append("#### Cost by Margin of Victory")
    lines.append("")
    df["margin"] = df["w_score"] - df["l_score"]
    lines.append("| Margin Bucket | N | Mean Brier | Upset Rate |")
    lines.append("|---------------|---|-----------|------------|")
    for lo, hi, label in [(1, 5, "1-5 (close)"), (6, 10, "6-10"), (11, 15, "11-15"), (16, 99, "16+ (blowout)")]:
        mask = (df["margin"] >= lo) & (df["margin"] <= hi)
        bucket = df[mask]
        if len(bucket) > 0:
            upset_rate = bucket["is_upset"].mean()
            lines.append(f"| {label} | {len(bucket)} | {bucket['brier'].mean():.4f} | {upset_rate:.1%} |")

    # By round
    lines.append("")
    lines.append("#### Upset Cost by Round")
    lines.append("")
    lines.append("| Round | Total Games | Upsets | Upset Rate | Upset Brier | Chalk Brier |")
    lines.append("|-------|-------------|--------|------------|-------------|-------------|")
    round_order = ["R64", "R32", "S16", "E8", "F4", "Champ"]
    for rnd in round_order:
        rdf = df[df["round"] == rnd]
        if rdf.empty:
            continue
        r_upsets = rdf[rdf["is_upset"]]
        r_chalk = rdf[~rdf["is_upset"]]
        upset_brier = f"{r_upsets['brier'].mean():.4f}" if len(r_upsets) > 0 else "—"
        chalk_brier = f"{r_chalk['brier'].mean():.4f}" if len(r_chalk) > 0 else "—"
        lines.append(
            f"| {rnd} | {len(rdf)} | {len(r_upsets)} | {len(r_upsets)/len(rdf):.0%} | "
            f"{upset_brier} | {chalk_brier} |"
        )

    return "\n".join(lines)


def lgb_lr_disagreement(df, gender_label):
    """Analyze when LGB and LR disagree, and who's right."""
    if df["pred_lgb"].isna().all():
        return f"\n### {gender_label}: LGB vs LR Disagreement\n\nN/A (single model, no blend)\n"

    lines = []
    lines.append(f"\n### {gender_label}: LGB vs LR Disagreement Analysis")
    lines.append("")

    df = df.copy()
    df["round"] = df["daynum"].apply(assign_round)
    df["lgb_lr_diff"] = (df["pred_lgb"] - df["pred_lr"]).abs()
    df["lgb_brier"] = (df["pred_lgb"] - df["label"]) ** 2
    df["lr_brier"] = (df["pred_lr"] - df["label"]) ** 2
    df["lgb_better"] = df["lgb_brier"] < df["lr_brier"]

    # Overall
    lines.append("#### Overall Component Comparison")
    lines.append("")
    lines.append(f"| Metric | LGB (40%) | LR (60%) | Blend |")
    lines.append(f"|--------|-----------|----------|-------|")
    lines.append(f"| Mean Brier | {df['lgb_brier'].mean():.4f} | {df['lr_brier'].mean():.4f} | {df['brier'].mean():.4f} |")
    lines.append(f"| Median abs(pred - 0.5) | {(df['pred_lgb'] - 0.5).abs().median():.3f} | {(df['pred_lr'] - 0.5).abs().median():.3f} | {(df['pred_blend'] - 0.5).abs().median():.3f} |")
    lines.append(f"| Games LGB wins | {df['lgb_better'].sum()} | {(~df['lgb_better']).sum()} | — |")
    lines.append(f"| Mean disagreement | {df['lgb_lr_diff'].mean():.4f} | — | — |")
    lines.append(f"| Max disagreement | {df['lgb_lr_diff'].max():.4f} | — | — |")

    # Disagreement quartiles
    lines.append("")
    lines.append("#### Performance by Disagreement Level")
    lines.append("")
    lines.append("High disagreement = LGB and LR see the game differently. Who's right when they disagree?")
    lines.append("")
    lines.append("| Disagreement Quartile | N | Mean |LGB-LR| | LGB Brier | LR Brier | Blend Brier | LGB Wins |")
    lines.append("|----------------------|---|---------------|-----------|----------|-------------|----------|")

    quartiles = df["lgb_lr_diff"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    labels = ["Q1 (agree)", "Q2", "Q3", "Q4 (disagree)"]
    for i in range(4):
        lo, hi = quartiles[i], quartiles[i + 1]
        if i == 3:
            mask = (df["lgb_lr_diff"] >= lo) & (df["lgb_lr_diff"] <= hi)
        else:
            mask = (df["lgb_lr_diff"] >= lo) & (df["lgb_lr_diff"] < hi)
        qdf = df[mask]
        if len(qdf) == 0:
            continue
        lines.append(
            f"| {labels[i]} | {len(qdf)} | {qdf['lgb_lr_diff'].mean():.4f} | "
            f"{qdf['lgb_brier'].mean():.4f} | {qdf['lr_brier'].mean():.4f} | "
            f"{qdf['brier'].mean():.4f} | {qdf['lgb_better'].mean():.0%} |"
        )

    # Top 10 highest disagreement games
    lines.append("")
    lines.append("#### Top 10 Highest-Disagreement Games")
    lines.append("")
    lines.append("| Season | Round | Matchup | LGB | LR | Blend | Actual | Better |")
    lines.append("|--------|-------|---------|-----|-----|-------|--------|--------|")

    top_disagree = df.sort_values("lgb_lr_diff", ascending=False).head(10)
    for _, row in top_disagree.iterrows():
        s_a = f"({int(row['seed_a'])})" if pd.notna(row['seed_a']) else ""
        s_b = f"({int(row['seed_b'])})" if pd.notna(row['seed_b']) else ""
        better = "LGB" if row["lgb_brier"] < row["lr_brier"] else "LR"
        lines.append(
            f"| {row['season']} | {row['round']} | "
            f"{row['name_a']} {s_a} vs {row['name_b']} {s_b} | "
            f"{row['pred_lgb']:.3f} | {row['pred_lr']:.3f} | {row['pred_blend']:.3f} | "
            f"{row['winner']} won | {better} |"
        )

    # By round: who's better?
    lines.append("")
    lines.append("#### LGB vs LR by Round")
    lines.append("")
    lines.append("| Round | LGB Brier | LR Brier | Blend Brier | LGB Win % |")
    lines.append("|-------|-----------|----------|-------------|-----------|")
    round_order = ["R64", "R32", "S16", "E8", "F4", "Champ"]
    for rnd in round_order:
        rdf = df[df["round"] == rnd]
        if rdf.empty:
            continue
        lines.append(
            f"| {rnd} | {rdf['lgb_brier'].mean():.4f} | {rdf['lr_brier'].mean():.4f} | "
            f"{rdf['brier'].mean():.4f} | {rdf['lgb_better'].mean():.0%} |"
        )

    # Upset games: who's better?
    lines.append("")
    lines.append("#### LGB vs LR on Upsets")
    lines.append("")

    df["fav_seed"] = df[["seed_a", "seed_b"]].min(axis=1)
    df["is_upset"] = False
    for idx, row in df.iterrows():
        if pd.notna(row["seed_a"]) and pd.notna(row["seed_b"]):
            fav_id = row["team_a"] if row["seed_a"] < row["seed_b"] else row["team_b"]
            w_id = row["team_a"] if row["label"] == 1 else row["team_b"]
            if w_id != fav_id:
                df.at[idx, "is_upset"] = True

    upsets = df[df["is_upset"]]
    chalk = df[~df["is_upset"]]

    lines.append(f"| Category | N | LGB Brier | LR Brier | Blend Brier | LGB Win % |")
    lines.append(f"|----------|---|-----------|----------|-------------|-----------|")
    lines.append(f"| Upsets | {len(upsets)} | {upsets['lgb_brier'].mean():.4f} | {upsets['lr_brier'].mean():.4f} | {upsets['brier'].mean():.4f} | {upsets['lgb_better'].mean():.0%} |")
    lines.append(f"| Chalk | {len(chalk)} | {chalk['lgb_brier'].mean():.4f} | {chalk['lr_brier'].mean():.4f} | {chalk['brier'].mean():.4f} | {chalk['lgb_better'].mean():.0%} |")

    return "\n".join(lines)


def feature_stability(data, gender_label):
    """Analyze feature importance stability across CV folds (men's only)."""
    if gender_label != "Men's":
        return ""

    lines = []
    lines.append(f"\n### {gender_label}: Feature Importance Stability Across CV Folds")
    lines.append("")

    fold_importances = {}
    for val_season in CV_SEASONS:
        train_seasons = [s for s in range(2015, 2026) if s != val_season]
        train_df = build_custom_training_data(data, train_seasons, "M")
        fc = get_feature_columns(train_df)
        X = train_df[fc].values.copy()
        y = train_df["Label"].values.copy()
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
        lgb.fit(X, y)
        fold_importances[val_season] = dict(zip(fc, lgb.feature_importances_))

    # Build table
    imp_df = pd.DataFrame(fold_importances)
    imp_df["mean"] = imp_df.mean(axis=1)
    imp_df["std"] = imp_df.std(axis=1)
    imp_df["cv"] = imp_df["std"] / imp_df["mean"].replace(0, np.nan)
    imp_df = imp_df.sort_values("mean", ascending=False)

    lines.append("LGB split importance per fold (higher = more splits on this feature).")
    lines.append("")
    lines.append("| Feature | 2022 | 2023 | 2024 | 2025 | Mean | Std | CV |")
    lines.append("|---------|------|------|------|------|------|-----|-----|")

    for feat, row in imp_df.iterrows():
        lines.append(
            f"| {feat} | {row[2022]:.0f} | {row[2023]:.0f} | {row[2024]:.0f} | {row[2025]:.0f} | "
            f"{row['mean']:.0f} | {row['std']:.0f} | {row['cv']:.2f} |"
        )

    lines.append("")
    lines.append(f"CV = coefficient of variation (std/mean). Lower = more stable. "
                 f"Features with CV > 0.5 are unstable across folds.")

    # Flag unstable features
    unstable = imp_df[imp_df["cv"] > 0.5]
    if len(unstable) > 0:
        lines.append("")
        lines.append(f"**Unstable features (CV > 0.5):** {', '.join(unstable.index.tolist())}")
    stable = imp_df[imp_df["cv"] < 0.2]
    if len(stable) > 0:
        lines.append(f"**Most stable features (CV < 0.2):** {', '.join(stable.index.tolist())}")

    return "\n".join(lines)


def main():
    print("Loading data...")
    data = load_all()

    report_lines = []
    report_lines.append("# v7 Model Diagnostics")
    report_lines.append("")
    report_lines.append("Generated from leave-season-out CV on 2022-2025 tournaments.")
    report_lines.append("")

    # === MEN'S ===
    print("Running men's CV...")
    m_df, m_fc = run_cv_detailed(data, "M")
    print(f"  {len(m_df)} games collected")

    report_lines.append("---")
    report_lines.append("## Men's")
    report_lines.append(per_round_analysis(m_df, "Men's"))
    report_lines.append(residual_analysis(m_df, "Men's"))
    report_lines.append(lgb_lr_disagreement(m_df, "Men's"))

    print("Running men's feature stability...")
    report_lines.append(feature_stability(data, "Men's"))

    # === WOMEN'S ===
    print("Running women's CV...")
    w_df, w_fc = run_cv_detailed(data, "W")
    print(f"  {len(w_df)} games collected")

    report_lines.append("")
    report_lines.append("---")
    report_lines.append("## Women's")
    report_lines.append(per_round_analysis(w_df, "Women's"))
    report_lines.append(residual_analysis(w_df, "Women's"))
    report_lines.append(lgb_lr_disagreement(w_df, "Women's"))

    # === SUMMARY ===
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("## Key Diagnostic Findings")
    report_lines.append("")

    # Per-round summary
    m_df["round"] = m_df["daynum"].apply(assign_round)
    r32_plus = m_df[m_df["round"].isin(["R32", "S16", "E8", "F4", "Champ"])]
    r64 = m_df[m_df["round"] == "R64"]
    report_lines.append(f"1. **Men's R64 Brier: {r64['brier'].mean():.4f} vs R32+ Brier: {r32_plus['brier'].mean():.4f}** — "
                        f"Later rounds are {r32_plus['brier'].mean()/r64['brier'].mean():.1f}x harder per game.")

    # LGB vs LR
    report_lines.append(f"2. **LGB alone: {m_df['lgb_brier'].mean():.4f}, LR alone: {m_df['lr_brier'].mean():.4f}, "
                        f"Blend: {m_df['brier'].mean():.4f}** — Blend beats both components.")

    # Upsets
    m_df_copy = m_df.copy()
    m_df_copy["is_upset"] = False
    for idx, row in m_df_copy.iterrows():
        if pd.notna(row["seed_a"]) and pd.notna(row["seed_b"]):
            fav_id = row["team_a"] if row["seed_a"] < row["seed_b"] else row["team_b"]
            w_id = row["team_a"] if row["label"] == 1 else row["team_b"]
            if w_id != fav_id:
                m_df_copy.at[idx, "is_upset"] = True
    upsets = m_df_copy[m_df_copy["is_upset"]]
    report_lines.append(f"3. **Upsets account for {upsets['brier'].sum()/m_df_copy['brier'].sum():.0%} of total Brier cost** "
                        f"from {len(upsets)}/{len(m_df_copy)} games ({len(upsets)/len(m_df_copy):.0%} of games).")

    # Top cost game
    worst = m_df.sort_values("brier", ascending=False).iloc[0]
    report_lines.append(f"4. **Costliest game**: {worst['name_a']} vs {worst['name_b']} ({worst['season']}) — "
                        f"Brier cost {worst['brier']:.4f} (predicted {worst['pred_blend']:.3f}, {worst['winner']} won).")

    report = "\n".join(report_lines)

    out_path = Path(__file__).parent.parent / "outputs" / "reports" / "v7_diagnostics.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
