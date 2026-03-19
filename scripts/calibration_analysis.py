"""Deep calibration, upset, and feature importance analysis comparing v5 and v7 models.

Generates outputs/reports/calibration_feature_analysis.md
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from marchmadness.data_loader import load_all
from marchmadness.features.builder import (
    build_team_features, build_matchup_features, build_training_data, get_feature_columns
)
from marchmadness.features import seeds, ordinals, elo, season_stats, efficiency, four_factors, adj_efficiency
from marchmadness.features.trank_clone import compute as compute_trank
from marchmadness.config import CV_SEASONS, PREDICTION_CLIP
from marchmadness.evaluation.metrics import brier_score, calibration_error, compute_accuracy

# ============================================================
# V5 and V7 configs
# ============================================================

def build_v5_team_features(data, season, gender):
    """V5 custom features — uses top-10 ordinal systems (not PCA)."""
    from marchmadness.features import torvik

    if gender == "W":
        team_df = seeds.compute(data, season, gender)
        if team_df.empty:
            return team_df
        elo_df = elo.compute(data, season, gender)
        if not elo_df.empty:
            team_df = team_df.merge(elo_df, on="TeamID", how="left")
        trank_df = compute_trank(data, season, gender)
        if not trank_df.empty:
            team_df = team_df.merge(trank_df[["TeamID", "TRank_barthag", "TRank_adjoe", "TRank_adjde"]], on="TeamID", how="left")
        return team_df

    # Men's v5: ordinal aggregates (named systems), Torvik, Elo, TRank clone + gaps
    team_df = seeds.compute(data, season, gender)
    if team_df.empty:
        return team_df

    # Individual ordinal systems (v5 used named systems, not PCA)
    TOP_SYSTEMS = ["POM", "WLK", "MOR", "DOL", "COL", "RPI"]
    ord_key = f"{gender}MasseyOrdinals"
    if ord_key in data:
        all_ord = data[ord_key]
        season_ord = all_ord[
            (all_ord["Season"] == season) &
            (all_ord["RankingDayNum"] >= 128)
        ]
        if not season_ord.empty:
            for sys_name in TOP_SYSTEMS:
                sys_data = season_ord[season_ord["SystemName"] == sys_name]
                if not sys_data.empty:
                    latest = sys_data.loc[sys_data.groupby("TeamID")["RankingDayNum"].idxmax()]
                    sys_df = latest[["TeamID", "OrdinalRank"]].rename(columns={"OrdinalRank": f"Ord_{sys_name}"})
                    team_df = team_df.merge(sys_df, on="TeamID", how="left")

            # Ordinal aggregates
            all_latest = season_ord.loc[season_ord.groupby(["TeamID", "SystemName"])["RankingDayNum"].idxmax()]
            ord_stats = all_latest.groupby("TeamID")["OrdinalRank"].agg(["mean", "std"]).reset_index()
            ord_stats.columns = ["TeamID", "OrdinalMean", "OrdinalStd"]
            team_df = team_df.merge(ord_stats, on="TeamID", how="left")

    # Torvik external
    torvik_df = torvik.compute(data, season, gender)
    if not torvik_df.empty:
        team_df = team_df.merge(torvik_df, on="TeamID", how="left")

    # Elo
    elo_df = elo.compute(data, season, gender)
    if not elo_df.empty:
        team_df = team_df.merge(elo_df, on="TeamID", how="left")

    # TRank clone
    trank_df = compute_trank(data, season, gender)
    if not trank_df.empty:
        team_df = team_df.merge(trank_df[["TeamID", "TRank_barthag", "TRank_adjoe", "TRank_adjde", "TRank_sos", "TRank_tempo"]], on="TeamID", how="left")

    # Disagreement features (same as v7)
    if "Torvik_barthag" in team_df.columns and "TRank_barthag" in team_df.columns:
        team_df["barthag_gap"] = team_df["Torvik_barthag"] - team_df["TRank_barthag"]
    if "Torvik_adjoe" in team_df.columns and "TRank_adjoe" in team_df.columns:
        team_df["adjoe_gap"] = team_df["Torvik_adjoe"] - team_df["TRank_adjoe"]
    if "Torvik_adjde" in team_df.columns and "TRank_adjde" in team_df.columns:
        team_df["adjde_gap"] = team_df["Torvik_adjde"] - team_df["TRank_adjde"]
    if "Torvik_sos" in team_df.columns and "TRank_sos" in team_df.columns:
        team_df["sos_gap"] = team_df["Torvik_sos"] - team_df["TRank_sos"]
    if "Torvik_adjt" in team_df.columns and "TRank_tempo" in team_df.columns:
        team_df["tempo_gap"] = team_df["Torvik_adjt"] - team_df["TRank_tempo"]
    if "Torvik_barthag" in team_df.columns and "Elo" in team_df.columns:
        elo_norm = 1 / (1 + 10 ** ((1500 - team_df["Elo"]) / 400))
        team_df["elo_barthag_gap"] = elo_norm - team_df["Torvik_barthag"]

    return team_df


def build_v7_team_features(data, season, gender):
    """V7 custom features — PCA ordinals, same as current research.py."""
    from marchmadness.features import torvik

    if gender == "W":
        team_df = seeds.compute(data, season, gender)
        if team_df.empty:
            return team_df
        elo_df = elo.compute(data, season, gender)
        if not elo_df.empty:
            team_df = team_df.merge(elo_df, on="TeamID", how="left")
        trank_df = compute_trank(data, season, gender)
        if not trank_df.empty:
            team_df = team_df.merge(trank_df[["TeamID", "TRank_barthag", "TRank_adjoe", "TRank_adjde", "TRank_tempo"]], on="TeamID", how="left")
        stats_df = season_stats.compute(data, season, gender)
        if not stats_df.empty:
            team_df = team_df.merge(stats_df[["TeamID", "PointDiff"]], on="TeamID", how="left")
        return team_df

    # Men's v7: PCA ordinals
    team_df = seeds.compute(data, season, gender)
    if team_df.empty:
        return team_df

    ord_key = f"{gender}MasseyOrdinals"
    if ord_key in data:
        from sklearn.decomposition import PCA
        all_ord = data[ord_key]
        season_ord = all_ord[
            (all_ord["Season"] == season) &
            (all_ord["RankingDayNum"] >= 133)
        ]
        if not season_ord.empty:
            sys_counts = season_ord.groupby("SystemName")["TeamID"].nunique()
            good_sys = sys_counts[sys_counts >= 250].index.tolist()
            latest = season_ord[season_ord["SystemName"].isin(good_sys)]
            latest = latest.loc[latest.groupby(["TeamID", "SystemName"])["RankingDayNum"].idxmax()]
            pivot = latest.pivot(index="TeamID", columns="SystemName", values="OrdinalRank")
            pivot = pivot.fillna(pivot.median())
            if pivot.shape[1] >= 5:
                n_components = min(2, pivot.shape[1])
                pca = PCA(n_components=n_components)
                comps = pca.fit_transform(pivot.values)
                pca_df = pd.DataFrame({"TeamID": pivot.index})
                for i in range(n_components):
                    pca_df[f"OrdPCA_{i}"] = comps[:, i]
                team_df = team_df.merge(pca_df, on="TeamID", how="left")

    from marchmadness.features import torvik as torvik_mod
    torvik_df = torvik_mod.compute(data, season, gender)
    if not torvik_df.empty:
        team_df = team_df.merge(torvik_df, on="TeamID", how="left")

    elo_df = elo.compute(data, season, gender)
    if not elo_df.empty:
        team_df = team_df.merge(elo_df, on="TeamID", how="left")

    trank_df = compute_trank(data, season, gender)
    if not trank_df.empty:
        team_df = team_df.merge(trank_df[["TeamID", "TRank_barthag", "TRank_adjoe", "TRank_adjde", "TRank_sos", "TRank_tempo"]], on="TeamID", how="left")

    if "Torvik_barthag" in team_df.columns and "TRank_barthag" in team_df.columns:
        team_df["barthag_gap"] = team_df["Torvik_barthag"] - team_df["TRank_barthag"]
    if "Torvik_adjoe" in team_df.columns and "TRank_adjoe" in team_df.columns:
        team_df["adjoe_gap"] = team_df["Torvik_adjoe"] - team_df["TRank_adjoe"]
    if "Torvik_adjde" in team_df.columns and "TRank_adjde" in team_df.columns:
        team_df["adjde_gap"] = team_df["Torvik_adjde"] - team_df["TRank_adjde"]
    if "Torvik_sos" in team_df.columns and "TRank_sos" in team_df.columns:
        team_df["sos_gap"] = team_df["Torvik_sos"] - team_df["TRank_sos"]
    if "Torvik_adjt" in team_df.columns and "TRank_tempo" in team_df.columns:
        team_df["tempo_gap"] = team_df["Torvik_adjt"] - team_df["TRank_tempo"]
    if "Torvik_barthag" in team_df.columns and "Elo" in team_df.columns:
        elo_norm = 1 / (1 + 10 ** ((1500 - team_df["Elo"]) / 400))
        team_df["elo_barthag_gap"] = elo_norm - team_df["Torvik_barthag"]

    return team_df


def build_training(data, seasons, gender, build_fn):
    """Build training data using a custom feature builder."""
    tourney_key = f"{gender}NCAATourneyCompactResults"
    tourney = data[tourney_key]
    seed_key = f"{gender}NCAATourneySeeds"
    seed_data = data[seed_key]

    all_rows = []
    for season in seasons:
        team_features = build_fn(data, season, gender)
        if team_features.empty:
            continue

        # Get seeds for this season
        season_seeds = seed_data[seed_data["Season"] == season]
        seed_map = {}
        for _, row in season_seeds.iterrows():
            seed_str = row["Seed"]
            seed_num = int(seed_str[1:3])
            seed_map[row["TeamID"]] = seed_num

        season_tourney = tourney[tourney["Season"] == season]
        for _, game in season_tourney.iterrows():
            w_id, l_id = game["WTeamID"], game["LTeamID"]
            team_a, team_b = min(w_id, l_id), max(w_id, l_id)
            label = 1 if w_id == team_a else 0

            matchup = build_matchup_features(team_features, team_a, team_b)
            if not matchup:
                continue

            matchup["Season"] = season
            matchup["Label"] = label
            matchup["DayNum"] = game["DayNum"]
            matchup["SampleWeight"] = 6.0
            matchup["seed_a_raw"] = seed_map.get(team_a, np.nan)
            matchup["seed_b_raw"] = seed_map.get(team_b, np.nan)
            matchup["TeamA"] = team_a
            matchup["TeamB"] = team_b
            all_rows.append(matchup)

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def prep_data(df):
    """Prepare feature matrix."""
    feat_cols = [c for c in df.columns if c not in
                 {"Season", "Label", "DayNum", "SampleWeight", "TeamA", "TeamB",
                  "seed_a_raw", "seed_b_raw"}]
    X = df[feat_cols].values.copy()
    y = df["Label"].values.copy()
    seasons = df["Season"].values.copy()
    med = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = med[i] if not np.isnan(med[i]) else 0
    return X, y, seasons, feat_cols, med


def run_cv_with_details(df, model_fn, blend_configs=None):
    """Run leave-season-out CV returning OOF predictions + feature importance.

    If blend_configs is provided, it's a list of (model_fn, weight) tuples
    and df is the same for all models.
    """
    feat_cols = [c for c in df.columns if c not in
                 {"Season", "Label", "DayNum", "SampleWeight", "TeamA", "TeamB",
                  "seed_a_raw", "seed_b_raw"}]
    X = df[feat_cols].values.copy()
    y = df["Label"].values.copy()
    seasons = df["Season"].values.copy()
    med = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = med[i] if not np.isnan(med[i]) else 0

    oof_preds = np.full(len(y), np.nan)
    all_importances = defaultdict(list)

    for val_season in CV_SEASONS:
        train_mask = seasons != val_season
        val_mask = seasons == val_season
        if val_mask.sum() == 0:
            continue

        if blend_configs:
            fold_pred = np.zeros(val_mask.sum())
            for m_fn, weight in blend_configs:
                model = m_fn()
                model.fit(X[train_mask], y[train_mask])
                preds = model.predict_proba(X[val_mask])[:, 1]
                fold_pred += weight * preds

                # Extract feature importance
                extract_importance(model, feat_cols, all_importances, weight)
            oof_preds[val_mask] = np.clip(fold_pred, *PREDICTION_CLIP)
        else:
            model = model_fn()
            model.fit(X[train_mask], y[train_mask])
            preds = model.predict_proba(X[val_mask])[:, 1]
            oof_preds[val_mask] = np.clip(preds, *PREDICTION_CLIP)
            extract_importance(model, feat_cols, all_importances, 1.0)

    # Average importances across folds
    avg_imp = {}
    for feat, vals in all_importances.items():
        avg_imp[feat] = np.mean(vals)

    return oof_preds, y, seasons, df, feat_cols, avg_imp


def extract_importance(model, feat_cols, all_importances, weight):
    """Extract feature importance from a model (LGB or Pipeline with LR)."""
    if isinstance(model, LGBMClassifier):
        imp = model.feature_importances_
        for j, col in enumerate(feat_cols):
            all_importances[col].append(imp[j] * weight)
    elif isinstance(model, Pipeline):
        # Get the logistic regression step
        lr = None
        for name, step in model.steps:
            if isinstance(step, LogisticRegression):
                lr = step
                break
        if lr is not None and hasattr(lr, 'coef_'):
            coefs = np.abs(lr.coef_[0])
            for j, col in enumerate(feat_cols):
                all_importances[col].append(coefs[j] * weight)


# ============================================================
# Analysis functions
# ============================================================

def seed_matchup_calibration(oof_preds, y, df, label):
    """Detailed calibration by seed matchup."""
    seed_a = df["seed_a_raw"].values
    seed_b = df["seed_b_raw"].values
    valid = ~np.isnan(oof_preds)

    matchups = [(1,16), (2,15), (3,14), (4,13), (5,12), (6,11), (7,10), (8,9)]
    rows = []

    for s_high, s_low in matchups:
        mask = valid & (
            ((seed_a == s_high) & (seed_b == s_low)) |
            ((seed_a == s_low) & (seed_b == s_high))
        )
        if mask.sum() == 0:
            continue

        m_preds = oof_preds[mask]
        m_labels = y[mask]

        # For consistency: pred of favorite winning
        # "favorite" = lower seed number
        # In our encoding: if seed_a < seed_b, then favorite = A, pred of fav = pred
        # if seed_a > seed_b, then favorite = B, pred of fav = 1 - pred
        fav_preds = []
        fav_won = []
        for idx in np.where(mask)[0]:
            sa, sb = seed_a[idx], seed_b[idx]
            p = oof_preds[idx]
            lab = y[idx]
            if sa < sb:  # A is favorite
                fav_preds.append(p)
                fav_won.append(lab)
            else:  # B is favorite
                fav_preds.append(1 - p)
                fav_won.append(1 - lab)

        fav_preds = np.array(fav_preds)
        fav_won = np.array(fav_won)
        actual_fav_rate = fav_won.mean()
        pred_fav_rate = fav_preds.mean()
        n_upsets = int((fav_won == 0).sum())

        rows.append({
            "matchup": f"{s_high}v{s_low}",
            "n": int(mask.sum()),
            "actual_fav_win%": f"{actual_fav_rate:.1%}",
            "pred_fav_win%": f"{pred_fav_rate:.1%}",
            "cal_error": f"{abs(pred_fav_rate - actual_fav_rate):.3f}",
            "cal_direction": "overconf" if pred_fav_rate > actual_fav_rate else "underconf",
            "upsets": n_upsets,
            "upset_rate": f"{n_upsets/mask.sum():.1%}",
            "brier": f"{brier_score(m_labels, m_preds):.4f}",
            "pred_mean": f"{fav_preds.mean():.3f}",
            "pred_std": f"{fav_preds.std():.3f}",
            "pred_min": f"{fav_preds.min():.3f}",
            "pred_max": f"{fav_preds.max():.3f}",
        })

    return rows


def upset_deep_dive(oof_preds, y, df, seasons):
    """Detailed upset analysis: which upsets were caught, which were missed."""
    seed_a = df["seed_a_raw"].values
    seed_b = df["seed_b_raw"].values
    valid = ~np.isnan(oof_preds)

    upset_details = []
    for idx in np.where(valid)[0]:
        sa, sb = seed_a[idx], seed_b[idx]
        if np.isnan(sa) or np.isnan(sb):
            continue
        seed_diff = abs(sa - sb)
        if seed_diff < 3:
            continue

        p = oof_preds[idx]
        lab = y[idx]
        season = seasons[idx]

        if sa < sb:  # A is favorite
            fav_pred = p
            fav_won = lab == 1
        else:
            fav_pred = 1 - p
            fav_won = lab == 0

        is_upset = not fav_won

        if is_upset:
            upset_details.append({
                "season": int(season),
                "matchup": f"{int(min(sa,sb))}v{int(max(sa,sb))}",
                "fav_seed": int(min(sa, sb)),
                "dog_seed": int(max(sa, sb)),
                "pred_fav": f"{fav_pred:.3f}",
                "predicted_upset": fav_pred < 0.5,
                "team_a": int(df.iloc[idx]["TeamA"]) if "TeamA" in df.columns else "?",
                "team_b": int(df.iloc[idx]["TeamB"]) if "TeamB" in df.columns else "?",
            })

    return upset_details


def calibration_by_confidence(oof_preds, y):
    """Calibration analysis binned by prediction confidence."""
    valid = ~np.isnan(oof_preds)
    p = oof_preds[valid]
    t = y[valid]

    # Use 20 bins for finer granularity
    ece, bins = calibration_error(t, p, n_bins=20)

    rows = []
    for b in bins:
        if b["count"] == 0:
            continue
        rows.append({
            "bin": f"{b['bin_center']:.2f}",
            "n": b["count"],
            "mean_pred": f"{b['mean_pred']:.3f}",
            "actual_rate": f"{b['actual_rate']:.3f}",
            "error": f"{abs(b['mean_pred'] - b['actual_rate']):.3f}",
        })

    return ece, rows


def five_twelve_deep_dive(oof_preds, y, df, seasons, label):
    """Deep dive into 5v12 matchups specifically."""
    seed_a = df["seed_a_raw"].values
    seed_b = df["seed_b_raw"].values
    valid = ~np.isnan(oof_preds)

    mask = valid & (
        ((seed_a == 5) & (seed_b == 12)) |
        ((seed_a == 12) & (seed_b == 5))
    )
    if mask.sum() == 0:
        return []

    games = []
    for idx in np.where(mask)[0]:
        sa, sb = seed_a[idx], seed_b[idx]
        p = oof_preds[idx]
        lab = y[idx]
        season = int(seasons[idx])

        if sa < sb:  # 5-seed is A
            fav_pred = p
            fav_won = lab == 1
        else:
            fav_pred = 1 - p
            fav_won = lab == 0

        games.append({
            "season": season,
            "team_a": int(df.iloc[idx]["TeamA"]) if "TeamA" in df.columns else "?",
            "team_b": int(df.iloc[idx]["TeamB"]) if "TeamB" in df.columns else "?",
            "5seed_team": int(df.iloc[idx]["TeamA"] if sa == 5 else df.iloc[idx]["TeamB"]) if "TeamA" in df.columns else "?",
            "12seed_team": int(df.iloc[idx]["TeamA"] if sa == 12 else df.iloc[idx]["TeamB"]) if "TeamA" in df.columns else "?",
            "pred_5seed": f"{fav_pred:.3f}",
            "5seed_won": fav_won,
            "upset": not fav_won,
        })

    return games


# ============================================================
# Main
# ============================================================

def main():
    print("Loading data...")
    data = load_all()

    # Load team names for readable output
    teams_m = data.get("MTeams", pd.DataFrame())
    teams_w = data.get("WTeams", pd.DataFrame())
    team_names = {}
    for _, row in teams_m.iterrows():
        team_names[row["TeamID"]] = row["TeamName"]
    for _, row in teams_w.iterrows():
        team_names[row["TeamID"]] = row["TeamName"]

    report = StringIO()
    def w(line=""):
        report.write(line + "\n")

    w("# Calibration, Upset & Feature Importance Analysis: v5 vs v7")
    w()
    w("Generated 2026-03-18. Leave-season-out CV on 2022-2025 tournaments.")
    w()

    for gender, gender_label in [("M", "Men's"), ("W", "Women's")]:
        w(f"---")
        w(f"# {gender_label}")
        w()

        # Build data for v5 and v7
        if gender == "M":
            start_year = 2015
        else:
            start_year = 2003
        seasons_list = list(range(start_year, 2026))

        print(f"Building {gender_label} v5 data...")
        v5_df = build_training(data, seasons_list, gender, build_v5_team_features)
        print(f"Building {gender_label} v7 data...")
        v7_df = build_training(data, seasons_list, gender, build_v7_team_features)

        # Define model configs
        if gender == "M":
            # v5: 55% LGB + 45% L2 LR
            v5_blend = [
                (lambda: LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.025,
                                        subsample=0.8, colsample_bytree=0.6, min_split_gain=0.005,
                                        random_state=42, verbose=-1), 0.55),
                (lambda: Pipeline([("s", StandardScaler()),
                                   ("lr", LogisticRegression(C=100.0, penalty='l2', solver='lbfgs', max_iter=2000))]), 0.45),
            ]
            # v7: 40% LGB + 60% L2 LR
            v7_blend = [
                (lambda: LGBMClassifier(n_estimators=250, max_depth=3, learning_rate=0.025,
                                        subsample=0.8, colsample_bytree=0.6, min_split_gain=0.005,
                                        random_state=42, verbose=-1), 0.40),
                (lambda: Pipeline([("s", StandardScaler()),
                                   ("lr", LogisticRegression(C=100.0, penalty='l2', solver='lbfgs', max_iter=2000))]), 0.60),
            ]
            print(f"  Running v5 CV (blend)...")
            v5_preds, v5_y, v5_seasons, v5_full, v5_feats, v5_imp = run_cv_with_details(v5_df, None, v5_blend)
            print(f"  Running v7 CV (blend)...")
            v7_preds, v7_y, v7_seasons, v7_full, v7_feats, v7_imp = run_cv_with_details(v7_df, None, v7_blend)
        else:
            # v5: L1 LR C=0.15
            v5_model = lambda: Pipeline([("s", StandardScaler()),
                                          ("lr", LogisticRegression(C=0.15, penalty='l1', solver='liblinear', max_iter=2000))])
            # v7: QT + L1 LR C=0.20
            v7_model = lambda: Pipeline([("q", QuantileTransformer(output_distribution='normal', n_quantiles=50)),
                                          ("lr", LogisticRegression(C=0.20, penalty='l1', solver='liblinear', max_iter=2000))])
            print(f"  Running v5 CV...")
            v5_preds, v5_y, v5_seasons, v5_full, v5_feats, v5_imp = run_cv_with_details(v5_df, v5_model)
            print(f"  Running v7 CV...")
            v7_preds, v7_y, v7_seasons, v7_full, v7_feats, v7_imp = run_cv_with_details(v7_df, v7_model)

        # Overall scores
        v5_valid = ~np.isnan(v5_preds)
        v7_valid = ~np.isnan(v7_preds)
        v5_brier = brier_score(v5_y[v5_valid], v5_preds[v5_valid])
        v7_brier = brier_score(v7_y[v7_valid], v7_preds[v7_valid])

        w(f"## Overall Scores")
        w(f"| Model | Brier | Accuracy | N |")
        w(f"|-------|-------|----------|---|")
        v5_acc = compute_accuracy(v5_y[v5_valid], v5_preds[v5_valid])
        v7_acc = compute_accuracy(v7_y[v7_valid], v7_preds[v7_valid])
        w(f"| v5 | {v5_brier:.4f} | {v5_acc:.1%} | {v5_valid.sum()} |")
        w(f"| v7 | {v7_brier:.4f} | {v7_acc:.1%} | {v7_valid.sum()} |")
        w(f"| delta | {v7_brier - v5_brier:+.4f} | {v7_acc - v5_acc:+.1%} | |")
        w()

        # Per-season breakdown
        w(f"## Per-Season Breakdown")
        w(f"| Season | v5 Brier | v7 Brier | Delta | v5 Acc | v7 Acc |")
        w(f"|--------|----------|----------|-------|--------|--------|")
        for s in CV_SEASONS:
            s5_mask = v5_valid & (v5_seasons == s)
            s7_mask = v7_valid & (v7_seasons == s)
            if s5_mask.sum() == 0 or s7_mask.sum() == 0:
                continue
            b5 = brier_score(v5_y[s5_mask], v5_preds[s5_mask])
            b7 = brier_score(v7_y[s7_mask], v7_preds[s7_mask])
            a5 = compute_accuracy(v5_y[s5_mask], v5_preds[s5_mask])
            a7 = compute_accuracy(v7_y[s7_mask], v7_preds[s7_mask])
            w(f"| {s} | {b5:.4f} | {b7:.4f} | {b7-b5:+.4f} | {a5:.1%} | {a7:.1%} |")
        w()

        # ============================================================
        # CALIBRATION BY SEED MATCHUP
        # ============================================================
        w(f"## Calibration by Seed Matchup (First Round)")
        w()
        w(f"Pred fav win% = model's average predicted probability that the favorite wins.")
        w(f"Actual fav win% = how often the favorite actually won in CV data.")
        w(f"Calibration error = |predicted - actual|. Direction: overconf = model too confident in favorite, underconf = model doesn't trust favorite enough.")
        w()

        w(f"### v5")
        v5_cal = seed_matchup_calibration(v5_preds, v5_y, v5_df, "v5")
        w(f"| Matchup | N | Pred Fav% | Actual Fav% | Cal Error | Direction | Upsets | Upset% | Brier |")
        w(f"|---------|---|-----------|-------------|-----------|-----------|--------|--------|-------|")
        for r in v5_cal:
            w(f"| {r['matchup']} | {r['n']} | {r['pred_fav_win%']} | {r['actual_fav_win%']} | {r['cal_error']} | {r['cal_direction']} | {r['upsets']} | {r['upset_rate']} | {r['brier']} |")
        w()

        w(f"### v7")
        v7_cal = seed_matchup_calibration(v7_preds, v7_y, v7_df, "v7")
        w(f"| Matchup | N | Pred Fav% | Actual Fav% | Cal Error | Direction | Upsets | Upset% | Brier |")
        w(f"|---------|---|-----------|-------------|-----------|-----------|--------|--------|-------|")
        for r in v7_cal:
            w(f"| {r['matchup']} | {r['n']} | {r['pred_fav_win%']} | {r['actual_fav_win%']} | {r['cal_error']} | {r['cal_direction']} | {r['upsets']} | {r['upset_rate']} | {r['brier']} |")
        w()

        # Side-by-side comparison
        w(f"### Calibration Comparison (v5 vs v7)")
        v5_map = {r['matchup']: r for r in v5_cal}
        v7_map = {r['matchup']: r for r in v7_cal}
        w(f"| Matchup | v5 Cal Error | v7 Cal Error | v5 Dir | v7 Dir | v5 Brier | v7 Brier | Better |")
        w(f"|---------|-------------|-------------|--------|--------|----------|----------|--------|")
        for mu in ["1v16", "2v15", "3v14", "4v13", "5v12", "6v11", "7v10", "8v9"]:
            if mu in v5_map and mu in v7_map:
                r5, r7 = v5_map[mu], v7_map[mu]
                better = "v7" if float(r7['brier']) < float(r5['brier']) else "v5" if float(r5['brier']) < float(r7['brier']) else "tie"
                w(f"| {mu} | {r5['cal_error']} | {r7['cal_error']} | {r5['cal_direction']} | {r7['cal_direction']} | {r5['brier']} | {r7['brier']} | {better} |")
        w()

        # ============================================================
        # 5v12 DEEP DIVE
        # ============================================================
        w(f"## 5v12 Matchup Deep Dive")
        w()
        w(f"5-12 matchups are historically the most volatile first-round pairing (upsets ~35% of the time).")
        w()

        w(f"### v5 game-by-game predictions")
        v5_512 = five_twelve_deep_dive(v5_preds, v5_y, v5_df, v5_seasons, "v5")
        w(f"| Season | 5-seed | 12-seed | Pred(5wins) | 5won? | Upset? |")
        w(f"|--------|--------|---------|-------------|-------|--------|")
        for g in sorted(v5_512, key=lambda x: x["season"]):
            s5 = team_names.get(g["5seed_team"], str(g["5seed_team"]))
            s12 = team_names.get(g["12seed_team"], str(g["12seed_team"]))
            w(f"| {g['season']} | {s5} | {s12} | {g['pred_5seed']} | {'Y' if g['5seed_won'] else 'N'} | {'**YES**' if g['upset'] else 'no'} |")
        w()

        w(f"### v7 game-by-game predictions")
        v7_512 = five_twelve_deep_dive(v7_preds, v7_y, v7_df, v7_seasons, "v7")
        w(f"| Season | 5-seed | 12-seed | Pred(5wins) | 5won? | Upset? |")
        w(f"|--------|--------|---------|-------------|-------|--------|")
        for g in sorted(v7_512, key=lambda x: x["season"]):
            s5 = team_names.get(g["5seed_team"], str(g["5seed_team"]))
            s12 = team_names.get(g["12seed_team"], str(g["12seed_team"]))
            w(f"| {g['season']} | {s5} | {s12} | {g['pred_5seed']} | {'Y' if g['5seed_won'] else 'N'} | {'**YES**' if g['upset'] else 'no'} |")
        w()

        # Compare v5 vs v7 on specific 5-12 games
        if v5_512 and v7_512:
            w(f"### v5 vs v7 prediction comparison on 5-12 games")
            # Match by season + teams
            v5_by_key = {(g["season"], g["5seed_team"], g["12seed_team"]): g for g in v5_512}
            v7_by_key = {(g["season"], g["5seed_team"], g["12seed_team"]): g for g in v7_512}
            common = set(v5_by_key.keys()) & set(v7_by_key.keys())
            if common:
                w(f"| Season | 5-seed | 12-seed | v5 Pred | v7 Pred | Delta | Result | Comment |")
                w(f"|--------|--------|---------|---------|---------|-------|--------|---------|")
                for key in sorted(common):
                    g5, g7 = v5_by_key[key], v7_by_key[key]
                    s5 = team_names.get(key[1], str(key[1]))
                    s12 = team_names.get(key[2], str(key[2]))
                    p5 = float(g5["pred_5seed"])
                    p7 = float(g7["pred_5seed"])
                    delta = p7 - p5
                    result = "5-seed won" if g5["5seed_won"] else "**12-SEED UPSET**"
                    if g5["upset"]:
                        if p7 < p5:
                            comment = "v7 better (lower fav confidence)"
                        else:
                            comment = "v5 better (lower fav confidence)"
                    else:
                        if p7 > p5:
                            comment = "v7 better (higher fav confidence)"
                        else:
                            comment = "v5 better (higher fav confidence)"
                    w(f"| {key[0]} | {s5} | {s12} | {p5:.3f} | {p7:.3f} | {delta:+.3f} | {result} | {comment} |")
                w()

        # ============================================================
        # OVERALL CALIBRATION CURVES
        # ============================================================
        w(f"## Calibration Curves (20-bin)")
        w()

        v5_ece, v5_bins = calibration_by_confidence(v5_preds, v5_y)
        v7_ece, v7_bins = calibration_by_confidence(v7_preds, v7_y)

        w(f"ECE: v5={v5_ece:.4f}, v7={v7_ece:.4f}")
        w()

        w(f"### v5 calibration")
        w(f"| Bin | N | Mean Pred | Actual Rate | Error |")
        w(f"|-----|---|-----------|-------------|-------|")
        for r in v5_bins:
            w(f"| {r['bin']} | {r['n']} | {r['mean_pred']} | {r['actual_rate']} | {r['error']} |")
        w()

        w(f"### v7 calibration")
        w(f"| Bin | N | Mean Pred | Actual Rate | Error |")
        w(f"|-----|---|-----------|-------------|-------|")
        for r in v7_bins:
            w(f"| {r['bin']} | {r['n']} | {r['mean_pred']} | {r['actual_rate']} | {r['error']} |")
        w()

        # ============================================================
        # UPSET ANALYSIS
        # ============================================================
        w(f"## Upset Analysis (seed diff >= 3)")
        w()

        v5_upsets = upset_deep_dive(v5_preds, v5_y, v5_df, v5_seasons)
        v7_upsets = upset_deep_dive(v7_preds, v7_y, v7_df, v7_seasons)

        v5_caught = sum(1 for u in v5_upsets if u["predicted_upset"])
        v7_caught = sum(1 for u in v7_upsets if u["predicted_upset"])

        w(f"| Model | Total Upsets | Predicted Correctly | Detection Rate |")
        w(f"|-------|-------------|--------------------|--------------------|")
        w(f"| v5 | {len(v5_upsets)} | {v5_caught} | {v5_caught/len(v5_upsets):.1%} |")
        w(f"| v7 | {len(v7_upsets)} | {v7_caught} | {v7_caught/len(v7_upsets):.1%} |")
        w()

        w(f"### Upsets by seed matchup")
        from collections import Counter
        v5_by_mu = Counter(u["matchup"] for u in v5_upsets)
        v7_by_mu = Counter(u["matchup"] for u in v7_upsets)
        v5_caught_by = Counter(u["matchup"] for u in v5_upsets if u["predicted_upset"])
        v7_caught_by = Counter(u["matchup"] for u in v7_upsets if u["predicted_upset"])
        all_mu = sorted(set(list(v5_by_mu.keys()) + list(v7_by_mu.keys())),
                        key=lambda x: int(x.split("v")[0]))
        w(f"| Matchup | v5 Upsets | v5 Caught | v7 Upsets | v7 Caught |")
        w(f"|---------|----------|-----------|----------|-----------|")
        for mu in all_mu:
            w(f"| {mu} | {v5_by_mu.get(mu,0)} | {v5_caught_by.get(mu,0)} | {v7_by_mu.get(mu,0)} | {v7_caught_by.get(mu,0)} |")
        w()

        # ============================================================
        # FEATURE IMPORTANCE
        # ============================================================
        w(f"## Feature Importance")
        w()

        w(f"### v5 features (importance = avg across CV folds)")
        w(f"For blended models: LGB uses split-based importance weighted by blend weight, LR uses |coefficient| weighted by blend weight.")
        w()
        v5_sorted = sorted(v5_imp.items(), key=lambda x: -x[1])
        w(f"| Rank | Feature | Importance |")
        w(f"|------|---------|------------|")
        for rank, (feat, imp) in enumerate(v5_sorted, 1):
            w(f"| {rank} | {feat} | {imp:.4f} |")
        w()

        w(f"### v7 features")
        v7_sorted = sorted(v7_imp.items(), key=lambda x: -x[1])
        w(f"| Rank | Feature | Importance |")
        w(f"|------|---------|------------|")
        for rank, (feat, imp) in enumerate(v7_sorted, 1):
            w(f"| {rank} | {feat} | {imp:.4f} |")
        w()

        # Feature comparison
        w(f"### Feature comparison")
        w()
        v5_set = set(v5_imp.keys())
        v7_set = set(v7_imp.keys())
        only_v5 = v5_set - v7_set
        only_v7 = v7_set - v5_set
        common_feats = v5_set & v7_set

        if only_v5:
            w(f"**Features only in v5**: {', '.join(sorted(only_v5))}")
        if only_v7:
            w(f"**Features only in v7**: {', '.join(sorted(only_v7))}")
        w()

        if common_feats:
            w(f"**Importance shift for common features (v7 - v5)**:")
            w(f"| Feature | v5 Imp | v7 Imp | Delta | Direction |")
            w(f"|---------|--------|--------|-------|-----------|")
            shifts = []
            for f in common_feats:
                delta = v7_imp[f] - v5_imp[f]
                shifts.append((f, v5_imp[f], v7_imp[f], delta))
            shifts.sort(key=lambda x: -abs(x[3]))
            for feat, v5i, v7i, delta in shifts:
                direction = "increased" if delta > 0 else "decreased"
                w(f"| {feat} | {v5i:.4f} | {v7i:.4f} | {delta:+.4f} | {direction} |")
            w()

    # Write report
    output_path = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "calibration_feature_analysis.md"
    output_path.write_text(report.getvalue(), encoding="utf-8")
    print(f"\nReport written to {output_path}")
    print(report.getvalue()[:2000])


if __name__ == "__main__":
    main()
