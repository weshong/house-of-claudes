"""Experiment 13: T-Rank clone with Bayesian optimization.

Tests the full T-Rank clone for both men's and women's, then uses
Bayesian optimization (Optuna) to tune the iterative efficiency
hyperparameters end-to-end on Brier score.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marchmadness.config import CV_SEASONS
from marchmadness.data_loader import load_all
from marchmadness.features.builder import (
    build_matchup_features, get_feature_columns
)
from marchmadness.features import seeds, ordinals
from marchmadness.features.trank_clone import compute as compute_trank
from marchmadness.evaluation.cross_val import leave_season_out_cv
from marchmadness.evaluation.metrics import brier_score
from lightgbm import LGBMClassifier


def build_trank_team_features(data, season, gender, trank_kwargs=None):
    """Build team features with T-Rank clone."""
    team_df = seeds.compute(data, season, gender)
    if team_df.empty:
        return team_df

    ord_df = ordinals.compute(data, season, gender)
    if not ord_df.empty:
        team_df = team_df.merge(ord_df, on="TeamID", how="left")

    kwargs = trank_kwargs or {}
    trank_df = compute_trank(data, season, gender, **kwargs)
    if not trank_df.empty:
        team_df = team_df.merge(trank_df, on="TeamID", how="left")

    return team_df


def build_trank_training_data(data, seasons, gender, trank_kwargs=None):
    """Build training data with T-Rank features."""
    tourney_key = f"{gender}NCAATourneyCompactResults"
    tourney = data[tourney_key]

    all_rows = []
    for season in seasons:
        team_features = build_trank_team_features(data, season, gender, trank_kwargs)
        if team_features.empty:
            continue

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
            all_rows.append(matchup)

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def prep_data(df):
    feat_cols = get_feature_columns(df)
    X = df[feat_cols].values.copy()
    y = df["Label"].values.copy()
    seasons = df["Season"].values.copy()
    med = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = med[i] if not np.isnan(med[i]) else 0
    return X, y, seasons, feat_cols


def eval_config(data, gender, seasons, trank_kwargs, model_fn, label=""):
    """Evaluate a T-Rank config with leave-season-out CV."""
    df = build_trank_training_data(data, seasons, gender, trank_kwargs)
    if df.empty:
        print(f"  {label}: NO DATA")
        return None, float('inf')

    X, y, s, fc = prep_data(df)
    model = model_fn()
    result = leave_season_out_cv(model, X, y, s, CV_SEASONS)
    brier = result["overall"]["brier_score"]
    print(f"  {label}: Brier={brier:.4f} ({len(fc)} features)")
    return result, brier


def main():
    print("Loading data...")
    data = load_all()

    # ============================================================
    # Part 1: T-Rank clone with default params
    # ============================================================
    print(f"\n{'='*70}")
    print("  PART 1: T-Rank Clone - Default Parameters")
    print(f"{'='*70}")

    # Women's
    print("\n--- Women's T-Rank clone ---")
    w_seasons = list(range(2010, 2026))
    for c_val in [0.01, 0.1, 0.5, 1.0]:
        eval_config(data, "W", w_seasons, {},
                    lambda c=c_val: Pipeline([("s", StandardScaler()),
                                              ("lr", LogisticRegression(C=c, max_iter=1000))]),
                    f"W TRank LR C={c_val}")

    eval_config(data, "W", w_seasons, {},
                lambda: LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
                "W TRank LGB d=3")

    # Men's (for comparison with real Torvik)
    print("\n--- Men's T-Rank clone (vs real Torvik @ 0.1543) ---")
    m_seasons = list(range(2003, 2026))
    for c_val in [0.1, 1.0]:
        eval_config(data, "M", m_seasons, {},
                    lambda c=c_val: Pipeline([("s", StandardScaler()),
                                              ("lr", LogisticRegression(C=c, max_iter=1000))]),
                    f"M TRank LR C={c_val}")

    eval_config(data, "M", m_seasons, {},
                lambda: LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
                "M TRank LGB d=3")

    # ============================================================
    # Part 2: Bayesian Optimization of T-Rank hyperparameters
    # ============================================================
    print(f"\n{'='*70}")
    print("  PART 2: Bayesian Optimization (Women's)")
    print(f"{'='*70}")

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  Installing optuna...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "-q"])
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_brier_so_far = float('inf')

    def objective(trial):
        nonlocal best_brier_so_far

        trank_kwargs = {
            "n_iterations": trial.suggest_int("n_iterations", 10, 40),
            "hca_factor": trial.suggest_float("hca_factor", 0.005, 0.025),
            "recency_full_window": trial.suggest_int("recency_full_window", 20, 60),
            "recency_decay_rate": trial.suggest_float("recency_decay_rate", 0.005, 0.02),
            "recency_floor": trial.suggest_float("recency_floor", 0.4, 0.8),
            "blowout_threshold": trial.suggest_int("blowout_threshold", 8, 20),
            "barthag_exponent": trial.suggest_float("barthag_exponent", 8.0, 16.0),
            "preseason_games": trial.suggest_int("preseason_games", 2, 10),
        }

        c_val = trial.suggest_categorical("lr_C", [0.01, 0.05, 0.1, 0.5, 1.0])

        df = build_trank_training_data(data, w_seasons, "W", trank_kwargs)
        if df.empty:
            return 1.0

        X, y, s, fc = prep_data(df)
        model = Pipeline([("s", StandardScaler()),
                          ("lr", LogisticRegression(C=c_val, max_iter=1000))])
        result = leave_season_out_cv(model, X, y, s, CV_SEASONS)
        brier = result["overall"]["brier_score"]

        if brier < best_brier_so_far:
            best_brier_so_far = brier
            print(f"  NEW BEST: Brier={brier:.4f} | "
                  f"hca={trank_kwargs['hca_factor']:.3f} "
                  f"recency_win={trank_kwargs['recency_full_window']} "
                  f"decay={trank_kwargs['recency_decay_rate']:.3f} "
                  f"floor={trank_kwargs['recency_floor']:.2f} "
                  f"blowout={trank_kwargs['blowout_threshold']} "
                  f"barthag_exp={trank_kwargs['barthag_exponent']:.1f} "
                  f"preseason={trank_kwargs['preseason_games']} "
                  f"C={c_val}")

        return brier

    print("\n  Running 50 Optuna trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    print(f"\n  Best Brier: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # Evaluate best params
    best_params = {k: v for k, v in study.best_params.items() if k != "lr_C"}
    best_c = study.best_params["lr_C"]

    print(f"\n--- Best Optuna config, full evaluation ---")
    eval_config(data, "W", w_seasons, best_params,
                lambda: Pipeline([("s", StandardScaler()),
                                  ("lr", LogisticRegression(C=best_c, max_iter=1000))]),
                f"W TRank-Optuna LR C={best_c}")

    # Also try LGB with best T-Rank params
    eval_config(data, "W", w_seasons, best_params,
                lambda: LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
                "W TRank-Optuna LGB d=3")

    # ============================================================
    # Part 3: Blend optimized T-Rank with tier2
    # ============================================================
    print(f"\n{'='*70}")
    print("  PART 3: Blend T-Rank with tier2")
    print(f"{'='*70}")

    from marchmadness.features.builder import build_training_data

    # Collect OOF from T-Rank and tier2
    def collect_oof(model_fn, X, y, seasons):
        preds = np.full(len(y), np.nan)
        for val_season in CV_SEASONS:
            train_mask = seasons != val_season
            val_mask = seasons == val_season
            if val_mask.sum() == 0:
                continue
            model = model_fn()
            model.fit(X[train_mask], y[train_mask])
            preds[val_mask] = model.predict_proba(X[val_mask])[:, 1]
        valid = ~np.isnan(preds)
        return preds[valid], y[valid]

    # T-Rank with best params
    df_trank = build_trank_training_data(data, w_seasons, "W", best_params)
    X_tr, y_tr, s_tr, _ = prep_data(df_trank)
    oof_trank, y_oof_tr = collect_oof(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=best_c, max_iter=1000))]),
        X_tr, y_tr, s_tr)

    # tier2 LR
    df_tier2 = build_training_data(data, list(range(1998, 2026)), "W", "tier2")
    X_t2, y_t2, s_t2, _ = prep_data(df_tier2)
    oof_tier2, y_oof_t2 = collect_oof(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
        X_t2, y_t2, s_t2)

    # iter_eff v1
    df_v1 = build_training_data(data, list(range(2003, 2026)), "W", "iter_eff")
    X_v1, y_v1, s_v1, _ = prep_data(df_v1)
    oof_v1, y_oof_v1 = collect_oof(
        lambda: Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
        X_v1, y_v1, s_v1)

    print(f"  T-Rank OOF: {brier_score(y_oof_tr, oof_trank):.4f}")
    print(f"  tier2 OOF: {brier_score(y_oof_t2, oof_tier2):.4f}")
    print(f"  iter_eff_v1 OOF: {brier_score(y_oof_v1, oof_v1):.4f}")

    # Blend T-Rank with tier2 (need same games — use CV games only from overlapping seasons)
    # T-Rank uses 2010+, tier2 uses 1998+, iter_eff uses 2003+
    # Find CV games in T-Rank set
    trank_cv_mask = np.isin(s_tr, CV_SEASONS)
    tier2_cv_mask = np.isin(s_t2, CV_SEASONS)

    # These should have the same CV games if same seasons covered
    n_tr = trank_cv_mask.sum()
    n_t2 = tier2_cv_mask.sum()
    print(f"  T-Rank CV games: {n_tr}, tier2 CV games: {n_t2}")

    if n_tr == n_t2:
        print("\n  Blend optimization:")
        for w_tr in np.arange(0.0, 1.05, 0.1):
            w_t2 = 1.0 - w_tr
            blend = w_tr * oof_trank + w_t2 * oof_tier2
            b = brier_score(y_oof_tr, blend)
            print(f"    TRank={w_tr:.0%} + tier2={w_t2:.0%}: Brier={b:.4f}")

    print(f"\n  REFERENCE: Women's current best = 0.1387")
    print(f"  REFERENCE: Men's current best = 0.1543")


if __name__ == "__main__":
    main()
