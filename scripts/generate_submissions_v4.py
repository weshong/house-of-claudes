"""Generate submissions v4 — best configs from experiments 9-15.

Men's: Torvik shallow LGB (d=3, n=400, lr=0.02) — Brier 0.1543
Women's: tier2 LR(70%) + iter_eff LR(30%) blend — Brier 0.1386
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marchmadness.config import DATA_DIR, SUBMISSIONS_DIR, PREDICTION_CLIP
from marchmadness.data_loader import load_all
from marchmadness.features.builder import (
    build_team_features, build_matchup_features, get_feature_columns, build_training_data
)
from marchmadness.submission import validate_submission


def train_and_get_medians(data, gender, feature_set, model_obj, start_year):
    """Train a model and return (model, feature_cols, col_medians)."""
    df = build_training_data(data, list(range(start_year, 2026)), gender, feature_set)
    fc = get_feature_columns(df)
    X = df[fc].values.copy()
    y = df["Label"].values.copy()
    med = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = med[i] if not np.isnan(med[i]) else 0
    model_obj.fit(X, y)
    return model_obj, fc, med


def predict_one(tf, team_a, team_b, fc, model, med):
    """Predict P(team_a wins) for a single matchup."""
    matchup = build_matchup_features(tf, team_a, team_b)
    if not matchup:
        return 0.5
    fv = np.array([[matchup.get(c, np.nan) for c in fc]])
    for i in range(fv.shape[1]):
        if np.isnan(fv[0, i]):
            fv[0, i] = med[i] if not np.isnan(med[i]) else 0
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(fv)[:, 1][0])
    return float(model.predict(fv)[0])


def generate_submission(data, stage):
    sample_sub = data[f"SampleSubmissionStage{stage}"]
    parts = sample_sub["ID"].str.split("_", expand=True)
    parsed = pd.DataFrame({
        "ID": sample_sub["ID"],
        "Season": parts[0].astype(int),
        "TeamA": parts[1].astype(int),
        "TeamB": parts[2].astype(int),
    })
    print(f"  Stage {stage}: {len(parsed)} matchups")

    # --- Men's: Torvik shallow LGB ---
    from lightgbm import LGBMClassifier
    print("  Training Men's Torvik LGB d=3...")
    m_lgb, m_lgb_fc, m_lgb_med = train_and_get_medians(
        data, "M", "torvik",
        LGBMClassifier(n_estimators=400, max_depth=3, learning_rate=0.02,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        start_year=2015
    )

    # --- Women's: tier2 LR C=0.1 (70%) + iter_eff LR C=0.1 (30%) ---
    print("  Training Women's models...")
    w_tier2, w_tier2_fc, w_tier2_med = train_and_get_medians(
        data, "W", "tier2",
        Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
        start_year=1998
    )
    w_iter, w_iter_fc, w_iter_med = train_and_get_medians(
        data, "W", "iter_eff",
        Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.1, max_iter=1000))]),
        start_year=2003
    )

    tf_cache = {}
    predictions = {}
    total = len(parsed)

    for i, (_, row) in enumerate(parsed.iterrows()):
        if (i + 1) % 10000 == 0:
            print(f"    {i+1}/{total}")

        ta, tb = row["TeamA"], row["TeamB"]
        season = row["Season"]
        gender = "M" if ta < 3000 else "W"

        if gender == "M":
            ck = (season, "M", "torvik")
            if ck not in tf_cache:
                tf_cache[ck] = build_team_features(data, season, "M", "torvik")
            pred = predict_one(tf_cache[ck], ta, tb, m_lgb_fc, m_lgb, m_lgb_med)
        else:
            # Blend: 70% tier2 + 30% iter_eff
            ck1 = (season, "W", "tier2")
            if ck1 not in tf_cache:
                tf_cache[ck1] = build_team_features(data, season, "W", "tier2")
            p1 = predict_one(tf_cache[ck1], ta, tb, w_tier2_fc, w_tier2, w_tier2_med)

            ck2 = (season, "W", "iter_eff")
            if ck2 not in tf_cache:
                tf_cache[ck2] = build_team_features(data, season, "W", "iter_eff")
            p2 = predict_one(tf_cache[ck2], ta, tb, w_iter_fc, w_iter, w_iter_med)

            pred = 0.70 * p1 + 0.30 * p2

        predictions[row["ID"]] = float(np.clip(pred, *PREDICTION_CLIP))

    submission = pd.DataFrame({
        "ID": sample_sub["ID"],
        "Pred": sample_sub["ID"].map(predictions).fillna(0.5),
    })

    errors = validate_submission(submission, sample_sub)
    if errors:
        print(f"  ERRORS: {errors}")
    else:
        print(f"  Validation OK")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SUBMISSIONS_DIR / f"submission_stage{stage}_v4_{ts}.csv"
    submission.to_csv(path, index=False)
    print(f"  Saved: {path}")
    print(f"  Rows={len(submission)}, Mean={submission['Pred'].mean():.4f}, Std={submission['Pred'].std():.4f}")
    return path


if __name__ == "__main__":
    print("Loading data...")
    data = load_all()
    print("\n=== Stage 2 ===")
    s2 = generate_submission(data, 2)
    print(f"\nDone!\n  {s2}")
