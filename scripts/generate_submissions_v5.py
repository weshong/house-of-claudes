"""Generate submissions v5 — current best configs.

Men's:  55% LGB d=3 + 45% L2 LR C=100, custom features (ordinals+Torvik+TRank+gaps) — Brier 0.1307
Women's: seeds + Elo + TRank clone, L1 LR C=0.15 — Brier 0.1365
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marchmadness.config import SUBMISSIONS_DIR, PREDICTION_CLIP
from marchmadness.data_loader import load_all
from marchmadness.features.builder import build_matchup_features, get_feature_columns
from marchmadness.submission import validate_submission

# Reuse the custom feature builder from research.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from research import build_custom_team_features, build_custom_training_data


def train_custom(data, gender, model_obj, start_year):
    """Train a model on custom features. Returns (model, feature_cols, medians)."""
    seasons = list(range(start_year, 2026))
    df = build_custom_training_data(data, seasons, gender)
    fc = get_feature_columns(df)
    X = df[fc].values.copy()
    y = df["Label"].values.copy()
    med = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        if mask.any():
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

    # --- Men's: 55% LGB + 45% LR blend on custom features ---
    print("  Training Men's LGB d=3...")
    m_lgb, m_fc, m_med = train_custom(
        data, "M",
        LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.025,
                       subsample=0.8, colsample_bytree=0.6, min_split_gain=0.005,
                       random_state=42, verbose=-1),
        start_year=2015
    )
    print("  Training Men's LR C=100...")
    m_lr, m_lr_fc, m_lr_med = train_custom(
        data, "M",
        Pipeline([("s", StandardScaler()),
                  ("lr", LogisticRegression(C=100.0, penalty='l2', solver='lbfgs', max_iter=2000))]),
        start_year=2015
    )

    # --- Women's: L1 LR C=0.15 on custom features ---
    print("  Training Women's LR L1 C=0.15...")
    w_lr, w_fc, w_med = train_custom(
        data, "W",
        Pipeline([("s", StandardScaler()),
                  ("lr", LogisticRegression(C=0.15, penalty='l1', solver='liblinear', max_iter=2000))]),
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

        ck = (season, gender)
        if ck not in tf_cache:
            tf_cache[ck] = build_custom_team_features(data, season, gender)

        tf = tf_cache[ck]

        if gender == "M":
            p_lgb = predict_one(tf, ta, tb, m_fc, m_lgb, m_med)
            p_lr = predict_one(tf, ta, tb, m_lr_fc, m_lr, m_lr_med)
            pred = 0.55 * p_lgb + 0.45 * p_lr
        else:
            pred = predict_one(tf, ta, tb, w_fc, w_lr, w_med)

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
    path = SUBMISSIONS_DIR / f"submission_stage2_v5_lgb55-lr45-blend_l1w_{ts}.csv"
    submission.to_csv(path, index=False)
    print(f"  Saved: {path}")
    print(f"  Rows={len(submission)}, Mean={submission['Pred'].mean():.4f}, Std={submission['Pred'].std():.4f}")

    # Stats for seeded matchups only
    seeded = submission[submission["Pred"] != 0.5]
    print(f"  Seeded matchups: {len(seeded)}, Mean={seeded['Pred'].mean():.4f}, Std={seeded['Pred'].std():.4f}")

    return path


if __name__ == "__main__":
    print("Loading data...")
    data = load_all()
    print("\n=== Stage 2 ===")
    s2 = generate_submission(data, 2)
    print(f"\nDone!\n  {s2}")
