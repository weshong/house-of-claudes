"""Generate submissions using best models from experiments.

Men's: Torvik LR + LGB ensemble (Brier 0.1607)
Women's: tier2 LR C=0.1 (Brier 0.1390)
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


def train_model(data, gender, feature_set, C=1.0, start_year=None):
    """Train a single model on all available data."""
    if start_year is None:
        start_year = 2015 if feature_set == "torvik" else (1985 if gender == "M" else 1998)

    training_df = build_training_data(
        data, list(range(start_year, 2026)), gender, feature_set
    )
    if training_df.empty:
        return None, None, None

    feature_cols = get_feature_columns(training_df)
    X = training_df[feature_cols].values.copy()
    y = training_df["Label"].values.copy()

    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
    ])
    model.fit(X, y)
    return model, feature_cols, col_medians


def train_lgb(data, gender, feature_set, start_year=None):
    """Train LightGBM model."""
    if start_year is None:
        start_year = 2015 if feature_set == "torvik" else (1985 if gender == "M" else 1998)

    training_df = build_training_data(
        data, list(range(start_year, 2026)), gender, feature_set
    )
    if training_df.empty:
        return None, None, None

    feature_cols = get_feature_columns(training_df)
    X = training_df[feature_cols].values.copy()
    y = training_df["Label"].values.copy()

    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0

    from lightgbm import LGBMClassifier
    model = LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    model.fit(X, y)
    return model, feature_cols, col_medians


def predict_matchup(team_features, team_a, team_b, feature_cols, model, col_medians):
    """Predict a single matchup."""
    matchup = build_matchup_features(team_features, team_a, team_b)
    if not matchup:
        return 0.5

    feat_values = np.array([[matchup.get(c, np.nan) for c in feature_cols]])
    # Fill NaN with training medians
    for i in range(feat_values.shape[1]):
        if np.isnan(feat_values[0, i]):
            feat_values[0, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0

    if hasattr(model, "predict_proba"):
        result = model.predict_proba(feat_values)[:, 1][0]
    else:
        result = model.predict(feat_values)[0]

    return float(np.clip(result, *PREDICTION_CLIP))


def generate_submission(data, stage):
    """Generate submission using best models for each gender."""
    sample_key = f"SampleSubmissionStage{stage}"
    sample_sub = data[sample_key]

    parts = sample_sub["ID"].str.split("_", expand=True)
    parsed = pd.DataFrame({
        "ID": sample_sub["ID"],
        "Season": parts[0].astype(int),
        "TeamA": parts[1].astype(int),
        "TeamB": parts[2].astype(int),
    })

    seasons = sorted(parsed["Season"].unique())
    print(f"  Stage {stage}: seasons {seasons}, {len(parsed)} matchups")

    # Train models
    print("  Training Men's models...")
    # Men's: torvik LR C=1.0 (62%) + torvik LGB (38%) + seeds LR (bonus)
    m_torvik_lr, m_torvik_lr_cols, m_torvik_lr_med = train_model(
        data, "M", "torvik", C=1.0, start_year=2015
    )
    m_torvik_lgb, m_torvik_lgb_cols, m_torvik_lgb_med = train_lgb(
        data, "M", "torvik", start_year=2015
    )
    m_seeds_lr, m_seeds_lr_cols, m_seeds_lr_med = train_model(
        data, "M", "seeds_only", C=1.0
    )

    print("  Training Women's models...")
    # Women's: tier2 LR C=0.1
    w_tier2_lr, w_tier2_lr_cols, w_tier2_lr_med = train_model(
        data, "W", "tier2", C=0.1
    )
    # Fallback: seeds only
    w_seeds_lr, w_seeds_lr_cols, w_seeds_lr_med = train_model(
        data, "W", "seeds_only", C=1.0
    )

    # Cache team features
    tf_cache = {}

    predictions = {}
    total = len(parsed)

    for i, (_, row) in enumerate(parsed.iterrows()):
        if (i + 1) % 10000 == 0:
            print(f"    Progress: {i+1}/{total}")

        team_a, team_b = row["TeamA"], row["TeamB"]
        season = row["Season"]
        gender = "M" if team_a < 3000 else "W"

        if gender == "M":
            # Torvik ensemble: 50% LR + 39% LGB + 11% seeds_LR
            # Torvik features
            cache_key = (season, "M", "torvik")
            if cache_key not in tf_cache:
                tf_cache[cache_key] = build_team_features(data, season, "M", "torvik")
            tf_torvik = tf_cache[cache_key]

            # Seeds features
            cache_key_s = (season, "M", "seeds_only")
            if cache_key_s not in tf_cache:
                tf_cache[cache_key_s] = build_team_features(data, season, "M", "seeds_only")
            tf_seeds = tf_cache[cache_key_s]

            p_torvik_lr = predict_matchup(tf_torvik, team_a, team_b,
                                          m_torvik_lr_cols, m_torvik_lr, m_torvik_lr_med)
            p_torvik_lgb = predict_matchup(tf_torvik, team_a, team_b,
                                           m_torvik_lgb_cols, m_torvik_lgb, m_torvik_lgb_med)
            p_seeds = predict_matchup(tf_seeds, team_a, team_b,
                                      m_seeds_lr_cols, m_seeds_lr, m_seeds_lr_med)

            pred = 0.50 * p_torvik_lr + 0.39 * p_torvik_lgb + 0.11 * p_seeds
        else:
            # Women's: tier2 LR C=0.1
            cache_key = (season, "W", "tier2")
            if cache_key not in tf_cache:
                tf_cache[cache_key] = build_team_features(data, season, "W", "tier2")
            tf_w = tf_cache[cache_key]

            if w_tier2_lr is not None:
                pred = predict_matchup(tf_w, team_a, team_b,
                                       w_tier2_lr_cols, w_tier2_lr, w_tier2_lr_med)
            else:
                # Fallback to seeds
                cache_key_ws = (season, "W", "seeds_only")
                if cache_key_ws not in tf_cache:
                    tf_cache[cache_key_ws] = build_team_features(data, season, "W", "seeds_only")
                tf_ws = tf_cache[cache_key_ws]
                pred = predict_matchup(tf_ws, team_a, team_b,
                                       w_seeds_lr_cols, w_seeds_lr, w_seeds_lr_med)

        predictions[row["ID"]] = float(np.clip(pred, *PREDICTION_CLIP))

    submission = pd.DataFrame({
        "ID": sample_sub["ID"],
        "Pred": sample_sub["ID"].map(predictions).fillna(0.5),
    })

    errors = validate_submission(submission, sample_sub)
    if errors:
        print(f"  VALIDATION ERRORS: {errors}")
    else:
        print(f"  Validation passed!")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_stage{stage}_torvik_v2_{timestamp}.csv"
    save_path = SUBMISSIONS_DIR / filename
    submission.to_csv(save_path, index=False)

    print(f"  Saved: {save_path}")
    print(f"  Rows: {len(submission)}, Mean: {submission['Pred'].mean():.4f}, "
          f"Std: {submission['Pred'].std():.4f}")

    # Print prediction distribution
    preds = submission["Pred"].values
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(preds, bins=bins)
    print(f"  Distribution: {dict(zip([f'{b:.1f}-{bins[i+1]:.1f}' for i, b in enumerate(bins[:-1])], hist))}")

    return save_path


if __name__ == "__main__":
    print("Loading data...")
    data = load_all()

    print("\n=== Stage 1 (Historical) ===")
    s1 = generate_submission(data, 1)

    print("\n=== Stage 2 (2026) ===")
    s2 = generate_submission(data, 2)

    print(f"\nDone! Submissions:")
    print(f"  Stage 1: {s1}")
    print(f"  Stage 2: {s2}")
