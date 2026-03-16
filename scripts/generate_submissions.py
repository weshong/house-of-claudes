"""Generate final competition submissions using best strategies."""

import numpy as np
import pandas as pd
from datetime import datetime
from itertools import combinations
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from marchmadness.config import (
    DATA_DIR, SUBMISSIONS_DIR, PREDICTION_CLIP, CV_SEASONS
)
from marchmadness.data_loader import load_all
from marchmadness.features.builder import (
    build_team_features, build_matchup_features, get_feature_columns, build_training_data
)
from marchmadness.submission import validate_submission


def train_seed_ensemble(data, gender):
    """Train the seed-only LR + LightGBM ensemble (our best Men's model)."""
    training_df = build_training_data(
        data, list(range(1985 if gender == "M" else 1998, 2026)),
        gender, "seeds_only"
    )
    if training_df.empty:
        return None, None

    feature_cols = get_feature_columns(training_df)
    X = training_df[feature_cols].values.copy()
    y = training_df["Label"].values.copy()

    # Handle NaN
    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0

    # Train LR
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])
    lr.fit(X, y)

    # Train LightGBM
    try:
        from lightgbm import LGBMClassifier
        lgb = LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        )
        lgb.fit(X, y)
        models = [("lr", lr, 0.5), ("lgb", lgb, 0.5)]
    except Exception:
        models = [("lr", lr, 1.0)]

    return models, feature_cols


def train_minimal_model(data, gender):
    """Train the minimal 3-feature LR (C=0.01)."""
    training_df = build_training_data(
        data, list(range(1985 if gender == "M" else 1998, 2026)),
        gender, "all"
    )
    if training_df.empty:
        return None, None, None

    all_feature_cols = get_feature_columns(training_df)
    target_cols = ["Seed_diff", "OrdinalMean_diff", "Elo_diff"]
    available = [c for c in target_cols if c in all_feature_cols]

    if not available:
        return None, None, None

    X = training_df[available].values.copy()
    y = training_df["Label"].values.copy()

    # Handle NaN
    col_medians = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs")),
    ])
    model.fit(X, y)

    return model, available, all_feature_cols


def predict_matchup(team_features, team_a, team_b, feature_cols, models_info, feature_set="seeds_only"):
    """Predict a single matchup using trained models."""
    matchup = build_matchup_features(team_features, team_a, team_b)
    if not matchup:
        return 0.5

    feat_values = np.array([[matchup.get(c, np.nan) for c in feature_cols]])
    feat_values = np.nan_to_num(feat_values, nan=0.0)

    if isinstance(models_info, list):
        # Ensemble of (name, model, weight) tuples
        preds = []
        weights = []
        for name, model, weight in models_info:
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(feat_values)[:, 1][0]
            else:
                pred = model.predict(feat_values)[0]
            preds.append(pred)
            weights.append(weight)
        result = np.average(preds, weights=weights)
    else:
        # Single model
        if hasattr(models_info, "predict_proba"):
            result = models_info.predict_proba(feat_values)[:, 1][0]
        else:
            result = models_info.predict(feat_values)[0]

    return float(np.clip(result, *PREDICTION_CLIP))


def generate_submission(data, stage, strategy="conservative"):
    """Generate a complete submission.

    strategy: "conservative" (seed ensemble) or "aggressive" (blend)
    """
    sample_key = f"SampleSubmissionStage{stage}"
    sample_sub = data[sample_key]

    # Parse IDs
    parts = sample_sub["ID"].str.split("_", expand=True)
    parsed = pd.DataFrame({
        "ID": sample_sub["ID"],
        "Season": parts[0].astype(int),
        "TeamA": parts[1].astype(int),
        "TeamB": parts[2].astype(int),
    })

    seasons = sorted(parsed["Season"].unique())
    print(f"  Generating Stage {stage} '{strategy}' for seasons: {seasons}")

    # Train models for each gender
    models = {}
    for gender in ["M", "W"]:
        print(f"  Training {gender} models...")
        seed_models, seed_cols = train_seed_ensemble(data, gender)
        models[f"{gender}_seed"] = (seed_models, seed_cols)

        if strategy == "aggressive":
            min_model, min_cols, all_cols = train_minimal_model(data, gender)
            models[f"{gender}_minimal"] = (min_model, min_cols)

    # Cache team features per (season, gender, feature_set)
    team_feature_cache = {}

    predictions = {}
    total = len(parsed)

    for i, (_, row) in enumerate(parsed.iterrows()):
        if (i + 1) % 10000 == 0:
            print(f"    Progress: {i+1}/{total}")

        team_a, team_b = row["TeamA"], row["TeamB"]
        season = row["Season"]
        gender = "M" if team_a < 3000 else "W"

        # Get seed features
        cache_key = (season, gender, "seeds_only")
        if cache_key not in team_feature_cache:
            team_feature_cache[cache_key] = build_team_features(data, season, gender, "seeds_only")
        seed_tf = team_feature_cache[cache_key]

        seed_models_info, seed_feature_cols = models[f"{gender}_seed"]

        if strategy == "conservative" or gender == "W":
            # Pure seed ensemble
            pred = predict_matchup(seed_tf, team_a, team_b, seed_feature_cols, seed_models_info)
        elif strategy == "aggressive":
            # Blend: 30% seed + 70% minimal
            seed_pred = predict_matchup(seed_tf, team_a, team_b, seed_feature_cols, seed_models_info)

            cache_key_all = (season, gender, "all")
            if cache_key_all not in team_feature_cache:
                team_feature_cache[cache_key_all] = build_team_features(data, season, gender, "all")
            all_tf = team_feature_cache[cache_key_all]

            min_model, min_cols = models[f"{gender}_minimal"]
            if min_model is not None:
                min_pred = predict_matchup(all_tf, team_a, team_b, min_cols, min_model)
                pred = float(np.clip(0.3 * seed_pred + 0.7 * min_pred, *PREDICTION_CLIP))
            else:
                pred = seed_pred
        else:
            pred = 0.5

        predictions[row["ID"]] = pred

    # Build submission
    submission = pd.DataFrame({
        "ID": sample_sub["ID"],
        "Pred": sample_sub["ID"].map(predictions).fillna(0.5),
    })

    # Validate
    errors = validate_submission(submission, sample_sub)
    if errors:
        print(f"  VALIDATION ERRORS: {errors}")
    else:
        print(f"  Validation passed!")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_stage{stage}_{strategy}_{timestamp}.csv"
    save_path = SUBMISSIONS_DIR / filename
    submission.to_csv(save_path, index=False)

    print(f"  Saved: {save_path}")
    print(f"  Rows: {len(submission)}, Mean pred: {submission['Pred'].mean():.4f}, "
          f"Std: {submission['Pred'].std():.4f}")

    return save_path


if __name__ == "__main__":
    print("Loading data...")
    data = load_all()

    print("\n=== Stage 1 (Historical Validation) ===")
    generate_submission(data, 1, "conservative")
    generate_submission(data, 1, "aggressive")

    print("\n=== Stage 2 (2026 Predictions) ===")
    generate_submission(data, 2, "conservative")
    generate_submission(data, 2, "aggressive")

    print("\nDone! Check outputs/submissions/ for files.")
