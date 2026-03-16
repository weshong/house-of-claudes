"""Generate and validate Kaggle submission files."""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from marchmadness.config import DATA_DIR, SUBMISSIONS_DIR, PREDICTION_CLIP, CURRENT_SEASON
from marchmadness.data_loader import load_all
from marchmadness.features.builder import build_team_features, build_matchup_features, get_feature_columns
from marchmadness.models.trainer import ModelTrainer


def parse_submission_ids(sample_sub: pd.DataFrame) -> pd.DataFrame:
    """Parse sample submission IDs into (Season, TeamA, TeamB)."""
    parts = sample_sub["ID"].str.split("_", expand=True)
    return pd.DataFrame({
        "ID": sample_sub["ID"],
        "Season": parts[0].astype(int),
        "TeamA": parts[1].astype(int),
        "TeamB": parts[2].astype(int),
    })


def validate_submission(submission: pd.DataFrame, sample_sub: pd.DataFrame) -> list[str]:
    """Validate submission format. Returns list of error messages (empty = valid)."""
    errors = []

    if list(submission.columns) != ["ID", "Pred"]:
        errors.append(f"Wrong columns: {list(submission.columns)}, expected ['ID', 'Pred']")
        return errors  # Can't check further with wrong columns

    if len(submission) != len(sample_sub):
        errors.append(f"Wrong row count: {len(submission)}, expected {len(sample_sub)}")

    if submission["Pred"].isna().any():
        errors.append(f"Found {submission['Pred'].isna().sum()} NaN predictions")

    if (submission["Pred"] < 0).any() or (submission["Pred"] > 1).any():
        errors.append("Predictions outside [0, 1] range")

    missing_ids = set(sample_sub["ID"]) - set(submission["ID"])
    if missing_ids:
        errors.append(f"Missing {len(missing_ids)} IDs")

    extra_ids = set(submission["ID"]) - set(sample_sub["ID"])
    if extra_ids:
        errors.append(f"Extra {len(extra_ids)} IDs not in sample")

    if submission["ID"].duplicated().any():
        errors.append(f"Found {submission['ID'].duplicated().sum()} duplicate IDs")

    return errors


def generate_submission(stage: int = 2, feature_set: str = "all",
                        model_type: str = "ensemble") -> Path:
    """Generate a submission file.

    Args:
        stage: 1 (historical) or 2 (2026 predictions)
        feature_set: Which features to use
        model_type: "ensemble", "logistic", etc.

    Returns:
        Path to the generated submission file.
    """
    data = load_all()

    # Load sample submission
    sample_key = f"SampleSubmissionStage{stage}"
    sample_sub = data[sample_key]
    parsed = parse_submission_ids(sample_sub)

    # Get unique seasons in this submission
    seasons = sorted(parsed["Season"].unique())
    print(f"Generating Stage {stage} submission for seasons: {seasons}")

    # Train model
    trainer = ModelTrainer(feature_set=feature_set)
    trainer.data = data

    # Build training data from all available tournament seasons
    X_train, y_train, seasons_train = trainer.build_training()
    trainer.train_final(X_train, y_train)

    # Generate predictions for each season in submission
    predictions = {}
    for season in seasons:
        # Determine gender based on TeamID range
        season_rows = parsed[parsed["Season"] == season]

        for _, row in season_rows.iterrows():
            team_a, team_b = row["TeamA"], row["TeamB"]

            # Determine gender
            gender = "M" if team_a < 3000 else "W"

            # Build team features
            team_feats = build_team_features(data, season, gender, feature_set)
            matchup = build_matchup_features(team_feats, team_a, team_b)

            if matchup:
                feat_values = np.array([[matchup.get(c, np.nan) for c in trainer.feature_cols]])
                # Fill NaN with 0
                feat_values = np.nan_to_num(feat_values, nan=0.0)
                pred = trainer.predict(feat_values)[0]
            else:
                pred = 0.5  # Default for missing data

            predictions[row["ID"]] = float(np.clip(pred, *PREDICTION_CLIP))

    # Build submission DataFrame
    submission = pd.DataFrame({
        "ID": sample_sub["ID"],
        "Pred": sample_sub["ID"].map(predictions).fillna(0.5),
    })

    # Validate
    errors = validate_submission(submission, sample_sub)
    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Submission validation passed!")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"submission_stage{stage}_{feature_set}_{timestamp}.csv"
    save_path = SUBMISSIONS_DIR / filename
    submission.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")

    # Summary stats
    print(f"  Rows: {len(submission)}")
    print(f"  Mean prediction: {submission['Pred'].mean():.4f}")
    print(f"  Std prediction: {submission['Pred'].std():.4f}")
    print(f"  Min: {submission['Pred'].min():.4f}, Max: {submission['Pred'].max():.4f}")

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2])
    parser.add_argument("--features", type=str, default="all")
    args = parser.parse_args()
    generate_submission(stage=args.stage, feature_set=args.features)
