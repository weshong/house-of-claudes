"""End-to-end pipeline: load -> features -> train -> eval -> submit."""

import argparse
import json
import numpy as np
from datetime import datetime

from marchmadness.config import CV_SEASONS, REPORTS_DIR, PREDICTION_CLIP
from marchmadness.models.trainer import ModelTrainer
from marchmadness.evaluation.metrics import compute_all_metrics
from marchmadness.evaluation.analysis import (
    per_round_analysis, per_seed_tier_analysis, upset_analysis, overconfidence_analysis
)
from marchmadness.evaluation.plots import plot_calibration, plot_prediction_distribution


def run_cv_pipeline(feature_set: str = "all", gender: str = "M"):
    """Run full CV pipeline and generate evaluation report."""
    print(f"=== CV Pipeline: {gender} {feature_set} ===")

    trainer = ModelTrainer(feature_set=feature_set, gender=gender)
    trainer.load_data()

    # Build training data
    X, y, seasons = trainer.build_training()
    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Feature columns: {trainer.feature_cols}")

    # Run CV for all models
    trainer.run_all_cv(X, y, seasons)

    # Get ensemble OOF predictions for analysis
    if len(trainer.cv_results) >= 2 and trainer.ensemble_weights is not None:
        oof_preds_list = [
            trainer.cv_results[name]["oof_predictions"]
            for name in trainer.cv_results
        ]
        oof_labels = list(trainer.cv_results.values())[0]["oof_labels"]
        oof_seasons = list(trainer.cv_results.values())[0]["oof_seasons"]

        ens_preds = np.clip(
            np.column_stack(oof_preds_list) @ trainer.ensemble_weights,
            *PREDICTION_CLIP
        )

        # Full metrics
        metrics = compute_all_metrics(oof_labels, ens_preds)
        print(f"\n=== Ensemble Metrics ===")
        print(f"Brier: {metrics['brier_score']:.4f}")
        print(f"Log Loss: {metrics['log_loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"ECE: {metrics['ece']:.4f}")

        # Calibration plot
        cal_path = plot_calibration(
            metrics["calibration_bins"],
            title=f"Calibration - {gender} {feature_set}",
        )
        print(f"Calibration plot: {cal_path}")

        # Prediction distribution
        dist_path = plot_prediction_distribution(
            ens_preds,
            title=f"Predictions - {gender} {feature_set}",
        )
        print(f"Distribution plot: {dist_path}")

        # Per-round analysis (need DayNum from training data)
        if "DayNum" in trainer.training_df.columns:
            # Get DayNum for OOF samples
            valid_mask = trainer.training_df["Season"].isin(CV_SEASONS)
            if valid_mask.sum() == len(oof_labels):
                day_nums = trainer.training_df.loc[valid_mask, "DayNum"].values
                round_results = per_round_analysis(oof_labels, ens_preds, day_nums)
                print(f"\n=== Per-Round Analysis ===")
                for rnd, res in round_results.items():
                    print(f"  {rnd}: Brier={res['brier_score']:.4f}, "
                          f"Acc={res['accuracy']:.4f}, n={res['n_games']}")

        # Seed-tier analysis
        if "seed_a" in (trainer.feature_cols if hasattr(trainer, 'feature_cols') else []):
            # Extract seed info from training df
            valid_df = trainer.training_df[trainer.training_df["Season"].isin(CV_SEASONS)]
            if "seed_a" in valid_df.columns and "seed_b" in valid_df.columns:
                seed_a = valid_df["seed_a"].values
                seed_b = valid_df["seed_b"].values
                seed_results = per_seed_tier_analysis(oof_labels, ens_preds, seed_a, seed_b)
                print(f"\n=== Seed Tier Analysis ===")
                for tier, res in seed_results.items():
                    print(f"  {tier}: Brier={res['brier_score']:.4f}, n={res['n_games']}")

                upset_results = upset_analysis(oof_labels, ens_preds, seed_a, seed_b)
                print(f"\n=== Upset Analysis ===")
                for k, v in upset_results.items():
                    print(f"  {k}: {v}")

        # Overconfidence
        oc_results = overconfidence_analysis(oof_labels, ens_preds)
        print(f"\n=== Overconfidence Analysis ===")
        for k, v in oc_results.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Log experiment
        trainer.log_experiment(
            f"{gender} {feature_set} ensemble",
            {**metrics, "features": trainer.feature_cols}
        )
        trainer.update_best_scores(metrics["brier_score"], f"{feature_set} ensemble")

    elif trainer.cv_results:
        # Single model result
        best_name = min(trainer.cv_results, key=lambda n: trainer.cv_results[n]["overall"]["brier_score"])
        best = trainer.cv_results[best_name]
        trainer.log_experiment(
            f"{gender} {feature_set} {best_name}",
            best["overall"]
        )
        trainer.update_best_scores(best["overall"]["brier_score"], f"{feature_set} {best_name}")

    return trainer


def run_final_training(feature_set: str = "all", gender: str = "M") -> ModelTrainer:
    """Train final models on all data."""
    print(f"\n=== Final Training: {gender} {feature_set} ===")
    trainer = ModelTrainer(feature_set=feature_set, gender=gender)
    trainer.load_data()
    X, y, _ = trainer.build_training()
    trainer.train_final(X, y)
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cv", "train", "full"], default="cv")
    parser.add_argument("--features", type=str, default="all")
    parser.add_argument("--gender", type=str, default="M", choices=["M", "W"])
    args = parser.parse_args()

    if args.mode == "cv":
        run_cv_pipeline(args.features, args.gender)
    elif args.mode == "train":
        run_final_training(args.features, args.gender)
    elif args.mode == "full":
        # Run CV first, then train final
        run_cv_pipeline(args.features, args.gender)
        run_final_training(args.features, args.gender)
