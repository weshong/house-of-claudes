"""Unified model training and prediction interface."""

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from marchmadness.config import (
    MODELS_DIR, PREDICTION_CLIP, CV_SEASONS, REPORTS_DIR, BEST_SCORES_FILE
)
from marchmadness.models import logistic, xgboost_model, lightgbm_model
from marchmadness.models.ensemble import optimize_weights, ensemble_predict
from marchmadness.evaluation.cross_val import leave_season_out_cv
from marchmadness.evaluation.metrics import compute_all_metrics
from marchmadness.features.builder import (
    build_training_data, build_prediction_matchups, get_feature_columns
)
from marchmadness.data_loader import load_all


MODEL_FACTORIES = {
    "logistic": logistic.create_model,
    "xgboost": xgboost_model.create_model,
    "lightgbm": lightgbm_model.create_model,
}


class ModelTrainer:
    """Manages training, CV, and ensemble for all models."""

    def __init__(self, feature_set: str = "all", gender: str = "M"):
        self.feature_set = feature_set
        self.gender = gender
        self.models = {}
        self.cv_results = {}
        self.ensemble_weights = None
        self.data = None
        self.training_df = None

    def load_data(self):
        """Load all competition data."""
        self.data = load_all()

    def build_training(self, seasons: list[int] | None = None,
                        include_regular_season: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Build training data. Returns (X, y, seasons, sample_weights)."""
        if self.data is None:
            self.load_data()

        if seasons is None:
            # Use all seasons with tournament data
            tourney_key = f"{self.gender}NCAATourneyCompactResults"
            available_seasons = sorted(self.data[tourney_key]["Season"].unique())
            seasons = available_seasons

        self.training_df = build_training_data(
            self.data, seasons, self.gender, self.feature_set,
            include_regular_season=include_regular_season
        )
        if self.training_df.empty:
            raise ValueError("No training data built")

        feature_cols = get_feature_columns(self.training_df)
        X = self.training_df[feature_cols].values.copy()
        y = self.training_df["Label"].values.copy()
        season_arr = self.training_df["Season"].values.copy()

        # Extract sample weights if available
        sample_weights = None
        if "SampleWeight" in self.training_df.columns:
            sample_weights = self.training_df["SampleWeight"].values.copy()

        # Handle NaN: fill with column median
        col_medians = np.nanmedian(X, axis=0)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = col_medians[i] if not np.isnan(col_medians[i]) else 0

        self.feature_cols = feature_cols
        return X, y, season_arr, sample_weights

    def run_cv(self, model_name: str, X: np.ndarray, y: np.ndarray,
               seasons: np.ndarray, calibrate: bool = False) -> dict:
        """Run leave-season-out CV for a single model."""
        factory = MODEL_FACTORIES[model_name]
        model = factory()

        cv_result = leave_season_out_cv(model, X, y, seasons, CV_SEASONS,
                                        calibrate=calibrate)
        self.cv_results[model_name] = cv_result

        print(f"\n{model_name} CV Results:")
        print(f"  Overall Brier: {cv_result['overall']['brier_score']:.4f}")
        print(f"  Overall Accuracy: {cv_result['overall']['accuracy']:.4f}")
        for season, metrics in cv_result["per_fold"].items():
            print(f"  Season {season}: Brier={metrics['brier_score']:.4f}, "
                  f"Acc={metrics['accuracy']:.4f}, n={metrics['n_val']}")

        return cv_result

    def run_all_cv(self, X: np.ndarray, y: np.ndarray, seasons: np.ndarray):
        """Run CV for all models and optimize ensemble."""
        for name in MODEL_FACTORIES:
            try:
                self.run_cv(name, X, y, seasons)
            except Exception as e:
                print(f"  {name} failed: {e}")

        # Optimize ensemble weights
        if len(self.cv_results) >= 2:
            oof_preds = [
                self.cv_results[name]["oof_predictions"]
                for name in self.cv_results
            ]
            oof_labels = list(self.cv_results.values())[0]["oof_labels"]

            self.ensemble_weights = optimize_weights(oof_preds, oof_labels)
            ens_pred = ensemble_predict(oof_preds, self.ensemble_weights)
            ens_pred = np.clip(ens_pred, *PREDICTION_CLIP)
            ens_metrics = compute_all_metrics(oof_labels, ens_pred)

            model_names = list(self.cv_results.keys())
            print(f"\nEnsemble weights: {dict(zip(model_names, self.ensemble_weights))}")
            print(f"Ensemble Brier: {ens_metrics['brier_score']:.4f}")
            print(f"Ensemble Accuracy: {ens_metrics['accuracy']:.4f}")

    def train_final(self, X: np.ndarray, y: np.ndarray):
        """Train all models on full dataset for final predictions."""
        for name, factory in MODEL_FACTORIES.items():
            model = factory()
            model.fit(X, y)
            self.models[name] = model

            # Save model
            model_path = MODELS_DIR / f"{name}_{self.gender}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"  Saved {name} to {model_path}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions."""
        preds_list = []
        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                preds = model.predict_proba(X)[:, 1]
            else:
                preds = model.predict(X)
            preds_list.append(preds)

        if self.ensemble_weights is not None and len(preds_list) == len(self.ensemble_weights):
            result = ensemble_predict(preds_list, self.ensemble_weights)
        else:
            result = np.mean(preds_list, axis=0)

        return np.clip(result, *PREDICTION_CLIP)

    def log_experiment(self, description: str, metrics: dict):
        """Append experiment results to the experiment log."""
        log_path = REPORTS_DIR / "experiment_log.md"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = f"\n## {timestamp} - {description}\n\n"
        entry += f"- Feature set: {self.feature_set}\n"
        entry += f"- Gender: {self.gender}\n"
        for key, val in metrics.items():
            if isinstance(val, float):
                entry += f"- {key}: {val:.4f}\n"
            else:
                entry += f"- {key}: {val}\n"
        entry += "\n---\n"

        with open(log_path, "a") as f:
            f.write(entry)

    def update_best_scores(self, brier: float, model_desc: str):
        """Update best scores tracker if this is a new best."""
        scores = {}
        if BEST_SCORES_FILE.exists():
            with open(BEST_SCORES_FILE) as f:
                scores = json.load(f)

        key = f"{self.gender}_{self.feature_set}"
        if key not in scores or brier < scores[key]["brier"]:
            scores[key] = {
                "brier": brier,
                "model": model_desc,
                "timestamp": datetime.now().isoformat(),
            }
            with open(BEST_SCORES_FILE, "w") as f:
                json.dump(scores, f, indent=2)
            print(f"  New best {key}: {brier:.4f}")
