"""Fixed evaluation harness for autoresearch. DO NOT MODIFY.

This is the ground truth evaluation — leave-season-out CV on Brier score.
The research.py file calls this to measure improvements.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from marchmadness.config import CV_SEASONS, PREDICTION_CLIP
from marchmadness.data_loader import load_all
from marchmadness.evaluation.metrics import brier_score


def prep_data(df, feat_cols):
    """Prepare feature matrix, handling NaNs."""
    X = df[feat_cols].values.copy()
    y = df["Label"].values.copy()
    seasons = df["Season"].values.copy()
    med = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        X[mask, i] = med[i] if not np.isnan(med[i]) else 0
    return X, y, seasons, med


def evaluate_single_model(build_data_fn, model_fn, data, gender, label=""):
    """Evaluate a single model with leave-season-out CV.

    Args:
        build_data_fn: callable(data, seasons, gender) -> DataFrame with features + Label + Season
        model_fn: callable() -> sklearn-compatible model
        data: loaded data dict
        gender: "M" or "W"
        label: display label

    Returns:
        dict with 'brier', 'per_fold', 'oof_preds', 'oof_labels', 'oof_seasons'
    """
    # Build training data for all available seasons
    all_seasons = build_data_fn(data, gender)
    if isinstance(all_seasons, pd.DataFrame) and all_seasons.empty:
        return {"brier": float("inf"), "per_fold": {}}

    df = all_seasons
    feat_cols = [c for c in df.columns if c not in {"Season", "Label", "DayNum", "SampleWeight", "TeamA", "TeamB"}]
    X, y, seasons, med = prep_data(df, feat_cols)

    oof_preds = np.full(len(y), np.nan)
    per_fold = {}

    for val_season in CV_SEASONS:
        train_mask = seasons != val_season
        val_mask = seasons == val_season
        if val_mask.sum() == 0:
            continue

        model = model_fn()
        model.fit(X[train_mask], y[train_mask])
        preds = model.predict_proba(X[val_mask])[:, 1]
        preds = np.clip(preds, *PREDICTION_CLIP)
        oof_preds[val_mask] = preds

        fold_brier = brier_score(y[val_mask], preds)
        per_fold[val_season] = fold_brier

    valid = ~np.isnan(oof_preds)
    if valid.sum() == 0:
        return {"brier": float("inf"), "per_fold": {}}

    overall_brier = brier_score(y[valid], oof_preds[valid])

    return {
        "brier": overall_brier,
        "per_fold": per_fold,
        "oof_preds": oof_preds[valid],
        "oof_labels": y[valid],
        "oof_seasons": seasons[valid],
        "n_features": len(feat_cols),
        "n_samples": int(valid.sum()),
    }


def evaluate_blend(results_list, weights, label=""):
    """Evaluate a weighted blend of OOF predictions.

    Args:
        results_list: list of dicts from evaluate_single_model (must have same CV games)
        weights: list of floats (must sum to 1.0)

    Returns:
        dict with 'brier'
    """
    if not results_list or any(r["brier"] == float("inf") for r in results_list):
        return {"brier": float("inf")}

    blend = sum(w * r["oof_preds"] for w, r in zip(weights, results_list))
    blend = np.clip(blend, *PREDICTION_CLIP)
    y = results_list[0]["oof_labels"]
    return {"brier": brier_score(y, blend)}


def load_data():
    """Load all competition data. Cached after first call."""
    return load_all()
