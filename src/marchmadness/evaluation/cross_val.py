"""Leave-season-out cross-validation framework."""

import numpy as np
import pandas as pd
from marchmadness.evaluation.metrics import compute_all_metrics


def leave_season_out_cv(model, X: pd.DataFrame, y: np.ndarray,
                        seasons: np.ndarray, cv_seasons: list[int]) -> dict:
    """Run leave-season-out cross-validation.

    Args:
        model: Scikit-learn compatible model with fit/predict_proba
        X: Feature matrix
        y: Labels (0 or 1)
        seasons: Season array aligned with X
        cv_seasons: List of seasons to use as validation folds

    Returns:
        Dict with:
        - per_fold: dict of season -> metrics
        - overall: aggregated metrics across all folds
        - oof_predictions: out-of-fold predictions array
        - oof_labels: corresponding true labels
        - oof_seasons: corresponding seasons
    """
    oof_preds = np.full(len(y), np.nan)
    per_fold = {}

    for val_season in cv_seasons:
        train_mask = seasons != val_season
        val_mask = seasons == val_season

        if val_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_val)[:, 1]
        else:
            preds = model.predict(X_val)

        oof_preds[val_mask] = preds
        fold_metrics = compute_all_metrics(y_val, preds)
        fold_metrics["n_train"] = int(train_mask.sum())
        fold_metrics["n_val"] = int(val_mask.sum())
        per_fold[val_season] = fold_metrics

    # Overall metrics on all OOF predictions
    valid_mask = ~np.isnan(oof_preds)
    overall = compute_all_metrics(y[valid_mask], oof_preds[valid_mask])

    return {
        "per_fold": per_fold,
        "overall": overall,
        "oof_predictions": oof_preds[valid_mask],
        "oof_labels": y[valid_mask],
        "oof_seasons": seasons[valid_mask],
    }
