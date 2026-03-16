"""Leave-season-out cross-validation framework."""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from marchmadness.evaluation.metrics import compute_all_metrics


def leave_season_out_cv(model, X: pd.DataFrame, y: np.ndarray,
                        seasons: np.ndarray, cv_seasons: list[int],
                        calibrate: bool = False,
                        sample_weights: np.ndarray | None = None,
                        shrinkage_alpha: float = 0.0) -> dict:
    """Run leave-season-out cross-validation.

    Args:
        model: Scikit-learn compatible model with fit/predict_proba
        X: Feature matrix
        y: Labels (0 or 1)
        seasons: Season array aligned with X
        cv_seasons: List of seasons to use as validation folds
        calibrate: If True, apply nested isotonic regression calibration
        sample_weights: Optional sample weights for training
        shrinkage_alpha: If > 0, shrink predictions toward 0.5

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

        if sample_weights is not None:
            w_train = sample_weights[train_mask]
            # Try passing sample_weight with pipeline-compatible naming
            try:
                model.fit(X_train, y_train, sample_weight=w_train)
            except (TypeError, ValueError):
                # For sklearn Pipeline, need stepname__sample_weight
                # Try to find the last step name
                if hasattr(model, 'steps'):
                    step_name = model.steps[-1][0]
                    model.fit(X_train, y_train, **{f"{step_name}__sample_weight": w_train})
                else:
                    model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_val)[:, 1]
        else:
            preds = model.predict(X_val)

        if calibrate:
            # Nested calibration: train calibrator on OOF predictions from
            # non-val, non-current training folds
            if hasattr(model, "predict_proba"):
                train_preds = model.predict_proba(X_train)[:, 1]
            else:
                train_preds = model.predict(X_train)
            iso = IsotonicRegression(y_min=0.025, y_max=0.975, out_of_bounds="clip")
            iso.fit(train_preds, y_train)
            preds = iso.predict(preds)

        # Shrinkage: pull predictions toward 0.5
        if shrinkage_alpha > 0:
            preds = preds * (1 - shrinkage_alpha) + 0.5 * shrinkage_alpha

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
