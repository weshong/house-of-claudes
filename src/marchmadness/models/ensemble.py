"""Ensemble model: optimized weighted average of base models."""

import numpy as np
from scipy.optimize import minimize
from marchmadness.evaluation.metrics import brier_score


def optimize_weights(predictions_list: list[np.ndarray], y_true: np.ndarray) -> np.ndarray:
    """Optimize ensemble weights to minimize Brier score.

    Args:
        predictions_list: List of prediction arrays from each model
        y_true: True labels

    Returns:
        Optimal weight array (sums to 1)
    """
    n_models = len(predictions_list)
    preds_matrix = np.column_stack(predictions_list)

    def objective(weights):
        ensemble_pred = np.clip(preds_matrix @ weights, 0, 1)
        return float(np.mean((np.asarray(y_true, dtype=float) - ensemble_pred) ** 2))

    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    # Bounds: each weight between 0 and 1
    bounds = [(0, 1)] * n_models
    # Initial guess: equal weights
    x0 = np.ones(n_models) / n_models

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        return x0  # Fall back to equal weights


def ensemble_predict(predictions_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Generate ensemble predictions using optimized weights."""
    preds_matrix = np.column_stack(predictions_list)
    return preds_matrix @ weights
