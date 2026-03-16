"""Evaluation metrics for tournament prediction."""

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier score (MSE of predicted probabilities vs outcomes)."""
    return brier_score_loss(y_true, y_pred)


def compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log loss, clipping predictions to avoid infinities."""
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return log_loss(y_true, y_pred_clipped)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute classification accuracy at a given threshold."""
    y_class = (y_pred >= threshold).astype(int)
    return accuracy_score(y_true, y_class)


def calibration_error(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Compute Expected Calibration Error (ECE) and per-bin stats.

    Returns:
        (ece, bins) where bins is a list of dicts with keys:
        bin_center, mean_pred, actual_rate, count
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    weighted_errors = []

    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = mask | (y_pred == bin_edges[i + 1])

        count = mask.sum()
        if count == 0:
            bins.append({
                "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
                "mean_pred": np.nan,
                "actual_rate": np.nan,
                "count": 0,
            })
            continue

        mean_pred = y_pred[mask].mean()
        actual_rate = y_true[mask].mean()
        error = abs(mean_pred - actual_rate)

        bins.append({
            "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2,
            "mean_pred": mean_pred,
            "actual_rate": actual_rate,
            "count": int(count),
        })
        weighted_errors.append(error * count)

    total = len(y_true)
    ece = sum(weighted_errors) / total if total > 0 else 0.0

    return ece, bins


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all standard metrics."""
    ece, cal_bins = calibration_error(y_true, y_pred)
    return {
        "brier_score": brier_score(y_true, y_pred),
        "log_loss": compute_log_loss(y_true, y_pred),
        "accuracy": compute_accuracy(y_true, y_pred),
        "ece": ece,
        "n_samples": len(y_true),
        "mean_prediction": float(y_pred.mean()),
        "calibration_bins": cal_bins,
    }
