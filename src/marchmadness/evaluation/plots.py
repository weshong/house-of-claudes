"""Evaluation plots: calibration curves, reliability diagrams."""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from marchmadness.config import REPORTS_DIR


def plot_calibration(bins: list[dict], title: str = "Calibration Plot",
                     save_path: Path | None = None) -> Path:
    """Plot reliability diagram from calibration bins.

    Args:
        bins: List of dicts from metrics.calibration_error()
        title: Plot title
        save_path: Where to save. Defaults to reports dir.

    Returns:
        Path to saved plot.
    """
    if save_path is None:
        save_path = REPORTS_DIR / "calibration.png"

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)

    # Actual calibration
    centers = [b["bin_center"] for b in bins if b["count"] > 0]
    actuals = [b["actual_rate"] for b in bins if b["count"] > 0]
    preds = [b["mean_pred"] for b in bins if b["count"] > 0]
    counts = [b["count"] for b in bins if b["count"] > 0]

    ax.bar(centers, actuals, width=0.08, alpha=0.3, label="Actual rate", color="steelblue")
    ax.scatter(preds, actuals, s=[c * 2 for c in counts], c="steelblue",
               label="Mean pred vs actual", zorder=5)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Actual Win Rate")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_prediction_distribution(y_pred: np.ndarray, title: str = "Prediction Distribution",
                                 save_path: Path | None = None) -> Path:
    """Plot histogram of predicted probabilities."""
    if save_path is None:
        save_path = REPORTS_DIR / "pred_distribution.png"

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(y_pred, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
