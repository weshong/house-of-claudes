"""Advanced analysis: per-round, per-seed-tier, upset detection."""

import numpy as np
import pandas as pd
from marchmadness.evaluation.metrics import brier_score, compute_accuracy


# DayNum ranges for tournament rounds (approximate, may vary by year)
ROUND_DAYNUM_RANGES = {
    "R64": (134, 135),
    "R32": (136, 137),
    "S16": (138, 139),
    "E8": (143, 144),
    "F4": (152, 152),
    "Championship": (154, 154),
}


def per_round_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                       day_nums: np.ndarray) -> dict[str, dict]:
    """Compute Brier score and accuracy for each tournament round."""
    results = {}
    for round_name, (day_min, day_max) in ROUND_DAYNUM_RANGES.items():
        mask = (day_nums >= day_min) & (day_nums <= day_max)
        if mask.sum() == 0:
            continue
        results[round_name] = {
            "brier_score": float(brier_score(y_true[mask], y_pred[mask])),
            "accuracy": float(compute_accuracy(y_true[mask], y_pred[mask])),
            "n_games": int(mask.sum()),
        }
    return results


def per_seed_tier_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                           seed_a: np.ndarray, seed_b: np.ndarray) -> dict[str, dict]:
    """Compute metrics for different seed matchup tiers."""
    seed_diff = np.abs(seed_a - seed_b)
    results = {}

    # Standard first-round matchups
    matchups = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
    for s_high, s_low in matchups:
        mask = (
            ((seed_a == s_high) & (seed_b == s_low)) |
            ((seed_a == s_low) & (seed_b == s_high))
        )
        if mask.sum() == 0:
            continue
        results[f"{s_high}v{s_low}"] = {
            "brier_score": float(brier_score(y_true[mask], y_pred[mask])),
            "accuracy": float(compute_accuracy(y_true[mask], y_pred[mask])),
            "n_games": int(mask.sum()),
        }

    # Close matchups (seed diff <= 3)
    close_mask = seed_diff <= 3
    if close_mask.sum() > 0:
        results["close (diff<=3)"] = {
            "brier_score": float(brier_score(y_true[close_mask], y_pred[close_mask])),
            "accuracy": float(compute_accuracy(y_true[close_mask], y_pred[close_mask])),
            "n_games": int(close_mask.sum()),
        }

    return results


def upset_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                   seed_a: np.ndarray, seed_b: np.ndarray) -> dict:
    """Analyze upset prediction performance.

    An upset occurs when the higher-seeded team (higher seed number = worse seed) wins.
    For our format: team_a has lower TeamID, not necessarily lower seed.
    """
    # Identify the favorite (lower seed number) and underdog
    # In our encoding: label=1 means lower TeamID won
    # We need to know which team had the better seed

    # Games where there's a clear seed difference
    seed_diff = seed_a - seed_b  # negative means A is favored
    significant_mask = np.abs(seed_diff) >= 5

    if significant_mask.sum() == 0:
        return {"upset_rate": 0.0, "n_significant_games": 0}

    sig_true = y_true[significant_mask]
    sig_pred = y_pred[significant_mask]
    sig_diff = seed_diff[significant_mask]

    # For each game, determine if an upset happened
    # If seed_diff < 0 (A favored), upset = label 0 (B won)
    # If seed_diff > 0 (B favored), upset = label 1 (A won)
    upset_happened = np.where(sig_diff < 0, sig_true == 0, sig_true == 1)
    upset_predicted = np.where(sig_diff < 0, sig_pred < 0.5, sig_pred > 0.5)

    n_upsets = upset_happened.sum()
    n_correct_upset_pred = (upset_happened & upset_predicted).sum()

    return {
        "n_significant_games": int(significant_mask.sum()),
        "n_actual_upsets": int(n_upsets),
        "actual_upset_rate": float(n_upsets / significant_mask.sum()),
        "upset_detection_rate": float(n_correct_upset_pred / n_upsets) if n_upsets > 0 else 0.0,
        "false_upset_predictions": int((~upset_happened & upset_predicted).sum()),
    }


def overconfidence_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                            threshold: float = 0.3) -> dict:
    """Analyze performance on confident predictions (|pred - 0.5| > threshold)."""
    confidence = np.abs(y_pred - 0.5)
    confident_mask = confidence > threshold

    if confident_mask.sum() == 0:
        return {"n_confident": 0}

    return {
        "n_confident": int(confident_mask.sum()),
        "pct_confident": float(confident_mask.mean()),
        "brier_confident": float(brier_score(y_true[confident_mask], y_pred[confident_mask])),
        "accuracy_confident": float(compute_accuracy(y_true[confident_mask], y_pred[confident_mask])),
        "brier_uncertain": float(brier_score(y_true[~confident_mask], y_pred[~confident_mask]))
        if (~confident_mask).sum() > 0 else None,
    }
