"""Integration tests for the training pipeline."""

import pytest
import numpy as np
from marchmadness.evaluation.metrics import (
    brier_score, compute_log_loss, compute_accuracy,
    calibration_error, compute_all_metrics
)
from marchmadness.evaluation.cross_val import leave_season_out_cv
from marchmadness.models.ensemble import optimize_weights, ensemble_predict


class TestMetrics:
    def test_brier_perfect(self):
        """Perfect predictions should have Brier score of 0."""
        y = np.array([0, 1, 1, 0])
        pred = np.array([0, 1, 1, 0], dtype=float)
        assert brier_score(y, pred) == pytest.approx(0.0)

    def test_brier_worst(self):
        """Worst predictions should have Brier score of 1."""
        y = np.array([0, 1])
        pred = np.array([1, 0], dtype=float)
        assert brier_score(y, pred) == pytest.approx(1.0)

    def test_brier_naive(self):
        """Always predicting 0.5 should give Brier of 0.25."""
        y = np.array([0, 1, 0, 1])
        pred = np.array([0.5, 0.5, 0.5, 0.5])
        assert brier_score(y, pred) == pytest.approx(0.25)

    def test_accuracy_perfect(self):
        y = np.array([0, 1, 1, 0])
        pred = np.array([0.1, 0.9, 0.8, 0.2])
        assert compute_accuracy(y, pred) == 1.0

    def test_calibration_error_returns_bins(self):
        y = np.random.randint(0, 2, 100)
        pred = np.random.uniform(0, 1, 100)
        ece, bins = calibration_error(y, pred)
        assert 0 <= ece <= 1
        assert len(bins) == 10

    def test_compute_all_metrics(self):
        y = np.random.randint(0, 2, 50)
        pred = np.random.uniform(0, 1, 50)
        metrics = compute_all_metrics(y, pred)
        assert "brier_score" in metrics
        assert "log_loss" in metrics
        assert "accuracy" in metrics
        assert "ece" in metrics


class TestEnsemble:
    def test_optimize_weights_sum_to_one(self):
        y = np.random.randint(0, 2, 100)
        pred1 = np.clip(y + np.random.normal(0, 0.3, 100), 0, 1)
        pred2 = np.clip(y + np.random.normal(0, 0.4, 100), 0, 1)
        weights = optimize_weights([pred1, pred2], y)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_ensemble_predict_shape(self):
        pred1 = np.random.uniform(0, 1, 50)
        pred2 = np.random.uniform(0, 1, 50)
        weights = np.array([0.6, 0.4])
        result = ensemble_predict([pred1, pred2], weights)
        assert result.shape == (50,)
        assert all(0 <= r <= 1 for r in result)


class TestCrossVal:
    def test_leave_season_out_basic(self):
        """Test CV with a simple sklearn model on synthetic data."""
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 3)
        y = (X[:, 0] > 0).astype(int)
        seasons = np.repeat([2022, 2023, 2024, 2025], n // 4)

        model = LogisticRegression()
        result = leave_season_out_cv(model, X, y, seasons, [2024, 2025])

        assert "per_fold" in result
        assert "overall" in result
        assert 2024 in result["per_fold"]
        assert 2025 in result["per_fold"]
        assert 0 <= result["overall"]["brier_score"] <= 1
