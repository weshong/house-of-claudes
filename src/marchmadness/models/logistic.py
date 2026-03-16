"""Logistic regression model."""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from marchmadness.config import LOGISTIC_PARAMS


def create_model() -> Pipeline:
    """Create logistic regression pipeline with standardization."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(**LOGISTIC_PARAMS)),
    ])
