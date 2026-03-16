"""XGBoost model with GPU support."""

from marchmadness.config import XGBOOST_PARAMS


def create_model():
    """Create XGBoost classifier. Falls back to CPU if GPU unavailable."""
    try:
        from xgboost import XGBClassifier
        # Try GPU first
        model = XGBClassifier(**XGBOOST_PARAMS)
        return model
    except Exception:
        from xgboost import XGBClassifier
        # Fall back to CPU
        params = {**XGBOOST_PARAMS}
        params["tree_method"] = "hist"
        params.pop("device", None)
        return XGBClassifier(**params)
