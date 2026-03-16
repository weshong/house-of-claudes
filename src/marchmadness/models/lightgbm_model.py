"""LightGBM model with GPU support."""

from marchmadness.config import LIGHTGBM_PARAMS


def create_model():
    """Create LightGBM classifier. Falls back to CPU if GPU unavailable."""
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(**LIGHTGBM_PARAMS)
        return model
    except Exception:
        from lightgbm import LGBMClassifier
        params = {**LIGHTGBM_PARAMS}
        params.pop("device", None)
        return LGBMClassifier(**params)
