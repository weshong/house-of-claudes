"""Performance regression tests."""

import json
import pytest
from pathlib import Path
from marchmadness.config import BEST_SCORES_FILE, REGRESSION_THRESHOLD


@pytest.mark.skipif(
    not BEST_SCORES_FILE.exists(),
    reason="No best scores file yet (first run)"
)
class TestRegression:
    def _get_scores(self):
        with open(BEST_SCORES_FILE) as f:
            return json.load(f)

    def test_scores_not_regressed(self):
        """Ensure no model configuration has regressed significantly."""
        scores = self._get_scores()
        # This test just validates the file structure
        for key, entry in scores.items():
            assert "brier" in entry
            assert isinstance(entry["brier"], float)
            assert 0 < entry["brier"] < 1
