"""Tests for submission format validation."""

import pytest
import pandas as pd
import numpy as np
from marchmadness.submission import validate_submission, parse_submission_ids


class TestSubmissionValidation:
    def make_sample(self, n=10):
        """Create a sample submission DataFrame."""
        ids = [f"2025_{1000+i}_{1100+i}" for i in range(n)]
        return pd.DataFrame({"ID": ids, "Pred": [0.5] * n})

    def test_valid_submission(self):
        sample = self.make_sample()
        submission = sample.copy()
        errors = validate_submission(submission, sample)
        assert errors == []

    def test_wrong_columns(self):
        sample = self.make_sample()
        submission = pd.DataFrame({"id": sample["ID"], "pred": sample["Pred"]})
        errors = validate_submission(submission, sample)
        assert any("columns" in e.lower() for e in errors)

    def test_missing_rows(self):
        sample = self.make_sample(10)
        submission = sample.head(5).copy()
        errors = validate_submission(submission, sample)
        assert any("row count" in e.lower() for e in errors)

    def test_nan_predictions(self):
        sample = self.make_sample()
        submission = sample.copy()
        submission.loc[0, "Pred"] = np.nan
        errors = validate_submission(submission, sample)
        assert any("nan" in e.lower() for e in errors)

    def test_out_of_range_predictions(self):
        sample = self.make_sample()
        submission = sample.copy()
        submission.loc[0, "Pred"] = 1.5
        errors = validate_submission(submission, sample)
        assert any("range" in e.lower() for e in errors)

    def test_duplicate_ids(self):
        sample = self.make_sample()
        submission = sample.copy()
        submission.loc[1, "ID"] = submission.loc[0, "ID"]
        errors = validate_submission(submission, sample)
        assert any("duplicate" in e.lower() for e in errors)


class TestParseSubmissionIds:
    def test_parse_basic(self):
        df = pd.DataFrame({"ID": ["2025_1000_1001", "2025_3000_3001"]})
        parsed = parse_submission_ids(df)
        assert parsed.loc[0, "Season"] == 2025
        assert parsed.loc[0, "TeamA"] == 1000
        assert parsed.loc[0, "TeamB"] == 1001
        assert parsed.loc[1, "TeamA"] == 3000
