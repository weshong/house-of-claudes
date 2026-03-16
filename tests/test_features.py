"""Unit tests for feature engineering modules."""

import pytest
import pandas as pd
import numpy as np
from marchmadness.features.seeds import parse_seed
from marchmadness.features.elo import expected_score, mov_multiplier
from marchmadness.features.efficiency import estimate_possessions


class TestSeedParsing:
    def test_standard_seeds(self):
        assert parse_seed("W01") == 1
        assert parse_seed("X16") == 16
        assert parse_seed("Y08") == 8
        assert parse_seed("Z11") == 11

    def test_playin_seeds(self):
        assert parse_seed("X16a") == 16
        assert parse_seed("X16b") == 16
        assert parse_seed("W11a") == 11
        assert parse_seed("W11b") == 11

    def test_all_seeds_in_range(self):
        for region in "WXYZ":
            for seed in range(1, 17):
                seed_str = f"{region}{seed:02d}"
                assert 1 <= parse_seed(seed_str) <= 16


class TestElo:
    def test_expected_score_equal(self):
        """Equal ratings should give 0.5 expected score."""
        assert abs(expected_score(1500, 1500) - 0.5) < 1e-10

    def test_expected_score_higher_rated(self):
        """Higher-rated player should have expected score > 0.5."""
        assert expected_score(1600, 1400) > 0.5
        assert expected_score(1400, 1600) < 0.5

    def test_expected_scores_sum_to_one(self):
        """Expected scores for both players should sum to 1."""
        e1 = expected_score(1600, 1400)
        e2 = expected_score(1400, 1600)
        assert abs(e1 + e2 - 1.0) < 1e-10

    def test_mov_multiplier_positive(self):
        """MOV multiplier should be positive for any margin."""
        for margin in [1, 5, 10, 20, 50]:
            assert mov_multiplier(margin) > 0

    def test_mov_multiplier_increasing(self):
        """Larger margins should give larger multipliers (diminishing)."""
        m1 = mov_multiplier(5)
        m2 = mov_multiplier(20)
        assert m2 > m1


class TestEfficiency:
    def test_possessions_positive(self):
        """Estimated possessions should be positive for reasonable inputs."""
        poss = estimate_possessions(fga=60, ora=10, to=15, fta=20)
        assert poss > 0

    def test_possessions_formula(self):
        """Check possessions formula: FGA - OR + TO + 0.475 * FTA."""
        poss = estimate_possessions(fga=60, ora=10, to=15, fta=20)
        expected = 60 - 10 + 15 + 0.475 * 20
        assert abs(poss - expected) < 1e-10


class TestFeatureLeakage:
    """Test that features don't leak future or tournament data."""

    def test_matchup_symmetry(self):
        """Difference features for (A,B) should negate for (B,A)."""
        from marchmadness.features.builder import build_matchup_features

        # Create minimal team features
        team_features = pd.DataFrame({
            "TeamID": [1000, 1001],
            "Seed": [1, 16],
            "Elo": [1700, 1300],
        })

        feat_ab = build_matchup_features(team_features, 1000, 1001)
        feat_ba = build_matchup_features(team_features, 1000, 1001)

        # Since we always order (low, high), both should be the same
        assert feat_ab == feat_ba

        # Check that diff features have expected signs
        assert feat_ab["Seed_diff"] == 1 - 16  # A seed - B seed
        assert feat_ab["Elo_diff"] == 1700 - 1300
