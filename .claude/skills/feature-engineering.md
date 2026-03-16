---
name: feature-engineering
description: Build and iterate on features for matchup prediction
user_invocable: true
---

# Feature Engineering Skill

Build, evaluate, and iterate on features for tournament matchup prediction.

## Architecture

- All features computed at **team-season level**, then combined into **matchup-level** features
- Each feature module exposes: `compute(data: dict[str, DataFrame], season: int) -> DataFrame` with columns `[TeamID, feature1, feature2, ...]`
- `builder.py` orchestrates all modules and creates matchup rows where TeamA_ID < TeamB_ID
- Matchup features: primarily differences (feat_A - feat_B), plus individual seed values

## Feature Priority

1. **Seeds** (`seeds.py`): Parse seed string → int. Features: seed_A, seed_B, seed_diff
2. **Massey Ordinals** (`ordinals.py`): Late-season (DayNum≥128) ranks from top systems. Mean rank, rank_diff
3. **Elo** (`elo.py`): Rolling Elo with K=20, MOV adjustment, season regression to mean
4. **Season Stats** (`season_stats.py`): Win%, PPG, PPG_allowed, point_diff, SOS
5. **Efficiency** (`efficiency.py`): Possessions, AdjO, AdjD, AdjEM, Tempo (needs detailed results, 2003+)
6. **Four Factors** (`four_factors.py`): eFG%, TO_rate, OR%, FT_rate (offense + defense)

## Critical Rules

- **No leakage**: Features for season S use ONLY season S regular season data (no tournament, no future seasons)
- **Symmetry**: Matchup (A, B) features must be the negated version of (B, A) features for difference features
- After training, log feature importances and drop zero-importance features in next iteration

## Iteration Workflow

1. Add new feature module or modify existing one
2. Run `python -m pytest tests/test_features.py` to validate
3. Run cross-validation to measure Brier score impact
4. Log results to experiment log
5. If improvement > 0.001, keep the feature; otherwise revert

## Commands

```bash
source /c/Users/wesho/miniforge3/etc/profile.d/conda.sh && conda activate marchmadness
cd /c/Users/wesho/Projects/house-of-claudes
python -c "from marchmadness.features.builder import build_features; df = build_features(); print(df.shape, df.columns.tolist())"
```
