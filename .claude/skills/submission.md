---
name: submission
description: Generate and validate Kaggle submission files
user_invocable: true
---

# Submission Skill

Generate, validate, and prepare submission files for Kaggle.

## Submission Format

- CSV with columns: `ID`, `Pred`
- ID format: `{Season}_{TeamIDLow}_{TeamIDHigh}`
- Pred: probability that lower TeamID wins (float in [0, 1])
- Men's TeamIDs: 1000-1999, Women's: 3000-3999

## Steps

1. Load `SampleSubmissionStage1.csv` or `SampleSubmissionStage2.csv` from `data/raw/`
2. Parse each ID to get (Season, TeamIDLow, TeamIDHigh)
3. Look up both teams' features for the given season
4. Run ensemble prediction for P(low wins)
5. Clip predictions to [0.025, 0.975]
6. Validate submission:
   - All IDs from sample are present
   - All Pred values in [0, 1], no NaN
   - Row count matches sample exactly
   - No duplicate IDs
7. Save to `outputs/submissions/submission_stage{1,2}_{timestamp}.csv`

## Sanity Checks

- For Stage 1: compute Brier score against known historical tournament results
- For 1-seed vs 16-seed matchups: Pred should be > 0.80
- For 8-seed vs 9-seed matchups: Pred should be close to 0.50
- Mean prediction should be close to 0.50 (roughly symmetric)

## Submission Strategy

Generate 2 submissions for final selection:
1. **Conservative**: Seed-heavy logistic regression with ordinals (lower variance)
2. **Aggressive**: Full ensemble with all features (potentially lower Brier, higher variance)

## Commands

```bash
source /c/Users/wesho/miniforge3/etc/profile.d/conda.sh && conda activate marchmadness
cd /c/Users/wesho/Projects/house-of-claudes
python -m marchmadness.submission --stage 1
python -m marchmadness.submission --stage 2
```
