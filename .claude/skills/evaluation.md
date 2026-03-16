---
name: evaluation
description: Comprehensive model evaluation beyond Brier score
user_invocable: true
---

# Evaluation Skill

Run comprehensive evaluation suite on model predictions.

## Metrics

| Metric | Target | Purpose |
|--------|--------|---------|
| Brier score | < 0.16 | Primary competition metric |
| Log loss | < 0.50 | Calibration-sensitive |
| Accuracy (@0.5) | > 70% | Sanity check |
| ECE (calibration) | < 0.05 | Reliability of probabilities |
| Upset detection | > 30% | Catching lower-seed wins |
| Overconfidence penalty | low | Right when confident |

## Analysis Breakdowns

1. **Per-round Brier**: R64, R32, S16, E8, F4, Championship
   - DayNum ranges: 134-135=R64, 136-137=R32, 138-139=S16, 143-144=E8, 152=F4, 154=Championship
2. **Per-seed-tier**: 1v16, 2v15, ..., 8v9, and closer matchups
3. **Men's vs Women's**: all metrics tracked separately
4. **Per-season**: Brier by CV fold to check consistency
5. **Calibration plot**: 10-bin reliability diagram (predicted vs actual win rate)
6. **Confidence distribution**: histogram of predicted probabilities

## Cross-Validation Protocol

Leave-season-out on [2022, 2023, 2024, 2025]:
- Train on all tournament matchups EXCEPT target season
- Predict target season tournament
- Report per-fold and mean scores

## Output

Save structured report to `outputs/reports/eval_{timestamp}.md` with:
- Summary table of all metrics
- Per-fold breakdown
- Feature importances (top 20)
- Calibration plot saved as PNG
- Recommendations for next iteration

## Commands

```bash
source /c/Users/wesho/miniforge3/etc/profile.d/conda.sh && conda activate marchmadness
cd /c/Users/wesho/Projects/house-of-claudes
python scripts/run_evaluation.py
```
