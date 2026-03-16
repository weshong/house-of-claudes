# March Machine Learning Mania 2026 - Autoresearch Agent

## Competition Overview
- **Competition**: [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
- **Task**: Predict P(lower TeamId wins) for every possible NCAA tournament matchup
- **Men's TeamIds**: 1000–1999, **Women's TeamIds**: 3000–3999
- **Submission**: CSV with `ID` (`2026_TeamIdLow_TeamIdHigh`) and `Pred` (probability)
- **Metric**: Brier score (MSE of predicted probs vs 0/1 outcomes) — **lower is better**
- **Prize**: $50,000
- **Deadline**: March 19, 2026
- **Stages**: Stage 1 = historical validation (2022–2025); Stage 2 = 2026 predictions
- **Must select 2 submissions** for final scoring

## Environment
- **Python**: Python 3.13 venv at `.venv/`
- **Activate**: `source /c/Users/wesho/Projects/house-of-claudes/.venv/Scripts/activate`
- **GPU**: RTX 4070 (12GB VRAM) — use for XGBoost (`tree_method='gpu_hist'`) and LightGBM (`device='gpu'`)
- **Data**: `data/raw/` — downloaded via `kaggle competitions download -c march-machine-learning-mania-2026`
- **Tests**: `pytest tests/` — run before any submission

## Agent Operating Principles

1. **Leave-season-out CV is ground truth** — never trust a single train/test split. Train on all seasons except target, predict target tournament.
2. **Log every experiment** — write to `outputs/reports/experiment_log.md` with timestamp, features, model params, CV Brier scores.
3. **Clip predictions to [0.025, 0.975]** — Brier score severely punishes extreme wrong predictions.
4. **Train men's and women's models separately** — different tournament dynamics and data availability.
5. **Massey ordinals are the highest-signal features** after seeds — they encode expert rating systems for free.
6. **Brier score targets**: baseline seed-only ~0.20, with ordinals ~0.17, good model ~0.16, competitive ~0.14.
7. **Ensemble conservatively** — simple weighted average of 2-3 models beats complex stacking for this small dataset (~70 tournament games/year).
8. **Check GitHub issues/PR comments** before each iteration cycle for collaborator feedback.

## Workflow

1. Load and validate data completeness (`python -m marchmadness.data_loader`)
2. Build seed-only baseline, measure Brier on 2022–2025 CV
3. Add Massey ordinal features, measure improvement
4. Add Elo ratings + season stats, measure improvement
5. Add efficiency/four-factors features (needs detailed results, 2003+), measure improvement
6. Train XGBoost and LightGBM alongside logistic regression
7. Build ensemble, optimize weights on CV Brier score
8. Run full evaluation suite (calibration, per-round, upset analysis)
9. Generate Stage 1 and Stage 2 submissions
10. Select 2 final submissions: one conservative (seed-heavy), one aggressive (full ensemble)

## Data Notes

- **Seed format**: e.g., "W01" = region W, seed 01; "X16a"/"X16b" for play-in games. Parse seed number as integer.
- **Massey ordinals**: filter to `RankingDayNum >= 128` (late season) for stable rankings. Top systems: POM (Pomeroy), SAG (Sagarin), WLK, WOL, MOR, DOL, COL, RPI.
- **Detailed results**: available from 2003 onward for men's; coverage varies for women's.
- **Not all teams in every season** — check `FirstD1Season` and `LastD1Season` in Teams files.
- **FGM includes 3-pointers** — points = 2×(FGM - FGM3) + 3×FGM3 + FTM, or equivalently Score column.

## Testing

- Run `pytest tests/` before any submission
- `test_regression.py` fails if Brier score regresses more than 0.005 from previous best stored in `outputs/best_scores.json`
- All submissions must pass format validation (correct IDs, Pred in [0,1], no NaN)

## Experiment Tracking

- **GitHub repo**: https://github.com/weshong/house-of-claudes
- Each experiment: descriptive commit with features, params, CV scores
- Push regularly for traceability
- Check issues/PR comments for collaborator feedback
- Log to `outputs/reports/experiment_log.md`

## Project Structure

```
src/marchmadness/
├── config.py          # Paths, constants, hyperparams
├── data_loader.py     # Load + cache all CSVs
├── features/
│   ├── seeds.py       # Seed parsing, seed-diff
│   ├── elo.py         # Elo ratings with MOV adjustment
│   ├── efficiency.py  # KenPom-style AdjO/AdjD/AdjEM/Tempo
│   ├── four_factors.py # eFG%, TO rate, OR%, FT rate
│   ├── ordinals.py    # Massey ordinal features
│   ├── season_stats.py # Win%, PPG, point diff, SOS
│   └── builder.py     # Orchestrate features → matchup DataFrame
├── models/
│   ├── logistic.py    # Logistic regression baseline
│   ├── xgboost_model.py # XGBoost (GPU)
│   ├── lightgbm_model.py # LightGBM (GPU)
│   ├── ensemble.py    # Weighted average ensemble
│   └── trainer.py     # Unified train/predict interface
├── evaluation/
│   ├── metrics.py     # Brier, log loss, accuracy, calibration
│   ├── analysis.py    # Per-round, per-seed-tier, upset analysis
│   ├── cross_val.py   # Leave-season-out CV
│   └── plots.py       # Calibration curves, reliability diagrams
└── submission.py      # Generate + validate submission CSVs
```
