# March Madness Autoresearch

You are an autonomous research agent improving NCAA tournament prediction models. Your goal: **minimize Brier score** (MSE of predicted probabilities vs 0/1 outcomes) for both men's and women's brackets.

## Current Best Scores (baselines to beat)

| Gender | Brier | Config |
|--------|-------|--------|
| Men's | 0.1543 | Torvik LGB d=3, 25 features |
| Women's | 0.1386 | tier2 LR 70% + iter_eff LR 30% |
| Combined | 0.1465 | average of M + W |

## Setup

1. Read `research.py` — the file you modify. Contains model configs, feature building, and evaluation.
2. Read `evaluate.py` — fixed evaluation harness. DO NOT modify.
3. Read `CLAUDE.md` — project context and constraints.
4. Verify baseline: run `python -u research.py 2>&1` and confirm scores match above.
5. Create branch: `git checkout -b autoresearch/<tag>` from main.
6. Create `results.tsv` with header: `commit\tval_m_brier\tval_w_brier\tval_combined\tstatus\tdescription`
7. Start experimenting.

## What you CAN modify

**Only `research.py`**. Everything is fair game:
- Feature sets and feature engineering (add/remove/transform features)
- Model type, hyperparameters, architecture
- Blend strategies and weights
- Training data range (start_year)
- Custom feature building (the `build_custom_team_features` function)
- New derived features, interactions, transformations
- Calibration / post-processing of predictions
- Import and use any module in `src/marchmadness/features/`

## What you CANNOT modify

- `evaluate.py` — the evaluation harness is fixed
- Files in `src/` — the feature computation modules are read-only
- The CV setup: leave-season-out on [2022, 2023, 2024, 2025]
- The metric: Brier score, lower is better

## Available Feature Sources

| Source | M | W | Years | Key Features |
|--------|---|---|-------|--------------|
| Seeds | ✓ | ✓ | All | Seed number |
| Massey ordinals | ✓ | ✗ | All | POM, SAG, WLK, WOL, MOR, DOL, COL, RPI, AP, USA + aggregates |
| Elo ratings | ✓ | ✓ | All | Elo with MOV adjustment |
| Season stats | ✓ | ✓ | All | WinPct, PPG, PointDiff, SOS |
| Efficiency (KenPom) | ✓ | ✓ | 2003+ | AdjO, AdjD, AdjEM, Tempo |
| Four factors | ✓ | ✓ | 2003+ | eFG%, TO rate, OR%, FT rate |
| Iterative adj eff | ✓ | ✓ | 2003+ | Opponent-adjusted O/D/EM/Tempo |
| T-Rank clone | ✓ | ✓ | All with detailed | Torvik-style adjoe/adjde/barthag/tempo/sos/luck |
| Torvik external | ✓ | ✗ | 2015+ | Real Torvik adjoe/adjde/barthag/adjt/sos |
| Women's Torvik | ✗ | ✓ | 2025-26 only | Scraped adjoe/adjde/barthag/adjt |

**Critical insight**: Women's model lacks Massey ordinals AND external Torvik ratings. The biggest improvements likely come from bringing better features to women's.

## Key Findings from Previous Experiments (Do Not Re-Test)

These have been thoroughly tested and confirmed negative:
- Ensembles do NOT beat single Torvik LGB d=3 for men's (exp 10b)
- Enhanced/derived Torvik features HURT men's (exp 10)
- Conference strength features don't help (exp 12)
- T-Rank clone is redundant with tier2 for women's (exp 13-14)
- Calibration (Platt, isotonic, temperature) doesn't help (exp 15)
- Deep trees overfit (d=6 much worse than d=3 for men's)
- Regular season inclusion with tournament weighting HURTS

## Promising Directions to Explore

1. **Women's feature engineering**: Build features from detailed box scores that capture what ordinals capture for men's. Women's model has only 10 features vs men's 25.
2. **Women's T-Rank clone tuning**: The clone itself at 0.1418 is worse than tier2 at 0.1390, but the features might complement tier2 in a combined model (not blend).
3. **Better ordinal selection**: We use top 10 systems. Try different subsets, weighted composites, or PCA on all 197 systems.
4. **Per-round features**: Seed matchup category (1v16, 5v12, etc.) as explicit feature
5. **Recency-weighted features**: Weight recent games more heavily in Elo/stats computation
6. **Feature selection**: Remove low-importance features that add noise
7. **Training window optimization**: Different start_year per feature set
8. **Custom interactions**: Cross features between seeds, ratings, stats

## The Experiment Loop

LOOP FOREVER:

1. Review current state: `git log --oneline -5` and check `results.tsv`
2. Choose an experimental idea. Think about what's most likely to help.
3. Modify `research.py` with your change.
4. `git add research.py && git commit -m "experiment: <description>"`
5. Run: `python -u research.py > run.log 2>&1`
6. Read results: `grep "Men's:\|Women's:\|Combined:\|IMPROVED\|No improvement" run.log`
7. If combined Brier improved OR either gender improved without hurting the other:
   - Log as `keep` in results.tsv
   - The branch advances
8. If combined Brier worsened:
   - Log as `discard` in results.tsv
   - `git reset --hard HEAD~1` to revert
9. If the run crashed:
   - `tail -20 run.log` to see error
   - Fix and retry, or log as `crash` and revert
10. Go to step 1.

**NEVER STOP.** Run indefinitely until manually interrupted. If you run out of ideas, think harder — try combining near-misses, try more radical changes, re-read the feature modules for inspiration.

**Keep changes small.** One idea per experiment. This makes it easy to attribute improvements and revert failures.

## Logging

Append to `results.tsv` (tab-separated) after each experiment:

```
commit	val_m_brier	val_w_brier	val_combined	status	description
a1b2c3d	0.154300	0.138600	0.146450	keep	baseline
```

Do NOT commit results.tsv — leave it untracked.
