# Experiment Journal - March ML Mania 2026

---

## Brier Score Leaderboard (Updated)

| Gender | Config | Brier | Timestamp |
|--------|--------|-------|-----------|
| M | **Torvik shallow LGB d=3 n=400 lr=0.02** | **0.1543** | 2026-03-16 14:00 |
| M | Torvik ensemble (LR 50% + LGB 39% + seeds 11%) | 0.1607 | 2026-03-16 13:00 |
| M | torvik_LR_1 + torvik_LGB (62/38) | 0.1610 | 2026-03-16 13:00 |
| M | torvik LR C=1.0 (single model) | 0.1703 | 2026-03-16 13:00 |
| M | seeds_only ensemble (LR+LGB) | 0.1915 | 2026-03-16 10:38 |
| W | **tier2 LR(70%) + iter_eff LR(30%) blend** | **0.1386** | 2026-03-16 22:00 |
| W | tier2_LR C=0.1 meta-ensemble | 0.1390 | 2026-03-16 12:00 |
| W | seeds_only LR | 0.1489 | 2026-03-16 10:39 |

### Improvement Timeline

| Experiment | Best M Brier | Best W Brier |
|------------|-------------|-------------|
| Exp 1: Seed baseline | 0.1915 | 0.1489 |
| Exp 6: Meta-ensemble | 0.1904 | 0.1390 |
| Exp 7: Torvik | 0.1607 | 0.1390 |
| Exp 8: Shallow trees | 0.1543 | 0.1387 |
| **Exp 9-15: T-Rank, calibration** | **0.1543** | **0.1386** |

Total improvement: M **0.0372** (19.4%), W **0.0103** (6.9%)

---

## Gap Analysis: Current Score vs Competition Winners

### The Numbers
| Metric | 2025 Winner | Our Best (Men's) | Our Best (Women's) | Gap |
|--------|-------------|-------------------|---------------------|-----|
| Brier | ~0.10 | 0.1543 | 0.1387 | 0.05 (M), 0.04 (W) |

### Where the Gap Comes From

A Brier score is MSE of predicted probabilities vs 0/1 outcomes. Each game contributes `(pred - outcome)²`:

- **Perfect prediction**: A 1-seed vs 16-seed game where we predict 0.97 and they win → (0.97-1)² = 0.0009. If we predict 0.80 → (0.80-1)² = 0.04. That single game costs us 40x more Brier.

- **The math of ~67 games**: With ~67 tournament games scored, each game contributes ~1.5% of total Brier. A 0.05 gap across 67 games means we're off by ~0.23 per-game on average in our probability predictions (√0.05 ≈ 0.22).

### Likely Sources of the 0.05 Gap

1. **Calibration on upsets (~0.015-0.02)**: Winners likely better predict which 8v9, 7v10, 5v12 matchups are true toss-ups vs slight favorites. Our model may be too confident on mild favorites.

2. **Late-round accuracy (~0.01-0.015)**: Elite 8 and Final Four matchups between strong teams are where probability estimates matter most. Small improvements here compound.

3. **Feature quality (~0.01-0.015)**: Winners may use additional data sources — coaching experience, injury reports, recent form trajectory, matchup-specific features (pace mismatch, 3-pt defense vs 3-pt shooting).

4. **Model sophistication (~0.005-0.01)**: Better ensemble methods, stacking, Bayesian approaches, or more careful cross-validation. Our single LGB is strong but not ensembled well.

### What 0.10 Brier Looks Like
- Average absolute error of ~0.32 per game (√0.10)
- On a 1v16 game (95% true): predicting ~0.92-0.97 (we do this fine)
- On a 5v12 game (65% true): predicting ~0.60-0.68 instead of our ~0.55-0.75
- On Elite 8 games: getting the close calls right more often

### Actionable Improvements for Tomorrow
1. **Calibration**: Apply isotonic regression or Platt scaling to CV predictions
2. **Better ensembles**: Blend models with diverse error patterns, not just diverse features
3. **More features**: Coaching tenure, conference strength, travel distance, rest days
4. **Women's Torvik**: Find women's adjusted efficiency ratings (Her Hoop Stats)
5. **Recency**: Weight late-season performance more heavily

---

## 2026-03-16 22:00 - Experiments 9-15: T-Rank Clone, Features, Calibration

### Exp 9: Enhanced Torvik Features & Women's LGB (NEGATIVE)
- Extra Torvik columns (WAB, elite_SOS, Qual_Barthag, etc.): 0.1550 → 0.1627 (hurt)
- Women's LGB d=3: 0.1440-0.1541 (all worse than LR 0.1390)
- Single Torvik LGB d=3 (0.1543) remains best for men's

### Exp 10-10b: Enhanced Features & Ensembles (NEGATIVE)
- Derived Torvik features (AdjEM, off_pct, percentiles, interactions): 0.1610 vs 0.1543 baseline (hurt)
- Best pairwise ensemble: 0.1562 — WORSE than single LGB (0.1543)
- All 3-model, 4-model, log-odds, seed prior blends: all worse
- **Confirmed: single model dominance for men's**

### Exp 11: Enhanced Iterative Efficiency v2 (NEGATIVE)
- Added recency, HCA, possession weighting to iter_eff
- Women's: v2 = 0.1426 vs v1 = 0.1411 (worse)
- V1 remains superior

### Exp 12: Conference Strength Features (NEGATIVE)
- Added conf_mean_strength, conf_rank, conf_size, team_strength_vs_conf
- Women's: 0.1394 vs 0.1390 baseline (no improvement)

### Exp 13: T-Rank Clone + Bayesian Optimization
- Built vectorized Torvik-style adjusted efficiency clone (works for both genders)
- Validated against real Torvik: 0.93-0.96 correlation
- **Vectorized version: 0.16s for 16 seasons (vs hours with iterrows)**
- Optuna optimization (50 trials): best Brier = 0.1418 for women's
- Best params: n_iter=10, hca=0.018, recency_win=31, decay=0.011, floor=0.43, blowout=9, barthag_exp=13.0, preseason=5, C=0.05
- Men's clone: 0.1868 (much worse than real Torvik 0.1543 — missing preseason priors, recruiting data)
- Blending with tier2: T-Rank adds 0% weight (fully redundant)

### Exp 14: T-Rank + tier2 Combined Features & 3-Way Blend
- tier2 + T-Rank combined: 0.1421 (worse than tier2 alone)
- 3-way blend tier2 + iter_eff + T-Rank: best = tier2 70% + iter_eff 30% + T-Rank 0% = **0.1386**
- T-Rank features built from same game data → no complementary signal

### Exp 15: Calibration Analysis
- Women's ECE = 0.0402, Men's ECE = 0.0582
- **Women's**: [0.2-0.3] bin badly miscalibrated (pred=0.256, actual=0.138)
- Temperature T=0.9: 0.1387 (tiny improvement) — slight overconfidence
- **Men's**: [0.6-0.8] range underconfident (pred ~0.7, actual ~0.87)
- No temperature, clipping, Platt, or isotonic method improves either model
- Models are already near-optimally calibrated for small dataset

### Updated Women's blend: 70/30 → Brier 0.1386 (v4 submission generated)

---

## 2026-03-16 21:20 - Kaggle Stage 2 Submission

Submitted `submission_stage2_v3_20260316_135520.csv` to Kaggle. Public score shows 0.0 — expected, as Kaggle confirms: "Your 2026 submissions will score 0.0 if you have submitted predictions in the right format. The leaderboard will be meaningful once the 2026 tournaments begin and Kaggle rescores your predictions."

**Submission quality check:**
- 132,133 total rows (all 2026 matchups)
- 4,556 matchups between seeded teams (the only ones that get scored)
- 4,550/4,556 seeded matchups have real predictions (99.9%)
- Only 6 seeded matchups at 0.5 default (minor missing features)
- Mean=0.5015, Std=0.0650 for all rows; Std=0.3476 for seeded matchups (good spread)

**Negative results from Day 1 (Experiments 5-8):**
- Regular season inclusion with 6x tournament weighting: hurts (0.2143 vs 0.1944) — domain shift too large
- Probability shrinkage toward 0.5: marginal/negative for well-calibrated models
- Custom Brier loss for LightGBM: implementation issues, abandoned
- XGBoost shallow trees: no improvement over LGB (0.1609 vs 0.1543)
- Random seed ensemble: no improvement — LGB is stable across seeds
- Deep features don't add ensemble diversity over existing Torvik models

---

## 2026-03-16 14:00 - Experiment 8: Shallow Trees + Iterative Efficiency (MAJOR BREAKTHROUGH)

### Discovery: Shallow Trees Massively Outperform Deep Trees

While testing Brier-optimized objectives, accidentally discovered that **shallow LightGBM trees (depth=3) outperform deep trees (depth=6) by 0.03 Brier points**. This is because:

1. With only ~750 games in training (2015-2025 tournaments), deep trees memorize noise
2. Shallow trees (2-3 splits) learn simple, generalizable rules like "better Torvik rating + better seed = higher win probability"
3. Combined with slow learning rate (0.02-0.03), this acts as strong regularization

### Shallow LGB Results (Men's, Torvik features)

| Config | Brier |
|--------|-------|
| LGB d=6 n=500 lr=0.05 (previous default) | 0.1848 |
| LGB d=4 n=300 lr=0.03 | 0.1565 |
| LGB d=3 n=300 lr=0.03 | 0.1551 |
| **LGB d=3 n=400 lr=0.02** | **0.1543** |
| LGB d=2 n=500 lr=0.03 | 0.1588 |

### Iterative Opponent-Adjusted Efficiency (Our KenPom Clone)

Built an iterative efficiency calculator that:
1. Computes raw OE/DE per game from box scores
2. Iteratively adjusts for opponent quality (15 iterations)
3. Re-centers to preserve league averages

Results:
- **Women's**: iter_eff LR C=0.1 = **0.1411** (vs 0.1489 seeds-only)
- **Women's combined**: iter_eff(28%) + tier2_LR(72%) = **0.1387**
- Men's: 0.1918 (worse than Torvik's 0.1703 — Torvik has better adjustments)

---

## 2026-03-16 13:00 - Experiment 7: Bart Torvik External Ratings (BREAKTHROUGH)

### Motivation

Research showed that past Kaggle winners universally cite external power ratings (KenPom, Sagarin, Torvik) as their most impactful features. Bart Torvik T-Rank provides free adjusted efficiency metrics similar to KenPom.

### Data

Downloaded Torvik team results CSVs for 2015-2026 from barttorvik.com. Each contains ~350 D1 teams with:
- **adjoe**: Adjusted offensive efficiency (points per 100 possessions, opponent-adjusted)
- **adjde**: Adjusted defensive efficiency
- **barthag**: Power rating (win probability against average team)
- **adjt**: Adjusted tempo
- **sos**: Strength of schedule

Team name matching: fuzzy matching + manual overrides. Matched 349-357/381 Kaggle teams per season (all tournament teams covered).

**Technical challenge**: Torvik CSVs pre-2023 have a header/data column count mismatch (`"Fun Rk, adjt"` quoted header vs two separate data columns), causing pandas to shift all columns. Fixed by reading by position instead of column name.

### Results (Men's, trained on 2015-2025, CV on 2022-2025)

| Config | Brier | Notes |
|--------|-------|-------|
| seeds_only LR (baseline, 2015+) | 0.1982 | Slightly worse than full history (0.1944) |
| tier1 LR C=0.01 (seeds + ordinals) | 0.1977 | |
| **torvik LR C=0.01** | **0.1843** | Seeds + ordinals + Torvik features |
| **torvik LR C=0.1** | **0.1719** | Less regularization helps with good features |
| **torvik LR C=1.0** | **0.1703** | Even better — Torvik features are high-signal |
| **torvik LGB** | **0.1848** | LGB overfits more than LR |
| tier2 LR C=0.01 (our features) | 0.1971 | Elo/stats much worse than Torvik |
| all LR C=0.01 (everything + Torvik) | 0.1870 | Too many features dilute signal |

### Best Ensembles

| Ensemble | Brier |
|----------|-------|
| torvik_LR_1 (62%) + torvik_LGB (38%) | **0.1610** |
| torvik_LR_01 (60%) + torvik_LGB (40%) | 0.1615 |
| torvik_LR_001 (50%) + torvik_LGB (50%) | 0.1643 |
| seeds_LR (43%) + torvik_LGB (57%) | 0.1656 |
| **All 9 models (best 3: seeds_LR 11%, torvik_LR 50%, torvik_LGB 39%)** | **0.1607** |

### Why This Is Such a Big Improvement

1. **Expert-calibrated features**: Torvik's adjusted efficiency metrics represent years of domain expertise in opponent adjustment, tempo normalization, and schedule strength correction. Our homegrown Elo and season stats can't compete.

2. **Opponent adjustment**: Raw PPG/FG% don't account for schedule difficulty. Torvik's AdjOE and AdjDE normalize for opponent quality, which is critical for cross-conference tournament matchups.

3. **Barthag**: This single feature (estimated probability of beating an average D1 team) is directly analogous to what we're trying to predict, making it an extremely efficient input for the model.

4. **Less regularization is better**: With high-quality features, the model can use more of the signal (C=1.0 beats C=0.01). This is the opposite of our findings with homegrown features, where more regularization was needed to prevent overfitting on noise.

---

## 2026-03-16 12:00 - Experiment 6: Feature Improvements + Meta-Ensemble

### Research Summary

Surveyed past Kaggle March Madness winners (2019-2025) and advanced approaches:
- **Winning Brier scores**: ~0.10-0.11 (combined M+W). Our 0.19 M-only is not yet competitive.
- **Winners used**: gradient-boosted trees + LR ensembles on **external ratings** (KenPom, Sagarin, Torvik). Deep learning has never won.
- **Key tricks**: tournament game weighting (6x), probability shrinkage, isotonic calibration on OOF
- **External data**: Bart Torvik T-Rank (free KenPom alternative), Massey Composite, ESPN BPI

### Changes Implemented

1. **Better ordinal aggregation**: Added OrdinalMedian, OrdinalMin, OrdinalMax (beyond just Mean/Std)
2. **Interaction features**: seed_ordinal_agreement, seed_ordinal_mismatch, elo_ordinal_agreement
3. **Sample weight support**: Tournament games can be weighted 6x vs regular season
4. **Probability shrinkage**: `pred = pred * (1 - alpha) + 0.5 * alpha`

### Results: Tournament Game Weighting (NEGATIVE)

Including regular season games with 6x tournament weighting **hurt badly**:
- seeds_only + RS (6x weight): Brier 0.2143 (vs 0.1944 baseline)
- The domain shift between regular season and tournament is too large. 6x weighting isn't enough to overcome it.

### Results: Shrinkage (MARGINAL/NEGATIVE)

For seed-only LR, shrinkage doesn't help — already well-calibrated:
- shrink=0.05: 0.1944 (same), shrink=0.10: 0.1946 (worse), shrink=0.15: 0.1952 (worse)

### Results: New Features (POSITIVE)

tier1 (seeds + improved ordinals) with C=0.01: **Brier 0.1938** (vs 0.1944 baseline LR)

### Results: Meta-Ensemble (NEW BEST!)

The breakthrough came from ensembling models trained on different feature sets:

**Men's Best: 0.1904** (seeds_LGB 43% + tier2_LR 57%)
- LightGBM on seeds makes different errors than LR on tier2 features
- Blending captures both calibration (from seeds LGB) and accuracy (from tier2 LR)
- Previous best was 0.1915 — improvement of 0.0011

**Women's Best: 0.1390** (tier2_LR C=0.1, 83% weight in ensemble)
- Massive improvement from 0.1489 — improvement of 0.0099!
- Elo + season stats are very predictive for women's basketball
- Seeds-only approach was leaving signal on the table
- Key: strong regularization (C=0.1) prevents overfitting

### Key Insight

The meta-ensemble works because different feature sets produce predictions with **different error patterns**. Even when one model is worse on its own, it can improve the ensemble by reducing correlated errors. The seeds-only LGB is critical not because it's the best model, but because its errors are most *different* from the feature-rich LR models.

---

## 2026-03-16 11:30 - Submissions Generated

Four submission files generated for both stages:

| Stage | Strategy | Rows | Mean Pred | Std Pred |
|-------|----------|------|-----------|----------|
| 1 (Historical) | conservative (seed ensemble) | 519,144 | 0.5011 | 0.0519 |
| 1 (Historical) | aggressive (30/70 blend) | 519,144 | 0.5011 | 0.0514 |
| 2 (2026) | conservative (seed ensemble) | 132,133 | 0.5017 | 0.0517 |
| 2 (2026) | aggressive (30/70 blend) | 132,133 | 0.5018 | 0.0526 |

All submissions passed validation. Mean predictions near 0.5 and narrow std (~0.05) reflect the well-calibrated, conservative nature of seed-based predictions.

---

## 2026-03-16 11:15 - Experiment 5: Team Embeddings via Neural Network (NEGATIVE RESULT)

### Hypothesis
Hand-engineered features (seeds, ordinals, Elo) are a bottleneck. A neural network trained on regular season game outcomes could learn latent team representations that capture strength, style, and dynamics that manual features miss.

### Setup
- **Architecture**: PyTorch embedding model
  - Team embedding layer: n_teams × 16 dimensions
  - Prediction head: Linear(32→64) → ReLU → Dropout(0.3) → Linear(64→32) → ReLU → Dropout(0.2) → Linear(32→1) → Sigmoid
  - Input: concatenation of two team embeddings (team A, team B)
- **Training**: BCE loss, Adam optimizer (lr=0.001, weight_decay=1e-4), 50 epochs, batch size 256
- **Data**: Regular season compact results from 5 most recent seasons (~25K games)
- **Evaluation**: Embedding features (Emb_0 through Emb_15) added to matchup features, tested with LR via leave-season-out CV

### Results (Men's)

| Config | Brier | Accuracy | ECE |
|--------|-------|----------|-----|
| seeds_only ensemble (baseline) | **0.1915** | 68.3% | 0.056 |
| Embeddings only (16-dim) | 0.2269 | 62.1% | 0.142 |
| Seeds + Embeddings | 0.2018 | 67.5% | 0.098 |

### Why It Failed

1. **Regular season ≠ tournament**: Embeddings learned from regular season game outcomes don't capture "tournament readiness." Teams play different competition levels in the regular season (conference games vs non-conference), and tournament play involves different dynamics (neutral sites, single elimination pressure, preparation time).

2. **Overfitting on small data**: With only ~350 teams per season, 16-dim embeddings have 5,600+ parameters for the embedding layer alone. The model likely memorized regular season patterns rather than learning generalizable representations.

3. **Information already captured**: The signal that embeddings could learn (team strength relative to opponents) is already well-captured by Massey ordinals, which aggregate 197 expert ranking systems. The ordinals are essentially a crowdsourced version of what we're trying to learn with a single neural network.

4. **Calibration degradation**: ECE jumped from 0.056 to 0.142, confirming the model produces overconfident predictions — the hallmark of overfitting on this small dataset.

### Takeaway

Simple neural embeddings from regular season data don't add value over expert ranking systems (ordinals). If pursuing deep learning for this problem, would need:
- **Pre-training on much larger dataset** (multiple sports, historical data back to 1985)
- **Game-level features as input** (not just team IDs) — e.g., box score sequences
- **Tournament-specific fine-tuning** to bridge the regular season → tournament domain gap
- **External data** (KenPom ratings, Sagarin, etc.) as additional training signal

---

## 2026-03-16 10:55 - Experiment 4: Model Blending

### Hypothesis
Blending a well-calibrated seed-only model with a feature-rich model should combine the calibration of the former with the accuracy of the latter.

### Setup
- Three base models: seed-only LR, minimal 3-feature LR (C=0.01), full 34-feature LR (C=0.01)
- Tested weighted blends at various ratios

### Results (Men's, Best Blends)

| Blend | Brier | Accuracy | ECE |
|-------|-------|----------|-----|
| 30% seed + 70% minimal | **0.1921** | 72.4% | 0.066 |
| 40% seed + 40% min + 20% full | 0.1922 | 71.6% | 0.065 |
| 40% seed + 60% minimal | 0.1922 | 72.0% | 0.075 |
| seed-only ensemble (Exp 1) | **0.1915** | 68.3% | 0.056 |

### Takeaway

Blending improves on individual feature-rich models (0.1921 vs 0.1925), but the **original seeds_only ensemble (LR 50% + LightGBM 50%) at 0.1915 remains our best Men's configuration**.

The ensemble of simple models beats a single complex model. This is consistent with the "diversity of errors" principle — LR and LightGBM make different types of mistakes on seed-only features, and averaging smooths them out.

---

## 2026-03-16 10:50 - Experiment 3: Feature Selection + Regularization Sweep

### Hypothesis
The key to improving on seed-only is finding the minimal set of features that adds signal without overfitting. Heavy regularization (low C) should help.

### Setup
- Tested 7 feature subsets × 5 C values = 35 configurations
- All using LogisticRegression with StandardScaler
- Leave-season-out CV on [2022-2025]

### Results (Top Configurations)

| Features | C | Brier | Accuracy | ECE |
|----------|---|-------|----------|-----|
| Seed_diff + OrdinalMean_diff + Elo_diff | 0.01 | **0.1925** | 71.3% | 0.077 |
| Seed_diff + POM + SAG + Elo | 0.01 | 0.1927 | 71.3% | 0.065 |
| Seed_diff + OrdinalMean + Elo + AdjEM + POM | 0.1 | 0.1934 | 70.9% | 0.074 |
| Seed_diff only | 0.01 | 0.1956 | 70.2% | 0.066 |
| Seed_diff only | 0.1 | 0.1951 | 70.2% | 0.071 |

### Key Finding

**Seed_diff + OrdinalMean_diff + Elo_diff with C=0.01** is the sweet spot for a single LR model: 0.1925 Brier. This is better than any single model from Experiment 1, but still worse than the seeds_only *ensemble* (0.1915).

---

## 2026-03-16 10:45 - Experiment 2: Isotonic Calibration (NEGATIVE RESULT)

### Hypothesis
Adding more features improves accuracy but hurts calibration (ECE). Isotonic regression calibration should fix this and let us benefit from the extra features.

### Setup
- Nested calibration within leave-season-out CV: train isotonic regression on training fold's OOF predictions, apply to validation predictions
- Tested across all 4 feature sets

### Results (Men's, LR only)

| Feature Set | Uncalibrated Brier | Calibrated Brier | Delta |
|-------------|-------------------|-----------------|-------|
| seeds_only | 0.1944 | 0.1966 | **+0.0022 (worse)** |
| tier1 | 0.1979 | 0.2015 | **+0.0036 (worse)** |
| tier2 | 0.1968 | 0.1977 | **+0.0009 (worse)** |
| all | 0.1985 | 0.2014 | **+0.0029 (worse)** |

### Why It Failed

Isotonic regression needs sufficient data to learn a reliable mapping from predicted → calibrated probabilities. With only ~2500 training games and 4 CV folds of 67 games each, the calibrator doesn't have enough data points to fit well — it ends up overfitting to training noise rather than improving calibration.

### Takeaway

Post-hoc calibration is not viable with this dataset size. Instead, we need models that are inherently well-calibrated. Logistic regression with strong regularization (low C) is the best approach.

---

## 2026-03-16 10:30 - Experiment 1: Baseline Feature Set Comparison

### Setup
- Leave-season-out CV on [2022, 2023, 2024, 2025]
- Models: Logistic Regression, XGBoost (GPU), LightGBM (GPU)
- Ensemble: scipy-optimized weighted average

### Results (Men's)

| Feature Set | Features | Brier | Accuracy | ECE | Best Model |
|-------------|----------|-------|----------|-----|------------|
| seeds_only | 3 | **0.1915** | 68.3% | 0.056 | LR+LGB 50/50 |
| tier1 (seeds+ordinals) | 15 | 0.1944 | 71.6% | 0.090 | LR 71% |
| tier2 (seeds+ord+elo+stats) | 22 | 0.1951 | 71.6% | 0.086 | LR 78% |
| all (+ efficiency + 4factors) | 34 | 0.1950 | 73.1% | 0.102 | LR 65% |

### Results (Women's)

| Feature Set | Features | Brier | Accuracy | ECE | Best Model |
|-------------|----------|-------|----------|-----|------------|
| seeds_only | 3 | **0.1489** | 77.2% | 0.067 | LR 100% |

### Key Insights

1. **More features = higher accuracy but WORSE Brier score.** This is the calibration paradox:
   - Seeds-only: best calibrated (ECE 0.056), best Brier
   - All features: best accuracy (73.1%), worst calibration (ECE 0.102)
   - Adding features helps pick the right winner more often, but makes predictions overconfident

2. **Logistic regression dominates** across all feature sets (65-100% ensemble weight). Tree models (XGBoost, LightGBM) consistently underperform on Brier — they produce more extreme predictions on this small dataset (~67 tournament games/year).

3. **Women's tournament is highly seed-predictable:**
   - 1v16 Brier: 0.0006 (essentially perfect)
   - Only 10.6% upset rate vs 22.4% for men's
   - Pure seed model achieves Brier 0.1489 — very strong

4. **Upset detection is poor across all models** (3-14%). Models almost never predict an upset, which is actually correct Brier strategy — predicting P=0.3 for a game with 22% upset rate is closer to optimal than predicting P=0.5.

5. **Season 2025 is consistently easiest** (lowest Brier across all configs), suggesting a "chalk" year. Season 2022 is hardest.

6. **Elite 8 is the hardest round** (Brier ~0.27-0.28). By this stage, remaining teams are all strong, making matchups harder to predict.

---

## 2026-03-16 10:00 - Exploratory Data Analysis

### Dataset Overview

28 datasets loaded from Kaggle (35 CSV files total, some not mapped):

| Category | Men's | Women's |
|----------|-------|---------|
| Teams | 381 | 379 |
| Regular Season (compact) | 198,577 games (1985-2026) | 142,507 games (1998-2026) |
| Regular Season (detailed) | 124,529 games (2003-2026) | 87,187 games (2010-2026) |
| Tournament (compact) | 2,585 games | 1,717 games |
| Tournament (detailed) | 1,449 games | 961 games |
| Seeds | 2,694 entries | 1,812 entries |
| Massey Ordinals | 5,865,001 rows (197 systems!) | separate file |

**Key data gaps:** Women's detailed results only go back to 2010 (vs 2003 for men's). This limits efficiency/four-factors features for women.

### Historical Upset Rates (Men's, 2003-2025)

| Matchup | Games | Upsets | Rate |
|---------|-------|--------|------|
| 1 vs 16 | 88 | 2 | **2.3%** |
| 2 vs 15 | 88 | 7 | 8.0% |
| 3 vs 14 | 88 | 10 | 11.4% |
| 4 vs 13 | 88 | 18 | 20.5% |
| 5 vs 12 | 88 | 34 | **38.6%** |
| 6 vs 11 | 88 | 39 | **44.3%** |
| 7 vs 10 | 87 | 33 | 37.9% |
| 8 vs 9 | 88 | 44 | **50.0%** |

5-12 and 6-11 matchups are the most upset-prone (besides the coin-flip 8-9). This is the "Cinderella zone" where mid-majors often have strong teams underseeded.

### Massey Ordinals Coverage

197 distinct ranking systems! Top systems with 24 seasons of coverage: AP, DOL, COL, MOR, POM, USA, WLK. SAG has 21 seasons.

Late-season rankings (DayNum ≥ 128) are most stable and predictive.

### Submission Requirements

- **Stage 1**: 519,144 matchups (historical validation across 4 seasons × 2 genders × all team pairs)
- **Stage 2**: 132,133 matchups (2026 season predictions)
