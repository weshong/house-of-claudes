# Experiment Journal - March ML Mania 2026

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

### Games Per Season

![Games per season](eda_games_per_season.png)

Men's average ~5,000 games/season. Women's slightly less. Notable dips in 2020-2021 due to COVID.

### Score Distributions (2020+)

![Score distributions](eda_score_distribution.png)

Men's average score ~72 points. Women's slightly lower. Both roughly normal distributions.

### Tournament Win Rate by Seed

![Seed win rates](eda_seed_win_rate.png)

Clear monotonic relationship: 1-seeds win ~85% of games, 16-seeds win ~15%. The gap is much more pronounced in women's basketball.

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

### Calibration Plots

![Calibration plot](calibration.png)
![Prediction distribution](pred_distribution.png)

### Per-Round Brier (Men's, seeds_only)

| Round | Brier | Accuracy | Games |
|-------|-------|----------|-------|
| R64 | 0.244 | 56.3% | 16 |
| R32 | 0.170 | 70.3% | 128 |
| S16 | 0.173 | 73.4% | 64 |
| E8 | **0.258** | 59.4% | 32 |
| F4 | 0.223 | 50.0% | 8 |
| Championship | 0.144 | 100% | 4 |

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

### Next Steps

- Apply isotonic regression calibration to feature-rich models
- Feature selection: seed_diff + OrdinalMean_diff + Elo_diff may be sufficient
- Use seeds-only for women, focus feature engineering on men
- Consider restricting training window to recent seasons (2010+)

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

Strong regularization (C=0.01) is critical — it prevents the model from overweighting noisy features.

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

### Strategy Decision

For final submission:
- **Men's "conservative"**: seeds_only ensemble (Brier 0.1915)
- **Men's "aggressive"**: 30% seed LR + 70% minimal LR blend (Brier 0.1921, but higher accuracy 72.4%)
- **Women's**: seeds_only LR (Brier 0.1489)

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

## Brier Score Leaderboard (Best Configurations)

| Gender | Config | Brier | Accuracy | Timestamp |
|--------|--------|-------|----------|-----------|
| M | seeds_only ensemble (LR+LGB) | **0.1915** | 68.3% | 2026-03-16 10:38 |
| M | 30% seed + 70% minimal blend | 0.1921 | 72.4% | 2026-03-16 10:55 |
| M | minimal 3-feat LR (C=0.01) | 0.1925 | 71.3% | 2026-03-16 10:50 |
| W | seeds_only LR | **0.1489** | 77.2% | 2026-03-16 10:39 |
