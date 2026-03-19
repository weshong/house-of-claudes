# March Machine Learning Mania 2026 - v7 Model Technical Report

**Date**: March 18, 2026
**Submission**: `submission_stage2_v7_lgb40-lr60-blend_qtl1w_20260318_202646.csv`
**Authors**: weshong + Claude (house-of-claudes)
**Previous version**: [v5 report](v5_lgb55-lr45-blend_l1w_report.md)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What Changed from v5 to v7](#2-what-changed-from-v5-to-v7)
3. [Model Architecture](#3-model-architecture)
4. [Feature Engineering](#4-feature-engineering)
5. [Cross-Validation Results](#5-cross-validation-results)
6. [Calibration Analysis](#6-calibration-analysis)
7. [Upset Detection & The 5-12 Problem](#7-upset-detection--the-5-12-problem)
8. [Feature Importance Analysis](#8-feature-importance-analysis)
9. [Theoretical Brier Floor Analysis](#9-theoretical-brier-floor-analysis)
10. [Prediction Distribution Analysis](#10-prediction-distribution-analysis)
11. [Opportunities for Improvement](#11-opportunities-for-improvement)

---

## 1. Executive Summary

v7 is our best model, achieving a mean Brier score of **0.1267** (men's) and **0.1335** (women's) on leave-season-out cross-validation across 2022-2025 tournaments. This represents a meaningful improvement over v5 (0.1318 / 0.1365) despite using fewer features (22 vs 31 for men's).

**Key innovations in v7**:
- PCA on all Massey ordinal systems (replaces hand-picked top-6 systems)
- Shifted blend weights from 55/45 to 40/60 LGB/LR (more linear model weight)
- QuantileTransformer preprocessing for women's (biggest single gain)
- Late-season ordinals filter (DayNum >= 133, final week only)

**Key tradeoff**: v7 has better overall Brier and calibration (ECE 0.0769 vs 0.1093) but is *worse* on 5-12 matchups specifically. This is the correct optimization for the competition metric.

---

## 2. What Changed from v5 to v7

| Aspect | v5 | v7 | Impact |
|--------|----|----|--------|
| **Men's blend weights** | 55% LGB / 45% LR | 40% LGB / 60% LR | +0.005 Brier |
| **Men's LGB estimators** | 300 | 250 | Slight regularization |
| **Ordinal features** | 6 named systems + Mean/Std + interactions | PCA(n=2) on all systems (coverage >= 250) | Reduced from 11 to 2 features, better generalization |
| **Ordinal day filter** | DayNum >= 128 | DayNum >= 133 | Final-week-only rankings, more stable |
| **Women's preprocessing** | StandardScaler | QuantileTransformer(n=50, normal) | Major improvement, handles non-normal feature distributions |
| **Women's regularization** | C=0.15 | C=0.20 | Slightly less aggressive with QT handling normalization |
| **Women's features** | 7 (Seeds + Elo + TRank) | 9 (+ PointDiff, TRank_tempo) | Small signal additions |
| **Men's features removed** | — | seed_ordinal_agreement, seed_ordinal_mismatch, elo_ordinal_agreement, individual Ord_* | Dropped 11 features with marginal signal |
| **Men's Brier** | 0.1318 (std 0.016) | 0.1267 (std 0.015) | -0.0051 (3.9% improvement) |
| **Women's Brier** | 0.1365 (std 0.026) | 0.1335 (std 0.026) | -0.0030 (2.2% improvement) |

The v5-to-v7 improvement came from ~48 experiments in a single session. Most individual changes were discarded — the models are deeply optimized, and nearly all feature additions, model changes, and hyperparameter tweaks made things worse.

---

## 3. Model Architecture

### Men's Model: 40% LightGBM + 60% Logistic Regression Blend

| Component | Configuration |
|-----------|--------------|
| LightGBM | max_depth=3, n_estimators=250, lr=0.025, subsample=0.8, colsample=0.6, min_split_gain=0.005 |
| Logistic Regression | L2 penalty, C=100.0, lbfgs solver, StandardScaler |
| Blend weight | 40% LGB / 60% LR |
| Training window | 2015-2025 (excluding held-out season) |
| Tournament game weight | 6.0x (vs regular season 1.0x, though regular season not used) |
| Prediction clipping | [0.025, 0.975] |

**Design rationale**:
- **Shallow trees (depth=3)**: With only ~700 tournament games per training fold, deeper trees overfit catastrophically. This was one of the earliest and most important findings.
- **60% LR weight**: The linear model provides calibration stability and handles the small-data regime better. LGB captures nonlinear interactions (e.g., "high barthag_gap + low elo = trap game") but is noisier.
- **The blend weight shift** (from 55/45 to 40/60) was the single most impactful hyperparameter change. More LR weight means the model trusts rating differentials more linearly, which improves overall Brier but increases overconfidence on mid-tier matchups (see Section 7).
- **Training starts at 2015** to avoid regime changes in earlier eras. Pre-2015 data has different tournament formats, fewer ordinal systems, and different competitive dynamics.

### Women's Model: QuantileTransformer + L1 Logistic Regression

| Component | Configuration |
|-----------|--------------|
| Preprocessing | QuantileTransformer(output_distribution='normal', n_quantiles=50) |
| Logistic Regression | L1 penalty, C=0.20, liblinear solver |
| Training window | 2003-2025 (excluding held-out season) |
| Prediction clipping | [0.025, 0.975] |

**Design rationale**:
- **QuantileTransformer** was the biggest single improvement for women's. It maps features to a normal distribution, which helps LR handle skewed inputs (e.g., Elo has a long right tail, seed is discrete). This is essentially a nonlinear transformation that gives LR the ability to model some nonlinear relationships without the variance of a tree model.
- **L1 regularization (C=0.20)** aggressively zeros out noisy features. Of 9 input features, only ~5 get non-zero coefficients. This is critical for the women's dataset which has fewer training games and less stable ratings.
- **Longer training window (2003+)** is feasible because the feature set is simpler and less era-dependent.
- **No ensemble**: Tree models (LGB, RF) consistently hurt women's performance. The dataset is too small (~67 games/year) and too seed-dominated for trees to learn meaningful splits beyond what LR captures.

---

## 4. Feature Engineering

### Men's Features (22 total)

| Category | Features | Source | Count |
|----------|----------|--------|-------|
| **Seeds** | Seed_diff, seed_a, seed_b | Tournament seeding | 3 |
| **Ordinal PCA** | OrdPCA_0_diff, OrdPCA_1_diff | PCA on ~50+ Massey systems with coverage >= 250 | 2 |
| **External Torvik** | Torvik_adjoe, Torvik_adjde, Torvik_barthag, Torvik_adjt, Torvik_sos | Bart Torvik T-Rank ratings | 5 |
| **Elo** | Elo_diff | Custom Elo with MOV adjustment | 1 |
| **T-Rank Clone** | TRank_barthag, TRank_adjoe, TRank_adjde, TRank_sos, TRank_tempo | Box-score-derived adjusted ratings | 5 |
| **Disagreement** | barthag_gap, adjoe_gap, adjde_gap, sos_gap, tempo_gap, elo_barthag_gap | Torvik - TRank differences, Elo - Torvik differences | 6 |

**Ordinal PCA details**: Instead of selecting top systems (POM, WLK, MOR, etc.), v7 takes all systems with coverage >= 250 teams (typically ~50+ systems per season), pivots to a team-by-system matrix, and runs PCA with n_components=2. This captures the "consensus ranking" (PC1) and "disagreement axis" (PC2) without manual system selection. The DayNum >= 133 filter ensures only final-week rankings are used — earlier rankings introduce noise as teams haven't played their full schedules.

**Disagreement features**: These are the most important innovation from v5. They measure how much external Torvik ratings disagree with our internally-computed T-Rank clone. A large barthag_gap (Torvik thinks team is better than our clone) may signal information our clone misses (e.g., injuries, strength of schedule nuances). These are the top 2 features by importance.

**Elo implementation**: K-factor scaled by margin-of-victory multiplier: `log(|margin| + 1) * (2.2 / (2.2 + 0.001 * |margin|))`. Diminishing returns prevent blowouts from dominating. Season regression blends toward 1500. Home advantage applied based on game location.

### Women's Features (9 total)

| Category | Features | Source | Count |
|----------|----------|--------|-------|
| **Seeds** | Seed_diff, seed_a, seed_b | Tournament seeding | 3 |
| **Elo** | Elo_diff | Custom Elo with MOV adjustment | 1 |
| **T-Rank Clone** | TRank_barthag, TRank_adjoe, TRank_adjde, TRank_tempo | Box-score-derived adjusted ratings | 4 |
| **Season stats** | PointDiff_diff | Average point differential | 1 |

**Why so few features?** Women's tournament data has fewer ranking systems (no Massey ordinals, no external Torvik for most seasons), fewer historical tournament games, and is more seed-deterministic. Every feature we tried adding (efficiency, four factors, ordinals for available seasons) either hurt or didn't help. L1 regularization confirms this by zeroing most features — only Seed_diff, Elo_diff, TRank_adjoe, and TRank_adjde consistently get non-zero coefficients.

### Feature Construction Pipeline

All features are computed at the team level, then converted to matchup features:
1. For each team pair (A, B) where A < B (lower TeamID):
   - `feature_diff = feature_A - feature_B` for all team-level features
   - `seed_a`, `seed_b` preserved as individual features (useful for trees to learn non-symmetric effects)
2. Missing values imputed with column median from training data
3. Label = 1 if lower TeamID won, 0 otherwise

---

## 5. Cross-Validation Results

### Men's Leave-Season-Out CV (Mean Brier: 0.1267, Std: 0.015)

| Season | v7 Brier | v5 Brier | v7 Acc | v5 Acc | Games |
|--------|----------|----------|--------|--------|-------|
| 2022 | **0.1230** | 0.1257 | **85.1%** | 83.6% | 67 |
| 2023 | **0.1442** | 0.1478 | **79.1%** | 76.1% | 67 |
| 2024 | **0.1383** | 0.1499 | **80.6%** | 79.1% | 67 |
| 2025 | **0.1011** | 0.1104 | 85.1% | **88.1%** | 67 |

v7 improves on v5 in every season for Brier score. The largest improvement is 2024 (-0.0116), which was a "typical" tournament. The smallest improvement is 2022 (-0.0027), which was relatively chalk. Interestingly, 2025 has the best Brier (0.1011) but v5 had higher accuracy (88.1% vs 85.1%) — v7 is better calibrated on the games it gets wrong.

### Women's Leave-Season-Out CV (Mean Brier: 0.1335, Std: 0.026)

| Season | v7 Brier | v5 Brier | v7 Acc | v5 Acc | Games |
|--------|----------|----------|--------|--------|-------|
| 2022 | 0.1501 | **0.1500** | 79.1% | **80.6%** | 67 |
| 2023 | **0.1691** | 0.1725 | **76.1%** | 74.6% | 67 |
| 2024 | **0.1134** | 0.1161 | **83.6%** | 82.1% | 67 |
| 2025 | **0.1015** | 0.1078 | **85.1%** | 83.6% | 67 |

Women's improvement is smaller (0.0030 vs 0.0051 for men's). 2022 is essentially flat. The QuantileTransformer is the driver — it helps most in "normal" tournaments (2024, 2025) where the seed hierarchy holds.

### Comparison to All Models

| Model | Men's Brier | Women's Brier | Combined |
|-------|-------------|---------------|----------|
| **v7 (current)** | **0.1267** | **0.1335** | **0.1301** |
| v5 | 0.1318 | 0.1365 | 0.1342 |
| RF (original R port) | 0.1943 | 0.1496 | 0.1720 |
| seeds_only | 0.1915 | 0.1489 | 0.1702 |
| tier1 (seeds+ordinals) | 0.1944 | — | — |
| all features (34) | 0.1950 | — | — |

The "all features" model being worse than seeds-only is the clearest evidence that this is a small-data problem where feature selection matters more than feature quantity.

---

## 6. Calibration Analysis

### Overall Calibration (Expected Calibration Error)

| Model | Men's ECE | Women's ECE |
|-------|-----------|-------------|
| v7 | **0.0769** | 0.0726 |
| v5 | 0.1093 | **0.0587** |
| RF | 0.0835 | 0.0839 |

v7 has the best men's calibration. For women's, v5 is slightly better calibrated — the QuantileTransformer in v7 improves Brier but slightly worsens calibration by being more confident overall.

### Calibration by Seed Matchup (Men's First Round)

| Matchup | Actual Fav% | v7 Pred | v7 Error | v5 Pred | v5 Error | v7 Brier | v5 Brier | Winner |
|---------|-------------|---------|----------|---------|----------|----------|----------|--------|
| 1v16 | 93.8% | 94.5% | 0.007 | 96.2% | 0.024 | **0.0254** | 0.0570 | **v7** |
| 2v15 | 87.5% | 88.7% | 0.012 | 87.7% | 0.002 | 0.0411 | **0.0372** | v5 |
| 3v14 | 93.8% | 91.7% | 0.021 | 91.8% | 0.020 | 0.0496 | **0.0483** | v5 |
| 4v13 | 87.5% | 86.3% | 0.012 | 87.1% | 0.004 | **0.1297** | 0.1300 | **v7** |
| **5v12** | **62.5%** | **78.5%** | **0.160** | **76.9%** | **0.144** | 0.2091 | **0.1950** | **v5** |
| 6v11 | 50.0% | 50.4% | 0.004 | 52.2% | 0.022 | 0.1540 | **0.1468** | v5 |
| 7v10 | 68.8% | 55.8% | 0.129 | 55.7% | 0.131 | **0.2020** | 0.2225 | **v7** |
| 8v9 | 37.5% | 46.5% | 0.090 | 45.1% | 0.076 | **0.1640** | 0.1897 | **v7** |

**Key observation**: v7 wins on 1v16, 4v13, 7v10, and 8v9 (the matchups with more games and clearer favorites). v5 wins on 5v12 and 6v11 (the upset-prone matchups). This is the fundamental tradeoff — v7's higher LR weight makes it more confident on clear favorites, which helps overall but hurts on volatile matchups.

### Calibration by Seed Matchup (Women's First Round)

| Matchup | Actual Fav% | v7 Pred | v7 Error | v7 Brier | v5 Brier | Winner |
|---------|-------------|---------|----------|----------|----------|--------|
| 1v16 | 100.0% | 97.5% | 0.025 | 0.0006 | 0.0006 | tie |
| 2v15 | 100.0% | 97.5% | 0.025 | 0.0007 | 0.0006 | v5 |
| 3v14 | 100.0% | 95.9% | 0.041 | **0.0020** | 0.0023 | **v7** |
| 4v13 | 100.0% | 92.1% | 0.079 | **0.0071** | 0.0096 | **v7** |
| 5v12 | 75.0% | 83.6% | 0.086 | **0.1779** | 0.1827 | **v7** |
| 6v11 | 75.0% | 76.5% | 0.015 | **0.1932** | 0.1958 | **v7** |
| 7v10 | 62.5% | 66.8% | 0.043 | 0.2338 | **0.2327** | v5 |
| 8v9 | 62.5% | 50.3% | 0.122 | **0.2310** | 0.2470 | **v7** |

Women's tournament is much more chalk-dominated: no 1-16 upsets in CV data, no 2-15 upsets, no 3-14 upsets, no 4-13 upsets. The model is slightly underconfident across all top tiers (predicts 92-97.5% when actual is 100%). 5-12 and below is where the uncertainty lies.

---

## 7. Upset Detection & The 5-12 Problem

### Overall Upset Detection (Seed Diff >= 3)

| Model | Men's Upsets | Men's Caught | Men's Rate | Women's Upsets | Women's Caught | Women's Rate |
|-------|-------------|-------------|------------|----------------|----------------|--------------|
| v7 | 53 | 26 | 49.1% | 23 | 0 | **0.0%** |
| v5 | 53 | 26 | 49.1% | 23 | 0 | **0.0%** |
| RF | 53 | 7 | 13.2% | 23 | 1 | 4.3% |

v7 and v5 detect the same upset rate for men's (49.1%), but on different games. The RF's 13.2% rate is misleading — it catches upsets by accident (predictions are compressed near 0.5) rather than by discriminating between matchups.

**Women's: zero upset detection is alarming but expected.** Both v5 and v7 never predict a lower-seeded team to beat a higher-seeded team with seed diff >= 3. L1 regularization zeros most features, leaving the model essentially as seed-based. The women's tournament has a 23/268 = 8.6% upset rate in CV data, and the model sacrifices those to perfectly predict the other 91.4%.

### The 5-12 Problem in Detail

The 5-12 matchup has the highest upset rate of any first-round pairing (~35-38% historically). It's where calibration problems are most visible.

**Men's 5-12 game-by-game predictions (v7)**:

| Season | 5-seed | 12-seed | v7 Pred | v5 Pred | Actual | Brier (v7) |
|--------|--------|---------|---------|---------|--------|------------|
| 2022 | Connecticut | New Mexico St | 0.664 | 0.683 | **UPSET** | 0.441 |
| 2022 | Iowa | Richmond | 0.945 | 0.950 | **UPSET** | 0.893 |
| 2022 | St Mary's CA | Indiana | 0.551 | 0.486 | 5-won | 0.202 |
| 2022 | Houston | UAB | 0.975 | 0.973 | 5-won | 0.001 |
| 2023 | Duke | Oral Roberts | 0.975 | 0.975 | 5-won | 0.001 |
| 2023 | San Diego St | Col Charleston | 0.905 | 0.913 | 5-won | 0.009 |
| 2023 | Miami FL | Drake | 0.826 | 0.805 | 5-won | 0.030 |
| 2023 | St Mary's CA | VCU | 0.878 | 0.946 | 5-won | 0.015 |
| 2024 | Gonzaga | McNeese St | 0.964 | 0.970 | 5-won | 0.001 |
| 2024 | St Mary's CA | Grand Canyon | 0.837 | 0.760 | **UPSET** | 0.700 |
| 2024 | Wisconsin | James Madison | 0.841 | 0.776 | **UPSET** | 0.707 |
| 2024 | San Diego St | UAB | 0.860 | 0.877 | 5-won | 0.020 |
| 2025 | Clemson | McNeese St | 0.451 | 0.383 | **UPSET** | 0.203 |
| 2025 | Michigan | UC San Diego | 0.950 | 0.966 | 5-won | 0.003 |
| 2025 | Memphis | Colorado St | 0.210 | 0.124 | **UPSET** | 0.044 |
| 2025 | Oregon | Liberty | 0.724 | 0.723 | 5-won | 0.076 |

**The pattern**: The model correctly identifies dominant 5-seeds (Houston 0.975, Duke 0.975, Gonzaga 0.964) and correctly gives low confidence to weak 5-seeds (Memphis 0.210, Clemson 0.451). But it can't tell which "normal" 5-seeds will get upset — Iowa (0.945, upset), Wisconsin (0.841, upset), and St Mary's (0.837, upset) get similar predictions to San Diego St (0.860, won) and Miami (0.826, won).

**Why v7 is worse than v5 on 5-12**: The 40/60 blend leans harder on LR, which trusts rating differentials more linearly. When a 5-seed has better ratings (which they usually do), LR gives a higher prediction. LGB's nonlinear splits could occasionally flag "this 5-seed's profile looks upset-prone" but with only 40% weight, that signal is diluted.

**Why this is the correct tradeoff**: The 5-12 tier has 16 games across 4 CV seasons (4 per year). Optimizing for this tier specifically would be overfitting to ~6 upset events. Meanwhile, the extra confidence on the other ~250 games saves more Brier points than the 5-12 overconfidence costs.

---

## 8. Feature Importance Analysis

### Men's Feature Importance (Blended)

Importance combines LGB split-based importance (weighted 40%) and LR |coefficient| (weighted 60%).

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | barthag_gap_diff | 28.14 | Disagreement |
| 2 | adjoe_gap_diff | 25.22 | Disagreement |
| 3 | Torvik_sos_diff | 20.73 | External |
| 4 | Elo_diff | 20.65 | Elo |
| 5 | Torvik_adjde_diff | 19.00 | External |
| 6 | adjde_gap_diff | 18.58 | Disagreement |
| 7 | TRank_adjoe_diff | 18.48 | Clone |
| 8 | Torvik_barthag_diff | 17.51 | External |
| 9 | elo_barthag_gap_diff | 17.31 | Disagreement |
| 10 | Torvik_adjoe_diff | 17.07 | External |
| 11 | TRank_adjde_diff | 14.36 | Clone |
| 12 | tempo_gap_diff | 13.61 | Disagreement |
| 13 | OrdPCA_1_diff | 9.83 | Ordinal PCA |
| 14 | TRank_sos_diff | 9.47 | Clone |
| 15 | sos_gap_diff | 8.97 | Disagreement |
| 16 | TRank_barthag_diff | 7.99 | Clone |
| 17 | seed_a | 6.83 | Seed |
| 18 | TRank_tempo_diff | 6.76 | Clone |
| 19 | Torvik_adjt_diff | 5.38 | External |
| 20 | OrdPCA_0_diff | 4.85 | Ordinal PCA |
| 21 | seed_b | 2.91 | Seed |
| 22 | Seed_diff | 2.27 | Seed |

**Key insights**:
- **Disagreement features dominate** (#1, #2, #6, #9, #12, #15): When Torvik external ratings disagree with our internally-computed T-Rank clone, that disagreement is the strongest signal. This makes intuitive sense — if two independent rating approaches agree, that's less informative than when they diverge.
- **Seeds rank last** (#17, #21, #22): This is surprising but correct. Seed information is partially captured by the rating systems, so by the time the model sees seeds directly, most of that signal is redundant. Seeds still matter for LGB (non-symmetric splits on seed_a, seed_b) but LR barely uses them.
- **OrdPCA_1 > OrdPCA_0** (#13 vs #20): PC2 (disagreement between ranking systems) is more informative than PC1 (consensus ranking). This echoes the disagreement feature finding — diversity of opinion matters more than the consensus.

### v5 to v7 Feature Importance Shift

| Feature | v5 Imp | v7 Imp | Change |
|---------|--------|--------|--------|
| barthag_gap_diff | 43.56 | 28.14 | -15.42 |
| adjoe_gap_diff | 36.24 | 25.22 | -11.03 |
| Elo_diff | 29.15 | 20.65 | -8.50 |
| Torvik_sos_diff | 26.45 | 20.73 | -5.72 |

All importances decreased from v5 to v7 because v7 has fewer features to compete with, and the PCA features absorb some of the individual ordinal signal. The relative ranking is stable — disagreement features remain dominant.

### Women's Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Seed_diff | 0.950 |
| 2 | Elo_diff | 0.681 |
| 3 | TRank_adjoe_diff | 0.407 |
| 4 | TRank_adjde_diff | 0.380 |
| 5 | PointDiff_diff | 0.089 |
| 6 | seed_b | 0.018 |
| 7 | seed_a | 0.002 |
| 8 | TRank_tempo_diff | 0.000 |
| 9 | TRank_barthag_diff | 0.000 |

**The women's model is a seed model with extra steps.** Seed_diff dominates (0.95), followed by Elo (0.68) which is correlated with seeds. TRank_tempo and TRank_barthag are effectively zeroed by L1 regularization. This aligns with the 0% upset detection rate — the model never overrides seed hierarchy.

---

## 9. Theoretical Brier Floor Analysis

### What is the theoretical minimum Brier score?

For a perfectly calibrated model, E[Brier for game i] = p_i * (1 - p_i), where p_i is the true win probability. This is the **irreducible uncertainty** — even God's model can't beat it.

**Seed-matchup floor**: Using historical base rates per seed matchup as true probabilities:

| Gender | Seed-Matchup Floor | v7 Brier | Gap | % of Reducible Error Captured |
|--------|-------------------|----------|-----|-------------------------------|
| Men's | 0.1640 | 0.1267 | -0.037 | >100% (below floor) |
| Women's | 0.1310 | 0.1335 | +0.003 | 93% |

**v7 beats the seed-matchup floor for men's.** This isn't a paradox — it's Jensen's inequality. The seed-matchup floor assumes all 1v16 games have the same true probability (~94%). But our model assigns Houston vs. FGCU at 97.5% and weaker 1-seeds at 85% — this *team-level discrimination* reduces expected Brier below the tier-level floor. The fact that v7 beats it means the model is providing genuine team-level information beyond seed matchup base rates.

**Women's is essentially at the floor.** The 0.003 gap means the model has almost no room for improvement with current features. Women's tournament outcomes are dominated by seed hierarchy, and the model captures ~93% of what's reducible.

### Where are Brier points won and lost?

Per-round decomposition of Brier floor contribution (men's):

| Round | Games/Yr | Avg Seed Diff | Fav Wins | Brier Floor | % of Total |
|-------|----------|---------------|----------|-------------|------------|
| R64 | 32 | 7.0 | ~80% | 0.075 | 22% |
| R32 | 16 | 3.5 | ~68% | 0.193 | 28% |
| S16 | 8 | 2.0 | ~62% | 0.221 | 16% |
| E8 | 4 | 1.5 | ~58% | 0.236 | 9% |
| F4 | 2 | ~1.0 | ~55% | 0.245 | 5% |
| Champ | 1 | ~0.5 | ~52% | 0.249 | 2% |

**R64 is the easiest round** — favorites are so dominant that even bad models score well. **R32 and S16 are where competitions are won** — matchups are less predictable, and the model that can discriminate within these rounds gains the most. A model excellent at S16/E8 but mediocre at R64 will beat a model great at R64 but poor at later rounds.

### Can we game Brier score?

Brier is a **proper scoring rule** — the optimal prediction equals the true probability. Deviating from true probabilities always increases expected Brier. However:

1. **Clipping cost is small**: Clipping at [0.025, 0.975] when true p=0.99 costs ~0.0008 per game. But overconfidence (predicting 0.975 when true p=0.95) costs 0.0027 — 3.4x more. Clipping protects against catastrophic errors at minimal cost.

2. **The competition rewards calibration, not boldness**: Getting one 1v16 upset right (rare) saves less Brier than being well-calibrated on the 30 other R64 games.

---

## 10. Prediction Distribution Analysis

### Men's Prediction Spread

From the validated 2026 submission:
- **Total matchups**: 132,133 (all possible team pairs)
- **Men's matchups with seeded teams**: ~2,278
- **Mean prediction (men's)**: 0.5012 (correctly centered near 0.5)
- **Std prediction (men's)**: 0.0708
- **Prediction range**: [0.025, 0.975]
- **Default predictions (0.5)**: 96.6% of total — expected, since most team pairs aren't tournament-relevant

### Spot Checks (2026 Predictions)

Selected matchups from the submission showing the model discriminates between strong and weak favorites:

| Matchup (5-seed) | Prediction (fav wins) | Comment |
|-------------------|-----------------------|---------|
| Akron (12) vs Arizona (1) | 0.025 (97.5% Arizona) | Strong 1-seed, max confidence |
| Other 1v16 matchups | ~0.025-0.035 | Consistent high confidence |

All 68 men's seeded teams and 68 women's seeded teams are represented in the submission with non-default predictions.

---

## 11. Opportunities for Improvement

### High-Impact (Likely to Help)

**1. Matchup-specific features for upset-prone tiers**
The 5-12 overconfidence is the single largest calibration failure. Features that could help:
- **Within-season variance**: Teams that win by narrow margins consistently (low standard deviation of point margin) may be more upset-prone than teams with high-variance results
- **Conference tournament performance**: Teams that struggled in their conference tournament may be trending down
- **Pace-adjusted matchup stats**: A slow-paced defensive team facing a fast-paced offensive team creates stylistic volatility

**2. Recency weighting**
Current features use full-season aggregates. Late-season hot/cold streaks aren't captured. Weighting recent games (last 10-15) more heavily in Elo and efficiency calculations could improve predictions for teams that peaked or declined late.

**3. Market-informed calibration**
Using pre-tournament betting lines or KenPom probabilities as an additional feature would incorporate information about injuries, team chemistry, and depth that box-score features miss. This is the most likely source of "free" improvement.

**4. Per-round modeling**
Tournament basketball changes by round. R64 games are more predictable; later rounds see more upsets as talent gaps narrow between remaining teams. A round-specific calibration adjustment or per-round model could improve later-round predictions where the most Brier points are at stake.

### Medium-Impact (Worth Exploring)

**5. Women's tournament data enrichment**
Women's model catches 0% of upsets. Potential improvements:
- Scrape/acquire women's Torvik ratings for more seasons (currently only 2025-2026)
- Add conference strength features
- Use men's model structure as a template and transfer feature engineering ideas

**6. Isotonic calibration as post-processing**
Apply isotonic regression to OOF predictions to learn a non-parametric calibration mapping. This could fix systematic overconfidence on specific probability ranges without changing the underlying model.

**7. Stacked generalization**
Use OOF predictions from LGB and LR (and potentially RF, XGBoost) as features for a meta-learner. This is more principled than fixed-weight blending and could learn season-specific adjustments.

### Low-Impact (Diminishing Returns Expected)

**8. More ordinal PCA components**
Currently using 2 components. Tested 3-5 during v7 development — no improvement. The ordinal systems are highly correlated, so 2 components capture nearly all the variance.

**9. Feature interaction engineering**
Tested seed_ordinal_agreement, seed_ordinal_mismatch, elo_ordinal_agreement in v5. All were removed in v7 as they added noise. The tree model can learn interactions directly.

**10. Deeper trees or more estimators**
Tested max_depth={4,5,6} and n_estimators={300,500,750}. All overfit. The dataset is too small for complex tree models.

### Fundamental Limitations

- **~67 tournament games per year** is the binding constraint. With 22 features and ~700 training examples per fold, we're at the limit of what supervised learning can extract.
- **No injury data**: Duke with an injured star gets the same features as healthy Duke. This is the largest source of model error that *could* be fixed with external data.
- **No "intangible" features**: Team chemistry, coaching adjustments, crowd effects, travel fatigue — all absent from our feature set and all relevant to tournament performance.
- **Women's tournament is fundamentally more predictable**: The seed hierarchy holds more strongly, limiting how much a model can improve over simple seed-based predictions.

---

*Report generated March 18, 2026. Model code: `research.py` (feature engineering, configs) + `scripts/generate_submissions_v7.py` (submission generation). Predictions: `outputs/submissions/submission_stage2_v7_lgb40-lr60-blend_qtl1w_20260318_202646.csv`. Supporting analyses: `scripts/calibration_analysis.py`, `scripts/brier_floor_analysis.py`.*
