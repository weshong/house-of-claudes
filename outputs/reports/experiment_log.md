# Experiment Log

## Leaderboard (Mean Brier, leave-season-out CV 2022-2025)

| Model | Men's | Women's |
|-------|-------|---------|
| **v5: LGB+LR blend / L1 LR** | **0.1318** | **0.1365** |
| seeds_only ensemble | 0.1915 | 0.1489 |
| tier1 (seeds+ordinals) | 0.1944 | — |
| all (full features) | 0.1950 | — |
| tier2 (seeds+elo+stats+ord) | 0.1951 | — |

---

## 2026-03-17 — v5: Best model — LGB+LR blend (M) / L1 LR (W) [CURRENT BEST]

Submitted to Kaggle as `submission_stage2_v5_lgb55-lr45-blend_l1w_20260317_222920.csv`.

### Men's: 55% LightGBM + 45% Logistic Regression blend
- **LGB**: max_depth=3, n_estimators=300, lr=0.025, subsample=0.8, colsample=0.6, min_split_gain=0.005
- **LR**: L2 penalty, C=100.0, lbfgs solver, StandardScaler
- **Training data**: 2015-2025 (excluding test season)
- **Features**: Custom builder — ordinal aggregates (POM, WLK, MOR, DOL, COL, RPI), Torvik ratings, T-Rank clone, Elo, feature gaps/disagreements between rating systems
- **Mean Brier: 0.1318** (std 0.0156)

| Season | Brier | Accuracy | N |
|--------|-------|----------|---|
| 2022 | 0.1256 | 83.6% | 67 |
| 2023 | 0.1461 | 79.1% | 67 |
| 2024 | 0.1464 | 79.1% | 67 |
| 2025 | 0.1090 | 88.1% | 67 |

Commentary: The blend of LGB and LR is key — LGB captures nonlinear interactions between rating systems while LR provides a stable calibrated backbone. Shallow trees (depth=3) prevent overfitting on ~700 training games. The 55/45 weight was tuned via grid search over blending ratios (tested 50/50 through 70/30). Custom features that measure disagreement between rating systems (e.g., seed-ordinal gap) add signal beyond raw ordinals.

### Women's: L1 Logistic Regression
- **LR**: L1 penalty, C=0.15, liblinear solver, StandardScaler
- **Training data**: 2003-2025 (excluding test season)
- **Features**: Seeds + Elo + T-Rank clone
- **Mean Brier: 0.1365** (std 0.0261)

| Season | Brier | Accuracy | N |
|--------|-------|----------|---|
| 2022 | 0.1500 | 80.6% | 67 |
| 2023 | 0.1724 | 74.6% | 67 |
| 2024 | 0.1162 | 82.1% | 67 |
| 2025 | 0.1074 | 83.6% | 67 |

Commentary: Women's benefits from strong L1 regularization (C=0.15) which aggressively zeros out noisy features. Fewer Massey ordinals are available for women's so a simpler feature set works better. Seeds are more predictive in the women's tournament (fewer upsets), so the model leans heavily on seed differential. Longer training window (2003+) helps given the smaller feature set.

---

## 2026-03-16 10:43:57 - M tier2 ensemble

- Feature set: tier2 (seeds + Elo + season stats + ordinal aggregates)
- Gender: M
- brier_score: 0.1951
- accuracy: 0.7164
- ece: 0.0864
- n_samples: 268
- features: Seed_diff, seed_a, seed_b, Elo_diff, WinPct_diff, PPG_diff, PPGAllowed_diff, PointDiff_diff, GamesPlayed_diff, SOS_diff, OrdinalMean_diff, OrdinalStd_diff + individual ordinals

---

## 2026-03-16 10:39:44 - W seeds_only ensemble

- Feature set: seeds_only
- Gender: W
- brier_score: 0.1489
- accuracy: 0.7724
- ece: 0.0666
- n_samples: 268
- features: Seed_diff, seed_a, seed_b

---

## 2026-03-16 10:39:17 - M tier1 ensemble

- Feature set: tier1 (seeds + ordinals)
- Gender: M
- brier_score: 0.1944
- accuracy: 0.7164
- ece: 0.0902
- n_samples: 268
- features: Seed_diff, seed_a, seed_b + 10 ordinal systems + OrdinalMean/Std

---

## 2026-03-16 10:38:41 - M all ensemble

- Feature set: all (34 features)
- Gender: M
- brier_score: 0.1950
- accuracy: 0.7313
- ece: 0.1021
- n_samples: 268
- features: Seeds, Elo, WinPct, PPG, PointDiff, SOS, 10 ordinals, AdjO/D/EM, Tempo, four factors

---

## 2026-03-16 10:36:05 - M seeds_only ensemble

- Feature set: seeds_only
- Gender: M
- brier_score: 0.1915
- accuracy: 0.6828
- ece: 0.0557
- n_samples: 268
- features: Seed_diff, seed_a, seed_b

---
