# March Machine Learning Mania 2026 - Model Technical Report

**Date**: March 17, 2026
**Submission**: `submission_stage2_v5_lgb55-lr45-blend_l1w_20260317_222920.csv`
**Authors**: weshong + Claude (house-of-claudes)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Model Architecture](#model-architecture)
3. [Cross-Validation Results](#cross-validation-results)
4. [Predicted Brackets](#predicted-brackets)
5. [Discrepancy Analysis vs Expert Consensus](#discrepancy-analysis)
6. [Implications and Improvement Opportunities](#implications)

---

## 1. Executive Summary

Our best-performing model achieves a mean Brier score of **0.1318** (men's) and **0.1365** (women's) on leave-season-out cross-validation across 2022-2025 tournaments. The model uses a blend of LightGBM and Logistic Regression for men's predictions and L1-regularized Logistic Regression for women's.

**Model predictions for 2026 champions:**
- **Men's**: (1) Florida over (1) Arizona (54.0%) - All four 1-seeds reach Final Four
- **Women's**: (1) UConn over (1) UCLA (58.2%) - All four 1-seeds reach Final Four

---

## 2. Model Architecture

### Men's Model: 55% LightGBM + 45% Logistic Regression Blend

| Component | Configuration |
|-----------|--------------|
| LightGBM | max_depth=3, n_estimators=300, lr=0.025, subsample=0.8, colsample=0.6, min_split_gain=0.005 |
| Logistic Regression | L2 penalty, C=100.0, lbfgs solver, StandardScaler |
| Blend weight | 55% LGB / 45% LR |
| Training window | 2015-2025 (excluding held-out season) |
| Prediction clipping | [0.025, 0.975] |

**Feature set (custom builder):**
- Seed differential and raw seeds
- Massey ordinal aggregates: POM, WLK, MOR, DOL, COL, RPI (excluded SAG/AP/USA for coverage)
- Torvik T-Rank ratings (external data, matched 349-356 of 381 teams per season)
- T-Rank clone (constructed from season stats)
- Elo ratings with margin-of-victory adjustment
- Feature gap/disagreement signals (e.g., seed-ordinal divergence)

**Design rationale:**
- Shallow trees (depth=3) are critical — deeper trees overfit on ~700 training games per fold
- The LR component provides calibration stability; LGB captures nonlinear rating interactions
- 55/45 blend was selected via grid search over {50/50, 55/45, 60/40, 65/35, 70/30}
- Training starts at 2015 to avoid regime changes in earlier eras

### Women's Model: L1 Logistic Regression

| Component | Configuration |
|-----------|--------------|
| Logistic Regression | L1 penalty, C=0.15, liblinear solver, StandardScaler |
| Training window | 2003-2025 (excluding held-out season) |
| Prediction clipping | [0.025, 0.975] |

**Feature set:**
- Seed differential and raw seeds
- Elo ratings
- T-Rank clone

**Design rationale:**
- Aggressive L1 regularization (C=0.15) zeros out noisy features automatically
- Fewer Massey ordinals available for women's tournament; simpler features work better
- Seeds are more predictive in women's (fewer upsets historically)
- Longer training window (2003+) is feasible given the smaller feature set

---

## 3. Cross-Validation Results

### Men's Leave-Season-Out CV (Mean Brier: 0.1318, Std: 0.0156)

| Season | Brier Score | Accuracy | Games |
|--------|-------------|----------|-------|
| 2022 | 0.1256 | 83.6% | 67 |
| 2023 | 0.1461 | 79.1% | 67 |
| 2024 | 0.1464 | 79.1% | 67 |
| 2025 | 0.1090 | 88.1% | 67 |

### Women's Leave-Season-Out CV (Mean Brier: 0.1365, Std: 0.0261)

| Season | Brier Score | Accuracy | Games |
|--------|-------------|----------|-------|
| 2022 | 0.1500 | 80.6% | 67 |
| 2023 | 0.1724 | 74.6% | 67 |
| 2024 | 0.1162 | 82.1% | 67 |
| 2025 | 0.1074 | 83.6% | 67 |

### Comparison to Baselines

| Model | Men's Brier | Women's Brier |
|-------|-------------|---------------|
| **v5 (current)** | **0.1318** | **0.1365** |
| seeds_only | 0.1915 | 0.1489 |
| tier1 (seeds+ordinals) | 0.1944 | -- |
| tier2 (seeds+elo+stats) | 0.1951 | -- |
| all features | 0.1950 | -- |
| Typical Kaggle competitive | ~0.14 | ~0.14 |

The v5 model improves over seeds-only by **31%** for men's and **8%** for women's. Notably, the "all features" model (34 features) performs *worse* than seeds-only — a clear sign of overfitting on this small dataset.

---

## 4. Predicted Brackets

### Men's Tournament

```
REGION W (EAST)                                          REGION X (SOUTH)

R64          R32          S16    E8                E8     S16          R32          R64

(1)Duke  ────┐                                                ┌──── (1)Florida
   97.5%     ├─(1)Duke──┐                        ┌(1)Florida─┤      97.5%
(16)Siena────┘  94.1%   │                        │   90.6%   └──── (16)Lehigh
                        ├(1)Duke──┐      ┌(1)Fla─┤
(8)Ohio St───┐  71.7%  │  85.3%  │      │ 76.1% │  80.8%   ┌──── (5)Vanderb.
   90.0%     ├(5)StJ's─┘         │      │       └(5)Vanderb┤      97.5%
(9)TCU ──────┘                   │      │                   └──── (4)Nebraska
                                 │      │                            92.1%
(5)St John's─┐                  │      │                   ┌──── (6)UNC
   92.2%     ├(5)StJ's─┐        │      │        ┌(3)Ill.──┤      78.0%
(12)N Iowa───┘  71.7%  │        │      │        │  94.7%   └──── (3)Illinois
                        ├(2)UConn┘      └(2)Hou─┤                   97.5%
(4)Kansas────┐  62.9%  │ 84.0%    59.2%  51.4% │  92.1%   ┌──── (10)Tex A&M
   89.8%     ├(3)MichSt┘                       └(2)Houst.─┤      57.6%
(13)Cal Bap──┘                                             └──── (2)Houston
                                                                     89.1%
(6)Louisv.───┐                                             ┌──── (7)St Mary's
   91.8%     ├(3)MichSt─┐                      ┌(10)TxA&M─┤
(11)S Fla────┘   62.9%  │                      │           └──── (15)Idaho
                         ├(2)UConn              │
(3)Mich St───┐   79.6%  │                      │
   93.8%     ├(2)UConn──┘                      │
(14)N Dak St─┘                                  │
                                                 │
(7)UCLA──────┐                                   │
   86.2%     ├(2)UConn──┐                       │
(10)UCF──────┘   79.6%  │                       │
                         │                       │
(2)UConn─────┐           │                       │
   97.5%     ├(2)UConn──┘                       │
(15)Furman───┘                                   │

         Duke wins Region W                Florida wins Region X


REGION Y (MIDWEST)                                       REGION Z (WEST)

(1)Michigan──┐                                           ┌──── (1)Arizona
   97.5%     ├(1)Mich──┐                        ┌(1)Ariz─┤      81.8%
(16)Howard───┘  96.3%  │                        │  94.9%  └──── (16)LIU
                       ├(1)Mich──┐      ┌(1)Ari─┤
(8)Georgia───┐  51.9%  │  89.3%  │      │ 65.3% │  57.8%  ┌──── (5)Wiscon.
   54.4%     ├(4)Bama──┘         │      │       └(4)Ark.──┤      94.9%
(9)St Louis──┘                   │      │                  └──── (4)Arkansas
                                 │      │                           92.8%
(5)Tex Tech──┐                   │      │                  ┌──── (11)NC State
   91.6%     ├(5)TxTch─┐        │      │       ┌(3)Gonz──┤      63.1%
(12)Akron────┘  51.9%  │        │      │       │  73.9%   └──── (3)Gonzaga
                       ├(1)Mich─┘      └(1)Ari─┤                   90.6%
(4)Alabama───┐  53.9%  │ 77.3%   51.8%  85.2% │  94.1%   ┌──── (10)Missouri
   95.0%     ├(6)Tenn──┘                       └(2)Purdue─┤      77.8%
(13)Hofstra──┘                                             └──── (2)Purdue
                                                                    97.5%
(6)Tennessee─┐                                             ┌──── (6)BYU
   95.0%     ├(6)Tenn──┐                       ┌(11)NCSt──┤
(11)Miami OH─┘  53.9%  │                       │   63.1%   └──── (8)Villanova
                       ├(2)IowaSt               │                    51.1%
(3)Virginia──┐  81.2%  │                       │
   93.2%     ├(2)IowSt─┘                      │
(14)Wr. St───┘                                  │
                                                 │
(7)Kentucky──┐                                   │
   76.6%     ├(10)SClra─┐                       │
(10)S Clara──┘   81.2%  │                       │
                         │                       │
(2)Iowa St───┐           │                       │
   97.5%     ├(2)IowaSt─┘                       │
(15)Tenn St──┘                                   │

        Michigan wins Region Y              Arizona wins Region Z


                            ┌─────── FINAL FOUR ───────┐
                            │                           │
                  (1) Duke [W]                (1) Michigan [Y]
                       vs                          vs
                  (1) Florida [X]             (1) Arizona [Z]
                       │                           │
                  (1) Florida                 (1) Arizona
                    53.7%                       55.1%
                       │                           │
                       └──── CHAMPIONSHIP ─────────┘
                            (1) Florida vs (1) Arizona

                        *** CHAMPION: (1) FLORIDA 54.0% ***
```

### Women's Tournament

```
REGION W                                                 REGION X

(1)UConn─────┐                                           ┌──── (1)S Carolina
   97.5%     ├(1)UConn─┐                       ┌(1)SC───┤      97.5%
(16)UTSA─────┘  97.5%  │                       │  96.2%  └──── (16)Samford
                       ├(1)UConn─┐     ┌(1)SC──┤
(8)Iowa St───┐  55.2%  │  93.8%  │     │ 86.3% │  62.0%  ┌──── (5)Mich St
   60.0%     ├(4)UNC───┘         │     │       └(4)Okla──┤      87.1%
(9)Syracuse──┘                   │     │                  └──── (4)Oklahoma
                                 │     │                          95.1%
(5)Maryland──┐                   │     │                  ┌──── (6)Wash.
   92.6%     ├(5)Maryl─┐        │     │       ┌(3)TCU───┤      70.5%
(12)Murray St┘  55.2%  │        │     │       │  80.2%   └──── (3)TCU
                       ├(1)UCon─┘     └(1)SC──┤                   97.5%
(4)N Carolina┐  70.7%  │ 90.8%   79.5% 56.1% │  86.5%   ┌──── (7)Georgia
   97.1%     ├(3)OhSt──┘                      └(2)Iowa──┤      67.7%
(13)W Illinois┘                                          └──── (2)Iowa
                                                                  97.5%
(6)Notre Dame┐                                            ┌──── (9)USC
   79.0%     ├(3)OhSt──┐                      ┌(9)USC───┤      64.1%
(11)Fairfield┘  70.7%  │                      │          └──── (8)Clemson
                       ├(3)OhSt                │
(3)Ohio St───┐  81.9%  │                      │
   97.5%     ├(2)Vandy─┘                      │
(14)Howard───┘                                 │
                                                │
(7)Illinois──┐                                  │
   60.0%     ├(2)Vandy──┐                      │
(10)Colorado─┘   81.9%  │                      │
                         │                      │
(2)Vanderbilt┐           │                      │
   97.5%     ├(2)Vandy──┘                      │
(15)High Pt──┘                                  │

       UConn wins Region W               S Carolina wins Region X


REGION Y                                                 REGION Z

(1)Texas─────┐                                           ┌──── (1)UCLA
   97.5%     ├(1)Texas─┐                       ┌(1)UCLA─┤      97.5%
(16)Mo St────┘  96.0%  │                       │  97.2%  └──── (16)Cal Bap
                       ├(1)Texas─┐     ┌(1)UCLA┤
(8)Oregon────┐  59.0%  │  87.0%  │     │ 91.9% │  63.3%  ┌──── (5)Ole Miss
   59.9%     ├(4)WVU───┘         │     │       └(4)Minn──┤      84.7%
(9)Va Tech───┘                   │     │                  └──── (4)Minnesota
                                 │     │                          95.6%
(5)Kentucky──┐                   │     │                  ┌──── (6)Baylor
   80.6%     ├(5)Kent──┐        │     │       ┌(3)Duke──┤      76.4%
(12)Jas. Mad.┘  59.0%  │        │     │       │  81.2%   └──── (3)Duke
                       ├(1)Tex──┘     └(1)UCLA┤                   97.5%
(4)W Virginia┐  77.2%  │ 76.5%   71.2% 66.1% │  94.2%   ┌──── (7)Tex Tech
   94.0%     ├(3)Louis─┘                      └(2)LSU───┤      53.4%
(13)Miami OH─┘                                           └──── (2)LSU
                                                                  97.5%
(6)Alabama───┐                                            ┌──── (11)Nebraska
   76.4%     ├(3)Louis──┐                     ┌(6)Bayl──┤      76.4%
(11)Rhode Is.┘  77.2%   │                     │          └──── (8)Okla St
                        ├(2)Mich               │                  57.6%
(3)Louisville┐  85.2%   │                     │
   97.3%     ├(2)Mich───┘                     │
(14)Vermont──┘                                 │
                                                │
(7)NC State──┐                                  │
   66.5%     ├(2)Mich───┐                      │
(10)Tennessee┘   85.2%  │                      │
                         │                      │
(2)Michigan──┐           │                      │
   97.5%     ├(2)Mich───┘                      │
(15)Holy Cr──┘                                  │

        Texas wins Region Y                UCLA wins Region Z


                            ┌─────── FINAL FOUR ───────┐
                            │                           │
                  (1) UConn [W]                (1) Texas [Y]
                       vs                          vs
                  (1) S Carolina [X]           (1) UCLA [Z]
                       │                           │
                  (1) UConn                   (1) UCLA
                    67.0%                       58.3%
                       │                           │
                       └──── CHAMPIONSHIP ─────────┘
                           (1) UConn vs (1) UCLA

                      *** CHAMPION: (1) UCONN 58.2% ***
```

---

## 5. Discrepancy Analysis: Our Model vs Expert Consensus

### 5.1 Men's Tournament

#### Overall Power Rankings Compared

| Rank | Our Model (Avg Win %) | Expert Consensus (Odds) | KenPom Rank |
|------|-----------------------|-------------------------|-------------|
| 1 | Michigan (90.3%) | Michigan (+325) | ~3-4 |
| 2 | Arizona (90.2%) | Duke (+333) | #1 |
| 3 | Duke (87.5%) | Arizona (+380) | ~2-3 |
| 4 | Purdue (87.4%) | Florida (+750) | ~5-6 |
| 5 | Florida (85.8%) | Houston (+900) | ~4-5 |
| 6 | Houston (83.9%) | Iowa State (+1800) | ~8-10 |
| 7 | Alabama (82.9%) | Illinois (+2200) | ~6-7 |
| 8 | Vanderbilt (82.6%) | Purdue (+2500) | ~7-8 |

#### Key Discrepancies

**1. Duke: Our model underrates Duke relative to consensus**

Our model ranks Duke 3rd (87.5% avg win prob) while KenPom has Duke as the clear #1 overall (best AdjD, top-4 AdjO). Experts and oddsmakers have Duke as co-favorite with Michigan (+333). Our model still has Duke winning Region W convincingly (84.0% over UConn in Elite Eight), but loses to Florida in the Final Four (46.3%).

*Possible explanation*: Duke guard Caleb Foster's injury may not be fully captured by our features, which rely on season-long ratings. KenPom's ratings reflect Duke's full-strength performance, while betting markets have started to discount them post-injury. Our model may coincidentally align with the injury-adjusted view.

*Implication*: Our model may be accidentally correct here — or it could be that our ordinal/Torvik features lag KenPom's real-time adjustments.

**2. Purdue: Our model is significantly higher on Purdue**

We rank Purdue 4th (87.4%), essentially tied with Duke. Oddsmakers have Purdue at +2500 (roughly 8th). KenPom rates Purdue's offense as #1 nationally. Our model has Purdue winning the Elite Eight of Region Z over Arizona (48.2% — it's close), but Arizona edges them.

*Possible explanation*: Purdue's #1 KenPom offense gets amplified through our ordinal features. The market may be down on Purdue due to historical tournament underperformance narrative and Zach Edey's departure.

*Implication*: This is a value bet if our model is right. Consider creating an alternate submission that boosts Purdue's probabilities.

**3. Vanderbilt: Model loves Vanderbilt (#5 seed, ranked 8th overall)**

Our model has Vanderbilt at 82.6% average win probability — well above their 5-seed. KenPom ranks Vanderbilt #12. Our model has them beating Nebraska (80.8%) and reaching the Sweet 16 before falling to Florida.

*Possible explanation*: Vanderbilt's efficiency numbers and ordinal rankings are strong despite a lower seed (SEC bias in seeding, or they peaked late). The model sees through the seed to the underlying metrics.

*Implication*: Vanderbilt as a 5-seed may represent one of the model's stronger "value" picks. ESPN BPI also notes McNeese as an upset candidate over Vanderbilt, which our model strongly disagrees with (97.5% Vanderbilt).

**4. Kansas (#4 seed): Model is low on Kansas**

Our model ranks Kansas at just 64.9% average win probability, 20th among tournament teams. The model has St. John's (#5) beating Kansas in the Round of 32 (71.7%). This is a significant upset call.

*Possible explanation*: Kansas's ordinal rankings and efficiency metrics may have declined from early season. St. John's is widely considered under-seeded (Nate Silver's COOPER model views them as a 2-3 seed quality team). Multiple expert models flag this seeding discrepancy.

*Implication*: This aligns with expert consensus — St. John's over Kansas is a legitimate upset pick.

**5. Kentucky (#7): Model thinks Kentucky is bad**

Our model has Santa Clara (#10) beating Kentucky 76.6% — one of the most extreme upset predictions. Missouri (#10) over Kentucky at 83.0%. Kentucky appears in the bottom tier of tournament teams by our model's ranking.

*Possible explanation*: Kentucky's underlying efficiency numbers may be poor despite brand recognition. This could be a case where our model sees through reputation to underlying performance, OR it could be a data issue with ordinal coverage.

*Verification needed*: Check Kentucky's ordinal rankings and Torvik data for potential data quality issues.

**6. Florida as champion: Model is more bullish than markets**

Our model picks Florida to win it all (54.0% over Arizona in the final). Markets have Florida at +750 (4th favorite). The model's path: Florida beats Houston (59.2%), then Duke/UConn winner (53.7%), then Arizona (54.0%).

*Possible explanation*: Florida's balanced profile (strong seed + strong ordinals + good Elo) performs well in the model's feature space. The model may not capture Duke's raw talent advantage that experts see.

**7. BYU (#6): Model thinks BYU is very weak**

NC State (#11) beats BYU 63.1%, Utah St (#9) over BYU at 72.7%, Missouri (#10) over BYU at 80.2%. The model has BYU as one of the weakest 6-seeds ever.

*Expert alignment*: ESPN BPI also flags NC State over BYU as a likely upset (~37% for NC State). Our model is even more aggressive on this call.

#### First-Round Upset Predictions Compared to ESPN BPI

| Upset Pick | Our Model | ESPN BPI | Consensus |
|------------|-----------|----------|-----------|
| (12) High Point over (5) Wisconsin | No (5.1%) | Yes (20%+) | Popular pick |
| (10) Texas A&M over (7) St Mary's | Yes (57.6%) | Notable | Moderate |
| (10) Missouri over (7) Miami FL | Yes (77.8%) | Notable | Moderate |
| (11) NC State over (6) BYU | Yes (63.1%) | Yes (~37%) | Popular |
| (11) VCU over (6) North Carolina | No (22.0%) | Notable | Popular |
| (12) Akron over (5) Texas Tech | No (8.4%) | Notable | Moderate |
| (12) McNeese over (5) Vanderbilt | No (2.5%) | ~18% | Moderate |
| (10) Santa Clara over (7) Kentucky | Yes (76.6%) | -- | Uncommon |

**Major disagreement**: High Point — experts love this as an upset, our model gives it essentially 0%. McNeese over Vanderbilt — experts give ~18%, we give 2.5%. Meanwhile, we're extremely high on Missouri and Santa Clara upsets that experts barely mention.

### 5.2 Women's Tournament

#### Power Rankings Compared

| Rank | Our Model | Betting Odds |
|------|-----------|-------------|
| 1 | UConn (95.8%) | UConn (-270, heavy favorite) |
| 2 | UCLA (94.3%) | UCLA (+550) |
| 3 | Texas (91.3%) | Texas (+700) |
| 4 | South Carolina (91.2%) | South Carolina (+800) |
| 5 | LSU (88.0%) | LSU (+1600) |

Our women's model aligns remarkably well with consensus. The top 4 are the same as the betting market, and UConn is the clear #1 in both. Our model gives UConn a 58.2% championship probability, consistent with their -270 odds (~73% implied, but that includes vig).

**Key discrepancy**: West Virginia (#4 seed) is our 6th-ranked team (85.1% avg win prob), which is much higher than expected for a 4-seed. This suggests WVU may be under-seeded.

**Upset pick alignment**: James Madison over Kentucky is the #1 expert upset pick for women's. Our model disagrees — Kentucky wins 80.6%. This could be a blind spot where our model's feature set misses JMU's late-season surge.

---

## 6. Implications and Improvement Opportunities

### 6.1 What the Discrepancies Tell Us

**Model strengths:**
- Strong alignment with markets on top-tier teams (especially women's)
- Correctly identifies under-seeded teams (St. John's, Vanderbilt) that experts also flag
- Conservative calibration prevents extreme predictions

**Model weaknesses:**
- Some predictions look like data quality issues (Kentucky at bottom, BYU extremely low)
- No injury adjustment — Duke with injured Caleb Foster is rated the same as healthy Duke
- No recency weighting — late-season hot streaks (VCU's 16-1 run, High Point's 30-4) don't get enough credit
- Cinderella teams systematically underrated — small-conference teams with great records (McNeese 28-5, High Point 30-4) get crushed by our ordinal features which favor major conferences

### 6.2 Specific Improvements to Consider

1. **Recency weighting**: Weight recent games (last 10-15) more heavily in Elo and efficiency calculations. Currently our features use full-season aggregates, missing late-season momentum.

2. **Injury adjustment**: Incorporate a manual injury flag or use line movement data as a proxy. Duke's championship probability should probably be lower than our model suggests given Foster's injury.

3. **Small-conference adjustment**: Our ordinal features inherently bias toward major conferences. Consider a "conference strength residual" feature — how much better is a team than their conference's average? High Point going 30-4 in a weak conference should still count for something.

4. **Ensemble with seed-based priors**: For Round of 64 matchups, historical seed matchup probabilities (e.g., 12-seeds beat 5-seeds ~35% of the time) provide a useful prior. Blending our model with these priors could improve calibration in early rounds.

5. **Market-informed calibration**: Use pre-tournament betting lines or KenPom probabilities as an additional feature or calibration target. Our model and the market agree on the top but diverge on the middle — the market may have information (injuries, team chemistry, depth) that our features miss.

6. **Per-round model**: Tournament basketball changes by round. Round of 64 games are more predictable (chalk holds); later rounds see more upsets as talent gaps narrow. A per-round model or round-specific calibration could improve performance.

7. **Verify data quality for outlier predictions**: Kentucky and BYU predictions are extreme enough to warrant checking for data issues (missing ordinals, stale Torvik data, Elo calculation errors). If these predictions are correct, they're bold contrarian calls; if they're data errors, they're costing us Brier score.

### 6.3 Submission Strategy

Given 2 days until deadline (March 19, 2026):

- **Submission 1 (current)**: v5 LGB+LR blend — our best CV score, trust the model
- **Submission 2 (recommended)**: Create a variant that:
  - Shrinks extreme upset predictions toward historical base rates
  - Manually adjusts Duke downward for Foster injury
  - Boosts High Point and McNeese upset probabilities toward ESPN BPI levels
  - This would be a "hedged" submission that blends model confidence with expert consensus

The optimal strategy for Brier score is to be well-calibrated rather than bold. If our model says 77.8% for Missouri over Miami FL but the true probability is ~50%, that costs us significantly. A hedged submission that pulls extreme predictions toward 50% may score better on average even if it misses some correct calls.

---

*Report generated March 17, 2026. Model code: `scripts/generate_submissions_v5.py`. Predictions: `outputs/submissions/submission_stage2_v5_lgb55-lr45-blend_l1w_20260317_222920.csv`.*
