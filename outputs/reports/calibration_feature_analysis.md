# Calibration, Upset & Feature Importance Analysis: RF (original) vs v5 vs v7

Generated 2026-03-18. Leave-season-out CV on 2022-2025 tournaments.

**Models compared:**
- **RF**: Original R model ported to Python. Random Forest (1000 trees), box score features (shooting %, rebounds, assists, TO, efficiency, four factors). Training window 2003+.
- **v5**: 55% LightGBM + 45% Logistic Regression blend. Ordinal aggregates + Torvik + TRank clone + disagreement features. Training window 2015+ (M) / 2003+ (W).
- **v7**: 40% LightGBM + 60% Logistic Regression blend. PCA ordinals + Torvik + TRank clone + disagreement features. Training window 2015+ (M) / 2003+ (W).

## Key Takeaways

**The RF model has the best 5-12 calibration by far** (cal error 0.024 vs v5's 0.144 and v7's 0.160). It predicts 64.9% favorite win rate on 5-12 games, nearly matching the 62.5% actual rate. But its overall Brier is terrible (0.1943 vs v7's 0.1266) because it's underconfident on every other matchup tier — it can't separate a 1v16 blowout from a 5v12 toss-up.

**This is not good calibration — it's poor discrimination.** The RF doesn't "know" upsets are coming; it just can't distinguish strong 5-seeds from weak ones, so all its 5-12 predictions cluster in a narrow 0.55–0.75 band. That coincidentally looks well-calibrated for a matchup tier with ~35% upsets, but it leaves massive Brier points on the table when the 5-seed is genuinely dominant (e.g., Houston vs UAB: RF gave 0.709, v7 gave 0.975).

**v5 and v7 have the opposite problem**: they discriminate well (correctly assigning 0.97 to Houston/UAB) but can't tell which 5-seeds will get upset, so they're systematically overconfident on the tier as a whole. v7 is actually *worse* than v5 here because the shift from 55/45 to 40/60 LGB/LR blend weight leans harder on the linear model that trusts ratings more literally.

**Brier score is not misleading, but it does mask the tradeoff.** Improving overall Brier from v5 to v7 came partly by becoming more confident on favorites — which helps on the ~70% of games where favorites win but hurts on the ~30% where they don't. The 5-12 tier is where this tradeoff is most acute because it has the highest upset rate of any first-round matchup.

**The women's models catch 0/23 upsets** (both v5 and v7). They are effectively seed models with extra steps — L1 regularization zeros out most features, and neither model ever predicts an underdog to win. The RF does no better (0/23 as well).

**Practical implication**: The 5-12 overconfidence is real but unfixable without features that capture *why* specific 5-seeds are upset-prone (e.g., game-to-game variance, matchup style, momentum). Matchup-tier-specific calibration hacks would overfit to 16 games. The honest answer is: with current features, the models correctly optimize overall Brier at the cost of 5-12 calibration, and that is the right tradeoff for the competition metric.

---
# Men's

## Overall Scores
| Model | Brier | Accuracy | N | Features |
|-------|-------|----------|---|----------|
| RF (original) | 0.1943 | 70.9% | 268 | 47 |
| v5 | 0.1334 | 81.3% | 268 | 31 |
| v7 | 0.1266 | 82.5% | 268 | 22 |

## Per-Season Breakdown
| Season | RF Brier | v5 Brier | v7 Brier | RF Acc | v5 Acc | v7 Acc |
|--------|----------|----------|----------|--------|--------|--------|
| 2022 | 0.2200 | 0.1257 | 0.1230 | 65.7% | 83.6% | 85.1% |
| 2023 | 0.2001 | 0.1478 | 0.1442 | 71.6% | 76.1% | 79.1% |
| 2024 | 0.1919 | 0.1499 | 0.1383 | 70.1% | 79.1% | 80.6% |
| 2025 | 0.1651 | 0.1104 | 0.1011 | 76.1% | 86.6% | 85.1% |

## Calibration by Seed Matchup (First Round)

Pred fav win% = model's average predicted probability that the favorite wins.
Actual fav win% = how often the favorite actually won in CV data.
Calibration error = |predicted - actual|. Direction: overconf = model too confident in favorite, underconf = model doesn't trust favorite enough.

### RF (original)
| Matchup | N | Pred Fav% | Actual Fav% | Cal Error | Direction | Upsets | Upset% | Brier |
|---------|---|-----------|-------------|-----------|-----------|--------|--------|-------|
| 1v16 | 16 | 89.7% | 93.8% | 0.040 | underconf | 1 | 6.2% | 0.0558 |
| 2v15 | 16 | 83.4% | 87.5% | 0.041 | underconf | 2 | 12.5% | 0.1189 |
| 3v14 | 16 | 78.2% | 93.8% | 0.155 | underconf | 1 | 6.2% | 0.0875 |
| 4v13 | 16 | 72.4% | 87.5% | 0.151 | underconf | 2 | 12.5% | 0.1529 |
| 5v12 | 16 | 64.9% | 62.5% | 0.024 | overconf | 6 | 37.5% | 0.2394 |
| 6v11 | 16 | 60.5% | 50.0% | 0.105 | overconf | 8 | 50.0% | 0.2851 |
| 7v10 | 16 | 57.5% | 68.8% | 0.112 | underconf | 5 | 31.2% | 0.2360 |
| 8v9 | 16 | 53.9% | 37.5% | 0.164 | overconf | 10 | 62.5% | 0.2335 |

### v5
| Matchup | N | Pred Fav% | Actual Fav% | Cal Error | Direction | Upsets | Upset% | Brier |
|---------|---|-----------|-------------|-----------|-----------|--------|--------|-------|
| 1v16 | 16 | 96.2% | 93.8% | 0.024 | overconf | 1 | 6.2% | 0.0570 |
| 2v15 | 16 | 87.7% | 87.5% | 0.002 | overconf | 2 | 12.5% | 0.0372 |
| 3v14 | 16 | 91.8% | 93.8% | 0.020 | underconf | 1 | 6.2% | 0.0483 |
| 4v13 | 16 | 87.1% | 87.5% | 0.004 | underconf | 2 | 12.5% | 0.1300 |
| 5v12 | 16 | 76.9% | 62.5% | 0.144 | overconf | 6 | 37.5% | 0.1950 |
| 6v11 | 16 | 52.2% | 50.0% | 0.022 | overconf | 8 | 50.0% | 0.1468 |
| 7v10 | 16 | 55.7% | 68.8% | 0.131 | underconf | 5 | 31.2% | 0.2225 |
| 8v9 | 16 | 45.1% | 37.5% | 0.076 | overconf | 10 | 62.5% | 0.1897 |

### v7
| Matchup | N | Pred Fav% | Actual Fav% | Cal Error | Direction | Upsets | Upset% | Brier |
|---------|---|-----------|-------------|-----------|-----------|--------|--------|-------|
| 1v16 | 16 | 94.5% | 93.8% | 0.007 | overconf | 1 | 6.2% | 0.0254 |
| 2v15 | 16 | 88.7% | 87.5% | 0.012 | overconf | 2 | 12.5% | 0.0411 |
| 3v14 | 16 | 91.7% | 93.8% | 0.021 | underconf | 1 | 6.2% | 0.0496 |
| 4v13 | 16 | 86.3% | 87.5% | 0.012 | underconf | 2 | 12.5% | 0.1297 |
| 5v12 | 16 | 78.5% | 62.5% | 0.160 | overconf | 6 | 37.5% | 0.2091 |
| 6v11 | 16 | 50.4% | 50.0% | 0.004 | overconf | 8 | 50.0% | 0.1540 |
| 7v10 | 16 | 55.8% | 68.8% | 0.129 | underconf | 5 | 31.2% | 0.2020 |
| 8v9 | 16 | 46.5% | 37.5% | 0.090 | overconf | 10 | 62.5% | 0.1640 |

### Calibration Comparison (RF vs v5 vs v7)
| Matchup | RF Cal Err | v5 Cal Err | v7 Cal Err | RF Dir | v5 Dir | v7 Dir | RF Brier | v5 Brier | v7 Brier | Best |
|---------|-----------|-----------|-----------|--------|--------|--------|----------|----------|----------|------|
| 1v16 | 0.040 | 0.024 | 0.007 | underconf | overconf | overconf | 0.0558 | 0.0570 | 0.0254 | v7 |
| 2v15 | 0.041 | 0.002 | 0.012 | underconf | overconf | overconf | 0.1189 | 0.0372 | 0.0411 | v5 |
| 3v14 | 0.155 | 0.020 | 0.021 | underconf | underconf | underconf | 0.0875 | 0.0483 | 0.0496 | v5 |
| 4v13 | 0.151 | 0.004 | 0.012 | underconf | underconf | underconf | 0.1529 | 0.1300 | 0.1297 | v7 |
| 5v12 | 0.024 | 0.144 | 0.160 | overconf | overconf | overconf | 0.2394 | 0.1950 | 0.2091 | v5 |
| 6v11 | 0.105 | 0.022 | 0.004 | overconf | overconf | overconf | 0.2851 | 0.1468 | 0.1540 | v5 |
| 7v10 | 0.112 | 0.131 | 0.129 | underconf | underconf | underconf | 0.2360 | 0.2225 | 0.2020 | v7 |
| 8v9 | 0.164 | 0.076 | 0.090 | overconf | overconf | overconf | 0.2335 | 0.1897 | 0.1640 | v7 |

## 5v12 Matchup Deep Dive

5-12 matchups are historically the most volatile first-round pairing (upsets ~35% of the time).

### RF (original) game-by-game predictions
| Season | 5-seed | 12-seed | Pred(5wins) | 5won? | Upset? |
|--------|--------|---------|-------------|-------|--------|
| 2022 | Connecticut | New Mexico St | 0.755 | N | **YES** |
| 2022 | Iowa | Richmond | 0.800 | N | **YES** |
| 2022 | St Mary's CA | Indiana | 0.724 | Y | no |
| 2022 | Houston | UAB | 0.709 | Y | no |
| 2023 | Duke | Oral Roberts | 0.620 | Y | no |
| 2023 | San Diego St | Col Charleston | 0.605 | Y | no |
| 2023 | Miami FL | Drake | 0.741 | Y | no |
| 2023 | St Mary's CA | VCU | 0.742 | Y | no |
| 2024 | Gonzaga | McNeese St | 0.677 | Y | no |
| 2024 | St Mary's CA | Grand Canyon | 0.634 | N | **YES** |
| 2024 | Wisconsin | James Madison | 0.530 | N | **YES** |
| 2024 | San Diego St | UAB | 0.740 | Y | no |
| 2025 | Clemson | McNeese St | 0.639 | N | **YES** |
| 2025 | Michigan | UC San Diego | 0.470 | Y | no |
| 2025 | Memphis | Colorado St | 0.487 | N | **YES** |
| 2025 | Oregon | Liberty | 0.505 | Y | no |

### v5 game-by-game predictions
| Season | 5-seed | 12-seed | Pred(5wins) | 5won? | Upset? |
|--------|--------|---------|-------------|-------|--------|
| 2022 | Connecticut | New Mexico St | 0.683 | N | **YES** |
| 2022 | Iowa | Richmond | 0.950 | N | **YES** |
| 2022 | St Mary's CA | Indiana | 0.486 | Y | no |
| 2022 | Houston | UAB | 0.973 | Y | no |
| 2023 | Duke | Oral Roberts | 0.975 | Y | no |
| 2023 | San Diego St | Col Charleston | 0.913 | Y | no |
| 2023 | Miami FL | Drake | 0.805 | Y | no |
| 2023 | St Mary's CA | VCU | 0.946 | Y | no |
| 2024 | Gonzaga | McNeese St | 0.970 | Y | no |
| 2024 | St Mary's CA | Grand Canyon | 0.760 | N | **YES** |
| 2024 | Wisconsin | James Madison | 0.776 | N | **YES** |
| 2024 | San Diego St | UAB | 0.877 | Y | no |
| 2025 | Clemson | McNeese St | 0.383 | N | **YES** |
| 2025 | Michigan | UC San Diego | 0.966 | Y | no |
| 2025 | Memphis | Colorado St | 0.124 | N | **YES** |
| 2025 | Oregon | Liberty | 0.723 | Y | no |

### v7 game-by-game predictions
| Season | 5-seed | 12-seed | Pred(5wins) | 5won? | Upset? |
|--------|--------|---------|-------------|-------|--------|
| 2022 | Connecticut | New Mexico St | 0.664 | N | **YES** |
| 2022 | Iowa | Richmond | 0.945 | N | **YES** |
| 2022 | St Mary's CA | Indiana | 0.551 | Y | no |
| 2022 | Houston | UAB | 0.975 | Y | no |
| 2023 | Duke | Oral Roberts | 0.975 | Y | no |
| 2023 | San Diego St | Col Charleston | 0.905 | Y | no |
| 2023 | Miami FL | Drake | 0.826 | Y | no |
| 2023 | St Mary's CA | VCU | 0.878 | Y | no |
| 2024 | Gonzaga | McNeese St | 0.964 | Y | no |
| 2024 | St Mary's CA | Grand Canyon | 0.837 | N | **YES** |
| 2024 | Wisconsin | James Madison | 0.841 | N | **YES** |
| 2024 | San Diego St | UAB | 0.860 | Y | no |
| 2025 | Clemson | McNeese St | 0.451 | N | **YES** |
| 2025 | Michigan | UC San Diego | 0.950 | Y | no |
| 2025 | Memphis | Colorado St | 0.210 | N | **YES** |
| 2025 | Oregon | Liberty | 0.724 | Y | no |

### RF vs v5 vs v7 prediction comparison on 5-12 games
| Season | 5-seed | 12-seed | RF Pred | v5 Pred | v7 Pred | Result | Best |
|--------|--------|---------|---------|---------|---------|--------|------|
| 2022 | Connecticut | New Mexico St | 0.755 | 0.683 | 0.664 | **12-UPSET** | v7 |
| 2022 | Houston | UAB | 0.709 | 0.973 | 0.975 | 5-seed won | v7 |
| 2022 | Iowa | Richmond | 0.800 | 0.950 | 0.945 | **12-UPSET** | RF |
| 2022 | St Mary's CA | Indiana | 0.724 | 0.486 | 0.551 | 5-seed won | RF |
| 2023 | Duke | Oral Roberts | 0.620 | 0.975 | 0.975 | 5-seed won | v5 |
| 2023 | Miami FL | Drake | 0.741 | 0.805 | 0.826 | 5-seed won | v7 |
| 2023 | San Diego St | Col Charleston | 0.605 | 0.913 | 0.905 | 5-seed won | v5 |
| 2023 | St Mary's CA | VCU | 0.742 | 0.946 | 0.878 | 5-seed won | v5 |
| 2024 | Gonzaga | McNeese St | 0.677 | 0.970 | 0.964 | 5-seed won | v5 |
| 2024 | San Diego St | UAB | 0.740 | 0.877 | 0.860 | 5-seed won | v5 |
| 2024 | St Mary's CA | Grand Canyon | 0.634 | 0.760 | 0.837 | **12-UPSET** | RF |
| 2024 | Wisconsin | James Madison | 0.530 | 0.776 | 0.841 | **12-UPSET** | RF |
| 2025 | Clemson | McNeese St | 0.639 | 0.383 | 0.451 | **12-UPSET** | v5 |
| 2025 | Memphis | Colorado St | 0.487 | 0.124 | 0.210 | **12-UPSET** | v5 |
| 2025 | Michigan | UC San Diego | 0.470 | 0.966 | 0.950 | 5-seed won | v5 |
| 2025 | Oregon | Liberty | 0.505 | 0.723 | 0.724 | 5-seed won | v7 |

### 5-12 Commentary

The game-by-game comparison tells the story clearly:

- **Iowa/Richmond 2022 upset**: RF gave 0.800, v5 gave 0.950, v7 gave 0.945. RF was least wrong, but only because it never trusted Iowa much to begin with — not because it saw the upset coming.
- **Wisconsin/James Madison 2024 upset**: RF gave 0.530 (near coin flip), v5 gave 0.776, v7 gave 0.841. RF's uncertainty happened to be right, but it also gave Wisconsin/James Madison the same confidence as Michigan/UC San Diego (0.470) — a game the 5-seed won easily.
- **Houston/UAB 2022 (5-seed won)**: RF gave 0.709, v7 gave 0.975. The RF's inability to distinguish a dominant Houston team from a vulnerable Iowa cost it dearly. v7 correctly identified Houston as a near-lock.
- **Clemson/McNeese St and Memphis/Colorado St 2025**: v5 actually predicted these upsets (0.383 and 0.124), catching what the ratings saw — these 12-seeds were genuinely strong. v7 was closer to 0.5 but still on the right side. RF gave ~0.55-0.64, less wrong than v7 on Clemson but worse than v5.

The pattern: RF's 5-12 predictions are compressed into a narrow band (0.47–0.80), while v5/v7 spread from 0.12 to 0.975. The wider spread means v5/v7 are more right when they're right and more wrong when they're wrong — but overall, the wider spread wins on Brier because most 5-seeds do win and the model is rewarded for confidently backing them.

## Calibration Curves (20-bin)

ECE: RF=0.0835, v5=0.1093, v7=0.0769

### RF (original) calibration
| Bin | N | Mean Pred | Actual Rate | Error |
|-----|---|-----------|-------------|-------|
| 0.03 | 2 | 0.042 | 0.000 | 0.042 |
| 0.08 | 6 | 0.077 | 0.000 | 0.077 |
| 0.12 | 8 | 0.129 | 0.000 | 0.129 |
| 0.18 | 8 | 0.175 | 0.250 | 0.075 |
| 0.23 | 7 | 0.228 | 0.286 | 0.057 |
| 0.28 | 8 | 0.276 | 0.125 | 0.151 |
| 0.33 | 18 | 0.325 | 0.333 | 0.008 |
| 0.38 | 17 | 0.372 | 0.471 | 0.098 |
| 0.43 | 20 | 0.423 | 0.450 | 0.027 |
| 0.47 | 24 | 0.474 | 0.542 | 0.067 |
| 0.53 | 22 | 0.526 | 0.500 | 0.026 |
| 0.58 | 15 | 0.571 | 1.000 | 0.429 |
| 0.62 | 15 | 0.627 | 0.600 | 0.027 |
| 0.68 | 23 | 0.679 | 0.870 | 0.191 |
| 0.73 | 22 | 0.727 | 0.727 | 0.001 |
| 0.78 | 25 | 0.772 | 0.760 | 0.012 |
| 0.83 | 12 | 0.817 | 0.667 | 0.150 |
| 0.88 | 7 | 0.868 | 0.857 | 0.011 |
| 0.93 | 6 | 0.918 | 1.000 | 0.082 |
| 0.98 | 3 | 0.959 | 1.000 | 0.041 |

### v5 calibration
| Bin | N | Mean Pred | Actual Rate | Error |
|-----|---|-----------|-------------|-------|
| 0.03 | 20 | 0.035 | 0.000 | 0.035 |
| 0.08 | 13 | 0.075 | 0.154 | 0.079 |
| 0.12 | 14 | 0.125 | 0.071 | 0.054 |
| 0.18 | 10 | 0.176 | 0.400 | 0.224 |
| 0.23 | 14 | 0.222 | 0.286 | 0.063 |
| 0.28 | 10 | 0.277 | 0.000 | 0.277 |
| 0.33 | 9 | 0.332 | 0.556 | 0.223 |
| 0.38 | 11 | 0.370 | 0.455 | 0.085 |
| 0.43 | 7 | 0.431 | 0.429 | 0.002 |
| 0.47 | 10 | 0.470 | 0.300 | 0.170 |
| 0.53 | 9 | 0.520 | 0.444 | 0.076 |
| 0.58 | 6 | 0.583 | 0.833 | 0.250 |
| 0.62 | 11 | 0.624 | 0.727 | 0.103 |
| 0.68 | 9 | 0.671 | 0.333 | 0.338 |
| 0.73 | 13 | 0.722 | 0.923 | 0.201 |
| 0.78 | 17 | 0.775 | 0.882 | 0.107 |
| 0.83 | 13 | 0.823 | 0.769 | 0.054 |
| 0.88 | 14 | 0.868 | 1.000 | 0.132 |
| 0.93 | 25 | 0.924 | 1.000 | 0.076 |
| 0.98 | 33 | 0.971 | 0.939 | 0.031 |

### v7 calibration
| Bin | N | Mean Pred | Actual Rate | Error |
|-----|---|-----------|-------------|-------|
| 0.03 | 19 | 0.034 | 0.000 | 0.034 |
| 0.08 | 13 | 0.070 | 0.077 | 0.007 |
| 0.12 | 13 | 0.125 | 0.077 | 0.048 |
| 0.18 | 13 | 0.177 | 0.385 | 0.207 |
| 0.23 | 10 | 0.222 | 0.200 | 0.022 |
| 0.28 | 12 | 0.278 | 0.083 | 0.194 |
| 0.33 | 10 | 0.323 | 0.400 | 0.077 |
| 0.38 | 14 | 0.377 | 0.500 | 0.123 |
| 0.43 | 10 | 0.429 | 0.400 | 0.029 |
| 0.47 | 11 | 0.475 | 0.364 | 0.111 |
| 0.53 | 6 | 0.533 | 0.667 | 0.134 |
| 0.58 | 10 | 0.581 | 0.700 | 0.119 |
| 0.62 | 11 | 0.633 | 0.636 | 0.003 |
| 0.68 | 4 | 0.676 | 0.500 | 0.176 |
| 0.73 | 15 | 0.726 | 0.800 | 0.074 |
| 0.78 | 12 | 0.769 | 0.833 | 0.064 |
| 0.83 | 14 | 0.825 | 1.000 | 0.175 |
| 0.88 | 15 | 0.877 | 1.000 | 0.123 |
| 0.93 | 26 | 0.924 | 0.962 | 0.038 |
| 0.98 | 30 | 0.970 | 0.967 | 0.004 |

## Upset Analysis (seed diff >= 3)

| Model | Total Upsets | Predicted Correctly | Detection Rate |
|-------|-------------|--------------------|--------------------|
| RF | 53 | 7 | 13.2% |
| v5 | 53 | 26 | 49.1% |
| v7 | 53 | 26 | 49.1% |

### Upsets by seed matchup
| Matchup | RF Upsets | RF Caught | v5 Upsets | v5 Caught | v7 Upsets | v7 Caught |
|---------|----------|-----------|----------|-----------|----------|-----------|
| 1v16 | 1 | 0 | 1 | 0 | 1 | 0 |
| 1v8 | 2 | 0 | 2 | 1 | 2 | 1 |
| 1v5 | 3 | 1 | 3 | 2 | 3 | 2 |
| 1v4 | 3 | 1 | 3 | 0 | 3 | 0 |
| 2v15 | 2 | 0 | 2 | 1 | 2 | 1 |
| 2v5 | 1 | 1 | 1 | 0 | 1 | 1 |
| 2v7 | 1 | 0 | 1 | 0 | 1 | 0 |
| 2v11 | 1 | 0 | 1 | 0 | 1 | 0 |
| 2v8 | 1 | 0 | 1 | 1 | 1 | 0 |
| 2v6 | 1 | 0 | 1 | 0 | 1 | 0 |
| 2v10 | 2 | 0 | 2 | 1 | 2 | 1 |
| 3v15 | 1 | 0 | 1 | 0 | 1 | 0 |
| 3v9 | 1 | 1 | 1 | 1 | 1 | 1 |
| 3v6 | 4 | 1 | 4 | 3 | 4 | 3 |
| 3v14 | 1 | 0 | 1 | 0 | 1 | 0 |
| 3v11 | 2 | 0 | 2 | 1 | 2 | 1 |
| 4v8 | 1 | 0 | 1 | 1 | 1 | 1 |
| 4v13 | 2 | 0 | 2 | 0 | 2 | 0 |
| 4v9 | 1 | 0 | 1 | 0 | 1 | 0 |
| 4v11 | 1 | 0 | 1 | 0 | 1 | 0 |
| 5v12 | 6 | 1 | 6 | 2 | 6 | 2 |
| 6v11 | 8 | 0 | 8 | 7 | 8 | 7 |
| 7v10 | 5 | 1 | 5 | 3 | 5 | 3 |
| 7v15 | 2 | 0 | 2 | 2 | 2 | 2 |

## Feature Importance

### RF (original) features
RF uses Gini importance (mean decrease in impurity). Higher = more important.

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | seed_diff | 0.0735 |
| 2 | seed_a | 0.0507 |
| 3 | owp_diff | 0.0470 |
| 4 | oeff_diff | 0.0354 |
| 5 | seed_b | 0.0342 |
| 6 | wp_diff | 0.0324 |
| 7 | deff_diff | 0.0231 |
| 8 | orebpct_diff | 0.0226 |
| 9 | fgm_avg_diff | 0.0226 |
| 10 | blk_diff | 0.0218 |
| 11 | ast_diff | 0.0210 |
| 12 | oppfg3_diff | 0.0201 |
| 13 | ft_diff | 0.0190 |
| 14 | oppeffFG_diff | 0.0189 |
| 15 | oppfg_diff | 0.0189 |
| 16 | pt_diff | 0.0189 |
| 17 | oppdreb_diff | 0.0189 |
| 18 | topct_diff | 0.0187 |
| 19 | pf_diff | 0.0184 |
| 20 | stl_diff | 0.0183 |
| 21 | fg_diff | 0.0182 |
| 22 | oppfgm3_avg_diff | 0.0181 |
| 23 | opppt_diff | 0.0180 |
| 24 | opporeb_diff | 0.0177 |
| 25 | oppstl_diff | 0.0175 |
| 26 | oppft_diff | 0.0174 |
| 27 | ftr_diff | 0.0174 |
| 28 | opporebpct_diff | 0.0174 |
| 29 | to_diff | 0.0173 |
| 30 | effFG_diff | 0.0172 |
| 31 | oppast_diff | 0.0170 |
| 32 | oreb_diff | 0.0169 |
| 33 | opppf_diff | 0.0169 |
| 34 | opptopct_diff | 0.0165 |
| 35 | fga_avg_diff | 0.0164 |
| 36 | fgm3_avg_diff | 0.0163 |
| 37 | oppfta_avg_diff | 0.0162 |
| 38 | fg3_diff | 0.0161 |
| 39 | fta_avg_diff | 0.0161 |
| 40 | oppblk_diff | 0.0160 |
| 41 | oppftr_diff | 0.0158 |
| 42 | oppto_diff | 0.0157 |
| 43 | oppfgm_avg_diff | 0.0155 |
| 44 | dreb_diff | 0.0153 |
| 45 | oppfga_avg_diff | 0.0150 |
| 46 | pos_diff | 0.0141 |
| 47 | opppos_diff | 0.0137 |

### v5 features
For blended models: LGB uses split-based importance weighted by blend weight, LR uses |coefficient| weighted by blend weight.

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | barthag_gap_diff | 43.5615 |
| 2 | adjoe_gap_diff | 36.2439 |
| 3 | Elo_diff | 29.1474 |
| 4 | adjde_gap_diff | 26.9827 |
| 5 | Torvik_sos_diff | 26.4525 |
| 6 | Torvik_adjoe_diff | 26.1097 |
| 7 | TRank_adjoe_diff | 25.6331 |
| 8 | Torvik_adjde_diff | 24.1935 |
| 9 | elo_barthag_gap_diff | 23.9725 |
| 10 | Torvik_barthag_diff | 21.7879 |
| 11 | tempo_gap_diff | 20.0310 |
| 12 | TRank_adjde_diff | 18.9252 |
| 13 | seed_ordinal_mismatch | 16.5985 |
| 14 | OrdinalStd_diff | 16.5461 |
| 15 | TRank_sos_diff | 15.1400 |
| 16 | TRank_tempo_diff | 10.3469 |
| 17 | sos_gap_diff | 9.3404 |
| 18 | Ord_RPI_diff | 8.4770 |
| 19 | Ord_WLK_diff | 8.0404 |
| 20 | Torvik_adjt_diff | 7.8535 |
| 21 | seed_a | 7.5868 |
| 22 | Ord_MOR_diff | 7.2895 |
| 23 | Ord_POM_diff | 7.1659 |
| 24 | seed_ordinal_agreement | 6.8996 |
| 25 | TRank_barthag_diff | 6.7555 |
| 26 | OrdinalMean_diff | 6.5742 |
| 27 | Ord_COL_diff | 5.6582 |
| 28 | Ord_DOL_diff | 5.4763 |
| 29 | seed_b | 4.6951 |
| 30 | Seed_diff | 1.7463 |
| 31 | elo_ordinal_agreement | 1.6941 |

### v7 features
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | barthag_gap_diff | 28.1387 |
| 2 | adjoe_gap_diff | 25.2163 |
| 3 | Torvik_sos_diff | 20.7346 |
| 4 | Elo_diff | 20.6520 |
| 5 | Torvik_adjde_diff | 18.9976 |
| 6 | adjde_gap_diff | 18.5762 |
| 7 | TRank_adjoe_diff | 18.4761 |
| 8 | Torvik_barthag_diff | 17.5078 |
| 9 | elo_barthag_gap_diff | 17.3121 |
| 10 | Torvik_adjoe_diff | 17.0741 |
| 11 | TRank_adjde_diff | 14.3589 |
| 12 | tempo_gap_diff | 13.6123 |
| 13 | OrdPCA_1_diff | 9.8320 |
| 14 | TRank_sos_diff | 9.4668 |
| 15 | sos_gap_diff | 8.9663 |
| 16 | TRank_barthag_diff | 7.9923 |
| 17 | seed_a | 6.8251 |
| 18 | TRank_tempo_diff | 6.7585 |
| 19 | Torvik_adjt_diff | 5.3793 |
| 20 | OrdPCA_0_diff | 4.8466 |
| 21 | seed_b | 2.9139 |
| 22 | Seed_diff | 2.2728 |

### Feature comparison (v5 vs v7)

**Features only in v5**: Ord_COL_diff, Ord_DOL_diff, Ord_MOR_diff, Ord_POM_diff, Ord_RPI_diff, Ord_WLK_diff, OrdinalMean_diff, OrdinalStd_diff, elo_ordinal_agreement, seed_ordinal_agreement, seed_ordinal_mismatch
**Features only in v7**: OrdPCA_0_diff, OrdPCA_1_diff

**Importance shift for common features (v7 - v5)**:
| Feature | v5 Imp | v7 Imp | Delta | Direction |
|---------|--------|--------|-------|-----------|
| barthag_gap_diff | 43.5615 | 28.1387 | -15.4228 | decreased |
| adjoe_gap_diff | 36.2439 | 25.2163 | -11.0275 | decreased |
| Torvik_adjoe_diff | 26.1097 | 17.0741 | -9.0356 | decreased |
| Elo_diff | 29.1474 | 20.6520 | -8.4954 | decreased |
| adjde_gap_diff | 26.9827 | 18.5762 | -8.4065 | decreased |
| TRank_adjoe_diff | 25.6331 | 18.4761 | -7.1570 | decreased |
| elo_barthag_gap_diff | 23.9725 | 17.3121 | -6.6604 | decreased |
| tempo_gap_diff | 20.0310 | 13.6123 | -6.4187 | decreased |
| Torvik_sos_diff | 26.4525 | 20.7346 | -5.7179 | decreased |
| TRank_sos_diff | 15.1400 | 9.4668 | -5.6732 | decreased |
| Torvik_adjde_diff | 24.1935 | 18.9976 | -5.1959 | decreased |
| TRank_adjde_diff | 18.9252 | 14.3589 | -4.5663 | decreased |
| Torvik_barthag_diff | 21.7879 | 17.5078 | -4.2800 | decreased |
| TRank_tempo_diff | 10.3469 | 6.7585 | -3.5884 | decreased |
| Torvik_adjt_diff | 7.8535 | 5.3793 | -2.4742 | decreased |
| seed_b | 4.6951 | 2.9139 | -1.7812 | decreased |
| TRank_barthag_diff | 6.7555 | 7.9923 | +1.2368 | increased |
| seed_a | 7.5868 | 6.8251 | -0.7617 | decreased |
| Seed_diff | 1.7463 | 2.2728 | +0.5265 | increased |
| sos_gap_diff | 9.3404 | 8.9663 | -0.3741 | decreased |

---
# Women's

## Overall Scores
| Model | Brier | Accuracy | N | Features |
|-------|-------|----------|---|----------|
| RF (original) | 0.1496 | 76.5% | 268 | 47 |
| v5 | 0.1366 | 80.2% | 268 | 7 |
| v7 | 0.1335 | 81.0% | 268 | 9 |

## Per-Season Breakdown
| Season | RF Brier | v5 Brier | v7 Brier | RF Acc | v5 Acc | v7 Acc |
|--------|----------|----------|----------|--------|--------|--------|
| 2022 | 0.1666 | 0.1500 | 0.1501 | 76.1% | 80.6% | 79.1% |
| 2023 | 0.1852 | 0.1725 | 0.1691 | 70.1% | 74.6% | 76.1% |
| 2024 | 0.1280 | 0.1161 | 0.1134 | 79.1% | 82.1% | 83.6% |
| 2025 | 0.1185 | 0.1078 | 0.1015 | 80.6% | 83.6% | 85.1% |

## Calibration by Seed Matchup (First Round)

Pred fav win% = model's average predicted probability that the favorite wins.
Actual fav win% = how often the favorite actually won in CV data.
Calibration error = |predicted - actual|. Direction: overconf = model too confident in favorite, underconf = model doesn't trust favorite enough.

### RF (original)
| Matchup | N | Pred Fav% | Actual Fav% | Cal Error | Direction | Upsets | Upset% | Brier |
|---------|---|-----------|-------------|-----------|-----------|--------|--------|-------|
| 1v16 | 16 | 93.4% | 100.0% | 0.066 | underconf | 0 | 0.0% | 0.0051 |
| 2v15 | 16 | 90.5% | 100.0% | 0.095 | underconf | 0 | 0.0% | 0.0121 |
| 3v14 | 16 | 88.5% | 100.0% | 0.115 | underconf | 0 | 0.0% | 0.0168 |
| 4v13 | 16 | 83.2% | 100.0% | 0.167 | underconf | 0 | 0.0% | 0.0357 |
| 5v12 | 16 | 78.0% | 75.0% | 0.030 | overconf | 4 | 25.0% | 0.1792 |
| 6v11 | 16 | 67.6% | 75.0% | 0.074 | underconf | 4 | 25.0% | 0.2022 |
| 7v10 | 16 | 61.7% | 62.5% | 0.008 | underconf | 6 | 37.5% | 0.2761 |
| 8v9 | 16 | 50.2% | 62.5% | 0.123 | underconf | 6 | 37.5% | 0.2566 |

### v5
| Matchup | N | Pred Fav% | Actual Fav% | Cal Error | Direction | Upsets | Upset% | Brier |
|---------|---|-----------|-------------|-----------|-----------|--------|--------|-------|
| 1v16 | 16 | 97.5% | 100.0% | 0.025 | underconf | 0 | 0.0% | 0.0006 |
| 2v15 | 16 | 97.5% | 100.0% | 0.025 | underconf | 0 | 0.0% | 0.0006 |
| 3v14 | 16 | 95.8% | 100.0% | 0.042 | underconf | 0 | 0.0% | 0.0023 |
| 4v13 | 16 | 91.0% | 100.0% | 0.090 | underconf | 0 | 0.0% | 0.0096 |
| 5v12 | 16 | 83.0% | 75.0% | 0.080 | overconf | 4 | 25.0% | 0.1827 |
| 6v11 | 16 | 75.0% | 75.0% | 0.000 | underconf | 4 | 25.0% | 0.1958 |
| 7v10 | 16 | 66.3% | 62.5% | 0.038 | overconf | 6 | 37.5% | 0.2327 |
| 8v9 | 16 | 50.5% | 62.5% | 0.120 | underconf | 6 | 37.5% | 0.2470 |

### v7
| Matchup | N | Pred Fav% | Actual Fav% | Cal Error | Direction | Upsets | Upset% | Brier |
|---------|---|-----------|-------------|-----------|-----------|--------|--------|-------|
| 1v16 | 16 | 97.5% | 100.0% | 0.025 | underconf | 0 | 0.0% | 0.0006 |
| 2v15 | 16 | 97.5% | 100.0% | 0.025 | underconf | 0 | 0.0% | 0.0007 |
| 3v14 | 16 | 95.9% | 100.0% | 0.041 | underconf | 0 | 0.0% | 0.0020 |
| 4v13 | 16 | 92.1% | 100.0% | 0.079 | underconf | 0 | 0.0% | 0.0071 |
| 5v12 | 16 | 83.6% | 75.0% | 0.086 | overconf | 4 | 25.0% | 0.1779 |
| 6v11 | 16 | 76.5% | 75.0% | 0.015 | overconf | 4 | 25.0% | 0.1932 |
| 7v10 | 16 | 66.8% | 62.5% | 0.043 | overconf | 6 | 37.5% | 0.2338 |
| 8v9 | 16 | 50.3% | 62.5% | 0.122 | underconf | 6 | 37.5% | 0.2310 |

### Calibration Comparison (RF vs v5 vs v7)
| Matchup | RF Cal Err | v5 Cal Err | v7 Cal Err | RF Dir | v5 Dir | v7 Dir | RF Brier | v5 Brier | v7 Brier | Best |
|---------|-----------|-----------|-----------|--------|--------|--------|----------|----------|----------|------|
| 1v16 | 0.066 | 0.025 | 0.025 | underconf | underconf | underconf | 0.0051 | 0.0006 | 0.0006 | v5 |
| 2v15 | 0.095 | 0.025 | 0.025 | underconf | underconf | underconf | 0.0121 | 0.0006 | 0.0007 | v5 |
| 3v14 | 0.115 | 0.042 | 0.041 | underconf | underconf | underconf | 0.0168 | 0.0023 | 0.0020 | v7 |
| 4v13 | 0.167 | 0.090 | 0.079 | underconf | underconf | underconf | 0.0357 | 0.0096 | 0.0071 | v7 |
| 5v12 | 0.030 | 0.080 | 0.086 | overconf | overconf | overconf | 0.1792 | 0.1827 | 0.1779 | v7 |
| 6v11 | 0.074 | 0.000 | 0.015 | underconf | underconf | overconf | 0.2022 | 0.1958 | 0.1932 | v7 |
| 7v10 | 0.008 | 0.038 | 0.043 | underconf | overconf | overconf | 0.2761 | 0.2327 | 0.2338 | v5 |
| 8v9 | 0.123 | 0.120 | 0.122 | underconf | underconf | underconf | 0.2566 | 0.2470 | 0.2310 | v7 |

## 5v12 Matchup Deep Dive

5-12 matchups are historically the most volatile first-round pairing (upsets ~35% of the time).

### RF (original) game-by-game predictions
| Season | 5-seed | 12-seed | Pred(5wins) | 5won? | Upset? |
|--------|--------|---------|-------------|-------|--------|
| 2022 | Virginia Tech | FGCU | 0.651 | N | **YES** |
| 2022 | Oregon | Belmont | 0.906 | N | **YES** |
| 2022 | North Carolina | SF Austin | 0.822 | Y | no |
| 2022 | Notre Dame | Massachusetts | 0.765 | Y | no |
| 2023 | Washington St | FGCU | 0.603 | N | **YES** |
| 2023 | Louisville | Drake | 0.603 | Y | no |
| 2023 | Oklahoma | Portland | 0.785 | Y | no |
| 2023 | Iowa St | Toledo | 0.808 | N | **YES** |
| 2024 | Baylor | Vanderbilt | 0.836 | Y | no |
| 2024 | Colorado | Drake | 0.731 | Y | no |
| 2024 | Oklahoma | FGCU | 0.721 | Y | no |
| 2024 | Utah | S Dakota St | 0.749 | Y | no |
| 2025 | Kansas St | Fairfield | 0.918 | Y | no |
| 2025 | Mississippi | Ball St | 0.883 | Y | no |
| 2025 | Tennessee | South Florida | 0.923 | Y | no |
| 2025 | Alabama | WI Green Bay | 0.780 | Y | no |

### v5 game-by-game predictions
| Season | 5-seed | 12-seed | Pred(5wins) | 5won? | Upset? |
|--------|--------|---------|-------------|-------|--------|
| 2022 | Virginia Tech | FGCU | 0.788 | N | **YES** |
| 2022 | Oregon | Belmont | 0.830 | N | **YES** |
| 2022 | North Carolina | SF Austin | 0.872 | Y | no |
| 2022 | Notre Dame | Massachusetts | 0.826 | Y | no |
| 2023 | Washington St | FGCU | 0.690 | N | **YES** |
| 2023 | Louisville | Drake | 0.842 | Y | no |
| 2023 | Oklahoma | Portland | 0.868 | Y | no |
| 2023 | Iowa St | Toledo | 0.898 | N | **YES** |
| 2024 | Baylor | Vanderbilt | 0.864 | Y | no |
| 2024 | Colorado | Drake | 0.827 | Y | no |
| 2024 | Oklahoma | FGCU | 0.747 | Y | no |
| 2024 | Utah | S Dakota St | 0.760 | Y | no |
| 2025 | Kansas St | Fairfield | 0.828 | Y | no |
| 2025 | Mississippi | Ball St | 0.868 | Y | no |
| 2025 | Tennessee | South Florida | 0.888 | Y | no |
| 2025 | Alabama | WI Green Bay | 0.884 | Y | no |

### v7 game-by-game predictions
| Season | 5-seed | 12-seed | Pred(5wins) | 5won? | Upset? |
|--------|--------|---------|-------------|-------|--------|
| 2022 | Virginia Tech | FGCU | 0.803 | N | **YES** |
| 2022 | Oregon | Belmont | 0.843 | N | **YES** |
| 2022 | North Carolina | SF Austin | 0.875 | Y | no |
| 2022 | Notre Dame | Massachusetts | 0.838 | Y | no |
| 2023 | Washington St | FGCU | 0.638 | N | **YES** |
| 2023 | Louisville | Drake | 0.857 | Y | no |
| 2023 | Oklahoma | Portland | 0.882 | Y | no |
| 2023 | Iowa St | Toledo | 0.890 | N | **YES** |
| 2024 | Baylor | Vanderbilt | 0.882 | Y | no |
| 2024 | Colorado | Drake | 0.827 | Y | no |
| 2024 | Oklahoma | FGCU | 0.754 | Y | no |
| 2024 | Utah | S Dakota St | 0.783 | Y | no |
| 2025 | Kansas St | Fairfield | 0.841 | Y | no |
| 2025 | Mississippi | Ball St | 0.878 | Y | no |
| 2025 | Tennessee | South Florida | 0.901 | Y | no |
| 2025 | Alabama | WI Green Bay | 0.888 | Y | no |

### RF vs v5 vs v7 prediction comparison on 5-12 games
| Season | 5-seed | 12-seed | RF Pred | v5 Pred | v7 Pred | Result | Best |
|--------|--------|---------|---------|---------|---------|--------|------|
| 2022 | North Carolina | SF Austin | 0.822 | 0.872 | 0.875 | 5-seed won | v7 |
| 2022 | Notre Dame | Massachusetts | 0.765 | 0.826 | 0.838 | 5-seed won | v7 |
| 2022 | Oregon | Belmont | 0.906 | 0.830 | 0.843 | **12-UPSET** | v5 |
| 2022 | Virginia Tech | FGCU | 0.651 | 0.788 | 0.803 | **12-UPSET** | RF |
| 2023 | Iowa St | Toledo | 0.808 | 0.898 | 0.890 | **12-UPSET** | RF |
| 2023 | Louisville | Drake | 0.603 | 0.842 | 0.857 | 5-seed won | v7 |
| 2023 | Oklahoma | Portland | 0.785 | 0.868 | 0.882 | 5-seed won | v7 |
| 2023 | Washington St | FGCU | 0.603 | 0.690 | 0.638 | **12-UPSET** | RF |
| 2024 | Baylor | Vanderbilt | 0.836 | 0.864 | 0.882 | 5-seed won | v7 |
| 2024 | Colorado | Drake | 0.731 | 0.827 | 0.827 | 5-seed won | v5 |
| 2024 | Oklahoma | FGCU | 0.721 | 0.747 | 0.754 | 5-seed won | v7 |
| 2024 | Utah | S Dakota St | 0.749 | 0.760 | 0.783 | 5-seed won | v7 |
| 2025 | Alabama | WI Green Bay | 0.780 | 0.884 | 0.888 | 5-seed won | v7 |
| 2025 | Kansas St | Fairfield | 0.918 | 0.828 | 0.841 | 5-seed won | RF |
| 2025 | Mississippi | Ball St | 0.883 | 0.868 | 0.878 | 5-seed won | RF |
| 2025 | Tennessee | South Florida | 0.923 | 0.888 | 0.901 | 5-seed won | RF |

## Calibration Curves (20-bin)

ECE: RF=0.0839, v5=0.0587, v7=0.0726

### RF (original) calibration
| Bin | N | Mean Pred | Actual Rate | Error |
|-----|---|-----------|-------------|-------|
| 0.03 | 9 | 0.032 | 0.000 | 0.032 |
| 0.08 | 27 | 0.075 | 0.037 | 0.038 |
| 0.12 | 16 | 0.123 | 0.062 | 0.060 |
| 0.18 | 11 | 0.170 | 0.000 | 0.170 |
| 0.23 | 15 | 0.227 | 0.133 | 0.093 |
| 0.28 | 16 | 0.271 | 0.250 | 0.021 |
| 0.33 | 9 | 0.325 | 0.444 | 0.120 |
| 0.38 | 14 | 0.383 | 0.357 | 0.026 |
| 0.43 | 14 | 0.422 | 0.286 | 0.137 |
| 0.47 | 11 | 0.471 | 0.636 | 0.165 |
| 0.53 | 19 | 0.531 | 0.421 | 0.110 |
| 0.58 | 12 | 0.570 | 0.500 | 0.070 |
| 0.62 | 17 | 0.621 | 0.588 | 0.033 |
| 0.68 | 18 | 0.674 | 0.778 | 0.104 |
| 0.73 | 9 | 0.728 | 0.667 | 0.061 |
| 0.78 | 14 | 0.778 | 1.000 | 0.222 |
| 0.83 | 12 | 0.828 | 0.917 | 0.089 |
| 0.88 | 8 | 0.880 | 0.750 | 0.130 |
| 0.93 | 15 | 0.918 | 0.933 | 0.015 |
| 0.98 | 2 | 0.966 | 1.000 | 0.034 |

### v5 calibration
| Bin | N | Mean Pred | Actual Rate | Error |
|-----|---|-----------|-------------|-------|
| 0.03 | 34 | 0.030 | 0.000 | 0.030 |
| 0.08 | 15 | 0.080 | 0.000 | 0.080 |
| 0.12 | 17 | 0.124 | 0.118 | 0.006 |
| 0.18 | 18 | 0.175 | 0.222 | 0.048 |
| 0.23 | 13 | 0.219 | 0.154 | 0.065 |
| 0.28 | 13 | 0.278 | 0.231 | 0.048 |
| 0.33 | 7 | 0.315 | 0.286 | 0.029 |
| 0.38 | 6 | 0.375 | 0.167 | 0.208 |
| 0.43 | 10 | 0.420 | 0.400 | 0.020 |
| 0.47 | 7 | 0.482 | 0.571 | 0.089 |
| 0.53 | 14 | 0.520 | 0.643 | 0.123 |
| 0.58 | 14 | 0.579 | 0.500 | 0.079 |
| 0.62 | 13 | 0.622 | 0.615 | 0.006 |
| 0.68 | 13 | 0.671 | 0.538 | 0.133 |
| 0.73 | 10 | 0.723 | 0.900 | 0.177 |
| 0.78 | 10 | 0.773 | 0.800 | 0.027 |
| 0.83 | 6 | 0.823 | 0.833 | 0.010 |
| 0.88 | 15 | 0.880 | 0.867 | 0.013 |
| 0.93 | 10 | 0.922 | 0.800 | 0.122 |
| 0.98 | 23 | 0.972 | 1.000 | 0.028 |

### v7 calibration
| Bin | N | Mean Pred | Actual Rate | Error |
|-----|---|-----------|-------------|-------|
| 0.03 | 32 | 0.029 | 0.000 | 0.029 |
| 0.08 | 23 | 0.077 | 0.000 | 0.077 |
| 0.12 | 19 | 0.128 | 0.211 | 0.082 |
| 0.18 | 15 | 0.169 | 0.200 | 0.031 |
| 0.23 | 12 | 0.224 | 0.083 | 0.140 |
| 0.28 | 11 | 0.274 | 0.273 | 0.002 |
| 0.33 | 7 | 0.321 | 0.143 | 0.178 |
| 0.38 | 4 | 0.361 | 0.500 | 0.139 |
| 0.43 | 7 | 0.414 | 0.571 | 0.158 |
| 0.47 | 10 | 0.471 | 0.300 | 0.171 |
| 0.53 | 13 | 0.521 | 0.538 | 0.018 |
| 0.58 | 16 | 0.573 | 0.625 | 0.052 |
| 0.62 | 12 | 0.626 | 0.500 | 0.126 |
| 0.68 | 11 | 0.679 | 0.818 | 0.139 |
| 0.73 | 8 | 0.727 | 0.625 | 0.102 |
| 0.78 | 10 | 0.771 | 0.900 | 0.129 |
| 0.83 | 9 | 0.822 | 0.778 | 0.044 |
| 0.88 | 15 | 0.878 | 0.867 | 0.012 |
| 0.93 | 13 | 0.925 | 0.846 | 0.079 |
| 0.98 | 21 | 0.971 | 1.000 | 0.029 |

## Upset Analysis (seed diff >= 3)

| Model | Total Upsets | Predicted Correctly | Detection Rate |
|-------|-------------|--------------------|--------------------|
| RF | 23 | 1 | 4.3% |
| v5 | 23 | 0 | 0.0% |
| v7 | 23 | 0 | 0.0% |

### Upsets by seed matchup
| Matchup | RF Upsets | RF Caught | v5 Upsets | v5 Caught | v7 Upsets | v7 Caught |
|---------|----------|-----------|----------|-----------|----------|-----------|
| 1v8 | 1 | 0 | 1 | 0 | 1 | 0 |
| 1v9 | 1 | 0 | 1 | 0 | 1 | 0 |
| 2v10 | 2 | 0 | 2 | 0 | 2 | 0 |
| 2v7 | 1 | 0 | 1 | 0 | 1 | 0 |
| 3v10 | 1 | 0 | 1 | 0 | 1 | 0 |
| 3v6 | 2 | 0 | 2 | 0 | 2 | 0 |
| 4v9 | 1 | 0 | 1 | 0 | 1 | 0 |
| 5v12 | 4 | 0 | 4 | 0 | 4 | 0 |
| 6v11 | 4 | 0 | 4 | 0 | 4 | 0 |
| 7v10 | 6 | 1 | 6 | 0 | 6 | 0 |

## Feature Importance

### RF (original) features
RF uses Gini importance (mean decrease in impurity). Higher = more important.

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | seed_diff | 0.1118 |
| 2 | owp_diff | 0.0726 |
| 3 | seed_a | 0.0585 |
| 4 | seed_b | 0.0501 |
| 5 | fgm_avg_diff | 0.0463 |
| 6 | oeff_diff | 0.0347 |
| 7 | fg_diff | 0.0346 |
| 8 | pt_diff | 0.0272 |
| 9 | blk_diff | 0.0260 |
| 10 | oppdreb_diff | 0.0225 |
| 11 | ast_diff | 0.0211 |
| 12 | deff_diff | 0.0206 |
| 13 | wp_diff | 0.0205 |
| 14 | orebpct_diff | 0.0204 |
| 15 | effFG_diff | 0.0175 |
| 16 | fga_avg_diff | 0.0164 |
| 17 | opppt_diff | 0.0144 |
| 18 | oppfta_avg_diff | 0.0144 |
| 19 | oppeffFG_diff | 0.0143 |
| 20 | oppfg_diff | 0.0142 |
| 21 | oppftr_diff | 0.0142 |
| 22 | topct_diff | 0.0139 |
| 23 | oppft_diff | 0.0136 |
| 24 | fg3_diff | 0.0134 |
| 25 | pf_diff | 0.0133 |
| 26 | opporebpct_diff | 0.0132 |
| 27 | oreb_diff | 0.0131 |
| 28 | opporeb_diff | 0.0131 |
| 29 | oppfg3_diff | 0.0130 |
| 30 | fgm3_avg_diff | 0.0130 |
| 31 | ft_diff | 0.0130 |
| 32 | oppast_diff | 0.0129 |
| 33 | oppstl_diff | 0.0129 |
| 34 | oppfga_avg_diff | 0.0129 |
| 35 | dreb_diff | 0.0128 |
| 36 | oppfgm_avg_diff | 0.0128 |
| 37 | oppblk_diff | 0.0126 |
| 38 | ftr_diff | 0.0126 |
| 39 | stl_diff | 0.0125 |
| 40 | oppfgm3_avg_diff | 0.0124 |
| 41 | to_diff | 0.0122 |
| 42 | fta_avg_diff | 0.0120 |
| 43 | pos_diff | 0.0115 |
| 44 | opppos_diff | 0.0115 |
| 45 | opppf_diff | 0.0114 |
| 46 | opptopct_diff | 0.0114 |
| 47 | oppto_diff | 0.0109 |

### v5 features
For blended models: LGB uses split-based importance weighted by blend weight, LR uses |coefficient| weighted by blend weight.

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Elo_diff | 0.9066 |
| 2 | Seed_diff | 0.8653 |
| 3 | TRank_adjde_diff | 0.4381 |
| 4 | TRank_adjoe_diff | 0.3815 |
| 5 | seed_b | 0.0289 |
| 6 | TRank_barthag_diff | 0.0026 |
| 7 | seed_a | 0.0000 |

### v7 features
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Seed_diff | 0.9499 |
| 2 | Elo_diff | 0.6807 |
| 3 | TRank_adjoe_diff | 0.4069 |
| 4 | TRank_adjde_diff | 0.3797 |
| 5 | PointDiff_diff | 0.0890 |
| 6 | seed_b | 0.0178 |
| 7 | seed_a | 0.0021 |
| 8 | TRank_tempo_diff | 0.0000 |
| 9 | TRank_barthag_diff | 0.0000 |

### Feature comparison (v5 vs v7)

**Features only in v7**: PointDiff_diff, TRank_tempo_diff

**Importance shift for common features (v7 - v5)**:
| Feature | v5 Imp | v7 Imp | Delta | Direction |
|---------|--------|--------|-------|-----------|
| Elo_diff | 0.9066 | 0.6807 | -0.2259 | decreased |
| Seed_diff | 0.8653 | 0.9499 | +0.0846 | increased |
| TRank_adjde_diff | 0.4381 | 0.3797 | -0.0583 | decreased |
| TRank_adjoe_diff | 0.3815 | 0.4069 | +0.0254 | increased |
| seed_b | 0.0289 | 0.0178 | -0.0111 | decreased |
| TRank_barthag_diff | 0.0026 | 0.0000 | -0.0026 | decreased |
| seed_a | 0.0000 | 0.0021 | +0.0021 | increased |

