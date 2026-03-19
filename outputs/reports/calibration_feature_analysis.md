# Calibration, Upset & Feature Importance Analysis: v5 vs v7

Generated 2026-03-18. Leave-season-out CV on 2022-2025 tournaments.

---
# Men's

## Overall Scores
| Model | Brier | Accuracy | N |
|-------|-------|----------|---|
| v5 | 0.1334 | 81.3% | 268 |
| v7 | 0.1266 | 82.5% | 268 |
| delta | -0.0068 | +1.1% | |

## Per-Season Breakdown
| Season | v5 Brier | v7 Brier | Delta | v5 Acc | v7 Acc |
|--------|----------|----------|-------|--------|--------|
| 2022 | 0.1257 | 0.1230 | -0.0027 | 83.6% | 85.1% |
| 2023 | 0.1478 | 0.1442 | -0.0036 | 76.1% | 79.1% |
| 2024 | 0.1499 | 0.1383 | -0.0116 | 79.1% | 80.6% |
| 2025 | 0.1104 | 0.1011 | -0.0093 | 86.6% | 85.1% |

## Calibration by Seed Matchup (First Round)

Pred fav win% = model's average predicted probability that the favorite wins.
Actual fav win% = how often the favorite actually won in CV data.
Calibration error = |predicted - actual|. Direction: overconf = model too confident in favorite, underconf = model doesn't trust favorite enough.

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

### Calibration Comparison (v5 vs v7)
| Matchup | v5 Cal Error | v7 Cal Error | v5 Dir | v7 Dir | v5 Brier | v7 Brier | Better |
|---------|-------------|-------------|--------|--------|----------|----------|--------|
| 1v16 | 0.024 | 0.007 | overconf | overconf | 0.0570 | 0.0254 | v7 |
| 2v15 | 0.002 | 0.012 | overconf | overconf | 0.0372 | 0.0411 | v5 |
| 3v14 | 0.020 | 0.021 | underconf | underconf | 0.0483 | 0.0496 | v5 |
| 4v13 | 0.004 | 0.012 | underconf | underconf | 0.1300 | 0.1297 | v7 |
| 5v12 | 0.144 | 0.160 | overconf | overconf | 0.1950 | 0.2091 | v5 |
| 6v11 | 0.022 | 0.004 | overconf | overconf | 0.1468 | 0.1540 | v5 |
| 7v10 | 0.131 | 0.129 | underconf | underconf | 0.2225 | 0.2020 | v7 |
| 8v9 | 0.076 | 0.090 | overconf | overconf | 0.1897 | 0.1640 | v7 |

## 5v12 Matchup Deep Dive

5-12 matchups are historically the most volatile first-round pairing (upsets ~35% of the time).

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

### v5 vs v7 prediction comparison on 5-12 games
| Season | 5-seed | 12-seed | v5 Pred | v7 Pred | Delta | Result | Comment |
|--------|--------|---------|---------|---------|-------|--------|---------|
| 2022 | Connecticut | New Mexico St | 0.683 | 0.664 | -0.019 | **12-SEED UPSET** | v7 better (lower fav confidence) |
| 2022 | Houston | UAB | 0.973 | 0.975 | +0.002 | 5-seed won | v7 better (higher fav confidence) |
| 2022 | Iowa | Richmond | 0.950 | 0.945 | -0.005 | **12-SEED UPSET** | v7 better (lower fav confidence) |
| 2022 | St Mary's CA | Indiana | 0.486 | 0.551 | +0.065 | 5-seed won | v7 better (higher fav confidence) |
| 2023 | Duke | Oral Roberts | 0.975 | 0.975 | +0.000 | 5-seed won | v5 better (higher fav confidence) |
| 2023 | Miami FL | Drake | 0.805 | 0.826 | +0.021 | 5-seed won | v7 better (higher fav confidence) |
| 2023 | San Diego St | Col Charleston | 0.913 | 0.905 | -0.008 | 5-seed won | v5 better (higher fav confidence) |
| 2023 | St Mary's CA | VCU | 0.946 | 0.878 | -0.068 | 5-seed won | v5 better (higher fav confidence) |
| 2024 | Gonzaga | McNeese St | 0.970 | 0.964 | -0.006 | 5-seed won | v5 better (higher fav confidence) |
| 2024 | San Diego St | UAB | 0.877 | 0.860 | -0.017 | 5-seed won | v5 better (higher fav confidence) |
| 2024 | St Mary's CA | Grand Canyon | 0.760 | 0.837 | +0.077 | **12-SEED UPSET** | v5 better (lower fav confidence) |
| 2024 | Wisconsin | James Madison | 0.776 | 0.841 | +0.065 | **12-SEED UPSET** | v5 better (lower fav confidence) |
| 2025 | Clemson | McNeese St | 0.383 | 0.451 | +0.068 | **12-SEED UPSET** | v5 better (lower fav confidence) |
| 2025 | Memphis | Colorado St | 0.124 | 0.210 | +0.086 | **12-SEED UPSET** | v5 better (lower fav confidence) |
| 2025 | Michigan | UC San Diego | 0.966 | 0.950 | -0.016 | 5-seed won | v5 better (higher fav confidence) |
| 2025 | Oregon | Liberty | 0.723 | 0.724 | +0.001 | 5-seed won | v7 better (higher fav confidence) |

## Calibration Curves (20-bin)

ECE: v5=0.1093, v7=0.0769

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
| v5 | 53 | 26 | 49.1% |
| v7 | 53 | 26 | 49.1% |

### Upsets by seed matchup
| Matchup | v5 Upsets | v5 Caught | v7 Upsets | v7 Caught |
|---------|----------|-----------|----------|-----------|
| 1v5 | 3 | 2 | 3 | 2 |
| 1v16 | 1 | 0 | 1 | 0 |
| 1v4 | 3 | 0 | 3 | 0 |
| 1v8 | 2 | 1 | 2 | 1 |
| 2v8 | 1 | 1 | 1 | 0 |
| 2v5 | 1 | 0 | 1 | 1 |
| 2v10 | 2 | 1 | 2 | 1 |
| 2v6 | 1 | 0 | 1 | 0 |
| 2v7 | 1 | 0 | 1 | 0 |
| 2v15 | 2 | 1 | 2 | 1 |
| 2v11 | 1 | 0 | 1 | 0 |
| 3v15 | 1 | 0 | 1 | 0 |
| 3v6 | 4 | 3 | 4 | 3 |
| 3v9 | 1 | 1 | 1 | 1 |
| 3v14 | 1 | 0 | 1 | 0 |
| 3v11 | 2 | 1 | 2 | 1 |
| 4v8 | 1 | 1 | 1 | 1 |
| 4v11 | 1 | 0 | 1 | 0 |
| 4v13 | 2 | 0 | 2 | 0 |
| 4v9 | 1 | 0 | 1 | 0 |
| 5v12 | 6 | 2 | 6 | 2 |
| 6v11 | 8 | 7 | 8 | 7 |
| 7v15 | 2 | 2 | 2 | 2 |
| 7v10 | 5 | 3 | 5 | 3 |

## Feature Importance

### v5 features (importance = avg across CV folds)
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

### Feature comparison

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
| Model | Brier | Accuracy | N |
|-------|-------|----------|---|
| v5 | 0.1366 | 80.2% | 268 |
| v7 | 0.1335 | 81.0% | 268 |
| delta | -0.0030 | +0.7% | |

## Per-Season Breakdown
| Season | v5 Brier | v7 Brier | Delta | v5 Acc | v7 Acc |
|--------|----------|----------|-------|--------|--------|
| 2022 | 0.1499 | 0.1501 | +0.0001 | 80.6% | 79.1% |
| 2023 | 0.1725 | 0.1691 | -0.0034 | 74.6% | 76.1% |
| 2024 | 0.1161 | 0.1134 | -0.0027 | 82.1% | 83.6% |
| 2025 | 0.1078 | 0.1015 | -0.0062 | 83.6% | 85.1% |

## Calibration by Seed Matchup (First Round)

Pred fav win% = model's average predicted probability that the favorite wins.
Actual fav win% = how often the favorite actually won in CV data.
Calibration error = |predicted - actual|. Direction: overconf = model too confident in favorite, underconf = model doesn't trust favorite enough.

### v5
| Matchup | N | Pred Fav% | Actual Fav% | Cal Error | Direction | Upsets | Upset% | Brier |
|---------|---|-----------|-------------|-----------|-----------|--------|--------|-------|
| 1v16 | 16 | 97.5% | 100.0% | 0.025 | underconf | 0 | 0.0% | 0.0006 |
| 2v15 | 16 | 97.5% | 100.0% | 0.025 | underconf | 0 | 0.0% | 0.0006 |
| 3v14 | 16 | 95.8% | 100.0% | 0.042 | underconf | 0 | 0.0% | 0.0023 |
| 4v13 | 16 | 91.0% | 100.0% | 0.090 | underconf | 0 | 0.0% | 0.0097 |
| 5v12 | 16 | 83.0% | 75.0% | 0.080 | overconf | 4 | 25.0% | 0.1827 |
| 6v11 | 16 | 75.0% | 75.0% | 0.000 | underconf | 4 | 25.0% | 0.1958 |
| 7v10 | 16 | 66.3% | 62.5% | 0.038 | overconf | 6 | 37.5% | 0.2326 |
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

### Calibration Comparison (v5 vs v7)
| Matchup | v5 Cal Error | v7 Cal Error | v5 Dir | v7 Dir | v5 Brier | v7 Brier | Better |
|---------|-------------|-------------|--------|--------|----------|----------|--------|
| 1v16 | 0.025 | 0.025 | underconf | underconf | 0.0006 | 0.0006 | tie |
| 2v15 | 0.025 | 0.025 | underconf | underconf | 0.0006 | 0.0007 | v5 |
| 3v14 | 0.042 | 0.041 | underconf | underconf | 0.0023 | 0.0020 | v7 |
| 4v13 | 0.090 | 0.079 | underconf | underconf | 0.0097 | 0.0071 | v7 |
| 5v12 | 0.080 | 0.086 | overconf | overconf | 0.1827 | 0.1779 | v7 |
| 6v11 | 0.000 | 0.015 | underconf | overconf | 0.1958 | 0.1932 | v7 |
| 7v10 | 0.038 | 0.043 | overconf | overconf | 0.2326 | 0.2338 | v5 |
| 8v9 | 0.120 | 0.122 | underconf | underconf | 0.2470 | 0.2310 | v7 |

## 5v12 Matchup Deep Dive

5-12 matchups are historically the most volatile first-round pairing (upsets ~35% of the time).

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

### v5 vs v7 prediction comparison on 5-12 games
| Season | 5-seed | 12-seed | v5 Pred | v7 Pred | Delta | Result | Comment |
|--------|--------|---------|---------|---------|-------|--------|---------|
| 2022 | North Carolina | SF Austin | 0.872 | 0.875 | +0.003 | 5-seed won | v7 better (higher fav confidence) |
| 2022 | Notre Dame | Massachusetts | 0.826 | 0.838 | +0.012 | 5-seed won | v7 better (higher fav confidence) |
| 2022 | Oregon | Belmont | 0.830 | 0.843 | +0.013 | **12-SEED UPSET** | v5 better (lower fav confidence) |
| 2022 | Virginia Tech | FGCU | 0.788 | 0.803 | +0.015 | **12-SEED UPSET** | v5 better (lower fav confidence) |
| 2023 | Iowa St | Toledo | 0.898 | 0.890 | -0.008 | **12-SEED UPSET** | v7 better (lower fav confidence) |
| 2023 | Louisville | Drake | 0.842 | 0.857 | +0.015 | 5-seed won | v7 better (higher fav confidence) |
| 2023 | Oklahoma | Portland | 0.868 | 0.882 | +0.014 | 5-seed won | v7 better (higher fav confidence) |
| 2023 | Washington St | FGCU | 0.690 | 0.638 | -0.052 | **12-SEED UPSET** | v7 better (lower fav confidence) |
| 2024 | Baylor | Vanderbilt | 0.864 | 0.882 | +0.018 | 5-seed won | v7 better (higher fav confidence) |
| 2024 | Colorado | Drake | 0.827 | 0.827 | +0.000 | 5-seed won | v5 better (higher fav confidence) |
| 2024 | Oklahoma | FGCU | 0.747 | 0.754 | +0.007 | 5-seed won | v7 better (higher fav confidence) |
| 2024 | Utah | S Dakota St | 0.760 | 0.783 | +0.023 | 5-seed won | v7 better (higher fav confidence) |
| 2025 | Alabama | WI Green Bay | 0.884 | 0.888 | +0.004 | 5-seed won | v7 better (higher fav confidence) |
| 2025 | Kansas St | Fairfield | 0.828 | 0.841 | +0.013 | 5-seed won | v7 better (higher fav confidence) |
| 2025 | Mississippi | Ball St | 0.868 | 0.878 | +0.010 | 5-seed won | v7 better (higher fav confidence) |
| 2025 | Tennessee | South Florida | 0.888 | 0.901 | +0.013 | 5-seed won | v7 better (higher fav confidence) |

## Calibration Curves (20-bin)

ECE: v5=0.0587, v7=0.0726

### v5 calibration
| Bin | N | Mean Pred | Actual Rate | Error |
|-----|---|-----------|-------------|-------|
| 0.03 | 34 | 0.030 | 0.000 | 0.030 |
| 0.08 | 15 | 0.080 | 0.000 | 0.080 |
| 0.12 | 17 | 0.124 | 0.118 | 0.006 |
| 0.18 | 18 | 0.175 | 0.222 | 0.047 |
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
| v5 | 23 | 0 | 0.0% |
| v7 | 23 | 0 | 0.0% |

### Upsets by seed matchup
| Matchup | v5 Upsets | v5 Caught | v7 Upsets | v7 Caught |
|---------|----------|-----------|----------|-----------|
| 1v9 | 1 | 0 | 1 | 0 |
| 1v8 | 1 | 0 | 1 | 0 |
| 2v10 | 2 | 0 | 2 | 0 |
| 2v7 | 1 | 0 | 1 | 0 |
| 3v6 | 2 | 0 | 2 | 0 |
| 3v10 | 1 | 0 | 1 | 0 |
| 4v9 | 1 | 0 | 1 | 0 |
| 5v12 | 4 | 0 | 4 | 0 |
| 6v11 | 4 | 0 | 4 | 0 |
| 7v10 | 6 | 0 | 6 | 0 |

## Feature Importance

### v5 features (importance = avg across CV folds)
For blended models: LGB uses split-based importance weighted by blend weight, LR uses |coefficient| weighted by blend weight.

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Elo_diff | 0.9068 |
| 2 | Seed_diff | 0.8650 |
| 3 | TRank_adjde_diff | 0.4379 |
| 4 | TRank_adjoe_diff | 0.3813 |
| 5 | seed_b | 0.0290 |
| 6 | TRank_barthag_diff | 0.0028 |
| 7 | seed_a | 0.0000 |

### v7 features
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Seed_diff | 0.9504 |
| 2 | Elo_diff | 0.6798 |
| 3 | TRank_adjoe_diff | 0.4070 |
| 4 | TRank_adjde_diff | 0.3797 |
| 5 | PointDiff_diff | 0.0894 |
| 6 | seed_b | 0.0179 |
| 7 | seed_a | 0.0021 |
| 8 | TRank_tempo_diff | 0.0000 |
| 9 | TRank_barthag_diff | 0.0000 |

### Feature comparison

**Features only in v7**: PointDiff_diff, TRank_tempo_diff

**Importance shift for common features (v7 - v5)**:
| Feature | v5 Imp | v7 Imp | Delta | Direction |
|---------|--------|--------|-------|-----------|
| Elo_diff | 0.9068 | 0.6798 | -0.2270 | decreased |
| Seed_diff | 0.8650 | 0.9504 | +0.0854 | increased |
| TRank_adjde_diff | 0.4379 | 0.3797 | -0.0582 | decreased |
| TRank_adjoe_diff | 0.3813 | 0.4070 | +0.0257 | increased |
| seed_b | 0.0290 | 0.0179 | -0.0111 | decreased |
| TRank_barthag_diff | 0.0028 | 0.0000 | -0.0028 | decreased |
| seed_a | 0.0000 | 0.0021 | +0.0021 | increased |

