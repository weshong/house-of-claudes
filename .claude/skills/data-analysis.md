---
name: data-analysis
description: Exploratory data analysis on competition datasets
user_invocable: true
---

# Data Analysis Skill

Run comprehensive EDA on the March Madness competition data.

## Steps

1. Load all CSVs from `data/raw/` using `src/marchmadness/data_loader.py`
2. For each table, print: shape, dtypes, null counts, value ranges, date/season coverage
3. For results tables (compact + detailed):
   - Games per season, average score, score distribution
   - Home vs away win rates
   - Overtime frequency
4. For MasseyOrdinals / WMasseyOrdinals:
   - List all distinct ranking systems (`SystemName`)
   - Count seasons covered by each system
   - Identify which systems have best coverage for recent seasons (2020+)
5. For Seeds tables:
   - Distribution of seed numbers
   - Historical upset rates by seed matchup (1v16, 2v15, etc.)
6. Cross-reference:
   - Teams in tournament seeds that are missing from results
   - Seasons with detailed results vs only compact
   - Women's data coverage gaps vs men's
7. Save output to `outputs/reports/data_exploration.md`

## Commands

```bash
source /c/Users/wesho/miniforge3/etc/profile.d/conda.sh && conda activate marchmadness
cd /c/Users/wesho/Projects/house-of-claudes
python -c "from marchmadness.data_loader import load_all; data = load_all(); [print(f'{k}: {v.shape}') for k,v in data.items()]"
```
