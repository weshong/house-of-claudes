"""Generate v7 submission — current best configs.

Men's:  40% LGB d=3 + 60% L2 LR C=100, PCA ordinals + Torvik + TRank + gaps — Brier 0.1267
Women's: QT + L1 LR C=0.20, seeds + Elo + TRank + PointDiff — Brier 0.1335
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marchmadness.config import SUBMISSIONS_DIR, PREDICTION_CLIP
from marchmadness.data_loader import load_all
from marchmadness.features.builder import build_matchup_features, get_feature_columns
from marchmadness.features.seeds import parse_seed
from marchmadness.submission import validate_submission

# Reuse the custom feature builder from research.py (v7 config)
sys.path.insert(0, str(Path(__file__).parent.parent))
from research import build_custom_team_features, build_custom_training_data


def train_custom(data, gender, model_obj, start_year):
    """Train a model on custom features. Returns (model, feature_cols, medians)."""
    seasons = list(range(start_year, 2026))
    df = build_custom_training_data(data, seasons, gender)
    fc = get_feature_columns(df)
    X = df[fc].values.copy()
    y = df["Label"].values.copy()
    med = np.nanmedian(X, axis=0)
    for i in range(X.shape[1]):
        mask = np.isnan(X[:, i])
        if mask.any():
            X[mask, i] = med[i] if not np.isnan(med[i]) else 0
    model_obj.fit(X, y)
    return model_obj, fc, med


def predict_one(tf, team_a, team_b, fc, model, med):
    """Predict P(team_a wins) for a single matchup."""
    matchup = build_matchup_features(tf, team_a, team_b)
    if not matchup:
        return 0.5
    fv = np.array([[matchup.get(c, np.nan) for c in fc]])
    for i in range(fv.shape[1]):
        if np.isnan(fv[0, i]):
            fv[0, i] = med[i] if not np.isnan(med[i]) else 0
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(fv)[:, 1][0])
    return float(model.predict(fv)[0])


def generate_submission(data, stage):
    sample_sub = data[f"SampleSubmissionStage{stage}"]
    parts = sample_sub["ID"].str.split("_", expand=True)
    parsed = pd.DataFrame({
        "ID": sample_sub["ID"],
        "Season": parts[0].astype(int),
        "TeamA": parts[1].astype(int),
        "TeamB": parts[2].astype(int),
    })
    print(f"  Stage {stage}: {len(parsed)} matchups")

    # --- Men's: 40% LGB + 60% LR blend on custom features (v7) ---
    print("  Training Men's LGB d=3 (n=250)...")
    m_lgb, m_fc, m_med = train_custom(
        data, "M",
        LGBMClassifier(n_estimators=250, max_depth=3, learning_rate=0.025,
                       subsample=0.8, colsample_bytree=0.6, min_split_gain=0.005,
                       random_state=42, verbose=-1),
        start_year=2015
    )
    print(f"    Features ({len(m_fc)}): {m_fc}")

    print("  Training Men's LR L2 C=100...")
    m_lr, m_lr_fc, m_lr_med = train_custom(
        data, "M",
        Pipeline([("s", StandardScaler()),
                  ("lr", LogisticRegression(C=100.0, penalty='l2', solver='lbfgs', max_iter=2000))]),
        start_year=2015
    )

    # --- Women's: QT + L1 LR C=0.20 on custom features (v7) ---
    print("  Training Women's QT + L1 LR C=0.20...")
    w_lr, w_fc, w_med = train_custom(
        data, "W",
        Pipeline([("q", QuantileTransformer(output_distribution='normal', n_quantiles=50)),
                  ("lr", LogisticRegression(C=0.20, penalty='l1', solver='liblinear', max_iter=2000))]),
        start_year=2003
    )
    print(f"    Features ({len(w_fc)}): {w_fc}")

    # Build team features for 2026
    print("  Building 2026 team features...")
    tf_cache = {}
    predictions = {}
    missing_teams = {"M": set(), "W": set()}
    total = len(parsed)

    for i, (_, row) in enumerate(parsed.iterrows()):
        if (i + 1) % 10000 == 0:
            print(f"    {i+1}/{total}")

        ta, tb = row["TeamA"], row["TeamB"]
        season = row["Season"]
        gender = "M" if ta < 3000 else "W"

        ck = (season, gender)
        if ck not in tf_cache:
            tf_cache[ck] = build_custom_team_features(data, season, gender)

        tf = tf_cache[ck]

        # Track missing teams
        if tf.empty or ta not in tf["TeamID"].values:
            missing_teams[gender].add(ta)
        if tf.empty or tb not in tf["TeamID"].values:
            missing_teams[gender].add(tb)

        if gender == "M":
            p_lgb = predict_one(tf, ta, tb, m_fc, m_lgb, m_med)
            p_lr = predict_one(tf, ta, tb, m_lr_fc, m_lr, m_lr_med)
            pred = 0.40 * p_lgb + 0.60 * p_lr  # v7 blend weights
        else:
            pred = predict_one(tf, ta, tb, w_fc, w_lr, w_med)

        predictions[row["ID"]] = float(np.clip(pred, *PREDICTION_CLIP))

    submission = pd.DataFrame({
        "ID": sample_sub["ID"],
        "Pred": sample_sub["ID"].map(predictions).fillna(0.5),
    })

    # ============================================================
    # VALIDATION
    # ============================================================
    print(f"\n  === VALIDATION ===")

    # Basic format validation
    errors = validate_submission(submission, sample_sub)
    if errors:
        print(f"  FORMAT ERRORS: {errors}")
        return None
    else:
        print(f"  Format: OK")

    # Check for missing teams
    if missing_teams["M"]:
        print(f"  WARNING: {len(missing_teams['M'])} men's teams missing features: {sorted(missing_teams['M'])[:10]}...")
    else:
        print(f"  Men's team coverage: 100%")
    if missing_teams["W"]:
        print(f"  WARNING: {len(missing_teams['W'])} women's teams missing features: {sorted(missing_teams['W'])[:10]}...")
    else:
        print(f"  Women's team coverage: 100%")

    # Check predictions are not all 0.5 (would indicate feature failure)
    n_default = (submission["Pred"] == 0.5).sum()
    print(f"  Default predictions (0.5): {n_default} / {len(submission)} ({n_default/len(submission):.1%})")

    # Split by gender
    m_mask = submission["ID"].str.split("_").str[1].astype(int) < 3000
    m_sub = submission[m_mask]
    w_sub = submission[~m_mask]
    print(f"  Men's matchups: {len(m_sub)}, Mean={m_sub['Pred'].mean():.4f}, Std={m_sub['Pred'].std():.4f}")
    print(f"  Women's matchups: {len(w_sub)}, Mean={w_sub['Pred'].mean():.4f}, Std={w_sub['Pred'].std():.4f}")

    # Check prediction distribution
    print(f"  Prediction range: [{submission['Pred'].min():.4f}, {submission['Pred'].max():.4f}]")
    print(f"  Clip bounds: {PREDICTION_CLIP}")

    # Check no NaN
    n_nan = submission["Pred"].isna().sum()
    if n_nan > 0:
        print(f"  ERROR: {n_nan} NaN predictions!")
        return None
    else:
        print(f"  NaN check: OK")

    # Verify expected team counts from sample submission
    all_teams_m = set()
    all_teams_w = set()
    for _, row in parsed.iterrows():
        if row["TeamA"] < 3000:
            all_teams_m.add(row["TeamA"])
            all_teams_m.add(row["TeamB"])
        else:
            all_teams_w.add(row["TeamA"])
            all_teams_w.add(row["TeamB"])
    print(f"  Unique teams in submission: {len(all_teams_m)} men's, {len(all_teams_w)} women's")

    # Check seed coverage for 2026
    seeds_m = data[f"MNCAATourneySeeds"]
    seeds_w = data[f"WNCAATourneySeeds"]
    seeded_m_2026 = set(seeds_m[seeds_m["Season"] == 2026]["TeamID"]) if 2026 in seeds_m["Season"].values else set()
    seeded_w_2026 = set(seeds_w[seeds_w["Season"] == 2026]["TeamID"]) if 2026 in seeds_w["Season"].values else set()
    if seeded_m_2026:
        missing_m = seeded_m_2026 - all_teams_m
        print(f"  2026 seeded men's teams in submission: {len(seeded_m_2026 - missing_m)}/{len(seeded_m_2026)}")
        if missing_m:
            print(f"    MISSING: {missing_m}")
    if seeded_w_2026:
        missing_w = seeded_w_2026 - all_teams_w
        print(f"  2026 seeded women's teams in submission: {len(seeded_w_2026 - missing_w)}/{len(seeded_w_2026)}")
        if missing_w:
            print(f"    MISSING: {missing_w}")

    # Spot-check some known matchups
    print(f"\n  === SPOT CHECK (sample predictions) ===")
    # Find some 1v16 and 5v12 type matchups if seeds are available
    if seeded_m_2026:
        seed_map = {}
        for _, row in seeds_m[seeds_m["Season"] == 2026].iterrows():
            seed_map[row["TeamID"]] = parse_seed(row["Seed"])
        teams_m = data["MTeams"]
        name_map = dict(zip(teams_m["TeamID"], teams_m["TeamName"]))

        # Show a few predictions for seeded matchups
        shown = 0
        for _, row in parsed[parsed["Season"] == 2026].iterrows():
            ta, tb = row["TeamA"], row["TeamB"]
            if ta in seed_map and tb in seed_map and ta < 3000:
                sa, sb = seed_map[ta], seed_map[tb]
                diff = abs(sa - sb)
                if diff >= 10 and shown < 3:  # Big seed diff
                    pred = predictions.get(row["ID"], 0.5)
                    na, nb = name_map.get(ta, ta), name_map.get(tb, tb)
                    print(f"    {na} ({sa}) vs {nb} ({sb}): {pred:.4f}")
                    shown += 1
                elif diff == 7 and shown < 6:  # 5v12 type
                    pred = predictions.get(row["ID"], 0.5)
                    na, nb = name_map.get(ta, ta), name_map.get(tb, tb)
                    print(f"    {na} ({sa}) vs {nb} ({sb}): {pred:.4f}")
                    shown += 1

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SUBMISSIONS_DIR / f"submission_stage2_v7_lgb40-lr60-blend_qtl1w_{ts}.csv"
    submission.to_csv(path, index=False)
    print(f"\n  Saved: {path}")

    return path


if __name__ == "__main__":
    print("Loading data...")
    data = load_all()
    print("\n=== Stage 2 (v7) ===")
    s2 = generate_submission(data, 2)
    if s2:
        print(f"\nDone! Submission at:\n  {s2}")
    else:
        print("\nFailed! Check errors above.")
