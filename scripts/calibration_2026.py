"""Generate calibration plots for v7, rf, and kaito models on 2026 tournament results."""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Load 2026 results ---
with open(ROOT / "data/live/results.json") as f:
    results = json.load(f)

# Build ground truth: ID -> outcome (1 if lower TeamId won, 0 otherwise)
actuals = {}
for game in results:
    w, l = game["winner_id"], game["loser_id"]
    low, high = min(w, l), max(w, l)
    game_id = f"2026_{low}_{high}"
    outcome = 1 if w == low else 0
    actuals[game_id] = outcome

print(f"Total 2026 games with results: {len(actuals)}")

# --- Load submissions ---
def load_submission(path):
    preds = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds[row["ID"]] = float(row["Pred"])
    return preds

submissions = {
    "v7 (LGB40-LR60)": ROOT / "outputs/submissions/submission_stage2_v7_lgb40-lr60-blend_qtl1w_20260318_202646.csv",
    "RF Baseline": ROOT / "outputs/submissions/baseline_rf_2026.csv",
    "Kaito510": ROOT / "outputs/submissions/kaito510_goto_winning.csv",
}

model_preds = {name: load_submission(path) for name, path in submissions.items()}

# --- Match predictions to results ---
def match_preds(preds_dict, actuals_dict):
    y_true, y_pred, ids = [], [], []
    for game_id, outcome in actuals_dict.items():
        if game_id in preds_dict:
            y_true.append(outcome)
            y_pred.append(preds_dict[game_id])
            ids.append(game_id)
    return np.array(y_true), np.array(y_pred), ids

# --- Calibration computation ---
def calibration_bins(y_true, y_pred, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
        else:
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        count = mask.sum()
        if count == 0:
            bins.append({"bin_center": (bin_edges[i] + bin_edges[i+1]) / 2,
                         "mean_pred": np.nan, "actual_rate": np.nan, "count": 0})
        else:
            bins.append({"bin_center": (bin_edges[i] + bin_edges[i+1]) / 2,
                         "mean_pred": y_pred[mask].mean(),
                         "actual_rate": y_true[mask].mean(),
                         "count": int(count)})
    return bins

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = ["#2196F3", "#4CAF50", "#FF9800"]

for idx, (name, preds_dict) in enumerate(model_preds.items()):
    y_true, y_pred, ids = match_preds(preds_dict, actuals)
    brier = np.mean((y_true - y_pred) ** 2)
    bins = calibration_bins(y_true, y_pred)

    ax = axes[idx]
    # Perfect calibration
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")

    # Bars and scatter
    centers = [b["bin_center"] for b in bins if b["count"] > 0]
    actual_rates = [b["actual_rate"] for b in bins if b["count"] > 0]
    mean_preds = [b["mean_pred"] for b in bins if b["count"] > 0]
    counts = [b["count"] for b in bins if b["count"] > 0]

    ax.bar(centers, actual_rates, width=0.08, alpha=0.3, color=colors[idx], label="Actual rate")
    ax.scatter(mean_preds, actual_rates, s=[max(c * 5, 20) for c in counts],
               c=colors[idx], edgecolors="white", zorder=5, label="Bins")

    # Annotate counts
    for mp, ar, c in zip(mean_preds, actual_rates, counts):
        ax.annotate(f"n={c}", (mp, ar), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=7, alpha=0.7)

    ax.set_xlabel("Predicted Probability (lower ID wins)")
    ax.set_ylabel("Actual Win Rate")
    ax.set_title(f"{name}\nBrier={brier:.4f} | {len(y_true)} games matched")
    ax.legend(fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    # Print summary
    missing = len(actuals) - len(ids)
    print(f"\n{name}: Brier={brier:.4f}, {len(ids)} matched, {missing} missing")
    # Compute ECE
    weighted_err = sum(abs(b["mean_pred"] - b["actual_rate"]) * b["count"]
                       for b in bins if b["count"] > 0)
    ece = weighted_err / len(y_true)
    print(f"  ECE={ece:.4f}")

fig.suptitle("2026 Tournament Calibration: Predicted vs Actual Win Rates", fontsize=14, y=1.02)
fig.tight_layout()
save_path = ROOT / "outputs/reports/calibration_2026_models.png"
save_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved to {save_path}")
