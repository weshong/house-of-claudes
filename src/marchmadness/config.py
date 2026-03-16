"""Project configuration: paths, constants, hyperparameters."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
SUBMISSIONS_DIR = OUTPUT_DIR / "submissions"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Ensure output dirs exist
for d in [MODELS_DIR, SUBMISSIONS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Competition constants
CURRENT_SEASON = 2026
MEN_TEAM_ID_RANGE = (1000, 1999)
WOMEN_TEAM_ID_RANGE = (3000, 3999)
PREDICTION_CLIP = (0.025, 0.975)

# CV seasons (leave-season-out)
CV_SEASONS = [2022, 2023, 2024, 2025]
# Training seasons for features requiring detailed results
DETAILED_RESULTS_START = 2003

# Massey ordinals: only use late-season rankings
ORDINALS_MIN_DAY = 128
# Top ranking systems to use as features
TOP_RANKING_SYSTEMS = ["POM", "SAG", "WLK", "WOL", "MOR", "DOL", "COL", "RPI", "AP", "USA"]

# Elo parameters
ELO_K = 20
ELO_HOME_ADV = 100
ELO_INITIAL = 1500
ELO_SEASON_REGRESSION = 0.75  # new = 0.75 * end + 0.25 * 1500

# Model hyperparameters
LOGISTIC_PARAMS = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}

XGBOOST_PARAMS = {
    "tree_method": "hist",
    "device": "cuda",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

LIGHTGBM_PARAMS = {
    "device": "gpu",
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

# Probability shrinkage: pred = pred * (1 - alpha) + 0.5 * alpha
# Shrinks predictions toward 0.5 to improve calibration
SHRINKAGE_ALPHA = 0.10

# Best scores tracking
BEST_SCORES_FILE = OUTPUT_DIR / "best_scores.json"
REGRESSION_THRESHOLD = 0.005
