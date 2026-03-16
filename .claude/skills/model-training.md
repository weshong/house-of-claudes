---
name: model-training
description: Train and tune prediction models with GPU support
user_invocable: true
---

# Model Training Skill

Train, tune, and compare models for tournament prediction.

## Models

### Logistic Regression (baseline)
```python
LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
```
- Standardize features first (StandardScaler)
- Fast to train, good regularizing effect in ensemble

### XGBoost (GPU)
```python
XGBClassifier(
    tree_method='gpu_hist', device='cuda',
    objective='binary:logistic', eval_metric='logloss',
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0
)
```

### LightGBM (GPU)
```python
LGBMClassifier(
    device='gpu', objective='binary', metric='binary_logloss',
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8
)
```

### Ensemble
- Weighted average of all models
- Optimize weights via `scipy.optimize.minimize` on CV Brier score
- Typical: ~0.3 logistic + 0.35 XGBoost + 0.35 LightGBM
- Always clip final predictions to [0.025, 0.975]

## Workflow

1. Run leave-season-out CV for each model individually
2. Compare Brier scores per fold and overall
3. Optimize ensemble weights on out-of-fold predictions
4. Retrain each model on all available data for final submission
5. Log all results to experiment log

## Hyperparameter Tuning (if time permits)

Use Optuna with leave-season-out CV as objective:
```python
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

## Commands

```bash
source /c/Users/wesho/miniforge3/etc/profile.d/conda.sh && conda activate marchmadness
cd /c/Users/wesho/Projects/house-of-claudes
python scripts/run_pipeline.py --mode train --cv
python scripts/run_pipeline.py --mode train --final
```
