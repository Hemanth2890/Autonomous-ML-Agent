"""
Hyperparameter optimization via Optuna, replacing the original
GridSearchCV approach. Defines search spaces per model family and
runs a TPE study minimizing cross-validated RMSE.
"""
from __future__ import annotations

import numpy as np
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import lightgbm as lgb
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _build_model(model_type: str, trial: optuna.Trial):
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=42, n_jobs=-1,
        )
    if model_type == "gradient_boost":
        return GradientBoostingRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            random_state=42,
        )
    if model_type == "elastic_net":
        return ElasticNet(
            alpha=trial.suggest_float("alpha", 1e-3, 10.0, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
            random_state=42,
        )
    if model_type == "xgboost":
        return xgb.XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=42, n_jobs=-1, verbosity=0,
        )
    if model_type == "lightgbm":
        return lgb.LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            random_state=42, verbosity=-1,
        )
    raise ValueError(f"No Optuna search space defined for '{model_type}'")


def tune_model(X: np.ndarray, y: np.ndarray, model_type: str,
                n_trials: int = 30, cv_folds: int = 5, seed: int = 42):
    """Run an Optuna TPE study to minimize CV RMSE for the given model family.

    Returns (best_model_fitted_on_full_data, best_params, best_cv_rmse, study).
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        model = _build_model(model_type, trial)
        scores = cross_val_score(
            model, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
        )
        return float(np.sqrt(-scores).mean())

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_trial_model = _build_model(model_type, optuna.trial.FixedTrial(best_params))
    best_trial_model.fit(X, y)

    return best_trial_model, best_params, study.best_value, study
