"""
Competing model agents for the regression AutoML pipeline.

Each agent wraps a single regression model family, trains it with
Stratified/KFold cross-validation, and reports back a structured
proposal (cv RMSE mean/std, fit time, params) so the JudgeAgent can
compare them on equal footing.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

MODEL_REGISTRY = {
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "gradient_boost": lambda: GradientBoostingRegressor(random_state=42),
    "elastic_net": lambda: ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42),
    "xgboost": lambda: xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        random_state=42, n_jobs=-1, verbosity=0,
    ),
    "lightgbm": lambda: lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.05, random_state=42, verbosity=-1
    ),
}


@dataclass
class ModelProposal:
    model_type: str
    cv_rmse_mean: float
    cv_rmse_std: float
    fit_time_seconds: float
    params: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "cv_rmse_mean": self.cv_rmse_mean,
            "cv_rmse_std": self.cv_rmse_std,
            "fit_time_seconds": self.fit_time_seconds,
            "params": self.params,
            "error": self.error,
        }


class BaseModelAgent:
    """An agent responsible for one model family in the competition."""

    def __init__(self, model_type: str, cv_folds: int = 5):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_type '{model_type}'. "
                              f"Options: {list(MODEL_REGISTRY)}")
        self.model_type = model_type
        self.cv_folds = cv_folds

    def propose_and_train(self, X: np.ndarray, y: np.ndarray) -> ModelProposal:
        start = time.time()
        try:
            model = MODEL_REGISTRY[self.model_type]()
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            neg_mse_scores = cross_val_score(
                model, X, y, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
            )
            rmse_scores = np.sqrt(-neg_mse_scores)
            elapsed = time.time() - start
            return ModelProposal(
                model_type=self.model_type,
                cv_rmse_mean=float(rmse_scores.mean()),
                cv_rmse_std=float(rmse_scores.std()),
                fit_time_seconds=round(elapsed, 4),
                params=model.get_params(),
            )
        except Exception as exc:  # noqa: BLE001
            return ModelProposal(
                model_type=self.model_type,
                cv_rmse_mean=float("inf"),
                cv_rmse_std=0.0,
                fit_time_seconds=round(time.time() - start, 4),
                error=str(exc),
            )

    def fit_full(self, X: np.ndarray, y: np.ndarray, params: dict | None = None):
        """Fit the model on the full training data, optionally with overridden params."""
        model = MODEL_REGISTRY[self.model_type]()
        if params:
            model.set_params(**{k: v for k, v in params.items() if k in model.get_params()})
        model.fit(X, y)
        return model
