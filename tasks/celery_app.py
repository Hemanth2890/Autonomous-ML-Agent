"""
Celery app for distributed/asynchronous training runs.

Broker/backend default to a local Redis instance (redis://localhost:6379)
but are configurable via env vars for real deployments. CELERY_TASK_ALWAYS_EAGER
can be set to run tasks synchronously in-process (useful for local dev/tests
when no Redis broker is running).
"""
from __future__ import annotations

import os
import datetime as dt

from celery import Celery
from sklearn.datasets import fetch_california_housing, load_diabetes
import pandas as pd

from agents.graph import run_pipeline
from db.models import SessionLocal, ExperimentRun, init_db

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
ALWAYS_EAGER = os.environ.get("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true"

celery_app = Celery("automl_tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.task_always_eager = ALWAYS_EAGER
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]

_ABALONE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "abalone.csv")
_ABALONE_COLUMNS = ["sex", "length", "diameter", "height", "whole_weight",
                     "shucked_weight", "viscera_weight", "shell_weight", "rings"]


def _load_abalone():
    df = pd.read_csv(_ABALONE_PATH, header=None, names=_ABALONE_COLUMNS)
    df = pd.get_dummies(df, columns=["sex"], drop_first=True)
    y = df.pop("rings").to_numpy(dtype=float)
    X = df.to_numpy(dtype=float)

    class _Bunch:
        pass

    bunch = _Bunch()
    bunch.data = X
    bunch.target = y
    return bunch


DATASETS = {
    "california_housing": fetch_california_housing,
    "diabetes": load_diabetes,
    "abalone": _load_abalone,
}


@celery_app.task(bind=True, name="tasks.run_automl")
def run_automl_task(self, dataset_name: str = "diabetes", optuna_trials: int = 30):
    init_db()
    db = SessionLocal()
    task_id = self.request.id or "eager-run"

    run = ExperimentRun(task_id=task_id, dataset_name=dataset_name, status="running")
    db.add(run)
    db.commit()

    try:
        loader = DATASETS[dataset_name]
        data = loader()
        report = run_pipeline(data.data, data.target, optuna_trials=optuna_trials)

        run.status = "completed"
        run.baseline_model = report["selected_model"]
        run.baseline_rmse = report["baseline_test_rmse"]
        run.tuned_model = report["selected_model"]
        run.tuned_rmse = report["tuned_test_rmse"]
        run.n_optuna_trials = report["optuna_trials"]
        run.best_params = report["best_params"]
        run.ranking = report["ranking"]
        run.reasoning = report["reasoning"]
        run.completed_at = dt.datetime.utcnow()
        db.commit()
        return report
    except Exception as exc:  # noqa: BLE001
        run.status = "failed"
        run.error = str(exc)
        db.commit()
        raise
    finally:
        db.close()
