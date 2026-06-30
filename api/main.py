"""
FastAPI service exposing the AutoML pipeline.

POST /train      -> kicks off an async Celery task, returns task_id
GET  /status/{id} -> Celery task state
GET  /results/{id} -> stored experiment row from the DB (once completed)
GET  /experiments  -> list all past experiment runs
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from celery.result import AsyncResult

from tasks.celery_app import celery_app, run_automl_task, DATASETS
from db.models import init_db, SessionLocal, ExperimentRun

app = FastAPI(
    title="Autonomous Multi-Agent AutoML API",
    description="LangGraph-orchestrated multi-agent AutoML with Optuna tuning, "
                 "Celery/Redis distributed execution, and Postgres experiment tracking.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    init_db()


class TrainRequest(BaseModel):
    dataset_name: str = "diabetes"
    optuna_trials: int = 30


@app.post("/train")
def train(req: TrainRequest):
    if req.dataset_name not in DATASETS:
        raise HTTPException(400, f"Unknown dataset '{req.dataset_name}'. "
                                  f"Options: {list(DATASETS)}")
    async_result = run_automl_task.delay(req.dataset_name, req.optuna_trials)
    return {"task_id": async_result.id, "status": "submitted"}


@app.get("/status/{task_id}")
def status(task_id: str):
    result = AsyncResult(task_id, app=celery_app)
    return {"task_id": task_id, "state": result.state}


@app.get("/results/{task_id}")
def results(task_id: str):
    db = SessionLocal()
    try:
        run = db.query(ExperimentRun).filter(ExperimentRun.task_id == task_id).first()
        if not run:
            raise HTTPException(404, "No experiment found for this task_id")
        return {
            "task_id": run.task_id,
            "dataset": run.dataset_name,
            "status": run.status,
            "selected_model": run.tuned_model,
            "baseline_rmse": run.baseline_rmse,
            "tuned_rmse": run.tuned_rmse,
            "optuna_trials": run.n_optuna_trials,
            "best_params": run.best_params,
            "ranking": run.ranking,
            "reasoning": run.reasoning,
            "error": run.error,
        }
    finally:
        db.close()


@app.get("/experiments")
def list_experiments():
    db = SessionLocal()
    try:
        runs = db.query(ExperimentRun).order_by(ExperimentRun.created_at.desc()).all()
        return [
            {"task_id": r.task_id, "dataset": r.dataset_name, "status": r.status,
             "tuned_rmse": r.tuned_rmse, "selected_model": r.tuned_model,
             "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in runs
        ]
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}
