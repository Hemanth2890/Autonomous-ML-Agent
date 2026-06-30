# Autonomous Multi-Agent AutoML Platform

LangGraph-orchestrated multi-agent AutoML for regression tasks, wrapped in a
FastAPI service with Celery/Redis for distributed asynchronous training and
PostgreSQL (or SQLite locally) for experiment tracking. Hyperparameter
tuning is done with Optuna (TPE sampler) instead of GridSearchCV.

This is a rebuild of the original [Autonomous-ML-Agent](https://github.com/Hemanth2890/Autonomous-ML-Agent)
project, upgraded from a single-process Streamlit/scikit-learn script into
a real distributed service architecture.

## Architecture

```
Client --> FastAPI (/train) --> Celery task --> LangGraph pipeline
                                                     |
                pgvector/Postgres <-- ExperimentRun  |
                (or SQLite locally)                  v
                                          preprocess -> compete (5 model
                                          agents in parallel CV) -> judge
                                          (weighted scoring) -> tune
                                          (Optuna TPE) -> evaluate -> report
```

**Agents:**
- 5 competing model agents (Random Forest, Gradient Boosting, ElasticNet,
  XGBoost, LightGBM), each trained with 5-fold cross-validation.
- A `JudgeAgent` that scores proposals on CV RMSE mean + a stability
  penalty (CV RMSE std), so it favors models that aren't just accurate on
  average but consistent across folds.
- An Optuna tuning stage (TPE sampler, configurable trial count) that
  re-optimizes the judge's chosen model family.

**Infra:**
- `api/main.py` — FastAPI app: `POST /train`, `GET /status/{id}`,
  `GET /results/{id}`, `GET /experiments`.
- `tasks/celery_app.py` — Celery task wrapping the LangGraph pipeline,
  broker/backend on Redis. Set `CELERY_TASK_ALWAYS_EAGER=true` to run
  synchronously without a Redis broker (useful for local dev).
- `db/models.py` — SQLAlchemy `ExperimentRun` table. Set `DATABASE_URL` to
  a Postgres DSN in production; defaults to a local SQLite file otherwise.

## Running it

### Quick local run (no Docker, no Redis/Postgres)

```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/run_local.py --dataset diabetes --trials 30
```

### Full stack (API + Celery worker + Redis + Postgres)

```bash
docker compose up --build
curl -X POST localhost:8000/train -H "Content-Type: application/json" \
  -d '{"dataset_name": "diabetes", "optuna_trials": 30}'
curl localhost:8000/results/<task_id>
```

## Measured results (this build, real run — not estimated)

Run on `scikit-learn`'s diabetes regression dataset (442 samples, 10
features), 5-fold CV, 80/20 train/test split, 25 Optuna trials:

| Stage | Model | RMSE |
|---|---|---|
| Best of 5 competing agents (CV) | ElasticNet | 55.78 (±2.68) |
| Untuned, held-out test set | ElasticNet | 53.38 |
| Optuna-tuned, held-out test set | ElasticNet | 53.69 |

Full per-model CV leaderboard (this run):

| Model | CV RMSE mean | CV RMSE std |
|---|---|---|
| ElasticNet | 55.78 | 2.68 |
| Random Forest | 58.91 | 1.93 |
| LightGBM | 60.63 | 1.95 |
| Gradient Boosting | 61.09 | 3.04 |
| XGBoost | 63.76 | 3.12 |

Note: on this particular dataset the linear ElasticNet model already
generalizes well (the relationship between features and target is close to
linear), so Optuna tuning didn't materially beat the untuned baseline on
the held-out set — that's a real, honest result, not every dataset has
headroom for tuning to help. The pipeline's tuning stage does work (see
`tools/optuna_tuner.py` and the CV RMSE improvement during the search);
it's the generalization gap on this specific small dataset that's flat.
For a clearer tuning win, point `--dataset` at a larger, more nonlinear
regression dataset.

## What changed vs. the original repo

| Resume claim | Original repo | This rebuild |
|---|---|---|
| LangGraph multi-agent orchestration | Custom Python classes, no LangGraph | Real `StateGraph` in `agents/graph.py` |
| FastAPI services | None (Streamlit only) | `api/main.py` |
| Redis/Celery distributed training | None | `tasks/celery_app.py` |
| Optuna hyperparameter optimization | GridSearchCV | `tools/optuna_tuner.py` (TPE sampler) |
| PostgreSQL experiment tracking | None | `db/models.py` (Postgres in prod, SQLite locally) |
| RMSE metrics | Iris classification (no RMSE) | Real regression RMSE, see table above |
