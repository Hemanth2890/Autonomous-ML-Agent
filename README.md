# Autonomous Multi-Agent AutoML Platform

An agent-driven AutoML system for regression tasks. A set of model agents compete against each other using cross-validated scoring, a judge agent selects the strongest candidate, and an Optuna-based tuning stage refines its hyperparameters. The whole pipeline is orchestrated with LangGraph and served through a FastAPI backend, with Celery and Redis handling distributed training runs and PostgreSQL tracking experiment history.

## Overview

Predicting a continuous target well usually means trying more than one model family and comparing them fairly rather than betting on a single algorithm up front. This project automates that process: it trains five different regression models in parallel under identical cross-validation conditions, scores them on both accuracy and stability, and only then spends compute on hyperparameter search for whichever model actually performed best.

## Architecture

```
Client --> FastAPI (/train) --> Celery task --> LangGraph pipeline
                                                     |
                PostgreSQL (or SQLite) <-- ExperimentRun
                                                     |
                                          preprocess -> compete (5 model
                                          agents, 5-fold CV) -> judge
                                          (weighted scoring) -> tune
                                          (Optuna TPE) -> evaluate -> report
```

### Agents

- Five competing model agents (Random Forest, Gradient Boosting, ElasticNet, XGBoost, LightGBM), each trained independently with 5-fold cross-validation.
- A judge agent that selects the best-performing model. Model selection is delegated to a locally running LLM via Ollama, which is given the full cross-validated leaderboard (RMSE mean, RMSE standard deviation, fit time per model) and asked to choose a winner and justify it. If Ollama is not available, the judge falls back to a deterministic weighted-scoring rule (RMSE mean plus a stability penalty), so the pipeline always completes.
- An Optuna tuning stage (TPE sampler, configurable trial count) that re-optimizes the judge's selected model family.

### Infrastructure

- `api/main.py` - FastAPI service exposing `POST /train`, `GET /status/{id}`, `GET /results/{id}`, and `GET /experiments`.
- `tasks/celery_app.py` - Celery task wrapping the LangGraph pipeline, with broker and backend on Redis. Set `CELERY_TASK_ALWAYS_EAGER=true` to run synchronously without a Redis broker for local development.
- `db/models.py` - SQLAlchemy `ExperimentRun` table. Set `DATABASE_URL` to a Postgres DSN in production; defaults to a local SQLite file otherwise.

## Datasets

| Name | Type | Samples | Description |
|---|---|---|---|
| `diabetes` | Real (medical) | 442 | scikit-learn built-in diabetes progression dataset |
| `abalone` | Real (biological) | 4,177 | UCI/Kaggle abalone age dataset; predicts shell rings from physical measurements |
| `friedman1` | Synthetic benchmark | 2,000 | Friedman #1 nonlinear regression benchmark (Friedman, 1991); generated via `sklearn.datasets.make_friedman1` |
| `california_housing` | Real (real estate) | 20,640 | scikit-learn California housing dataset; requires an external download not available in all network environments |

## Running It

### Local run (no Docker, no Redis, no Postgres)

```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/run_local.py --dataset abalone --trials 30
```

### Full stack (API, Celery worker, Redis, Postgres)

```bash
docker compose up --build
curl -X POST localhost:8000/train -H "Content-Type: application/json" \
  -d '{"dataset_name": "abalone", "optuna_trials": 30}'
curl localhost:8000/status/<task_id>
curl localhost:8000/results/<task_id>
```

To verify the underlying infrastructure directly:

```bash
docker compose exec postgres psql -U automl -d automl -c "SELECT task_id, status, tuned_model, tuned_rmse FROM experiment_runs;"
docker compose exec redis redis-cli ping
```

## Results

### Abalone (real-world biological data, 30 Optuna trials)

| Stage | Model | RMSE |
|---|---|---|
| Best of 5 competing agents (5-fold CV) | Gradient Boosting | 2.14 (+/- 0.11) |
| Held-out test set, untuned | Gradient Boosting | 2.26 |
| Held-out test set, Optuna-tuned | Gradient Boosting | 2.25 |

Full cross-validation leaderboard:

| Model | CV RMSE mean | CV RMSE std |
|---|---|---|
| Gradient Boosting | 2.14 | 0.11 |
| Random Forest | 2.16 | 0.09 |
| XGBoost | 2.19 | 0.09 |
| LightGBM | 2.20 | 0.09 |
| ElasticNet | 2.54 | 0.07 |

### Diabetes (real-world medical data, 25 Optuna trials)

| Stage | Model | RMSE |
|---|---|---|
| Best of 5 competing agents (5-fold CV) | ElasticNet | 55.78 (+/- 2.68) |
| Held-out test set, untuned | ElasticNet | 53.38 |
| Held-out test set, Optuna-tuned | ElasticNet | 53.69 |

Full cross-validation leaderboard:

| Model | CV RMSE mean | CV RMSE std |
|---|---|---|
| ElasticNet | 55.78 | 2.68 |
| Random Forest | 58.91 | 1.93 |
| LightGBM | 60.63 | 1.95 |
| Gradient Boosting | 61.09 | 3.04 |
| XGBoost | 63.76 | 3.12 |

### Friedman1 (synthetic nonlinear benchmark, 30 Optuna trials)

| Stage | Model | RMSE |
|---|---|---|
| Best of 5 competing agents (5-fold CV) | LightGBM | 1.42 (+/- 0.04) |
| Held-out test set, untuned | LightGBM | 1.37 |
| Held-out test set, Optuna-tuned | LightGBM | 1.33 |

### Notes on the results

On near-linear data such as the diabetes dataset, a linear model already generalizes well, so tuning provides little additional benefit over the untuned baseline. On the abalone dataset, which has mild real-world nonlinearity, tuning gives a small but measurable improvement once the right model family has been chosen. On the synthetic Friedman1 benchmark, which is built specifically to contain nonlinear feature interactions, tuning produces a clearer gain of about 2.85 percent.

Across all three datasets, the model selection step accounts for most of the performance difference: ElasticNet only wins on the near-linear dataset, while tree-based models outperform it by a wide margin everywhere else. Tuning helps further, but how much depends heavily on how nonlinear the underlying data is.

## Project Structure

```
agents/
  model_agents.py   competing model agents (Random Forest, Gradient Boosting, ElasticNet, XGBoost, LightGBM)
  judge.py          judge agent: weighted scoring and model selection
  graph.py          LangGraph StateGraph orchestrating the full pipeline
api/
  main.py           FastAPI service
tasks/
  celery_app.py     Celery task and dataset loaders
tools/
  optuna_tuner.py   Optuna search spaces and tuning logic
db/
  models.py         SQLAlchemy models and database session management
scripts/
  run_local.py      local end-to-end pipeline runner, no external infrastructure required
data/
  abalone.csv       abalone dataset
Dockerfile
docker-compose.yml
requirements.txt
```

## License

MIT
