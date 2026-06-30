"""
Local end-to-end run of the LangGraph pipeline, no Celery/Redis/Postgres
infra required. Use this to validate the pipeline and capture real
baseline-vs-tuned RMSE numbers (e.g. for documentation or a resume bullet).

Usage:
    python scripts/run_local.py --dataset diabetes --trials 30
"""
from __future__ import annotations

import argparse
import json
import time

from sklearn.datasets import fetch_california_housing, load_diabetes

from agents.graph import run_pipeline
from tasks.celery_app import _load_abalone

DATASETS = {
    "california_housing": fetch_california_housing,
    "diabetes": load_diabetes,
    "abalone": _load_abalone,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), default="diabetes")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    data = DATASETS[args.dataset]()
    print(f"Dataset: {args.dataset}  |  X shape: {data.data.shape}")

    start = time.time()
    report = run_pipeline(data.data, data.target, optuna_trials=args.trials)
    elapsed = time.time() - start

    report["wall_clock_seconds"] = round(elapsed, 2)
    print("\n=== Pipeline Report ===")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
