"""
Experiment-tracking database layer.

Uses Postgres in production (set DATABASE_URL, e.g.
postgresql+psycopg2://user:pass@host:5432/automl) and falls back to a
local SQLite file when DATABASE_URL is not set, so the project runs
out-of-the-box without external infra while staying production-ready.
"""
from __future__ import annotations

import os
import datetime as dt

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./automl_experiments.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    dataset_name = Column(String)
    status = Column(String, default="pending")  # pending|running|completed|failed
    baseline_model = Column(String, nullable=True)
    baseline_rmse = Column(Float, nullable=True)
    tuned_model = Column(String, nullable=True)
    tuned_rmse = Column(Float, nullable=True)
    n_optuna_trials = Column(Integer, nullable=True)
    best_params = Column(JSON, nullable=True)
    ranking = Column(JSON, nullable=True)
    reasoning = Column(String, nullable=True)
    error = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
