"""
LangGraph orchestration for the AutoML pipeline.

Graph flow:
  preprocess -> compete (fan-out model agents) -> judge ->
  tune (Optuna on the judge's chosen model) -> evaluate -> report

State is a typed dict threaded through every node so each agent only
reads/writes the keys it owns.
"""
from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from langgraph.graph import StateGraph, END

from agents.model_agents import BaseModelAgent
from agents.judge import JudgeAgent
from tools.optuna_tuner import tune_model

CANDIDATE_MODELS = ["random_forest", "gradient_boost", "elastic_net", "xgboost", "lightgbm"]


class PipelineState(TypedDict, total=False):
    X: np.ndarray
    y: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    proposals: list[dict[str, Any]]
    decision: dict[str, Any]
    optuna_trials: int
    tuned_params: dict[str, Any]
    tuned_cv_rmse: float
    tuned_test_rmse: float
    baseline_test_rmse: float
    report: dict[str, Any]
    _tuned_model: Any


def node_preprocess(state: PipelineState) -> PipelineState:
    X_train, X_test, y_train, y_test = train_test_split(
        state["X"], state["y"], test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return {**state, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test}


def node_compete(state: PipelineState) -> PipelineState:
    proposals = []
    for model_type in CANDIDATE_MODELS:
        agent = BaseModelAgent(model_type)
        proposal = agent.propose_and_train(state["X_train"], state["y_train"])
        proposals.append(proposal.to_dict())
    return {**state, "proposals": proposals}


def node_judge(state: PipelineState) -> PipelineState:
    judge = JudgeAgent()
    decision = judge.decide(state["proposals"])

    # Baseline test-set RMSE using the judge's pick, untuned.
    winner_type = decision["selected_model"]
    baseline_agent = BaseModelAgent(winner_type)
    baseline_model = baseline_agent.fit_full(state["X_train"], state["y_train"])
    baseline_pred = baseline_model.predict(state["X_test"])
    baseline_test_rmse = float(np.sqrt(mean_squared_error(state["y_test"], baseline_pred)))

    return {**state, "decision": decision, "baseline_test_rmse": baseline_test_rmse}


def node_tune(state: PipelineState) -> PipelineState:
    winner_type = state["decision"]["selected_model"]
    n_trials = state.get("optuna_trials", 30)
    best_model, best_params, best_cv_rmse, _study = tune_model(
        state["X_train"], state["y_train"], winner_type, n_trials=n_trials
    )
    return {**state, "tuned_params": best_params, "tuned_cv_rmse": best_cv_rmse,
            "_tuned_model": best_model}


def node_evaluate(state: PipelineState) -> PipelineState:
    model = state["_tuned_model"]
    pred = model.predict(state["X_test"])
    tuned_test_rmse = float(np.sqrt(mean_squared_error(state["y_test"], pred)))
    return {**state, "tuned_test_rmse": tuned_test_rmse}


def node_report(state: PipelineState) -> PipelineState:
    baseline = state["baseline_test_rmse"]
    tuned = state["tuned_test_rmse"]
    improvement_pct = 100.0 * (baseline - tuned) / baseline if baseline else 0.0
    report = {
        "selected_model": state["decision"]["selected_model"],
        "ranking": state["decision"]["ranking"],
        "reasoning": state["decision"]["reasoning"],
        "baseline_test_rmse": round(baseline, 5),
        "tuned_cv_rmse": round(state["tuned_cv_rmse"], 5),
        "tuned_test_rmse": round(tuned, 5),
        "improvement_pct": round(improvement_pct, 2),
        "best_params": state["tuned_params"],
        "optuna_trials": state.get("optuna_trials", 30),
    }
    return {**state, "report": report}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("preprocess", node_preprocess)
    graph.add_node("compete", node_compete)
    graph.add_node("judge", node_judge)
    graph.add_node("tune", node_tune)
    graph.add_node("evaluate", node_evaluate)
    graph.add_node("report", node_report)

    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "compete")
    graph.add_edge("compete", "judge")
    graph.add_edge("judge", "tune")
    graph.add_edge("tune", "evaluate")
    graph.add_edge("evaluate", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_pipeline(X: np.ndarray, y: np.ndarray, optuna_trials: int = 30) -> dict[str, Any]:
    app = build_graph()
    final_state = app.invoke({"X": X, "y": y, "optuna_trials": optuna_trials})
    return final_state["report"]
