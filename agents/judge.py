"""
JudgeAgent: uses a locally running LLM (via Ollama) to decide which
competing model to select, given each model's cross-validated metrics.
The LLM is shown the full leaderboard and asked to pick a winner and
explain its reasoning -- this is the actual decision point, not just a
post-hoc explanation of a formula's output.

Requires Ollama running locally (https://ollama.com) with a model pulled,
e.g. `ollama pull llama3`. No API key, no per-request billing, fully
offline. If Ollama is not reachable, falls back to a deterministic
weighted-scoring rule so the pipeline still works.
"""
from __future__ import annotations

import json
import os
from typing import Any


class JudgeAgent:
    def __init__(self, stability_penalty_weight: float = 0.5,
                 model: str = "llama3", ollama_host: str | None = None):
        self.stability_penalty_weight = stability_penalty_weight
        self.model = model
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def decide(self, proposals: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [p for p in proposals if p.get("error") is None]
        if not valid:
            raise RuntimeError("All model agents failed to train.")

        try:
            return self._decide_with_llm(valid)
        except Exception as exc:  # noqa: BLE001
            # Ollama not running / model not pulled / parse failure --
            # fall back rather than crash the pipeline.
            fallback = self._decide_deterministic(valid)
            fallback["reasoning"] = (
                f"[LLM judge unavailable ({exc}); used deterministic fallback] "
                + fallback["reasoning"]
            )
            return fallback

    def _decide_with_llm(self, valid: list[dict[str, Any]]) -> dict[str, Any]:
        import ollama

        client = ollama.Client(host=self.ollama_host)

        leaderboard = [
            {"model_type": p["model_type"],
             "cv_rmse_mean": round(p["cv_rmse_mean"], 5),
             "cv_rmse_std": round(p["cv_rmse_std"], 5),
             "fit_time_seconds": p.get("fit_time_seconds")}
            for p in valid
        ]

        prompt = (
            "You are selecting the best regression model from a set of "
            "candidates that were each evaluated with 5-fold cross-validation. "
            "Lower RMSE is better. Lower RMSE std means the model is more "
            "stable across folds, which matters for generalization, not just "
            "raw accuracy. Here is the leaderboard:\n\n"
            f"{json.dumps(leaderboard, indent=2)}\n\n"
            "Pick the single best model considering both accuracy and "
            "stability (a model with the lowest mean RMSE but high variance "
            "across folds may be a worse choice than a slightly higher-RMSE "
            "model that is more consistent). Respond with ONLY a JSON object "
            "in this exact format, no other text:\n"
            '{"selected_model": "<model_type>", "reasoning": "<2-3 sentence '
            'explanation referencing the actual numbers>"}'
        )

        response = client.generate(
            model=self.model,
            prompt=prompt,
            format="json",
            options={"temperature": 0.2},
        )

        decision = json.loads(response["response"])

        selected_type = decision["selected_model"]
        selected = next((p for p in valid if p["model_type"] == selected_type), None)
        if selected is None:
            raise ValueError(f"LLM selected unknown model_type '{selected_type}'")

        ranking = sorted(
            [{"model_type": p["model_type"],
              "cv_rmse_mean": round(p["cv_rmse_mean"], 5),
              "cv_rmse_std": round(p["cv_rmse_std"], 5)} for p in valid],
            key=lambda r: r["cv_rmse_mean"],
        )

        return {
            "selected_model": selected_type,
            "selected_score": round(selected["cv_rmse_mean"], 5),
            "baseline_cv_rmse": selected["cv_rmse_mean"],
            "ranking": ranking,
            "reasoning": decision["reasoning"],
            "decision_method": "llm",
        }

    def _decide_deterministic(self, valid: list[dict[str, Any]]) -> dict[str, Any]:
        scored = []
        for p in valid:
            score = p["cv_rmse_mean"] + self.stability_penalty_weight * p["cv_rmse_std"]
            scored.append((score, p))
        scored.sort(key=lambda t: t[0])

        best_score, best = scored[0]
        ranking = [
            {"model_type": p["model_type"], "score": round(s, 5),
             "cv_rmse_mean": round(p["cv_rmse_mean"], 5),
             "cv_rmse_std": round(p["cv_rmse_std"], 5)}
            for s, p in scored
        ]

        reasoning = (
            f"Selected '{best['model_type']}' with cross-validated RMSE "
            f"{best['cv_rmse_mean']:.4f} (+/-{best['cv_rmse_std']:.4f}). "
            f"It had the lowest stability-penalized score ({best_score:.4f}) "
            f"among {len(valid)} candidates: "
            + ", ".join(f"{p['model_type']}={p['cv_rmse_mean']:.4f}" for p in valid)
            + "."
        )

        return {
            "selected_model": best["model_type"],
            "selected_score": round(best_score, 5),
            "baseline_cv_rmse": best["cv_rmse_mean"],
            "ranking": ranking,
            "reasoning": reasoning,
            "decision_method": "deterministic",
        }