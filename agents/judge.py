from utils.llm_helper import LLMHelper


class JudgeAgent:

    def __init__(self):
        self.llm = LLMHelper()

    def decide(self, proposals):

        for proposal in proposals:
            m = proposal["metrics"]


            final_score = (
                0.5 * m["f1_mean"] +
                0.2 * m["precision_mean"] +
                0.2 * m["recall_mean"] +
                0.05 * m["accuracy_mean"] -
                0.05 * m["complexity_penalty"] -
                0.05 * m["stability_penalty"]
            )

            imbalance_ratio = m.get("imbalance_ratio", 0)

            # If dataset is highly imbalanced AND F1 is much lower than accuracy,
            # model is likely collapsing to majority class
            if imbalance_ratio > 0.6:
                if m["accuracy_mean"] - m["f1_mean"] > 0.15:
                    final_score *= 0.75  # Penalize collapse

            proposal["final_score"] = final_score

        # Select best model
        best = max(proposals, key=lambda x: x["final_score"])

        explanation_prompt = f"""
You are an expert ML system evaluator.

The selected model is {best['model_type']}.

Cross-validation metrics:
- Mean Accuracy: {best['metrics']['accuracy_mean']:.4f}
- Mean Macro F1: {best['metrics']['f1_mean']:.4f}
- Accuracy Std Dev: {best['metrics']['accuracy_std']:.4f}
- Imbalance Ratio: {best['metrics'].get('imbalance_ratio', 0):.4f}

Explain in 3 concise sentences why this model was selected
based on balanced performance, stability, and complexity.
Do NOT invent numbers.
"""

        explanation = self.llm.generate(explanation_prompt, max_new_tokens=120)

        decision = {
            "selected_model": best["model_type"],
            "final_score": round(best["final_score"], 4),
            "reason": explanation.strip()
        }

        return decision
