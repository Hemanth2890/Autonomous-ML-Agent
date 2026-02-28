# Autonomous Multi-Agent AutoML System

A hybrid AutoML system combining multiple ML models with LLM-based reasoning and intelligent model selection.

---

## Features

- Multi-agent model competition (RF, Logistic, SVM, Gradient Boost)
- Automatic class imbalance detection
- Adaptive metric switching (macro-F1 priority)
- Collapse detection & penalty
- Hyperparameter tuning
- 5-Fold Cross-Validated evaluation
- Confusion matrix & ROC curves
- Feature importance visualization
- Download trained model
- Streamlit frontend

---

## Architecture

1. Preprocessing Pipeline
2. Multiple Model Agents
3. Imbalance-Aware Training
4. Intelligent Judge Selection
5. Cross-Validated Evaluation
6. Visualization & Reporting

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Autonomous-ML-Agent.git
cd Autonomous-ML-Agent
pip install -r requirements.txt

streamlit run frontend.py
