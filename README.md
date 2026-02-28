# 🚀 Autonomous Multi-Agent AutoML System

An imbalance-aware, multi-agent AutoML framework that intelligently selects and evaluates classification models using cross-validated macro-F1 optimization and LLM-driven reasoning.

---

## 🧠 Overview

This project implements a hybrid AutoML architecture that combines:

- Classical Machine Learning (Scikit-Learn)
- Multi-Agent Model Competition
- Imbalance-Aware Metric Switching
- Cross-Validated Evaluation
- LLM-Based Structured Reasoning
- Interactive Streamlit Dashboard

Instead of selecting models purely by accuracy, the system dynamically adapts to dataset characteristics such as class imbalance and stability variance.

---

## 🏗 System Architecture

The pipeline consists of:

### 1️⃣ Preprocessing Engine
- Automatic numeric & categorical handling
- Missing value imputation
- One-hot encoding
- Feature scaling
- Date parsing support

### 2️⃣ Competing Model Agents
Each agent independently trains and evaluates:

- Random Forest
- Logistic Regression
- Support Vector Machine
- Gradient Boosting

Using:
- Stratified 5-Fold Cross Validation
- Macro-F1 Optimization
- Stability (Std Dev) Monitoring
- Imbalance Ratio Detection

---

### 3️⃣ Intelligent Judge Agent

A weighted scoring system selects the best model based on:

- Accuracy
- Macro F1 Score
- Precision & Recall
- Stability (variance across folds)
- Complexity penalty
- Collapse detection (for majority-class bias)

Additionally, an LLM generates structured reasoning explaining the selection.

---

### 4️⃣ Hyperparameter Optimization
GridSearchCV is applied to the selected model for fine-tuning.

---

### 5️⃣ Honest Evaluation Strategy

All performance metrics are computed using:

✔ 5-Fold Cross-Validated Predictions  
✔ Cross-Validated Confusion Matrix  
✔ Per-Class Accuracy  
✔ Multi-Class ROC Curves  
✔ Classification Reports  

No train-test leakage.

---

## 📊 Dashboard Features

The Streamlit frontend allows users to:

- Upload any CSV classification dataset
- Select target column
- Run automated model competition
- View leaderboard
- Inspect confusion matrix
- Analyze per-class performance
- Visualize ROC curves
- View feature importance (tree models)
- Download trained model

---

## 🧪 Evaluation Philosophy

This project emphasizes:

- Robustness over raw accuracy
- Macro-F1 for imbalanced datasets
- Stability-aware scoring
- Detection of model collapse behavior
- Honest generalization via cross-validation

---

## ⚙️ Installation

```bash
git clone https://github.com/Hemanth2890/Autonomous-ML-Agent.git
cd Autonomous-ML-Agent
pip install -r requirements.txt
streamlit run frontend.py
