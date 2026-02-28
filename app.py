import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from transformers import logging as hf_logging

from agents.model_agents import BaseModelAgent
from agents.judge import JudgeAgent
from tools.model_tool import tune_model


hf_logging.set_verbosity_error()



def clean_dataframe(df, target_column):

    df = df.copy()

    for col in df.columns:
        if col != target_column:
            df[col] = pd.to_numeric(df[col], errors="coerce")


    df = df.dropna()

    df[target_column] = df[target_column].astype(int)

    return df


if __name__ == "__main__":


    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target


    df = df.astype("object")

    df.iloc[0, 0] = None
    df.iloc[1, 1] = "invalid"


    rf_agent = BaseModelAgent("random_forest")
    log_agent = BaseModelAgent("logistic")
    svm_agent = BaseModelAgent("svm")
    gb_agent = BaseModelAgent("gradient_boost")

    rf_proposal = rf_agent.propose_and_train(df.copy(), "target")
    log_proposal = log_agent.propose_and_train(df.copy(), "target")
    svm_proposal = svm_agent.propose_and_train(df.copy(), "target")
    gb_proposal = gb_agent.propose_and_train(df.copy(), "target")

    print("\n--- Model Proposals ---")
    print("RandomForest Proposal:\n", rf_proposal)
    print("\nLogistic Proposal:\n", log_proposal)
    print("\nSVM Proposal:\n", svm_proposal)
    print("\nGradient Boosting Proposal:\n", gb_proposal)

    judge = JudgeAgent()
    decision = judge.decide([rf_proposal, log_proposal, svm_proposal, gb_proposal])

    print("\n--- Final Decision ---")
    print(decision)

    best_model_type = decision["selected_model"]

    
    clean_df = clean_dataframe(df, "target")

    X = clean_df.drop(columns=["target"])
    y = clean_df["target"]

    print("\n--- Hyperparameter Tuning ---")

    best_model, best_params = tune_model(clean_df, "target", best_model_type)

    print("Best Parameters Found:", best_params)

    y_pred = best_model.predict(X)

  
    cm = confusion_matrix(y, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


    if best_model_type == "random_forest":

        importances = best_model.feature_importances_

        plt.figure()
        plt.bar(range(len(importances)), importances)
        plt.xticks(range(len(importances)), X.columns, rotation=45)
        plt.title("Feature Importance")
        plt.show()

    accuracy_full = best_model.score(X, y)

    class_totals = cm.sum(axis=1)
    class_correct = cm.diagonal()
    class_accuracy = class_correct / class_totals

    weakest_class = class_accuracy.argmin()
    weakest_accuracy = class_accuracy.min()

    print("\n--- Model Self-Critique ---")
    print(
        f"The selected model achieved an overall training accuracy of {accuracy_full:.4f}. "
        f"The weakest performing class was Class {weakest_class} "
        f"with accuracy {weakest_accuracy:.4f}. "
        "Performance may improve by collecting more samples for this class "
        "or tuning model hyperparameters further."
    )