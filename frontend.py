import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_predict

from agents.model_agents import BaseModelAgent
from agents.judge import JudgeAgent
from tools.model_tool import tune_model
from utils.preprocessing import preprocess_dataframe



st.set_page_config(page_title="Autonomous AutoML", layout="wide")

st.title("Autonomous Multi-Agent AutoML System")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)

  
    if st.button("Run AutoML"):

   
        with st.spinner("Preprocessing dataset..."):
            clean_df = preprocess_dataframe(df.copy(), target_column)


        with st.spinner("Running model competition..."):

            agents = [
                BaseModelAgent("random_forest"),
                BaseModelAgent("logistic"),
                BaseModelAgent("svm"),
                BaseModelAgent("gradient_boost")
            ]

            proposals = [
                agent.propose_and_train(clean_df.copy(), target_column)
                for agent in agents
            ]

            judge = JudgeAgent()
            decision = judge.decide(proposals)

        st.success("Model Selection Complete")

        st.subheader("Model Leaderboard")

        leaderboard = pd.DataFrame([
            {
                "Model": p["model_type"],
                "Accuracy": p["metrics"]["accuracy_mean"],
                "Macro F1": p["metrics"]["f1_mean"],
                "Stability (Std)": p["metrics"]["accuracy_std"],
                "Imbalance Ratio": p["metrics"].get("imbalance_ratio", 0)
            }
            for p in proposals
        ])

        leaderboard = leaderboard.sort_values("Macro F1", ascending=False)
        st.dataframe(leaderboard)


        st.subheader("Final Model Selected")
        st.json(decision)

        with st.spinner("Tuning hyperparameters..."):
            best_model, best_params = tune_model(
                clean_df.copy(),
                target_column,
                decision["selected_model"]
            )

        st.subheader("Best Hyperparameters")
        st.json(best_params)


        X = clean_df.drop(columns=[target_column])
        y = clean_df[target_column]

        with st.spinner("Evaluating with 5-fold Cross Validation..."):

            y_pred = cross_val_predict(best_model, X, y, cv=5)

        st.subheader("Confusion Matrix (Cross-Validated)")

        cm = confusion_matrix(y, y_pred)

        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")

        st.pyplot(fig_cm)

        st.subheader("Per-Class Accuracy")

        class_totals = cm.sum(axis=1)
        class_correct = cm.diagonal()
        class_accuracy = class_correct / class_totals

        class_df = pd.DataFrame({
            "Class": np.arange(len(class_accuracy)),
            "Accuracy": class_accuracy
        })

        st.dataframe(class_df)


        st.subheader("Classification Report")

        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        st.dataframe(report_df)


        st.subheader("ROC Curve (Cross-Validated)")

        if hasattr(best_model, "predict_proba"):

            y_bin = label_binarize(y, classes=np.unique(y))

            y_score = cross_val_predict(
                best_model,
                X,
                y,
                cv=5,
                method="predict_proba"
            )

            fig_roc, ax_roc = plt.subplots()

            for i in range(y_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

            ax_roc.plot([0, 1], [0, 1], "k--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend()

            st.pyplot(fig_roc)

        else:
            st.info("ROC curve not available for this model.")

        if hasattr(best_model, "feature_importances_"):

            st.subheader("Feature Importance")

            best_model.fit(X, y)

            importances = best_model.feature_importances_
            feature_names = X.columns

            feat_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values("Importance", ascending=False).head(10)

            fig_imp, ax_imp = plt.subplots()
            sns.barplot(
                data=feat_df,
                x="Importance",
                y="Feature",
                ax=ax_imp
            )

            st.pyplot(fig_imp)

        best_model.fit(X, y)

        st.subheader("Download Trained Model")

        model_bytes = pickle.dumps(best_model)

        st.download_button(
            label="Download Model (.pkl)",
            data=model_bytes,
            file_name="trained_model.pkl",
            mime="application/octet-stream"
        )

        st.success("AutoML Pipeline Completed Successfully")