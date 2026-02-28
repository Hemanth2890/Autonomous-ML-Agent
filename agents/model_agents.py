from tools.model_tool import train_model
from utils.llm_helper import LLMHelper
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class BaseModelAgent:

    def __init__(self, model_type):
        self.model_type = model_type
        self.llm = LLMHelper()

    def propose_and_train(self, df, target_column):

        dataset_info = self._analyze_dataset(df, target_column)

        try:
            model, metrics = train_model(df.copy(), target_column, self.model_type)

        except Exception as e:

            print(f"[{self.model_type}] Training failed. Error:", str(e))
            print(f"[{self.model_type}] Applying automatic preprocessing...")

            df = df.copy()

            # -----------------------------
            # Separate target and features
            # -----------------------------
            y = df[target_column]
            X = df.drop(columns=[target_column])

            # -----------------------------
            # Encode target if categorical
            # -----------------------------
            if y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y)

            # -----------------------------
            # 🔥 Detect and convert DATE columns
            # -----------------------------
            for col in X.columns:
                try:
                    parsed = pd.to_datetime(X[col], errors="raise")
                    X[col + "_year"] = parsed.dt.year
                    X[col + "_month"] = parsed.dt.month
                    X[col + "_day"] = parsed.dt.day
                    X = X.drop(columns=[col])
                except:
                    pass

            # -----------------------------
            # Identify feature types again
            # -----------------------------
            categorical_cols = X.select_dtypes(include=["object"]).columns
            numeric_cols = X.select_dtypes(exclude=["object"]).columns

            # ===============================
            # NUMERIC FEATURES
            # ===============================
            if len(numeric_cols) > 0:
                num_imputer = SimpleImputer(strategy="mean")
                X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

                scaler = StandardScaler()
                X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

            # ===============================
            # CATEGORICAL FEATURES
            # ===============================
            if len(categorical_cols) > 0:
                cat_imputer = SimpleImputer(strategy="most_frequent")
                X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

            # -----------------------------
            # Final safety check
            # -----------------------------
            if X.isnull().sum().sum() > 0:
                raise ValueError("NaN values remain after preprocessing.")

            df_clean = pd.concat(
                [pd.DataFrame(X), pd.Series(y, name=target_column)],
                axis=1
            )

            if df_clean.empty:
                raise ValueError("Dataset became empty after preprocessing.")

            print(f"[{self.model_type}] Retrying training...")

            model, metrics = train_model(df_clean, target_column, self.model_type)

        proposal = {
            "model_type": self.model_type,
            "metrics": metrics,
            "dataset_analysis": dataset_info,
            "argument": self._generate_argument(dataset_info, metrics)
        }

        return proposal

    def _analyze_dataset(self, df, target_column):

        return {
            "num_samples": df.shape[0],
            "num_features": df.shape[1] - 1,
            "num_classes": df[target_column].nunique()
        }

    def _generate_argument(self, dataset_info, metrics):

        prompt = f"""
You are a senior machine learning engineer.

IMPORTANT:
- Only discuss the model: {self.model_type}.
- Do not mention other algorithms.
- Use only provided metrics.
- Do not invent statistics.

Dataset:
Samples: {dataset_info['num_samples']}
Features: {dataset_info['num_features']}
Classes: {dataset_info['num_classes']}

Performance:
Accuracy Mean: {metrics['accuracy_mean']:.4f}
Accuracy Std: {metrics['accuracy_std']:.4f}
F1 Mean: {metrics['f1_mean']:.4f}
Precision Mean: {metrics['precision_mean']:.4f}
Recall Mean: {metrics['recall_mean']:.4f}

Write exactly 3 professional sentences evaluating suitability.
"""

        response = self.llm.generate(prompt, max_new_tokens=120)

        return response.strip()
