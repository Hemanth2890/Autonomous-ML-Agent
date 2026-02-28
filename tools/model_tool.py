import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def train_model(df, target_column, model_type="random_forest"):

    X = df.drop(columns=[target_column])
    y = df[target_column]


    class_distribution = y.value_counts(normalize=True)
    imbalance_ratio = class_distribution.max()

    use_balancing = imbalance_ratio > 0.6


    if model_type == "random_forest":
        model = RandomForestClassifier(
            class_weight="balanced" if use_balancing else None,
            random_state=42
        )
        complexity_penalty = 0.7

    elif model_type == "logistic":
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced" if use_balancing else None
        )
        complexity_penalty = 0.3

    elif model_type == "svm":
        model = SVC(
            probability=True,
            random_state=42,
            class_weight="balanced" if use_balancing else None
        )
        complexity_penalty = 0.5

    elif model_type == "gradient_boost":
        model = GradientBoostingClassifier(random_state=42)
        complexity_penalty = 0.6

    else:
        raise ValueError("Unsupported model type")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro"
    ]

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring=scoring,
        error_score="raise"
    )

    accuracy_mean = float(np.mean(cv_results["test_accuracy"]))
    accuracy_std = float(np.std(cv_results["test_accuracy"]))

    precision_mean = float(np.mean(cv_results["test_precision_macro"]))
    recall_mean = float(np.mean(cv_results["test_recall_macro"]))
    f1_mean = float(np.mean(cv_results["test_f1_macro"]))

    metrics = {
        "accuracy_mean": accuracy_mean,
        "accuracy_std": accuracy_std,
        "precision_mean": precision_mean,
        "recall_mean": recall_mean,
        "f1_mean": f1_mean,
        "complexity_penalty": complexity_penalty,
        "stability_penalty": accuracy_std,
        "imbalance_ratio": float(imbalance_ratio)
    }

    model.fit(X, y)

    return model, metrics



def tune_model(df, target_column, model_type):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    class_distribution = y.value_counts(normalize=True)
    imbalance_ratio = class_distribution.max()
    use_balancing = imbalance_ratio > 0.6

    if model_type == "random_forest":

        model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced" if use_balancing else None
        )

        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5]
        }

    elif model_type == "logistic":

        model = LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight="balanced" if use_balancing else None
        )

        param_grid = {
            "C": [0.1, 1, 10]
        }

    elif model_type == "svm":

        model = SVC(
            probability=True,
            random_state=42,
            class_weight="balanced" if use_balancing else None
        )

        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }

    elif model_type == "gradient_boost":

        model = GradientBoostingClassifier(random_state=42)

        param_grid = {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        }

    else:
        raise ValueError("Unsupported model type")

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="f1_macro",   
        n_jobs=1,
        error_score="raise"
    )

    grid.fit(X, y)

    return grid.best_estimator_, grid.best_params_
