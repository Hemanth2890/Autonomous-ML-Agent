import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_dataframe(df, target_column):

    df = df.copy()

    y = df[target_column]
    X = df.drop(columns=[target_column])

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    for col in X.columns:
        try:
            parsed = pd.to_datetime(X[col], errors="raise")
            X[col + "_year"] = parsed.dt.year
            X[col + "_month"] = parsed.dt.month
            X[col + "_day"] = parsed.dt.day
            X = X.drop(columns=[col])
        except:
            pass

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    # Numeric
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy="mean")
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Categorical
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return pd.concat(
        [pd.DataFrame(X), pd.Series(y, name=target_column)],
        axis=1
    )
