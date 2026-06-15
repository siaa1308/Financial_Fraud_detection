import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


DATA_DIR = "data"
SAVE_DIR = "processed"
os.makedirs(SAVE_DIR, exist_ok=True)


TARGET = "is_sar"

DROP_COLS = [
    "fraud_type",
    "suspicious_reason",
    "alert_type_join",
    "tran_id",
    "alert_id",
    "label_available_at"
]


def load_and_preprocess(file_name, save_name):
    print(f"\nProcessing {file_name}")

    df = pd.read_csv(os.path.join(DATA_DIR, file_name))

    # Sort by time
    df = df.sort_values("tran_timestamp").reset_index(drop=True)

    # Target
    y = df[TARGET].copy()

    # Drop columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop + [TARGET])

    # Optional: derive time features
    X["hour"] = pd.to_datetime(df["tran_timestamp"], unit="s", errors="coerce").dt.hour.fillna(0)
    X["dayofweek"] = pd.to_datetime(df["tran_timestamp"], unit="s", errors="coerce").dt.dayofweek.fillna(0)

    # Remove raw timestamp after feature extraction
    if "tran_timestamp" in X.columns:
        X = X.drop(columns=["tran_timestamp"])

    # Split indexes (time-based)
    n = len(df)
    train_end = int(0.70 * n)
    val_end   = int(0.85 * n)

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    # Detect column types
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    # Pipelines
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    # Fit only on train
    X_train = preprocessor.fit_transform(X_train)
    X_val   = preprocessor.transform(X_val)
    X_test  = preprocessor.transform(X_test)

    # Save
    data = {
        "X_train": X_train,
        "y_train": y_train.values,
        "X_val": X_val,
        "y_val": y_val.values,
        "X_test": X_test,
        "y_test": y_test.values,
        "preprocessor": preprocessor
    }

    joblib.dump(data, os.path.join(SAVE_DIR, save_name))
    print(f"Saved -> {save_name}")


# Run all banks
load_and_preprocess("Bank_A.csv", "bankA.pkl")
load_and_preprocess("Bank_B.csv", "bankB.pkl")
load_and_preprocess("Bank_C.csv", "bankC.pkl")