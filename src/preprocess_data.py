import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Column groups
NUMERIC_OBJECTS = ["humidity", "wind_speed", "pressure"]
NUMERIC_FEATURES = [
    "temperature", "irradiance", "panel_age", "maintenance_count",
    "soiling_ratio", "voltage", "current", "module_temperature", "cloud_coverage"
]
CATEGORICAL_FEATURES = ["string_id", "error_code", "installation_type"]


def convert_numeric_objects(df):
    for col in NUMERIC_OBJECTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def impute_and_encode(df, is_train=True, imputers=None, encoders=None):
    df = df.copy()
    if imputers is None: imputers = {}
    if encoders is None: encoders = {}

    # Impute numeric
    for col in df.select_dtypes(include="number").columns:
        if is_train:
            imputer = SimpleImputer(strategy="median")
            df[col] = imputer.fit_transform(df[[col]])
            imputers[col] = imputer
        else:
            df[col] = imputers[col].transform(df[[col]])

    # Impute + encode categoricals
    cat_cols = CATEGORICAL_FEATURES
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("missing")
            if is_train:
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                df[[col]] = encoder.fit_transform(df[[col]])
                encoders[col] = encoder
            else:
                df[[col]] = encoders[col].transform(df[[col]])

    return df, imputers, encoders


def drop_highly_correlated(df, threshold=0.76):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"📉 Dropping {len(to_drop)} highly correlated features: {to_drop}")
    return df.drop(columns=to_drop), to_drop


def preprocess_and_save():
    os.makedirs("data/processed", exist_ok=True)

    train = pd.read_csv("data/raw/train.csv", index_col=0)
    test = pd.read_csv("data/raw/test.csv", index_col=0)

    # Basic cleanup
    train = convert_numeric_objects(train)
    test = convert_numeric_objects(test)

    # Final encoding/imputation
    train, imputers, encoders = impute_and_encode(train, is_train=True)
    test, _, _ = impute_and_encode(test, is_train=False, imputers=imputers, encoders=encoders)

    # Drop correlated features
    train, dropped = drop_highly_correlated(train)
    test = test.drop(columns=dropped, errors="ignore")

    # Save
    train.to_csv("data/processed/train_clean.csv")
    test.to_csv("data/processed/test_clean.csv")
    print("✅ Saved processed data to data/processed/")


if __name__ == "__main__":
    preprocess_and_save()
