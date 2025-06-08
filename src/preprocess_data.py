import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

NUMERIC_OBJECTS = ["humidity", "wind_speed", "pressure"]
NUMERIC_FEATURES = [
    "temperature", "irradiance", "panel_age", "maintenance_count",
    "soiling_ratio", "voltage", "current", "module_temperature", "cloud_coverage"
]
CATEGORICAL_FEATURES = ["string_id", "error_code", "installation_type"]

DERIVED_FEATURES = {
    "power_output": lambda df: df["voltage"] * df["current"],
    "temp_diff": lambda df: df["module_temperature"] - df["temperature"],
    "soiled_irradiance": lambda df: df["irradiance"] * (1 - df["soiling_ratio"]),
    "has_error": lambda df: df["error_code"].notnull().astype(int),
}


def convert_numeric_objects(df):
    for col in NUMERIC_OBJECTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_derived_features(df):
    for name, func in DERIVED_FEATURES.items():
        df[name] = func(df)
    return df


def impute_and_encode(df, is_train=True, encoders=None, imputers=None):
    df = df.copy()

    # Impute numeric
    if imputers is None:
        imputers = {}
    for col in NUMERIC_OBJECTS + NUMERIC_FEATURES:
        if is_train:
            imputer = SimpleImputer(strategy="median")
            df[col] = imputer.fit_transform(df[[col]])
            imputers[col] = imputer
        else:
            df[col] = imputers[col].transform(df[[col]])

    # Impute + encode categorical
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_FEATURES:
        if is_train:
            df[col] = df[col].fillna("missing")
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[[col]] = encoder.fit_transform(df[[col]])
            encoders[col] = encoder
        else:
            df[col] = df[col].fillna("missing")
            df[[col]] = encoders[col].transform(df[[col]])

    return df, imputers, encoders


def preprocess_and_save():
    os.makedirs("data/processed", exist_ok=True)

    train = pd.read_csv("data/raw/train.csv", index_col=0)
    test = pd.read_csv("data/raw/test.csv", index_col=0)

    # Step 1: convert object-like numerics
    train = convert_numeric_objects(train)
    test = convert_numeric_objects(test)

    # Step 2: feature engineering
    train = add_derived_features(train)
    test = add_derived_features(test)

    # Step 3: impute and encode
    train_clean, imputers, encoders = impute_and_encode(train, is_train=True)
    test_clean, _, _ = impute_and_encode(test, is_train=False, imputers=imputers, encoders=encoders)

    # Save
    train_clean.to_csv("data/processed/train_clean.csv")
    test_clean.to_csv("data/processed/test_clean.csv")
    print("✅ Saved processed data to data/processed/")


if __name__ == "__main__":
    preprocess_and_save()
