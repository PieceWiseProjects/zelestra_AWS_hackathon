import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Feature definitions
NUMERIC_OBJECTS = ["humidity", "wind_speed", "pressure"]
NUMERIC_FEATURES = [
    "temperature", "irradiance", "panel_age", "maintenance_count",
    "soiling_ratio", "voltage", "current", "module_temperature", "cloud_coverage"
]
CATEGORICAL_FEATURES = ["string_id", "error_code", "installation_type"]

# Derived features
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

    if imputers is None:
        imputers = {}
    for col in NUMERIC_OBJECTS + NUMERIC_FEATURES:
        if is_train:
            imp = SimpleImputer(strategy="median")
            df[col] = imp.fit_transform(df[[col]])
            imputers[col] = imp
        else:
            df[col] = imputers[col].transform(df[[col]])

    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("missing")
        if is_train:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[[col]] = enc.fit_transform(df[[col]])
            encoders[col] = enc
        else:
            df[[col]] = encoders[col].transform(df[[col]])

    return df, imputers, encoders

def add_group_features(df, id_col="string_id"):
    group_cols = {
        "power_output": ["mean", "std"],
        "voltage": ["mean", "std"],
        "current": ["mean", "std"],
        "irradiance": ["mean", "std"]
    }

    if "efficiency" in df.columns:
        group_cols["efficiency"] = ["mean", "std"]

    group_stats = df.groupby(id_col).agg(group_cols)
    group_stats.columns = [f"{a}_{b}_by_{id_col}" for a, b in group_stats.columns]
    return df.merge(group_stats, left_on=id_col, right_index=True, how="left")

def preprocess_and_save():
    os.makedirs("data/processed", exist_ok=True)

    train = pd.read_csv("data/raw/train.csv", index_col=0)
    test = pd.read_csv("data/raw/test.csv", index_col=0)

    train = convert_numeric_objects(train)
    test = convert_numeric_objects(test)

    train = add_derived_features(train)
    test = add_derived_features(test)

    train_clean, imputers, encoders = impute_and_encode(train, is_train=True)
    test_clean, _, _ = impute_and_encode(test, is_train=False, imputers=imputers, encoders=encoders)

    train_clean.to_csv("data/processed/train_clean.csv")
    test_clean.to_csv("data/processed/test_clean.csv")
    print("✅ Saved: baseline clean datasets")

    train_grouped = add_group_features(train_clean)
    test_grouped = add_group_features(test_clean)

    train_grouped.to_csv("data/processed/train_grouped.csv")
    test_grouped.to_csv("data/processed/test_grouped.csv")
    print("✅ Saved: group-aware datasets")

if __name__ == "__main__":
    preprocess_and_save()
