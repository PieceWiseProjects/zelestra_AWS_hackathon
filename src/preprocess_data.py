import os
import json
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from catboost import CatBoostRegressor

# Configuration
NUMERIC_OBJECTS = ["humidity", "wind_speed", "pressure"]
NUMERIC_FEATURES = [
    "temperature", "irradiance", "panel_age", "maintenance_count",
    "soiling_ratio", "voltage", "current", "module_temperature", "cloud_coverage"
]
CATEGORICAL_FEATURES = ["string_id", "error_code", "installation_type"]
TARGET = "efficiency"
CORR_THRESHOLD = 0.8
IMPORTANCE_THRESHOLD = 0.002


def convert_object_numerics(df):
    for col in NUMERIC_OBJECTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def impute_numeric_columns(df, is_train=True, imputers=None):
    df = df.copy()
    if imputers is None:
        imputers = {}
    for col in df.select_dtypes(include="number").columns:
        if is_train:
            imputer = SimpleImputer(strategy="median")
            df[col] = imputer.fit_transform(df[[col]])
            imputers[col] = imputer
        else:
            df[col] = imputers[col].transform(df[[col]])
    return df, imputers


def engineer_features(df):
    if {"voltage", "current"}.issubset(df.columns):
        df["power"] = df["voltage"] * df["current"]
    if {"module_temperature", "irradiance"}.issubset(df.columns):
        df["temp_irradiance_ratio"] = df["module_temperature"] / (df["irradiance"] + 1e-3)
    if {"irradiance", "soiling_ratio"}.issubset(df.columns):
        df["soiling_effect"] = df["irradiance"] * df["soiling_ratio"]
    if {"irradiance", "cloud_coverage"}.issubset(df.columns):
        df["irradiance_penalty"] = df["irradiance"] * (1 - df["cloud_coverage"])
    if {"panel_age", "maintenance_count"}.issubset(df.columns):
        df["age_maintenance_ratio"] = df["panel_age"] / (df["maintenance_count"] + 1e-3)
    if {"temperature", "humidity"}.issubset(df.columns):
        df["temp_humidity_interaction"] = df["temperature"] * df["humidity"]
    if {"module_temperature", "temperature"}.issubset(df.columns):
        df["temp_module_temp_diff"] = df["module_temperature"] - df["temperature"]
    return df


def drop_highly_correlated(df, threshold=CORR_THRESHOLD):
    corr = df.select_dtypes(include="number").corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"📉 Dropping {len(to_drop)} highly correlated features: {to_drop}")
    return df.drop(columns=to_drop), to_drop


def select_important_features(X, y, cat_cols, threshold=IMPORTANCE_THRESHOLD):
    cat_indices = [X.columns.get_loc(col) for col in cat_cols if col in X.columns]
    model = CatBoostRegressor(iterations=300, learning_rate=0.1, depth=6, task_type="CPU", verbose=0, random_state=42)
    model.fit(X, y, cat_features=cat_indices)

    importances = model.get_feature_importance(prettified=True)
    selected_names = importances[importances["Importances"] > threshold]["Feature Id"].tolist()

    # Force-include all categorical columns if missing
    for col in cat_cols:
        if col in X.columns and col not in selected_names:
            selected_names.append(col)

    print(f"✅ Selected {len(selected_names)} features above importance threshold (including categoricals)")
    return X[selected_names], selected_names


def process_and_save(train_path, test_path, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)
    test_ids = test.index.copy()

    # Convert numeric object types
    train = convert_object_numerics(train)
    test = convert_object_numerics(test)

    # Feature engineering
    train = engineer_features(train)
    test = engineer_features(test)

    # Imputation
    train, imputers = impute_numeric_columns(train, is_train=True)
    test, _ = impute_numeric_columns(test, is_train=False, imputers=imputers)

    # Correlation-based drop
    train, dropped = drop_highly_correlated(train)
    test = test.drop(columns=dropped, errors="ignore")

    # Categorical cleanup
    for df in [train, test]:
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna("missing").astype(str)

    # Save present categorical columns
    cat_cols_present = [col for col in CATEGORICAL_FEATURES if col in train.columns]
    with open(os.path.join(out_dir, "catboost_categorical.json"), "w") as f:
        json.dump(cat_cols_present, f)

    # Feature selection
    if TARGET in train.columns:
        X, y = train.drop(columns=TARGET), train[TARGET]
        X_selected, keep_cols = select_important_features(X, y, cat_cols_present)
        train = X_selected.copy()
        train[TARGET] = y
        test = test[keep_cols]

        # Save selected feature names (optional)
        with open(os.path.join(out_dir, "feature_names.json"), "w") as f:
            json.dump(keep_cols, f)

    # Save processed datasets
    train.to_csv(os.path.join(out_dir, "train_clean.csv"))
    test.loc[test_ids].to_csv(os.path.join(out_dir, "test_clean.csv"))

    print("✅ Feature pipeline completed and saved!")


if __name__ == "__main__":
    process_and_save("data/raw/train.csv", "data/raw/test.csv")
