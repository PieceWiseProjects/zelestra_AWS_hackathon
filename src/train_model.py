import os
import gc
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

# Feature definitions
NUM_COLS = [
    "temperature", "irradiance", "panel_age", "maintenance_count", "soiling_ratio",
    "voltage", "current", "module_temperature", "cloud_coverage",
    "humidity", "wind_speed", "pressure",
    "power_output", "temp_diff", "soiled_irradiance", "has_error"
]
CAT_COLS = ["string_id", "error_code", "installation_type"]

def load_clean_data():
    df = pd.read_csv("data/processed/train_clean.csv", index_col=0)
    X = df.drop(columns=["efficiency"])
    y = df["efficiency"]
    return X, y

def get_preprocessor():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=-1))
    ])
    return ColumnTransformer([
        ("num", num_pipeline, NUM_COLS),
        ("cat", cat_pipeline, CAT_COLS)
    ])

def evaluate_model(name, model, X, y, preprocessor):
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    print(f"\n🔍 Evaluating {name} with 3-fold cross-validation:")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        print(f"➡️ Fold {fold+1}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        try:
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))  # ✅ compatible version
            print(f"   Fold {fold+1} RMSE: {rmse:.4f}")
            scores.append(rmse)
        except Exception as e:
            print(f"⚠️ {name} failed on fold {fold+1}: {e}")
            return None, float("inf")

        gc.collect()

    mean_rmse = np.mean(scores)
    print(f"✅ {name} Mean RMSE: {mean_rmse:.4f}")
    return pipe, mean_rmse

def train_best_model():
    X, y = load_clean_data()
    preprocessor = get_preprocessor()

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            device="gpu",
            gpu_use_dp=False,
            verbose=10,
            random_state=42
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            task_type="GPU",
            devices="0",
            verbose=10,
            random_state=42
        )
    }

    best_score = float("inf")
    best_pipeline = None
    best_name = ""

    for name, model in models.items():
        pipe, score = evaluate_model(name, model, X, y, preprocessor)
        if score < best_score:
            best_score = score
            best_pipeline = pipe
            best_name = name

    if best_pipeline:
        print(f"\n🏆 Best Model: {best_name} with RMSE = {best_score:.4f}")
        print("📦 Retraining on full dataset...")
        best_pipeline.fit(X, y)

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_pipeline, "models/model.pkl")
        print("✅ Saved best model to models/model.pkl")
    else:
        print("❌ All models failed. Please check your setup.")

if __name__ == "__main__":
    train_best_model()
