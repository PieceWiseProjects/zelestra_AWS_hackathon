import os

# Force LightGBM to use NVIDIA GPU if multiple OpenCL devices exist
os.environ["GPU_DEVICE_ID"] = "0"  # Usually 0 for NVIDIA
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def load_data(path):
    df = pd.read_csv(path, index_col=0)
    X = df.drop(columns=["efficiency"])
    y = df["efficiency"]
    return X, y

def cross_validate(name, model, X, y):
    print(f"\n🚀 Training {name}...")

    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"  Fold {fold+1} RMSE: {rmse:.4f}")
        rmses.append(rmse)

    mean_rmse = np.mean(rmses)
    print(f"✅ {name} Mean RMSE: {mean_rmse:.4f}")
    return model, mean_rmse

def get_models():
    return {
        # "RandomForest": RandomForestRegressor(
        #     n_estimators=1000,
        #     n_jobs=-1,
        #     max_depth=12,
        #     random_state=42
        # ),
        # "LightGBM": LGBMRegressor(
        #     n_estimators=1000,
        #     learning_rate=0.03,
        #     num_leaves=31,
        #     device="gpu",
        #     gpu_use_dp=False,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     random_state=42
        # ),
        "CatBoost": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            task_type="GPU",
            devices="0",
            verbose=0,
            random_state=42
        )
    }

def main():
    os.makedirs("models", exist_ok=True)
    print("📄 Loading clean enriched data")
    X, y = load_data("data/processed/train_clean.csv")

    best_score = float("inf")
    best_model = None
    best_name = ""

    for name, model in get_models().items():
        try:
            model_instance, score = cross_validate(name, model, X, y)
            if score < best_score:
                best_score = score
                best_model = model_instance
                best_name = name
        except Exception as e:
            print(f"⚠️ {name} failed: {e}")

    print(f"\n🏆 Best Model: {best_name} with RMSE = {best_score:.4f}")
    best_model.fit(X, y)
    joblib.dump(best_model, "models/best_model.pkl")
    print("✅ Saved best model to models/best_model.pkl")

if __name__ == "__main__":
    main()
