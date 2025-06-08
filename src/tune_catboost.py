import os

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_data():
    df = pd.read_csv("data/processed/train_clean.csv", index_col=0)
    X = df.drop(columns=["efficiency"])
    y = df["efficiency"]
    return X, y


def objective(trial):
    X, y = load_data()
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    params = {
        "iterations": trial.suggest_int("iterations", 500, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "random_state": 42,
        "loss_function": "RMSE",
        "task_type": "GPU",
        "devices": "0",
        "verbose": 0
    }

    rmses = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostRegressor(**params)
        model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=0
        )
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        rmses.append(rmse)

    return np.mean(rmses)


def main():
    print("🔍 Starting Optuna tuning for CatBoost...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, timeout=1800)

    print("\n✅ Best trial:")
    print(f"  RMSE: {study.best_value:.5f}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Retrain on full data
    X, y = load_data()
    model = CatBoostRegressor(
            **study.best_params,
            loss_function="RMSE",
            task_type="GPU",
            devices="0",
            verbose=100
    )
    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_catboost_tuned.pkl")
    print("✅ Saved tuned CatBoost model to models/model_catboost_tuned.pkl")


if __name__ == "__main__":
    main()
