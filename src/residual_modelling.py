import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# Paths
TRAIN_PATH = "data/processed/train_clean.csv"
CAT_PATH = "data/processed/catboost_categorical.json"
MODEL_DIR = "models"
PRIMARY_MODEL_PATH = os.path.join(MODEL_DIR, "model_primary.pkl")
RESIDUAL_MODEL_PATH = os.path.join(MODEL_DIR, "model_residual.pkl")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, "model_2stage_ensemble.pkl")

N_SPLITS = 4
SEED = 42


def load_data():
    df = pd.read_csv(TRAIN_PATH, index_col=0)
    X = df.drop(columns=["efficiency"])
    y = df["efficiency"]
    groups = df["string_id"] if "string_id" in df.columns else None
    with open(CAT_PATH) as f:
        cat_cols = json.load(f)
    cat_indices = [X.columns.get_loc(col) for col in cat_cols if col in X.columns]
    return X, y, groups, cat_cols, cat_indices


def train_catboost(X, y, cat_indices, groups, model_name):
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        task_type="CPU",
        loss_function="RMSE",
        random_state=SEED,
        verbose=0
    )

    cv = GroupKFold(n_splits=N_SPLITS)
    preds = np.zeros(len(X))
    for train_idx, val_idx in cv.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for idx in cat_indices:
            col = X.columns[idx]
            X_train[col] = X_train[col].astype(str).fillna("missing")
            X_val[col] = X_val[col].astype(str).fillna("missing")

        model.fit(X_train, y_train, cat_features=cat_indices)
        preds[val_idx] = model.predict(X_val)

    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    print(f"✅ CV RMSE ({model_name}): {rmse:.5f}")
    return model.fit(X, y, cat_features=cat_indices), preds


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y, groups, cat_cols, cat_indices = load_data()

    print("\n📘 Stage 1: Primary Model")
    model_1, preds_1 = train_catboost(X, y, cat_indices, groups, "primary")
    joblib.dump(model_1, PRIMARY_MODEL_PATH)

    print("\n📗 Stage 2: Residual Model")
    residuals = y - preds_1
    model_2, preds_2 = train_catboost(X, residuals, cat_indices, groups, "residual")
    joblib.dump(model_2, RESIDUAL_MODEL_PATH)

    final_preds = preds_1 + preds_2
    final_mse = mean_squared_error(y, final_preds)
    final_rmse = np.sqrt(final_mse)
    print(f"\n🏁 Final 2-Stage RMSE: {final_rmse:.5f}")

    joblib.dump((model_1, model_2), ENSEMBLE_MODEL_PATH)
    print(f"💾 Saved 2-stage ensemble to {ENSEMBLE_MODEL_PATH}")


if __name__ == "__main__":
    main()
