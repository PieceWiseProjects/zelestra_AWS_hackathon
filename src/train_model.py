import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# Constants
MODEL_NAME = "catboost_native_cat"
MODEL_DIR = "models"
LOG_FILE = "models/model_eval_log.csv"
DATA_PATH = "data/processed/train_clean.csv"
CAT_FEATURES_PATH = "data/processed/catboost_categorical.json"
N_SPLITS = 4


def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0)
    X = df.drop(columns=["efficiency"])
    y = df["efficiency"]
    groups = df["string_id"] if "string_id" in df.columns else None

    with open(CAT_FEATURES_PATH) as f:
        cat_features = json.load(f)
    cat_feature_indices = [X.columns.get_loc(col) for col in cat_features if col in X.columns]

    return X, y, groups, cat_feature_indices


def cross_validate(model, X, y, cat_indices, groups=None):
    print(f"\n🚀 Training {MODEL_NAME}...")

    kf = GroupKFold(n_splits=N_SPLITS) if groups is not None else KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    print(f"🔍 Using {'GroupKFold' if groups is not None else 'KFold'} with {N_SPLITS} splits")
    rmses = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups)):
        X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 🛠️ Fix: Convert categorical features to string and fillna
        for cat_idx in cat_indices:
            col = X.columns[cat_idx]
            X_train[col] = X_train[col].astype(str).fillna("missing")
            X_val[col] = X_val[col].astype(str).fillna("missing")

        model.fit(X_train, y_train, cat_features=cat_indices, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"  Fold {fold + 1} RMSE: {rmse:.5f}")
        rmses.append(rmse)

    mean_rmse = np.mean(rmses)
    print(f"✅ {MODEL_NAME} Mean RMSE: {mean_rmse:.5f}")
    return model, mean_rmse, rmses



def get_model():
    return CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        task_type="GPU",
        devices="0",
        verbose=0,
        loss_function="RMSE",
        random_state=42
    )


def log_evaluation(mean_rmse, fold_rmses, params):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": MODEL_NAME,
        "mean_rmse": round(mean_rmse, 6),
        "fold_rmses": json.dumps([round(r, 6) for r in fold_rmses]),
        "params": json.dumps(params)
    }
    if not os.path.exists(LOG_FILE):
        pd.DataFrame([row]).to_csv(LOG_FILE, index=False)
    else:
        pd.concat([pd.read_csv(LOG_FILE), pd.DataFrame([row])]).to_csv(LOG_FILE, index=False)
    print(f"📄 Logged results to {LOG_FILE}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y, groups, cat_feature_indices = load_data()

    model = get_model()
    trained_model, mean_rmse, fold_rmses = cross_validate(model, X, y, cat_feature_indices, groups)

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pkl")
    joblib.dump(trained_model, model_path)
    print(f"✅ Saved model to {model_path}")

    # Log evaluation
    log_evaluation(mean_rmse, fold_rmses, trained_model.get_params())


if __name__ == "__main__":
    main()
