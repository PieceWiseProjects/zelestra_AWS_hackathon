import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import joblib
import os

def load_data(path):
    df = pd.read_csv(path, index_col=0)
    X = df.drop(columns=["efficiency"])
    y = df["efficiency"]
    return X, y

def train_lgbm(X, y):
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        device="gpu",
        gpu_use_dp=False,
        random_state=42,
        verbose=-1
    )
    return model

def evaluate_model(name, X, y):
    print(f"\n🚀 Training {name} LightGBM model...")

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = train_lgbm(X_train, y_train)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        print(f"  Fold {fold+1} RMSE: {rmse:.4f}")
        rmses.append(rmse)

    mean_rmse = np.mean(rmses)
    print(f"✅ {name} mean RMSE: {mean_rmse:.4f}")
    return mean_rmse

def train_best_model(X, y):
    model = train_lgbm(X, y)
    model.fit(X, y)
    return model

def main():
    os.makedirs("models", exist_ok=True)

    print("🔍 Loading baseline and group-aware datasets...")
    X_base, y_base = load_data("data/processed/train_clean.csv")
    X_group, y_group = load_data("data/processed/train_grouped.csv")

    rmse_base = evaluate_model("Baseline", X_base, y_base)
    rmse_group = evaluate_model("Group-aware", X_group, y_group)

    print("\n📊 Final RMSE Comparison:")
    print(f"  Baseline     : {rmse_base:.4f}")
    print(f"  Group-aware  : {rmse_group:.4f}")
    print(f"  Improvement  : {rmse_base - rmse_group:.4f}")

    if rmse_group < rmse_base:
        print("🏆 Group-aware model selected as best.")
        best_model = train_best_model(X_group, y_group)
    else:
        print("🏆 Baseline model selected as best.")
        best_model = train_best_model(X_base, y_base)

    joblib.dump(best_model, "models/best_model.pkl")
    print("✅ Saved best model to models/best_model.pkl")

if __name__ == "__main__":
    main()
