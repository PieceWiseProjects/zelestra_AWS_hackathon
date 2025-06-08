import pandas as pd
import joblib

def main():
    print("📦 Loading model...")
    model = joblib.load("models/best_model.pkl")

    print("📄 Loading test and train column reference...")
    df_test = pd.read_csv("data/processed/test_grouped.csv", index_col=0)
    df_train = pd.read_csv("data/processed/train_grouped.csv", index_col=0)

    # Get expected feature columns (excluding target)
    expected_features = df_train.drop(columns=["efficiency"]).columns.tolist()

    print("🧼 Aligning test columns...")
    for col in expected_features:
        if col not in df_test.columns:
            df_test[col] = 0.0  # or df_test[col] = df_test[col].mean() if you prefer

    df_test = df_test[expected_features]  # exact column order match

    print("🔮 Making predictions...")
    preds = model.predict(df_test)

    submission = pd.DataFrame({
        "id": df_test.index,
        "efficiency": preds
    })
    submission.to_csv("outputs/submission.csv", index=False)
    print("✅ Saved submission.csv")

if __name__ == "__main__":
    main()
