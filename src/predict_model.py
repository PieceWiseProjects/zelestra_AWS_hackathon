import os
import pandas as pd
import joblib

def predict_and_save():
    # Load cleaned test data
    test = pd.read_csv("data/processed/test_clean.csv", index_col=0)

    print(f"loading best model from models/model_catboost_tuned.pkl")
    # Load trained pipeline
    model = joblib.load("models/model_catboost_tuned.pkl")
    print(f" {model.__class__.__name__ } loaded successfully")
    # Predict
    print("🔍 Running inference...")
    preds = model.predict(test)

    # Create submission DataFrame
    submission = pd.DataFrame({
        "id": test.index,
        "efficiency": preds
    })

    os.makedirs("outputs", exist_ok=True)
    submission.to_csv("outputs/submission.csv", index=False)
    print("✅ Submission saved to outputs/submission.csv")

if __name__ == "__main__":
    predict_and_save()
