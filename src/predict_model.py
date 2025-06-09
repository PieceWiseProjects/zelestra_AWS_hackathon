import datetime
import json
import os

import joblib
import pandas as pd
from catboost import Pool

# Constants (aligned with train_model.py)
MODEL_PATH = "models/model_2stage_ensemble.pkl"
TEST_PATH = "data/processed/test_clean.csv"
CAT_FEATURES_PATH = "data/processed/catboost_categorical.json"
OUTPUT_DIR = "outputs"
SUBMISSION_TEMPLATE = "submission_{timestamp}.csv"


def load_test_data():
    test = pd.read_csv(TEST_PATH, index_col=0)
    print(f"📄 Loaded test data: {test.shape[0]} rows, {test.shape[1]} columns")
    return test


def load_model():
    print(f"📦 Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"✅ Loaded model: {type(model).__name__}")
    return model


def validate_columns(model, test_df):
    if hasattr(model, "feature_names_"):
        expected = list(model.feature_names_)
        actual = list(test_df.columns)
        if expected != actual:
            print("⚠️ WARNING: Model and test data column mismatch")
            print("  → Model expects:", expected)
            print("  → Test provides :", actual)


def predict(model, test_df):
    print("🔍 Running inference...")

    # Load categorical feature names
    with open(CAT_FEATURES_PATH) as f:
        cat_columns = json.load(f)

    # Fix NaNs and enforce string type for categorical columns
    for col in cat_columns:
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna("missing").astype(str)

    # Create Pool for prediction
    test_pool = Pool(data=test_df, cat_features=cat_columns)
    preds = model.predict(test_pool)

    assert len(preds) == len(test_df), "❌ Prediction count mismatch with test data rows!"
    return preds


def save_submission(ids, preds):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = SUBMISSION_TEMPLATE.format(timestamp=timestamp)
    path_timestamped = os.path.join(OUTPUT_DIR, file_name)
    path_latest = os.path.join(OUTPUT_DIR, "submission.csv")

    submission_df = pd.DataFrame({
        "id": ids,
        "efficiency": preds
    })
    submission_df.to_csv(path_timestamped, index=False)
    submission_df.to_csv(path_latest, index=False)

    print(f"✅ Submission saved as:\n  • Latest   → {path_latest}\n  • Timestamped → {path_timestamped}")


def main():
    test_df = load_test_data()
    model = load_model()
    validate_columns(model, test_df)
    preds = predict(model, test_df)
    save_submission(test_df.index, preds)


if __name__ == "__main__":
    main()
