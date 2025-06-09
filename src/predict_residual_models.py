import os
import json
import joblib
import pandas as pd
from catboost import Pool

# Paths
TEST_PATH = "data/processed/test_clean.csv"
CAT_PATH = "data/processed/catboost_categorical.json"
ENSEMBLE_MODEL_PATH = "models/model_2stage_ensemble.pkl"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "submission_residual_ensemble.csv")


def load_test_data():
    test_df = pd.read_csv(TEST_PATH, index_col=0)
    with open(CAT_PATH) as f:
        cat_cols = json.load(f)
    cat_cols = [col for col in cat_cols if col in test_df.columns]
    for col in cat_cols:
        test_df[col] = test_df[col].fillna("missing").astype(str)
    return test_df, cat_cols


def predict_ensemble(model_1, model_2, test_df, cat_cols):
    pool = Pool(test_df, cat_features=cat_cols)
    pred1 = model_1.predict(pool)
    pred2 = model_2.predict(pool)
    return pred1 + pred2


def save_submission(ids, predictions):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    submission = pd.DataFrame({
        "id": ids,
        "efficiency": predictions
    })
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Submission saved to {OUTPUT_FILE}")


def main():
    print("📦 Loading 2-stage ensemble model...")
    model_1, model_2 = joblib.load(ENSEMBLE_MODEL_PATH)

    print("📄 Loading test data...")
    test_df, cat_cols = load_test_data()

    print("🔍 Running inference with residual ensemble...")
    preds = predict_ensemble(model_1, model_2, test_df, cat_cols)

    save_submission(test_df.index, preds)


if __name__ == "__main__":
    main()
