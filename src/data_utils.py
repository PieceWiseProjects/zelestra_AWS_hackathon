import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Columns to process
NUM_COLS = [
    "temperature", "irradiance", "panel_age", "maintenance_count", "soiling_ratio",
    "voltage", "current", "module_temperature", "cloud_coverage"
]

CAT_COLS = ["string_id", "error_code", "installation_type"]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def convert_object_numerics(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["humidity", "wind_speed", "pressure"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_preprocessor():
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, NUM_COLS),
        ("cat", cat_pipeline, CAT_COLS)
    ], remainder="drop")

    return preprocessor


def split_target(df: pd.DataFrame):
    return df.drop(columns=["efficiency"]), df["efficiency"]
