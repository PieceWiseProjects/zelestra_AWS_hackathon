import os
import pandas as pd
from ydata_profiling import ProfileReport

def generate_report(csv_path, output_html):
    df = pd.read_csv(csv_path, index_col=0)
    profile = ProfileReport(df, title=f"EDA Report for {csv_path}", explorative=True)
    profile.to_file(output_html)
    print(f"✅ EDA report saved to: {output_html}")

def generate_all():
    os.makedirs("outputs", exist_ok=True)

    generate_report("data/processed/train_clean.csv", "outputs/eda_report_train.html")
    generate_report("data/processed/test_clean.csv", "outputs/eda_report_test.html")

if __name__ == "__main__":
    generate_all()
