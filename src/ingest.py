import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_CSV_PATH = Path("data/telco_churn_raw.csv")
DB_PATH = Path("data/churn.db")


def download_and_clean() -> pd.DataFrame:
    print("Downloading dataset...")
    df = pd.read_csv(DATA_URL)
    df.to_csv(RAW_CSV_PATH, index=False)

    # 1. Clean TotalCharges: spaces exist when tenure == 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # 2. Standardize target column to binary 0/1
    df["ChurnFlag"] = (df["Churn"].str.strip() == "Yes").astype(int)

    # 3. Feature Engineering: domain-driven signals
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["ActiveServicesCount"] = (df[service_cols] == "Yes").sum(axis=1)

    # Ratio of monthly bill to average historical monthly spend
    avg_spend = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"])
    df["ChargeToAvgRatio"] = np.round(df["MonthlyCharges"] / np.maximum(avg_spend, 1.0), 3)

    # Flag high-risk onboarding window (first 6 months)
    df["IsNewCustomer"] = (df["tenure"] <= 6).astype(int)

    return df


def save_to_sqlite(df: pd.DataFrame) -> None:
    print(f"Writing {len(df)} cleaned rows to {DB_PATH}...")
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("customers", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_churn ON customers(ChurnFlag);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_contract ON customers(Contract);")
    print("Database ingestion complete.")


if __name__ == "__main__":
    cleaned_df = download_and_clean()
    save_to_sqlite(cleaned_df)
