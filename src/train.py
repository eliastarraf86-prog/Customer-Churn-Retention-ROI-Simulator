import json
import sqlite3
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DB_PATH = Path("data/churn.db")
MODELS_DIR = Path("models")


def load_data() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM customers", conn)


def main():
    df = load_data()

    drop_cols = ["customerID", "Churn", "ChurnFlag"]
    X = df.drop(columns=drop_cols)
    y = df["ChurnFlag"]

    num_cols = [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges",
        "ActiveServicesCount", "ChargeToAvgRatio", "IsNewCustomer"
    ]
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_name = None
    best_score = -1.0
    best_pipeline = None

    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", clf)])
        cv_roc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        cv_pr = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision").mean()

        pipe.fit(X_train, y_train)
        test_probs = pipe.predict_proba(X_test)[:, 1]
        test_roc = roc_auc_score(y_test, test_probs)
        test_pr = average_precision_score(y_test, test_probs)

        results[name] = {
            "cv_roc_auc": round(float(cv_roc), 4),
            "cv_pr_auc": round(float(cv_pr), 4),
            "test_roc_auc": round(float(test_roc), 4),
            "test_pr_auc": round(float(test_pr), 4),
        }
        print(f"{name}: CV ROC-AUC={cv_roc:.4f} | Test ROC-AUC={test_roc:.4f} | Test PR-AUC={test_pr:.4f}")

        if cv_pr > best_score:
            best_score = cv_pr
            best_name = name
            best_pipeline = pipe

    joblib.dump(best_pipeline, MODELS_DIR / "churn_pipeline.joblib")

    test_eval = X_test.copy()
    test_eval["customerID"] = df.loc[X_test.index, "customerID"]
    test_eval["ActualChurn"] = y_test.values
    test_eval["ChurnProbability"] = best_pipeline.predict_proba(X_test)[:, 1]
    test_eval.to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    summary = {"selected_model": best_name, "metrics": results}
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved best model ({best_name}) and test predictions to models/")


if __name__ == "__main__":
    main()
