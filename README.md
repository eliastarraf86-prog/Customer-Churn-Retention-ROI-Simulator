# Customer Churn & Retention ROI Simulator

End-to-end customer churn prediction pipeline and cost-sensitive retention decision simulator built with Python, SQLite, scikit-learn, FastAPI, and vanilla HTML/CSS/JS.

## Results Summary
- **Dataset & ETL (`src/ingest.py`, `sql/analysis.sql`):** Cleans 7,043 Telco customer records, resolves `TotalCharges` on zero-tenure accounts, engineers `ActiveServicesCount`, `ChargeToAvgRatio`, and `IsNewCustomer`, and loads indexed tables into SQLite (`data/churn.db`).
- **Model Selection (`src/train.py`):** Compares Logistic Regression and Histogram-Based Gradient Boosting via 5-fold stratified cross-validation inside a leak-free `ColumnTransformer` pipeline. Logistic Regression achieved **0.8457 Test ROC-AUC** and **0.6524 Test PR-AUC**.
- **Retention Simulator (`api.py` + `frontend/` & `app.py`):** Evaluates probability thresholds against retention offer cost, acceptance rate, and customer LTV on unseen test data (`n = 1,409`), surfacing optimal threshold ROI and single-customer risk scoring.

## Quickstart
1. `python -m venv venv`
2. `.\venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. `python src/ingest.py`
5. `python src/train.py`
6. `uvicorn api:app --reload` (open `http://127.0.0.1:8000`)
