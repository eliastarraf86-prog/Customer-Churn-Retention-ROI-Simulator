from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title='Churn & Retention ROI API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

EVAL_DF = pd.read_csv('models/test_predictions.csv')
PIPELINE = joblib.load('models/churn_pipeline.joblib')
SAMPLE_TEMPLATE = EVAL_DF.drop(columns=['customerID', 'ActualChurn', 'ChurnProbability']).iloc[[0]].copy()


class SimulationRequest(BaseModel):
    threshold: float = 0.35
    offer_cost: float = 30.0
    save_rate_pct: float = 35.0
    ltv_value: float = 600.0


class CustomerPredictionRequest(BaseModel):
    tenure: int = 4
    monthly_charges: float = 85.0
    contract: str = 'Month-to-month'
    internet_service: str = 'Fiber optic'
    tech_support: str = 'No'
    payment_method: str = 'Electronic check'
    threshold: float = 0.35


def calc_metrics(df: pd.DataFrame, thresh: float, offer_cost: float, save_rate: float, ltv: float) -> dict:
    targeted = df['ChurnProbability'] >= thresh
    actual_churn = df['ActualChurn'] == 1

    tp = int((targeted & actual_churn).sum())
    fp = int((targeted & ~actual_churn).sum())
    fn = int((~targeted & actual_churn).sum())

    total_targeted = tp + fp
    campaign_cost = total_targeted * offer_cost
    saved_customers = tp * save_rate
    net_savings = (saved_customers * ltv) - campaign_cost

    precision = (tp / total_targeted) if total_targeted > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        'threshold': round(float(thresh), 2),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'targeted': total_targeted,
        'campaign_cost': round(float(campaign_cost), 2),
        'saved_customers': round(float(saved_customers), 1),
        'net_savings': round(float(net_savings), 2),
        'precision': round(float(precision * 100), 1),
        'recall': round(float(recall * 100), 1),
    }


@app.post('/api/simulate')
def simulate(req: SimulationRequest):
    save_rate = req.save_rate_pct / 100.0
    current = calc_metrics(EVAL_DF, req.threshold, req.offer_cost, save_rate, req.ltv_value)

    thresholds = np.round(np.linspace(0.05, 0.90, 86), 2)
    curve = [calc_metrics(EVAL_DF, float(t), req.offer_cost, save_rate, req.ltv_value) for t in thresholds]
    best = max(curve, key=lambda item: item['net_savings'])

    return {
        'current': current,
        'optimal': best,
        'curve': curve,
    }


@app.post('/api/predict-customer')
def predict_customer(req: CustomerPredictionRequest):
    row = SAMPLE_TEMPLATE.copy()
    row['tenure'] = req.tenure
    row['MonthlyCharges'] = req.monthly_charges
    row['TotalCharges'] = req.tenure * req.monthly_charges
    row['Contract'] = req.contract
    row['InternetService'] = req.internet_service
    row['TechSupport'] = req.tech_support
    row['PaymentMethod'] = req.payment_method
    row['IsNewCustomer'] = int(req.tenure <= 6)
    row['ChargeToAvgRatio'] = 1.0

    prob = float(PIPELINE.predict_proba(row)[0, 1])
    trigger = prob >= req.threshold
    return {
        'churn_probability': round(prob * 100, 1),
        'trigger_offer': trigger,
        'action': 'TRIGGER RETENTION OFFER' if trigger else 'NO ACTION NEEDED',
    }


frontend_dir = Path('frontend')
if frontend_dir.exists():
    app.mount('/', StaticFiles(directory='frontend', html=True), name='frontend')
