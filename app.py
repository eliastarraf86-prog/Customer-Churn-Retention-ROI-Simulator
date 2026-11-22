import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Customer Churn & Retention Simulator", layout="wide")

@st.cache_data
def load_artifacts():
    eval_df = pd.read_csv("models/test_predictions.csv")
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    return eval_df, metrics

@st.cache_resource
def load_model():
    return joblib.load("models/churn_pipeline.joblib")

eval_df, metrics = load_artifacts()
pipeline = load_model()

st.title("Customer Churn & Retention ROI Simulator")
st.caption("Threshold optimization and cost-sensitive retention modeling on unseen test customers (n = 1,409)")

# Sidebar: Business Assumptions
st.sidebar.header("Retention Economics")
threshold = st.sidebar.slider("Churn Alert Threshold", 0.05, 0.90, 0.35, 0.01)
offer_cost = st.sidebar.number_input("Retention Offer Cost ($ / customer)", min_value=5, max_value=500, value=30, step=5)
save_rate = st.sidebar.slider("Offer Acceptance / Save Rate (%)", 5, 90, 35, 5) / 100.0
ltv_value = st.sidebar.number_input("Retained Customer LTV ($)", min_value=100, max_value=5000, value=600, step=50)

def compute_financials(df: pd.DataFrame, thresh: float):
    targeted = df["ChurnProbability"] >= thresh
    actual_churn = df["ActualChurn"] == 1

    tp = int((targeted & actual_churn).sum())
    fp = int((targeted & ~actual_churn).sum())
    fn = int((~targeted & actual_churn).sum())

    total_targeted = tp + fp
    campaign_cost = total_targeted * offer_cost
    saved_customers = tp * save_rate
    gross_saved_value = saved_customers * ltv_value
    net_savings = gross_saved_value - campaign_cost

    precision = tp / total_targeted if total_targeted > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "targeted": total_targeted,
        "campaign_cost": campaign_cost,
        "saved_customers": saved_customers,
        "net_savings": net_savings,
        "precision": precision,
        "recall": recall,
    }

current = compute_financials(eval_df, threshold)

thresholds = np.linspace(0.05, 0.90, 86)
sweep = [compute_financials(eval_df, t) for t in thresholds]
net_curve = [s["net_savings"] for s in sweep]
best_idx = int(np.argmax(net_curve))
best_thresh = float(thresholds[best_idx])
best_savings = net_curve[best_idx]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Net Financial Impact", f"${current['net_savings']:,.0f}", f"Optimal: ${best_savings:,.0f} @ {best_thresh:.2f}")
c2.metric("Customers Targeted", f"{current['targeted']:,}", f"Precision: {current['precision']:.1%}")
c3.metric("Churners Caught (Recall)", f"{current['tp']} / {current['tp'] + current['fn']}", f"Recall: {current['recall']:.1%}")
c4.metric("Campaign Spend", f"${current['campaign_cost']:,.0f}", f"Est. Saved: {current['saved_customers']:.1f} users")

fig = go.Figure()
fig.add_trace(go.Scatter(x=thresholds, y=net_curve, mode="lines", name="Net Savings ($)"))
fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"Selected ({threshold:.2f})")
fig.add_vline(x=best_thresh, line_dash="dot", line_color="green", annotation_text=f"Optimal ({best_thresh:.2f})")
fig.update_layout(
    title="Net Retention Profit vs. Probability Threshold",
    xaxis_title="Probability Threshold",
    yaxis_title="Net Profit ($)",
    height=380,
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Single-Customer Risk & Action Simulator")
col_a, col_b, col_c = st.columns(3)
tenure = col_a.slider("Tenure (Months)", 0, 72, 4)
monthly_charges = col_b.slider("Monthly Charges ($)", 18.0, 120.0, 85.0)
contract = col_c.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

col_d, col_e, col_f = st.columns(3)
internet = col_d.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
tech_support = col_e.selectbox("Tech Support", ["No", "Yes", "No internet service"])
payment = col_f.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

sample = eval_df.drop(columns=["customerID", "ActualChurn", "ChurnProbability"]).iloc[[0]].copy()
sample["tenure"] = tenure
sample["MonthlyCharges"] = monthly_charges
sample["TotalCharges"] = tenure * monthly_charges
sample["Contract"] = contract
sample["InternetService"] = internet
sample["TechSupport"] = tech_support
sample["PaymentMethod"] = payment
sample["IsNewCustomer"] = int(tenure <= 6)
sample["ChargeToAvgRatio"] = 1.0

prob = float(pipeline.predict_proba(sample)[0, 1])
action = "TRIGGER RETENTION OFFER" if prob >= threshold else "NO ACTION NEEDED"
st.info(f"**Predicted Churn Probability:** `{prob:.1%}`  |  **Decision at Threshold {threshold:.2f}:** **{action}**")
