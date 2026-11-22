const els = {
  threshold: document.getElementById("threshold"),
  thresholdVal: document.getElementById("thresholdVal"),
  offerCost: document.getElementById("offerCost"),
  saveRate: document.getElementById("saveRate"),
  saveRateVal: document.getElementById("saveRateVal"),
  ltvValue: document.getElementById("ltvValue"),
  optimalPill: document.getElementById("optimalPill"),
  netSavings: document.getElementById("netSavings"),
  targetedCount: document.getElementById("targetedCount"),
  precisionSub: document.getElementById("precisionSub"),
  caughtCount: document.getElementById("caughtCount"),
  recallSub: document.getElementById("recallSub"),
  campaignSpend: document.getElementById("campaignSpend"),
  savedUsersSub: document.getElementById("savedUsersSub"),
  canvas: document.getElementById("roiChart"),
  custTenure: document.getElementById("custTenure"),
  custTenureVal: document.getElementById("custTenureVal"),
  custCharges: document.getElementById("custCharges"),
  custContract: document.getElementById("custContract"),
  custInternet: document.getElementById("custInternet"),
  custSupport: document.getElementById("custSupport"),
  custPayment: document.getElementById("custPayment"),
  customerResult: document.getElementById("customerResult"),
};

function fmtMoney(num) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(num);
}

function drawCurve(curve, selectedThresh, optimalThresh) {
  const ctx = els.canvas.getContext("2d");
  const w = els.canvas.width;
  const h = els.canvas.height;
  ctx.clearRect(0, 0, w, h);

  const pad = { top: 20, right: 24, bottom: 36, left: 68 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const vals = curve.map((d) => d.net_savings);
  const minVal = Math.min(0, ...vals);
  const maxVal = Math.max(1000, ...vals);
  const span = maxVal - minVal || 1;

  const xForIdx = (i) => pad.left + (i / Math.max(curve.length - 1, 1)) * plotW;
  const yForVal = (v) => pad.top + plotH - ((v - minVal) / span) * plotH;

  ctx.strokeStyle = "#1e293b";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#94a3b8";
  ctx.font = "12px Segoe UI, sans-serif";

  for (let g = 0; g <= 4; g++) {
    const yVal = minVal + (g / 4) * span;
    const yPos = yForVal(yVal);
    ctx.beginPath();
    ctx.moveTo(pad.left, yPos);
    ctx.lineTo(w - pad.right, yPos);
    ctx.stroke();
    ctx.fillText(fmtMoney(yVal), 8, yPos + 4);
  }

  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  curve.forEach((pt, idx) => {
    const x = xForIdx(idx);
    const y = yForVal(pt.net_savings);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  const drawMarker = (thresh, color) => {
    const idx = curve.findIndex((d) => d.threshold >= thresh);
    if (idx >= 0) {
      const x = xForIdx(idx);
      ctx.strokeStyle = color;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, h - pad.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  };

  drawMarker(optimalThresh, "#34d399");
  drawMarker(selectedThresh, "#f8fafc");

  ctx.fillStyle = "#94a3b8";
  ctx.fillText("Threshold 0.05", pad.left, h - 10);
  ctx.fillText("Threshold 0.90", w - pad.right - 82, h - 10);
}

async function refreshSimulation() {
  const threshold = parseFloat(els.threshold.value);
  const offerCost = parseFloat(els.offerCost.value) || 30;
  const saveRatePct = parseFloat(els.saveRate.value);
  const ltvValue = parseFloat(els.ltvValue.value) || 600;

  els.thresholdVal.textContent = threshold.toFixed(2);
  els.saveRateVal.textContent = saveRatePct + "%";

  const res = await fetch("/api/simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      threshold,
      offer_cost: offerCost,
      save_rate_pct: saveRatePct,
      ltv_value: ltvValue,
    }),
  });
  const data = await res.json();
  const c = data.current;
  const opt = data.optimal;

  els.optimalPill.textContent = "Optimal: " + fmtMoney(opt.net_savings) + " @ " + opt.threshold.toFixed(2);
  els.netSavings.textContent = fmtMoney(c.net_savings);
  els.targetedCount.textContent = c.targeted.toLocaleString();
  els.precisionSub.textContent = "Precision: " + c.precision + "%";
  els.caughtCount.textContent = c.tp + " / " + (c.tp + c.fn);
  els.recallSub.textContent = "Recall: " + c.recall + "%";
  els.campaignSpend.textContent = fmtMoney(c.campaign_cost);
  els.savedUsersSub.textContent = "Est. Saved: " + c.saved_customers + " users";

  drawCurve(data.curve, threshold, opt.threshold);
  refreshCustomer();
}

async function refreshCustomer() {
  const tenure = parseInt(els.custTenure.value, 10);
  els.custTenureVal.textContent = tenure;

  const res = await fetch("/api/predict-customer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      tenure,
      monthly_charges: parseFloat(els.custCharges.value) || 85,
      contract: els.custContract.value,
      internet_service: els.custInternet.value,
      tech_support: els.custSupport.value,
      payment_method: els.custPayment.value,
      threshold: parseFloat(els.threshold.value),
    }),
  });
  const pred = await res.json();
  els.customerResult.className = "decision-banner " + (pred.trigger_offer ? "offer" : "no-offer");
  els.customerResult.textContent =
    "Predicted Churn Probability: " + pred.churn_probability + "%  |  Decision: " + pred.action;
}

[
  els.threshold,
  els.offerCost,
  els.saveRate,
  els.ltvValue,
  els.custTenure,
  els.custCharges,
  els.custContract,
  els.custInternet,
  els.custSupport,
  els.custPayment,
].forEach((input) => input.addEventListener("input", refreshSimulation));

refreshSimulation();