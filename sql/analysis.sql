-- 1. Overall Churn Rate & Revenue at Risk
SELECT 
    COUNT(*) AS total_customers,
    SUM(ChurnFlag) AS churned_customers,
    ROUND(AVG(ChurnFlag) * 100, 2) AS churn_rate_pct,
    ROUND(SUM(CASE WHEN ChurnFlag = 1 THEN MonthlyCharges ELSE 0 END), 2) AS monthly_revenue_lost
FROM customers;

-- 2. Churn Rate by Contract Type
SELECT 
    Contract,
    COUNT(*) AS customer_count,
    ROUND(AVG(ChurnFlag) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_bill
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 3. New Customers (<= 6 months) vs. Tenured Customers
SELECT 
    IsNewCustomer,
    COUNT(*) AS customers,
    ROUND(AVG(ChurnFlag) * 100, 2) AS churn_rate_pct
FROM customers
GROUP BY IsNewCustomer;
