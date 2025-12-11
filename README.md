# E-Commerce ML Pipeline - Business Intelligence & Predictive Analytics

An advanced Machine Learning pipeline for e-commerce businesses that predicts sales, customer churn, provides product recommendations, and generates actionable business insights.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

### 2. Run the Pipeline

```bash
python "# E-COMMERCE ML PIPELINE (synthetic demo.py"
```

## 📊 What This Pipeline Does

### **Predictions:**
- ✅ **Sales Prediction**: Predicts order value based on product features, discounts, delivery time, etc.
- ✅ **Churn Prediction**: Identifies customers at risk of churning
- ✅ **Product Recommendations**: Suggests similar products to customers

### **Business Insights:**
- 💰 Revenue optimization strategies
- 👥 Customer segmentation and retention hints
- 📦 Product performance analysis
- ⚙️ Operational improvements
- 🚀 Growth opportunities
- 🚨 Risk alerts

### **Tracking & Analytics:**
- Model performance metrics (Accuracy, R², ROC-AUC)
- Feature importance analysis
- Business KPIs dashboard
- High-risk customer lists

## 📁 Output Files

All results are saved in `ecom_ml_output/` folder:

1. **Models:**
   - `sales_regression_model.joblib` - Sales prediction model
   - `churn_classification_model.joblib` - Churn prediction model
   - `recommendation_engine.joblib` - Product recommendation engine

2. **Data:**
   - `orders.csv` - All order data
   - `products.csv` - Product catalog
   - `customers_with_predictions.csv` - Customers with churn probabilities

3. **Insights:**
   - `business_insights.json` - All business recommendations
   - `performance_dashboard.json` - KPI dashboard
   - `high_risk_customers.csv` - Customers needing immediate attention
   - `feature_importance_sales.csv` - What drives sales
   - `feature_importance_churn.csv` - What causes churn

## 🔄 Using Your Own Data

To use your real e-commerce data, replace the data generation section (lines 17-52) with:

```python
# Load your data
orders = pd.read_csv("your_orders.csv", parse_dates=["order_date"])
products = pd.read_csv("your_products.csv")
customers = pd.read_csv("your_customers.csv")

# Ensure your data has these columns:
# Orders: order_id, customer_id, product_id, order_date, quantity, order_value, etc.
# Products: product_id, category, base_price, rating
# Customers: customer_id, signup_date, lifetime_value (or similar)
```

## 💡 Key Features Analyzed

### Sales Prediction Features:
- Quantity ordered
- Discount percentage
- Base price
- Product rating
- Delivery delay
- Time of day, day of week, month
- Profit margin

### Churn Prediction Features:
- Customer lifetime value
- Total orders
- Average order value
- Days since last order (recency)
- Total spent
- Average rating given
- Return rate
- Customer age (days since signup)

## 📈 Using the Predictions

### Load a trained model:

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('ecom_ml_output/sales_regression_model.joblib')
scaler = joblib.load('ecom_ml_output/scaler_regression.joblib')

# Make predictions
new_order_features = [[quantity, discount, price, rating, delay, hour, day, month, margin]]
scaled_features = scaler.transform(new_order_features)
predicted_value = model.predict(scaled_features)
# E-Commerce ML Pipeline — Business Intelligence & Predictive Analytics

An end-to-end ML pipeline for e-commerce that trains sales and churn models, builds a product recommender, and generates business insights and a Streamlit dashboard.

This repository now supports:
- Automatic detection of common dataset formats (Superstore-style orders or Amazon product/review exports)
- A `--file` CLI option to explicitly pick any file in `data/`
- Automatic synthesis of orders/customers for product/review-only files (for exploration)
- A Streamlit dashboard (`dashboard.py`) that reads outputs from `ecom_ml_output/`

---

## Quick Start

1. Create and activate the Python 3.12 virtual environment (recommended):

```powershell
python -m venv .venv312
& .\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the pipeline (auto-detects best file in `data/`):

```powershell
$env:PYTHONIOENCODING='utf-8'
python -u .\ecommerce_demo.py
```

Or explicitly run a file in `data/`:

```powershell
python -u .\ecommerce_demo.py --file data\my_supplier.csv
```

4. Start the dashboard (after pipeline finishes):

```powershell
streamlit run dashboard.py --server.port 8501
# open http://localhost:8501
```

---

## What this project produces

All outputs are written to `ecom_ml_output/` and models to `models/`:

- Models
   - `sales_regression_model.joblib`
   - `churn_classification_model.joblib`
   - `recommendation_engine.joblib`
- Data / artifacts
   - `orders.csv`, `products.csv`, `customers_with_predictions.csv`
   - `high_risk_customers.csv`
   - `business_insights.json`, `performance_dashboard.json`
   - `feature_importance_sales.csv`, `feature_importance_churn.csv`

These are used by the Streamlit dashboard for KPIs and visualizations.

---

## Supported Input Types

- Full order datasets (recommended): must contain at least `order_id`, `order_date`, `customer_id`, `product_id`, `quantity`, and `order_value` (or `base_price` + discount columns).
- Product/review exports (e.g., Amazon): the pipeline will detect this format and synthesize orders/customers for exploratory analysis (one order per review, synthetic dates). Useful for demos — not production-grade.

If your file uses unusual column names, use `--file` to point to it, or add a mapping JSON (see Advanced section) for repeatable runs.

---

## Important Notes & Caveats

- Synthetic data: When the pipeline synthesizes orders/customers the churn and lifetime metrics are approximate. Use real order datasets for trustworthy business metrics.
- Degenerate labels: If churn labels are mostly one class (e.g., 100% churn), accuracy can be misleading — check class counts and ROC-AUC.
- Mapping: `map_columns()` handles many common synonyms (e.g., `discounted_price`, `actual_price`, `qty`, `units`, `user_id`). If mapping fails, provide a mapping JSON.

---

## Examples: Using the trained models

Load the sales model and make a prediction:

```python
import joblib
import numpy as np

model = joblib.load('ecom_ml_output/sales_regression_model.joblib')
scaler = joblib.load('models/sales_scaler.pkl')

# Example feature vector: [quantity, discount_pct, base_price, rating, delivery_delay, hour_of_day, day_of_week, month, profit_margin]
features = np.array([[1, 10.0, 1999.0, 4.2, 0, 14, 2, 11, 20.0]])
scaled = scaler.transform(features)
pred = model.predict(scaled)
print(f"Predicted order value: ₹{pred[0]:.2f}")
```

Get churn probability for a customer:

```python
clf = joblib.load('ecom_ml_output/churn_classification_model.joblib')
scaler_clf = joblib.load('models/churn_scaler.pkl')
features = [[0.0, 5, 2000.0, 45]]  # example: [lifetime_value, total_orders, avg_order_value, recency]
scaled = scaler_clf.transform(features)
prob = clf.predict_proba(scaled)[0][1]
print(f"Churn probability: {prob:.2%}")
```

---

## Advanced: per-supplier mapping (recommended for many suppliers)

Create a JSON file under `mappings/` named after the supplier (e.g. `mappings/my_supplier.json`) with keys mapping standard fields to possible source names. Example:

```json
{
   "order_id": ["order_id","txn_id"],
   "order_date": ["order_date","txn_date","date"],
   "customer_id": ["customer_id","user_id","buyer_id"],
   "product_id": ["product_id","sku"],
   "quantity": ["quantity","qty","units"],
   "order_value": ["order_value","amount","total","discounted_price"]
}
```

The pipeline will attempt to apply a mapping automatically if you add this feature (ask me to enable mapping JSON auto-loader).

---

Happy analyzing! 🚀

