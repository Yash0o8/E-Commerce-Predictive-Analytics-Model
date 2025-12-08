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
print(f"Predicted order value: ₹{predicted_value[0]:.2f}")
```

### Get churn predictions:

```python
churn_model = joblib.load('ecom_ml_output/churn_classification_model.joblib')
churn_scaler = joblib.load('ecom_ml_output/scaler_classification.joblib')

# Predict churn probability
customer_features = [[lifetime_value, total_orders, avg_order_value, recency, ...]]
scaled = churn_scaler.transform(customer_features)
churn_prob = churn_model.predict_proba(scaled)[0][1]
print(f"Churn probability: {churn_prob:.2%}")
```

## 🎯 Action Items After Running

1. **Review high_risk_customers.csv** - Launch retention campaigns
2. **Check business_insights.json** - Implement recommended strategies
3. **Monitor feature importance** - Understand what drives your business
4. **Update models monthly** - Retrain with new data for accuracy
5. **A/B test recommendations** - Validate pricing and discount strategies

## 📞 Need Help?

The script will output:
- ✅ Model performance metrics
- ✅ Business KPIs
- ✅ Actionable insights with hints
- ✅ Risk alerts
- ✅ All saved files location

## 🔄 Maintenance

- **Weekly**: Review high-risk customers
- **Monthly**: Retrain models with new data
- **Quarterly**: Review feature importance changes
- **Ongoing**: Implement and test recommended strategies

---

**Happy analyzing! 🚀**
