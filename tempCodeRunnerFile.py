# E-COMMERCE ML PIPELINE - Advanced Predictive Analytics & Business Intelligence
# Requirements: pandas, numpy, scikit-learn, joblib, matplotlib, seaborn (optional for visualization)
# Install dependencies: pip install pandas numpy scikit-learn joblib matplotlib seaborn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.neighbors import NearestNeighbors
import joblib
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("E-COMMERCE ML PIPELINE - Predictive Analytics & Business Intelligence")
print("="*80)

# ========== CONFIGURATION: USE YOUR OWN DATA ==========
# Set this to True to use your own data files, False to use synthetic data
USE_YOUR_DATA = True  # ⬅️ CHANGED TO True - Using your data!

# File paths for your data (only used if USE_YOUR_DATA = True)
DATA_FOLDER = "data"  # Folder containing your data files (or "." for current folder)
ORDERS_FILE = "superstore.csv"  # Your single file with all data
PRODUCTS_FILE = "superstore.csv"  # Same file (all data in one file)
CUSTOMERS_FILE = "superstore.csv"  # Same file (all data in one file)

# Excel sheet names (only used if files are Excel format)
# If your Excel files have multiple sheets, specify the sheet name here (None = first sheet)
ORDERS_SHEET = None  # Sheet name for orders, or None for first sheet
PRODUCTS_SHEET = None  # Sheet name for products, or None for first sheet
CUSTOMERS_SHEET = None  # Sheet name for customers, or None for first sheet

# Reference date for calculating churn (usually today's date or last known date)
REFERENCE_DATE = datetime(2024, 12, 1)  # ⬅️ CHANGE THIS to your reference date

print(f"\n📊 Data Source: {'YOUR OWN DATA' if USE_YOUR_DATA else 'SYNTHETIC DATA'}")

# ========== HELPER FUNCTION: Map column names automatically ==========
def map_columns(df, mapping_dict):
    """Map column names from various formats to standard format"""
    df_mapped = df.copy()
    
    for standard_name, possible_names in mapping_dict.items():
        # Check if standard name already exists
        if standard_name in df_mapped.columns:
            continue
        
        # Try to find matching column
        for col in df_mapped.columns:
            col_str = str(col).strip()
            # Check exact match or partial match
            for possible in possible_names:
                if possible.lower() in col_str.lower() or col_str.lower() in possible.lower():
                    df_mapped.rename(columns={col: standard_name}, inplace=True)
                    print(f"   ✓ Mapped '{col}' → '{standard_name}'")
                    break
    
    return df_mapped

# ========== HELPER FUNCTION: Load CSV or Excel files ==========
def load_data_file(file_path, date_columns=None, sheet_name=None, column_mapping=None):
    """
    Automatically detect and load CSV or Excel files
    Supports: .csv, .xlsx, .xls files
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext in ['.xlsx', '.xls']:
            # Load Excel file
            print(f"   📊 Loading Excel file: {os.path.basename(file_path)}")
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path, sheet_name=0)  # First sheet
            # Parse date columns if specified
            if date_columns:
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
        elif file_ext == '.csv':
            # Load CSV file
            print(f"   📄 Loading CSV file: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path, parse_dates=date_columns if date_columns else None)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .csv, .xlsx, .xls")
        
        # Apply column mapping if provided
        if column_mapping:
            df = map_columns(df, column_mapping)
            # Re-parse dates after mapping
            if date_columns:
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    except Exception as e:
        raise Exception(f"Error loading {file_path}: {str(e)}")

# ========== 1) DATA LOADING ==========
if USE_YOUR_DATA:
    print(f"\n[1/8] Loading your data from {DATA_FOLDER}/ folder...")
    try:
        # Construct file paths
        orders_path = os.path.join(DATA_FOLDER, ORDERS_FILE)
        products_path = os.path.join(DATA_FOLDER, PRODUCTS_FILE)
        customers_path = os.path.join(DATA_FOLDER, CUSTOMERS_FILE)
        
        # Column mappings for common e-commerce file formats
        orders_column_mapping = {
            'order_id': ['Order.ID', 'Order ID', 'order_id', 'Order_ID'],
            'order_date': ['Order.Date', 'Order Date', 'Date', 'order_date', 'Order_Date'],
            'customer_id': ['Customer.ID', 'Customer ID', 'Customer.', 'customer_id', 'Customer_ID'],
            'product_id': ['Product.ID', 'Product ID', 'product_id', 'Product_ID'],
            'quantity': ['Quantity', 'quantity'],
            'order_value': ['Sales', 'sales', 'Order Value', 'order_value'],
            'discount_pct': ['Discount', 'discount'],
            'ship_date': ['Ship.Date', 'Ship Date', 'ship_date'],
        }
        
        products_column_mapping = {
            'product_id': ['Product.ID', 'Product ID', 'product_id', 'Product_ID'],
            'category': ['Category', 'category'],
            'base_price': ['Sales', 'sales', 'Price', 'price'],
            'rating': ['Rating', 'rating'],
        }
        
        customers_column_mapping = {
            'customer_id': ['Customer.ID', 'Customer ID', 'Customer.', 'customer_id', 'Customer_ID'],
            'city': ['City', 'city'],
            'signup_date': ['Order.Date', 'Order Date', 'Date', 'First Order'],
            'lifetime_value': ['Sales', 'sales'],
        }
        
        # Load data files (automatically detects CSV or Excel and maps columns)
        orders = load_data_file(orders_path, date_columns=["order_date"], sheet_name=ORDERS_SHEET, column_mapping=orders_column_mapping)
        products = load_data_file(products_path, sheet_name=PRODUCTS_SHEET, column_mapping=products_column_mapping)
        customers = load_data_file(customers_path, date_columns=["signup_date"], sheet_name=CUSTOMERS_SHEET, column_mapping=customers_column_mapping)
        
        print(f"✓ Loaded {len(orders)} orders, {len(products)} products, {len(customers)} customers")
        
        # Data validation and cleaning
        print("   Validating and cleaning data...")
        
        # Ensure required columns exist (create missing ones with defaults)
        if "stock_quantity" not in products.columns:
            products["stock_quantity"] = 100
            print("   ⚠️  Added missing 'stock_quantity' column with default value 100")
        
        if "manufacturing_cost" not in products.columns:
            products["manufacturing_cost"] = products["base_price"] * 0.5
            print("   ⚠️  Added missing 'manufacturing_cost' column (estimated as 50% of base_price)")
        
        if "city" not in customers.columns:
            customers["city"] = "Unknown"
            print("   ⚠️  Added missing 'city' column")
        
        if "age_group" not in customers.columns:
            customers["age_group"] = "Unknown"
            print("   ⚠️  Added missing 'age_group' column")
        
        # Merge product info into orders if not already merged
        if "base_price" not in orders.columns:
            orders = orders.merge(products[["product_id", "base_price", "category", "rating"]], 
                                on="product_id", how="left")
        
        # Calculate missing order fields
        if "price_after_discount" not in orders.columns:
            if "discount_pct" not in orders.columns:
                orders["discount_pct"] = 0
            orders["price_after_discount"] = orders["base_price"] * (1 - orders["discount_pct"]/100)
        
        if "order_value" not in orders.columns:
            orders["order_value"] = orders["price_after_discount"] * orders["quantity"]
        
        if "delivery_time_days" not in orders.columns:
            orders["delivery_time_days"] = 3  # Default 3 days
            print("   ⚠️  Added missing 'delivery_time_days' column with default value 3")
        
        if "delivery_delay" not in orders.columns:
            orders["delivery_delay"] = (orders["delivery_time_days"] - 2).clip(lower=0)
        
        if "payment_method" not in orders.columns:
            orders["payment_method"] = "card"
            print("   ⚠️  Added missing 'payment_method' column")
        
        if "failed_payment_attempts" not in orders.columns:
            orders["failed_payment_attempts"] = 0
            print("   ⚠️  Added missing 'failed_payment_attempts' column")
        
        if "is_returned" not in orders.columns:
            orders["is_returned"] = 0
            print("   ⚠️  Added missing 'is_returned' column")
        
        if "customer_rating" not in orders.columns:
            orders["customer_rating"] = 4  # Default rating
            print("   ⚠️  Added missing 'customer_rating' column with default value 4")
        
        # Extract date features
        orders["hour_of_day"] = pd.to_datetime(orders["order_date"]).dt.hour
        orders["day_of_week"] = pd.to_datetime(orders["order_date"]).dt.dayofweek
        orders["month"] = pd.to_datetime(orders["order_date"]).dt.month
        
        # Calculate profit margin if possible
        if "manufacturing_cost" in products.columns:
            products_merged = products[["product_id", "manufacturing_cost"]].merge(
                orders[["product_id", "price_after_discount"]], on="product_id", how="right"
            )
            orders["profit_margin"] = np.round(
                (orders["price_after_discount"] - products_merged["manufacturing_cost"]) / orders["price_after_discount"] * 100, 
                2
            )
        else:
            orders["profit_margin"] = np.round((orders["price_after_discount"] - orders["base_price"] * 0.6) / orders["price_after_discount"] * 100, 2)
        
        # Set reference date from user config
        reference_date = REFERENCE_DATE
        
        print(f"✓ Data validation complete!")
        print(f"   Orders date range: {orders['order_date'].min()} to {orders['order_date'].max()}")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find data file: {e}")
        print(f"\n📋 Please ensure you have the following files in '{DATA_FOLDER}/' folder:")
        print(f"   - {ORDERS_FILE}")
        print(f"   - {PRODUCTS_FILE}")
        print(f"   - {CUSTOMERS_FILE}")
        print(f"\n💡 Check DATA_PREPARATION_GUIDE.md for the required format.")
        print(f"   Or set USE_YOUR_DATA = False to use synthetic data for testing.")
        raise
    
    except Exception as e:
        print(f"\n❌ ERROR loading data: {e}")
        print(f"\n💡 Please check:")
        print(f"   1. CSV files are in the correct format")
        print(f"   2. Required columns are present (see DATA_PREPARATION_GUIDE.md)")
        print(f"   3. Date columns are in YYYY-MM-DD format")
        raise

else:
    # ========== GENERATE SYNTHETIC DATA ==========
    print("\n[1/8] Generating synthetic e-commerce data...")
    reference_date = REFERENCE_DATE
    
    n_products = 200
    products = pd.DataFrame({
        "product_id": [f"P{1000+i}" for i in range(n_products)],
        "category": np.random.choice(["Electronics","Clothing","Home","Beauty","Toys","Sports"], size=n_products),
        "base_price": np.round(np.random.uniform(100, 20000, size=n_products), 2),
        "rating": np.round(np.random.uniform(2.5, 5.0, size=n_products), 2),
        "stock_quantity": np.random.randint(10, 500, size=n_products),
        "manufacturing_cost": np.round(np.random.uniform(50, 10000, size=n_products), 2)
    })
    
    n_customers = 2000
    customers = pd.DataFrame({
        "customer_id": [f"C{20000+i}" for i in range(n_customers)],
        "city": np.random.choice(["Mumbai","Delhi","Bengaluru","Pune","Hyderabad","Chennai","Kolkata"], size=n_customers),
        "signup_date": [(datetime(2020,1,1) + timedelta(days=int(np.random.exponential(400)))).date() for _ in range(n_customers)],
        "lifetime_value": np.round(np.random.exponential(3000, size=n_customers), 2),
        "age_group": np.random.choice(["18-25","26-35","36-45","46-55","55+"], size=n_customers, p=[0.15,0.35,0.25,0.15,0.10])
    })
    customers["is_high_value"] = (customers["lifetime_value"] > customers["lifetime_value"].quantile(0.85)).astype(int)
    
    n_orders = 12000
    order_dates = [datetime(2023,1,1) + timedelta(days=int(np.random.exponential(400))) for _ in range(n_orders)]
    orders = pd.DataFrame({
        "order_id": [f"O{300000+i}" for i in range(n_orders)],
        "customer_id": np.random.choice(customers["customer_id"], size=n_orders),
        "product_id": np.random.choice(products["product_id"], size=n_orders),
        "order_date": order_dates,
        "quantity": np.random.randint(1, 5, size=n_orders),
        "discount_pct": np.round(np.random.choice([0,5,10,15,20,30], size=n_orders, p=[0.4,0.2,0.15,0.1,0.1,0.05]), 2)
    })
    
    # Merge product info into orders
    orders = orders.merge(products[["product_id","base_price","category","rating"]], on="product_id", how="left")
    orders["price_after_discount"] = np.round(orders["base_price"] * (1 - orders["discount_pct"]/100), 2)
    orders["order_value"] = np.round(orders["price_after_discount"] * orders["quantity"], 2)
    orders["delivery_time_days"] = np.maximum(0, np.random.normal(loc=3, scale=2, size=n_orders).astype(int))
    orders["delivery_delay"] = (orders["delivery_time_days"] - 2).clip(lower=0)
    orders["payment_method"] = np.random.choice(["card","upi","cod","netbanking"], size=n_orders, p=[0.45,0.35,0.15,0.05])
    orders["failed_payment_attempts"] = np.random.poisson(0.2, size=n_orders)
    orders["is_returned"] = (np.random.rand(n_orders) < 0.08).astype(int)
    orders["customer_rating"] = np.random.choice([1,2,3,4,5], size=n_orders, p=[0.02,0.03,0.10,0.35,0.50])
    
    # Additional features for better predictions
    orders["hour_of_day"] = pd.to_datetime(orders["order_date"]).dt.hour
    orders["day_of_week"] = pd.to_datetime(orders["order_date"]).dt.dayofweek
    orders["month"] = pd.to_datetime(orders["order_date"]).dt.month
    orders["profit_margin"] = np.round((orders["price_after_discount"] - orders["base_price"] * 0.6) / orders["price_after_discount"] * 100, 2)
    
    print(f"✓ Generated {len(products)} products, {len(customers)} customers, {len(orders)} orders")

# ========== 2) CHURN LABELING ==========
print("\n[2/8] Computing customer churn labels...")
last_order = orders.groupby("customer_id")["order_date"].max().reset_index().rename(columns={"order_date":"last_order_date"})
customers = customers.merge(last_order, on="customer_id", how="left")
customers["days_since_last_order"] = (reference_date - pd.to_datetime(customers["last_order_date"])).dt.days.fillna(9999).astype(int)
customers["churned"] = (customers["days_since_last_order"] > 180).astype(int)
print(f"✓ Churn rate: {customers['churned'].mean():.2%}")

# ========== 3) SALES PREDICTION MODEL (Enhanced) ==========
print("\n[3/8] Training Sales Prediction Model...")
reg_features = ["quantity", "discount_pct", "base_price", "rating", "delivery_delay", 
                "hour_of_day", "day_of_week", "month", "profit_margin"]
X_reg = orders[reg_features].copy()
y_reg = orders["order_value"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
scaler_reg = StandardScaler().fit(Xr_train)
Xr_train_s = scaler_reg.transform(Xr_train)
Xr_test_s = scaler_reg.transform(Xr_test)

# Using Gradient Boosting for better predictions
reg_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
reg_model.fit(Xr_train_s, yr_train)
yr_pred = reg_model.predict(Xr_test_s)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))  # Fixed for newer scikit-learn versions
r2 = r2_score(yr_test, yr_pred)

print(f"✓ Sales Prediction - MAE: ₹{mae:.2f}, RMSE: ₹{rmse:.2f}, R²: {r2:.3f}")

# Feature importance for sales
feature_importance_reg = pd.DataFrame({
    'feature': reg_features,
    'importance': reg_model.feature_importances_
}).sort_values('importance', ascending=False)

# ========== 4) CHURN PREDICTION MODEL (Enhanced) ==========
print("\n[4/8] Training Customer Churn Prediction Model...")
order_stats = orders.groupby("customer_id").agg(
    total_orders=("order_id", "count"),
    avg_order_value=("order_value", "mean"),
    total_spent=("order_value", "sum"),
    avg_rating_given=("customer_rating", "mean"),
    return_rate=("is_returned", "mean"),
    avg_discount=("discount_pct", "mean")
).reset_index()

customers_enhanced = customers.merge(order_stats, on="customer_id", how="left").fillna({
    "total_orders": 0, "avg_order_value": 0, "total_spent": 0, 
    "avg_rating_given": 0, "return_rate": 0, "avg_discount": 0
})
customers_enhanced["recency"] = customers_enhanced["days_since_last_order"]
customers_enhanced["customer_age_days"] = (pd.to_datetime(reference_date) - pd.to_datetime(customers_enhanced["signup_date"])).dt.days

clf_features = ["lifetime_value", "total_orders", "avg_order_value", "recency", 
                "total_spent", "avg_rating_given", "return_rate", "customer_age_days"]
X_clf = customers_enhanced[clf_features].copy()
y_clf = customers_enhanced["churned"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
scaler_clf = StandardScaler().fit(Xc_train)
Xc_train_s = scaler_clf.transform(Xc_train)
Xc_test_s = scaler_clf.transform(Xc_test)

clf_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=10, random_state=42, n_jobs=-1)
clf_model.fit(Xc_train_s, yc_train)
yc_pred = clf_model.predict(Xc_test_s)
yc_proba = clf_model.predict_proba(Xc_test_s)[:, 1]

acc = accuracy_score(yc_test, yc_pred)
roc = roc_auc_score(yc_test, yc_proba)
precision = precision_score(yc_test, yc_pred)
recall = recall_score(yc_test, yc_pred)
f1 = f1_score(yc_test, yc_pred)

print(f"✓ Churn Prediction - Accuracy: {acc:.3f}, ROC-AUC: {roc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# Feature importance for churn
feature_importance_clf = pd.DataFrame({
    'feature': clf_features,
    'importance': clf_model.feature_importances_
}).sort_values('importance', ascending=False)

# ========== 5) PRODUCT RECOMMENDATION SYSTEM ==========
print("\n[5/8] Building Product Recommendation System...")
prod_feat = products.copy()
prod_feat["price_log"] = np.log1p(prod_feat["base_price"])
# Use sparse_output for newer scikit-learn versions (fallback to sparse for older versions)
try:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
except TypeError:
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
cat_enc = ohe.fit_transform(prod_feat[["category"]])
prod_X = np.hstack([prod_feat[["price_log", "rating"]].values, cat_enc])
knn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(prod_X)

def recommend_similar(product_id, top_n=5):
    """Recommend similar products"""
    try:
        idx = prod_feat.index[prod_feat["product_id"] == product_id].tolist()[0]
        distances, indices = knn.kneighbors([prod_X[idx]], n_neighbors=top_n+1)
        recs = prod_feat.iloc[indices[0]].copy()
        recs["similarity"] = 1 - distances[0]
        recs = recs[recs["product_id"] != product_id].head(top_n)
        return recs[["product_id", "category", "base_price", "rating", "similarity"]]
    except:
        return pd.DataFrame()

print("✓ Recommendation system ready")

# ========== 6) COMPREHENSIVE BUSINESS INSIGHTS & ACTIONABLE HINTS ==========
print("\n[6/8] Generating Business Insights & Actionable Recommendations...")

def generate_comprehensive_insights(orders_df, products_df, customers_df, reg_model, clf_model, 
                                   scaler_reg, scaler_clf, feature_importance_reg, feature_importance_clf):
    """Generate comprehensive business insights with actionable recommendations"""
    insights = {
        "revenue_insights": [],
        "customer_insights": [],
        "product_insights": [],
        "operational_insights": [],
        "growth_opportunities": [],
        "risk_alerts": []
    }
    
    cutoff_90 = reference_date - timedelta(days=90)
    cutoff_30 = reference_date - timedelta(days=30)
    recent_orders_90 = orders_df[orders_df["order_date"] >= cutoff_90]
    recent_orders_30 = orders_df[orders_df["order_date"] >= cutoff_30]
    
    # === REVENUE INSIGHTS ===
    if not recent_orders_90.empty:
        total_revenue_90 = recent_orders_90["order_value"].sum()
        total_revenue_30 = recent_orders_30["order_value"].sum() if not recent_orders_30.empty else 0
        
        insights["revenue_insights"].append({
            "metric": "Total Revenue (90 days)",
            "value": f"₹{total_revenue_90:,.2f}",
            "hint": f"Focus on maintaining this revenue momentum. Consider A/B testing pricing strategies."
        })
        
        # Category analysis
        cat_revenue = recent_orders_90.groupby("category")["order_value"].sum().sort_values(ascending=False)
        top_cat = cat_revenue.index[0]
        top_cat_revenue = cat_revenue.iloc[0]
        cat_pct = (top_cat_revenue / total_revenue_90) * 100
        
        insights["revenue_insights"].append({
            "metric": "Top Revenue Category",
            "value": f"{top_cat}: ₹{top_cat_revenue:,.2f} ({cat_pct:.1f}%)",
            "hint": f"🎯 ACTION: Increase inventory and marketing for '{top_cat}'. Consider bundling with complementary products."
        })
        
        # Discount analysis
        discount_impact = recent_orders_90.groupby("discount_pct").agg({
            "order_value": ["mean", "sum", "count"]
        }).reset_index()
        discount_impact.columns = ["discount_pct", "avg_order_value", "total_revenue", "order_count"]
        
        optimal_discount = discount_impact.loc[discount_impact["total_revenue"].idxmax(), "discount_pct"]
        
        insights["revenue_insights"].append({
            "metric": "Optimal Discount Strategy",
            "value": f"{optimal_discount}% discount drives highest revenue",
            "hint": f"💰 ACTION: Test {optimal_discount}% discount campaigns for underperforming products. Monitor profit margins."
        })
    
    # === CUSTOMER INSIGHTS ===
    customers_with_predictions = customers_df.copy()
    customers_with_predictions = customers_with_predictions.merge(order_stats, on="customer_id", how="left").fillna({
        "total_orders": 0, "avg_order_value": 0, "total_spent": 0,
        "avg_rating_given": 0, "return_rate": 0, "avg_discount": 0
    })
    customers_with_predictions["recency"] = customers_with_predictions["days_since_last_order"]
    customers_with_predictions["customer_age_days"] = (
        pd.to_datetime(reference_date) - pd.to_datetime(customers_with_predictions["signup_date"])
    ).dt.days
    
    # Churn predictions for all customers
    X_all = customers_with_predictions[clf_features]
    customers_with_predictions["churn_probability"] = clf_model.predict_proba(
        scaler_clf.transform(X_all)
    )[:, 1]
    
    high_risk_customers = customers_with_predictions[
        (customers_with_predictions["churn_probability"] > 0.7) & 
        (customers_with_predictions["total_orders"] > 0)
    ].sort_values("total_spent", ascending=False).head(10)
    
    if not high_risk_customers.empty:
        top_at_risk = high_risk_customers.iloc[0]
        total_at_risk_value = high_risk_customers["total_spent"].sum()
        
        insights["customer_insights"].append({
            "metric": "High-Value At-Risk Customers",
            "value": f"{len(high_risk_customers)} customers (Total Value: ₹{total_at_risk_value:,.2f})",
            "hint": f"🚨 URGENT: Launch retention campaign for top {len(high_risk_customers)} customers. "
                   f"Offer personalized discounts or loyalty rewards. Estimated save: ₹{total_at_risk_value * 0.3:,.2f}"
        })
    
    # Customer segments
    high_value = customers_with_predictions[customers_with_predictions["lifetime_value"] > 
                                           customers_with_predictions["lifetime_value"].quantile(0.9)]
    
    insights["customer_insights"].append({
        "metric": "High-Value Customers",
        "value": f"{len(high_value)} customers (Top 10%)",
        "hint": f"⭐ ACTION: Create VIP program. Send exclusive early access, premium support, and special offers."
    })
    
    # === PRODUCT INSIGHTS ===
    if not recent_orders_90.empty:
        prod_sales = recent_orders_90.groupby("product_id").agg({
            "quantity": "sum",
            "order_value": "sum",
            "rating": "mean"
        }).reset_index()
        prod_sales.columns = ["product_id", "units_sold", "revenue", "avg_rating"]
        prod_sales = prod_sales.merge(products_df[["product_id", "base_price", "category"]], on="product_id")
        prod_sales = prod_sales.sort_values("revenue", ascending=False)
        
        top_products = prod_sales.head(5)
        worst_products = prod_sales[prod_sales["revenue"] > 0].tail(5)
        
        insights["product_insights"].append({
            "metric": "Top 5 Products (Revenue)",
            "value": ", ".join(top_products["product_id"].tolist()),
            "hint": f"🏆 ACTION: Ensure adequate stock. Create product bundles. Use as anchor products in marketing."
        })
        
        if not worst_products.empty:
            insights["product_insights"].append({
                "metric": "Underperforming Products",
                "value": ", ".join(worst_products["product_id"].tolist()),
                "hint": f"⚠️ ACTION: Review pricing strategy. Consider promotions, bundling, or discontinuing low-margin items."
            })
    
    # === OPERATIONAL INSIGHTS ===
    if not orders_df.empty:
        # Delivery performance
        avg_delivery = orders_df["delivery_time_days"].mean()
        delayed_orders = orders_df[orders_df["delivery_delay"] > 2]
        delay_return_correlation = orders_df.groupby("delivery_delay").agg({
            "is_returned": "mean",
            "customer_rating": "mean"
        }).reset_index()
        
        insights["operational_insights"].append({
            "metric": "Average Delivery Time",
            "value": f"{avg_delivery:.1f} days",
            "hint": f"📦 ACTION: Target <3 days. Delays of {delayed_orders['delivery_delay'].mean():.1f} days correlate with "
                   f"{delayed_orders['is_returned'].mean():.1%} return rate. Optimize logistics or offer express shipping."
        })
        
        # Payment issues
        payment_failures = orders_df[orders_df["failed_payment_attempts"] > 0]
        if not payment_failures.empty:
            insights["operational_insights"].append({
                "metric": "Payment Failures",
                "value": f"{len(payment_failures)} orders ({len(payment_failures)/len(orders_df)*100:.1f}%)",
                "hint": f"💳 ACTION: Improve payment gateway reliability. Offer multiple payment options. "
                       f"Potential lost revenue: ₹{payment_failures['order_value'].sum():,.2f}"
            })
        
        # Return analysis
        return_rate = orders_df["is_returned"].mean()
        return_by_category = orders_df.groupby("category")["is_returned"].mean().sort_values(ascending=False)
        
        insights["operational_insights"].append({
            "metric": "Overall Return Rate",
            "value": f"{return_rate:.2%}",
            "hint": f"📦 ACTION: Target <5%. Highest returns in '{return_by_category.index[0]}' ({return_by_category.iloc[0]:.2%}). "
                   f"Improve product descriptions, quality checks, and sizing guides."
        })
    
    # === GROWTH OPPORTUNITIES ===
    # Price optimization
    insights["growth_opportunities"].append({
        "opportunity": "Dynamic Pricing",
        "description": "Top features affecting sales: " + ", ".join(feature_importance_reg.head(3)["feature"].tolist()),
        "hint": f"💡 ACTION: Implement dynamic pricing based on demand, time of day, and customer segments. "
               f"Expected revenue increase: 5-15%"
    })
    
    # Customer acquisition
    avg_customer_value = customers_with_predictions["lifetime_value"].mean()
    insights["growth_opportunities"].append({
        "opportunity": "Customer Acquisition",
        "description": f"Average Customer Lifetime Value: ₹{avg_customer_value:,.2f}",
        "hint": f"💡 ACTION: Invest in marketing channels that bring customers with LTV > ₹{avg_customer_value:,.2f}. "
               f"Focus on '{customers_df['city'].mode()[0]}' market for expansion."
    })
    
    # Cross-selling
    insights["growth_opportunities"].append({
        "opportunity": "Cross-Selling & Upselling",
        "description": "Average order value can be increased through recommendations",
        "hint": f"💡 ACTION: Implement 'Customers who bought X also bought Y' recommendations. "
               f"Target: Increase AOV by 10-20%"
    })
    
    # === RISK ALERTS ===
    # Stockout risk
    if "stock_quantity" in products_df.columns:
        low_stock = products_df[products_df["stock_quantity"] < 20]
        if not low_stock.empty:
            insights["risk_alerts"].append({
                "risk": "Low Stock Alert",
                "severity": "Medium",
                "description": f"{len(low_stock)} products running low on stock",
                "hint": f"🚨 ACTION: Reorder immediately: {', '.join(low_stock['product_id'].head(5).tolist())}"
            })
    
    # Churn risk
    churn_rate = customers_with_predictions["churn_probability"].mean()
    if churn_rate > 0.3:
        insights["risk_alerts"].append({
            "risk": "High Churn Risk",
            "severity": "High",
            "description": f"Average churn probability: {churn_rate:.1%}",
            "hint": f"🚨 URGENT: Launch customer retention program. Focus on top {len(high_risk_customers)} at-risk customers."
        })
    
    return insights, customers_with_predictions, high_risk_customers

insights_dict, customers_predicted, high_risk = generate_comprehensive_insights(
    orders, products, customers, reg_model, clf_model, scaler_reg, scaler_clf,
    feature_importance_reg, feature_importance_clf
)

# ========== 7) PERFORMANCE TRACKING DASHBOARD ==========
print("\n[7/8] Creating Performance Tracking Dashboard...")

def create_tracking_dashboard(orders_df, customers_df, reg_model, clf_model, 
                             feature_importance_reg, feature_importance_clf):
    """Create comprehensive tracking metrics"""
    dashboard = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_performance": {
            "sales_prediction": {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2_score": float(r2),
                "status": "Excellent" if r2 > 0.85 else "Good" if r2 > 0.70 else "Fair"
            },
            "churn_prediction": {
                "accuracy": float(acc),
                "roc_auc": float(roc),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "status": "Excellent" if roc > 0.85 else "Good" if roc > 0.70 else "Fair"
            }
        },
        "business_kpis": {
            "total_orders": int(len(orders_df)),
            "total_customers": int(len(customers_df)),
            "total_revenue": float(orders_df["order_value"].sum()),
            "average_order_value": float(orders_df["order_value"].mean()),
            "customer_churn_rate": float(customers_df["churned"].mean()),
            "return_rate": float(orders_df["is_returned"].mean()),
            "average_delivery_time": float(orders_df["delivery_time_days"].mean())
        },
        "top_features_sales": feature_importance_reg.head(5).to_dict('records'),
        "top_features_churn": feature_importance_clf.head(5).to_dict('records')
    }
    return dashboard

dashboard = create_tracking_dashboard(orders, customers, reg_model, clf_model,
                                     feature_importance_reg, feature_importance_clf)

# ========== 8) SAVE ALL ARTIFACTS ==========
print("\n[8/8] Saving models and outputs...")
SAVE_DIR = "ecom_ml_output"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save data
orders.to_csv(os.path.join(SAVE_DIR, "orders.csv"), index=False)
products.to_csv(os.path.join(SAVE_DIR, "products.csv"), index=False)
customers_predicted.to_csv(os.path.join(SAVE_DIR, "customers_with_predictions.csv"), index=False)

# Save models
joblib.dump(reg_model, os.path.join(SAVE_DIR, "sales_regression_model.joblib"))
joblib.dump(clf_model, os.path.join(SAVE_DIR, "churn_classification_model.joblib"))
joblib.dump(scaler_reg, os.path.join(SAVE_DIR, "scaler_regression.joblib"))
joblib.dump(scaler_clf, os.path.join(SAVE_DIR, "scaler_classification.joblib"))
joblib.dump(knn, os.path.join(SAVE_DIR, "recommendation_engine.joblib"))
joblib.dump(ohe, os.path.join(SAVE_DIR, "product_category_encoder.joblib"))

# Save insights and dashboard
with open(os.path.join(SAVE_DIR, "business_insights.json"), "w") as f:
    json.dump(insights_dict, f, indent=2, default=str)

with open(os.path.join(SAVE_DIR, "performance_dashboard.json"), "w") as f:
    json.dump(dashboard, f, indent=2, default=str)

# Save high-risk customers
if not high_risk.empty:
    high_risk.to_csv(os.path.join(SAVE_DIR, "high_risk_customers.csv"), index=False)

# Save feature importance
feature_importance_reg.to_csv(os.path.join(SAVE_DIR, "feature_importance_sales.csv"), index=False)
feature_importance_clf.to_csv(os.path.join(SAVE_DIR, "feature_importance_churn.csv"), index=False)

print(f"✓ All files saved to: {SAVE_DIR}/")

# ========== 9) COMPREHENSIVE OUTPUT REPORT ==========
print("\n" + "="*80)
print("BUSINESS INTELLIGENCE REPORT")
print("="*80)

print("\n" + "-"*80)
print("MODEL PERFORMANCE METRICS")
print("-"*80)
print(f"\n📊 Sales Prediction Model:")
print(f"   • Mean Absolute Error: ₹{mae:,.2f}")
print(f"   • Root Mean Squared Error: ₹{rmse:,.2f}")
print(f"   • R² Score: {r2:.3f} ({'✅ Excellent' if r2 > 0.85 else '✅ Good' if r2 > 0.70 else '⚠️ Needs Improvement'})")

print(f"\n👥 Churn Prediction Model:")
print(f"   • Accuracy: {acc:.3f}")
print(f"   • ROC-AUC Score: {roc:.3f} ({'✅ Excellent' if roc > 0.85 else '✅ Good' if roc > 0.70 else '⚠️ Needs Improvement'})")
print(f"   • Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")

print("\n" + "-"*80)
print("KEY BUSINESS METRICS (KPIs)")
print("-"*80)
print(f"   • Total Revenue: ₹{dashboard['business_kpis']['total_revenue']:,.2f}")
print(f"   • Average Order Value: ₹{dashboard['business_kpis']['average_order_value']:,.2f}")
print(f"   • Total Customers: {dashboard['business_kpis']['total_customers']:,}")
print(f"   • Customer Churn Rate: {dashboard['business_kpis']['customer_churn_rate']:.2%}")
print(f"   • Return Rate: {dashboard['business_kpis']['return_rate']:.2%}")
print(f"   • Average Delivery Time: {dashboard['business_kpis']['average_delivery_time']:.1f} days")

print("\n" + "-"*80)
print("TOP FEATURES AFFECTING SALES")
print("-"*80)
for idx, row in feature_importance_reg.head(5).iterrows():
    print(f"   {idx+1}. {row['feature']}: {row['importance']:.3f}")

print("\n" + "-"*80)
print("TOP FEATURES AFFECTING CUSTOMER CHURN")
print("-"*80)
for idx, row in feature_importance_clf.head(5).iterrows():
    print(f"   {idx+1}. {row['feature']}: {row['importance']:.3f}")

print("\n" + "-"*80)
print("ACTIONABLE BUSINESS INSIGHTS & HINTS")
print("-"*80)

print("\n💰 REVENUE INSIGHTS:")
for insight in insights_dict["revenue_insights"][:3]:
    print(f"   • {insight['metric']}: {insight['value']}")
    print(f"     💡 {insight['hint']}")

print("\n👥 CUSTOMER INSIGHTS:")
for insight in insights_dict["customer_insights"][:3]:
    print(f"   • {insight['metric']}: {insight['value']}")
    print(f"     💡 {insight['hint']}")

print("\n📦 PRODUCT INSIGHTS:")
for insight in insights_dict["product_insights"][:2]:
    print(f"   • {insight['metric']}: {insight['value']}")
    print(f"     💡 {insight['hint']}")

print("\n⚙️  OPERATIONAL INSIGHTS:")
for insight in insights_dict["operational_insights"][:3]:
    print(f"   • {insight['metric']}: {insight['value']}")
    print(f"     💡 {insight['hint']}")

print("\n🚀 GROWTH OPPORTUNITIES:")
for opp in insights_dict["growth_opportunities"][:3]:
    print(f"   • {opp['opportunity']}: {opp['description']}")
    print(f"     💡 {opp['hint']}")

if insights_dict["risk_alerts"]:
    print("\n🚨 RISK ALERTS:")
    for risk in insights_dict["risk_alerts"]:
        print(f"   • [{risk['severity']}] {risk['risk']}: {risk['description']}")
        print(f"     💡 {risk['hint']}")

print("\n" + "-"*80)
print("EXAMPLE PRODUCT RECOMMENDATIONS")
print("-"*80)
if len(products) > 0:
    sample_product = products['product_id'].iloc[0]
    recs = recommend_similar(sample_product, top_n=5)
    if not recs.empty:
        print(f"\n   Products similar to {sample_product}:")
        print(recs.to_string(index=False))

print("\n" + "="*80)
print(f"✅ All outputs saved to: {SAVE_DIR}/")
print(f"📁 Files generated: {len(os.listdir(SAVE_DIR))} files")
print("="*80)

print("\n💼 NEXT STEPS:")
print("   1. Review high_risk_customers.csv for retention campaigns")
print("   2. Implement recommendations from business insights")
print("   3. Monitor feature importance to track changing patterns")
print("   4. Update models monthly with new data for better predictions")
print("   5. A/B test suggested pricing and discount strategies")
