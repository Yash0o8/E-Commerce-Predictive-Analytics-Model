"""
E-Commerce Predictive Model - Interactive Streamlit Dashboard
Displays insights, metrics, and visualizations like Power BI
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime
import numpy as np

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="E-Commerce ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-value {
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            opacity: 0.9;
        }
    </style>
""", unsafe_allow_html=True)

# ========== LOAD DATA ==========
@st.cache_resource
def load_data():
    output_dir = "ecom_ml_output"
    
    data = {
        "orders": None,
        "products": None,
        "customers": None,
        "insights": None,
        "dashboard": None
    }
    
    # Load CSV files
    if os.path.exists(os.path.join(output_dir, "orders.csv")):
        data["orders"] = pd.read_csv(os.path.join(output_dir, "orders.csv"))
        data["orders"]["order_date"] = pd.to_datetime(data["orders"]["order_date"], errors='coerce')
    
    if os.path.exists(os.path.join(output_dir, "products.csv")):
        data["products"] = pd.read_csv(os.path.join(output_dir, "products.csv"))
    
    if os.path.exists(os.path.join(output_dir, "customers.csv")):
        data["customers"] = pd.read_csv(os.path.join(output_dir, "customers.csv"))
    
    # Load JSON insights
    if os.path.exists(os.path.join(output_dir, "business_insights.json")):
        with open(os.path.join(output_dir, "business_insights.json")) as f:
            data["insights"] = json.load(f)
    
    if os.path.exists(os.path.join(output_dir, "performance_dashboard.json")):
        with open(os.path.join(output_dir, "performance_dashboard.json")) as f:
            data["dashboard"] = json.load(f)
    
    return data

# Load data
data = load_data()
orders = data["orders"]
products = data["products"]
customers = data["customers"]
insights = data["insights"]
dashboard = data["dashboard"]

# ========== HEADER ==========
st.title("🎯 E-COMMERCE PREDICTIVE ANALYTICS DASHBOARD")
st.markdown("Real-time Business Intelligence & ML Insights")

# ========== SIDEBAR FILTERS ==========
st.sidebar.header("📋 Filters & Options")

if orders is not None and "order_date" in orders.columns:
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(orders["order_date"].min().date(), orders["order_date"].max().date()),
        min_value=orders["order_date"].min().date(),
        max_value=orders["order_date"].max().date()
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        orders_filtered = orders[(orders["order_date"].dt.date >= start_date) & 
                                (orders["order_date"].dt.date <= end_date)]
    else:
        orders_filtered = orders
else:
    orders_filtered = orders

# Category filter
if products is not None and "category" in products.columns:
    categories = products["category"].unique()
    selected_categories = st.sidebar.multiselect(
        "Select Categories",
        options=categories,
        default=list(categories)[:3] if len(categories) > 0 else []
    )
    if selected_categories:
        product_ids = products[products["category"].isin(selected_categories)]["product_id"].tolist()
        orders_filtered = orders_filtered[orders_filtered["product_id"].isin(product_ids)]

# ========== KPI METRICS ==========
st.header("📊 Key Performance Indicators")

if dashboard and "business_kpis" in dashboard:
    kpis = dashboard["business_kpis"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=f"₹{kpis.get('total_revenue', 0):,.0f}",
            delta=f"Orders: {kpis.get('total_orders', 0):,}"
        )
    
    with col2:
        st.metric(
            label="👥 Total Customers",
            value=f"{kpis.get('total_customers', 0):,}",
            delta=f"Churn Rate: {kpis.get('customer_churn_rate', 0):.1%}"
        )
    
    with col3:
        st.metric(
            label="📦 Avg Order Value",
            value=f"₹{kpis.get('average_order_value', 0):,.2f}",
            delta=f"Return Rate: {kpis.get('return_rate', 0):.2%}"
        )
    
    with col4:
        st.metric(
            label="📅 Avg Delivery Time",
            value=f"{kpis.get('average_delivery_time', 0):.1f} days",
            delta="Target: < 3 days"
        )

# ========== MODEL PERFORMANCE ==========
st.header("🤖 Model Performance")

col1, col2 = st.columns(2)

if dashboard and "model_performance" in dashboard:
    perf = dashboard["model_performance"]
    
    with col1:
        st.subheader("📈 Sales Prediction Model")
        sales_perf = perf.get("sales_prediction", {})
        st.metric("R² Score", f"{sales_perf.get('r2_score', 0):.3f}", "Good" if sales_perf.get('r2_score', 0) > 0.7 else "Fair")
        st.metric("MAE", f"₹{sales_perf.get('mae', 0):.2f}")
        st.metric("RMSE", f"₹{sales_perf.get('rmse', 0):.2f}")
    
    with col2:
        st.subheader("👥 Churn Prediction Model")
        churn_perf = perf.get("churn_prediction", {})
        st.metric("Accuracy", f"{churn_perf.get('accuracy', 0):.3f}")
        st.metric("Precision", f"{churn_perf.get('precision', 0):.3f}")
        st.metric("Recall", f"{churn_perf.get('recall', 0):.3f}")

# ========== VISUALIZATIONS ==========
st.header("📉 Sales Analytics")

col1, col2 = st.columns(2)

# Revenue by Category
with col1:
    if orders_filtered is not None and "category" in orders_filtered.columns and "order_value" in orders_filtered.columns:
        category_revenue = orders_filtered.groupby("category")["order_value"].sum().sort_values(ascending=False).head(10)
        
        fig = go.Figure(data=[
            go.Bar(x=category_revenue.index, y=category_revenue.values, 
                   marker=dict(color=category_revenue.values, colorscale="Viridis"))
        ])
        fig.update_layout(
            title="💰 Top 10 Categories by Revenue",
            xaxis_title="Category",
            yaxis_title="Revenue (₹)",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# Orders by Day of Week
with col2:
    if orders_filtered is not None and "day_of_week" in orders_filtered.columns:
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_counts = orders_filtered["day_of_week"].value_counts().sort_index()
        day_labels = [day_names[i] if i < len(day_names) else f"Day {i}" for i in day_counts.index]
        
        fig = go.Figure(data=[
            go.Bar(x=day_labels, y=day_counts.values, 
                   marker=dict(color=day_counts.values, colorscale="Blues"))
        ])
        fig.update_layout(
            title="📊 Orders by Day of Week",
            xaxis_title="Day",
            yaxis_title="Number of Orders",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== REVENUE TRENDS ==========
st.header("📈 Revenue Trends")

if orders_filtered is not None and "order_date" in orders_filtered.columns and "order_value" in orders_filtered.columns:
    daily_revenue = orders_filtered.groupby(orders_filtered["order_date"].dt.date)["order_value"].sum().reset_index()
    daily_revenue.columns = ["date", "revenue"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_revenue["date"],
        y=daily_revenue["revenue"],
        mode="lines+markers",
        name="Daily Revenue",
        fill="tozeroy",
        line=dict(color="#667eea", width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="💹 Daily Revenue Trend",
        xaxis_title="Date",
        yaxis_title="Revenue (₹)",
        height=400,
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== FEATURE IMPORTANCE ==========
st.header("🎯 Feature Importance for Predictions")

col1, col2 = st.columns(2)

if dashboard and "top_features_sales" in dashboard:
    with col1:
        features_sales = dashboard["top_features_sales"]
        feature_names = [f["feature"] for f in features_sales]
        feature_values = [f["importance"] for f in features_sales]
        
        fig = go.Figure(data=[
            go.Bar(x=feature_values, y=feature_names, orientation="h",
                   marker=dict(color=feature_values, colorscale="Greens"))
        ])
        fig.update_layout(
            title="📊 Top Features Affecting Sales",
            xaxis_title="Importance Score",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

if dashboard and "top_features_churn" in dashboard:
    with col2:
        features_churn = dashboard["top_features_churn"]
        feature_names = [f["feature"] for f in features_churn]
        feature_values = [f["importance"] for f in features_churn]
        
        fig = go.Figure(data=[
            go.Bar(x=feature_values, y=feature_names, orientation="h",
                   marker=dict(color=feature_values, colorscale="Reds"))
        ])
        fig.update_layout(
            title="👥 Top Features Affecting Churn",
            xaxis_title="Importance Score",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# ========== BUSINESS INSIGHTS ==========
st.header("💡 Actionable Business Insights")

if insights:
    # Customer Insights
    if insights.get("customer_insights"):
        st.subheader("👥 Customer Insights")
        for insight in insights["customer_insights"][:3]:
            st.info(f"**{insight['metric']}**: {insight['value']}\n\n{insight['hint']}")
    
    # Operational Insights
    if insights.get("operational_insights"):
        st.subheader("⚙️ Operational Insights")
        for insight in insights["operational_insights"][:2]:
            st.warning(f"**{insight['metric']}**: {insight['value']}\n\n{insight['hint']}")
    
    # Growth Opportunities
    if insights.get("growth_opportunities"):
        st.subheader("🚀 Growth Opportunities")
        for opp in insights["growth_opportunities"][:3]:
            st.success(f"**{opp['opportunity']}**: {opp['description']}\n\n{opp['hint']}")
    
    # Risk Alerts
    if insights.get("risk_alerts"):
        st.subheader("🚨 Risk Alerts")
        for risk in insights["risk_alerts"]:
            st.error(f"**[{risk['severity']}] {risk['risk']}**: {risk['description']}\n\n{risk['hint']}")

# ========== HIGH-RISK CUSTOMERS ==========
st.header("⚠️ High-Risk Customers (At-Risk of Churn)")

high_risk_path = os.path.join("ecom_ml_output", "high_risk_customers.csv")
if os.path.exists(high_risk_path):
    high_risk_df = pd.read_csv(high_risk_path)
    st.dataframe(high_risk_df.head(10), use_container_width=True)
    st.download_button(
        label="📥 Download High-Risk Customers CSV",
        data=high_risk_df.to_csv(index=False),
        file_name="high_risk_customers.csv",
        mime="text/csv"
    )

# ========== DATA DOWNLOAD ==========
st.header("📥 Download Reports")

col1, col2, col3 = st.columns(3)

if orders is not None:
    with col1:
        csv_orders = orders.to_csv(index=False)
        st.download_button(
            label="📊 Download Orders Data",
            data=csv_orders,
            file_name="orders.csv",
            mime="text/csv"
        )

if products is not None:
    with col2:
        csv_products = products.to_csv(index=False)
        st.download_button(
            label="📦 Download Products Data",
            data=csv_products,
            file_name="products.csv",
            mime="text/csv"
        )

if customers is not None:
    with col3:
        csv_customers = customers.to_csv(index=False)
        st.download_button(
            label="👥 Download Customers Data",
            data=csv_customers,
            file_name="customers.csv",
            mime="text/csv"
        )

# ========== FOOTER ==========
st.divider()
st.markdown("""
    <div style='text-align: center; opacity: 0.7; margin-top: 20px;'>
        <p>📊 E-Commerce ML Dashboard | Last Updated: {}</p>
        <p>Powered by Streamlit, Plotly & ML Models</p>
    </div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
