"""
Helper script to load and split a single Excel file with all columns
This automatically creates orders, products, and customers dataframes
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_split_excel_data(file_path, sheet_name=None):
    """
    Load a single Excel file and automatically split into orders, products, customers
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name (None = first sheet)
    
    Returns:
        orders_df, products_df, customers_df
    """
    print(f"Loading Excel file: {file_path}")
    
    # Load the Excel file
    if sheet_name:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        df = pd.read_excel(file_path, sheet_name=0)
    
    print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
    print(f"Columns found: {', '.join(df.columns[:10])}...")
    
    # ========== CREATE ORDERS DATAFRAME ==========
    orders = pd.DataFrame()
    
    # Map order columns (handle various column name variations)
    column_mapping_orders = {
        'order_id': ['Order.ID', 'Order ID', 'order_id', 'Order_ID', 'orderId'],
        'order_date': ['Date', 'Order Date', 'order_date', 'Order Date', 'Order_Date', 'è®å½æ Order.'],
        'customer_id': ['Customer.', 'Customer ID', 'customer_id', 'Customer_ID', 'Customer'],
        'product_id': ['Product.ID', 'Product ID', 'product_id', 'Product_ID'],
        'quantity': ['Quantity', 'quantity'],
        'sales': ['Sales', 'sales', 'order_value'],
        'discount': ['Discount', 'discount', 'discount_pct'],
        'profit': ['Profit', 'profit'],
        'ship_date': ['Ship. Date', 'Ship Date', 'ship_date', 'Ship_Date'],
        'ship_mode': ['Ship. Mode', 'Ship Mode', 'ship_mode'],
    }
    
    # Find and map columns
    for target_col, possible_names in column_mapping_orders.items():
        for col_name in df.columns:
            if any(name in str(col_name) for name in possible_names):
                orders[target_col] = df[col_name]
                print(f"   Mapped '{col_name}' → '{target_col}'")
                break
    
    # Create order_value from sales if not found
    if 'order_value' not in orders.columns and 'sales' in orders.columns:
        orders['order_value'] = orders['sales']
    
    # Create order_id if missing (use row number)
    if 'order_id' not in orders.columns or orders['order_id'].isna().all():
        orders['order_id'] = df.get('Row.ID', pd.Series(range(1, len(df) + 1)))
        orders['order_id'] = 'O' + orders['order_id'].astype(str)
    
    # Create customer_id if missing
    if 'customer_id' not in orders.columns or orders['customer_id'].isna().all():
        # Try to create from segment or other customer columns
        orders['customer_id'] = 'C' + pd.Series(range(1, len(df) + 1)).astype(str)
    
    # Create product_id if missing
    if 'product_id' not in orders.columns or orders['product_id'].isna().all():
        orders['product_id'] = 'P' + pd.Series(range(1, len(df) + 1)).astype(str)
    
    # Parse dates
    date_cols = ['order_date', 'ship_date']
    for col in date_cols:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors='coerce')
    
    # Calculate missing fields
    if 'quantity' not in orders.columns:
        orders['quantity'] = 1
    
    if 'discount' not in orders.columns:
        orders['discount_pct'] = 0
    else:
        orders['discount_pct'] = orders['discount'] * 100 if orders['discount'].max() < 1 else orders['discount']
    
    # Calculate delivery_time_days
    if 'ship_date' in orders.columns and 'order_date' in orders.columns:
        orders['delivery_time_days'] = (orders['ship_date'] - orders['order_date']).dt.days
        orders['delivery_time_days'] = orders['delivery_time_days'].fillna(3).astype(int)
    else:
        orders['delivery_time_days'] = 3
    
    orders['delivery_delay'] = (orders['delivery_time_days'] - 2).clip(lower=0)
    
    # Add other required fields
    orders['payment_method'] = df.get('Payment Method', 'card')
    orders['failed_payment_attempts'] = 0
    orders['is_returned'] = 0
    orders['customer_rating'] = 4
    
    print(f"✓ Created orders dataframe: {len(orders)} rows")
    
    # ========== CREATE PRODUCTS DATAFRAME ==========
    products = pd.DataFrame()
    
    # Get unique products
    if 'product_id' in orders.columns:
        products['product_id'] = orders['product_id'].unique()
    else:
        products['product_id'] = df.get('Product.ID', pd.Series(range(1, len(df) + 1))).unique()
        products['product_id'] = 'P' + products['product_id'].astype(str)
    
    # Map product columns
    column_mapping_products = {
        'category': ['Category', 'category'],
        'base_price': ['Sales', 'sales', 'Price', 'price'],
        'rating': ['Rating', 'rating'],
        'product_name': ['Product. N.', 'Product Name', 'Product_Name', 'Product'],
    }
    
    # Get product information from first occurrence
    for target_col, possible_names in column_mapping_products.items():
        for col_name in df.columns:
            if any(name in str(col_name) for name in possible_names):
                product_info = df.groupby(orders['product_id'] if 'product_id' in orders.columns else df.get('Product.ID', range(len(df))))[col_name].first()
                products[target_col] = products['product_id'].map(product_info)
                print(f"   Mapped '{col_name}' → '{target_col}'")
                break
    
    # If category not found, try Sub. CategYear
    if 'category' not in products.columns or products['category'].isna().all():
        if 'Sub. CategYear' in df.columns:
            product_info = df.groupby(orders['product_id'] if 'product_id' in orders.columns else df.get('Product.ID', range(len(df))))['Sub. CategYear'].first()
            products['category'] = products['product_id'].map(product_info)
        else:
            products['category'] = 'General'
    
    # Calculate base_price from sales if needed
    if 'base_price' not in products.columns or products['base_price'].isna().all():
        if 'sales' in orders.columns:
            product_price = orders.groupby('product_id')['sales'].mean()
            products['base_price'] = products['product_id'].map(product_price)
        else:
            products['base_price'] = 1000  # Default
    
    # Default rating if not found
    if 'rating' not in products.columns or products['rating'].isna().all():
        products['rating'] = 4.0
    
    # Add optional fields
    products['stock_quantity'] = 100
    products['manufacturing_cost'] = products['base_price'] * 0.5
    
    print(f"✓ Created products dataframe: {len(products)} unique products")
    
    # ========== CREATE CUSTOMERS DATAFRAME ==========
    customers = pd.DataFrame()
    
    # Get unique customers
    if 'customer_id' in orders.columns:
        customers['customer_id'] = orders['customer_id'].unique()
    else:
        customers['customer_id'] = df.get('Customer.', pd.Series(range(1, len(df) + 1))).unique()
        customers['customer_id'] = 'C' + customers['customer_id'].astype(str)
    
    # Map customer columns
    column_mapping_customers = {
        'city': ['City', 'city'],
        'region': ['Region', 'region'],
        'country': ['Country', 'country'],
        'state': ['State', 'state'],
        'segment': ['Segment', 'segment'],
    }
    
    # Get customer information from first occurrence
    customer_id_col = orders['customer_id'] if 'customer_id' in orders.columns else df.get('Customer.', range(len(df)))
    
    for target_col, possible_names in column_mapping_customers.items():
        for col_name in df.columns:
            if any(name in str(col_name) for name in possible_names):
                customer_info = df.groupby(customer_id_col)[col_name].first()
                customers[target_col] = customers['customer_id'].map(customer_info)
                print(f"   Mapped '{col_name}' → '{target_col}'")
                break
    
    # Calculate lifetime_value from sales
    if 'sales' in orders.columns and 'customer_id' in orders.columns:
        customers['lifetime_value'] = orders.groupby('customer_id')['sales'].sum()
        customers['lifetime_value'] = customers['customer_id'].map(customers['lifetime_value'])
    else:
        customers['lifetime_value'] = 5000  # Default
    
    # Create signup_date (estimate from first order)
    if 'order_date' in orders.columns and 'customer_id' in orders.columns:
        first_order = orders.groupby('customer_id')['order_date'].min()
        customers['signup_date'] = customers['customer_id'].map(first_order)
        customers['signup_date'] = pd.to_datetime(customers['signup_date'], errors='coerce')
    else:
        customers['signup_date'] = datetime(2020, 1, 1)
    
    # Fill missing values
    if 'city' not in customers.columns or customers['city'].isna().all():
        customers['city'] = df.get('City', 'Unknown').iloc[0] if 'City' in df.columns else 'Unknown'
    
    customers['age_group'] = 'Unknown'
    
    print(f"✓ Created customers dataframe: {len(customers)} unique customers")
    
    return orders, products, customers

if __name__ == "__main__":
    # Test the function
    print("Testing Excel file loader...")
    print("Usage: Use this function in the main script to load your single Excel file")
