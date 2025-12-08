"""
Data Validation Helper Script
Run this before the main pipeline to check if your data is ready
"""

import pandas as pd
import os

def validate_data(data_folder="data"):
    """Validate e-commerce data files before running ML pipeline"""
    
    print("="*80)
    print("DATA VALIDATION TOOL")
    print("="*80)
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"\n❌ ERROR: Folder '{data_folder}' not found!")
        print(f"   Please create the folder and place your CSV files there.")
        return False
    
    files = {
        "orders": "orders.csv",
        "products": "products.csv",
        "customers": "customers.csv"
    }
    
    all_valid = True
    
    # Check each file
    for file_type, filename in files.items():
        filepath = os.path.join(data_folder, filename)
        
        print(f"\n{'='*80}")
        print(f"Checking {file_type.upper()} file: {filename}")
        print(f"{'='*80}")
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            all_valid = False
            continue
        
        try:
            # Try to load the file
            if file_type == "orders":
                df = pd.read_csv(filepath, parse_dates=["order_date"])
                required_cols = ["order_id", "customer_id", "product_id", "order_date", "quantity", "order_value"]
            elif file_type == "products":
                df = pd.read_csv(filepath)
                required_cols = ["product_id", "category", "base_price", "rating"]
            else:  # customers
                df = pd.read_csv(filepath, parse_dates=["signup_date"])
                required_cols = ["customer_id", "signup_date", "lifetime_value"]
            
            print(f"✅ File loaded successfully")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {len(df.columns)}")
            
            # Check required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Missing required columns: {', '.join(missing_cols)}")
                all_valid = False
            else:
                print(f"✅ All required columns present")
            
            # Check for missing values in required columns
            for col in required_cols:
                if col in df.columns:
                    missing_count = df[col].isna().sum()
                    if missing_count > 0:
                        print(f"⚠️  Column '{col}' has {missing_count} missing values ({missing_count/len(df)*100:.1f}%)")
                        if missing_count / len(df) > 0.1:  # More than 10% missing
                            print(f"   ⚠️  WARNING: High percentage of missing values!")
            
            # Specific validations
            if file_type == "orders":
                # Check date range
                if "order_date" in df.columns:
                    date_min = df["order_date"].min()
                    date_max = df["order_date"].max()
                    print(f"   Date range: {date_min} to {date_max}")
                
                # Check numeric columns
                if "quantity" in df.columns:
                    if (df["quantity"] < 1).any():
                        print(f"⚠️  Warning: Some quantities are less than 1")
                
                if "order_value" in df.columns:
                    if (df["order_value"] < 0).any():
                        print(f"⚠️  Warning: Some order values are negative")
                    print(f"   Total revenue: ₹{df['order_value'].sum():,.2f}")
            
            elif file_type == "products":
                # Check price range
                if "base_price" in df.columns:
                    print(f"   Price range: ₹{df['base_price'].min():,.2f} to ₹{df['base_price'].max():,.2f}")
                
                # Check rating range
                if "rating" in df.columns:
                    if (df["rating"] < 1).any() or (df["rating"] > 5).any():
                        print(f"⚠️  Warning: Some ratings outside 1-5 range")
            
            elif file_type == "customers":
                # Check lifetime value
                if "lifetime_value" in df.columns:
                    print(f"   LTV range: ₹{df['lifetime_value'].min():,.2f} to ₹{df['lifetime_value'].max():,.2f}")
                    print(f"   Average LTV: ₹{df['lifetime_value'].mean():,.2f}")
            
            print(f"✅ {file_type} validation complete")
            
        except Exception as e:
            print(f"❌ ERROR loading {filename}: {e}")
            all_valid = False
    
    # Cross-file validation
    print(f"\n{'='*80}")
    print("CROSS-FILE VALIDATION")
    print(f"{'='*80}")
    
    try:
        orders = pd.read_csv(os.path.join(data_folder, "orders.csv"))
        products = pd.read_csv(os.path.join(data_folder, "products.csv"))
        customers = pd.read_csv(os.path.join(data_folder, "customers.csv"))
        
        # Check customer IDs in orders match customers file
        if "customer_id" in orders.columns and "customer_id" in customers.columns:
            missing_customers = orders[~orders["customer_id"].isin(customers["customer_id"])]["customer_id"].unique()
            if len(missing_customers) > 0:
                print(f"⚠️  Warning: {len(missing_customers)} customer IDs in orders not found in customers file")
                if len(missing_customers) <= 10:
                    print(f"   Missing IDs: {', '.join(map(str, missing_customers[:10]))}")
        
        # Check product IDs in orders match products file
        if "product_id" in orders.columns and "product_id" in products.columns:
            missing_products = orders[~orders["product_id"].isin(products["product_id"])]["product_id"].unique()
            if len(missing_products) > 0:
                print(f"⚠️  Warning: {len(missing_products)} product IDs in orders not found in products file")
                if len(missing_products) <= 10:
                    print(f"   Missing IDs: {', '.join(map(str, missing_products[:10]))}")
        
        print(f"✅ Cross-file validation complete")
        
    except Exception as e:
        print(f"⚠️  Could not perform cross-file validation: {e}")
    
    # Final summary
    print(f"\n{'='*80}")
    if all_valid:
        print("✅ ALL CHECKS PASSED! Your data is ready to use.")
        print(f"\nNext step: Set USE_YOUR_DATA = True in the main script and run it!")
    else:
        print("❌ SOME ISSUES FOUND. Please fix the errors above before running the pipeline.")
    print(f"{'='*80}")
    
    return all_valid

if __name__ == "__main__":
    # You can change "data" to your data folder path
    validate_data("data")
