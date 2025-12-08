"""Quick script to move the Excel file to data folder"""
import os
import shutil

# Create data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

# Copy the file
source = 'data_templates/superstore.xl'
destination = 'data/superstore.xlsx'

try:
    shutil.copy2(source, destination)
    print(f"✅ Successfully copied {source} to {destination}")
    
    # Verify it exists
    if os.path.exists(destination):
        size = os.path.getsize(destination)
        print(f"✅ File exists in data folder! Size: {size:,} bytes")
    else:
        print("❌ File not found in destination")
        
except Exception as e:
    print(f"❌ Error: {e}")
