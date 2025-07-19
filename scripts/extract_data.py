#!/usr/bin/env python3
"""
Script to extract data from Airtable.
Run this first to download images and create metadata.csv
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.airtable_extractor import AirtableExtractor
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Check if credentials are set
    required_vars = ['AIRTABLE_API_KEY', 'AIRTABLE_BASE_ID', 'AIRTABLE_TABLE_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these in your .env file:")
        print("AIRTABLE_API_KEY=your_api_key")
        print("AIRTABLE_BASE_ID=your_base_id") 
        print("AIRTABLE_TABLE_NAME=your_table_name")
        return 1
    
    try:
        print("🔄 Starting Airtable data extraction...")
        
        # Initialize extractor
        extractor = AirtableExtractor()
        
        # Extract data
        df = extractor.extract_all_data()
        
        print(f"✅ Successfully extracted {len(df)} records")
        print(f"📁 Data saved to ./data/raw/")
        print(f"📊 Metadata saved to ./data/raw/metadata.csv")
        
        # Show summary
        print("\n📈 Dataset Summary:")
        print(f"   - Total records: {len(df)}")
        print(f"   - Products: {df['product'].value_counts().to_dict()}")
        print(f"   - Regions: {df['region'].value_counts().to_dict()}")
        print(f"   - Volume range: {df['volume_ml'].min():.1f} - {df['volume_ml'].max():.1f} ml")
        print(f"   - Age range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during data extraction: {e}")
        return 1

if __name__ == "__main__":
    exit(main())