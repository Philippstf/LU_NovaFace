#!/usr/bin/env python3
"""
Airtable Data Import Script for Nuva Face
==========================================

Dieses Script lädt Daten von Airtable herunter und bereitet sie für das Training vor.

Usage:
    python scripts/import_airtable_data.py
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from data.airtable_extractor import AirtableExtractor
import pandas as pd


def main():
    """Main function to import Airtable data."""
    print("🚀 Starting Airtable data import...")
    
    try:
        # Initialize extractor
        print("📡 Connecting to Airtable...")
        extractor = AirtableExtractor()
        
        # Extract all data
        print("📥 Downloading data and images...")
        df = extractor.extract_all_data(data_dir="./data")
        
        print(f"✅ Successfully imported {len(df)} records!")
        print(f"📊 Data saved to: ./data/raw/metadata.csv")
        print(f"🖼️  Images saved to: ./data/raw/before/ and ./data/raw/after/")
        
        # Show data preview
        print("\n📋 Data Preview:")
        print("="*50)
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print("\nFirst few records:")
        print(df[['record_id', 'gender', 'region', 'product', 'volume_ml']].head())
        
        # Data quality check
        print("\n🔍 Data Quality Check:")
        print("="*50)
        print(f"Records with both images: {len(df)}")
        print(f"Missing values:")
        for col in ['gender', 'region', 'product', 'volume_ml']:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing} missing")
            
        print("\n🎯 Next Steps:")
        print("1. Check your .env file has correct Airtable credentials")
        print("2. Review the downloaded images in ./data/raw/")
        print("3. Run data preprocessing: python scripts/preprocess_data.py")
        
    except ValueError as e:
        print("❌ Configuration Error:")
        print(f"   {e}")
        print("\n🔧 Solution:")
        print("1. Copy .env.example to .env")
        print("2. Add your Airtable credentials to .env")
        print("3. Ask the person who gave you access for:")
        print("   - API Key (Personal Access Token)")
        print("   - Base ID (from the URL)")
        print("   - Table Name")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🆘 Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify Airtable credentials")
        print("3. Ensure you have access to the shared base")


if __name__ == "__main__":
    main()