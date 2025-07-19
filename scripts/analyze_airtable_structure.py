#!/usr/bin/env python3
"""
Analyze Airtable structure to understand multiple images per person
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from data.airtable_extractor import AirtableExtractor
import pandas as pd


def analyze_image_structure():
    """Analyze how many images each person has."""
    print("🔍 Analyzing Airtable image structure...")
    
    try:
        extractor = AirtableExtractor()
        records = extractor.fetch_records()
        
        print(f"📊 Found {len(records)} total records")
        print("\n🖼️  Image analysis per person:")
        print("="*60)
        
        for i, record in enumerate(records[:10]):  # Show first 10 for analysis
            fields = record.get('fields', {})
            record_id = record['id']
            
            before_images = fields.get('Vorher-Foto', [])
            after_images = fields.get('Nachher-Foto', [])
            
            print(f"Person {i+1} (ID: {record_id[:10]}...):")
            print(f"  Vorher-Bilder: {len(before_images)}")
            print(f"  Nachher-Bilder: {len(after_images)}")
            
            # Show image filenames
            if before_images:
                print(f"  Vorher-Dateien: {[img.get('filename', 'unknown') for img in before_images[:3]]}{'...' if len(before_images) > 3 else ''}")
            if after_images:
                print(f"  Nachher-Dateien: {[img.get('filename', 'unknown') for img in after_images[:3]]}{'...' if len(after_images) > 3 else ''}")
            print()
            
        # Overall statistics
        total_before = sum(len(record.get('fields', {}).get('Vorher-Foto', [])) for record in records)
        total_after = sum(len(record.get('fields', {}).get('Nachher-Foto', [])) for record in records)
        
        print(f"📈 Gesamtstatistik:")
        print(f"  Personen insgesamt: {len(records)}")
        print(f"  Vorher-Bilder insgesamt: {total_before}")
        print(f"  Nachher-Bilder insgesamt: {total_after}")
        print(f"  Durchschnitt Vorher/Person: {total_before/len(records):.1f}")
        print(f"  Durchschnitt Nachher/Person: {total_after/len(records):.1f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    analyze_image_structure()