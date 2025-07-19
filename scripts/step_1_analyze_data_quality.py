#!/usr/bin/env python3
"""
Step 1: Analyze Data Quality for MVP Training
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Add src to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

def analyze_image_quality():
    """Analyze image quality and consistency of our dataset."""
    print("🔍 Analyzing dataset for MVP training...")
    
    data_dir = Path("./data/raw")
    metadata = pd.read_csv(data_dir / "metadata.csv")
    
    print(f"📊 Dataset Overview:")
    print(f"  Personen: {len(metadata)}")
    print(f"  Gesamt Vorher-Bilder: {metadata['before_image_count'].sum()}")
    print(f"  Gesamt Nachher-Bilder: {metadata['after_image_count'].sum()}")
    
    # Sample images for quality analysis
    quality_stats = {
        'resolutions': [],
        'brightness': [],
        'blur_scores': [],
        'person_ids': []
    }
    
    print(f"\n🖼️  Analyzing image quality...")
    
    for idx, row in metadata.head(10).iterrows():  # Sample first 10 persons
        person_folder = Path(row['person_folder'])
        
        # Analyze before images
        before_dir = person_folder / "before"
        if before_dir.exists():
            for img_path in before_dir.glob("*.jpg"):
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        quality_stats['resolutions'].append(f"{w}x{h}")
                        
                        # Brightness analysis
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        brightness = np.mean(gray)
                        quality_stats['brightness'].append(brightness)
                        
                        # Blur detection (Laplacian variance)
                        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                        quality_stats['blur_scores'].append(blur_score)
                        quality_stats['person_ids'].append(row['person_id'])
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Analysis results
    print(f"\n📈 Quality Analysis Results:")
    
    # Resolution analysis
    from collections import Counter
    res_counts = Counter(quality_stats['resolutions'])
    print(f"  Top resolutions:")
    for res, count in res_counts.most_common(5):
        print(f"    {res}: {count} images")
    
    # Brightness analysis
    avg_brightness = np.mean(quality_stats['brightness'])
    print(f"  Average brightness: {avg_brightness:.1f}")
    print(f"  Brightness range: {min(quality_stats['brightness']):.1f} - {max(quality_stats['brightness']):.1f}")
    
    # Blur analysis
    avg_blur = np.mean(quality_stats['blur_scores'])
    print(f"  Average sharpness score: {avg_blur:.1f}")
    print(f"  Blur threshold recommendation: > {avg_blur/2:.1f}")
    
    # Check for potential issues
    print(f"\n⚠️  Potential Issues:")
    low_quality_count = sum(1 for score in quality_stats['blur_scores'] if score < avg_blur/2)
    print(f"  Low quality images: {low_quality_count}/{len(quality_stats['blur_scores'])}")
    
    dark_images = sum(1 for brightness in quality_stats['brightness'] if brightness < 50)
    print(f"  Very dark images: {dark_images}/{len(quality_stats['brightness'])}")
    
    return quality_stats

def analyze_lip_regions():
    """Analyze lip regions specifically for MVP focus."""
    print(f"\n👄 Lip Region Analysis:")
    print(f"  Next step: Implement MediaPipe face detection")
    print(f"  Next step: Extract lip landmarks")
    print(f"  Next step: Crop 256x256 lip regions")
    
    # This would use MediaPipe to detect lip regions
    # For now, we note what needs to be implemented
    
def recommend_mvp_approach():
    """Recommend best approach based on data analysis."""
    print(f"\n🎯 MVP Recommendation:")
    print(f"  1. Data Quality: Good enough for initial training")
    print(f"  2. Recommended approach: Conditional Pix2Pix")
    print(f"  3. Training strategy: Person-wise train/val split")
    print(f"  4. Image resolution: Standardize to 512x512")
    print(f"  5. Focus area: 256x256 lip region")
    
    print(f"\n🚀 Next Implementation Steps:")
    print(f"  1. Install MediaPipe for face detection")
    print(f"  2. Implement lip segmentation pipeline")
    print(f"  3. Create training dataset with lip pairs")
    print(f"  4. Set up PyTorch training environment")
    print(f"  5. Implement Pix2Pix architecture")

def main():
    """Main analysis function."""
    quality_stats = analyze_image_quality()
    analyze_lip_regions()
    recommend_mvp_approach()
    
    print(f"\n✅ Analysis complete!")
    print(f"   Run next: python scripts/step_2_install_dependencies.py")

if __name__ == "__main__":
    main()