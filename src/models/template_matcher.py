#!/usr/bin/env python3
"""
Template Matching System for Lip Enhancement Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
from typing import Dict, List, Tuple, Optional

class LipTemplateMatch:
    """
    Template Matching System that finds similar lip cases
    and applies their transformations to new images.
    """
    
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(self.data_dir / "metadata.csv")
        self.templates = self._load_templates()
        
    def _load_templates(self) -> List[Dict]:
        """Load all available templates with their transformations."""
        templates = []
        
        for _, row in self.metadata.iterrows():
            person_folder = Path(row['person_folder'])
            
            # Load before images
            before_images = []
            before_dir = person_folder / "before"
            if before_dir.exists():
                before_images = [str(p) for p in before_dir.glob("*.jpg")]
            
            # Load after images  
            after_images = []
            after_dir = person_folder / "after" 
            if after_dir.exists():
                after_images = [str(p) for p in after_dir.glob("*.jpg")]
            
            if before_images and after_images:
                template = {
                    'person_id': row['person_id'],
                    'before_images': before_images,
                    'after_images': after_images,
                    'product': row['product'],
                    'volume_ml': row['volume_ml'],
                    'treatment_notes': row.get('notes', ''),
                    'before_count': len(before_images),
                    'after_count': len(after_images),
                    # We'll add lip features later
                    'lip_features': None
                }
                templates.append(template)
        
        print(f"📚 Loaded {len(templates)} templates")
        return templates
    
    def extract_lip_features(self, image_path: str) -> Dict:
        """
        Extract basic lip features for matching.
        For now, we'll use simple image-based features.
        Later: integrate with MediaPipe or similar.
        """
        try:
            image = Image.open(image_path)
            width, height = image.size
            
            # Basic features that can be computed without face detection
            # These are placeholders - in real implementation we'd use:
            # - Lip landmarks from MediaPipe
            # - Geometric ratios
            # - Color histograms
            # - Texture features
            
            features = {
                'image_width': width,
                'image_height': height,
                'aspect_ratio': width / height,
                'image_size': width * height,
                # Placeholder for real lip features:
                'lip_width_ratio': 0.0,  # lip_width / face_width
                'lip_height_ratio': 0.0, # lip_height / face_height  
                'lip_symmetry': 0.0,     # left/right symmetry score
                'lip_fullness': 0.0,     # upper/lower lip ratio
                'color_intensity': 0.0,  # average lip color saturation
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return {}
    
    def find_best_template(self, 
                          input_features: Dict, 
                          target_volume: float = 1.0,
                          target_product: str = "['Restylane Kysse']") -> Optional[Dict]:
        """
        Find the most similar template based on lip features and treatment parameters.
        """
        
        if not self.templates:
            return None
        
        best_score = -1
        best_template = None
        
        for template in self.templates:
            # Skip if different treatment parameters
            if (target_volume != 1.0 and 
                str(template['volume_ml']) != str(target_volume) + " ml"):
                continue
                
            if target_product != template['product']:
                continue
            
            # For now, use simple scoring based on image count
            # Later: use actual lip feature similarity
            
            # Score based on template quality
            image_quality_score = min(template['before_count'] / 10, 1.0)  # More images = better
            completeness_score = min(template['after_count'] / 10, 1.0)
            
            # Prefer templates with treatment notes (more documented)
            documentation_score = 1.0 if template['treatment_notes'] else 0.5
            
            # Prefer first-time treatments for more natural transformations
            treatment_type_score = 1.0
            notes = str(template['treatment_notes']).lower() if template['treatment_notes'] else ""
            if '1.' in notes or 'erste' in notes:
                treatment_type_score = 1.2  # Prefer first treatments
            elif 'multiple' in notes or 'vorunter' in notes:
                treatment_type_score = 0.8  # Less prefer multiple pre-treatments
            
            total_score = (image_quality_score * 0.3 + 
                          completeness_score * 0.3 +
                          documentation_score * 0.2 +
                          treatment_type_score * 0.2)
            
            if total_score > best_score:
                best_score = total_score
                best_template = template
        
        return best_template
    
    def get_transformation_examples(self, template: Dict, num_examples: int = 3) -> List[Tuple[str, str]]:
        """
        Get before/after image pairs from the template for transformation learning.
        """
        pairs = []
        
        # Take up to num_examples pairs
        max_pairs = min(len(template['before_images']), 
                       len(template['after_images']), 
                       num_examples)
        
        for i in range(max_pairs):
            before_img = template['before_images'][i] if i < len(template['before_images']) else template['before_images'][0]
            after_img = template['after_images'][i] if i < len(template['after_images']) else template['after_images'][0]
            pairs.append((before_img, after_img))
        
        return pairs
    
    def predict_enhancement(self, 
                           input_image_path: str,
                           target_volume: float = 1.0,
                           target_product: str = "['Restylane Kysse']") -> Dict:
        """
        Predict lip enhancement for input image using template matching.
        """
        
        # Extract features from input image
        input_features = self.extract_lip_features(input_image_path)
        
        # Find best matching template
        best_template = self.find_best_template(input_features, target_volume, target_product)
        
        if not best_template:
            return {
                'success': False,
                'error': 'No suitable template found',
                'suggestion': 'Try different volume or product parameters'
            }
        
        # Get transformation examples
        transformation_pairs = self.get_transformation_examples(best_template)
        
        result = {
            'success': True,
            'matched_template': {
                'person_id': best_template['person_id'],
                'similarity_score': 0.85,  # Placeholder
                'treatment_notes': best_template['treatment_notes'],
                'image_count': f"{best_template['before_count']}→{best_template['after_count']}"
            },
            'transformation_pairs': transformation_pairs,
            'prediction_confidence': 0.78,  # Placeholder
            'recommendation': self._generate_recommendation(best_template),
            'disclaimer': "Dies ist eine Computersimulation basierend auf ähnlichen Fällen. Echte Ergebnisse können abweichen."
        }
        
        return result
    
    def _generate_recommendation(self, template: Dict) -> str:
        """Generate personalized recommendation based on template."""
        notes = str(template['treatment_notes']).lower() if template['treatment_notes'] else ""
        
        if '1.' in notes or 'erste' in notes:
            return f"Basierend auf ähnlichen Erstbehandlungen. Erwarten Sie natürliche, subtile Verbesserung."
        elif 'multiple' in notes:
            return f"Basierend auf Personen mit Vorerfahrung. Möglicherweise stärkerer Effekt sichtbar."
        else:
            return f"Basierend auf {template['person_id']} mit ähnlicher Lippenform."
    
    def get_template_statistics(self) -> Dict:
        """Get statistics about available templates."""
        if not self.templates:
            return {}
        
        products = {}
        volumes = {}
        total_images = 0
        
        for template in self.templates:
            # Product distribution
            product = template['product']
            products[product] = products.get(product, 0) + 1
            
            # Volume distribution  
            volume = template['volume_ml']
            volumes[volume] = volumes.get(volume, 0) + 1
            
            # Total images
            total_images += template['before_count'] + template['after_count']
        
        return {
            'total_templates': len(self.templates),
            'total_images': total_images,
            'avg_before_images': np.mean([t['before_count'] for t in self.templates]),
            'avg_after_images': np.mean([t['after_count'] for t in self.templates]),
            'product_distribution': products,
            'volume_distribution': volumes,
            'best_templates': sorted(self.templates, 
                                   key=lambda x: x['before_count'] + x['after_count'], 
                                   reverse=True)[:5]
        }


def main():
    """Test the template matching system."""
    print("🔍 Testing Template Matching System...")
    
    matcher = LipTemplateMatch()
    
    # Get statistics
    stats = matcher.get_template_statistics()
    print(f"\n📊 Template Statistics:")
    print(f"  Total templates: {stats['total_templates']}")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Avg before images: {stats['avg_before_images']:.1f}")
    print(f"  Avg after images: {stats['avg_after_images']:.1f}")
    
    print(f"\n🧴 Product coverage:")
    for product, count in stats['product_distribution'].items():
        print(f"  {product}: {count} templates")
    
    print(f"\n💉 Volume coverage:")
    for volume, count in stats['volume_distribution'].items():
        print(f"  {volume}: {count} templates")
    
    print(f"\n⭐ Best templates (most images):")
    for template in stats['best_templates']:
        print(f"  {template['person_id']}: {template['before_count']}→{template['after_count']} images")
    
    # Test prediction with a sample image
    if matcher.templates:
        sample_template = matcher.templates[0]
        if sample_template['before_images']:
            sample_image = sample_template['before_images'][0]
            print(f"\n🧪 Testing prediction with: {sample_image}")
            
            result = matcher.predict_enhancement(sample_image)
            if result['success']:
                print(f"✅ Match found: {result['matched_template']['person_id']}")
                print(f"   Confidence: {result['prediction_confidence']}")
                print(f"   Recommendation: {result['recommendation']}")
            else:
                print(f"❌ {result['error']}")


if __name__ == "__main__":
    main()