#!/usr/bin/env python3
"""
Lip Enhancement Engine - Transform customer's actual image
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from typing import Tuple, List, Dict, Optional
import json

class LipEnhancer:
    """
    Enhances lip appearance in customer's own image using template guidance.
    """
    
    def __init__(self):
        self.lip_landmark_indices = list(range(48, 68))  # MediaPipe lip landmarks
        
    def enhance_customer_lips(self, 
                             input_image_path: str,
                             template_guidance: Dict,
                             volume_ml: float = 1.0,
                             product_type: str = "restylane_kysse") -> str:
        """
        Main function to enhance customer's lips using template guidance.
        """
        
        # Load and process input image
        image = Image.open(input_image_path)
        image_array = np.array(image)
        
        # For MVP: Use simplified approach without MediaPipe
        # (MediaPipe would require large dependencies)
        enhanced_image = self._enhance_lips_geometric(
            image_array, 
            volume_ml, 
            product_type,
            template_guidance
        )
        
        # Save enhanced image
        output_path = input_image_path.replace('.jpg', '_enhanced.jpg')
        enhanced_pil = Image.fromarray(enhanced_image.astype(np.uint8))
        enhanced_pil.save(output_path, quality=95)
        
        return output_path
    
    def _enhance_lips_geometric(self, 
                               image: np.ndarray,
                               volume_ml: float,
                               product_type: str,
                               template_guidance: Dict) -> np.ndarray:
        """
        Geometric lip enhancement without face detection.
        Uses center-region detection and radial expansion.
        """
        
        height, width = image.shape[:2]
        
        # Estimate lip region (center-lower part of image)
        # This is a simplified approach - in production we'd use MediaPipe
        lip_center_y = int(height * 0.75)  # Assume lips are at 75% down
        lip_center_x = int(width * 0.5)    # Center horizontally
        
        # Define lip region bounds
        lip_width = int(width * 0.25)      # Lips ~25% of image width
        lip_height = int(height * 0.08)    # Lips ~8% of image height
        
        # Create enhancement mask
        enhancement_mask = self._create_lip_enhancement_mask(
            (height, width),
            (lip_center_x, lip_center_y),
            (lip_width, lip_height),
            volume_ml
        )
        
        # Apply geometric transformation
        enhanced_image = self._apply_lip_warping(
            image,
            enhancement_mask,
            volume_ml
        )
        
        # Apply color/texture enhancements
        enhanced_image = self._apply_product_effects(
            enhanced_image,
            enhancement_mask,
            product_type
        )
        
        return enhanced_image
    
    def _create_lip_enhancement_mask(self,
                                   image_shape: Tuple[int, int],
                                   center: Tuple[int, int],
                                   size: Tuple[int, int],
                                   volume_ml: float) -> np.ndarray:
        """
        Create a smooth enhancement mask for lip region.
        """
        height, width = image_shape
        center_x, center_y = center
        lip_width, lip_height = size
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Create elliptical lip mask
        ellipse_mask = (
            ((x - center_x) / lip_width) ** 2 + 
            ((y - center_y) / lip_height) ** 2
        ) <= 1.0
        
        # Create smooth falloff
        distance_from_center = np.sqrt(
            ((x - center_x) / lip_width) ** 2 + 
            ((y - center_y) / lip_height) ** 2
        )
        
        # Enhancement strength based on volume
        max_enhancement = volume_ml * 0.3  # 1ml = 30% max enhancement
        
        # Smooth falloff from center to edge
        enhancement_strength = np.where(
            distance_from_center <= 1.0,
            max_enhancement * (1.0 - distance_from_center ** 2),
            0.0
        )
        
        return enhancement_strength
    
    def _apply_lip_warping(self,
                          image: np.ndarray,
                          enhancement_mask: np.ndarray,
                          volume_ml: float) -> np.ndarray:
        """
        Apply geometric warping to enlarge lips.
        """
        height, width = image.shape[:2]
        
        # Find center of enhancement
        enhancement_indices = np.where(enhancement_mask > 0)
        if len(enhancement_indices[0]) == 0:
            return image  # No enhancement region found
        
        center_y = int(np.mean(enhancement_indices[0]))
        center_x = int(np.mean(enhancement_indices[1]))
        
        # Create displacement field
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Calculate displacement vectors
        dx = x_coords - center_x
        dy = y_coords - center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Avoid division by zero
        distance = np.maximum(distance, 1e-6)
        
        # Displacement strength based on enhancement mask
        displacement_strength = enhancement_mask * volume_ml * 0.1
        
        # Calculate new coordinates (expand outward)
        new_x = x_coords - dx * displacement_strength / distance
        new_y = y_coords - dy * displacement_strength / distance
        
        # Ensure coordinates are within bounds
        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)
        
        # Apply warping using cv2.remap
        warped_image = cv2.remap(
            image,
            new_x.astype(np.float32),
            new_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return warped_image
    
    def _apply_product_effects(self,
                              image: np.ndarray,
                              enhancement_mask: np.ndarray,
                              product_type: str) -> np.ndarray:
        """
        Apply product-specific color and texture effects.
        """
        
        # Convert to PIL for easier manipulation
        pil_image = Image.fromarray(image)
        
        # Create mask for PIL operations
        mask_pil = Image.fromarray((enhancement_mask * 255).astype(np.uint8))
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=2))
        
        if product_type.lower() in ['restylane_kysse', 'restylane']:
            # Restylane Kysse: Natural enhancement with subtle shine
            enhanced = self._apply_restylane_effect(pil_image, mask_pil)
            
        elif product_type.lower() in ['redensity', 'redensity_ii']:
            # Redensity: More natural, less glossy
            enhanced = self._apply_redensity_effect(pil_image, mask_pil)
            
        elif product_type.lower() in ['teoxane', 'teoxane_lips']:
            # Teoxane: Smoother, more defined
            enhanced = self._apply_teoxane_effect(pil_image, mask_pil)
            
        else:
            # Default enhancement
            enhanced = self._apply_default_effect(pil_image, mask_pil)
        
        return np.array(enhanced)
    
    def _apply_restylane_effect(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Apply Restylane Kysse specific effects."""
        
        # Slight color enhancement (more rosy)
        enhancer = ImageEnhance.Color(image)
        color_enhanced = enhancer.enhance(1.1)  # 10% more saturation
        
        # Slight brightness increase (natural glow)
        enhancer = ImageEnhance.Brightness(color_enhanced)
        bright_enhanced = enhancer.enhance(1.05)  # 5% brighter
        
        # Blend with original using mask
        result = Image.composite(bright_enhanced, image, mask)
        
        return result
    
    def _apply_redensity_effect(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Apply Redensity II specific effects."""
        
        # Very subtle enhancement - more natural
        enhancer = ImageEnhance.Color(image)
        color_enhanced = enhancer.enhance(1.05)  # 5% more saturation
        
        result = Image.composite(color_enhanced, image, mask)
        return result
    
    def _apply_teoxane_effect(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Apply Teoxane Lips specific effects."""
        
        # Smoother appearance with slight gloss
        # Apply slight blur for smoothness
        smoothed = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Enhance brightness more than Restylane
        enhancer = ImageEnhance.Brightness(smoothed)
        bright_enhanced = enhancer.enhance(1.08)  # 8% brighter
        
        result = Image.composite(bright_enhanced, image, mask)
        return result
    
    def _apply_default_effect(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Apply default enhancement effects."""
        
        enhancer = ImageEnhance.Color(image)
        enhanced = enhancer.enhance(1.1)
        
        result = Image.composite(enhanced, image, mask)
        return result
    
    def create_before_after_comparison(self,
                                     original_path: str,
                                     enhanced_path: str,
                                     output_path: str) -> str:
        """
        Create a side-by-side before/after comparison image.
        """
        
        original = Image.open(original_path)
        enhanced = Image.open(enhanced_path)
        
        # Resize to consistent size
        size = (400, 400)
        original_resized = original.resize(size, Image.Resampling.LANCZOS)
        enhanced_resized = enhanced.resize(size, Image.Resampling.LANCZOS)
        
        # Create comparison image
        comparison = Image.new('RGB', (800, 400), 'white')
        comparison.paste(original_resized, (0, 0))
        comparison.paste(enhanced_resized, (400, 0))
        
        # Save comparison
        comparison.save(output_path, quality=95)
        
        return output_path


def main():
    """Test the lip enhancer."""
    print("🧪 Testing Lip Enhancement...")
    
    enhancer = LipEnhancer()
    
    # Test with a sample image (if available)
    sample_images = list(Path("./data/raw").glob("*/before/*.jpg"))
    
    if sample_images:
        sample_image = str(sample_images[0])
        print(f"Testing with: {sample_image}")
        
        # Create mock template guidance
        template_guidance = {
            'person_id': 'test',
            'product': 'restylane_kysse',
            'volume_ml': 1.0
        }
        
        try:
            enhanced_path = enhancer.enhance_customer_lips(
                sample_image,
                template_guidance,
                volume_ml=1.0,
                product_type="restylane_kysse"
            )
            
            print(f"✅ Enhanced image saved: {enhanced_path}")
            
            # Create comparison
            comparison_path = enhanced_path.replace('_enhanced.jpg', '_comparison.jpg')
            enhancer.create_before_after_comparison(
                sample_image,
                enhanced_path,
                comparison_path
            )
            
            print(f"📊 Comparison saved: {comparison_path}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("No sample images found in ./data/raw/*/before/")


if __name__ == "__main__":
    from pathlib import Path
    main()