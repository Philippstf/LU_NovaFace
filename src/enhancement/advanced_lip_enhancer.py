#!/usr/bin/env python3
"""
Advanced Lip Enhancement Engine with better detection and more realistic results
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import cv2
from typing import Tuple, List, Dict, Optional
import json

class AdvancedLipEnhancer:
    """
    Advanced lip enhancement using better detection and more realistic warping.
    """
    
    def __init__(self):
        self.face_cascade = None
        self.mouth_cascade = None
        self._load_cascades()
        
    def _load_cascades(self):
        """Load OpenCV Haar cascades for face/mouth detection."""
        try:
            # Try to load pre-trained cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            # For mouth, we'll use a simple approach since mouth cascade is less reliable
        except Exception as e:
            print(f"Warning: Could not load face cascades: {e}")
            print("Falling back to geometric estimation")
    
    def enhance_customer_lips(self, 
                             input_image_path: str,
                             template_guidance: Dict,
                             volume_ml: float = 1.0,
                             product_type: str = "restylane_kysse") -> str:
        """
        Enhanced lip enhancement with better detection and realism.
        """
        
        # Load and process input image
        image = Image.open(input_image_path)
        image_array = np.array(image)
        
        # Detect face and lip region more accurately
        face_region, lip_region = self._detect_face_and_lips(image_array)
        
        if face_region is None:
            print("⚠️  Could not detect face, using fallback method")
            enhanced_image = self._enhance_lips_fallback(image_array, volume_ml, product_type)
        else:
            print(f"✅ Face detected at {face_region}, lip region: {lip_region}")
            enhanced_image = self._enhance_lips_advanced(
                image_array, 
                face_region,
                lip_region,
                volume_ml, 
                product_type,
                template_guidance
            )
        
        # Save enhanced image
        output_path = input_image_path.replace('.jpg', '_enhanced.jpg')
        enhanced_pil = Image.fromarray(enhanced_image.astype(np.uint8))
        enhanced_pil.save(output_path, quality=95)
        
        return output_path
    
    def _detect_face_and_lips(self, image: np.ndarray) -> Tuple[Optional[Tuple], Optional[Tuple]]:
        """
        Detect face and estimate lip region using OpenCV.
        """
        if self.face_cascade is None:
            return None, None
            
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(100, 100)
            )
            
            if len(faces) == 0:
                return None, None
            
            # Take the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            
            # Estimate lip region within face
            # Lips are typically in the lower 1/3 of face, centered horizontally
            lip_y_start = y + int(h * 0.65)  # Start at 65% down the face
            lip_y_end = y + int(h * 0.85)    # End at 85% down the face
            lip_x_start = x + int(w * 0.25)  # Start at 25% from left
            lip_x_end = x + int(w * 0.75)    # End at 75% from left
            
            face_region = (x, y, w, h)
            lip_region = (lip_x_start, lip_y_start, lip_x_end - lip_x_start, lip_y_end - lip_y_start)
            
            return face_region, lip_region
            
        except Exception as e:
            print(f"Face detection failed: {e}")
            return None, None
    
    def _enhance_lips_advanced(self,
                              image: np.ndarray,
                              face_region: Tuple[int, int, int, int],
                              lip_region: Tuple[int, int, int, int],
                              volume_ml: float,
                              product_type: str,
                              template_guidance: Dict) -> np.ndarray:
        """
        Advanced lip enhancement using detected face/lip regions.
        """
        
        height, width = image.shape[:2]
        face_x, face_y, face_w, face_h = face_region
        lip_x, lip_y, lip_w, lip_h = lip_region
        
        # Calculate enhancement parameters based on volume
        # More conservative scaling for realism
        max_expansion = min(volume_ml * 0.15, 0.3)  # Max 30% expansion even for 2ml
        
        # Create more sophisticated enhancement mask
        enhancement_mask = self._create_sophisticated_lip_mask(
            (height, width),
            lip_region,
            max_expansion
        )
        
        # Apply warping with face-aware constraints
        warped_image = self._apply_face_aware_warping(
            image,
            enhancement_mask,
            face_region,
            lip_region,
            volume_ml
        )
        
        # Apply subtle product effects
        enhanced_image = self._apply_subtle_product_effects(
            warped_image,
            enhancement_mask,
            product_type,
            volume_ml
        )
        
        return enhanced_image
    
    def _create_sophisticated_lip_mask(self,
                                     image_shape: Tuple[int, int],
                                     lip_region: Tuple[int, int, int, int],
                                     max_expansion: float) -> np.ndarray:
        """
        Create a more sophisticated lip enhancement mask.
        """
        height, width = image_shape
        lip_x, lip_y, lip_w, lip_h = lip_region
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Create lip center
        center_x = lip_x + lip_w // 2
        center_y = lip_y + lip_h // 2
        
        # Create elliptical lip mask with more realistic proportions
        # Lips are wider than they are tall
        ellipse_w = lip_w * 0.6  # Slightly smaller than detected region
        ellipse_h = lip_h * 0.8  # More vertical coverage
        
        # Distance from center (elliptical)
        distance = (
            ((x - center_x) / ellipse_w) ** 2 + 
            ((y - center_y) / ellipse_h) ** 2
        )
        
        # Create smooth falloff mask
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Inner region (strongest enhancement)
        inner_mask = distance <= 0.3
        mask[inner_mask] = max_expansion
        
        # Transition region (smooth falloff)
        transition_mask = (distance > 0.3) & (distance <= 1.0)
        transition_strength = 1.0 - (distance - 0.3) / 0.7  # Linear falloff
        mask[transition_mask] = max_expansion * transition_strength[transition_mask]
        
        # Smooth the mask to avoid artifacts
        mask = cv2.GaussianBlur(mask, (21, 21), 5)
        
        return mask
    
    def _apply_face_aware_warping(self,
                                 image: np.ndarray,
                                 enhancement_mask: np.ndarray,
                                 face_region: Tuple[int, int, int, int],
                                 lip_region: Tuple[int, int, int, int],
                                 volume_ml: float) -> np.ndarray:
        """
        Apply warping that's aware of face structure to avoid distortion.
        """
        height, width = image.shape[:2]
        face_x, face_y, face_w, face_h = face_region
        lip_x, lip_y, lip_w, lip_h = lip_region
        
        # Create displacement field
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        # Calculate lip center
        center_x = lip_x + lip_w // 2
        center_y = lip_y + lip_h // 2
        
        # Calculate displacement vectors (radial expansion from lip center)
        dx = x_coords - center_x
        dy = y_coords - center_y
        distance = np.sqrt(dx**2 + dy**2)
        distance = np.maximum(distance, 1e-6)  # Avoid division by zero
        
        # Scale displacement based on enhancement mask and distance
        displacement_strength = enhancement_mask * 0.5  # Reduced strength for realism
        
        # Limit displacement to face region only
        face_mask = (
            (x_coords >= face_x) & (x_coords < face_x + face_w) &
            (y_coords >= face_y) & (y_coords < face_y + face_h)
        )
        displacement_strength = displacement_strength * face_mask
        
        # Calculate new coordinates (expand outward from lip center)
        new_x = x_coords - dx * displacement_strength / distance
        new_y = y_coords - dy * displacement_strength / distance
        
        # Ensure coordinates are within bounds
        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)
        
        # Apply warping using more sophisticated interpolation
        warped_image = cv2.remap(
            image,
            new_x.astype(np.float32),
            new_y.astype(np.float32),
            cv2.INTER_CUBIC,  # Better interpolation
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return warped_image
    
    def _apply_subtle_product_effects(self,
                                    image: np.ndarray,
                                    enhancement_mask: np.ndarray,
                                    product_type: str,
                                    volume_ml: float) -> np.ndarray:
        """
        Apply very subtle product-specific effects.
        """
        
        # Convert to PIL for processing
        pil_image = Image.fromarray(image)
        
        # Create mask for blending
        mask_pil = Image.fromarray((enhancement_mask * 255).astype(np.uint8))
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=3))
        
        # Very subtle enhancements
        if 'restylane' in product_type.lower():
            # Restylane: Slight increase in saturation and smoothness
            enhancer = ImageEnhance.Color(pil_image)
            color_enhanced = enhancer.enhance(1.03)  # Only 3% more saturation
            
            enhancer = ImageEnhance.Brightness(color_enhanced)
            enhanced = enhancer.enhance(1.02)  # Only 2% brighter
            
        elif 'redensity' in product_type.lower():
            # Redensity: Very natural, minimal change
            enhancer = ImageEnhance.Color(pil_image)
            enhanced = enhancer.enhance(1.01)  # Only 1% more saturation
            
        elif 'teoxane' in product_type.lower():
            # Teoxane: Slight smoothing effect
            smoothed = pil_image.filter(ImageFilter.GaussianBlur(radius=0.3))
            enhancer = ImageEnhance.Brightness(smoothed)
            enhanced = enhancer.enhance(1.03)
            
        else:
            enhanced = pil_image
        
        # Blend very subtly with original
        result = Image.composite(enhanced, pil_image, mask_pil)
        
        return np.array(result)
    
    def _enhance_lips_fallback(self,
                              image: np.ndarray,
                              volume_ml: float,
                              product_type: str) -> np.ndarray:
        """
        Fallback method when face detection fails - improved version.
        """
        height, width = image.shape[:2]
        
        # More conservative estimation
        estimated_lip_center_y = int(height * 0.72)  # Slightly higher
        estimated_lip_center_x = int(width * 0.5)
        
        # Smaller estimated lip region
        lip_width = int(width * 0.15)   # Reduced from 0.25
        lip_height = int(height * 0.04) # Reduced from 0.08
        
        # Create more conservative enhancement
        enhancement_mask = self._create_conservative_mask(
            (height, width),
            (estimated_lip_center_x, estimated_lip_center_y),
            (lip_width, lip_height),
            volume_ml
        )
        
        # Apply very gentle warping
        enhanced = self._apply_gentle_warping(image, enhancement_mask, volume_ml)
        
        return enhanced
    
    def _create_conservative_mask(self,
                                image_shape: Tuple[int, int],
                                center: Tuple[int, int],
                                size: Tuple[int, int],
                                volume_ml: float) -> np.ndarray:
        """
        Create a very conservative enhancement mask for fallback.
        """
        height, width = image_shape
        center_x, center_y = center
        lip_width, lip_height = size
        
        y, x = np.ogrid[:height, :width]
        
        # Much more conservative enhancement
        max_enhancement = volume_ml * 0.08  # Reduced from 0.3
        
        # Smaller ellipse
        distance = (
            ((x - center_x) / lip_width) ** 2 + 
            ((y - center_y) / lip_height) ** 2
        )
        
        # Only enhance very close to center
        enhancement = np.where(
            distance <= 0.5,  # Smaller radius
            max_enhancement * (1.0 - distance * 2),  # Faster falloff
            0.0
        )
        
        # Heavy smoothing
        enhancement = cv2.GaussianBlur(enhancement.astype(np.float32), (31, 31), 8)
        
        return enhancement
    
    def _apply_gentle_warping(self,
                            image: np.ndarray,
                            enhancement_mask: np.ndarray,
                            volume_ml: float) -> np.ndarray:
        """
        Apply very gentle warping for fallback method.
        """
        height, width = image.shape[:2]
        
        # Find enhancement center
        enhancement_indices = np.where(enhancement_mask > 0)
        if len(enhancement_indices[0]) == 0:
            return image
        
        center_y = int(np.mean(enhancement_indices[0]))
        center_x = int(np.mean(enhancement_indices[1]))
        
        # Create very gentle displacement
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        
        dx = x_coords - center_x
        dy = y_coords - center_y
        distance = np.sqrt(dx**2 + dy**2)
        distance = np.maximum(distance, 1e-6)
        
        # Much gentler displacement
        displacement_strength = enhancement_mask * 0.2  # Very gentle
        
        new_x = x_coords - dx * displacement_strength / distance
        new_y = y_coords - dy * displacement_strength / distance
        
        new_x = np.clip(new_x, 0, width - 1)
        new_y = np.clip(new_y, 0, height - 1)
        
        # Use linear interpolation for smoother result
        warped = cv2.remap(
            image,
            new_x.astype(np.float32),
            new_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return warped
    
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
        
        # Create comparison image with labels
        comparison = Image.new('RGB', (820, 450), 'white')
        
        # Paste images
        comparison.paste(original_resized, (10, 40))
        comparison.paste(enhanced_resized, (410, 40))
        
        # Add labels
        draw = ImageDraw.Draw(comparison)
        try:
            # Try to use a better font if available
            from PIL import ImageFont
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = None
        
        draw.text((150, 10), "VORHER", fill='black', font=font, anchor="mm")
        draw.text((550, 10), "NACHHER", fill='black', font=font, anchor="mm")
        
        comparison.save(output_path, quality=95)
        
        return output_path


def main():
    """Test the advanced lip enhancer."""
    print("🧪 Testing Advanced Lip Enhancement...")
    
    enhancer = AdvancedLipEnhancer()
    
    # Test with a sample image
    from pathlib import Path
    sample_images = list(Path("./data/raw").glob("*/before/*.jpg"))
    
    if sample_images:
        sample_image = str(sample_images[0])
        print(f"Testing with: {sample_image}")
        
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
            import traceback
            traceback.print_exc()
    else:
        print("No sample images found")


if __name__ == "__main__":
    main()