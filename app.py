#!/usr/bin/env python3
"""
Nuva Face MVP - Template-Based Lip Enhancement Prediction
Web Interface
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import base64
from PIL import Image
import io
import tempfile

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from models.template_matcher import LipTemplateMatch
from enhancement.advanced_lip_enhancer import AdvancedLipEnhancer

app = Flask(__name__)
CORS(app)

# Initialize template matcher and advanced lip enhancer
template_matcher = LipTemplateMatch()
lip_enhancer = AdvancedLipEnhancer()

@app.route('/')
def index():
    """Main page with upload interface."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_enhancement():
    """API endpoint for lip enhancement prediction."""
    try:
        # Get uploaded image
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get parameters
        volume_ml = float(request.form.get('volume', 1.0))
        product = request.form.get('product', "['Restylane Kysse']")
        
        # Save uploaded image to persistent location
        import uuid
        upload_id = str(uuid.uuid4())
        uploads_dir = project_root / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        original_path = uploads_dir / f"{upload_id}_original.jpg"
        image_file.save(str(original_path))
        
        try:
            # Run prediction
            result = template_matcher.predict_enhancement(
                str(original_path), 
                target_volume=volume_ml,
                target_product=product
            )
            
            if result['success']:
                # Get template examples for display
                template_info = result['matched_template']
                transformation_pairs = result['transformation_pairs']
                
                # 🚀 NEW: Enhance customer's actual image
                try:
                    enhanced_image_path = lip_enhancer.enhance_customer_lips(
                        str(original_path),
                        result,  # Template guidance
                        volume_ml=volume_ml,
                        product_type=product.strip("[]'")
                    )
                    
                    # Create before/after comparison
                    comparison_path = enhanced_image_path.replace('_enhanced.jpg', '_comparison.jpg')
                    lip_enhancer.create_before_after_comparison(
                        str(original_path),
                        enhanced_image_path,
                        comparison_path
                    )
                    
                    # Convert paths for serving (relative to project root)
                    enhanced_rel = os.path.relpath(enhanced_image_path, project_root)
                    comparison_rel = os.path.relpath(comparison_path, project_root)
                    original_rel = os.path.relpath(str(original_path), project_root)
                    
                    customer_result = {
                        'original': f'/serve_image/{original_rel.replace("/", "__")}',
                        'enhanced': f'/serve_image/{enhanced_rel.replace("/", "__")}',
                        'comparison': f'/serve_image/{comparison_rel.replace("/", "__")}'
                    }
                    
                    print(f"✅ Customer result created:")
                    print(f"   Original: {customer_result['original']}")
                    print(f"   Enhanced: {customer_result['enhanced']}")
                    
                except Exception as e:
                    print(f"Enhancement error: {e}")
                    import traceback
                    traceback.print_exc()
                    customer_result = None
                
                # Convert template example paths to URLs for frontend
                example_images = []
                for before_path, after_path in transformation_pairs[:3]:  # Show max 3 examples
                    # Create relative paths for serving
                    before_rel = os.path.relpath(before_path, project_root)
                    after_rel = os.path.relpath(after_path, project_root)
                    
                    example_images.append({
                        'before': f'/serve_image/{before_rel.replace("/", "__")}',
                        'after': f'/serve_image/{after_rel.replace("/", "__")}',
                    })
                
                response = {
                    'success': True,
                    'customer_result': customer_result,  # 🚀 NEW: Customer's enhanced image
                    'prediction': {
                        'matched_person': template_info['person_id'],
                        'similarity_score': template_info['similarity_score'],
                        'confidence': result['prediction_confidence'],
                        'treatment_notes': template_info['treatment_notes'],
                        'image_count': template_info['image_count']
                    },
                    'examples': example_images,
                    'recommendation': result['recommendation'],
                    'disclaimer': result['disclaimer'],
                    'parameters': {
                        'volume': f"{volume_ml}ml",
                        'product': product
                    }
                }
                
                return jsonify(response)
            else:
                return jsonify({
                    'success': False,
                    'error': result['error'],
                    'suggestion': result.get('suggestion', '')
                }), 404
                
        finally:
            # Keep files for serving (don't delete)
            pass
            
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/serve_image/<path:image_path>')
def serve_image(image_path):
    """Serve images from data directory."""
    # Convert back from URL-safe format
    actual_path = image_path.replace('__', '/')
    
    # Security: ensure path is within project
    full_path = project_root / actual_path
    if not str(full_path).startswith(str(project_root)):
        return "Access denied", 403
    
    if full_path.exists():
        return send_from_directory(full_path.parent, full_path.name)
    else:
        return "Image not found", 404

@app.route('/api/stats')
def get_stats():
    """Get template database statistics."""
    stats = template_matcher.get_template_statistics()
    return jsonify(stats)

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'templates_loaded': len(template_matcher.templates),
        'service': 'Nuva Face Template Matcher'
    })

@app.route('/api/cleanup')
def cleanup_uploads():
    """Clean up old upload files (older than 1 hour)."""
    try:
        uploads_dir = project_root / "uploads"
        if not uploads_dir.exists():
            return jsonify({'message': 'No uploads directory'})
        
        import time
        current_time = time.time()
        deleted_count = 0
        
        for file_path in uploads_dir.glob("*"):
            if file_path.is_file():
                # Delete files older than 1 hour (3600 seconds)
                if current_time - file_path.stat().st_mtime > 3600:
                    file_path.unlink()
                    deleted_count += 1
        
        return jsonify({
            'message': f'Deleted {deleted_count} old files',
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory for HTML
    templates_dir = project_root / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting Nuva Face MVP Server...")
    print(f"📚 Loaded {len(template_matcher.templates)} templates")
    print("🌐 Server will be available at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)