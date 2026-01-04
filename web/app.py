"""
Flask Web Application for MaizeAttentionNet Inference.

This web app provides a user-friendly interface for classifying maize leaf
diseases using the trained MaizeAttentionNet model.

Deep Learning Course Project - 400L

Usage:
    python web/app.py
    
Then open http://localhost:8123 in your browser.
"""

import os
import sys
from io import BytesIO
from datetime import datetime

# Add parent directory to path for model import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torchvision import transforms

from model import MaizeAttentionNet

# ============================================================
# Flask App Configuration
# ============================================================
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ============================================================
# Model Loading
# ============================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_maize_model.pth')
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
DEVICE = None
MODEL = None


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model():
    """Load the trained MaizeAttentionNet model."""
    global MODEL, DEVICE, CLASS_NAMES
    
    DEVICE = get_device()
    print(f"Using device: {DEVICE}")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Please train the model first using: python train.py")
        return False
    
    try:
        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # Get class names from checkpoint if available
        if 'class_names' in checkpoint:
            CLASS_NAMES = checkpoint['class_names']
        
        num_classes = len(CLASS_NAMES)
        
        # Create and load model
        MODEL = MaizeAttentionNet(num_classes=num_classes)
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL = MODEL.to(DEVICE)
        MODEL.eval()
        
        print("Model loaded successfully!")
        print(f"Classes: {CLASS_NAMES}")
        print(f"Best accuracy: {checkpoint.get('best_acc', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


# ============================================================
# Image Processing
# ============================================================
def get_transform():
    """Get the image transform for inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def process_image(image_bytes):
    """
    Process an uploaded image for model inference.
    
    Args:
        image_bytes: Raw bytes of the uploaded image
        
    Returns:
        torch.Tensor: Processed image tensor ready for model
    """
    try:
        # Open image from bytes
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms
        transform = get_transform()
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return tensor.to(DEVICE)
        
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")


# ============================================================
# Prediction
# ============================================================
def predict(image_tensor):
    """
    Run inference on a processed image.
    
    Args:
        image_tensor: Processed image tensor
        
    Returns:
        tuple: (predicted_class, confidence, all_probabilities)
    """
    with torch.no_grad():
        outputs = MODEL(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
        
        return (
            CLASS_NAMES[predicted_idx.item()],
            confidence.item() * 100,
            {CLASS_NAMES[i]: float(all_probs[i] * 100) for i in range(len(CLASS_NAMES))}
        )


# ============================================================
# Disease Information
# ============================================================
DISEASE_INFO = {
    'Blight': {
        'description': 'Northern Corn Leaf Blight is a fungal disease caused by Exserohilum turcicum.',
        'symptoms': 'Long, elliptical, grayish-green or tan lesions on leaves.',
        'treatment': 'Apply fungicides, use resistant hybrids, practice crop rotation.',
        'severity': 'moderate',
        'icon': 'bi-exclamation-triangle'
    },
    'Common_Rust': {
        'description': 'Common Rust is caused by the fungus Puccinia sorghi.',
        'symptoms': 'Small, circular to elongated, reddish-brown pustules on both leaf surfaces.',
        'treatment': 'Use resistant hybrids, apply fungicides if severe.',
        'severity': 'moderate',
        'icon': 'bi-exclamation-triangle'
    },
    'Gray_Leaf_Spot': {
        'description': 'Gray Leaf Spot is caused by the fungus Cercospora zeae-maydis.',
        'symptoms': 'Rectangular, tan to gray lesions that run parallel to leaf veins.',
        'treatment': 'Crop rotation, tillage, fungicides, and resistant hybrids.',
        'severity': 'high',
        'icon': 'bi-exclamation-octagon'
    },
    'Healthy': {
        'description': 'No disease detected. The leaf appears to be healthy.',
        'symptoms': 'No visible symptoms of disease.',
        'treatment': 'Continue regular crop management practices.',
        'severity': 'none',
        'icon': 'bi-check-circle'
    }
}


# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    """Render the main page."""
    model_loaded = MODEL is not None
    return render_template('index.html', 
                          model_loaded=model_loaded,
                          class_names=CLASS_NAMES)


@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Handle image upload and return prediction.
    
    Expects a POST request with an image file.
    Returns JSON with prediction results.
    """
    # Check if model is loaded
    if MODEL is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    file_ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
        }), 400
    
    try:
        # Read and process the image
        image_bytes = file.read()
        image_tensor = process_image(image_bytes)
        
        # Get prediction
        predicted_class, confidence, all_probs = predict(image_tensor)
        
        # Get disease information
        disease_info = DISEASE_INFO.get(predicted_class, {})
        
        return jsonify({
            'success': True,
            'prediction': {
                'class': predicted_class.replace('_', ' '),
                'confidence': round(confidence, 2),
                'all_probabilities': {k.replace('_', ' '): round(v, 2) for k, v in all_probs.items()}
            },
            'disease_info': disease_info,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': str(DEVICE) if DEVICE else None,
        'classes': CLASS_NAMES
    })


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("MaizeAttentionNet - Web Inference Server")
    print("=" * 60)
    
    # Load model on startup
    if load_model():
        print("\nStarting Flask server...")
        print("Open http://localhost:8123 in your browser")
        print("=" * 60)
        
        app.run(host='0.0.0.0', port=8123, debug=False)
    else:
        print("\nFailed to load model. Please ensure:")
        print("1. You have trained the model: python train.py")
        print("2. The model file exists: best_maize_model.pth")
        print("\nExiting...")
