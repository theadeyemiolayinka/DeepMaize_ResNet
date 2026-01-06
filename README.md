# DeepMaize ResNet: Maize Leaf Disease Classification with Spatial Attention

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A custom Convolutional Neural Network with **Spatial Attention** for classifying maize (corn) leaf diseases. Built from scratch without using pre-trained backbones, this project demonstrates deep learning fundamentals while achieving competitive performance.

**Deep Learning Course Project - 400 Level**

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Web Deployment](#web-deployment)
- [Visualization](#visualization)
- [Benchmarking](#benchmarking)
- [Robustness](#robustness)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a custom CNN architecture called **MaizeAttentionNet** for classifying maize leaf diseases into four categories:

1. **Blight** - Leaf blight disease
2. **Common Rust** - Common rust infection
3. **Gray Leaf Spot** - Gray leaf spot disease
4. **Healthy** - Healthy maize leaves

### Key Features

- **Custom Architecture**: 4 convolutional blocks built entirely from scratch
- **Spatial Attention**: Novel attention mechanism using channel-wise pooling
- **No Pre-trained Weights**: Backbone built without torchvision.models
- **Baseline Comparison**: Includes MobileNetV2 pre-trained baseline
- **Comprehensive Evaluation**: Confusion matrix, ROC curves, and metrics
- **Cross-Platform**: Works on CUDA, Apple MPS, and CPU

---

## Architecture

### MaizeAttentionNet

```
Input Image (3, 224, 224)
        │
        ▼
┌─────────────────────────┐
│   Conv Block 1 (64)     │  Conv2D → BatchNorm → ReLU → MaxPool
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Conv Block 2 (128)    │  Conv2D → BatchNorm → ReLU → MaxPool
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Conv Block 3 (256)    │  Conv2D → BatchNorm → ReLU → MaxPool
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Conv Block 4 (512)    │  Conv2D → BatchNorm → ReLU → MaxPool
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   Spatial Attention     │  AvgPool + MaxPool → Conv → Sigmoid
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Global Average Pool    │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   FC Classifier         │  512 → 256 → 128 → num_classes
│   (with Dropout)        │
└─────────────────────────┘
        │
        ▼
    Output (4 classes)
```

### Spatial Attention Module

The attention module computes importance weights for spatial locations:

1. **Channel-wise Average Pooling**: Captures global context
2. **Channel-wise Max Pooling**: Captures salient features
3. **Concatenation + Convolution**: Fuses pooled features
4. **Sigmoid Activation**: Generates attention map (0-1)
5. **Element-wise Multiplication**: Applies attention to features

---

## Dataset

We use the [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset) from Kaggle.

### Download Dataset

```bash
# Install Kaggle CLI (if not installed)
pip install kaggle

# Configure Kaggle API credentials
# Place kaggle.json in ~/.kaggle/

# Download and extract dataset
kaggle datasets download -d smaranjitghose/corn-or-maize-leaf-disease-dataset --unzip

# Move to project directory
mv corn-or-maize-leaf-disease-dataset/* ./data/
```

### Expected Directory Structure

```
data/
├── Blight/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── Common_Rust/
│   ├── image_001.jpg
│   └── ...
├── Gray_Leaf_Spot/
│   ├── image_001.jpg
│   └── ...
└── Healthy/
    ├── image_001.jpg
    └── ...
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster training
- (Optional) Apple Silicon Mac for MPS acceleration

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepMaize_ResNet.git
cd DeepMaize_ResNet

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

---

## Usage

### 1. Training

Train the MaizeAttentionNet model:

```bash
python train.py
```

**Configuration** (modify in `train.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_DIR` | `./data` | Path to dataset |
| `BATCH_SIZE` | `32` | Training batch size |
| `NUM_EPOCHS` | `25` | Number of training epochs |
| `LEARNING_RATE` | `0.001` | Initial learning rate |
| `VAL_SPLIT` | `0.2` | Validation split ratio |
| `DROPOUT_RATE` | `0.5` | Dropout probability |

**Output**:
- `best_maize_model.pth` - Best model weights
- `training_history.pth` - Training metrics history

### 2. Evaluation

Evaluate the trained model and compare with baseline:

```bash
python evaluate.py
```

**Output** (saved to `./evaluation_results/`):
- `confusion_matrix_maize.png` - Confusion matrix for our model
- `confusion_matrix_baseline.png` - Confusion matrix for MobileNetV2
- `roc_curves_maize.png` - ROC curves for our model
- `roc_curves_baseline.png` - ROC curves for MobileNetV2
- `model_comparison.png` - Side-by-side metric comparison
- `evaluation_results.pth` - All metrics in PyTorch format

### 3. Inference (Single Image)

```python
import torch
from PIL import Image
from torchvision import transforms
from model import MaizeAttentionNet

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MaizeAttentionNet(num_classes=4)
checkpoint = torch.load('best_maize_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('path/to/leaf_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
print(f"Prediction: {class_names[predicted_class]}")
print(f"Confidence: {probabilities[0][predicted_class]:.2%}")
```

---

## Web Deployment

Deploy the trained model as a web application with a beautiful UI.

### Running the Web App

```bash
# Make sure the model is trained first
python train.py

# Start the Flask server
python web/app.py

# Open in browser
# http://localhost:8123
```

### Web App Features

- **Drag & Drop Upload**: Easy image upload with preview
- **Real-time Analysis**: Instant disease classification
- **Detailed Results**: Confidence scores, disease information, symptoms, and treatment recommendations
- **Cross-Platform**: Works on any device with a web browser

### API Endpoint

```bash
# POST /predict
curl -X POST -F "file=@leaf_image.jpg" http://localhost:8123/predict

# Response:
{
  "success": true,
  "prediction": {
    "class": "Healthy",
    "confidence": 98.5,
    "all_probabilities": {...}
  },
  "disease_info": {...}
}
```

---

## Visualization

Visualize model predictions using Grad-CAM (Gradient-weighted Class Activation Mapping).

### Running Grad-CAM Analysis

```bash
python visualize.py
```

**Output** (saved to `./evaluation_results/gradcam/`):
- `gradcam_analysis.png` - Single image analysis with 3 panels:
  - (A) Original Image
  - (B) Spatial Attention Map (from model's attention layer)
  - (C) Grad-CAM Overlay (highlighting regions important for classification)
- `gradcam_batch_analysis.png` - Comparison across all 4 disease classes

### Understanding the Visualizations

| Visualization | Description |
|---------------|-------------|
| **Original Image** | The input maize leaf image |
| **Spatial Attention** | Model's learned attention weights highlighting important regions |
| **Grad-CAM Overlay** | Gradient-based heatmap showing which regions influenced the prediction |

---

## Benchmarking

Compare model efficiency between MaizeAttentionNet and MobileNetV2 baseline.

### Running Benchmarks

```bash
python benchmark.py
```

**Metrics Computed**:

| Metric | Description |
|--------|-------------|
| **Parameters** | Total model parameters (in millions) |
| **FLOPs** | Floating point operations for a single 224×224 input |
| **Model Size** | Disk space of the saved model file (MB) |
| **CPU Latency** | Average inference time on CPU (ms) |
| **GPU/MPS Latency** | Average inference time on GPU or Apple Silicon (ms) |

**Output**:
- Console: Formatted comparison table
- `./evaluation_results/efficiency_results.txt` - Detailed benchmark report with hardware info

### Hardware Detection

The benchmark script automatically detects and logs:
- CPU model/name
- GPU name and memory (NVIDIA CUDA or Apple MPS)
- PyTorch and CUDA versions

---

## Robustness

Evaluate model robustness against image corruption and noise.

### Running Robustness Tests

```bash
python robustness.py
```

### Noise Types Tested

| Noise Type | Description | Real-world Scenario |
|------------|-------------|--------------------|
| **Gaussian Noise** | Random pixel-level noise | Low-light camera grain |
| **Motion Blur** | Directional blur | Shaky hands, camera movement |
| **Brightness Drop** | Reduced image brightness | Shadows, underexposure |

### Noise Intensity Levels

- **Low**: Mild corruption (33% intensity)
- **Medium**: Moderate corruption (66% intensity)
- **High**: Severe corruption (100% intensity)

### Output

- `./evaluation_results/robustness_chart.png` - Line graphs comparing model accuracy vs noise intensity
- `./evaluation_results/robustness_results.txt` - Detailed results with accuracy drop analysis
- `./evaluation_results/robustness_results.pth` - Raw results for further analysis

### Interpretation

The robustness evaluation tests whether the **Spatial Attention mechanism** helps maintain accuracy under challenging conditions. A more robust model will show:
- Smaller accuracy drops as noise increases
- Better performance compared to baseline at higher noise levels

---

## Results

### Model Comparison

| Metric | MaizeAttentionNet (Ours) | MobileNetV2 (Baseline) |
|--------|--------------------------|------------------------|
| Accuracy | XX.XX% | XX.XX% |
| Precision | X.XXXX | X.XXXX |
| Recall | X.XXXX | X.XXXX |
| F1-Score | X.XXXX | X.XXXX |

*Results will be populated after training*

### Visualizations

After running `evaluate.py`, check the `evaluation_results/` directory for:

- **Confusion Matrix**: Shows per-class classification performance
- **ROC Curves**: Area Under Curve (AUC) for each class
- **Model Comparison**: Bar chart comparing all metrics

---

## Project Structure

```
DeepMaize_ResNet/
├── model.py                 # MaizeAttentionNet architecture
├── train.py                 # Training script
├── evaluate.py              # Evaluation and comparison script
├── visualize.py             # Grad-CAM visualization script
├── benchmark.py             # Model efficiency benchmarking
├── robustness.py            # Robustness evaluation script
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── .gitignore               # Git ignore rules
├── LICENSE                  # MIT License
├── data/                    # Dataset directory (not tracked)
│   ├── Blight/
│   ├── Common_Rust/
│   ├── Gray_Leaf_Spot/
│   └── Healthy/
├── web/                     # Web application
│   ├── app.py               # Flask server
│   ├── templates/
│   │   └── index.html       # Web UI template
│   └── static/
│       └── css/
│           └── style.css    # Custom styles
└── evaluation_results/      # Generated evaluation outputs
    ├── confusion_matrix_maize.png
    ├── confusion_matrix_baseline.png
    ├── roc_curves_maize.png
    ├── roc_curves_baseline.png
    ├── model_comparison.png
    ├── efficiency_results.txt
    ├── robustness_chart.png
    ├── robustness_results.txt
    └── gradcam/
        ├── gradcam_analysis.png
        └── gradcam_batch_analysis.png
```

---

## Technical Details

### Data Augmentation

Training transforms include:
- Random Rotation (±30°)
- Random Horizontal/Vertical Flip
- Color Jitter (brightness, contrast, saturation, hue)
- Random Crop (256 → 224)
- Normalization (ImageNet mean/std)

### Training Configuration

- **Optimizer**: Adam with weight decay (L2 regularization)
- **Loss Function**: CrossEntropyLoss
- **Learning Rate Scheduler**: StepLR (decay every 7 epochs)
- **Reproducibility**: `torch.manual_seed(42)`

### Device Compatibility

```python
# Automatic device selection
if torch.cuda.is_available():
    device = 'cuda'      # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = 'mps'       # Apple Silicon
else:
    device = 'cpu'       # Fallback
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset) by Smaranjit Ghose
- PyTorch Team for the excellent deep learning framework
- Course instructors for guidance and support

---

## Contact

For questions or feedback, please open an issue on GitHub.

---

<p align="center">
  Made with ❤️ for New Horizons Deep Learning Course Project
</p>
