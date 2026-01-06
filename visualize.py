"""
Grad-CAM Visualization for MaizeAttentionNet.

This script implements Gradient-weighted Class Activation Mapping (Grad-CAM)
to visualize which regions of a maize leaf image the model focuses on when
making disease classification predictions.

Features:
- Automatic hook registration on the final convolutional layer
- Grad-CAM heatmap overlay generation
- Spatial attention map extraction
- Batch visualization across all disease classes

Deep Learning Course Project - 400L
"""

import os
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Import our custom model
from model import MaizeAttentionNet


# ============================================================
# Configuration
# ============================================================
# ImageNet normalization values (used during training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default paths
DATA_DIR = './data'
MODEL_PATH = './best_maize_model.pth'
OUTPUT_DIR = './evaluation_results/gradcam'


# ============================================================
# Device Selection
# ============================================================
def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        torch.device: CUDA, MPS (Apple Silicon), or CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Grad-CAM Implementation
# ============================================================
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Implements Grad-CAM to visualize which regions of an input image
    are important for a CNN's prediction. Automatically finds and hooks
    the final convolutional layer of the model.
    
    Reference:
        Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks
        via Gradient-based Localization" (ICCV 2017)
    
    Args:
        model (nn.Module): The trained model
        target_layer (nn.Module, optional): Specific layer to hook.
            If None, automatically finds the final conv layer.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None
    ):
        self.model = model
        self.model.eval()
        
        # Storage for activations and gradients
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        
        # Find target layer if not specified
        if target_layer is None:
            self.target_layer = self._find_final_conv_layer()
        else:
            self.target_layer = target_layer
        
        # Register hooks
        self._register_hooks()
    
    def _find_final_conv_layer(self) -> nn.Module:
        """
        Automatically find the final convolutional layer in the model.
        
        For MaizeAttentionNet, this is conv_block4.conv (512 channels).
        
        Returns:
            nn.Module: The final convolutional layer
        """
        # For MaizeAttentionNet, we know the structure
        # The final conv layer before attention is conv_block4.conv
        if hasattr(self.model, 'conv_block4'):
            return self.model.conv_block4.conv
        
        # Generic fallback: find last Conv2d layer
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        
        if last_conv is None:
            raise ValueError("No convolutional layer found in the model!")
        
        return last_conv
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input image.
        
        Args:
            input_tensor (torch.Tensor): Preprocessed input image (1, C, H, W)
            target_class (int, optional): Target class index. If None,
                uses the predicted class.
        
        Returns:
            np.ndarray: Grad-CAM heatmap of shape (H, W) with values in [0, 1]
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Enable gradients for input
        input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero out gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)
        
        # Global average pooling of gradients -> channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        
        # ReLU to keep only positive contributions
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class, output


# ============================================================
# Image Processing Utilities
# ============================================================
def get_val_transform() -> transforms.Compose:
    """Get validation/inference transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize a tensor image using ImageNet statistics.
    
    Args:
        tensor (torch.Tensor): Normalized image tensor (C, H, W)
    
    Returns:
        np.ndarray: Denormalized image as uint8 array (H, W, C)
    """
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    
    image = tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    return (image * 255).astype(np.uint8)


def apply_colormap(
    heatmap: np.ndarray,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Apply a colormap to a heatmap.
    
    Args:
        heatmap (np.ndarray): 2D array with values in [0, 1]
        colormap (str): Matplotlib colormap name
    
    Returns:
        np.ndarray: RGB image as uint8 array (H, W, 3)
    """
    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    return (colored * 255).astype(np.uint8)


def create_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay a heatmap on an image.
    
    Args:
        image (np.ndarray): Original image (H, W, 3)
        heatmap (np.ndarray): Heatmap (can be any size, will be resized)
        alpha (float): Blending factor for heatmap
    
    Returns:
        np.ndarray: Overlaid image (H, W, 3)
    """
    # Resize heatmap to match image size
    h, w = image.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray(heatmap).resize((w, h), Image.BILINEAR)
    )
    
    # Blend
    overlay = (1 - alpha) * image + alpha * heatmap_resized
    return overlay.astype(np.uint8)


def load_model(
    model_path: str,
    device: torch.device
) -> Tuple[MaizeAttentionNet, List[str]]:
    """
    Load trained MaizeAttentionNet model from checkpoint.
    
    Args:
        model_path (str): Path to the checkpoint file
        device (torch.device): Device to load model on
    
    Returns:
        tuple: (model, class_names)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get class names from checkpoint
    class_names = checkpoint.get('class_names', 
        ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy'])
    
    # Create model
    model = MaizeAttentionNet(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, class_names


def preprocess_image(
    image_path: str,
    device: torch.device
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load and preprocess an image for inference.
    
    Args:
        image_path (str): Path to the image file
        device (torch.device): Device for the tensor
    
    Returns:
        tuple: (preprocessed_tensor, original_pil_image)
    """
    # Load original image
    original = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    transform = get_val_transform()
    tensor = transform(original).unsqueeze(0).to(device)
    
    return tensor, original


# ============================================================
# Spatial Attention Extraction
# ============================================================
def get_spatial_attention_map(
    model: MaizeAttentionNet,
    input_tensor: torch.Tensor
) -> np.ndarray:
    """
    Extract the spatial attention map from the model.
    
    Uses the model's built-in get_attention_map method.
    
    Args:
        model (MaizeAttentionNet): The model
        input_tensor (torch.Tensor): Preprocessed input image (1, C, H, W)
    
    Returns:
        np.ndarray: Attention map normalized to [0, 1]
    """
    model.eval()
    with torch.no_grad():
        attention_map, _ = model.get_attention_map(input_tensor)
    
    # Squeeze and normalize
    attention = attention_map.squeeze().cpu().numpy()
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    return attention


# ============================================================
# Visualization Functions
# ============================================================
def visualize_single_image(
    image_path: str,
    model: MaizeAttentionNet,
    class_names: List[str],
    device: torch.device,
    output_path: str = 'gradcam_analysis.png',
    target_class: Optional[int] = None
) -> Dict:
    """
    Visualize Grad-CAM and attention for a single image.
    
    Creates a figure with 3 side-by-side subplots:
    (A) Original Image
    (B) Spatial Attention Map
    (C) Grad-CAM Overlay
    
    Args:
        image_path (str): Path to the input image
        model (MaizeAttentionNet): Trained model
        class_names (List[str]): List of class names
        device (torch.device): Computation device
        output_path (str): Path to save the output figure
        target_class (int, optional): Target class for Grad-CAM
    
    Returns:
        dict: Contains predicted class, confidence, and paths
    """
    # Preprocess image
    input_tensor, original_pil = preprocess_image(image_path, device)
    
    # Get predictions
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Generate Grad-CAM
    gradcam = GradCAM(model)
    cam, used_class, _ = gradcam.generate(input_tensor.clone(), target_class)
    
    # Get spatial attention map
    attention_map = get_spatial_attention_map(model, input_tensor)
    
    # Prepare images for visualization
    original_np = np.array(original_pil.resize((224, 224)))
    
    # Apply colormap to heatmaps
    cam_colored = apply_colormap(cam)
    attention_colored = apply_colormap(attention_map)
    
    # Create overlays
    gradcam_overlay = create_overlay(original_np, cam_colored, alpha=0.5)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'Grad-CAM Analysis | Predicted: {class_names[pred_class]} ({confidence*100:.1f}%)',
        fontsize=14, fontweight='bold'
    )
    
    # (A) Original Image
    axes[0].imshow(original_np)
    axes[0].set_title('(A) Original Image', fontsize=12)
    axes[0].axis('off')
    
    # (B) Spatial Attention Map
    im_attention = axes[1].imshow(attention_map, cmap='jet')
    axes[1].set_title('(B) Spatial Attention Map', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im_attention, ax=axes[1], fraction=0.046, pad=0.04)
    
    # (C) Grad-CAM Overlay
    axes[2].imshow(gradcam_overlay)
    axes[2].set_title('(C) Grad-CAM Overlay', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved Grad-CAM analysis to: {output_path}")
    
    return {
        'predicted_class': class_names[pred_class],
        'confidence': confidence,
        'output_path': output_path
    }


def visualize_batch(
    data_dir: str,
    model: MaizeAttentionNet,
    class_names: List[str],
    device: torch.device,
    output_path: str = 'gradcam_batch_analysis.png',
    seed: int = 42
) -> Dict:
    """
    Visualize Grad-CAM for one sample from each disease class.
    
    Creates a comprehensive grid visualization showing:
    - Original image, Spatial Attention, and Grad-CAM for each class
    
    Args:
        data_dir (str): Path to the data directory
        model (MaizeAttentionNet): Trained model
        class_names (List[str]): List of class names
        device (torch.device): Computation device
        output_path (str): Path to save the output figure
        seed (int): Random seed for sample selection
    
    Returns:
        dict: Summary of visualizations for each class
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Collect one sample from each class
    samples = {}
    for class_name in class_names:
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue
        
        # Get all images in the class directory
        image_files = list(class_dir.glob('*.jpg')) + \
                      list(class_dir.glob('*.jpeg')) + \
                      list(class_dir.glob('*.png')) + \
                      list(class_dir.glob('*.JPG'))
        
        if image_files:
            samples[class_name] = random.choice(image_files)
    
    if not samples:
        raise ValueError("No samples found in the data directory!")
    
    # Create figure: 4 rows (one per class) x 3 cols (original, attention, gradcam)
    num_classes = len(samples)
    fig, axes = plt.subplots(num_classes, 3, figsize=(15, 5 * num_classes))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(
        'Grad-CAM Analysis Across Disease Classes',
        fontsize=16, fontweight='bold', y=1.02
    )
    
    results = {}
    gradcam = GradCAM(model)
    
    for idx, (class_name, image_path) in enumerate(samples.items()):
        # Preprocess image
        input_tensor, original_pil = preprocess_image(str(image_path), device)
        
        # Get predictions
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        # Generate Grad-CAM (for predicted class)
        # Need to create new GradCAM instance for fresh hooks
        gradcam = GradCAM(model)
        cam, _, _ = gradcam.generate(input_tensor.clone(), pred_class)
        
        # Get spatial attention map
        attention_map = get_spatial_attention_map(model, input_tensor)
        
        # Prepare images
        original_np = np.array(original_pil.resize((224, 224)))
        cam_colored = apply_colormap(cam)
        gradcam_overlay = create_overlay(original_np, cam_colored, alpha=0.5)
        
        # Determine correctness
        is_correct = class_names[pred_class] == class_name
        status = "✓" if is_correct else "✗"
        
        # Plot original
        axes[idx, 0].imshow(original_np)
        axes[idx, 0].set_title(
            f'{class_name}\nPred: {class_names[pred_class]} ({confidence*100:.1f}%) {status}',
            fontsize=10
        )
        axes[idx, 0].axis('off')
        
        # Plot attention
        im_att = axes[idx, 1].imshow(attention_map, cmap='jet')
        axes[idx, 1].set_title('Spatial Attention', fontsize=10)
        axes[idx, 1].axis('off')
        
        # Plot Grad-CAM
        axes[idx, 2].imshow(gradcam_overlay)
        axes[idx, 2].set_title('Grad-CAM Overlay', fontsize=10)
        axes[idx, 2].axis('off')
        
        results[class_name] = {
            'image_path': str(image_path),
            'predicted_class': class_names[pred_class],
            'confidence': confidence,
            'correct': is_correct
        }
    
    # Add column titles
    col_titles = ['(A) Original Image', '(B) Spatial Attention', '(C) Grad-CAM']
    for ax, title in zip(axes[0], col_titles):
        ax.annotate(
            title, xy=(0.5, 1.15), xycoords='axes fraction',
            ha='center', fontsize=12, fontweight='bold'
        )
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved batch Grad-CAM analysis to: {output_path}")
    
    return results


# ============================================================
# Main Entry Point
# ============================================================
def main():
    """
    Main function to run Grad-CAM visualization.
    """
    print("=" * 60)
    print("Grad-CAM Visualization for MaizeAttentionNet")
    print("=" * 60)
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model, class_names = load_model(MODEL_PATH, device)
    print(f"Classes: {class_names}")
    
    # Run batch visualization (one sample per class)
    print("\n" + "-" * 40)
    print("Running Grad-CAM visualization for all classes...")
    
    batch_output = os.path.join(OUTPUT_DIR, 'gradcam_batch_analysis.png')
    batch_results = visualize_batch(
        data_dir=DATA_DIR,
        model=model,
        class_names=class_names,
        device=device,
        output_path=batch_output
    )
    
    # Print results summary
    print("\n" + "-" * 40)
    print("Results Summary:")
    print("-" * 40)
    
    for class_name, result in batch_results.items():
        status = "✓" if result['correct'] else "✗"
        print(f"  {class_name}: Predicted={result['predicted_class']} "
              f"({result['confidence']*100:.1f}%) {status}")
    
    # Also run single image visualization on a random sample
    print("\n" + "-" * 40)
    print("Running single image Grad-CAM analysis...")
    
    # Pick a random sample from any class
    all_images = []
    for class_name in class_names:
        class_dir = Path(DATA_DIR) / class_name
        if class_dir.exists():
            all_images.extend(class_dir.glob('*.jpg'))
            all_images.extend(class_dir.glob('*.JPG'))
            all_images.extend(class_dir.glob('*.png'))
    
    if all_images:
        random.seed(42)
        sample_image = random.choice(all_images)
        single_output = os.path.join(OUTPUT_DIR, 'gradcam_analysis.png')
        
        visualize_single_image(
            image_path=str(sample_image),
            model=model,
            class_names=class_names,
            device=device,
            output_path=single_output
        )
    
    print("\n" + "=" * 60)
    print("Grad-CAM Visualization Complete!")
    print(f"Output saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
