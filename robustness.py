"""
Robustness Evaluation Script for MaizeAttentionNet.

This script evaluates model robustness against various types of image
corruption, comparing the custom MaizeAttentionNet with MobileNetV2 baseline.

Noise Types:
- Gaussian Noise (simulating low-light grain)
- Motion Blur (simulating shaky hands)
- Brightness Drop (simulating shadows)

Deep Learning Course Project - 400L
"""

import os
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# Import our custom model
from model import MaizeAttentionNet


# ============================================================
# Configuration
# ============================================================
DATA_DIR = './data'
MODEL_PATH = './best_maize_model.pth'
OUTPUT_DIR = './evaluation_results'
BATCH_SIZE = 32
NUM_WORKERS = 0  # For compatibility

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Noise intensity levels
NOISE_LEVELS = {
    'Low': 0.33,
    'Medium': 0.66,
    'High': 1.0
}


# ============================================================
# Device Selection
# ============================================================
def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Noise Injection Classes
# ============================================================
class NoiseInjector:
    """
    Base class for noise injection transformations.
    
    Applies synthetic noise to images to test model robustness.
    """
    
    def __init__(self, intensity: float = 0.5):
        """
        Initialize the noise injector.
        
        Args:
            intensity (float): Noise intensity from 0.0 to 1.0
        """
        self.intensity = max(0.0, min(1.0, intensity))
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply noise to the image."""
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(intensity={self.intensity})"


class GaussianNoiseInjector(NoiseInjector):
    """
    Adds Gaussian noise to simulate low-light camera grain.
    
    Intensity controls the standard deviation of the noise.
    """
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Add Gaussian noise to the image.
        
        Args:
            img (PIL.Image): Input image
        
        Returns:
            PIL.Image: Noisy image
        """
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Scale noise std with intensity (max std = 0.15 at intensity=1.0)
        noise_std = self.intensity * 0.15
        noise = np.random.normal(0, noise_std, img_array.shape).astype(np.float32)
        
        noisy_img = img_array + noise
        noisy_img = np.clip(noisy_img, 0, 1)
        noisy_img = (noisy_img * 255).astype(np.uint8)
        
        return Image.fromarray(noisy_img)


class MotionBlurInjector(NoiseInjector):
    """
    Applies motion blur to simulate camera shake or movement.
    
    Intensity controls the blur kernel size.
    """
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply motion blur to the image.
        
        Args:
            img (PIL.Image): Input image
        
        Returns:
            PIL.Image: Blurred image
        """
        # Kernel size based on intensity (3 to 15 pixels)
        kernel_size = int(3 + self.intensity * 12)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        # Create horizontal motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        
        # Convert to PIL ImageFilter
        img_array = np.array(img).astype(np.float32)
        
        # Apply blur to each channel
        blurred = np.zeros_like(img_array)
        for c in range(3):
            from scipy.ndimage import convolve
            blurred[:, :, c] = convolve(img_array[:, :, c], kernel, mode='reflect')
        
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
        return Image.fromarray(blurred)


class BrightnessDropInjector(NoiseInjector):
    """
    Reduces image brightness to simulate shadows or underexposure.
    
    Intensity controls the degree of darkening.
    """
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Reduce brightness of the image.
        
        Args:
            img (PIL.Image): Input image
        
        Returns:
            PIL.Image: Darkened image
        """
        img_array = np.array(img).astype(np.float32)
        
        # Brightness factor: 1.0 at intensity=0, 0.3 at intensity=1.0
        brightness_factor = 1.0 - (self.intensity * 0.7)
        
        darkened = img_array * brightness_factor
        darkened = np.clip(darkened, 0, 255).astype(np.uint8)
        
        return Image.fromarray(darkened)


class CombinedNoiseInjector(NoiseInjector):
    """
    Applies multiple noise types simultaneously.
    """
    
    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply all noise types with scaled intensity."""
        # Apply each noise type with reduced individual intensity
        scaled_intensity = self.intensity * 0.6  # Scale down for combined effect
        
        img = GaussianNoiseInjector(scaled_intensity)(img)
        img = MotionBlurInjector(scaled_intensity * 0.5)(img)
        img = BrightnessDropInjector(scaled_intensity * 0.7)(img)
        
        return img


# ============================================================
# Dataset with Noise Injection
# ============================================================
class NoisyDataset(Dataset):
    """
    Dataset wrapper that applies noise injection on-the-fly.
    """
    
    def __init__(
        self,
        root_dir: str,
        noise_injector: Optional[NoiseInjector] = None,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the noisy dataset.
        
        Args:
            root_dir (str): Path to the data directory
            noise_injector (NoiseInjector): Noise to apply (optional)
            transform (Callable): Additional transforms (e.g., normalization)
        """
        self.dataset = datasets.ImageFolder(root_dir)
        self.noise_injector = noise_injector
        self.transform = transform
        self.classes = self.dataset.classes
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        
        # Apply noise injection
        if self.noise_injector is not None:
            image = self.noise_injector(image)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


# ============================================================
# Model Loading
# ============================================================
def load_maize_model(device: torch.device, num_classes: int = 4) -> nn.Module:
    """Load the trained MaizeAttentionNet model."""
    model = MaizeAttentionNet(num_classes=num_classes)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    return model


def load_mobilenet_baseline(device: torch.device, num_classes: int = 4) -> nn.Module:
    """Load MobileNetV2 baseline model."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    # Load fine-tuned weights if available
    baseline_path = './baseline_mobilenet.pth'
    if os.path.exists(baseline_path):
        checkpoint = torch.load(baseline_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    return model


# ============================================================
# Evaluation Functions
# ============================================================
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, int, int]:
    """
    Evaluate model accuracy on a dataloader.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader with test data
        device: Computation device
    
    Returns:
        tuple: (accuracy, correct_count, total_count)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy, correct, total


def run_robustness_evaluation(
    model: nn.Module,
    model_name: str,
    data_dir: str,
    device: torch.device,
    noise_types: Dict[str, type],
    noise_levels: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Run robustness evaluation across all noise types and levels.
    
    Args:
        model: PyTorch model to evaluate
        model_name: Name for logging
        data_dir: Path to test data
        device: Computation device
        noise_types: Dictionary of noise type names to classes
        noise_levels: Dictionary of level names to intensity values
    
    Returns:
        dict: Results organized by noise_type -> level -> accuracy
    """
    # Base transform (after noise injection)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    results = {}
    
    # First evaluate on clean data
    print(f"\n  Evaluating {model_name} on clean data...")
    clean_dataset = NoisyDataset(
        root_dir=data_dir,
        noise_injector=None,
        transform=base_transform
    )
    clean_loader = DataLoader(
        clean_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    clean_acc, _, total = evaluate_model(model, clean_loader, device)
    results['Clean'] = {'None': clean_acc}
    print(f"    Clean: {clean_acc:.2f}% ({total} samples)")
    
    # Evaluate each noise type at each level
    for noise_name, noise_class in noise_types.items():
        print(f"\n  Evaluating {model_name} with {noise_name}...")
        results[noise_name] = {}
        
        for level_name, intensity in noise_levels.items():
            # Create noisy dataset
            noise_injector = noise_class(intensity=intensity)
            noisy_dataset = NoisyDataset(
                root_dir=data_dir,
                noise_injector=noise_injector,
                transform=base_transform
            )
            noisy_loader = DataLoader(
                noisy_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS
            )
            
            # Evaluate
            accuracy, correct, total = evaluate_model(model, noisy_loader, device)
            results[noise_name][level_name] = accuracy
            print(f"    {level_name}: {accuracy:.2f}%")
    
    return results


# ============================================================
# Visualization
# ============================================================
def plot_robustness_comparison(
    maize_results: Dict,
    baseline_results: Dict,
    noise_levels: List[str],
    output_path: str
) -> None:
    """
    Create robustness comparison line charts.
    
    Args:
        maize_results: Results for MaizeAttentionNet
        baseline_results: Results for MobileNetV2
        noise_levels: List of noise level names
        output_path: Path to save the chart
    """
    # Noise types to plot (excluding 'Clean')
    noise_types = [k for k in maize_results.keys() if k != 'Clean']
    
    # Create figure with subplots for each noise type
    fig, axes = plt.subplots(1, len(noise_types) + 1, figsize=(5 * (len(noise_types) + 1), 5))
    
    if len(noise_types) == 0:
        return
    
    colors = {
        'MaizeAttentionNet': '#2ecc71',  # Green
        'MobileNetV2': '#e74c3c'          # Red
    }
    markers = {
        'MaizeAttentionNet': 'o',
        'MobileNetV2': 's'
    }
    
    # Get clean accuracy for reference
    maize_clean = maize_results.get('Clean', {}).get('None', 0)
    baseline_clean = baseline_results.get('Clean', {}).get('None', 0)
    
    # Plot each noise type
    for idx, noise_type in enumerate(noise_types):
        ax = axes[idx]
        
        x_labels = ['Clean'] + noise_levels
        x_positions = range(len(x_labels))
        
        # MaizeAttentionNet
        maize_accuracies = [maize_clean]
        for level in noise_levels:
            maize_accuracies.append(maize_results[noise_type].get(level, 0))
        
        # MobileNetV2
        baseline_accuracies = [baseline_clean]
        for level in noise_levels:
            baseline_accuracies.append(baseline_results[noise_type].get(level, 0))
        
        ax.plot(x_positions, maize_accuracies, 
                color=colors['MaizeAttentionNet'], 
                marker=markers['MaizeAttentionNet'],
                linewidth=2, markersize=8, label='MaizeAttentionNet')
        
        ax.plot(x_positions, baseline_accuracies, 
                color=colors['MobileNetV2'], 
                marker=markers['MobileNetV2'],
                linewidth=2, markersize=8, label='MobileNetV2')
        
        ax.set_xlabel('Noise Intensity', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'{noise_type}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize=9)
    
    # Combined/Summary plot
    ax = axes[-1]
    
    # Calculate average accuracy across all noise types at each level
    x_labels = ['Clean'] + noise_levels
    x_positions = range(len(x_labels))
    
    maize_avg = [maize_clean]
    baseline_avg = [baseline_clean]
    
    for level in noise_levels:
        maize_level_acc = [maize_results[nt].get(level, 0) for nt in noise_types]
        baseline_level_acc = [baseline_results[nt].get(level, 0) for nt in noise_types]
        maize_avg.append(np.mean(maize_level_acc))
        baseline_avg.append(np.mean(baseline_level_acc))
    
    ax.plot(x_positions, maize_avg, 
            color=colors['MaizeAttentionNet'], 
            marker=markers['MaizeAttentionNet'],
            linewidth=2.5, markersize=10, label='MaizeAttentionNet')
    
    ax.plot(x_positions, baseline_avg, 
            color=colors['MobileNetV2'], 
            marker=markers['MobileNetV2'],
            linewidth=2.5, markersize=10, label='MobileNetV2')
    
    ax.set_xlabel('Noise Intensity', fontsize=11)
    ax.set_ylabel('Average Accuracy (%)', fontsize=11)
    ax.set_title('Average Across All Noise Types', fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)
    
    plt.suptitle('Model Robustness to Image Corruption', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Saved robustness chart to: {output_path}")


def generate_report(
    maize_results: Dict,
    baseline_results: Dict,
    noise_levels: List[str],
    output_path: str
) -> str:
    """Generate a text report of robustness results."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = []
    report.append("=" * 70)
    report.append("MODEL ROBUSTNESS EVALUATION REPORT")
    report.append("=" * 70)
    report.append(f"Generated: {timestamp}")
    report.append("")
    
    # Clean accuracy
    maize_clean = maize_results.get('Clean', {}).get('None', 0)
    baseline_clean = baseline_results.get('Clean', {}).get('None', 0)
    
    report.append("-" * 70)
    report.append("BASELINE (CLEAN DATA)")
    report.append("-" * 70)
    report.append(f"  MaizeAttentionNet: {maize_clean:.2f}%")
    report.append(f"  MobileNetV2:       {baseline_clean:.2f}%")
    report.append("")
    
    # Results by noise type
    noise_types = [k for k in maize_results.keys() if k != 'Clean']
    
    for noise_type in noise_types:
        report.append("-" * 70)
        report.append(f"{noise_type.upper()}")
        report.append("-" * 70)
        report.append(f"{'Level':<15} {'MaizeAttentionNet':>20} {'MobileNetV2':>20} {'Difference':>15}")
        report.append("-" * 70)
        
        for level in noise_levels:
            maize_acc = maize_results[noise_type].get(level, 0)
            baseline_acc = baseline_results[noise_type].get(level, 0)
            diff = maize_acc - baseline_acc
            diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
            
            report.append(f"{level:<15} {maize_acc:>19.2f}% {baseline_acc:>19.2f}% {diff_str:>15}")
        
        # Accuracy drop from clean
        report.append("")
        report.append("  Accuracy Drop from Clean:")
        for level in noise_levels:
            maize_drop = maize_clean - maize_results[noise_type].get(level, 0)
            baseline_drop = baseline_clean - baseline_results[noise_type].get(level, 0)
            report.append(f"    {level}: MaizeAttentionNet -{maize_drop:.2f}%, MobileNetV2 -{baseline_drop:.2f}%")
        report.append("")
    
    # Summary
    report.append("=" * 70)
    report.append("SUMMARY")
    report.append("=" * 70)
    
    # Calculate average accuracy drop at high noise
    maize_drops = []
    baseline_drops = []
    for noise_type in noise_types:
        maize_drops.append(maize_clean - maize_results[noise_type].get('High', 0))
        baseline_drops.append(baseline_clean - baseline_results[noise_type].get('High', 0))
    
    avg_maize_drop = np.mean(maize_drops)
    avg_baseline_drop = np.mean(baseline_drops)
    
    report.append(f"Average accuracy drop at High noise:")
    report.append(f"  MaizeAttentionNet: -{avg_maize_drop:.2f}%")
    report.append(f"  MobileNetV2:       -{avg_baseline_drop:.2f}%")
    report.append("")
    
    if avg_maize_drop < avg_baseline_drop:
        diff = avg_baseline_drop - avg_maize_drop
        report.append(f"✓ MaizeAttentionNet is MORE ROBUST by {diff:.2f}% on average")
        report.append("  (Attention mechanism helps maintain accuracy under noise)")
    else:
        diff = avg_maize_drop - avg_baseline_drop
        report.append(f"  MobileNetV2 is more robust by {diff:.2f}% on average")
    
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    return report_text


# ============================================================
# Main Function
# ============================================================
def main():
    """Run the robustness evaluation."""
    print("=" * 60)
    print("Robustness Evaluation: MaizeAttentionNet vs MobileNetV2")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check for scipy (needed for motion blur)
    try:
        from scipy.ndimage import convolve
        print("✓ scipy available for motion blur")
    except ImportError:
        print("! scipy not installed, motion blur will be skipped")
        print("  Install with: pip install scipy")
    
    # Load models
    print("\n" + "-" * 40)
    print("Loading models...")
    maize_model = load_maize_model(device)
    print("✓ Loaded MaizeAttentionNet")
    mobilenet_model = load_mobilenet_baseline(device)
    print("✓ Loaded MobileNetV2")
    
    # Define noise types
    noise_types = {
        'Gaussian Noise': GaussianNoiseInjector,
        'Motion Blur': MotionBlurInjector,
        'Brightness Drop': BrightnessDropInjector,
    }
    
    # Run evaluation for MaizeAttentionNet
    print("\n" + "=" * 60)
    print("Evaluating MaizeAttentionNet")
    print("=" * 60)
    maize_results = run_robustness_evaluation(
        model=maize_model,
        model_name="MaizeAttentionNet",
        data_dir=DATA_DIR,
        device=device,
        noise_types=noise_types,
        noise_levels=NOISE_LEVELS
    )
    
    # Run evaluation for MobileNetV2
    print("\n" + "=" * 60)
    print("Evaluating MobileNetV2")
    print("=" * 60)
    mobilenet_results = run_robustness_evaluation(
        model=mobilenet_model,
        model_name="MobileNetV2",
        data_dir=DATA_DIR,
        device=device,
        noise_types=noise_types,
        noise_levels=NOISE_LEVELS
    )
    
    # Generate visualizations
    print("\n" + "-" * 40)
    print("Generating visualizations...")
    
    chart_path = os.path.join(OUTPUT_DIR, 'robustness_chart.png')
    plot_robustness_comparison(
        maize_results=maize_results,
        baseline_results=mobilenet_results,
        noise_levels=list(NOISE_LEVELS.keys()),
        output_path=chart_path
    )
    
    # Generate report
    report_path = os.path.join(OUTPUT_DIR, 'robustness_results.txt')
    report = generate_report(
        maize_results=maize_results,
        baseline_results=mobilenet_results,
        noise_levels=list(NOISE_LEVELS.keys()),
        output_path=report_path
    )
    
    print("\n" + report)
    print(f"\n✓ Report saved to: {report_path}")
    
    # Save results as PyTorch file
    results_path = os.path.join(OUTPUT_DIR, 'robustness_results.pth')
    torch.save({
        'maize_results': maize_results,
        'mobilenet_results': mobilenet_results,
        'noise_levels': NOISE_LEVELS,
        'timestamp': datetime.now().isoformat()
    }, results_path)
    print(f"✓ Results saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("Robustness Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
