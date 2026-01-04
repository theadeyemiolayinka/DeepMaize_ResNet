"""
Evaluation Script for MaizeAttentionNet and Baseline Comparison.

This script evaluates the trained MaizeAttentionNet model and compares it
against a pre-trained MobileNetV2 baseline. Generates confusion matrix,
ROC curves, and calculates key metrics.

Deep Learning Course Project - 400L
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# Import our custom model
from model import MaizeAttentionNet

# ============================================================
# Reproducibility
# ============================================================
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# Device Selection
# ============================================================
def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


# ============================================================
# Data Loading for Evaluation
# ============================================================
def get_test_loader(data_dir, batch_size=32):
    """
    Load test data with validation transforms (no augmentation).
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for evaluation
    
    Returns:
        tuple: (dataloader, class_names, dataset_size)
    """
    # Standard transforms for evaluation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader, dataset.classes, len(dataset)


# ============================================================
# Model Evaluation Function
# ============================================================
def evaluate_model(model, dataloader, device, num_classes):
    """
    Evaluate a model and return predictions and probabilities.
    
    Args:
        model: The model to evaluate
        dataloader: Test data loader
        device: Device to run on
        num_classes: Number of classes
    
    Returns:
        tuple: (all_labels, all_preds, all_probs)
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


# ============================================================
# Metrics Calculation
# ============================================================
def calculate_metrics(y_true, y_pred, class_names):
    """
    Calculate and print classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
    
    Returns:
        dict: Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print("\n" + "=" * 50)
    print("Classification Metrics")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\n" + "-" * 50)
    print("Detailed Classification Report:")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return metrics


# ============================================================
# Confusion Matrix Visualization
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """
    Generate and save a confusion matrix plot.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        title: Plot title
        save_path: Path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


# ============================================================
# ROC Curve Visualization
# ============================================================
def plot_roc_curves(y_true, y_probs, class_names, title, save_path):
    """
    Generate and save ROC curves for multi-class classification.
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities (shape: [n_samples, n_classes])
        class_names: List of class names
        title: Plot title
        save_path: Path to save the figure
    """
    num_classes = len(class_names)
    
    # Binarize labels for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, num_classes))
    
    # Calculate ROC curve for each class
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(
            fpr, tpr,
            color=color,
            linewidth=2,
            label=f'{class_name} (AUC = {roc_auc:.3f})'
        )
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curves saved to: {save_path}")


# ============================================================
# Load MaizeAttentionNet Model
# ============================================================
def load_maize_model(model_path, num_classes, device):
    """
    Load the trained MaizeAttentionNet model.
    
    Args:
        model_path: Path to saved model weights
        num_classes: Number of classes
        device: Device to load model on
    
    Returns:
        model: Loaded model
    """
    model = MaizeAttentionNet(num_classes=num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded MaizeAttentionNet from: {model_path}")
    print(f"Best training accuracy: {checkpoint.get('best_acc', 'N/A')}")
    
    return model


# ============================================================
# Baseline: MobileNetV2 (Pre-trained)
# ============================================================
def create_mobilenet_baseline(num_classes, device):
    """
    Create a pre-trained MobileNetV2 model for baseline comparison.
    
    NOTE: This uses torchvision.models for the baseline only.
    Our custom MaizeAttentionNet is built from scratch.
    
    Args:
        num_classes: Number of output classes
        device: Device to load model on
    
    Returns:
        model: MobileNetV2 model with modified classifier
    """
    print("\nCreating MobileNetV2 baseline (pre-trained on ImageNet)...")
    
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Modify the classifier for our number of classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV2 parameters: {total_params:,}")
    
    return model


def train_baseline_briefly(model, dataloader, device, num_epochs=5):
    """
    Briefly fine-tune the MobileNetV2 baseline for fair comparison.
    
    Args:
        model: MobileNetV2 model
        dataloader: Training data loader
        device: Device
        num_epochs: Number of fine-tuning epochs
    
    Returns:
        model: Fine-tuned model
    """
    print(f"\nFine-tuning MobileNetV2 baseline for {num_epochs} epochs...")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total
        print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    
    return model


# ============================================================
# Comparison Visualization
# ============================================================
def plot_comparison(maize_metrics, baseline_metrics, save_path):
    """
    Create a comparison bar chart between models.
    
    Args:
        maize_metrics: Metrics from MaizeAttentionNet
        baseline_metrics: Metrics from MobileNetV2 baseline
        save_path: Path to save the figure
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    maize_values = [
        maize_metrics['accuracy'],
        maize_metrics['precision'],
        maize_metrics['recall'],
        maize_metrics['f1_score']
    ]
    baseline_values = [
        baseline_metrics['accuracy'],
        baseline_metrics['precision'],
        baseline_metrics['recall'],
        baseline_metrics['f1_score']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, maize_values, width, label='MaizeAttentionNet (Ours)', color='#2ecc71')
    bars2 = ax.bar(x + width/2, baseline_values, width, label='MobileNetV2 (Baseline)', color='#3498db')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison: MaizeAttentionNet vs MobileNetV2', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison chart saved to: {save_path}")


# ============================================================
# Main Evaluation Script
# ============================================================
def main():
    """Main evaluation pipeline."""
    
    # ================== Configuration ==================
    DATA_DIR = './data'
    MODEL_PATH = 'best_maize_model.pth'
    OUTPUT_DIR = './evaluation_results'
    BATCH_SIZE = 32
    FINE_TUNE_EPOCHS = 5  # Brief fine-tuning for baseline
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("MaizeAttentionNet - Model Evaluation")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get device
    device = get_device()
    
    # ================== Load Data ==================
    print("\n" + "-" * 40)
    print("Loading evaluation data...")
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found!")
        return
    
    dataloader, class_names, dataset_size = get_test_loader(DATA_DIR, BATCH_SIZE)
    num_classes = len(class_names)
    
    print(f"Dataset: {dataset_size} images")
    print(f"Classes ({num_classes}): {class_names}")
    
    # ================== Evaluate MaizeAttentionNet ==================
    print("\n" + "=" * 60)
    print("Evaluating MaizeAttentionNet (Our Model)")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("Please train the model first using train.py")
        return
    
    maize_model = load_maize_model(MODEL_PATH, num_classes, device)
    
    maize_labels, maize_preds, maize_probs = evaluate_model(
        maize_model, dataloader, device, num_classes
    )
    
    maize_metrics = calculate_metrics(maize_labels, maize_preds, class_names)
    
    # Generate visualizations for MaizeAttentionNet
    plot_confusion_matrix(
        maize_labels, maize_preds, class_names,
        'Confusion Matrix - MaizeAttentionNet',
        os.path.join(OUTPUT_DIR, 'confusion_matrix_maize.png')
    )
    
    plot_roc_curves(
        maize_labels, maize_probs, class_names,
        'ROC Curves - MaizeAttentionNet',
        os.path.join(OUTPUT_DIR, 'roc_curves_maize.png')
    )
    
    # ================== Evaluate MobileNetV2 Baseline ==================
    print("\n" + "=" * 60)
    print("Evaluating MobileNetV2 Baseline (Pre-trained)")
    print("=" * 60)
    
    baseline_model = create_mobilenet_baseline(num_classes, device)
    
    # Fine-tune briefly for fair comparison
    baseline_model = train_baseline_briefly(
        baseline_model, dataloader, device, FINE_TUNE_EPOCHS
    )
    
    baseline_labels, baseline_preds, baseline_probs = evaluate_model(
        baseline_model, dataloader, device, num_classes
    )
    
    baseline_metrics = calculate_metrics(baseline_labels, baseline_preds, class_names)
    
    # Generate visualizations for baseline
    plot_confusion_matrix(
        baseline_labels, baseline_preds, class_names,
        'Confusion Matrix - MobileNetV2 (Baseline)',
        os.path.join(OUTPUT_DIR, 'confusion_matrix_baseline.png')
    )
    
    plot_roc_curves(
        baseline_labels, baseline_probs, class_names,
        'ROC Curves - MobileNetV2 (Baseline)',
        os.path.join(OUTPUT_DIR, 'roc_curves_baseline.png')
    )
    
    # ================== Model Comparison ==================
    print("\n" + "=" * 60)
    print("Model Comparison Summary")
    print("=" * 60)
    
    print("\n{:<25} {:>15} {:>15}".format(
        "Metric", "MaizeAttentionNet", "MobileNetV2"
    ))
    print("-" * 55)
    print("{:<25} {:>15.4f} {:>15.4f}".format(
        "Accuracy", maize_metrics['accuracy'], baseline_metrics['accuracy']
    ))
    print("{:<25} {:>15.4f} {:>15.4f}".format(
        "Precision", maize_metrics['precision'], baseline_metrics['precision']
    ))
    print("{:<25} {:>15.4f} {:>15.4f}".format(
        "Recall", maize_metrics['recall'], baseline_metrics['recall']
    ))
    print("{:<25} {:>15.4f} {:>15.4f}".format(
        "F1-Score", maize_metrics['f1_score'], baseline_metrics['f1_score']
    ))
    
    # Determine winner
    if maize_metrics['accuracy'] > baseline_metrics['accuracy']:
        print("\n✓ MaizeAttentionNet outperforms the baseline!")
    elif maize_metrics['accuracy'] < baseline_metrics['accuracy']:
        print("\n→ Baseline performs better. Consider training for more epochs.")
    else:
        print("\n→ Both models perform equally.")
    
    # Create comparison chart
    plot_comparison(
        maize_metrics, baseline_metrics,
        os.path.join(OUTPUT_DIR, 'model_comparison.png')
    )
    
    # ================== Save Results ==================
    results = {
        'maize_metrics': maize_metrics,
        'baseline_metrics': baseline_metrics,
        'class_names': class_names,
        'dataset_size': dataset_size
    }
    
    results_path = os.path.join(OUTPUT_DIR, 'evaluation_results.pth')
    torch.save(results, results_path)
    print(f"\nResults saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print(f"All visualizations saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
