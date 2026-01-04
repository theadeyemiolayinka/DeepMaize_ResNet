"""
Training Script for MaizeAttentionNet: Maize Disease Classification.

This script trains the custom MaizeAttentionNet model on maize leaf images
for disease classification. Includes data augmentation, training/validation
loops, and best model checkpointing.

Deep Learning Course Project - 400L
"""

import os
import time
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Import our custom model
from model import MaizeAttentionNet

# ============================================================
# Reproducibility - Set random seeds
# ============================================================
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# Device Selection - Works on Windows (CUDA), Mac (MPS), or CPU
# ============================================================
def get_device():
    """
    Get the best available device for training.
    
    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device: The selected device
    """
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
# Data Transforms with Augmentation
# ============================================================
def get_transforms():
    """
    Get data transforms for training and validation.
    
    Training: Augmentation + Normalization
    Validation: Only Resize + Normalization
    
    Returns:
        dict: Dictionary containing 'train' and 'val' transforms
    """
    # ImageNet mean and std (commonly used for transfer learning compatibility)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    }
    
    return data_transforms


# ============================================================
# Data Loading
# ============================================================
def load_data(data_dir, batch_size=32, val_split=0.2, num_workers=0):
    """
    Load and prepare the dataset with train/validation split.
    
    Expects data organized as:
        data_dir/
            class_1/
                img1.jpg
                img2.jpg
                ...
            class_2/
                ...
    
    Args:
        data_dir (str): Path to the data directory
        batch_size (int): Batch size for data loaders
        val_split (float): Fraction of data for validation
        num_workers (int): Number of worker processes for data loading
    
    Returns:
        tuple: (dataloaders, dataset_sizes, class_names)
    """
    data_transforms = get_transforms()
    
    # Load the full dataset with training transforms first
    full_dataset = datasets.ImageFolder(data_dir)
    
    # Get class names
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    print(f"\nDataset loaded from: {data_dir}")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Total samples: {len(full_dataset)}")
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Split the dataset
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # Apply appropriate transforms
    # We need to create new datasets with transforms since random_split
    # doesn't allow changing transforms
    train_dataset_with_transform = TransformDataset(
        train_dataset, 
        data_transforms['train']
    )
    val_dataset_with_transform = TransformDataset(
        val_dataset, 
        data_transforms['val']
    )
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(
            train_dataset_with_transform,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        ),
        'val': DataLoader(
            val_dataset_with_transform,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    }
    
    dataset_sizes = {
        'train': train_size,
        'val': val_size
    }
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    return dataloaders, dataset_sizes, class_names


class TransformDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset to apply transforms to a subset.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# Training Function
# ============================================================
def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device='cpu',
    save_path='best_maize_model.pth'
):
    """
    Train the model and save the best weights.
    
    Args:
        model: The neural network model
        dataloaders: Dictionary of data loaders ('train', 'val')
        dataset_sizes: Dictionary of dataset sizes
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of training epochs
        device: Device to train on
        save_path: Path to save the best model
    
    Returns:
        model: Trained model with best weights loaded
        history: Dictionary containing training history
    """
    since = time.time()
    
    # Track best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                # Track gradients only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only in train phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Step scheduler after training epoch
            if phase == 'train' and scheduler is not None:
                scheduler.step()
            
            # Calculate epoch metrics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f"{phase.capitalize():5} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")
            
            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # Save the best model weights
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc.item(),
                    'class_names': dataloaders['train'].dataset.subset.dataset.classes
                }, save_path)
                print(f"  ✓ New best model saved! (Acc: {best_acc:.4f})")
    
    # Calculate training time
    time_elapsed = time.time() - since
    print("\n" + "=" * 60)
    print(f"Training Complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc:.4f} (Epoch {best_epoch})")
    print(f"Best model saved to: {save_path}")
    print("=" * 60)
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history


# ============================================================
# Main Training Script
# ============================================================
def main():
    """
    Main function to run the training pipeline.
    """
    # ================== Configuration ==================
    DATA_DIR = './data'           # Path to dataset
    BATCH_SIZE = 32               # Batch size
    NUM_EPOCHS = 25               # Number of training epochs
    LEARNING_RATE = 0.001         # Initial learning rate
    WEIGHT_DECAY = 1e-4           # L2 regularization
    VAL_SPLIT = 0.2               # Validation split ratio
    NUM_WORKERS = 0               # Data loading workers (0 for Windows compatibility)
    DROPOUT_RATE = 0.5            # Dropout rate
    SAVE_PATH = 'best_maize_model.pth'
    
    # ================== Setup ==================
    print("=" * 60)
    print("MaizeAttentionNet - Disease Classification Training")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get device
    device = get_device()
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\nError: Data directory '{DATA_DIR}' not found!")
        print("Please ensure your data is organized as:")
        print("  data/")
        print("    class_1/")
        print("      image1.jpg")
        print("      ...")
        print("    class_2/")
        print("      ...")
        return
    
    # ================== Load Data ==================
    dataloaders, dataset_sizes, class_names = load_data(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        num_workers=NUM_WORKERS
    )
    
    num_classes = len(class_names)
    
    # ================== Create Model ==================
    print("\n" + "-" * 40)
    print("Creating MaizeAttentionNet model...")
    
    model = MaizeAttentionNet(
        num_classes=num_classes,
        in_channels=3,
        dropout_rate=DROPOUT_RATE
    )
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ================== Loss & Optimizer ==================
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler - reduce LR on plateau
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.1
    )
    
    print(f"\nOptimizer: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"Loss Function: CrossEntropyLoss")
    print(f"LR Scheduler: StepLR (step_size=7, gamma=0.1)")
    
    # ================== Train Model ==================
    model, history = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=device,
        save_path=SAVE_PATH
    )
    
    # ================== Save Training History ==================
    history_path = 'training_history.pth'
    torch.save(history, history_path)
    print(f"\nTraining history saved to: {history_path}")
    
    # Print final summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Final Training Accuracy:   {history['train_acc'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"Best Validation Accuracy:  {max(history['val_acc']):.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
