"""
MaizeAttentionNet: A custom CNN with Spatial Attention for Maize Image Classification.

This module implements a deep learning model from scratch using PyTorch,
featuring 4 convolutional blocks and a spatial attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Computes an attention map by applying channel-wise average pooling and
    max pooling, concatenating the results, and passing through a convolutional
    layer to produce a spatial attention map that highlights important regions.
    
    Args:
        kernel_size (int): Kernel size for the convolutional layer. Default: 7
    """
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        
        # Ensure kernel size is odd for proper padding
        assert kernel_size % 2 == 1, "Kernel size must be odd for same padding"
        
        padding = kernel_size // 2
        
        # Convolution to combine avg and max pooled features
        # Input: 2 channels (avg + max), Output: 1 channel (attention map)
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spatial attention.
        
        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Attention-weighted feature map of shape (B, C, H, W)
        """
        # Channel-wise average pooling: (B, C, H, W) -> (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Channel-wise max pooling: (B, C, H, W) -> (B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension: (B, 2, H, W)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Generate attention map: (B, 2, H, W) -> (B, 1, H, W)
        attention_map = self.sigmoid(self.conv(pooled))
        
        # Apply attention to input feature map
        return x * attention_map


class ConvBlock(nn.Module):
    """
    Convolutional Block: Conv2D -> BatchNorm -> ReLU -> MaxPool.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size for convolution. Default: 3
        pool_size (int): Kernel size for max pooling. Default: 2
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2
    ):
        super(ConvBlock, self).__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False  # BatchNorm handles bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H/2, W/2)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class MaizeAttentionNet(nn.Module):
    """
    MaizeAttentionNet: Custom CNN with Spatial Attention for Maize Classification.
    
    Architecture:
        - 4 Convolutional Blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool)
        - Spatial Attention Module applied after conv blocks
        - Global Average Pooling
        - Fully Connected Layers with Dropout
    
    Args:
        num_classes (int): Number of output classes. Default: 4
        in_channels (int): Number of input image channels. Default: 3 (RGB)
        dropout_rate (float): Dropout probability. Default: 0.5
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        in_channels: int = 3,
        dropout_rate: float = 0.5
    ):
        super(MaizeAttentionNet, self).__init__()
        
        self.num_classes = num_classes
        
        # ============================================================
        # Convolutional Backbone (4 blocks, built from scratch)
        # ============================================================
        
        # Block 1: 3 -> 64 channels
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64)
        
        # Block 2: 64 -> 128 channels
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        
        # Block 3: 128 -> 256 channels
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        
        # Block 4: 256 -> 512 channels
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        
        # ============================================================
        # Spatial Attention Module
        # ============================================================
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # ============================================================
        # Global Average Pooling
        # ============================================================
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ============================================================
        # Fully Connected Classifier
        # ============================================================
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He (Kaiming) initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MaizeAttentionNet.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
                              Typically (B, 3, 224, 224) for RGB images
            
        Returns:
            torch.Tensor: Class logits of shape (B, num_classes)
        """
        # Pass through convolutional backbone
        x = self.conv_block1(x)  # (B, 64, H/2, W/2)
        x = self.conv_block2(x)  # (B, 128, H/4, W/4)
        x = self.conv_block3(x)  # (B, 256, H/8, W/8)
        x = self.conv_block4(x)  # (B, 512, H/16, W/16)
        
        # Apply spatial attention
        x = self.spatial_attention(x)  # (B, 512, H/16, W/16)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (B, 512, 1, 1)
        
        # Classification head
        x = self.classifier(x)  # (B, num_classes)
        
        return x
    
    def get_attention_map(self, x: torch.Tensor) -> tuple:
        """
        Get the spatial attention map for visualization.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            
        Returns:
            tuple: (attention_map, output_logits)
                - attention_map: Tensor of shape (B, 1, H/16, W/16)
                - output_logits: Tensor of shape (B, num_classes)
        """
        # Pass through convolutional backbone
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # Compute attention map
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        attention_map = self.spatial_attention.sigmoid(
            self.spatial_attention.conv(pooled)
        )
        
        # Apply attention and continue forward pass
        x = x * attention_map
        x = self.global_avg_pool(x)
        output = self.classifier(x)
        
        return attention_map, output


def create_model(
    num_classes: int = 4,
    in_channels: int = 3,
    dropout_rate: float = 0.5,
    pretrained: bool = False
) -> MaizeAttentionNet:
    """
    Factory function to create a MaizeAttentionNet model.
    
    Args:
        num_classes (int): Number of output classes. Default: 4
        in_channels (int): Number of input channels. Default: 3
        dropout_rate (float): Dropout probability. Default: 0.5
        pretrained (bool): Not used (no pretrained weights available). Default: False
    
    Returns:
        MaizeAttentionNet: Initialized model
    """
    if pretrained:
        print("Warning: No pretrained weights available for MaizeAttentionNet. "
              "Initializing with random weights.")
    
    return MaizeAttentionNet(
        num_classes=num_classes,
        in_channels=in_channels,
        dropout_rate=dropout_rate
    )


# ============================================================
# Example usage and model summary
# ============================================================
if __name__ == "__main__":
    # Create model
    model = MaizeAttentionNet(num_classes=4)
    
    # Print model architecture
    print("=" * 60)
    print("MaizeAttentionNet Architecture")
    print("=" * 60)
    print(model)
    print("=" * 60)
    
    # Test with dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    output = model(dummy_input)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Get attention map
    attention_map, logits = model.get_attention_map(dummy_input)
    print(f"Attention map shape: {attention_map.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
