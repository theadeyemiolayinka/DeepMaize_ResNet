"""
Model Benchmarking Script for MaizeAttentionNet.

This script performs comprehensive efficiency benchmarking comparing
the custom MaizeAttentionNet with MobileNetV2 baseline.

Metrics computed:
- Total Parameters (Millions)
- FLOPs (Floating Point Operations)
- Inference Latency (CPU and GPU/MPS)
- Model Size on Disk (MB)

Deep Learning Course Project - 400L
"""

import os
import sys
import time
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models

# Import our custom model
from model import MaizeAttentionNet

# Try to import thop for FLOPs calculation
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: 'thop' not installed. Install with: pip install thop")
    print("FLOPs calculation will be skipped.\n")


# ============================================================
# Configuration
# ============================================================
MODEL_PATH = './best_maize_model.pth'
OUTPUT_DIR = './evaluation_results'
OUTPUT_FILE = 'efficiency_results.txt'
INPUT_SIZE = (1, 3, 224, 224)  # Batch, Channels, Height, Width
NUM_WARMUP_RUNS = 10
NUM_BENCHMARK_RUNS = 100


# ============================================================
# Device and Hardware Detection
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


def get_hardware_info() -> Dict[str, str]:
    """
    Get hardware information for the benchmark report.
    
    Returns:
        dict: Hardware information including CPU, GPU names
    """
    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
    }
    
    # Get CPU info
    if platform.system() == 'Darwin':  # macOS
        try:
            # Get CPU brand on macOS
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True
            )
            info['cpu_name'] = result.stdout.strip() or 'Unknown'
        except:
            info['cpu_name'] = platform.processor() or 'Unknown'
        
        # Check for Apple Silicon
        try:
            result = subprocess.run(
                ['sysctl', '-n', 'hw.optional.arm64'],
                capture_output=True, text=True
            )
            if result.stdout.strip() == '1':
                # Get chip name
                result2 = subprocess.run(
                    ['system_profiler', 'SPHardwareDataType'],
                    capture_output=True, text=True
                )
                for line in result2.stdout.split('\n'):
                    if 'Chip' in line:
                        info['cpu_name'] = line.split(':')[1].strip()
                        break
        except:
            pass
            
    elif platform.system() == 'Linux':
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        info['cpu_name'] = line.split(':')[1].strip()
                        break
        except:
            info['cpu_name'] = 'Unknown'
    elif platform.system() == 'Windows':
        info['cpu_name'] = platform.processor()
    else:
        info['cpu_name'] = 'Unknown'
    
    # Get GPU info
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        info['cuda_version'] = torch.version.cuda or 'N/A'
    elif torch.backends.mps.is_available():
        info['gpu_name'] = 'Apple MPS (Metal Performance Shaders)'
        info['gpu_memory'] = 'Unified Memory'
        info['cuda_version'] = 'N/A (MPS)'
    else:
        info['gpu_name'] = 'None (CPU only)'
        info['gpu_memory'] = 'N/A'
        info['cuda_version'] = 'N/A'
    
    return info


# ============================================================
# Model Loading
# ============================================================
def load_maize_model(device: torch.device, num_classes: int = 4) -> nn.Module:
    """
    Load the trained MaizeAttentionNet model.
    
    Args:
        device: Device to load model on
        num_classes: Number of output classes
    
    Returns:
        nn.Module: Loaded model
    """
    model = MaizeAttentionNet(num_classes=num_classes)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded MaizeAttentionNet from: {MODEL_PATH}")
    else:
        print(f"! MaizeAttentionNet checkpoint not found, using random weights")
    
    model = model.to(device)
    model.eval()
    return model


def load_mobilenet_baseline(device: torch.device, num_classes: int = 4) -> nn.Module:
    """
    Load MobileNetV2 as baseline model.
    
    Args:
        device: Device to load model on
        num_classes: Number of output classes
    
    Returns:
        nn.Module: MobileNetV2 model
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Modify classifier for our number of classes
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    model = model.to(device)
    model.eval()
    print("✓ Loaded MobileNetV2 baseline")
    return model


# ============================================================
# Benchmarking Functions
# ============================================================
def count_parameters(model: nn.Module) -> Tuple[int, float]:
    """
    Count total and trainable parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (total_params, params_in_millions)
    """
    total_params = sum(p.numel() for p in model.parameters())
    params_millions = total_params / 1e6
    return total_params, params_millions


def calculate_flops(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: torch.device
) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculate FLOPs using thop library.
    
    Note: Creates a fresh copy of the model on CPU for calculation
    to avoid MPS float64 compatibility issues.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (B, C, H, W)
        device: Computation device (unused, kept for API compatibility)
    
    Returns:
        tuple: (flops_value, formatted_string)
    """
    if not THOP_AVAILABLE:
        return None, 'N/A (thop not installed)'
    
    import copy
    
    try:
        # Create a fresh copy of the model on CPU
        # This avoids issues with MPS not supporting float64
        model_cpu = copy.deepcopy(model).to('cpu')
        model_cpu.eval()
        
        # Ensure all parameters are float32
        model_cpu = model_cpu.float()
        
        dummy_input = torch.randn(input_size, dtype=torch.float32)
        
        flops, params = profile(model_cpu, inputs=(dummy_input,), verbose=False)
        flops_formatted, _ = clever_format([flops, params], "%.2f")
        
        # Clean up the copy
        del model_cpu
        
        return flops, flops_formatted
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        return None, 'Error'


def measure_inference_latency(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: torch.device,
    num_warmup: int = NUM_WARMUP_RUNS,
    num_runs: int = NUM_BENCHMARK_RUNS
) -> Tuple[float, float, float]:
    """
    Measure inference latency with warm-up runs.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Computation device
        num_warmup: Number of warm-up iterations
        num_runs: Number of benchmark iterations
    
    Returns:
        tuple: (mean_ms, std_ms, min_ms)
    """
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warm-up runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    # Synchronize if using GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    # Benchmark runs
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            
            # Synchronize for accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
    
    import numpy as np
    latencies = np.array(latencies)
    return float(np.mean(latencies)), float(np.std(latencies)), float(np.min(latencies))


def get_model_size_mb(model: nn.Module, temp_path: str = '/tmp/temp_model.pth') -> float:
    """
    Get the model size on disk in MB.
    
    Args:
        model: PyTorch model
        temp_path: Temporary path to save model
    
    Returns:
        float: Model size in megabytes
    """
    # Save model state dict
    torch.save(model.state_dict(), temp_path)
    
    # Get file size
    size_bytes = os.path.getsize(temp_path)
    size_mb = size_bytes / (1024 * 1024)
    
    # Clean up
    os.remove(temp_path)
    
    return size_mb


def get_saved_model_size_mb(model_path: str) -> Optional[float]:
    """
    Get the size of a saved model file.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        float or None: Model size in MB, or None if file doesn't exist
    """
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        return size_bytes / (1024 * 1024)
    return None


# ============================================================
# Report Generation
# ============================================================
def generate_report(
    maize_metrics: Dict,
    mobilenet_metrics: Dict,
    hardware_info: Dict,
    output_path: str
) -> str:
    """
    Generate a formatted benchmark report.
    
    Args:
        maize_metrics: Metrics for MaizeAttentionNet
        mobilenet_metrics: Metrics for MobileNetV2
        hardware_info: Hardware information
        output_path: Path to save the report
    
    Returns:
        str: The formatted report
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Build report
    report = []
    report.append("=" * 70)
    report.append("MODEL EFFICIENCY BENCHMARK REPORT")
    report.append("=" * 70)
    report.append(f"Generated: {timestamp}")
    report.append("")
    
    # Hardware section
    report.append("-" * 70)
    report.append("HARDWARE INFORMATION")
    report.append("-" * 70)
    report.append(f"  Platform:        {hardware_info['platform']} ({hardware_info['platform_version'][:50]}...)")
    report.append(f"  CPU:             {hardware_info['cpu_name']}")
    report.append(f"  GPU:             {hardware_info['gpu_name']}")
    report.append(f"  GPU Memory:      {hardware_info['gpu_memory']}")
    report.append(f"  CUDA Version:    {hardware_info['cuda_version']}")
    report.append(f"  PyTorch Version: {hardware_info['pytorch_version']}")
    report.append(f"  Python Version:  {hardware_info['python_version']}")
    report.append("")
    
    # Benchmark configuration
    report.append("-" * 70)
    report.append("BENCHMARK CONFIGURATION")
    report.append("-" * 70)
    report.append(f"  Input Size:      {INPUT_SIZE}")
    report.append(f"  Warm-up Runs:    {NUM_WARMUP_RUNS}")
    report.append(f"  Benchmark Runs:  {NUM_BENCHMARK_RUNS}")
    report.append("")
    
    # Results table
    report.append("-" * 70)
    report.append("BENCHMARK RESULTS")
    report.append("-" * 70)
    report.append("")
    
    # Table header
    header = f"{'Metric':<30} {'MaizeAttentionNet':>18} {'MobileNetV2':>18}"
    separator = "-" * 70
    
    report.append(header)
    report.append(separator)
    
    # Parameters
    maize_params = f"{maize_metrics['params_millions']:.2f} M"
    mobile_params = f"{mobilenet_metrics['params_millions']:.2f} M"
    report.append(f"{'Parameters':<30} {maize_params:>18} {mobile_params:>18}")
    
    # FLOPs
    maize_flops = maize_metrics.get('flops_formatted', 'N/A')
    mobile_flops = mobilenet_metrics.get('flops_formatted', 'N/A')
    report.append(f"{'FLOPs':<30} {maize_flops:>18} {mobile_flops:>18}")
    
    # Model Size
    maize_size = f"{maize_metrics['model_size_mb']:.2f} MB"
    mobile_size = f"{mobilenet_metrics['model_size_mb']:.2f} MB"
    report.append(f"{'Model Size (disk)':<30} {maize_size:>18} {mobile_size:>18}")
    
    report.append(separator)
    
    # CPU Latency
    if 'cpu_latency_mean' in maize_metrics:
        maize_cpu = f"{maize_metrics['cpu_latency_mean']:.2f} ± {maize_metrics['cpu_latency_std']:.2f} ms"
        mobile_cpu = f"{mobilenet_metrics['cpu_latency_mean']:.2f} ± {mobilenet_metrics['cpu_latency_std']:.2f} ms"
        report.append(f"{'Latency (CPU)':<30} {maize_cpu:>18} {mobile_cpu:>18}")
    
    # GPU/MPS Latency
    if 'gpu_latency_mean' in maize_metrics:
        device_name = 'MPS' if hardware_info['gpu_name'].startswith('Apple') else 'GPU'
        maize_gpu = f"{maize_metrics['gpu_latency_mean']:.2f} ± {maize_metrics['gpu_latency_std']:.2f} ms"
        mobile_gpu = f"{mobilenet_metrics['gpu_latency_mean']:.2f} ± {mobilenet_metrics['gpu_latency_std']:.2f} ms"
        report.append(f"{'Latency (' + device_name + ')':<30} {maize_gpu:>18} {mobile_gpu:>18}")
    
    report.append(separator)
    report.append("")
    
    # Analysis section
    report.append("-" * 70)
    report.append("ANALYSIS")
    report.append("-" * 70)
    
    # Parameter comparison
    param_ratio = maize_metrics['params_millions'] / mobilenet_metrics['params_millions']
    if param_ratio < 1:
        report.append(f"  • MaizeAttentionNet has {1/param_ratio:.1f}x fewer parameters")
    else:
        report.append(f"  • MaizeAttentionNet has {param_ratio:.1f}x more parameters")
    
    # Size comparison
    size_ratio = maize_metrics['model_size_mb'] / mobilenet_metrics['model_size_mb']
    if size_ratio < 1:
        report.append(f"  • MaizeAttentionNet is {1/size_ratio:.1f}x smaller on disk")
    else:
        report.append(f"  • MaizeAttentionNet is {size_ratio:.1f}x larger on disk")
    
    # Latency comparison (GPU/MPS)
    if 'gpu_latency_mean' in maize_metrics:
        latency_ratio = maize_metrics['gpu_latency_mean'] / mobilenet_metrics['gpu_latency_mean']
        if latency_ratio < 1:
            report.append(f"  • MaizeAttentionNet is {1/latency_ratio:.1f}x faster on GPU/MPS")
        else:
            report.append(f"  • MaizeAttentionNet is {latency_ratio:.1f}x slower on GPU/MPS")
    
    report.append("")
    report.append("=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    # Join and save
    report_text = "\n".join(report)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    return report_text


# ============================================================
# Main Benchmark Function
# ============================================================
def run_benchmark() -> Dict:
    """
    Run the complete benchmarking suite.
    
    Returns:
        dict: All benchmark results
    """
    print("=" * 60)
    print("Model Efficiency Benchmark")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get hardware info
    print("Detecting hardware...")
    hardware_info = get_hardware_info()
    print(f"  CPU: {hardware_info['cpu_name']}")
    print(f"  GPU: {hardware_info['gpu_name']}")
    print()
    
    # Get devices
    gpu_device = get_device()
    cpu_device = torch.device('cpu')
    
    print(f"Primary compute device: {gpu_device}")
    print()
    
    # Load models
    print("-" * 40)
    print("Loading models...")
    maize_model = load_maize_model(gpu_device)
    mobilenet_model = load_mobilenet_baseline(gpu_device)
    print()
    
    # Initialize metrics dictionaries
    maize_metrics = {}
    mobilenet_metrics = {}
    
    # Count parameters
    print("-" * 40)
    print("Counting parameters...")
    maize_total, maize_millions = count_parameters(maize_model)
    mobilenet_total, mobilenet_millions = count_parameters(mobilenet_model)
    
    maize_metrics['total_params'] = maize_total
    maize_metrics['params_millions'] = maize_millions
    mobilenet_metrics['total_params'] = mobilenet_total
    mobilenet_metrics['params_millions'] = mobilenet_millions
    
    print(f"  MaizeAttentionNet: {maize_millions:.2f}M parameters")
    print(f"  MobileNetV2:       {mobilenet_millions:.2f}M parameters")
    print()
    
    # Calculate FLOPs
    print("-" * 40)
    print("Calculating FLOPs...")
    maize_flops, maize_flops_fmt = calculate_flops(maize_model, INPUT_SIZE, gpu_device)
    mobilenet_flops, mobilenet_flops_fmt = calculate_flops(mobilenet_model, INPUT_SIZE, gpu_device)
    
    maize_metrics['flops'] = maize_flops
    maize_metrics['flops_formatted'] = maize_flops_fmt
    mobilenet_metrics['flops'] = mobilenet_flops
    mobilenet_metrics['flops_formatted'] = mobilenet_flops_fmt
    
    print(f"  MaizeAttentionNet: {maize_flops_fmt}")
    print(f"  MobileNetV2:       {mobilenet_flops_fmt}")
    print()
    
    # Get model sizes
    print("-" * 40)
    print("Measuring model sizes...")
    maize_size = get_model_size_mb(maize_model)
    mobilenet_size = get_model_size_mb(mobilenet_model)
    
    # Also get the actual saved model size if available
    saved_size = get_saved_model_size_mb(MODEL_PATH)
    if saved_size:
        print(f"  MaizeAttentionNet (saved checkpoint): {saved_size:.2f} MB")
    
    maize_metrics['model_size_mb'] = maize_size
    mobilenet_metrics['model_size_mb'] = mobilenet_size
    
    print(f"  MaizeAttentionNet: {maize_size:.2f} MB")
    print(f"  MobileNetV2:       {mobilenet_size:.2f} MB")
    print()
    
    # Measure CPU latency
    print("-" * 40)
    print(f"Measuring CPU latency ({NUM_BENCHMARK_RUNS} runs)...")
    
    # Move models to CPU
    maize_cpu = maize_model.to(cpu_device)
    mobilenet_cpu = mobilenet_model.to(cpu_device)
    
    maize_cpu_mean, maize_cpu_std, maize_cpu_min = measure_inference_latency(
        maize_cpu, INPUT_SIZE, cpu_device
    )
    mobilenet_cpu_mean, mobilenet_cpu_std, mobilenet_cpu_min = measure_inference_latency(
        mobilenet_cpu, INPUT_SIZE, cpu_device
    )
    
    maize_metrics['cpu_latency_mean'] = maize_cpu_mean
    maize_metrics['cpu_latency_std'] = maize_cpu_std
    maize_metrics['cpu_latency_min'] = maize_cpu_min
    mobilenet_metrics['cpu_latency_mean'] = mobilenet_cpu_mean
    mobilenet_metrics['cpu_latency_std'] = mobilenet_cpu_std
    mobilenet_metrics['cpu_latency_min'] = mobilenet_cpu_min
    
    print(f"  MaizeAttentionNet: {maize_cpu_mean:.2f} ± {maize_cpu_std:.2f} ms")
    print(f"  MobileNetV2:       {mobilenet_cpu_mean:.2f} ± {mobilenet_cpu_std:.2f} ms")
    print()
    
    # Measure GPU/MPS latency if available
    if gpu_device.type != 'cpu':
        device_name = 'MPS' if gpu_device.type == 'mps' else 'GPU'
        print("-" * 40)
        print(f"Measuring {device_name} latency ({NUM_BENCHMARK_RUNS} runs)...")
        
        # Reload fresh models for GPU/MPS testing to ensure clean state
        # This avoids any issues from previous CPU testing
        maize_gpu = load_maize_model(gpu_device)
        mobilenet_gpu = load_mobilenet_baseline(gpu_device)
        
        maize_gpu_mean, maize_gpu_std, maize_gpu_min = measure_inference_latency(
            maize_gpu, INPUT_SIZE, gpu_device
        )
        mobilenet_gpu_mean, mobilenet_gpu_std, mobilenet_gpu_min = measure_inference_latency(
            mobilenet_gpu, INPUT_SIZE, gpu_device
        )
        
        maize_metrics['gpu_latency_mean'] = maize_gpu_mean
        maize_metrics['gpu_latency_std'] = maize_gpu_std
        maize_metrics['gpu_latency_min'] = maize_gpu_min
        mobilenet_metrics['gpu_latency_mean'] = mobilenet_gpu_mean
        mobilenet_metrics['gpu_latency_std'] = mobilenet_gpu_std
        mobilenet_metrics['gpu_latency_min'] = mobilenet_gpu_min
        
        print(f"  MaizeAttentionNet: {maize_gpu_mean:.2f} ± {maize_gpu_std:.2f} ms")
        print(f"  MobileNetV2:       {mobilenet_gpu_mean:.2f} ± {mobilenet_gpu_std:.2f} ms")
        print()
    
    # Generate report
    print("-" * 40)
    print("Generating report...")
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    report = generate_report(maize_metrics, mobilenet_metrics, hardware_info, output_path)
    
    print()
    print(report)
    print()
    print(f"✓ Report saved to: {output_path}")
    
    return {
        'maize_metrics': maize_metrics,
        'mobilenet_metrics': mobilenet_metrics,
        'hardware_info': hardware_info
    }


# ============================================================
# Entry Point
# ============================================================
def main():
    """Main entry point."""
    results = run_benchmark()
    
    print()
    print("=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
