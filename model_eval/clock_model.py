import torch
import time
import gc
from typing import Dict, Any, Optional
import numpy as np
from contextlib import contextmanager
from tqdm import tqdm

@contextmanager
def torch_timer():
    """Context manager for timing CUDA operations"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start

# def get_gpu_memory_info():
#     """Get current GPU memory usage"""
#     if torch.cuda.is_available():
#         return {
#             'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
#             'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
#             'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
#             'cached': torch.cuda.memory_cached() / 1024**3,  # GB
#         }
#     return {'allocated': 0, 'reserved': 0, 'max_allocated': 0, 'cached': 0}

def get_gpu_memory_info(device=None):
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated(device) / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved(device) / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated(device) / 1024**3,  # GB
            'cached': torch.cuda.memory_cached(device) / 1024**3,  # GB
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0, 'cached': 0}

# def reset_memory_tracking():
#     """Reset memory tracking"""
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.reset_peak_memory_stats()
#         gc.collect()

def reset_memory_tracking(device=None):
    """Reset memory tracking for specified device"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # This clears cache for all devices
        torch.cuda.reset_peak_memory_stats(device)  # This can be device-specific
        gc.collect()

def analyze_model_performance(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    num_batches: int = 10,
    warmup_batches: int = 3,
    loss_fn: Optional[callable] = None,
    model_forward_args: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze model performance for both training and evaluation modes.
    
    Args:
        model: PyTorch model to analyze
        loader: DataLoader with batches
        device: Device to run analysis on ('cuda' or 'cpu')
        num_batches: Number of batches to analyze
        warmup_batches: Number of warmup batches (excluded from timing)
        loss_fn: Loss function for training mode (if None, uses F.mse_loss)
        model_forward_args: Additional arguments for model forward pass
        
    Returns:
        Dictionary containing performance metrics
    """
    
    if loss_fn is None:
        import torch.nn.functional as F
        loss_fn = lambda pred, target: F.mse_loss(pred, target)
    
    if model_forward_args is None:
        model_forward_args = {}
    
    model = model.to(device)
    results = {
        'device': device,
        'num_batches_analyzed': num_batches,
        'warmup_batches': warmup_batches,
    }
    
    # Get model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results['model_info'] = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_params_millions': total_params / 1e6,
        'trainable_params_millions': trainable_params / 1e6,
    }
    
    # Prepare batches
    batches = []
    print("Preparing batches...")
    for i, batch in tqdm(enumerate(loader)):
        if i >= num_batches + warmup_batches:
            break
        # Move batch to device
        if hasattr(batch, 'to'):
            batch = batch.to(device)
        else:
            # Handle different batch types
            if isinstance(batch, (list, tuple)):
                batch = [b.to(device) if hasattr(b, 'to') else b for b in batch]
            elif isinstance(batch, dict):
                batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
        batches.append(batch)
    
    if len(batches) < num_batches + warmup_batches:
        print(f"Warning: Only {len(batches)} batches available, reducing analysis size")
        num_batches = max(1, len(batches) - warmup_batches)
    
    # Analyze evaluation mode
    print("Analyzing evaluation mode...")
    results['eval_mode'] = analyze_mode(
        model, batches, 'eval', device, num_batches, warmup_batches, 
        loss_fn, model_forward_args
    )
    
    # Analyze training mode
    print("Analyzing training mode...")
    results['train_mode'] = analyze_mode(
        model, batches, 'train', device, num_batches, warmup_batches,
        loss_fn, model_forward_args
    )
    
    # Calculate speedup/memory differences
    results['comparison'] = {
        'eval_vs_train_speed_ratio': results['eval_mode']['avg_forward_time'] / results['train_mode']['avg_total_time'],
        'train_memory_overhead_gb': results['train_mode']['avg_memory_allocated'] - results['eval_mode']['avg_memory_allocated'],
        'train_memory_overhead_ratio': results['train_mode']['avg_memory_allocated'] / max(results['eval_mode']['avg_memory_allocated'], 1e-6),
    }

    #reset memory 
    reset_memory_tracking(device=device)
    
    return results

def analyze_mode(
    model: torch.nn.Module,
    batches: list,
    mode: str,
    device: str,
    num_batches: int,
    warmup_batches: int,
    loss_fn: callable,
    model_forward_args: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze performance for a specific mode (train/eval)"""
    
    # Set model mode
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    forward_times = []
    backward_times = []
    total_times = []
    memory_stats = []
    batch_sizes = []
    
    print('inputting graph_batch!')
    print(f"Running in {mode} mode...")
    for i, batch in tqdm(enumerate(batches)):
        reset_memory_tracking(device=device)

        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch.max().item() + 1
        forward_args = {'z' : batch.z, 'pos': batch.pos, 'batch': batch.batch}
        # forward_args = {'z' : batch.z, 'pos': batch.pos, 'batch': batch.batch, 'graph_batch': batch}
        target = batch.y if hasattr(batch, 'y') else torch.randn(batch_size, 1, device=device)
        
        # Extract batch info for your specific data format
        # if hasattr(batch, 'z') and hasattr(batch, 'pos') and hasattr(batch, 'batch'):
        #     # PyTorch Geometric format
        #     batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch.max().item() + 1
        #     forward_args = (batch.z, batch.pos, batch.batch)
        #     target = batch.y if hasattr(batch, 'y') else torch.randn(batch_size, 1, device=device)
        # else:
        #     # Handle other batch formats
        #     if isinstance(batch, (list, tuple)):
        #         forward_args = batch[:-1]  # Assume last element is target
        #         target = batch[-1]
        #         batch_size = target.size(0)
        #     else:
        #         # Fallback - you may need to customize this
        #         batch_size = 32
        #         forward_args = (batch,)
        #         target = torch.randn(batch_size, 1, device=device)
        
        batch_sizes.append(batch_size)
        
        # Apply model forward args
        # forward_args = (*forward_args, *model_forward_args.values()) if model_forward_args else forward_args
        
        total_start = time.perf_counter()
        
        # Forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_start = time.perf_counter()
        
        with torch.set_grad_enabled(mode == 'train'):
            if model_forward_args:
                output = model(**forward_args, **model_forward_args)
            else:
                output = model(**forward_args)
            # if isinstance(forward_args, tuple) and len(forward_args) > 0:
            #     output = model(*forward_args)
            # else:
            #     output = model(forward_args)
            
            # Handle model output format
            if isinstance(output, dict):
                pred = output.get('y', output.get('prediction', list(output.values())[0]))
            else:
                pred = output
            
            # Ensure target and prediction are compatible
            if pred.dim() != target.dim():
                if pred.dim() > target.dim():
                    target = target.view(-1, 1) if target.dim() == 1 else target
                else:
                    pred = pred.view(-1, 1) if pred.dim() == 1 else pred
            
            if pred.size(0) != target.size(0):
                min_size = min(pred.size(0), target.size(0))
                pred = pred[:min_size]
                target = target[:min_size]
            
            loss = loss_fn(pred, target)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_end = time.perf_counter()
        forward_time = forward_end - forward_start
        
        # Backward pass (only in training mode)
        backward_time = 0
        if mode == 'train':
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_start = time.perf_counter()
            
            model.zero_grad()
            loss.backward()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_end = time.perf_counter()
            backward_time = backward_end - backward_start
        
        total_end = time.perf_counter()
        total_time = total_end - total_start
        
        # Skip warmup batches for timing
        if i >= warmup_batches:
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            total_times.append(total_time)
            memory_stats.append(get_gpu_memory_info(device=device))
    
    # Calculate statistics
    analyzed_batch_sizes = batch_sizes[warmup_batches:]
    avg_batch_size = np.mean(analyzed_batch_sizes)
    
    # Extract memory statistics
    allocated_memory = [m['allocated'] for m in memory_stats]
    reserved_memory = [m['reserved'] for m in memory_stats]
    max_allocated_memory = [m['max_allocated'] for m in memory_stats]
    cached_memory = [m['cached'] for m in memory_stats]
    
    return {
        'avg_forward_time': np.mean(forward_times),
        'std_forward_time': np.std(forward_times),
        'avg_backward_time': np.mean(backward_times) if backward_times else 0,
        'std_backward_time': np.std(backward_times) if backward_times else 0,
        'avg_total_time': np.mean(total_times),
        'std_total_time': np.std(total_times),
        'avg_batch_size': avg_batch_size,
        'min_batch_size': np.min(analyzed_batch_sizes),
        'max_batch_size': np.max(analyzed_batch_sizes),
        'std_batch_size': np.std(analyzed_batch_sizes),
        'throughput_samples_per_sec': avg_batch_size / np.mean(total_times),
        
        # Detailed memory statistics
        'memory_allocated': {
            'avg': np.mean(allocated_memory),
            'min': np.min(allocated_memory),
            'max': np.max(allocated_memory),
            'std': np.std(allocated_memory),
            'per_sample_avg': np.mean(allocated_memory) / avg_batch_size,
        },
        'memory_reserved': {
            'avg': np.mean(reserved_memory),
            'min': np.min(reserved_memory),
            'max': np.max(reserved_memory),
            'std': np.std(reserved_memory),
            'per_sample_avg': np.mean(reserved_memory) / avg_batch_size,
        },
        'memory_cached': {
            'avg': np.mean(cached_memory),
            'min': np.min(cached_memory),
            'max': np.max(cached_memory),
            'std': np.std(cached_memory),
        },
        'peak_memory_usage': max(max_allocated_memory),
        
        # Legacy fields for compatibility
        'avg_memory_allocated': np.mean(allocated_memory),
        'max_memory_allocated': max(allocated_memory),
        'avg_memory_reserved': np.mean(reserved_memory),
        
        'detailed_times': {
            'forward_times': forward_times,
            'backward_times': backward_times,
            'total_times': total_times,
        },
        'memory_stats': memory_stats,
        'batch_sizes': analyzed_batch_sizes,
    }

def print_performance_summary(results: Dict[str, Any]):
    """Print a formatted summary of performance results"""
    
    print("=" * 80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Model info
    print(f"\nModel Information:")
    print(f"  Total Parameters: {results['model_info']['total_params_millions']:.2f}M")
    print(f"  Trainable Parameters: {results['model_info']['trainable_params_millions']:.2f}M")
    print(f"  Device: {results['device']}")
    print(f"  Batches Analyzed: {results['num_batches_analyzed']} (+ {results['warmup_batches']} warmup)")
    
    # Performance comparison table
    print(f"\n{'Mode':<10} {'Forward (ms)':<15} {'Backward (ms)':<15} {'Total (ms)':<12} {'Throughput':<15} {'Memory (GB)':<12}")
    print("-" * 90)
    
    for mode_name, mode_data in [('Eval', results['eval_mode']), ('Train', results['train_mode'])]:
        forward_ms = mode_data['avg_forward_time'] * 1000
        backward_ms = mode_data['avg_backward_time'] * 1000
        total_ms = mode_data['avg_total_time'] * 1000
        throughput = mode_data['throughput_samples_per_sec']
        memory_gb = mode_data['avg_memory_allocated']
        
        print(f"{mode_name:<10} {forward_ms:<15.2f} {backward_ms:<15.2f} {total_ms:<12.2f} {throughput:<15.1f} {memory_gb:<12.4f}")
    
    # Comparison metrics
    print(f"\nPerformance Comparison:")
    print(f"  Eval vs Train Speed Ratio: {results['comparison']['eval_vs_train_speed_ratio']:.3f}x")
    print(f"  Training Memory Overhead: {results['comparison']['train_memory_overhead_gb']:.3f} GB")
    print(f"  Training Memory Ratio: {results['comparison']['train_memory_overhead_ratio']:.3f}x")

def print_detailed_memory_breakdown(results: Dict[str, Any]):
    """Print a detailed breakdown of memory usage statistics"""
    
    print("=" * 80)
    print("DETAILED MEMORY ANALYSIS")
    print("=" * 80)
    
    # Helper function to format memory values
    def format_memory(gb_value, unit='GB'):
        if unit == 'GB':
            return f"{gb_value:.3f} GB"
        elif unit == 'MB':
            return f"{gb_value * 1024:.1f} MB"
        else:
            return f"{gb_value:.3f} GB"
    
    # Helper function to format memory stats with std dev
    def format_memory_stats(stats_dict, unit='GB'):
        if unit == 'MB':
            avg = stats_dict['avg'] * 1024
            std = stats_dict['std'] * 1024
            min_val = stats_dict['min'] * 1024
            max_val = stats_dict['max'] * 1024
            per_sample = stats_dict.get('per_sample_avg', 0) * 1024
            return f"{avg:.1f} ± {std:.1f} MB (range: {min_val:.1f} - {max_val:.1f} MB)" + (f", {per_sample:.2f} MB/sample" if per_sample > 0 else "")
        else:
            avg = stats_dict['avg']
            std = stats_dict['std']
            min_val = stats_dict['min']
            max_val = stats_dict['max']
            per_sample = stats_dict.get('per_sample_avg', 0)
            return f"{avg:.3f} ± {std:.3f} GB (range: {min_val:.3f} - {max_val:.3f} GB)" + (f", {per_sample:.4f} GB/sample" if per_sample > 0 else "")
    
    print(f"\nBatch Size Statistics:")
    for mode_name, mode_data in [('Evaluation', results['eval_mode']), ('Training', results['train_mode'])]:
        print(f"  {mode_name} Mode:")
        print(f"    Average Batch Size: {mode_data['avg_batch_size']:.1f}")
        print(f"    Batch Size Range: {mode_data['min_batch_size']} - {mode_data['max_batch_size']}")
        print(f"    Batch Size Std Dev: {mode_data['std_batch_size']:.2f}")
    
    print(f"\n{'Mode':<12} {'Memory Type':<15} {'Statistics':<60}")
    print("-" * 100)
    
    for mode_name, mode_data in [('Evaluation', results['eval_mode']), ('Training', results['train_mode'])]:
        # Allocated Memory
        allocated_stats = format_memory_stats(mode_data['memory_allocated'], 'MB')
        print(f"{mode_name:<12} {'Allocated':<15} {allocated_stats}")
        
        # Reserved Memory  
        reserved_stats = format_memory_stats(mode_data['memory_reserved'], 'MB')
        print(f"{'':>12} {'Reserved':<15} {reserved_stats}")
        
        # Cached Memory
        cached_stats = format_memory_stats(mode_data['memory_cached'], 'MB')
        print(f"{'':>12} {'Cached':<15} {cached_stats}")
        
        # Peak Memory
        peak_memory = mode_data['peak_memory_usage'] * 1024  # Convert to MB
        print(f"{'':>12} {'Peak Usage':<15} {peak_memory:.1f} MB")
        
        # Memory Efficiency
        if mode_data['memory_reserved']['avg'] > 0:
            efficiency = (mode_data['memory_allocated']['avg'] / mode_data['memory_reserved']['avg']) * 100
            print(f"{'':>12} {'Efficiency':<15} {efficiency:.1f}% (allocated/reserved)")
        
        print("-" * 100)
    
    # Memory comparison between modes
    print(f"\nMemory Comparison (Training vs Evaluation):")
    train_allocated = results['train_mode']['memory_allocated']['avg']
    eval_allocated = results['eval_mode']['memory_allocated']['avg']
    train_reserved = results['train_mode']['memory_reserved']['avg']
    eval_reserved = results['eval_mode']['memory_reserved']['avg']
    
    allocated_overhead = (train_allocated - eval_allocated) * 1024  # MB
    allocated_ratio = train_allocated / max(eval_allocated, 1e-6)
    reserved_overhead = (train_reserved - eval_reserved) * 1024  # MB
    reserved_ratio = train_reserved / max(eval_reserved, 1e-6)
    
    print(f"  Allocated Memory Overhead: +{allocated_overhead:.1f} MB ({allocated_ratio:.2f}x)")
    print(f"  Reserved Memory Overhead: +{reserved_overhead:.1f} MB ({reserved_ratio:.2f}x)")
    
    # Per-sample analysis
    print(f"\nPer-Sample Memory Usage:")
    for mode_name, mode_data in [('Evaluation', results['eval_mode']), ('Training', results['train_mode'])]:
        per_sample_allocated = mode_data['memory_allocated']['per_sample_avg'] * 1024  # MB
        per_sample_reserved = mode_data['memory_reserved']['per_sample_avg'] * 1024  # MB
        print(f"  {mode_name} Mode:")
        print(f"    Allocated: {per_sample_allocated:.2f} MB/sample")
        print(f"    Reserved: {per_sample_reserved:.2f} MB/sample")
    
    # Memory usage distribution
    print(f"\nMemory Usage Distribution:")
    for mode_name, mode_data in [('Evaluation', results['eval_mode']), ('Training', results['train_mode'])]:
        allocated_stats = mode_data['memory_allocated']
        cv_allocated = (allocated_stats['std'] / allocated_stats['avg']) * 100 if allocated_stats['avg'] > 0 else 0
        
        reserved_stats = mode_data['memory_reserved']
        cv_reserved = (reserved_stats['std'] / reserved_stats['avg']) * 100 if reserved_stats['avg'] > 0 else 0
        
        print(f"  {mode_name} Mode:")
        print(f"    Allocated Memory Variability: {cv_allocated:.1f}% (coefficient of variation)")
        print(f"    Reserved Memory Variability: {cv_reserved:.1f}% (coefficient of variation)")
    
    # Recommendations
    print(f"\nRecommendations:")
    train_peak = results['train_mode']['peak_memory_usage'] * 1024  # MB
    eval_peak = results['eval_mode']['peak_memory_usage'] * 1024   # MB
    max_peak = max(train_peak, eval_peak)
    
    print(f"  Minimum GPU Memory Needed: {max_peak:.0f} MB ({max_peak/1024:.1f} GB)")
    
    # Suggest optimal batch size based on memory usage
    train_per_sample = results['train_mode']['memory_allocated']['per_sample_avg'] * 1024  # MB
    if train_per_sample > 0:
        # Assuming user wants to stay under 80% of common GPU memory sizes
        common_gpu_sizes = [4096, 6144, 8192, 11264, 16384, 24576, 32768, 40960, 81920]  # MB
        print(f"  Suggested Maximum Batch Sizes (80% memory usage):")
        for gpu_mem in common_gpu_sizes:
            if gpu_mem > max_peak:
                max_batch = int((gpu_mem * 0.8) / train_per_sample)
                print(f"    {gpu_mem//1024}GB GPU: ~{max_batch} samples/batch")
                if len([g for g in common_gpu_sizes if g > max_peak]) > 3:
                    break

# Usage example:
def compare_models(model1, model2, loader, device='cuda', **kwargs):
    """Compare two models"""
    print("Analyzing Model 1...")
    results1 = analyze_model_performance(model1, loader, device=device, **kwargs)
    print_performance_summary(results1)
    
    print("\n" + "="*80)
    print("Analyzing Model 2...")
    results2 = analyze_model_performance(model2, loader, device=device, **kwargs)
    print_performance_summary(results2)
    
    # Cross-model comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"Model 1 vs Model 2 (Eval Mode):")
    print(f"  Speed Ratio: {results1['eval_mode']['avg_total_time'] / results2['eval_mode']['avg_total_time']:.2f}x")
    print(f"  Memory Ratio: {results1['eval_mode']['avg_memory_allocated'] / results2['eval_mode']['avg_memory_allocated']:.2f}x")
    print(f"  Throughput Ratio: {results1['eval_mode']['throughput_samples_per_sec'] / results2['eval_mode']['throughput_samples_per_sec']:.2f}x")
    
    return results1, results2

