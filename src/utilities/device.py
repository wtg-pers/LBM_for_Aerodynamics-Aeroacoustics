"""
Device Management Module for LBM Solver

This module handles CPU/GPU library selection and GPU device assignment.
Supports multi-GPU systems by allowing specific GPU selection.

Author: LBM Development Team
Date: 2026-02
"""

from typing import TYPE_CHECKING, Optional, Tuple
import os

if TYPE_CHECKING:
    from types import ModuleType


def setup_library(MODE: str, device_id: Optional[int] = None) -> 'ModuleType':
    """Setup computation library (NumPy or CuPy)
    
    Args:
        MODE: 'cpu' or 'gpu'
        device_id: GPU device ID to use (0, 1, 2, ...). If None, uses GPU 0.
                  Only used when MODE='gpu'.  [dimensionless]
    
    Returns:
        xp: Array module (numpy or cupy)
        
    Examples:
        >>> xp = setup_library('gpu', device_id=2)  # Use GPU 2
        >>> xp = setup_library('cpu')               # Use CPU
    """
    if MODE == 'cpu':
        import numpy as xp
        print(f"MODE: CPU")
        return xp
    
    else:  # GPU mode
        try:
            import cupy as xp
            
            # Set GPU device
            if device_id is not None:
                xp.cuda.Device(device_id).use()
                actual_device = device_id
            else:
                # Default to GPU 0
                xp.cuda.Device(0).use()
                actual_device = 0
            
            # Verify GPU is accessible
            xp.cuda.Device(actual_device).compute_capability
            
            # Get GPU name for informative output
            device = xp.cuda.Device(actual_device)
            device_name = device.attributes.get('name', 'Unknown').decode('utf-8')
            total_memory = device.mem_info[1] / 1e9  # Total memory in GB
            
            print(f"MODE: GPU")
            print(f"  Device ID: {actual_device}")
            print(f"  Device Name: {device_name}")
            print(f"  Total Memory: {total_memory:.2f} GB")
            
            return xp
            
        except ImportError:
            print(f"Warning: CuPy not installed. Falling back to NumPy (CPU mode).")
            import numpy as xp
            return xp
            
        except Exception as e:
            print(f"Warning: Cannot use GPU {device_id if device_id is not None else 0}: {e}")
            print("Falling back to NumPy (CPU mode).")
            import numpy as xp
            return xp


def get_available_gpus() -> Tuple[int, list]:
    """Get information about available GPUs
    
    Returns:
        Tuple of:
            - num_gpus: Number of available GPUs  [dimensionless]
            - gpu_info: List of (device_id, name, memory_GB) tuples
    
    Example:
        >>> num_gpus, info = get_available_gpus()
        >>> print(f"Found {num_gpus} GPUs")
        >>> for dev_id, name, mem in info:
        ...     print(f"  GPU {dev_id}: {name}, {mem:.2f} GB")
    """
    try:
        import cupy as cp
        
        num_gpus = cp.cuda.runtime.getDeviceCount()
        gpu_info = []
        
        for i in range(num_gpus):
            device = cp.cuda.Device(i)
            name = device.attributes.get('name', 'Unknown').decode('utf-8')
            total_memory = device.mem_info[1] / 1e9  # GB
            gpu_info.append((i, name, total_memory))
        
        return num_gpus, gpu_info
        
    except ImportError:
        return 0, []
    except Exception:
        return 0, []


def print_gpu_info():
    """Print available GPU information
    
    Useful for checking what GPUs are available before running simulation.
    """
    num_gpus, gpu_info = get_available_gpus()
    
    if num_gpus == 0:
        print("No GPUs available (or CuPy not installed)")
        return
    
    print(f"\nAvailable GPUs: {num_gpus}")
    print("-" * 60)
    for dev_id, name, memory in gpu_info:
        print(f"  GPU {dev_id}: {name}")
        print(f"           Memory: {memory:.2f} GB")
    print("-" * 60)


def set_gpu_memory_limit(device_id: int, limit_fraction: float = 0.9):
    """Set memory pool limit for specific GPU
    
    Prevents CuPy from allocating all GPU memory, which can be useful
    when running multiple simulations or sharing GPU with other processes.
    
    Args:
        device_id: GPU device ID  [dimensionless]
        limit_fraction: Fraction of total memory to use (0.0-1.0)  [dimensionless]
    
    Example:
        >>> set_gpu_memory_limit(device_id=1, limit_fraction=0.8)  # Use 80% of GPU 1
    """
    try:
        import cupy as cp
        
        device = cp.cuda.Device(device_id)
        total_memory = device.mem_info[1]  # Total memory in bytes
        limit_bytes = int(total_memory * limit_fraction)
        
        # Set memory pool limit
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=limit_bytes)
        
        print(f"  GPU {device_id} memory limit set to {limit_fraction*100:.0f}% "
              f"({limit_bytes/1e9:.2f} GB)")
        
    except ImportError:
        print("Warning: CuPy not installed, cannot set memory limit")
    except Exception as e:
        print(f"Warning: Could not set memory limit: {e}")