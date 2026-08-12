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

    if MODE != 'gpu':
        raise ValueError(
            f"device_mode must be 'cpu' or 'gpu', got {MODE!r}")

    # GPU mode. A GPU request must never silently degrade to NumPy: the
    # run would drop the fused kernels, esoteric path and SGS in one go
    # and complete with a one-line note buried in the setup log.
    try:
        import cupy as xp
    except ImportError as e:
        raise RuntimeError(
            "device_mode='gpu' but CuPy is not importable. Install CuPy "
            "or set simulation device_mode: 'cpu' explicitly.") from e

    actual_device = device_id if device_id is not None else 0
    try:
        xp.cuda.Device(actual_device).use()

        # Verify GPU is accessible
        xp.cuda.Device(actual_device).compute_capability

        # Get GPU name for informative output (version-safe method)
        device_props = xp.cuda.runtime.getDeviceProperties(actual_device)
        device_name_raw = device_props['name']

        # Handle both bytes (old CuPy) and str (new CuPy)
        if isinstance(device_name_raw, bytes):
            device_name = device_name_raw.decode('utf-8')
        else:
            device_name = device_name_raw

        device = xp.cuda.Device(actual_device)
        total_memory = device.mem_info[1] / 1e9  # Total memory in GB
    except Exception as e:
        raise RuntimeError(
            f"device_mode='gpu' but GPU {actual_device} is unusable: {e}. "
            "Fix the device/driver (or CUDA_VISIBLE_DEVICES) or set "
            "device_mode: 'cpu' explicitly.") from e

    print(f"MODE: GPU")
    print(f"  Device ID: {actual_device}")
    print(f"  Device Name: {device_name}")
    print(f"  Total Memory: {total_memory:.2f} GB")

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
            # Use getDeviceProperties for better compatibility
            device_props = cp.cuda.runtime.getDeviceProperties(i)
            device_name_raw = device_props['name']
            
            # Handle both bytes (old CuPy) and str (new CuPy)
            if isinstance(device_name_raw, bytes):
                name = device_name_raw.decode('utf-8')
            else:
                name = device_name_raw
            
            device = cp.cuda.Device(i)
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