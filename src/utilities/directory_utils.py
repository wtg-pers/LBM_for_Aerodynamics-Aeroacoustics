"""
Directory Management Utilities for LBM Solver

This module provides utility functions for managing output directories,
including creation, clearing, and validation.

Author: LBM Development Team
Date: 2026-01
"""

import os
import glob
from typing import List, Optional


def clear_directory(directory: str, patterns: List[str], 
                    verbose: bool = True) -> int:
    """Clear files matching patterns from a directory
    
    Removes files that match any of the specified glob patterns.
    Does not remove subdirectories.
    
    Args:
        directory: Target directory path
        patterns: List of glob patterns (e.g., ['*.vti', '*.pvd', '*.csv'])
        verbose: Print warnings for files that couldn't be removed
        
    Returns:
        Number of files successfully removed
        
    Examples:
        >>> clear_directory('./results', ['*.vti', '*.pvd'])
        Removed 25 files
        25
        
        >>> clear_directory('./checkpoints', ['*.npz'], verbose=False)
        3
    """
    if not os.path.exists(directory):
        return 0
    
    removed_count = 0
    
    for pattern in patterns:
        # Find files matching pattern
        files = glob.glob(os.path.join(directory, pattern))
        
        for filepath in files:
            # Skip directories
            if os.path.isdir(filepath):
                continue
                
            try:
                os.remove(filepath)
                removed_count += 1
            except OSError as e:
                if verbose:
                    print(f"    Warning: Could not remove {filepath}: {e}")
    
    return removed_count


def setup_output_directories(output_dir: str, 
                              checkpoint_dir: str, 
                              csv_dir: Optional[str] = None,
                              clear_previous: bool = False, 
                              is_restart: bool = False,
                              verbose: bool = True) -> None:
    """Create output directories and optionally clear previous results
    
    Creates all necessary output directories if they don't exist.
    Optionally clears previous results (VTK, checkpoints, CSV files).
    
    Safety: Never clears files when is_restart=True, regardless of
    clear_previous setting.
    
    Args:
        output_dir: VTK output directory path
        checkpoint_dir: Checkpoint directory path
        csv_dir: CSV output directory path (optional)
        clear_previous: Whether to clear existing files
        is_restart: If True, never clear (safety measure for restart)
        verbose: Print status messages
        
    Examples:
        >>> setup_output_directories(
        ...     output_dir='./results/vtk',
        ...     checkpoint_dir='./checkpoints',
        ...     csv_dir='./results/csv',
        ...     clear_previous=True,
        ...     is_restart=False
        ... )
        Clearing previous results...
          Removed 15 VTK files from ./results/vtk
          Removed 3 checkpoint files from ./checkpoints
          Removed 1 CSV files from ./results/csv
    """
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    
    # Safety: Never clear on restart
    if is_restart:
        if verbose and clear_previous:
            print("  Note: clear_previous ignored during restart (safety measure)")
        return
    
    # Clear previous results if requested
    if clear_previous:
        if verbose:
            print(f"  Clearing previous results...")
        
        # Clear VTK files
        vtk_patterns = ['*.vti', '*.vtk', '*.pvd']
        vtk_count = clear_directory(output_dir, vtk_patterns, verbose=False)
        if verbose and vtk_count > 0:
            print(f"    Removed {vtk_count} VTK files from {output_dir}")
        
        # Clear checkpoint files
        ckpt_patterns = ['*.npz']
        ckpt_count = clear_directory(checkpoint_dir, ckpt_patterns, verbose=False)
        if verbose and ckpt_count > 0:
            print(f"    Removed {ckpt_count} checkpoint files from {checkpoint_dir}")
        
        # Clear CSV files
        if csv_dir:
            csv_patterns = ['*.csv']
            csv_count = clear_directory(csv_dir, csv_patterns, verbose=False)
            if verbose and csv_count > 0:
                print(f"    Removed {csv_count} CSV files from {csv_dir}")


def ensure_directory(path: str) -> str:
    """Ensure a directory exists, creating it if necessary
    
    Args:
        path: Directory path to ensure exists
        
    Returns:
        The same path (for chaining)
        
    Examples:
        >>> filepath = os.path.join(ensure_directory('./results/csv'), 'data.csv')
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_directory_size(directory: str, patterns: Optional[List[str]] = None) -> dict:
    """Get size information for a directory
    
    Args:
        directory: Directory path to analyze
        patterns: Optional list of glob patterns to filter files
                 If None, includes all files
        
    Returns:
        Dictionary with:
            - total_bytes: Total size in bytes
            - total_mb: Total size in megabytes
            - file_count: Number of files
            - files: List of (filename, size_bytes) tuples
    """
    if not os.path.exists(directory):
        return {
            'total_bytes': 0,
            'total_mb': 0.0,
            'file_count': 0,
            'files': []
        }
    
    files_info = []
    total_bytes = 0
    
    if patterns:
        # Only files matching patterns
        for pattern in patterns:
            for filepath in glob.glob(os.path.join(directory, pattern)):
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    files_info.append((os.path.basename(filepath), size))
                    total_bytes += size
    else:
        # All files in directory
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                files_info.append((filename, size))
                total_bytes += size
    
    return {
        'total_bytes': total_bytes,
        'total_mb': total_bytes / (1024 * 1024),
        'file_count': len(files_info),
        'files': sorted(files_info, key=lambda x: x[1], reverse=True)
    }


def print_directory_summary(directories: dict, verbose: bool = True) -> None:
    """Print summary of output directories
    
    Args:
        directories: Dictionary mapping names to paths
                    e.g., {'VTK': './results/vtk', 'Checkpoints': './checkpoints'}
        verbose: If True, print detailed info; if False, print condensed
    """
    print("Output Directories:")
    
    for name, path in directories.items():
        exists = os.path.exists(path)
        if exists:
            info = get_directory_size(path)
            if verbose:
                print(f"  {name}: {path}")
                print(f"    Files: {info['file_count']}, Size: {info['total_mb']:.2f} MB")
            else:
                print(f"  {name}: {path} ({info['file_count']} files, {info['total_mb']:.2f} MB)")
        else:
            print(f"  {name}: {path} (will be created)")