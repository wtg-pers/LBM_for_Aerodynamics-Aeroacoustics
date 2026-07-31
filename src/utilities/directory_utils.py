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


def setup_output_directories(output_dir: Optional[str],
                              checkpoint_dir: Optional[str],
                              csv_dir: Optional[str] = None,
                              clear_previous: bool = False,
                              is_restart: bool = False,
                              verbose: bool = True,
                              sweep_dirs: Optional[list] = None) -> None:
    """Create output directories for ACTIVE channels; optionally clear.

    A None output_dir / checkpoint_dir means the channel is disabled for
    this run and its directory is NOT created (unconditionally creating
    empty vtk/checkpoints dirs was pure noise). clear_previous still
    sweeps the on-disk locations of DISABLED channels via `sweep_dirs`
    so stale files cannot mix into a later run of the same folder.

    Safety: Never clears files when is_restart=True, regardless of
    clear_previous setting.

    Args:
        output_dir: VTK output directory path, or None (vtk disabled)
        checkpoint_dir: Checkpoint directory path, or None (disabled)
        csv_dir: CSV output directory path (optional)
        clear_previous: Whether to clear existing files
        is_restart: If True, never clear (safety measure for restart)
        verbose: Print status messages
        sweep_dirs: optional [vtk_dir, checkpoint_dir] on-disk locations
            to clear even when the channel is disabled (missing dirs are
            no-ops; clear_directory tolerates them)
    """
    # Create directories only for channels that will write
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if checkpoint_dir:
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

        vtk_sweep = (sweep_dirs[0] if sweep_dirs else output_dir)
        ckpt_sweep = (sweep_dirs[1] if sweep_dirs else checkpoint_dir)

        # Clear VTK files
        if vtk_sweep:
            vtk_patterns = ['*.vti', '*.vtk', '*.pvd']
            vtk_count = clear_directory(vtk_sweep, vtk_patterns, verbose=False)
            if verbose and vtk_count > 0:
                print(f"    Removed {vtk_count} VTK files from {vtk_sweep}")

        # Clear checkpoint files
        if ckpt_sweep:
            ckpt_patterns = ['*.npz']
            ckpt_count = clear_directory(ckpt_sweep, ckpt_patterns,
                                         verbose=False)
            if verbose and ckpt_count > 0:
                print(f"    Removed {ckpt_count} checkpoint files "
                      f"from {ckpt_sweep}")

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