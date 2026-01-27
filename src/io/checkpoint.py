"""
Checkpoint Module for LBM Solver (Restart Functionality)

This module provides save/load functionality for simulation checkpoints,
enabling restart from a saved state.

Saved Data:
    - Distribution function f (essential for restart)
    - Macroscopic fields (rho, u) for verification
    - Simulation metadata (step, time, parameters)
    - Configuration snapshot

File Format:
    Uses NumPy's compressed .npz format for efficient storage.
    Typical compression ratio: 50-80% reduction.

Usage:
    # Save checkpoint
    checkpoint.save(step=5000, f=f, rho=rho, u=u, config=sim_params)
    
    # Load and restart
    state = checkpoint.load('checkpoint_005000.npz')
    f = state['f']
    start_step = state['step']

Author: LBM Development Team
Date: 2026-01
"""

import os
import json
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Dict, Any, Union
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class CheckpointManager:
    """Manages simulation checkpoints for restart capability
    
    Saves the complete simulation state to compressed NumPy files,
    allowing restart from any saved checkpoint.
    
    Attributes:
        output_dir: Directory for checkpoint files
        prefix: Filename prefix
        keep_last_n: Number of recent checkpoints to keep (0=keep all)
        
    Example:
        >>> ckpt = CheckpointManager('./checkpoints', keep_last_n=3)
        >>> ckpt.save(step=1000, f=f_old, rho=rho, u=u)
        >>> state = ckpt.load_latest()
        >>> f_restart = xp.asarray(state['f'])
    """
    
    def __init__(self, 
                 output_dir: str,
                 prefix: str = 'checkpoint',
                 keep_last_n: int = 0,
                 xp: 'ModuleType' = None) -> None:
        """Initialize checkpoint manager
        
        Args:
            output_dir: Directory for checkpoint files
            prefix: Filename prefix
            keep_last_n: Keep only last N checkpoints (0=keep all)
            xp: Array module (numpy or cupy), for loading
        """
        self.output_dir = output_dir
        self.prefix = prefix
        self.keep_last_n = keep_last_n
        self.xp = xp if xp is not None else np
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Track saved checkpoints
        self.saved_files: list = []
    
    def _to_numpy(self, arr: 'npt.NDArray') -> np.ndarray:
        """Convert CuPy array to NumPy if necessary"""
        if hasattr(arr, 'get'):  # CuPy array
            return arr.get()
        return np.asarray(arr)
    
    def save(self,
             step: int,
             f: 'npt.NDArray',
             rho: Optional['npt.NDArray'] = None,
             u: Optional['npt.NDArray'] = None,
             tau: Optional[float] = None,
             config: Optional[Dict] = None,
             extra_data: Optional[Dict] = None) -> str:
        """Save simulation checkpoint
        
        Args:
            step: Current time step  [dimensionless]
            f: Distribution function (Q, Nx, Ny, Nz)  [dimensionless]
            rho: Density field (Nx, Ny, Nz)  [dimensionless, optional]
            u: Velocity field (3, Nx, Ny, Nz)  [lattice units, optional]
            tau: Relaxation time  [dimensionless, optional]
            config: Simulation configuration dictionary
            extra_data: Additional data to save
            
        Returns:
            Path to saved checkpoint file
        """
        filename = f"{self.prefix}_{step:08d}.npz"
        filepath = os.path.join(self.output_dir, filename)
        
        # Prepare data dictionary
        save_dict = {
            'f': self._to_numpy(f),
            'step': np.array(step),
            'timestamp': np.array(datetime.now().isoformat()),
        }
        
        if rho is not None:
            save_dict['rho'] = self._to_numpy(rho)
        
        if u is not None:
            save_dict['u'] = self._to_numpy(u)
        
        if tau is not None:
            save_dict['tau'] = np.array(tau)
        
        if config is not None:
            # Store config as JSON string
            save_dict['config_json'] = np.array(json.dumps(config))
        
        if extra_data is not None:
            for key, value in extra_data.items():
                if isinstance(value, np.ndarray) or hasattr(value, 'get'):
                    save_dict[f'extra_{key}'] = self._to_numpy(value)
                else:
                    save_dict[f'extra_{key}'] = np.array(value)
        
        # Save compressed
        np.savez_compressed(filepath, **save_dict)
        
        self.saved_files.append(filepath)
        
        # Cleanup old checkpoints
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()
        
        # Get file size
        file_size_mb = os.path.getsize(filepath) / 1e6
        
        print(f"  Checkpoint saved: {filename} ({file_size_mb:.2f} MB)")
        
        return filepath
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N"""
        if len(self.saved_files) > self.keep_last_n:
            files_to_remove = self.saved_files[:-self.keep_last_n]
            for f in files_to_remove:
                if os.path.exists(f):
                    os.remove(f)
                    print(f"  Removed old checkpoint: {os.path.basename(f)}")
            self.saved_files = self.saved_files[-self.keep_last_n:]
    
    def load(self, filepath: str) -> Dict[str, Any]:
        """Load checkpoint from file
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Dictionary containing:
                - 'f': Distribution function (numpy array)
                - 'step': Time step
                - 'rho': Density (if saved)
                - 'u': Velocity (if saved)
                - 'tau': Relaxation time (if saved)
                - 'config': Configuration dict (if saved)
                - 'timestamp': When checkpoint was created
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        
        result = {
            'f': data['f'],
            'step': int(data['step']),
        }
        
        if 'timestamp' in data:
            result['timestamp'] = str(data['timestamp'])
        
        if 'rho' in data:
            result['rho'] = data['rho']
        
        if 'u' in data:
            result['u'] = data['u']
        
        if 'tau' in data:
            result['tau'] = float(data['tau'])
        
        if 'config_json' in data:
            result['config'] = json.loads(str(data['config_json']))
        
        # Load extra data
        for key in data.files:
            if key.startswith('extra_'):
                result[key[6:]] = data[key]  # Remove 'extra_' prefix
        
        print(f"  Checkpoint loaded: {os.path.basename(filepath)}")
        print(f"    Step: {result['step']}")
        if 'timestamp' in result:
            print(f"    Created: {result['timestamp']}")
        
        return result
    
    def load_latest(self) -> Dict[str, Any]:
        """Load the most recent checkpoint
        
        Returns:
            Checkpoint data dictionary
            
        Raises:
            FileNotFoundError: If no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.output_dir}")
        
        # Sort by step number and get latest
        latest = sorted(checkpoints, key=lambda x: x['step'])[-1]
        
        return self.load(latest['filepath'])
    
    def load_by_step(self, step: int) -> Dict[str, Any]:
        """Load checkpoint for specific step
        
        Args:
            step: Time step to load
            
        Returns:
            Checkpoint data dictionary
        """
        filename = f"{self.prefix}_{step:08d}.npz"
        filepath = os.path.join(self.output_dir, filename)
        
        return self.load(filepath)
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints
        
        Returns:
            List of dictionaries with checkpoint info
        """
        checkpoints = []
        
        for f in os.listdir(self.output_dir):
            if f.startswith(self.prefix) and f.endswith('.npz'):
                filepath = os.path.join(self.output_dir, f)
                
                # Extract step from filename
                try:
                    step_str = f.replace(self.prefix + '_', '').replace('.npz', '')
                    step = int(step_str)
                except ValueError:
                    continue
                
                file_size = os.path.getsize(filepath) / 1e6  # MB
                
                checkpoints.append({
                    'filename': f,
                    'filepath': filepath,
                    'step': step,
                    'size_MB': file_size
                })
        
        return sorted(checkpoints, key=lambda x: x['step'])
    
    def print_available(self) -> None:
        """Print list of available checkpoints"""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            print(f"No checkpoints found in {self.output_dir}")
            return
        
        print(f"\nAvailable checkpoints in {self.output_dir}:")
        print("-" * 50)
        for ckpt in checkpoints:
            print(f"  Step {ckpt['step']:8d}: {ckpt['filename']} ({ckpt['size_MB']:.2f} MB)")
        print("-" * 50)
    
    def get_size_estimate(self, f_shape: tuple, include_macros: bool = True) -> Dict[str, float]:
        """Estimate checkpoint file size
        
        Args:
            f_shape: Shape of distribution function (Q, Nx, Ny, Nz)
            include_macros: Include rho and u in estimate
            
        Returns:
            Size estimates in MB
        """
        Q, Nx, Ny, Nz = f_shape
        n_points = Nx * Ny * Nz
        
        # f: float64
        f_size = Q * n_points * 8
        
        # macros: rho (1) + u (3) = 4 fields
        macro_size = 4 * n_points * 8 if include_macros else 0
        
        total_raw = f_size + macro_size
        
        # Compression ratio (typical for LBM data)
        compressed_size = total_raw * 0.4  # ~60% reduction typical
        
        return {
            'raw_MB': total_raw / 1e6,
            'estimated_MB': compressed_size / 1e6,
            'f_shape': f_shape
        }


def create_restart_info(checkpoint_path: str, 
                        original_max_steps: int,
                        new_max_steps: int) -> Dict:
    """Create restart information summary
    
    Args:
        checkpoint_path: Path to loaded checkpoint
        original_max_steps: Original simulation max steps
        new_max_steps: New total max steps
        
    Returns:
        Info dictionary for logging
    """
    return {
        'restart_from': checkpoint_path,
        'original_max_steps': original_max_steps,
        'new_max_steps': new_max_steps,
        'additional_steps': new_max_steps - original_max_steps
    }
