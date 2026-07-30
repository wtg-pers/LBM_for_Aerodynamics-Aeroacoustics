"""
Checkpoint Module for LBM Solver (Restart Functionality)

This module provides save/load functionality for simulation checkpoints,
enabling restart from a saved state.

Supports both 2D and 3D simulations.

Saved Data:
    - Distribution function f (essential for restart)
    - Macroscopic fields (rho, u) for verification
    - Simulation metadata (step, time, parameters)
    - Configuration snapshot

File Format:
    Uses NumPy's compressed .npz format for efficient storage.
    Typical compression ratio: 50-80% reduction.

Author: LBM Development Team
Date: 2026-02
"""

import os
import json
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Dict, Any, Union, Tuple
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class CheckpointManager:
    """Manages simulation checkpoints for restart capability
    
    Supports both 2D and 3D simulations.
    """
    
    def __init__(self,
                 output_dir: str,
                 prefix: str = 'checkpoint',
                 keep_last_n: int = 0,
                 xp: 'ModuleType' = None,
                 create_dir: bool = True) -> None:
        """Initialize checkpoint manager
        
        Args:
            output_dir: Directory for checkpoint files
            prefix: Filename prefix
            keep_last_n: Keep only last N checkpoints (0=keep all)
            xp: Array module (numpy or cupy)
        """
        self.output_dir = output_dir
        self.prefix = prefix
        self.keep_last_n = keep_last_n
        self.xp = xp if xp is not None else np

        # create_dir=False: non-IO MPI ranks keep the manager for RESTORE
        # (reading existing files) without creating directories.
        if create_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.saved_files: list = []
    
    def _to_numpy(self, arr: 'npt.NDArray') -> np.ndarray:
        """Convert CuPy array to NumPy if necessary"""
        if hasattr(arr, 'get'):
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
            step: Current time step
            f: Distribution function (Q, Nx, Ny) or (Q, Nx, Ny, Nz)
            rho: Density field (optional)
            u: Velocity field (optional)
            tau: Relaxation time (optional)
            config: Simulation configuration dictionary
            extra_data: Additional data to save
            
        Returns:
            Path to saved checkpoint file
        """
        filename = f"{self.prefix}_{step:08d}.npz"
        filepath = os.path.join(self.output_dir, filename)
        
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
            save_dict['config_json'] = np.array(json.dumps(config))
        
        if extra_data is not None:
            for key, value in extra_data.items():
                if isinstance(value, np.ndarray) or hasattr(value, 'get'):
                    save_dict[f'extra_{key}'] = self._to_numpy(value)
                else:
                    save_dict[f'extra_{key}'] = np.array(value)
        
        np.savez_compressed(filepath, **save_dict)
        self.saved_files.append(filepath)
        
        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()
        
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
        """Load checkpoint from file"""
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
        
        for key in data.files:
            if key.startswith('extra_'):
                result[key[6:]] = data[key]
        
        print(f"  Checkpoint loaded: {os.path.basename(filepath)}")
        print(f"    Step: {result['step']}")
        if 'timestamp' in result:
            print(f"    Created: {result['timestamp']}")
        
        return result
    
    def load_latest(self) -> Dict[str, Any]:
        """Load the most recent checkpoint"""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.output_dir}")
        
        latest = sorted(checkpoints, key=lambda x: x['step'])[-1]
        return self.load(latest['filepath'])
    
    def load_by_step(self, step: int) -> Dict[str, Any]:
        """Load checkpoint for specific step"""
        filename = f"{self.prefix}_{step:08d}.npz"
        filepath = os.path.join(self.output_dir, filename)
        return self.load(filepath)
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints"""
        checkpoints = []
        
        if not os.path.exists(self.output_dir):
            return checkpoints
        
        for f in os.listdir(self.output_dir):
            if f.startswith(self.prefix) and f.endswith('.npz'):
                filepath = os.path.join(self.output_dir, f)
                
                try:
                    step_str = f.replace(self.prefix + '_', '').replace('.npz', '')
                    step = int(step_str)
                except ValueError:
                    continue
                
                file_size = os.path.getsize(filepath) / 1e6
                
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
        
        Supports both 2D (Q, Nx, Ny) and 3D (Q, Nx, Ny, Nz) shapes.
        
        Args:
            f_shape: Shape of distribution function
            include_macros: Include rho and u in estimate
            
        Returns:
            Size estimates in MB
        """
        # Handle both 2D and 3D
        if len(f_shape) == 3:
            # 2D: (Q, Nx, Ny)
            Q, Nx, Ny = f_shape
            n_points = Nx * Ny
            dim = 2
        else:
            # 3D: (Q, Nx, Ny, Nz)
            Q, Nx, Ny, Nz = f_shape
            n_points = Nx * Ny * Nz
            dim = 3
        
        # f: float64
        f_size = Q * n_points * 8
        
        # macros: rho (1) + u (dim) = (dim+1) fields
        macro_size = (dim + 1) * n_points * 8 if include_macros else 0
        
        total_raw = f_size + macro_size
        compressed_size = total_raw * 0.4  # ~60% reduction typical
        
        return {
            'raw_MB': total_raw / 1e6,
            'estimated_MB': compressed_size / 1e6,
            'f_shape': f_shape
        }


def create_restart_info(checkpoint_path: str, 
                        original_max_steps: int,
                        new_max_steps: int) -> Dict:
    """Create restart information summary"""
    return {
        'restart_from': checkpoint_path,
        'original_max_steps': original_max_steps,
        'new_max_steps': new_max_steps,
        'additional_steps': new_max_steps - original_max_steps
    }