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
import zipfile
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Dict, Any, Union, Tuple, Callable, Iterable, Iterator
import numpy as np

#: host chunk size for streaming device -> file copies [elements per chunk]
#: (~64 MB of float32); the writer never holds more than one chunk.
STREAM_CHUNK_ELEMS = 16 * 1024 * 1024


class LazyArray:
    """An array that is PRODUCED while the checkpoint is written.

    `chunks()` must yield C-contiguous host ndarrays whose bytes, in
    order, are the C-order buffer of the array of `shape`/`dtype` (any
    piece shape is fine -- whole axis-0 rows, or (x-slab, Ny, Nz) pieces of
    one q of a (Q, Nx, Ny, Nz) f, which is how the levels stream). Producing
    on demand is what bounds the writer's host memory to ONE chunk: the old
    path materialised every level's f on the host before np.savez ran
    (float32 27 x 265 M nodes = 29 GB + rho/u + zlib buffers for the
    6-level ROBIN grid -> host OOM at the first checkpoint, robin/16 s12).
    """

    def __init__(self, shape, dtype, chunks: Callable[[], Iterable[np.ndarray]]):
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self._chunks = chunks

    def chunks(self) -> Iterator[np.ndarray]:
        return iter(self._chunks())

    def materialize(self) -> np.ndarray:
        out = np.empty(self.shape, self.dtype)
        flat = out.reshape(-1)
        i = 0
        for c in self.chunks():
            c = np.ascontiguousarray(np.asarray(c, dtype=self.dtype)).reshape(-1)
            flat[i:i + c.size] = c
            i += c.size
        if i != flat.size:
            raise ValueError(f"LazyArray produced {i} elements, declared {flat.size}")
        return out


def _device_chunks(arr, n_elems: int = STREAM_CHUNK_ELEMS) -> Iterator[np.ndarray]:
    """Slices of a device (CuPy) array along axis 0, one host copy at a time."""
    rows = max(1, n_elems // max(1, int(np.prod(arr.shape[1:]))))
    for i in range(0, arr.shape[0], rows):
        yield np.ascontiguousarray(arr[i:i + rows].get())


def write_npz_streaming(filepath: str, entries: Dict[str, Any],
                        compress: bool = True) -> None:
    """np.savez(_compressed)-compatible writer that STREAMS.

    Same container (zip of '<key>.npy' members, .npy header v1.0) so
    np.load / CheckpointManager.load read the result exactly as before.
    Values may be: numpy arrays / scalars (written whole), CuPy arrays
    (copied to the host in axis-0 slices), or LazyArray (produced chunk
    by chunk while the member is being written).
    """
    from numpy.lib import format as npf
    comp = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(filepath, mode='w', compression=comp,
                         allowZip64=True) as zf:
        for key, val in entries.items():
            name = key + '.npy'
            if isinstance(val, LazyArray):
                shape, dtype, it = val.shape, val.dtype, val.chunks()
            elif hasattr(val, 'get') and hasattr(val, 'shape') \
                    and not isinstance(val, np.ndarray):        # CuPy
                shape, dtype = tuple(val.shape), np.dtype(val.dtype)
                it = _device_chunks(val) if val.ndim >= 1 else iter([val.get()])
            else:
                arr = val if isinstance(val, np.ndarray) else np.asarray(val)
                with zf.open(name, 'w', force_zip64=True) as fp:
                    npf.write_array(fp, arr, allow_pickle=False)
                continue
            with zf.open(name, 'w', force_zip64=True) as fp:
                npf.write_array_header_1_0(
                    fp, {'descr': npf.dtype_to_descr(dtype),
                         'fortran_order': False, 'shape': tuple(shape)})
                need = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                nbytes = 0
                for c in it:
                    c = np.ascontiguousarray(np.asarray(c, dtype=dtype))
                    mv = memoryview(c).cast('B')
                    nbytes += len(mv)
                    fp.write(mv)
                if nbytes != need:
                    raise ValueError(f"{key}: streamed {nbytes} bytes, declared {need}")


class _LazyState(Mapping):
    """CheckpointManager.load result: keys resolved on ACCESS, never cached.

    A restart that holds every level's f on the host at once (the old
    eager dict) needs the same ~30 GB the writer needed; resolving
    'f_level_k' when the initializer asks for it keeps the host at one
    level (initializer._restart_mlg hands each array straight to the
    device / the slab slicer and drops it)."""

    def __init__(self, npz, filename: str):
        self._npz = npz
        self._files = list(npz.files)
        self.filename = filename

    @staticmethod
    def _public(member: str) -> str:
        return member[6:] if member.startswith('extra_') else member

    def _member(self, key: str) -> Optional[str]:
        if key == 'config':
            return 'config_json' if 'config_json' in self._files else None
        for cand in (key, 'extra_' + key):
            if cand in self._files:
                return cand
        return None

    def __getitem__(self, key: str):
        m = self._member(key)
        if m is None:
            raise KeyError(key)
        v = self._npz[m]
        if key == 'step':
            return int(v)
        if key == 'tau':
            return float(v)
        if key == 'timestamp':
            return str(v)
        if key == 'config':
            return json.loads(str(v))
        return v

    def __contains__(self, key) -> bool:
        return self._member(key) is not None

    def __iter__(self):
        seen = set()
        for m in self._files:
            k = 'config' if m == 'config_json' else self._public(m)
            if k not in seen:
                seen.add(k)
                yield k

    def __len__(self) -> int:
        return len(set(self.__iter__()))

    def close(self) -> None:
        try:
            self._npz.close()
        except Exception:
            pass


def _say(msg: str) -> None:
    """Print without corrupting an active tqdm progress bar.

    A plain print() while the run-loop bar is live gets appended to the
    bar's line (the 'Checkpoint saved' notice stretched it sideways).
    tqdm.write() prints ABOVE active bars and degrades to a normal print
    when no bar exists / tqdm is unavailable."""
    try:
        from tqdm import tqdm
        tqdm.write(msg)
    except Exception:
        print(msg)

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
        # (reading existing files) without creating directories. Creation
        # is LAZY (first save_checkpoint): building the manager alone must
        # not leave an empty checkpoints/ dir on runs that never save.
        self._create_dir = bool(create_dir)
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
        os.makedirs(self.output_dir, exist_ok=True)  # lazy (see __init__)
        filename = f"{self.prefix}_{step:08d}.npz"
        filepath = os.path.join(self.output_dir, filename)

        # STREAMED write (robin/16 s12): arrays / LazyArray producers are
        # passed through untouched and copied to the host one chunk at a
        # time while their .npy member is written -- the host never holds
        # a whole level's f, let alone every level's. Key order is the
        # order the members are produced: 'f' first, then the fine-level
        # extras in block order (the MPI streams are consumed in exactly
        # this order on every rank, mpi_output._checkpoint_payload).
        save_dict = {
            'f': f,
            'step': np.array(step),
            'timestamp': np.array(datetime.now().isoformat()),
        }
        
        if rho is not None:
            save_dict['rho'] = rho
        
        if u is not None:
            save_dict['u'] = u
        
        if tau is not None:
            save_dict['tau'] = np.array(tau)
        
        if config is not None:
            save_dict['config_json'] = np.array(json.dumps(config))
        
        if extra_data is not None:
            for key, value in extra_data.items():
                if isinstance(value, (np.ndarray, LazyArray)) or hasattr(value, 'get'):
                    save_dict[f'extra_{key}'] = value
                else:
                    save_dict[f'extra_{key}'] = np.array(value)
        
        write_npz_streaming(filepath, save_dict)
        # re-saving the same step must not leave a duplicate entry —
        # keep_last_n pruning would otherwise count the path twice and
        # delete the file it just wrote (seen: emergency + final save at
        # the same stop step with keep_last_n=1)
        if filepath in self.saved_files:
            self.saved_files.remove(filepath)
        self.saved_files.append(filepath)

        if self.keep_last_n > 0:
            self._cleanup_old_checkpoints()
        
        file_size_mb = os.path.getsize(filepath) / 1e6
        _say(f"  Checkpoint saved: {filename} ({file_size_mb:.2f} MB)")
        
        return filepath
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N"""
        if len(self.saved_files) > self.keep_last_n:
            files_to_remove = self.saved_files[:-self.keep_last_n]
            for f in files_to_remove:
                if os.path.exists(f):
                    os.remove(f)
                    _say(f"  Removed old checkpoint: {os.path.basename(f)}")
            self.saved_files = self.saved_files[-self.keep_last_n:]
    
    def load(self, filepath: str) -> 'Mapping[str, Any]':
        """Load checkpoint from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        data = np.load(filepath, allow_pickle=True)
        # LAZY mapping (robin/16 s12): same keys as the old eager dict
        # ('f', 'step', 'timestamp', 'rho', 'u', 'tau', 'config', extras
        # without their 'extra_' prefix) but each array is read from the
        # npz when accessed and not retained here.
        result = _LazyState(data, filepath)
        if 'f' not in result or 'step' not in result:
            raise KeyError(f"{filepath}: not a checkpoint (no 'f'/'step')")
        
        print(f"  Checkpoint loaded: {os.path.basename(filepath)}")
        print(f"    Step: {result['step']}")
        if 'timestamp' in result:
            print(f"    Created: {result['timestamp']}")
        
        return result
    
    def load_latest(self) -> 'Mapping[str, Any]':
        """Load the most recent checkpoint"""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {self.output_dir}")
        
        latest = sorted(checkpoints, key=lambda x: x['step'])[-1]
        return self.load(latest['filepath'])
    
    def load_by_step(self, step: int) -> 'Mapping[str, Any]':
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