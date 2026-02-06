"""
VTK Output Module for LBM Solver

This module provides VTK file output for ParaView visualization.
Uses VTK XML ImageData format (.vti) with appended binary data.

Supports both 2D and 3D simulations. For 2D, data is written
as a single z-slice (Nz=1).

Supported Output Variables:
    - density (ρ): Scalar field  [dimensionless, ρ/ρ_0]
    - pressure (p): Scalar field  [lattice units, p = ρ·c_s²]
    - velocity (u): Vector field  [lattice units, Δx/Δt]
    - velocity_magnitude: Scalar field  [lattice units]
    - solid_mask: Integer field (0=fluid, 1=solid)

File Format:
    VTK XML ImageData (.vti) with appended binary encoding.
    This format is compatible with all ParaView versions.

Author: LBM Development Team
Date: 2026-02
"""

import os
import struct
import re
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class VTKWriter:
    """VTK ImageData (.vti) writer with appended binary format
    
    Writes 2D/3D scalar and vector fields to VTK XML format for ParaView.
    Uses appended binary encoding for compatibility and efficiency.
    
    For 2D data, arrays are expanded to 3D with Nz=1.
    
    Attributes:
        output_dir: Directory for output files
        precision: Data precision ('float32' or 'float64')
        dim: Spatial dimension (2 or 3)
        
    Example:
        >>> writer = VTKWriter('./results', domain_shape=(100, 40))  # 2D
        >>> writer = VTKWriter('./results', domain_shape=(100, 40, 40))  # 3D
        >>> writer.write(step=1000, rho=rho, u=u, solid_mask=mask)
    """
    
    def __init__(self, 
                 output_dir: str,
                 domain_shape: tuple,
                 precision: str = 'float32',
                 compression_level: int = 0,
                 origin: tuple = None,
                 spacing: tuple = None) -> None:
        """Initialize VTK writer
        
        Args:
            output_dir: Output directory path
            domain_shape: (Nx, Ny) for 2D or (Nx, Ny, Nz) for 3D  [lattice units]
            precision: 'float32' or 'float64'
            compression_level: Compression level (currently unused)
            origin: Grid origin  [physical or lattice units]
            spacing: Grid spacing  [physical or lattice units]
        """
        self.output_dir = output_dir
        
        # Handle 2D vs 3D
        if len(domain_shape) == 2:
            self.Nx, self.Ny = domain_shape
            self.Nz = 1
            self.dim = 2
        else:
            self.Nx, self.Ny, self.Nz = domain_shape
            self.dim = 3
        
        self.precision = precision
        
        # Set origin and spacing with defaults
        if origin is None:
            origin = (0.0, 0.0, 0.0)
        elif len(origin) == 2:
            origin = (origin[0], origin[1], 0.0)
        self.origin = origin
        
        if spacing is None:
            spacing = (1.0, 1.0, 1.0)
        elif len(spacing) == 2:
            spacing = (spacing[0], spacing[1], 1.0)
        self.spacing = spacing
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine numpy dtype and VTK type string
        if precision == 'float32':
            self.dtype = np.float32
            self.vtk_type = 'Float32'
        else:
            self.dtype = np.float64
            self.vtk_type = 'Float64'
        
        # For time series (PVD file)
        self.time_steps: List[tuple] = []
        
        # Scan existing VTK files to support restart
        self._scan_existing_files()
    
    def _scan_existing_files(self, prefix: str = 'lbm') -> None:
        """Scan output directory for existing VTK files"""
        if not os.path.exists(self.output_dir):
            return
        
        pattern = re.compile(rf'^{prefix}_(\d+)\.vti$')
        
        for filename in os.listdir(self.output_dir):
            match = pattern.match(filename)
            if match:
                step = int(match.group(1))
                self.time_steps.append((float(step), filename))
        
        self.time_steps.sort(key=lambda x: x[0])
        
        if self.time_steps:
            print(f"    Found {len(self.time_steps)} existing VTK files")
            print(f"    Step range: {int(self.time_steps[0][0])} -> {int(self.time_steps[-1][0])}")
    
    def _to_numpy(self, arr: 'npt.NDArray') -> np.ndarray:
        """Convert CuPy array to NumPy if necessary"""
        if hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)
    
    def _expand_2d_to_3d(self, arr: np.ndarray, is_vector: bool = False) -> np.ndarray:
        """Expand 2D array to 3D for VTK output
        
        Args:
            arr: 2D array (Nx, Ny) or 2D vector (2, Nx, Ny)
            is_vector: If True, array is a vector field
            
        Returns:
            3D array (Nx, Ny, 1) or 3D vector (3, Nx, Ny, 1)
        """
        if is_vector:
            # (2, Nx, Ny) -> (3, Nx, Ny, 1)
            arr_3d = np.zeros((3, arr.shape[1], arr.shape[2], 1), dtype=arr.dtype)
            arr_3d[0, :, :, 0] = arr[0]  # u_x
            arr_3d[1, :, :, 0] = arr[1]  # u_y
            # arr_3d[2] = 0 (u_z)
            return arr_3d
        else:
            # (Nx, Ny) -> (Nx, Ny, 1)
            return arr[:, :, np.newaxis]
    
    def write(self, 
              step: int,
              rho: Optional['npt.NDArray'] = None,
              u: Optional['npt.NDArray'] = None,
              solid_mask: Optional['npt.NDArray'] = None,
              time: Optional[float] = None,
              prefix: str = 'lbm') -> str:
        """Write VTK file for current time step
        
        Args:
            step: Time step number
            rho: Density field (Nx, Ny[, Nz])  [dimensionless]
            u: Velocity field (dim, Nx, Ny[, Nz])  [lattice units]
            solid_mask: Solid mask (Nx, Ny[, Nz]), True=solid
            time: Physical time (optional, for PVD)
            prefix: Filename prefix
            
        Returns:
            Path to written file
        """
        filename = f"{prefix}_{step:08d}.vti"
        filepath = os.path.join(self.output_dir, filename)
        
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        ox, oy, oz = self.origin
        dx, dy, dz = self.spacing
        
        # Collect data arrays: (name, vtk_type, num_components, data_bytes)
        data_arrays = []
        
        if rho is not None:
            rho_np = self._to_numpy(rho).astype(self.dtype)
            
            # Expand 2D to 3D if needed
            if self.dim == 2:
                rho_np = self._expand_2d_to_3d(rho_np, is_vector=False)
            
            # VTK expects x to vary fastest
            # Transpose (Nx, Ny, Nz) -> (Nz, Ny, Nx) then flatten
            rho_vtk = np.ascontiguousarray(rho_np.transpose(2, 1, 0)).ravel()
            data_arrays.append(('density', self.vtk_type, 1, rho_vtk.tobytes()))
            
            # Pressure: p = ρ * c_s² = ρ / 3
            pressure = (rho_vtk / 3.0).astype(self.dtype)
            data_arrays.append(('pressure', self.vtk_type, 1, pressure.tobytes()))
        
        if u is not None:
            u_np = self._to_numpy(u).astype(self.dtype)
            
            # Expand 2D to 3D if needed
            if self.dim == 2:
                u_np = self._expand_2d_to_3d(u_np, is_vector=True)
            
            # Transpose spatial dimensions: (3, Nx, Ny, Nz) -> (3, Nz, Ny, Nx)
            u_transposed = u_np.transpose(0, 3, 2, 1)
            
            # Interleave components: (Nz, Ny, Nx, 3)
            u_interleaved = np.ascontiguousarray(u_transposed.transpose(1, 2, 3, 0))
            data_arrays.append(('velocity', self.vtk_type, 3, u_interleaved.ravel().tobytes()))
            
            # Velocity magnitude
            u_mag = np.sqrt(np.sum(u_np**2, axis=0))
            u_mag_vtk = np.ascontiguousarray(u_mag.transpose(2, 1, 0)).ravel()
            data_arrays.append(('velocity_magnitude', self.vtk_type, 1, u_mag_vtk.astype(self.dtype).tobytes()))
        
        if solid_mask is not None:
            mask_np = self._to_numpy(solid_mask).astype(np.int8)
            
            # Expand 2D to 3D if needed
            if self.dim == 2:
                mask_np = self._expand_2d_to_3d(mask_np, is_vector=False)
            
            mask_vtk = np.ascontiguousarray(mask_np.transpose(2, 1, 0)).ravel()
            data_arrays.append(('solid_mask', 'Int8', 1, mask_vtk.tobytes()))
        
        # Build XML with correct extent
        xml_lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
            f'  <ImageData WholeExtent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}" '
            f'Origin="{ox} {oy} {oz}" Spacing="{dx} {dy} {dz}">',
            f'    <Piece Extent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}">',
            '      <PointData>',
        ]
        
        # Add DataArray entries with offset references
        offset = 0
        for name, vtype, ncomp, data_bytes in data_arrays:
            xml_lines.append(
                f'        <DataArray type="{vtype}" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="appended" offset="{offset}"/>'
            )
            offset += 8 + len(data_bytes)  # 8 bytes for size header
        
        xml_lines.extend([
            '      </PointData>',
            '      <CellData>',
            '      </CellData>',
            '    </Piece>',
            '  </ImageData>',
            '  <AppendedData encoding="raw">',
            '   _',
        ])
        
        # Write file
        with open(filepath, 'wb') as f:
            header_text = '\n'.join(xml_lines)
            f.write(header_text.encode('ascii'))
            
            for name, vtype, ncomp, data_bytes in data_arrays:
                size_header = struct.pack('<Q', len(data_bytes))
                f.write(size_header)
                f.write(data_bytes)
            
            footer = '\n  </AppendedData>\n</VTKFile>\n'
            f.write(footer.encode('ascii'))
        
        # Record for PVD
        if time is None:
            time = float(step)
        
        self.time_steps = [(t, fn) for t, fn in self.time_steps if t != time]
        self.time_steps.append((time, filename))
        self.time_steps.sort(key=lambda x: x[0])
        
        return filepath
    
    def write_pvd(self, pvd_filename: str = 'simulation.pvd') -> str:
        """Write PVD collection file for time series
        
        Args:
            pvd_filename: Name of PVD file
            
        Returns:
            Path to written PVD file
        """
        filepath = os.path.join(self.output_dir, pvd_filename)
        
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1">',
            '  <Collection>',
        ]
        
        for time, filename in self.time_steps:
            lines.append(f'    <DataSet timestep="{time}" file="{filename}"/>')
        
        lines.extend([
            '  </Collection>',
            '</VTKFile>',
        ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        return filepath
    
    def get_info(self) -> str:
        """Return information about the writer"""
        dim_str = "2D" if self.dim == 2 else "3D"
        return (f"VTKWriter ({dim_str}): {self.output_dir}\n"
                f"  Domain: {self.Nx} x {self.Ny} x {self.Nz}\n"
                f"  Precision: {self.precision}\n"
                f"  Files written: {len(self.time_steps)}")
    
    def get_file_size_estimate(self, 
                               include_rho: bool = True,
                               include_u: bool = True,
                               include_mask: bool = True) -> Dict[str, float]:
        """Estimate output file size
        
        Args:
            include_rho: Include density field (also adds pressure)
            include_u: Include velocity field (also adds magnitude)
            include_mask: Include solid mask
            
        Returns:
            Dictionary with size estimates:
                - raw_MB: Uncompressed data size
                - estimated_MB: Estimated file size
                - n_points: Number of grid points
        """
        n_points = self.Nx * self.Ny * self.Nz
        bytes_per_float = 4 if self.precision == 'float32' else 8
        
        total_raw = 0
        if include_rho:
            # Density + Pressure (2 scalar fields)
            total_raw += n_points * bytes_per_float * 2
        if include_u:
            # Velocity (3 components) + Magnitude (1 scalar)
            total_raw += n_points * bytes_per_float * 4
        if include_mask:
            # Solid mask (1 byte per point)
            total_raw += n_points * 1
        
        return {
            'raw_MB': total_raw / 1e6,
            'estimated_MB': total_raw / 1e6,  # No compression currently
            'n_points': n_points
        }


class VTKWriterASCII:
    """Legacy ASCII VTK writer for debugging
    
    Writes VTK legacy format (.vtk) in ASCII.
    Slower and larger files, but human-readable.
    Use for debugging or when binary formats fail.
    
    Supports both 2D and 3D data.
    """
    
    def __init__(self, output_dir: str, domain_shape: tuple) -> None:
        self.output_dir = output_dir
        
        # Handle 2D vs 3D
        if len(domain_shape) == 2:
            self.Nx, self.Ny = domain_shape
            self.Nz = 1
            self.dim = 2
        else:
            self.Nx, self.Ny, self.Nz = domain_shape
            self.dim = 3
        
        os.makedirs(output_dir, exist_ok=True)
        self.time_steps = []
        self._scan_existing_files()
    
    def _scan_existing_files(self, prefix: str = 'lbm') -> None:
        """Scan for existing files"""
        if not os.path.exists(self.output_dir):
            return
        
        pattern = re.compile(rf'^{prefix}_(\d+)\.vtk$')
        
        for filename in os.listdir(self.output_dir):
            match = pattern.match(filename)
            if match:
                step = int(match.group(1))
                self.time_steps.append((float(step), filename))
        
        self.time_steps.sort(key=lambda x: x[0])
        
        if self.time_steps:
            print(f"    Found {len(self.time_steps)} existing VTK files")
    
    def _to_numpy(self, arr):
        if hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)
    
    def write(self, step: int, rho=None, u=None, solid_mask=None, 
              time=None, prefix='lbm') -> str:
        """Write legacy ASCII VTK file"""
        filename = f"{prefix}_{step:08d}.vtk"
        filepath = os.path.join(self.output_dir, filename)
        
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        n_points = Nx * Ny * Nz
        
        with open(filepath, 'w') as f:
            # Header
            f.write('# vtk DataFile Version 3.0\n')
            f.write(f'LBM Output Step {step}\n')
            f.write('ASCII\n')
            f.write('DATASET STRUCTURED_POINTS\n')
            f.write(f'DIMENSIONS {Nx} {Ny} {Nz}\n')
            f.write('ORIGIN 0 0 0\n')
            f.write('SPACING 1 1 1\n')
            f.write(f'POINT_DATA {n_points}\n')
            
            # Density
            if rho is not None:
                rho_np = self._to_numpy(rho).astype(np.float32)
                if self.dim == 2:
                    rho_np = rho_np[:, :, np.newaxis]
                
                f.write('SCALARS density float 1\n')
                f.write('LOOKUP_TABLE default\n')
                for k in range(Nz):
                    for j in range(Ny):
                        for i in range(Nx):
                            f.write(f'{rho_np[i,j,k]:.6g}\n')
            
            # Velocity
            if u is not None:
                u_np = self._to_numpy(u).astype(np.float32)
                if self.dim == 2:
                    # Expand (2, Nx, Ny) -> (3, Nx, Ny, 1)
                    u_3d = np.zeros((3, Nx, Ny, 1), dtype=np.float32)
                    u_3d[0, :, :, 0] = u_np[0]
                    u_3d[1, :, :, 0] = u_np[1]
                    u_np = u_3d
                
                f.write('VECTORS velocity float\n')
                for k in range(Nz):
                    for j in range(Ny):
                        for i in range(Nx):
                            f.write(f'{u_np[0,i,j,k]:.6g} {u_np[1,i,j,k]:.6g} {u_np[2,i,j,k]:.6g}\n')
            
            # Solid mask
            if solid_mask is not None:
                mask_np = self._to_numpy(solid_mask).astype(np.int8)
                if self.dim == 2:
                    mask_np = mask_np[:, :, np.newaxis]
                
                f.write('SCALARS solid_mask int 1\n')
                f.write('LOOKUP_TABLE default\n')
                for k in range(Nz):
                    for j in range(Ny):
                        for i in range(Nx):
                            f.write(f'{mask_np[i,j,k]}\n')
        
        if time is None:
            time = float(step)
        self.time_steps.append((time, filename))
        
        return filepath
    
    def write_pvd(self, pvd_filename: str = 'simulation.pvd') -> str:
        """Write PVD file for time series"""
        filepath = os.path.join(self.output_dir, pvd_filename)
        
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1">',
            '  <Collection>',
        ]
        
        for time, filename in sorted(set(self.time_steps)):
            lines.append(f'    <DataSet timestep="{time}" file="{filename}"/>')
        
        lines.extend([
            '  </Collection>',
            '</VTKFile>',
        ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        return filepath
    
    def get_file_size_estimate(self, **kwargs) -> Dict[str, float]:
        """Estimate file size (rough estimate for ASCII format)
        
        Returns:
            Dictionary with size estimates
        """
        n_points = self.Nx * self.Ny * self.Nz
        # ASCII format: ~50 bytes per point (rough estimate)
        return {
            'raw_MB': n_points * 50 / 1e6,
            'estimated_MB': n_points * 50 / 1e6,
            'n_points': n_points
        }