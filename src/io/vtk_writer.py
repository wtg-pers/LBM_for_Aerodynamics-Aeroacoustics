"""
VTK Output Module for LBM Solver

This module provides VTK file output for ParaView visualization.
Uses VTK XML ImageData format (.vti) with optional compression.

Supported Output Variables:
    - density (ρ): Scalar field  [dimensionless, ρ/ρ_0]
    - pressure (p): Scalar field  [lattice units, p = ρ·c_s²]
    - velocity (u): Vector field  [lattice units, Δx/Δt]
    - velocity_magnitude: Scalar field  [lattice units]
    - solid_mask: Integer field (0=fluid, 1=solid)

File Size Optimization:
    - Binary encoding (base64)
    - zlib compression (configurable level 1-9)
    - float32 precision (optional)

References:
    - VTK File Formats: https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
    - ParaView: https://www.paraview.org/

Author: LBM Development Team
Date: 2026-01
"""

import os
import base64
import zlib
import struct
from typing import TYPE_CHECKING, Optional, Dict, List, Union
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class VTKWriter:
    """VTK ImageData (.vti) writer with compression support
    
    Writes 3D scalar and vector fields to VTK XML format for ParaView.
    Supports binary encoding and zlib compression for reduced file size.
    
    Attributes:
        output_dir: Directory for output files
        precision: Data precision ('float32' or 'float64')
        compression_level: zlib compression (0=none, 1-9=compressed)
        
    Example:
        >>> writer = VTKWriter('./results', precision='float32', compression_level=6)
        >>> writer.write(step=1000, rho=rho, u=u, solid_mask=mask)
    """
    
    def __init__(self, 
                 output_dir: str,
                 domain_shape: tuple,
                 precision: str = 'float32',
                 compression_level: int = 6,
                 origin: tuple = (0.0, 0.0, 0.0),
                 spacing: tuple = (1.0, 1.0, 1.0)) -> None:
        """Initialize VTK writer
        
        Args:
            output_dir: Output directory path
            domain_shape: (Nx, Ny, Nz) grid dimensions
            precision: 'float32' or 'float64'
            compression_level: 0 (no compression) to 9 (max compression)
            origin: Grid origin (x0, y0, z0)  [physical units or lattice units]
            spacing: Grid spacing (dx, dy, dz)  [physical units or lattice units]
        """
        self.output_dir = output_dir
        self.Nx, self.Ny, self.Nz = domain_shape
        self.precision = precision
        self.compression_level = compression_level
        self.origin = origin
        self.spacing = spacing
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine numpy dtype
        self.dtype = np.float32 if precision == 'float32' else np.float64
        self.vtk_type = 'Float32' if precision == 'float32' else 'Float64'
        
        # For time series (PVD file)
        self.time_steps: List[tuple] = []  # [(time, filename), ...]
    
    def _to_numpy(self, arr: 'npt.NDArray') -> np.ndarray:
        """Convert CuPy array to NumPy if necessary"""
        if hasattr(arr, 'get'):  # CuPy array
            return arr.get()
        return np.asarray(arr)
    
    def _prepare_scalar(self, data: 'npt.NDArray') -> np.ndarray:
        """Prepare scalar field for VTK output
        
        VTK expects Fortran order (column-major) for structured grids.
        
        Args:
            data: Scalar field, shape (Nx, Ny, Nz)
            
        Returns:
            Flattened array in Fortran order
        """
        data = self._to_numpy(data).astype(self.dtype)
        # VTK uses Fortran ordering (x varies fastest)
        return np.asfortranarray(data).ravel(order='F')
    
    def _prepare_vector(self, data: 'npt.NDArray') -> np.ndarray:
        """Prepare vector field for VTK output
        
        Args:
            data: Vector field, shape (3, Nx, Ny, Nz)
            
        Returns:
            Interleaved array [vx0,vy0,vz0, vx1,vy1,vz1, ...]
        """
        data = self._to_numpy(data).astype(self.dtype)
        # Reshape to (Nx, Ny, Nz, 3) then flatten in Fortran order
        vector = np.moveaxis(data, 0, -1)  # (Nx, Ny, Nz, 3)
        return np.asfortranarray(vector).ravel(order='F')
    
    def _encode_data(self, data: np.ndarray) -> str:
        """Encode data as base64 with optional compression
        
        VTK XML format uses base64 encoding for binary data.
        With compression, data is: [header][compressed_data]
        Header contains: num_blocks, block_size, last_block_size, compressed_sizes
        
        Args:
            data: NumPy array to encode
            
        Returns:
            Base64 encoded string
        """
        raw_bytes = data.tobytes()
        
        if self.compression_level > 0:
            # Compressed format with header
            compressed = zlib.compress(raw_bytes, self.compression_level)
            
            # Header: [num_blocks, uncompressed_block_size, last_block_size, compressed_size]
            # All as 32-bit unsigned integers
            header = struct.pack('<4I', 
                                 1,                    # num_blocks
                                 len(raw_bytes),      # uncompressed size
                                 len(raw_bytes),      # last block size (same for 1 block)
                                 len(compressed))     # compressed size
            
            encoded = base64.b64encode(header + compressed).decode('ascii')
        else:
            # Uncompressed: just prepend size header
            header = struct.pack('<I', len(raw_bytes))
            encoded = base64.b64encode(header + raw_bytes).decode('ascii')
        
        return encoded
    
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
            rho: Density field (Nx, Ny, Nz)  [dimensionless]
            u: Velocity field (3, Nx, Ny, Nz)  [lattice units]
            solid_mask: Solid mask (Nx, Ny, Nz), True=solid
            time: Physical time (optional, for PVD)
            prefix: Filename prefix
            
        Returns:
            Path to written file
        """
        filename = f"{prefix}_{step:08d}.vti"
        filepath = os.path.join(self.output_dir, filename)
        
        # Build XML content
        xml_lines = self._build_vti_header()
        
        # Add data arrays
        data_arrays = []
        
        if rho is not None:
            # Density
            rho_data = self._prepare_scalar(rho)
            data_arrays.append(('density', 'Scalars', 1, rho_data))
            
            # Pressure: p = ρ * c_s² = ρ / 3
            pressure = rho_data / 3.0
            data_arrays.append(('pressure', 'Scalars', 1, pressure))
        
        if u is not None:
            # Velocity vector
            u_data = self._prepare_vector(u)
            data_arrays.append(('velocity', 'Vectors', 3, u_data))
            
            # Velocity magnitude
            u_np = self._to_numpy(u)
            u_mag = np.sqrt(np.sum(u_np**2, axis=0))
            u_mag_data = self._prepare_scalar(u_mag)
            data_arrays.append(('velocity_magnitude', 'Scalars', 1, u_mag_data))
        
        if solid_mask is not None:
            # Solid mask as integer (0=fluid, 1=solid)
            mask_np = self._to_numpy(solid_mask).astype(np.int8)
            mask_data = np.asfortranarray(mask_np).ravel(order='F')
            data_arrays.append(('solid_mask', 'Scalars', 1, mask_data))
        
        # Write PointData section
        xml_lines.append('      <PointData>')
        
        for name, attr_type, num_components, data in data_arrays:
            if name == 'solid_mask':
                vtype = 'Int8'
            else:
                vtype = self.vtk_type
            
            xml_lines.append(
                f'        <DataArray type="{vtype}" Name="{name}" '
                f'NumberOfComponents="{num_components}" format="binary"'
                + (f' compression="zlib"' if self.compression_level > 0 else '') + '>'
            )
            xml_lines.append('          ' + self._encode_data(data))
            xml_lines.append('        </DataArray>')
        
        xml_lines.append('      </PointData>')
        xml_lines.append('      <CellData></CellData>')
        
        # Close tags
        xml_lines.extend(self._build_vti_footer())
        
        # Write file
        with open(filepath, 'w') as f:
            f.write('\n'.join(xml_lines))
        
        # Record for PVD
        if time is None:
            time = float(step)
        self.time_steps.append((time, filename))
        
        return filepath
    
    def _build_vti_header(self) -> List[str]:
        """Build VTI file header"""
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        ox, oy, oz = self.origin
        dx, dy, dz = self.spacing
        
        # WholeExtent is 0-indexed, point-based (Nx points = Nx-1 cells, but for point data use Nx)
        # For PointData on structured grid: extent goes from 0 to N-1
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian"'
            + (' compressor="vtkZLibDataCompressor"' if self.compression_level > 0 else '') + '>',
            f'  <ImageData WholeExtent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}" '
            f'Origin="{ox} {oy} {oz}" Spacing="{dx} {dy} {dz}">',
            f'    <Piece Extent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}">',
        ]
        return lines
    
    def _build_vti_footer(self) -> List[str]:
        """Build VTI file footer"""
        return [
            '    </Piece>',
            '  </ImageData>',
            '</VTKFile>'
        ]
    
    def write_pvd(self, filename: str = 'simulation.pvd') -> str:
        """Write ParaView Data (PVD) file for time series
        
        PVD files allow ParaView to load all time steps as an animation.
        
        Args:
            filename: PVD filename
            
        Returns:
            Path to PVD file
        """
        filepath = os.path.join(self.output_dir, filename)
        
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            '  <Collection>'
        ]
        
        for time, vti_file in self.time_steps:
            lines.append(f'    <DataSet timestep="{time}" file="{vti_file}"/>')
        
        lines.extend([
            '  </Collection>',
            '</VTKFile>'
        ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        return filepath
    
    def get_file_size_estimate(self, 
                                include_rho: bool = True,
                                include_u: bool = True,
                                include_mask: bool = True) -> Dict[str, float]:
        """Estimate output file size
        
        Returns:
            Dictionary with size estimates in MB
        """
        n_points = self.Nx * self.Ny * self.Nz
        bytes_per_float = 4 if self.precision == 'float32' else 8
        
        # Raw sizes
        scalar_size = n_points * bytes_per_float
        vector_size = n_points * 3 * bytes_per_float
        mask_size = n_points * 1  # int8
        
        total_raw = 0
        if include_rho:
            total_raw += scalar_size * 2  # density + pressure
        if include_u:
            total_raw += vector_size + scalar_size  # velocity + magnitude
        if include_mask:
            total_raw += mask_size
        
        # Compression ratio estimate (typical for CFD data)
        if self.compression_level > 0:
            compression_ratio = 0.3  # ~70% reduction typical
        else:
            compression_ratio = 1.35  # base64 expansion
        
        estimated_size = total_raw * compression_ratio
        
        return {
            'raw_MB': total_raw / 1e6,
            'estimated_MB': estimated_size / 1e6,
            'n_points': n_points
        }


class VTKWriterLegacy:
    """Legacy VTK writer (.vtk format)
    
    Simpler format but no compression. Use VTKWriter for better performance.
    Kept for compatibility with older ParaView versions.
    """
    
    def __init__(self, output_dir: str, domain_shape: tuple) -> None:
        self.output_dir = output_dir
        self.Nx, self.Ny, self.Nz = domain_shape
        os.makedirs(output_dir, exist_ok=True)
    
    def _to_numpy(self, arr):
        if hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)
    
    def write(self, step: int, rho=None, u=None, solid_mask=None, prefix='lbm') -> str:
        """Write legacy VTK file"""
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
                f.write('SCALARS density float 1\n')
                f.write('LOOKUP_TABLE default\n')
                for k in range(Nz):
                    for j in range(Ny):
                        for i in range(Nx):
                            f.write(f'{rho_np[i,j,k]:.6f}\n')
            
            # Velocity
            if u is not None:
                u_np = self._to_numpy(u).astype(np.float32)
                f.write('VECTORS velocity float\n')
                for k in range(Nz):
                    for j in range(Ny):
                        for i in range(Nx):
                            f.write(f'{u_np[0,i,j,k]:.6f} {u_np[1,i,j,k]:.6f} {u_np[2,i,j,k]:.6f}\n')
        
        return filepath
