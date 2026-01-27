"""
VTK Output Module for LBM Solver

This module provides VTK file output for ParaView visualization.
Uses VTK XML ImageData format (.vti) with appended binary data.

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
Date: 2026-01
"""

import os
import struct
import re
from typing import TYPE_CHECKING, Optional, Dict, List
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class VTKWriter:
    """VTK ImageData (.vti) writer with appended binary format
    
    Writes 3D scalar and vector fields to VTK XML format for ParaView.
    Uses appended binary encoding for compatibility and efficiency.
    
    Attributes:
        output_dir: Directory for output files
        precision: Data precision ('float32' or 'float64')
        
    Example:
        >>> writer = VTKWriter('./results', domain_shape=(100,40,40))
        >>> writer.write(step=1000, rho=rho, u=u, solid_mask=mask)
    """
    
    def __init__(self, 
                 output_dir: str,
                 domain_shape: tuple,
                 precision: str = 'float32',
                 compression_level: int = 0,
                 origin: tuple = (0.0, 0.0, 0.0),
                 spacing: tuple = (1.0, 1.0, 1.0)) -> None:
        """Initialize VTK writer
        
        Args:
            output_dir: Output directory path
            domain_shape: (Nx, Ny, Nz) grid dimensions  [lattice units]
            precision: 'float32' or 'float64'
            compression_level: Compression level (currently unused)
            origin: Grid origin (x0, y0, z0)  [physical or lattice units]
            spacing: Grid spacing (dx, dy, dz)  [physical or lattice units]
        """
        self.output_dir = output_dir
        self.Nx, self.Ny, self.Nz = domain_shape
        self.precision = precision
        self.origin = origin
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
            print(f"    Step range: {int(self.time_steps[0][0])} → {int(self.time_steps[-1][0])}")
    
    def _to_numpy(self, arr: 'npt.NDArray') -> np.ndarray:
        """Convert CuPy array to NumPy if necessary"""
        if hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)
    
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
        
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        ox, oy, oz = self.origin
        dx, dy, dz = self.spacing
        
        # Collect data arrays: (name, vtk_type, num_components, data_bytes)
        data_arrays = []
        
        if rho is not None:
            # Density - C order, then reshape for VTK (x fastest)
            rho_np = self._to_numpy(rho).astype(self.dtype)
            # VTK expects x to vary fastest, but our array is (Nx, Ny, Nz) in C order
            # Transpose to (Nz, Ny, Nx) then flatten in C order = x varies fastest
            rho_vtk = np.ascontiguousarray(rho_np.transpose(2, 1, 0)).ravel()
            data_arrays.append(('density', self.vtk_type, 1, rho_vtk.tobytes()))
            
            # Pressure: p = ρ * c_s² = ρ / 3
            pressure = (rho_vtk / 3.0).astype(self.dtype)
            data_arrays.append(('pressure', self.vtk_type, 1, pressure.tobytes()))
        
        if u is not None:
            # Velocity vector (3, Nx, Ny, Nz)
            u_np = self._to_numpy(u).astype(self.dtype)
            
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
            mask_vtk = np.ascontiguousarray(mask_np.transpose(2, 1, 0)).ravel()
            data_arrays.append(('solid_mask', 'Int8', 1, mask_vtk.tobytes()))
        
        # Build XML with correct extent
        # VTK ImageData extent: number of points in each dimension
        # For Nx cells, we have Nx points (cell-centered data treated as point data)
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
            offset += 8 + len(data_bytes)
        
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
        
        self.time_steps = [(t, f) for t, f in self.time_steps if t != time]
        self.time_steps.append((time, filename))
        self.time_steps.sort(key=lambda x: x[0])
        
        return filepath
    
    def write_pvd(self, filename: str = 'simulation.pvd') -> str:
        """Write ParaView Data (PVD) file for time series"""
        filepath = os.path.join(self.output_dir, filename)
        
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            '  <Collection>'
        ]
        
        for time, vti_file in sorted(self.time_steps, key=lambda x: x[0]):
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
        """Estimate output file size"""
        n_points = self.Nx * self.Ny * self.Nz
        bytes_per_float = 4 if self.precision == 'float32' else 8
        
        total_raw = 0
        if include_rho:
            total_raw += n_points * bytes_per_float * 2
        if include_u:
            total_raw += n_points * bytes_per_float * 4
        if include_mask:
            total_raw += n_points * 1
        
        return {
            'raw_MB': total_raw / 1e6,
            'estimated_MB': total_raw / 1e6,
            'n_points': n_points
        }


class VTKWriterASCII:
    """Legacy ASCII VTK writer (.vtk format)
    
    Simplest format, compatible with all VTK readers.
    Use for debugging or when binary formats fail.
    """
    
    def __init__(self, output_dir: str, domain_shape: tuple) -> None:
        self.output_dir = output_dir
        self.Nx, self.Ny, self.Nz = domain_shape
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
                f.write('SCALARS density float 1\n')
                f.write('LOOKUP_TABLE default\n')
                # VTK STRUCTURED_POINTS: x varies fastest, then y, then z
                for k in range(Nz):
                    for j in range(Ny):
                        for i in range(Nx):
                            f.write(f'{rho_np[i,j,k]:.6g}\n')
                
                # Pressure
                f.write('SCALARS pressure float 1\n')
                f.write('LOOKUP_TABLE default\n')
                for k in range(Nz):
                    for j in range(Ny):
                        for i in range(Nx):
                            f.write(f'{rho_np[i,j,k]/3.0:.6g}\n')
            
            # Velocity
            if u is not None:
                u_np = self._to_numpy(u).astype(np.float32)
                f.write('VECTORS velocity float\n')
                for k in range(Nz):
                    for j in range(Ny):
                        for i in range(Nx):
                            f.write(f'{u_np[0,i,j,k]:.6g} {u_np[1,i,j,k]:.6g} {u_np[2,i,j,k]:.6g}\n')
                
                # Velocity magnitude
                u_mag = np.sqrt(np.sum(u_np**2, axis=0))
                f.write('SCALARS velocity_magnitude float 1\n')
                f.write('LOOKUP_TABLE default\n')
                for k in range(Nz):
                    for j in range(Ny):
                        for i in range(Nx):
                            f.write(f'{u_mag[i,j,k]:.6g}\n')
            
            # Solid mask
            if solid_mask is not None:
                mask_np = self._to_numpy(solid_mask).astype(np.int32)
                f.write('SCALARS solid_mask int 1\n')
                f.write('LOOKUP_TABLE default\n')
                for k in range(Nz):
                    for j in range(Ny):
                        for i in range(Nx):
                            f.write(f'{mask_np[i,j,k]}\n')
        
        if time is None:
            time = float(step)
        self.time_steps = [(t, fn) for t, fn in self.time_steps if t != time]
        self.time_steps.append((time, filename))
        self.time_steps.sort(key=lambda x: x[0])
        
        return filepath
    
    def write_pvd(self, filename: str = 'simulation.pvd') -> str:
        """Write PVD file for time series"""
        filepath = os.path.join(self.output_dir, filename)
        
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1">',
            '  <Collection>'
        ]
        
        for time, vtk_file in sorted(self.time_steps, key=lambda x: x[0]):
            lines.append(f'    <DataSet timestep="{time}" file="{vtk_file}"/>')
        
        lines.extend([
            '  </Collection>',
            '</VTKFile>'
        ])
        
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        
        return filepath
    
    def get_file_size_estimate(self, **kwargs) -> Dict[str, float]:
        """Estimate file size (rough)"""
        n_points = self.Nx * self.Ny * self.Nz
        return {
            'raw_MB': n_points * 50 / 1e6,  # ~50 bytes per point in ASCII
            'estimated_MB': n_points * 50 / 1e6,
            'n_points': n_points
        }
