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
    from src.io.field_units import FieldUnits


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
                 spacing: tuple = None,
                 units: Optional['FieldUnits'] = None) -> None:
        """Initialize VTK writer

        Args:
            output_dir: Output directory path
            domain_shape: (Nx, Ny) for 2D or (Nx, Ny, Nz) for 3D  [lattice units]
            precision: 'float32' or 'float64'
            compression_level: Compression level (currently unused)
            origin: Grid origin  [physical or lattice units]
            spacing: Grid spacing  [physical or lattice units]
            units: FieldUnits value converter (None = lattice passthrough)
        """
        self.output_dir = output_dir
        self.units = units
        
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
    
    def _units_convert(self, name: str, arr: np.ndarray):
        """Apply the FieldUnits value/name mapping (identity when unset)."""
        if self.units is None:
            return name, arr
        return self.units.convert(name, arr)

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
              extra_scalars: Optional[Dict[str, 'npt.NDArray']] = None,
              extra_vectors: Optional[Dict[str, 'npt.NDArray']] = None,
              time: Optional[float] = None,
              prefix: str = 'lbm') -> str:
        """Write VTK file for current time step
        
        Standard fields:
            - density (ρ)        ← from rho
            - velocity (u)       ← from u
            - solid_mask         ← from solid_mask

        Derived fields removed (use ParaView Calculator):
            - pressure:           p = density / 3
            - velocity_magnitude: mag(velocity)

        Args:
            step: Time step number
            rho: Density field (Nx, Ny[, Nz])  [dimensionless]
            u: Velocity field (dim, Nx, Ny[, Nz])  [lattice units]
            solid_mask: Solid mask (Nx, Ny[, Nz]), True=solid
            extra_scalars: Additional scalar fields {name: array(Nx,Ny,Nz)}
            extra_vectors: Additional vector fields {name: array(3,Nx,Ny,Nz)}
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
            name, rho_np = self._units_convert('density', self._to_numpy(rho))
            rho_np = rho_np.astype(self.dtype)

            # Expand 2D to 3D if needed
            if self.dim == 2:
                rho_np = self._expand_2d_to_3d(rho_np, is_vector=False)

            # VTK expects x to vary fastest
            # Transpose (Nx, Ny, Nz) -> (Nz, Ny, Nx) then flatten
            rho_vtk = np.ascontiguousarray(rho_np.transpose(2, 1, 0)).ravel()
            data_arrays.append((name, self.vtk_type, 1, rho_vtk.tobytes()))
            # Pressure: p = ρ * c_s² = ρ / 3

        if u is not None:
            name, u_np = self._units_convert('velocity', self._to_numpy(u))
            u_np = u_np.astype(self.dtype)

            # Expand 2D to 3D if needed
            if self.dim == 2:
                u_np = self._expand_2d_to_3d(u_np, is_vector=True)

            # Transpose spatial dimensions: (3, Nx, Ny, Nz) -> (3, Nz, Ny, Nx)
            u_transposed = u_np.transpose(0, 3, 2, 1)

            # Interleave components: (Nz, Ny, Nx, 3)
            u_interleaved = np.ascontiguousarray(u_transposed.transpose(1, 2, 3, 0))
            data_arrays.append((name, self.vtk_type, 3, u_interleaved.ravel().tobytes()))
        
        if solid_mask is not None:
            mask_np = self._to_numpy(solid_mask).astype(np.int8)
            
            # Expand 2D to 3D if needed
            if self.dim == 2:
                mask_np = self._expand_2d_to_3d(mask_np, is_vector=False)
            
            mask_vtk = np.ascontiguousarray(mask_np.transpose(2, 1, 0)).ravel()
            data_arrays.append(('solid_mask', 'Int8', 1, mask_vtk.tobytes()))
        
        # --- Extra scalar fields ---
        if extra_scalars is not None:
            for name, field in extra_scalars.items():
                name, field_np = self._units_convert(
                    name, self._to_numpy(field))
                field_np = field_np.astype(self.dtype)
                if self.dim == 2:
                    field_np = self._expand_2d_to_3d(field_np, is_vector=False)
                field_vtk = np.ascontiguousarray(
                    field_np.transpose(2, 1, 0)
                ).ravel()
                data_arrays.append(
                    (name, self.vtk_type, 1, field_vtk.tobytes())
                )

        # --- Extra vector fields ---
        if extra_vectors is not None:
            for name, field in extra_vectors.items():
                name, field_np = self._units_convert(
                    name, self._to_numpy(field))
                field_np = field_np.astype(self.dtype)
                if self.dim == 2:
                    field_np = self._expand_2d_to_3d(field_np, is_vector=True)
                field_t = field_np.transpose(0, 3, 2, 1)  # (3,Nx,Ny,Nz) → (3,Nz,Ny,Nx)
                field_interleaved = np.ascontiguousarray(
                    field_t.transpose(1, 2, 3, 0)  # → (Nz,Ny,Nx,3)
                )
                data_arrays.append(
                    (name, self.vtk_type, 3, field_interleaved.ravel().tobytes())
                )
        
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
                               include_mask: bool = True,
                               n_extra_scalars: int = 0,
                               n_extra_vectors: int = 0) -> Dict[str, float]:
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
            # Density only (pressure removed — derivable in ParaView)
            total_raw += n_points * bytes_per_float * 1
        if include_u:
            # Velocity (3 components) only (magnitude removed — derivable)
            total_raw += n_points * bytes_per_float * 3
        if include_mask:
            # Solid mask (1 byte per point)
            total_raw += n_points * 1
        # Extra fields
        total_raw += n_points * bytes_per_float * n_extra_scalars
        total_raw += n_points * bytes_per_float * 3 * n_extra_vectors
        
        return {
            'raw_MB': total_raw / 1e6,
            'estimated_MB': total_raw / 1e6,  # No compression currently
            'n_points': n_points
        }
