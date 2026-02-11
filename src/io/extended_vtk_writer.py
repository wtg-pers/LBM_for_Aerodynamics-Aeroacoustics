"""
Extended VTK Writer with Vortex Identification Fields

This module extends the base VTKWriter to support additional post-processed
scalar and vector fields, particularly vortex identification criteria
(Q-criterion, Lambda2, vorticity) for iso-surface visualization in ParaView.

ParaView Visualization Guide:
============================

Q-criterion iso-surface:
    1. Open .vti file in ParaView
    2. Filters → Contour
    3. Contour By → "Q_criterion"
    4. Set iso-value (start with Q* ≈ 1.0, adjust up/down)
    5. Color by "velocity_magnitude" for physical insight

Lambda2 iso-surface:
    1. Open .vti file in ParaView
    2. Filters → Contour
    3. Contour By → "Lambda2"
    4. Set iso-value to a NEGATIVE number (e.g., λ₂* ≈ -1.0)
    5. Color by "vorticity_magnitude"

Iso-volume (volume rendering):
    1. Filters → Threshold
    2. Scalars → "Q_criterion"
    3. Set Lower threshold > 0 (e.g., Q* > 0.5)
    4. Apply → shows only vortex regions as solid volume

References:
    - ParaView: https://www.paraview.org/
    - VTK XML format: https://vtk.org/doc/nightly/html/

Author: LBM Development Team
Date: 2026-02
"""

import os
import struct
import re
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class ExtendedVTKWriter:
    """VTK ImageData writer supporting arbitrary scalar/vector fields

    Extends the base VTK writer concept to accept any number of additional
    fields beyond the standard rho, u, and solid_mask.

    Key Features:
        - Standard LBM fields (ρ, u, p)
        - Vortex criteria (Q, λ₂, |ω|) computed on-the-fly
        - Arbitrary user-defined scalar/vector fields
        - Body force field F(x) from actuator line
        - PVD time-series collection file

    Usage:
        >>> writer = ExtendedVTKWriter('./results', (Nx, Ny, Nz))
        >>> writer.write(
        ...     step=1000,
        ...     rho=rho,
        ...     u=u,
        ...     solid_mask=mask,
        ...     extra_scalars={'Q_criterion': Q, 'Lambda2': lam2},
        ...     extra_vectors={'vorticity': omega, 'body_force': F},
        ... )
    """

    def __init__(
        self,
        output_dir: str,
        domain_shape: tuple,
        precision: str = 'float32',
        origin: Optional[tuple] = None,
        spacing: Optional[tuple] = None,
    ) -> None:
        """Initialize extended VTK writer

        Args:
            output_dir: Directory for output files
            domain_shape: (Nx, Ny) or (Nx, Ny, Nz)
            precision: 'float32' or 'float64'
            origin: Grid origin (default (0,0,0))
            spacing: Grid spacing (default (1,1,1))
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Handle 2D/3D
        if len(domain_shape) == 2:
            self.Nx, self.Ny = domain_shape
            self.Nz = 1
            self.dim = 2
        else:
            self.Nx, self.Ny, self.Nz = domain_shape
            self.dim = 3

        self.precision = precision
        self.dtype = np.float32 if precision == 'float32' else np.float64
        self.vtk_type = 'Float32' if precision == 'float32' else 'Float64'

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

        self.time_steps: List[Tuple[float, str]] = []
        self._scan_existing_files()

    def _scan_existing_files(self, prefix: str = 'lbm') -> None:
        """Scan for existing VTI files in output directory"""
        if not os.path.exists(self.output_dir):
            return
        pattern = re.compile(rf'^{prefix}_(\d+)\.vti$')
        for filename in os.listdir(self.output_dir):
            match = pattern.match(filename)
            if match:
                step = int(match.group(1))
                self.time_steps.append((float(step), filename))
        self.time_steps.sort(key=lambda x: x[0])

    @staticmethod
    def _to_numpy(arr: Any) -> np.ndarray:
        """Convert CuPy/JAX array to NumPy"""
        if hasattr(arr, 'get'):
            return arr.get()
        return np.asarray(arr)

    def _expand_2d(self, arr: np.ndarray, is_vector: bool = False) -> np.ndarray:
        """Expand 2D array to 3D (Nz=1) for VTK"""
        if self.dim == 3:
            return arr
        if is_vector:
            # (2, Nx, Ny) → (3, Nx, Ny, 1)
            dim = arr.shape[0]
            arr_3d = np.zeros((3, self.Nx, self.Ny, 1), dtype=arr.dtype)
            arr_3d[:dim, :, :, 0] = arr
            return arr_3d
        else:
            # (Nx, Ny) → (Nx, Ny, 1)
            return arr[:, :, np.newaxis]

    def _prepare_scalar(self, data: np.ndarray) -> bytes:
        """Prepare scalar field for VTK: transpose to VTK order and serialize

        VTK ImageData expects x varying fastest:
            (Nx, Ny, Nz) → transpose(2,1,0) → (Nz, Ny, Nx) → flatten
        """
        data_np = self._to_numpy(data).astype(self.dtype)
        if self.dim == 2:
            data_np = self._expand_2d(data_np, is_vector=False)
        vtk_data = np.ascontiguousarray(data_np.transpose(2, 1, 0)).ravel()
        return vtk_data.tobytes()

    def _prepare_vector(self, data: np.ndarray) -> bytes:
        """Prepare vector field for VTK: transpose and interleave components

        Input: (3, Nx, Ny, Nz) or (2, Nx, Ny) for 2D
        VTK: interleaved (Nz, Ny, Nx, 3) → flatten
        """
        data_np = self._to_numpy(data).astype(self.dtype)
        if self.dim == 2:
            data_np = self._expand_2d(data_np, is_vector=True)
        # (3, Nx, Ny, Nz) → (3, Nz, Ny, Nx) → (Nz, Ny, Nx, 3) → flatten
        transposed = data_np.transpose(0, 3, 2, 1)
        interleaved = np.ascontiguousarray(transposed.transpose(1, 2, 3, 0))
        return interleaved.ravel().tobytes()

    # -----------------------------------------------------------------
    # Main write method
    # -----------------------------------------------------------------

    def write(
        self,
        step: int,
        rho: Optional['npt.NDArray'] = None,
        u: Optional['npt.NDArray'] = None,
        solid_mask: Optional['npt.NDArray'] = None,
        extra_scalars: Optional[Dict[str, 'npt.NDArray']] = None,
        extra_vectors: Optional[Dict[str, 'npt.NDArray']] = None,
        time: Optional[float] = None,
        prefix: str = 'lbm',
    ) -> str:
        """Write VTK file with standard and extra fields

        Standard fields (automatically derived):
            - density (ρ)
            - pressure (p = ρ·cs²)
            - velocity (u)
            - velocity_magnitude (|u|)
            - solid_mask

        Extra fields (user-provided):
            - extra_scalars: dict of name → array (Nx, Ny, Nz)
            - extra_vectors: dict of name → array (3, Nx, Ny, Nz)

        Args:
            step: Time step number
            rho: Density field  [dimensionless]
            u: Velocity field  [Δx/Δt]
            solid_mask: Solid mask (True=solid)
            extra_scalars: Additional scalar fields
            extra_vectors: Additional vector fields
            time: Physical time for PVD
            prefix: Filename prefix

        Returns:
            Path to written file
        """
        filename = f"{prefix}_{step:08d}.vti"
        filepath = os.path.join(self.output_dir, filename)

        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        ox, oy, oz = self.origin
        dx, dy, dz = self.spacing

        # Collect all data arrays: (name, vtk_type, ncomp, bytes)
        data_arrays: List[Tuple[str, str, int, bytes]] = []

        # --- Standard fields ---
        if rho is not None:
            data_arrays.append(('density', self.vtk_type, 1,
                                self._prepare_scalar(rho)))
            # Pressure: p = ρ · cs² = ρ / 3
            rho_np = self._to_numpy(rho).astype(self.dtype)
            pressure = rho_np / 3.0
            data_arrays.append(('pressure', self.vtk_type, 1,
                                self._prepare_scalar(pressure)))

        if u is not None:
            data_arrays.append(('velocity', self.vtk_type, 3,
                                self._prepare_vector(u)))
            # Velocity magnitude
            u_np = self._to_numpy(u).astype(np.float64)
            u_mag = np.sqrt(np.sum(u_np**2, axis=0)).astype(self.dtype)
            data_arrays.append(('velocity_magnitude', self.vtk_type, 1,
                                self._prepare_scalar(u_mag)))

        if solid_mask is not None:
            mask_np = self._to_numpy(solid_mask).astype(np.int8)
            if self.dim == 2:
                mask_np = mask_np[:, :, np.newaxis]
            mask_vtk = np.ascontiguousarray(mask_np.transpose(2, 1, 0)).ravel()
            data_arrays.append(('solid_mask', 'Int8', 1, mask_vtk.tobytes()))

        # --- Extra scalar fields ---
        if extra_scalars is not None:
            for name, field in extra_scalars.items():
                data_arrays.append((name, self.vtk_type, 1,
                                    self._prepare_scalar(field)))

        # --- Extra vector fields ---
        if extra_vectors is not None:
            for name, field in extra_vectors.items():
                data_arrays.append((name, self.vtk_type, 3,
                                    self._prepare_vector(field)))

        # --- Build VTK XML ---
        xml_lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="ImageData" version="1.0" '
            'byte_order="LittleEndian" header_type="UInt64">',
            f'  <ImageData WholeExtent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}" '
            f'Origin="{ox} {oy} {oz}" Spacing="{dx} {dy} {dz}">',
            f'    <Piece Extent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}">',
            '      <PointData>',
        ]

        offset = 0
        for name, vtype, ncomp, data_bytes in data_arrays:
            xml_lines.append(
                f'        <DataArray type="{vtype}" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="appended" '
                f'offset="{offset}"/>'
            )
            offset += 8 + len(data_bytes)  # 8 bytes UInt64 size header

        xml_lines.extend([
            '      </PointData>',
            '      <CellData>',
            '      </CellData>',
            '    </Piece>',
            '  </ImageData>',
            '  <AppendedData encoding="raw">',
            '   _',
        ])

        # --- Write binary file ---
        with open(filepath, 'wb') as f:
            header_text = '\n'.join(xml_lines)
            f.write(header_text.encode('ascii'))

            for name, vtype, ncomp, data_bytes in data_arrays:
                f.write(struct.pack('<Q', len(data_bytes)))
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
        """Write PVD collection file for time-series animation"""
        filepath = os.path.join(self.output_dir, pvd_filename)
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1">',
            '  <Collection>',
        ]
        for time, filename in self.time_steps:
            lines.append(
                f'    <DataSet timestep="{time}" file="{filename}"/>'
            )
        lines.extend([
            '  </Collection>',
            '</VTKFile>',
        ])
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        return filepath

    def get_file_size_estimate(
        self,
        n_extra_scalars: int = 0,
        n_extra_vectors: int = 0,
    ) -> Dict[str, float]:
        """Estimate output file size per step"""
        n_pts = self.Nx * self.Ny * self.Nz
        bpf = 4 if self.precision == 'float32' else 8

        # Standard: density, pressure, velocity(3), magnitude, mask
        n_scalars = 3 + n_extra_scalars  # density, pressure, magnitude + extras
        n_vectors = 1 + n_extra_vectors  # velocity + extras
        n_masks = 1

        total = n_pts * (n_scalars * bpf + n_vectors * 3 * bpf + n_masks)
        return {
            'raw_MB': total / 1e6,
            'estimated_MB': total / 1e6,
            'n_points': n_pts,
        }

    def get_info(self) -> str:
        dim_str = "2D" if self.dim == 2 else "3D"
        return (
            f"ExtendedVTKWriter ({dim_str}): {self.output_dir}\n"
            f"  Domain: {self.Nx} × {self.Ny} × {self.Nz}\n"
            f"  Precision: {self.precision}\n"
            f"  Files written: {len(self.time_steps)}"
        )