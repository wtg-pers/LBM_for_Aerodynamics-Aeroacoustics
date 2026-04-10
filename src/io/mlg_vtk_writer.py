"""
Multi-Level Grid VTK Writer — AMR Output for ParaView

Directory Structure:
    vtk/
    ├── lbm_00005000.vth              ← .vth in root (open this in ParaView)
    ├── level0/
    │   └── lbm_00005000_level0.vti   ← Level 0 .vti
    └── level1/
        └── lbm_00005000_level1.vti   ← Level 1 .vti

ParaView Usage:
    Open the .vth file → all levels appear at correct positions.

Author: LBM Development Team
Date: 2026-04
"""

import os
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict

import numpy as np

if TYPE_CHECKING:
    from src.grid.overlap_manager import OverlapManager
    from src.grid.level_scaling import LevelScaler


class MLGVTKWriter:
    """Multi-level VTK writer with AMR hierarchy support.

    Creates per-level .vti files in level subdirectories and a .vth
    meta-file in the root vtk directory.

    Args:
        output_dir: Root VTK output directory.
        coarse_shape: Level 0 domain dimensions (Nx, Ny, Nz).
        overlap_mgr: OverlapManager with all level pair geometries.
        scaler: LevelScaler for per-level dx values.
        num_levels: Total number of grid levels.
        precision: Data precision ('float32' or 'float64').
    """

    REFINE_RATIO: int = 2

    def __init__(
        self,
        output_dir: str,
        coarse_shape: Tuple[int, int, int],
        overlap_mgr: 'OverlapManager',
        scaler: 'LevelScaler',
        num_levels: int,
        precision: str = 'float32',
    ) -> None:
        self.output_dir = output_dir
        self._num_levels = num_levels
        self._overlap_mgr = overlap_mgr
        self._scaler = scaler
        self._precision = precision
        self._dtype = np.float32 if precision == 'float32' else np.float64
        self._vtk_type = 'Float32' if precision == 'float32' else 'Float64'

        os.makedirs(output_dir, exist_ok=True)

        # ── Create level subdirectories ──────────────────────────
        self._level_dirs: List[str] = []
        for k in range(num_levels):
            level_dir = os.path.join(output_dir, f'level{k}')
            os.makedirs(level_dir, exist_ok=True)
            self._level_dirs.append(level_dir)

        # ── Create vth subdirectory ──────────────────────────────
        self._vth_dir = os.path.join(output_dir, 'vth')
        os.makedirs(self._vth_dir, exist_ok=True)

        # ── Per-level metadata ───────────────────────────────────
        self._level_info: List[Dict] = []

        # Level 0
        Nx, Ny, Nz = coarse_shape
        self._level_info.append({
            'shape': (Nx, Ny, Nz),
            'origin': (0.0, 0.0, 0.0),
            'spacing': (1.0, 1.0, 1.0),
            'amr_box': (0, Nx - 2, 0, Ny - 2, 0, Nz - 2),
        })

        # Fine levels — origin computed from parent's origin + spacing
        for k in range(1, num_levels):
            region = overlap_mgr.get_region(k - 1)
            lu = scaler.get_level_units(k)
            dx_k = lu.dx  # this level's spacing

            fdc = region.fine_domain_coarse
            Nx_f, Ny_f, Nz_f = region.fine_shape

            # Parent level's physical origin and spacing
            parent = self._level_info[k - 1]
            px, py, pz = parent['origin']
            pdx, pdy, pdz = parent['spacing']

            # Physical origin = parent_origin + fdc_start × parent_spacing
            origin = (
                px + fdc.x_start * pdx,
                py + fdc.y_start * pdy,
                pz + fdc.z_start * pdz,
            )

            # AMR box in global cell indices at this level's resolution
            # Global position = origin / dx_k (convert physical to this level's cells)
            amr_lo_x = round(origin[0] / dx_k)
            amr_lo_y = round(origin[1] / dx_k)
            amr_lo_z = round(origin[2] / dx_k)
            amr_box = (
                amr_lo_x, amr_lo_x + Nx_f - 2,
                amr_lo_y, amr_lo_y + Ny_f - 2,
                amr_lo_z, amr_lo_z + Nz_f - 2,
            )

            self._level_info.append({
                'shape': (Nx_f, Ny_f, Nz_f),
                'origin': origin,
                'spacing': (dx_k, dx_k, dx_k),
                'amr_box': amr_box,
            })

        # ── Time-series tracking ─────────────────────────────────
        self.time_steps: List[Tuple[float, str]] = []
        self._scan_existing_files()

    # =================================================================
    # Public: Write
    # =================================================================

    def write(
        self,
        step: int,
        mlg: object,
        time: Optional[float] = None,
        prefix: str = 'lbm',
    ) -> str:
        """Write all levels as .vti files + .vth AMR meta-file.

        .vti files → vtk/level{k}/ subdirectories
        .vth file  → vtk/ root directory

        Args:
            step: Timestep number.
            mlg: MultiLevelGrid object.
            time: Physical time (defaults to step number).
            prefix: Filename prefix.

        Returns:
            Path to the .vth file.
        """
        if time is None:
            time = float(step)

        vti_relative_paths = []

        # ── Write per-level .vti files ───────────────────────────
        for k in range(self._num_levels):
            level_sim = mlg.get_level(k)
            info = self._level_info[k]

            rho = level_sim.rho
            u = level_sim.u
            if rho is None or u is None:
                continue

            # Extract solid mask from obstacle BC (if present)
            solid_mask = None
            if (level_sim.obstacle_bc is not None
                    and hasattr(level_sim.obstacle_bc, 'solid_mask')):
                solid_mask = level_sim.obstacle_bc.solid_mask

            # File in level subdirectory
            vti_filename = f"{prefix}_{step:08d}_level{k}.vti"
            vti_filepath = os.path.join(self._level_dirs[k], vti_filename)

            self._write_vti(
                filepath=vti_filepath,
                shape=info['shape'],
                origin=info['origin'],
                spacing=info['spacing'],
                rho=rho,
                u=u,
                solid_mask=solid_mask,
            )

            # Relative path from .vth location (vtk/vth/) to .vti (vtk/level{k}/)
            vti_relative_paths.append(f"../level{k}/{vti_filename}")

        # ── Write .vth AMR meta-file in vth/ subdir ──────────────
        vth_filename = f"{prefix}_{step:08d}.vth"
        vth_filepath = os.path.join(self._vth_dir, vth_filename)
        self._write_vth(vth_filepath, vti_relative_paths, step)

        # ── Track time-series ────────────────────────────────────
        self.time_steps = [
            (t, fn) for t, fn in self.time_steps if t != time
        ]
        self.time_steps.append((time, vth_filename))
        self.time_steps.sort(key=lambda x: x[0])

        return vth_filepath

    # =================================================================
    # Private: .vti writer (per-level)
    # =================================================================

    def _write_vti(
        self,
        filepath: str,
        shape: Tuple[int, int, int],
        origin: Tuple[float, float, float],
        spacing: Tuple[float, float, float],
        rho: np.ndarray,
        u: np.ndarray,
        solid_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Write a single level's data as VTK ImageData (.vti).

        Array convention: rho shape = (Nx, Ny, Nz),
                          u shape = (dim, Nx, Ny, Nz).
        VTK expects Fortran ordering (x varies fastest), so we
        transpose to (Nz, Ny, Nx) before writing.
        """
        import struct

        Nx, Ny, Nz = shape

        # ── Convert to numpy ─────────────────────────────────────
        if hasattr(rho, 'get'):
            rho_np = rho.get()
            u_np = u.get()
        else:
            rho_np = np.asarray(rho)
            u_np = np.asarray(u)

        dtype = self._dtype
        vtk_type = self._vtk_type

        # ── Prepare data arrays (VTK Fortran order) ──────────────
        rho_flat = np.ascontiguousarray(
            rho_np.transpose(2, 1, 0)
        ).astype(dtype).tobytes()

        u_interleaved = np.ascontiguousarray(
            u_np.transpose(3, 2, 1, 0)
        ).astype(dtype).tobytes()

        # ── Build XML ────────────────────────────────────────────
        ox, oy, oz = origin
        sx, sy, sz = spacing

        data_arrays = [
            ('density', vtk_type, 1, rho_flat),
            ('velocity', vtk_type, 3, u_interleaved),
        ]

        if solid_mask is not None:
            if hasattr(solid_mask, 'get'):
                mask_np = solid_mask.get()
            else:
                mask_np = np.asarray(solid_mask)
            mask_flat = np.ascontiguousarray(
                mask_np.astype(np.int8).transpose(2, 1, 0)
            ).tobytes()
            data_arrays.append(('solid_mask', 'Int8', 1, mask_flat))

        xml_lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="ImageData" version="0.1" '
            'byte_order="LittleEndian" header_type="UInt64">',
            f'  <ImageData WholeExtent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}" '
            f'Origin="{ox} {oy} {oz}" '
            f'Spacing="{sx} {sy} {sz}">',
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
            offset += 8 + len(data_bytes)

        xml_lines.extend([
            '      </PointData>',
            '      <CellData/>',
            '    </Piece>',
            '  </ImageData>',
            '  <AppendedData encoding="raw">',
            '   _',
        ])

        # ── Write binary file ────────────────────────────────────
        with open(filepath, 'wb') as f:
            header_text = '\n'.join(xml_lines)
            f.write(header_text.encode('ascii'))

            for name, vtype, ncomp, data_bytes in data_arrays:
                f.write(struct.pack('<Q', len(data_bytes)))
                f.write(data_bytes)

            footer = '\n  </AppendedData>\n</VTKFile>\n'
            f.write(footer.encode('ascii'))

    # =================================================================
    # Private: .vth writer (AMR hierarchy)
    # =================================================================

    def _write_vth(
        self,
        filepath: str,
        vti_relative_paths: List[str],
        step: int,
    ) -> None:
        """Write vtkHierarchicalBoxDataSet (.vth) meta-file.

        File references use relative paths from the .vth location:
            level0/lbm_00005000_level0.vti
            level1/lbm_00005000_level1.vti
        """
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="vtkHierarchicalBoxDataSet" version="1.1" '
            'byte_order="LittleEndian">',
            '  <vtkHierarchicalBoxDataSet origin="0 0 0" '
            'grid_description="XYZ">',
        ]

        for k, (info, rel_path) in enumerate(
            zip(self._level_info, vti_relative_paths)
        ):
            sx, sy, sz = info['spacing']
            bx0, bx1, by0, by1, bz0, bz1 = info['amr_box']

            lines.append(
                f'    <Block level="{k}" spacing="{sx} {sy} {sz}">'
            )
            lines.append(
                f'      <DataSet index="0" '
                f'amr_box="{bx0} {bx1} {by0} {by1} {bz0} {bz1}" '
                f'file="{rel_path}"/>'
            )
            lines.append('    </Block>')

        lines.extend([
            '  </vtkHierarchicalBoxDataSet>',
            '</VTKFile>',
        ])

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

    # =================================================================
    # Utility
    # =================================================================

    def _scan_existing_files(self) -> None:
        """Scan for existing .vth files in vth subdirectory."""
        if not os.path.exists(self._vth_dir):
            return
        pattern = re.compile(r'^lbm_(\d+)\.vth$')
        for filename in os.listdir(self._vth_dir):
            match = pattern.match(filename)
            if match:
                step = int(match.group(1))
                self.time_steps.append((float(step), filename))
        self.time_steps.sort(key=lambda x: x[0])

    def get_info(self) -> str:
        """Return human-readable info."""
        lines = [f"MLGVTKWriter: {self._num_levels} levels"]
        for k, info in enumerate(self._level_info):
            Nx, Ny, Nz = info['shape']
            sx, sy, sz = info['spacing']
            ox, oy, oz = info['origin']
            lines.append(
                f"  Level {k}: {Nx}×{Ny}×{Nz}, "
                f"dx={sx}, origin=({ox},{oy},{oz})"
            )
        return '\n'.join(lines)

    def get_file_size_estimate(self) -> Dict[str, float]:
        """Estimate total output file size per timestep."""
        bytes_per_float = 4 if self._precision == 'float32' else 8
        total = 0
        for info in self._level_info:
            Nx, Ny, Nz = info['shape']
            n = Nx * Ny * Nz
            total += n * bytes_per_float * 5
        return {
            'total_MB': total / 1e6,
            'per_level_MB': [
                info['shape'][0] * info['shape'][1] * info['shape'][2]
                * bytes_per_float * 5 / 1e6
                for info in self._level_info
            ],
        }