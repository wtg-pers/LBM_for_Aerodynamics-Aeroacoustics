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
        blocks: Optional[List] = None,
    ) -> None:
        self.output_dir = output_dir
        self._num_levels = num_levels
        self._overlap_mgr = overlap_mgr
        self._scaler = scaler
        self._blocks = blocks
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

        # ── Per-BLOCK metadata, level-major ──────────────────────
        # Taken straight from the block tree when one is supplied: setup has
        # already computed every origin, and re-deriving them here was a second
        # source of truth. The legacy chain path keeps the old derivation so
        # callers that predate the tree still work.
        self._block_info: List[Dict] = []
        if blocks is not None:
            for b in blocks:
                Nx_f, Ny_f, Nz_f = b.shape
                lo = [int(round(o / b.spacing)) for o in b.origin]
                self._block_info.append({
                    'level': b.level, 'index': b.index, 'name': b.name,
                    'shape': (Nx_f, Ny_f, Nz_f),
                    'origin': tuple(float(o) for o in b.origin),
                    'spacing': (b.spacing,) * 3,
                    'amr_box': (lo[0], lo[0] + Nx_f - 2,
                                lo[1], lo[1] + Ny_f - 2,
                                lo[2], lo[2] + Nz_f - 2),
                })
        else:
            Nx, Ny, Nz = coarse_shape
            self._block_info.append({
                'level': 0, 'index': 0, 'name': 'L0',
                'shape': (Nx, Ny, Nz),
                'origin': (0.0, 0.0, 0.0),
                'spacing': (1.0, 1.0, 1.0),
                'amr_box': (0, Nx - 2, 0, Ny - 2, 0, Nz - 2),
            })
            for k in range(1, num_levels):
                region = overlap_mgr.get_region(k - 1)
                dx_k = scaler.get_level_units(k).dx
                fdc = region.fine_domain_coarse
                Nx_f, Ny_f, Nz_f = region.fine_shape
                parent = self._block_info[k - 1]
                px, py, pz = parent['origin']
                pdx, pdy, pdz = parent['spacing']
                origin = (px + fdc.x_start * pdx,
                          py + fdc.y_start * pdy,
                          pz + fdc.z_start * pdz)
                lo = [round(origin[d] / dx_k) for d in range(3)]
                self._block_info.append({
                    'level': k, 'index': 0, 'name': f'L{k}',
                    'shape': (Nx_f, Ny_f, Nz_f),
                    'origin': origin,
                    'spacing': (dx_k, dx_k, dx_k),
                    'amr_box': (lo[0], lo[0] + Nx_f - 2,
                                lo[1], lo[1] + Ny_f - 2,
                                lo[2], lo[2] + Nz_f - 2),
                })

        # How many blocks share each level — decides whether filenames need a
        # block suffix (they must NOT get one for chains: byte-identical output)
        self._per_level = [0] * num_levels
        for info in self._block_info:
            self._per_level[info['level']] += 1

        # Back-compat alias (read-only users indexed this by level)
        self._level_info = self._block_info

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

        vti_relative_paths = []      # parallel to the blocks actually written
        written_info = []

        # ── Write per-BLOCK .vti files ───────────────────────────
        # Enumerate the grid's own blocks when it has a tree; otherwise fall
        # back to one grid per level (duck-typed MPI views take this path).
        if hasattr(mlg, 'iter_blocks') and self._blocks is not None:
            _pairs = [(b.sim, info) for b, info
                      in zip(mlg.iter_blocks(), self._block_info)]
        else:
            _pairs = [(mlg.get_level(i['level']), i) for i in self._block_info]

        for level_sim, info in _pairs:
            k = info['level']

            rho = level_sim.rho
            u = level_sim.u
            if rho is None or u is None:
                continue

            # Extract solid mask. Single-GPU passes a Simulation and it hangs
            # off the obstacle BC; the MPI path passes a gathered view that
            # carries `.solid_mask` directly (mpi_output._LevelView). Reading
            # only the first shape silently dropped the array from every MPI
            # VTK, which is what makes the body render as fluid — solid cells
            # hold bounce DEPOSITS, so their macroscopic u is large garbage
            # and there is nothing left to blank it with.
            solid_mask = getattr(level_sim, 'solid_mask', None)
            if solid_mask is None:
                _ob = getattr(level_sim, 'obstacle_bc', None)
                solid_mask = getattr(_ob, 'solid_mask', None)

            # SGS eddy viscosity (allocated only when SGS enabled).
            extras = {}
            if getattr(level_sim, 'nu_t', None) is not None:
                extras['nu_t'] = level_sim.nu_t

            # File in level subdirectory. No block suffix when the level has
            # only one block, so chain runs keep byte-identical filenames.
            _sfx = "" if self._per_level[k] <= 1 else f"_b{info['index']}"
            vti_filename = f"{prefix}_{step:08d}_level{k}{_sfx}.vti"
            vti_filepath = os.path.join(self._level_dirs[k], vti_filename)

            self._write_vti(
                filepath=vti_filepath,
                shape=info['shape'],
                origin=info['origin'],
                spacing=info['spacing'],
                rho=rho,
                u=u,
                solid_mask=solid_mask,
                extras=extras,
            )

            # Relative path from .vth location (vtk/vth/) to .vti (vtk/level{k}/)
            vti_relative_paths.append(f"../level{k}/{vti_filename}")
            written_info.append(info)

        # ── Write .vth AMR meta-file in vth/ subdir ──────────────
        vth_filename = f"{prefix}_{step:08d}.vth"
        vth_filepath = os.path.join(self._vth_dir, vth_filename)
        self._write_vth(vth_filepath, vti_relative_paths, step,
                        written_info)

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
        extras: Optional[dict] = None,
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

        if extras:
            for name, field in extras.items():
                if hasattr(field, 'get'):
                    field_np = field.get()
                else:
                    field_np = np.asarray(field)
                field_flat = np.ascontiguousarray(
                    field_np.transpose(2, 1, 0)
                ).astype(dtype).tobytes()
                data_arrays.append((name, vtk_type, 1, field_flat))

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
        infos: Optional[List[Dict]] = None,
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

        # vtkHierarchicalBoxDataSet groups DataSets under one <Block> per
        # LEVEL, indexed within that level — exactly what a level with several
        # refinement blocks needs. A chain emits one index="0" per Block, so
        # its .vth bytes are unchanged.
        _infos = self._block_info if infos is None else infos
        by_level: Dict[int, List] = {}
        for info, rel_path in zip(_infos, vti_relative_paths):
            by_level.setdefault(info['level'], []).append((info, rel_path))

        for k in sorted(by_level):
            entries = by_level[k]
            sx, sy, sz = entries[0][0]['spacing']
            lines.append(
                f'    <Block level="{k}" spacing="{sx} {sy} {sz}">'
            )
            for j, (info, rel_path) in enumerate(entries):
                bx0, bx1, by0, by1, bz0, bz1 = info['amr_box']
                lines.append(
                    f'      <DataSet index="{j}" '
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
        nb = len(self._block_info)
        lines = [f"MLGVTKWriter: {self._num_levels} levels"
                 + ("" if nb == self._num_levels else f", {nb} blocks")]
        for info in self._block_info:
            Nx, Ny, Nz = info['shape']
            sx, sy, sz = info['spacing']
            ox, oy, oz = info['origin']
            k = info['level']
            tag = (f"Level {k}" if self._per_level[k] <= 1
                   else f"Level {k}.b{info['index']} '{info['name']}'")
            lines.append(
                f"  {tag}: {Nx}×{Ny}×{Nz}, "
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