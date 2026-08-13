"""
Multi-Level Grid VTK Writer (2D) — MultiBlock Output for ParaView

Uses vtkMultiBlockDataSet (.vtm) rather than vtkHierarchicalBoxDataSet
(.vth) because VTK's AMR format is inherently 3D and silently fails
to open with 2D slab geometries.

Each level is a separate block; ParaView displays them all at their
correct (x, y) locations via the per-VTI origin/spacing. Overlapping
regions between levels draw on top of each other (no AMR blanking);
to see one level at a time, toggle visibility in ParaView.

Directory Structure:
    vtk/
    ├── vtm/
    │   └── lbm_00005000.vtm            ← open this in ParaView
    ├── level0/
    │   └── lbm_00005000_level0.vti
    ├── level1/
    │   └── lbm_00005000_level1.vti
    └── level2/
        └── lbm_00005000_level2.vti

Array convention (2D):
    rho: (Nx, Ny)
    u:   (2, Nx, Ny)
    solid_mask: (Nx, Ny)  (bool)

Author: LBM Development Team
Date: 2026-04
"""

import os
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict

import numpy as np

if TYPE_CHECKING:
    from src.grid.overlap_manager_2d import OverlapManager2D
    from src.grid.level_scaling import LevelScaler


class MLGVTKWriter2D:
    """Multi-level 2D VTK writer with AMR hierarchy support.

    Args:
        output_dir: Root VTK output directory.
        coarse_shape: Level 0 domain dimensions (Nx, Ny).
        overlap_mgr: OverlapManager2D with all level pair geometries.
        scaler: LevelScaler for per-level dx values.
        num_levels: Total number of grid levels.
        precision: Data precision ('float32' or 'float64').
    """

    REFINE_RATIO: int = 2

    def __init__(
        self,
        output_dir: str,
        coarse_shape: Tuple[int, int],
        overlap_mgr: 'OverlapManager2D',
        scaler: 'LevelScaler',
        num_levels: int,
        precision: str = 'float32',
        units: Optional[object] = None,
    ) -> None:
        self.output_dir = output_dir
        self.units = units  # FieldUnits (None = lattice passthrough)
        self._num_levels = num_levels
        self._overlap_mgr = overlap_mgr
        self._scaler = scaler
        self._precision = precision
        self._dtype = np.float32 if precision == 'float32' else np.float64
        self._vtk_type = 'Float32' if precision == 'float32' else 'Float64'

        os.makedirs(output_dir, exist_ok=True)

        # ── Level subdirectories ─────────────────────────────────
        self._level_dirs: List[str] = []
        for k in range(num_levels):
            d = os.path.join(output_dir, f'level{k}')
            os.makedirs(d, exist_ok=True)
            self._level_dirs.append(d)

        self._vtm_dir = os.path.join(output_dir, 'vtm')
        os.makedirs(self._vtm_dir, exist_ok=True)

        # ── Per-level metadata ───────────────────────────────────
        self._level_info: List[Dict] = []

        Nx, Ny = coarse_shape
        self._level_info.append({
            'shape': (Nx, Ny),
            'origin': (0.0, 0.0),
            'spacing': (1.0, 1.0),
            'amr_box': (0, Nx - 2, 0, Ny - 2),
        })

        # Fine levels — origin = parent_origin + fdc_start × parent_spacing
        for k in range(1, num_levels):
            region = overlap_mgr.get_region(k - 1)
            lu = scaler.get_level_units(k)
            dx_k = lu.dx

            fdc = region.fine_domain_coarse
            Nx_f, Ny_f = region.fine_shape

            parent = self._level_info[k - 1]
            px, py = parent['origin']
            pdx, pdy = parent['spacing']

            origin = (
                px + fdc.x_start * pdx,
                py + fdc.y_start * pdy,
            )

            amr_lo_x = round(origin[0] / dx_k)
            amr_lo_y = round(origin[1] / dx_k)
            amr_box = (
                amr_lo_x, amr_lo_x + Nx_f - 2,
                amr_lo_y, amr_lo_y + Ny_f - 2,
            )

            self._level_info.append({
                'shape': (Nx_f, Ny_f),
                'origin': origin,
                'spacing': (dx_k, dx_k),
                'amr_box': amr_box,
            })

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
        if time is None:
            time = float(step)

        vti_relative_paths = []

        for k in range(self._num_levels):
            level_sim = mlg.get_level(k)
            info = self._level_info[k]

            rho = level_sim.rho
            u = level_sim.u
            if rho is None or u is None:
                continue

            solid_mask = None
            if (level_sim.obstacle_bc is not None
                    and hasattr(level_sim.obstacle_bc, 'solid_mask')):
                solid_mask = level_sim.obstacle_bc.solid_mask

            extras = {}
            if getattr(level_sim, 'nu_t', None) is not None:
                extras['nu_t'] = level_sim.nu_t

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
                extras=extras,
            )

            vti_relative_paths.append(f"../level{k}/{vti_filename}")

        vtm_filename = f"{prefix}_{step:08d}.vtm"
        vtm_filepath = os.path.join(self._vtm_dir, vtm_filename)
        self._write_vtm(vtm_filepath, vti_relative_paths, step)

        self.time_steps = [
            (t, fn) for t, fn in self.time_steps if t != time
        ]
        self.time_steps.append((time, vtm_filename))
        self.time_steps.sort(key=lambda x: x[0])

        return vtm_filepath

    # =================================================================
    # Private: .vti writer (per-level, 2D)
    # =================================================================

    def _write_vti(
        self,
        filepath: str,
        shape: Tuple[int, int],
        origin: Tuple[float, float],
        spacing: Tuple[float, float],
        rho: np.ndarray,
        u: np.ndarray,
        solid_mask: Optional[np.ndarray] = None,
        extras: Optional[dict] = None,
    ) -> None:
        """Write a single 2D level's data as VTK ImageData (.vti).

        Array convention: rho (Nx, Ny), u (2, Nx, Ny).
        VTK ImageData is always 3D — we use z-extent = 0 (single slice).
        VTK expects Fortran ordering (x varies fastest) → (Ny, Nx) for rho
        after transpose, (Ny, Nx, 2) for velocity interleaved.
        """
        import struct

        Nx, Ny = shape

        if hasattr(rho, 'get'):
            rho_np = rho.get()
            u_np = u.get()
        else:
            rho_np = np.asarray(rho)
            u_np = np.asarray(u)

        # Field-value unit mapping (output.units; identity when unset).
        # spacing[0] = level spacing in L0 units (nu_t is level-dependent).
        rho_name, u_name = 'density', 'velocity'
        if self.units is not None:
            rho_name, rho_np = self.units.convert(rho_name, rho_np,
                                                  spacing=spacing[0])
            u_name, u_np = self.units.convert(u_name, u_np,
                                              spacing=spacing[0])

        dtype = self._dtype
        vtk_type = self._vtk_type

        # rho shape (Nx, Ny) → Fortran-order (Ny, Nx)
        rho_flat = np.ascontiguousarray(
            rho_np.transpose(1, 0)
        ).astype(dtype).tobytes()

        # u shape (2, Nx, Ny) → (Ny, Nx, 2) with z-component zero-padded
        # ParaView expects 3-component vectors for velocity
        u2 = u_np.transpose(2, 1, 0).astype(dtype)  # (Ny, Nx, 2)
        u3 = np.zeros((Ny, Nx, 3), dtype=dtype)
        u3[..., 0] = u2[..., 0]
        u3[..., 1] = u2[..., 1]
        u_interleaved = np.ascontiguousarray(u3).tobytes()

        ox, oy = origin
        sx, sy = spacing

        data_arrays = [
            (rho_name, vtk_type, 1, rho_flat),
            (u_name, vtk_type, 3, u_interleaved),
        ]

        if solid_mask is not None:
            if hasattr(solid_mask, 'get'):
                mask_np = solid_mask.get()
            else:
                mask_np = np.asarray(solid_mask)
            mask_flat = np.ascontiguousarray(
                mask_np.astype(np.int8).transpose(1, 0)
            ).tobytes()
            data_arrays.append(('solid_mask', 'Int8', 1, mask_flat))

        if extras:
            for name, field in extras.items():
                if hasattr(field, 'get'):
                    field_np = field.get()
                else:
                    field_np = np.asarray(field)
                if self.units is not None:
                    name, field_np = self.units.convert(name, field_np,
                                                        spacing=spacing[0])
                field_flat = np.ascontiguousarray(
                    field_np.transpose(1, 0)
                ).astype(dtype).tobytes()
                data_arrays.append((name, vtk_type, 1, field_flat))

        # VTK 2D = 3D with z-extent 0..0 (single slice)
        whole_extent = f"0 {Nx-1} 0 {Ny-1} 0 0"
        xml_lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="ImageData" version="0.1" '
            'byte_order="LittleEndian" header_type="UInt64">',
            f'  <ImageData WholeExtent="{whole_extent}" '
            f'Origin="{ox} {oy} 0.0" '
            f'Spacing="{sx} {sy} 1.0">',
            f'    <Piece Extent="{whole_extent}">',
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

        with open(filepath, 'wb') as f:
            header_text = '\n'.join(xml_lines)
            f.write(header_text.encode('ascii'))
            for name, vtype, ncomp, data_bytes in data_arrays:
                f.write(struct.pack('<Q', len(data_bytes)))
                f.write(data_bytes)
            footer = '\n  </AppendedData>\n</VTKFile>\n'
            f.write(footer.encode('ascii'))

    # =================================================================
    # Private: .vth writer (AMR hierarchy, 2D)
    # =================================================================

    def _write_vtm(
        self,
        filepath: str,
        vti_relative_paths: List[str],
        step: int,
    ) -> None:
        """Write vtkMultiBlockDataSet (.vtm) meta-file.

        Simple container — each level is a block pointing at its own
        .vti. Level origin/spacing are embedded in each VTI so ParaView
        places them correctly in world coordinates.
        """
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="vtkMultiBlockDataSet" version="1.0" '
            'byte_order="LittleEndian">',
            '  <vtkMultiBlockDataSet>',
        ]

        for k, rel_path in enumerate(vti_relative_paths):
            lines.append(f'    <Block index="{k}" name="Level_{k}">')
            lines.append(
                f'      <DataSet index="0" file="{rel_path}"/>'
            )
            lines.append('    </Block>')

        lines.extend([
            '  </vtkMultiBlockDataSet>',
            '</VTKFile>',
        ])

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

    # =================================================================
    # Utility
    # =================================================================

    def _scan_existing_files(self) -> None:
        if not os.path.exists(self._vtm_dir):
            return
        pattern = re.compile(r'^lbm_(\d+)\.vtm$')
        for filename in os.listdir(self._vtm_dir):
            match = pattern.match(filename)
            if match:
                step = int(match.group(1))
                self.time_steps.append((float(step), filename))
        self.time_steps.sort(key=lambda x: x[0])

    def get_info(self) -> str:
        lines = [f"MLGVTKWriter2D: {self._num_levels} levels"]
        for k, info in enumerate(self._level_info):
            Nx, Ny = info['shape']
            sx, sy = info['spacing']
            ox, oy = info['origin']
            lines.append(
                f"  Level {k}: {Nx}×{Ny}, "
                f"dx={sx}, origin=({ox},{oy})"
            )
        return '\n'.join(lines)

    def get_file_size_estimate(self) -> Dict[str, float]:
        bytes_per_float = 4 if self._precision == 'float32' else 8
        total = 0
        for info in self._level_info:
            Nx, Ny = info['shape']
            n = Nx * Ny
            # rho (1) + velocity (3) + mask (small) ≈ 4 floats/cell
            total += n * bytes_per_float * 4
        return {
            'total_MB': total / 1e6,
            'per_level_MB': [
                info['shape'][0] * info['shape'][1] * bytes_per_float * 4 / 1e6
                for info in self._level_info
            ],
        }
