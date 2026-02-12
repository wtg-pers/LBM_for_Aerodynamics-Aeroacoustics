"""
Marker VTK Writer — VTP (PolyData) output for Actuator Line markers

Writes marker particles as VTK PolyData (.vtp) files, enabling
ParaView visualization of blade tip trajectories, per-marker forces,
angles of attack, and other BEM diagnostics.

File Format:
    VTK XML PolyData (.vtp) with appended binary encoding.
    Each marker is a Vertex with associated point data arrays.

Data Flow:
    ActuatorLineModel._last_positions   → marker XYZ coordinates
    ActuatorLineModel._last_forces_global → force vectors (Fx, Fy, Fz)
    ActuatorLineModel._last_bem_result   → α, φ, CL, CD, u_rel, etc.

ParaView Usage:
    1. Open .vtp files alongside .vti flow field files
    2. Apply "Glyph" filter with arrows to visualize force vectors
    3. Color markers by blade_id, alpha, CL, etc.
    4. Animate to see blade rotation and force evolution

Author: LBM Development Team
Date: 2026-02
"""

import os
import struct
from typing import TYPE_CHECKING, Optional, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class MarkerVTPWriter:
    """Write actuator line marker data as VTK PolyData (.vtp) files

    Each marker particle becomes a VTK Vertex with associated
    scalar and vector point data.

    Supported Data:
        Scalars:
            - blade_id:   Blade index (0, 1, ..., N_b-1)  [dimensionless]
            - marker_id:  Local marker index along blade   [dimensionless]
            - radius:     Radial position from hub         [lu]
            - alpha:      Angle of attack                  [degrees]
            - phi:        Flow angle                       [degrees]
            - CL, CD:     Lift/drag coefficients           [dimensionless]
            - u_rel:      Relative velocity magnitude      [Δx/Δt]
            - Re:         Chord Reynolds number            [dimensionless]
            - F_n, F_theta: Normal/tangential forces       [lattice force]
            - F_L, F_D:   Lift/drag forces                 [lattice force]
            - active:     Marker active flag                [0 or 1]

        Vectors:
            - force:      Global force vector (Fx, Fy, Fz) [lattice force]

    Example:
        >>> writer = MarkerVTPWriter('./results/markers')
        >>> writer.write(
        ...     step=1000,
        ...     positions=al_model._last_positions,
        ...     scalars={'alpha': bem.alpha, 'CL': bem.CL},
        ...     vectors={'force': al_model._last_forces_global},
        ... )
    """

    def __init__(
        self,
        output_dir: str,
        precision: str = 'float32',
    ) -> None:
        """Initialize marker VTP writer

        Args:
            output_dir: Directory for .vtp output files
            precision: 'float32' or 'float64'
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.precision = precision
        self.dtype = np.float32 if precision == 'float32' else np.float64
        self.vtk_type = 'Float32' if precision == 'float32' else 'Float64'

        self.time_steps: List[Tuple[float, str]] = []

    def write(
        self,
        step: int,
        positions: 'npt.NDArray',
        scalars: Optional[Dict[str, 'npt.NDArray']] = None,
        vectors: Optional[Dict[str, 'npt.NDArray']] = None,
        time: Optional[float] = None,
        prefix: str = 'markers',
    ) -> str:
        """Write marker data as VTP PolyData file

        Args:
            step: Time step number
            positions: Marker positions, shape (N_markers, 3)  [lu]
            scalars: Dict of name → array(N_markers,)
            vectors: Dict of name → array(N_markers, 3)
            time: Physical time for PVD collection
            prefix: Filename prefix

        Returns:
            Path to written file
        """
        filename = f"{prefix}_{step:08d}.vtp"
        filepath = os.path.join(self.output_dir, filename)

        n_markers = positions.shape[0]

        # --- Prepare data arrays: (name, vtk_type, ncomp, bytes) ---
        data_arrays: List[Tuple[str, str, int, bytes]] = []

        # Positions (Points array) — always Float32/64
        pos_np = np.ascontiguousarray(
            positions.astype(self.dtype)
        )  # (N, 3)
        pos_bytes = pos_np.tobytes()

        # Connectivity: each marker is a single-vertex cell
        # offsets[i] = i + 1 (each vertex cell has exactly 1 point)
        connectivity = np.arange(n_markers, dtype=np.int64)
        offsets = np.arange(1, n_markers + 1, dtype=np.int64)

        conn_bytes = connectivity.tobytes()
        offset_bytes = offsets.tobytes()

        # Scalar fields
        if scalars is None:
            scalars = {}
        for name, arr in scalars.items():
            arr_np = np.ascontiguousarray(
                np.asarray(arr).astype(self.dtype)
            )
            data_arrays.append((name, self.vtk_type, 1, arr_np.tobytes()))

        # Vector fields
        if vectors is None:
            vectors = {}
        for name, arr in vectors.items():
            arr_np = np.ascontiguousarray(
                np.asarray(arr).astype(self.dtype)
            )  # (N, 3)
            data_arrays.append((name, self.vtk_type, 3, arr_np.tobytes()))

        # --- Build VTP XML ---
        xml_lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="PolyData" version="1.0" '
            'byte_order="LittleEndian" header_type="UInt64">',
            '  <PolyData>',
            f'    <Piece NumberOfPoints="{n_markers}" '
            f'NumberOfVerts="{n_markers}" '
            f'NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">',
        ]

        # --- Points ---
        # Points go first in the appended data
        offset = 0

        xml_lines.append('      <Points>')
        xml_lines.append(
            f'        <DataArray type="{self.vtk_type}" '
            f'NumberOfComponents="3" format="appended" '
            f'offset="{offset}"/>'
        )
        offset += 8 + len(pos_bytes)
        xml_lines.append('      </Points>')

        # --- PointData (scalars + vectors) ---
        xml_lines.append('      <PointData>')
        for name, vtype, ncomp, data_bytes in data_arrays:
            xml_lines.append(
                f'        <DataArray type="{vtype}" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="appended" '
                f'offset="{offset}"/>'
            )
            offset += 8 + len(data_bytes)
        xml_lines.append('      </PointData>')

        # --- Verts (connectivity) ---
        xml_lines.append('      <Verts>')
        xml_lines.append(
            f'        <DataArray type="Int64" Name="connectivity" '
            f'format="appended" offset="{offset}"/>'
        )
        offset += 8 + len(conn_bytes)
        xml_lines.append(
            f'        <DataArray type="Int64" Name="offsets" '
            f'format="appended" offset="{offset}"/>'
        )
        offset += 8 + len(offset_bytes)
        xml_lines.append('      </Verts>')

        # Close tags
        xml_lines.extend([
            '      <Lines/>',
            '      <Strips/>',
            '      <Polys/>',
            '    </Piece>',
            '  </PolyData>',
            '  <AppendedData encoding="raw">',
            '   _',
        ])

        # --- Write binary file ---
        with open(filepath, 'wb') as f:
            header_text = '\n'.join(xml_lines)
            f.write(header_text.encode('ascii'))

            # Points
            f.write(struct.pack('<Q', len(pos_bytes)))
            f.write(pos_bytes)

            # Scalar and vector data
            for name, vtype, ncomp, data_bytes in data_arrays:
                f.write(struct.pack('<Q', len(data_bytes)))
                f.write(data_bytes)

            # Connectivity and offsets
            f.write(struct.pack('<Q', len(conn_bytes)))
            f.write(conn_bytes)
            f.write(struct.pack('<Q', len(offset_bytes)))
            f.write(offset_bytes)

            footer = '\n  </AppendedData>\n</VTKFile>\n'
            f.write(footer.encode('ascii'))

        # Record for PVD
        if time is None:
            time = float(step)
        self.time_steps = [(t, fn) for t, fn in self.time_steps if t != time]
        self.time_steps.append((time, filename))
        self.time_steps.sort(key=lambda x: x[0])

        return filepath

    def write_from_al_model(
        self,
        step: int,
        al_model: 'ActuatorLineModel',
        time: Optional[float] = None,
        prefix: str = 'markers',
    ) -> Optional[str]:
        """Convenience method: extract all data from ActuatorLineModel

        Extracts positions, forces, and BEM diagnostics from the AL model's
        last computed step and writes them as a single VTP file.

        Args:
            step: Time step number
            al_model: ActuatorLineModel with completed step data
            time: Physical time
            prefix: Filename prefix

        Returns:
            Path to written file, or None if no data available
        """
        if al_model._last_positions is None:
            return None

        positions = al_model._last_positions  # (N_total, 3) [lu]
        rotor = al_model.rotor

        # --- Build scalar fields ---
        scalars: Dict[str, np.ndarray] = {}

        # Blade and marker identification
        n_per = rotor.markers_per_blade  # [dimensionless]
        blade_ids = np.repeat(
            np.arange(rotor.n_blades), n_per
        ).astype(np.float32)  # [dimensionless]
        marker_ids = np.tile(
            np.arange(n_per), rotor.n_blades
        ).astype(np.float32)  # [dimensionless]

        scalars['blade_id'] = blade_ids
        scalars['marker_id'] = marker_ids

        # Radial position
        scalars['radius'] = rotor.get_all_marker_radii()  # [lu]

        # Active flag
        scalars['active'] = rotor.get_all_marker_active().astype(
            np.float32
        )  # [0 or 1]

        # BEM results (if available)
        bem = al_model._last_bem_result
        if bem is not None:
            scalars['alpha'] = bem.alpha        # [degrees]
            scalars['phi'] = bem.phi            # [degrees]
            scalars['CL'] = bem.CL              # [dimensionless]
            scalars['CD'] = bem.CD              # [dimensionless]
            scalars['u_rel'] = bem.u_rel        # [Δx/Δt]
            scalars['Re'] = bem.Re              # [dimensionless]
            scalars['F_L'] = bem.F_L            # [lattice force]
            scalars['F_D'] = bem.F_D            # [lattice force]
            scalars['F_n'] = bem.F_n            # [lattice force]
            scalars['F_theta'] = bem.F_theta    # [lattice force]

        # --- Build vector fields ---
        vectors: Dict[str, np.ndarray] = {}

        if al_model._last_forces_global is not None:
            vectors['force'] = al_model._last_forces_global  # (N, 3) [lattice force]

        return self.write(
            step=step,
            positions=positions,
            scalars=scalars,
            vectors=vectors,
            time=time,
            prefix=prefix,
        )

    def write_pvd(self, pvd_filename: str = 'markers.pvd') -> str:
        """Write PVD collection file for time-series animation

        Returns:
            Path to written PVD file
        """
        filepath = os.path.join(self.output_dir, pvd_filename)
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1">',
            '  <Collection>',
        ]
        for time_val, filename in self.time_steps:
            lines.append(
                f'    <DataSet timestep="{time_val}" file="{filename}"/>'
            )
        lines.extend([
            '  </Collection>',
            '</VTKFile>',
        ])
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        return filepath

    def get_info(self) -> str:
        """Return information about the writer"""
        return (
            f"MarkerVTPWriter: {self.output_dir}\n"
            f"  Precision: {self.precision}\n"
            f"  Files written: {len(self.time_steps)}"
        )