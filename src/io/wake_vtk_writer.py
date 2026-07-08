"""Wake filament VTK export for the ALM smearing-correction wake.

Writes the trailed-vortex wake used by the Dağ/Kleine correction as VTP polyline
filaments, one polyline per panel edge per blade, so it can be inspected in
ParaView (color by wake_age or blade_id).

Supported wakes (whichever is active on the model):
  - Kleine free wake  : model._kleine_wake[k].rings   (convected rings)
  - Dağ prescribed helix: model._dag_helix_rings[k]   (analytic helix)

Each "ring" is (n_edge, 3) node positions [lu]; a filament for edge j is the
polyline through rings[0][j], rings[1][j], …, newest→oldest.
"""
import os
import numpy as np


def _collect_filaments(al_model):
    """Return (filaments, blade_ids): list of (L,3) polylines + their blade index."""
    rotor = getattr(al_model, 'rotor', None)
    n_blades = int(getattr(rotor, 'n_blades', 1)) if rotor is not None else 1
    kleine = getattr(al_model, '_kleine_wake', {}) or {}
    helix = getattr(al_model, '_dag_helix_rings', {}) or {}
    filaments, blade_ids = [], []
    for k in range(n_blades):
        rings = None
        kw = kleine.get(k)
        if kw is not None and len(getattr(kw, 'rings', [])) >= 2:
            rings = kw.rings
        elif helix.get(k) is not None and len(helix[k]) >= 2:
            rings = helix[k]
        if rings is None:
            continue
        R = np.asarray(rings, dtype=np.float64)      # (L, n_edge, 3)
        if R.ndim != 3 or R.shape[0] < 2:
            continue
        L, ne, _ = R.shape
        for j in range(ne):
            filaments.append(R[:, j, :])             # (L,3)
            blade_ids.append(k)
    return filaments, blade_ids


def write_wake_vtp(al_model, step, out_dir, prefix='wake', origin=None, spacing=1.0):
    """Write the active correction wake to <out_dir>/<prefix>_<step>.vtp.

    origin/spacing: optional fine-level→global transform (global = origin +
    local·spacing), matching the marker VTK so the wake overlays the flow field.
    Returns the path, or None if no wake geometry is present (e.g. straight-kernel
    Dağ / Kleine, or fewer than 2 rings shed yet).
    """
    filaments, blade_ids = _collect_filaments(al_model)
    if not filaments:
        return None

    pts, conn, offsets, age, bid = [], [], [], [], []
    off = 0
    for fil, k in zip(filaments, blade_ids):
        Lp = fil.shape[0]
        for a in range(Lp):
            pts.append(fil[a]); conn.append(off + a); age.append(a); bid.append(k)
        off += Lp
        offsets.append(off)
    pts = np.asarray(pts, dtype=np.float64)
    if origin is not None:                       # fine local → global coords
        pts = np.asarray(origin, float)[None, :] + pts * float(spacing)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_{int(step):07d}.vtp")
    with open(path, 'w') as f:
        f.write('<?xml version="1.0"?>\n'
                '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">\n'
                ' <PolyData>\n')
        f.write(f'  <Piece NumberOfPoints="{len(pts)}" NumberOfVerts="0" '
                f'NumberOfLines="{len(filaments)}" NumberOfStrips="0" NumberOfPolys="0">\n')
        f.write('   <Points>\n    <DataArray type="Float64" NumberOfComponents="3" '
                'format="ascii">\n     ')
        f.write(' '.join(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" for p in pts))
        f.write('\n    </DataArray>\n   </Points>\n')
        f.write('   <Lines>\n    <DataArray type="Int64" Name="connectivity" format="ascii">\n     ')
        f.write(' '.join(str(c) for c in conn))
        f.write('\n    </DataArray>\n    <DataArray type="Int64" Name="offsets" format="ascii">\n     ')
        f.write(' '.join(str(o) for o in offsets))
        f.write('\n    </DataArray>\n   </Lines>\n')
        f.write('   <PointData Scalars="wake_age">\n')
        f.write('    <DataArray type="Int64" Name="wake_age" format="ascii">\n     ')
        f.write(' '.join(str(a) for a in age))
        f.write('\n    </DataArray>\n')
        f.write('    <DataArray type="Int64" Name="blade_id" format="ascii">\n     ')
        f.write(' '.join(str(b) for b in bid))
        f.write('\n    </DataArray>\n   </PointData>\n')
        f.write('  </Piece>\n </PolyData>\n</VTKFile>\n')
    return path
