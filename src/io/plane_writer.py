"""
Plane output channel — axis-aligned slice time series across the full
MLG hierarchy (.vti per block + .vtm per step + one .pvd per plane).

WHY
---
Full-field VTK at every step is I/O-impossible, but ACOUSTIC visualization
(p' wavefronts radiating from a rotor) needs dense-in-time output over a
2D cut, not the volume. This channel writes an axis-aligned plane every
`interval` L0 steps while the volume channel keeps its own coarse cadence.

Same no-stall pattern as the probe channel: every sampled step each
block's slice is copied GPU->GPU into a ring buffer (no host sync); every
`flush_every` rows ONE device->host copy drains the buffers and the files
are written. Unlike probes the payload is a full plane, so the flush does
real disk work — size it via `fields` ("p" only by default), `interval`,
and the per-plane byte estimate printed at bind.

ALL LEVELS BY DEFAULT
---------------------
A plane cuts the whole AMR hierarchy: every block ON EVERY LEVEL whose
extent contains the plane gets its own .vti series (mirrors the volume
.vth output). Per step a .vtm multiblock collects the block slices and
one <name>.pvd indexes the .vtm time series — ParaView loads a single
file and overlays fine over coarse (no blanking, same convention as the
2D MLG .vtm output). `level: k` restricts to one level when wanted.

The position snaps ONCE on the COARSEST intersecting level's grid; by
nesting that coordinate lies exactly on every finer level's grid too, so
all block slices are strictly coplanar. Snapping to a finer grid instead
would leave the coarse levels without a node on the plane.

Coordinates follow the volume writers exactly: origins/spacings in the
global L0-lu frame, so plane files overlay the .vth volume output. Field
VALUES follow output.units via the shared FieldUnits (phys default:
p_pa / velocity_ms — both level-invariant constants).

Config (output.planes — a LIST, several planes allowed):
    "planes": [
        {"normal": "z",          # slice axis: "x" | "y" | "z"
         "position": 172.0,      # along the normal
         "units": "lu",          # "lu" (global L0, default) | "m"
         "level": "all",         # default "all"; int k = that level only
         "fields": ["p"],        # "p" (rho->pressure) and/or "u"
         "interval": 1,          # sample every N L0 steps
         "flush_every": 64,      # rows buffered per flush
         "buffer_mb": 96,        # device ring-buffer cap (clamps rows)
         "name": "wave",         # dir/file stem (default plane<i>)
         "index": "both"},       # "both" | "vth" | "vtm" — which index
    ]                            #   families to write (see _INDEX_MODES;
                                 #   "vth" saves one .vtm/step/plane)
Unknown keys, a 2D domain, an empty level, or a position outside every
block are hard errors (a requested channel must never silently vanish).

Scope: 3D only. Single-process runs use PlaneWriterManager; the MPI
runner uses MPIPlaneWriterManager (owner-rank strips: every rank writes
the .vti pieces of its own slabs, rank 0 indexes them all from the
replicated piece table — zero communication). The probe channel is
still single-process and stays a hard error under MPI.

Author: LBM Development Team
Date: 2026-08
"""

import glob
import json
import os
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.io.probe_writer import grid_block_views

__all__ = ["parse_plane_config", "PlaneWriterManager",
           "MPIPlaneWriterManager"]

_ALLOWED_KEYS = {"normal", "position", "units", "level", "fields",
                 "interval", "flush_every", "buffer_mb", "name", "index"}

# Which index families to write next to the .vti payload:
#   "both" (default)  .vtm/.pvd AND .vth/.vth.series
#   "vth"             AMR view only — at interval=1 this saves one .vtm
#                     per plane per sampled step (octo8-scale: ~10^5
#                     files), losing only the .vtm fallback for steps
#                     whose strip layout no longer matches (rank-count
#                     change on restart, crash-partial steps)
#   "vtm"             multiblock view only (no AMR blanking)
_INDEX_MODES = ("both", "vth", "vtm")

# Device ring-buffer budget per plane [MB]. flush_every is CLAMPED so the
# buffer never exceeds this: a production multi-level plane can reach
# several MB/step, and 64 buffered rows of that silently competing with
# the lattice for GPU memory is exactly the failure mode the init-memory
# audits keep finding. Flushing more often costs nothing extra in total
# transfer — it only shortens the batch.
_DEFAULT_BUFFER_MB = 96.0
_AXES = {"x": 0, "y": 1, "z": 2}


def parse_plane_config(output_cfg: Dict[str, Any],
                       dim: int) -> Optional[List[Dict[str, Any]]]:
    """Validate output.planes -> list of normalized dicts (None = off)."""
    cfg = output_cfg.get('planes')
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        cfg = [cfg]
    if not isinstance(cfg, (list, tuple)) or not cfg:
        raise ValueError(
            "output.planes must be a non-empty list of plane dicts")
    if dim != 3:
        raise ValueError(
            "output.planes needs a 3D domain — a 2D run's full field IS "
            "a plane already (use the volume channel)")

    planes = []
    for i, pc in enumerate(cfg):
        if not isinstance(pc, dict):
            raise ValueError(f"output.planes[{i}] must be a dict")
        unknown = sorted(set(pc) - _ALLOWED_KEYS)
        if unknown:
            raise ValueError(
                f"output.planes[{i}]: unknown key(s) {unknown} "
                f"(allowed: {sorted(_ALLOWED_KEYS)})")
        normal = pc.get('normal')
        if normal not in _AXES:
            raise ValueError(
                f"output.planes[{i}].normal={normal!r}: use 'x'|'y'|'z'")
        if 'position' not in pc:
            raise ValueError(f"output.planes[{i}]: 'position' is required")
        units = pc.get('units', 'lu')
        if units not in ('lu', 'm'):
            raise ValueError(
                f"output.planes[{i}].units={units!r}: 'lu' or 'm'")
        fields = list(pc.get('fields', ['p']))
        bad = [f for f in fields if f not in ('p', 'u')]
        if bad or not fields:
            raise ValueError(
                f"output.planes[{i}].fields={fields}: subset of ['p','u']")
        level = pc.get('level', 'all')
        if level != 'all':
            level = int(level)
            if level < 0:
                raise ValueError(
                    f"output.planes[{i}].level={level}: >= 0 or 'all'")
        interval = int(pc.get('interval', 1))
        flush_every = int(pc.get('flush_every', 64))
        buffer_mb = float(pc.get('buffer_mb', _DEFAULT_BUFFER_MB))
        if interval < 1 or flush_every < 1 or buffer_mb <= 0:
            raise ValueError(
                f"output.planes[{i}]: interval/flush_every >= 1, "
                "buffer_mb > 0")
        name = str(pc.get('name', f'plane{i}'))
        if not re.fullmatch(r'[A-Za-z0-9_\-]+', name):
            raise ValueError(
                f"output.planes[{i}].name={name!r}: [A-Za-z0-9_-] only")
        index = pc.get('index', 'both')
        if index not in _INDEX_MODES:
            raise ValueError(
                f"output.planes[{i}].index={index!r}: use "
                f"{'|'.join(_INDEX_MODES)}")
        planes.append({'normal': normal, 'position': float(pc['position']),
                       'units': units, 'level': level, 'fields': fields,
                       'interval': interval, 'flush_every': flush_every,
                       'buffer_mb': buffer_mb, 'name': name,
                       'index': index})
    names = [p['name'] for p in planes]
    if len(set(names)) != len(names):
        raise ValueError(f"output.planes: duplicate name(s) in {names}")
    return planes


def _snap_position(pc: Dict[str, Any], ax: int, pos: float,
                   blocks: List[Dict[str, Any]]
                   ) -> Tuple[float, List[Dict[str, Any]]]:
    """Snap `pos` onto the coarsest intersecting level's grid.

    Snapping once on the coarsest level keeps every finer level's slice
    strictly coplanar (nesting). `blocks` needs only origin/spacing/
    shape/level per entry — under MPI it is the GLOBAL replicated block
    list, so the snapped coordinate is rank-invariant by construction.

    Returns (snapped position, level-filtered candidate blocks); raises
    the channel's hard errors (empty level / no intersecting extent).
    """
    if pc['level'] == 'all':
        cands = list(blocks)
    else:
        cands = [b for b in blocks if b['level'] == pc['level']]
        if not cands:
            raise ValueError(
                f"output.planes '{pc['name']}': level "
                f"{pc['level']} does not exist in this grid")
    coarse = [b for b in cands
              if (b['origin'][ax] - 1e-9 <= pos
                  <= b['origin'][ax]
                  + (b['shape'][ax] - 1) * b['spacing'] + 1e-9)]
    if not coarse:
        raise ValueError(
            f"output.planes '{pc['name']}': {pc['normal']}="
            f"{pc['position']} ({pc['units']}) intersects no "
            "block extent")
    k_min = min(b['level'] for b in coarse)
    anchor = next(b for b in coarse if b['level'] == k_min)
    node0 = int(round((pos - anchor['origin'][ax]) / anchor['spacing']))
    snapped = anchor['origin'][ax] + node0 * anchor['spacing']
    return snapped, cands


def _block_node(snapped: float, blk: Dict[str, Any],
                ax: int) -> Optional[int]:
    """Node index of the snapped plane on this block's grid, or None
    when the plane is off-grid or outside the block's extent."""
    t = (snapped - blk['origin'][ax]) / blk['spacing']
    node = int(round(t))
    if abs(t - node) > 1e-6:
        return None
    if not (0 <= node <= blk['shape'][ax] - 1):
        return None
    return node


def _solid_slice(sim: Any, ax: int, node: int) -> Optional[np.ndarray]:
    """Host bool slice of this block's solid mask at the plane node.

    None when the block has no obstacle (or no solid cell on the slice).
    Same source as the volume channel (obstacle_bc.solid_mask) so both
    channels overwrite the same cells with the rest state."""
    sm = getattr(getattr(sim, 'obstacle_bc', None), 'solid_mask', None)
    if sm is None:
        return None
    sl = tuple(node if d == ax else slice(None) for d in range(3))
    arr = sm[sl]
    arr = arr.get() if hasattr(arr, 'get') else np.asarray(arr)
    m = np.asarray(arr, dtype=bool)
    return m if m.any() else None


class _PlaneSeries:
    """One (plane, block) slice series: ring buffers + .vti sink."""

    def __init__(self, plane_name: str, blk: Dict[str, Any],
                 ax: int, node: int, out_dir: str, units: Any,
                 dtype_np: Any) -> None:
        self.sim = blk['sim']
        self.ax = ax
        self.node = node
        # 'tag' override: the MPI manager qualifies split strips of one
        # block; unsplit pieces keep the single-process stem so a series
        # stays continuous across single-GPU and MPI runs.
        tag = blk.get('tag') or (f"L{blk['level']}" if blk['index'] == 0
                                 else f"L{blk['level']}b{blk['index']}")
        self.tag = tag
        self.stem = f"{plane_name}_{tag}"
        self.dir = out_dir
        self.units = units          # FieldUnits or None
        self.dtype = dtype_np
        self.level = blk['level']

        shape3 = [blk['shape'][0], blk['shape'][1], blk['shape'][2]]
        shape3[ax] = 1
        self.shape3 = tuple(shape3)          # written extent (plane = 1 node)
        self.slice2 = tuple(n for d, n in enumerate(blk['shape']) if d != ax)
        origin = [float(o) for o in blk['origin']]
        origin[ax] = blk['origin'][ax] + node * blk['spacing']
        self.origin = tuple(origin)          # global L0-lu, overlays .vth
        self.spacing = blk['spacing']

        self._slicer = tuple(node if d == ax else slice(None)
                             for d in range(3))
        self._buf_p: Optional[Any] = None    # (flush, n1, n2)
        self._buf_u: Optional[Any] = None    # (flush, 3, n1, n2)
        # Static solid-cell mask (host bool, slice2 shape) — flush()
        # overwrites those cells with the seeded rest state, the volume
        # channel's convention (mpi_output._mask_solid, 3b3f080): on the
        # esoteric paths solid macros are path-dependent garbage 35x the
        # flow range, and a body-crossing plane would hand ParaView's
        # colour bar the body instead of the flow.
        self.mask2: Optional[np.ndarray] = None

    def filename(self, step: int) -> str:
        return f"{self.stem}_{step:08d}.vti"

    # ── per-step (device only) ───────────────────────────────────
    def row_bytes(self, fields: List[str]) -> int:
        """Device bytes ONE sampled step adds to this series' buffers."""
        n1, n2 = self.slice2
        item = np.dtype(self.dtype).itemsize
        per = n1 * n2 * item
        return per * (('p' in fields) + 3 * ('u' in fields))

    def alloc(self, xp: Any, flush_every: int, fields: List[str]) -> int:
        """Allocate ring buffers; returns the buffer size in bytes."""
        n1, n2 = self.slice2
        nbytes = 0
        if 'p' in fields:
            self._buf_p = xp.empty((flush_every, n1, n2), dtype=self.dtype)
            nbytes += self._buf_p.nbytes
        if 'u' in fields:
            self._buf_u = xp.empty((flush_every, 3, n1, n2),
                                   dtype=self.dtype)
            nbytes += self._buf_u.nbytes
        return nbytes

    def record(self, cursor: int) -> bool:
        """Copy this step's slice into the ring buffer (GPU->GPU)."""
        if self._buf_p is not None:
            rho = self.sim.rho
            if rho is None:
                return False
            self._buf_p[cursor] = rho[self._slicer]
        if self._buf_u is not None:
            u = self.sim.u
            if u is None:
                return False
            self._buf_u[cursor] = u[(slice(None),) + self._slicer]
        return True

    # ── seam columns (MPI split strips, see _sync_seams) ─────────
    def seam_pack(self, n: int, j: int) -> np.ndarray:
        """First in-plane column along split dim j, rows :n (host,
        packed p then u) — the strip's first OWNED column."""
        parts = []
        if self._buf_p is not None:
            b = self._buf_p[:n, 0, :] if j == 0 else self._buf_p[:n, :, 0]
            parts.append((b.get() if hasattr(b, 'get')
                          else np.asarray(b)).ravel())
        if self._buf_u is not None:
            b = (self._buf_u[:n, :, 0, :] if j == 0
                 else self._buf_u[:n, :, :, 0])
            parts.append((b.get() if hasattr(b, 'get')
                          else np.asarray(b)).ravel())
        if not parts:
            return np.empty(0, self.dtype)
        return np.ascontiguousarray(np.concatenate(parts))

    def seam_nbytes(self, n: int, j: int) -> int:
        n2 = self.slice2[1 - j]
        item = np.dtype(self.dtype).itemsize
        vals = n * n2 * ((self._buf_p is not None)
                         + 3 * (self._buf_u is not None))
        return vals * item

    def seam_apply(self, n: int, j: int, data: np.ndarray) -> None:
        """Overwrite the LAST (ext) column along split dim j, rows :n
        with the neighbour strip's packed first column."""
        xp_dev = hasattr(self._buf_p if self._buf_p is not None
                         else self._buf_u, 'get')
        if xp_dev:
            import cupy
        off = 0
        if self._buf_p is not None:
            tgt = (self._buf_p[:n, -1, :] if j == 0
                   else self._buf_p[:n, :, -1])
            src = data[off:off + tgt.size].reshape(tgt.shape)
            tgt[...] = cupy.asarray(src) if xp_dev else src
            off += tgt.size
        if self._buf_u is not None:
            tgt = (self._buf_u[:n, :, -1, :] if j == 0
                   else self._buf_u[:n, :, :, -1])
            src = data[off:off + tgt.size].reshape(tgt.shape)
            tgt[...] = cupy.asarray(src) if xp_dev else src

    # ── flush (one D2H + files) ──────────────────────────────────
    def flush(self, steps: List[int]) -> None:
        n = len(steps)
        if n == 0:
            return
        p_host = u_host = None
        if self._buf_p is not None:
            b = self._buf_p[:n]
            p_host = b.get() if hasattr(b, 'get') else np.asarray(b)
            if self.mask2 is not None:
                p_host[:, self.mask2] = 1.0          # rest-state density
        if self._buf_u is not None:
            b = self._buf_u[:n]
            u_host = b.get() if hasattr(b, 'get') else np.asarray(b)
            if self.mask2 is not None:
                u_host[:, :, self.mask2] = 0.0
        for r, step in enumerate(steps):
            self._write_vti(os.path.join(self.dir, self.filename(step)),
                            None if p_host is None else p_host[r],
                            None if u_host is None else u_host[r])

    def scan_existing(self) -> List[int]:
        """Steps already on disk from a previous (restarted) run."""
        rx = re.compile(re.escape(self.stem) + r'_(\d{8})\.vti$')
        steps = []
        for path in glob.glob(os.path.join(self.dir, f"{self.stem}_*.vti")):
            m = rx.search(os.path.basename(path))
            if m:
                steps.append(int(m.group(1)))
        return steps

    def _write_vti(self, filepath: str, p2: Optional[np.ndarray],
                   u2: Optional[np.ndarray]) -> None:
        """One plane snapshot as ImageData (extent 1 along the normal).

        Same conventions as the volume writers: PointData, appended raw
        binary, Fortran order (x fastest), origin/spacing in L0-lu.
        """
        Nx, Ny, Nz = self.shape3
        vtk_type = ('Float32' if self.dtype == np.float32 else 'Float64')

        data_arrays = []
        if p2 is not None:
            name, arr = 'density', p2
            if self.units is not None:
                name, arr = self.units.convert(name, arr,
                                               spacing=self.spacing)
            a3 = np.expand_dims(np.asarray(arr), self.ax)
            flat = np.ascontiguousarray(
                a3.transpose(2, 1, 0)).astype(self.dtype).tobytes()
            data_arrays.append((name, vtk_type, 1, flat))
        if u2 is not None:
            name, arr = 'velocity', u2
            if self.units is not None:
                name, arr = self.units.convert(name, arr,
                                               spacing=self.spacing)
            a4 = np.expand_dims(np.asarray(arr), self.ax + 1)  # (3,Nx,Ny,Nz)
            inter = np.ascontiguousarray(
                a4.transpose(3, 2, 1, 0)).astype(self.dtype).tobytes()
            data_arrays.append((name, vtk_type, 3, inter))

        ox, oy, oz = self.origin
        s = self.spacing
        xml = ['<?xml version="1.0"?>',
               '<VTKFile type="ImageData" version="0.1" '
               'byte_order="LittleEndian" header_type="UInt64">',
               f'  <ImageData WholeExtent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}" '
               f'Origin="{ox} {oy} {oz}" Spacing="{s} {s} {s}">',
               f'    <Piece Extent="0 {Nx-1} 0 {Ny-1} 0 {Nz-1}">',
               '      <PointData>']
        offset = 0
        for name, vtype, ncomp, data in data_arrays:
            xml.append(f'        <DataArray type="{vtype}" Name="{name}" '
                       f'NumberOfComponents="{ncomp}" format="appended" '
                       f'offset="{offset}"/>')
            offset += 8 + len(data)
        xml += ['      </PointData>', '      <CellData/>', '    </Piece>',
                '  </ImageData>', '  <AppendedData encoding="raw">', '   _']
        with open(filepath, 'wb') as f:
            f.write('\n'.join(xml).encode('ascii'))
            for name, vtype, ncomp, data in data_arrays:
                f.write(struct.pack('<Q', len(data)))
                f.write(data)
            f.write('\n  </AppendedData>\n</VTKFile>\n'.encode('ascii'))



_GDESC = {0: "YZ", 1: "XZ", 2: "XY"}       # plane normal axis -> 2D AMR


def _write_vth_file(path: str, name: str, step: int, ax: int,
                    entries: List[Dict[str, Any]]) -> None:
    """Per-step vtkOverlappingAMR index (.vth) over the plane's pieces.

    THE user-facing composed product: ParaView blanks coarse cells under
    finer boxes automatically (the volume .vth experience), at native
    fine resolution, with zero extra payload — the .vti pieces are the
    same files the .vtm references. Flat (plane-normal) axis follows the
    empirically validated 2D-AMR convention: grid_description in
    {YZ, XZ, XY} and amr_box "0 -1" on the flat axis (vtk 9.3 reader
    generates the blanking ghost array; verified: L1-footprint L0 cells
    hidden). In-plane boxes are GLOBAL cell indices at each level —
    node-aligned origins make them exact integers; split strips carry a
    one-node overlap (ext) so the cell boxes tile without a seam gap.
    """
    by_level: Dict[int, List[Dict[str, Any]]] = {}
    for e in entries:
        by_level.setdefault(e['level'], []).append(e)
    flat_coord = entries[0]['origin'][ax]
    org = [0.0, 0.0, 0.0]
    org[ax] = float(flat_coord)
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="vtkHierarchicalBoxDataSet" version="1.1" '
             'byte_order="LittleEndian">',
             f'  <vtkHierarchicalBoxDataSet '
             f'origin="{org[0]:g} {org[1]:g} {org[2]:g}" '
             f'grid_description="{_GDESC[ax]}">']
    for k in sorted(by_level):
        sp = by_level[k][0]['spacing']
        lines.append(f'    <Block level="{k}" '
                     f'spacing="{sp:g} {sp:g} {sp:g}">')
        for j, e in enumerate(by_level[k]):
            box = []
            for d in range(3):
                if d == ax:
                    box += [0, -1]
                    continue
                i0 = int(round(e['origin'][d] / e['spacing']))
                box += [i0, i0 + int(e['shape3'][d]) - 2]
            fn = f"{name}_{e['tag']}_{step:08d}.vti"
            lines.append(f'      <DataSet index="{j}" '
                         f'amr_box="{box[0]} {box[1]} {box[2]} {box[3]} '
                         f'{box[4]} {box[5]}" file="{fn}"/>')
        lines.append('    </Block>')
    lines += ['  </vtkHierarchicalBoxDataSet>', '</VTKFile>', '']
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def _write_vth_series(p: Dict[str, Any]) -> None:
    """<plane>.vth.series -- the time-series entry point for the AMR
    view. A .pvd collection CANNOT reference AMR datasets (ParaView's
    PVD reader has no vth dispatch -- errors on open, 0814 user
    report), so the composed series uses the .series meta-file
    mechanism instead (ParaView >= 5.5; the trailing .vth.series name
    picks the AMR reader). Replaces the short-lived _amr.pvd, which is
    removed if present so a broken index never lingers."""
    vth_steps = sorted(p.get('written_vth', ()))
    if not vth_steps:
        return
    doc = {"file-series-version": "1.0",
           "files": [{"name": f"{p['name']}_{st:08d}.vth",
                      "time": float(st)} for st in vth_steps]}
    with open(os.path.join(p['dir'], f"{p['name']}.vth.series"),
              'w') as f:
        json.dump(doc, f, indent=1)
    try:
        os.remove(os.path.join(p['dir'], f"{p['name']}_amr.pvd"))
    except OSError:
        pass


def _write_pvd_file(p: Dict[str, Any]) -> None:
    """Time-series indexes for one plane, honouring the `index` knob:
    <plane>.pvd over the .vtm series (skipped for index='vth') and
    <plane>.vth.series (skips itself when no .vth was written)."""
    if p.get('index', 'both') != 'vth':
        lines = ['<?xml version="1.0"?>',
                 '<VTKFile type="Collection" version="0.1" '
                 'byte_order="LittleEndian">',
                 '  <Collection>']
        for step in sorted(p['written']):
            lines.append(f'    <DataSet timestep="{float(step)}" group="" '
                         f'part="0" file="{p["name"]}_{step:08d}.vtm"/>')
        lines += ['  </Collection>', '</VTKFile>', '']
        with open(os.path.join(p['dir'], f"{p['name']}.pvd"), 'w') as f:
            f.write('\n'.join(lines))
    _write_vth_series(p)


def _entry_point(p: Dict[str, Any]) -> str:
    """The file a user opens for this plane's time series (summary line)."""
    ext = ('.vth.series' if p.get('index', 'both') == 'vth' else '.pvd')
    return os.path.join(p['dir'], p['name'] + ext)


def _vti_nodes(path: str) -> Optional[Tuple[int, int, int]]:
    """Node counts from a .vti header (first 2 KB), or None."""
    try:
        with open(path, 'rb') as f:
            head = f.read(2048).decode('ascii', 'replace')
        m = re.search(r'WholeExtent="([^"]+)"', head)
        if not m:
            return None
        we = [int(v) for v in m.group(1).split()]
        return (we[1] - we[0] + 1, we[3] - we[2] + 1, we[5] - we[4] + 1)
    except OSError:
        return None


def _index_hist_vth(p: Dict[str, Any], steps) -> None:
    """Index historical steps as .vth too — only where the archive
    matches the CURRENT piece geometry: every referenced .vti exists
    and one representative header carries the current node counts
    (split strips written before the +1-node ext would tile with a
    seam AND disagree with the amr_box). Non-matching steps keep the
    .vtm-only view — the .vtm rebuild tolerates layout changes, an
    AMR index cannot."""
    geo = p.get('vth_geo')
    if not geo or p.get('index', 'both') == 'vtm':
        return
    probe = next((e for e in geo if re.search(r's\d+$', e['tag'])),
                 geo[0])
    for step in sorted(steps):
        paths = {e['tag']: os.path.join(
            p['dir'], f"{p['name']}_{e['tag']}_{step:08d}.vti")
            for e in geo}
        if not all(os.path.isfile(v) for v in paths.values()):
            continue
        if _vti_nodes(paths[probe['tag']]) != tuple(probe['shape3']):
            continue
        _write_vth_file(
            os.path.join(p['dir'], f"{p['name']}_{step:08d}.vth"),
            p['name'], step, p['ax'], geo)
        p.setdefault('written_vth', set()).add(step)


class PlaneWriterManager:
    """Drives every configured plane: bind -> per-step record -> flush.

    Binding is lazy (first sample) for the same reason as the probes:
    the Simulation/MultiLevelGrid only reaches the OutputManager through
    process(step, sim), and a bad plane spec must die at step 0.
    """

    def __init__(self, cfg: List[Dict[str, Any]], vtk_dir: str,
                 unit_converter: Any, units: Any,
                 precision: str = 'float32') -> None:
        """
        Args:
            cfg: Output of parse_plane_config (non-None).
            vtk_dir: VTK output root; planes go under <vtk_dir>/planes/.
            unit_converter: UnitConverter (for 'm' positions, dt).
            units: FieldUnits shared with the volume writers (or None).
            precision: buffer/file dtype ('float32' | 'float64').
        """
        self._cfg = cfg
        self._root = os.path.join(vtk_dir, 'planes')
        self._uc = unit_converter
        self._units = units
        self._dtype = np.float32 if precision == 'float32' else np.float64
        self._bound = False
        # per plane: {'name', 'dir', 'series': [...], 'interval',
        #             'flush_every', 'cursor', 'steps', 'written': set}
        self._planes: List[Dict[str, Any]] = []

    # ── public ───────────────────────────────────────────────────
    def sample(self, step: int, target: Any) -> None:
        """Record slices due this step; flush any full ring buffer."""
        if not self._bound:
            self._bind(target, append=(step > 0))
        for p in self._planes:
            if step % p['interval'] != 0:
                continue
            ok = all(s.record(p['cursor']) for s in p['series'])
            if not ok:            # rho/u not allocated yet (first steps)
                continue
            p['steps'].append(step)
            p['cursor'] += 1
            if p['cursor'] >= p['flush_every']:
                self._flush_plane(p)

    def finalize(self) -> None:
        if not self._bound:
            return
        for p in self._planes:
            self._flush_plane(p)
            print(f"  Plane '{p['name']}': {len(p['written'])} steps x "
                  f"{len(p['series'])} block(s) -> {_entry_point(p)}")

    # ── private ──────────────────────────────────────────────────
    def _flush_plane(self, p: Dict[str, Any]) -> None:
        steps = p['steps']
        for s in p['series']:
            s.flush(steps)
        for step in steps:
            self._write_vtm(p, step)
            self._write_vth(p, step)
        p['written'].update(steps)
        p['cursor'] = 0
        p['steps'] = []
        if steps:
            self._write_pvd(p)

    def _write_vth(self, p: Dict[str, Any], step: int) -> None:
        geo = p.get('vth_geo')
        if not geo or p.get('index', 'both') == 'vtm':
            return
        _write_vth_file(
            os.path.join(p['dir'], f"{p['name']}_{step:08d}.vth"),
            p['name'], step, p['ax'], geo)
        p.setdefault('written_vth', set()).add(step)

    def _write_vtm(self, p: Dict[str, Any], step: int) -> None:
        """Per-step multiblock collecting every block slice (fine drawn
        over coarse in load order; no blanking — 2D MLG .vtm precedent)."""
        if p.get('index', 'both') == 'vth':
            return
        lines = ['<?xml version="1.0"?>',
                 '<VTKFile type="vtkMultiBlockDataSet" version="1.0" '
                 'byte_order="LittleEndian">',
                 '  <vtkMultiBlockDataSet>']
        for i, s in enumerate(p['series']):
            lines.append(f'    <DataSet index="{i}" name="{s.tag}" '
                         f'file="{s.filename(step)}"/>')
        lines += ['  </vtkMultiBlockDataSet>', '</VTKFile>', '']
        path = os.path.join(p['dir'], f"{p['name']}_{step:08d}.vtm")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def _write_pvd(self, p: Dict[str, Any]) -> None:
        _write_pvd_file(p)

    def _bind(self, target: Any, append: bool) -> None:
        blocks = grid_block_views(target)
        xp = blocks[0]['sim'].xp
        for pc in self._cfg:
            ax = _AXES[pc['normal']]
            pos = pc['position']
            if pc['units'] == 'm':
                pos = pos / self._uc.dx_phys
            # Snap ONCE on the coarsest intersecting level — by nesting
            # that coordinate lies exactly on every finer grid, keeping
            # all block slices strictly coplanar.
            snapped, cands = _snap_position(pc, ax, pos, blocks)

            series = []
            for b in cands:
                node = _block_node(snapped, b, ax)
                if node is None:      # off this block's grid or extent
                    continue
                out_dir = os.path.join(self._root, pc['name'])
                os.makedirs(out_dir, exist_ok=True)
                s = _PlaneSeries(pc['name'], b, ax, node,
                                 out_dir, self._units, self._dtype)
                s.mask2 = _solid_slice(b['sim'], ax, node)
                series.append(s)
            if not series:
                raise ValueError(
                    f"output.planes '{pc['name']}': {pc['normal']}="
                    f"{pc['position']} ({pc['units']}) intersects no "
                    f"block at level={pc['level']}")
            # Clamp flush_every to the device-buffer budget: the total
            # transfer is interval-invariant, so a shorter batch costs
            # nothing except flush frequency — while an unclamped buffer
            # on a production multi-level plane silently steals hundreds
            # of MB from the lattice.
            row_bytes = sum(s.row_bytes(pc['fields']) for s in series)
            cap = max(1, int(pc['buffer_mb'] * 1e6 // max(row_bytes, 1)))
            eff_flush = min(pc['flush_every'], cap)
            nbytes = sum(s.alloc(xp, eff_flush, pc['fields'])
                         for s in series)
            p = {'name': pc['name'], 'dir': series[0].dir,
                 'series': series, 'interval': pc['interval'],
                 'flush_every': eff_flush, 'ax': ax,
                 'index': pc['index'],
                 'vth_geo': [
                     {'level': s_.level, 'tag': s_.tag,
                      'origin': s_.origin, 'spacing': s_.spacing,
                      'shape3': s_.shape3} for s_ in series],
                 'cursor': 0, 'steps': [], 'written': set()}
            if append:
                for s in series:
                    p['written'].update(s.scan_existing())
                if p['written']:
                    # A units switch against the existing slice series is
                    # a hard error (same rule as the volume writers).
                    if self._units is not None:
                        from src.io.field_units import assert_series_units
                        s0 = series[0]
                        assert_series_units(
                            os.path.join(s0.dir, s0.filename(
                                max(p['written']))),
                            self._units.mode)
                    # vtm/pvd of prior steps may predate a block-set
                    # change; rebuild from what is actually on disk now.
                    for step in sorted(p['written']):
                        self._write_vtm(p, step)
                    _index_hist_vth(p, p['written'])
                    self._write_pvd(p)
            self._planes.append(p)
            lv_str = ('all' if pc['level'] == 'all' else f"L{pc['level']}")
            clamp = ("" if eff_flush == pc['flush_every'] else
                     f" (clamped from {pc['flush_every']} by "
                     f"buffer_mb={pc['buffer_mb']:g})")
            print(f"  Plane '{pc['name']}': {pc['normal']}={snapped:.2f} "
                  f"L0-lu, levels={lv_str} -> {len(series)} block "
                  f"series, fields={pc['fields']}, every "
                  f"{pc['interval']} step(s), index={pc['index']}")
            print(f"    disk {row_bytes / 1e6:.2f} MB/sampled-step | "
                  f"buffer {nbytes / 1e6:.1f} MB = {eff_flush} rows"
                  f"{clamp}")
        self._bound = True


# =====================================================================
# MPI (distributed runner) path — owner-rank strips
# =====================================================================
class _SlabView:
    """Owned-region device views of one LocalLevel (ghosts trimmed) —
    the minimal 'sim' duck _PlaneSeries reads (.rho / .u). The views
    are taken once: LocalLevel allocates rho/u a single time and the
    kernel writes into them in place every step."""

    __slots__ = ("rho", "u")

    def __init__(self, rho: Any, u: Any) -> None:
        self.rho = rho
        self.u = u


class MPIPlaneWriterManager:
    """Owner-rank plane sampling for the distributed (MPI) runner.

    The runner's own principle — schedule from the replicated tree,
    work from ownership — applied to this I/O channel:

      * the PIECE TABLE (which strip of which block holds part of a
        plane) is computed on every rank from replicated data only:
        the global block geometry captured at build time plus the
        runner's ownership table. Rank-invariant, zero communication.
      * each rank ring-buffers and writes ONLY its own strips (.vti
        pieces; GPU->GPU per sampled step, one D2H per flush — the
        same no-stall pattern as the single-process channel).
      * the IO rank (0) writes the per-step .vtm index and the .pvd
        from the FULL piece table — including pieces it does not own.
        A fresh .vtm may briefly reference a piece another rank has
        not flushed yet; the set converges at that rank's flush and
        always by finalize().

    Strips: blocks are decomposed 1D along `runner.axis`. A plane
    NORMAL to that axis has exactly ONE owning rank per block, and the
    piece keeps the single-process file stem — the series stays
    continuous across single-GPU and MPI runs (restart included). A
    plane PARALLEL to the axis splits into per-rank strips tagged
    `s<start>`; each .vti carries its strip origin/extent, so the
    pieces overlay seamlessly in ParaView (and an unsplit block keeps
    the plain stem here too).

    The flush cadence is derived from the WHOLE table (the worst
    rank's row bytes against buffer_mb), so every rank flushes at the
    same sampled row — rank-invariant, like every other cadence in
    the MPI pipeline.
    """

    def __init__(self, cfg: List[Dict[str, Any]], vtk_dir: str,
                 unit_converter: Any, units: Any, precision: str,
                 blocks_meta: List[Dict[str, Any]], rank: int,
                 n_ranks: int) -> None:
        """
        Args:
            cfg: Output of parse_plane_config (non-None).
            vtk_dir: VTK output root; planes go under <vtk_dir>/planes/.
            unit_converter: UnitConverter (for 'm' positions).
            units: FieldUnits shared with the volume writers (or None).
            precision: buffer/file dtype ('float32' | 'float64').
            blocks_meta: GLOBAL block list, level-major (entry index ==
                runner uid), dicts with uid/origin/spacing/shape/level/
                index — captured from the replicated build BEFORE it is
                dropped (the runner keeps topology only, no geometry).
            rank, n_ranks: this rank / world size.
        """
        self._cfg = cfg
        self._root = os.path.join(vtk_dir, 'planes')
        self._uc = unit_converter
        self._units = units
        self._dtype = np.float32 if precision == 'float32' else np.float64
        self._meta = list(blocks_meta)
        self._rank = int(rank)
        self._nr = int(n_ranks)
        self._bound = False
        self._planes: List[Dict[str, Any]] = []
        self._mask2: Dict[Tuple[str, int], np.ndarray] = {}

    def capture_solid_masks(self, mlg: Any) -> None:
        """Solid-mask plane slices, captured while the replicated build
        still exists (the runner drops the obstacle arrays; only tiny
        link tables survive). Host bool, (plane name, uid) keyed; the
        bind cuts each rank's strip out of them. Every rank captures
        identically — the masks are static and replicated."""
        blocks = grid_block_views(mlg)
        for pc in self._cfg:
            ax = _AXES[pc['normal']]
            pos = pc['position']
            if pc['units'] == 'm':
                pos = pos / self._uc.dx_phys
            snapped, _ = _snap_position(pc, ax, pos, self._meta)
            for m in self._meta:
                node = _block_node(snapped, m, ax)
                if node is None:
                    continue
                mk = _solid_slice(blocks[m['uid']]['sim'], ax, node)
                if mk is not None:
                    self._mask2[(pc['name'], m['uid'])] = mk

    # ── public (same surface as PlaneWriterManager) ──────────────
    def sample(self, step: int, target: Any = None) -> None:
        """Record strips due this step; flush any full ring buffer.

        `target` is the DistributedMLGRunner (the MPI loop's process()
        argument). Binding is lazy for the same reason as the
        single-process channel: a bad spec must die at step 0.
        """
        if not self._bound:
            self._bind(target, append=(step > 0))
        for p in self._planes:
            if step % p['interval'] != 0:
                continue
            for s in p['series']:
                if not s.record(p['cursor']):
                    # Slab rho/u are allocated with the level and written
                    # in place every step — a miss here is a wiring bug,
                    # and skipping would silently desync the step lists
                    # across ranks (vtm would index missing pieces).
                    raise RuntimeError(
                        f"plane '{p['name']}': slab macro view missing "
                        f"at step {step} (rank {self._rank})")
            p['steps'].append(step)
            p['cursor'] += 1
            if p['cursor'] >= p['flush_every']:
                self._flush_plane(p)

    def finalize(self) -> None:
        if not self._bound:
            return
        for p in self._planes:
            self._flush_plane(p)
            if self._rank == 0:
                print(f"  Plane '{p['name']}': {len(p['written'])} steps "
                      f"x {len(p['pieces'])} piece(s) -> {_entry_point(p)}")

    # ── private ──────────────────────────────────────────────────
    def _flush_plane(self, p: Dict[str, Any]) -> None:
        steps = p['steps']
        self._sync_seams(p, len(steps))
        for s in p['series']:
            s.flush(steps)
        if self._rank == 0:
            for step in steps:
                self._write_vtm(p, step)
                self._write_vth(p, step)
            p['written'].update(steps)
            if steps:
                self._write_pvd(p)
        p['cursor'] = 0
        p['steps'] = []

    def _write_vth(self, p: Dict[str, Any], step: int) -> None:
        geo = p.get('vth_geo')
        if not geo or p.get('index', 'both') == 'vtm':
            return
        _write_vth_file(
            os.path.join(p['dir'], f"{p['name']}_{step:08d}.vth"),
            p['name'], step, p['ax'], geo)
        p.setdefault('written_vth', set()).add(step)

    def _sync_seams(self, p: Dict[str, Any], n: int) -> None:
        """Make each ext column the NEXT strip's first OWNED column.

        The ext (+1 node) column is recorded from this rank's ghost
        mirror, which is bit-identical on plain bulk rows but goes
        STALE on coupling-band rows (measured: L1 top C2F row -- the
        owner's C2F scatter is not mirrored into neighbours' ghosts
        before sampling). At flush the true column is sitting in the
        next strip owner's ring buffer as its piece's first owned
        column, so a single Sendrecv per seam per flush replaces the
        ghost copy with the owner's -- bit-identical by construction,
        the same flush-time-only communication contract as the probe
        channel's rank-0 assembly. The schedule is computed on every
        rank from the replicated piece table (rank-invariant)."""
        if n == 0 or self._comm is None or p.get('seam_j', -1) < 0:
            return
        j = p['seam_j']
        reqs, apply_q = [], []
        for k, pe in enumerate(p['pieces']):
            if not pe.get('ext'):
                continue
            nxt = next(q for q in p['pieces']
                       if q['uid'] == pe['uid']
                       and q['start'] == pe['start'] + pe['count'])
            tag = 700 + k
            if self._rank == pe['rank']:
                sr = pe['_series']
                buf = np.empty(sr.seam_nbytes(n, j) //
                               np.dtype(sr.dtype).itemsize, sr.dtype)
                reqs.append(self._comm.Irecv(buf, source=nxt['rank'],
                                             tag=tag))
                apply_q.append((sr, buf))
            if self._rank == nxt['rank']:
                col = nxt['_series'].seam_pack(n, j)
                reqs.append(self._comm.Isend(col, dest=pe['rank'],
                                             tag=tag))
        if reqs:
            from mpi4py import MPI as _MPI
            _MPI.Request.Waitall(reqs)
        for sr, buf in apply_q:
            sr.seam_apply(n, j, buf)

    def _write_vtm(self, p: Dict[str, Any], step: int) -> None:
        """Per-step multiblock over ALL ranks' pieces (piece-table
        order = (uid, rank), fine drawn over coarse in load order)."""
        if p.get('index', 'both') == 'vth':
            return
        lines = ['<?xml version="1.0"?>',
                 '<VTKFile type="vtkMultiBlockDataSet" version="1.0" '
                 'byte_order="LittleEndian">',
                 '  <vtkMultiBlockDataSet>']
        for i, pe in enumerate(p['pieces']):
            fname = f"{p['name']}_{pe['tag']}_{step:08d}.vti"
            lines.append(f'    <DataSet index="{i}" name="{pe["tag"]}" '
                         f'file="{fname}"/>')
        lines += ['  </vtkMultiBlockDataSet>', '</VTKFile>', '']
        path = os.path.join(p['dir'], f"{p['name']}_{step:08d}.vtm")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def _write_vtm_from_files(self, p: Dict[str, Any], step: int,
                              basenames: List[str]) -> None:
        """Historical .vtm rebuilt from what is ON DISK for `step` — a
        restart may follow a run with a different rank count, whose
        strip split (and so piece names) differs from the current
        table. Those steps keep their own pieces."""
        if p.get('index', 'both') == 'vth':
            return
        lines = ['<?xml version="1.0"?>',
                 '<VTKFile type="vtkMultiBlockDataSet" version="1.0" '
                 'byte_order="LittleEndian">',
                 '  <vtkMultiBlockDataSet>']
        for i, fname in enumerate(sorted(basenames)):
            tag = fname[len(p['name']) + 1:-13]   # drop "name_" and "_<8d>.vti"
            lines.append(f'    <DataSet index="{i}" name="{tag}" '
                         f'file="{fname}"/>')
        lines += ['  </vtkMultiBlockDataSet>', '</VTKFile>', '']
        path = os.path.join(p['dir'], f"{p['name']}_{step:08d}.vtm")
        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    def _write_pvd(self, p: Dict[str, Any]) -> None:
        _write_pvd_file(p)

    def _piece_table(self, pc: Dict[str, Any], ax: int, snapped: float,
                     runner: Any) -> List[Dict[str, Any]]:
        """Every (block, rank) strip holding part of this plane, in
        (uid, rank) order — identical on every rank by construction."""
        if pc['level'] == 'all':
            cands = self._meta
        else:
            cands = [m for m in self._meta if m['level'] == pc['level']]
        axis = runner.axis
        pieces = []
        for m in cands:
            node = _block_node(snapped, m, ax)
            if node is None:
                continue
            owners = [(r, s, c)
                      for r, (s, c) in enumerate(runner.range_table[m['uid']])
                      if c > 0]
            if ax == axis:                     # single owner: node's rank
                owners = [(r, s, c) for r, s, c in owners
                          if s <= node < s + c]
            base = (f"L{m['level']}" if m['index'] == 0
                    else f"L{m['level']}b{m['index']}")
            split = len(owners) > 1
            ext_n = m['shape'][axis]
            for r, s, c in owners:
                # split strips: +1 node on the high side (except the
                # block end) so adjacent pieces SHARE the boundary node
                # -> AMR cell boxes tile with no seam gap. The extra
                # node is the neighbour's first owned node, read from
                # this rank's halo-synced ghost column (bit-identical
                # mirror).
                ext = 1 if (split and s + c < ext_n) else 0
                pieces.append({'uid': m['uid'], 'meta': m, 'rank': r,
                               'start': s, 'count': c, 'ext': ext,
                               'node': node,
                               'tag': base + (f"s{s}" if split else "")})
        return pieces

    def _piece_row_bytes(self, pe: Dict[str, Any], fields: List[str],
                         ax: int, axis: int) -> int:
        shape = list(pe['meta']['shape'])
        shape[axis] = pe['count'] + pe.get('ext', 0)
        n1, n2 = [n for d, n in enumerate(shape) if d != ax]
        item = np.dtype(self._dtype).itemsize
        return n1 * n2 * item * (('p' in fields) + 3 * ('u' in fields))

    def _existing_steps(self, name: str,
                        out_dir: str) -> Dict[int, List[str]]:
        """step -> .vti basenames already on disk for this plane (any
        strip layout — tags are constrained to L\\d+(b\\d+)?(s\\d+)?)."""
        rx = re.compile(re.escape(name)
                        + r'_L\d+(?:b\d+)?(?:s\d+)?_(\d{8})\.vti$')
        found: Dict[int, List[str]] = {}
        for path in glob.glob(os.path.join(out_dir, f"{name}_*.vti")):
            base = os.path.basename(path)
            mm = rx.fullmatch(base)
            if mm:
                found.setdefault(int(mm.group(1)), []).append(base)
        return found

    def _bind(self, runner: Any, append: bool) -> None:
        if runner is None or not hasattr(runner, 'range_table'):
            raise ValueError(
                "MPIPlaneWriterManager.sample needs the distributed "
                f"runner as target, got {type(runner).__name__}")
        axis = runner.axis
        self._comm = getattr(runner, '_comm', None)
        xp = None
        for pc in self._cfg:
            ax = _AXES[pc['normal']]
            pos = pc['position']
            if pc['units'] == 'm':
                pos = pos / self._uc.dx_phys
            snapped, _ = _snap_position(pc, ax, pos, self._meta)
            pieces = self._piece_table(pc, ax, snapped, runner)
            if not pieces:
                raise ValueError(
                    f"output.planes '{pc['name']}': {pc['normal']}="
                    f"{pc['position']} ({pc['units']}) intersects no "
                    f"block at level={pc['level']}")

            out_dir = os.path.join(self._root, pc['name'])
            os.makedirs(out_dir, exist_ok=True)

            # ── my series (owned strips only) ──
            series = []
            for pe in pieces:
                if pe['rank'] != self._rank:
                    continue
                uid = pe['uid']
                if not runner.owns[uid]:       # table/owns must agree
                    raise RuntimeError(
                        f"plane '{pc['name']}': piece table says rank "
                        f"{self._rank} owns block uid {uid}, runner "
                        "disagrees")
                L = runner.lv[uid]
                part = runner.parts[uid]
                own = list(part.owned_local())
                if pe.get('ext'):
                    _a = own[axis]
                    own[axis] = slice(_a.start, _a.stop + pe['ext'])
                own = tuple(own)
                view = _SlabView(L.rho[own], L.u[(slice(None),) + own])
                if xp is None:
                    import cupy
                    xp = cupy.get_array_module(L.rho)
                m = pe['meta']
                shape = list(m['shape'])
                shape[axis] = pe['count'] + pe.get('ext', 0)
                origin = [float(v) for v in m['origin']]
                origin[axis] += pe['start'] * m['spacing']
                node_local = (pe['node'] - pe['start'] if ax == axis
                              else pe['node'])
                blk = {'sim': view, 'origin': tuple(origin),
                       'spacing': m['spacing'], 'shape': tuple(shape),
                       'level': m['level'], 'index': m['index'],
                       'tag': pe['tag']}
                s = _PlaneSeries(pc['name'], blk, ax, node_local,
                                 out_dir, self._units, self._dtype)
                pe['_series'] = s
                mask2 = self._mask2.get((pc['name'], uid))
                if mask2 is not None and ax != axis:
                    # cut this rank's strip out of the full-block slice
                    # (the strip splits the in-plane dim = decomp axis)
                    j = [d for d in range(3) if d != ax].index(axis)
                    sl = [slice(None), slice(None)]
                    sl[j] = slice(pe['start'], pe['start'] + pe['count']
                                  + pe.get('ext', 0))
                    mask2 = mask2[tuple(sl)]
                    mask2 = mask2 if mask2.any() else None
                s.mask2 = mask2
                series.append(s)

            # ── rank-invariant flush cadence (worst rank's rows) ──
            per_rank: Dict[int, int] = {}
            for pe in pieces:
                per_rank[pe['rank']] = (per_rank.get(pe['rank'], 0)
                                        + self._piece_row_bytes(
                                            pe, pc['fields'], ax, axis))
            cap = min(max(1, int(pc['buffer_mb'] * 1e6 // rb))
                      for rb in per_rank.values())
            eff_flush = min(pc['flush_every'], cap)
            nbytes = sum(s.alloc(xp, eff_flush, pc['fields'])
                         for s in series) if series else 0

            vgeo = []
            for pe in pieces:
                m = pe['meta']
                sh = list(m['shape'])
                sh[axis] = pe['count'] + pe.get('ext', 0)
                sh[ax] = 1
                og = [float(v) for v in m['origin']]
                og[axis] += pe['start'] * m['spacing']
                og[ax] = float(snapped)
                vgeo.append({'level': m['level'], 'tag': pe['tag'],
                             'origin': tuple(og),
                             'spacing': float(m['spacing']),
                             'shape3': tuple(int(v) for v in sh)})
            p = {'name': pc['name'], 'dir': out_dir, 'series': series,
                 'pieces': pieces, 'interval': pc['interval'],
                 'flush_every': eff_flush, 'ax': ax, 'vth_geo': vgeo,
                 'index': pc['index'],
                 'seam_j': ([d for d in range(3) if d != ax].index(axis)
                            if ax != axis else -1),
                 'cursor': 0, 'steps': [], 'written': set()}
            if append and self._rank == 0:
                hist = self._existing_steps(pc['name'], out_dir)
                if hist:
                    if self._units is not None:
                        from src.io.field_units import assert_series_units
                        last = max(hist)
                        assert_series_units(
                            os.path.join(out_dir, sorted(hist[last])[0]),
                            self._units.mode)
                    # vtm/pvd of prior steps may predate a strip-layout
                    # or block-set change; rebuild from the disk state.
                    for step, files in sorted(hist.items()):
                        self._write_vtm_from_files(p, step, files)
                    p['written'].update(hist)
                    _index_hist_vth(p, hist)
                    self._write_pvd(p)
            self._planes.append(p)

            if self._rank == 0:
                mine = len(series)
                total_mb = sum(per_rank.values()) / 1e6
                lv_str = ('all' if pc['level'] == 'all'
                          else f"L{pc['level']}")
                clamp = ("" if eff_flush == pc['flush_every'] else
                         f" (clamped from {pc['flush_every']} by "
                         f"buffer_mb={pc['buffer_mb']:g}, worst rank)")
                print(f"  Plane '{pc['name']}': {pc['normal']}="
                      f"{snapped:.2f} L0-lu, levels={lv_str} -> "
                      f"{len(pieces)} piece(s) over {self._nr} rank(s) "
                      f"(rank0 owns {mine}), fields={pc['fields']}, "
                      f"every {pc['interval']} step(s), "
                      f"index={pc['index']}")
                print(f"    disk {total_mb:.2f} MB/sampled-step | "
                      f"rank0 buffer {nbytes / 1e6:.1f} MB = "
                      f"{eff_flush} rows{clamp}")
        self._bound = True
