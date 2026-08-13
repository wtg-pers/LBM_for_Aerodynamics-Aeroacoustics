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
p_prime_pa / velocity_ms — both level-invariant constants).

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
         "name": "wave"},        # dir/file stem (default plane<i>)
    ]
Unknown keys, a 2D domain, an empty level, or a position outside every
block are hard errors (a requested channel must never silently vanish).

Scope: single-process runs, 3D only. MPI is rejected at setup with the
probe channel (owner-rank sampling is the follow-up step).

Author: LBM Development Team
Date: 2026-08
"""

import glob
import os
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.io.probe_writer import grid_block_views

__all__ = ["parse_plane_config", "PlaneWriterManager"]

_ALLOWED_KEYS = {"normal", "position", "units", "level", "fields",
                 "interval", "flush_every", "buffer_mb", "name"}

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
        planes.append({'normal': normal, 'position': float(pc['position']),
                       'units': units, 'level': level, 'fields': fields,
                       'interval': interval, 'flush_every': flush_every,
                       'buffer_mb': buffer_mb, 'name': name})
    names = [p['name'] for p in planes]
    if len(set(names)) != len(names):
        raise ValueError(f"output.planes: duplicate name(s) in {names}")
    return planes


class _PlaneSeries:
    """One (plane, block) slice series: ring buffers + .vti sink."""

    def __init__(self, plane_name: str, blk: Dict[str, Any],
                 ax: int, node: int, out_dir: str, units: Any,
                 dtype_np: Any) -> None:
        self.sim = blk['sim']
        self.ax = ax
        self.node = node
        tag = (f"L{blk['level']}" if blk['index'] == 0
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

    # ── flush (one D2H + files) ──────────────────────────────────
    def flush(self, steps: List[int]) -> None:
        n = len(steps)
        if n == 0:
            return
        p_host = u_host = None
        if self._buf_p is not None:
            b = self._buf_p[:n]
            p_host = b.get() if hasattr(b, 'get') else np.asarray(b)
        if self._buf_u is not None:
            b = self._buf_u[:n]
            u_host = b.get() if hasattr(b, 'get') else np.asarray(b)
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
                  f"{len(p['series'])} block(s) -> "
                  f"{os.path.join(p['dir'], p['name'] + '.pvd')}")

    # ── private ──────────────────────────────────────────────────
    def _flush_plane(self, p: Dict[str, Any]) -> None:
        steps = p['steps']
        for s in p['series']:
            s.flush(steps)
        for step in steps:
            self._write_vtm(p, step)
        p['written'].update(steps)
        p['cursor'] = 0
        p['steps'] = []
        if steps:
            self._write_pvd(p)

    def _write_vtm(self, p: Dict[str, Any], step: int) -> None:
        """Per-step multiblock collecting every block slice (fine drawn
        over coarse in load order; no blanking — 2D MLG .vtm precedent)."""
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

    def _bind(self, target: Any, append: bool) -> None:
        blocks = grid_block_views(target)
        xp = blocks[0]['sim'].xp
        for pc in self._cfg:
            ax = _AXES[pc['normal']]
            pos = pc['position']
            if pc['units'] == 'm':
                pos = pos / self._uc.dx_phys
            if pc['level'] == 'all':
                cands = list(blocks)
            else:
                cands = [b for b in blocks if b['level'] == pc['level']]
                if not cands:
                    raise ValueError(
                        f"output.planes '{pc['name']}': level "
                        f"{pc['level']} does not exist in this grid")

            # Snap ONCE on the coarsest intersecting level — by nesting
            # that coordinate lies exactly on every finer grid, keeping
            # all block slices strictly coplanar.
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
            node0 = int(round((pos - anchor['origin'][ax])
                              / anchor['spacing']))
            snapped = anchor['origin'][ax] + node0 * anchor['spacing']

            series = []
            for b in cands:
                t = (snapped - b['origin'][ax]) / b['spacing']
                node = int(round(t))
                if abs(t - node) > 1e-6:      # not on this block's grid
                    continue
                if not (0 <= node <= b['shape'][ax] - 1):
                    continue
                out_dir = os.path.join(self._root, pc['name'])
                os.makedirs(out_dir, exist_ok=True)
                series.append(_PlaneSeries(pc['name'], b, ax, node,
                                           out_dir, self._units,
                                           self._dtype))
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
                 'flush_every': eff_flush,
                 'cursor': 0, 'steps': [], 'written': set()}
            if append:
                for s in series:
                    p['written'].update(s.scan_existing())
                # vtm/pvd of prior steps may predate a block-set change;
                # rebuild them from what is actually on disk now.
                for step in sorted(p['written']):
                    self._write_vtm(p, step)
                if p['written']:
                    self._write_pvd(p)
            self._planes.append(p)
            lv_str = ('all' if pc['level'] == 'all' else f"L{pc['level']}")
            clamp = ("" if eff_flush == pc['flush_every'] else
                     f" (clamped from {pc['flush_every']} by "
                     f"buffer_mb={pc['buffer_mb']:g})")
            print(f"  Plane '{pc['name']}': {pc['normal']}={snapped:.2f} "
                  f"L0-lu, levels={lv_str} -> {len(series)} block "
                  f"series, fields={pc['fields']}, every "
                  f"{pc['interval']} step(s)")
            print(f"    disk {row_bytes / 1e6:.2f} MB/sampled-step | "
                  f"buffer {nbytes / 1e6:.1f} MB = {eff_flush} rows"
                  f"{clamp}")
        self._bound = True
