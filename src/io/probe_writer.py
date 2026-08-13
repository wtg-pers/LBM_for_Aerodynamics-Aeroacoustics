"""
Pressure probe channel — fixed-point p(t) time series (virtual microphones).

WHY
---
Acoustic post-processing (p' spectra, audible rendering) needs p at a few
FIXED locations at the FULL L0 time-step cadence. The VTK channel cannot
serve this: full-field files at every step are I/O-impossible, and the
usual output_interval (~10^3-10^4 steps) aliases every acoustic frequency
of interest. This channel samples single nodes every step into a small
device-side buffer and appends to CSV only every `flush_every` rows, so
the per-step cost is one tiny gather kernel — no GPU->CPU sync, no stall
(same reasoning as the nan_trap default-OFF note in setup).

WHERE A PROBE LIVES
-------------------
Nearest node on the FINEST block whose extent contains the point. The
coordinate frame is the global L0 lattice frame — identical to the frame
the MLG .vti/.vth files declare (block.origin + i * block.spacing), so a
position picked in ParaView can be pasted into the config verbatim.
`units: "m"` divides by dx_phys first for physically-specified mics.

Sampling is at the L0 cadence (once per coarse step, after the full MLG
cycle), giving a uniform series with fs = 1 / (interval * dt_phys(L0)).

UNITS
-----
The CSV stores p in Pa relative to the lattice reference density rho0 = 1:
    p_pa = cs_lu^2 * (rho_lu - 1) * rho_phys * (dx_phys / dt_phys)^2
p' is deliberately NOT formed here: subtract the per-probe time mean in
post (ParaView Calculator / numpy), where the averaging window is a
post-processing choice, not a solver one.

Config (output.probes):
    "probes": {
        "points": [[x, y, z], ...],   # dim-length each
        "units": "lu",                # "lu" (L0 lattice, default) | "m"
        "interval": 1,                # sample every N L0 steps
        "flush_every": 2048,          # rows buffered per CSV append
    }
Unknown keys and malformed points are hard errors: a typo must never
silently disable a requested output channel.

Scope: single-process runs (single-grid + MLG, 2D/3D). MPI is rejected at
setup — under the distributed runner the probe's owner rank and its local
index space need the partition map (ownership rule: sample on the owning
rank), which is a follow-up step.

Author: LBM Development Team
Date: 2026-08
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["parse_probe_config", "PressureProbeManager", "grid_block_views"]


def grid_block_views(target: Any) -> List[Dict[str, Any]]:
    """Uniform (sim, origin, spacing, shape, level, name) views of a target.

    MultiLevelGrid exposes the block tree; a plain Simulation becomes the
    single L0 pseudo-block. origin/spacing are the same values the .vti
    files declare, so sampling frames match ParaView exactly. Shared by
    the probe and plane channels.
    """
    if hasattr(target, 'iter_blocks'):
        return [{'sim': b.sim, 'origin': b.origin, 'spacing': b.spacing,
                 'shape': b.shape, 'level': b.level, 'index': b.index,
                 'name': b.label}
                for b in target.iter_blocks()]
    return [{'sim': target, 'origin': (0.0, 0.0, 0.0), 'spacing': 1.0,
             'shape': target.domain_shape, 'level': 0, 'index': 0,
             'name': 'L0'}]

_ALLOWED_KEYS = {"enabled", "points", "units", "interval", "flush_every"}

# Fine-block faces are C2F coupling bands (interpolated, not computed —
# see feedback on MLG region padding); a probe this close to one reads
# coupling artifacts, so warn.
_EDGE_GUARD = 4


def parse_probe_config(output_cfg: Dict[str, Any],
                       dim: int) -> Optional[Dict[str, Any]]:
    """Validate output.probes -> normalized dict, or None (channel off).

    Args:
        output_cfg: The config's `output` block.
        dim: Domain dimension (2 or 3) — each point must have dim coords.

    Returns:
        {'points', 'units', 'interval', 'flush_every'} or None.
    """
    cfg = output_cfg.get('probes')
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise ValueError(
            "output.probes must be a dict: "
            "{'points': [[x, y, z], ...], 'units': 'lu'|'m', ...}")
    unknown = sorted(set(cfg) - _ALLOWED_KEYS)
    if unknown:
        raise ValueError(
            f"output.probes: unknown key(s) {unknown} "
            f"(allowed: {sorted(_ALLOWED_KEYS)})")
    if not cfg.get('enabled', True):
        return None

    pts = cfg.get('points')
    if not pts:
        raise ValueError(
            "output.probes.points is required and must be non-empty")
    points: List[Tuple[float, ...]] = []
    for i, p in enumerate(pts):
        q = tuple(float(c) for c in p)
        if len(q) != dim:
            raise ValueError(
                f"output.probes.points[{i}] has {len(q)} coordinates, "
                f"but the domain is {dim}D")
        points.append(q)

    units = cfg.get('units', 'lu')
    if units not in ('lu', 'm'):
        raise ValueError(
            f"output.probes.units={units!r}: use 'lu' (global L0 lattice, "
            "the ParaView frame) or 'm' (physical)")

    interval = int(cfg.get('interval', 1))
    flush_every = int(cfg.get('flush_every', 2048))
    if interval < 1:
        raise ValueError(f"output.probes.interval={interval}: must be >= 1")
    if flush_every < 1:
        raise ValueError(
            f"output.probes.flush_every={flush_every}: must be >= 1")

    return {'points': points, 'units': units,
            'interval': interval, 'flush_every': flush_every}


class PressureProbeManager:
    """Samples rho at fixed nodes every `interval` L0 steps into a device
    buffer; flushes p [Pa] rows to <csv_dir>/probes.csv per `flush_every`.

    Binding (point -> block/node resolution) is lazy — it happens on the
    first sample() call, because the Simulation/MultiLevelGrid object only
    reaches the OutputManager through process(step, sim). A bad probe
    coordinate therefore raises at step 0, before any long run time.
    """

    def __init__(self, cfg: Dict[str, Any], csv_dir: str,
                 unit_converter: Any, dim: int) -> None:
        """
        Args:
            cfg: Output of parse_probe_config (non-None).
            csv_dir: Directory for probes.csv / probes_meta.csv.
            unit_converter: UnitConverter (dx_phys, dt_phys, cs, rho_phys).
            dim: Domain dimension (2 or 3).
        """
        self._points_in = cfg['points']
        self._units: str = cfg['units']
        self.interval: int = cfg['interval']
        self._flush_every: int = cfg['flush_every']
        self._uc = unit_converter
        self._dim = dim

        self.csv_path = os.path.join(csv_dir, 'probes.csv')
        self.meta_path = os.path.join(csv_dir, 'probes_meta.csv')

        dxdt = unit_converter.dx_phys / unit_converter.dt_phys
        # [Pa] per unit lattice density deviation (rho0 = 1)
        self._p_conv: float = (unit_converter.cs ** 2
                               * unit_converter.rho_phys * dxdt * dxdt)
        self._rho0: float = 1.0
        self.fs_hz: float = 1.0 / (self.interval * unit_converter.dt_phys)

        self._bound = False
        self._groups: List[Dict[str, Any]] = []
        self._buf: Optional[Any] = None
        self._cursor: int = 0
        self._steps: List[int] = []
        self._rows_written: int = 0

    # =================================================================
    # Public: per-step channel
    # =================================================================

    def sample(self, step: int, target: Any) -> None:
        """Record one row (rho at every probe) into the device buffer.

        Args:
            step: Current L0 step index.
            target: Simulation or MultiLevelGrid (post-advance).
        """
        if step % self.interval != 0:
            return
        if not self._bound:
            # First sampled step > 0 == restarted run -> append (rotor CSV
            # follows the same restart-preserving convention).
            self._bind(target, append=(step > 0))
        if any(g['sim'].rho is None for g in self._groups):
            return
        row = self._cursor
        for g in self._groups:
            self._buf[row, g['cols']] = g['sim'].rho[g['idx']]
        self._steps.append(step)
        self._cursor += 1
        if self._cursor >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        """Append buffered rows to CSV (one D2H copy)."""
        if self._cursor == 0:
            return
        vals = self._buf[:self._cursor]
        vals = vals.get() if hasattr(vals, 'get') else np.asarray(vals)
        p_pa = (vals - self._rho0) * self._p_conv
        steps = np.asarray(self._steps, dtype=np.float64)
        t_s = steps * self._uc.dt_phys
        out = np.column_stack([steps, t_s, p_pa])
        n_p = p_pa.shape[1]
        with open(self.csv_path, 'a') as f:
            np.savetxt(f, out, delimiter=',',
                       fmt=['%d', '%.9e'] + ['%.9e'] * n_p)
        self._rows_written += self._cursor
        self._cursor = 0
        self._steps = []

    def finalize(self) -> None:
        """Flush the tail and report the channel summary."""
        if not self._bound:
            return
        self.flush()
        print(f"  Probes: {self._rows_written} rows -> {self.csv_path} "
              f"(fs = {self.fs_hz:.1f} Hz)")

    # =================================================================
    # Private: binding
    # =================================================================

    def _bind(self, target: Any, append: bool) -> None:
        """Resolve every probe to (block, node index); open the CSVs."""
        uc, dim = self._uc, self._dim
        if self._units == 'm':
            points_lu = [tuple(c / uc.dx_phys for c in p)
                         for p in self._points_in]
        else:
            points_lu = [p for p in self._points_in]

        blocks = grid_block_views(target)
        records = []
        for pid, p in enumerate(points_lu):
            rec = self._locate(pid, p, blocks)
            records.append(rec)

        # Group per Simulation for one vectorized gather per block.
        by_sim: Dict[int, Dict[str, Any]] = {}
        xp = records[0]['sim'].xp
        for rec in records:
            g = by_sim.setdefault(id(rec['sim']), {
                'sim': rec['sim'], 'cols_h': [], 'idx_h': []})
            g['cols_h'].append(rec['pid'])
            g['idx_h'].append(rec['idx'])
        self._groups = []
        for g in by_sim.values():
            idx_arr = np.asarray(g['idx_h'], dtype=np.int64)  # (n, dim)
            self._groups.append({
                'sim': g['sim'],
                'cols': xp.asarray(np.asarray(g['cols_h'], dtype=np.int64)),
                'idx': tuple(xp.asarray(idx_arr[:, d]) for d in range(dim)),
            })
        self._buf = xp.empty((self._flush_every, len(records)),
                             dtype=xp.float64)

        self._write_meta(records)
        if not append or not os.path.exists(self.csv_path):
            header = 'step,time_s,' + ','.join(
                f'p{r["pid"]}_pa' for r in records)
            with open(self.csv_path, 'w') as f:
                f.write(header + '\n')
        for r in records:
            pos = ','.join(f'{c:.2f}' for c in r['snapped_lu'])
            print(f"  Probe {r['pid']}: {r['name']} node {r['idx']} "
                  f"@ ({pos}) L0-lu")
        print(f"  Probe channel: {len(records)} probes, every "
              f"{self.interval} step(s), fs = {self.fs_hz:.1f} Hz "
              f"-> {self.csv_path}")
        self._bound = True

    def _locate(self, pid: int, p_lu: Tuple[float, ...],
                blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Nearest node on the finest block containing the point."""
        best = None
        for b in blocks:
            idx = tuple(
                int(round((p_lu[d] - b['origin'][d]) / b['spacing']))
                for d in range(self._dim))
            inside = all(0 <= idx[d] <= b['shape'][d] - 1
                         for d in range(self._dim))
            if inside and (best is None or b['level'] > best['level']):
                best = {'pid': pid, 'sim': b['sim'], 'idx': idx,
                        'level': b['level'], 'name': b['name'],
                        'shape': b['shape'], 'origin': b['origin'],
                        'spacing': b['spacing']}
        if best is None:
            raise ValueError(
                f"output.probes.points[{pid}] = {p_lu} (L0 lu) lies "
                "outside every grid block")
        best['snapped_lu'] = tuple(
            best['origin'][d] + best['idx'][d] * best['spacing']
            for d in range(self._dim))
        if best['level'] > 0:
            gap = min(min(best['idx'][d], best['shape'][d] - 1 - best['idx'][d])
                      for d in range(self._dim))
            if gap < _EDGE_GUARD:
                print(f"  WARNING: probe {pid} is {gap} node(s) from a "
                      f"{best['name']} block face — inside/near the C2F "
                      "coupling band; move it inward for clean acoustics")
        return best

    def _write_meta(self, records: List[Dict[str, Any]]) -> None:
        """Snapped positions + conversion constants (post-processing keys)."""
        uc, dim = self._uc, self._dim
        ax = 'xyz'[:dim]
        cols = (['probe', 'level', 'block']
                + [f'i{a}' for a in ax]
                + [f'{a}_lu' for a in ax]
                + [f'{a}_m' for a in ax]
                + [f'req_{a}' for a in ax])
        with open(self.meta_path, 'w') as f:
            f.write("# pressure probe metadata\n")
            f.write("# p_pa = cs_lu^2*(rho_lu - 1)*rho_phys"
                    "*(dx_phys/dt_phys)^2 ; p' = p - mean(p) in post\n")
            f.write(f"# dx_phys_m={uc.dx_phys:.9e}, "
                    f"dt_phys_s={uc.dt_phys:.9e} (L0), "
                    f"rho_phys={uc.rho_phys}, "
                    f"p_conv_pa_per_drho={self._p_conv:.9e}\n")
            f.write(f"# sample_interval={self.interval}, "
                    f"fs_hz={self.fs_hz:.6e}\n")
            f.write(','.join(cols) + '\n')
            for r in records:
                row = ([str(r['pid']), str(r['level']), r['name']]
                       + [str(i) for i in r['idx']]
                       + [f'{c:.6f}' for c in r['snapped_lu']]
                       + [f'{c * uc.dx_phys:.9e}' for c in r['snapped_lu']]
                       + [f'{c:.6f}' for c in self._points_in[r['pid']]])
                f.write(','.join(row) + '\n')
