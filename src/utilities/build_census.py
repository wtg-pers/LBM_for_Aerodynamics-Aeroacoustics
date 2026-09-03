"""Build-phase memory census (mpi_axis/06).

One stderr line per call: host RSS + CuPy pool used/held + raw device
used/total, tagged with the build stage and MPI rank. Purpose: localize
the g6-class replicated-build peak (rank SIGKILL = host OOM killer needs
HOST numbers, which the runner's device-only census cannot see).

Env-gated: set LBM_BUILD_CENSUS=1 to enable (default off — zero overhead
and zero output on every existing path).

Usage:
    from src.utilities.build_census import build_census
    build_census("setup L3 sim built")
"""
from __future__ import annotations

import os
import sys


def _rank() -> str:
    for k in ('OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'PMIX_RANK',
              'MV2_COMM_WORLD_RANK', 'SLURM_PROCID'):
        v = os.environ.get(k)
        if v is not None:
            return v
    return '?'


def _rss_gib() -> float:
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 2 ** 20   # kB -> GiB
    except OSError:
        pass
    return -1.0


def _host_top(top: int = 8) -> str:
    """gc-walk numpy census (LBM_BUILD_CENSUS=2): top host arrays by
    (dtype, shape) class — the host twin of runner._mem_census."""
    import gc
    import numpy as np
    rows = {}
    seen = set()
    for o in gc.get_objects():
        if isinstance(o, np.ndarray):
            base = o.base if isinstance(o.base, np.ndarray) else o
            if id(base) in seen:
                continue
            seen.add(id(base))
            key = (str(base.dtype), base.shape)
            n, b = rows.get(key, (0, 0))
            rows[key] = (n + 1, b + base.nbytes)
    out = []
    for (dt, shp), (n, b) in sorted(rows.items(),
                                    key=lambda kv: -kv[1][1])[:top]:
        if b < 64 * 2 ** 20:
            break
        out.append(f"\n[census]   host {b / 2**30:7.3f} GiB x{n:<3d} "
                   f"{dt:<9s} {shp}")
    return ''.join(out)


def build_census(tag: str) -> None:
    lvl = os.environ.get('LBM_BUILD_CENSUS', '0')
    if lvl not in ('1', '2'):
        return
    dev = ''
    try:
        import cupy as cp
        mp = cp.get_default_memory_pool()
        pp = cp.get_default_pinned_memory_pool()
        try:
            pinned_gib = pp.total_bytes() / 2 ** 30
        except AttributeError:      # older CuPy: no accounting API
            pinned_gib = -1.0
        free_b, total_b = cp.cuda.Device().mem_info
        dev = (f" | dev pool {mp.used_bytes() / 2**30:.2f}"
               f"/{mp.total_bytes() / 2**30:.2f} GiB"
               f", pinned {pinned_gib:.2f} GiB"
               f", raw {(total_b - free_b) / 2**30:.2f}"
               f"/{total_b / 2**30:.2f} GiB")
    except Exception:
        dev = ' | dev n/a'
    extra = _host_top() if lvl == '2' else ''
    print(f"[census] rank{_rank()} {tag}: host RSS {_rss_gib():.2f} GiB"
          f"{dev}{extra}", file=sys.stderr, flush=True)
