"""Plane composer — per-level plane pieces -> ONE L0-grid .vti per step.

The plane channel writes RAW per-level (and per-rank-strip) .vti pieces:
correct as an archive, but in ParaView the levels overlap (no AMR
blanking in a .vtm) and parallel planes appear split into s<start>
strips. What a reader usually wants is the SAME view ParaView gives when
slicing the volume .vth: one composed section.

NOTE: the channel now ALSO writes a native per-step .vth AMR index
(<plane>_<step>.vth + <plane>_amr.pvd) -- open that in ParaView for the
composed finest-resolution view with automatic coarse-cell blanking.
This tool remains for L0-lattice resampled single .vti products and for
archives that predate the .vth output.

This tool composes OFFLINE (zero runtime cost, works on finished or
in-progress result trees): for every step it merges all pieces of a
plane onto the plane's L0 node grid and, because the level hierarchy is
NODE-ALIGNED, injects each finer level's node-coincident values over the
coarser ones EXACTLY (no interpolation; finest level wins). Under fine
boxes the L0 nodes therefore carry the fine solution sampled at L0
density — the "slice the .vth, read it on the level-0 lattice" product.

Output: <plane_dir>/composed/<plane>_L0c_<step>.vti (+ .pvd series).
Coordinates and spacing stay global L0-lu, arrays and units unchanged.

Usage (메인 디렉토리에서):
    python -m src.io.plane_compose <.../vtk/planes>          # 전 plane
    python -m src.io.plane_compose <.../planes/x_wing_center>
    python -m src.io.plane_compose <dir> --steps 100:200     # 스텝 구간

Author: LBM Development Team
Date: 2026-08
"""

import argparse
import os
import re
import struct
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_PIECE_RE = re.compile(r"^(?P<plane>.+)_(?P<tag>L\d+(?:b\d+)?(?:s\d+)?)"
                       r"_(?P<step>\d{8})\.vti$")


def _read_vti(path: str):
    """(origin[3], spacing, extent_counts[3], {name: array}) — arrays are
    (nx,ny,nz) scalars or (3,nx,ny,nz) vectors, float32 raw."""
    raw = open(path, "rb").read()
    i = raw.index(b'<AppendedData encoding="raw">')
    xml = raw[:i].decode("ascii", "replace")
    base = raw.index(b"_", i + len(b'<AppendedData encoding="raw">')) + 1
    we = [int(v) for v in
          re.search(r'WholeExtent="([^"]+)"', xml).group(1).split()]
    n = [we[1] - we[0] + 1, we[3] - we[2] + 1, we[5] - we[4] + 1]
    org = [float(v) for v in
           re.search(r'Origin="([^"]+)"', xml).group(1).split()]
    spc = float(re.search(r'Spacing="([^"]+)"', xml).group(1).split()[0])
    out = {}
    for _t, name, ncomp, off in re.findall(
            r'<DataArray type="([^"]+)" Name="([^"]+)" '
            r'NumberOfComponents="(\d+)" format="appended" offset="(\d+)"/>',
            xml):
        off = int(off)
        nb = struct.unpack("<Q", raw[base + off:base + off + 8])[0]
        buf = np.frombuffer(raw[base + off + 8:base + off + 8 + nb],
                            dtype="<f4")
        if int(ncomp) == 3:
            out[name] = buf.reshape(n[2], n[1], n[0], 3).transpose(3, 2, 1, 0)
        else:
            out[name] = buf.reshape(n[2], n[1], n[0]).transpose(2, 1, 0)
    return org, spc, n, out


def _write_vti(path: str, origin, n, arrays: Dict[str, np.ndarray]) -> None:
    """Minimal ImageData writer (spacing 1 = L0 lu, appended raw f32)."""
    heads, blobs, off = [], [], 0
    for name, a in arrays.items():
        if a.ndim == 4:
            # (3,nx,ny,nz) -> (nz,ny,nx,3) interleaved
            payload = np.ascontiguousarray(
                a.transpose(3, 2, 1, 0)).astype("<f4").tobytes()
            nc = 3
        else:
            payload = np.ascontiguousarray(
                a.transpose(2, 1, 0)).astype("<f4").tobytes()
            nc = 1
        heads.append(
            f'    <DataArray type="Float32" Name="{name}" '
            f'NumberOfComponents="{nc}" format="appended" '
            f'offset="{off}"/>')
        blobs.append(struct.pack("<Q", len(payload)) + payload)
        off += 8 + len(payload)
    ext = f"0 {n[0]-1} 0 {n[1]-1} 0 {n[2]-1}"
    xml = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="1.0" '
        'byte_order="LittleEndian" header_type="UInt64">\n'
        f' <ImageData WholeExtent="{ext}" '
        f'Origin="{origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}" '
        'Spacing="1 1 1">\n'
        f'  <Piece Extent="{ext}">\n'
        '   <PointData>\n' + "\n".join(heads) + '\n'
        '   </PointData>\n  </Piece>\n </ImageData>\n'
        ' <AppendedData encoding="raw">\n_')
    with open(path, "wb") as f:
        f.write(xml.encode("ascii"))
        for b in blobs:
            f.write(b)
        f.write(b"\n </AppendedData>\n</VTKFile>\n")


def _axis_maps(org: float, spc: float, cnt: int, t0: float, tn: int,
               tol: float = 1e-6) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Piece-node indices landing EXACTLY on target (L0, spacing 1) nodes.

    Returns (piece_idx, target_idx) or None if nothing aligns."""
    j = np.arange(cnt)
    pos = org + j * spc
    ti = np.round(pos - t0)
    ok = (np.abs(pos - (t0 + ti)) < tol) & (ti >= 0) & (ti < tn)
    if not ok.any():
        return None
    return j[ok], ti[ok].astype(np.int64)


def compose_plane_dir(pdir: str, steps: Optional[range] = None,
                      out_sub: str = "composed") -> int:
    by_step: Dict[int, List[Tuple[float, str]]] = {}
    plane_name = None
    for f in sorted(os.listdir(pdir)):
        m = _PIECE_RE.match(f)
        if not m:
            continue
        st = int(m.group("step"))
        if steps is not None and st not in steps:
            continue
        plane_name = m.group("plane")
        lvl = int(re.match(r"L(\d+)", m.group("tag")).group(1))
        by_step.setdefault(st, []).append((lvl, os.path.join(pdir, f)))
    if not by_step:
        return 0
    odir = os.path.join(pdir, out_sub)
    os.makedirs(odir, exist_ok=True)
    written = []
    for st in sorted(by_step):
        pieces = [(lvl, *_read_vti(p)) for lvl, p in sorted(by_step[st])]
        l0 = [p for p in pieces if p[0] == 0]
        if not l0:
            print(f"  [skip] step {st}: no L0 piece (need level 'all' "
                  "or 0 in the plane config)")
            continue
        # target grid = union of L0 pieces (integer L0-lu lattice)
        flat = next(d for d in range(3) if all(p[3][d] == 1 for p in l0))
        t_org = [min(p[1][d] for p in l0) for d in range(3)]
        t_n = [1 if d == flat else
               int(round(max(p[1][d] + (p[3][d] - 1) * p[2]
                             for p in l0) - t_org[d])) + 1
               for d in range(3)]
        names = set.intersection(*(set(p[4]) for p in pieces))
        comp = {}
        for name in sorted(names):
            ref = l0[0][4][name]
            shape = ((3,) + tuple(t_n)) if ref.ndim == 4 else tuple(t_n)
            comp[name] = np.zeros(shape, np.float32)
        # coarse -> fine injection order; node-aligned = exact
        for lvl, org, spc, n, arrs in sorted(pieces, key=lambda p: p[0]):
            maps = []
            for d in range(3):
                if d == flat:
                    maps.append((np.array([0]), np.array([0])))
                    continue
                m = _axis_maps(org[d], spc, n[d], t_org[d], t_n[d])
                if m is None:
                    maps = None
                    break
                maps.append(m)
            if maps is None:
                continue
            (jx, tx), (jy, ty), (jz, tz) = maps
            src = np.ix_(jx, jy, jz)
            dst = np.ix_(tx, ty, tz)
            for name in comp:
                a = arrs[name]
                if a.ndim == 4:
                    for c in range(3):
                        comp[name][c][dst] = a[c][src]
                else:
                    comp[name][dst] = a[src]
        out = os.path.join(odir, f"{plane_name}_L0c_{st:08d}.vti")
        _write_vti(out, t_org, t_n, comp)
        written.append((st, os.path.basename(out)))
    if written:
        with open(os.path.join(odir, f"{plane_name}_L0c.pvd"), "w") as f:
            f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" '
                    'version="0.1">\n <Collection>\n')
            for st, fn in written:
                f.write(f'  <DataSet timestep="{st}" file="{fn}"/>\n')
            f.write(' </Collection>\n</VTKFile>\n')
    return len(written)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", help="planes/ 루트 또는 개별 plane 디렉토리")
    ap.add_argument("--steps", default=None,
                    help="a:b (파이썬 range 반개구간)")
    ap.add_argument("--out", default="composed")
    a = ap.parse_args(argv)
    steps = None
    if a.steps:
        lo, hi = a.steps.split(":")
        steps = range(int(lo), int(hi))
    dirs = []
    if any(_PIECE_RE.match(f) for f in os.listdir(a.root)):
        dirs = [a.root]
    else:
        dirs = [os.path.join(a.root, d) for d in sorted(os.listdir(a.root))
                if os.path.isdir(os.path.join(a.root, d))
                and d != a.out]
    total = 0
    for d in dirs:
        n = compose_plane_dir(d, steps, a.out)
        print(f"  {os.path.basename(d)}: {n} composed steps")
        total += n
    return 0 if total else 1


if __name__ == "__main__":
    sys.exit(main())
