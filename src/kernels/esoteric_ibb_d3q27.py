"""Esoteric-path Bouzidi IBB for D3Q27 — two-phase deposit rewrite (S5).

Core insight (PLAN.md, cross-checked against eso_mem_force's t_next slot
convention): the esoteric fused kernel's implicit HWBB is "re-read your own
deposit" — at the completed step's parity, the post-collision outgoing
population f*_d(x) toward a SOLID neighbor sits at a known slot/address, and
the NEXT step's LOAD reads exactly that address back as the bounced
population. Bouzidi IBB therefore needs NO change to the fused kernel: a
sparse post-pass overwrites each toward-solid deposit with the Bouzidi
combination, and the next LOAD picks it up.

Deposit address algebra (completed step parity `even = ((t_done & 1)==0)`,
esoteric paired ordering, pair (i, i+1) with i odd):
    dir e odd  : deposit at cell (y + c_e), slot = even ? e : e+1
    dir e even : deposit at cell  y       , slot = even ? e : e-1
This matches the STORE in esoteric_d3q27 and the (t_next) convention in
eso_mem_force (t_next = t_done + 1).

Bouzidi linear (identical float expressions to interpolated_wall_d3q27):
    q >= 1/2:  val = (1/(2q)) f*_d(x)  + ((2q-1)/(2q)) f*_dbar(x)
    q <  1/2:  val = 2q f*_d(x) + (1-2q) f*_d(x - c_d)
All three reads are deposits of the SAME completed step, at x or x +- c_d
(pair axis, +-1 cell) => MPI ghost >= 2 suffices (S6).

Two phases are STRUCTURALLY required: in a thin gap (solid on both sides,
e.g. a 1-cell fluid channel) link (x, d)'s Bouzidi READ of f*_dbar(x) is
link (x, dbar)'s WRITE address — a single fused pass races. Phase 1
(gather) computes every link value from pristine deposits into link_val;
phase 2 (scatter) writes them back. Each deposit address belongs to exactly
one link, so scatter is write-race-free and may read its own old value.

MEM force is fused into scatter (f64 atomics, owned-clip): after the
rewrite the original f*_d is gone, so F = sum_links c_d (f*_d + val) must
be computed in the same pass. q = 0.5 gives val == f*_d bit-exactly
(1/(2*0.5) = 1.0, coefficient of the partner read is exactly 0.0), so the
whole pass — f evolution AND force — degenerates to the eso HWBB path
bit-for-bit (gate G2).

Links: standard sparse triple (link_cell, link_dir, link_q) from
InterpolatedBounceBack._build_sparse_links with ONLY the direction remapped
std -> esoteric (cells and q untouched, q already solid-upstream
sanitized). ASCII-only kernel source (nvrtc POSIX locale).

Author: LBM Development Team
Date: 2026-07
"""

from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

from src.kernels.esoteric_d3q27 import CX_ESO, CY_ESO, CZ_ESO, STD_TO_ESO_MAP

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


def _fmt_int(arr) -> str:
    return ", ".join(str(int(v)) for v in arr)


def _kernel_source() -> str:
    tables = f"""
static __device__ const int CXE[27] = {{{_fmt_int(CX_ESO)}}};
static __device__ const int CYE[27] = {{{_fmt_int(CY_ESO)}}};
static __device__ const int CZE[27] = {{{_fmt_int(CZ_ESO)}}};
"""
    return tables + r"""
__device__ __forceinline__ long long eso_dep_addr(
    const int e, const int yx, const int yy, const int yz,
    const int even,
    const int Nx, const int Ny, const int Nz, const long long N)
{
    // Deposit address of direction e leaving cell y at completed parity.
    int slot, ax, ay, az;
    if (e & 1) {                       // odd: deposited at y + c_e
        slot = even ? e : (e + 1);
        ax = yx + CXE[e]; ax += (ax < 0) ? Nx : ((ax >= Nx) ? -Nx : 0);
        ay = yy + CYE[e]; ay += (ay < 0) ? Ny : ((ay >= Ny) ? -Ny : 0);
        az = yz + CZE[e]; az += (az < 0) ? Nz : ((az >= Nz) ? -Nz : 0);
    } else {                           // even: deposited at y itself
        slot = even ? e : (e - 1);
        ax = yx; ay = yy; az = yz;
    }
    long long cell = ((long long)ax * Ny + ay) * Nz + az;
    return (long long)slot * N + cell;
}

extern "C" __global__ void eso_ibb_gather(
    const float* __restrict__ f,             // (27, N) esoteric memory
    const int*   __restrict__ link_cell,     // (n_links,) flat cell idx
    const signed char* __restrict__ link_dir, // (n_links,) ESO dir 1..26
    const float* __restrict__ link_q,        // (n_links,) q in (0, 1]
    float*       __restrict__ link_val,      // (n_links,) out
    const int n_links,
    const int Nx, const int Ny, const int Nz,
    const int t_done)                        // parity of the completed step
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= n_links) return;

    const long long N = (long long)Nx * Ny * Nz;
    const int even = ((t_done & 1) == 0);
    const long long cell = (long long)link_cell[l];
    const int d = (int)link_dir[l];
    const float qv = link_q[l];

    int iz = (int)(cell % Nz);
    int iy = (int)((cell / Nz) % Ny);
    int ix = (int)(cell / ((long long)Nz * (long long)Ny));

    float f_d = f[eso_dep_addr(d, ix, iy, iz, even, Nx, Ny, Nz, N)];

    float val;
    if (qv >= 0.5f) {
        int dbar = (d & 1) ? (d + 1) : (d - 1);
        float f_db = f[eso_dep_addr(dbar, ix, iy, iz, even, Nx, Ny, Nz, N)];
        float two_q  = 2.0f * qv;
        float inv_2q = 1.0f / two_q;
        val = inv_2q * f_d + (two_q - 1.0f) * inv_2q * f_db;
    } else {
        int ux = ix - CXE[d]; ux += (ux < 0) ? Nx : ((ux >= Nx) ? -Nx : 0);
        int uy = iy - CYE[d]; uy += (uy < 0) ? Ny : ((uy >= Ny) ? -Ny : 0);
        int uz = iz - CZE[d]; uz += (uz < 0) ? Nz : ((uz >= Nz) ? -Nz : 0);
        float f_up = f[eso_dep_addr(d, ux, uy, uz, even, Nx, Ny, Nz, N)];
        val = 2.0f * qv * f_d + (1.0f - 2.0f * qv) * f_up;
    }
    link_val[l] = val;
}

extern "C" __global__ void eso_ibb_scatter(
    float*       __restrict__ f,             // (27, N) esoteric memory
    const int*   __restrict__ link_cell,
    const signed char* __restrict__ link_dir,
    const float* __restrict__ link_val,
    double*      __restrict__ force_out,     // (3,) f64 accumulator or NULL
    const int n_links,
    const int Nx, const int Ny, const int Nz,
    const int t_done,
    const int cx0, const int cx1,            // owned clip [x0, x1) etc.
    const int cy0, const int cy1,
    const int cz0, const int cz1)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= n_links) return;

    const long long N = (long long)Nx * Ny * Nz;
    const int even = ((t_done & 1) == 0);
    const long long cell = (long long)link_cell[l];
    const int d = (int)link_dir[l];

    int iz = (int)(cell % Nz);
    int iy = (int)((cell / Nz) % Ny);
    int ix = (int)(cell / ((long long)Nz * (long long)Ny));

    long long addr = eso_dep_addr(d, ix, iy, iz, even, Nx, Ny, Nz, N);
    float f_old = f[addr];      // own address: no other link touches it
    float v = link_val[l];
    f[addr] = v;

    // MEM force on the body: F += c_d * (f*_d + f_bounced). Fluid-cell
    // ownership clip keeps MPI rank contributions exclusive (S6).
    if (force_out != NULL &&
        ix >= cx0 && ix < cx1 && iy >= cy0 && iy < cy1 &&
        iz >= cz0 && iz < cz1) {
        double m = (double)f_old + (double)v;
        if (CXE[d] != 0) atomicAdd(&force_out[0], m * (double)CXE[d]);
        if (CYE[d] != 0) atomicAdd(&force_out[1], m * (double)CYE[d]);
        if (CZE[d] != 0) atomicAdd(&force_out[2], m * (double)CZE[d]);
    }
}
"""


class EsotericIBBD3Q27:
    """Two-phase Bouzidi deposit rewrite on the esoteric path.

    Built once per level from an InterpolatedBounceBack's sparse links;
    `rewrite(f, t_done)` is called right after each fused-kernel launch
    (t_done = the step index the kernel just executed). The per-step MEM
    force accumulates into a (3,) float64 device buffer read via
    `last_force()`.
    """

    def __init__(
        self,
        xp: 'ModuleType',
        link_cell: 'npt.NDArray',
        link_dir_eso: 'npt.NDArray',
        link_q: 'npt.NDArray',
        domain_shape: Tuple[int, int, int],
        clip_bounds: Optional[Sequence[Tuple[int, int]]] = None,
        block_size: int = 256,
    ) -> None:
        self.xp = xp
        self.link_cell = xp.ascontiguousarray(link_cell, dtype=xp.int32)
        self.link_dir = xp.ascontiguousarray(link_dir_eso, dtype=xp.int8)
        self.link_q = xp.ascontiguousarray(link_q, dtype=xp.float32)
        self.n_links = int(self.link_cell.size)
        self.domain_shape = tuple(int(s) for s in domain_shape)
        if clip_bounds is None:
            clip_bounds = [(0, s) for s in self.domain_shape]
        self.clip_bounds = [(int(a), int(b)) for a, b in clip_bounds]

        self._block_size = block_size
        self._link_val = xp.empty(self.n_links, dtype=xp.float32)
        self._force_out = xp.zeros(3, dtype=xp.float64)
        self._k_gather = None
        self._k_scatter = None

    # ------------------------------------------------------------------
    @classmethod
    def from_interpolated_bc(
        cls,
        xp: 'ModuleType',
        ibb,
        domain_shape: Tuple[int, int, int],
        clip_bounds: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> 'EsotericIBBD3Q27':
        """Build from an InterpolatedBounceBack (3D sparse) instance.

        Only the direction is remapped std -> esoteric ordering; cells and
        (solid-upstream-sanitized) q are reused untouched.
        """
        if getattr(ibb, 'link_cell', None) is None:
            raise ValueError(
                "esoteric IBB needs the sparse link arrays "
                "(InterpolatedBounceBack 3D use_sparse path)")
        lut = xp.asarray(np.asarray(STD_TO_ESO_MAP, dtype=np.int8))
        link_dir_eso = lut[ibb.link_dir.astype(xp.int64)]
        return cls(xp, ibb.link_cell, link_dir_eso, ibb.link_q,
                   domain_shape, clip_bounds=clip_bounds)

    # ------------------------------------------------------------------
    def _compile(self) -> None:
        import cupy as cp
        src = _kernel_source()
        assert all(ord(ch) < 128 for ch in src)  # nvrtc POSIX locale rule
        # --fmad=false: keep the Bouzidi expressions IEEE mul/add (no FMA
        # contraction) so the pass is bit-reproducible against the numpy
        # f32 reference (gate G0). Cost is negligible on a sparse pass.
        opts = ('--fmad=false',)
        self._k_gather = cp.RawKernel(src, 'eso_ibb_gather', options=opts)
        self._k_scatter = cp.RawKernel(src, 'eso_ibb_scatter', options=opts)

    def rewrite(self, f: 'npt.NDArray', t_done: int) -> None:
        """Overwrite toward-solid deposits with Bouzidi values (2 launches).

        Args:
            f: esoteric single buffer, shape (27, Nx, Ny, Nz) float32.
            t_done: index of the step the fused kernel JUST executed
                (parity selects the deposit slots of that store).
        """
        if self.n_links == 0:
            return
        if self._k_gather is None:
            self._compile()
        import cupy as cp
        nx, ny, nz = self.domain_shape
        grid = ((self.n_links + self._block_size - 1) // self._block_size,)
        blk = (self._block_size,)
        cb = self.clip_bounds

        self._k_gather(grid, blk, (
            f, self.link_cell, self.link_dir, self.link_q, self._link_val,
            cp.int32(self.n_links),
            cp.int32(nx), cp.int32(ny), cp.int32(nz), cp.int32(t_done)))

        self._force_out.fill(0)
        self._k_scatter(grid, blk, (
            f, self.link_cell, self.link_dir, self._link_val,
            self._force_out,
            cp.int32(self.n_links),
            cp.int32(nx), cp.int32(ny), cp.int32(nz), cp.int32(t_done),
            cp.int32(cb[0][0]), cp.int32(cb[0][1]),
            cp.int32(cb[1][0]), cp.int32(cb[1][1]),
            cp.int32(cb[2][0]), cp.int32(cb[2][1])))

    def last_force(self) -> np.ndarray:
        """MEM force of the most recent rewrite, host float64 (3,)."""
        return self.xp.asnumpy(self._force_out)


def build_slab_ibb(
    xp: 'ModuleType',
    ibb,
    part,
    global_shape: Tuple[int, int, int],
) -> Optional['EsotericIBBD3Q27']:
    """Slab-filtered, rebased EsotericIBBD3Q27 for one rank's LocalLevel (S6).

    Filter: links whose FLUID CELL lies at slab-local axis layer
    rel in [3, local-4] (periodic wrap), which REQUIRES ghost >= 5.

    Dependency-hop accounting (why 3 layers, not 1 — PLAN's "ghost>=2
    suffices" undercounted this; found by the NR=2 --verify ulp failure):
      hop 0: after each pre-advance sync the whole slab is owner-truth,
             but the OUTERMOST layer (rel 0 / local-1) computes garbage
             (its LOAD wraps around the slab) -> compute-correct cells:
             rel in [1, local-2];
      hop 1: a kernel STORE writes slots at writer+-1 -> post-advance
             slot-correct addresses: rel in [2, local-3];
      hop 2: the rewrite READS slots at link+-1 -> input-consistent
             links: rel in [3, local-4]; their rewrites land at [2,...],
             so post-rewrite owner-consistent slots: rel in [4, local-5];
      halo:  the v1 band gather reconstructs std f of the owned edge band
             (width g) and READS slots one cell beyond it — reach
             [g-1, 2g] — which must sit inside [4, local-5] -> g >= 5.
    Links outside [3, local-4] are the neighbor's responsibility: their
    rewritten slots live in ghost layers that every consumer (loads,
    coupling gathers, band sends — all post-sync in the runner) sees only
    AFTER the halo refreshed them with the owner's consistent values.

    Ownership clip for the fused MEM force = the OWNED axis range (rank
    contributions exclusive; Allreduce outside).

    Host-side int64 arithmetic, slab-local int32 cells. Returns None when
    no link survives the filter (level slab away from the body).
    """
    nx, ny, nz = (int(s) for s in global_shape)
    cell = ibb.link_cell.get() if hasattr(ibb.link_cell, 'get') \
        else np.asarray(ibb.link_cell)
    dir_std = ibb.link_dir.get() if hasattr(ibb.link_dir, 'get') \
        else np.asarray(ibb.link_dir)
    link_q = ibb.link_q.get() if hasattr(ibb.link_q, 'get') \
        else np.asarray(ibb.link_q)
    cell = cell.astype(np.int64)

    gx = cell // (ny * nz)
    gy = (cell // nz) % ny
    gz = cell % nz
    coords = [gx, gy, gz]

    ax = int(part.axis)
    g = int(part.ghost)
    n_ax = (nx, ny, nz)[ax]
    local_dims = tuple(int(d) for d in part.local_shape)

    if g == 0:
        # NR == 1: the slab IS the domain (no ghosts) — keep everything.
        keep = np.ones(cell.shape[0], dtype=bool)
        rel = coords[ax]
    else:
        if g < 5:
            raise ValueError(
                f"esoteric IBB under MPI needs ghost >= 5 (got {g}): the "
                f"deposit rewrite adds a dependency hop on top of the "
                f"kernel store, so the halo band gather's [g-1, 2g] reach "
                f"only sees rewrite-consistent slots from g = 5 up "
                f"(see build_slab_ibb docstring).")
        local_ax = int(part.own_count) + 2 * g
        rel = (coords[ax] - (int(part.own_start) - g)) % n_ax
        keep = (rel >= 3) & (rel <= local_ax - 4)

    if not np.any(keep):
        return None

    lc = [coords[0][keep], coords[1][keep], coords[2][keep]]
    lc[ax] = rel[keep]
    ly, lz = local_dims[1], local_dims[2]
    local_flat = (lc[0] * ly + lc[1]) * lz + lc[2]
    assert local_flat.max() < np.prod(local_dims)

    lut = np.asarray(STD_TO_ESO_MAP, dtype=np.int8)
    dir_eso = lut[dir_std.astype(np.int64)[keep]]

    clip = [(0, d) for d in local_dims]
    clip[ax] = (g, g + int(part.own_count))
    return EsotericIBBD3Q27(
        xp,
        xp.asarray(local_flat.astype(np.int32)),
        xp.asarray(dir_eso),
        xp.asarray(link_q[keep]),
        local_dims,
        clip_bounds=clip,
    )
