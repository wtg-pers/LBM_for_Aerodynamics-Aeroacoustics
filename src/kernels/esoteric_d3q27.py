"""
Esoteric Pull Kernel for D3Q27 (Lehmann 2022)

Single-buffer in-place streaming + collision in one kernel launch.
Eliminates f_post, f_new, and Python BC overhead.

Key properties:
    - f array: 1 copy only (in-place read/write)
    - Bounce-back: implicit (SOLID nodes skip -> auto reversal)
    - BC: inside kernel (node_type flag)
    - Race-free: each memory address accessed by exactly 1 thread

Direction ordering: paired opposites (i, i+1) for i = 1,3,5,...,25
    This is REQUIRED for the Esoteric scheme to work.

Reference:
    Lehmann, "Esoteric Pull and Esoteric Push", Computation 10(6), 2022

Author: LBM Development Team
Date: 2026-04
"""

from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


# ============================================================
# Esoteric D3Q27 direction ordering (paired opposites)
# ============================================================
# Standard D3Q27 indices -> Esoteric paired order
# Pairs: (1,2), (3,4), (5,6), (7,10), (8,9), (11,14), (12,13),
#         (15,18), (16,17), (19,26), (20,25), (21,24), (22,23)

_STD_TO_ESO = [0, 1,2, 3,4, 5,6, 7,10, 8,9, 11,14, 12,13, 15,18, 16,17, 19,26, 20,25, 21,24, 22,23]

# Build Esoteric lattice constants
def _build_esoteric_lattice():
    """Build D3Q27 lattice constants in Esoteric paired ordering."""
    # Standard D3Q27
    cx_std = [0, 1,-1,0,0,0,0, 1,-1,1,-1,1,-1,1,-1, 0,0,0,0, 1,-1,1,-1,1,-1,1,-1]
    cy_std = [0, 0,0,1,-1,0,0, 1,1,-1,-1,0,0,0,0, 1,-1,1,-1, 1,1,-1,-1,1,1,-1,-1]
    cz_std = [0, 0,0,0,0,1,-1, 0,0,0,0,1,1,-1,-1, 1,1,-1,-1, 1,1,1,1,-1,-1,-1,-1]
    w_std = [8/27] + [2/27]*6 + [1/54]*12 + [1/216]*8

    cx_eso = [cx_std[_STD_TO_ESO[q]] for q in range(27)]
    cy_eso = [cy_std[_STD_TO_ESO[q]] for q in range(27)]
    cz_eso = [cz_std[_STD_TO_ESO[q]] for q in range(27)]
    w_eso = [w_std[_STD_TO_ESO[q]] for q in range(27)]

    # Verify pairing
    for p in range(13):
        i = 2*p + 1
        assert cx_eso[i] == -cx_eso[i+1], f"Pair ({i},{i+1}) not opposite in x"
        assert cy_eso[i] == -cy_eso[i+1], f"Pair ({i},{i+1}) not opposite in y"
        assert cz_eso[i] == -cz_eso[i+1], f"Pair ({i},{i+1}) not opposite in z"

    return cx_eso, cy_eso, cz_eso, w_eso

CX_ESO, CY_ESO, CZ_ESO, W_ESO = _build_esoteric_lattice()

# Mapping arrays for converting between standard and esoteric ordering
ESO_TO_STD = [_STD_TO_ESO[q] for q in range(27)]  # eso index -> std index
STD_TO_ESO_MAP = [0]*27
for eso_idx, std_idx in enumerate(_STD_TO_ESO):
    STD_TO_ESO_MAP[std_idx] = eso_idx


# ============================================================
# Node type constants
# ============================================================
NODE_FLUID = 0
NODE_SOLID = 1
NODE_EQ_BC = 2       # Equilibrium BC (f = f_eq at prescribed rho, u)
NODE_NEUMANN = 3     # Zero-gradient (copy from interior, no collision)
NODE_SPONGE = 4      # Collision + blending toward target


# ============================================================
# CUDA Kernel
# ============================================================

def _fmt_array(arr):
    """Format Python list as C array initializer."""
    return ', '.join(f'{v}f' if isinstance(v, float) else str(v) for v in arr)

_ESOTERIC_BGK_KERNEL = r'''
extern "C" __global__
void esoteric_bgk_d3q27(
    float*        __restrict__ f,          // (27, N) single buffer, in-place
    float*        __restrict__ rho_out,    // (N,)
    float*        __restrict__ u_out,      // (3, N)
    const char*   __restrict__ node_type,  // (N,) 0=fluid, 1=solid, 2=eq_bc, 3=neumann
    const float*  __restrict__ bc_rho,     // (N,) BC target rho
    const float*  __restrict__ bc_ux,      // (N,) BC target ux
    const float*  __restrict__ bc_uy,      // (N,) BC target uy
    const float*  __restrict__ bc_uz,      // (N,) BC target uz
    const bool*   __restrict__ needs_bounce, // (27, N) boundary link mask, or NULL
    float*        __restrict__ force_out,    // (3,) force accumulator (atomicAdd), or NULL
    const float omega,
    const int Nx, const int Ny, const int Nz,
    const int t_step                       // time step (parity: t_step & 1)
) {
    // 64-bit indexing: q*N+idx overflows int32 above ~79.5M nodes/level
    // (see project int32 kernel ceiling; cumulant esoteric already 64-bit).
    long long N = (long long)Nx * (long long)Ny * (long long)Nz;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int type = node_type[idx];
    if (type == 1) return;  // SOLID: skip (implicit bounce-back)

    // 3D index (C-contiguous: x varies slowest)
    long long ix = idx / ((long long)Ny * Nz);
    long long rem = idx - ix * (long long)Ny * Nz;
    long long iy = rem / Nz;
    long long iz = rem - iy * Nz;

    // Esoteric D3Q27 lattice constants (paired ordering)
    const int cx[27] = {''' + _fmt_array(CX_ESO) + r'''};
    const int cy[27] = {''' + _fmt_array(CY_ESO) + r'''};
    const int cz[27] = {''' + _fmt_array(CZ_ESO) + r'''};
    const float w[27] = {''' + _fmt_array(W_ESO) + r'''};

    int is_odd = t_step & 1;

    // ========================================================
    // LOAD (Esoteric Pull streaming part 2/2)
    // ========================================================
    float fhn[27];
    fhn[0] = f[0 * N + idx];  // rest: always local

    for (int p = 0; p < 13; p++) {
        int i = 2 * p + 1;  // odd index: 1, 3, 5, ..., 25

        // Neighbor in direction i
        long long nx = (ix + cx[i] + Nx) % Nx;
        long long ny = (iy + cy[i] + Ny) % Ny;
        long long nz = (iz + cz[i] + Nz) % Nz;
        long long j_i = nx * (long long)Ny * Nz + ny * (long long)Nz + nz;

        if (is_odd) {
            fhn[i]   = f[i     * N + idx];   // dir i from slot i at self
            fhn[i+1] = f[(i+1) * N + j_i];   // dir i+1 from slot i+1 at neighbor
        } else {
            fhn[i]   = f[(i+1) * N + idx];   // dir i from slot i+1 at self
            fhn[i+1] = f[i     * N + j_i];   // dir i+1 from slot i at neighbor
        }
    }

    // ========================================================
    // MACROSCOPIC
    // ========================================================
    float rho = 0.0f;
    float mom_x = 0.0f, mom_y = 0.0f, mom_z = 0.0f;
    for (int q = 0; q < 27; q++) {
        rho += fhn[q];
        mom_x += cx[q] * fhn[q];
        mom_y += cy[q] * fhn[q];
        mom_z += cz[q] * fhn[q];
    }
    float inv_rho = 1.0f / rho;
    float ux = mom_x * inv_rho;
    float uy = mom_y * inv_rho;
    float uz = mom_z * inv_rho;

    // ========================================================
    // BC + COLLISION
    // ========================================================
    if (type == 2) {
        // EQUILIBRIUM BC: override rho, u with target, compute f_eq
        rho = bc_rho[idx];
        ux = bc_ux[idx]; uy = bc_uy[idx]; uz = bc_uz[idx];
        float usqr = ux*ux + uy*uy + uz*uz;
        for (int q = 0; q < 27; q++) {
            float cu = (float)cx[q]*ux + (float)cy[q]*uy + (float)cz[q]*uz;
            fhn[q] = w[q] * rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usqr);
        }
    }
    else if (type == 0) {
        // FLUID: BGK collision
        float usqr = ux*ux + uy*uy + uz*uz;
        for (int q = 0; q < 27; q++) {
            float cu = (float)cx[q]*ux + (float)cy[q]*uy + (float)cz[q]*uz;
            float f_eq = w[q] * rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usqr);
            fhn[q] = fhn[q] - omega * (fhn[q] - f_eq);
        }
    }
    else if (type == 4) {
        // SPONGE: collision + blending toward target equilibrium
        // Target = f_eq(bc_rho, bc_u) with sigma blending
        float sigma = bc_uz[idx];  // reuse bc_uz as sigma for sponge nodes
        float rho_t = bc_rho[idx];
        float ux_t = bc_ux[idx], uy_t = bc_uy[idx];

        // Collision first (same as FLUID)
        float usqr = ux*ux + uy*uy + uz*uz;
        for (int q = 0; q < 27; q++) {
            float cu = (float)cx[q]*ux + (float)cy[q]*uy + (float)cz[q]*uz;
            float f_eq = w[q] * rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usqr);
            fhn[q] = fhn[q] - omega * (fhn[q] - f_eq);
        }

        // Blending toward target
        float usqr_t = ux_t*ux_t + uy_t*uy_t;
        for (int q = 0; q < 27; q++) {
            float cu_t = (float)cx[q]*ux_t + (float)cy[q]*uy_t;
            float f_target = w[q] * rho_t * (1.0f + 3.0f*cu_t + 4.5f*cu_t*cu_t - 1.5f*usqr_t);
            fhn[q] = (1.0f - sigma) * fhn[q] + sigma * f_target;
        }
    }
    // type == 3 (NEUMANN): no collision, pass through loaded values

    // Write macroscopic
    rho_out[idx] = rho;
    u_out[0 * N + idx] = ux;
    u_out[1 * N + idx] = uy;
    u_out[2 * N + idx] = uz;

    // ========================================================
    // MEM FORCE (optional, after collision, before store)
    // fhn[] contains post-collision distributions in correct
    // physical direction ordering -> force calculation is clean.
    // ========================================================
    if (force_out != NULL && needs_bounce != NULL && type == 0) {
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;
        for (int q = 1; q < 27; q++) {
            if (needs_bounce[q * N + idx]) {
                fx += 2.0f * (float)cx[q] * fhn[q];
                fy += 2.0f * (float)cy[q] * fhn[q];
                fz += 2.0f * (float)cz[q] * fhn[q];
            }
        }
        if (fx != 0.0f) atomicAdd(&force_out[0], fx);
        if (fy != 0.0f) atomicAdd(&force_out[1], fy);
        if (fz != 0.0f) atomicAdd(&force_out[2], fz);
    }

    // ========================================================
    // STORE (Esoteric Pull streaming part 1/2)
    // ========================================================
    f[0 * N + idx] = fhn[0];  // rest: always local

    for (int p = 0; p < 13; p++) {
        int i = 2 * p + 1;

        long long nx = (ix + cx[i] + Nx) % Nx;
        long long ny = (iy + cy[i] + Ny) % Ny;
        long long nz = (iz + cz[i] + Nz) % Nz;
        long long j_i = nx * (long long)Ny * Nz + ny * (long long)Nz + nz;

        if (is_odd) {
            f[(i+1) * N + j_i] = fhn[i];     // dir i -> slot i+1 at neighbor
            f[i     * N + idx] = fhn[i+1];   // dir i+1 -> slot i at self
        } else {
            f[i     * N + j_i] = fhn[i];     // dir i -> slot i at neighbor
            f[(i+1) * N + idx] = fhn[i+1];   // dir i+1 -> slot i+1 at self
        }
    }
}
'''


# ============================================================
# Python Wrapper
# ============================================================

class EsotericBGKKernelD3Q27:
    """Esoteric Pull BGK kernel for D3Q27.

    Single-buffer in-place streaming + collision + BC in one launch.

    Usage:
        >>> kernel = EsotericBGKKernelD3Q27()
        >>> kernel.launch(f, rho, u, node_type, bc_rho, bc_ux, bc_uy, bc_uz,
        ...               omega, Nx, Ny, Nz, t_step)
    """

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel = None

    def _compile(self) -> None:
        import cupy as cp
        self._kernel = cp.RawKernel(
            _ESOTERIC_BGK_KERNEL,
            'esoteric_bgk_d3q27',
            options=('--use_fast_math',),
        )

    def launch(
        self,
        f: 'npt.NDArray',
        rho_out: 'npt.NDArray',
        u_out: 'npt.NDArray',
        node_type: 'npt.NDArray',
        bc_rho: 'npt.NDArray',
        bc_ux: 'npt.NDArray',
        bc_uy: 'npt.NDArray',
        bc_uz: 'npt.NDArray',
        omega: float,
        Nx: int, Ny: int, Nz: int,
        t_step: int,
        needs_bounce: Optional['npt.NDArray'] = None,
        force_out: Optional['npt.NDArray'] = None,
    ) -> None:
        if self._kernel is None:
            self._compile()

        import cupy as cp
        N = Nx * Ny * Nz
        grid = (N + self._block_size - 1) // self._block_size

        nb_arg = needs_bounce if needs_bounce is not None else cp.int32(0)
        fo_arg = force_out if force_out is not None else cp.int32(0)

        self._kernel(
            (grid,), (self._block_size,),
            (f, rho_out, u_out, node_type,
             bc_rho, bc_ux, bc_uy, bc_uz,
             nb_arg, fo_arg,
             cp.float32(omega),
             cp.int32(Nx), cp.int32(Ny), cp.int32(Nz),
             cp.int32(t_step)),
        )


def init_f_esoteric(xp, f_physical: 'npt.NDArray', t_start: int = 0) -> 'npt.NDArray':
    """Initialize f in Esoteric memory layout for a given start step.

    The Esoteric scheme stores distributions with alternating slot swaps.
    At even steps, slot i contains direction i+1's value (and vice versa).
    At odd steps, slots match directions directly.

    Args:
        xp: Array module
        f_physical: Distribution in physical ordering (27, Nx, Ny, Nz)
                    Already in Esoteric direction ordering.
        t_start: Starting time step (0 = even)

    Returns:
        f_memory: Distribution in Esoteric memory layout
    """
    # The first kernel LOAD at step t_start must reconstruct f_physical exactly.
    # It reads one slot locally and the paired slot from the direction-i
    # neighbour (x + c_i), so the paired slot must be pre-streamed (rolled by
    # +c_i). A plain local swap (the previous implementation) omitted this roll
    # and produced a per-direction one-cell offset at step 0. c_i is the
    # Esoteric direction-i velocity (CX_ESO[i], ...).
    #   even LOAD: fhn[i]<-mem[i+1]@self, fhn[i+1]<-mem[i]@(x+c_i)
    #            => mem[i+1]=f[i] (local),  mem[i]=roll(f[i+1], +c_i)
    #   odd  LOAD: fhn[i]<-mem[i]@self,   fhn[i+1]<-mem[i+1]@(x+c_i)
    #            => mem[i]=f[i]   (local),  mem[i+1]=roll(f[i+1], +c_i)
    f_mem = xp.copy(f_physical)
    even = (t_start % 2 == 0)
    ndim = f_physical.ndim - 1              # spatial dims (3 for D3Q27 field)
    axes = tuple(range(1, f_physical.ndim))
    for p in range(13):
        i = 2 * p + 1
        ci = (int(CX_ESO[i]), int(CY_ESO[i]), int(CZ_ESO[i]))[:ndim]
        rolled = xp.roll(f_physical[i + 1], shift=ci, axis=tuple(a - 1 for a in axes))
        if even:
            f_mem[i + 1] = f_physical[i]
            f_mem[i] = rolled
        else:
            f_mem[i] = f_physical[i]
            f_mem[i + 1] = rolled
    return f_mem


def convert_f_std_to_esoteric(xp, f_std: 'npt.NDArray') -> 'npt.NDArray':
    """Convert f from standard D3Q27 ordering to Esoteric paired ordering.

    Args:
        xp: Array module
        f_std: Distribution in standard ordering (27, Nx, Ny, Nz)

    Returns:
        f_eso: Distribution in Esoteric ordering (27, Nx, Ny, Nz)
    """
    f_eso = xp.empty_like(f_std)
    for eso_q in range(27):
        std_q = _STD_TO_ESO[eso_q]
        f_eso[eso_q] = f_std[std_q]
    return f_eso


def convert_f_esoteric_to_std(xp, f_eso: 'npt.NDArray') -> 'npt.NDArray':
    """Convert f from Esoteric ordering back to standard D3Q27 ordering.

    Args:
        xp: Array module
        f_eso: Distribution in Esoteric ordering (27, Nx, Ny, Nz)

    Returns:
        f_std: Distribution in standard ordering (27, Nx, Ny, Nz)
    """
    f_std = xp.empty_like(f_eso)
    for eso_q in range(27):
        std_q = _STD_TO_ESO[eso_q]
        f_std[std_q] = f_eso[eso_q]
    return f_std


# ============================================================
# Gather / Scatter: Esoteric memory <-> physical distribution
# (bridge for MLG coupling, checkpointing; pure permutation+roll
#  = bit-exact roundtrip)
# ============================================================

def esoteric_gather_physical(xp, f_mem: 'npt.NDArray', t_step: int) -> 'npt.NDArray':
    """Reconstruct the physical distribution (Esoteric direction ordering)
    from Esoteric memory, as the kernel LOAD at step `t_step` would read it.

        even LOAD: f[i]@x = mem[i+1]@x ; f[i+1]@x = mem[i]@(x+c_i)
        odd  LOAD: f[i]@x = mem[i]@x   ; f[i+1]@x = mem[i+1]@(x+c_i)

    "value at x comes from array at x+c_i" == xp.roll(arr, shift=-c_i).
    Exact inverse of init_f_esoteric / esoteric_scatter_physical.
    """
    f_phys = xp.empty_like(f_mem)
    f_phys[0] = f_mem[0]
    even = (t_step % 2 == 0)
    ndim = f_mem.ndim - 1
    ax = tuple(range(ndim))
    for p in range(13):
        i = 2 * p + 1
        neg = tuple(-c for c in (CX_ESO[i], CY_ESO[i], CZ_ESO[i])[:ndim])
        if even:
            f_phys[i] = f_mem[i + 1]
            f_phys[i + 1] = xp.roll(f_mem[i], shift=neg, axis=ax)
        else:
            f_phys[i] = f_mem[i]
            f_phys[i + 1] = xp.roll(f_mem[i + 1], shift=neg, axis=ax)
    return f_phys


def esoteric_scatter_physical(xp, f_phys: 'npt.NDArray', t_step: int) -> 'npt.NDArray':
    """Place a physical distribution (Esoteric ordering) into Esoteric memory
    so the kernel LOAD at step `t_step` reads exactly f_phys. Same mapping as
    init_f_esoteric (kept as the single implementation)."""
    return init_f_esoteric(xp, f_phys, t_start=t_step)


def esoteric_gather_std(xp, f_mem: 'npt.NDArray', t_step: int) -> 'npt.NDArray':
    """Esoteric memory -> physical f in STANDARD D3Q27 ordering.

    FUSED single-allocation version (gather + reorder in one pass, writing
    each slot directly into its standard index). The previous two-step
    gather_physical -> convert transiently held TWO f-sized arrays, which
    OOM'd the D40 checkpoint gather on a 24GB card (2026-07-10 cluster run).
    Values are identical to the two-step path (pure permutation + roll).
    """
    out = xp.empty_like(f_mem)
    even = (t_step % 2 == 0)
    ndim = f_mem.ndim - 1
    ax = tuple(range(ndim))
    out[_STD_TO_ESO[0]] = f_mem[0]
    for p in range(13):
        i = 2 * p + 1
        neg = tuple(-c for c in (CX_ESO[i], CY_ESO[i], CZ_ESO[i])[:ndim])
        if even:
            out[_STD_TO_ESO[i]] = f_mem[i + 1]
            out[_STD_TO_ESO[i + 1]] = xp.roll(f_mem[i], shift=neg, axis=ax)
        else:
            out[_STD_TO_ESO[i]] = f_mem[i]
            out[_STD_TO_ESO[i + 1]] = xp.roll(f_mem[i + 1], shift=neg, axis=ax)
    return out


def esoteric_scatter_std(xp, f_std: 'npt.NDArray', t_step: int) -> 'npt.NDArray':
    """Physical f in STANDARD ordering -> Esoteric memory layout."""
    return init_f_esoteric(
        xp, convert_f_std_to_esoteric(xp, f_std), t_start=t_step)


# ============================================================
# REGION-scoped gather/scatter (Phase e2, patch 15)
# Same slot/parity mapping as the full versions, restricted to a
# spatial region (supports strided slices, e.g. the F2C 0::R read).
# Wrap semantics for the +-c_i shifted access match the kernel's
# periodic index arithmetic exactly ((i + c + N) % N).
# ============================================================

def _region_axis_indices(xp, sl: slice, n: int):
    """Integer indices selected by `sl` on an axis of length n."""
    return xp.arange(*sl.indices(n), dtype=xp.int64)


def _region_ix(xp, mem_shape, region, shift=(0, 0, 0)):
    """Open-mesh advanced indices for region (+shift, wrapped) per axis."""
    Nx, Ny, Nz = mem_shape
    ix = (_region_axis_indices(xp, region[0], Nx) + shift[0]) % Nx
    iy = (_region_axis_indices(xp, region[1], Ny) + shift[1]) % Ny
    iz = (_region_axis_indices(xp, region[2], Nz) + shift[2]) % Nz
    return ix[:, None, None], iy[None, :, None], iz[None, None, :]


def esoteric_gather_std_region(xp, f_mem: 'npt.NDArray', t_step: int,
                               region) -> 'npt.NDArray':
    """Gather the physical f (STANDARD ordering) on a spatial region only.

    region: 3-tuple of slices into the (Nx, Ny, Nz) grid (strided OK).
    Returns (27, *region_shape). Equals esoteric_gather_std(...)[:, region].
    """
    shape = f_mem.shape[1:]
    ix0, iy0, iz0 = _region_ix(xp, shape, region)
    rs = (len(ix0.ravel()), len(iy0.ravel()), len(iz0.ravel()))
    out = xp.empty((27,) + rs, dtype=f_mem.dtype)
    even = (t_step % 2 == 0)
    out[_STD_TO_ESO[0]] = f_mem[0][ix0, iy0, iz0]
    for p in range(13):
        i = 2 * p + 1
        ci = (CX_ESO[i], CY_ESO[i], CZ_ESO[i])
        ixs, iys, izs = _region_ix(xp, shape, region, shift=ci)
        std_i, std_ip1 = _STD_TO_ESO[i], _STD_TO_ESO[i + 1]
        if even:
            out[std_i] = f_mem[i + 1][ix0, iy0, iz0]
            out[std_ip1] = f_mem[i][ixs, iys, izs]
        else:
            out[std_i] = f_mem[i][ix0, iy0, iz0]
            out[std_ip1] = f_mem[i + 1][ixs, iys, izs]
    return out


def esoteric_scatter_std_region(xp, f_mem: 'npt.NDArray',
                                values: 'npt.NDArray', t_step: int,
                                region) -> None:
    """Scatter physical values (STANDARD ordering, region-shaped) into the
    Esoteric memory IN PLACE, so the next LOAD at `t_step` reads them.

    Inverse of esoteric_gather_std_region on the same region: writes the
    local slot at region and the paired slot at region+c_i (wrapped).
    """
    shape = f_mem.shape[1:]
    ix0, iy0, iz0 = _region_ix(xp, shape, region)
    even = (t_step % 2 == 0)
    f_mem[0][ix0, iy0, iz0] = values[_STD_TO_ESO[0]]
    for p in range(13):
        i = 2 * p + 1
        ci = (CX_ESO[i], CY_ESO[i], CZ_ESO[i])
        ixs, iys, izs = _region_ix(xp, shape, region, shift=ci)
        std_i, std_ip1 = _STD_TO_ESO[i], _STD_TO_ESO[i + 1]
        if even:
            f_mem[i + 1][ix0, iy0, iz0] = values[std_i]
            f_mem[i][ixs, iys, izs] = values[std_ip1]
        else:
            f_mem[i][ix0, iy0, iz0] = values[std_i]
            f_mem[i + 1][ixs, iys, izs] = values[std_ip1]


# ============================================================
# Esoteric macro pre-pass kernel: LOAD + (rho, u), no collide/store.
# Used by the WALE/dyn_smag pre-pass and the ALM 2-pass, which need
# the current-step macroscopic fields BEFORE the collision launch.
# NOTE: computes on every node incl. solids (garbage there) -- same
# as the standard MacroKernel pre-pass; ALM/rotor cases have no solids.
# ============================================================

_ESOTERIC_MACRO_KERNEL = r'''
extern "C" __global__
void esoteric_macro_d3q27(
    const float* __restrict__ f,        // (27, N) Esoteric memory
    float*       __restrict__ rho_out,  // (N,)
    float*       __restrict__ u_out,    // (3, N)
    const int Nx, const int Ny, const int Nz,
    const int t_step
) {
    long long N = (long long)Nx * (long long)Ny * (long long)Nz;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    long long ix = idx / ((long long)Ny * Nz);
    long long rem = idx - ix * (long long)Ny * Nz;
    long long iy = rem / Nz;
    long long iz = rem - iy * Nz;

    const int cx[27] = {''' + _fmt_array(CX_ESO) + r'''};
    const int cy[27] = {''' + _fmt_array(CY_ESO) + r'''};
    const int cz[27] = {''' + _fmt_array(CZ_ESO) + r'''};

    int is_odd = t_step & 1;

    float fhn[27];
    fhn[0] = f[0 * N + idx];
    for (int p = 0; p < 13; p++) {
        int i = 2 * p + 1;
        long long nx = (ix + cx[i] + Nx) % Nx;
        long long ny = (iy + cy[i] + Ny) % Ny;
        long long nz = (iz + cz[i] + Nz) % Nz;
        long long j_i = nx * (long long)Ny * Nz + ny * (long long)Nz + nz;
        if (is_odd) {
            fhn[i]   = f[i     * N + idx];
            fhn[i+1] = f[(i+1) * N + j_i];
        } else {
            fhn[i]   = f[(i+1) * N + idx];
            fhn[i+1] = f[i     * N + j_i];
        }
    }

    float rho = 0.0f, mx = 0.0f, my = 0.0f, mz = 0.0f;
    for (int q = 0; q < 27; q++) {
        float fq = fhn[q];
        rho += fq; mx += cx[q]*fq; my += cy[q]*fq; mz += cz[q]*fq;
    }
    float inv_rho = 1.0f / rho;
    rho_out[idx] = rho;
    u_out[0 * N + idx] = mx * inv_rho;
    u_out[1 * N + idx] = my * inv_rho;
    u_out[2 * N + idx] = mz * inv_rho;
}
'''


class EsotericMacroKernelD3Q27:
    """Macro-only pre-pass on Esoteric memory (rho, u; no collide/store)."""

    def __init__(self, block_size: int = 256) -> None:
        self._block_size = block_size
        self._kernel = None

    def launch(self, f: 'npt.NDArray', rho_out: 'npt.NDArray',
               u_out: 'npt.NDArray', Nx: int, Ny: int, Nz: int,
               t_step: int) -> None:
        import cupy as cp
        if self._kernel is None:
            self._kernel = cp.RawKernel(
                _ESOTERIC_MACRO_KERNEL, "esoteric_macro_d3q27",
                options=('--use_fast_math',))
        N = Nx * Ny * Nz
        grid = (N + self._block_size - 1) // self._block_size
        self._kernel((grid,), (self._block_size,),
                     (f, rho_out, u_out,
                      cp.int32(Nx), cp.int32(Ny), cp.int32(Nz),
                      cp.int32(t_step)))
