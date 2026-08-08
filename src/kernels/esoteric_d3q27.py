"""
Esoteric Pull Kernel for D3Q27 (Lehmann 2022)

Single-buffer in-place streaming + collision in one kernel launch.
Eliminates f_post, f_new, and Python BC overhead.

Key properties:
    - f array: 1 copy only (in-place read/write)
    - Bounce-back: implicit HWBB (SOLID nodes skip; fluid LOAD reads the
      parity-swapped slot at solid neighbors = own t-1 outgoing reflected.
      NOTE a bare skip WITHOUT the swapped-slot read is a 2-step-delayed
      bounce, not HWBB — found by eso_mem_force_twin_gate)
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
NODE_TRANSIT = 5     # v2 halo ghost mailbox: skipped like SOLID but WITHOUT
                     # the implicit-bounce semantics in neighboring loads
                     # (a SOLID ghost plane would reflect since the HWBB fix
                     # reads parity-swapped slots at SOLID neighbors)


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
    const int t_step,                      // time step (parity: t_step & 1)
    const float*  __restrict__ force,      // (3, N) body force (ALM), or NULL
    const float*  __restrict__ nu_t_in,    // (N,) SGS eddy viscosity, or NULL
    float*        __restrict__ nu_t_out    // (N,) nu_t diagnostics, or NULL
) {
    // 64-bit indexing: q*N+idx overflows int32 above ~79.5M nodes/level
    // (see project int32 kernel ceiling; cumulant esoteric already 64-bit).
    long long N = (long long)Nx * (long long)Ny * (long long)Nz;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int type = node_type[idx];
    if (type == 1 || type == 5) return;  // SOLID (implicit HWBB) / TRANSIT

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

        // Implicit halfway bounce-back (twin-gate fix): a SOLID neighbor
        // never stores, so the slot the normal LOAD reads there is STALE
        // (holds the t-2 deposit => a 2-step-DELAYED bounce, not HWBB;
        // caught by eso_mem_force_twin_gate as an O(f_i - f_opp) transient
        // divergence vs the standard path). Our own t-1 outgoing toward
        // that solid sits in the PARITY-SWAPPED slot at the same address;
        // reading it there is exact textbook HWBB:
        //     f_in[opp](x, t) = f_out[dir](x, t-1).
        // Pure-fluid cells never take these branches (bit-neutral).
        long long mx = (ix - cx[i] + Nx) % Nx;   // upstream nbr x - c_i
        long long my = (iy - cy[i] + Ny) % Ny;
        long long mz = (iz - cz[i] + Nz) % Nz;
        long long j_ip = mx * (long long)Ny * Nz + my * (long long)Nz + mz;
        bool nb_dn = node_type[j_i]  == 1;   // x + c_i solid: bounce dir i+1
        bool nb_up = node_type[j_ip] == 1;   // x - c_i solid: bounce dir i

        if (is_odd) {
            fhn[i]   = nb_up ? f[(i+1) * N + idx]
                             : f[i     * N + idx];   // dir i from slot i at self
            fhn[i+1] = nb_dn ? f[i     * N + j_i]
                             : f[(i+1) * N + j_i];   // dir i+1 from slot i+1 at neighbor
        } else {
            fhn[i]   = nb_up ? f[i     * N + idx]
                             : f[(i+1) * N + idx];   // dir i from slot i+1 at self
            fhn[i+1] = nb_dn ? f[(i+1) * N + j_i]
                             : f[i     * N + j_i];   // dir i+1 from slot i at neighbor
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

    // Body force (ALM): half-force velocity shift (same scheme as the
    // esoteric cumulant kernel; Guo source term added after collision).
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;
    if (force != NULL) {
        Fx = force[0 * N + idx]; Fy = force[1 * N + idx]; Fz = force[2 * N + idx];
        float h = 0.5f * inv_rho;
        ux += Fx * h; uy += Fy * h; uz += Fz * h;
    }

    // SGS: local relaxation rate from eddy viscosity (dyn_smag pre-pass)
    float om = omega;
    if (nu_t_in != NULL) {
        float nu0 = (1.0f / omega - 0.5f) / 3.0f;
        om = 1.0f / (3.0f * (nu0 + nu_t_in[idx]) + 0.5f);
        if (nu_t_out != NULL) nu_t_out[idx] = nu_t_in[idx];
    }

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
        // FLUID: BGK collision (om = local rate incl. SGS nu_t)
        float usqr = ux*ux + uy*uy + uz*uz;
        for (int q = 0; q < 27; q++) {
            float cu = (float)cx[q]*ux + (float)cy[q]*uy + (float)cz[q]*uz;
            float f_eq = w[q] * rho * (1.0f + 3.0f*cu + 4.5f*cu*cu - 1.5f*usqr);
            fhn[q] = fhn[q] - om * (fhn[q] - f_eq);
        }
        // Guo source term (matches the cumulant kernel's Step 9)
        if (force != NULL) {
            float pf = 1.0f - 0.5f * om;
            for (int q = 0; q < 27; q++) {
                float cxq = (float)cx[q], cyq = (float)cy[q], czq = (float)cz[q];
                float ci_F = cxq*Fx + cyq*Fy + czq*Fz;
                float ci_u = cxq*ux + cyq*uy + czq*uz;
                float u_F  = ux*Fx + uy*Fy + uz*Fz;
                fhn[q] += pf * w[q] * ((ci_F - u_F)*3.0f + ci_u*ci_F*9.0f);
            }
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
            fhn[q] = fhn[q] - om * (fhn[q] - f_eq);
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
        force: Optional['npt.NDArray'] = None,
        nu_t_in: Optional['npt.NDArray'] = None,
        nu_t_out: Optional['npt.NDArray'] = None,
    ) -> None:
        if self._kernel is None:
            self._compile()

        import cupy as cp
        N = Nx * Ny * Nz
        grid = (N + self._block_size - 1) // self._block_size

        nb_arg = needs_bounce if needs_bounce is not None else cp.int32(0)
        fo_arg = force_out if force_out is not None else cp.int32(0)
        f_arg = force if force is not None else cp.int32(0)
        nti_arg = nu_t_in if nu_t_in is not None else cp.int32(0)
        nto_arg = nu_t_out if nu_t_out is not None else cp.int32(0)

        self._kernel(
            (grid,), (self._block_size,),
            (f, rho_out, u_out, node_type,
             bc_rho, bc_ux, bc_uy, bc_uz,
             nb_arg, fo_arg,
             cp.float32(omega),
             cp.int32(Nx), cp.int32(Ny), cp.int32(Nz),
             cp.int32(t_step),
             f_arg, nti_arg, nto_arg),
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


def eso_seed_solid_bounce_ic(xp, f_mem, node_type, get_dir, t0: int) -> None:
    """Seed the implicit-HWBB bounce slots for a FRESH-IC memory build.

    The fixed implicit-HWBB LOAD reads the parity-swapped slot at/toward a
    SOLID neighbor — the previous step's deposit of the cell's own outgoing
    population. At t0 no deposit exists yet; this defines it as the IC's
    incoming population ("the deposit of step t0-1"), which makes the
    step-t0 collision input at bounce links exactly the IC — the same
    convention as the standard path (whose step-t0 collision also consumes
    the IC as-is). With this seed the esoteric and standard trajectories
    are bit-identical from step t0 on (eso_mem_force_twin_gate).

    MUST NOT be called on checkpoint restore: gather/scatter is a pure
    permutation bijection over ALL slots, so scattering the checkpointed
    std f at the restored parity already reproduces the true deposits
    bit-exactly — seeding there would overwrite them with IC-rule values.

    Only slots read exclusively by the kernel's solid branches are written
    (derived in the kernel LOAD comment): pure-fluid cells and fluid-fluid
    slots are untouched, so solid-free cases are bit-neutral.

    Args:
        xp: array module
        f_mem: (27, Nx, Ny, Nz) esoteric memory (modified in place)
        node_type: (Nx, Ny, Nz) int8 node types (1 = SOLID)
        get_dir: callable(std_q) -> IC population of STANDARD direction
                 std_q as a (Nx, Ny, Nz) array OR a python/0-d scalar
                 (uniform IC). Lazy per-direction access keeps the peak at
                 ~2 direction fields (the slab builders chunk the full f).
        t0: start step (parity)
    """
    solid = (node_type == NODE_SOLID)
    if not bool(solid.any()):
        return
    even = (int(t0) % 2 == 0)
    axes = (0, 1, 2)
    for p in range(13):
        i = 2 * p + 1
        ci = (int(CX_ESO[i]), int(CY_ESO[i]), int(CZ_ESO[i]))
        f_i = get_dir(_STD_TO_ESO[i])        # eso dir i   (toward solid)
        f_ip = get_dir(_STD_TO_ESO[i + 1])   # eso dir i+1 (opposite)
        # Write ONLY link pairs whose partner cell is non-solid. The solid
        # branches are read exclusively by FLUID cells, so this covers every
        # actual read — and it keeps every written value sourced from a
        # fluid-cell entry of the IC. That matters for rebuild identity: the
        # slab builders re-seed from gather(already-seeded memory), where
        # SOLID-cell entries are stale junk; seeding solid-to-solid links
        # from them would break the bit match with the direct-IC build
        # (found by the sphere --verify leg).
        part_solid = xp.roll(solid, shift=ci, axis=axes)  # x - c_i is solid
        dn = solid & ~part_solid             # solid s with fluid x = s - c_i
        up = part_solid & ~solid             # fluid x with solid at x - c_i
        slot_dn = (i + 1) if even else i     # read at the solid cell
        slot_up = i if even else (i + 1)     # read at the fluid cell
        if getattr(f_ip, "ndim", 0) == 0 or not hasattr(f_ip, "shape"):
            f_mem[slot_dn][dn] = f_ip        # uniform IC: roll is identity
            f_mem[slot_up][up] = f_i
        else:
            rolled = xp.roll(f_ip, shift=ci, axis=axes)
            f_mem[slot_dn][dn] = rolled[dn]
            f_mem[slot_up][up] = f_i[up]


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


_MEM_FORCE_CACHE = {}


def _mem_force_kernel_source():
    """MEM (momentum-exchange) force on SOLID bodies from ESOTERIC memory.

    Halfway-BB convention (matches mem_force_d3q27 on the standard path:
    F = sum 2 c_i f_i^out over bounce links). In Esoteric Pull the
    post-collision outgoing population f*_i(x_fluid) IS the slot the STORE
    deposited: at completed-step parity T-1,
        toward-solid dir i   -> slot (i+1 if T even else i) AT the solid cell
        toward-solid dir i+1 -> slot (i   if T even else i+1) AT x itself.
    Owned-clip bounds make rank contributions exclusive (allreduce outside).
    atomicAdd accumulation: DIAGNOSTIC tier (log-cadence CL/CD), not physics.
    ASCII only."""
    cxp = [CX_ESO[2 * p + 1] for p in range(13)]
    cyp = [CY_ESO[2 * p + 1] for p in range(13)]
    czp = [CZ_ESO[2 * p + 1] for p in range(13)]
    fmt = lambda a: ", ".join(str(int(v)) for v in a)
    return f"""
extern "C" __global__ void eso_mem_force(
    const float* __restrict__ f,
    const signed char* __restrict__ nt,
    double* __restrict__ force_out,
    const int Nx, const int Ny, const int Nz,
    const int t_next,
    const int cx0, const int cx1,
    const int cy0, const int cy1,
    const int cz0, const int cz1)
{{
    const int CXP[13] = {{{fmt(cxp)}}};
    const int CYP[13] = {{{fmt(cyp)}}};
    const int CZP[13] = {{{fmt(czp)}}};
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long N = (long long)Nx * Ny * Nz;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    if (idx < N) {{
        int gz = (int)(idx % Nz);
        int gy = (int)((idx / Nz) % Ny);
        int gx = (int)(idx / ((long long)Nz * Ny));
        bool owned = (gx >= cx0 && gx < cx1 && gy >= cy0 && gy < cy1 &&
                      gz >= cz0 && gz < cz1);
        if (owned && nt[idx] != 1) {{
            const int even_next = ((t_next & 1) == 0);
            for (int p = 0; p < 13; p++) {{
                int i = 2 * p + 1;
                int sx = gx + CXP[p]; sx += (sx < 0) ? Nx : ((sx >= Nx) ? -Nx : 0);
                int sy = gy + CYP[p]; sy += (sy < 0) ? Ny : ((sy >= Ny) ? -Ny : 0);
                int sz = gz + CZP[p]; sz += (sz < 0) ? Nz : ((sz >= Nz) ? -Nz : 0);
                long long nb = ((long long)sx * Ny + sy) * Nz + sz;
                if (nt[nb] == 1) {{
                    int slot = even_next ? (i + 1) : i;
                    double fq = 2.0 * (double)f[(long long)slot * N + nb];
                    fx += fq * CXP[p]; fy += fq * CYP[p]; fz += fq * CZP[p];
                }}
                int mx = gx - CXP[p]; mx += (mx < 0) ? Nx : ((mx >= Nx) ? -Nx : 0);
                int my = gy - CYP[p]; my += (my < 0) ? Ny : ((my >= Ny) ? -Ny : 0);
                int mz = gz - CZP[p]; mz += (mz < 0) ? Nz : ((mz >= Nz) ? -Nz : 0);
                long long mb = ((long long)mx * Ny + my) * Nz + mz;
                if (nt[mb] == 1) {{
                    int slot2 = even_next ? i : (i + 1);
                    double fq = 2.0 * (double)f[(long long)slot2 * N + idx];
                    fx -= fq * CXP[p]; fy -= fq * CYP[p]; fz -= fq * CZP[p];
                }}
            }}
        }}
    }}
    __shared__ double sh[3 * 256];
    int t = threadIdx.x;
    sh[t] = fx; sh[256 + t] = fy; sh[512 + t] = fz;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {{
        if (t < st) {{
            sh[t] += sh[t + st];
            sh[256 + t] += sh[256 + t + st];
            sh[512 + t] += sh[512 + t + st];
        }}
        __syncthreads();
    }}
    if (t == 0) {{
        atomicAdd(&force_out[0], sh[0]);
        atomicAdd(&force_out[1], sh[256]);
        atomicAdd(&force_out[2], sh[512]);
    }}
}}
"""


def eso_mem_force(xp, f_mem, node_type_flat, t_next: int, clip_bounds):
    """(Fx, Fy, Fz) numpy f64 over OWNED fluid cells (see kernel doc)."""
    import numpy as _np
    if "k" not in _MEM_FORCE_CACHE:
        src = _mem_force_kernel_source()
        assert all(ord(ch) < 128 for ch in src)
        _MEM_FORCE_CACHE["k"] = xp.RawKernel(src, "eso_mem_force")
    Nx, Ny, Nz = f_mem.shape[1:]
    out = xp.zeros(3, xp.float64)
    N = Nx * Ny * Nz
    bs = 256
    cb = clip_bounds
    _MEM_FORCE_CACHE["k"](
        ((N + bs - 1) // bs,), (bs,),
        (f_mem, node_type_flat, out,
         _np.int32(Nx), _np.int32(Ny), _np.int32(Nz), _np.int32(t_next),
         _np.int32(cb[0][0]), _np.int32(cb[0][1]),
         _np.int32(cb[1][0]), _np.int32(cb[1][1]),
         _np.int32(cb[2][0]), _np.int32(cb[2][1])))
    return xp.asnumpy(out)


_REGION_KERNEL_CACHE = {}


def _region_kernel_source():
    """Single-launch region gather/scatter (M5 pass 4). The python fancy-
    index path costs 27 advanced-indexing kernels + 13 index-array builds
    per call; at D40 the coupling block gathers alone move ~GB per call.
    These kernels do the identical permutation in ONE launch (exact copies,
    bit-equal by construction). ASCII only (nvrtc POSIX locale)."""
    cxp = [CX_ESO[2 * p + 1] for p in range(13)]
    cyp = [CY_ESO[2 * p + 1] for p in range(13)]
    czp = [CZ_ESO[2 * p + 1] for p in range(13)]
    stdi = [_STD_TO_ESO[2 * p + 1] for p in range(13)]
    stdip = [_STD_TO_ESO[2 * p + 2] for p in range(13)]
    fmt = lambda a: ", ".join(str(int(v)) for v in a)
    tables = f"""
    const int CXP[13] = {{{fmt(cxp)}}};
    const int CYP[13] = {{{fmt(cyp)}}};
    const int CZP[13] = {{{fmt(czp)}}};
    const int STDI[13] = {{{fmt(stdi)}}};
    const int STDIP[13] = {{{fmt(stdip)}}};
"""
    head = """
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long M = (long long)nxr * nyr * nzr;
    if (idx >= M) return;
    int cz = (int)(idx % nzr);
    int cy = (int)((idx / nzr) % nyr);
    int cx = (int)(idx / ((long long)nzr * nyr));
    int gx = (x0 + cx * xs) % Nx;
    int gy = (y0 + cy * ys) % Ny;
    int gz = (z0 + cz * zs) % Nz;
    const long long N = (long long)Nx * Ny * Nz;
    const long long b0 = ((long long)gx * Ny + gy) * Nz + gz;
"""
    loop_pre = """
    for (int p = 0; p < 13; p++) {
        int i = 2 * p + 1;
        int sx = gx + CXP[p]; if (sx < 0) sx += Nx; if (sx >= Nx) sx -= Nx;
        int sy = gy + CYP[p]; if (sy < 0) sy += Ny; if (sy >= Ny) sy -= Ny;
        int sz = gz + CZP[p]; if (sz < 0) sz += Nz; if (sz >= Nz) sz -= Nz;
        const long long bs = ((long long)sx * Ny + sy) * Nz + sz;
        const long long o_i = (long long)STDI[p] * M + idx;
        const long long o_ip = (long long)STDIP[p] * M + idx;
"""
    args = """
    const int Nx, const int Ny, const int Nz,
    const int x0, const int xs, const int nxr,
    const int y0, const int ys, const int nyr,
    const int z0, const int zs, const int nzr,
    const int is_odd)
"""
    gather = ("""
extern "C" __global__ void eso_gather_region(
    const float* __restrict__ f, float* __restrict__ out,""" + args + "{"
        + tables + head + """
    out[idx] = f[b0];
""" + loop_pre + """
        if (is_odd) {
            out[o_i]  = f[(long long)i * N + b0];
            out[o_ip] = f[(long long)(i + 1) * N + bs];
        } else {
            out[o_i]  = f[(long long)(i + 1) * N + b0];
            out[o_ip] = f[(long long)i * N + bs];
        }
    }
}
""")
    scatter = ("""
extern "C" __global__ void eso_scatter_region(
    float* __restrict__ f, const float* __restrict__ vals,""" + args + "{"
        + tables + head + """
    f[b0] = vals[idx];
""" + loop_pre + """
        if (is_odd) {
            f[(long long)i * N + b0]        = vals[o_i];
            f[(long long)(i + 1) * N + bs]  = vals[o_ip];
        } else {
            f[(long long)(i + 1) * N + b0]  = vals[o_i];
            f[(long long)i * N + bs]        = vals[o_ip];
        }
    }
}
""")
    # skip-solid variant (repair track patch 12): a SOLID cell's std-ordered
    # population entries alias the LIVE implicit-HWBB deposit slots of its
    # fluid neighbours (proven by f1d_scatter_alias_proof), so a coupling
    # write of FOREIGN values (F2C restriction / C2F interpolation) at solid
    # cells bounces garbage off the wall on the next step. Faithful
    # same-state transfers (MPI halo bands) must keep writing them — the
    # deposits are real payload there — hence a separate kernel, opt-in.
    scatter_ns = ("""
extern "C" __global__ void eso_scatter_region_ns(
    float* __restrict__ f, const float* __restrict__ vals,
    const signed char* __restrict__ nt,""" + args + "{"
        + tables + head + """
    if (nt[b0] == 1) return;
    f[b0] = vals[idx];
""" + loop_pre + """
        if (is_odd) {
            f[(long long)i * N + b0]        = vals[o_i];
            f[(long long)(i + 1) * N + bs]  = vals[o_ip];
        } else {
            f[(long long)(i + 1) * N + b0]  = vals[o_i];
            f[(long long)i * N + bs]        = vals[o_ip];
        }
    }
}
""")
    return gather + scatter + scatter_ns


def _get_region_kernels(cp):
    if "k" not in _REGION_KERNEL_CACHE:
        src = _region_kernel_source()
        assert all(ord(ch) < 128 for ch in src)
        mod = cp.RawModule(code=src)
        _REGION_KERNEL_CACHE["k"] = (
            mod.get_function("eso_gather_region"),
            mod.get_function("eso_scatter_region"),
            mod.get_function("eso_scatter_region_ns"))
    return _REGION_KERNEL_CACHE["k"]


def _region_kernel_params(region, shape):
    """(start, step, count)*3 for the kernel, or None -> python fallback."""
    ps = []
    for sl, N in zip(region, shape):
        if not isinstance(sl, slice):
            return None
        if (sl.start is not None and sl.start < 0) or \
           (sl.stop is not None and sl.stop > N):
            return None                       # out-of-bounds wrap semantics
        start, stop, step = sl.indices(N)
        if step <= 0:
            return None
        ps += [start, step, max(0, (stop - start + step - 1) // step)]
    return ps


def esoteric_gather_std_region(xp, f_mem: 'npt.NDArray', t_step: int,
                               region) -> 'npt.NDArray':
    """Gather the physical f (STANDARD ordering) on a spatial region only.

    region: 3-tuple of slices into the (Nx, Ny, Nz) grid (strided OK).
    Returns (27, *region_shape). Equals esoteric_gather_std(...)[:, region].
    """
    shape = f_mem.shape[1:]
    if xp.__name__ == 'cupy' and f_mem.dtype == xp.float32 \
            and f_mem.flags.c_contiguous:
        ps = _region_kernel_params(region, shape)
        if ps is not None:
            import numpy as _np
            gk, _, _ = _get_region_kernels(xp)
            out = xp.empty((27, ps[2], ps[5], ps[8]), xp.float32)
            M = int(ps[2]) * int(ps[5]) * int(ps[8])
            if M:
                bs_ = 256
                gk(((M + bs_ - 1) // bs_,), (bs_,),
                   (f_mem, out,
                    _np.int32(shape[0]), _np.int32(shape[1]),
                    _np.int32(shape[2]),
                    *(_np.int32(v) for v in ps), _np.int32(t_step & 1)))
            return out
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
                                region, skip_solid_nt=None) -> None:
    """Scatter physical values (STANDARD ordering, region-shaped) into the
    Esoteric memory IN PLACE, so the next LOAD at `t_step` reads them.

    Inverse of esoteric_gather_std_region on the same region: writes the
    local slot at region and the paired slot at region+c_i (wrapped).

    skip_solid_nt (patch 12): raveled int8 node-type of the FULL domain.
    When given, entries belonging to SOLID region cells are NOT written.
    A solid cell's std-ordered population entries alias the LIVE
    implicit-HWBB deposit slots of its fluid neighbours (the deposit IS
    the population streamed into the wall cell — f1d proof), so coupling
    writes of FOREIGN values (F2C restriction, C2F interpolation) at
    solid cells put garbage on the wall bounce. Pass it from every
    coupling scatter. Do NOT pass it for faithful same-parity transfers
    of an identical wall state (MPI halo bands): there the solid entries
    carry the real deposits and must be written.
    """
    shape = f_mem.shape[1:]
    if xp.__name__ == 'cupy' and f_mem.dtype == xp.float32 \
            and f_mem.flags.c_contiguous and values.dtype == xp.float32:
        ps = _region_kernel_params(region, shape)
        if ps is not None:
            import numpy as _np
            _, sk, skns = _get_region_kernels(xp)
            vals = xp.ascontiguousarray(values)
            M = int(ps[2]) * int(ps[5]) * int(ps[8])
            if M:
                bs_ = 256
                dims = (_np.int32(shape[0]), _np.int32(shape[1]),
                        _np.int32(shape[2]))
                tail = (*(_np.int32(v) for v in ps), _np.int32(t_step & 1))
                if skip_solid_nt is None:
                    sk(((M + bs_ - 1) // bs_,), (bs_,),
                       (f_mem, vals, *dims, *tail))
                else:
                    nt = xp.ascontiguousarray(
                        skip_solid_nt.ravel().astype(xp.int8, copy=False))
                    skns(((M + bs_ - 1) // bs_,), (bs_,),
                         (f_mem, vals, nt, *dims, *tail))
            return
    ix0, iy0, iz0 = _region_ix(xp, shape, region)
    even = (t_step % 2 == 0)
    keep = None
    if skip_solid_nt is not None:
        nt3 = skip_solid_nt.reshape(shape)
        keep = nt3[ix0, iy0, iz0] != 1          # broadcast to region shape

    def _put(dst, sl, src):
        if keep is None:
            dst[sl] = src
        else:
            dst[sl] = xp.where(keep, src, dst[sl])

    _put(f_mem[0], (ix0, iy0, iz0), values[_STD_TO_ESO[0]])
    for p in range(13):
        i = 2 * p + 1
        ci = (CX_ESO[i], CY_ESO[i], CZ_ESO[i])
        ixs, iys, izs = _region_ix(xp, shape, region, shift=ci)
        std_i, std_ip1 = _STD_TO_ESO[i], _STD_TO_ESO[i + 1]
        if even:
            _put(f_mem[i + 1], (ix0, iy0, iz0), values[std_i])
            _put(f_mem[i], (ixs, iys, izs), values[std_ip1])
        else:
            _put(f_mem[i], (ix0, iy0, iz0), values[std_i])
            _put(f_mem[i + 1], (ixs, iys, izs), values[std_ip1])


# ============================================================
# Esoteric macro pre-pass kernel: LOAD + (rho, u), no collide/store.
# Used by the WALE/dyn_smag pre-pass and the ALM 2-pass, which need
# the current-step macroscopic fields BEFORE the collision launch.
# LOAD replica includes the implicit-HWBB bounce branches (patch 12,
# f1e gate): without node_type the wall-adjacent bounced populations
# came from the parity-dead slot (one substep STALE), giving the SGS
# pre-pass a systematically wrong wall-layer u on the esoteric path
# only. Solid cells still produce garbage (consumers mask) -- same
# as the standard MacroKernel pre-pass.
# ============================================================

_ESOTERIC_MACRO_KERNEL = r'''
extern "C" __global__
void esoteric_macro_d3q27(
    const float* __restrict__ f,        // (27, N) Esoteric memory
    const char*  __restrict__ node_type, // (N,) 1 = SOLID
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
        // bounce addressing: verbatim from the fused kernel LOAD
        long long mx = (ix - cx[i] + Nx) % Nx;
        long long my = (iy - cy[i] + Ny) % Ny;
        long long mz = (iz - cz[i] + Nz) % Nz;
        long long j_ip = mx * (long long)Ny * Nz + my * (long long)Nz + mz;
        bool nb_dn = node_type[j_i]  == 1;
        bool nb_up = node_type[j_ip] == 1;
        if (is_odd) {
            fhn[i]   = nb_up ? f[(i+1) * N + idx] : f[i     * N + idx];
            fhn[i+1] = nb_dn ? f[i     * N + j_i] : f[(i+1) * N + j_i];
        } else {
            fhn[i]   = nb_up ? f[i     * N + idx] : f[(i+1) * N + idx];
            fhn[i+1] = nb_dn ? f[(i+1) * N + j_i] : f[i     * N + j_i];
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

    def launch(self, f: 'npt.NDArray', node_type: 'npt.NDArray',
               rho_out: 'npt.NDArray', u_out: 'npt.NDArray',
               Nx: int, Ny: int, Nz: int, t_step: int) -> None:
        import cupy as cp
        if self._kernel is None:
            self._kernel = cp.RawKernel(
                _ESOTERIC_MACRO_KERNEL, "esoteric_macro_d3q27",
                options=('--use_fast_math',))
        N = Nx * Ny * Nz
        grid = (N + self._block_size - 1) // self._block_size
        self._kernel((grid,), (self._block_size,),
                     (f, node_type, rho_out, u_out,
                      cp.int32(Nx), cp.int32(Ny), cp.int32(Nz),
                      cp.int32(t_step)))
