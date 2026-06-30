"""
Force Calculation Module using Momentum Exchange Method (MEM)

This module implements the Momentum Exchange Method for computing
hydrodynamic forces on solid obstacles in LBM simulations.

Supports both 2D and 3D domains.

Physical Principle:
==================
The momentum exchange method calculates forces by summing the momentum
transferred at fluid-solid boundary links during bounce-back.

For half-way bounce-back, at each boundary link:
    F_link = 2 * c_i * f_i^post(x_fluid)

where:
    c_i = lattice velocity vector  [Δx/Δt]
    f_i^post = post-collision distribution  [dimensionless]
    x_fluid = fluid node adjacent to solid

Total force on obstacle:
    F_total = Σ F_link (sum over all boundary links)

Force Coefficients:
==================
2D:
    Drag coefficient: C_D = F_x / (0.5 * ρ * U² * D)
    Lift coefficient: C_L = F_y / (0.5 * ρ * U² * D)
    Reference area: D (per unit span)

3D:
    Drag coefficient: C_D = F_x / (0.5 * ρ * U² * A)
    Lift coefficient: C_L = F_y / (0.5 * ρ * U² * A)
    Reference area: A = D * L_z

References:
==========
- Ladd, J. Fluid Mech. 271, 1994 (original MEM)
- Mei et al., Phys. Fluids 14, 2002 (improved MEM)
- Kruger et al., "The Lattice Boltzmann Method", Springer 2017, Ch. 11

Author: LBM Development Team
Date: 2026-02
"""

import os
import csv
from datetime import datetime
from typing import TYPE_CHECKING, Tuple, Optional, Dict, List, Any, Union

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class MomentumExchangeForce:
    """Momentum Exchange Method for Force Calculation
    
    Computes hydrodynamic forces on solid obstacles using the momentum
    exchange at fluid-solid boundary links.
    
    Supports both 2D and 3D domains.
    
    Algorithm:
    ----------
    1. Identify boundary links: fluid nodes with solid neighbors
    2. For each boundary link (fluid node x_f, direction i → solid):
       F_link = 2 * c_i * f_i^post(x_f)
    3. Sum all F_link to get total force
    
    Example:
        >>> force_calc = MomentumExchangeForce(xp, lattice, solid_mask)
        >>> Fx, Fy = force_calc.compute(f_post)  # 2D
        >>> Fx, Fy, Fz = force_calc.compute(f_post)  # 3D
    """
    
    def __init__(self, xp: 'ModuleType',
                 lattice: 'Lattice',
                 solid_mask: 'npt.NDArray',
                 wall_bc: Optional['HalfwayBounceBack'] = None,
                 save_link_forces: bool = False) -> None:
        """Initialize momentum exchange force calculator

        Args:
            xp: Array module (numpy or cupy)
            lattice: Lattice model (D2Q9, D3Q27, etc.)
            solid_mask: Boolean solid mask, True = solid
            wall_bc: Optional wall BC instance. Two roles:
                1. Reuse its precomputed `needs_bounce` mask (HWBB or IBB).
                2. If it has a `q_fraction` attribute (InterpolatedBounceBack),
                   switch to the Bouzidi-aware per-link force formula.
                   Otherwise fall back to the HWBB formula 2·c_i·f̃_i.
            save_link_forces: If True, precompute flat per-link metadata
                (fluid node coords, direction, wall-hit point, normal) and
                allow `compute_per_link()` calls for surface-distribution
                post-processing. Adds a small setup cost; zero run-time cost
                unless `compute_per_link()` is invoked. 2D only.
        """
        self.xp = xp
        self.lattice = lattice
        self.dim = lattice.dim
        self.Q = lattice.Q
        self.c = xp.asarray(lattice.c, dtype=xp.float64)  # (dim, Q)
        self.opp = xp.asarray(lattice.opp)
        self.solid_mask = xp.asarray(solid_mask, dtype=bool)
        self.shape = solid_mask.shape

        # Reuse boundary link info from wall BC if provided
        if wall_bc is not None and hasattr(wall_bc, 'needs_bounce'):
            self.needs_bounce = wall_bc.needs_bounce
        else:
            self._precompute_boundary_links()

        # Count boundary links for info
        self.n_boundary_links = int(xp.sum(self.needs_bounce))

        # ── IBB-aware force formula (when wall_bc is InterpolatedBounceBack) ──
        # Bouzidi MEM (Caiazzo-Junk, Kruger Ch. 11):
        #   q ≥ 1/2: F_link = c_i · [(2q+1)/(2q) f̃_i + (2q-1)/(2q) f̃_ī]
        #   q < 1/2: F_link = c_i · [(1+2q) f̃_i(x_f) + (1-2q) f̃_i(x_f − c_i)]
        # In 3D sparse mode the dense q_fraction is released — we use the
        # sparse triple (link_cell, link_dir, link_q) instead.
        self._q_fraction = None
        self._ibb_aware = False
        self._wall_bc_ref = wall_bc
        if wall_bc is not None and hasattr(wall_bc, 'q_fraction'):
            sparse_ok = all(
                hasattr(wall_bc, n) for n in
                ('link_cell', 'link_dir', 'link_q', 'n_links')
            )
            dense_q = getattr(wall_bc, 'q_fraction', None)
            if dense_q is not None or sparse_ok:
                self._q_fraction = dense_q   # may be None in sparse 3D mode
                self._ibb_aware = True

        # Try CUDA kernel
        # 3D HWBB         → MEMForceKernelD3Q27 (existing)
        # 2D HWBB         → MEMForceKernelD2Q9  (new)
        # 2D IBB          → MEMForceIBBKernelD2Q9 (new) — used by _compute_ibb path
        self._cuda_kernel = None
        self._cuda_kernel_ibb = None
        if xp.__name__ == 'cupy':
            if self.dim == 3:
                try:
                    from src.kernels.mem_force_d3q27 import MEMForceKernelD3Q27
                    self._cuda_kernel = MEMForceKernelD3Q27()
                except Exception:
                    pass
            elif self.dim == 2 and self.Q == 9:
                try:
                    from src.kernels.mem_force_d2q9 import (
                        MEMForceKernelD2Q9, MEMForceIBBKernelD2Q9,
                    )
                    self._cuda_kernel = MEMForceKernelD2Q9()
                    if self._ibb_aware:
                        self._cuda_kernel_ibb = MEMForceIBBKernelD2Q9()
                except Exception:
                    pass

        # Storage for time history
        self.force_history: List[Dict[str, Any]] = []

        # ── Per-link metadata for surface-distribution post-processing ──
        self._save_link_forces = bool(save_link_forces) and self.dim == 2
        if self._save_link_forces:
            self._precompute_link_metadata()
    
    def _precompute_boundary_links(self) -> None:
        """Precompute fluid-solid boundary links
        
        For each direction i, identify fluid nodes whose neighbor
        in direction i is solid. These are the boundary links.
        
        needs_bounce[i, x, y(, z)] = True means:
            - (x, y(, z)) is a fluid node
            - neighbor in direction i is a solid node
            - Distribution f_i at this node bounces back
        """
        xp = self.xp
        c = self.c.astype(xp.int32)
        solid = self.solid_mask
        
        if self.dim == 2:
            self._precompute_boundary_links_2d(c, solid)
        else:
            self._precompute_boundary_links_3d(c, solid)
    
    def _precompute_boundary_links_2d(self, c: 'npt.NDArray', solid: 'npt.NDArray') -> None:
        """Precompute boundary links for 2D domain"""
        xp = self.xp
        Nx, Ny = self.shape
        self.needs_bounce = xp.zeros((self.Q,) + self.shape, dtype=bool)
        
        for i in range(self.Q):
            if i == 0:  # Rest direction
                continue
            
            cx, cy = int(c[0, i]), int(c[1, i])
            
            # Shift solid mask by -c_i
            shifted_solid = xp.roll(
                xp.roll(solid, -cx, axis=0),
                -cy, axis=1)
            
            # Boundary link: fluid node where neighbor is solid
            self.needs_bounce[i] = (~solid) & shifted_solid
    
    def _precompute_boundary_links_3d(self, c: 'npt.NDArray', solid: 'npt.NDArray') -> None:
        """Precompute boundary links for 3D domain"""
        xp = self.xp
        Nx, Ny, Nz = self.shape
        self.needs_bounce = xp.zeros((self.Q,) + self.shape, dtype=bool)
        
        for i in range(self.Q):
            if i == 0:
                continue
            
            cx, cy, cz = int(c[0, i]), int(c[1, i]), int(c[2, i])
            
            shifted_solid = xp.roll(
                xp.roll(
                    xp.roll(solid, -cx, axis=0),
                    -cy, axis=1),
                -cz, axis=2)
            
            self.needs_bounce[i] = (~solid) & shifted_solid
    
    def compute(self, f_post: 'npt.NDArray') -> Tuple:
        """Compute total force on solid using momentum exchange.

        Mathematical Formula:
        ---------------------
        Per boundary link (fluid node x_f, direction c_i pointing into solid):

            HWBB (or IBB with q=0.5 sentinel):
                F_link = 2 · c_i · f̃_i(x_f)

            Bouzidi IBB (linear), q < 1/2:
                F_link = c_i · [(1+2q) · f̃_i(x_f) + (1-2q) · f̃_i(x_f − c_i)]

            Bouzidi IBB (linear), q ≥ 1/2:
                F_link = c_i · [((2q+1)/(2q)) · f̃_i(x_f)
                                + ((2q-1)/(2q)) · f̃_ī(x_f)]

        The IBB-aware branches are taken only when a
        `InterpolatedBounceBack` was passed as `wall_bc` at construction
        (2D only). Otherwise the standard HWBB formula is used.

        Both IBB formulas reduce to 2·c_i·f̃_i at q = 1/2, so the q=0.5
        sentinel on "unknown" links is consistent with HWBB treatment.

        Args:
            f_post: Post-collision distribution, shape (Q, Nx, Ny)
                    or (Q, Nx, Ny, Nz)  [dimensionless].
                    MUST be post-collision, pre-streaming.

        Returns:
            (Fx, Fy) for 2D or (Fx, Fy, Fz) for 3D  [lattice units]
        """
        if self._ibb_aware:
            return self._compute_ibb(f_post)

        # Try CUDA kernel (HWBB) — both 2D and 3D
        if self._cuda_kernel is not None:
            N = 1
            for d in f_post.shape[1:]:
                N *= d
            return self._cuda_kernel.compute(f_post, self.needs_bounce, N)

        # HWBB fallback: Python loop (no CUDA)
        xp = self.xp
        c = self.c

        forces = []
        for d in range(self.dim):
            F_d = 0.0
            for i in range(1, self.Q):
                f_boundary = f_post[i] * self.needs_bounce[i]
                F_d += 2.0 * float(c[d, i]) * float(xp.sum(f_boundary))
            forces.append(F_d)

        return tuple(forces)

    def _compute_ibb(self, f_post: 'npt.NDArray') -> Tuple:
        """Bouzidi-aware per-link momentum exchange (2D and 3D).

        See `compute()` docstring for the formula. The q>=1/2 branch needs
        only local f̃_i and f̃_ī; the q<1/2 branch also needs the upstream
        post-collision value f̃_i(x_f − c_i).

        Three paths:
            * 2D dense  (CUDA kernel if available)  ← MEMForceIBBKernelD2Q9
            * 3D / 2D sparse (link_cell + link_dir + link_q)
            * dense fallback (xp.roll over full f_post)
        """
        # 2D CUDA fast path.
        if self._cuda_kernel_ibb is not None and self.dim == 2:
            Nx, Ny = f_post.shape[1], f_post.shape[2]
            return self._cuda_kernel_ibb.compute(
                f_post, self.needs_bounce, self._q_fraction, Nx, Ny,
            )

        # Sparse path (used by 3D after _build_sparse_links releases q_fraction;
        # also works for any wall_bc that exposes link_cell/link_dir/link_q).
        wb = self._wall_bc_ref
        sparse_ok = (
            wb is not None
            and getattr(wb, 'n_links', 0) > 0
            and getattr(wb, 'link_cell', None) is not None
            and getattr(wb, 'link_dir', None) is not None
            and getattr(wb, 'link_q', None) is not None
        )
        if sparse_ok:
            return self._compute_ibb_sparse(f_post)

        # Dense Python fallback (q_fraction array required).
        xp = self.xp
        c = self.c
        opp = self.opp
        q_frac = self._q_fraction
        if q_frac is None:
            raise RuntimeError(
                "IBB force compute: neither sparse links nor dense q_fraction "
                "are available on wall_bc."
            )

        F = [0.0] * self.dim
        for i in range(1, self.Q):
            mask = self.needs_bounce[i]
            if not bool(xp.any(mask)):
                continue

            i_opp = int(opp[i])
            q_i = q_frac[i]

            # q ≥ 1/2 — local only
            mask_ge = mask & (q_i >= 0.5)
            if bool(xp.any(mask_ge)):
                q_ge = q_i[mask_ge]
                two_q = 2.0 * q_ge
                f_link_sum = xp.sum(
                    ((two_q + 1.0) / two_q) * f_post[i][mask_ge]
                    + ((two_q - 1.0) / two_q) * f_post[i_opp][mask_ge]
                )
                for d in range(self.dim):
                    F[d] += float(c[d, i]) * float(f_link_sum)

            # q < 1/2 — needs upstream f̃_i at x_f − c_i
            mask_lt = mask & (q_i < 0.5)
            if bool(xp.any(mask_lt)):
                if self.dim == 2:
                    cx, cy = int(c[0, i]), int(c[1, i])
                    f_upstream = xp.roll(
                        xp.roll(f_post[i], +cx, axis=0), +cy, axis=1,
                    )
                else:  # 3D
                    cx, cy, cz = (int(c[0, i]), int(c[1, i]), int(c[2, i]))
                    f_upstream = xp.roll(
                        xp.roll(
                            xp.roll(f_post[i], +cx, axis=0),
                            +cy, axis=1),
                        +cz, axis=2,
                    )
                q_lt = q_i[mask_lt]
                f_link_sum = xp.sum(
                    (1.0 + 2.0 * q_lt) * f_post[i][mask_lt]
                    + (1.0 - 2.0 * q_lt) * f_upstream[mask_lt]
                )
                for d in range(self.dim):
                    F[d] += float(c[d, i]) * float(f_link_sum)

        return tuple(F)

    def _compute_ibb_sparse(self, f_post: 'npt.NDArray') -> Tuple:
        """Bouzidi MEM, sparse-link variant (no full xp.roll). Works in 2D & 3D.

        Per link k = (cell_k, dir_k, q_k):
            ix, iy[, iz] = unflatten(cell_k)        # C-contiguous
            f̃_i           = f_post[dir_k, cell_k]
            f̃_ī           = f_post[opp[dir_k], cell_k]
            f̃_i(x_f-c_i) = f_post[dir_k, upstream_k]
        Vectorised across all links.
        """
        xp = self.xp
        wb = self._wall_bc_ref
        link_cell = wb.link_cell           # (n_links,) int32
        link_dir  = wb.link_dir.astype(xp.int32)
        link_q    = wb.link_q              # (n_links,) float32

        f_flat = f_post.reshape(self.Q, -1)
        opp_dir = self.opp[link_dir]

        fp_i  = f_flat[link_dir, link_cell]
        fp_oi = f_flat[opp_dir,  link_cell]

        # Reconstruct (ix, iy[, iz]) → upstream cell index, with periodic wrap.
        c_int = self.c.astype(xp.int32)         # (dim, Q)
        if self.dim == 2:
            Nx, Ny = self.shape
            ix = link_cell // Ny
            iy = link_cell - ix * Ny
            cx = c_int[0, link_dir]
            cy = c_int[1, link_dir]
            up_ix = (ix - cx) % Nx
            up_iy = (iy - cy) % Ny
            up_cell = up_ix * Ny + up_iy
        else:
            Nx, Ny, Nz = self.shape
            N_yz = Ny * Nz
            ix = link_cell // N_yz
            rem = link_cell - ix * N_yz
            iy = rem // Nz
            iz = rem - iy * Nz
            cx = c_int[0, link_dir]
            cy = c_int[1, link_dir]
            cz = c_int[2, link_dir]
            up_ix = (ix - cx) % Nx
            up_iy = (iy - cy) % Ny
            up_iz = (iz - cz) % Nz
            up_cell = up_ix * N_yz + up_iy * Nz + up_iz
        fp_i_up = f_flat[link_dir, up_cell]

        two_q = 2.0 * link_q
        F_link = xp.where(
            link_q >= 0.5,
            ((two_q + 1.0) / two_q) * fp_i + ((two_q - 1.0) / two_q) * fp_oi,
            (1.0 + two_q) * fp_i + (1.0 - two_q) * fp_i_up,
        )

        c_f = self.c                           # (dim, Q) float64
        forces = []
        for d in range(self.dim):
            cd = c_f[d, link_dir]
            forces.append(float(xp.sum(cd * F_link)))
        return tuple(forces)

    # =====================================================================
    # Per-link metadata + per-link force output (for surface distribution)
    # =====================================================================

    def _precompute_link_metadata(self) -> None:
        """Build flat per-link metadata for 2D surface-distribution output.

        Produces host (numpy) arrays of length N_links, in a fixed order:

            link_dir:    direction index i             (int32)
            link_xf,yf:  fluid-node lattice coords     (int32)
            link_cx,cy:  lattice velocity c_i          (float32)
            link_xw,yw:  wall-hit point  x_f + q·c_i   (float32)
            link_nx,ny:  inward wall-normal = +c_i/|c| (float32)
                         (c_i points from fluid into solid, so +c_i/|c|
                          is the inward body-surface normal; this sign
                          makes F_link · n̂ positive at stagnation)
            link_dl:     reference link "area"         (float32)
                         = 1 / |c_i|  (so ∫ dl ≈ perimeter)

        Wall-hit point uses the IBB q_fraction if available; otherwise q=0.5
        (HWBB assumption). Normal is approximated as +c_i/|c_i| — the
        unit-vector from fluid toward solid (= inward wall normal) —
        which averages out when binning links for Cp/Cf distributions.
        """
        import numpy as _np

        # Pull needs_bounce to host (may be cupy)
        nb_np = (self.needs_bounce.get()
                 if hasattr(self.needs_bounce, 'get')
                 else _np.asarray(self.needs_bounce))
        c_np = (self.c.get() if hasattr(self.c, 'get')
                else _np.asarray(self.c)).astype(_np.float32)

        if self._q_fraction is not None:
            q_np = (self._q_fraction.get()
                    if hasattr(self._q_fraction, 'get')
                    else _np.asarray(self._q_fraction)).astype(_np.float32)
        else:
            # HWBB: all q=0.5
            q_np = _np.full_like(nb_np, 0.5, dtype=_np.float32)

        dirs, xs, ys = [], [], []
        for i in range(1, self.Q):
            idx = _np.argwhere(nb_np[i])
            if idx.size == 0:
                continue
            dirs.append(_np.full(idx.shape[0], i, dtype=_np.int32))
            xs.append(idx[:, 0].astype(_np.int32))
            ys.append(idx[:, 1].astype(_np.int32))

        if not dirs:
            # No links at all
            self.link_dir = _np.zeros(0, dtype=_np.int32)
            self.link_xf = _np.zeros(0, dtype=_np.int32)
            self.link_yf = _np.zeros(0, dtype=_np.int32)
            self.link_cx = self.link_cy = _np.zeros(0, dtype=_np.float32)
            self.link_xw = self.link_yw = _np.zeros(0, dtype=_np.float32)
            self.link_nx = self.link_ny = _np.zeros(0, dtype=_np.float32)
            self.link_dl = _np.zeros(0, dtype=_np.float32)
            return

        link_dir = _np.concatenate(dirs)
        link_xf = _np.concatenate(xs)
        link_yf = _np.concatenate(ys)
        link_cx = c_np[0, link_dir]
        link_cy = c_np[1, link_dir]
        link_q = q_np[link_dir, link_xf, link_yf]

        link_xw = link_xf.astype(_np.float32) + link_q * link_cx
        link_yw = link_yf.astype(_np.float32) + link_q * link_cy
        c_norm = _np.hypot(link_cx, link_cy)
        link_nx = link_cx / c_norm          # inward normal = +c_i/|c|
        link_ny = link_cy / c_norm
        link_dl = (1.0 / c_norm).astype(_np.float32)  # proxy surface weight

        # Per-link lattice weight w_i (needed by post-processor to subtract
        # the quiescent-fluid baseline 2·w_i·ρ_∞ so Cp matches the textbook
        # definition with Cp=0 at free stream).
        w_np = (self.lattice.w.get() if hasattr(self.lattice.w, 'get')
                else _np.asarray(self.lattice.w)).astype(_np.float32)
        link_w = w_np[link_dir]

        self.link_dir = link_dir
        self.link_xf = link_xf
        self.link_yf = link_yf
        self.link_cx = link_cx.astype(_np.float32)
        self.link_cy = link_cy.astype(_np.float32)
        self.link_xw = link_xw
        self.link_yw = link_yw
        self.link_nx = link_nx.astype(_np.float32)
        self.link_ny = link_ny.astype(_np.float32)
        self.link_dl = link_dl
        self.link_w  = link_w

    def compute_per_link(self, f_post: 'npt.NDArray'
                         ) -> Tuple['npt.NDArray', 'npt.NDArray']:
        """Return (F_link, ρ_link) per boundary link.

        F_link: shape (N_links, dim). Momentum-exchange force per link.
        ρ_link: shape (N_links,). Density Σ_i f_post[i] at the fluid node
                adjacent to wall for each link — the cleaner raw input for
                density-based surface-pressure post-processing:
                    p_wall ≈ ρ_link · cs²,  Cp = (p − p_∞)/q.

        Order matches `self.link_*`. HWBB or IBB formula used depending on
        `self._ibb_aware`. This path runs only when `save_link_forces=True`
        was passed to __init__; otherwise raises RuntimeError.
        """
        if not self._save_link_forces:
            raise RuntimeError(
                "compute_per_link() requires save_link_forces=True at "
                "MomentumExchangeForce construction."
            )

        import numpy as _np
        xp = self.xp
        c = self.c
        opp = self.opp

        N = len(self.link_dir)
        if N == 0:
            return (
                _np.zeros((0, self.dim), dtype=_np.float32),
                _np.zeros(0, dtype=_np.float32),
            )

        # Gather f_post values at link fluid nodes (host numpy).
        # For both branches we need f_post[dir, xf, yf]; for q<1/2 also
        # f_post[dir, xf - cx, yf - cy] (upstream).
        def _to_host(arr):
            return arr.get() if hasattr(arr, 'get') else _np.asarray(arr)

        f_host = _to_host(f_post)          # (Q, Nx, Ny)  float
        Nx, Ny = f_host.shape[1], f_host.shape[2]

        i_arr = self.link_dir
        x_arr = self.link_xf
        y_arr = self.link_yf
        cx_arr = self.link_cx.astype(_np.int32)
        cy_arr = self.link_cy.astype(_np.int32)

        f_i = f_host[i_arr, x_arr, y_arr]
        # opp direction index per link
        opp_host = opp.get() if hasattr(opp, 'get') else _np.asarray(opp)
        i_opp_arr = opp_host[i_arr]
        f_opp = f_host[i_opp_arr, x_arr, y_arr]

        # Upstream x_f - c_i (wrap-safe; boundary links shouldn't wrap).
        xu = (x_arr - cx_arr) % Nx
        yu = (y_arr - cy_arr) % Ny
        f_up = f_host[i_arr, xu, yu]

        # Per-cell ρ at each link's fluid node.  f_post is post-collision
        # and mass-conservative, so ρ = Σ_i f_post[i] at that cell equals
        # the macroscopic density.
        rho_field = f_host.sum(axis=0)          # (Nx, Ny)
        rho_link  = rho_field[x_arr, y_arr].astype(_np.float32)  # (N,)

        if self._ibb_aware:
            q_host = _to_host(self._q_fraction).astype(_np.float32)
            q = q_host[i_arr, x_arr, y_arr]
        else:
            q = _np.full(N, 0.5, dtype=_np.float32)

        # Branch on q
        lt = q < 0.5
        ge = ~lt
        f_link_scalar = _np.zeros(N, dtype=_np.float32)
        two_q = 2.0 * q
        # q >= 0.5
        f_link_scalar[ge] = (
            (two_q[ge] + 1.0) / two_q[ge] * f_i[ge]
            + (two_q[ge] - 1.0) / two_q[ge] * f_opp[ge]
        )
        # q < 0.5
        f_link_scalar[lt] = (
            (1.0 + 2.0 * q[lt]) * f_i[lt]
            + (1.0 - 2.0 * q[lt]) * f_up[lt]
        )

        F_link = _np.empty((N, self.dim), dtype=_np.float32)
        F_link[:, 0] = self.link_cx * f_link_scalar
        F_link[:, 1] = self.link_cy * f_link_scalar
        return F_link, rho_link

    def get_link_metadata(self) -> Dict[str, 'npt.NDArray']:
        """Return per-link metadata (host numpy). Requires save_link_forces=True."""
        if not self._save_link_forces:
            raise RuntimeError("save_link_forces was not enabled.")
        return {
            'link_dir': self.link_dir,
            'link_xf':  self.link_xf,
            'link_yf':  self.link_yf,
            'link_cx':  self.link_cx,
            'link_cy':  self.link_cy,
            'link_xw':  self.link_xw,
            'link_yw':  self.link_yw,
            'link_nx':  self.link_nx,
            'link_ny':  self.link_ny,
            'link_dl':  self.link_dl,
            'link_w':   self.link_w,
        }

    def get_coefficients(self, forces: Tuple,
                         rho_ref: float = 1.0,
                         u_ref: float = 0.1,
                         char_length: float = 20.0,
                         span_length: float = 1.0) -> Dict[str, float]:
        """Convert forces to dimensionless coefficients
        
        Args:
            forces: (Fx, Fy) or (Fx, Fy, Fz)  [lattice units]
            rho_ref: Reference density  [dimensionless]
            u_ref: Reference velocity  [Δx/Δt]
            char_length: Characteristic length D  [Δx]
            span_length: Span length (for 3D, use Lz)  [Δx]
        
        Returns:
            Dictionary with Cd, Cl (and Cz for 3D)
        """
        # Reference area: D for 2D, D*Lz for 3D
        if self.dim == 2:
            A_ref = char_length  # Per unit span
        else:
            A_ref = char_length * span_length
        
        # Dynamic pressure: q = 0.5 * ρ * U²
        q = 0.5 * rho_ref * u_ref**2
        
        # Force coefficients
        Cd = forces[0] / (q * A_ref) if abs(q * A_ref) > 1e-10 else 0.0
        Cl = forces[1] / (q * A_ref) if abs(q * A_ref) > 1e-10 else 0.0
        
        result = {'Cd': Cd, 'Cl': Cl}
        
        if self.dim == 3 and len(forces) > 2:
            Cz = forces[2] / (q * A_ref) if abs(q * A_ref) > 1e-10 else 0.0
            result['Cz'] = Cz
        
        return result
    
    def get_info(self) -> str:
        """Return information about the force calculator"""
        dim_str = "2D" if self.dim == 2 else "3D"
        scheme = "Bouzidi-aware (linear IBB)" if self._ibb_aware else "HWBB"
        return (f"MomentumExchangeForce ({dim_str}, {scheme}):\n"
                f"  Boundary links: {self.n_boundary_links:,}")


class ForceManager:
    """High-level manager for force calculation and logging
    
    Combines MomentumExchangeForce with logging, statistics,
    and coefficient computation.
    
    Supports both 2D and 3D domains.
    
    Example:
        >>> force_mgr = ForceManager(xp, lattice, solid_mask, config)
        >>> force_mgr.initialize()
        >>> # In time loop:
        >>> force_mgr.compute_and_log(step, f_post)
        >>> # After simulation:
        >>> stats = force_mgr.get_final_statistics()
    """
    
    def __init__(self, xp: 'ModuleType',
                 lattice: 'Lattice',
                 solid_mask: 'npt.NDArray',
                 config: Dict[str, Any],
                 wall_bc: Optional['HalfwayBounceBack'] = None,
                 csv_dir: str = './results/csv',
                 internal_geometry: Optional[Dict[str, Any]] = None) -> None:
        """Initialize force manager

        Args:
            xp: Array module
            lattice: Lattice model
            solid_mask: Solid mask
            config: Force configuration dictionary
            wall_bc: Optional wall BC (reuses boundary link info)
            csv_dir: Directory for CSV output
            internal_geometry: config['internal_geometry'] dict, used at
                close() time to auto-write surface_cp.dat / .vtp / .csv
                for 2D obstacles.  Omitted → no surface-distribution output.
        """
        self.xp = xp
        self.config = config
        self.csv_dir = csv_dir
        self.dim = lattice.dim
        
        # Parse config
        self.enabled = config.get('enabled', True)
        self.interval = config.get('interval', 1)
        self.start_step = config.get('start_step', 0)
        
        ref_config = config.get('reference', {})
        self.rho_ref = ref_config.get('rho', 1.0)
        self.u_ref = ref_config.get('velocity', 0.1)
        self.char_length = ref_config.get('char_length', 20.0)
        self.span_length = ref_config.get('span_length', 1.0)
        
        log_config = config.get('log', {})
        self.log_enabled = log_config.get('enabled', True)
        self.log_filename = log_config.get('filename', 'force_history')

        # Surface-distribution option.
        # Default: auto-on for 2D obstacles (cheap — a few MB per run).
        # Caller may still force it off via config['save_link_forces']=False.
        self._internal_geometry = dict(internal_geometry or {})
        has_2d_obstacle = (
            self.dim == 2
            and any(isinstance(v, dict) and v.get('enabled', False)
                    for v in self._internal_geometry.values())
        )
        self.save_link_forces = bool(
            config.get('save_link_forces', has_2d_obstacle)
        )

        # Create force calculator
        self.force_calc = MomentumExchangeForce(
            xp, lattice, solid_mask, wall_bc,
            save_link_forces=self.save_link_forces,
        )

        # History storage
        self.history: List[Dict[str, Any]] = []

        # Per-link storage (only populated when save_link_forces=True)
        self._link_forces_steps: List[int] = []
        self._link_forces_buffer: List['npt.NDArray'] = []
        self._link_rho_buffer:    List['npt.NDArray'] = []

        # CSV file handle
        self._csv_file = None
        self._csv_writer = None
    
    def initialize(self) -> None:
        """Print force manager info. CSV is opened later by open_csv()."""
        if not self.enabled:
            print("  Force calculation: disabled")
            return

        print(f"  {self.force_calc.get_info()}")
        print(f"  Reference: ρ={self.rho_ref}, U={self.u_ref}, D={self.char_length}, Lz={self.span_length}")
        print(f"  Interval: every {self.interval} steps (start from step {self.start_step})")

    def open_csv(self, start_step: int = 0) -> None:
        """Open CSV log file. Called by SolverInitializer after start_step is known.

        Args:
            start_step: First step of this run. If > 0 (restart),
                        existing CSV rows up to start_step are preserved
                        and new data is appended.
        """
        if not self.enabled or not self.log_enabled:
            return

        os.makedirs(self.csv_dir, exist_ok=True)
        csv_path = os.path.join(self.csv_dir, f"{self.log_filename}.csv")

        if self.dim == 2:
            fieldnames = ['step', 'Fx', 'Fy', 'Cd', 'Cl']
        else:
            fieldnames = ['step', 'Fx', 'Fy', 'Fz', 'Cd', 'Cl', 'Cz']

        if start_step > 0 and os.path.exists(csv_path):
            # Restart: keep rows with step < start_step
            kept_lines = []
            with open(csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row['step']) < start_step:
                        kept_lines.append(row)

            self._csv_file = open(csv_path, 'w', newline='')
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=fieldnames,
            )
            self._csv_writer.writeheader()
            for row in kept_lines:
                self._csv_writer.writerow(row)
            self._csv_file.flush()
            print(f"  Force CSV: {csv_path} (kept {len(kept_lines)} rows, "
                  f"appending from step {start_step})")
        else:
            self._csv_file = open(csv_path, 'w', newline='')
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=fieldnames,
            )
            self._csv_writer.writeheader()
            print(f"  Force CSV: {csv_path}")
    
    def should_compute(self, step: int) -> bool:
        """Check if force should be computed at this step"""
        if not self.enabled:
            return False
        if step < self.start_step:
            return False
        return step % self.interval == 0
    
    def compute_and_log(self, step: int, f_post: 'npt.NDArray',
                        verbose: bool = False) -> Optional[Dict[str, Any]]:
        """Compute forces and log to CSV
        
        Args:
            step: Current time step
            f_post: Post-collision distribution
            verbose: Print results to console
        
        Returns:
            Dictionary with forces and coefficients, or None if skipped
        """
        if not self.should_compute(step):
            return None
        
        # Compute forces
        forces = self.force_calc.compute(f_post)
        
        # Get coefficients
        coeffs = self.force_calc.get_coefficients(
            forces,
            rho_ref=self.rho_ref,
            u_ref=self.u_ref,
            char_length=self.char_length,
            span_length=self.span_length
        )
        
        # Build result dictionary
        if self.dim == 2:
            result = {
                'step': step,
                'Fx': forces[0],
                'Fy': forces[1],
                'Cd': coeffs['Cd'],
                'Cl': coeffs['Cl']
            }
        else:
            result = {
                'step': step,
                'Fx': forces[0],
                'Fy': forces[1],
                'Fz': forces[2],
                'Cd': coeffs['Cd'],
                'Cl': coeffs['Cl'],
                'Cz': coeffs.get('Cz', 0.0)
            }
        
        # Store in history
        self.history.append(result)

        # Per-link snapshot for surface distribution
        if self.save_link_forces:
            F_link, rho_link = self.force_calc.compute_per_link(f_post)
            self._link_forces_steps.append(int(step))
            self._link_forces_buffer.append(F_link)
            self._link_rho_buffer.append(rho_link)

        # Write to CSV
        if self._csv_writer is not None:
            self._csv_writer.writerow(result)
            self._csv_file.flush()

        if verbose:
            print(f"  Step {step}: Cd={coeffs['Cd']:.4f}, Cl={coeffs['Cl']:.4f}")

        return result
    
    def get_final_statistics(self) -> Dict[str, Any]:
        """Compute final statistics from force history
        
        Returns:
            Dictionary with mean, std, min, max for each coefficient
        """
        import numpy as np
        
        if len(self.history) == 0:
            return {}
        
        Cd_values = [h['Cd'] for h in self.history]
        Cl_values = [h['Cl'] for h in self.history]
        
        stats = {
            'Cd_mean': float(np.mean(Cd_values)),
            'Cd_std': float(np.std(Cd_values)),
            'Cd_min': float(np.min(Cd_values)),
            'Cd_max': float(np.max(Cd_values)),
            'Cl_mean': float(np.mean(Cl_values)),
            'Cl_std': float(np.std(Cl_values)),
            'Cl_min': float(np.min(Cl_values)),
            'Cl_max': float(np.max(Cl_values)),
            'n_samples': len(self.history)
        }
        
        if self.dim == 3 and 'Cz' in self.history[0]:
            Cz_values = [h['Cz'] for h in self.history]
            stats.update({
                'Cz_mean': float(np.mean(Cz_values)),
                'Cz_std': float(np.std(Cz_values)),
            })
        
        return stats
    
    def print_summary(self, window: Optional[int] = None) -> None:
        """Print force summary statistics
        
        Args:
            window: Number of steps to average (None = last 50% of data)
        """
        if not self.enabled:
            print("  Force calculation: disabled")
            return
        
        if len(self.history) == 0:
            print("  Force calculation: no data recorded")
            return
        
        # Use last 50% of data by default (skip transient)
        if window is None:
            window = max(1, len(self.history) // 2)
        
        # Get statistics from recent data
        import numpy as np
        recent = self.history[-window:]
        
        Cd_values = np.array([h['Cd'] for h in recent])
        Cl_values = np.array([h['Cl'] for h in recent])
        Fx_values = np.array([h['Fx'] for h in recent])
        Fy_values = np.array([h['Fy'] for h in recent])
        
        print("\n" + "="*60)
        print(" Force Summary (time-averaged)")
        print("="*60)
        print(f"  Samples: {len(recent)} (last {window} of {len(self.history)} total)")
        print(f"  Drag coefficient: Cd = {np.mean(Cd_values):.4f} ± {np.std(Cd_values):.4f}")
        print(f"  Lift coefficient: Cl = {np.mean(Cl_values):.4f} ± {np.std(Cl_values):.4f}")
        print(f"  Lift RMS:         Cl_rms = {np.sqrt(np.mean(Cl_values**2)):.4f}")
        print(f"  Force (mean):     Fx = {np.mean(Fx_values):.6f}, Fy = {np.mean(Fy_values):.6f}")
        
        if self.dim == 3 and 'Fz' in recent[0]:
            Fz_values = np.array([h['Fz'] for h in recent])
            print(f"                    Fz = {np.mean(Fz_values):.6f}")
        
        print("="*60)
    
    def _extract_obstacle_meta(self) -> Dict[str, Any]:
        """Obstacle chord/center/AoA dumped alongside link metadata.

        Pulled from `internal_geometry` as supplied at construction. When
        MLG is active, setup passes the *fine-level* geometry config so
        these values are already in finest-lattice units (matching the
        per-link `link_xw`/`link_yw`). For single-level runs the values
        are simply L0 lu, which is the same lattice as the links.
        """
        import numpy as _np
        out: Dict[str, Any] = {}
        geom = self._internal_geometry
        for key in ('airfoil', 'circle', 'cylinder', 'sphere', 'box'):
            cfg = geom.get(key)
            if not (isinstance(cfg, dict) and cfg.get('enabled', False)):
                continue
            if 'center' in cfg:
                out['obstacle_center'] = _np.asarray(
                    cfg['center'], dtype=_np.float32,
                )
            if 'chord' in cfg:
                out['obstacle_chord'] = _np.float32(cfg['chord'])
            if 'radius' in cfg:
                out['obstacle_radius'] = _np.float32(cfg['radius'])
            if 'angle_of_attack' in cfg:
                out['obstacle_aoa_deg'] = _np.float32(cfg['angle_of_attack'])
            out['obstacle_type'] = _np.array(key)
            break
        return out

    def close(self) -> None:
        """Close CSV file and dump per-link surface data if enabled."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

        # Per-link raw NPZ dump only. Cp/Cf time-window selection is left to
        # the post-processor `src.utilities.surface_distribution` because the
        # correct averaging window depends on force_history.csv inspection.
        if self.save_link_forces and self._link_forces_buffer:
            import numpy as _np
            os.makedirs(self.csv_dir, exist_ok=True)
            meta = self.force_calc.get_link_metadata()
            obstacle_meta = self._extract_obstacle_meta()
            F_link_arr = _np.stack(self._link_forces_buffer, axis=0)  # (S, N, 2)
            rho_link_arr = _np.stack(self._link_rho_buffer, axis=0)    # (S, N)
            steps_arr = _np.asarray(self._link_forces_steps, dtype=_np.int64)

            npz_path = os.path.join(self.csv_dir, 'surface_link_forces.npz')
            _np.savez_compressed(
                npz_path,
                steps=steps_arr,
                F_link=F_link_arr,
                rho_link=rho_link_arr,
                **meta,
                **obstacle_meta,
            )
            print(f"  Surface link forces saved: {npz_path} "
                  f"({len(self._link_forces_buffer)} samples × "
                  f"{F_link_arr.shape[1]} links)")
    
    def get_info(self) -> str:
        """Return information string"""
        dim_str = "2D" if self.dim == 2 else "3D"
        return (f"ForceManager ({dim_str}):\n"
                f"  Enabled: {self.enabled}\n"
                f"  Interval: {self.interval}\n"
                f"  Start step: {self.start_step}\n"
                f"  {self.force_calc.get_info()}")


# =============================================================================
# Strouhal Number Calculation
# =============================================================================

def compute_strouhal_number(force_history: List[Dict[str, Any]],
                            char_length: float,
                            u_ref: float,
                            component: str = 'Cl',
                            min_periods: int = 3) -> Optional[float]:
    """Compute Strouhal number from force time history using FFT
    
    The Strouhal number characterizes vortex shedding frequency:
        St = f * D / U
    
    where:
        f = dominant frequency of lift oscillation  [1/Δt]
        D = characteristic length (diameter)  [Δx]
        U = reference velocity  [Δx/Δt]
    
    Algorithm:
        1. Extract force component time series
        2. Apply FFT to find dominant frequency
        3. Compute St = f * D / U
    
    Args:
        force_history: List of force records with 'step' and component
        char_length: Characteristic length D  [Δx]
        u_ref: Reference velocity U  [Δx/Δt]
        component: Force component to analyze ('Cl', 'Fy', etc.)
        min_periods: Minimum number of oscillation periods required
        
    Returns:
        Strouhal number, or None if insufficient data
        
    Example:
        >>> St = compute_strouhal_number(force_mgr.history, D=20, u_ref=0.1)
        >>> print(f"Strouhal number: {St:.4f}")  # Expected ~0.16-0.21 for cylinder
    """
    import numpy as np
    
    if len(force_history) < 100:
        return None  # Not enough data
    
    # Extract component values
    if component not in force_history[0]:
        return None
    
    values = np.array([h[component] for h in force_history])
    steps = np.array([h['step'] for h in force_history])
    
    # Determine time step (assume uniform spacing)
    dt = 1.0  # In lattice units, one step = Δt = 1
    if len(steps) > 1:
        dt = float(steps[1] - steps[0])
    
    N = len(values)
    
    # Remove mean (DC component)
    values_centered = values - np.mean(values)
    
    # Apply window function to reduce spectral leakage
    window = np.hanning(N)
    values_windowed = values_centered * window
    
    # FFT
    fft_vals = np.fft.rfft(values_windowed)
    freqs = np.fft.rfftfreq(N, d=dt)
    
    # Find dominant frequency (excluding DC)
    magnitudes = np.abs(fft_vals)
    magnitudes[0] = 0  # Ignore DC
    
    # Find peak
    peak_idx = np.argmax(magnitudes)
    if peak_idx == 0:
        return None  # No oscillation detected
    
    f_dominant = freqs[peak_idx]  # [1/Δt]
    
    # Check if we have enough periods
    T_period = 1.0 / f_dominant if f_dominant > 0 else float('inf')
    n_periods = (N * dt) / T_period
    
    if n_periods < min_periods:
        return None  # Insufficient number of periods
    
    # Strouhal number: St = f * D / U
    St = f_dominant * char_length / u_ref
    
    return float(St)