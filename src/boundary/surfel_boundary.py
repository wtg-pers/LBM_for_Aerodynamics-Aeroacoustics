"""Production adapter for the surfel wall-model boundary (S8a, patch 47).

Wires the verified surfel stack (surfel_geometry / surfel_transport /
surfel.SurfelFacets / kernels.surfel_d3q27) into the solver's
`wall_bc = "surfel"` path. The testbed
patch_notes/surfel/channel_wmles_surfel.py is the validated reference —
this adapter reproduces its construction and per-step call order exactly
(gate s13 pins the parity), it does not re-derive anything.

Scope (patch_notes/surfel/46 sec. 1): STL geometry, standard advance
path only. Since S8a-2 one instance is built PER MLG LEVEL that carries
the body: `triangles_lu`, `shape` and `nu_lu` are all level-local, so
facet areas [lu_k^2], prism dV [lu_k^3] and the facet force [lu_k units]
come out in that level's lattice units by construction (the MLG force
rebind scales the reference lengths by 2^k to match). tau-model band
(S8b) and esoteric streaming (post-S8c) stay guarded with explicit
errors.

STRUCTURAL NOTE: unlike HWBB/IBB (post-stream corrections), the surfel
scheme REPLACES streaming — populations are volumetric (cut cells carry
dV < 1) and `SurfelKernelD3Q27.advect` transports them with facet
interception. Hence Simulation dispatches to `_advance_surfel` instead
of hooking `apply_with_reset` after `streaming.compute`.

Defaults below are the campaign-final configuration
(patch_notes/surfel/45 sec. 3): Musker law, h = 3, viscous fallback,
log friction direction, dv_min = 1e-6 (the measured small-cell floor —
without it one roundoff-dV cell per solid slice passes the live test
and n = N/dV explodes; PLAN "클러스터 런" note).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.boundary.surfel import SurfelFacets
from src.boundary.surfel_geometry import build_surfels
from src.boundary.surfel_transport import (
    C27, build_prism_tables, fluid_fraction_by_march,
)
from src.boundary.wall_function import WALL_LAWS
from src.kernels.surfel_d3q27 import SurfelKernelD3Q27

#: campaign-final defaults (patch 45 sec. 3 / testbed CLI defaults)
_DEFAULTS = dict(
    law='musker', h_law=3.0, y_plus_min=30.0, sample_h=0.5,
    friction_dir='log', fallback='viscous', mode='wallmodel',
    dv_min=1e-6, march_axis=2, orient='as_is',
    tau_model=False, collide='kernel',
)

#: D3Q27 weights in surfel_transport's C27 ordering, derived from |c|^2
#: (order-independent by construction).
_W27 = np.array([8 / 27, 2 / 27, 1 / 54, 1 / 216])[
    np.einsum('qd,qd->q', C27, C27)]


class SurfelBoundary:
    """Surfel wall boundary bound to ONE grid level.

    Args:
        xp: array module (cupy expected — the surfel kernel is CUDA).
        shape: THIS level's grid shape (NX, NY, NZ).
        triangles_lu: (vertices_lu, faces) in THIS grid's lattice units —
            exactly the `geom_info['triangles_lu']` contract of
            geometry_manager (fine levels pass the re-localized frame
            built by create_fine_level_geometry_config).
        nu_lu: molecular viscosity in THIS level's lattice units (wall
            law input — nu_lu doubles per refinement level under
            acoustic scaling; passing the L0 value on a fine level halves
            every sampled y+).
        cfg: the `internal_geometry.<type>.surfel` sub-dict (optional
            overrides of _DEFAULTS; unknown keys rejected).
    """

    kind = 'surfel'

    def __init__(self, xp, shape: Tuple[int, int, int], triangles_lu,
                 nu_lu: float, cfg: dict | None = None):
        cfg = dict(cfg or {})
        unknown = set(cfg) - set(_DEFAULTS)
        if unknown:
            raise ValueError(f"surfel config: unknown keys {sorted(unknown)}")
        p = {**_DEFAULTS, **cfg}
        if p['orient'] == 'auto':
            # recurring-mistake list: orient="auto" on open meshes flips
            # normals silently. STL bodies are watertight-checked upstream
            # and carry CCW winding — 'as_is' is the only sanctioned value.
            raise ValueError("surfel: orient='auto' is forbidden "
                             "(open-mesh normal flips; use 'as_is')")

        self.xp = xp
        self.shape = tuple(int(s) for s in shape)
        if len(self.shape) != 3:
            raise ValueError("surfel wall BC is D3Q27-only (S8a)")

        verts, faces = triangles_lu
        sf = build_surfels(self.shape, (np.asarray(verts), np.asarray(faces)),
                           orient=p['orient'], return_polygons=True)
        tb = build_prism_tables(sf, self.shape)
        dv_raw = np.clip(
            fluid_fraction_by_march(tb, sf, self.shape,
                                    axis=int(p['march_axis'])), 0.0, 1.0)
        # small-cell floor (module docstring): measured, not negotiable
        self.dv_min = float(p['dv_min'])
        dV = np.where(dv_raw > self.dv_min, dv_raw, 0.0)
        live = dV > 0.0
        self.dropped_vol = float(dv_raw.sum() - dV.sum())

        self.facets = SurfelFacets(
            sf, tb, self.shape,
            mode=p['mode'], sample_h=float(p['sample_h']), live=live,
            law=WALL_LAWS[p['law']](), h_law=float(p['h_law']),
            nu=float(nu_lu), y_plus_min=float(p['y_plus_min']),
            friction_dir=p['friction_dir'], fallback=p['fallback'],
            wm_filter=None,
        )
        self.kernel = SurfelKernelD3Q27(self.facets)
        self.N = self.kernel.N
        if self.N != int(np.prod(self.shape)):
            raise RuntimeError("surfel kernel N != prod(shape)")

        self.n_facets = int(len(sf['area']))
        self.nnz = int(tb['cell'].size) if 'cell' in tb else -1
        # kept for the surface writer (S8o): polygons + triangle identity
        self.surfels = sf
        self.triangles_lu = (np.asarray(verts), np.asarray(faces))
        self.n_faces = int(len(self.triangles_lu[1]))
        # filled by setup when the config gives a freestream (Cp/Cf refs)
        self.q_inf = None
        self.p_ref = 1.0 / 3.0            # lattice p_inf = rho0 * cs^2
        # surface-output frame: verts_out = origin + verts * spacing.
        # Identity on L0; MLG setup binds the level's L0-lu origin/dx so
        # the surface VTK overlays the per-level volume output.
        self.coord_origin = (0.0, 0.0, 0.0)
        self.coord_spacing = 1.0

        self.dV_h, self.live_h = dV, live
        self.d_dV = xp.asarray(dV.ravel())
        self.d_live = xp.asarray(live.ravel().astype(np.uint8))
        self.d_dead = xp.asarray(~live.ravel())     # bool, for zero_dead
        self._solid_mask_dev = None       # lazy (ForceManager contract)
        # provenance for the setup summary / log header
        self.params = p

        # ── tau-model band (S8b, patch 52; campaign config carried over:
        #    FS_CLASSES v3, src wall, inject→apply, no cut-cell injection)
        self.tau_model_on = bool(p['tau_model'])
        self._taum_summary = ""
        self._d_CC = None                 # lazy (tau_sgs moment matrix)
        # collide path (patch 55): 'kernel' = fused macro+SGS+collide in
        # ONE launch (CumulantCollideKernelD3Q27, gate s10) — the host
        # collide was measured at 87% of the surfel step (testbed note).
        # 'host' = the reference chain (gates [D]/[S] pin it).
        if p['collide'] not in ('kernel', 'host'):
            raise ValueError(f"surfel collide: unknown '{p['collide']}'")
        self.collide_path = p['collide']
        if self.tau_model_on:
            self._build_tau_band()

    # ------------------------------------------------------------ tau-model
    def _build_tau_band(self) -> None:
        """Static tau-model band for THIS level's facet geometry (S8b).

        Curved-surface generalization of the channel testbed wiring
        (channel_wmles_surfel.py, patches 25~40); every campaign rule is
        carried over, the only new inputs are geometric:

          eta   per cell = normal projection onto the NEAREST facet
                (exact for planes; curvature error O(offset^2/R) —
                negligible at wall-model resolutions).
          y1    per facet = distance to the first LIVE cell center along
                the facet normal (cut or full — matches the channel's
                first-fluid-NODE definition; the class tables key on it).
          f_s   = f_s_class(eta, y1(nearest facet)) — FS_CLASSES v3,
                zero free parameters (patch 40).
          band  = FULL (dV = 1) AND live cells with f_s > 0 — cut cells
                are never injected (measured runaway, tau_model.py
                deviation 5; patch 29 gating).
          tau_w per cell = facet-area-weighted average over facets near
                the cell's wall FOOT POINT (the curved analog of the
                channel's per-column area average), gathered at runtime
                from kernel.tau_out (one substep stale — patch 27).

        The shear direction t_hat is NOT static: Tamaki's xi is the
        wall-parallel velocity direction, evaluated per substep in
        inject_tau_model (the channel testbed froze xi = +x, valid for
        its mean-flow frame — deviation-from-testbed, not from source).
        """
        from scipy.ndimage import binary_dilation
        from scipy.spatial import cKDTree

        from src.turbulence.tau_model import (
            FS_CLASSES, FS_CLASSES_VERSION,
        )

        xp = self.xp
        nx, ny, nz = self.shape
        cen = np.asarray(self.facets.centroid, dtype=np.float64)
        nrm = np.asarray(self.facets.normal, dtype=np.float64)
        area = np.asarray(self.facets.area, dtype=np.float64)
        n_f = cen.shape[0]

        # per-facet y1: first live cell center along the normal ray.
        # Sample the ray at 0.05..2.55 (cells), accept the first live
        # cell whose CENTER projects positively onto the normal.
        y1_f = np.full(n_f, 0.5)
        found = np.zeros(n_f, dtype=bool)
        for s in np.arange(0.05, 2.56, 0.1):
            if found.all():
                break
            pos = np.rint(cen + s * nrm).astype(np.int64)
            np.clip(pos[:, 0], 0, nx - 1, out=pos[:, 0])
            np.clip(pos[:, 1], 0, ny - 1, out=pos[:, 1])
            np.clip(pos[:, 2], 0, nz - 1, out=pos[:, 2])
            lv = self.live_h[pos[:, 0], pos[:, 1], pos[:, 2]]
            proj = np.einsum('fd,fd->f', pos - cen, nrm)
            hit = (~found) & lv & (proj > 0.05)
            y1_f[hit] = proj[hit]
            found |= hit
        self._taum_y1_unresolved = int((~found).sum())

        # band candidates: full+live cells within reach of the surface.
        # Cheap prefilter: dilate the facet-coupled (prism CSR) cell set
        # by the table depth, then keep full+live.
        eta_max = 6.5
        near = np.zeros(self.shape, dtype=bool)
        csr_cells = np.unique(self.facets._t_cell)
        near.ravel()[csr_cells] = True
        near = binary_dilation(near, iterations=int(np.ceil(eta_max)))
        full = self.dV_h > 1.0 - 1e-6
        cand = near & full & self.live_h
        cxyz = np.argwhere(cand).astype(np.float64)

        tree = cKDTree(cen)
        _, jn = tree.query(cxyz, k=1, workers=-1)
        eta = np.einsum('md,md->m', cxyz - cen[jn], nrm[jn])
        # vectorize f_s_class over per-cell y1 by linear blend of the
        # class tables (same math as tau_model.f_s_class, batched):
        keys = sorted(FS_CLASSES)
        y1c = np.clip(y1_f[jn], keys[0], keys[-1])
        fs_k = {k: np.interp(eta, FS_CLASSES[k][0], FS_CLASSES[k][1],
                             left=FS_CLASSES[k][1][0], right=0.0)
                for k in keys}
        fs = np.zeros_like(eta)
        for lo, hi in zip(keys[:-1], keys[1:]):
            m = (y1c >= lo) & (y1c <= hi)
            w = np.where(hi > lo, (y1c - lo) / max(hi - lo, 1e-30), 0.0)
            fs[m] = ((1.0 - w[m]) * fs_k[lo][m] + w[m] * fs_k[hi][m])
        keep = (eta > 0.05) & (fs > 0.0)
        cxyz, jn, eta, fs = cxyz[keep], jn[keep], eta[keep], fs[keep]
        M = int(cxyz.shape[0])
        if M == 0:
            # a level whose body region holds no full band cell (deeply
            # under-resolved) — nothing to inject; report, do not crash
            self.tau_model_on = False
            self._taum_summary = "tau-model requested but band is EMPTY"
            return

        # tau_w gather: facets near each cell's wall foot point,
        # area-weighted (curved analog of the channel column average).
        foot = cxyz - eta[:, None] * nrm[jn]
        neigh = tree.query_ball_point(foot, r=1.5, workers=-1)
        indptr = np.zeros(M + 1, dtype=np.int64)
        for i, lst in enumerate(neigh):
            indptr[i + 1] = indptr[i] + (len(lst) if lst else 1)
        fidx = np.empty(indptr[-1], dtype=np.int64)
        for i, lst in enumerate(neigh):
            if lst:
                fidx[indptr[i]:indptr[i + 1]] = lst
            else:
                fidx[indptr[i]] = jn[i]       # fallback: nearest facet
        w_area = area[fidx]
        row = np.repeat(np.arange(M), np.diff(indptr))
        wsum = np.bincount(row, weights=w_area, minlength=M)
        w_norm = w_area / np.maximum(wsum[row], 1e-30)

        import cupyx.scipy.sparse as _cs
        Wmat = _cs.csr_matrix(
            (xp.asarray(w_norm),
             xp.asarray(fidx.astype(np.int32)),
             xp.asarray(indptr.astype(np.int32))),
            shape=(M, n_f))
        self._tb_W = Wmat
        flat = (cxyz[:, 0].astype(np.int64) * ny
                + cxyz[:, 1].astype(np.int64)) * nz \
            + cxyz[:, 2].astype(np.int64)
        self.d_tb_cells = xp.asarray(flat)
        self.d_tb_fs = xp.asarray(fs.astype(np.float32))
        self.d_tb_normal = xp.asarray(
            np.ascontiguousarray(nrm[jn]).astype(np.float32))
        self.d_tb_C = xp.asarray(C27.astype(np.float32))
        self.d_tb_Wl = xp.asarray(_W27.astype(np.float32))
        self._sig_last = None
        # band-start provenance (patch 29: log what the mask DID) +
        # FS table fingerprint (patch 40: values are data, labels lie)
        nodes = ",".join(f"{v:g}" for v in FS_CLASSES[0.5][1])
        self._taum_summary = (
            f"tau-model ON: {M} band cells, band-start eta="
            f"{eta.min():.2f}, eta-max {eta.max():.2f}, "
            f"y1 [{y1_f.min():.2f},{y1_f.max():.2f}] "
            f"(unresolved {self._taum_y1_unresolved}), "
            f"fs {FS_CLASSES_VERSION} nodes(0.5)=[{nodes}]")

    def inject_tau_model(self, f_post, u, tau_bar: float) -> None:
        """Post-collision Hermite-2 injection, BEFORE the facet apply.

        Order is load-bearing (patch 27, measured): apply-first lets the
        injected wall-crossing populations drain outside the facet force
        accounting (closure 1.37); inject-first closes at 1.000.

        sigma^m = tau_w f_s (Eq. 50, src wall — bounded by tau_w), with
        tau_w from the PREVIOUS substep's kernel.tau_out (one substep
        stale; step 0 injects 0). t_hat = unit wall-parallel component
        of the LOCAL cell velocity (Tamaki's xi); cells with no
        tangential velocity inject 0 (stagnation: direction undefined,
        tau_w ~ 0 there anyway). dPi = -sigma^m / tau_bar with the
        LEVEL's scalar molecular tau (tau_model.py deviations 1~2).
        """
        xp = self.xp
        idx = self.d_tb_cells
        nvec = self.d_tb_normal                       # (M, 3) f32
        ub = u.reshape(3, -1)[:, idx].T.astype(xp.float32, copy=False)
        un = (ub * nvec).sum(axis=1)
        t = ub - un[:, None] * nvec
        tn = xp.sqrt((t * t).sum(axis=1))
        that = t / xp.maximum(tn, xp.float32(1e-30))[:, None]
        ok = (tn > 1e-10).astype(xp.float32)

        tau_cell = self._tb_W @ self.kernel.tau_out   # (M,) f64
        sig = tau_cell.astype(xp.float32) * self.d_tb_fs * ok
        dPi = -(sig / xp.float32(tau_bar))            # (M,)

        h = (xp.float32(9.0) * self.d_tb_Wl[:, None]
             * (self.d_tb_C @ that.T) * (self.d_tb_C @ nvec.T))  # (27, M)
        f_post.reshape(f_post.shape[0], -1)[:, idx] += h * dPi[None, :]
        self._sig_last = sig

    # -------------------------------------------------------------- sgs
    def tau_sgs(self, f, rho, u, tau0: float, Cs: float):
        """Per-cell tau_total, Stiebler moment-based Smagorinsky (S8b-2).

        Straight port of the testbed's tau_sgs (channel_wmles_surfel.py
        — the configuration every campaign table was measured with):
        Pi^neq from the PRE-collision populations as raw second central
        moments (second-order cumulants == second-order central moments),
        Q = sqrt(2 Pi:Pi), tau via src.turbulence.tau_total. Dead cells
        keep tau0 by where() — their raw u is NaN (0/0) and the NaN
        rides through pi harmlessly because where() SELECTS, never
        multiplies.
        """
        from src.turbulence.strain_rate import pi_norm_3d
        from src.turbulence.tau_total import tau_total_smagorinsky

        xp = self.xp
        if self._d_CC is None:
            cc = np.stack([C27[:, 0] * C27[:, 0], C27[:, 1] * C27[:, 1],
                           C27[:, 2] * C27[:, 2], C27[:, 0] * C27[:, 1],
                           C27[:, 0] * C27[:, 2], C27[:, 1] * C27[:, 2]])
            self._d_CC = xp.asarray(cc.astype(np.float32))
        n = f.reshape(f.shape[0], -1)
        M = self._d_CC @ n                                # (6, N)
        rho_f = rho.reshape(-1)
        ux, uy, uz = (u.reshape(3, -1)[i] for i in range(3))
        th = xp.float32(1.0 / 3.0)
        pi = (M[0] - rho_f * ux * ux - rho_f * th,
              M[1] - rho_f * uy * uy - rho_f * th,
              M[2] - rho_f * uz * uz - rho_f * th,
              M[3] - rho_f * ux * uy,
              M[4] - rho_f * ux * uz,
              M[5] - rho_f * uy * uz)
        Q = pi_norm_3d(*pi)
        tau = tau_total_smagorinsky(tau0, Q, Cs)
        tau = xp.where(self.d_live > 0, tau, xp.float32(tau0))
        return tau.reshape(rho.shape)

    # ---------------------------------------------------------------- step
    def sanitize_macro(self, rho, u):
        """Benign (rho=1, u=0) state wherever the division is undefined.

        Guard = live AND rho > 1e-20, matching the testbed's OWN
        macroscopic() (channel_wmles_surfel.py): live cells can hold
        rho ~ 0 too (measured: 36 such cells in the parity arm's initial
        field), and the production Macroscopic divides unguarded there.
        Always where(), never mask-multiply: NaN * 0 = NaN.
        """
        xp = self.xp
        ok = (self.d_live.reshape(self.shape) > 0) & (rho > 1e-20)
        # f32 contract: the production Macroscopic promotes u to float64
        # (int32 c in the tensordot — measured), while the surfel CUDA
        # kernel and the testbed macro are float32. Cast here, once.
        rho_s = xp.where(ok, rho, rho.dtype.type(1.0)).astype(
            xp.float32, copy=False)
        u_s = xp.where(ok[None], u, u.dtype.type(0.0)).astype(
            xp.float32, copy=False)
        return rho_s, u_s

    def mask_post(self, f_post):
        """Zero populations on non-live cells (post-collision hygiene)."""
        xp = self.xp
        flat = f_post.reshape(f_post.shape[0], -1)
        flat *= (self.d_live > 0).astype(flat.dtype)[None, :]

    def zero_dead(self, f):
        """Zero dead-cell populations by ASSIGNMENT (NaN-safe).

        For fields that may carry NaN on dead cells — the MLG F2C
        restriction writes feq(rho=0/0) through the body (patch 50) —
        mask-multiply would keep the NaN (NaN * 0 = NaN, the documented
        trap); indexed assignment removes it.
        """
        f.reshape(f.shape[0], -1)[:, self.d_dead] = 0.0

    def apply_and_advect(self, f_post, f_new, rho, u):
        """Facet exchange then volumetric transport (replaces streaming).

        f_post: post-collision populations (Q,)+shape — modified in place
            by the facet interception; f_new receives the streamed field.
        rho/u: macroscopics of the raw field. The testbed's own macro
            guards the division (dead cells: rho = 0, u = 0); the
            production Macroscopic divides unguarded, so dead-cell u may
            be NaN — zero it with where() to match the testbed input
            contract exactly (rho stays raw: dead rho = 0 both ways).
        """
        xp = self.xp
        if f_post.dtype != xp.float32:
            raise TypeError("surfel path is float32-only (CUDA kernel "
                            f"contract); got {f_post.dtype}")
        q = f_post.shape[0]
        rho_flat = rho.ravel().astype(xp.float32, copy=False)
        ok = (self.d_live > 0) & (rho_flat > 1e-20)
        u_flat = u.reshape(u.shape[0], -1)
        u_in = xp.where(ok[None, :], u_flat, u.dtype.type(0.0)).astype(
            xp.float32, copy=False)
        # kernel.apply RETURNS (Q, force) — the CUDA path does NOT write
        # the python facets' _force state (recurring mistake #4 family:
        # the first smoke read exact 0.0 through facets.last_force()).
        # Capture the return DEVICE-side; last_force() converts lazily —
        # a per-substep .get() here was a host sync every substep
        # (patch 55; ForceManager only reads every `interval` steps).
        _, F = self.kernel.apply(f_post.reshape(q, -1), rho_flat,
                                 u_in, self.d_live)
        self._force = F
        self.kernel.advect(f_post.reshape(q, -1), self.d_dV, self.d_live,
                           f_new.reshape(q, -1))

    @property
    def solid_mask(self):
        """Bool mask (device), True = no fluid volume on this level.

        Exists for the ForceManager constructor contract (the MLG force
        rebind reads `obstacle_bc.solid_mask`). The surfel force itself
        comes from `last_force()` (facet ledger) — this mask never enters
        a momentum-exchange link formula.
        """
        if self._solid_mask_dev is None:
            self._solid_mask_dev = self.xp.asarray(~self.live_h)
        return self._solid_mask_dev

    # ---------------------------------------------------------------- force
    def last_force(self):
        """Force on the body from the LAST apply (kernel return, host np).

        NOT facets.last_force(): that state belongs to the python
        SurfelFacets.apply path, which the CUDA kernel does not touch.
        The device->host conversion happens HERE (caller cadence =
        ForceManager interval), not per substep (patch 55).
        """
        F = getattr(self, '_force', None)
        if F is None:
            return np.zeros(3)
        return np.asarray(F.get() if hasattr(F, 'get') else F, dtype=float)

    def facet_traction(self, *a, **k):
        return self.facets.facet_traction(*a, **k)

    # ------------------------------------------------------------- surface
    def write_surface(self, path: str) -> int:
        """Write per-triangle surface loads (Cp/Cf) as legacy-VTK POLYDATA.

        S8o writer on the production path. The traction MUST come from the
        CUDA kernel's G buffers passed EXPLICITLY — facet_traction() with
        no args reads the python-path `_last` state the kernel never
        writes (recurring mistake #4; the same trap the force readout hit
        in patch 48). Triangle-level aggregation is the user-facing form:
        area-weighted onto the original STL triangles, immune to sliver
        noise (surfel_surface_writer docstring). Cp is built from p_use =
        p_state in wallmodel mode (the normal traction carries the
        spurious dp — patch 07, ~79% of signal).
        """
        from src.io.surfel_surface_writer import (
            aggregate_to_triangles, write_triangle_surface,
        )
        if getattr(self, '_force', None) is None:
            return 0                       # no facet pass yet this run
        G_in = np.asarray(self.kernel.G_in.get(), dtype=np.float64)
        G_out = np.asarray(self.kernel.G_out.get(), dtype=np.float64)
        t = self.facets.facet_traction(G_in=G_in, G_out=G_out)
        fields = {'p_use': t['p_use'], 'dp': t['dp'],
                  'tau_mag': t['tau_mag'], 'tau': t['tau'],
                  'traction': t['traction']}
        if self.q_inf:
            fields['Cp'] = (t['p_use'] - self.p_ref) / float(self.q_inf)
            fields['Cf'] = t['tau_mag'] / float(self.q_inf)
        a_tri, agg = aggregate_to_triangles(self.surfels, fields,
                                            self.n_faces)
        # level-local -> global (L0 lu) frame; identity on a single grid
        verts, faces = self.triangles_lu
        verts_out = (np.asarray(verts, dtype=np.float64)
                     * float(self.coord_spacing)
                     + np.asarray(self.coord_origin, dtype=np.float64))
        return write_triangle_surface(path, (verts_out, faces), a_tri, agg)

    # ------------------------------------------------------------- reporting
    def summary(self) -> str:
        p = self.params
        s = (f"surfel wall BC (S8a): {self.n_facets} facets, "
             f"prism nnz {self.nnz}, dropped dV {self.dropped_vol:.3g}, "
             f"law {p['law']} h={p['h_law']:g} fallback {p['fallback']} "
             f"dv_min {p['dv_min']:g}")
        if self._taum_summary:
            s += "\n            " + self._taum_summary
        return s


def check_level_coupling_bands(sb: SurfelBoundary, region,
                               span_through_axis=None, level=None) -> None:
    """S8a-2 guards: surfel state vs an MLG fine level's coupling bands.

    Two hard failure modes the body-vs-band mask check cannot see:

    1. span-through prism x z coupling: the prism/advect wrap along the
       span axis is PHYSICAL periodicity only when the fine face IS a
       domain face (boundary-flush, zero-width band — commit 2a19360).
       A z coupling interface through the facets would interpolate
       through the wall.
    2. band invasion by the surfel stencils: the C2F/F2C band (fine-
       domain edge .. fine_region edge) is overwritten by coupling every
       sub-step. Facet-coupled prism cells, partial (dV < 1) cells, or
       the wall-law sample points landing there would read/write
       coupling-owned data.

    `region` needs only `REFINE_RATIO`, `fine_region`,
    `fine_domain_coarse` (.<ax>_start/_end) and `flush_faces` — the
    OverlapRegion contract (gate s13 [L] fabricates a stub).

    Raises:
        ValueError: on any violation (never a silent downgrade).
    """
    tag = f"Level {level}" if level is not None else "fine level"
    ratio = int(getattr(region, 'REFINE_RATIO', 2))
    fr, fd = region.fine_region, region.fine_domain_coarse

    if span_through_axis:
        ax = str(span_through_axis)
        flush = getattr(region, 'flush_faces', {})
        if not (flush.get(f'{ax}_min', False)
                and flush.get(f'{ax}_max', False)):
            raise ValueError(
                f"{tag}: wall_bc='surfel' with span_through_axis="
                f"'{ax}' needs the fine region boundary-flush on BOTH "
                f"{ax} faces (fine face == domain face, no {ax} coupling "
                f"interface): the prism/advect wrap must be physical "
                f"periodicity, not a C2F/F2C band. Extend the region to "
                f"the full {ax} span.")

    # band widths [fine cells] per (axis, side); 0 on flush faces
    widths = []
    for ax_i, ax in enumerate('xyz'):
        w_lo = (getattr(fr, f'{ax}_start') - getattr(fd, f'{ax}_start')) \
            * ratio
        w_hi = (getattr(fd, f'{ax}_end') - getattr(fr, f'{ax}_end')) \
            * ratio
        widths.append((ax_i, ax, int(w_lo), int(w_hi)))

    # stencil footprints (host, level-local cells)
    cells = np.unique(sb.facets._t_cell)                 # prism CSR cells
    csr_xyz = np.stack(np.unravel_index(cells, sb.shape), axis=1)
    # partial cells (tolerance: the march's float64 column accumulation
    # may leave far-field fluid at 1 - eps, which is not a cut cell)
    part_xyz = np.argwhere(sb.live_h & (sb.dV_h < 1.0 - 1e-9))
    reach = max(float(sb.facets.h_law), float(sb.facets.sample_h)) + 1.0
    samp_xyz = (np.asarray(sb.facets.centroid, dtype=np.float64)
                + np.asarray(sb.facets.normal, dtype=np.float64) * reach)

    checks = (('prism-coupled cell', csr_xyz),
              ('partial (dV<1) cell', part_xyz),
              (f'wall-law sample point (reach {reach:g} cells)', samp_xyz))
    for ax_i, ax, w_lo, w_hi in widths:
        n_ax = sb.shape[ax_i]
        for name, xyz in checks:
            if xyz.size == 0:
                continue
            a = xyz[:, ax_i]
            # zero-width sides are boundary-flush (the fine face IS a
            # domain face): no band exists there, and geometry may
            # legitimately sit on/beyond the face (span-through prism,
            # boundary-clipped polygons) — skip, don't flag.
            n_bad = int((np.sum(a < w_lo) if w_lo > 0 else 0)
                        + (np.sum(a > n_ax - 1 - w_hi) if w_hi > 0 else 0))
            if n_bad:
                raise ValueError(
                    f"{tag}: {n_bad} surfel {name}(s) inside the C2F/F2C "
                    f"coupling band on axis {ax} (band widths lo={w_lo}/"
                    f"hi={w_hi} fine cells). Coupling would overwrite or "
                    f"feed the facet exchange. Enlarge "
                    f"mlg.levels[{level}].region so the band clears the "
                    f"surfel stencils (MLG region padding rule).")
