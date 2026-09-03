"""Surfel V2 — band-restricted sandwich bridge (patch 83, design = 63 §1).

Per substep on an INTERIOR surfel level (no domain faces):

    1. stage   : esoteric_gather_std_region(parity t) of the STAGE box
                 (physical std pre-collide f_t on the box scratch)
    2. std     : the production surfel chain on the box (fused collide
                 kernel -> zero_dead -> tau-band inject -> facet apply +
                 advect) — band cells get the EXACT std-path treatment
    3. fused   : esoteric cumulant kernel over the WHOLE level, in place
                 (all-fluid tags; band cells receive wrong values and
                 dead cells NaN — both covered by 4, and every neighbor
                 that pulls from them is inside the deposit box: 63 §1
                 containment accounting, verified at build)
    4. deposit : esoteric_scatter_std_region(parity t+1) of the DEPOSIT
                 box from the box chain's advect output

Key fact (63 §0): outside the static rewrite set R the fused streaming
is IDENTICAL to surfel_advect, and the collide closed forms agree
(63 §2 E-C1); the box chain is the std path bit-for-bit on its inputs.
Rate policy mirrors the V1 std chain exactly: w3..w10 = collision.omega_3
(the std fused collide kernel's omega_high slot receives omega_3 in
_advance_surfel_kernel), lambda_lim = 0.

Scope (83 §1, 0904): interior fine levels only — bc_manager must carry
zero faces/sponges/corners, no trip, no partial body. L0 (domain faces)
stays on the V1 full-staging bridge. Opt-in via LBM_SURFEL_V2=1.
"""
from __future__ import annotations

import numpy as np

from src.boundary.surfel_eso import (
    rewrite_mask, stage_and_deposit_boxes, verify_containment)


def _host(a, dtype=None):
    a = a.get() if hasattr(a, 'get') else np.asarray(a)
    return np.asarray(a, dtype=dtype) if dtype is not None else np.asarray(a)


def band_eligible(sim) -> str:
    """'' if this bridge sim can run the band sandwich, else the reason."""
    bm = sim.bc_manager
    if bm.n_face_bcs or bm.sponge_layers or bm.corner_bc is not None:
        return "domain faces/sponge/corners (L0-class level: V1 keeps it)"
    if getattr(sim, '_trip', None) is not None:
        return "trip forcing is wing-band state (not wired for V2)"
    sb = sim.obstacle_bc
    if getattr(sb, 'partial_body', False):
        return "partial-body is out of scope (83 §0)"
    if getattr(sb.kernel, 'wm_mode', 0) != 0:
        return "wall-model input filter carries per-facet state"
    if not sim._surfel_use_kernel():
        return "host collide chain (V2 mirrors the kernel chain only)"
    return ""


class SurfelBandBridge:
    """Static band boxes + box-local surfel chain + fused eso bulk."""

    def __init__(self, sim) -> None:
        xp = sim.xp
        sb = sim.obstacle_bc
        k = sb.kernel
        shape = tuple(sim.domain_shape)
        n = int(np.prod(shape))

        # ── deviation support M (63 §0; g/Q support == kernel.sup) ──
        M = np.zeros(n, dtype=bool)
        M |= (_host(sb.dV_h).reshape(-1) != 1.0)
        M |= ~_host(sb.live_h).reshape(-1).astype(bool)
        M[_host(k.sup, np.int64)] = True
        M[_host(k.cell, np.int64)] = True
        if getattr(sb, 'tau_model_on', False):
            M[_host(sb.d_tb_cells, np.int64)] = True
        M3 = M.reshape(shape)
        stage0, dep0 = stage_and_deposit_boxes(M3, stage_margin=2,
                                               deposit_margin=1)
        # per-axis containment contract: a FULL-WIDTH axis is fine (the
        # box %wrap equals the level %wrap — the slab full-axis
        # argument, and on an MPI slab it matches the E6 halo model); a
        # PARTIAL axis must both sit strictly inside the level (1-cell
        # advect-pull clearance) and contain every facet's wall-law
        # sample envelope. An axis failing either test is PROMOTED to
        # full width (monotone — at most 3 promotions), which always
        # satisfies both via the wrap.
        cen = _host(k.cen)
        nrm = _host(k.nrm)
        base = (max(float(k.f.sample_h),
                    float(getattr(k.f, 'p_sample_h', None) or 0.0))
                + float(k.h_law))
        pg = float(getattr(k, 'pg_ds', 0.0) or 0.0) \
            if int(getattr(k, 'pg_on', 0) or 0) else 0.0
        stage, dep = list(stage0), list(dep0)
        for a in range(3):
            full = slice(0, shape[a])
            if stage[a].start <= 0 or stage[a].stop >= shape[a]:
                stage[a], dep[a] = full, full
                continue
            reach = base * np.abs(nrm[:, a]) + pg + 1.0
            if ((cen[:, a] - reach < stage[a].start).any()
                    or (cen[:, a] + reach > stage[a].stop).any()):
                stage[a], dep[a] = full, full   # envelope wraps: full
        stage, dep = tuple(stage), tuple(dep)
        verify_containment(M3, stage, dep)
        self.stage = stage
        self.dep = dep
        self.stage_shape = tuple(s.stop - s.start for s in stage)
        self.n_box = int(np.prod(self.stage_shape))
        self.dep_local = tuple(
            slice(dep[a].start - stage[a].start,
                  dep[a].stop - stage[a].start) for a in range(3))

        self.sb_box = _build_box_surfel(sb, stage)

        # ── fused eso kernel (bulk) + all-fluid tags + scratch ──
        from src.kernels.esoteric_cumulant_d3q27 import (
            EsotericCumulantKernelD3Q27)
        sgs_on = bool(sim._sgs_cfg.get("enabled"))
        model = str(sim._sgs_cfg.get("model", "off")) if sgs_on else "off"
        if model not in ("off", "smagorinsky"):
            raise NotImplementedError(
                f"V2 sandwich SGS '{model}' (smagorinsky/off only — "
                "mirrors the V1 surfel kernel chain)")
        self._eso = EsotericCumulantKernelD3Q27(sgs_model=model)
        self._Cs = float(sim._sgs_cfg.get("Cs", 0.0)) if sgs_on else 0.0
        self._nt0 = xp.zeros(n, dtype=xp.int8)          # all NODE_FLUID
        self._bz = xp.zeros(n, dtype=xp.float32)        # unread bc slots
        self._rho_lvl = xp.zeros(n, dtype=xp.float32)
        self._u_lvl = xp.zeros((3, n), dtype=xp.float32)

        # box chain scratch (std fused collide kernel, V1's — 패치 55)
        from src.kernels.cumulant_d3q27 import CumulantCollideKernelD3Q27
        self._ck = CumulantCollideKernelD3Q27(
            sgs_model=('smagorinsky' if sgs_on else 'off'))
        self._rho_box = xp.zeros(self.n_box, dtype=xp.float32)
        self._u_box = xp.zeros((3, self.n_box), dtype=xp.float32)
        self._nut_box = (xp.zeros(self.n_box, dtype=xp.float32)
                         if sgs_on else None)
        self._o1 = 1.0 / float(sim.tau)
        self._ob = float(sim.collision.omega_bulk)
        self._o3 = float(sim.collision.omega_3)

    # ─────────────────────────────────────────────────────────────────
    def advance(self, sim) -> None:
        from src.kernels.esoteric_d3q27 import (
            esoteric_gather_std_region, esoteric_scatter_std_region)
        xp = sim.xp
        t = sim._esoteric_step
        ss = self.stage_shape
        q = 27

        # 1. stage: physical std pre-collide f_t on the box
        f_st = esoteric_gather_std_region(xp, sim.f, t, self.stage)

        # 2. std chain on the box (V1's kernel chain, box arrays)
        f_post = xp.empty_like(f_st)
        self._ck.launch(
            f_st.reshape(q, -1), f_post.reshape(q, -1),
            self._rho_box, self._u_box, None,
            self._o1, self._ob, self._o3, self.n_box,
            Cs=self._Cs, nu_t_out=self._nut_box)
        sbb = self.sb_box
        if self._nut_box is not None:
            sbb.zero_dead_scalar(self._nut_box)
        sbb.zero_dead(f_post)
        rho3 = self._rho_box.reshape(ss)
        u3 = self._u_box.reshape((3,) + ss)
        if getattr(sbb, 'tau_model_on', False):
            sbb.inject_tau_model(f_post, u3, sim.tau)
        f_new = xp.empty_like(f_st)
        sbb.apply_and_advect(f_post, f_new, rho3, u3)
        del f_post

        # 3. fused eso kernel, whole level, in place (band overwritten
        #    by 4; dead-cell NaN contained inside the deposit box)
        nx, ny, nz = sim.domain_shape
        nut_lvl = (sim.nu_t.reshape(-1) if sim.nu_t is not None else None)
        self._eso.launch(
            sim.f, self._rho_lvl, self._u_lvl, self._nt0,
            self._bz, self._bz, self._bz, self._bz,
            self._o1, self._ob, self._o3,
            nx, ny, nz, t_step=t, Cs=self._Cs, nu_t_out=nut_lvl)

        # 4. deposit the box chain's advect output (parity t+1)
        vals = f_new[(slice(None),) + self.dep_local]
        esoteric_scatter_std_region(xp, sim.f, xp.ascontiguousarray(vals),
                                    t + 1, self.dep)
        del f_new

        # output fields: bulk from the fused kernel, band from the box
        # chain, dead zeroed (fused writes NaN macro on f=0 cells)
        rho_out = self._rho_lvl.reshape(sim.domain_shape)
        u_out = self._u_lvl.reshape((3,) + sim.domain_shape)
        rho_out[self.dep] = rho3[self.dep_local]
        u_out[(slice(None),) + self.dep] = u3[(slice(None),) + self.dep_local]
        if nut_lvl is not None:
            nut3 = nut_lvl.reshape(sim.domain_shape)
            nut3[self.dep] = self._nut_box.reshape(ss)[self.dep_local]
        sb = sim.obstacle_bc
        sb.zero_dead_scalar(self._rho_lvl)
        sb.zero_dead_scalar(self._u_lvl[0])
        sb.zero_dead_scalar(self._u_lvl[1])
        sb.zero_dead_scalar(self._u_lvl[2])
        if nut_lvl is not None:
            sb.zero_dead_scalar(nut_lvl)
        sim.rho = rho_out
        sim.u = u_out
        # force ledger: the box chain's apply return IS the body force
        # (identical facet set) — hand it to the level's boundary object
        # so ForceManager/last_force read it unchanged
        sb._force = getattr(sbb, '_force', None)
        sb._force_merge = getattr(sbb, '_force_merge', None)
        sim._esoteric_step += 1
        sim.step_count += 1


def _build_box_surfel(sb, stage):
    """Full SurfelBoundary re-addressed onto an interior box.

    Every facet's CSR/tau-band cell lies inside the stage box by the
    band construction (M covers them, stage ⊇ box(M)+2), so NO facet is
    dropped — this is pure re-addressing (the build_slab_surfel pattern
    with a 3-axis window and no wrap), asserted below.
    """
    from types import SimpleNamespace
    from src.boundary.surfel_boundary import SurfelBoundary
    from src.kernels.surfel_d3q27 import SurfelKernelD3Q27

    xp = sb.xp
    k = sb.kernel
    full_shape = sb.shape
    box_shape = tuple(s.stop - s.start for s in stage)
    org = np.array([s.start for s in stage], dtype=np.int64)
    n_box = int(np.prod(box_shape))

    def remap_flat(cells):
        c = _host(cells, np.int64)
        Nx, Ny, Nz = full_shape
        co = np.stack([c // (Ny * Nz), (c // Nz) % Ny, c % Nz])
        loc = co - org[:, None]
        ok = ((loc >= 0)
              & (loc < np.array(box_shape, dtype=np.int64)[:, None]))
        if not ok.all():
            raise AssertionError(
                "box surfel: a facet cell escapes the stage box "
                "(band mask must cover the CSR — 83 §1)")
        return (loc[0] * box_shape[1] + loc[1]) * box_shape[2] + loc[2]

    def cells_slice(arr):
        lead = arr.shape[:-1]
        a3 = arr.reshape(lead + full_shape)
        return (np if isinstance(arr, np.ndarray) else xp).ascontiguousarray(
            a3[(slice(None),) * len(lead) + stage].reshape(lead + (n_box,)))

    sk = SurfelKernelD3Q27.__new__(SurfelKernelD3Q27)
    sk._k = k._k
    sk.block = k.block
    for name in ('mode', 'law_id', 'law_iters', 'h_law', 'nu',
                 'y_plus_min', 'fric_dir', 'fb_mode', 'wm_mode', 'wm_tf',
                 'pg_on', 'pg_ds', 'p_h', '_per'):
        setattr(sk, name, getattr(k, name))
    sk.n_f = int(k.n_f)
    sk.shape = box_shape
    sk.N = n_box
    # facet-major arrays: SHARED with the full kernel (every facet is
    # kept, order identical) — per-facet OUTPUT/STATE arrays shared too,
    # so the force ledger, surface channels, tau-band persistence and
    # the wall-law utau seed flow through the existing readers untouched
    sk.indptr = k.indptr                       # all facets kept verbatim
    sk.cell = xp.asarray(remap_flat(k.cell).astype(np.int32))
    sk.wgt = k.wgt
    sk.nrm = k.nrm
    sk.area = k.area
    cen_box = _host(k.cen).copy()
    cen_box -= org[None, :]
    sk.cen = xp.asarray(np.ascontiguousarray(cen_box))
    sk.Vsum = k.Vsum
    sk.gamma = k.gamma
    sk.crease = k.crease
    sup_box = remap_flat(k.sup)
    order = np.argsort(sup_box, kind='stable')
    sk.sup = sup_box[order]
    sk.n_sup = int(sk.sup.size)
    qmap_box = np.full(n_box, -1, dtype=np.int32)
    qmap_box[sk.sup] = np.arange(sk.n_sup, dtype=np.int32)
    sk.qmap = xp.asarray(qmap_box)
    sk.cellc = xp.asarray(qmap_box[_host(sk.cell, np.int64)])
    sk.g_field = k.g_field[:, xp.asarray(order)]
    sk.G_in = k.G_in
    sk.G_out = k.G_out
    sk.Q = None
    sk.tau_out = k.tau_out
    sk.fb_out = k.fb_out
    sk.rho_out = k.rho_out
    sk.rho_out2 = k.rho_out2
    sk.u_wm = k.u_wm
    sk.utau_prev = k.utau_prev
    sk._wm_seed = getattr(k, '_wm_seed', 1)
    sk.f = SimpleNamespace(
        sample_h=k.f.sample_h,
        p_sample_h=getattr(k.f, 'p_sample_h', None),
        cdotn=np.ascontiguousarray(np.asarray(k.f.cdotn)))

    # inherit the INPUT's class: on an MPI slab the sb is a
    # SlabSurfelBoundary whose inject/last_force overrides consume the
    # slab tau-band field set — the box re-address must keep them
    out = type(sb).__new__(type(sb))
    out.xp = xp
    out.shape = box_shape
    out.kernel = sk
    out.n_facets = sk.n_f
    out.d_live = cells_slice(sb.d_live)
    out.d_dead = cells_slice(sb.d_dead)
    out.d_dV = cells_slice(sb.d_dV)
    out.dV_h = cells_slice(np.ascontiguousarray(
        sb.dV_h.reshape(-1))).reshape(box_shape)
    out.live_h = cells_slice(np.ascontiguousarray(
        sb.live_h.reshape(-1))).reshape(box_shape)
    out._solid_mask_dev = None
    out._d_CC = None
    out._force = None
    for name in ('params', 'dv_min', 'tau_model_on', 'collide_path',
                 'q_inf', 'p_ref', 'coord_origin', 'coord_spacing',
                 'p_sample_h', 'kh_star'):
        setattr(out, name, getattr(sb, name))
    out.facets = sk.f
    if getattr(sb, 'tau_model_on', False):
        # cells re-addressed to box coords; every facet-space tau-band
        # array (full OR slab-extended convention) shared verbatim
        out.d_tb_cells = xp.asarray(remap_flat(sb.d_tb_cells))
        for name in ('d_tb_fs', 'd_tb_normal', 'd_tb_C', 'd_tb_Wl',
                     '_tb_W', '_tb_tau_ext', 'd_tb_local_slots',
                     'd_tb_local_rows', 'tb_needed_gids',
                     'tb_phantom_gids'):
            if hasattr(sb, name):
                setattr(out, name, getattr(sb, name))
    return out
