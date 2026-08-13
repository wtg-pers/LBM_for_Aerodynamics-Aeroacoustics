"""MPIOutputManager — the full OutputManager pipeline on the MPI runner.

Subclass of OutputManager that overrides the C2 data seams with collective
implementations (src/parallel/output.py gather backend + scalar Allreduces).
One cadence/CSV/coefficient/finalize source for both entry paths.

Contract (deadlock safety):
  - Every collective seam is reached by EVERY rank at the same step: all
    cadence flags are replicated values (CLI/config), and the convergence
    'stop' verdict is computed from Allreduced scalars (identical monitor
    state on every rank) plus a 1-byte MAX-allreduce belt.
  - rank != 0 joins every gather/reduction and skips only file writes
    (its writers/CSV paths are None from io_role='silent'; _io_rank=False
    guards the rest).

Semantic notes vs the single-GPU path (documented, intentional):
  - Conservation/convergence field statistics EXCLUDE solid cells (the
    esoteric kernel leaves rho/u uninitialized there; the slab arrays are
    cp.empty garbage). Conservation baselines are re-initialized from the
    collective fluid-only masses at bind time, so drift is measured
    against the run's start exactly like the single path.
  - Checkpoint rho/u are the assembled L0 fields (solid cells carry the
    same garbage semantics as the single esoteric path; restore reads f
    only).
"""

from __future__ import annotations

from typing import Optional

import os
import numpy as np

from src.solver.output_manager import OutputManager
from src.parallel.output import (
    _gather_block, _BlockView, _LevelView, _MLGView)
from src.parallel.runner import TAG_FIELD, TAG_CKPT, TAG_MACRO


class MPIOutputManager(OutputManager):

    def __init__(self, comm=None, rank: int = 0, nr: int = 1, mpi_mod=None,
                 log_every: Optional[int] = None,
                 vtk_every: int = 0, ckpt_every: int = 0,
                 vtk_fields_last: int = 0,
                 dense_csv_path: Optional[str] = None,
                 **base_kwargs) -> None:
        super().__init__(**base_kwargs)
        self._comm, self._rank, self._nr = comm, rank, nr
        self._MPI = mpi_mod
        self._io_rank = (rank == 0)
        if log_every is not None:
            self.log_interval = max(1, int(log_every))
        self._vtk_every = int(vtk_every or 0)
        self._ckpt_every = int(ckpt_every or 0)
        self._vtk_fields_last = int(vtk_fields_last or 0)
        self._dense_csv_path = dense_csv_path if self._io_rank else None
        self._dense_csv = None
        self._runner = None
        self._level_shapes = None
        self._solid_masks = None
        self._tier = 'flow'
        self._flow_cache = None
        self._t_last = None
        self._t0 = None

    # ── binding ─────────────────────────────────────────────────────

    def bind_runner(self, runner, level_shapes, solid_masks=None,
                    dist_init: bool = False, start_step: int = 0) -> None:
        """Attach the runner + replicated static metadata. COLLECTIVE
        (conservation re-baseline does a SUM Allreduce)."""
        self._runner = runner
        self._level_shapes = [tuple(s) for s in level_shapes]
        self._solid_masks = solid_masks
        if runner.has_alm:
            self._tier = 'alm'
        elif runner.body_block is not None:
            self._tier = 'body'
        else:
            self._tier = 'flow'

        if self.conservation_mgr is not None and dist_init:
            # No full-field baseline exists under --dist-init; refuse
            # to silently report garbage drift.
            if self._io_rank:
                print("[output] conservation: DISABLED under --dist-init "
                      "(no full-field baseline)")
            self.conservation_mgr = None
        # NOTE (S4-C finding, 2026-07-31): do NOT re-baseline M0 from
        # _cv_masses() here. At bind time no kernel has run yet, so the
        # slab rho buffers are unwritten (collective "mass" ~1.3% of the
        # real one) — the re-baseline replaced the initializer's valid
        # full-field M0 and every later drift read ~7705%. The
        # initializer's M0 stands (same fluid-only basis); if a CV ever
        # arrives uninitialized, check_from_mass() self-baselines at the
        # first check with VALID collective masses.

        if self._dense_csv_path:
            import os
            append = start_step > 0 and os.path.exists(self._dense_csv_path)
            self._dense_csv = open(self._dense_csv_path,
                                   "a" if append else "w")
            if not append:
                if self._tier == 'alm':
                    self._dense_csv.write(
                        "step,time_lu,thrust,torque,power,C_T,C_P,FM\n")
                elif self._tier == 'body':
                    self._dense_csv.write("step,Fx,Fy,Fz,CD,CL,CS\n")
                else:
                    self._dense_csv.write("step,rho_mean,u_max\n")

    # ── collective primitives ───────────────────────────────────────

    def _allreduce(self, vec: np.ndarray, op=None) -> np.ndarray:
        if self._nr <= 1:
            return vec
        out = np.empty_like(vec)
        if op is None:
            self._comm.Allreduce(vec, out)
        else:
            self._comm.Allreduce(vec, out, op=op)
        return out

    def _l0_owned(self):
        """(rho, u, fluid_mask) on the OWNED L0 slab (device arrays)."""
        r = self._runner
        p = r.parts[0]
        L = r.lv[0]
        sl = p.owned_local()
        rho = L.rho[sl]
        u = L.u[(slice(None),) + sl]
        fluid = (L.nt.reshape(L.dims)[sl] != 1)
        return rho, u, fluid

    def _cv_masses(self) -> dict:
        """Fluid-only CV masses: slab partials + one SUM Allreduce."""
        r = self._runner
        p = r.parts[0]
        rho, _u, fluid = self._l0_owned()
        rho_f = rho * fluid
        names, partials = [], []
        for name, cv in self.conservation_mgr.cv_items():
            lo = [cv.x0, cv.y0, cv.z0]
            hi = [cv.x1, cv.y1, cv.z1]
            a = p.axis
            lo_a = max(lo[a], p.own_start)
            hi_a = min(hi[a], p.own_start + p.own_count - 1)
            names.append(name)
            if lo_a > hi_a:
                partials.append(0.0)
                continue
            idx = [slice(lo[d], hi[d] + 1) for d in range(3)]
            idx[a] = slice(lo_a - p.own_start, hi_a - p.own_start + 1)
            partials.append(float(rho_f[tuple(idx)].sum()))
        out = self._allreduce(np.asarray(partials, dtype=np.float64))
        return dict(zip(names, out.tolist()))

    def _flow_stats(self):
        """COLLECTIVE: owned FLUID-cell rho mean + |u| max over the finest
        level (solid slab buffers are uninitialized — masked).

        Summed over every block of that level this rank owns: with several
        blocks per level, reading only the last one would silently report a
        fraction of the domain."""
        r = self._runner
        s_loc = n_loc = m_loc = 0.0
        for uid in r.blocks_at(r.NL - 1):
            if not r.owns[uid]:
                continue
            L = r.lv[uid]
            sl = r.parts[uid].owned_local()
            fluid = (L.nt.reshape(L.dims)[sl] != 1)
            rho = L.rho[sl]
            u = L.u[(slice(None),) + sl]
            s_loc += float(rho[fluid].sum())
            n_loc += float(fluid.sum())
            usq = (u * u).sum(axis=0)
            try:
                import cupy as cp
                usq = cp.where(fluid, usq, 0.0)
            except Exception:
                usq = np.where(fluid, usq, 0.0)
            m_loc = max(m_loc, float(usq.max()) ** 0.5)
        sums = self._allreduce(np.array([s_loc, n_loc], dtype=np.float64))
        mx = self._allreduce(np.array([m_loc], dtype=np.float64),
                             op=self._MPI.MAX if self._nr > 1 else None)
        return sums[0] / max(sums[1], 1.0), float(mx[0])

    # ── seam overrides: force ───────────────────────────────────────

    def _force_inputs(self, step, sim):
        F_loc = self._runner.mem_force_local()
        F_tot = self._allreduce(np.asarray(F_loc, dtype=np.float64))
        self._last_F = F_tot          # dense-CSV body row source
        return None, F_tot

    # ── seam overrides: progress / dense CSV ────────────────────────

    def start(self, start_step: int, end_step: int) -> None:
        import time as _time
        self._start_step = start_step
        self._end_step = end_step
        self._pbar = None
        self._start_time = _time.perf_counter()
        self._t0 = self._start_time
        self._t_last = self._start_time

    def process(self, step, sim):
        # collectives that feed the progress line must run on EVERY rank
        # (never behind a rank/pbar guard)
        log_due = bool(self.log_interval) and step % self.log_interval == 0
        if self._tier == 'flow' and log_due:
            self._flow_cache = self._flow_stats()
        # Rotor CSV + progress line fire at log_interval, marker VTP at the
        # VTK cadence; every one of them reads a model this rank may not own.
        if self._tier == 'alm' and (log_due or self._vtk_due(step)):
            self._runner.sync_alm_reporting()
        action = super().process(step, sim)
        # progress line + dense CSV AFTER the base channels so the force
        # caches (_last_F/_last_Cd) are from THIS step's force interval,
        # not the previous one (single force source, no lag).
        if log_due:
            self._progress_line(step)
        # stop-verdict belt: monitor state is rank-invariant by
        # construction (all feeds are Allreduced scalars); this 1-double
        # MAX makes divergence in control flow structurally impossible.
        if (self._nr > 1 and step % self.check_interval == 0
                and step > self._start_step):
            flag = self._allreduce(
                np.array([1.0 if action == 'stop' else 0.0]),
                op=self._MPI.MAX)
            if flag[0] > 0.0:
                action = 'stop'
        return action

    def _update_progress(self, step, sim) -> None:
        # replaced by _progress_line at the END of process() (force caches
        # must be fresh); the base per-step tqdm has no MPI counterpart
        return

    def _progress_line(self, step) -> None:
        import time as _time
        now = _time.perf_counter()
        sps = (now - self._t_last) / max(self.log_interval, 1)
        self._t_last = now
        if not self._io_rank:
            return
        line = f"[mpi] step {step}/{self._end_step - 1}"
        eta_h = (self._end_step - 1 - step) * sps / 3600.0
        line += f"  {sps:.3f}s/step  ETA {eta_h:.2f}h"
        row = None
        if self._tier == 'alm' and self.al_model is not None:
            # A multi-rotor manager returns {name: perf}; report the ROTOR SUM
            # for thrust/torque/power and the mean coefficient, so the live
            # line means the same thing for one rotor or eight.
            _models = getattr(self.al_model, 'models', None)
            if _models:
                _ps = [m.get_rotor_performance() for m in _models]
                _ps = [p for p in _ps if 'C_T' in p]
                if not _ps:
                    return
                _n = len(_ps)
                perf = {
                    'time': _ps[0].get('time', 0.0),
                    'thrust': sum(p['thrust'] for p in _ps),
                    'torque': sum(p['torque'] for p in _ps),
                    'power': sum(p['power'] for p in _ps),
                    'C_T': sum(p['C_T'] for p in _ps) / _n,
                    'C_P': sum(p['C_P'] for p in _ps) / _n,
                    'FM': sum(p.get('FM', 0.0) for p in _ps) / _n,
                }
            else:
                perf = self.al_model.get_rotor_performance()
                if 'C_T' not in perf:
                    return
            line += (f"  CT={perf['C_T']:.6e}  CP={perf['C_P']:.6e}"
                     f"  FM={perf['FM']:.4f}")
            row = (f"{step},{perf['time']},{perf['thrust']},"
                   f"{perf['torque']},{perf['power']},"
                   f"{perf['C_T']},{perf['C_P']},{perf['FM']}\n")
        elif self._tier == 'body':
            line += (f"  CD={self._last_Cd:.4f}  CL={self._last_Cl:.4f}"
                     f"  CS={self._last_Cz:.4f}")
            F = getattr(self, '_last_F', None)
            if F is not None:
                row = (f"{step},{F[0]},{F[1]},{F[2]},"
                       f"{self._last_Cd},{self._last_Cl},"
                       f"{self._last_Cz}\n")
        elif self._flow_cache is not None:
            rho_m, u_max = self._flow_cache
            line += f"  rho={rho_m:.6f}  u_max={u_max:.5f}"
            row = f"{step},{rho_m},{u_max}\n"
        if self._dense_csv is not None and row is not None:
            self._dense_csv.write(row)
            self._dense_csv.flush()
        print(line, flush=True)

    # ── seam overrides: VTK ─────────────────────────────────────────

    def _vtk_due(self, step) -> bool:
        return bool(self._vtk_every) and step % self._vtk_every == 0

    def _mask_solid(self, uid, rho, u):
        """Single-GPU solid-node convention on a gathered block (rank 0).

        The slab rho/u buffers are never written at SOLID nodes (module
        docstring), so what the gather assembles there is meaningless —
        measured rho = 0 and |u| = sqrt(3) against a physical u_max ~ 5e-2,
        i.e. 35x the real range. Every auto-scaled colour bar in ParaView
        then reads the body instead of the flow, which is exactly the
        symptom `solid_mask` was added for (3b3f080): the mask lets you
        threshold the values away, it does not stop them landing in the
        file. The single-GPU path shows the seeded rest state there, so
        write that.

        Applied in ONE place because the L0 gather is done from three call
        sites (VTK views, checkpoint payload, finalize) and each drifting
        on its own is how this class of defect is born.
        """
        if self._solid_masks is None or uid >= len(self._solid_masks):
            return
        sm = self._solid_masks[uid]
        if sm is None:
            return
        s = np.asarray(sm, dtype=bool)
        if s.shape != rho.shape:
            return
        rho[s] = 1.0
        u[:, s] = 0.0

    def _gather_l0_macros(self):
        """COLLECTIVE L0 rho/u gather -> (rho, u) on rank 0, (None, None) off."""
        comm, rank, nr, r = self._comm, self._rank, self._nr, self._runner
        rho0 = _gather_block(comm, rank, nr, r.parts[0],
                             r.lv[0].rho[None], TAG_MACRO, 1)
        u0 = _gather_block(comm, rank, nr, r.parts[0], r.lv[0].u,
                           TAG_MACRO + 2, 3)
        if rank != 0:
            return None, None
        self._mask_solid(0, rho0[0], u0)
        return rho0[0], u0

    def _gather_views(self):
        """COLLECTIVE per-BLOCK rho/u/nut gathers -> rank0 _MLGView.

        Every rank joins every block's gather (a rank owning nothing of a
        block contributes a zero-length piece) and the nu_t presence flag is
        read off the replicated build, not off this rank's slab — otherwise
        the collective would depend on who owns what."""
        comm, rank, nr, r = self._comm, self._rank, self._nr, self._runner
        views = []
        for uid, b in enumerate(r.blocks):
            L = r.lv[uid] if r.owns[uid] else None
            part = r.parts[uid]
            tag = TAG_FIELD + 8 * uid
            rho = _gather_block(comm, rank, nr, part,
                                None if L is None else L.rho[None], tag, 1)
            u = _gather_block(comm, rank, nr, part,
                              None if L is None else L.u, tag + 2, 3)
            nut = None
            if r.has_nut[uid]:
                nut = _gather_block(
                    comm, rank, nr, part,
                    None if L is None
                    else L.nut.reshape((1,) + tuple(L.dims)), tag + 4, 1)
            if rank == 0:
                sm = (self._solid_masks[uid]
                      if (self._solid_masks is not None
                          and uid < len(self._solid_masks)) else None)
                self._mask_solid(uid, rho[0], u)
                views.append(_BlockView(b.level, b.index, uid, _LevelView(
                    rho[0], u, nu_t=(None if nut is None else nut[0]),
                    solid_mask=sm)))
        return _MLGView(views) if rank == 0 else None

    def _sim_vtk_target(self, step, sim):
        return self._gather_views()

    def _write_vtk(self, step, sim) -> None:
        if not self._vtk_due(step):
            return
        fields = (self._vtk_fields_last <= 0 or
                  step > (self._end_step - 1)
                  - self._vtk_fields_last * self._vtk_every)
        target = self._gather_views() if fields else None
        if self._io_rank:
            if fields and self.mlg_vtk_writer is not None \
                    and target is not None:
                self.mlg_vtk_writer.write(step=step, mlg=target,
                                          time=float(step))
            if (self.marker_vtk_writer is not None
                    and self.al_model is not None):
                self._write_markers(step)
        if self._nr > 1:
            self._comm.Barrier()

    # ── seam overrides: checkpoint ──────────────────────────────────

    def _ckpt_due(self, step) -> bool:
        return (bool(self._ckpt_every) and step > 0
                and step % self._ckpt_every == 0)

    def _checkpoint_payload(self, sim, include_extra: bool = True):
        """COLLECTIVE per-block owned_f_std gathers -> rank0 payload / None.

        Key rule mirrors the single-GPU writer exactly: no block suffix when
        the level holds one block, so a chain writes the SAME key set as
        before and either reader loads either file."""
        comm, rank, nr, r = self._comm, self._rank, self._nr, self._runner
        f_blocks = []
        for uid in range(len(r.blocks)):
            fk = _gather_block(comm, rank, nr, r.parts[uid],
                               r.owned_f_std_block(uid), TAG_CKPT + 4 * uid,
                               r.n_pop, pre_sliced=True)
            f_blocks.append(fk)
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        # eso implicit-wall mailbox (L0-only until fine walls exist;
        # eso_wall §4-5b): cell-local state OUTSIDE the f slots. Gather
        # each rank's OWNED strips (collective — must run on every rank)
        # and assemble the global buffer on rank 0 under the SAME npz
        # keys as the single-GPU writer, so either reader restarts
        # either file.
        wall_mask = int(getattr(r.lv[0], 'wall_mask', 0)) \
            if r.owns[0] else 0
        wall_strips = None
        if include_extra and wall_mask:
            import cupy as cp
            from src.kernels.esoteric_d3q27 import eso_wall_mail_layout
            p0 = r.parts[0]
            lay_l, _ = eso_wall_mail_layout(wall_mask,
                                            tuple(p0.local_shape))
            own_sl = slice(p0.ghost, p0.ghost + p0.own_count)
            strips = {}
            for bit, (off, area, (l1, l2), axes) in lay_l.items():
                seg = r.lv[0].wall_mail[off:off + 9 * area] \
                    .reshape(9, l1, l2)
                sl = [slice(None)] * 3
                sl[1 + axes.index(r.axis)] = own_sl
                strips[bit] = cp.asnumpy(seg[tuple(sl)])
            wall_strips = comm.gather(strips, root=0) if comm is not None \
                else [strips]

        # Restore reads f only, so checkpoint rho/u are informational — but
        # "informational" is not a licence to differ from the single path.
        rho_c, u_c = self._gather_l0_macros()
        if rank != 0:
            return None
        extra = None
        if include_extra:
            extra = {"num_levels": r.NL}
            per = {}
            for b in r.blocks:
                per[b.level] = per.get(b.level, 0) + 1
            # Unconditional, matching OutputManager._build_checkpoint_extra:
            # it writes these whenever the grid exposes blocks, so gating
            # them on `> NL` made a one-block-per-level chain produce a
            # DIFFERENT key set under MPI than on one GPU. Nothing reads
            # them today, which is precisely why the drift went unnoticed.
            extra["num_blocks"] = len(r.blocks)
            extra["block_levels"] = [b.level for b in r.blocks]
            for uid, b in enumerate(r.blocks):
                if b.level == 0:
                    continue
                sfx = "" if per[b.level] <= 1 else f"_b{b.index}"
                extra[f"f_level_{b.level}{sfx}"] = f_blocks[uid]
            if wall_mask:
                import numpy as _np
                from src.kernels.esoteric_d3q27 import eso_wall_mail_layout
                p0 = r.parts[0]
                g_dims = list(p0.local_shape)
                g_dims[r.axis] = sum(c for _s, c in r.range_table[0])
                lay_g, tot_g = eso_wall_mail_layout(wall_mask,
                                                    tuple(g_dims))
                gm = _np.zeros(tot_g, _np.float32)
                for rk, st in enumerate(wall_strips):
                    s0, cnt = r.range_table[0][rk]
                    if cnt == 0 or not st:
                        continue
                    for bit, arr in st.items():
                        off, area, (g1, g2), axes = lay_g[bit]
                        seg = gm[off:off + 9 * area].reshape(9, g1, g2)
                        sl = [slice(None)] * 3
                        sl[1 + axes.index(r.axis)] = slice(s0, s0 + cnt)
                        seg[tuple(sl)] = arr
                extra["wall_mask_L0"] = wall_mask
                extra["wall_mail_L0"] = gm
        return {'f': f_blocks[0], 'extra': extra,
                'rho': rho_c, 'u': u_c}

    def _save_checkpoint(self, step, sim) -> None:
        if not self._ckpt_due(step):
            return
        payload = self._checkpoint_payload(sim)
        if payload is not None and self.checkpoint_mgr is not None:
            self.checkpoint_mgr.save(
                step=step, f=payload['f'], rho=payload['rho'],
                u=payload['u'], tau=self.tau, config=self.sim_params,
                extra_data=payload['extra'])
        # set on EVERY rank (payload is None off-rank0): finalize's
        # skip-duplicate decision must be rank-invariant or the payload
        # gather deadlocks
        self._last_ckpt_step = step
        if self._nr > 1:
            self._comm.Barrier()

    def _emergency_checkpoint(self, step, sim,
                              include_extra: bool = True) -> None:
        payload = self._checkpoint_payload(sim, include_extra=include_extra)
        if payload is not None and self.checkpoint_mgr is not None:
            kwargs = dict(step=step, f=payload['f'], rho=payload['rho'],
                          u=payload['u'], tau=self.tau,
                          config=self.sim_params)
            if include_extra:
                kwargs['extra_data'] = payload['extra']
            self.checkpoint_mgr.save(**kwargs)
        self._last_ckpt_step = step
        if self._nr > 1:
            self._comm.Barrier()

    # ── seam overrides: conservation / convergence ──────────────────

    def _check_conservation(self, step, sim) -> None:
        if self.conservation_mgr is None:
            return
        masses = self._cv_masses()
        results = self.conservation_mgr.check_from_masses(
            masses, step, verbose=(self.conservation_mgr.verbose > 0))
        if results.get('domain'):
            self._last_drift = results['domain']['mass_drift_percent']

    def _feed_convergence(self, step, sim) -> None:
        rho, u, fluid = self._l0_owned()
        usq = (u * u).sum(axis=0)
        e_loc = float((0.5 * rho * usq)[fluid].sum())
        n_loc = float(fluid.sum())
        rho_f = rho[fluid]
        nan_loc = 1.0 if bool(
            (~np.isfinite(rho_f.get() if hasattr(rho_f, 'get')
                          else rho_f)).any()) else 0.0
        dev_loc = float(abs(rho_f - 1.0).max()) if n_loc else 0.0
        rmin_loc = float(rho_f.min()) if n_loc else 1.0
        rmax_loc = float(rho_f.max()) if n_loc else 1.0
        umax_loc = float(usq[fluid].max()) ** 0.5 if n_loc else 0.0

        sums = self._allreduce(np.array([e_loc, n_loc]))
        op_max = self._MPI.MAX if self._nr > 1 else None
        mx = self._allreduce(
            np.array([nan_loc, dev_loc, rmax_loc, -rmin_loc, umax_loc]),
            op=op_max)
        E = sums[0] / max(sums[1], 1.0)
        self.conv_monitor.feed_energy_value(step, E)
        self.conv_monitor.feed_divergence_scalars({
            'rho_nan_inf': mx[0] > 0.0,
            'rho_dev': float(mx[1]),
            'rho_max': float(mx[2]),
            'rho_min': float(-mx[3]),
            'u_max': float(mx[4]),
        })

    # ── seam overrides: finalize ────────────────────────────────────

    def _level_cell_counts(self, sim):
        # One entry per LEVEL, summed over that level's blocks: the caller
        # uses the list index as the 2^k sub-step exponent, so it must stay
        # level-indexed even when a level holds several grids.
        r = self._runner
        counts = [0] * r.NL
        for uid, b in enumerate(r.blocks):
            n = 1
            for d in r.parts[uid].global_shape:
                n *= int(d)
            counts[b.level] += n
        return counts

    def _free_transient_buffers(self, sim) -> None:
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _final_fields(self, sim):
        return self._gather_l0_macros()

    def _field_nan_flag(self, rho_final) -> bool:
        rho, _u, fluid = self._l0_owned()
        rho_f = rho[fluid]
        arr = rho_f.get() if hasattr(rho_f, 'get') else np.asarray(rho_f)
        loc = 1.0 if bool((~np.isfinite(arr)).any()) else 0.0
        out = self._allreduce(np.array([loc]),
                              op=self._MPI.MAX if self._nr > 1 else None)
        return out[0] > 0.0

    def _final_conservation(self, rho_final, final_step) -> None:
        if self.conservation_mgr is None:
            return
        print(f"\n[8] Final Conservation Analysis")
        masses = self._cv_masses()
        self.conservation_mgr.check_from_masses(masses, final_step,
                                                verbose=True)
        self.conservation_mgr.close()

    def finalize(self, sim):
        if self._tier == 'alm':
            self._runner.sync_alm_reporting()
        result = super().finalize(sim)
        if self._dense_csv is not None:
            self._dense_csv.close()
            self._dense_csv = None
        if self._nr > 1:
            self._comm.Barrier()
        return result
