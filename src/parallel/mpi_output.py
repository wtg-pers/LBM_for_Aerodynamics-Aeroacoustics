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
from src.parallel.output import _gather_level, _LevelView, _MLGView


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
        if runner.model is not None:
            self._tier = 'alm'
        elif runner.body_level is not None:
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
        """COLLECTIVE: owned FLUID-cell rho mean + |u| max on the finest
        level (solid slab buffers are uninitialized — masked)."""
        r = self._runner
        L = r.lv[-1]
        sl = r.parts[-1].owned_local()
        fluid = (L.nt.reshape(L.dims)[sl] != 1)
        rho = L.rho[sl]
        u = L.u[(slice(None),) + sl]
        s_loc = float(rho[fluid].sum())
        n_loc = float(fluid.sum())
        usq = (u * u).sum(axis=0)
        try:
            import cupy as cp
            usq = cp.where(fluid, usq, 0.0)
        except Exception:
            usq = np.where(fluid, usq, 0.0)
        m_loc = float(usq.max()) ** 0.5
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

    def _gather_views(self):
        """COLLECTIVE per-level rho/u/nut gathers -> rank0 _MLGView."""
        comm, rank, nr, r = self._comm, self._rank, self._nr, self._runner
        views = []
        for k in range(r.NL):
            L = r.lv[k]
            rho = _gather_level(comm, rank, nr, r, L.rho[None], k,
                                tag0=300 + 4 * k)
            u = _gather_level(comm, rank, nr, r, L.u, k, tag0=302 + 4 * k)
            nut = None
            if getattr(L, "nut", None) is not None:
                nut = _gather_level(
                    comm, rank, nr, r,
                    L.nut.reshape((1,) + tuple(L.dims)), k, tag0=600 + 2 * k)
            if rank == 0:
                sm = (self._solid_masks[k]
                      if (self._solid_masks is not None
                          and k < len(self._solid_masks)) else None)
                views.append(_LevelView(
                    rho[0], u, nu_t=(None if nut is None else nut[0]),
                    solid_mask=sm))
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
        """COLLECTIVE owned_f_std gathers -> rank0 payload dict / None."""
        comm, rank, nr, r = self._comm, self._rank, self._nr, self._runner
        f_levels = []
        for k in range(r.NL):
            fk = _gather_level(comm, rank, nr, r, r.owned_f_std(k), k,
                               tag0=400 + 2 * k, pre_sliced=True)
            f_levels.append(fk)
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        rho0 = _gather_level(comm, rank, nr, r, r.lv[0].rho[None], 0,
                             tag0=460)
        u0 = _gather_level(comm, rank, nr, r, r.lv[0].u, 0, tag0=462)
        if rank != 0:
            return None
        extra = None
        if include_extra:
            # step_convention marker: 'index' = 0-based last-processed
            # label (C8 unified). Pre-unification main_mpi checkpoints
            # (step = advance count) lack it — restore prints a note.
            extra = {"num_levels": r.NL, "step_convention": "index"}
            for k in range(1, r.NL):
                extra[f"f_level_{k}"] = f_levels[k]
        return {'f': f_levels[0], 'extra': extra,
                'rho': rho0[0], 'u': u0}

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
        counts = []
        for shape in self._level_shapes:
            n = 1
            for d in shape:
                n *= int(d)
            counts.append(n)
        return counts

    def _free_transient_buffers(self, sim) -> None:
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    def _final_fields(self, sim):
        comm, rank, nr, r = self._comm, self._rank, self._nr, self._runner
        rho0 = _gather_level(comm, rank, nr, r, r.lv[0].rho[None], 0,
                             tag0=460)
        u0 = _gather_level(comm, rank, nr, r, r.lv[0].u, 0, tag0=462)
        if rank != 0:
            return None, None
        return rho0[0], u0

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
        result = super().finalize(sim)
        if self._dense_csv is not None:
            self._dense_csv.close()
            self._dense_csv = None
        if self._nr > 1:
            self._comm.Barrier()
        return result
