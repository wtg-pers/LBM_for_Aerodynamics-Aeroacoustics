"""Rank-0 assembled field output for the MPI runner (patch 17 M5b).

VTK and checkpoints reuse the PRODUCTION writers byte-for-byte: rank 0 keeps
the setup's MLGVTKWriter and CheckpointManager (built during its own full
build), every rank sends its owned slabs (host-staged numpy), and rank 0
assembles the global per-level arrays and calls the same write()/save() the
single-GPU loop calls. Checkpoints are therefore restartable by BOTH the
single-GPU path and the MPI runner (each rank restores the global state
through the normal initializer, then slab-extracts as always).

All entry points are COLLECTIVE — every rank must call them at the same
coarse step (non-root ranks only send).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cupy as cp


def _gather_level(comm, rank: int, nr: int, runner, arr_local,
                  k: int, tag0: int, pre_sliced: bool = False):
    """Assemble owned slabs of a (C, x, y, z) LOCAL array on rank 0.

    pre_sliced=True when the caller already passes OWNED-only data
    (e.g. runner.owned_f_std) — slicing twice drops 2*ghost rows per
    rank (caught by the bench5 npz shape check)."""
    part = runner.parts[k]
    own = (arr_local if pre_sliced
           else arr_local[(slice(None),) + part.owned_local()])
    own = cp.asnumpy(own) if hasattr(own, "get") else np.asarray(own)
    if rank != 0:
        comm.send((own.shape, own.dtype.str), dest=0, tag=tag0)
        comm.Send(np.ascontiguousarray(own), dest=0, tag=tag0 + 1)
        return None
    pieces = [own]
    for r in range(1, nr):
        shape, dt = comm.recv(source=r, tag=tag0)
        buf = np.empty(shape, dtype=np.dtype(dt))
        comm.Recv(buf, source=r, tag=tag0 + 1)
        pieces.append(buf)
    return np.concatenate(pieces, axis=1 + runner.axis)


class _MaskCarrier:
    """Minimal duck for MLGVTKWriter's `obstacle_bc.solid_mask` read."""

    def __init__(self, solid_mask):
        self.solid_mask = solid_mask


class _LevelView:
    """Duck-typed level for MLGVTKWriter (needs .rho/.u/.nu_t/.obstacle_bc)."""

    def __init__(self, rho, u, nu_t=None, solid_mask=None):
        self.rho = rho
        self.u = u
        self.nu_t = nu_t
        # solid_mask is static -> captured once at bridge construction on
        # rank 0 (pre-fix this was hardwired None, silently dropping the
        # solid_mask array from every MPI VTK snapshot).
        self.obstacle_bc = (None if solid_mask is None
                            else _MaskCarrier(solid_mask))


class _MLGView:
    def __init__(self, levels):
        self._levels = levels

    def get_level(self, k):
        return self._levels[k]


class Rank0OutputBridge:
    """Collective VTK / checkpoint output through the production writers."""

    def __init__(self, comm, rank: int, nr: int,
                 mlg_vtk_writer=None, checkpoint_mgr=None,
                 sim_params: Optional[dict] = None,
                 tau: float = 0.5,
                 marker_vtk_writer=None,
                 alm_marker_origin=None,
                 alm_marker_spacing=None,
                 solid_masks: Optional[list] = None) -> None:
        self._comm, self._rank, self._nr = comm, rank, nr
        self._vtk = mlg_vtk_writer
        self._ckpt = checkpoint_mgr
        self._sim_params = sim_params or {}
        self._tau = tau
        self._marker_vtk = marker_vtk_writer
        self._m_origin = alm_marker_origin
        self._m_spacing = alm_marker_spacing
        # Per-level static solid masks (host numpy or None), rank 0 only.
        self._solid_masks = solid_masks

    # ── VTK: assembled rho/u per level -> MLGVTKWriter.write ─────────
    def write_vtk(self, step: int, runner, fields: bool = True) -> None:
        """fields=False skips the level gathers + field files (the per-
        snapshot GB cost) but still writes the marker VTPs. The flag must be
        computed identically on every rank (the gathers are collective)."""
        comm, rank, nr = self._comm, self._rank, self._nr
        if fields:
            views = []
            for k in range(runner.NL):
                L = runner.lv[k]
                rho = _gather_level(comm, rank, nr, runner,
                                    L.rho[None], k, tag0=300 + 4 * k)
                u = _gather_level(comm, rank, nr, runner,
                                  L.u, k, tag0=302 + 4 * k)
                # SGS eddy viscosity (dyn_smag levels allocate .nut, flat N —
                # config-uniform across ranks, so the gather stays collective)
                nut = None
                if getattr(L, "nut", None) is not None:
                    nut = _gather_level(
                        comm, rank, nr, runner,
                        L.nut.reshape((1,) + tuple(L.dims)), k,
                        tag0=600 + 2 * k)
                if rank == 0:
                    sm = (self._solid_masks[k]
                          if (self._solid_masks is not None
                              and k < len(self._solid_masks)) else None)
                    views.append(_LevelView(
                        rho[0], u, nu_t=(None if nut is None else nut[0]),
                        solid_mask=sm))
            if rank == 0 and self._vtk is not None:
                self._vtk.write(step, _MLGView(views), time=float(step))
        self._write_markers(step, runner)
        comm.Barrier()

    def _write_markers(self, step: int, runner) -> None:
        """ALM marker VTP from rank 0's REPLICATED model — no comms needed:
        the M3 hooks keep positions/_last_positions in GLOBAL fine coords on
        every rank, so rank 0's copy is the exact production state. Same
        fine->L0 transform as OutputManager._write_markers."""
        if self._rank != 0 or self._marker_vtk is None or \
                runner.model is None:
            return
        model = runner.model
        orig = model._last_positions
        if self._m_origin is not None and orig is not None:
            ox, oy, oz = self._m_origin
            dxs = self._m_spacing
            t = orig.copy()
            t[:, 0] = ox + orig[:, 0] * dxs
            t[:, 1] = oy + orig[:, 1] * dxs
            t[:, 2] = oz + orig[:, 2] * dxs
            model._last_positions = t
        try:
            self._marker_vtk.write_from_al_model(
                step=step, al_model=model, time=float(step))
        finally:
            if self._m_origin is not None and orig is not None:
                model._last_positions = orig

    # ── Checkpoint: assembled std f -> CheckpointManager.save ────────
    def save_checkpoint(self, step: int, runner) -> None:
        comm, rank, nr = self._comm, self._rank, self._nr
        f_levels = []
        for k in range(runner.NL):
            fk = _gather_level(comm, rank, nr, runner,
                               runner.owned_f_std(k), k, tag0=400 + 2 * k,
                               pre_sliced=True)
            f_levels.append(fk)                      # None on non-root
            cp.get_default_memory_pool().free_all_blocks()
        rho0 = _gather_level(comm, rank, nr, runner,
                             runner.lv[0].rho[None], 0, tag0=460)
        u0 = _gather_level(comm, rank, nr, runner,
                           runner.lv[0].u, 0, tag0=462)
        if rank == 0 and self._ckpt is not None:
            extra = {"num_levels": runner.NL}
            for k in range(1, runner.NL):
                extra[f"f_level_{k}"] = f_levels[k]
            self._ckpt.save(step=step, f=f_levels[0], rho=rho0[0], u=u0,
                            tau=self._tau, config=self._sim_params,
                            extra_data=extra)
        comm.Barrier()
