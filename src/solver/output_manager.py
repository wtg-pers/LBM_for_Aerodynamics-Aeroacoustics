"""
Output Manager — Per-step I/O, Monitoring, and Finalization

This module extracts ALL output/monitoring logic from the time loop:
    - Progress bar (tqdm)
    - Force calculation (MEM)
    - VTK output (domain + marker)
    - Conservation check
    - Rotor performance CSV
    - Convergence monitoring (→ 'stop' signal)
    - Checkpoint saving
    - Post-loop summary and finalization

Design:
    process(step, sim) handles one step's worth of I/O.
    finalize(sim) handles post-loop cleanup.
    main.py only needs:
        for step in range(start_step, end_step):
            sim.advance()
            if output.process(step, sim) == 'stop': break
        output.finalize(sim)

Author: LBM Development Team
Date: 2026-03
"""

import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from tqdm import tqdm

if TYPE_CHECKING:
    from types import ModuleType
    from src.solver.simulation import Simulation
    from src.io.vtk_writer import VTKWriter
    from src.io.marker_vtk_writer import MarkerVTPWriter
    from src.io.checkpoint import CheckpointManager
    from src.io.probe_writer import PressureProbeManager
    from src.io.plane_writer import PlaneWriterManager
    from src.utilities.flux_utils import ConservationManager
    from src.utilities.force_calculator import ForceManager
    from src.utilities.convergence import ConvergenceMonitor


class OutputManager:
    """Manages all simulation output, monitoring, and diagnostics.

    Owns: progress bar, VTK/checkpoint/CSV writers, conservation/
    convergence monitors, force calculator. Provides a clean two-method
    interface: process() per step, finalize() after loop.

    Example:
        >>> output = OutputManager(...)
        >>> output.start(start_step, end_step)
        >>> for step in range(start_step, end_step):
        ...     sim.advance()
        ...     if output.process(step, sim) == 'stop':
        ...         break
        >>> output.finalize(sim)
    """

    def __init__(
        self,
        xp: 'ModuleType',
        macroscopic: Any,
        lattice: Any,
        sim_params: Dict[str, Any],
        # ── I/O writers ──
        vtk_writer: Optional['VTKWriter'] = None,
        marker_vtk_writer: Optional['MarkerVTPWriter'] = None,
        checkpoint_mgr: Optional['CheckpointManager'] = None,
        probe_mgr: Optional['PressureProbeManager'] = None,
        plane_mgr: Optional['PlaneWriterManager'] = None,
        # ── Monitors ──
        conservation_mgr: Optional['ConservationManager'] = None,
        force_mgr: Optional['ForceManager'] = None,
        conv_monitor: Optional['ConvergenceMonitor'] = None,
        # ── ALM reference ──
        al_model: Optional[Any] = None,
        # ── Intervals ──
        output_interval: int = 500,
        log_interval: int = 100,
        check_interval: int = 500,
        checkpoint_interval: int = 2000,
        # ── Misc ──
        tau: float = 0.6,
        solid_mask_np: Optional[Any] = None,
        perf_csv_path: Optional[str] = None,
        blade_csv_dir: Optional[str] = None,
        domain_shape: Optional[tuple] = None,
        # ── References for finalize summary ──
        L_ref_lu: Optional[float] = None,
        u_ref_lu: Optional[float] = None,
        config_path: Optional[str] = None,
        mlg_vtk_writer: Optional[object] = None,
        mlg_force_level: Optional[int] = None,
        mlg_force_block: Optional[int] = None,
        alm_marker_origin: Optional[tuple] = None,
        alm_marker_spacing: Optional[float] = None,
    ) -> None:
        """Initialize OutputManager with all I/O components.

        Args:
            xp: Array module (numpy or cupy)
            macroscopic: Macroscopic calculator (for finalize recompute)
            lattice: Lattice model (for dim, Q)
            sim_params: Simulation parameters dict (for checkpoint metadata)
            vtk_writer: VTK domain writer (optional)
            marker_vtk_writer: Marker VTP writer for ALM (optional)
            checkpoint_mgr: Checkpoint save/load manager (optional)
            probe_mgr: Pressure probe channel (optional)
            plane_mgr: Plane slice channel (optional)
            conservation_mgr: Mass conservation monitor (optional)
            force_mgr: MEM force calculator (optional)
            conv_monitor: Convergence/divergence monitor (optional)
            al_model: Actuator Line model reference (optional)
            output_interval: Steps between VTK outputs  [steps]
            check_interval: Steps between periodic checks  [steps]
            checkpoint_interval: Steps between checkpoints  [steps]
            tau: Relaxation time  [Δt] (for checkpoint metadata)
            solid_mask_np: Solid mask as numpy array (for VTK)
            perf_csv_path: Path to rotor performance CSV (optional)
            domain_shape: Grid dimensions (Nx, Ny[, Nz])  [lu]
            L_ref_lu: Reference length  [Δx] (for Strouhal)
            u_ref_lu: Reference velocity  [Δx/Δt] (for Strouhal)
            config_path: Path to config file (for restart hint)
        """
        self.xp = xp
        self.macroscopic = macroscopic
        self.lattice = lattice
        self.sim_params = sim_params

        # ── I/O writers ──
        self.vtk_writer = vtk_writer
        self.marker_vtk_writer = marker_vtk_writer
        self.checkpoint_mgr = checkpoint_mgr
        self.probe_mgr = probe_mgr
        self.plane_mgr = plane_mgr

        # ── Monitors ──
        self.conservation_mgr = conservation_mgr
        self.force_mgr = force_mgr
        self.conv_monitor = conv_monitor

        # ── ALM ──
        self.al_model = al_model
        self._alm_marker_origin = alm_marker_origin   # (ox, oy, oz) in L0 lu
        self._alm_marker_spacing = alm_marker_spacing  # dx_fine in L0 lu
        self._is_multi_rotor: bool = False
        if al_model is not None:
            from src.actuator.actuator_line import MultiRotorManager
            self._is_multi_rotor = isinstance(al_model, MultiRotorManager)

        # File-IO ownership. True on the single-GPU path; the MPI subclass
        # sets it False on rank != 0 (those ranks still join every collective
        # data seam but never write files).
        self._io_rank: bool = True

        # ── Intervals ──
        self.output_interval = output_interval
        self.log_interval = log_interval
        self.check_interval = check_interval
        self.checkpoint_interval = checkpoint_interval

        # ── Misc ──
        self.tau = tau
        self.solid_mask_np = solid_mask_np
        self.perf_csv_path = perf_csv_path
        self.blade_csv_dir = blade_csv_dir
        self.domain_shape = domain_shape
        self.L_ref_lu = L_ref_lu
        self.u_ref_lu = u_ref_lu
        self.config_path = config_path
        self.mlg_vtk_writer = mlg_vtk_writer
        self._mlg_force_level = mlg_force_level
        self._mlg_force_block = mlg_force_block

        # ── Progress tracking state ──
        self._last_drift: float = 0.0
        self._last_Cd: float = 0.0
        self._last_Cl: float = 0.0
        self._last_Cz: float = 0.0
        self._last_ct: float = 0.0
        self._last_cp: float = 0.0
        self._last_thrust_lu: float = 0.0
        self._last_power_lu: float = 0.0
        self._last_rev: float = 0.0

        # ── Time loop state (set by start()) ──
        self._pbar: Optional[tqdm] = None
        self._start_step: int = 0
        self._end_step: int = 0
        self._start_time: float = 0.0

    # =====================================================================
    # Lifecycle
    # =====================================================================

    def start(self, start_step: int, end_step: int) -> None:
        """Initialize time loop tracking. Call before the for-loop.

        Args:
            start_step: First step index  [steps]
            end_step: One past last step  [steps]
        """
        self._start_step = start_step
        self._end_step = end_step
        self._start_time = time.perf_counter()

        custom_format = (
            "{l_bar}{bar:10}|{n_fmt}/{total_fmt} "
            "{rate_fmt} [{elapsed}{postfix}]"
        )
        self._pbar = tqdm(
            total=end_step - start_step,
            unit="step",
            ncols=99,
            bar_format=custom_format,
        )

    def process(self, step: int, sim: 'Simulation') -> str:
        """Handle all per-step I/O. Call after sim.advance().

        Processing order:
            1. Force calculation (MEM)
            2. Progress bar update
            3. VTK output
            4. Periodic checks (conservation, rotor CSV, convergence)
            5. Checkpoint

        Args:
            step: Current timestep index  [steps]
            sim: Simulation object (provides rho, u, f, f_post, body_force)

        Returns:
            'continue' — normal operation
            'stop'     — convergence/divergence triggered termination
        """
        action = 'continue'
        # Last step this manager actually processed — finalize labels the
        # final VTK/checkpoint with THIS, not _end_step-1: on an early
        # convergence/divergence stop the old label pointed 'end-1' at
        # state frozen at the stop step (a restart-poisoning mismatch;
        # with keep_last_n it even pruned the emergency checkpoint).
        self._last_processed_step = step

        # ─── 0. Pressure probes (every-step acoustic channel) ────
        # Before every other channel: a device-buffer gather with no
        # host sync, so it must never sit behind an interval gate.
        if self.probe_mgr is not None:
            self.probe_mgr.sample(step, sim)

        # ─── 0b. Plane slices (dense-in-time acoustic cuts) ──────
        # Same ring-buffer pattern; the flush inside does real disk work
        # at flush_every cadence, not per step.
        if self.plane_mgr is not None:
            self.plane_mgr.sample(step, sim)

        # ─── 1. Progress bar ──────────────────────────────────────
        self._update_progress(step, sim)

        # ─── 2. Logging (rotor perf + blade diag) ────────────────
        if step % self.log_interval == 0 and step > self._start_step:
            self._log_rotor_performance(step, sim)
            self._log_blade_diagnostics(step, sim)

        # ─── 2b. MEM force (independent gate at force_mgr.interval) ──
        # Decoupled from log_interval so users can sample forces densely
        # (e.g. every 10 steps) for Cp/Strouhal post-processing while
        # keeping rotor/blade logging at a coarser cadence.
        if (self.force_mgr is not None
                and step > self._start_step
                and step >= self.force_mgr.start_step
                and step % self.force_mgr.interval == 0):
            self._process_force(step, sim)

        # ─── 3. VTK output ───────────────────────────────────────
        self._write_vtk(step, sim)

        # ─── 4. Periodic checks ──────────────────────────────────
        if step % self.check_interval == 0 and step > self._start_step:
            self._check_conservation(step, sim)
            action = self._check_convergence(step, sim)

        # ─── 6. Checkpoint ────────────────────────────────────────
        self._save_checkpoint(step, sim)

        return action

    def finalize(self, sim: 'Simulation') -> Optional[bool]:
        """Post-loop cleanup, final output, and summary.

        Handles:
            - Close progress bar
            - Max-steps convergence check
            - Performance summary (MLUPS)
            - Final VTK + checkpoint
            - Conservation analysis
            - Force summary + Strouhal number
            - Rotor performance summary
            - Convergence summary
            - Final status

        Args:
            sim: Simulation object

        Returns:
            True if completed normally, False if diverged/unstable,
            None if no convergence info available.
        """
        # Close progress bar
        if self._pbar is not None:
            self._pbar.close()

        # Probe/plane tails: flush buffered rows before any summary path.
        if self.probe_mgr is not None:
            self.probe_mgr.finalize()
        if self.plane_mgr is not None:
            self.plane_mgr.finalize()

        elapsed = time.perf_counter() - self._start_time
        total_steps = self._end_step - self._start_step
        final_step = getattr(self, '_last_processed_step',
                             self._end_step - 1)

        # ── Max steps check ──
        if self.conv_monitor is not None and self.conv_monitor.enabled:
            if (not self.conv_monitor.converged
                    and not self.conv_monitor.diverged):
                self.conv_monitor.mark_max_steps()
                if self.conv_monitor.on_max_steps == 'warn':
                    print(f"\n  ⚠ Max steps reached without convergence")
                elif self.conv_monitor.on_max_steps == 'error':
                    print(f"\n  ✗ ERROR: Max steps without convergence")

        # ── MLUPS ──
        # For MLG: each level k advances 2^k times per coarse step.
        # Total lattice updates per coarse step = Σ N_k × 2^k
        level_counts = self._level_cell_counts(sim)
        is_mlg = level_counts is not None
        num_levels = len(level_counts) if is_mlg else 1
        if is_mlg:
            updates_per_step = 0
            for k, n_k in enumerate(level_counts):
                updates_per_step += n_k * (2 ** k)
        elif self.domain_shape is not None:
            updates_per_step = 1
            for n in self.domain_shape:
                updates_per_step *= n
        else:
            updates_per_step = 0
        mlups = (updates_per_step * total_steps) / elapsed / 1e6 if elapsed > 0 else 0.0

        # ── Save performance to CSV ──
        # Determine CSV directory from existing managers
        if self._io_rank:
            _csv_dir = None
            if self.force_mgr is not None:
                _csv_dir = getattr(self.force_mgr, 'csv_dir', None)
            if _csv_dir is None and self.conservation_mgr is not None:
                _csv_dir = getattr(self.conservation_mgr, 'csv_dir', None)
            if _csv_dir is None:
                _csv_dir = './results/csv'
            perf_path = os.path.join(_csv_dir, 'performance.csv')
            try:
                _csv_dir = os.path.dirname(perf_path)
                if _csv_dir:
                    os.makedirs(_csv_dir, exist_ok=True)
                with open(perf_path, 'w', newline='') as pf:
                    import csv as _csv
                    pw = _csv.writer(pf)
                    pw.writerow(['start_step', 'end_step', 'total_steps',
                                 'elapsed_sec', 'updates_per_step', 'mlups',
                                 'is_mlg', 'num_levels'])
                    pw.writerow([
                        self._start_step, self._end_step, total_steps,
                        f"{elapsed:.3f}", updates_per_step, f"{mlups:.2f}",
                        is_mlg, num_levels,
                    ])
            except Exception:
                pass  # non-critical

        # ── Summary header ──
        print("\n" + "=" * 70)
        print(f"[7] Summary")
        print(f"  Completed: step {self._start_step} → {final_step}")
        print(f"  Time: {elapsed:.2f}s | MLUPS: {mlups:.2f}")
        if is_mlg:
            print(f"  (updates/coarse step: {updates_per_step:,})")

        # ── Free WALE pre-pass buffers before heavy post-processing ──
        self._free_transient_buffers(sim)

        # ── Final macroscopic ──
        rho_final, u_final = self._final_fields(sim)

        # ── Final MLG VTK ──
        if self.mlg_vtk_writer is not None:
            from src.grid.multi_level_grid import MultiLevelGrid
            target = self._sim_vtk_target(final_step, sim)
            if target is not None and (is_mlg
                                       or isinstance(target, MultiLevelGrid)):
                self.mlg_vtk_writer.write(
                    step=final_step, mlg=target, time=float(final_step))
            # final surfel surface loads (patch 68): the cadence path
            # wrote them mid-run; the final snapshot used to drop them —
            # the 10k span runs ended with no surface file at the
            # verdict step. OUTSIDE the rank-0-only branch: the MPI
            # override is collective (every rank ships owned facets).
            self._write_final_surface(final_step, target)

        # ── Final VTK (single grid only) ──
        if self.vtk_writer is not None and self.mlg_vtk_writer is None:
            extra_vectors_vtk = {}
            if sim.body_force is not None:
                extra_vectors_vtk['body_force'] = sim.body_force

            self.vtk_writer.write(
                step=final_step, rho=rho_final, u=u_final,
                solid_mask=self.solid_mask_np,
                extra_vectors=extra_vectors_vtk if extra_vectors_vtk else None,
                time=float(final_step),
            )
            self.vtk_writer.write_pvd('simulation.pvd')

        # ── Final markers (both single grid and MLG) ──
        if (self.marker_vtk_writer is not None
                and self.al_model is not None):
            self._write_markers(final_step)
            self.marker_vtk_writer.write_pvd()

        # ── Final checkpoint ── (skipped when this exact step was already
        # checkpointed — e.g. the divergence emergency save; a duplicate
        # save of identical state is wasted IO and, with keep_last_n, once
        # deleted the file it had just written)
        if (self.checkpoint_mgr is not None
                and final_step != getattr(self, '_last_ckpt_step', None)):
            payload = self._checkpoint_payload(sim)
            if payload is not None:
                self.checkpoint_mgr.save(
                    step=final_step, f=payload['f'], rho=rho_final,
                    u=u_final, tau=self.tau, config=self.sim_params,
                    extra_data=payload['extra'],
                )

        # ── Final conservation ──
        self._final_conservation(rho_final, final_step)

        # ── Force summary ──
        if self.force_mgr is not None:
            self.force_mgr.print_summary()

            from src.utilities.force_calculator import compute_strouhal_number

            St = compute_strouhal_number(
                force_history=self.force_mgr.history,
                char_length=self.L_ref_lu,
                u_ref=self.u_ref_lu,
                component='Cl',
                min_periods=3,
            )

            if St is not None:
                print(f"\n  Strouhal number: St = {St:.4f}")
            else:
                print(f"\n  Strouhal number: insufficient data for FFT")

            self.force_mgr.close()

        # ── Rotor performance summary ──
        self._print_rotor_summary()

        # ── Convergence summary ──
        if self.conv_monitor is not None and self.conv_monitor.enabled:
            self.conv_monitor.print_summary()
            self.conv_monitor.close()

        # ── Final status ──
        return self._print_final_status(rho_final)

    # =====================================================================
    # Per-step helpers (private)
    # =====================================================================

    def _process_force(self, step: int, sim: 'Simulation') -> None:
        """MEM force calculation. Called at force_mgr.interval by process().

        Caller is responsible for the step-modulo check; this function only
        guards against (a) no force manager, (b) step below start_step.

        For MLG: uses the finest level's f_post where the obstacle exists,
        because L0's f_post is captured before F→C coupling and does not
        reflect the fine-grid solution.
        """
        if self.force_mgr is None:
            return
        if step < self.force_mgr.start_step:
            return

        f_post, forces_override = self._force_inputs(step, sim)
        force_result = self.force_mgr.compute_and_log(
            step, f_post, verbose=False, forces_override=forces_override,
        )
        if force_result:
            self._last_Cd = force_result['Cd']
            self._last_Cl = force_result['Cl']
            self._last_Cz = force_result.get('Cz', 0.0)
            if self.conv_monitor is not None:
                self.conv_monitor.feed_force(
                    step, self._last_Cd, self._last_Cl,
                )

    def _force_inputs(self, step: int, sim: 'Simulation'):
        """Data seam: (f_post, forces_override) for the MEM force channel.

        Single-GPU: pick the level carrying the obstacle; esoteric levels
        have no f_post buffer (single-buffer in-place) so the force comes
        from the same eso_mem_force kernel the MPI runner uses, fed through
        forces_override (coefficients/CSV stay on the one path).
        MPI override: Allreduce(mem_force_local) as forces_override.
        """
        if hasattr(sim, 'get_level'):
            # _mlg_force_level is None when the obstacle sits on L0 (e.g.
            # num_levels=1): resolve to the LEVEL, not the MLG wrapper —
            # the wrapper lacks _use_esoteric/eso_body_force, so an eso L0
            # would fall through to f_post=None and crash the MEM channel.
            _fb = getattr(self, '_mlg_force_block', None)
            lvl = (sim.block_by_uid(_fb).sim
                   if _fb is not None and hasattr(sim, 'block_by_uid')
                   else sim.get_level(self._mlg_force_level or 0))
        else:
            lvl = sim

        if getattr(lvl, '_use_esoteric', False) \
                and not getattr(lvl, '_use_surfel', False):
            # surfel + eso is the residency bridge (patch_notes/surfel/
            # 63): its force is the facet ledger fed by the std chain,
            # NOT the HWBB MEM formula — route it like the std path.
            return None, lvl.eso_body_force()
        return lvl.f_post, None

    def _update_progress(self, step: int, sim: 'Simulation') -> None:
        """Update tqdm progress bar with relevant metrics."""
        if self._pbar is None:
            return

        self._pbar.update(1)

        if step % 10 != 0:
            return

        if self.al_model is not None:
            # NOTE: thrust/power are shown as the dimensionless, grid-invariant
            # C_T and C_P — NOT the raw lattice force/power. thrust_lu/power_lu are
            # in FINEST-level lattice units and scale as (resolution)ⁿ, so they
            # jump between a 4-level and 5-level grid for identical physics —
            # misleading to watch. C_T=T/(ρ·A·(ωR)²), C_P=P/(ρ·A·(ωR)³) cancel that
            # scale exactly. (Lattice values are still logged to the CSVs.)
            if self._is_multi_rotor:
                perf = self.al_model.get_rotor_performance(rotor_idx=0)
                self._pbar.set_postfix({
                    'rev': f"{perf.get('revolutions', 0):.2f}",
                    'C_T': f"{perf.get('C_T', 0):.5f}",
                    'C_P': f"{perf.get('C_P', 0):.5f}",
                    'rotors': f"{self.al_model.n_rotors}",
                    'drift': f"{self._last_drift:+.3f}%",
                })
            else:
                self._pbar.set_postfix({
                    'rev': f"{self.al_model.rotor.n_revolutions:.2f}",
                    'C_T': f"{self._last_ct:.5f}",
                    'C_P': f"{self._last_cp:.5f}",
                    'drift': f"{self._last_drift:+.3f}%",
                })
        elif self.force_mgr is not None:
            self._pbar.set_postfix({
                'Cd': f"{self._last_Cd:.3f}",
                'Cl': f"{self._last_Cl:.3f}",
            })
        else:
            self._pbar.set_postfix({
                'ρ': f"{float(sim.rho.mean()):.4f}",
                'drift': f"{self._last_drift:+.4f}%",
            })

    def _vtk_due(self, step: int) -> bool:
        """Cadence seam for the VTK channel. The MPI subclass replaces the
        writer-presence gate with a replicated flag (rank != 0 has no
        writer but must still join the collective gathers)."""
        if self.vtk_writer is None:
            return False
        return step % self.output_interval == 0

    def _sim_vtk_target(self, step: int, sim: 'Simulation'):
        """Data seam: the object the VTK writers read fields from.

        Single-GPU: the sim itself. MPI override: collective per-level
        gathers -> assembled duck views on rank 0, None elsewhere (skip the
        write, then meet the barrier)."""
        return sim

    def _write_final_surface(self, step: int, target) -> None:
        """Single-GPU surfel surface at the final step (patch 68); the
        MPI manager overrides with the collective gather."""
        if target is None or not hasattr(target, 'iter_blocks'):
            return
        import os as _os
        surf = [b for b in reversed(list(target.iter_blocks()))
                if getattr(getattr(b.sim, 'obstacle_bc', None),
                           'kind', None) == 'surfel']
        if not surf:
            return
        top = max(b.level for b in surf)
        sel = [b for b in surf if b.level == top]
        for b in sel:
            tag = "" if len(sel) == 1 else f"_b{b.index}"
            b.sim.obstacle_bc.write_surface(_os.path.join(
                self.mlg_vtk_writer.output_dir,
                f"surface{tag}_{step:08d}.vtk"))

    def _write_vtk(self, step: int, sim: 'Simulation') -> None:
        """Write VTK output (domain + markers) at output_interval."""
        if not self._vtk_due(step):
            return
        target = self._sim_vtk_target(step, sim)
        if target is None:
            return

        # Field suppression window (output.vtk.fields_start_step): markers
        # only until the start step — see setup.build_output_manager.
        if step < getattr(self, 'vtk_fields_start_step', 0):
            if (self.marker_vtk_writer is not None
                    and self.al_model is not None):
                self._write_markers(step)
            return

        # ── MLG: write per-level .vti + .vth ───────────────
        if self.mlg_vtk_writer is not None:
            from src.grid.multi_level_grid import MultiLevelGrid
            if isinstance(target, MultiLevelGrid):
                self.mlg_vtk_writer.write(step=step, mlg=target,
                                          time=float(step))
                # Surfel surface loads (S8a-2): from the FINEST level
                # carrying the surfel BC (facet accounting = wing level).
                # The adapter transforms verts to the global L0-lu frame
                # (coord_origin/spacing bound in setup), so the file
                # overlays the .vth volume output.
                import os as _os
                # Finest block carrying a surfel BC. Several blocks on a level
                # may each have one, so the file name gets a block tag when
                # that happens (a chain keeps the original name).
                _grids = (list(target.iter_blocks())
                          if hasattr(target, 'iter_blocks')
                          else [(None, target.get_level(_k))
                                for _k in range(target.num_levels)])
                _surf = [b for b in reversed(_grids)
                         if getattr(getattr(b.sim if hasattr(b, 'sim') else b[1],
                                            'obstacle_bc', None),
                                    'kind', None) == 'surfel'] \
                    if hasattr(target, 'iter_blocks') else []
                if _surf:
                    _top = max(b.level for b in _surf)
                    _sel = [b for b in _surf if b.level == _top]
                    for _b in _sel:
                        _tag = "" if len(_sel) == 1 else f"_b{_b.index}"
                        _b.sim.obstacle_bc.write_surface(_os.path.join(
                            self.mlg_vtk_writer.output_dir,
                            f"surface{_tag}_{step:08d}.vtk"))
                # Write ALM markers (before returning)
                if (self.marker_vtk_writer is not None
                        and self.al_model is not None):
                    self._write_markers(step)
                return

        extra_vectors_vtk = {}
        if target.body_force is not None:
            extra_vectors_vtk['body_force'] = target.body_force

        extra_scalars_vtk = {}
        # Eddy viscosity (allocated only when SGS is enabled).
        if getattr(target, 'nu_t', None) is not None:
            extra_scalars_vtk['nu_t'] = target.nu_t

        self.vtk_writer.write(
            step=step, rho=target.rho, u=target.u,
            solid_mask=self.solid_mask_np,
            extra_scalars=extra_scalars_vtk if extra_scalars_vtk else None,
            extra_vectors=extra_vectors_vtk if extra_vectors_vtk else None,
            time=float(step),
        )

        # ── Surfel wall boundary: surface loads on the STL triangles ──
        # (S8a single-GPU path only — the MLG branch returned above.)
        _ob = getattr(target, 'obstacle_bc', None)
        if getattr(_ob, 'kind', None) == 'surfel':
            import os as _os
            _p = _os.path.join(self.vtk_writer.output_dir,
                               f"surface_{step:08d}.vtk")
            _ob.write_surface(_p)

        if (self.marker_vtk_writer is not None
                and self.al_model is not None):
            self.marker_vtk_writer.write_from_al_model(
                step=step, al_model=self.al_model, time=float(step),
            )

    def _write_markers(self, step: int) -> None:
        """Write ALM marker VTP, transforming coords for MLG fine level."""
        # The fine-local → global L0 transform lives in the writer now. Doing
        # it here meant assigning to al_model._last_positions, which is a
        # read-only property on MultiRotorManager (it vstacks its models'
        # arrays) — that path raised AttributeError the moment a multi-rotor
        # ALM sat on a fine level.
        self.marker_vtk_writer.write_from_al_model(
            step=step, al_model=self.al_model, time=float(step),
            origin=self._alm_marker_origin,
            spacing=(self._alm_marker_spacing
                     if self._alm_marker_origin is not None else 1.0),
        )

        # Correction wake filaments (Kleine free / Dağ prescribed helix) → ParaView.
        # write_wake_vtp itself returns None when there is no stored wake
        # (straight kernel, <2 rings), so a real failure here must raise —
        # the old blanket except silently dropped every wake file.
        from src.io.wake_vtk_writer import write_wake_vtp
        wdir = os.path.join(
            getattr(self.marker_vtk_writer, 'output_dir', '.'), 'wake')
        write_wake_vtp(
            self.al_model, step, wdir,
            origin=self._alm_marker_origin,
            spacing=(self._alm_marker_spacing
                     if self._alm_marker_origin is not None else 1.0))

    def _check_conservation(self, step: int, sim: 'Simulation') -> None:
        """Run mass conservation check."""
        if self.conservation_mgr is None:
            return

        results = self.conservation_mgr.check(
            sim.rho, step,
            verbose=(self.conservation_mgr.verbose > 0),
        )
        if results.get('domain'):
            self._last_drift = results['domain']['mass_drift_percent']

    def _log_rotor_performance(self, step: int, sim: 'Simulation') -> None:
        """Log rotor performance to CSV (single or multi-rotor).

        Writes dimensional forces + normalization parameters so any
        coefficient convention can be applied in post-processing:
            C_T(prop)      = thrust_lu / (rho_ref * n_lu^2 * D_lu^4)
            C_T(rotorcraft)= thrust_lu / (rho_ref * area_lu * tip_speed_lu^2)
        """
        if self.al_model is None or self.perf_csv_path is None:
            return

        models = (
            [self.al_model.models[i] for i in range(self.al_model.n_rotors)]
            if self._is_multi_rotor
            else [self.al_model]
        )
        for model in models:
            perf = log_rotor_performance_row(model, self.perf_csv_path, step)
            self._last_ct  = perf.get('C_T', 0)
            self._last_cp  = perf.get('C_P', 0)
            self._last_thrust_lu = perf.get('thrust', 0)
            self._last_power_lu  = perf.get('power', 0)
            self._last_rev = perf.get('revolutions', 0)

    def _log_blade_diagnostics(self, step: int, sim: 'Simulation') -> None:
        """Log per-marker BEM diagnostics to individual CSV files.

        File layout:
            csv/blade_diagnostics/0.csv   — marker 0 (all blades, all steps)
            csv/blade_diagnostics/1.csv   — marker 1
            ...

        Columns per file:
            step, revolutions, blade, r_R, r_lu, chord_lu, twist,
            u_n, u_theta, u_rel, phi, alpha, Re, CL, CD,
            F_n, F_theta, F_L, F_D
        """
        import os
        if self.al_model is None or self.blade_csv_dir is None:
            return

        models = (
            [self.al_model.models[i] for i in range(self.al_model.n_rotors)]
            if self._is_multi_rotor
            else [self.al_model]
        )

        for model in models:
            log_blade_diagnostics_rows(model, self.blade_csv_dir, step)

    def _check_convergence(
        self, step: int, sim: 'Simulation',
    ) -> str:
        """Run convergence/divergence check. Returns 'continue' or 'stop'."""
        if self.conv_monitor is None or not self.conv_monitor.enabled:
            return 'continue'

        self._feed_convergence(step, sim)
        conv_status = self.conv_monitor.check(step)

        if conv_status['diverged']:
            print(f"\n  ⚠ DIVERGENCE at step {step}: "
                  f"{conv_status['diverge_reason']}")
            if self.conv_monitor.on_diverged == 'stop_with_checkpoint':
                self._emergency_checkpoint(step, sim, include_extra=True)
            return 'stop'

        if conv_status['converged']:
            print(f"\n  ✓ CONVERGED at step {step}")
            self.conv_monitor.print_summary()
            if self.conv_monitor.on_converged == 'checkpoint_and_stop':
                # NOTE(behavior-frozen): historically this save omitted
                # extra_data, so MLG fine-level f is MISSING from
                # converge-checkpoints (unlike the diverge save). Preserved
                # as-is here; fix scheduled with the C6 channel work.
                self._emergency_checkpoint(step, sim, include_extra=False)
            if self.conv_monitor.on_converged != 'continue':
                return 'stop'

        return 'continue'

    def _feed_convergence(self, step: int, sim: 'Simulation') -> None:
        """Data seam: feed the convergence monitor from full fields.

        MPI override: owned-fluid partial reductions -> Allreduce ->
        feed_energy_value / feed_divergence_scalars on EVERY rank (the
        monitor state must advance identically for a rank-invariant
        'stop' verdict)."""
        self.conv_monitor.feed_energy(step, sim.rho, sim.u)
        self.conv_monitor.feed_divergence_check(sim.rho, sim.u)

    def _emergency_checkpoint(self, step: int, sim: 'Simulation',
                              include_extra: bool = True) -> None:
        """Diverge/converge stop checkpoint (shared shape of the two
        historical inline save blocks)."""
        if self.checkpoint_mgr is None:
            return
        payload = self._checkpoint_payload(sim, include_extra=include_extra)
        if payload is not None:
            kwargs = dict(step=step, f=payload['f'], rho=sim.rho, u=sim.u,
                          tau=self.tau, config=self.sim_params)
            if include_extra:
                kwargs['extra_data'] = payload['extra']
            self.checkpoint_mgr.save(**kwargs)
        self._last_ckpt_step = step
    
    @staticmethod
    def _f_checkpoint_cpu(sim_like):
        """Level f for checkpointing, moved to host IMMEDIATELY.

        Esoteric levels gather a full physical copy on the GPU; converting
        to host per level (freeing the GPU copy before the next level) keeps
        the checkpoint's GPU footprint at ~one level's f. Passing the GPU
        arrays through to save() retained all levels simultaneously (~10 GB
        at D40) and OOM'd the first checkpoint (2026-07-10 cluster run).
        Non-GPU arrays pass through unchanged.
        """
        import numpy as _np
        sim = (sim_like.get_level(0) if hasattr(sim_like, 'get_level')
               else sim_like)
        if getattr(sim, '_use_esoteric', False):
            # SLAB-STREAMED gather -> host: a full-field GPU gather is one
            # f-sized block (1.9-2.9 GB/level at D40) that the fragmented
            # pool cannot reuse across levels -- the first D40 checkpoint
            # OOM'd exactly here (2026-07-10 cluster). Gathering x-slabs via
            # the bit-exact region gather caps the GPU transient at ~0.5 GB.
            from src.kernels.esoteric_d3q27 import esoteric_gather_std_region
            xp = sim.xp
            Nx, Ny, Nz = sim.domain_shape
            out = _np.empty((sim.f.shape[0], Nx, Ny, Nz), dtype=sim.f.dtype)
            step = max(1, 3_000_000 // max(Ny * Nz, 1))
            full = (slice(None), slice(None))
            for x0 in range(0, Nx, step):
                sl = slice(x0, min(x0 + step, Nx))
                g = esoteric_gather_std_region(
                    xp, sim.f, sim._esoteric_step, (sl,) + full)
                out[:, sl] = g.get() if hasattr(g, 'get') else g
                del g
            if xp.__name__ == 'cupy':
                import cupy
                cupy.get_default_memory_pool().free_all_blocks()
            return out
        f = sim.physical_f
        return f.get() if hasattr(f, 'get') else f

    def _build_checkpoint_extra(self, sim) -> dict:
        """Build extra_data dict for MLG checkpoint (fine level f arrays)
        and the eso domain-wall mailbox (single-grid, eso_wall §4-3)."""
        extra = {}
        # eso implicit domain wall: the face mailbox is state OUTSIDE the
        # 27N f slots (the std f gather is a LOAD-view bijection of those
        # slots only) -> extra npz keys, validated on restore by the
        # initializer. L{k} naming per PLAN §2-4; walls are single-grid-
        # only until §4-5·6, so only L0 can carry a mask today.
        root = sim.get_level(0) if hasattr(sim, 'get_level') else sim
        if getattr(root, '_eso_wall_mask', 0):
            extra['wall_mask_L0'] = int(root._eso_wall_mask)
            extra['wall_mail_L0'] = root._eso_wall_mail
        if self.mlg_vtk_writer is not None:
            from src.grid.multi_level_grid import MultiLevelGrid
            if isinstance(sim, MultiLevelGrid):
                extra['num_levels'] = sim.num_levels
                _blks = list(sim.iter_blocks()) if hasattr(sim, 'iter_blocks') \
                    else None
                if _blks is not None:
                    # Key rule: no block suffix when the level holds one block,
                    # so a chain writes the SAME key set as before and old
                    # readers still load new files.
                    _per = {}
                    for _b in _blks:
                        _per[_b.level] = _per.get(_b.level, 0) + 1
                    extra['num_blocks'] = len(_blks)
                    extra['block_levels'] = [b.level for b in _blks]
                    for _b in _blks:
                        if _b.level == 0:
                            continue
                        _sfx = "" if _per[_b.level] <= 1 else f"_b{_b.index}"
                        extra[f'f_level_{_b.level}{_sfx}'] = \
                            self._f_checkpoint_cpu(_b.sim)
                        # open-face track: a flush fine wall's mailbox is
                        # per-level state — same key rule as f_level_*.
                        if getattr(_b.sim, '_eso_wall_mask', 0):
                            extra[f'wall_mask_L{_b.level}{_sfx}'] = \
                                int(_b.sim._eso_wall_mask)
                            extra[f'wall_mail_L{_b.level}{_sfx}'] = \
                                _b.sim._eso_wall_mail
                for k in ([] if _blks is not None else range(1, sim.num_levels)):
                    # Host-immediate per level: keeps the GPU transient bounded to ONE
                    # level's gather instead of retaining every level's physical copy
                    # until save() (OOM'd the D40 checkpoint on a 24GB card).
                    extra[f'f_level_{k}'] = self._f_checkpoint_cpu(sim.get_level(k))
        return extra if extra else None

    def _ckpt_due(self, step: int) -> bool:
        """Cadence seam for the checkpoint channel (MPI: replicated flag —
        rank != 0 has no checkpoint writer but joins the gathers)."""
        if self.checkpoint_mgr is None:
            return False
        return step > 0 and step % self.checkpoint_interval == 0

    def _checkpoint_payload(self, sim: 'Simulation',
                            include_extra: bool = True):
        """Data seam: host f (+ MLG fine-level extras) for a checkpoint.

        Single-GPU: slab-streamed host gather per level. MPI override:
        collective owned_f_std gathers -> dict on rank 0, None elsewhere."""
        return {
            'f': self._f_checkpoint_cpu(sim),
            'extra': (self._build_checkpoint_extra(sim)
                      if include_extra else None),
        }

    def _save_checkpoint(self, step: int, sim: 'Simulation') -> None:
        if not self._ckpt_due(step):
            return
        try:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

        payload = self._checkpoint_payload(sim)
        if payload is not None:
            self.checkpoint_mgr.save(
                step=step, f=payload['f'], rho=sim.rho, u=sim.u,
                tau=self.tau, config=self.sim_params,
                extra_data=payload['extra'],
            )
        # rank-invariant bookkeeping (cadence + mgr presence are
        # replicated): finalize skips its final save when this step was
        # already checkpointed
        self._last_ckpt_step = step

    # =====================================================================
    # Finalize helpers (private)
    # =====================================================================

    def _level_cell_counts(self, sim) -> Optional[list]:
        """Data seam: per-level GLOBAL cell counts for MLUPS, or None for a
        single grid. MPI override: products of runner.parts[k] global
        shapes (no comm needed)."""
        from src.grid.multi_level_grid import MultiLevelGrid
        if isinstance(sim, MultiLevelGrid):
            # One entry per LEVEL, summed over that level's blocks: the
            # caller uses the list index as the 2^k sub-step exponent, so it
            # must stay level-indexed even when a level holds several grids.
            counts = [0] * sim.num_levels
            _grids = ([(b.level, b.sim) for b in sim.iter_blocks()]
                      if hasattr(sim, 'iter_blocks')
                      else [(k, sim.get_level(k))
                            for k in range(sim.num_levels)])
            for k, sim_k in _grids:
                n_k = 1
                for d in sim_k.domain_shape:
                    n_k *= d
                counts[k] += n_k
            return counts
        return None

    def _free_transient_buffers(self, sim) -> None:
        """Free WALE pre-pass buffers before heavy post-processing.

        On 3-level MLG these hold ~1 GB; not freeing them can OOM the
        following macroscopic.compute() (cupy tensordot upcast
        intermediate). MPI override: no-op (the runner already freed the
        source levels)."""
        def _free_wale_buffers(s):
            for attr in ('_u_buf', '_rho_buf', '_nu_t_in'):
                if hasattr(s, attr):
                    setattr(s, attr, None)
        from src.grid.multi_level_grid import MultiLevelGrid
        if isinstance(sim, MultiLevelGrid):
            for _level in sim._levels:
                _free_wale_buffers(_level)
        else:
            _free_wale_buffers(sim)
        try:
            import cupy as _cp
        except ImportError:      # CPU run — nothing to free
            pass
        else:
            _cp.get_default_memory_pool().free_all_blocks()

    def _final_fields(self, sim):
        """Data seam: (rho_final, u_final) for the finalize channels.

        Fused kernels already updated sim.rho / sim.u during the last step;
        reuse those instead of running tensordot again (saves ~5 GB peak).
        MPI override: collective L0 gather -> rank 0 arrays, (None, None)
        elsewhere."""
        if (getattr(sim, 'rho', None) is not None
                and getattr(sim, 'u', None) is not None):
            return sim.rho, sim.u
        return self.macroscopic.compute(sim.f)

    def _field_nan_flag(self, rho_final) -> bool:
        """Data seam: NaN/Inf presence in the final density field. MPI
        override: LOR-allreduced flag so finalize() returns the same
        verdict on every rank."""
        return bool(self.xp.isnan(rho_final).any()
                    or self.xp.isinf(rho_final).any())

    def _final_conservation(self, rho_final, final_step: int) -> None:
        """Data seam: the finalize conservation check. MPI override:
        collective fluid-only CV masses -> check_from_masses."""
        if self.conservation_mgr is None:
            return
        print(f"\n[8] Final Conservation Analysis")
        self.conservation_mgr.check(
            rho_final, final_step, verbose=True,
        )
        self.conservation_mgr.close()

    def _print_rotor_summary(self) -> None:
        """Print final rotor performance (single or multi)."""
        import math
        if self.al_model is None:
            return

        def _derived_coeffs(model, perf: dict) -> str:
            """Compute both propeller and rotorcraft C_T/C_P from dimensional."""
            rotor        = model.rotor
            rho          = model.rho_ref
            R            = float(rotor.radius)
            omega        = float(abs(rotor.omega))
            A            = math.pi * R ** 2
            tip          = omega * R
            n            = omega / (2.0 * math.pi)
            D            = 2.0 * R
            T            = perf.get('thrust', 0)
            Q            = perf.get('torque', 0)
            P            = perf.get('power', 0)
            denom_rc     = rho * A * tip ** 2
            denom_pr_T   = rho * n ** 2 * D ** 4
            denom_pr_P   = rho * n ** 3 * D ** 5
            CT_rc = T / denom_rc  if denom_rc   > 0 else float('nan')
            CT_pr = T / denom_pr_T if denom_pr_T > 0 else float('nan')
            CP_rc = P / (denom_rc * tip) if denom_rc * tip > 0 else float('nan')
            CP_pr = P / denom_pr_P if denom_pr_P > 0 else float('nan')
            return (
                f"  T_lu={T:.6f}  Q_lu={Q:.6f}  P_lu={P:.6f}\n"
                f"  C_T(rotorcraft)={CT_rc:.4f}  "
                f"C_T(propeller)={CT_pr:.4f}\n"
                f"  C_P(rotorcraft)={CP_rc:.4f}  "
                f"C_P(propeller)={CP_pr:.4f}"
            )

        if self._is_multi_rotor:
            print(f"\n[9] Multi-Rotor Performance "
                  f"({self.al_model.n_rotors} rotors)")
            for i, name in enumerate(self.al_model.names):
                model = self.al_model.models[i]
                perf  = model.get_rotor_performance()
                print(f"  [{i}] {name}:  Rev={perf.get('revolutions', 0):.2f}")
                print(_derived_coeffs(model, perf))
        else:
            perf = self.al_model.get_rotor_performance()
            print(f"\n[9] Rotor Performance")
            print(f"  Revolutions: {perf.get('revolutions', 0):.2f}")
            print(_derived_coeffs(self.al_model, perf))

    def _print_postrun_hints(self) -> None:
        """Restart + post-processing hints. Called on success paths
        (CONVERGED, normal completion) — skipped on DIVERGED / instability
        because those produce data the user shouldn't trust by default."""
        if not self._io_rank:
            return
        if not self.config_path:
            return
        print(f"\nTo continue: python main.py --config {self.config_path} "
              f"--restart-latest --extend 10000")

        if self.force_mgr is None:
            return
        csv_dir = getattr(self.force_mgr, 'csv_dir', None)
        if csv_dir is None:
            return
        import os as _os
        npz_path = _os.path.join(csv_dir, 'surface_link_forces.npz')
        if not _os.path.exists(npz_path):
            return
        run_dir = _os.path.dirname(csv_dir.rstrip('/'))
        print(
            f"\nTo extract surface Cp (edit --window-start to a step "
            f"past transient, see force_history.csv):\n"
            f"  python -m src.utilities.surface_distribution "
            f"{run_dir} {self.config_path} "
            f"--window-start <STEP>"
        )

    def _print_final_status(self, rho_final: Any) -> Optional[bool]:
        """Print final status and return success/failure.

        Returns:
            True if completed normally
            False if diverged or unstable
        """
        from src.utilities.convergence import ConvergenceStatus

        if self.conv_monitor is not None:
            final_status = self.conv_monitor.get_status()
        else:
            final_status = None

        if final_status == ConvergenceStatus.CONVERGED:
            print("\n" + "=" * 70)
            print(f" ✅ Simulation CONVERGED at step "
                  f"{self.conv_monitor.converged_step}!")
            print("=" * 70)
            self._print_postrun_hints()
            return True

        if final_status == ConvergenceStatus.DIVERGED:
            print("\n" + "=" * 70)
            print(" ❌ Simulation DIVERGED!")
            print("=" * 70)
            return False

        # No convergence monitor or still running
        if self._field_nan_flag(rho_final):
            print("\n  ❌ INSTABILITY DETECTED!")
            return False

        print("\n" + "=" * 70)
        print(" ✓ Simulation completed.")
        print("=" * 70)
        self._print_postrun_hints()
        return True

# =====================================================================
# Module-level CSV row appenders (single source for the single-GPU
# OutputManager loop AND main_mpi's rank-0 lockstep loop — the MPI path
# used to leave rotor_performance.csv / blade_diagnostics header-only).
# =====================================================================
def log_rotor_performance_row(model, path: str, step: int) -> dict:
    """Append one rotor_performance.csv row (schema = setup._perf_csv_header).

    Returns the model's get_rotor_performance() dict so callers can reuse it.
    """
    import math
    perf = model.get_rotor_performance()
    rotor = model.rotor
    rho_ref = model.rho_ref
    R_lu = float(rotor.radius)
    omega_lu = float(abs(rotor.omega))
    time_lt = float(perf.get('time', 0))
    with open(path, 'a') as fh:
        fh.write(
            f"{step},{time_lt:.6f},{time_lt * model.dt_phys:.9f},"
            f"{perf.get('revolutions', 0):.6f},"
            f"{perf.get('thrust', 0):.9f},"
            f"{perf.get('torque', 0):.9f},"
            f"{perf.get('power', 0):.9f},"
            f"{rho_ref:.6f},{math.pi * R_lu ** 2:.6f},"
            f"{omega_lu * R_lu:.9f},{omega_lu:.9f},"
            f"{R_lu:.6f},{2.0 * R_lu:.6f},"
            f"{omega_lu / (2.0 * math.pi):.9f}\n"
        )
    return perf


def log_blade_diagnostics_rows(model, csv_dir: str, step: int) -> None:
    """Append one row per (blade, marker) to csv_dir/<marker>.csv
    (schema = setup._blade_csv_header, incl. per-axis kernel widths)."""
    import os
    rev = model.rotor.n_revolutions
    for bi in range(model.rotor.n_blades):
        diag = model.get_blade_diagnostics(blade_idx=bi)
        if 'error' in diag:
            continue
        for j in range(len(diag['r'])):
            path = os.path.join(csv_dir, f'{j}.csv')
            with open(path, 'a') as fh:
                fh.write(
                    f"{step},{rev:.6f},{bi},"
                    f"{diag['r_R'][j]:.4f},"
                    f"{diag['r'][j]:.4f},"
                    f"{diag['chord'][j]:.4f},"
                    f"{diag['epsilon'][j]:.4f},"
                    f"{diag['twist'][j]:.3f},"
                    f"{diag['u_n'][j]:.6f},"
                    f"{diag['u_theta'][j]:.6f},"
                    f"{diag['u_rel'][j]:.6f},"
                    f"{diag['phi'][j]:.3f},"
                    f"{diag['alpha'][j]:.3f},"
                    f"{diag['Re'][j]:.1f},"
                    f"{diag['CL'][j]:.5f},"
                    f"{diag['CD'][j]:.5f},"
                    f"{diag['F_n'][j]:.6f},"
                    f"{diag['F_theta'][j]:.6f},"
                    f"{diag['F_L'][j]:.6f},"
                    f"{diag['F_D'][j]:.6f},"
                    f"{diag['eps_c'][j]:.4f},"
                    f"{diag['eps_t'][j]:.4f},"
                    f"{diag['eps_r'][j]:.4f},"
                    f"{diag['eps_samp_c'][j]:.4f},"
                    f"{diag['eps_samp_t'][j]:.4f},"
                    f"{diag['eps_samp_r'][j]:.4f}\n"
                )
