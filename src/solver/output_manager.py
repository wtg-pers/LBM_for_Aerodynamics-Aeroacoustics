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
        # ── Monitors ──
        conservation_mgr: Optional['ConservationManager'] = None,
        force_mgr: Optional['ForceManager'] = None,
        conv_monitor: Optional['ConvergenceMonitor'] = None,
        # ── ALM reference ──
        al_model: Optional[Any] = None,
        # ── Intervals ──
        output_interval: int = 500,
        check_interval: int = 500,
        checkpoint_interval: int = 2000,
        # ── Misc ──
        tau: float = 0.6,
        solid_mask_np: Optional[Any] = None,
        perf_csv_path: Optional[str] = None,
        domain_shape: Optional[tuple] = None,
        # ── References for finalize summary ──
        L_ref_lu: Optional[float] = None,
        u_ref_lu: Optional[float] = None,
        config_path: Optional[str] = None,
        mlg_vtk_writer: Optional[object] = None,
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

        # ── Monitors ──
        self.conservation_mgr = conservation_mgr
        self.force_mgr = force_mgr
        self.conv_monitor = conv_monitor

        # ── ALM ──
        self.al_model = al_model
        self._is_multi_rotor: bool = False
        if al_model is not None:
            try:
                from src.actuator.actuator_line import MultiRotorManager
                self._is_multi_rotor = isinstance(al_model, MultiRotorManager)
            except ImportError:
                pass

        # ── Intervals ──
        self.output_interval = output_interval
        self.check_interval = check_interval
        self.checkpoint_interval = checkpoint_interval

        # ── Misc ──
        self.tau = tau
        self.solid_mask_np = solid_mask_np
        self.perf_csv_path = perf_csv_path
        self.domain_shape = domain_shape
        self.L_ref_lu = L_ref_lu
        self.u_ref_lu = u_ref_lu
        self.config_path = config_path
        self.mlg_vtk_writer = mlg_vtk_writer

        # ── Progress tracking state ──
        self._last_drift: float = 0.0
        self._last_Cd: float = 0.0
        self._last_Cl: float = 0.0
        self._last_ct: float = 0.0
        self._last_cp: float = 0.0
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

        # ─── 1. Force Calculation (MEM) ───────────────────────────
        self._process_force(step, sim)

        # ─── 2. Progress bar ──────────────────────────────────────
        self._update_progress(step, sim)

        # ─── 3. VTK output ───────────────────────────────────────
        self._write_vtk(step, sim)

        # ─── 4. Periodic checks ──────────────────────────────────
        if step % self.check_interval == 0 and step > self._start_step:
            self._check_conservation(step, sim)
            self._log_rotor_performance(step, sim)
            action = self._check_convergence(step, sim)

        # ─── 5. Checkpoint ────────────────────────────────────────
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

        elapsed = time.perf_counter() - self._start_time
        total_steps = self._end_step - self._start_step
        final_step = self._end_step - 1

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
        if self.domain_shape is not None:
            total_cells = 1
            for n in self.domain_shape:
                total_cells *= n
        else:
            total_cells = 0
        mlups = (total_cells * total_steps) / elapsed / 1e6 if elapsed > 0 else 0.0

        # ── Summary header ──
        print("\n" + "=" * 70)
        print(f"[7] Summary")
        print(f"  Completed: step {self._start_step} → {final_step}")
        print(f"  Time: {elapsed:.2f}s | MLUPS: {mlups:.2f}")

        # ── Final macroscopic recompute ──
        rho_final, u_final = self.macroscopic.compute(sim.f)

        # ── Final MLG VTK ──
        if self.mlg_vtk_writer is not None:
            from src.grid.multi_level_grid import MultiLevelGrid
            if isinstance(sim, MultiLevelGrid):
                self.mlg_vtk_writer.write(
                    step=final_step, mlg=sim, time=float(final_step))

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

            if (self.marker_vtk_writer is not None
                    and self.al_model is not None):
                self.marker_vtk_writer.write_from_al_model(
                    step=final_step,
                    al_model=self.al_model,
                    time=float(final_step),
                )
                self.marker_vtk_writer.write_pvd()

        # ── Final checkpoint ──
        if self.checkpoint_mgr is not None:
            self.checkpoint_mgr.save(
                step=final_step, f=sim.f, rho=rho_final,
                u=u_final, tau=self.tau, config=self.sim_params,
                extra_data=self._build_checkpoint_extra(sim),
            )

        # ── Final conservation ──
        if self.conservation_mgr is not None:
            print(f"\n[8] Final Conservation Analysis")
            self.conservation_mgr.check(
                rho_final, final_step, verbose=True,
            )
            self.conservation_mgr.close()

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
        """MEM force calculation (if enabled and at correct interval)."""
        if self.force_mgr is None:
            return
        if not self.force_mgr.should_compute(step):
            return

        force_result = self.force_mgr.compute_and_log(
            step, sim.f_post, verbose=False,
        )
        if force_result:
            self._last_Cd = force_result['Cd']
            self._last_Cl = force_result['Cl']
            if self.conv_monitor is not None:
                self.conv_monitor.feed_force(
                    step, self._last_Cd, self._last_Cl,
                )

    def _update_progress(self, step: int, sim: 'Simulation') -> None:
        """Update tqdm progress bar with relevant metrics."""
        if self._pbar is None:
            return

        self._pbar.update(1)

        if step % 10 != 0:
            return

        if self.al_model is not None:
            if self._is_multi_rotor:
                perf = self.al_model.get_rotor_performance(rotor_idx=0)
                self._pbar.set_postfix({
                    'rev': f"{perf.get('revolutions', 0):.2f}",
                    'C_T': f"{perf.get('C_T', 0):.3f}",
                    'rotors': f"{self.al_model.n_rotors}",
                    'drift': f"{self._last_drift:+.3f}%",
                })
            else:
                self._pbar.set_postfix({
                    'rev': f"{self.al_model.rotor.n_revolutions:.2f}",
                    'C_T': f"{self._last_ct:.3f}",
                    'C_P': f"{self._last_cp:.3f}",
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

    def _write_vtk(self, step: int, sim: 'Simulation') -> None:
        """Write VTK output (domain + markers) at output_interval."""
        if self.vtk_writer is None:
            return
        if step % self.output_interval != 0:
            return

        # ── MLG: write per-level .vti + .vth ───────────────
        if self.mlg_vtk_writer is not None:
            from src.grid.multi_level_grid import MultiLevelGrid
            if isinstance(sim, MultiLevelGrid):
                self.mlg_vtk_writer.write(step=step, mlg=sim, time=float(step))
                # Skip redundant single-grid .vti (level0.vti already covers it)
                return

        extra_vectors_vtk = {}
        if sim.body_force is not None:
            extra_vectors_vtk['body_force'] = sim.body_force

        self.vtk_writer.write(
            step=step, rho=sim.rho, u=sim.u,
            solid_mask=self.solid_mask_np,
            extra_vectors=extra_vectors_vtk if extra_vectors_vtk else None,
            time=float(step),
        )

        if (self.marker_vtk_writer is not None
                and self.al_model is not None):
            self.marker_vtk_writer.write_from_al_model(
                step=step, al_model=self.al_model, time=float(step),
            )

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
        """Log rotor performance to CSV (single or multi-rotor)."""
        if self.al_model is None or self.perf_csv_path is None:
            return

        if self._is_multi_rotor:
            for ri, rname in enumerate(self.al_model.names):
                perf = self.al_model.get_rotor_performance(rotor_idx=ri)
                self._last_ct = perf.get('C_T', 0)
                self._last_cp = perf.get('C_P', 0)
                self._last_rev = perf.get('revolutions', 0)
                with open(self.perf_csv_path, 'a') as fh:
                    fh.write(
                        f"{step},{rname},{self._last_rev:.6f},"
                        f"{self._last_ct:.6f},{self._last_cp:.6f}\n"
                    )
        else:
            perf = self.al_model.get_rotor_performance()
            self._last_ct = perf.get('C_T', 0)
            self._last_cp = perf.get('C_P', 0)
            self._last_rev = perf.get('revolutions', 0)
            with open(self.perf_csv_path, 'a') as fh:
                fh.write(
                    f"{step},{self._last_rev:.6f},"
                    f"{self._last_ct:.6f},{self._last_cp:.6f}\n"
                )

    def _check_convergence(
        self, step: int, sim: 'Simulation',
    ) -> str:
        """Run convergence/divergence check. Returns 'continue' or 'stop'."""
        if self.conv_monitor is None or not self.conv_monitor.enabled:
            return 'continue'

        self.conv_monitor.feed_energy(step, sim.rho, sim.u)
        self.conv_monitor.feed_divergence_check(sim.rho, sim.u)
        conv_status = self.conv_monitor.check(step)

        if conv_status['diverged']:
            print(f"\n  ⚠ DIVERGENCE at step {step}: "
                  f"{conv_status['diverge_reason']}")
            if self.conv_monitor.on_diverged == 'stop_with_checkpoint':
                if self.checkpoint_mgr is not None:
                    self.checkpoint_mgr.save(
                        step=step, f=sim.f,
                        rho=sim.rho, u=sim.u,
                        tau=self.tau, config=self.sim_params,
                        extra_data=self._build_checkpoint_extra(sim),
                    )
            return 'stop'

        if conv_status['converged']:
            print(f"\n  ✓ CONVERGED at step {step}")
            self.conv_monitor.print_summary()
            if self.conv_monitor.on_converged == 'checkpoint_and_stop':
                if self.checkpoint_mgr is not None:
                    self.checkpoint_mgr.save(
                        step=step, f=sim.f,
                        rho=sim.rho, u=sim.u,
                        tau=self.tau, config=self.sim_params,
                    )
            if self.conv_monitor.on_converged != 'continue':
                return 'stop'

        return 'continue'
    
    def _build_checkpoint_extra(self, sim) -> dict:
        """Build extra_data dict for MLG checkpoint (fine level f arrays)."""
        extra = {}
        if self.mlg_vtk_writer is not None:
            from src.grid.multi_level_grid import MultiLevelGrid
            if isinstance(sim, MultiLevelGrid):
                extra['num_levels'] = sim.num_levels
                for k in range(1, sim.num_levels):
                    extra[f'f_level_{k}'] = sim.get_level(k).f
        return extra if extra else None

    def _save_checkpoint(self, step: int, sim: 'Simulation') -> None:
        if self.checkpoint_mgr is None:
            return
        if step <= 0 or step % self.checkpoint_interval != 0:
            return

        self.checkpoint_mgr.save(
            step=step, f=sim.f, rho=sim.rho, u=sim.u,
            tau=self.tau, config=self.sim_params,
            extra_data=self._build_checkpoint_extra(sim),
        )

    # =====================================================================
    # Finalize helpers (private)
    # =====================================================================

    def _print_rotor_summary(self) -> None:
        """Print final rotor performance (single or multi)."""
        if self.al_model is None:
            return

        if self._is_multi_rotor:
            print(f"\n[9] Multi-Rotor Performance "
                  f"({self.al_model.n_rotors} rotors)")
            for i, name in enumerate(self.al_model.names):
                perf = self.al_model.get_rotor_performance(rotor_idx=i)
                print(f"  [{i}] {name}:")
                print(f"      Rev={perf.get('revolutions', 0):.2f}, "
                      f"C_T={perf.get('C_T', 0):.4f}, "
                      f"C_P={perf.get('C_P', 0):.4f}")
        else:
            perf = self.al_model.get_rotor_performance()
            print(f"\n[9] Rotor Performance")
            print(f"  Revolutions: {perf.get('revolutions', 0):.2f}")
            print(f"  C_T = {perf.get('C_T', 0):.4f}")
            print(f"  C_P = {perf.get('C_P', 0):.4f}")

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
            return True

        if final_status == ConvergenceStatus.DIVERGED:
            print("\n" + "=" * 70)
            print(" ❌ Simulation DIVERGED!")
            print("=" * 70)
            return False

        # No convergence monitor or still running
        if (self.xp.isnan(rho_final).any()
                or self.xp.isinf(rho_final).any()):
            print("\n  ❌ INSTABILITY DETECTED!")
            return False

        print("\n" + "=" * 70)
        print(" ✓ Simulation completed.")
        print("=" * 70)
        if self.config_path:
            print(f"\nTo continue: python main.py --config {self.config_path} "
                  f"--restart-latest --extend 10000")

        return True