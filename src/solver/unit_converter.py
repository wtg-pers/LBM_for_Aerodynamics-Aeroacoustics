"""
Unit Converter for LBM Simulations

Converts between physical (SI) and lattice units.
All lattice parameters are derived from user inputs -- never manually specified.

User Inputs:
    physics:  rho [kg/m3], U_inf [m/s], Re [-], L_char [m]
    grid:     Nx, Ny, Nz [cells], resolution [cells/L_char]
    numerics: u_max [-] (max velocity in lattice units)
    ALM:      rpm, radius [m], chord [m], hub_center [lu], ...

Auto-Derived:
    dx_phys  = L_char / resolution                    [m]
    U_max    = max(U_inf, omega*R, ...)               [m/s]
    dt_phys  = u_max * dx_phys / U_max                [s]
    nu_phys  = U_ref * L_ref / Re                     [m2/s]
    nu_lu    = nu_phys * dt_phys / dx_phys^2           [-]
    tau      = 0.5 + 3 * nu_lu                        [-]

References:
    Kruger et al., "The Lattice Boltzmann Method", Ch. 3
    OpenLB UnitConverter, Palabos IncomprFlowParam

Author: LBM Development Team
Date: 2026-04
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np


class UnitConverter:
    """Lattice-Boltzmann unit converter.

    Computes all conversion factors from physical inputs + numerical choices.
    Performs stability checks automatically.

    Usage:
        >>> uc = UnitConverter(
        ...     physics={'rho': 1.225, 'U_inf': 10.0, 'Re': 75000, 'L_char': 0.2286},
        ...     grid={'Nx': 200, 'Ny': 100, 'Nz': 100, 'resolution': 25},
        ...     numerics={'u_max': 0.1},
        ... )
        >>> print(uc.tau, uc.nu_lu, uc.dt_phys)
    """

    def __init__(
        self,
        physics: Dict[str, Any],
        grid: Dict[str, Any],
        numerics: Dict[str, Any],
        actuator_line: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._physics = physics
        self._grid = grid
        self._numerics = numerics
        self._al_cfg = actuator_line

        # -- Grid --
        self.Nx: int = grid['Nx']
        self.Ny: int = grid['Ny']
        self.Nz: Optional[int] = grid.get('Nz')
        self.resolution: int = grid['resolution']

        if self.Nz is not None:
            self.domain_shape: Tuple[int, ...] = (self.Nx, self.Ny, self.Nz)
            self.dim = 3
        else:
            self.domain_shape = (self.Nx, self.Ny)
            self.dim = 2

        # -- Physical inputs --
        self.rho_phys: float = physics.get('rho', 1.0)
        self.U_inf: float = physics.get('U_inf', 0.0)
        self.L_char: float = physics['L_char']

        # -- Numerics --
        # acoustic_scaling=True derives u_max so LBM's effective speed of
        # sound matches c_s_phys (default 340 m/s, air STP). This is required
        # for aeroacoustic accuracy. If False (default), u_max is used as-is.
        self._acoustic_scaling: bool = bool(numerics.get('acoustic_scaling', False))
        self.c_s_phys: float = float(numerics.get('c_s_phys', 340.0))
        self.u_max: float = numerics.get('u_max', 0.1)

        # -- Step 1: dx_phys --
        self.dx_phys: float = self.L_char / self.resolution

        # -- Step 2: Detect U_max + ALM reference quantities --
        # For ALM, propeller Reynolds is conventionally defined at 75% span:
        #     V_75  = 0.75 * tip_speed  (rotational velocity at r/R=0.75)
        #     c_75  = chord at r/R=0.75 (linear-interpolated from blade sections)
        # We expose these as ref_velocity / ref_length for Re computation.
        self.omega_phys: float = 0.0
        self.R_phys: float = 0.0
        self.tip_speed: float = 0.0
        self.chord_phys: float = 0.0       # mean chord (kept for back-compat)
        self.chord_ref:  float = 0.0       # chord at r/R = 0.75
        self._has_alm = False

        if actuator_line is not None and actuator_line.get('rotor'):
            rotor_cfg = actuator_line['rotor']
            rpm = rotor_cfg.get('rpm', 0)
            self.omega_phys = rpm * 2.0 * np.pi / 60.0
            self.R_phys = rotor_cfg.get('radius', 0.0)
            self.tip_speed = self.omega_phys * self.R_phys
            self._has_alm = True

            blade_cfg = rotor_cfg.get('blade', {})
            sections = blade_cfg.get('sections', [])
            if sections:
                chords = [s.get('chord', 0.0) for s in sections]
                rs     = [s.get('r',     0.0) for s in sections]
                self.chord_phys = float(np.mean(chords))
                # chord at r/R = 0.75
                if self.R_phys > 0:
                    r_arr = np.asarray(rs, dtype=float)
                    c_arr = np.asarray(chords, dtype=float)
                    r_target = 0.75 * self.R_phys
                    if r_arr.size >= 2 and r_arr.min() <= r_target <= r_arr.max():
                        # Sort by r for monotonic interp
                        order = np.argsort(r_arr)
                        self.chord_ref = float(
                            np.interp(r_target, r_arr[order], c_arr[order])
                        )
                    else:
                        idx = int(np.argmin(np.abs(r_arr - r_target)))
                        self.chord_ref = float(c_arr[idx])
                else:
                    self.chord_ref = self.chord_phys
            else:
                self.chord_phys = blade_cfg.get('chord', 0.0)
                self.chord_ref = self.chord_phys

        self.U_max_phys: float = max(self.U_inf, self.tip_speed)
        if self.U_max_phys < 1e-30:
            raise ValueError(
                "U_max_phys ~ 0: either U_inf or tip_speed must be > 0"
            )

        # -- Step 2b: acoustic-scaling u_max override --
        # If enabled, derive u_max so that LBM's effective physical speed of
        # sound (c_s_lu * dx/dt) equals c_s_phys.
        #     dt = dx / (c_s_phys * sqrt(3))   (acoustic constraint)
        #     dt = u_max * dx / U_max_phys     (UnitConverter convention)
        # =>  u_max = U_max_phys / (c_s_phys * sqrt(3))
        if self._acoustic_scaling:
            cs_lu_local = 1.0 / np.sqrt(3.0)
            u_max_acoustic = self.U_max_phys * cs_lu_local / self.c_s_phys
            Ma_lat = u_max_acoustic / cs_lu_local
            if Ma_lat > 0.30:
                raise ValueError(
                    f"acoustic_scaling=True would set u_max = {u_max_acoustic:.4f}"
                    f" giving Ma_lattice = {Ma_lat:.3f} > 0.3 (LBM unstable).\n"
                    f"  U_max_phys = {self.U_max_phys:.2f} m/s,"
                    f" c_s_phys = {self.c_s_phys} m/s\n"
                    f"  Either lower U_max_phys (RPM or U_inf), or disable"
                    f" acoustic_scaling and accept incorrect c_s."
                )
            self.u_max = u_max_acoustic

        # -- Step 3: dt_phys --
        self.dt_phys: float = self.u_max * self.dx_phys / self.U_max_phys

        # -- Step 4: viscosity / Re --
        # Only physics.nu (physical kinematic viscosity, m^2/s) is accepted.
        # Re is derived: Re = U_ref * L_ref / nu_phys
        # (tau and Re-direct input paths were removed deliberately so all
        #  cases share a single, unambiguous physical-units convention.)
        nu_input = physics.get('nu')
        if nu_input is None:
            raise ValueError(
                "physics.nu (kinematic viscosity, m^2/s) is required.\n"
                "  Re/tau input paths are no longer supported."
            )
        self.nu_phys: float = float(nu_input)

        if self._has_alm:
            # Propeller-convention Re reference: 75% span (V_75 × c_75).
            _Re_U_ref_def = (0.75 * self.tip_speed
                             if self.tip_speed > 0 else self.U_inf)
            _Re_L_ref_def = (self.chord_ref
                             if self.chord_ref > 0 else self.L_char)
        else:
            _Re_U_ref_def = self.U_inf if self.U_inf > 0 else self.U_max_phys
            _Re_L_ref_def = self.L_char

        Re_U_ref = physics.get('Re_U_ref', _Re_U_ref_def)
        Re_L_ref = physics.get('Re_L_ref', _Re_L_ref_def)
        self.Re: float = (Re_U_ref * Re_L_ref / self.nu_phys
                          if self.nu_phys > 0 else 0.0)
        self._Re_U_ref = Re_U_ref
        self._Re_L_ref = Re_L_ref

        # -- Step 5: Lattice parameters --
        self.nu_lu: float = self.nu_phys * self.dt_phys / (self.dx_phys ** 2)
        self.tau: float = 0.5 + 3.0 * self.nu_lu
        self.cs: float = 1.0 / np.sqrt(3.0)
        self.Ma: float = self.u_max / self.cs

        # -- Step 6: ALM lattice parameters --
        self.omega_lu: float = self.omega_phys * self.dt_phys
        self.R_lu: float = self.R_phys / self.dx_phys if self.dx_phys > 0 else 0
        self.tip_speed_lu: float = self.omega_lu * self.R_lu
        self.Ma_tip: float = self.tip_speed_lu / self.cs if self.tip_speed > 0 else 0
        self.U_inf_lu: float = self.U_inf * self.dt_phys / self.dx_phys

        # -- Step 7: Derived info --
        total_cells = 1
        for n in self.domain_shape:
            total_cells *= n
        self.total_cells: int = total_cells
        self.domain_phys = tuple(n * self.dx_phys for n in self.domain_shape)

        if self.omega_lu > 0:
            self.steps_per_rev: int = int(2 * np.pi / self.omega_lu)
        else:
            self.steps_per_rev = 0

        self.Re_D: float = self.U_inf * self.L_char / self.nu_phys if self.nu_phys > 0 else 0

        # -- Step 8: Stability checks --
        self._warnings = []
        self._check_stability()

    def _check_stability(self) -> None:
        if self.tau <= 0.5:
            raise ValueError(
                f"UNSTABLE: tau = {self.tau:.6f} <= 0.5\n"
                f"  Increase resolution or decrease Re"
            )
        if self.tau < 0.501:
            self._warnings.append(
                f"  Warning: tau = {self.tau:.6f} is very close to 0.5"
            )
        if self.Ma > 0.3:
            raise ValueError(
                f"COMPRESSIBILITY ERROR: Ma = {self.Ma:.3f} > 0.3\n"
                f"  Decrease u_max (current: {self.u_max})"
            )
        if self.Ma_tip > 0.3:
            raise ValueError(
                f"COMPRESSIBILITY ERROR: Ma_tip = {self.Ma_tip:.3f} > 0.3\n"
                f"  Decrease u_max (current: {self.u_max})"
            )

    # -- Conversion helpers --

    def phys_to_lu_length(self, x_phys: float) -> float:
        return x_phys / self.dx_phys

    def lu_to_phys_length(self, x_lu: float) -> float:
        return x_lu * self.dx_phys

    def phys_to_lu_velocity(self, u_phys: float) -> float:
        return u_phys * self.dt_phys / self.dx_phys

    def lu_to_phys_velocity(self, u_lu: float) -> float:
        return u_lu * self.dx_phys / self.dt_phys

    def phys_to_lu_time(self, t_phys: float) -> float:
        return t_phys / self.dt_phys

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Unit Conversion Summary")
        print(f"{'='*60}")
        print(f"  Physical:")
        print(f"    rho = {self.rho_phys} kg/m3")
        print(f"    nu = {self.nu_phys:.6e} m2/s")
        print(f"    U_inf = {self.U_inf} m/s")
        print(f"    L_char = {self.L_char} m")
        print(f"    Re = {self.Re:.0f} "
              f"(U_ref={self._Re_U_ref:.2f} m/s, "
              f"L_ref={self._Re_L_ref*1000:.2f} mm)")
        if self._has_alm:
            print(f"    tip_speed = {self.tip_speed:.2f} m/s")
        print(f"  Grid:")
        print(f"    {'x'.join(str(n) for n in self.domain_shape)}"
              f" = {self.total_cells:,} cells")
        print(f"    resolution = {self.resolution} cells/L_char")
        print(f"  Conversion:")
        print(f"    dx = {self.dx_phys*1000:.4f} mm")
        print(f"    dt = {self.dt_phys*1e6:.4f} us")
        print(f"  Lattice:")
        print(f"    tau = {self.tau:.6f}")
        print(f"    nu_lu = {self.nu_lu:.6e}")
        if self._acoustic_scaling:
            cs_eff = self.cs * self.dx_phys / self.dt_phys
            print(f"    u_max = {self.u_max:.6f} "
                  f"[acoustic-derived, c_s_phys={self.c_s_phys}]"
                  f" (Ma = {self.Ma:.4f})")
            print(f"    c_s LBM eff = {cs_eff:.2f} m/s (target {self.c_s_phys})")
        else:
            print(f"    u_max = {self.u_max} (Ma = {self.Ma:.4f})")
        print(f"    U_inf_lu = {self.U_inf_lu:.6f}")
        if self._has_alm:
            print(f"    omega_lu = {self.omega_lu:.6f} rad/lt")
            print(f"    R_lu = {self.R_lu:.2f} cells")
            print(f"    tip_speed_lu = {self.tip_speed_lu:.4f}"
                  f" (Ma_tip = {self.Ma_tip:.4f})")
            print(f"    steps/rev = {self.steps_per_rev}")
        if self.Re_D > 0 and self._has_alm:
            print(f"  Info:")
            print(f"    Re_D (diameter) = {self.Re_D:.0f}")
        for w in self._warnings:
            print(w)
        print(f"{'='*60}")
