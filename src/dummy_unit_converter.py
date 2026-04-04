# # =============================================================================
# # LBM Unit Converter
# # =============================================================================
# #
# # Conversion between physical units (SI) and lattice units.
# #
# # Design Philosophy:
# #   - User inputs PHYSICAL values only (SI units)
# #   - Lattice parameters (dx, dt, τ) are computed automatically
# #   - Re takes priority over viscosity if both provided
# #
# # Reference:
# #   - OpenLB UnitConverter
# #   - Krüger et al., "The Lattice Boltzmann Method" (2017), Chapter 7
# #
# # =============================================================================

# from __future__ import annotations
# import numpy as np
# from typing import Optional, Union, Dict, Any
# import warnings


# # =============================================================================
# # Lattice Constants (Standard for D2Q9, D3Q19, D3Q27)
# # =============================================================================
# CS2: float = 1.0 / 3.0       # c_s² = 1/3  [Δx²/Δt²]
# CS: float = np.sqrt(CS2)     # c_s ≈ 0.577  [Δx/Δt]
# INV_CS2: float = 3.0         # 1/c_s² = 3


# # =============================================================================
# # UnitConverter Class
# # =============================================================================
# class UnitConverter:
#     """
#     Conversion between physical (SI) and lattice units for LBM simulations.
    
#     User Input (Physical):
#         - char_length      [m]      Characteristic length (L_ref)
#         - char_velocity    [m/s]    Characteristic velocity (U_ref)
#         - Re               [-]      Reynolds number (priority) OR
#         - viscosity        [m²/s]   Kinematic viscosity
#         - density          [kg/m³]  Density
    
#     User Input (Discretization):
#         - resolution       [-]      N = L_ref / dx
#         - lattice_velocity [-]      u_lu (controls numerical accuracy)
    
#     Auto-computed:
#         - dx, dt           [m], [s]     Grid spacing, time step
#         - τ, ω             [-]          Relaxation time/frequency
#         - All conversion factors
    
#     Example:
#         >>> converter = UnitConverter(
#         ...     char_length=0.1,
#         ...     char_velocity=10.0,
#         ...     Re=1000,
#         ...     density=1.205,
#         ...     resolution=64,
#         ...     lattice_velocity=0.05
#         ... )
#         >>> converter.print_summary()
#     """
    
#     def __init__(
#         self,
#         char_length: float,           # [m]
#         char_velocity: float,         # [m/s]
#         density: float,               # [kg/m³]
#         resolution: int,              # [-] N = L_ref / dx
#         lattice_velocity: float,      # [-] u_lu
#         Re: Optional[float] = None,   # [-] Reynolds number (priority)
#         viscosity: Optional[float] = None,  # [m²/s] kinematic viscosity
#     ):
#         """
#         Initialize UnitConverter.
        
#         Args:
#             char_length: Characteristic length L_ref [m]
#             char_velocity: Characteristic velocity U_ref [m/s]
#             density: Fluid density [kg/m³]
#             resolution: Grid cells per characteristic length (N = L_ref/dx)
#             lattice_velocity: Lattice velocity u_lu (controls accuracy, ~CFL)
#             Re: Reynolds number (takes priority if both Re and viscosity given)
#             viscosity: Kinematic viscosity [m²/s]
        
#         Raises:
#             ValueError: If neither Re nor viscosity is provided
#             ValueError: If resolution <= 0 or lattice_velocity <= 0
#         """
#         # =====================================================================
#         # Validate inputs
#         # =====================================================================
#         if Re is None and viscosity is None:
#             raise ValueError("Either 'Re' or 'viscosity' must be provided")
        
#         if resolution <= 0:
#             raise ValueError(f"resolution must be > 0, got {resolution}")
        
#         if lattice_velocity <= 0:
#             raise ValueError(f"lattice_velocity must be > 0, got {lattice_velocity}")
        
#         # =====================================================================
#         # Handle Re vs viscosity priority
#         # =====================================================================
#         if Re is not None and viscosity is not None:
#             # Both provided: Re takes priority
#             computed_viscosity = char_velocity * char_length / Re
#             warnings.warn(
#                 f"Both Re and viscosity provided. Using Re={Re} (priority).\n"
#                 f"  → Computed viscosity: {computed_viscosity:.6e} m²/s\n"
#                 f"  → Provided viscosity: {viscosity:.6e} m²/s (ignored)",
#                 UserWarning
#             )
#             viscosity = computed_viscosity
#         elif Re is not None:
#             # Only Re provided
#             viscosity = char_velocity * char_length / Re
#         else:
#             # Only viscosity provided
#             Re = char_velocity * char_length / viscosity
        
#         # =====================================================================
#         # Store physical parameters
#         # =====================================================================
#         self._char_length = char_length           # [m]
#         self._char_velocity = char_velocity       # [m/s]
#         self._viscosity = viscosity               # [m²/s]
#         self._density = density                   # [kg/m³]
#         self._Re = Re                             # [-]
        
#         # =====================================================================
#         # Store discretization parameters
#         # =====================================================================
#         self._resolution = resolution             # [-]
#         self._lattice_velocity = lattice_velocity # [-] u_lu
        
#         # =====================================================================
#         # Compute conversion factors
#         # =====================================================================
#         # dx = L_ref / N  [m/lu]
#         self._dx = char_length / resolution
        
#         # dt = u_lu × dx / U_ref  [s/lt]
#         self._dt = lattice_velocity * self._dx / char_velocity
        
#         # =====================================================================
#         # Compute lattice parameters
#         # =====================================================================
#         # ν_lu = ν_phys × dt / dx²  [lu²/lt]
#         self._nu_lu = viscosity * self._dt / (self._dx ** 2)
        
#         # τ = 0.5 + ν_lu / c_s² = 0.5 + 3 × ν_lu  [-]
#         self._tau = 0.5 + self._nu_lu * INV_CS2
        
#         # ω = 1 / τ  [-]
#         self._omega = 1.0 / self._tau
        
#         # =====================================================================
#         # Compute all conversion factors
#         # =====================================================================
#         self._C_length = self._dx                              # [m/lu]
#         self._C_time = self._dt                                # [s/lt]
#         self._C_velocity = self._dx / self._dt                 # [m·s⁻¹]
#         self._C_density = density                              # [kg·m⁻³]
#         self._C_viscosity = self._dx**2 / self._dt             # [m²·s⁻¹]
#         self._C_force = density * self._dx**4 / self._dt**2    # [N]
#         self._C_torque = density * self._dx**5 / self._dt**2   # [N·m]
#         self._C_pressure = density * self._dx**2 / self._dt**2 # [Pa]
#         self._C_mass = density * self._dx**3                   # [kg]
        
#         # =====================================================================
#         # Stability check
#         # =====================================================================
#         self._check_stability()
    
#     # =========================================================================
#     # Factory classmethod from config dict
#     # =========================================================================
#     @classmethod
#     def from_config(cls, config: Dict[str, Any]) -> UnitConverter:
#         """
#         Create UnitConverter from configuration dictionary.
        
#         Expected config structure:
#             {
#                 "physics": {
#                     "char_length": 0.1,      # [m]
#                     "char_velocity": 10.0,   # [m/s]
#                     "Re": 1000,              # [-] or "viscosity": 1e-5
#                     "density": 1.205         # [kg/m³]
#                 },
#                 "discretization": {
#                     "resolution": 64,        # [-]
#                     "lattice_velocity": 0.05 # [-]
#                 }
#             }
        
#         Args:
#             config: Configuration dictionary
        
#         Returns:
#             UnitConverter instance
#         """
#         physics = config["physics"]
#         discretization = config["discretization"]
        
#         return cls(
#             char_length=physics["char_length"],
#             char_velocity=physics["char_velocity"],
#             density=physics["density"],
#             resolution=discretization["resolution"],
#             lattice_velocity=discretization["lattice_velocity"],
#             Re=physics.get("Re"),
#             viscosity=physics.get("viscosity"),
#         )
    
#     # =========================================================================
#     # Stability check
#     # =========================================================================
#     def _check_stability(self) -> None:
#         """Check and warn about stability issues."""
#         warnings_list = []
        
#         # τ check
#         if self._tau <= 0.5:
#             warnings_list.append(
#                 f"CRITICAL: τ = {self._tau:.4f} ≤ 0.5 → UNSTABLE!"
#             )
#         elif self._tau < 0.52:
#             warnings_list.append(
#                 f"WARNING: τ = {self._tau:.4f} < 0.52 → marginally stable"
#             )
        
#         if self._tau > 2.0:
#             warnings_list.append(
#                 f"WARNING: τ = {self._tau:.4f} > 2.0 → accuracy degradation"
#             )
        
#         # u_lu check (Ma_lattice ≈ u_lu / 0.577)
#         ma_lattice = self._lattice_velocity / CS
#         if ma_lattice > 0.3:
#             warnings_list.append(
#                 f"CRITICAL: Ma_lattice = {ma_lattice:.3f} > 0.3 → compressibility error!"
#             )
#         elif ma_lattice > 0.1:
#             warnings_list.append(
#                 f"WARNING: Ma_lattice = {ma_lattice:.3f} > 0.1 → ~{ma_lattice**2*100:.1f}% density error"
#             )
        
#         # Print warnings
#         if warnings_list:
#             print("\n" + "=" * 60)
#             print(" ⚠️  STABILITY WARNINGS")
#             print("=" * 60)
#             for w in warnings_list:
#                 print(f"  {w}")
#             print("=" * 60 + "\n")
    
#     # =========================================================================
#     # Properties: Physical parameters (user input)
#     # =========================================================================
#     @property
#     def char_length(self) -> float:
#         """Characteristic length L_ref [m]"""
#         return self._char_length
    
#     @property
#     def char_velocity(self) -> float:
#         """Characteristic velocity U_ref [m/s]"""
#         return self._char_velocity
    
#     @property
#     def viscosity(self) -> float:
#         """Kinematic viscosity ν [m²/s]"""
#         return self._viscosity
    
#     @property
#     def density(self) -> float:
#         """Density ρ [kg/m³]"""
#         return self._density
    
#     @property
#     def Re(self) -> float:
#         """Reynolds number Re = U×L/ν [-]"""
#         return self._Re
    
#     # =========================================================================
#     # Properties: Discretization parameters
#     # =========================================================================
#     @property
#     def resolution(self) -> int:
#         """Resolution N = L_ref/dx [-]"""
#         return self._resolution
    
#     @property
#     def lattice_velocity(self) -> float:
#         """Lattice velocity u_lu [-]"""
#         return self._lattice_velocity
    
#     @property
#     def dx(self) -> float:
#         """Grid spacing [m]"""
#         return self._dx
    
#     @property
#     def dt(self) -> float:
#         """Time step [s]"""
#         return self._dt
    
#     # =========================================================================
#     # Properties: Lattice parameters (auto-computed)
#     # =========================================================================
#     @property
#     def nu_lu(self) -> float:
#         """Lattice kinematic viscosity ν_lu [lu²/lt]"""
#         return self._nu_lu
    
#     @property
#     def tau(self) -> float:
#         """Relaxation time τ [-]"""
#         return self._tau
    
#     @property
#     def omega(self) -> float:
#         """Relaxation frequency ω = 1/τ [-]"""
#         return self._omega
    
#     @property
#     def Ma_lattice(self) -> float:
#         """Lattice Mach number u_lu/c_s [-]"""
#         return self._lattice_velocity / CS
    
#     # =========================================================================
#     # Properties: Conversion factors
#     # =========================================================================
#     @property
#     def C_length(self) -> float:
#         """Length conversion factor [m/lu]"""
#         return self._C_length
    
#     @property
#     def C_time(self) -> float:
#         """Time conversion factor [s/lt]"""
#         return self._C_time
    
#     @property
#     def C_velocity(self) -> float:
#         """Velocity conversion factor [m·s⁻¹]"""
#         return self._C_velocity
    
#     @property
#     def C_force(self) -> float:
#         """Force conversion factor [N]"""
#         return self._C_force
    
#     @property
#     def C_torque(self) -> float:
#         """Torque conversion factor [N·m]"""
#         return self._C_torque
    
#     @property
#     def C_pressure(self) -> float:
#         """Pressure conversion factor [Pa]"""
#         return self._C_pressure
    
#     # =========================================================================
#     # Conversion methods: Physical → Lattice
#     # =========================================================================
#     def to_lattice_length(self, phys: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """Convert length: [m] → [lu]"""
#         return phys / self._C_length
    
#     def to_lattice_time(self, phys: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """Convert time: [s] → [lt]"""
#         return phys / self._C_time
    
#     def to_lattice_velocity(self, phys: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """Convert velocity: [m/s] → [lu/lt]"""
#         return phys / self._C_velocity
    
#     def to_lattice_force(self, phys: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """Convert force: [N] → [lu]"""
#         return phys / self._C_force
    
#     def to_lattice_torque(self, phys: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """Convert torque: [N·m] → [lu]"""
#         return phys / self._C_torque
    
#     def to_lattice_pressure(self, phys: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """Convert pressure: [Pa] → [lu]"""
#         return phys / self._C_pressure
    
#     # =========================================================================
#     # Conversion methods: Lattice → Physical
#     # =========================================================================
#     def to_phys_length(self, lattice: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """Convert length: [lu] → [m]"""
#         return lattice * self._C_length
    
#     def to_phys_time(self, lattice: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """Convert time: [lt] → [s]"""
#         return lattice * self._C_time
    
#     def to_phys_velocity(self, lattice: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """Convert velocity: [lu/lt] → [m/s]"""
#         return lattice * self._C_velocity
    
#     def to_phys_force(self, lattice: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """Convert force: [lu] → [N]"""
#         return lattice * self._C_force
    
#     def to_phys_torque(self, lattice: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """Convert torque: [lu] → [N·m]"""
#         return lattice * self._C_torque
    
#     def to_phys_pressure(self, lattice: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
#         """Convert pressure: [lu] → [Pa]"""
#         return lattice * self._C_pressure
    
#     # =========================================================================
#     # Utility methods
#     # =========================================================================
#     def steps_for_time(self, phys_time: float) -> int:
#         """Calculate number of steps for given physical time [s]."""
#         return int(np.ceil(phys_time / self._dt))
    
#     def steps_for_convective_time(self, t_star: float) -> int:
#         """Calculate steps for convective time units (t* = t × U/L)."""
#         phys_time = t_star * self._char_length / self._char_velocity
#         return self.steps_for_time(phys_time)
    
#     def get_convective_time(self, steps: int) -> float:
#         """Get convective time units for given steps."""
#         phys_time = steps * self._dt
#         return phys_time * self._char_velocity / self._char_length
    
#     # =========================================================================
#     # Print summary
#     # =========================================================================
#     def print_summary(self) -> None:
#         """Print unit converter summary."""
#         print(self._format_summary())
    
#     def _format_summary(self) -> str:
#         """Format summary string."""
#         lines = [
#             "",
#             "=" * 70,
#             " UnitConverter Summary",
#             "=" * 70,
#             "",
#             " Physical Parameters (User Input):",
#             f"   Characteristic length   L_ref = {self._char_length} m",
#             f"   Characteristic velocity U_ref = {self._char_velocity} m/s",
#             f"   Kinematic viscosity     ν     = {self._viscosity:.6e} m²/s",
#             f"   Density                 ρ     = {self._density} kg/m³",
#             f"   Reynolds number         Re    = {self._Re:.1f}",
#             "",
#             " Discretization (User Input):",
#             f"   Resolution              N     = {self._resolution}",
#             f"   Lattice velocity        u_lu  = {self._lattice_velocity}",
#             "",
#             " Lattice Parameters (Auto-computed):",
#             f"   Grid spacing            dx    = {self._dx:.6e} m",
#             f"   Time step               dt    = {self._dt:.6e} s",
#             f"   Lattice viscosity       ν_lu  = {self._nu_lu:.6f}",
#             f"   Relaxation time         τ     = {self._tau:.6f}",
#             f"   Relaxation frequency    ω     = {self._omega:.6f}",
#             f"   Lattice Mach            Ma_lu = {self.Ma_lattice:.4f}",
#             "",
#             " Conversion Factors:",
#             f"   C_length   = {self._C_length:.6e} m/lu",
#             f"   C_time     = {self._C_time:.6e} s/lt",
#             f"   C_velocity = {self._C_velocity:.6e} m/s",
#             f"   C_force    = {self._C_force:.6e} N",
#             f"   C_pressure = {self._C_pressure:.6e} Pa",
#             "",
#             " Derived Info:",
#             f"   1000 steps = {1000 * self._dt:.6f} s",
#             f"   1000 steps = {self.get_convective_time(1000):.3f} t*",
#             f"   Density error ≈ {self.Ma_lattice**2 * 100:.2f}%",
#             "=" * 70,
#         ]
        
#         # Stability indicator
#         if self._tau <= 0.5:
#             lines.append("❌ UNSTABLE: τ ≤ 0.5")
#         elif self._tau < 0.52:
#             lines.append(f"⚠️  MARGINAL: τ = {self._tau:.4f}")
#         else:
#             lines.append(f"✅ STABLE: τ = {self._tau:.4f}")
        
#         return "\n".join(lines)
    
#     def __repr__(self) -> str:
#         return f"UnitConverter(Re={self.Re:.1f}, N={self.resolution}, τ={self.tau:.4f})"


# # =============================================================================
# # Test
# # =============================================================================
# if __name__ == "__main__":
#     # Example 1: Using Re
#     print("\n" + "=" * 70)
#     print(" Example 1: Cylinder Flow (Re=1000)")
#     print("=" * 70)
    
#     converter1 = UnitConverter(
#         char_length=0.1,        # D = 0.1 m
#         char_velocity=10.0,     # U = 10 m/s
#         density=1.205,          # Air
#         resolution=64,          # N = 64
#         lattice_velocity=0.05,  # u_lu = 0.05
#         Re=1000,                # Re = 1000
#     )
#     converter1.print_summary()
    
#     # Example 2: Using viscosity
#     print("\n" + "=" * 70)
#     print(" Example 2: Using viscosity directly")
#     print("=" * 70)
    
#     converter2 = UnitConverter(
#         char_length=0.1,
#         char_velocity=10.0,
#         density=1.205,
#         resolution=64,
#         lattice_velocity=0.05,
#         viscosity=1.5e-5,       # Air at 20°C
#     )
#     converter2.print_summary()
    
#     # Example 3: From config dict
#     print("\n" + "=" * 70)
#     print(" Example 3: From config dictionary")
#     print("=" * 70)
    
#     config = {
#         "physics": {
#             "char_length": 0.25,
#             "char_velocity": 40.0,
#             "Re": 5000,
#             "density": 1.205,
#         },
#         "discretization": {
#             "resolution": 72,
#             "lattice_velocity": 0.08,
#         },
#     }
    
#     converter3 = UnitConverter.from_config(config)
#     converter3.print_summary()
    
#     # Conversion examples
#     print("\n Conversion Examples:")
#     print(f"   10 m/s → {converter3.to_lattice_velocity(10.0):.4f} lu/lt")
#     print(f"   0.05 lu/lt → {converter3.to_phys_velocity(0.05):.2f} m/s")
#     print(f"   1 N → {converter3.to_lattice_force(1.0):.6e} lu")
#     print(f"   0.01 s → {converter3.steps_for_time(0.01)} steps")