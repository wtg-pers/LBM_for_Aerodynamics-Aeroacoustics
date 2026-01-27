import os, sys, time
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from src.io.config_loader import ConfigLoader
from src.utilities.device import setup_library
from src.lattice.d3q27 import D3Q27
from src.domain.domain import Domain
from src.equilibrium.equilibrium import Maxwellian
from src.macroscopic.compute import Macroscopic
from src.collision.bgk import BGK
from src.streaming.stream import StreamingPull

# Boundary imports
from src.boundary.base import (
    BoundaryLocation, 
    BoundaryManager,
    create_boundary_from_config,
    create_all_boundaries_from_config
)
from src.boundary.inlet import EquilibriumInlet
from src.boundary.outlet import CharacteristicOutlet
from src.boundary.wall import HalfwayBounceBack, create_cylinder_mask

from src.utilities.check_conservation import ConservationChecker
from src.utilities.lattice_validation import LatticeValidator


# =============================================================================
# Mass Flux Utilities (for open boundary verification)
# =============================================================================
def compute_mass_flux(xp, rho, u, face='west'):
    """경계면을 통과하는 질량 플럭스 계산
    
    Mass flux: ṁ = ∫∫ ρ * u_n dA
    Discretized: ṁ = Σ ρ[face] * u_normal[face]
    
    Args:
        xp: Array module (numpy or cupy)
        rho: Density field, shape (Nx, Ny, Nz)  [dimensionless]
        u: Velocity field, shape (3, Nx, Ny, Nz)  [lattice units]
        face: Boundary face ('west' or 'east')
        
    Returns:
        float: Total mass flux through the face  [lattice units: mass/time]
    """
    if face == 'west':
        # West face: x = 0, normal direction = +x
        return float(xp.sum(rho[0, :, :] * u[0, 0, :, :]))
    elif face == 'east':
        # East face: x = Nx-1, normal direction = +x (outward)
        return float(xp.sum(rho[-1, :, :] * u[0, -1, :, :]))
    else:
        raise ValueError(f"Unknown face: {face}. Use 'west' or 'east'.")


def verify_mass_flux_balance(xp, rho, u, verbose=True):
    """Verify mass flux balance at inlet/outlet (for open boundaries)
    
    At steady state: ṁ_in ≈ ṁ_out
    
    Args:
        xp: Array module
        rho: Density field
        u: Velocity field
        verbose: Print results if True
        
    Returns:
        tuple: (flux_inlet, flux_outlet, imbalance_percent)
    """
    flux_inlet = compute_mass_flux(xp, rho, u, 'west')
    flux_outlet = compute_mass_flux(xp, rho, u, 'east')
    imbalance = (flux_inlet - flux_outlet) / (abs(flux_inlet) + 1e-10) * 100

    if verbose:
        print(f"Inlet flux:  {flux_inlet:.6f}")
        print(f"Outlet flux: {flux_outlet:.6f}")
        print(f"Imbalance:   {imbalance:.4f}%")
    
    return flux_inlet, flux_outlet, imbalance


# =============================================================================
# Main Simulation
# =============================================================================
def main():
    print("="*70)
    print(" Initializing LBM Solver...")
    print("="*70)

    # =========================================================================
    # Configuration Loading
    # =========================================================================
    config_path = './configs/input_config.py'
    config_loader = ConfigLoader(config_path)

    sim_params = config_loader.get_simulation_params()
    device_mode = sim_params.get('device_mode')
    lattice_model = sim_params.get('lattice_model', 'D3Q27')
    domain_config = sim_params.get('domain', {})
    physics_config = sim_params.get('physics', {})

    xp = setup_library(device_mode)
    lattice = D3Q27(xp)

    # =========================================================================
    # Lattice Validation
    # =========================================================================
    print("\n[0] Validating Lattice Model...")
    validator = LatticeValidator(xp)
    is_valid, validation_results = validator.validate_all(
        lattice.c, lattice.w, lattice.cs2, verbose=True
    )
    if not is_valid:
        raise RuntimeError("Lattice validation failed! Check lattice definition.")
    
    # =========================================================================
    # Domain Setup
    # =========================================================================
    Nx = domain_config.get('Nx')
    Ny = domain_config.get('Ny')
    Nz = domain_config.get('Nz')
    domain = Domain(lattice, xp, Nx, Ny, Nz)
    domain_shape = (Nx, Ny, Nz)

    print(f"\n[1] Domain Setup")
    print(f"  Grid: {Nx} x {Ny} x {Nz}")
    print(f"  Total cells: {Nx*Ny*Nz:,}")

    # =========================================================================
    # Boundary Conditions
    # =========================================================================
    print(f"\n[2] Boundary Conditions")

    west_bc = config_loader.get_boundary_config('west')
    print(f"  West (inlet): {west_bc}")
    inlet = EquilibriumInlet(
        xp, lattice, 
        'west',
        velocity=west_bc['velocity'],
        density=1.0,
        shape=domain_shape
    )

    east_bc = config_loader.get_boundary_config('east')
    print(f"  East (outlet): {east_bc}")
    outlet = CharacteristicOutlet(
        xp, lattice,
        'east',
        rho_target=east_bc['rho'],
        relax_coeff=east_bc['k'],
        shape=domain_shape
    )

    bc_manager = BoundaryManager()
    bc_manager.add(inlet)
    bc_manager.add(outlet)
    print(f"  → Inlet: Equilibrium BC, u = {west_bc['velocity']}")
    print(f"  → Outlet: Characteristic BC, K = {east_bc['k']}")
    print(f"  → Y/Z faces: Periodic (implicit via streaming)")

    # Internal obstacle (cylinder)
    cylinder_mask = create_cylinder_mask(
        xp, domain_shape,
        center=(Nx//4, Ny//2),
        radius=Ny//10,
        axis='z'
    )
    wall_bc = HalfwayBounceBack(xp, lattice, cylinder_mask)
    print(f"\n  Internal obstacle:")
    print(f"  {wall_bc.get_info()}")


    # =========================================================================
    # Physics Parameters
    # =========================================================================
    Re = physics_config.get('Re')
    u_init = physics_config.get('u_init')
    char_length = physics_config.get('characteristic_length')

    # Kinematic viscosity from Re = U*L/ν  →  ν = U*L/Re
    nu = u_init * char_length / Re    # [lattice units: Δx²/Δt]

     # Relaxation time from ν = c_s² * (τ - 0.5)  →  τ = 3ν + 0.5
    tau = 3.0 * nu + 0.5              # [dimensionless]

    max_steps = sim_params.get('time', {}).get('max_steps', 10000)
    output_interval = sim_params.get('time', {}).get('output_interval', 500)

    print(f"\n[3] Physics Parameters")
    print(f"  Re = {Re}")
    print(f"  u_init = {u_init} [Δx/Δt]")
    print(f"  L_char = {char_length} [Δx]")
    print(f"  ν = {nu:.6f} [Δx²/Δt]")
    print(f"  τ = {tau:.6f}")


    # =========================================================================
    # Initialize Components
    # =========================================================================
    streaming = StreamingPull(xp, lattice, domain_shape)
    conservation = ConservationChecker(xp, lattice)
    eq = Maxwellian(xp, lattice, domain)
    macro = Macroscopic(xp, lattice)
    collision = BGK(xp)
    
    # ======================================================================
    # initial condition
    # ======================================================================
    print(f"\n[4] Initializing Flow Field...")

    rho0 = xp.ones_like(domain.rho, dtype=xp.float64)  # [dimensionless]
    u0 = xp.zeros_like(domain.u, dtype=xp.float64)     # [lattice units]

    u0[0] = u_init

    # Gaussian perturbation
    # x = xp.linspace(0, 1, Nx)
    # y = xp.linspace(0, 1, Ny)
    # z = xp.linspace(0, 1, Nz)
    # X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    # u0[0] = 0.05 * xp.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2))

    # Initialize distribution to equilibrium
    f_old = eq.compute(rho0, u0)
    f_new = xp.empty_like(f_old)
    f_post = xp.empty_like(f_old)

    initial_total_mass = float(xp.sum(f_old))
    print(f"  Initial total mass: {initial_total_mass:.6f}")
    print(f"  Initialization complete!")
    
    # =========================================================================
    # Time Loop
    # =========================================================================
    print(f"\n[5] Running Simulation: {max_steps} time steps")
    print(f"    Output interval: {output_interval} steps")
    print("="*70)

    # Run simulation
    start_time = time.perf_counter()
    custom_format = "{l_bar}{bar:8}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]"
    pbar = tqdm(range(max_steps), ncols=72, bar_format=custom_format)
    
    flux_history = []

    for step in pbar:
        # ---------------------------------------------------------------------
        # Step 1: Compute Macroscopic Variables
        # ---------------------------------------------------------------------
        rho, u = macro.compute(f_old)

        # ---------------------------------------------------------------------
        # Step 2: Compute Equilibrium Distribution
        # ---------------------------------------------------------------------
        f_eq = eq.compute(rho, u)
        
        # ---------------------------------------------------------------------
        # Step 3: Collision (BGK)
        #   f_post = f - (1/τ)(f - f_eq)
        # ---------------------------------------------------------------------
        f_post[:] = collision.collide(f_old, f_eq, tau)

        # ---------------------------------------------------------------------
        # Step 4: Streaming (Pull scheme)
        #   f_new[i, x] = f_post[i, x - c_i]
        # ---------------------------------------------------------------------
        streaming.compute(f_post, f_new)

        # ---------------------------------------------------------------------
        # Step 5: Boundary Conditions (applied AFTER streaming)
        # ---------------------------------------------------------------------
        bc_manager.apply_all(f_new)
        wall_bc.apply_with_reset(f_new, f_post)

        # ---------------------------------------------------------------------
        # Step 6: Swap buffers for next iteration
        # ---------------------------------------------------------------------
        f_old, f_new = f_new, f_old

        # ---------------------------------------------------------------------
        # Output / Monitoring
        # ---------------------------------------------------------------------
        if step % output_interval == 0:
            flux_in = compute_mass_flux(xp, rho, u, 'west')
            flux_out = compute_mass_flux(xp, rho, u, 'east')
            flux_history.append((step, flux_in, flux_out))
            
            pbar.set_postfix({
                'ρ_avg': f"{float(rho.mean()):.4f}",
                'in': f"{flux_in:.1f}",
                'out': f"{flux_out:.1f}"
            }, refresh=False)

    elapsed = time.perf_counter() - start_time
    mlups = (Nx * Ny * Nz * max_steps) / elapsed / 1e6
    
    print("="*70)
    print(f"\n[6] Performance Summary")
    print(f"  Elapsed time: {elapsed:.3f} s")
    print(f"  MLUPS: {mlups:.2f} (Million Lattice Updates Per Second)")
    
    # =========================================================================
    # Final Analysis
    # =========================================================================
    print(f"\n[7] Final State Analysis")
    
    rho_final, u_final = macro.compute(f_old)
    
    print(f"\n  Density [ρ/ρ_0]:")
    print(f"    Range: [{float(rho_final.min()):.6f}, {float(rho_final.max()):.6f}]")
    print(f"    Mean:  {float(rho_final.mean()):.6f}")
    
    print(f"\n  Velocity u_x [Δx/Δt]:")
    print(f"    Range: [{float(u_final[0].min()):.6f}, {float(u_final[0].max()):.6f}]")
    print(f"    Mean:  {float(u_final[0].mean()):.6f}")
    
    # =========================================================================
    # Mass Flux Balance Verification
    # =========================================================================
    print(f"\n[8] Mass Flux Balance (Open Boundary Verification)")
    print("-"*60)
    print("  NOTE: For open boundaries (inlet/outlet), total mass is NOT")
    print("  conserved. Instead, we verify MASS FLUX BALANCE at steady state.")
    print("-"*60)
    
    flux_inlet, flux_outlet, imbalance = verify_mass_flux_balance(
        xp, rho_final, u_final, verbose=True
    )
    
    # Check convergence to steady state
    if len(flux_history) >= 2:
        _, last_in, last_out = flux_history[-1]
        _, prev_in, prev_out = flux_history[-2]
        
        flux_change = abs(last_out - prev_out) / (abs(prev_out) + 1e-10)
        print(f"\n  Flux change (last interval): {flux_change*100:.4f}%")
        
        if flux_change < 0.01:
            print("  → Steady state likely reached ✓")
        else:
            print("  → Still evolving, may need more steps")
    
    # Final verdict
    if abs(imbalance) < 5.0:
        print(f"\n  ✅ Mass flux balanced within 5% tolerance!")
    else:
        print(f"\n  ⚠️  Mass flux imbalance > 5%")
        print("     Possible causes:")
        print("     - Simulation needs more steps to reach steady state")
        print("     - Boundary conditions need adjustment")
    
    # Stability check
    if xp.isnan(rho_final).any() or xp.isinf(rho_final).any():
        print("\n  ❌ INSTABILITY DETECTED: NaN or Inf values found!")
        return False
    
    print("\n" + "="*70)
    print(" Simulation completed successfully!")
    print("="*70)

    return True


if __name__ == "__main__":
    main()