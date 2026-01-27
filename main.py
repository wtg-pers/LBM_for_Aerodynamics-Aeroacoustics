import os, sys, time, argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from src.io.config_loader import ConfigLoader
from src.io.vtk_writer import VTKWriter
from src.io.checkpoint import CheckpointManager
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
# Command Line Arguments
# =============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LBM Solver for Aerodynamics/Aeroacoustics'
    )
    parser.add_argument(
        '--config', type=str, default='./configs/input_config.py',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--restart', type=str, default=None,
        help='Path to checkpoint file for restart'
    )
    parser.add_argument(
        '--restart-latest', action='store_true',
        help='Restart from latest checkpoint'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./results',
        help='Output directory for VTK files'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default='./checkpoints',
        help='Directory for checkpoint files'
    )
    parser.add_argument(
        '--no-vtk', action='store_true',
        help='Disable VTK output'
    )
    
    return parser.parse_args()


# =============================================================================
# Main Simulation
# =============================================================================
def main():
    args = parse_args()

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
    # lattice_model = sim_params.get('lattice_model', 'D3Q27')
    domain_config = sim_params.get('domain', {})
    physics_config = sim_params.get('physics', {})
    time_config = sim_params.get('time', {})

    xp = setup_library(device_mode)
    lattice = D3Q27(xp)

    # =========================================================================
    # Lattice Validation
    # =========================================================================
    print("\n[0] Validating Lattice Model...")
    validator = LatticeValidator(xp)
    is_valid, _ = validator.validate_all(
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

    # Convert mask for VTK output (ensure NumPy array)
    solid_mask_np = cylinder_mask.get() if hasattr(cylinder_mask, 'get') else cylinder_mask

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

    max_steps = time_config.get('time', {}).get('max_steps', 10000)
    output_interval = time_config.get('time', {}).get('output_interval', 500)
    checkpoint_interval = time_config.get('checkpoint_interval', 2000)

    print(f"\n[3] Physics Parameters")
    print(f"  Re = {Re}")
    print(f"  u_init = {u_init} [Δx/Δt]")
    print(f"  L_char = {char_length} [Δx]")
    print(f"  ν = {nu:.6f} [Δx²/Δt]")
    print(f"  τ = {tau:.6f}")

    # =========================================================================
    # Initialize I/O Modules
    # =========================================================================
    print(f"\n[4] I/O Setup")
    
    # VTK Writer (compressed .vti format)
    if not args.no_vtk:
        vtk_writer = VTKWriter(
            output_dir=args.output_dir,
            domain_shape=domain_shape,
            precision='float32',      # 50% smaller than float64
            compression_level=6       # zlib compression (0-9)
        )
        size_est = vtk_writer.get_file_size_estimate()
        print(f"  VTK output: {args.output_dir}")
        print(f"    Estimated file size: {size_est['estimated_MB']:.2f} MB per snapshot")
    else:
        vtk_writer = None
        print("  VTK output: DISABLED")
    
    # Checkpoint Manager
    checkpoint_mgr = CheckpointManager(
        output_dir=args.checkpoint_dir,
        prefix='checkpoint',
        keep_last_n=3,  # Keep only last 3 checkpoints to save disk space
        xp=xp
    )
    ckpt_est = checkpoint_mgr.get_size_estimate((lattice.Q, Nx, Ny, Nz))
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print(f"    Estimated size: {ckpt_est['estimated_MB']:.2f} MB per checkpoint")


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
    start_step = 0
    
    if args.restart_latest:
        print(f"\n[5] Restarting from latest checkpoint...")
        checkpoint_mgr.print_available()
        state = checkpoint_mgr.load_latest()
        f_old = xp.asarray(state['f'])
        start_step = state['step']
        print(f"  Resuming from step {start_step}")
        
    elif args.restart:
        print(f"\n[5] Restarting from: {args.restart}")
        state = checkpoint_mgr.load(args.restart)
        f_old = xp.asarray(state['f'])
        start_step = state['step']
        print(f"  Resuming from step {start_step}")
        
    else:
        print(f"\n[5] Initializing Flow Field...")
        
        # Initial conditions: uniform density, uniform x-velocity
        rho0 = xp.ones((Nx, Ny, Nz), dtype=xp.float64)   # [dimensionless]
        u0 = xp.zeros((3, Nx, Ny, Nz), dtype=xp.float64) # [lattice units]
        u0[0] = u_init  # u_x = u_init everywhere
        
        # Initialize distribution to equilibrium
        f_old = eq.compute(rho0, u0)
        print(f"  Initial total mass: {float(xp.sum(f_old)):.6f}")
    
    # Allocate work arrays
    f_new = xp.empty_like(f_old)
    f_post = xp.empty_like(f_old)
    
    # =========================================================================
    # Time Loop
    # =========================================================================
    print(f"\n[6] Running Simulation: {max_steps} time steps")
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

            # VTK output
            if vtk_writer is not None:
                vtk_writer.write(
                    step=step,
                    rho=rho,
                    u=u,
                    solid_mask=solid_mask_np,
                    time=float(step)
                )
        
        # -----------------------------------------------------------------
        # Checkpoint (periodic save)
        # -----------------------------------------------------------------
        if step > 0 and step % checkpoint_interval == 0:
            checkpoint_mgr.save(
                step=step,
                f=f_old,
                rho=rho,
                u=u,
                tau=tau,
                config=sim_params
            )

    elapsed = time.perf_counter() - start_time
    actual_steps = max_steps - start_step
    mlups = (Nx * Ny * Nz * actual_steps) / elapsed / 1e6
    
    print("="*70)
    print(f"[7] Performance: {elapsed:.2f}s, {mlups:.2f} MLUPS")
    
    # Final macroscopic state
    rho_final, u_final = macro.compute(f_old)
    
    # Final VTK output
    if vtk_writer is not None:
        vtk_writer.write(
            step=max_steps, 
            rho=rho_final, 
            u=u_final, 
            solid_mask=solid_mask_np, 
            time=float(max_steps)
        )
        pvd_path = vtk_writer.write_pvd('simulation.pvd')
        print(f"  PVD file written: {pvd_path}")
        print(f"  Open in ParaView to view time series animation")
    
    # Final checkpoint
    checkpoint_mgr.save(
        step=max_steps, 
        f=f_old, 
        rho=rho_final, 
        u=u_final, 
        tau=tau, 
        config=sim_params
    )
    
    # =========================================================================
    # Mass Flux Verification
    # =========================================================================
    print(f"\n[8] Final Mass Flux Balance")
    print("-"*60)
    verify_mass_flux_balance(xp, rho_final, u_final, verbose=True)
    
    # Stability check
    if xp.isnan(rho_final).any() or xp.isinf(rho_final).any():
        print("\n  ❌ INSTABILITY DETECTED: NaN or Inf values!")
        return False
    
    print("\n" + "="*70)
    print(" Simulation completed successfully!")
    print("="*70)
    return True


if __name__ == "__main__":
    main()