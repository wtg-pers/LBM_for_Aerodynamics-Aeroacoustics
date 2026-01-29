import os, sys, time, glob, argparse
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
from src.boundary.wall import HalfwayBounceBack
from src.boundary.geometry import create_cylinder_mask

from src.utilities.check_conservation import ConservationChecker
from src.utilities.lattice_validation import LatticeValidator


# =============================================================================
# Directory Management Utilities
# =============================================================================

def clear_directory(directory: str, patterns: list, verbose: bool = True) -> int:
    """Clear files matching patterns from a directory
    
    Args:
        directory: Target directory path
        patterns: List of glob patterns (e.g., ['*.vti', '*.pvd'])
        verbose: Print information about removed files
        
    Returns:
        Number of files removed
    """
    if not os.path.exists(directory):
        return 0
    
    removed_count = 0
    for pattern in patterns:
        files = glob.glob(os.path.join(directory, pattern))
        for filepath in files:
            try:
                os.remove(filepath)
                removed_count += 1
            except OSError as e:
                if verbose:
                    print(f"    Warning: Could not remove {filepath}: {e}")
    
    return removed_count


def setup_output_directories(output_dir: str, checkpoint_dir: str, 
                              clear_previous: bool, is_restart: bool,
                              verbose: bool = True) -> None:
    """Create output directories and optionally clear previous results
    
    Args:
        output_dir: VTK output directory
        checkpoint_dir: Checkpoint directory
        clear_previous: Whether to clear existing files
        is_restart: If True, never clear (safety measure)
        verbose: Print status messages
    """
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Safety: Never clear on restart
    if is_restart:
        if verbose and clear_previous:
            print("  Note: clear_previous ignored during restart (safety measure)")
        return
    
    # Clear previous results if requested
    if clear_previous:
        if verbose:
            print(f"  Clearing previous results...")
        
        # Clear VTK files
        vtk_count = clear_directory(output_dir, ['*.vti', '*.vtk', '*.pvd'], verbose=False)
        if verbose and vtk_count > 0:
            print(f"    Removed {vtk_count} VTK files from {output_dir}")
        
        # Clear checkpoint files
        ckpt_count = clear_directory(checkpoint_dir, ['*.npz'], verbose=False)
        if verbose and ckpt_count > 0:
            print(f"    Removed {ckpt_count} checkpoint files from {checkpoint_dir}")



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
        return float(xp.sum(rho[0, :, :] * u[0, 0, :, :]))
    elif face == 'east':
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
    """Parse command line arguments
    
    Priority: CLI arguments > config file settings
    """
    parser = argparse.ArgumentParser(
        description='LBM Solver for Aerodynamics/Aeroacoustics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Normal run with config settings
  python main.py --clear                      # Force clear previous results
  python main.py --restart-latest             # Continue from latest checkpoint
  python main.py --restart-latest --extend 5000
        """
    )
    
    # Config file
    parser.add_argument(
        '--config', type=str, default='./configs/input_config.py',
        help='Path to configuration file'
    )
    
    # Restart options
    parser.add_argument(
        '--restart', type=str, default=None,
        help='Path to checkpoint file for restart'
    )
    parser.add_argument(
        '--restart-latest', action='store_true',
        help='Restart from latest checkpoint'
    )
    parser.add_argument(
        '--extend', type=int, default=None,
        help='Extend simulation by N additional steps from restart point'
    )
    parser.add_argument(
        '--max-steps', type=int, default=None,
        help='Override max_steps from config (absolute end step)'
    )
    
    # Directory overrides (CLI > config)
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Override VTK output directory (default: from config)'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default=None,
        help='Override checkpoint directory (default: from config)'
    )
    
    # Clear option (CLI flag forces clear regardless of config)
    parser.add_argument(
        '--clear', action='store_true',
        help='Clear previous results before starting (override config)'
    )
    
    # Disable outputs
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
    print(" LBM Solver for Aerodynamics & Aeroacoustics")
    print("="*70)

    # =========================================================================
    # Configuration Loading
    # =========================================================================
    config_loader = ConfigLoader(args.config)

    sim_params = config_loader.get_simulation_params()
    device_mode = sim_params.get('device_mode')
    # lattice_model = sim_params.get('lattice_model', 'D3Q27')
    domain_config = sim_params.get('domain', {})
    physics_config = sim_params.get('physics', {})
    time_config = sim_params.get('time', {})

    # Output configuration
    output_config = config_loader.config.get('output', {})
    vtk_config = output_config.get('vtk', {})
    checkpoint_config = output_config.get('checkpoint', {})

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
        raise RuntimeError("Lattice validation failed!")
    
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
    # Physics Parameters
    # =========================================================================
    Re = physics_config.get('Re')
    u_init = physics_config.get('u_init')
    char_length = physics_config.get('characteristic_length')

    # Kinematic viscosity from Re = U*L/ν  →  ν = U*L/Re
    nu = u_init * char_length / Re    # [lattice units: Δx²/Δt]

     # Relaxation time from ν = c_s² * (τ - 0.5)  →  τ = 3ν + 0.5
    tau = 3.0 * nu + 0.5              # [dimensionless]

    config_max_steps = time_config.get('max_steps', 10000)
    output_interval = time_config.get('output_interval', 500)
    checkpoint_interval = time_config.get('checkpoint_interval', 2000)

    print(f"\n[2] Physics Parameters")
    print(f"  Re = {Re}")
    print(f"  u_init = {u_init} [Δx/Δt]")
    print(f"  L_char = {char_length} [Δx]")
    print(f"  ν = {nu:.6f} [Δx²/Δt]")
    print(f"  τ = {tau:.6f}")

    # =========================================================================
    # Boundary Conditions
    # =========================================================================
    print(f"\n[3] Boundary Conditions")

    west_bc = config_loader.get_boundary_config('west')
    inlet = EquilibriumInlet(
        xp, lattice, 
        'west',
        velocity=west_bc['velocity'],
        density=1.0,
        shape=domain_shape
    )

    east_bc = config_loader.get_boundary_config('east')
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

    # Internal obstacle (cylinder)
    mask = create_cylinder_mask(
        xp, domain_shape,
        center=(Nx//5, Ny//2),
        radius=(char_length//2),
        axis='z',
        axis_range=(0, Nz-1)
    )
    wall_bc = HalfwayBounceBack(xp, lattice, mask)
    print(f"  → Internal obstacle: {wall_bc.get_info()}")

    # Convert mask for VTK output (ensure NumPy array)
    solid_mask_np = mask.get() if hasattr(mask, 'get') else mask

    # =========================================================================
    # Initialize I/O Modules
    # =========================================================================
    print(f"\n[4] I/O Setup")

    # Directory settings: CLI > config > default
    output_dir = args.output_dir or output_config.get('output_dir', './results')
    checkpoint_dir = args.checkpoint_dir or output_config.get('checkpoint_dir', './checkpoints')
    
    # Clear settings: CLI --clear overrides config
    is_restart = args.restart_latest or args.restart is not None
    clear_previous = args.clear or output_config.get('clear_previous', False)
    
    print(f"  VTK output dir: {output_dir}")
    print(f"  Checkpoint dir: {checkpoint_dir}")

    # Setup directories (create and optionally clear)
    setup_output_directories(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        clear_previous=clear_previous,
        is_restart=is_restart
    )
    
    # VTK Writer
    vtk_enabled = vtk_config.get('enabled', True) and not args.no_vtk
    if vtk_enabled:
        vtk_writer = VTKWriter(
            output_dir=output_dir,
            domain_shape=domain_shape,
            precision=vtk_config.get('precision', 'float32'),
            compression_level=vtk_config.get('compression_level', 0)
        )
        size_est = vtk_writer.get_file_size_estimate()
        print(f"  VTK: enabled ({size_est['estimated_MB']:.2f} MB/file)")
    else:
        vtk_writer = None
        print("  VTK: disabled")
    
    # Checkpoint Manager
    checkpoint_enabled = checkpoint_config.get('enabled', True)
    if checkpoint_enabled:
        checkpoint_mgr = CheckpointManager(
            output_dir=checkpoint_dir,
            prefix='checkpoint',
            keep_last_n=checkpoint_config.get('keep_last_n', 3),
            xp=xp
        )
        ckpt_est = checkpoint_mgr.get_size_estimate((lattice.Q, Nx, Ny, Nz))
        print(f"  Checkpoint: enabled ({ckpt_est['estimated_MB']:.2f} MB/file, keep last {checkpoint_config.get('keep_last_n', 3)})")
    else:
        checkpoint_mgr = None
        print("  Checkpoint: disabled")


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
        if checkpoint_mgr is None:
            raise RuntimeError("Cannot restart: checkpoints are disabled in config")
        checkpoint_mgr.print_available()
        state = checkpoint_mgr.load_latest()
        f_old = xp.asarray(state['f'])
        
        completed_step = state['step']
        start_step = completed_step + 1
        
        print(f"  Loaded step {completed_step}, resuming from step {start_step}")
        
    elif args.restart:
        print(f"\n[5] Restarting from: {args.restart}")
        if checkpoint_mgr is None:
            checkpoint_mgr = CheckpointManager(output_dir=checkpoint_dir, xp=xp)
        state = checkpoint_mgr.load(args.restart)
        f_old = xp.asarray(state['f'])
        
        completed_step = state['step']
        start_step = completed_step + 1
        
        print(f"  Loaded step {completed_step}, resuming from step {start_step}")
        
    else:
        print(f"\n[5] Initializing Flow Field (Fresh Start)...")
        
        rho0 = xp.ones((Nx, Ny, Nz), dtype=xp.float64)
        u0 = xp.zeros((3, Nx, Ny, Nz), dtype=xp.float64)
        u0[0] = u_init
        
        f_old = eq.compute(rho0, u0)
        print(f"  Initial total mass: {float(xp.sum(f_old)):.6f}")
    
    # =========================================================================
    # Determine End Step (max_steps)
    # =========================================================================
    if args.max_steps is not None:
        end_step = args.max_steps
        print(f"\n  End step (--max-steps): {end_step}")
    elif args.extend is not None:
        end_step = start_step + args.extend
        print(f"\n  Extending by {args.extend} steps: {start_step} → {end_step}")
    else:
        end_step = config_max_steps
        print(f"\n  End step (from config): {end_step}")
    
    if start_step >= end_step:
        print(f"\n  ⚠️  start_step ({start_step}) >= end_step ({end_step})")
        print(f"      Use --extend N or --max-steps N to continue")
        return True
    
    total_steps = end_step - start_step
    print(f"  Steps to run: {total_steps} ({start_step} → {end_step - 1})")
    
    f_new = xp.empty_like(f_old)
    f_post = xp.empty_like(f_old)
    
    # =========================================================================
    # Time Loop
    # =========================================================================
    print(f"\n[6] Running Simulation")
    print("="*70)

    # Run simulation
    start_time = time.perf_counter()
    custom_format = "{l_bar}{bar:5}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]"
    pbar = tqdm(range(start_step, end_step), 
                unit="step",
                ncols=72, bar_format=custom_format)
    
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
        # ---------------------------------------------------------------------
        f_post[:] = collision.collide(f_old, f_eq, tau)

        # ---------------------------------------------------------------------
        # Step 4: Streaming (Pull scheme)
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
            
            pbar.set_postfix({
                'ρ': f"{float(rho.mean()):.4f}",
                'in': f"{flux_in:.2f}",
                'out': f"{flux_out:.2f}"
            })
            
            if vtk_writer is not None:
                vtk_writer.write(step=step, rho=rho, u=u, 
                                 solid_mask=solid_mask_np, time=float(step))

        # Checkpoint
        if checkpoint_mgr is not None and step > 0 and step % checkpoint_interval == 0:
            checkpoint_mgr.save(step=step, f=f_old, rho=rho, u=u, 
                               tau=tau, config=sim_params)

    elapsed = time.perf_counter() - start_time
    mlups = (Nx * Ny * Nz * total_steps) / elapsed / 1e6

    # =========================================================================
    # Final Output
    # =========================================================================
    final_step = end_step - 1
    
    print("\n" + "="*70)
    print(f"[7] Summary")
    print(f"  Completed: step {start_step} → {final_step}")
    print(f"  Time: {elapsed:.2f}s | MLUPS: {mlups:.2f}")
    
    rho_final, u_final = macro.compute(f_old)
    
    if vtk_writer is not None:
        vtk_writer.write(step=final_step, rho=rho_final, u=u_final,
                         solid_mask=solid_mask_np, time=float(final_step))
        vtk_writer.write_pvd('simulation.pvd')
    
    if checkpoint_mgr is not None:
        checkpoint_mgr.save(step=final_step, f=f_old, rho=rho_final, 
                           u=u_final, tau=tau, config=sim_params)
    
    # Mass flux verification
    print(f"\n[8] Mass Flux Balance")
    verify_mass_flux_balance(xp, rho_final, u_final, verbose=True)
    
    if xp.isnan(rho_final).any() or xp.isinf(rho_final).any():
        print("\n  ❌ INSTABILITY DETECTED!")
        return False
    
    print("\n" + "="*70)
    print(" ✓ Simulation completed successfully!")
    print("="*70)
    print(f"\nTo continue: python main.py --restart-latest --extend 10000")
    
    return True


if __name__ == "__main__":
    main()