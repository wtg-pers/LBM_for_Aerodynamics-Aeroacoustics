import os, sys, time, glob, argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

# =============================================================================
# Core LBM Components
# =============================================================================
from src.lattice.d3q27 import D3Q27
from src.domain.domain import Domain
from src.equilibrium.equilibrium import Maxwellian
from src.macroscopic.compute import Macroscopic
from src.collision.bgk import BGK
from src.streaming.stream import StreamingPull

# =============================================================================
# I/O Modules
# =============================================================================
from src.io.config_loader import ConfigLoader
from src.io.vtk_writer import VTKWriter
from src.io.checkpoint import CheckpointManager
from src.io.args_parser import parse_args

# =============================================================================
# Boundary Conditions
# =============================================================================
from src.boundary.base import (
    BoundaryManager,
    create_boundary_from_config,
    create_all_boundaries_from_config
)
from src.boundary.wall import HalfwayBounceBack
from src.boundary.domain_wall import DomainWallManager
from src.boundary.geometry import create_cylinder_mask
from src.boundary.geometry import create_sphere_mask

# =============================================================================
# Utilities
# =============================================================================
from src.utilities.device import setup_library
from src.utilities.check_conservation import ConservationChecker
from src.utilities.lattice_validation import LatticeValidator
from src.utilities.directory_utils import setup_output_directories
# from src.utilities.flux_utils import (
#     compute_mass_flux, 
#     verify_mass_flux_balance,
#     MassConservationTracker
# )
from src.utilities.flux_utils import (
    ControlVolumeChecker,
    create_domain_cv,
    create_obstacle_cv
)




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

    # Kinematic viscosity from Re = U*L/ν  →  ν = U*L/Re [lattice units: Δx²/Δt]
    nu = u_init * char_length / Re

     # Relaxation time from ν = c_s² * (τ - 0.5)  →  τ = 3ν + 0.5 [dimensionless]
    tau = 3.0 * nu + 0.5

    config_max_steps = time_config.get('max_steps', 10000)
    output_interval = time_config.get('output_interval', 500)
    checkpoint_interval = time_config.get('checkpoint_interval', 2000)

    print(f"\n[2] Physics Parameters")
    print(f"  Re = {Re}")
    print(f"  u_init = {u_init} [Δx/Δt], L_char = {char_length} [Δx]")
    print(f"  ν = {nu:.6f} [Δx²/Δt], τ = {tau:.6f}")

    # =========================================================================
    # Boundary Conditions (Factory Pattern)
    # =========================================================================
    print(f"\n[3] Boundary Conditions")
    
    # Get boundaries config
    boundaries_config = config_loader.config.get('boundaries', {})
    
    # Separate boundaries by method
    # - Wall methods (bounce_back): need f_post, use DomainWallManager
    # - Inlet methods: equilibrium, non_equilibrium (applied LAST for corner priority)
    # - Farfield/outlet methods: characteristic, convective, etc. (applied FIRST)
    inlet_boundaries = {}    # inlet BCs (applied last → corner priority)
    farfield_boundaries = {} # farfield/outlet BCs (applied first)
    wall_boundaries = {}     # bounce_back walls (separate handling with f_post)
    
    inlet_methods = ['equilibrium', 'non_equilibrium', 'eq', 'neq', 'non_eq']
    wall_methods = ['bounce_back', 'wall', 'hwbb', 'halfway']
    skip_methods = ['periodic', 'none', '']
    # Everything else is farfield/outlet: characteristic, convective, extrapolation, open, farfield
    
    for bc_name, bc_config in boundaries_config.items():
        method = bc_config.get('method', '').lower()
        
        # Legacy support: if 'type' exists but no 'method'
        if not method:
            bc_type = bc_config.get('type', '').lower()
            if bc_type == 'wall':
                method = 'bounce_back'
            elif bc_type == 'inlet':
                method = bc_config.get('method', 'non_equilibrium').lower()
            elif bc_type in ['outlet', 'open']:
                method = bc_config.get('method', 'characteristic').lower()
        
        if method in wall_methods:
            wall_boundaries[bc_name] = bc_config
        elif method in inlet_methods:
            inlet_boundaries[bc_name] = bc_config
        elif method not in skip_methods:
            farfield_boundaries[bc_name] = bc_config
    
    # Create BCs with correct application order:
    # 1. Farfield/outlet FIRST (general open boundaries)
    # 2. Inlet LAST (overwrites corners → inlet has priority at edges)
    #
    # This ensures: at corner (x=0, y=0), inlet condition is applied,
    # not farfield. Physically correct for external flow.
    
    bc_manager = BoundaryManager()
    print("  Boundaries (in application order):")
    
    # --- Step 1: Add farfield/outlet BCs first ---
    for bc_name, bc_config in farfield_boundaries.items():
        bc = create_boundary_from_config(xp, lattice, bc_name, bc_config, domain_shape)
        if bc is not None:
            bc_manager.add(bc)
            loc = bc_config.get('location', bc_name)
            method = bc_config.get('method', 'characteristic')
            rho = bc_config.get('rho', 1.0)
            k = bc_config.get('k', bc_config.get('relax_coeff', 0.1))
            print(f"    {bc_name}: {method} at {loc}, ρ={rho}, k={k}")
    
    # --- Step 2: Add inlet BCs last (corner priority) ---
    for bc_name, bc_config in inlet_boundaries.items():
        bc = create_boundary_from_config(xp, lattice, bc_name, bc_config, domain_shape)
        if bc is not None:
            bc_manager.add(bc)
            loc = bc_config.get('location', bc_name)
            method = bc_config.get('method', 'non_equilibrium')
            velocity = bc_config.get('velocity', 0.1)
            rho = bc_config.get('rho', 1.0)
            print(f"    {bc_name}: {method} at {loc}, u={velocity}, ρ={rho} [corner priority]")
    
    if len(bc_manager) == 0:
        print("    (none)")
    
    # Create domain wall BCs
    print("  Domain Walls:")
    wall_locations = [bc_config.get('location') for bc_config in wall_boundaries.values()]
    if wall_locations:
        domain_walls = DomainWallManager(
            xp, lattice, domain_shape,
            walls=wall_locations,
            exclude_inlet_outlet=False  # Let other BCs overwrite at corners
        )
        print(f"    {domain_walls.get_info()}")
    else:
        domain_walls = None
        print("    (none - using periodic for unlisted boundaries)")

    # Internal obstacle (from config or default)
    internal_geom = config_loader.config.get('internal_geometry', {})
    obstacle_type = 'sphere'
    obstacle_config = internal_geom.get(obstacle_type, {})
    
    if obstacle_config.get('enabled', False):
        center = obstacle_config.get('center', (Nx//5, Ny//2, Nz//2))
        radius = obstacle_config.get('radius', char_length//2)
        
        mask = create_sphere_mask(
            xp, domain_shape,
            center=center,
            radius=radius
        )
        obstacle_bc = HalfwayBounceBack(xp, lattice, mask)
        print(f"  Internal Obstacle:")
        print(f"    center={center}, R={radius}")
        print(f"    {obstacle_bc.get_info()}")
    # else:
    #     # Default cylinder (backward compatibility)
    #     mask = create_cylinder_mask(
    #         xp, domain_shape,
    #         center=(Nx//5, Ny//2),
    #         radius=(char_length//2),
    #         axis='z',
    #         axis_range=(0, Nz-1)
    #     )
    #     obstacle_bc = HalfwayBounceBack(xp, lattice, mask)
    #     print(f"  Internal Obstacle (default):")
    #     print(f"    {obstacle_bc.get_info()}")

    # Convert mask for VTK output (ensure NumPy array)
    solid_mask_np = mask.get() if hasattr(mask, 'get') else mask

    # =========================================================================
    # Initialize I/O Modules
    # =========================================================================
    print(f"\n[4] I/O Setup")

    # Directory settings: CLI > config > default
    output_dir = args.output_dir or output_config.get('output_dir', './results/vtk')
    checkpoint_dir = args.checkpoint_dir or output_config.get('checkpoint_dir', './checkpoints')
    csv_dir = args.csv_dir or output_config.get('csv_dir', './results/csv')
    
    # Clear settings: CLI --clear overrides config
    is_restart = args.restart_latest or args.restart is not None
    clear_previous = args.clear or output_config.get('clear_previous', False)
    
    print(f"  VTK output dir: {output_dir}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  CSV output dir: {csv_dir}")

    # Setup directories (create and optionally clear)
    setup_output_directories(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        csv_dir=csv_dir,
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
    # Initialize Conservation Checkers (AFTER f_old is ready)
    # =========================================================================
    print(f"\n[5.1] Initializing Conservation Checkers...")
    
    # Compute initial macroscopic fields from f_old
    rho_init, u_init_field = macro.compute(f_old)
    initial_mass = float(xp.sum(rho_init[~mask])) if mask is not None else float(xp.sum(rho_init))
    print(f"  Initial fluid mass: {initial_mass:.6f}")
    
    # 1. Domain-wide conservation checker (all boundary fluxes)
    cv_domain = create_domain_cv(xp, domain_shape, mask)
    cv_domain.initialize(rho_init, step=start_step)
    print(f"  Domain CV initialized: bounds=[0:{Nx-1}, 0:{Ny-1}, 0:{Nz-1}]")
    
    # 2. Local checker around obstacle (optional, for detailed analysis)
    cv_obstacle = create_obstacle_cv(
        xp, domain_shape,
        obstacle_center=center,
        obstacle_radius=radius,
        margin=3.0 * radius,  # margin = 3R around obstacle
        solid_mask=mask
    )
    cv_obstacle.initialize(rho_init, step=start_step)
    print(f"  Obstacle CV initialized: margin={3.0 * radius:.1f} around cylinder")
    
    # =========================================================================
    # Time Loop
    # =========================================================================
    print(f"\n[6] Running Simulation")
    print("="*70)

    # Run simulation
    start_time = time.perf_counter()
    # custom_format = "{l_bar}{bar:5}|{n_fmt}/{total_fmt}[{elapsed}{postfix}]"
    custom_format = "{l_bar}{bar:15}|{n_fmt} [{elapsed}{postfix}]"
    pbar = tqdm(range(start_step, end_step), 
                unit="step",
                ncols=70, bar_format=custom_format)
    
    # For progress bar updates
    last_result = {'relative_error': 0.0}

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
        if domain_walls is not None:
            domain_walls.apply_all(f_new, f_post)   # Walls at y,z boundaries (FIRST)
        bc_manager.apply_all(f_new)              # Inlet/Outlet (overwrites wall at corners)
        obstacle_bc.apply_with_reset(f_new, f_post)  # Internal cylinder
        # ---------------------------------------------------------------------
        # Step 6: Swap buffers for next iteration
        # ---------------------------------------------------------------------
        f_old, f_new = f_new, f_old

        # ---------------------------------------------------------------------
        # Output / Monitoring
        # ---------------------------------------------------------------------
        if step % 10 == 0:
            pbar.set_postfix({
                'ρ': f"{float(rho.mean()):.4f}",
                'drift': f"{last_result.get('mass_drift_percent', 0.0):+.4f}%"
            })

        if step % output_interval == 0 and step > start_step:
            # Domain-wide conservation
            # last_result = cv_domain.check(rho, u, step, verbose=True)

            # Local conservation around obstacle (optional - uncomment if needed)
            # cv_obstacle.check(rho, u, step, verbose=True)
            
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
    
    # =========================================================================
    # Mass Conservation Report
    # =========================================================================
    print(f"\n[8] Final Mass Conservation Analysis")
    print("-" * 60)
    
    # Final conservation check
    final_result = cv_domain.check(rho_final, u_final, step=final_step, verbose=True)
    
    # Total mass change from start
    final_mass = float(xp.sum(rho_final[~mask])) if mask is not None else float(xp.sum(rho_final))
    mass_change_total = final_mass - initial_mass
    mass_change_percent = mass_change_total / initial_mass * 100
    
    print(f"\n  Overall Mass Change (entire simulation):")
    print(f"    Initial mass:  {initial_mass:.6f}")
    print(f"    Final mass:    {final_mass:.6f}")
    print(f"    Change:        {mass_change_total:+.6e} ({mass_change_percent:+.4f}%)")
    
    if abs(mass_change_percent) < 0.01:
        print(f"    Assessment:    ✓ Excellent global conservation")
    elif abs(mass_change_percent) < 0.1:
        print(f"    Assessment:    ✓ Good global conservation")
    elif abs(mass_change_percent) < 1.0:
        print(f"    Assessment:    ⚠ Acceptable, minor drift")
    else:
        print(f"    Assessment:    ⚠ Significant mass drift, check BCs")
    
    # Stability check
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