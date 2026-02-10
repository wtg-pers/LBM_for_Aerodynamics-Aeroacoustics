import os, sys, time, glob, argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

# =============================================================================
# Core LBM Components
# =============================================================================
from src.lattice import get_lattice
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
)
from src.boundary.wall import HalfwayBounceBack
from src.boundary.domain_wall import DomainWallManager

from src.boundary.geometry_manager import create_geometry_mask, validate_geometry_config

# =============================================================================
# Utilities
# =============================================================================
from src.utilities.device import setup_library
from src.utilities.lattice_validation import LatticeValidator
from src.utilities.directory_utils import setup_output_directories
from src.utilities.flux_utils import ConservationManager
from src.utilities.force_calculator import ForceManager
from src.utilities.convergence import ConvergenceMonitor, ConvergenceStatus


# =============================================================================
# Main Simulation
# =============================================================================
def main():
    args = parse_args()

    # GPU 목록 출력 후 종료
    if args.list_gpus:
        from src.utilities.device import print_gpu_info
        print_gpu_info()
        return

    print("="*70)
    print(" LBM Solver for Aerodynamics & Aeroacoustics")
    print("="*70)

    # =========================================================================
    # Configuration Loading
    # =========================================================================
    config_loader = ConfigLoader(args.config)

    sim_params = config_loader.get_simulation_params()
    device_mode = sim_params.get('device_mode')
    device_id = args.gpu if args.gpu is not None else sim_params.get('device_id', 0)

    domain_config = sim_params.get('domain', {})
    physics_config = sim_params.get('physics', {})
    time_config = sim_params.get('time', {})

    # Output configuration
    output_config = config_loader.config.get('output', {})
    vtk_config = output_config.get('vtk', {})
    checkpoint_config = output_config.get('checkpoint', {})
    
    # Conservation configuration
    conservation_config = config_loader.config.get('conservation', {})
    
    # Force calculation configuration
    force_config = config_loader.config.get('force_calculation', {})

    conv_config = config_loader.config.get('convergence', {})

    xp = setup_library(device_mode, device_id=device_id)

    lattice_model = sim_params.get('lattice_model', 'D3Q27')
    lattice = get_lattice(lattice_model, xp)
    dimension = sim_params.get('dimension')

    # =========================================================================
    # Lattice Validation
    # =========================================================================
    print(f"\n[0] Validating Lattice Model ({lattice_model})...")
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
    Nz = domain_config.get('Nz')  # None for 2D
    if lattice.dim == 2:
        domain = Domain(lattice, xp, Nx, Ny, None)
        domain_shape = (Nx, Ny)
        print(f"\n[1] Domain Setup (2D)")
        print(f"  Grid: {Nx} x {Ny}")
        print(f"  Total cells: {Nx*Ny:,}")
    else:
        domain = Domain(lattice, xp, Nx, Ny, Nz)
        domain_shape = (Nx, Ny, Nz)
        print(f"\n[1] Domain Setup (3D)")
        print(f"  Grid: {Nx} x {Ny} x {Nz}")
        print(f"  Total cells: {Nx*Ny*Nz:,}")

    # =========================================================================
    # Physics Parameters
    # =========================================================================
    Re = physics_config.get('Re')
    u_init = physics_config.get('u_init')
    char_length = physics_config.get('characteristic_length')

    nu = u_init * char_length / Re
    tau = 3.0 * nu + 0.5

    config_max_steps = time_config.get('max_steps', 10000)
    output_interval = time_config.get('output_interval', 500)
    checkpoint_interval = time_config.get('checkpoint_interval', 2000)
    force_interval = time_config.get('probe_interval', force_config.get('interval', 10))

    print(f"\n[2] Physics Parameters")
    print(f"  Re = {Re}")
    print(f"  u_init = {u_init} [Δx/Δt], L_char = {char_length} [Δx]")
    print(f"  ν = {nu:.6f} [Δx²/Δt], τ = {tau:.6f}")

    # =========================================================================
    # Boundary Conditions
    # =========================================================================
    print(f"\n[3] Boundary Conditions")
    
    boundaries_config = config_loader.config.get('boundaries', {})
    
    # Separate boundaries by method
    inlet_boundaries = {}
    farfield_boundaries = {}
    wall_boundaries = {}
    
    inlet_methods = ['equilibrium', 'non_equilibrium', 'eq', 'neq', 'non_eq']
    wall_methods = ['bounce_back', 'wall', 'hwbb', 'halfway']
    skip_methods = ['periodic', 'none', '']
    
    for bc_name, bc_config in boundaries_config.items():
        method = bc_config.get('method', '').lower()
        
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
    
    bc_manager = BoundaryManager()
    print("  Boundaries (in application order):")
    
    # Add farfield/outlet BCs first
    for bc_name, bc_config in farfield_boundaries.items():
        bc = create_boundary_from_config(xp, lattice, bc_name, bc_config, domain_shape)
        if bc is not None:
            bc_manager.add(bc)
            loc = bc_config.get('location', bc_name)
            method = bc_config.get('method', 'characteristic')
            rho = bc_config.get('rho', 1.0)
            k = bc_config.get('k', bc_config.get('relax_coeff', 0.1))
            print(f"    {bc_name}: {method} at {loc}, ρ={rho}, k={k}")
    
    # Add inlet BCs last (corner priority)
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
    
    # Domain wall BCs
    print("  Domain Walls:")
    wall_locations = [bc_config.get('location') for bc_config in wall_boundaries.values()]
    if wall_locations:
        domain_walls = DomainWallManager(
            xp, lattice, domain_shape,
            walls=wall_locations,
            exclude_inlet_outlet=False
        )
        print(f"    {domain_walls.get_info()}")
    else:
        domain_walls = None
        print("    (none - using periodic for unlisted boundaries)")

    # Internal obstacle
    internal_geom = config_loader.config.get('internal_geometry', {})
    is_valid, msg = validate_geometry_config(internal_geom, domain_shape, lattice.dim)
    if not is_valid:
        raise ValueError(f"Invalid geometry configuration: {msg}")
    
    mask, geom_info = create_geometry_mask(
        xp, lattice, domain_shape, 
        internal_geom, 
        characteristic_length=char_length,
        verbose=True
    )

    if geom_info['type'] != 'none':
        obstacle_bc = HalfwayBounceBack(xp, lattice, mask)
    else:
        obstacle_bc = None

    solid_mask_np = mask.get() if hasattr(mask, 'get') else mask

    # =========================================================================
    # Initialize I/O Modules
    # =========================================================================
    print(f"\n[4] I/O Setup")

    output_dir = args.output_dir or output_config.get('output_dir', './results/vtk')
    checkpoint_dir = args.checkpoint_dir or output_config.get('checkpoint_dir', './checkpoints')
    csv_dir = args.csv_dir or output_config.get('csv_dir', './results/csv')
    
    is_restart = args.restart_latest or args.restart is not None
    clear_previous = args.clear or output_config.get('clear_previous', False)
    
    print(f"  VTK output dir: {output_dir}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  CSV output dir: {csv_dir}")

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

        if dimension == 2:
            ckpt_est = checkpoint_mgr.get_size_estimate((lattice.Q, Nx, Ny))
        else:
            ckpt_est = checkpoint_mgr.get_size_estimate((lattice.Q, Nx, Ny, Nz))
        print(f"  Checkpoint: enabled ({ckpt_est['estimated_MB']:.2f} MB/file)")
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
    
    # =========================================================================
    # Initial Condition
    # =========================================================================
    start_step = 0
    
    if args.restart_latest:
        print(f"\n[5] Restarting from latest checkpoint...")
        if checkpoint_mgr is None:
            raise RuntimeError("Cannot restart: checkpoints are disabled")
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
        
        # 2D/3D compatible initialization
        if lattice.dim == 2:
            rho0 = xp.ones((Nx, Ny), dtype=xp.float64)
            u0 = xp.zeros((2, Nx, Ny), dtype=xp.float64)
            u0[0] = u_init  # x-velocity
        else:
            rho0 = xp.ones((Nx, Ny, Nz), dtype=xp.float64)
            u0 = xp.zeros((3, Nx, Ny, Nz), dtype=xp.float64)
            u0[0] = u_init  # x-velocity
        
        f_old = eq.compute(rho0, u0)
        print(f"  Initial total mass: {float(xp.sum(f_old)):.6f}")
    
    # =========================================================================
    # Determine End Step
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
    # Initialize Conservation Manager
    # =========================================================================
    print(f"\n[5.1] Conservation Check Setup")
    
    # Determine check interval
    check_interval = conservation_config.get('check_interval', 0)
    if check_interval == 0:
        check_interval = output_interval
    
    # Create manager
    conservation_mgr = ConservationManager(
        xp=xp,
        domain_shape=domain_shape,
        config=conservation_config,
        solid_mask=mask,
        csv_dir=csv_dir
    )
    
    print(f"  {conservation_mgr.get_info()}")
    
    # Initialize with initial density
    rho_init, _ = macro.compute(f_old)
    conservation_mgr.initialize(rho_init, step=start_step)
    
    # =========================================================================
    # Initialize Force Calculator (NEW)
    # =========================================================================
    print(f"\n[5.2] Force Calculation Setup")
    
    force_enabled = force_config.get('enabled', False) and obstacle_bc is not None
    
    if force_enabled and not args.no_force:
        # Build force config with reference values
        ref_config = force_config.get('reference', {})

        # 2D/3D compatible span length
        if lattice.dim == 2:
            default_span = 1  # 2D: unit span
        else:
            default_span = Nz

        force_calc_config = {
            'enabled': True,
            'interval': force_interval,
            'start_step': force_config.get('start_step', 0),
            'reference': {
                'rho': ref_config.get('rho', 1.0),
                'velocity': ref_config.get('velocity', u_init),
                'char_length': ref_config.get('char_length', char_length),
                'span_length': ref_config.get('span_length', default_span),
            },
            'log': {
                'enabled': True,
                'filename': 'force_history'
            }
        }
        
        force_mgr = ForceManager(
            xp=xp,
            lattice=lattice,
            solid_mask=mask,
            config=force_calc_config,
            wall_bc=obstacle_bc,  # Reuse precomputed boundary links
            csv_dir=csv_dir
        )
        force_mgr.initialize()
    else:
        force_mgr = None
        if obstacle_bc is None:
            print("  Force calculation: disabled (no obstacle)")
        elif args.no_force:
            print("  Force calculation: disabled (--no-force)")
        else:
            print("  Force calculation: disabled (config)")
    

    # =============================================================================
    # [3] CONVERGENCE MONITOR INITIALIZATION
    # =============================================================================
    print(f"\n[5.3] Convergence Monitor Setup")

    has_obstacle = (obstacle_bc is not None) and (force_mgr is not None)
    conv_monitor = ConvergenceMonitor(
        config=conv_config,
        has_obstacle=has_obstacle,
        csv_dir=csv_dir,
    )

    if conv_monitor.enabled:
        conv_monitor.initialize(
            char_length=char_length,
            u_ref=u_init,
        )
    else:
        print("  Convergence monitor: disabled")
        
    # =========================================================================
    # Time Loop
    # =========================================================================
    print(f"\n[6] Running Simulation")
    print("="*70)

    start_time = time.perf_counter()
    custom_format = "{l_bar}{bar:10}|{n_fmt}/{total_fmt} {rate_fmt} [{elapsed}{postfix}]"
    pbar = tqdm(range(start_step, end_step), 
                unit="step",
                ncols=99, bar_format=custom_format)
    
    last_drift = 0.0
    last_Cd = 0.0
    last_Cl = 0.0

    for step in pbar:
        # Step 1: Compute Macroscopic Variables
        rho, u = macro.compute(f_old)

        # Step 2: Compute Equilibrium Distribution
        f_eq = eq.compute(rho, u)
        
        # Step 3: Collision (BGK)
        f_post[:] = collision.collide(f_old, f_eq, tau)

        # Step 4: Streaming (Pull scheme)
        streaming.compute(f_post, f_new)

        # Step 5: Boundary Conditions
        if domain_walls is not None:
            domain_walls.apply_all(f_new, f_post)
        bc_manager.apply_all(f_new)
        if obstacle_bc is not None:
            obstacle_bc.apply_with_reset(f_new, f_post)
        
        # =====================================================================
        # Step 5.5: Force Calculation (using f_post, before BC application)
        # =====================================================================
        # Note: Force is calculated from f_post (post-collision, pre-streaming)
        # which represents the momentum being transferred at boundary links
        if force_mgr is not None and force_mgr.should_compute(step):
            force_result = force_mgr.compute_and_log(step, f_post, verbose=False)
            if force_result:
                last_Cd = force_result['Cd']
                last_Cl = force_result['Cl']

                # --- Feed to convergence monitor ---
                conv_monitor.feed_force(step, last_Cd, last_Cl)
        
        # Step 6: Swap buffers
        f_old, f_new = f_new, f_old

        # Output / Monitoring
        if step % 10 == 0:
            if force_mgr is not None:
                postfix = {
                    'Cd': f"{last_Cd:.3f}",
                    'Cl': f"{last_Cl:.3f}",
                }                       
                pbar.set_postfix(postfix)
            else:
                pbar.set_postfix({
                    'ρ': f"{float(rho.mean()):.4f}",
                    'drift': f"{last_drift:+.4f}%"
                })
        
        # VTK output
        if vtk_writer is not None and step % output_interval == 0:
            vtk_writer.write(step=step, rho=rho, u=u, 
                                solid_mask=solid_mask_np, time=float(step))

        if step % check_interval == 0 and step > start_step:
            # Conservation check
            results = conservation_mgr.check(rho, step, verbose=(conservation_mgr.verbose > 0))
            if results.get('domain'):
                last_drift = results['domain']['mass_drift_percent']

            # --- Convergence check ---                            
            if conv_monitor.enabled:                                
                conv_monitor.feed_energy(step, rho, u)               
                conv_monitor.feed_divergence_check(rho, u)          
                conv_status = conv_monitor.check(step)            
                                                                    
                # Handle convergence/divergence actions              
                if conv_status['diverged']:                      
                    print(f"\n  ⚠ DIVERGENCE at step {step}: "      
                            f"{conv_status['diverge_reason']}")        
                    if conv_monitor.on_diverged == 'stop_with_checkpoint':
                        if checkpoint_mgr is not None:               
                            checkpoint_mgr.save(step=step, f=f_old,  
                                                rho=rho, u=u,         
                                                tau=tau, config=sim_params)
                    break                                            
                                                                    
                if conv_status['converged']:                       
                    print(f"\n  ✓ CONVERGED at step {step}")         
                    conv_monitor.print_summary()                     
                    if conv_monitor.on_converged == 'checkpoint_and_stop':
                        if checkpoint_mgr is not None:               
                            checkpoint_mgr.save(step=step, f=f_old,  
                                                rho=rho, u=u,         
                                                tau=tau, config=sim_params)
                    if conv_monitor.on_converged != 'continue':     
                        break                                        

        # Checkpoint
        if checkpoint_mgr is not None and step > 0 and step % checkpoint_interval == 0:
            checkpoint_mgr.save(step=step, f=f_old, rho=rho, u=u, 
                               tau=tau, config=sim_params)

    elapsed = time.perf_counter() - start_time

    # --- Max steps 도달 체크 ---
    if conv_monitor.enabled and not conv_monitor.converged and not conv_monitor.diverged:
        conv_monitor.mark_max_steps()
        if conv_monitor.on_max_steps == 'warn':
            print(f"\n  ⚠ Max steps reached without convergence")
        elif conv_monitor.on_max_steps == 'error':
            print(f"\n  ✗ ERROR: Max steps without convergence")
    
    # 2D/3D compatible MLUPS
    if lattice.dim == 2:
        total_cells = Nx * Ny
    else:
        total_cells = Nx * Ny * Nz
    mlups = (total_cells * total_steps) / elapsed / 1e6

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
    
    # Final conservation check
    print(f"\n[8] Final Conservation Analysis")
    final_results = conservation_mgr.check(rho_final, final_step, verbose=True)
    
    # Close CSV
    conservation_mgr.close()
    
    # =========================================================================
    # Force Summary (NEW)
    # =========================================================================
    if force_mgr is not None:
        force_mgr.print_summary()

        # Strouhal number 계산
        from src.utilities.force_calculator import compute_strouhal_number
        
        St = compute_strouhal_number(
            force_history=force_mgr.history,
            char_length=char_length,
            u_ref=u_init,
            component='Cl',
            min_periods=3
        )
        
        if St is not None:
            print(f"\n  Strouhal number: St = {St:.4f}")
            print(f"  (Expected for Re=100: St ≈ 0.16-0.17)")
            print(f"  (Expected for Re=150: St ≈ 0.18-0.19)")
        else:
            print(f"\n  Strouhal number: insufficient data for FFT")

        force_mgr.close()
    
    
    # =========================================================================
    # Convergence Summary
    # =========================================================================
    if conv_monitor.enabled:
        conv_monitor.print_summary()
        conv_monitor.close()

    # =========================================================================
    # Final Status Message (updated)
    # =========================================================================
    final_status = conv_monitor.get_status() if conv_monitor else None
    
    if final_status == ConvergenceStatus.CONVERGED:
        print("\n" + "="*70)
        print(f" ✅ Simulation CONVERGED at step {conv_monitor.converged_step}!")
        print("="*70)
    elif final_status == ConvergenceStatus.DIVERGED:
        print("\n" + "="*70)
        print(" ❌ Simulation DIVERGED!")
        print("="*70)
        return False
    else:
        # Check legacy NaN/Inf (fallback if convergence monitor is disabled)
        if xp.isnan(rho_final).any() or xp.isinf(rho_final).any():
            print("\n  ❌ INSTABILITY DETECTED!")
            return False
        
        print("\n" + "="*70)
        print(" ✓ Simulation completed.")
        print("="*70)
        print(f"\nTo continue: python main.py --restart-latest --extend 10000")
    
    return True


if __name__ == "__main__":
    main()