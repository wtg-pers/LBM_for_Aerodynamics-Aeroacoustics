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


def compute_mass_flux(xp, rho, u, face='west'):
    """경계면을 통과하는 질량 플럭스 계산"""
    if face == 'west':
        return float(xp.sum(rho[0, :, :] * u[0, 0, :, :]))
    elif face == 'east':
        return float(xp.sum(rho[-1, :, :] * u[0, -1, :, :]))

    # 검증
    flux_inlet = compute_mass_flux(xp, rho, u, 'west')
    flux_outlet = compute_mass_flux(xp, rho, u, 'east')
    imbalance = (flux_inlet - flux_outlet) / flux_inlet * 100

    print(f"Inlet flux: {flux_inlet:.4f}")
    print(f"Outlet flux: {flux_outlet:.4f}")
    print(f"Imbalance: {imbalance:.2f}%")  # 정상상태에서 ~0%



def main():
    print("="*70)
    print(" Initializing...")
    config_path = './configs/input_config.py'
    config_loader = ConfigLoader(config_path)

    sim_params = config_loader.get_simulation_params()
    device_mode = sim_params.get('device_mode')
    lattice_model = sim_params.get('lattice_model', 'D3Q27')
    domain_config = sim_params.get('domain', {})
    physics_config = sim_params.get('physics', {})

    xp = setup_library(device_mode)
    lattice = D3Q27(xp)

    Nx = domain_config.get('Nx')
    Ny = domain_config.get('Ny')
    Nz = domain_config.get('Nz')
    domain = Domain(lattice, xp, Nx, Ny, Nz)
    domain_shape = (Nx, Ny, Nz)
    print(f"\n[Setup]")
    print(f"  Grid: {Nx} x {Ny} x {Nz}")
    print(f"  Total cells: {Nx*Ny*Nz:,}")

    west_bc = config_loader.get_boundary_config('west')
    print(f"\n[Boundary Config - west]")
    print(f"  {west_bc}")
    inlet = EquilibriumInlet(
        xp, lattice, 
        'west',
        velocity=west_bc['velocity'],
        density=1.0,
        shape=domain_shape
    )

    east_bc = config_loader.get_boundary_config('east')
    print(f"\n[Boundary Config - east]")
    print(f"  {east_bc}")
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
    print(f"\n[Boundary Conditions]")
    print(f"  Inlet (West): Equilibrium, velocity = {west_bc['velocity']}")
    print(f"  Outlet (East): Characteristic, K = {east_bc['k']}")
    print(f"  - Y/Z directions: Periodic (implicit)")

    cylinder_mask = create_cylinder_mask(xp, domain_shape,
                                         center=(Nx//4, Ny//2),
                                         radius=Ny//10,
                                         axis='z')
    wall_bc = HalfwayBounceBack(xp, lattice, cylinder_mask)
    print(wall_bc.get_info())


    Re = physics_config.get('Re')
    u_init = physics_config.get('u_init')
    char_length = physics_config.get('characteristic_length')

    max_steps = sim_params.get('time', {}).get('max_steps', 10000)
    output_interval = sim_params.get('time', {}).get('output_interval', 500)
    print(f"\n[Physics]")
    print(f"  Re: {Re}")
    print(f"  u_initial: {u_init}")
    print(f"  L_char: {char_length}")

    # Initialize componets
    streaming = StreamingPull(xp, lattice, domain_shape)
    conservation = ConservationChecker(xp, lattice)
    eq = Maxwellian(xp, lattice, domain)
    macro = Macroscopic(xp, lattice)
    collision = BGK(xp)

    print("...done.")
    print("="*70)
    
    # ======================================================================
    # initial condition
    # ======================================================================
    print("\n[1] Initializing with Gaussian velocity perturbation...")
    rho0 = xp.ones_like(domain.rho, dtype=xp.float64)
    u0 = xp.zeros_like(domain.u, dtype=xp.float64)

    u0[0] = u_init

    # Gaussian perturbation
    # x = xp.linspace(0, 1, Nx)
    # y = xp.linspace(0, 1, Ny)
    # z = xp.linspace(0, 1, Nz)
    # X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    # u0[0] = 0.05 * xp.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2))

    f = eq.compute(rho0, u0)  # domain.f
    f_temp = xp.empty_like(f)
    f_post = xp.empty_like(f)

    initial_total_mass = float(xp.sum(f))
    print(f"  Initial total mass: {initial_total_mass:.6f}")
    print("Initialization complete!")
    # conservation.set_reference(f)
    
    # Physics parameters
    nu = u_init * char_length / Re
    tau = 3.0 * nu + 0.5

    print(f"\n[2] Running {max_steps} time steps...")
    print(f"    τ = {tau}, ν = {(1/3)*(tau-0.5):.6f}")
    # ======================================================================

    # Run simulation
    start_time = time.perf_counter()
    pbar = tqdm(range(max_steps), ncols=72)
    
    flux_history = []

    for step in pbar:
        # 1. Macroscopic
        rho, u = macro.compute(f)

        # 2. Equilibrium
        f_eq = eq.compute(rho, u)
        
        # 3. Collision
        f_post[:] = collision.collide(f, f_eq, tau)

        # 4. Streaming
        streaming.compute(f_post, f_temp)
        f, f_temp = f_temp, f

        # 5. Boundary Conditions (AFTER streaming)
        bc_manager.apply_all(f)
        wall_bc.apply_with_reset(f, f_post)

        # pbar.set_postfix(avg_rho=f"{rho.mean():.6f}")
        # Track mass flux periodically
        if step % output_interval == 0:
            flux_in = compute_mass_flux(xp, rho, u, 'west')
            flux_out = compute_mass_flux(xp, rho, u, 'east')
            flux_history.append((step, flux_in, flux_out))
            
            # Update progress bar
            pbar.set_postfix({
                'ρ_avg': f"{float(rho.mean()):.3f}"
                # 'flux_in': f"{flux_in:.2f}",
                # 'flux_out': f"{flux_out:.2f}"
            })

    elapsed = time.perf_counter() - start_time
    mlups = (Nx * Ny * Nz * max_steps) / elapsed / 1e6
    
    print(f"\n[3] Performance:")
    print(f"    Elapsed: {elapsed:.3f} s")
    print(f"    MLUPS:   {mlups:.2f}")
    
    # ======================================================================
    # Final Analysis (Appropriate for Open Boundaries)
    # ======================================================================
    print(f"\n[4] Final State Analysis:")
    
    rho_final, u_final = macro.compute(f)
    
    print(f"\n  Density:")
    print(f"    Range: [{float(rho_final.min()):.6f}, {float(rho_final.max()):.6f}]")
    print(f"    Mean:  {float(rho_final.mean()):.6f}")
    
    print(f"\n  Velocity (u_x):")
    print(f"    Range: [{float(u_final[0].min()):.6f}, {float(u_final[0].max()):.6f}]")
    print(f"    Mean:  {float(u_final[0].mean()):.6f}")
    
    # ======================================================================
    # Mass Flux Balance Check (Correct for Open BCs!)
    # ======================================================================
    print(f"\n[5] Mass Flux Balance Check:")
    print("="*60)
    print("  NOTE: For open boundaries (inlet/outlet), total mass is NOT")
    print("  conserved. Instead, we check MASS FLUX BALANCE at steady state.")
    print("="*60)
    
    flux_inlet = compute_mass_flux(xp, rho_final, u_final, 'west')
    flux_outlet = compute_mass_flux(xp, rho_final, u_final, 'east')
    flux_imbalance = (flux_inlet - flux_outlet) / (abs(flux_inlet) + 1e-10)
    
    print(f"\n  Inlet mass flux:  {flux_inlet:.6f}")
    print(f"  Outlet mass flux: {flux_outlet:.6f}")
    print(f"  Imbalance: {flux_imbalance*100:.4f}%")
    
    # Check if steady state is reached
    if len(flux_history) >= 2:
        _, last_in, last_out = flux_history[-1]
        _, prev_in, prev_out = flux_history[-2]
        
        flux_change = abs(last_out - prev_out) / (abs(prev_out) + 1e-10)
        print(f"\n  Flux change (last interval): {flux_change*100:.4f}%")
        
        if flux_change < 0.01:  # Less than 1% change
            print("  → Steady state likely reached ✓")
        else:
            print("  → Still evolving, may need more steps")
    
    # Final verdict
    if abs(flux_imbalance) < 0.05:  # Less than 5% imbalance
        print(f"\n✅ Mass flux balanced within 5% tolerance!")
    else:
        print(f"\n⚠️ Mass flux imbalance > 5%")
        print("   This may indicate:")
        print("   - Simulation needs more steps to reach steady state")
        print("   - Boundary conditions need adjustment")
    
    # Check for instability
    if xp.isnan(rho_final).any() or xp.isinf(rho_final).any():
        print("\n❌ INSTABILITY DETECTED: NaN or Inf values found!")
        return False
    
    print("\n" + "="*70)
    print(" Simulation completed successfully!")
    print("="*70)


    return None


if __name__ == "__main__":
    main()