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

from src.boundary.inlet import EquilibriumInlet

from src.utilities.check_conservation import ConservationChecker


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
    print(west_bc)
    inlet = EquilibriumInlet(xp, lattice, west_bc,
                             west_bc['velocity'],
                             density=1.0,
                             shape=domain_shape)

    Re = physics_config.get('Re')
    u_init = physics_config.get('u_init')
    char_length = physics_config.get('characteristic_length')

    max_steps = sim_params.get('time', {}).get('max_steps', 10000)
    output_interval = sim_params.get('time', {}).get('output_interval', 500)
    print(f"Re: {Re}")
    print(f"u_initial: {u_init}")
    print(f"L_char: {char_length}")

    streaming = StreamingPull(xp, lattice, domain_shape)
    conservation = ConservationChecker(xp, lattice)
    eq = Maxwellian(xp, lattice, domain)
    macro = Macroscopic(xp, lattice)
    collision = BGK(xp)

    print("...done.")
    print("="*70)
    
    # ======================================================================
    # ======================================================================
    print("\n[1] Initializing with Gaussian velocity perturbation...")
    # initial condition
    rho0 = xp.ones_like(domain.rho, dtype=xp.float64)
    u0 = xp.zeros_like(domain.u, dtype=xp.float64)

    x = xp.linspace(0, 1, Nx)
    y = xp.linspace(0, 1, Ny)
    z = xp.linspace(0, 1, Nz)
    X, Y, Z = xp.meshgrid(x, y, z, indexing='ij')
    
    # Gaussian perturbation
    u0[0] = 0.05 * xp.exp(-50 * ((X - 0.5)**2 + (Y - 0.5)**2 + (Z - 0.5)**2))

    f = eq.compute(rho0, u0)  # domain.f
    f_temp = xp.empty_like(f)
    print("Initialization complete!")

    conservation.set_reference(f)
    # ======================================================================

    # Physics parameters
    nu = u_init * char_length / Re
    tau = 3.0 * nu + 0.5

    print(f"\n[2] Running {max_steps} time steps...")
    print(f"    τ = {tau}, ν = {(1/3)*(tau-0.5):.6f}")
    print(f"    Boundary: Periodic (all directions)")

    # Run simulation
    start_time = time.perf_counter()
    pbar = tqdm(range(max_steps))

    for i in pbar:
        rho, u = macro.compute(f)
        f_eq = eq.compute(rho, u)
        f_post = collision.collide(f, f_eq, tau)

        streaming.compute(f_post, f_temp)
        f, f_temp = f_temp, f

        pbar.set_postfix(avg_rho=f"{rho.mean():.6f}")

    elapsed = time.perf_counter() - start_time
    mlups = (Nx * Ny * Nz * max_steps) / elapsed / 1e6
    
    print(f"\n[3] Performance:")
    print(f"    Elapsed: {elapsed:.3f} s")
    print(f"    MLUPS:   {mlups:.2f}")
    
    print(f"\n[4] Conservation Check (after {max_steps} steps):")
    is_conserved, details = conservation.check_conservation(f)


    return None


if __name__ == "__main__":
    main()