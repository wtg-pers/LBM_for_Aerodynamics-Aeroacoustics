import os, sys


sys.path.insert(0, os.path.dirname(__file__))

from src.io.config_loader import ConfigLoader
from src.utilities.device import setup_library
from src.lattice.d3q27 import D3Q27
from src.domain.domain import Domain
from src.equilibrium.equilibrium import Maxwellian
from src.macroscopic.compute import Macroscopic
from src.collision.bgk import BGK
from src.streaming.stream import STREAMROLL


def main():
    print("="*70)
    print(" Initializing...")
    config_path = './configs/test.json'
    config_loader = ConfigLoader(config_path)

    sim_params = config_loader.get_simulation_params()
    device_mode = sim_params.get('device_mode')
    lattice_model = sim_params.get('lattice_model', 'D3Q27')
    domain_config = sim_params.get('domain', {})
    physics_config = sim_params.get('physics', {})

    Nx = domain_config.get('Nx')
    Ny = domain_config.get('Ny')
    Nz = domain_config.get('Nz')

    Re = physics_config.get('Re')
    u_init = physics_config.get('u_init')
    char_length = physics_config.get('characteristic_length')

    max_steps = sim_params.get('time', {}).get('max_steps', 10000)
    output_interval = sim_params.get('time', {}).get('output_interval', 500)
    print("...done.")
    print("="*70)

    xp = setup_library(device_mode)
    lattice = D3Q27(xp)
    domain = Domain(lattice, xp, Nx, Ny, Nz)
    domain_shape = (Nx, Ny, Nz)
    # ======================================================================
    # ======================================================================

    # Physics parameters
    nu = u_init * char_length / Re
    tau = 3.0 * nu + 0.5

    eq = Maxwellian(xp, lattice, domain)
    macro = Macroscopic(xp, lattice)
    collision = BGK(xp)

    # initial condition
    rho0 = xp.ones_like(domain.rho, dtype=xp.float64)
    u0 = xp.zeros_like(domain.u, dtype=xp.float64)


    print("\nInitializing flow field...")
    u0[0, Nx//2, Ny//2, Nz//2] = 0.05
    
    f = eq.compute(rho0, u0)  # domain.f
    f_temp = xp.empty_like(f)
    print("Initialization complete!")

    streaming = STREAMROLL(xp, lattice)

    for i in range(max_steps):
        rho_i, u_i = macro.compute(f)
        f_eq_i = eq.compute(rho_i, u_i)
        f_post = collision.collide(f, f_eq_i, tau)

        streaming.compute(f_post, f_temp)
        f, f_temp = f_temp, f

        if i % output_interval == 0:
            umag = xp.sqrt(u_i[0]**2 + u_i[1]**2 + u_i[2]**2)
            print(f"t={i:6d} rho_mean={rho_i.mean().item():.12f} u_max={umag.max().item():.6e}")


    return None


if __name__ == "__main__":
    main()