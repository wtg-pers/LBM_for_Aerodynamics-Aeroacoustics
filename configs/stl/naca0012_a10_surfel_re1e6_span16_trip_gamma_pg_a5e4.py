"""A-sweep arm amp=5e-4 (patch 66 pre-registration, run on the
gamma+PG base — patch 81 §8 verdict).

ONE knob over naca0012_a10_surfel_re1e6_span16_trip_gamma_pg:
trip_forcing.amp_lu = 5e-4 (base 1e-3). Discriminator: the fore
suction deficit (x/c 0.02-0.3, integral -0.072 vs NeuralFoil at
matched alpha) is the LEADING candidate for trip-arm invasiveness —
the sustained body-force band (0.05-0.15) coincides with the largest
dCp deficit. If the fore deficit scales with amplitude, confirmed;
remedy = minimum effective amplitude on the gamma+PG base. Guard:
separation must NOT return (patch 67: trip kills the rear-half form
drag — watch Cf_x sign x/c 0.5-0.8 and Cd).

Run (cluster, 2x24 GiB) — --mca pml ucx REQUIRED:
    cd /home/users/wtg1/_hd2/00_LBM_solver/0730
    LBM_ESOTERIC=1 mpirun --mca pml ucx -n 2 python main.py --mpi \
        --config configs/stl/naca0012_a10_surfel_re1e6_span16_trip_gamma_pg_a5e4.py \
        --axis z --ghost 4 --cuda-aware 1 --gpu 0,1
"""
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "gamma_pg_base",
    os.path.join(_here, "naca0012_a10_surfel_re1e6_span16_trip_gamma_pg.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

config = dict(_m.config)
_folder = "results_naca0012_a10_surfel_re1e6_span16_trip_gamma_pg_a5e4"
config["output"] = dict(config["output"],
                        output_dir=f"./{_folder}/vtk",
                        checkpoint_dir=f"./{_folder}/checkpoints",
                        csv_dir=f"./{_folder}/csv")
config["trip_forcing"] = dict(config["trip_forcing"], amp_lu=5e-4)
