"""Octo-8 hover 5000 RPM — MLG 2-level, 8 rotors on the FINE level.

This is the config that the multi-rotor fine-level ALM work unlocks: before it,
putting a fine box over the rotors deleted their force silently (F2C overwrote
the coarse excised region with fine data that never saw the ALM).

L1 encloses the WHOLE vehicle, not just the rotors. That is forced, not
preferred: the arms and booms run through any box drawn around the rotor disks,
and a body crossing the coupling band is rejected by _check_body_vs_coupling_band.

    L0  D/40  = 11.43 mm   324x496x200 = 32.1 M
    L1  D/80  =  5.71 mm   381x689x131 = 34.4 M   (rotor + airframe)
    -> 66.5 M cells ~ 13.8 GB (esoteric) + ~7.4 GB ALM force buffers

⚠ 로컬 실행 금지 — 본 런은 클러스터에서. 여기서는 빌드 검증과 초소형 스모크만.

Run (single GPU):
    LBM_ESOTERIC=1 python main.py \\
        --config configs/octo8/octo8_hover_5000rpm_mlg2.py --gpu 2

Run (multi-GPU; multi-rotor ALM is MPI-capable since patch 17 M5/80abeca):
    LBM_ESOTERIC=1 mpirun -n 4 python main.py \\
        --config configs/octo8/octo8_hover_5000rpm_mlg2.py --gpu 0,1,2,3
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from _octo8_hover_base import build_config

config = build_config(rpm=5000.0, n_rev=40, vtk_deg=30.0,
                      vtk_fields_last_rev=5, wall_bc="ibb")

# Whole-vehicle fine box, in L0 lattice units.
#   body bbox  x[80.0,261.2] y[80.8,415.2] z[59.8,115.6]
#   rotor disks x[130.1,259.6] y[115.3,380.7] z=83.1 (R=20)
config["mlg"] = {
    "enabled": True, "num_levels": 2, "overlap_width": 2,
    "interpolation": "cubic", "filter_level": 1,
    "levels": [
        {},
        # x_max 264 -> 270: at 264 the FRONT rotors' Gaussian support reached
        # 2.6 fine cells past the excised boundary, so part of what they
        # deposit would have been restricted out. The placement guard only
        # warns about support (it decides levels on the DISK alone), so this
        # has to be set here.
        {"regions": [{"name": "vehicle",
                      "x_min": 78, "x_max": 270,
                      "y_min": 78, "y_max": 418,
                      "z_min": 57, "z_max": 118}]},
    ],
}
