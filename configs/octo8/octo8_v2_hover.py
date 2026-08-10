"""Octo-8 v2 — 4-level MLG, 14 m field, ground-resolved outwash. **100 rev.**

The production configuration. v1 (`octo8_v1_hover.py`) was the 5-rev smoke
that got the structure running; everything below is what the smoke and the
rig campaign changed about it.

    far-field   14.0 x 14.0 m  (vehicle centre +- 7 m), ground to inflow
    ground+wake D/40  (dx 11.430 mm) out to +- 5 m — the whole outwash field
    기체        D/80  (dx  5.715 mm)
    로터        D/160 (dx  2.857 mm), one block per rotor

    L0 D/20  616x616x100    =  37.95 M
    L1 D/40  883x883x139    = 108.60 M
    L2 D/80  401x701x141    =  39.64 M
    L3 D/160 229x229x69 x8  =  28.95 M
    ------------------------------------
    총 215.1 M cells | 645.3 M updates / coarse step

What changed from v1, and why
-----------------------------
1. **L1 reaches the GROUND and spans +-5 m.** v1's L1 floated 503 mm above
   the ground, so the wall jet — the thing the wide field exists to measure —
   was resolved only at L0 (22.86 mm). It now starts at z=0 and covers the
   full +-5 m influence radius the user measured in another solver. The
   vehicle's downwash and the radial ground flow are therefore both at D/40
   or finer, continuously.

2. **L0 grown to +-7 m** so L1 (+-5 m) does not touch the domain face. That
   is not cosmetic: fine levels are built with `boundaries_config={}` — they
   have NO domain BC of their own. A fine region flush against the domain
   face gets no C2F inflow (flush faces skip coupling by design) and no BC
   either, leaving the kernel's %N wrap exposed — outwash leaving +x would
   re-enter at -x. 2 m of L0 margin keeps L1 interior.

3. **Sides are `neumann`, not `sponge`.** A sponge force-damps inside its
   20-cell layer, which destroys the quantity being measured — how far the
   wall jet actually reaches. Zero-gradient lets it leave. The cost is
   reflectivity: when the target becomes acoustics (toroidal-propeller
   noise), this must go back to sponge or a non-reflecting BC.

4. **`overlap_width` stays 2.** Daeian et al. use 4 for a curved interface,
   so it was tested: `patch_notes/wall_coupling/gates/ow4_sweep.py` measured
   ow=4 as +0.06 pt (ibb) / +0.07 pt (hwbb) — i.e. no effect, within one SEM,
   for +23% cells. The residual is read-side, and a wider band does not
   remove the body from the stencil.

Known, quantified limitation
----------------------------
The 8 rotor blocks cut the airframe on 3-4 faces each (measured: 41,490 STL
vertices inside every block). The arm sits just under the disk and the ALM
Gaussian support needs 0.25 D below it, so no block can clear the arm, and
the arms run to the booms and the booms to the fuselage — no finite block
avoids the cut. `wall_coupling.mode='allow'` (NOT 'exclude': this case is
IBB, where the C2F wall exclusion was measured to make things worse, not
better — patch_notes/wall_coupling/01 §3a).

Consequence: rotor thrust is unaffected (measured 7e-6 relative), but
**airframe download carries ~5% uncertainty**. Judge body load against a
whole-vehicle-fine run, not against this one.

Memory / runtime
----------------
Running state is 7.0 GB per rank at 4 ranks — comfortable. The REPLICATED
build would be ~42 GB and does not fit a 24 GB card, so this runs with
`--dist-init`. That is no longer a restriction: the restore path became
slab-scoped, so --dist-init and restart now coexist (gate:
`mpi_blocks_gate.py` case `restart/dist-init`). The only thing given up is
the conservation diagnostic.

Run (cluster 1, 4x RTX 4090):
    LBM_ESOTERIC=1 mpirun -n 4 python main.py \\
        --config configs/octo8/octo8_v2_hover.py --gpu 0,1,2,3 \\
        --cuda-aware 1 --dist-init

Restart (checkpoints every 5 rev):
    LBM_ESOTERIC=1 mpirun -n 4 python main.py \\
        --config configs/octo8/octo8_v2_hover.py --gpu 0,1,2,3 \\
        --cuda-aware 1 --dist-init --restart-latest
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _octo8_hover_base import (  # noqa: E402
    build_config, build_mlg_4level, mlg_report)

D_LU_0 = 20                       # L0: rotor diameter in cells
HALF_XY_MM = 7000.0               # L0 half-extent about the vehicle centre
L1_HALF_MM = 5000.0               # ground-resolved outwash radius
N_RADIAL = 48                     # delta_r 3.7 mm <= eps floor 5.71 mm @D/160
N_REV = 100

# full-field VTK 를 전 구간 30° 로 뽑는다(사용자 확정). 용량:
#   1,207 스냅샷 x 212.7 M 셀 x ~32 B/셀 = **약 7~8 TB**
# (32 B/셀 = density+pressure+velocity(3)+velocity_magnitude 4B 씩 + nu_t
#  + solid_mask; farfield40 실측 1.7 GB / 52.3 M 셀 = 32.5 B/셀 과 일치)
# KISTI /scratch 100 TB 에 들어간다. **/home01 은 64 GB 라 절대 불가** —
# 리포와 출력을 /scratch 에 두어야 한다.
# 줄이려면 vtk_deg 를 키우거나 아래를 마지막 N 바퀴로 낮춘다.
VTK_FIELDS_REV = N_REV       # 전 구간
config = build_config(rpm=5000.0, n_rev=N_REV, n_radial=N_RADIAL,
                      vtk_deg=30.0, vtk_fields_last_rev=VTK_FIELDS_REV,
                      wall_bc="ibb",
                      d_lu=D_LU_0, half_xy_mm=HALF_XY_MM, side_bc="neumann",
                      # 8기 전부 y축(날개 방향)에 평행하게 정렬해 시작.
                      # e_perp=-y_hat 이라 theta=pi/2 가 -y, +pi 더한 2번째
                      # 블레이드가 +y. 스태거를 포기하는 대신 초기 위상이 통제된다.
                      theta0=np.pi / 2)

_info = build_mlg_4level(config, d_lu0=D_LU_0, half_xy_mm=HALF_XY_MM,
                         l1_half_mm=L1_HALF_MM, overlap_width=2,
                         pad2=2.0, l1_zmin=2.0,
                         # 로터 블록이 기체를 자른다(헤더 참조). 'exclude' 가
                         # 아니라 'allow': 이 케이스는 IBB 이고, C2F 벽 제외는
                         # 오히려 나쁘다고 실측됐다(wall_coupling/01 3a).
                         wall_coupling_mode="allow")

_tag = "result_octo8_v2"
config["output"]["output_dir"] = "./%s/vtk" % _tag
config["output"]["checkpoint_dir"] = "./%s/checkpoints" % _tag
config["output"]["csv_dir"] = "./%s/csv" % _tag


def _fine_n(lo, hi, k, ow=2):
    """Fine nodes per axis INCLUDING the coupling band (region*2^k + 4*ow + 1)."""
    return [int(round((hi[i] - lo[i]) * 2 ** k)) + 4 * ow + 1
            for i in range(3)]


if __name__ == "__main__":
    mlg_report(config, _info, D_LU_0, L1_HALF_MM, N_REV, tag="octo8 v2")
    _n_out = config['time']['max_steps'] // config['time']['output_interval']
    _n_fld = int(VTK_FIELDS_REV * 628 / config['time']['output_interval'])
    print(f"  steps {config['time']['max_steps']:,} ({N_REV} rev) | "
          f"VTK every {config['time']['output_interval']} | "
          f"ckpt every {config['time']['checkpoint_interval']:,}")
    print(f"  VTK: {_n_out} 출력 중 full-field {min(_n_fld, _n_out)}개 "
          f"~ {min(_n_fld, _n_out) * _info['total_cells'] * 32 / 1e12:.1f} TB")
