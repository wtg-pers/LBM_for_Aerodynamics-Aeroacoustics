#!/usr/bin/env bash
# HVAB collective sweep: 6 / 8 / 12 deg, 22 rev each, 2-rank on GPUs 0,1.
# (10 deg = existing 25-rev archive; mixing rank counts is legitimate —
#  decomposition is bit-identical to single, final acceptance test.)
#
# Sequential: each case starts when the previous finishes; a completed
# case (CSV already reaching the final logged step) is skipped, so the
# script is safe to re-launch after an interruption (the interrupted
# case itself restarts from step 0 — dist-init has no mid-run restart).
# Post-processing runs at the end over whatever cases completed.
#
#   nohup ./run_collective_sweep.sh > sweep_driver.log 2>&1 &

set -u
STEPS=27654          # 22 rev x 1257
LOGEVERY=64          # last logged step = 27648
VTKEVERY=1257        # rev-locked markers/fields; last two: 26397, 27654
LASTLOG=27648
MPIRUN="mpirun --mca pml ucx -x LBM_ESOTERIC=1"

run_case () {
    local deg="$1" cfg="$2" csv="$3"
    if [ -f "$csv" ] && awk -F, -v L="$LASTLOG" \
         'END {exit !($1 >= L)}' "$csv" 2>/dev/null; then
        echo "[sweep] c${deg}: already complete ($csv) — skipping"
        return 0
    fi
    echo "[sweep] c${deg}: starting $(date '+%F %T')"
    $MPIRUN -n 2 python main_mpi.py \
        --config "$cfg" \
        --steps "$STEPS" --log-every "$LOGEVERY" --cuda-aware 1 \
        --dist-init --devices 0,1 \
        --vtk-every "$VTKEVERY" --csv "$csv" 2>&1 | tee "sweep_c${deg}.log"
    local rc=${PIPESTATUS[0]}
    echo "[sweep] c${deg}: finished rc=${rc} $(date '+%F %T')"
    return "$rc"
}

backfill_10 () {
    # The 25-rev 10-deg archive predates marker VTP output. Restart from
    # its last checkpoint and run 2 extra revs (steps -> 33939) so marker
    # snapshots 32682/33939 exist for the spanwise comparison. Replicated
    # init (restart forbids --dist-init). Skipped if markers already there.
    local rdir="result_hvab_hover_c10.0_M650_mlg5_D40_farfield40_farfield40_eso_mpi4"
    if [ -f "$rdir/vtk/markers/markers_00033939.vtp" ]; then
        echo "[sweep] c10 backfill: markers present — skipping"; return 0
    fi
    if ! ls "$rdir"/checkpoints/checkpoint_*.npz >/dev/null 2>&1; then
        echo "[sweep] c10 backfill: no checkpoint found — spanwise will lack 10deg"
        return 0
    fi
    echo "[sweep] c10 backfill: +2 rev from latest checkpoint $(date '+%F %T')"
    $MPIRUN -n 2 python main_mpi.py \
        --config configs/hvab/hvab_hover_c10_farfield40_eso_mpi4.py \
        --steps 33939 --log-every "$LOGEVERY" --cuda-aware 1 \
        --restart-latest --devices 0,1 \
        --vtk-every "$VTKEVERY" --csv mpi4_case1_full.csv 2>&1 | tee sweep_c10_backfill.log
}

FAILED=""
run_case 6  configs/hvab/hvab_hover_c6_farfield40_eso_mpi2.py  sweep_c6.csv  || FAILED="$FAILED 6"
run_case 8  configs/hvab/hvab_hover_c8_farfield40_eso_mpi2.py  sweep_c8.csv  || FAILED="$FAILED 8"
run_case 12 configs/hvab/hvab_hover_c12_farfield40_eso_mpi2.py sweep_c12.csv || FAILED="$FAILED 12"
backfill_10 || FAILED="$FAILED 10bf"

echo "[sweep] post-processing (failed cases:${FAILED:- none})"
python src/utilities/postprocess_collective_sweep.py --out sweep_analysis
echo "[sweep] done $(date '+%F %T') — see sweep_analysis/"
