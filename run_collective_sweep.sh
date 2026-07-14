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

FAILED=""
run_case 6  configs/hvab/hvab_hover_c6_farfield40_eso_mpi2.py  sweep_c6.csv  || FAILED="$FAILED 6"
run_case 8  configs/hvab/hvab_hover_c8_farfield40_eso_mpi2.py  sweep_c8.csv  || FAILED="$FAILED 8"
run_case 10 configs/hvab/hvab_hover_c10_farfield40_eso_c10_mpi2.py sweep_c10.csv || FAILED="$FAILED 10"
run_case 12 configs/hvab/hvab_hover_c12_farfield40_eso_mpi2.py sweep_c12.csv || FAILED="$FAILED 12"

echo "[sweep] post-processing (failed cases:${FAILED:- none})"
python src/utilities/postprocess_collective_sweep.py --out sweep_analysis
echo "[sweep] done $(date '+%F %T') — see sweep_analysis/"
