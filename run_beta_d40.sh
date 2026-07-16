#!/usr/bin/env bash
# beta-kernel D40 matrix: case 1' (pure ALM, Wendland) + case 4'
# (archB + KSAS + Wendland full chain), 25 rev each, 2-rank on GPUs 0,1.
# Judged against the existing GAUSSIAN case-1 / case-4 25-rev archives
# (mixing rank counts is legitimate — decomposition is bit-identical).
#
# Sequential with completed-skip (same convention as run_collective_sweep.sh):
# a case whose CSV already reaches the final logged step is skipped, so the
# script is safe to re-launch after an interruption (the interrupted case
# itself restarts from step 0 — dist-init has no mid-run restart).
#
#   nohup ./run_beta_d40.sh > beta_d40_driver.log 2>&1 &

set -u
STEPS=31425          # 25 rev x 1257
LOGEVERY=64          # last logged step = 31424
VTKEVERY=1257        # rev-locked marker VTP / fields (spanwise M2cn analysis)
LASTLOG=31424
MPIRUN="mpirun --mca pml ucx -x LBM_ESOTERIC=1"

run_case () {
    local tag="$1" cfg="$2" csv="$3"
    if [ -f "$csv" ] && awk -F, -v L="$LASTLOG" \
         'END {exit !($1 >= L)}' "$csv" 2>/dev/null; then
        echo "[beta] ${tag}: already complete ($csv) — skipping"
        return 0
    fi
    echo "[beta] ${tag}: starting $(date '+%F %T')"
    $MPIRUN -n 2 python main_mpi.py \
        --config "$cfg" \
        --steps "$STEPS" --log-every "$LOGEVERY" --cuda-aware 1 \
        --dist-init --devices 0,1 \
        --vtk-every "$VTKEVERY" --csv "$csv" 2>&1 | tee "beta_${tag}.log"
    local rc=${PIPESTATUS[0]}
    echo "[beta] ${tag}: finished rc=${rc} $(date '+%F %T')"
    return "$rc"
}

FAILED=""
run_case case1w configs/hvab/hvab_hover_c10_farfield40_eso_wendland.py \
    beta_case1w.csv || FAILED="$FAILED case1w"
run_case case4w configs/hvab/hvab_hover_c10_farfield40_eso_archB_ksas_wendland.py \
    beta_case4w.csv || FAILED="$FAILED case4w"

echo "[beta] done $(date '+%F %T') — failed cases:${FAILED:- none}"
echo "[beta] analysis: compare vs the gaussian case-1/case-4 archives"
echo "        (collect_alm_runs.py / analyze_4case.py lineage; success"
echo "        criteria in patch_notes/alm_beta_kernel/00_handoff.md 5.4)"
