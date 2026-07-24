#!/usr/bin/env bash
# HART-II hover collective sweep (Workshop Task 3 insurance set, 2026-07-23):
#   archB + Shen g0.3 formulation on mlg4, c = 3/6/9/12/15 deg, 25 rev each,
#   ONE CASE AT A TIME on 4 GPUs. Judge: FM vs CT/sigma (last-5-rev).
# Completed-skip = safe to re-launch. Field VTK kept light (last 2 revs) —
# wake is not the metric here.
# If time-critical, trim to 20 rev: STEPS=25140 (la-anchor converges by ~16).
#
# FINAL STEP stashes each result dir's checkpoints/ one level up as
# hart2_<tag>_checkpoint so a single rsync of result_*/ skips them.
# Restore:  mv hart2_c03_checkpoint result_*c03.0_*archB_shen_g030/checkpoints
#
#   nohup ./run_hart2_archb_sweep.sh > hart2_archb_sweep_driver.log 2>&1 &

set -u
STEPS=31425          # 25 rev x 1257
MPIRUN="mpirun --mca pml ucx -x LBM_ESOTERIC=1"

run () {
    local tag="$1"
    local csv="hart2_archB_${tag}.csv"
    if [ -f "$csv" ] && awk -F, -v L="$STEPS" \
         'END {exit !($1 >= L)}' "$csv" 2>/dev/null; then
        echo "[hart2] ${tag}: already complete — skipping"
        return 0
    fi
    echo "[hart2] ${tag} (4 GPUs): start $(date '+%F %T')"
    $MPIRUN -n 4 python main_mpi.py \
        --config "configs/hart2/hart2_hover_archB_shen_g030_${tag}.py" \
        --steps "$STEPS" --log-every 64 --cuda-aware 1 --dist-init \
        --devices 0,1,2,3 \
        --vtk-every 1257 --vtk-fields-last 2 --ckpt-every "$STEPS" \
        --csv "$csv" > "hart2_archB_${tag}.log" 2>&1
    echo "[hart2] ${tag}: rc=$? $(date '+%F %T')"
}

stash_checkpoint () {
    local tag="$1"
    local d
    d=$(ls -d result_hart2_hover_${tag}.0_*archB_shen_g030 2>/dev/null | head -1)
    if [ -n "${d}" ] && [ -d "${d}/checkpoints" ]; then
        mv "${d}/checkpoints" "./hart2_${tag}_checkpoint"
        echo "[hart2] stash: ${d}/checkpoints -> hart2_${tag}_checkpoint"
    else
        echo "[hart2] stash: ${tag} checkpoints not found — skipped"
    fi
}

for t in c09 c06 c12 c03 c15; do
    run "$t"
done
for t in c03 c06 c09 c12 c15; do
    stash_checkpoint "$t"
done
echo "[hart2] all done $(date '+%F %T')"
