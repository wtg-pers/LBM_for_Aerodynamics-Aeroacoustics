#!/usr/bin/env bash
# Tip-loss sweep on the FINAL formulation (mlg4 + K17L + SGSoff + LineAverage):
#   pure Prandtl = shen g100 (g=1.0 == Prandtl), then Shen g050, g030.
# ONE CASE AT A TIME on ALL 4 GPUs (4-rank, user 2026-07-23), 25 rev each;
# judge with LAST-5-REV averages. Field VTK: last 5 revs (wake statistics
# over 5 rev-locked realizations). Completed-skip = safe to re-launch.
#
# FINAL STEP (checkpoint stash): each result dir's checkpoints/ is renamed
# to ./shen_<tag>_checkpoint and moved ONE LEVEL UP (next to the result
# dirs), so a single rsync of result_*/ skips the multi-GB checkpoints.
# Restore:  mv shen_g100_checkpoint result_*shen_g100/checkpoints   (etc.)
#
#   nohup ./run_tip_loss_sweep.sh > tip_loss_sweep_driver.log 2>&1 &

set -u
STEPS=31425          # 25 rev x 1257
LASTLOG=31425
MPIRUN="mpirun --mca pml ucx -x LBM_ESOTERIC=1"

run () {
    local tag="$1"
    local csv="shen_${tag}.csv"
    if [ -f "$csv" ] && awk -F, -v L="$LASTLOG" \
         'END {exit !($1 >= L)}' "$csv" 2>/dev/null; then
        echo "[tiploss] ${tag}: already complete — skipping"
        return 0
    fi
    echo "[tiploss] ${tag} (4 GPUs): start $(date '+%F %T')"
    $MPIRUN -n 4 python main_mpi.py \
        --config "configs/hvab/hvab_hover_c10_farfield40_eso_mlg4_k17l_shen_${tag}.py" \
        --steps "$STEPS" --log-every 64 --cuda-aware 1 --dist-init \
        --devices 0,1,2,3 \
        --vtk-every 1257 --vtk-fields-last 5 --ckpt-every "$STEPS" \
        --csv "$csv" > "shen_${tag}.log" 2>&1
    echo "[tiploss] ${tag}: rc=$? $(date '+%F %T')"
}

stash_checkpoint () {
    local tag="$1"
    local d
    d=$(ls -d result_*eso_mlg4_k17l_shen_${tag} 2>/dev/null | head -1)
    if [ -n "${d}" ] && [ -d "${d}/checkpoints" ]; then
        mv "${d}/checkpoints" "./shen_${tag}_checkpoint"
        echo "[tiploss] stash: ${d}/checkpoints -> shen_${tag}_checkpoint"
    else
        echo "[tiploss] stash: ${tag} checkpoints not found — skipped"
    fi
}

run g100
run g050
run g030

for t in g100 g050 g030; do
    stash_checkpoint "$t"
done
echo "[tiploss] all done $(date '+%F %T')"
