#!/usr/bin/env bash
# HVAB workshop collective sweep — archB + Shen g0.3 mlg4 (c10 already done):
#   c12 -> c08 -> c06 (information-value order: EXP FM peak side first),
#   25 rev each, ONE CASE AT A TIME on 4 GPUs, last-5-rev judging.
# Field VTK last 5 revs (wake-age figures; SG references TH08/TH10/TH12).
# Completed-skip = safe to re-launch.
#
#   nohup ./run_hvab_archb_collective_sweep.sh > hvab_archb_sweep_driver.log 2>&1 &

set -u
STEPS=31425          # 25 rev x 1257
MPIRUN="mpirun --mca pml ucx -x LBM_ESOTERIC=1"

run () {
    local tag="$1"    # c06 | c08 | c12
    local csv="archB_mlg4_shen_g030_${tag}.csv"
    if [ -f "$csv" ] && awk -F, -v L="$STEPS" \
         'END {exit !($1 >= L)}' "$csv" 2>/dev/null; then
        echo "[hvab-archb] ${tag}: already complete — skipping"
        return 0
    fi
    echo "[hvab-archb] ${tag} (4 GPUs): start $(date '+%F %T')"
    $MPIRUN -n 4 python main_mpi.py \
        --config "configs/hvab/hvab_hover_${tag}_farfield40_eso_archB_ksas_mlg4_shen_g030.py" \
        --steps "$STEPS" --log-every 64 --cuda-aware 1 --dist-init \
        --devices 0,1,2,3 \
        --vtk-every 1257 --vtk-fields-last 5 --ckpt-every "$STEPS" \
        --csv "$csv" > "archB_mlg4_shen_g030_${tag}.log" 2>&1
    echo "[hvab-archb] ${tag}: rc=$? $(date '+%F %T')"
}

stash_checkpoint () {
    local tag="$1"
    local d
    d=$(ls -d result_hvab_hover_${tag}.0_*archB_ksas_mlg4_shen_g030 2>/dev/null | head -1)
    if [ -n "${d}" ] && [ -d "${d}/checkpoints" ]; then
        mv "${d}/checkpoints" "./hvab_archb_${tag}_checkpoint"
        echo "[hvab-archb] stash: ${d}/checkpoints -> hvab_archb_${tag}_checkpoint"
    else
        echo "[hvab-archb] stash: ${tag} checkpoints not found — skipped"
    fi
}

for t in c12 c08 c06; do
    run "$t"
done
for t in c12 c08 c06; do
    stash_checkpoint "$t"
done
echo "[hvab-archb] all done $(date '+%F %T')"
