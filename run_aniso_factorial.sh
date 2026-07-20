#!/usr/bin/env bash
# Pure-ALM 2x2 sampling×spreading anisotropy factorial (D40, 20 rev).
#   Chain A on GPUs 0,1: s1 (iso,iso) -> s2 (aniso samp, iso spread)
#   Chain B on GPUs 2,3: s3 (iso samp, aniso spread) -> s4 (both aniso)
# Two chains run in parallel; the two cases within a chain run sequentially.
# Per case: markers every rev (20), field VTK last 2 revs only, checkpoint at
# the final step only, dense CT csv. Completed-skip = safe to re-launch.
#
#   nohup ./run_aniso_factorial.sh > aniso_factorial_driver.log 2>&1 &

set -u
STEPS=25140          # 20 rev x 1257
LASTLOG=25140
MPIRUN="mpirun --mca pml ucx -x LBM_ESOTERIC=1"

run () {
    local tag="$1" dev="$2"
    local csv="fact_${tag}.csv"
    if [ -f "$csv" ] && awk -F, -v L="$LASTLOG" \
         'END {exit !($1 >= L)}' "$csv" 2>/dev/null; then
        echo "[fact] ${tag}: already complete — skipping"
        return 0
    fi
    echo "[fact] ${tag} (GPUs ${dev}): start $(date '+%F %T')"
    $MPIRUN -n 2 python main_mpi.py \
        --config "configs/hvab/hvab_hover_c10_farfield40_eso_fact_${tag}.py" \
        --steps "$STEPS" --log-every 64 --cuda-aware 1 --dist-init \
        --devices "$dev" \
        --vtk-every 1257 --vtk-fields-last 2 --ckpt-every "$STEPS" \
        --csv "$csv" > "fact_${tag}.log" 2>&1
    echo "[fact] ${tag}: rc=$? $(date '+%F %T')"
}

# Chain A (GPUs 0,1): s1 then s2
( run s1_isoiso     0,1 ; run s2_anisosamp   0,1 ) &
CHAIN_A=$!
# Chain B (GPUs 2,3): s3 then s4
( run s3_anisospread 2,3 ; run s4_bothaniso  2,3 ) &
CHAIN_B=$!

wait "$CHAIN_A" "$CHAIN_B"
echo "[fact] all done $(date '+%F %T')"
