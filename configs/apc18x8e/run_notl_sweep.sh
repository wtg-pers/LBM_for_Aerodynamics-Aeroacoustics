#!/usr/bin/env bash
# APC 18x8E notl RPM 스윕 — GPU 2/3 두 큐 병렬, 큐 내부는 순차.
#
# Usage (src/ 있는 메인 디렉토리에서):
#   bash configs/apc18x8e/run_notl_sweep.sh
#
# 큐 구성 override (예: 2446 재실행 제외):
#   RPMS_GPU2="4446" RPMS_GPU3="3460 5446" bash configs/apc18x8e/run_notl_sweep.sh
#
# VTK: 마커 VTP는 전 구간 10.02도(35스텝)마다, full-field는 마지막 5바퀴
# (180 이벤트 = 5 rev x 36프레임)만. full-field 1장 ~1.6GB(65M셀 x 6스칼라)
# -> 케이스당 ~280GB. 디스크 확인 후 실행할 것.
#
# 주의: config가 clear_previous=True라 같은 케이스 재실행 시 기존 결과
# 폴더(result_apc18x8e_hover_*rpm_mlg4_notl)가 삭제된다 — 2446 기존 런을
# 보존하려면 폴더를 먼저 옮기거나 위 override로 2446을 큐에서 빼라.
set -euo pipefail

STEPS=31425            # 25 rev (1257 steps/rev)
VTK_EVERY=35           # 10.02 deg
VTK_FIELDS_LAST=180    # full-field는 마지막 5 rev만 (5 x 36)
LOG_EVERY=64

RPMS_GPU2=${RPMS_GPU2:-"2446 4446"}
RPMS_GPU3=${RPMS_GPU3:-"3460 5446"}

run_queue() {
  local gpu=$1; shift
  local rpm tag
  for rpm in "$@"; do
    tag="apc18x8e_${rpm}_notl"
    echo "[gpu${gpu}] ${tag} start $(date '+%m/%d %H:%M:%S')"
    LBM_ESOTERIC=1 python main.py --mpi \
      --config "configs/apc18x8e/apc18x8e_hover_${rpm}rpm_notl.py" \
      --gpu "${gpu}" --steps "${STEPS}" --log-every "${LOG_EVERY}" \
      --vtk-every "${VTK_EVERY}" --vtk-fields-last "${VTK_FIELDS_LAST}" \
      --ckpt-every "${STEPS}" --csv "${tag}.csv" \
      > "${tag}.log" 2>&1
    echo "[gpu${gpu}] ${tag} done  $(date '+%m/%d %H:%M:%S')"
  done
}

run_queue 2 ${RPMS_GPU2} & PID2=$!
run_queue 3 ${RPMS_GPU3} & PID3=$!

status=0
wait "${PID2}" || { echo "[gpu2] queue FAILED (see *_notl.log)"; status=1; }
wait "${PID3}" || { echo "[gpu3] queue FAILED (see *_notl.log)"; status=1; }

[ "${status}" -eq 0 ] && echo "sweep complete" || echo "sweep finished with FAILURES"
exit "${status}"
