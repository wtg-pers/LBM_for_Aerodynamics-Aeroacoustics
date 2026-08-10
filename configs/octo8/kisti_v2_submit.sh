#!/bin/bash
# ============================================================================
# octo8 v2 — KISTI SLURM 제출 스크립트 (자동 재제출 포함)
# ============================================================================
# 100 rev = 62,800 coarse step 은 큐 wall-clock 상한 안에 안 들어간다.
# 이 스크립트는 상한에 걸리기 전에 **자기 자신을 다시 제출**하고, 다음 잡은
# --restart-latest 로 이어 받는다. 체크포인트는 5 rev(3,140 step)마다 떨어진다.
#
#   sbatch configs/octo8/kisti_v2_submit.sh          # 최초 제출 (이후 자동)
#   scancel <jobid>                                  # 연쇄 중단
#   touch STOP_OCTO8_V2                              # 다음 턴에 우아하게 종료
#
# ★ --dist-init + --restart-latest 조합은 2026-08-10 에 지원됐다. 그 전에는
#   복원이 랭크마다 전 도메인을 device 에 올려 두 옵션이 배타였고, 그러면
#   이 런은 KISTI 에서 원리적으로 불가능했다(복제 빌드 ~42 GB > V100).
#   게이트: patch_notes/mlg_blocks/gates/mpi_blocks_gate.py :: restart/dist-init
# ============================================================================

# ─────────────────────────────────────────────────────────────
#  ① 여기만 채우면 된다 (클러스터 고유값)
# ─────────────────────────────────────────────────────────────
# 파티션 (전부 wall-clock 48h). v2 = 212.7 M 셀 / 100 rev = 4.02e13 업데이트,
# LBM 은 대역폭 지배라 총 대역폭으로 환산한 추정:
#     cas_v100_2     V100 x2   1.8 TB/s   13.8 GB/rank   1.6~5.2 일  (재제출 필수)
#     cas_v100nv_8   V100 x8   7.2 TB/s    3.5 GB/rank   10~31 h
#   ★ amd_a100nv_8   A100 x8  ~16 TB/s    3.5 GB/rank   4~14 h   <- 한 잡에 끝남
#     amd_h200nv_8   H200 x8  ~38 TB/s    3.5 GB/rank   1.8~5.9 h
PARTITION="amd_a100nv_8"
WALLTIME="48:00:00"     # sinfo 기준 상한. 초과분은 자동 재제출로 이어받는다
ACCOUNT=""              # sacctmgr 상 Def Acct 가 자동 적용되면 빈 값으로 둔다
COMMENT=""              # KISTI 가 응용분야 코드를 요구하면 채운다
CUDA_MODULE="cuda/12.9.1"            # cupy-cuda12x 대응
# python 모듈: KISTI 는 3.12.4 가 deprecated 라 **로드해도 3.14.2 로
# 리다이렉트**된다. venv 는 setup_env.sh 가 고른 인터프리터로 이미 만들어져
# 있으므로 여기서는 굳이 python 모듈을 로드하지 않는다(venv 가 이긴다).
PY_MODULE=""
GCC_MODULE="gcc/15.2.0"              # ★ cudampi 가 컴파일러를 먼저 요구한다
MPI_MODULE="cudampi/openmpi-4.1.8"   # ★ cudampi/* 여야 --cuda-aware 1 이 산다
# venv 경로는 리포 위치에서 유도한다 — 사용자가 폴더명을 바꿔도 따라간다
# (setup_env.sh 는 자기 옆에 venv_lbm 을 만든다 = 리포의 부모)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$(dirname "$REPO_DIR")/venv_lbm/bin/activate"
NGPU=8

# 복제 빌드는 랭크당 ~42 GB. A100 80GB 면 들어가므로 DIST_INIT="" 로 두면
# conservation 진단이 살아난다. 40GB 판이면 --dist-init 이 필수.
# 대화형 잡에서 nvidia-smi 로 확인 후 결정할 것.
DIST_INIT="--dist-init"

# ─────────────────────────────────────────────────────────────
#  ② SLURM 지시자 — ①의 값이 여기 반영되도록 재제출 시 --export 로 넘긴다
# ─────────────────────────────────────────────────────────────
#SBATCH --job-name=octo8_v2
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --output=logs/octo8_v2_%j.out
#SBATCH --error=logs/octo8_v2_%j.out
# (partition/time/account/comment 는 아래 자동 재제출에서 CLI 로 넘긴다 —
#  #SBATCH 는 변수 확장이 안 되므로 ①을 단일 소스로 유지하려면 이 방법뿐이다)

set -u
cd "$REPO_DIR"
CFG="configs/octo8/octo8_v2_hover.py"
STOPFILE="STOP_OCTO8_V2"

# ─────────────────────────────────────────────────────────────
#  전제 확인 — 안 채웠으면 여기서 시끄럽게 죽는다
# ─────────────────────────────────────────────────────────────
if [[ -z "$PARTITION" ]]; then
    echo "ERROR: ① PARTITION 을 채우세요.  sinfo -o '%P %l %G' 로 확인" >&2
    exit 1
fi
[[ -f "$VENV" ]] || { echo "ERROR: venv 없음: $VENV (setup_env.sh 먼저)" >&2; exit 1; }

mkdir -p logs

# ─────────────────────────────────────────────────────────────
#  환경
# ─────────────────────────────────────────────────────────────
module purge
# ★ 순서 중요: cudampi 는 컴파일러가 먼저 로드돼 있어야 한다
#   ("COMPILER env vars were not properly defined")
[[ -n "$GCC_MODULE"  ]] && module load "$GCC_MODULE"
[[ -n "$PY_MODULE"   ]] && module load "$PY_MODULE"
[[ -n "$CUDA_MODULE" ]] && module load "$CUDA_MODULE"
module load "$MPI_MODULE"
module list 2>&1
source "$VENV"

export LBM_ESOTERIC=1
# CUDA-aware 경로. UCX 가 GPU 를 못 잡으면 --cuda-aware 0 으로 폴백해도
# **정확도는 동일**하고 대역폭만 손해다.
export OMPI_MCA_pml=ucx

echo "=== $(date) | job ${SLURM_JOB_ID:-none} on $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ─────────────────────────────────────────────────────────────
#  이어받기 판정 — 체크포인트가 있으면 restart
# ─────────────────────────────────────────────────────────────
CKPT_DIR="./result_octo8_v2/checkpoints"
RESTART=""
if compgen -G "$CKPT_DIR/*.npz" > /dev/null; then
    RESTART="--restart-latest"
    echo "[submit] 체크포인트 발견 → --restart-latest"
else
    echo "[submit] 체크포인트 없음 → 신규 시작"
fi

GPULIST=$(seq -s, 0 $((NGPU-1)))
mpirun -n "$NGPU" python main.py \
    --config "$CFG" --gpu "$GPULIST" --cuda-aware 1 $DIST_INIT $RESTART
RC=$?
echo "=== $(date) | solver exit $RC ==="

# ─────────────────────────────────────────────────────────────
#  완료 판정 + 자동 재제출
# ─────────────────────────────────────────────────────────────
if [[ -f "$STOPFILE" ]]; then
    echo "[submit] $STOPFILE 존재 — 연쇄 종료"; exit 0
fi
if [[ $RC -ne 0 ]]; then
    echo "[submit] 솔버가 $RC 로 죽음 — 재제출하지 않음 (로그 확인)"; exit $RC
fi

# 마지막 체크포인트 step 이 목표에 도달했나 (config 가 단일 소스)
DONE=$(python - <<'PYEOF'
import glob, os, re, sys, importlib.util
spec = importlib.util.spec_from_file_location(
    "c", "configs/octo8/octo8_v2_hover.py")
m = importlib.util.module_from_spec(spec); sys.modules["c"] = m
spec.loader.exec_module(m)
target = int(m.config["time"]["max_steps"]) - 1
fs = glob.glob("./result_octo8_v2/checkpoints/*.npz")
last = max((int(re.findall(r"(\d+)", os.path.basename(f))[-1]) for f in fs),
           default=-1)
print("1" if last >= target else "0", last, target)
PYEOF
)
read -r FLAG LAST TARGET <<< "$DONE"
echo "[submit] 마지막 체크포인트 step=$LAST / 목표 $TARGET"
if [[ "$FLAG" == "1" ]]; then
    echo "[submit] 100 rev 완료 — 연쇄 종료"; exit 0
fi

SUB=(sbatch -p "$PARTITION" -t "$WALLTIME")
[[ -n "$ACCOUNT" ]] && SUB+=(-A "$ACCOUNT")
[[ -n "$COMMENT" ]] && SUB+=(--comment "$COMMENT")
echo "[submit] 이어서 재제출: ${SUB[*]} $0"
"${SUB[@]}" "$0"
