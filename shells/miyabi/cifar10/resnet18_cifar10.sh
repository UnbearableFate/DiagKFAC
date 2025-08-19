#!/bin/bash

#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=8:mpiprocs=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -m ae

# --- Safety & strict mode ---
set -eEuo pipefail
trap 'echo "[ERROR] Failed at line $LINENO" >&2' ERR

# --- Cluster topology ---
: "${PBS_NODEFILE:?PBS_NODEFILE is not set}"
NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# master address and port (allow override via env)
MASTER_ADDR=${MASTER_ADDR:-$(head -n 1 "$PBS_NODEFILE")}
MASTER_PORT=${MASTER_PORT:-20201}

# compute local node-rank
readarray -t hosts < <(sort -u "$PBS_NODEFILE")
NODE_RANK=0
for idx in "${!hosts[@]}"; do
  if [[ "${hosts[idx]}" == "$(hostname)" ]]; then
    NODE_RANK=$idx
    break
  fi
done

# --- Paths ---
timestamp=$(date "+%Y%m%d%H%M%S")
WORKSPACE="/work/xg24i002/x10041/DiagKFAC"
script_path="$WORKSPACE/main.py"
PYTHON_ROOT=/work/xg24i002/x10041/miniconda3/envs/diagkfac/bin
LOG_ROOT="$WORKSPACE/logs/cifar10_${timestamp}"
mkdir -p "$LOG_ROOT"

export MASTER_PORT MASTER_ADDR

echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE  WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  NODE_RANK=$NODE_RANK"
echo "LOG_ROOT=$LOG_ROOT"

# --- Helper: run one mpirun/torchrun job with logging & marker ---
run_exp() {
  local opt="$1"    # e.g. adamw / sgd
  local precond="$2"         # e.g. kfac / diag_kfac
  local epochs="$3"         # e.g. 90
  shift 3

  echo "==== [${opt}_${precond}] $(date) ===="

  if mpirun \
    --mca mpi_abort_print_stack 1 \
    --report-bindings \
    "${PYTHON_ROOT}/torchrun" \
      --rdzv-backend=c10d \
      --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      --nnodes="${NNODES}" \
      --nproc-per-node="${NPROC_PER_NODE}" \
      --node-rank="${NODE_RANK}" \
      "${script_path}" \
      --timestamp="${timestamp}" \
      --experiment-name="Aug18_why_slow" \
      --model='resnet18Cifar' \
      --dataset='cifar10' \
      --epochs "${epochs}" \
      --batch-size 256 \
      --opt "${opt}" \
      --lr 0.001 \
      --weight-decay 0.001  \
      --clip-grad-norm 5.0 \
      --workers 16 \
      --preconditioner "${precond}" \
      --kfac-factor-update-steps 10 \
      --kfac-inv-update-steps 100 \
      --kfac-damping 0.003 \
      --kfac-kl-clip 0.002 \
      --amp \
      ; then
    echo "ok" > "${LOG_ROOT}/${opt}_${precond}_${epochs}.SUCCESS"
    echo "==== [${opt}_${precond}_${epochs}] DONE $(date) ===="
  else
    echo "[ERROR] ${opt}_${precond}_${epochs} FAILED" >&2
    exit 1
  fi
}

# --- Runs ---
#run_exp "adamw" "none" 100
run_exp "adamw" "kfac" 47
#run_exp "adamw" "diag_kfac" 49
#run_exp "adafisher" "none" 56