#!/bin/bash

#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=4:mpiprocs=1
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -m abe

# --- Safety & strict mode ---
set -euo pipefail

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
LOG_ROOT="$WORKSPACE/logs/imagenet_${timestamp}"
mkdir -p "$LOG_ROOT"

export MASTER_PORT MASTER_ADDR

echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE  WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  NODE_RANK=$NODE_RANK"
echo "LOG_ROOT=$LOG_ROOT"

# --- Helper: run one mpirun/torchrun job with logging & marker ---
run_exp() {
  local exp_name="$1"        # e.g. kfac / diag_kfac
  local precond="$2"         # e.g. kfac / diag_kfac
  shift 2
  local extra_args=("$@")    # any extra args

  echo "==== [${exp_name}] $(date) ===="

  mpirun \
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
      --experiment-name="${exp_name}" \
      --model='resnet50' \
      --dataset='imagenet' \
      --epochs 10 \
      --opt adamw \
      --clip-grad-norm 5.0 \
      --amp \
      --preconditioner "${precond}" \
      --kfac-factor-update-steps 10 \
      --kfac-inv-update-steps 100 \
      --kfac-damping 0.001 \
      --kfac-kl-clip 0.001 \
      "${extra_args[@]}"

  # If mpirun returned 0, drop a success marker for this run
  echo "ok" > "${LOG_ROOT}/${exp_name}.SUCCESS"
  echo "==== [${exp_name}] DONE $(date) ===="
}

# --- Runs ---
run_exp "kfac" "kfac"
run_exp "diag_kfac" "diag_kfac"

# --- Summary ---
status=0
for exp in kfac diag_kfac; do
  if [[ -f "${LOG_ROOT}/${exp}.SUCCESS" ]]; then
    echo "[SUMMARY] ${exp}: SUCCESS"
  else
    echo "[SUMMARY] ${exp}: MISSING SUCCESS MARKER" >&2
    status=1
  fi
done
exit $status