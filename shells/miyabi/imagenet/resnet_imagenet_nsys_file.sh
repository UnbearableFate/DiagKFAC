#!/bin/bash

#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=8:mpiprocs=1
#PBS -l walltime=01:30:00
#PBS -j oe
#PBS -m abe

# --- Safety & strict mode ---
set -eEuo pipefail
trap 'echo "[ERROR] Failed at line $LINENO" >&2' ERR

# --- Profiling (Nsight Systems) ---
# Dedicated NSYS profiling script. Customize arguments via NSYS_ARGS env var if needed.
NSYS_ARGS=${NSYS_ARGS:-"--trace=cuda,cublas,cudnn,nvtx,osrt,mpi,ucx --sample=cpu --cpuctxsw=process-tree --force-overwrite=true --mpi-impl=openmpi"}

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
script_path="$WORKSPACE/main_file.py"
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

  echo "==== [${exp_name}] $(date) ===="

  if mpirun \
    --mca mpi_abort_print_stack 1 \
    --report-bindings \
    bash -lc "\
      set -euo pipefail; \
      nsys profile ${NSYS_ARGS} \
        -o \"${LOG_ROOT}/${exp_name}_nsys_rank\${OMPI_COMM_WORLD_RANK}\" \
        ${PYTHON_ROOT}/python \
          \"${script_path}\" \
          --timestamp=\"${timestamp}\" \
          --experiment-name=\"${exp_name}\" \
          --model='resnet18Cifar' \
          --dataset='cifar10' \
          --epochs 20 \
          --opt adamw \
          --clip-grad-norm 5.0 \
          --amp \
          --preconditioner \"${precond}\" \
          --kfac-factor-update-steps 2 \
          --kfac-inv-update-steps 10 \
          --kfac-damping 0.001 \
          --kfac-kl-clip 0.001"; then
    echo "ok" > "${LOG_ROOT}/${exp_name}.SUCCESS"
    echo "==== [${exp_name}] DONE $(date) ===="
  else
    echo "[ERROR] ${exp_name} FAILED" >&2
    exit 1
  fi
}

# --- Runs ---
#run_exp "kfac" "kfac"
run_exp "diag_kfac" "diag_kfac"

# --- Summary ---
status=0
for exp in diag_kfac; do
  if [[ -f "${LOG_ROOT}/${exp}.SUCCESS" ]]; then
    echo "[SUMMARY] ${exp}: SUCCESS"
  else
    echo "[SUMMARY] ${exp}: MISSING SUCCESS MARKER" >&2
    status=1
  fi
done
exit $status