#!/bin/bash

#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -m ae

# --- Safety & strict mode ---
set -eEuo pipefail
trap 'echo "[ERROR] Failed at line $LINENO" >&2' ERR

# --- Paths ---
timestamp=$(date "+%Y%m%d%H%M%S")
WORKSPACE="/work/xg24i002/x10041/DiagKFAC"
script_path="$WORKSPACE/main_local.py"
PYTHON_ROOT=/work/xg24i002/x10041/miniconda3/envs/diagkfac/bin
LOG_ROOT="$WORKSPACE/logs/cifar10_${timestamp}"
mkdir -p "$LOG_ROOT"

# --- Helper: run one mpirun/torchrun job with logging & marker ---
run_exp() {
  local opt="$1"    # e.g. adamw / sgd
  local precond="$2"         # e.g. kfac / diag_kfac
  local epochs="$3"         # e.g. 90
  shift 3

  echo "==== [${opt}_${precond}] $(date) ===="

  if ${PYTHON_ROOT}/python \
      "${script_path}" \
      --timestamp="${timestamp}" \
      --experiment-name="single_node" \
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
run_exp "adamw" "kfac" 2
#run_exp "adamw" "diag_kfac" 49
#run_exp "adafisher" "none" 56