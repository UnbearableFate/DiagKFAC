#!/bin/bash

#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -m ae

# --- Safety & strict mode ---
set -eEuo pipefail
trap 'echo "[ERROR] Failed at line $LINENO" >&2' ERR

# --- Configuration ---
CONFIG_FILE="${CONFIG_FILE:-configs/resnet18_cifar10_single_node.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"  # 如果为空，将运行所有实验
RUN_MODE="${RUN_MODE:-single}"  # single 或 batch

# --- Single Node Configuration ---
# No distributed training configuration needed for direct Python execution

# --- Paths ---
timestamp=$(date "+%Y%m%d%H%M%S")
WORKSPACE="/home/yu/workspace/DiagKFAC"
script_path="$WORKSPACE/main_local.py"
PYTHON_ROOT=/home/yu/miniconda3/envs/py313/bin
LOG_ROOT="$WORKSPACE/logs/${timestamp}"
mkdir -p "$LOG_ROOT"

echo "==================== JOB INFO ===================="
echo "CONFIG_FILE=$CONFIG_FILE"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "RUN_MODE=$RUN_MODE"
echo "SINGLE NODE TRAINING (Direct Python execution)"
echo "LOG_ROOT=$LOG_ROOT"
echo "TIMESTAMP=$timestamp"
echo "=================================================="

# --- Helper: run one experiment with YAML config ---
run_experiment() {
  local exp_name="$1"
  
  echo ""
  echo "==== [${exp_name}] $(date) ===="
  
  local log_file="${LOG_ROOT}/${exp_name}.log"
  local success_marker="${LOG_ROOT}/${exp_name}.SUCCESS"
  local error_marker="${LOG_ROOT}/${exp_name}.ERROR"
  
  # Direct Python execution without torchrun
  if "${PYTHON_ROOT}/python" \
      "${script_path}" \
      --config "${CONFIG_FILE}" \
      --experiment-name "${exp_name}" \
      --timestamp "${timestamp}" \
      ; then
    
    echo "SUCCESS" > "${success_marker}"
    echo "==== [${exp_name}] COMPLETED $(date) ===="
    return 0
  else
    echo "FAILED" > "${error_marker}"
    echo "[ERROR] ${exp_name} FAILED" >&2
    return 1
  fi
}

# --- Helper: get experiment list from YAML ---
get_experiments() {
  cd "$WORKSPACE"
  "${PYTHON_ROOT}/python" yaml_experiment_parser.py "$CONFIG_FILE" --list-experiments | grep "^  - " | sed 's/^  - //'
}

# --- Main execution logic ---
cd "$WORKSPACE"

if [[ "$RUN_MODE" == "batch" ]] || [[ -z "$EXPERIMENT_NAME" ]]; then
  # 批量运行模式：运行所有实验
  echo "Running in BATCH mode - will run all experiments in config file"
  
  # 获取实验列表
  readarray -t experiments < <(get_experiments)
  
  if [[ ${#experiments[@]} -eq 0 ]]; then
    echo "[ERROR] No experiments found in $CONFIG_FILE" >&2
    exit 1
  fi
  
  echo "Found ${#experiments[@]} experiments:"
  printf '  - %s\n' "${experiments[@]}"
  echo ""
  
  # 运行统计
  total_experiments=${#experiments[@]}
  successful_experiments=()
  failed_experiments=()
  
  # 依次运行每个实验
  for exp_name in "${experiments[@]}"; do
    echo "Running experiment $((${#successful_experiments[@]} + ${#failed_experiments[@]} + 1))/$total_experiments: $exp_name"
    
    if run_experiment "$exp_name"; then
      successful_experiments+=("$exp_name")
    else
      failed_experiments+=("$exp_name")
      # 在单个实验失败时立即停止
      echo "[ERROR] Experiment '$exp_name' failed, stopping batch execution"
      break
    fi
  done
  
  # 打印总结
  echo ""
  echo "==================== BATCH SUMMARY ===================="
  echo "Total experiments: $total_experiments"
  echo "Successful: ${#successful_experiments[@]}"
  echo "Failed: ${#failed_experiments[@]}"
  
  if [[ ${#successful_experiments[@]} -gt 0 ]]; then
    echo ""
    echo "Successful experiments:"
    printf '  ✓ %s\n' "${successful_experiments[@]}"
  fi
  
  if [[ ${#failed_experiments[@]} -gt 0 ]]; then
    echo ""
    echo "Failed experiments:"
    printf '  ✗ %s\n' "${failed_experiments[@]}"
  fi
  
  echo "========================================================="
  
  # 如果有任何实验失败，则退出失败
  if [[ ${#failed_experiments[@]} -gt 0 ]]; then
    echo "========================================================="
    exit 1
  fi

else
  # 单实验运行模式
  echo "Running in SINGLE mode - experiment: $EXPERIMENT_NAME"
  
  # 验证实验是否存在
  if ! get_experiments | grep -q "^${EXPERIMENT_NAME}$"; then
    echo "[ERROR] Experiment '$EXPERIMENT_NAME' not found in $CONFIG_FILE"
    echo "Available experiments:"
    get_experiments | sed 's/^/  /'
    exit 1
  fi
  
  run_experiment "$EXPERIMENT_NAME"
fi

echo ""
echo "Job completed at $(date)"
echo "Logs saved to: $LOG_ROOT"
