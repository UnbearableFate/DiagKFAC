#!/bin/bash

#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=8:mpiprocs=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -m ae

# --- Safety & strict mode ---
set -eEuo pipefail
trap 'echo "[ERROR] Failed at line $LINENO" >&2' ERR

# --- Configuration ---
CONFIG_FILE="${CONFIG_FILE:-configs/swin_b_cifar100_multi_node.yaml}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"  # 如果为空，将运行所有实验
RUN_MODE="${RUN_MODE:-single}"  # single 或 batch

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
LOG_ROOT="$WORKSPACE/logs/${timestamp}"
mkdir -p "$LOG_ROOT"

export MASTER_PORT MASTER_ADDR

echo "==================== JOB INFO ===================="
echo "CONFIG_FILE=$CONFIG_FILE"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "RUN_MODE=$RUN_MODE"
echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE  WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  NODE_RANK=$NODE_RANK"
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
      --config "${CONFIG_FILE}" \
      --experiment-name "${exp_name}" \
      --timestamp "${timestamp}" \
      2>&1 | tee "${log_file}"; then
    
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
      # 在分布式环境中，一个实验失败就停止
      echo "[ERROR] Experiment failed, stopping batch execution" >&2
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
    echo "========================================================="
    exit 1
  fi
  
  echo "========================================================="

else
  # 单实验运行模式
  echo "Running in SINGLE mode - experiment: $EXPERIMENT_NAME"
  
  # 验证实验是否存在
  if ! get_experiments | grep -q "^${EXPERIMENT_NAME}$"; then
    echo "[ERROR] Experiment '$EXPERIMENT_NAME' not found in $CONFIG_FILE" >&2
    echo "Available experiments:"
    get_experiments | sed 's/^/  /'
    exit 1
  fi
  
  run_experiment "$EXPERIMENT_NAME"
fi

echo ""
echo "Job completed at $(date)"
echo "Logs saved to: $LOG_ROOT"
