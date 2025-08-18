#!/bin/bash

#PBS -q debug-g
#PBS -W group_list=xg24i002
#PBS -l select=8:mpiprocs=1
#PBS -l walltime=00:30:00
#PBS -j oe

NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)
NPROC_PER_NODE=1
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# master address and port
MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
MASTER_PORT=20201

# compute local node-rank
hosts=( $(sort -u "$PBS_NODEFILE") )
NODE_RANK=0
for idx in "${!hosts[@]}"; do
  if [[ "${hosts[idx]}" == "$(hostname)" ]]; then
    NODE_RANK=$idx
    break
  fi
done

echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE  WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  NODE_RANK=$NODE_RANK"

timestamp=$(date "+%Y%m%d%H%M%S")

WORKSPACE="/work/xg24i002/x10041/DiagKFAC"
script_path="$WORKSPACE/main.py"

PYTHON_ROOT=/work/xg24i002/x10041/miniconda3/envs/diagkfac/bin

export MASTER_PORT
export MASTER_ADDR

mpirun --mca mpi_abort_print_stack 1 \
 --report-bindings \
${PYTHON_ROOT}/torchrun \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR":"$MASTER_PORT" \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  $script_path \
  --timestamp="$timestamp" \
  --experiment-name="diag_kfac" \
  --model='resnet18Cifar' \
  --dataset='cifar10' \
  --epochs 45 \
  --batch-size 256 \
  --opt adamw \
  --lr 0.001 \
  --weight-decay 0.001  \
  --norm-weight-decay 0.0 \
  --bias-weight-decay 0.0 \
  --transformer-embedding-decay 0.0 \
  --lr-scheduler onecycle \
  --pct-start 0.2 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.8 \
  --clip-grad-norm 5.0 \
  --cutmix-alpha 1.0 \
  --random-erase 0.25 \
  --interpolation bicubic \
  --auto-augment ta_wide \
  --val-resize-size 224 \
  --workers 8 \
  --preconditioner diag_kfac \
  --kfac-factor-update-steps 10 \
  --kfac-inv-update-steps 100 \
  --kfac-damping 0.003 \
  --kfac-kl-clip 0.002 \
  --amp \