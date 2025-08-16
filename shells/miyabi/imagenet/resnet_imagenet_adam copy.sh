#!/bin/bash

#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=16:mpiprocs=1
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -m abe 

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
    --experiment-name="adamw" \
    --model='resnet50' \
    --dataset='imagenet' \
    --data-path='/work/xg24i002/share/datasets/imagenet1k' \
    --epochs 90 \
    --batch-size 512 \
    --opt adamw \
    --lr 0.001 \
    --weight-decay 0.05  \
    --norm-weight-decay 0.0 \
    --bias-weight-decay 0.0 \
    --transformer-embedding-decay 0.0 \
    --lr-scheduler onecycle \
    --lr-min 0.00001 \
    --pct-start 0.2 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.8 \
    --clip-grad-norm 8.0 \
    --cutmix-alpha 1.0 \
    --random-erase 0.25 \
    --interpolation bicubic \
    --auto-augment ta_wide \
    --workers 8 \
    --amp \
    #--model-ema \
            
