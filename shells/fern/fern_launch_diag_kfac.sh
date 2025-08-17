#!/bin/bash
current_time=$(date "+%Y%m%d%H%M")

export MASTER_ADDR=fern02
export MASTER_PORT=29400
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export NCCL_P2P_DISABLE=1

mpirun --host fern02,fern01 \
 -np 2 -map-by ppr:1:node \
 -x MASTER_ADDR -x MASTER_PORT \
 /home/yu/miniconda3/envs/py313/bin/python \
 main_mpi.py \
 --timestamp="$current_time" \
 --experiment-name="diag_kfac_test_ddp" \
 --model='resnet18Cifar' \
 --dataset='cifar10' \
 --epochs 10 \
 --batch-size 256 \
 --opt adamw \
 --lr 0.001 \
 --weight-decay 0.001  \
 --clip-grad-norm 2.0 \
 --workers 8 \
 --preconditioner diag_kfac \
 --kfac-factor-update-steps 10 \
 --kfac-inv-update-steps 100 \
 --kfac-damping 0.003 \
 --kfac-kl-clip 0.008 \
 --amp