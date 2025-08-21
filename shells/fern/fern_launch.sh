#!/bin/bash
current_time=$(date "+%Y%m%d%H%M")

export MASTER_ADDR=fern02
export MASTER_PORT=29400

mpirun --host fern02,fern01 \
 -np 2 -map-by ppr:1:node \
 -x MASTER_ADDR -x MASTER_PORT \
 /home/yu/miniconda3/envs/py313/bin/python \
 main_mpi.py \
 --config configs/resnet18_cifar10_single_node.yaml \
 --experiment-name "resnet18_adamw_test"