#!/bin/bash
current_time=$(date "+%Y%m%d%H%M")

/home/yu/miniconda3/envs/py313/bin/python \
 main_local.py \
 --config configs/cct_cifar100_single_node.yaml \
 --experiment-name "cct_diag_kfac_test" \
 --timestamp $current_time