#!/bin/bash
current_time=$(date "+%Y%m%d%H%M")
CUDA_LAUNCH_BLOCKING=1 TORCH_SHOW_CPP_STACKTRACES=1 python \
 main_local.py \
 --config configs/focalnet_cifar100_single_node.yaml \
 --experiment-name "focalnet_adamw_test" \
 --timestamp $current_time