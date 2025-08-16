#!/bin/bash
current_time=$(date "+%Y%m%d%H%M")

export MASTER_ADDR=fern02
export MASTER_PORT=29400

mpirun --host fern02,fern01 \
 -np 2 -map-by ppr:1:node \
 -x MASTER_ADDR -x MASTER_PORT \
 /home/yu/miniconda3/envs/py313/bin/python \
 main.py \
 --timestamp="$current_time" \
 --experiment-name="adam_baseline" \
 --model='resnet' \
 --layers=18 \
 --dataset='cifar10' \
 --epochs 100 \
 --batch-size 256 \
 --opt adamw \
 --lr 0.001 \
 --weight-decay 0.001  \
 --norm-weight-decay 0.0 \
 --bias-weight-decay 0.0 \
 --transformer-embedding-decay 0.0 \
 --lr-scheduler onecycle \
 --lr-min 0.00001 \
 --pct-start 0.2 \
 --label-smoothing 0.1 \
 --mixup-alpha 0.8 \
 --clip-grad-norm 2.0 \
 --cutmix-alpha 1.0 \
 --random-erase 0.25 \
 --interpolation bicubic \
 --auto-augment ta_wide \
 --val-resize-size 224 \
 --workers 8 \
 --amp \
 #--model-ema \