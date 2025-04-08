#!/bin/bash
current_time=$(date "+%Y%m%d%H%M")

python main.py \
 --timestamp="$current_time" \
 --experiment-name="swift_resnet" \
 --model='resnet' \
 --layers=18 \
 --dataset='cifar10' \
 --epochs 10 \
 --batch-size 256 \
 --opt adamw \
 --lr 0.002 \
 --weight-decay 0.001  \
 --norm-weight-decay 0.0 \
 --bias-weight-decay 0.0 \
 --transformer-embedding-decay 0.0 \
 --lr-scheduler cosineannealinglr \
 --lr-min 0.00001 \
 --lr-warmup-method linear \
 --lr-warmup-epochs 20 \
 --lr-warmup-decay 0.01 \
 --amp \
 --label-smoothing 0.1 \
 --mixup-alpha 0.8 \
 --clip-grad-norm 5.0 \
 --cutmix-alpha 1.0 \
 --random-erase 0.25 \
 --interpolation bicubic \
 --auto-augment ta_wide \
 --model-ema \
 --ra-sampler \
 --ra-reps 4 \
 --val-resize-size 224 \