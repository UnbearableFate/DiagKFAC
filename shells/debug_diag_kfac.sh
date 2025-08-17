#!/bin/bash
current_time=$(date "+%Y%m%d%H%M")

python \
 main_local.py \
 --timestamp="$current_time" \
 --experiment-name="kfac" \
 --model='resnet18Cifar' \
 --dataset='cifar100' \
 --epochs 60 \
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
 --preconditioner kfac \
 --kfac-factor-update-steps 10 \
 --kfac-inv-update-steps 100 \
 --kfac-damping 0.003 \
 --kfac-kl-clip 0.0015 \
 --amp
 #--model-ema \
 #--ra-sampler \
 #--ra-reps 4 \