#!/bin/bash

#PBS -q debug-g
#PBS -W group_list=xg24i002
#PBS -l select=4:mpiprocs=1
#PBS -l walltime=00:10:00
#PBS -j oe

PYTHON_ROOT=/work/xg24i002/x10041/miniconda3/envs/diagkfac/bin
timestamp=$(date "+%Y%m%d%H%M%S")
WORKSPACE="/work/xg24i002/x10041/DiagKFAC"
script_path="$WORKSPACE/main_file.py"

mpirun --mca mpi_abort_print_stack 1 \
 --report-bindings \
  ${PYTHON_ROOT}/python \
    $script_path \
    --timestamp="$timestamp" \
    --experiment-name="adam_baseline" \
    --model='resnet18Cifar' \
    --dataset='cifar10' \
    --epochs 10 \
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
            
