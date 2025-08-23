#!/bin/bash

#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=05:00:00
#PBS -j oe
#PBS -m ae

# Configuration
WORKSPACE="/work/xg24i002/x10041/DiagKFAC"
PYTHON_HOME="/work/xg24i002/x10041/miniconda3/envs/diagkfac/bin"
SCRIPT="/work/xg24i002/x10041/DiagKFAC/Language_Model/run_exp.py"
EMBEDDINGS="/work/xg24i002/x10041/data/gpt1-data/embeddings.npz"
DATA_FOLDER="/work/xg24i002/x10041/data/gpt1-data/"

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
# Common parameters
COMMON_ARGS="--layers 4 --batch_size 32 --weight_decay 0.1 --clip_norm 1.0 --log 1 --log_dir $WORKSPACE/gpt-logs/$TIMESTAMP --embeddings $EMBEDDINGS --data_folder $DATA_FOLDER"

echo "Starting experiments..."

# Experiment 2: AdamW with DiagKFAC
echo "Running DiagKFAC experiment..."
$PYTHON_HOME/python $SCRIPT $COMMON_ARGS \
    --lr 0.00005 --epochs 49 --optimizer adamw \
    --preconditioner diag_kfac --curvature_update_interval 10 --preconditioner_upd_interval 100 \
    --exp_id diag_kfac

echo "Running KFAC experiment..."
# Experiment 2: AdamW with KFAC
$PYTHON_HOME/python $SCRIPT $COMMON_ARGS \
    --lr 0.00005 --epochs 44 --optimizer adamw \
    --preconditioner hd_kfac --curvature_update_interval 10 --preconditioner_upd_interval 100 \
    --exp_id kfac

# Experiment 1: AdamW baseline
echo "Running AdamW experiment..."
$PYTHON_HOME/python $SCRIPT $COMMON_ARGS \
    --lr 0.00005 --epochs 60 --optimizer adamw \
    --exp_id adamw_baseline

# Experiment 3: AdaFisherW
echo "Running AdaFisherW experiment..."
$PYTHON_HOME/python $SCRIPT $COMMON_ARGS \
    --lr 0.0001 --epochs 57 --optimizer AdaFisherW \
    --damping 1e-3 --gamma1 0.92 --gamma2 0.008 \
    --exp_id adafisherw

# Experiment 5: Shampoo
echo "Running Shampoo experiment..."
$PYTHON_HOME/python $SCRIPT $COMMON_ARGS \
    --lr 0.00005 --epochs 50 --optimizer Shampoo --damping 1e-12 --momentum 0.9 \
    --curvature_update_interval 10 --preconditioner_upd_interval 100 --ema_decay -1 \
    --exp_id shampoo

# # # Experiment 3: AdaHessian
# # echo "Running AdaHessian experiment..."
# # $PYTHON_HOME/python $SCRIPT $COMMON_ARGS \
# #     --lr 0.15 --epochs 19 --optimizer AdaHessian \
# #     --exp_id adahessian

# echo "All experiments completed!"