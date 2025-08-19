#!/bin/bash
python /home/yu/workspace/DiagKFAC/Language_Model/run_exp.py --layers 4 --batch_size 8 --lr 0.00001 --weight_decay 0.1 --log 1 --epochs 10 --optimizer adamw \
       --preconditioner diag_kfac --curvature_update_interval 10 --preconditioner_upd_interval 100 \
       --embeddings "/home/yu/workspace/data/gpt1-data/embeddings.npz" \
       --data_folder "/home/yu/workspace/data/gpt1-data/"