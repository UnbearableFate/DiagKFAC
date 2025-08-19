#!/bin/bash
python /home/yu/workspace/DiagKFAC/Language_Model/run_exp.py --layers 4 --batch_size 8 --weight_decay 0.1 --lr 0.0001 --log 1 --epochs 50 --optimizer AdaFisherW \
       --damping 1e-3 --gamma1 0.92 --gamma2 0.008 --device cuda \
       --embeddings "/home/yu/workspace/data/gpt1-data/embeddings.npz" \
       --data_folder "/home/yu/workspace/data/gpt1-data/"