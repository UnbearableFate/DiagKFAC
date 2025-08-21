#!/bin/bash

# ==============================================================================
# GPT1 实验配置文件
# 在这里修改参数，然后使用 source experiment_config.sh 加载配置
# ==============================================================================

# --- 环境配置 ---
export DATA_FOLDER="/home/yu/workspace/data/gpt1-data/"
export EMBEDDINGS="/home/yu/workspace/data/gpt1-data/embeddings.npz"
export DEVICE="cuda"

# --- 通用训练参数 ---
export LAYERS=4                    # 模型层数
export WEIGHT_DECAY=0.1            # 权重衰减
export LOG_ENABLED=1               # 是否记录日志 (0/1)

# --- DiagKFAC 实验参数 ---
export DIAG_KFAC_BATCH_SIZE=8      # 批次大小
export DIAG_KFAC_EPOCHS=10         # 训练轮数
export DIAG_KFAC_LR=0.00001        # 学习率

# --- AdaFisherW 实验参数 ---
export ADAFISHERW_BATCH_SIZE=8     # 批次大小
export ADAFISHERW_EPOCHS=50        # 训练轮数
export ADAFISHERW_LR=0.0001        # 学习率

# --- AdaHessian 实验参数 ---
export ADAHESSIAN_BATCH_SIZE=32    # 批次大小
export ADAHESSIAN_EPOCHS=18        # 训练轮数
export ADAHESSIAN_LR=0.15          # 学习率

# --- Shampoo 实验参数 ---
export SHAMPOO_BATCH_SIZE=32       # 批次大小
export SHAMPOO_EPOCHS=12           # 训练轮数
export SHAMPOO_LR=0.003            # 学习率

# --- AdamW 实验参数 ---
export ADAMW_BATCH_SIZE=32         # 批次大小
export ADAMW_EPOCHS=55             # 训练轮数
export ADAMW_LR=0.00001            # 学习率

# ==============================================================================
# 使用方法:
# 1. 修改上面的参数
# 2. 运行: source experiment_config.sh
# 3. 运行: ./run_all_experiments.sh [实验名称]
#
# 或者一次性运行:
# source experiment_config.sh && ./run_all_experiments.sh diag_kfac
# ==============================================================================

echo "实验配置已加载:"
echo "  通用参数: LAYERS=$LAYERS, WEIGHT_DECAY=$WEIGHT_DECAY"
echo "  DiagKFAC: batch_size=$DIAG_KFAC_BATCH_SIZE, epochs=$DIAG_KFAC_EPOCHS, lr=$DIAG_KFAC_LR"
echo "  设备: $DEVICE"
