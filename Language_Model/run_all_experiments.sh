#!/bin/bash

# ==============================================================================
# GPT1 实验集成脚本
# 整合所有优化器实验：DiagKFAC、AdaFisherW、AdaHessian、Shampoo、AdamW
# ==============================================================================

set -euo pipefail

# --- 配置区域 ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EXP_SCRIPT="$SCRIPT_DIR/run_exp.py"
DATA_FOLDER="${DATA_FOLDER:-/home/yu/workspace/data/gpt1-data/}"
EMBEDDINGS="${EMBEDDINGS:-/home/yu/workspace/data/gpt1-data/embeddings.npz}"
DEVICE="${DEVICE:-cuda}"

# --- 通用训练参数 ---
# 这些参数可以统一修改以应用到所有实验
LAYERS="${LAYERS:-4}"                    # 模型层数
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"      # 权重衰减
LOG_ENABLED="${LOG_ENABLED:-1}"          # 是否记录日志

# --- 各实验特定参数 ---
# DiagKFAC 实验参数
DIAG_KFAC_BATCH_SIZE="${DIAG_KFAC_BATCH_SIZE:-8}"
DIAG_KFAC_EPOCHS="${DIAG_KFAC_EPOCHS:-10}"
DIAG_KFAC_LR="${DIAG_KFAC_LR:-0.00001}"

# AdaFisherW 实验参数  
ADAFISHERW_BATCH_SIZE="${ADAFISHERW_BATCH_SIZE:-8}"
ADAFISHERW_EPOCHS="${ADAFISHERW_EPOCHS:-50}"
ADAFISHERW_LR="${ADAFISHERW_LR:-0.0001}"

# AdaHessian 实验参数
ADAHESSIAN_BATCH_SIZE="${ADAHESSIAN_BATCH_SIZE:-32}"
ADAHESSIAN_EPOCHS="${ADAHESSIAN_EPOCHS:-18}"
ADAHESSIAN_LR="${ADAHESSIAN_LR:-0.15}"

# Shampoo 实验参数
SHAMPOO_BATCH_SIZE="${SHAMPOO_BATCH_SIZE:-32}"
SHAMPOO_EPOCHS="${SHAMPOO_EPOCHS:-12}"
SHAMPOO_LR="${SHAMPOO_LR:-0.003}"

# AdamW 实验参数
ADAMW_BATCH_SIZE="${ADAMW_BATCH_SIZE:-32}"
ADAMW_EPOCHS="${ADAMW_EPOCHS:-55}"
ADAMW_LR="${ADAMW_LR:-0.00001}"

# --- 颜色输出 ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- 帮助函数 ---
print_usage() {
    echo "用法: $0 [实验名称] [选项]"
    echo ""
    echo "可用实验:"
    echo "  1. diag_kfac    - DiagKFAC 优化器 (已测试，推荐)"
    echo "  2. adafisherw   - AdaFisherW 优化器"
    echo "  3. adahessian   - AdaHessian 优化器"
    echo "  4. shampoo      - Shampoo 优化器"
    echo "  5. adamw        - AdamW 优化器"
    echo "  6. all          - 运行所有实验"
    echo ""
    echo "通用参数 (应用于所有实验):"
    echo "  --layers N             模型层数 (默认: $LAYERS)"
    echo "  --weight-decay F       权重衰减 (默认: $WEIGHT_DECAY)"
    echo "  --log 0|1              是否记录日志 (默认: $LOG_ENABLED)"
    echo ""
    echo "各实验特定参数:"
    echo "  DiagKFAC:"
    echo "    --diag-kfac-batch-size N    (默认: $DIAG_KFAC_BATCH_SIZE)"
    echo "    --diag-kfac-epochs N        (默认: $DIAG_KFAC_EPOCHS)"
    echo "    --diag-kfac-lr F            (默认: $DIAG_KFAC_LR)"
    echo ""
    echo "  AdaFisherW:"
    echo "    --adafisherw-batch-size N   (默认: $ADAFISHERW_BATCH_SIZE)"
    echo "    --adafisherw-epochs N       (默认: $ADAFISHERW_EPOCHS)"
    echo "    --adafisherw-lr F           (默认: $ADAFISHERW_LR)"
    echo ""
    echo "  AdaHessian:"
    echo "    --adahessian-batch-size N   (默认: $ADAHESSIAN_BATCH_SIZE)"
    echo "    --adahessian-epochs N       (默认: $ADAHESSIAN_EPOCHS)"
    echo "    --adahessian-lr F           (默认: $ADAHESSIAN_LR)"
    echo ""
    echo "  Shampoo:"
    echo "    --shampoo-batch-size N      (默认: $SHAMPOO_BATCH_SIZE)"
    echo "    --shampoo-epochs N          (默认: $SHAMPOO_EPOCHS)"
    echo "    --shampoo-lr F              (默认: $SHAMPOO_LR)"
    echo ""
    echo "  AdamW:"
    echo "    --adamw-batch-size N        (默认: $ADAMW_BATCH_SIZE)"
    echo "    --adamw-epochs N            (默认: $ADAMW_EPOCHS)"
    echo "    --adamw-lr F                (默认: $ADAMW_LR)"
    echo ""
    echo "环境选项:"
    echo "  --data-folder PATH     数据文件夹路径 (默认: $DATA_FOLDER)"
    echo "  --embeddings PATH      嵌入文件路径 (默认: $EMBEDDINGS)"
    echo "  --device DEVICE        设备 (默认: $DEVICE)"
    echo "  --help, -h             显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 diag_kfac"
    echo "  $0 all --layers 6 --weight-decay 0.05"
    echo "  $0 adafisherw --diag-kfac-epochs 20 --device cpu"
    echo ""
    echo "注意: 环境变量也可以用来设置参数，例如:"
    echo "  LAYERS=6 WEIGHT_DECAY=0.05 $0 all"
}

print_header() {
    echo -e "${BLUE}==================== $1 ====================${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# --- 验证环境 ---
check_environment() {
    print_info "检查环境..."
    
    if [[ ! -f "$RUN_EXP_SCRIPT" ]]; then
        print_error "找不到 run_exp.py 脚本: $RUN_EXP_SCRIPT"
        exit 1
    fi
    
    if [[ ! -d "$DATA_FOLDER" ]]; then
        print_warning "数据文件夹不存在: $DATA_FOLDER"
        print_info "请确保设置了正确的数据路径"
    fi
    
    if [[ ! -f "$EMBEDDINGS" ]]; then
        print_warning "嵌入文件不存在: $EMBEDDINGS"
        print_info "请确保设置了正确的嵌入文件路径"
    fi
    
    if ! python -c "import torch; print('PyTorch 可用')" 2>/dev/null; then
        print_error "PyTorch 未安装或不可用"
        exit 1
    fi
    
    if [[ "$DEVICE" == "cuda" ]] && ! python -c "import torch; assert torch.cuda.is_available(); print('CUDA 可用')" 2>/dev/null; then
        print_warning "CUDA 不可用，将使用 CPU"
        DEVICE="cpu"
    fi
    
    print_success "环境检查完成"
}

# --- 实验配置 ---
run_diag_kfac() {
    print_header "运行 DiagKFAC 实验"
    python "$RUN_EXP_SCRIPT" \
        --layers "$LAYERS" \
        --batch_size "$DIAG_KFAC_BATCH_SIZE" \
        --lr "$DIAG_KFAC_LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --log "$LOG_ENABLED" \
        --epochs "$DIAG_KFAC_EPOCHS" \
        --optimizer adamw \
        --preconditioner diag_kfac \
        --curvature_update_interval 10 \
        --preconditioner_upd_interval 100 \
        --embeddings "$EMBEDDINGS" \
        --data_folder "$DATA_FOLDER" \
        --device "$DEVICE"
}

run_adafisherw() {
    print_header "运行 AdaFisherW 实验"
    python "$RUN_EXP_SCRIPT" \
        --layers "$LAYERS" \
        --batch_size "$ADAFISHERW_BATCH_SIZE" \
        --weight_decay "$WEIGHT_DECAY" \
        --lr "$ADAFISHERW_LR" \
        --log "$LOG_ENABLED" \
        --epochs "$ADAFISHERW_EPOCHS" \
        --optimizer AdaFisherW \
        --damping 1e-3 \
        --gamma1 0.92 \
        --gamma2 0.008 \
        --device "$DEVICE" \
        --embeddings "$EMBEDDINGS" \
        --data_folder "$DATA_FOLDER"
}

run_adahessian() {
    print_header "运行 AdaHessian 实验"
    python "$RUN_EXP_SCRIPT" \
        --layers "$LAYERS" \
        --batch_size "$ADAHESSIAN_BATCH_SIZE" \
        --weight_decay "$WEIGHT_DECAY" \
        --lr "$ADAHESSIAN_LR" \
        --log "$LOG_ENABLED" \
        --epochs "$ADAHESSIAN_EPOCHS" \
        --optimizer AdaHessian \
        --device "$DEVICE" \
        --embeddings "$EMBEDDINGS" \
        --data_folder "$DATA_FOLDER"
}

run_shampoo() {
    print_header "运行 Shampoo 实验"
    python "$RUN_EXP_SCRIPT" \
        --layers "$LAYERS" \
        --batch_size "$SHAMPOO_BATCH_SIZE" \
        --lr "$SHAMPOO_LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --log "$LOG_ENABLED" \
        --epochs "$SHAMPOO_EPOCHS" \
        --optimizer Shampoo \
        --damping 1e-12 \
        --momentum 0.9 \
        --curvature_update_interval 1 \
        --ema_decay -1 \
        --device "$DEVICE" \
        --embeddings "$EMBEDDINGS" \
        --data_folder "$DATA_FOLDER"
}

run_adamw() {
    print_header "运行 AdamW 实验"
    python "$RUN_EXP_SCRIPT" \
        --layers "$LAYERS" \
        --batch_size "$ADAMW_BATCH_SIZE" \
        --lr "$ADAMW_LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --log "$LOG_ENABLED" \
        --epochs "$ADAMW_EPOCHS" \
        --optimizer adamw \
        --device "$DEVICE" \
        --embeddings "$EMBEDDINGS" \
        --data_folder "$DATA_FOLDER"
}

# --- 运行单个实验 ---
run_experiment() {
    local experiment="$1"
    local start_time=$(date +%s)
    
    print_info "开始运行实验: $experiment"
    print_info "开始时间: $(date)"
    
    case "$experiment" in
        "diag_kfac"|"1")
            run_diag_kfac
            ;;
        "adafisherw"|"2")
            run_adafisherw
            ;;
        "adahessian"|"3")
            run_adahessian
            ;;
        "shampoo"|"4")
            run_shampoo
            ;;
        "adamw"|"5")
            run_adamw
            ;;
        *)
            print_error "未知实验: $experiment"
            print_usage
            exit 1
            ;;
    esac
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ $? -eq 0 ]]; then
        print_success "实验 $experiment 完成! 用时: ${duration}秒"
    else
        print_error "实验 $experiment 失败!"
        exit 1
    fi
}

# --- 运行所有实验 ---
run_all_experiments() {
    print_header "运行所有实验"
    
    local experiments=("diag_kfac" "adafisherw" "adahessian" "shampoo" "adamw")
    local successful=()
    local failed=()
    local total_start_time=$(date +%s)
    
    for experiment in "${experiments[@]}"; do
        print_info "运行实验 $((${#successful[@]} + ${#failed[@]} + 1))/${#experiments[@]}: $experiment"
        
        if run_experiment "$experiment"; then
            successful+=("$experiment")
        else
            failed+=("$experiment")
            print_warning "实验 $experiment 失败，继续运行下一个实验..."
        fi
        
        echo ""
    done
    
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - total_start_time))
    
    # 打印总结
    print_header "实验总结"
    print_info "总耗时: ${total_duration}秒"
    print_info "成功: ${#successful[@]}/${#experiments[@]}"
    print_info "失败: ${#failed[@]}/${#experiments[@]}"
    
    if [[ ${#successful[@]} -gt 0 ]]; then
        echo ""
        print_success "成功的实验:"
        for exp in "${successful[@]}"; do
            echo "  ✓ $exp"
        done
    fi
    
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo ""
        print_error "失败的实验:"
        for exp in "${failed[@]}"; do
            echo "  ✗ $exp"
        done
    fi
}

# --- 参数解析 ---
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            # 环境参数
            --data-folder)
                DATA_FOLDER="$2"
                shift 2
                ;;
            --embeddings)
                EMBEDDINGS="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            # 通用参数
            --layers)
                LAYERS="$2"
                shift 2
                ;;
            --weight-decay)
                WEIGHT_DECAY="$2"
                shift 2
                ;;
            --log)
                LOG_ENABLED="$2"
                shift 2
                ;;
            # DiagKFAC 参数
            --diag-kfac-batch-size)
                DIAG_KFAC_BATCH_SIZE="$2"
                shift 2
                ;;
            --diag-kfac-epochs)
                DIAG_KFAC_EPOCHS="$2"
                shift 2
                ;;
            --diag-kfac-lr)
                DIAG_KFAC_LR="$2"
                shift 2
                ;;
            # AdaFisherW 参数
            --adafisherw-batch-size)
                ADAFISHERW_BATCH_SIZE="$2"
                shift 2
                ;;
            --adafisherw-epochs)
                ADAFISHERW_EPOCHS="$2"
                shift 2
                ;;
            --adafisherw-lr)
                ADAFISHERW_LR="$2"
                shift 2
                ;;
            # AdaHessian 参数
            --adahessian-batch-size)
                ADAHESSIAN_BATCH_SIZE="$2"
                shift 2
                ;;
            --adahessian-epochs)
                ADAHESSIAN_EPOCHS="$2"
                shift 2
                ;;
            --adahessian-lr)
                ADAHESSIAN_LR="$2"
                shift 2
                ;;
            # Shampoo 参数
            --shampoo-batch-size)
                SHAMPOO_BATCH_SIZE="$2"
                shift 2
                ;;
            --shampoo-epochs)
                SHAMPOO_EPOCHS="$2"
                shift 2
                ;;
            --shampoo-lr)
                SHAMPOO_LR="$2"
                shift 2
                ;;
            # AdamW 参数
            --adamw-batch-size)
                ADAMW_BATCH_SIZE="$2"
                shift 2
                ;;
            --adamw-epochs)
                ADAMW_EPOCHS="$2"
                shift 2
                ;;
            --adamw-lr)
                ADAMW_LR="$2"
                shift 2
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            -*)
                print_error "未知选项: $1"
                print_usage
                exit 1
                ;;
            *)
                EXPERIMENT="$1"
                shift
                ;;
        esac
    done
}

# --- 主函数 ---
main() {
    local EXPERIMENT=""
    
    # 解析参数
    parse_arguments "$@"
    
    # 如果没有指定实验，显示帮助
    if [[ -z "$EXPERIMENT" ]]; then
        print_usage
        exit 1
    fi
    
    # 检查环境
    check_environment
    
    echo ""
    print_info "配置信息:"
    print_info "  数据文件夹: $DATA_FOLDER"
    print_info "  嵌入文件: $EMBEDDINGS"
    print_info "  设备: $DEVICE"
    print_info ""
    print_info "通用训练参数:"
    print_info "  层数: $LAYERS"
    print_info "  权重衰减: $WEIGHT_DECAY"
    print_info "  记录日志: $LOG_ENABLED"
    print_info ""
    print_info "各实验参数:"
    print_info "  DiagKFAC: batch_size=$DIAG_KFAC_BATCH_SIZE, epochs=$DIAG_KFAC_EPOCHS, lr=$DIAG_KFAC_LR"
    print_info "  AdaFisherW: batch_size=$ADAFISHERW_BATCH_SIZE, epochs=$ADAFISHERW_EPOCHS, lr=$ADAFISHERW_LR"
    print_info "  AdaHessian: batch_size=$ADAHESSIAN_BATCH_SIZE, epochs=$ADAHESSIAN_EPOCHS, lr=$ADAHESSIAN_LR"
    print_info "  Shampoo: batch_size=$SHAMPOO_BATCH_SIZE, epochs=$SHAMPOO_EPOCHS, lr=$SHAMPOO_LR"
    print_info "  AdamW: batch_size=$ADAMW_BATCH_SIZE, epochs=$ADAMW_EPOCHS, lr=$ADAMW_LR"
    echo ""
    
    # 运行实验
    if [[ "$EXPERIMENT" == "all" ]]; then
        run_all_experiments
    else
        run_experiment "$EXPERIMENT"
    fi
    
    print_success "所有任务完成!"
}

# --- 脚本入口 ---
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
