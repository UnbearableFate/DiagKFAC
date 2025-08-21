# GPT1 实验集成脚本使用指南

## 文件说明

- `run_all_experiments.sh`: 主实验脚本，整合了所有优化器实验
- `experiment_config.sh`: 配置文件，用于统一设置实验参数
- 原始脚本：`diag_kfac.sh`, `AdaFisherW.sh`, `AdaHessian.sh`, `Shampoo.sh`, `adamw.sh`

## 快速开始

### 1. 运行单个实验
```bash
# 运行 DiagKFAC 实验（推荐，已测试）
./run_all_experiments.sh diag_kfac

# 运行其他实验
./run_all_experiments.sh adafisherw
./run_all_experiments.sh adahessian
./run_all_experiments.sh shampoo
./run_all_experiments.sh adamw
```

### 2. 运行所有实验
```bash
./run_all_experiments.sh all
```

### 3. 使用配置文件
```bash
# 修改 experiment_config.sh 中的参数
# 然后加载配置并运行
source experiment_config.sh && ./run_all_experiments.sh diag_kfac
```

## 参数配置

### 通用参数（应用于所有实验）
- `LAYERS`: 模型层数（默认：4）
- `WEIGHT_DECAY`: 权重衰减（默认：0.1）
- `LOG_ENABLED`: 是否记录日志（默认：1）

### 实验特定参数
每个优化器都有独立的 batch_size、epochs 和 learning_rate 参数：

| 优化器 | Batch Size | Epochs | Learning Rate |
|--------|------------|--------|---------------|
| DiagKFAC | 8 | 10 | 0.00001 |
| AdaFisherW | 8 | 50 | 0.0001 |
| AdaHessian | 32 | 18 | 0.15 |
| Shampoo | 32 | 12 | 0.003 |
| AdamW | 32 | 55 | 0.00001 |

## 参数修改方法

### 方法1：命令行参数
```bash
# 修改通用参数
./run_all_experiments.sh diag_kfac --layers 6 --weight-decay 0.05

# 修改特定实验参数
./run_all_experiments.sh diag_kfac --diag-kfac-epochs 20 --diag-kfac-batch-size 16
```

### 方法2：环境变量
```bash
# 设置环境变量
export LAYERS=6
export WEIGHT_DECAY=0.05
export DIAG_KFAC_EPOCHS=20

# 运行实验
./run_all_experiments.sh diag_kfac
```

### 方法3：修改配置文件
1. 编辑 `experiment_config.sh`
2. 修改需要的参数
3. 运行：`source experiment_config.sh && ./run_all_experiments.sh [实验名称]`

## 查看帮助
```bash
./run_all_experiments.sh --help
```

## 注意事项

1. **DiagKFAC** 是经过测试的实验，推荐优先使用
2. 确保数据路径正确设置
3. 根据GPU内存调整batch_size
4. 实验日志会保存到默认日志目录
5. 如果CUDA不可用，脚本会自动切换到CPU

## 示例

```bash
# 快速测试（使用默认参数）
./run_all_experiments.sh diag_kfac

# 自定义参数运行
./run_all_experiments.sh diag_kfac --layers 6 --diag-kfac-epochs 20

# 批量运行所有实验
./run_all_experiments.sh all --layers 4 --weight-decay 0.1

# 使用配置文件
source experiment_config.sh && ./run_all_experiments.sh all
```
