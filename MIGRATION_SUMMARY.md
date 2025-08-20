# DiagKFAC 配置系统改造总结

## 改造概述

我们成功将原有的基于 `argparse.ArgumentParser` 的配置系统改造为支持 YAML 配置文件的灵活系统，同时保持了向后兼容性。

## 主要改进

### 1. 更好的参数组织和重用
**之前**：每个实验都需要在shell脚本中硬编码所有参数
```bash
run_exp "adamw" "kfac" 107
# 需要在函数内部硬编码所有其他参数
```

**现在**：使用YAML配置文件，支持配置继承和重用
```yaml
base_config: &base_config
  model: 'swin_s'
  dataset: 'cifar100'
  batch_size: 256
  # ... 其他通用参数

experiments:
  - name: "adamw_kfac_107"
    <<: *base_config
    opt: 'adamw'
    preconditioner: 'kfac'
    epochs: 107
    # 只需要指定特定参数
```

### 2. 优化器参数兼容性
**之前**：所有优化器使用相同的参数结构，容易出错
```bash
# 所有实验都使用相同的参数，不管优化器是否支持
--lr 0.001 --weight-decay 0.001 --clip-grad-norm 5.0
```

**现在**：每个优化器可以有专门的参数配置
```yaml
# AdaFisher 特定参数
adafisher_config:
  opt: 'adafisher'
  gamma: 0.8
  lamb: 1e-3

# KFAC 特定参数  
kfac_config:
  preconditioner: 'kfac'
  kfac_factor_update_steps: 12
  kfac_inv_update_steps: 128
  kfac_damping: 0.003
```

### 3. 更灵活的实验管理
**之前**：修改实验需要编辑shell脚本
```bash
# 需要修改shell脚本来添加新实验
run_exp "adamw" "none" 200
run_exp "adamw" "kfac" 107
run_exp "adafisherw" "none" 137
```

**现在**：通过YAML文件轻松管理实验
```bash
# 列出所有可用实验
python yaml_experiment_parser.py configs/swin_s_cifar100_experiments.yaml --list-experiments

# 运行特定实验
./run_single_experiment.sh configs/swin_s_cifar100_experiments.yaml adamw_kfac_107

# 批量运行实验组
CONFIG_FILE=configs/swin_s_cifar100_experiments.yaml EXPERIMENT_GROUP=swin_experiments qsub shells/miyabi/swin_s_cifar100_yaml.sh
```

## 文件对比

### 原有结构
```
shells/miyabi/swin_s_cifar100_new.sh    # 硬编码的实验脚本
trainning_kit/common.py                 # 基于argparse的配置
```

### 新的结构
```
configs/                               # 新增：YAML配置目录
├── swin_s_cifar100_experiments.yaml  # 对应原脚本的实验配置
└── comprehensive_experiments.yaml    # 扩展的实验配置

trainning_kit/
├── common_yaml_config.py            # 改造：支持YAML的配置解析
└── yaml_config_parser.py            # 新增：YAML解析核心逻辑

shells/miyabi/
└── swin_s_cifar100_yaml.sh         # 新增：支持YAML的PBS脚本

yaml_experiment_parser.py             # 新增：YAML实验管理工具
run_single_experiment.sh              # 新增：单实验运行脚本
main_yaml.py                          # 新增：支持YAML的主脚本（可选）
```

## 向后兼容性

### 原有代码无需修改
```python
# 这行代码仍然有效，但现在支持YAML配置
from trainning_kit.common_yaml_config import merged_args_parser
args = merged_args_parser()
```

### 原有命令行方式仍然可用
```bash
# 传统方式
python main.py --model swin_s --dataset cifar100 --opt adamw --epochs 100

# 新的YAML方式
python main.py --config configs/my_config.yaml --experiment-name my_experiment
```

## 使用示例

### 1. 快速开始
```bash
# 查看可用实验
python yaml_experiment_parser.py configs/swin_s_cifar100_experiments.yaml --list-experiments

# 运行单个实验
./run_single_experiment.sh configs/swin_s_cifar100_experiments.yaml adamw_none_200
```

### 2. 批量实验
```bash
# 使用PBS脚本运行多个实验
export CONFIG_FILE=configs/swin_s_cifar100_experiments.yaml
export EXPERIMENT_GROUP=swin_experiments
qsub shells/miyabi/swin_s_cifar100_yaml.sh
```

### 3. 创建新实验
```yaml
# 在YAML文件中添加新实验
- name: "my_new_experiment"
  <<: *base_config
  opt: 'adamw'
  preconditioner: 'diag_kfac'
  epochs: 150
  lr: 0.002
```

## 主要优势

1. **可维护性**：配置和代码分离，易于维护
2. **可扩展性**：轻松添加新的优化器和参数
3. **可重用性**：通过YAML锚点重用配置
4. **可读性**：配置文件比shell脚本更易读
5. **类型安全**：自动验证参数类型和兼容性
6. **向后兼容**：现有代码无需修改

## 迁移建议

1. **立即可用**：现有代码无需修改即可使用
2. **逐步迁移**：可以逐个实验迁移到YAML配置
3. **混合使用**：可以同时使用YAML配置和命令行参数
4. **测试验证**：使用 `run_single_experiment.sh` 测试新配置

这个改造为您的实验管理提供了更大的灵活性，同时保持了系统的简单性和兼容性。
