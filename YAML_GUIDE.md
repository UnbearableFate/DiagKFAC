# YAML配置系统使用指南

## 概述

我们已经将原有的基于 `argparse` 的配置系统改造为支持 YAML 配置文件的灵活系统。这样做的好处包括：

1. **更好的组织性**：可以将相关实验组织在一起
2. **配置重用**：通过 YAML 的锚点和别名机制重用配置
3. **优化器参数兼容性**：不同优化器的特定参数都能正确处理
4. **向后兼容**：仍然支持原有的命令行参数方式

## 文件结构

```
DiagKFAC/
├── configs/                              # YAML配置文件目录
│   ├── swin_s_cifar100_experiments.yaml # Swin Transformer实验配置
│   └── comprehensive_experiments.yaml   # 全面的优化器对比配置
├── trainning_kit/
│   ├── common_yaml_config.py            # 改造后的配置解析（主要接口）
│   └── yaml_config_parser.py            # YAML解析核心逻辑
├── shells/miyabi/
│   └── swin_s_cifar100_yaml.sh         # 支持YAML的PBS脚本
├── main_yaml.py                         # 支持YAML的主训练脚本
└── yaml_experiment_parser.py            # YAML实验解析工具
```

## 使用方法

### 1. 命令行方式（保持兼容）

```bash
# 原有方式仍然可用
python main_yaml.py --model swin_s --dataset cifar100 --opt adamw --epochs 100

# 指定YAML配置文件 + 命令行覆盖
python main_yaml.py --config configs/swin_s_cifar100_experiments.yaml --experiment-name adamw_none_200
```

### 2. YAML配置文件方式

```bash
# 使用YAML配置运行特定实验
python main_yaml.py --config configs/swin_s_cifar100_experiments.yaml --experiment-name "adamw_kfac_107"

# 使用PBS脚本运行多个实验
CONFIG_FILE=configs/swin_s_cifar100_experiments.yaml EXPERIMENT_GROUP=swin_experiments qsub shells/miyabi/swin_s_cifar100_yaml.sh
```

### 3. 查看和管理实验

```bash
# 列出配置文件中的所有实验
python yaml_experiment_parser.py configs/swin_s_cifar100_experiments.yaml --list-experiments

# 列出实验组
python yaml_experiment_parser.py configs/comprehensive_experiments.yaml --list-groups

# 查看特定实验的配置
python yaml_experiment_parser.py configs/swin_s_cifar100_experiments.yaml --experiment "adamw_kfac_107"

# 生成特定实验的命令行参数
python yaml_experiment_parser.py configs/swin_s_cifar100_experiments.yaml --generate-args "adamw_kfac_107"
```

## YAML配置文件格式

### 基本格式

```yaml
# 基础配置（使用YAML锚点）
base_config: &base_config
  model: 'swin_s'
  dataset: 'cifar100'
  batch_size: 256
  lr: 0.001

# 实验列表
experiments:
  - name: "experiment_1"
    <<: *base_config  # 继承基础配置
    opt: 'adamw'
    epochs: 100
    
  - name: "experiment_2"
    <<: *base_config
    opt: 'sgd'
    epochs: 200
    momentum: 0.9
```

### 优化器特定参数

```yaml
# AdamW实验
adamw_experiment:
  opt: 'adamw'
  lr: 0.001
  weight_decay: 0.01

# SGD实验
sgd_experiment:
  opt: 'sgd'
  lr: 0.1
  momentum: 0.9
  weight_decay: 1e-4

# KFAC预条件器实验
kfac_experiment:
  preconditioner: 'kfac'
  kfac_factor_update_steps: 12
  kfac_inv_update_steps: 128
  kfac_damping: 0.003
  kfac_kl_clip: 0.001

# AdaFisher实验
adafisher_experiment:
  opt: 'adafisher'
  gamma: 0.8
  lamb: 1e-3

# Shampoo实验
shampoo_experiment:
  opt: 'shampoo'
  shampoo_damping: 1e-3
  shampoo_preconditioner_upd_interval: 10
```

## 迁移指南

### 从原有脚本迁移

原有的shell脚本：
```bash
run_exp "adamw" "kfac" 107
```

对应的YAML配置：
```yaml
- name: "adamw_kfac_107"
  model: 'swin_s'
  dataset: 'cifar100'
  opt: 'adamw'
  preconditioner: 'kfac'
  epochs: 107
  batch_size: 256
  lr: 0.001
  kfac_factor_update_steps: 12
  kfac_inv_update_steps: 128
  kfac_damping: 0.0022
  kfac_kl_clip: 0.0018
```

### 外部代码适配

如果您的代码使用了 `merged_args_parser()`，无需修改：

```python
# 这行代码保持不变，但现在支持YAML了
from trainning_kit.common_yaml_config import merged_args_parser
args = merged_args_parser()
```

## 高级功能

### 1. 配置继承和模板

```yaml
# 定义模板
optimizers:
  adamw: &adamw_config
    opt: 'adamw'
    lr: 0.001
    weight_decay: 0.01

datasets:
  cifar100: &cifar100_config
    dataset: 'cifar100'
    batch_size: 256

# 使用模板
experiments:
  - name: "my_experiment"
    <<: [*adamw_config, *cifar100_config]
    model: 'resnet18'
    epochs: 100
```

### 2. 条件配置

```yaml
# 可以根据不同条件定义不同的配置组
small_scale_experiments: &small_scale
  epochs: 50
  batch_size: 128

large_scale_experiments: &large_scale
  epochs: 200
  batch_size: 512

experiments:
  quick_test:
    - name: "quick_adamw"
      <<: *small_scale
      opt: 'adamw'
      
  full_experiments:
    - name: "full_adamw"
      <<: *large_scale
      opt: 'adamw'
```

## 最佳实践

1. **配置文件组织**：将相关实验放在同一个配置文件中
2. **使用锚点**：重复的配置使用YAML锚点和别名
3. **命名规范**：实验名称包含关键信息（优化器_预条件器_epochs）
4. **参数验证**：系统会自动验证优化器和参数的兼容性
5. **渐进迁移**：可以逐步从命令行方式迁移到YAML方式

## 故障排除

### 常见问题

1. **YAML语法错误**：使用在线YAML验证器检查语法
2. **参数不匹配**：系统会提示不兼容的优化器参数组合
3. **实验未找到**：使用 `--list-experiments` 查看可用实验
4. **路径问题**：确保配置文件路径正确

### 调试命令

```bash
# 验证YAML文件语法
python -c "import yaml; yaml.safe_load(open('configs/your_config.yaml'))"

# 查看解析后的配置
python yaml_experiment_parser.py configs/your_config.yaml --experiment your_experiment_name

# 测试生成的参数
python yaml_experiment_parser.py configs/your_config.yaml --generate-args your_experiment_name
```
