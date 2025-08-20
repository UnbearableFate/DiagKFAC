import os
import yaml
import datetime
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass


PROJECT_NAME = "DiagKFAC"

PRIVATE_DATA_ROOT = { # for small datasets like cifar10
    'fern' : "/home/yu/data",
    'mac' : "/Users/unbearablefate/workspace/data",
    'pegasus' : "/work/NBB/yu_mingzhe/data",
    'miyabi' : "/work/xg24i002/x10041/data",
}

WORKSPACE_ROOT = {
    'fern' : "/home/yu/workspace/" + PROJECT_NAME,
    'mac' : "/Users/unbearablefate/workspace/" + PROJECT_NAME,
    'pegasus' : "/work/NBB/yu_mingzhe/" + PROJECT_NAME,
    'miyabi' : "/work/xg24i002/x10041/" + PROJECT_NAME,
}


@dataclass
class DefaultConfig:
    """定义所有默认配置"""
    # Basic training parameters
    data_path: Optional[str] = None
    model: Optional[str] = None
    device: str = "cuda"
    batch_size: int = 256
    epochs: int = 90
    workers: int = 16
    
    # Optimizer parameters
    opt: str = "sgd"
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.03
    norm_weight_decay: Optional[float] = None
    bias_weight_decay: Optional[float] = None
    transformer_embedding_decay: Optional[float] = None
    
    # Training configuration
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    
    # Learning rate scheduler
    lr_scheduler: str = "onecycle"
    lr_warmup_epochs: int = 0
    lr_warmup_method: str = "constant"
    lr_warmup_decay: float = 0.01
    lr_step_size: int = 30
    lr_gamma: float = 0.1
    pct_start: float = 0.2
    lr_min: float = 1e-5
    
    # I/O and logging
    print_freq: int = 100
    output_dir: str = "./out"
    resume: str = ""
    start_epoch: int = 0
    log_dir: str = "./logs/torch_cifar10"
    checkpoint_format: str = "checkpoint_{epoch}.pth.tar"
    checkpoint_freq: int = 10
    
    # Training features
    cache_dataset: bool = False
    sync_bn: bool = False
    test_only: bool = False
    auto_augment: str = "ta_wide"
    ra_magnitude: int = 9
    augmix_severity: int = 3
    random_erase: float = 0.25
    amp: bool = False
    
    # Distributed training
    world_size: int = 1
    dist_url: str = "env://"
    local_rank: int = 0
    
    # Model EMA
    model_ema: bool = False
    model_ema_steps: int = 32
    model_ema_decay: float = 0.99998
    
    # Misc
    use_deterministic_algorithms: bool = False
    interpolation: str = "bicubic"
    val_resize_size: int = 256
    val_crop_size: int = 224
    train_crop_size: int = 224
    clip_grad_norm: Optional[float] = None
    ra_sampler: bool = False
    ra_reps: int = 3
    weights: Optional[str] = None
    backend: str = "PIL"
    use_v2: bool = False
    
    # Additional parameters
    no_cuda: bool = False
    seed: int = 42
    val_batch_size: int = 256
    batches_per_allreduce: int = 1
    
    # KFAC parameters
    kfac_inv_update_steps: int = 10
    kfac_factor_update_steps: int = 1
    kfac_update_steps_alpha: float = 10
    kfac_update_steps_decay: Optional[list] = None
    kfac_inv_method: bool = False
    kfac_factor_decay: float = 0.95
    kfac_damping: float = 0.003
    kfac_damping_alpha: float = 0.5
    kfac_damping_decay: Optional[list] = None
    kfac_kl_clip: float = 0.001
    kfac_skip_layers: list = None
    kfac_colocate_factors: bool = True
    kfac_strategy: str = "comm-opt"
    kfac_grad_worker_fraction: float = 0.25
    
    # AdaFisher parameters
    gamma: float = 0.8
    lamb: float = 1e-3
    
    # Shampoo parameters
    shampoo_damping: float = 1e-3
    shampoo_preconditioner_upd_interval: int = 10
    shampoo_curvature_update_interval: int = 10
    shampoo_ema_decay: float = -1
    
    # Experiment parameters
    timestamp: str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    unified_experiment_name: str = "single_node_test"
    recover: bool = False
    preconditioner: str = "none"
    degree_noniid: float = 0
    dataset: str = "cifar10"
    backpack: bool = False
    
    def __post_init__(self):
        if self.kfac_skip_layers is None:
            self.kfac_skip_layers = []


class YAMLConfigParser:
    """YAML配置文件解析器"""
    
    def __init__(self):
        self.default_config = DefaultConfig()
        
    def load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        return config or {}
    
    def merge_config(self, yaml_config: Dict[str, Any], cli_args: Optional[argparse.Namespace] = None) -> argparse.Namespace:
        """合并YAML配置和命令行参数"""
        # 从默认配置开始
        final_config = self.default_config.__dict__.copy()
        
        # 应用YAML配置
        if yaml_config:
            final_config.update(yaml_config)
        
        # 如果有命令行参数，优先使用命令行参数（除了None值）
        if cli_args:
            for key, value in cli_args.__dict__.items():
                if value is not None and hasattr(cli_args, key):
                    final_config[key] = value
        
        # 创建argparse.Namespace对象
        args = argparse.Namespace()
        for key, value in final_config.items():
            setattr(args, key, value)
            
        return args
    
    def validate_config(self, args: argparse.Namespace) -> argparse.Namespace:
        """验证和后处理配置"""
        # 验证优化器选择
        valid_optimizers = ["sgd", "adamw", "adafisher", "adafisherw", "adafactor", "adahessian", "shampoo"]
        if args.opt not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {args.opt}. Must be one of {valid_optimizers}")
        
        # 验证数据集选择
        valid_datasets = ["imagenet", "cifar10", "cifar100", "fashionmnist"]
        if args.dataset not in valid_datasets:
            raise ValueError(f"Invalid dataset: {args.dataset}. Must be one of {valid_datasets}")
        
        # 验证预条件器选择
        valid_preconditioners = ["kfac", "diag_kfac", "none"]
        if args.preconditioner not in valid_preconditioners:
            raise ValueError(f"Invalid preconditioner: {args.preconditioner}. Must be one of {valid_preconditioners}")
        
        return args
    
    def auto_configure_paths(self, args: argparse.Namespace) -> argparse.Namespace:
        """自动配置路径"""
        # 检测当前系统
        current_system = None
        for system, workspace_root in WORKSPACE_ROOT.items():
            if os.path.exists(workspace_root):
                current_system = system
                break
        
        if current_system is None:
            raise ValueError("No valid system found in WORKSPACE_ROOT. Please set the correct paths.")
        
        # 处理环境变量中的 LOCAL_RANK
        if "LOCAL_RANK" in os.environ:
            args.local_rank = int(os.environ["LOCAL_RANK"])
        
        # 自动设置数据路径
        if args.data_path is None and args.dataset in ["cifar10", "fashionmnist", "minist", "cifar100"]:
            if current_system in PRIVATE_DATA_ROOT:
                args.data_path = os.path.join(PRIVATE_DATA_ROOT[current_system], args.dataset)
        
        if args.dataset == "imagenet":
            args.data_path = "/work/xg24i002/share/datasets/imagenet1k"
        
        # 设置工作空间路径
        args.workspace_path = WORKSPACE_ROOT[current_system]
        
        # 切换到工作目录
        os.chdir(args.workspace_path)
        
        return args


def create_minimal_cli_parser() -> argparse.ArgumentParser:
    """创建最小的命令行参数解析器，主要用于指定配置文件"""
    parser = argparse.ArgumentParser(description="YAML-based Training Configuration", add_help=True)
    
    # 核心参数：配置文件路径
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    
    # 允许覆盖的关键参数
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--dataset", type=str, help="Override dataset name")
    parser.add_argument("--opt", type=str, help="Override optimizer")
    parser.add_argument("--preconditioner", type=str, help="Override preconditioner")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, dest="batch_size", help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--timestamp", type=str, help="Override timestamp")
    parser.add_argument("--experiment-name", type=str, dest="experiment_name", help="Override experiment name")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--test-only", action="store_true", dest="test_only", help="Only test the model")
    
    return parser


def merged_args_parser(add_help=True) -> argparse.Namespace:
    """
    主要的配置解析函数，兼容原有接口
    优先级: 命令行参数 > YAML配置 > 默认值
    """
    cli_parser = create_minimal_cli_parser()
    cli_args = cli_parser.parse_args()
    
    yaml_parser = YAMLConfigParser()
    
    # 如果指定了配置文件，加载它
    yaml_config = {}
    if cli_args.config:
        full_yaml_config = yaml_parser.load_yaml_config(cli_args.config)
        
        # 如果指定了实验名称，从experiments列表中提取特定实验
        if cli_args.experiment_name and 'experiments' in full_yaml_config:
            experiment_found = False
            for exp in full_yaml_config['experiments']:
                if exp.get('name') == cli_args.experiment_name:
                    yaml_config = exp.copy()
                    # 移除name字段，因为它不是训练参数
                    yaml_config.pop('name', None)
                    experiment_found = True
                    break
            
            if not experiment_found:
                available_experiments = [exp.get('name', 'unnamed') for exp in full_yaml_config['experiments']]
                raise ValueError(f"Experiment '{cli_args.experiment_name}' not found. Available experiments: {available_experiments}")
        else:
            # 没有指定实验名称，使用整个配置（但排除experiments列表）
            yaml_config = full_yaml_config.copy()
            yaml_config.pop('experiments', None)  # 移除experiments列表避免干扰
    
    # 合并配置
    args = yaml_parser.merge_config(yaml_config, cli_args)
    
    # 验证配置
    args = yaml_parser.validate_config(args)
    
    # 自动配置路径
    args = yaml_parser.auto_configure_paths(args)
    
    return args


# 为了兼容性，保留原有函数名
def get_args_from_yaml(config_path: str, cli_overrides: Optional[Dict[str, Any]] = None) -> argparse.Namespace:
    """
    从YAML文件加载配置的便捷函数
    
    Args:
        config_path: YAML配置文件路径
        cli_overrides: 可选的覆盖参数字典
    
    Returns:
        argparse.Namespace: 配置对象
    """
    yaml_parser = YAMLConfigParser()
    yaml_config = yaml_parser.load_yaml_config(config_path)
    
    # 将覆盖参数转换为Namespace
    cli_args = None
    if cli_overrides:
        cli_args = argparse.Namespace()
        for key, value in cli_overrides.items():
            setattr(cli_args, key, value)
    
    # 合并配置
    args = yaml_parser.merge_config(yaml_config, cli_args)
    
    # 验证和后处理
    args = yaml_parser.validate_config(args)
    args = yaml_parser.auto_configure_paths(args)
    
    return args
