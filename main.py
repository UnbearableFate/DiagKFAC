#!/usr/bin/env python3
"""
纯YAML配置驱动的主训练脚本
基于 trainning_kit/yaml_config_parser.py 构建
"""

import datetime
import sys
import os
import torch.distributed as dist
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainning_kit.yaml_config_parser import merged_args_parser
from trainning_kit.trainer import Trainer


def main():
    """
    主函数 - 使用yaml_config_parser的完整功能
    支持以下用法：
    1. python main_yaml_pure.py --config configs/xxx.yaml
    2. python main_yaml_pure.py --config configs/xxx.yaml --experiment-name xxx
    3. python main_yaml_pure.py --model xxx --dataset xxx --opt xxx (传统方式)
    """
    # 使用yaml_config_parser中的merged_args_parser
    # 它已经包含了完整的命令行解析和YAML配置加载逻辑
    args = merged_args_parser()
    
    # 创建并运行训练器
    trainer = Trainer(args)
    
    if args.test_only:
        print("Running in test-only mode...")
        trainer.test_only()
    else:
        print("Starting training...")
        trainer.train_and_evaluate()

if __name__ == "__main__":
    
    if os.getenv('MASTER_ADDR') is not None:
        timeout = datetime.timedelta(seconds=30)
        rank = os.getenv("RANK")
        world_size = os.getenv("WORLD_SIZE")
        
        if rank is None and world_size is None:
            world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
            rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))

            if world_size == -1 or rank == -1:
                from mpi4py import MPI
                rank = MPI.COMM_WORLD.Get_rank()
                world_size = MPI.COMM_WORLD.Get_size()
        
        if world_size < 1:
            raise ValueError("WORLD_SIZE must be greater than 0")
        
        print(f"World Size: {world_size}, World Rank: {rank} hostname {os.uname().nodename}")
        dist.init_process_group(backend='nccl', timeout=timeout, rank=rank, world_size=world_size)
        print("Distributed process group initialized.")
        print(f"World size: {dist.get_world_size()}, Rank: {dist.get_rank()}")
        if dist.get_rank() == 0:
            import logging
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    main()
    
    if dist.is_initialized():
        dist.destroy_process_group()