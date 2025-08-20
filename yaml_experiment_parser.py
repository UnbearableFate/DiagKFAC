#!/usr/bin/env python3
"""
YAML实验解析和运行工具
用于从YAML配置文件中解析实验并生成对应的命令行参数
"""

import yaml
import argparse
import sys
import os
from typing import Dict, Any, List, Optional
from copy import deepcopy


class YAMLExperimentParser:
    """YAML实验配置解析器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_experiment_names(self, group: Optional[str] = None) -> List[str]:
        """获取实验名称列表"""
        names = []
        experiments = self.config.get('experiments', {})
        
        if isinstance(experiments, list):
            # 简单的实验列表
            for exp in experiments:
                names.append(exp.get('name', 'unnamed'))
        elif isinstance(experiments, dict):
            # 分组的实验
            if group:
                if group in experiments:
                    for exp in experiments[group]:
                        names.append(exp.get('name', 'unnamed'))
            else:
                # 返回所有组的所有实验
                for group_name, group_exps in experiments.items():
                    for exp in group_exps:
                        names.append(exp.get('name', 'unnamed'))
        
        return names
    
    def get_experiment_config(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """获取特定实验的配置"""
        experiments = self.config.get('experiments', {})
        
        if isinstance(experiments, list):
            for exp in experiments:
                if exp.get('name') == experiment_name:
                    return exp
        elif isinstance(experiments, dict):
            for group_name, group_exps in experiments.items():
                for exp in group_exps:
                    if exp.get('name') == experiment_name:
                        return exp
        
        return None
    
    def resolve_inheritance(self, exp_config: Dict[str, Any]) -> Dict[str, Any]:
        """解析YAML继承（锚点和别名）"""
        # deepcopy 已经处理了YAML的锚点和别名
        return deepcopy(exp_config)
    
    def generate_args_string(self, experiment_name: str) -> Optional[str]:
        """为特定实验生成命令行参数字符串"""
        exp_config = self.get_experiment_config(experiment_name)
        if not exp_config:
            return None
        
        # 解析继承
        resolved_config = self.resolve_inheritance(exp_config)
        
        args_list = []
        
        # 处理每个配置项
        for key, value in resolved_config.items():
            if key == 'name':
                continue  # 跳过name字段
                
            # 转换为命令行参数格式
            arg_key = f"--{key.replace('_', '-')}"
            
            if isinstance(value, bool):
                if value:
                    args_list.append(arg_key)
            elif isinstance(value, list):
                if value:  # 非空列表
                    args_list.extend([arg_key] + [str(v) for v in value])
            elif value is not None:
                args_list.extend([arg_key, str(value)])
        
        return ' '.join(args_list)
    
    def list_experiments(self, group: Optional[str] = None) -> None:
        """列出可用的实验"""
        names = self.get_experiment_names(group)
        print(f"Available experiments{f' in group {group}' if group else ''}:")
        for name in names:
            print(f"  - {name}")
    
    def list_groups(self) -> None:
        """列出可用的实验组"""
        experiments = self.config.get('experiments', {})
        if isinstance(experiments, dict):
            print("Available experiment groups:")
            for group_name in experiments.keys():
                print(f"  - {group_name}")
        else:
            print("No experiment groups found (experiments are in a flat list)")


def main():
    parser = argparse.ArgumentParser(description="YAML实验配置解析工具")
    parser.add_argument("config", help="YAML配置文件路径")
    parser.add_argument("--list-experiments", action="store_true", help="列出所有实验名称")
    parser.add_argument("--list-groups", action="store_true", help="列出所有实验组")
    parser.add_argument("--group", help="指定实验组")
    parser.add_argument("--experiment", help="获取特定实验的参数")
    parser.add_argument("--generate-args", help="为指定实验生成命令行参数")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"错误：配置文件 {args.config} 不存在", file=sys.stderr)
        sys.exit(1)
    
    try:
        parser_tool = YAMLExperimentParser(args.config)
        
        if args.list_groups:
            parser_tool.list_groups()
        elif args.list_experiments:
            parser_tool.list_experiments(args.group)
        elif args.experiment:
            config = parser_tool.get_experiment_config(args.experiment)
            if config:
                print(yaml.dump(config, default_flow_style=False))
            else:
                print(f"实验 '{args.experiment}' 未找到", file=sys.stderr)
                sys.exit(1)
        elif args.generate_args:
            args_string = parser_tool.generate_args_string(args.generate_args)
            if args_string:
                print(args_string)
            else:
                print(f"实验 '{args.generate_args}' 未找到", file=sys.stderr)
                sys.exit(1)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
