#!/usr/bin/env python3
"""
训练时长分析脚本
统计不同优化器+预处理器组合的平均epoch训练时间，并计算需要设定的epoch数
"""

import os
import re
import glob
from collections import defaultdict
from typing import Dict, List, Tuple

def extract_train_time_per_epoch(log_file_path: str) -> float:
    """
    从日志文件中提取train time / epoch的值
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        train time per epoch的值，如果未找到则返回None
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 寻找"train time / epoch:"行
        pattern = r'train time / epoch:\s+([\d.]+)'
        match = re.search(pattern, content)
        
        if match:
            return float(match.group(1))
        else:
            print(f"警告: 在文件 {log_file_path} 中未找到 'train time / epoch' 信息")
            return None
            
    except Exception as e:
        print(f"错误: 读取文件 {log_file_path} 时出错: {e}")
        return None

def parse_optimizer_preconditioner(folder_name: str) -> Tuple[str, str]:
    """
    从文件夹名称解析优化器和预处理器
    
    Args:
        folder_name: 形如 "adafisher_none_2" 的文件夹名称
        
    Returns:
        (optimizer, preconditioner) 元组
    """
    # 移除末尾的数字后缀（如_2）
    parts = folder_name.split('_')
    if parts[-1].isdigit():
        parts = parts[:-1]
    
    if len(parts) >= 2:
        optimizer = parts[0]
        preconditioner = '_'.join(parts[1:])
        return optimizer, preconditioner
    else:
        return folder_name, 'unknown'

def collect_training_times(base_path: str) -> Dict[str, List[float]]:
    """
    收集所有优化器+预处理器组合的训练时间
    
    Args:
        base_path: logs/ResNet_cifar10/resnet18_cifar10_experiments 路径
        
    Returns:
        字典，键为 "optimizer_preconditioner"，值为时间列表
    """
    results = defaultdict(list)
    
    # 遍历所有优化器+预处理器组合的文件夹
    pattern = os.path.join(base_path, '*')
    for folder_path in glob.glob(pattern):
        if not os.path.isdir(folder_path):
            continue
            
        folder_name = os.path.basename(folder_path)
        optimizer, preconditioner = parse_optimizer_preconditioner(folder_name)
        combination_key = f"{optimizer}_{preconditioner}"
        
        print(f"处理组合: {combination_key} (文件夹: {folder_name})")
        
        # 遍历该组合下的所有时间戳文件夹
        timestamp_pattern = os.path.join(folder_path, '*')
        for timestamp_folder in glob.glob(timestamp_pattern):
            if not os.path.isdir(timestamp_folder):
                continue
                
            log_file = os.path.join(timestamp_folder, 'log.txt')
            if os.path.exists(log_file):
                train_time = extract_train_time_per_epoch(log_file)
                if train_time is not None:
                    results[combination_key].append(train_time)
                    print(f"  - 时间戳 {os.path.basename(timestamp_folder)}: {train_time:.6f} 秒/epoch")
    
    return results

def calculate_statistics(training_times: Dict[str, List[float]]) -> Dict[str, float]:
    """
    计算每个组合的平均训练时间
    
    Args:
        training_times: 原始训练时间数据
        
    Returns:
        每个组合的平均时间
    """
    averages = {}
    
    print("\n=== 统计结果 ===")
    for combination, times in training_times.items():
        if times:
            avg_time = sum(times) / len(times)
            averages[combination] = avg_time
            print(f"{combination:20}: {avg_time:.6f} 秒/epoch (基于 {len(times)} 次运行)")
        else:
            print(f"{combination:20}: 无有效数据")
    
    return averages

def calculate_recommended_epochs(averages: Dict[str, float], baseline_epochs: int = 200) -> Dict[str, int]:
    """
    基于adamw_none的基准计算推荐的epoch数
    
    Args:
        averages: 各组合的平均训练时间
        baseline_epochs: adamw_none的基准epoch数
        
    Returns:
        各组合推荐的epoch数
    """
    if 'adamw_none' not in averages:
        print("错误: 未找到 adamw_none 的数据作为基准")
        return {}
    
    baseline_time = averages['adamw_none']
    recommendations = {}
    
    print(f"\n=== 推荐Epoch数 (基准: adamw_none = {baseline_epochs} epochs) ===")
    print(f"adamw_none 平均训练时间: {baseline_time:.6f} 秒/epoch")
    print()
    
    for combination, avg_time in averages.items():
        # 计算推荐epoch数: adamw_none的平均时间 * 200 / 当前组合的平均时间
        recommended_epochs = int(baseline_time * baseline_epochs / avg_time)
        recommendations[combination] = recommended_epochs
        
        speedup_ratio = baseline_time / avg_time
        print(f"{combination:20}: {recommended_epochs:3d} epochs (速度比 {speedup_ratio:.2f}x)")
    
    return recommendations

def main():
    """主函数"""
    # 设置路径
    base_path = "/home/yu/workspace/DiagKFAC/logs/ResNet_cifar10/resnet18_cifar10_experiments"
    
    if not os.path.exists(base_path):
        print(f"错误: 路径 {base_path} 不存在")
        return
    
    print(f"分析路径: {base_path}")
    print("=" * 60)
    
    # 收集训练时间数据
    training_times = collect_training_times(base_path)
    
    if not training_times:
        print("错误: 未找到任何训练时间数据")
        return
    
    # 计算平均值
    averages = calculate_statistics(training_times)
    
    # 计算推荐的epoch数
    recommendations = calculate_recommended_epochs(averages)
    
    # 保存结果到文件
    output_file = "training_analysis_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("训练时长分析结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("平均训练时间 (秒/epoch):\n")
        for combination, avg_time in averages.items():
            f.write(f"{combination:20}: {avg_time:.6f}\n")
        
        f.write(f"\n推荐Epoch数 (基准: adamw_none = 200 epochs):\n")
        for combination, epochs in recommendations.items():
            speedup = averages['adamw_none'] / averages[combination] if combination in averages else 1.0
            f.write(f"{combination:20}: {epochs:3d} epochs (速度比 {speedup:.2f}x)\n")
    
    print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
