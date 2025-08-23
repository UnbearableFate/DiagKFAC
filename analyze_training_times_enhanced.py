#!/usr/bin/env python3
"""
训练时长分析脚本（增强版）
统计不同优化器+预处理器组合的平均epoch训练时间，并计算需要设定的epoch数
支持CSV输出和可视化
"""

import os
import re
import glob
import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse

def extract_metrics_from_log(log_file_path: str) -> Tuple[float, float, float, float]:
    """
    从日志文件中提取训练指标
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        (train_time_per_epoch, max_cuda_memory, final_accuracy, max_accuracy) 元组，如果未找到则返回None
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 寻找"train time / epoch:"行
        time_pattern = r'train time / epoch:\s+([\d.]+)'
        time_match = re.search(time_pattern, content)
        
        # 寻找"max cuda memory:"行
        memory_pattern = r'max cuda memory:\s+([\d.]+)\s+GB'
        memory_match = re.search(memory_pattern, content)
        
        # 寻找"final accuracy:"行
        final_acc_pattern = r'final accuracy:\s+([\d.]+)'
        final_acc_match = re.search(final_acc_pattern, content)
        
        # 寻找"max accuracy:"行
        max_acc_pattern = r'max accuracy:\s+([\d.]+)'
        max_acc_match = re.search(max_acc_pattern, content)
        
        train_time = float(time_match.group(1)) if time_match else None
        max_memory = float(memory_match.group(1)) if memory_match else None
        final_accuracy = float(final_acc_match.group(1)) if final_acc_match else None
        max_accuracy = float(max_acc_match.group(1)) if max_acc_match else None
        
        if train_time is None:
            print(f"警告: 在文件 {log_file_path} 中未找到 'train time / epoch' 信息")
        if max_memory is None:
            print(f"警告: 在文件 {log_file_path} 中未找到 'max cuda memory' 信息")
        if final_accuracy is None:
            print(f"警告: 在文件 {log_file_path} 中未找到 'final accuracy' 信息")
        if max_accuracy is None:
            print(f"警告: 在文件 {log_file_path} 中未找到 'max accuracy' 信息")
            
        return train_time, max_memory, final_accuracy, max_accuracy
            
    except Exception as e:
        print(f"错误: 读取文件 {log_file_path} 时出错: {e}")
        return None, None, None, None

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

def collect_training_metrics(base_path: str, verbose: bool = True) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """
    收集所有优化器+预处理器组合的训练指标
    
    Args:
        base_path: logs/ResNet_cifar10/resnet18_cifar10_experiments 路径
        verbose: 是否显示详细信息
        
    Returns:
        (训练时间字典, 内存使用字典, 最终准确率字典, 最大准确率字典)，键为 "optimizer_preconditioner"，值为相应指标的列表
    """
    training_times = defaultdict(list)
    memory_usage = defaultdict(list)
    final_accuracies = defaultdict(list)
    max_accuracies = defaultdict(list)
    
    # 遍历所有优化器+预处理器组合的文件夹
    pattern = os.path.join(base_path, '*')
    for folder_path in glob.glob(pattern):
        if not os.path.isdir(folder_path):
            continue
            
        folder_name = os.path.basename(folder_path)
        optimizer, preconditioner = parse_optimizer_preconditioner(folder_name)
        combination_key = f"{optimizer}_{preconditioner}"
        
        if verbose:
            print(f"处理组合: {combination_key} (文件夹: {folder_name})")
        
        # 遍历该组合下的所有时间戳文件夹
        timestamp_pattern = os.path.join(folder_path, '*')
        for timestamp_folder in glob.glob(timestamp_pattern):
            if not os.path.isdir(timestamp_folder):
                continue
                
            log_file = os.path.join(timestamp_folder, 'log.txt')
            if os.path.exists(log_file):
                train_time, max_memory, final_acc, max_acc = extract_metrics_from_log(log_file)
                timestamp_name = os.path.basename(timestamp_folder)
                
                if verbose:
                    print(f"  - 时间戳 {timestamp_name}:", end="")
                
                if train_time is not None:
                    training_times[combination_key].append(train_time)
                    if verbose:
                        print(f" {train_time:.6f}秒/epoch", end="")
                        
                if max_memory is not None:
                    memory_usage[combination_key].append(max_memory)
                    if verbose:
                        print(f", {max_memory:.4f}GB", end="")
                
                if final_acc is not None:
                    final_accuracies[combination_key].append(final_acc)
                    if verbose:
                        print(f", 最终准确率{final_acc:.2f}%", end="")
                
                if max_acc is not None:
                    max_accuracies[combination_key].append(max_acc)
                    if verbose:
                        print(f", 最大准确率{max_acc:.2f}%", end="")
                
                if verbose:
                    print()  # 换行
    
    return training_times, memory_usage, final_accuracies, max_accuracies

def calculate_statistics(training_times: Dict[str, List[float]], memory_usage: Dict[str, List[float]], 
                        final_accuracies: Dict[str, List[float]], max_accuracies: Dict[str, List[float]], 
                        verbose: bool = True) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
    """
    计算每个组合的统计信息
    
    Args:
        training_times: 原始训练时间数据
        memory_usage: 原始内存使用数据
        final_accuracies: 原始最终准确率数据
        max_accuracies: 原始最大准确率数据
        verbose: 是否显示详细信息
        
    Returns:
        (时间统计信息, 内存统计信息, 最终准确率统计信息, 最大准确率统计信息)
    """
    time_statistics = {}
    memory_statistics = {}
    final_acc_statistics = {}
    max_acc_statistics = {}
    
    def calculate_stats(data_list):
        if not data_list:
            return {}
        avg = sum(data_list) / len(data_list)
        min_val = min(data_list)
        max_val = max(data_list)
        std_dev = (sum((x - avg) ** 2 for x in data_list) / len(data_list)) ** 0.5
        return {
            'average': avg,
            'min': min_val,
            'max': max_val,
            'std_dev': std_dev,
            'count': len(data_list),
            'values': data_list
        }
    
    if verbose:
        print("\n=== 训练时间统计结果 ===")
    
    # 处理训练时间统计
    for combination, times in training_times.items():
        time_statistics[combination] = calculate_stats(times)
        if verbose and times:
            stats = time_statistics[combination]
            print(f"{combination:20}: {stats['average']:.6f}±{stats['std_dev']:.6f} 秒/epoch (基于 {stats['count']} 次运行)")
            if stats['count'] > 1:
                print(f"{' ' * 20}  范围: {stats['min']:.6f} - {stats['max']:.6f}")
        elif verbose:
            print(f"{combination:20}: 无有效时间数据")
    
    if verbose:
        print("\n=== CUDA内存使用统计结果 ===")
    
    # 处理内存使用统计
    for combination, memories in memory_usage.items():
        memory_statistics[combination] = calculate_stats(memories)
        if verbose and memories:
            stats = memory_statistics[combination]
            print(f"{combination:20}: {stats['average']:.4f}±{stats['std_dev']:.4f} GB (基于 {stats['count']} 次运行)")
            if stats['count'] > 1:
                print(f"{' ' * 20}  范围: {stats['min']:.4f} - {stats['max']:.4f} GB")
        elif verbose:
            print(f"{combination:20}: 无有效内存数据")
    
    if verbose:
        print("\n=== 最终准确率统计结果 ===")
    
    # 处理最终准确率统计
    for combination, accuracies in final_accuracies.items():
        final_acc_statistics[combination] = calculate_stats(accuracies)
        if verbose and accuracies:
            stats = final_acc_statistics[combination]
            print(f"{combination:20}: {stats['average']:.2f}±{stats['std_dev']:.2f}% (基于 {stats['count']} 次运行)")
            if stats['count'] > 1:
                print(f"{' ' * 20}  范围: {stats['min']:.2f}% - {stats['max']:.2f}%")
        elif verbose:
            print(f"{combination:20}: 无有效最终准确率数据")
    
    if verbose:
        print("\n=== 最大准确率统计结果 ===")
    
    # 处理最大准确率统计
    for combination, accuracies in max_accuracies.items():
        max_acc_statistics[combination] = calculate_stats(accuracies)
        if verbose and accuracies:
            stats = max_acc_statistics[combination]
            print(f"{combination:20}: {stats['average']:.2f}±{stats['std_dev']:.2f}% (基于 {stats['count']} 次运行)")
            if stats['count'] > 1:
                print(f"{' ' * 20}  范围: {stats['min']:.2f}% - {stats['max']:.2f}%")
        elif verbose:
            print(f"{combination:20}: 无有效最大准确率数据")
    
    return time_statistics, memory_statistics, final_acc_statistics, max_acc_statistics

def calculate_recommended_epochs(statistics: Dict[str, Dict], baseline_epochs: int = 200, baseline_optimizer: str = 'adamw_none') -> Dict[str, int]:
    """
    基于基准优化器计算推荐的epoch数
    
    Args:
        statistics: 各组合的统计信息
        baseline_epochs: 基准epoch数
        baseline_optimizer: 基准优化器名称
        
    Returns:
        各组合推荐的epoch数
    """
    if baseline_optimizer not in statistics:
        print(f"错误: 未找到 {baseline_optimizer} 的数据作为基准")
        return {}
    
    baseline_time = statistics[baseline_optimizer]['average']
    recommendations = {}
    
    print(f"\n=== 推荐Epoch数 (基准: {baseline_optimizer} = {baseline_epochs} epochs) ===")
    print(f"{baseline_optimizer} 平均训练时间: {baseline_time:.6f} 秒/epoch")
    print()
    
    for combination, stats in statistics.items():
        avg_time = stats['average']
        # 计算推荐epoch数: 基准优化器的平均时间 * 基准epochs / 当前组合的平均时间
        recommended_epochs = int(baseline_time * baseline_epochs / avg_time)
        recommendations[combination] = recommended_epochs
        
        speedup_ratio = baseline_time / avg_time
        print(f"{combination:20}: {recommended_epochs:3d} epochs (速度比 {speedup_ratio:.2f}x)")
    
    return recommendations

def save_csv_results(time_statistics: Dict[str, Dict], memory_statistics: Dict[str, Dict], 
                    final_acc_statistics: Dict[str, Dict], max_acc_statistics: Dict[str, Dict],
                    recommendations: Dict[str, int], output_file: str):
    """
    保存结果到CSV文件
    
    Args:
        time_statistics: 时间统计信息
        memory_statistics: 内存统计信息
        final_acc_statistics: 最终准确率统计信息
        max_acc_statistics: 最大准确率统计信息
        recommendations: 推荐epoch数
        output_file: 输出文件路径
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'optimizer_preconditioner', 
            'average_time_per_epoch', 'time_std_dev', 'min_time', 'max_time', 'time_run_count',
            'average_memory_gb', 'memory_std_dev', 'min_memory', 'max_memory', 'memory_run_count',
            'average_final_accuracy', 'final_acc_std_dev', 'min_final_acc', 'max_final_acc', 'final_acc_run_count',
            'average_max_accuracy', 'max_acc_std_dev', 'min_max_acc', 'max_max_acc', 'max_acc_run_count',
            'recommended_epochs', 'speedup_ratio'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        baseline_time = time_statistics.get('adamw_none', {}).get('average', 1.0)
        
        # 获取所有组合的并集
        all_combinations = set(time_statistics.keys()) | set(memory_statistics.keys()) | \
                          set(final_acc_statistics.keys()) | set(max_acc_statistics.keys())
        
        for combination in sorted(all_combinations):
            time_stats = time_statistics.get(combination, {})
            memory_stats = memory_statistics.get(combination, {})
            final_acc_stats = final_acc_statistics.get(combination, {})
            max_acc_stats = max_acc_statistics.get(combination, {})
            
            speedup_ratio = baseline_time / time_stats.get('average', baseline_time) if time_stats.get('average') else 1.0
            
            writer.writerow({
                'optimizer_preconditioner': combination,
                'average_time_per_epoch': f"{time_stats.get('average', 0):.6f}",
                'time_std_dev': f"{time_stats.get('std_dev', 0):.6f}",
                'min_time': f"{time_stats.get('min', 0):.6f}",
                'max_time': f"{time_stats.get('max', 0):.6f}",
                'time_run_count': time_stats.get('count', 0),
                'average_memory_gb': f"{memory_stats.get('average', 0):.4f}",
                'memory_std_dev': f"{memory_stats.get('std_dev', 0):.4f}",
                'min_memory': f"{memory_stats.get('min', 0):.4f}",
                'max_memory': f"{memory_stats.get('max', 0):.4f}",
                'memory_run_count': memory_stats.get('count', 0),
                'average_final_accuracy': f"{final_acc_stats.get('average', 0):.2f}",
                'final_acc_std_dev': f"{final_acc_stats.get('std_dev', 0):.2f}",
                'min_final_acc': f"{final_acc_stats.get('min', 0):.2f}",
                'max_final_acc': f"{final_acc_stats.get('max', 0):.2f}",
                'final_acc_run_count': final_acc_stats.get('count', 0),
                'average_max_accuracy': f"{max_acc_stats.get('average', 0):.2f}",
                'max_acc_std_dev': f"{max_acc_stats.get('std_dev', 0):.2f}",
                'min_max_acc': f"{max_acc_stats.get('min', 0):.2f}",
                'max_max_acc': f"{max_acc_stats.get('max', 0):.2f}",
                'max_acc_run_count': max_acc_stats.get('count', 0),
                'recommended_epochs': recommendations.get(combination, 0),
                'speedup_ratio': f"{speedup_ratio:.2f}"
            })

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析训练时长并计算推荐epoch数')
    parser.add_argument('--base-path', default='/home/yu/workspace/DiagKFAC/logs/ResNet_cifar10/resnet18_cifar10_experiments',
                        help='日志文件基础路径')
    parser.add_argument('--baseline-epochs', type=int, default=200,
                        help='基准epoch数')
    parser.add_argument('--baseline-optimizer', default='adamw_none',
                        help='基准优化器名称')
    parser.add_argument('--output-txt', default='training_analysis_results.txt',
                        help='文本结果输出文件')
    parser.add_argument('--output-csv', default='training_analysis_results.csv',
                        help='CSV结果输出文件')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式，减少输出')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.base_path):
        print(f"错误: 路径 {args.base_path} 不存在")
        return
    
    verbose = not args.quiet
    
    if verbose:
        print(f"分析路径: {args.base_path}")
        print("=" * 60)
    
    # 收集训练指标数据
    training_times, memory_usage, final_accuracies, max_accuracies = collect_training_metrics(args.base_path, verbose)
    
    if not training_times and not memory_usage and not final_accuracies and not max_accuracies:
        print("错误: 未找到任何训练数据")
        return
    
    # 计算统计信息
    time_statistics, memory_statistics, final_acc_statistics, max_acc_statistics = calculate_statistics(
        training_times, memory_usage, final_accuracies, max_accuracies, verbose)
    
    # 计算推荐的epoch数
    recommendations = calculate_recommended_epochs(time_statistics, args.baseline_epochs, args.baseline_optimizer)
    
    # 保存文本结果
    output_txt = os.path.join(args.base_path, args.output_txt)
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("训练时长、内存使用和准确率分析结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("训练时间详细统计信息:\n")
        for combination, stats in time_statistics.items():
            if stats:
                f.write(f"{combination:20}: {stats['average']:.6f}±{stats['std_dev']:.6f} 秒/epoch ")
                f.write(f"(基于 {stats['count']} 次运行)\n")
                if stats['count'] > 1:
                    f.write(f"{' ' * 20}  范围: {stats['min']:.6f} - {stats['max']:.6f}\n")
        
        f.write(f"\nCUDA内存使用详细统计信息:\n")
        for combination, stats in memory_statistics.items():
            if stats:
                f.write(f"{combination:20}: {stats['average']:.4f}±{stats['std_dev']:.4f} GB ")
                f.write(f"(基于 {stats['count']} 次运行)\n")
                if stats['count'] > 1:
                    f.write(f"{' ' * 20}  范围: {stats['min']:.4f} - {stats['max']:.4f} GB\n")
        
        f.write(f"\n最终准确率详细统计信息:\n")
        for combination, stats in final_acc_statistics.items():
            if stats:
                f.write(f"{combination:20}: {stats['average']:.2f}±{stats['std_dev']:.2f}% ")
                f.write(f"(基于 {stats['count']} 次运行)\n")
                if stats['count'] > 1:
                    f.write(f"{' ' * 20}  范围: {stats['min']:.2f}% - {stats['max']:.2f}%\n")
        
        f.write(f"\n最大准确率详细统计信息:\n")
        for combination, stats in max_acc_statistics.items():
            if stats:
                f.write(f"{combination:20}: {stats['average']:.2f}±{stats['std_dev']:.2f}% ")
                f.write(f"(基于 {stats['count']} 次运行)\n")
                if stats['count'] > 1:
                    f.write(f"{' ' * 20}  范围: {stats['min']:.2f}% - {stats['max']:.2f}%\n")
        
        f.write(f"\n推荐Epoch数 (基准: {args.baseline_optimizer} = {args.baseline_epochs} epochs):\n")
        baseline_time = time_statistics.get(args.baseline_optimizer, {}).get('average', 1.0)
        for combination, epochs in recommendations.items():
            speedup = baseline_time / time_statistics[combination]['average'] if combination in time_statistics and time_statistics[combination] else 1.0
            f.write(f"{combination:20}: {epochs:3d} epochs (速度比 {speedup:.2f}x)\n")
    
    # 保存CSV结果
    output_dir = os.path.join(args.base_path, args.output_csv)
    save_csv_results(time_statistics, memory_statistics, final_acc_statistics, max_acc_statistics, recommendations, output_dir)

    if verbose:
        print(f"\n文本结果已保存到: {args.output_txt}")
        print(f"CSV结果已保存到: {output_dir}")

if __name__ == "__main__":
    main()
