#!/usr/bin/env python3
"""
快速内存使用分析脚本
按内存使用排序显示各优化器组合
"""

import csv

def analyze_memory_usage(csv_file: str = 'training_analysis_results.csv'):
    """
    分析内存使用情况并按内存使用量排序
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        # 按平均内存使用量排序
        data_sorted = sorted(data, key=lambda x: float(x['average_memory_gb']))
        
        print("=== 按平均CUDA内存使用量排序 ===")
        print(f"{'优化器组合':20} {'平均内存(GB)':>12} {'时间(秒/epoch)':>15} {'推荐Epochs':>10} {'速度比':>8}")
        print("-" * 70)
        
        for row in data_sorted:
            print(f"{row['optimizer_preconditioner']:20} "
                  f"{float(row['average_memory_gb']):10.4f}   "
                  f"{float(row['average_time_per_epoch']):13.6f}   "
                  f"{int(row['recommended_epochs']):8d}   "
                  f"{float(row['speedup_ratio']):6.2f}x")
        
        print("\n=== 内存使用分析总结 ===")
        min_memory = min(data, key=lambda x: float(x['average_memory_gb']))
        max_memory = max(data, key=lambda x: float(x['average_memory_gb']))
        
        print(f"最低内存使用: {min_memory['optimizer_preconditioner']} - {float(min_memory['average_memory_gb']):.4f} GB")
        print(f"最高内存使用: {max_memory['optimizer_preconditioner']} - {float(max_memory['average_memory_gb']):.4f} GB")
        print(f"内存使用倍数差: {float(max_memory['average_memory_gb']) / float(min_memory['average_memory_gb']):.2f}x")
        
        # 分析内存使用与速度的关系
        print("\n=== 内存效率分析 ===")
        print("(速度比 / 内存使用量 比值，越高越好)")
        efficiency_data = []
        for row in data:
            efficiency = float(row['speedup_ratio']) / float(row['average_memory_gb'])
            efficiency_data.append((row['optimizer_preconditioner'], efficiency))
        
        efficiency_data.sort(key=lambda x: x[1], reverse=True)
        for name, efficiency in efficiency_data:
            print(f"{name:20}: {efficiency:.3f}")
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
        print("请先运行 analyze_training_times_enhanced.py 生成数据")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    analyze_memory_usage()
