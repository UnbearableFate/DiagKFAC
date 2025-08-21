#!/usr/bin/env python3
"""
准确率分析脚本
专门分析各优化器组合的准确率表现
"""

import csv
import argparse

def analyze_accuracy_from_csv(csv_file: str = 'training_analysis_results.csv'):
    """
    从CSV文件分析准确率数据
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        # 过滤有准确率数据的记录
        accuracy_data = [row for row in data if float(row.get('average_final_accuracy', 0)) > 0]
        
        if not accuracy_data:
            print("错误: CSV文件中没有找到准确率数据")
            return
        
        print("=== 按最终准确率排序 ===")
        final_acc_sorted = sorted(accuracy_data, key=lambda x: float(x['average_final_accuracy']), reverse=True)
        print(f"{'优化器组合':20} {'最终准确率(%)':>12} {'最大准确率(%)':>12} {'训练时间':>12} {'内存使用':>10}")
        print("-" * 75)
        
        for row in final_acc_sorted:
            print(f"{row['optimizer_preconditioner']:20} "
                  f"{float(row['average_final_accuracy']):10.2f}%   "
                  f"{float(row['average_max_accuracy']):10.2f}%   "
                  f"{float(row['average_time_per_epoch']):10.6f}s  "
                  f"{float(row['average_memory_gb']):8.4f}GB")
        
        print("\n=== 按最大准确率排序 ===")
        max_acc_sorted = sorted(accuracy_data, key=lambda x: float(x['average_max_accuracy']), reverse=True)
        print(f"{'优化器组合':20} {'最大准确率(%)':>12} {'最终准确率(%)':>12} {'训练时间':>12} {'内存使用':>10}")
        print("-" * 75)
        
        for row in max_acc_sorted:
            print(f"{row['optimizer_preconditioner']:20} "
                  f"{float(row['average_max_accuracy']):10.2f}%   "
                  f"{float(row['average_final_accuracy']):10.2f}%   "
                  f"{float(row['average_time_per_epoch']):10.6f}s  "
                  f"{float(row['average_memory_gb']):8.4f}GB")
        
        print("\n=== 准确率分析总结 ===")
        best_final = max(accuracy_data, key=lambda x: float(x['average_final_accuracy']))
        best_max = max(accuracy_data, key=lambda x: float(x['average_max_accuracy']))
        worst_final = min(accuracy_data, key=lambda x: float(x['average_final_accuracy']))
        
        print(f"最佳最终准确率: {best_final['optimizer_preconditioner']} - {float(best_final['average_final_accuracy']):.2f}%")
        print(f"最佳最大准确率: {best_max['optimizer_preconditioner']} - {float(best_max['average_max_accuracy']):.2f}%")
        print(f"最差最终准确率: {worst_final['optimizer_preconditioner']} - {float(worst_final['average_final_accuracy']):.2f}%")
        
        # 计算准确率差异
        final_acc_range = float(best_final['average_final_accuracy']) - float(worst_final['average_final_accuracy'])
        print(f"最终准确率差异: {final_acc_range:.2f}个百分点")
        
        # 分析准确率与其他指标的关系
        print("\n=== 综合效率分析 ===")
        print("(最终准确率 / 训练时间，越高越好)")
        efficiency_data = []
        for row in accuracy_data:
            efficiency = float(row['average_final_accuracy']) / float(row['average_time_per_epoch'])
            efficiency_data.append((row['optimizer_preconditioner'], efficiency, 
                                  float(row['average_final_accuracy']), float(row['average_time_per_epoch'])))
        
        efficiency_data.sort(key=lambda x: x[1], reverse=True)
        print(f"{'优化器组合':20} {'效率比值':>10} {'准确率':>10} {'时间':>10}")
        print("-" * 55)
        for name, efficiency, accuracy, time in efficiency_data:
            print(f"{name:20}: {efficiency:8.3f}   {accuracy:6.2f}%   {time:8.6f}s")
        
        print("\n=== 内存效率分析（准确率/内存） ===")
        memory_efficiency_data = []
        for row in accuracy_data:
            mem_efficiency = float(row['average_final_accuracy']) / float(row['average_memory_gb'])
            memory_efficiency_data.append((row['optimizer_preconditioner'], mem_efficiency,
                                         float(row['average_final_accuracy']), float(row['average_memory_gb'])))
        
        memory_efficiency_data.sort(key=lambda x: x[1], reverse=True)
        print(f"{'优化器组合':20} {'效率比值':>10} {'准确率':>10} {'内存':>8}")
        print("-" * 53)
        for name, efficiency, accuracy, memory in memory_efficiency_data:
            print(f"{name:20}: {efficiency:8.3f}   {accuracy:6.2f}%   {memory:6.4f}GB")
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
        print("请先运行 analyze_training_times_enhanced.py 生成数据")
    except Exception as e:
        print(f"错误: {e}")

def main():
    parser = argparse.ArgumentParser(description='分析准确率数据')
    parser.add_argument('--csv-file', default='training_analysis_results.csv',
                        help='输入的CSV文件路径')
    
    args = parser.parse_args()
    analyze_accuracy_from_csv(args.csv_file)

if __name__ == "__main__":
    main()
