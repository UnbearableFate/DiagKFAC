#!/usr/bin/env python3
"""
综合分析报告生成脚本
生成包含所有指标的简洁摘要报告
"""

import csv
import argparse

def generate_summary_report(csv_file: str = 'training_analysis_results.csv'):
    """
    生成综合分析摘要报告
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        print("=" * 80)
        print(" " * 25 + "训练分析综合报告")
        print("=" * 80)
        
        # 按训练时间排序的摘要表格
        print("\n=== 完整性能对比表 ===")
        print(f"{'优化器组合':15} {'时间(s/epoch)':>12} {'内存(GB)':>10} {'最终准确率(%)':>12} {'推荐Epochs':>10} {'速度比':>8}")
        print("-" * 75)
        
        # 过滤有时间数据的记录并按时间排序
        time_data = [row for row in data if float(row.get('average_time_per_epoch', 0)) > 0]
        time_data.sort(key=lambda x: float(x['average_time_per_epoch']))
        
        for row in time_data:
            optimizer = row['optimizer_preconditioner']
            time_per_epoch = float(row['average_time_per_epoch'])
            memory = float(row['average_memory_gb'])
            final_acc = float(row['average_final_accuracy'])
            epochs = int(row['recommended_epochs'])
            speedup = float(row['speedup_ratio'])
            
            acc_str = f"{final_acc:.1f}" if final_acc > 0 else "N/A"
            
            print(f"{optimizer:15} {time_per_epoch:10.6f}   {memory:8.4f}   {acc_str:>10}   {epochs:8d}   {speedup:6.2f}x")
        
        # 统计摘要
        print(f"\n=== 统计摘要 ===")
        if time_data:
            fastest = min(time_data, key=lambda x: float(x['average_time_per_epoch']))
            slowest = max(time_data, key=lambda x: float(x['average_time_per_epoch']))
            
            print(f"最快优化器: {fastest['optimizer_preconditioner']} ({float(fastest['average_time_per_epoch']):.6f} 秒/epoch)")
            print(f"最慢优化器: {slowest['optimizer_preconditioner']} ({float(slowest['average_time_per_epoch']):.6f} 秒/epoch)")
            print(f"速度差异: {float(slowest['average_time_per_epoch']) / float(fastest['average_time_per_epoch']):.2f}x")
            
            # 内存使用分析
            min_memory = min(time_data, key=lambda x: float(x['average_memory_gb']))
            max_memory = max(time_data, key=lambda x: float(x['average_memory_gb']))
            
            print(f"\n最低内存: {min_memory['optimizer_preconditioner']} ({float(min_memory['average_memory_gb']):.4f} GB)")
            print(f"最高内存: {max_memory['optimizer_preconditioner']} ({float(max_memory['average_memory_gb']):.4f} GB)")
            print(f"内存差异: {float(max_memory['average_memory_gb']) / float(min_memory['average_memory_gb']):.2f}x")
        
        # 准确率分析
        accuracy_data = [row for row in data if float(row.get('average_final_accuracy', 0)) > 0]
        if accuracy_data:
            print(f"\n=== 准确率分析 ===")
            best_acc = max(accuracy_data, key=lambda x: float(x['average_final_accuracy']))
            worst_acc = min(accuracy_data, key=lambda x: float(x['average_final_accuracy']))
            
            print(f"最佳准确率: {best_acc['optimizer_preconditioner']} ({float(best_acc['average_final_accuracy']):.2f}%)")
            print(f"最差准确率: {worst_acc['optimizer_preconditioner']} ({float(worst_acc['average_final_accuracy']):.2f}%)")
            print(f"准确率差异: {float(best_acc['average_final_accuracy']) - float(worst_acc['average_final_accuracy']):.2f}个百分点")
            
            # 性价比分析（准确率/时间）
            print(f"\n=== TOP 3 性价比 (准确率/时间) ===")
            efficiency_data = []
            for row in accuracy_data:
                acc = float(row['average_final_accuracy'])
                time = float(row['average_time_per_epoch'])
                efficiency = acc / time
                efficiency_data.append((row['optimizer_preconditioner'], efficiency, acc, time))
            
            efficiency_data.sort(key=lambda x: x[1], reverse=True)
            for i, (name, eff, acc, time) in enumerate(efficiency_data[:3], 1):
                print(f"{i}. {name:15}: {eff:6.3f} ({acc:5.2f}% / {time:6.6f}s)")
        
        # 推荐建议
        print(f"\n=== 使用建议 ===")
        if accuracy_data:
            best_overall = max(accuracy_data, key=lambda x: float(x['average_final_accuracy']))
            print(f"🏆 最佳准确率: {best_overall['optimizer_preconditioner']} - 追求最高性能时使用")
            
        if time_data:
            fastest_optimizer = min(time_data, key=lambda x: float(x['average_time_per_epoch']))
            print(f"⚡ 最快训练: {fastest_optimizer['optimizer_preconditioner']} - 快速实验或资源受限时使用")
            
            lowest_memory = min(time_data, key=lambda x: float(x['average_memory_gb']))
            print(f"💾 最低内存: {lowest_memory['optimizer_preconditioner']} - 内存受限时使用")
        
        if accuracy_data and len(efficiency_data) > 0:
            best_efficiency = efficiency_data[0]
            print(f"⚖️  最佳平衡: {best_efficiency[0]} - 准确率和速度的最佳平衡")
        
        print("=" * 80)
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {csv_file}")
        print("请先运行 analyze_training_times_enhanced.py 生成数据")
    except Exception as e:
        print(f"错误: {e}")

def main():
    parser = argparse.ArgumentParser(description='生成综合分析报告')
    parser.add_argument('--csv-file', default='training_analysis_results.csv',
                        help='输入的CSV文件路径')
    
    args = parser.parse_args()
    generate_summary_report(args.csv_file)

if __name__ == "__main__":
    main()
