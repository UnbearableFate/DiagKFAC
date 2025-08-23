#!/usr/bin/env python3
"""
简单测试脚本，验证OneCycleLR集成是否正确
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt

# 创建一个简单的模型
model = nn.Linear(10, 1)
optimizer = AdamW(model.parameters(), lr=0.001)

# 模拟训练设置
epochs = 3
steps_per_epoch = 100
total_steps = epochs * steps_per_epoch

# 创建OneCycleLR调度器
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,  # 最大学习率
    total_steps=total_steps,
    pct_start=0.3,  # 30%的时间用于升高学习率
    anneal_strategy='cos',  # 余弦退火
    div_factor=25.0,
    final_div_factor=1e4
)

# 记录学习率变化
learning_rates = []
steps = []

print("测试OneCycleLR调度器...")
print(f"总步数: {total_steps}")
print(f"初始学习率: {optimizer.param_groups[0]['lr']:.6f}")

step = 0
for epoch in range(epochs):
    for batch_idx in range(steps_per_epoch):
        # 模拟训练步骤
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        steps.append(step)
        
        # 更新调度器
        scheduler.step()
        step += 1
        
        if batch_idx % 25 == 0:
            print(f"Epoch {epoch}, Step {batch_idx}, LR: {current_lr:.6f}")

print(f"最终学习率: {optimizer.param_groups[0]['lr']:.6f}")

# 绘制学习率变化曲线
plt.figure(figsize=(10, 6))
plt.plot(steps, learning_rates)
plt.xlabel('训练步数')
plt.ylabel('学习率')
plt.title('OneCycleLR学习率变化曲线')
plt.grid(True)
plt.savefig('/work/xg24i002/x10041/DiagKFAC/onecycle_lr_curve.png', dpi=300, bbox_inches='tight')
print("学习率变化曲线已保存为 onecycle_lr_curve.png")
