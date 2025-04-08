import torch
import torch.nn as nn
from torch.func import functional_call, grad
from torch import vmap

# 定义一个简单模型，其中包含一个全连接层 fc1
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型、损失函数
input_dim = 10
hidden_dim = 20
output_dim = 1
model = SimpleModel(input_dim, hidden_dim, output_dim)
loss_fn = nn.MSELoss()

# 将模型参数转换为字典，新的 API 使用字典来管理参数
params = dict(model.named_parameters())

# 定义一个函数，输入单个样本 x 和 target，返回 loss
def compute_loss(params, x, target):
    # 使用 functional_call 对模型进行无状态调用，传入模型和参数，不需要额外传入 buffers
    output = functional_call(model, params, x.unsqueeze(0))
    loss = loss_fn(output, target.unsqueeze(0))
    return loss

# 假设有一个 mini-batch 数据
M = 32  # batch size
inputs = torch.randn(M, input_dim)
targets = torch.randn(M, output_dim)

# 利用 torch.func.grad 和 vmap 在 batch 维度上计算每个样本对模型参数的梯度
per_sample_grad_fn = vmap(grad(compute_loss), in_dims=(None, 0, 0))
# 得到的 per_sample_grads 是一个字典，与 params 的键一致，每个值的形状为 (M, *parameter.shape)
per_sample_grads = per_sample_grad_fn(params, inputs, targets)

# 假设我们只关注 fc1.weight 的梯度
grad_fc1_weight = per_sample_grads['fc1.weight']  # shape: (M, out_features, in_features)

# 将 fc1.weight 的 per-sample 梯度重构为矩阵 U，形状为 (N_l, M)
# 其中 N_l = out_features * in_features
M, out_features, in_features = grad_fc1_weight.shape
Nl = out_features * in_features
# 对每个样本的梯度展平，并转置，使得每一列对应一个样本
U = grad_fc1_weight.reshape(M, -1).transpose(0, 1)

print("U 的形状:", U.shape)