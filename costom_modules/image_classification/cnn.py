import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

class ResNetForCIFAR10(nn.Module):
    def __init__(self, layers = 18):
        super(ResNetForCIFAR10, self).__init__()
        # 加载预训练的resnet模型

        if layers == 18:
            self.model = models.resnet18()
        if layers == 34:
            self.model = models.resnet34()
        if layers == 50:
            self.model = models.resnet50()
        if layers == 101:
            self.model = models.resnet101()
        # 修改第一层卷积核大小和步长
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 修改最后的全连接层，以适应 CIFAR-10 的10个类别
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)
        self.model_name = f"resnet{layers}"

    def forward(self, x):
        return self.model(x)
    
class MLP(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=64, num_hidden_layers=5, output_size=10):
        super(MLP, self).__init__()
        layers = [nn.Flatten()]

        # Add the first layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Add hidden layers
        for _ in range(num_hidden_layers - 1):  # subtract 1 because the first layer is already added
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Add the output layer
        layers.append(nn.Linear(hidden_size, output_size))
        layers.append(nn.Softmax(dim=1))

        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.layers(x)

class SimpleCNN(nn.Module):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)