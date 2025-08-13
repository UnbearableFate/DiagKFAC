from costom_modules.cnn import ResNetForCIFAR10, MLP
from .resnet_cifar import (ResNet18 as resnet18_cifar, ResNet34 as resnet34_cifar, ResNet50
as resnet50_cifar, ResNet101 as resnet101_cifar, ResNet152 as resnet152_cifar)
def create_model(args):
    if args.model == "resnet":
        model = ResNetForCIFAR10(args.layers)
    elif args.model == "mlp":
        model = MLP(hidden_size=128, num_hidden_layers=8)
    elif args.model == "resnet18Cifar":
        model = resnet18_cifar(num_classes=10)
    elif args.model == "resnet34Cifar":
        model = resnet34_cifar(num_classes=10)
    elif args.model == "resnet50Cifar":
        model = resnet50_cifar(num_classes=10)
    elif args.model == "resnet101Cifar":
        model = resnet101_cifar(num_classes=10)
    elif args.model == "resnet152Cifar":
        model = resnet152_cifar(num_classes=10)

    return model