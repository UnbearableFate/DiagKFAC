import os
import datetime
import argparse

"""
ompi_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', -1))
ompi_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', -1))

from mpi4py import MPI
if ompi_world_size == -1 or ompi_world_rank == -1:
    ompi_world_rank = MPI.COMM_WORLD.Get_rank()
    ompi_world_size = MPI.COMM_WORLD.Get_size()
"""

PROJECT_NAME = "FNGD"

PRIVATE_DATA_ROOT = { # for small datasets like cifar10
    'fern' : "/home/yu/data",
    'mac' : "/Users/unbearablefate/workspace/data",
    'pegasus' : "/work/NBB/yu_mingzhe/data",
    'miyabi' : "/work/xg24i002/x10041/data",
}

WORKSPACE_ROOT = {
    'fern' : "/home/yu/workspace/+" + PROJECT_NAME,
    'mac' : "/Users/unbearablefate/workspace/" + PROJECT_NAME,
    'pegasus' : "/work/NBB/yu_mingzhe/" + PROJECT_NAME,
    'miyabi' : "/work/xg24i002/x10041/" + PROJECT_NAME,
}

def merged_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Merged Training Parser", add_help=add_help)

    # ===== 来自第一个函数的参数 =====
    parser.add_argument("--data-path", default=None, type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu; default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu, total batch size = NGPU x batch_size")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--norm-weight-decay", default=None, type=float, help="weight decay for Normalization layers (default: None, same as --wd)")
    parser.add_argument("--bias-weight-decay", default=None, type=float, help="weight decay for bias parameters (default: None, same as --wd)")
    parser.add_argument("--transformer-embedding-decay", default=None, type=float, help="weight decay for transformer embeddings (default: None, same as --wd)")
    parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing")
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="learning rate scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="number of warmup epochs (default: 0)")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="warmup method (default: constant)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="learning rate warmup decay")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum learning rate (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./out", type=str, help="directory to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path to checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--cache-dataset", dest="cache_dataset", action="store_true", help="cache dataset for faster initialization")
    parser.add_argument("--sync-bn", dest="sync_bn", action="store_true", help="use synchronized batch normalization")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="only test the model")
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
    parser.add_argument("--amp", action="store_true", help="use torch.cuda.amp for mixed precision training")
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url for distributed training setup")
    parser.add_argument("--model-ema", action="store_true", help="enable exponential moving average for model parameters")
    parser.add_argument("--model-ema-steps", type=int, default=32, help="iterations between EMA updates (default: 32)")
    parser.add_argument("--model-ema-decay", type=float, default=0.99998, help="EMA decay factor (default: 0.99998)")
    parser.add_argument("--use-deterministic-algorithms", action="store_true", help="force the use of deterministic algorithms")
    parser.add_argument("--interpolation", default="bilinear", type=str, help="interpolation method (default: bilinear)")
    parser.add_argument("--val-resize-size", default=256, type=int, help="resize size for validation (default: 256)")
    parser.add_argument("--val-crop-size", default=224, type=int, help="central crop size for validation (default: 224)")
    parser.add_argument("--train-crop-size", default=224, type=int, help="random crop size for training (default: 224)")
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="maximum gradient norm (default: None)")
    parser.add_argument("--ra-sampler", action="store_true", help="use repeated augmentation in training")
    parser.add_argument("--ra-reps", default=3, type=int, help="number of repeated augmentations (default: 3)")
    parser.add_argument("--weights", default=None, type=str, help="weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="backend for transforms: PIL or tensor (case insensitive)")
    parser.add_argument("--use-v2", action="store_true", help="use V2 transforms")

    # ===== 增加第二个函数中独有的参数 =====
    parser.add_argument("--log-dir", default="./logs/torch_cifar10", help="TensorBoard / checkpoint directory")
    parser.add_argument("--checkpoint-format", default="checkpoint_{epoch}.pth.tar", help="checkpoint file format")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disable CUDA training")
    parser.add_argument("--seed", type=int, default=42, metavar="S", help="random seed (default: 42)")
    parser.add_argument("--layers", type=int, default=34, help="number of layers in ResNet (default: 18)")  # 新增
    parser.add_argument("--val-batch-size", type=int, default=128, help="validation batch size (default: 128)")
    parser.add_argument("--batches-per-allreduce", type=int, default=1, help="number of local batches before allreduce")
    parser.add_argument("--checkpoint-freq", type=int, default=10, help="frequency of saving checkpoints (in epochs)")
    
    # ------ KFAC 相关参数（仅出现在第二个函数中） ------
    parser.add_argument("--kfac-inv-update-steps", type=int, default=10, help="iters between KFAC inverse updates (0 disables KFAC)")
    parser.add_argument("--kfac-factor-update-steps", type=int, default=1, help="iters between KFAC covariance updates")
    parser.add_argument("--kfac-update-steps-alpha", type=float, default=10, help="KFAC update step multiplier")
    parser.add_argument("--kfac-update-steps-decay", nargs="+", type=int, default=None, help="KFAC update step decay schedule")
    parser.add_argument("--kfac-inv-method", action="store_true", default=False, help="use inverse KFAC update (instead of eigen)")
    parser.add_argument("--kfac-factor-decay", type=float, default=0.95, help="alpha for covariance accumulation (default: 0.95)")
    parser.add_argument("--kfac-damping", type=float, default=0.003, help="KFAC damping factor (default: 0.003)")
    parser.add_argument("--kfac-damping-alpha", type=float, default=0.5, help="KFAC damping decay factor (default: 0.5)")
    parser.add_argument("--kfac-damping-decay", nargs="+", type=int, default=None, help="KFAC damping decay schedule")
    parser.add_argument("--kfac-kl-clip", type=float, default=0.001, help="KL clip (default: 0.001)")
    parser.add_argument("--kfac-skip-layers", nargs="+", type=str, default=[], help="layer types to ignore for KFAC")
    parser.add_argument("--kfac-colocate-factors", action="store_true", default=True, help="compute A and G for a layer on the same worker")
    parser.add_argument("--kfac-strategy", type=str, default="comm-opt", help="KFAC communication strategy (comm-opt, mem-opt, or hybrid_opt)")
    parser.add_argument("--kfac-grad-worker-fraction", type=float, default=0.25, help="fraction of workers for gradient computation in HYBRID_OPT")
    
    # ------ 分布式和实验相关参数 ------
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--timestamp", type=str, default=datetime.datetime.now().strftime('%Y%m%d_%H%M'), help="experiment timestamp")
    parser.add_argument("--experiment-name", type=str, default="cifar10_resnet", help="experiment name")
    parser.add_argument("--recover", action="store_true", default=False, help="recover from checkpoint")
    parser.add_argument("--not-kfac", action="store_true", default=False, help="disable KFAC")
    parser.add_argument("--degree-noniid", type=float, default=0, help="degree of non-iid data distribution")
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=["imagenet", "cifar10", "fashionmnist"],
        help="dataset name (default: imagenet)",
    )

    parser.add_argument(
        "--system",
        default="fern",
        type=str,
        choices=["fern","mac", "pegasus", "miyabi"],
        help="system name (default: fern)",
    )

    # 处理环境变量中分布式的 LOCAL_RANK（若存在）
    args = parser.parse_args()
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if args.data_path is None and args.dataset in ["cifar10", "fashionmnist,minist"] and args.system in ["fern", "mac", "pegasus", "miyabi"]:
        args.data_path = os.path.join(PRIVATE_DATA_ROOT[args.system], args.dataset)
        if not os.path.exists(args.data_path):
            raise ValueError(f"Dataset path {args.data_path} does not exist. Please set the correct path.")

    # print args
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    return args