import datetime
import os
import time
import warnings

from trainning_kit import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
from trainning_kit  import utils
from .sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from .transforms import get_mixup_cutmix
from .data_preparation import DataPreparer
import torch.distributed as dist
from costom_modules.cnn import ResNetForCIFAR10, MLP, SimpleCNN

class Trainer:
    def __init__(self,args):
        self.init(args)
    
    def init(self, args):
        self.args = args
        if args.output_dir:
            utils.mkdir(args.output_dir)

        #utils.init_distributed_mode(args)
        args.distributed = False
        print(args)

        device = torch.device(args.device)
        self.device = device

        if args.use_deterministic_algorithms:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True

        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self.data_manager = DataPreparer(args,self.world_size,self.rank)

        print("Creating model")
        if args.model in torchvision.models.list_models():
            self.model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=self.data_manager.num_classes)
        else:
            self.model = create_model(args)
        self.model.to(device)

        if args.distributed and args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        self.init_optimizer(args)

        model_ema = None
        if args.model_ema:
            # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
            # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
            #
            # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
            # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
            # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
            adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
            alpha = 1.0 - args.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            model_ema = utils.ExponentialMovingAverage(self.model_without_ddp, device=device, decay=1.0 - alpha)

        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            self.model_without_ddp.load_state_dict(checkpoint["model"])
            if not args.test_only:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if model_ema:
                model_ema.load_state_dict(checkpoint["model_ema"])
            if self.scaler:
                self.scaler.load_state_dict(checkpoint["scaler"])

        self.model_ema = model_ema

        print(f"==== Training Configuration Summary ====")
        print(f"Model Type           : {type(self.model).__name__}")
        print(f"Total Parameters     : {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable Parameters : {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Device               : {self.device}")
        print(f"Use AMP              : {args.amp}")
        print(f"Loss Function        : {type(self.criterion).__name__}")
        print(f"Optimizer            : {type(self.optimizer).__name__}")
        print(f"  - Learning Rate    : {self.optimizer.param_groups[0]['lr']}")
        print(f"  - Weight Decay     : {self.optimizer.param_groups[0]['weight_decay']}")
        print(f"Scheduler            : {type(self.lr_scheduler).__name__}")
        print(f"Batch Size           : {args.batch_size}")
        print(f"Epochs               : {args.epochs}")
        print(f"Distributed          : {args.distributed}")
        print(f"EMA Model Used       : {'Yes' if self.model_ema else 'No'}")
        print(f"Output Directory     : {args.output_dir}")
        print(f"Dataset Size        : {len(self.data_manager.train_loader.dataset)}")
        print(f"Number of Classes    : {self.data_manager.num_classes}")
        print("=" * 60)

    def init_optimizer(self, args):
        self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        custom_keys_weight_decay = []
        if args.bias_weight_decay is not None:
            custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
        if args.transformer_embedding_decay is not None:
            for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
                custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
        parameters = utils.set_weight_decay(
            self.model,
            args.weight_decay,
            norm_weight_decay=args.norm_weight_decay,
            custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
        )

        opt_name = args.opt.lower()
        if opt_name.startswith("sgd"):
            optimizer = torch.optim.SGD(
                parameters,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov="nesterov" in opt_name,
            )
        elif opt_name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

        scaler = torch.amp.GradScaler("cuda") if args.amp else None

        args.lr_scheduler = args.lr_scheduler.lower()
        if args.lr_scheduler == "steplr":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
            )
        elif args.lr_scheduler == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                "are supported."
            )

        if args.lr_warmup_epochs > 0:
            if args.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            elif args.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            else:
                raise RuntimeError(
                    f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
                )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler

        self.model_without_ddp = self.model
        if args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model_without_ddp = self.model.module
        
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler




    def train_and_evaluate(self):
        args = self.args
        train_sampler = self.data_manager.train_sampler
        optimizer = self.optimizer
        model_ema = self.model_ema
        scaler = self.scaler
        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            self.train_one_epoch(epoch, args)
            self.lr_scheduler.step()
            self.evaluate()
            if model_ema:
                self.evaluate(log_suffix="EMA")
            if args.output_dir:
                checkpoint = {
                    "model": self.model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                if model_ema:
                    checkpoint["model_ema"] = model_ema.state_dict()
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")
    
    def test_only(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if self.model_ema:
            self.evaluate(log_suffix="EMA")
        else:
            self.evaluate()
        return

    def evaluate(self, print_freq=100, log_suffix=""):
        criterion = self.criterion
        data_loader = self.data_manager.test_loader
        model = self.model
        device = self.device
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = f"Test: {log_suffix}"

        num_processed_samples = 0
        with torch.inference_mode():
            for image, target in metric_logger.log_every(data_loader, print_freq, header):
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(image)
                loss = criterion(output, target)

                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                # FIXME need to take into account that the datasets
                # could have been padded in distributed setup
                batch_size = image.shape[0]
                metric_logger.update(loss=loss.item())
                metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
                metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
                num_processed_samples += batch_size
        # gather the stats from all processes

        num_processed_samples = utils.reduce_across_processes(num_processed_samples)
        if (
            hasattr(data_loader.dataset, "__len__")
            and len(data_loader.dataset) != num_processed_samples
            and torch.distributed.get_rank() == 0
        ):
            # See FIXME above
            warnings.warn(
                f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
                "samples were used for the validation, which might bias the results. "
                "Try adjusting the batch size and / or the world size. "
                "Setting the world size to 1 is always a safe bet."
            )

        metric_logger.synchronize_between_processes()

        print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
        return metric_logger.acc1.global_avg
    
    def train_one_epoch(self,epoch, args):
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        data_loader = self.data_manager.train_loader
        device = self.device
        model_ema = self.model_ema
        scaler = self.scaler
        
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

        header = f"Epoch: [{epoch}]"
        for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            start_time = time.time()
            image, target = image.to(device), target.to(device)
            with torch.amp.autocast('cuda'):
                output = model(image)
                loss = criterion(output, target)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            if model_ema and i % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < args.lr_warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def create_model(args):
    if args.model == "resnet":
        model = ResNetForCIFAR10(args.layers)
    elif args.model == "mlp":
        model = MLP(hidden_size=64, num_hidden_layers=8)

    return model