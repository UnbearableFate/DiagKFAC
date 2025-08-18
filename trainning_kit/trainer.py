import datetime
import math
import os
import time
from typing import Callable
import warnings

from costom_modules.image_classification import get_network as create_image_classification_model
import kfac
from optimizers.AdaFisher import AdaFisher
from trainning_kit import presets
import torch
from trainning_kit  import utils
from .sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from .transforms import get_mixup_cutmix
from .data_preparation import DataPreparer
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from .common import WORKSPACE_ROOT
from kfac.dia_kfac.preconditioner import DiagKFACPreconditioner

class Trainer:
    def __init__(self,args ,model=None):
        self.init(args , model)

    def init(self, args , model):
        self.args = args
        
        if dist.is_initialized() and dist.get_world_size() > 1:
            args.distributed = True
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            utils.setup_for_distributed(self.rank == 0)
        else:
            args.distributed = False
            self.world_size = 1
            self.rank = 0

        if args.output_dir and self.rank == 0:
            utils.mkdir(args.output_dir)

        device = torch.device(args.device)
        self.device = device

        if args.use_deterministic_algorithms:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True

        self.data_manager = DataPreparer(args,self.world_size,self.rank)

        if model is None:
            self.model = create_image_classification_model(args.model, num_classes=self.data_manager.num_classes)
        else:
            self.model = model
        
        self.model.to(device)

        if args.distributed and args.sync_bn:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.model_without_ddp = self.model
        if args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model_without_ddp = self.model.module
        
        self.scaler = torch.amp.GradScaler("cuda") if args.amp else None

        self.optimizer, self.lr_scheduler = self.init_optimizer(args)
        if args.preconditioner in["kfac","diag_kfac"]:
            self.preconditioner, self.preconditioner_scheduler = self.init_preconditioner(args)
        else:
            self.preconditioner = None
            self.preconditioner_scheduler = None

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

        if self.rank == 0:
            self.summary_writer = self.init_logger(args)
        else:
            self.summary_writer = None

        self.train_function = self.train_one_epoch
        self.train_total_time = 0
    
    def init_logger(self ,args):
        model_name = type(self.model).__name__ if not self.args.distributed else self.model_without_ddp.__class__.__name__
        log_dir  = os.path.join(args.workspace_path,"logs",
                                f"{model_name}_{args.dataset}",
                                f"{args.opt}_{args.preconditioner}_{args.experiment_name}",
                                args.timestamp)
        summary_writer = SummaryWriter(log_dir) 
        
        print(f"==== Training Configuration Summary ====")
        config_lines = []
        config_lines.append(f"Model Type           : {type(self.model).__name__ if not self.args.distributed else self.model_without_ddp.__class__.__name__}")
        config_lines.append(f"Total Parameters     : {sum(p.numel() for p in self.model.parameters()):,}")
        config_lines.append(f"Trainable Parameters : {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        config_lines.append(f"Device               : {self.device}")
        config_lines.append(f"Use AMP              : {args.amp}")
        config_lines.append(f"Loss Function        : {type(self.criterion).__name__}")
        config_lines.append(f"Optimizer            : {type(self.optimizer).__name__}")
        config_lines.append(f"Preconditioner       : {str(self.preconditioner) if self.preconditioner else 'None'}")
        config_lines.append(f"  - Learning Rate    : {args.lr}")
        config_lines.append(f"  - Weight Decay     : {args.weight_decay}")
        config_lines.append(f"Scheduler            : {type(self.lr_scheduler).__name__}")
        config_lines.append(f"Batch Size           : {args.batch_size}")
        config_lines.append(f"Epochs               : {args.epochs}")
        config_lines.append(f"Distributed          : {args.distributed}")
        config_lines.append(f"EMA Model Used       : {'Yes' if self.model_ema else 'No'}")
        config_lines.append(f"Output Directory     : {args.output_dir}")
        config_lines.append(f"Dataset Size         : {len(self.data_manager.train_loader.dataset)}")
        config_lines.append(f"Number of Classes    : {self.data_manager.num_classes}")
        config_lines.append(f"World Size           : {self.world_size}")
        config_lines.append(f"Rank                 : {self.rank}")
        config_lines.append("=" * 60)

        for line in config_lines:
            print(line)
        # 写入到log_dir下的log.txt
        config_path = os.path.join(log_dir, "log.txt")
        with open(config_path, "w") as f:
            for line in config_lines:
                f.write(line + "\n")
            f.write("\n==== Arguments ====\n")
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")

        self.config_path = config_path
        return summary_writer

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

        """
        if self.world_size > 1:
            args.lr = args.lr * math.sqrt(self.world_size)
            args.lr_min = args.lr_min * math.sqrt(self.world_size)
        """

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
        elif opt_name == 'adafisher':
            optimizer = AdaFisher(
                self.model,
                lr=args.lr,
                weight_decay=args.weight_decay,
                dist= args.distributed,
            )
        else:
            raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

        args.lr_scheduler = args.lr_scheduler.lower()
        if args.lr_scheduler == "steplr":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
            )
        elif args.lr_scheduler == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
        elif args.lr_scheduler == "onecycle":
            main_lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr,
                total_steps=args.epochs * len(self.data_manager.train_loader),
                pct_start=args.pct_start,
                cycle_momentum=False if opt_name == "adafisher" else True
            )
            args.lr_warmup_epochs = 0
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

        return optimizer, lr_scheduler

    def init_preconditioner(self,args):
        if args.preconditioner == "kfac":
            preconditioner = kfac.preconditioner.KFACPreconditioner(
                self.model,
                factor_update_steps=args.kfac_factor_update_steps,
                inv_update_steps=args.kfac_inv_update_steps,
                damping=args.kfac_damping,
                factor_decay=args.kfac_factor_decay,
                kl_clip=args.kfac_kl_clip,
                lr=lambda x: self.optimizer.param_groups[0]['lr'],
                accumulation_steps=args.batches_per_allreduce,
                allreduce_bucket_cap_mb=25,
                colocate_factors=args.kfac_colocate_factors,
                compute_method=kfac.enums.ComputeMethod.EIGEN,
                grad_worker_fraction=kfac.enums.DistributedStrategy.COMM_OPT,
                grad_scaler=self.scaler.get_scale if self.scaler else None,
                skip_layers=args.kfac_skip_layers,
                compute_eigenvalue_outer_product=False,
            )
        elif args.preconditioner == "diag_kfac":
            preconditioner = DiagKFACPreconditioner(
                self.model,
                factor_update_steps=args.kfac_factor_update_steps,
                inv_update_steps=args.kfac_inv_update_steps,
                damping=args.kfac_damping,
                factor_decay=args.kfac_factor_decay,
                kl_clip=args.kfac_kl_clip,
                lr=lambda x: self.optimizer.param_groups[0]['lr'],
                accumulation_steps=args.batches_per_allreduce,
                allreduce_bucket_cap_mb=25,
                colocate_factors=args.kfac_colocate_factors,
                compute_method=kfac.enums.ComputeMethod.EIGEN,
                grad_worker_fraction=kfac.enums.DistributedStrategy.COMM_OPT,
                grad_scaler=self.scaler.get_scale if self.scaler else None,
                skip_layers=args.kfac_skip_layers,
                compute_eigenvalue_outer_product=False,
                split_num=4,
            )

        def get_lambda(
            alpha: int,
            epochs: list[int] | None,
        ) -> Callable[[int], float]:
            """Create lambda function for param scheduler."""
            if epochs is None:
                _epochs = []
            else:
                _epochs = epochs

            def scale(epoch: int) -> float:
                """Compute current scale factor using epoch."""
                factor = 1.0
                for e in _epochs:
                    if epoch >= e:
                        factor *= alpha
                return factor

            return scale

        kfac_param_scheduler = kfac.scheduler.LambdaParamScheduler(
            preconditioner,
            damping_lambda=get_lambda(
                args.kfac_damping_alpha,
                args.kfac_damping_decay,
            ),
            factor_update_steps_lambda=get_lambda(
                args.kfac_update_steps_alpha,
                args.kfac_update_steps_decay,
            ),
            inv_update_steps_lambda=get_lambda(
                args.kfac_update_steps_alpha,
                args.kfac_update_steps_decay,
            ),
        )
        return preconditioner, kfac_param_scheduler

    def train_and_evaluate(self):
        args = self.args
        train_sampler = self.data_manager.train_sampler
        optimizer = self.optimizer
        model_ema = self.model_ema
        scaler = self.scaler
        print(f"Start training at {datetime.datetime.now()} at rank {self.rank}")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            epoch_start_time = time.time()
            self.train_function(epoch)
            self.train_total_time += time.time() - epoch_start_time
            if args.lr_scheduler != "onecycle":
                self.lr_scheduler.step()
            self.evaluate(epoch=epoch)
            if model_ema:
                self.evaluate(epoch=epoch,log_suffix="EMA")
            if args.output_dir:
                checkpoint = {
                    "model": self.model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "preconditioner": self.preconditioner.state_dict() if self.preconditioner else None,
                    "timestamp": self.args.timestamp,
                    "training_time": self.train_total_time,
                    "epoch": epoch,
                    "args": args,
                }
                if model_ema:
                    checkpoint["model_ema"] = model_ema.state_dict()
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                #utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"checkpoint_{self.args.timestamp}.pth"))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str} ({total_time / args.epochs:.2f} s / epoch) at rank {self.rank}")

    def test_only(self):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if self.model_ema:
            self.evaluate(log_suffix="EMA")
        else:
            self.evaluate()
        return

    def evaluate(self,epoch=0, print_freq=100, log_suffix=""):
        criterion = self.criterion
        data_loader = self.data_manager.test_loader
        model = self.model
        device = self.device
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ", rank=self.rank)
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
        
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                f"Test/acc1{log_suffix}_vs_epoch", metric_logger.acc1.global_avg, epoch
            )
            self.summary_writer.add_scalar(
                f"Test/acc5{log_suffix}_vs_epoch", metric_logger.acc5.global_avg, epoch
            )
            self.summary_writer.add_scalar(
                f"Test/acc1{log_suffix}_vs_time", metric_logger.acc1.global_avg, self.train_total_time
            )
            self.summary_writer.add_scalar(
                f"Test/acc5{log_suffix}_vs_time", metric_logger.acc5.global_avg, self.train_total_time
            )
            print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
        
        return metric_logger.acc1.global_avg
    
    def train_one_epoch(self,epoch):
        args = self.args
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        data_loader = self.data_manager.train_loader
        device = self.device
        model_ema = self.model_ema
        scaler = self.scaler
        
        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ",rank=self.rank)
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

        # Additional monitoring meters
        if self.scaler is not None:
            metric_logger.add_meter("loss_scale", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("grad_norm", utils.SmoothedValue(window_size=1, fmt="{value}"))

        # Track AMP overflow events by observing loss_scale decreases
        self._amp_overflow_count = 0
        _prev_loss_scale = float(self.scaler.get_scale()) if self.scaler is not None else None

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
                if self.preconditioner is not None:
                    self.preconditioner.step()
                    self.preconditioner_scheduler.step()

                # Unscale before any grad norm computation / clipping
                scaler.unscale_(optimizer)
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                # Compute grad norm (L2) safely
                with torch.no_grad():
                    sq_sum = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            sq_sum += float(p.grad.detach().data.norm(2).item() ** 2)
                    grad_norm_val = math.sqrt(sq_sum) if sq_sum > 0.0 else 0.0

                scaler.step(optimizer)
                scaler.update()

                # Log current loss scale and detect overflow via scale drop
                current_scale = float(self.scaler.get_scale())
                metric_logger.update(loss_scale=current_scale)
                if _prev_loss_scale is not None and current_scale < _prev_loss_scale:
                    self._amp_overflow_count += 1
                _prev_loss_scale = current_scale
            else:
                loss.backward()
                if self.preconditioner is not None:
                    self.preconditioner.step()
                    self.preconditioner_scheduler.step()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                # Compute grad norm (L2) safely
                with torch.no_grad():
                    sq_sum = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            sq_sum += float(p.grad.detach().data.norm(2).item() ** 2)
                    grad_norm_val = math.sqrt(sq_sum) if sq_sum > 0.0 else 0.0

                optimizer.step()

            if model_ema and i % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < args.lr_warmup_epochs:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)

            if self.args.lr_scheduler == "onecycle":
                self.lr_scheduler.step()

            acc1 = utils.accuracy(output, target, topk=(1,))
            batch_size = image.shape[0]
            metric_logger.update(
                loss=loss.item(),
                lr=optimizer.param_groups[0]["lr"],
            )
            metric_logger.meters["acc1"].update(acc1[0].item(), n=batch_size)
            metric_logger.meters["grad_norm"].update(grad_norm_val)
            metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        self.write_training_summary(epoch, metric_logger)

    def write_training_summary(self, epoch, metric_logger):
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(
                f"Train/loss", metric_logger.loss.global_avg, epoch
            )
            self.summary_writer.add_scalar(
                f"Train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            # Training accuracy
            if hasattr(metric_logger, "acc1"):
                self.summary_writer.add_scalar(
                    f"Train/acc1", metric_logger.acc1.global_avg, epoch
                )

            # Grad norm (L2)
            if "grad_norm" in metric_logger.meters:
                self.summary_writer.add_scalar(
                    f"Train/grad_norm", metric_logger.meters["grad_norm"].global_avg, epoch
                )

            # Loss scale (AMP)
            if self.scaler is not None and "loss_scale" in metric_logger.meters:
                self.summary_writer.add_scalar(
                    f"Train/loss_scale", metric_logger.meters["loss_scale"].global_avg, epoch
                )

            # Optimizer momentum (if present)
            momentum = self.optimizer.param_groups[0].get("momentum", None)
            if momentum is not None:
                self.summary_writer.add_scalar(
                    f"Train/momentum", float(momentum), epoch
                )

            # AMP overflow events counted in this epoch
            if hasattr(self, "_amp_overflow_count"):
                self.summary_writer.add_scalar(
                    f"Train/amp_overflow_count", int(self._amp_overflow_count), epoch
                )

            # Sample BatchNorm running stats from the first BN module found
            bn_module = None
            model_ref = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            for m in model_ref.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    bn_module = m
                    break
            if bn_module is not None and hasattr(bn_module, 'running_mean') and hasattr(bn_module, 'running_var'):
                try:
                    rm = bn_module.running_mean.detach().abs().mean().item()
                    rv = bn_module.running_var.detach().mean().item()
                    self.summary_writer.add_scalar(f"Train/bn0_running_mean_abs", rm, epoch)
                    self.summary_writer.add_scalar(f"Train/bn0_running_var", rv, epoch)
                except Exception:
                    pass

            self.summary_writer.add_scalar(f"Train/max_allocate_memory", torch.cuda.max_memory_allocated() / 1024.0 / 1024.0, epoch)
            self.summary_writer.add_scalar(f"Train/img_per_second", metric_logger.meters["img/s"].global_avg, epoch)