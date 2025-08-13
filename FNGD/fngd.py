from trainning_kit.trainer import Trainer, utils
import time
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap
from backpack import extend,backpack
from backpack.extensions import BatchGrad


class FngdTrainer (Trainer):
    def __init__(self, args ,model=None):
        super().__init__(args, model=model)
        if args.backpack:
            self.model = extend(self.model, use_converter=True)
            self.criterion = extend(self.criterion)
            self.train_function = self.train_one_epoch_backpack
        else:
            self.train_function = self.train_one_epoch_torch
        self.precondition_coeffs = {}
        self.damping = args.kfac_damping
    
    def train_one_epoch_backpack(self, epoch):
        args = self.args
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

        header = f"backpack_Epoch: [{epoch}]"
        for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            start_time = time.time()
            # Move data to device
            image, target = image.to(device), target.to(device)
            
            # Optionally use AMP for forward pass (backward will be computed in full precision)
            output = model(image)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            # Use Backpack to compute per-sample gradients
            # 注意：确保提前已安装并导入了 backpack。这里我们动态导入。
           
            with backpack(BatchGrad()):
                loss.backward()
            
            for name, param in model.named_parameters():
                if hasattr(param, "grad_batch"):
                    # 计算本层 per-sample 梯度矩阵 U
                    # 假设 grad_batch 的形状为 (M, ...) ，其中 M 是 batch 大小，
                    # 将除 batch 维度外的部分展平得到维数 N，然后 U 的形状就是 (N, M)
                    M = param.grad_batch.shape[0]
                    U = param.grad_batch.reshape(M, -1).transpose(0, 1)  # shape: (N, M)
                    
                    # 使用每一层独立的 coefficient，保存时 key 为 "coefficient_" + name
                    coeff_key = f"coefficient_{name}"
                    if epoch == args.start_epoch:
                        coeff = self.compute_coefficient(U)
                        if coeff_key in self.precondition_coeffs:
                            # 如果已经计算过，则计算当前epoch的平均
                            old_coeff = self.precondition_coeffs[coeff_key]
                            coeff = (old_coeff * i + coeff) / (i + 1)
                        self.precondition_coeffs[coeff_key] = coeff
                    else:
                        coeff = self.precondition_coeffs.get(coeff_key)
                        if coeff is None:
                            raise ValueError(f"Coefficient for {name} not found. Please run the first epoch to compute it.")
                    
                    # 利用计算好的系数，对本层的梯度进行预调节：
                    # 根据公式：pre_grad = (1/(damping * M)) * U @ v, 然后 reshape 成参数原始形状
                    pre_grad = self.precondition_gradient(U, coeff).reshape(param.grad_batch.shape[1:])
                    param.grad = pre_grad
                    # 清除 grad_batch 以便下一次迭代
                    del param.grad_batch

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
        if epoch == args.start_epoch:
            print(f"Precondition coefficients: {self.precondition_coeffs.keys()}")

    def train_one_epoch_torch(self, epoch):
        args = self.args
        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        data_loader = self.data_manager.train_loader
        device = self.device
        model_ema = self.model_ema

        model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
        metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

        header = f"torch_Epoch: [{epoch}]"
        for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
            start_time = time.time()
            image, target = image.to(device), target.to(device)

            with torch.no_grad():
                output = model(image)
                loss = criterion(output, target)

            params = dict(model.named_parameters())

            model.eval()  # Set once per batch
            def sample_loss(params, x, y):
                output = functional_call(model, params, x.unsqueeze(0))
                return criterion(output, y.unsqueeze(0))

            per_sample_grad_fn = vmap(grad(sample_loss), in_dims=(None, 0, 0))

            with torch.no_grad():
                per_sample_grads = per_sample_grad_fn(params, image, target)

            def process_grad(name, grad_tensor):
                M = grad_tensor.shape[0]
                U = grad_tensor.reshape(M, -1).transpose(0, 1)

                coeff_key = f"coefficient_{name}"
                if epoch == args.start_epoch:
                    coeff = self.compute_coefficient(U)
                    if coeff_key in self.precondition_coeffs:
                        old_coeff = self.precondition_coeffs[coeff_key]
                        coeff = (old_coeff * i + coeff) / (i + 1)
                    self.precondition_coeffs[coeff_key] = coeff
                else:
                    coeff = self.precondition_coeffs.get(coeff_key)
                    if coeff is None:
                        raise ValueError(f"Coefficient for {name} not computed in first epoch.")

                pre_grad = self.precondition_gradient(U, coeff).reshape(grad_tensor.shape[1:])
                return pre_grad

            for name, param in model.named_parameters():
                if name in per_sample_grads:
                    param.grad = process_grad(name, per_sample_grads[name])

            del per_sample_grads, params
            torch.cuda.empty_cache()  # 清理显存

            optimizer.step()

            if model_ema and i % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < args.lr_warmup_epochs:
                    model_ema.n_averaged.fill_(0)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time)) 

        self.write_training_summary(epoch,start_time , metric_logger)

    def compute_coefficient(self, U):
        """计算并返回系数向量 v，根据公式：
           v = 1 - (damping * I + (1/M)*(U^T U))^{-1} * (U^T * g),
           U 的形状为 (N, M)，N 为展开后参数维数，M 为 batch size。
        """
        damping = self.damping
        M = U.shape[1]
        I_M = torch.eye(M, device=U.device, dtype=U.dtype)
        UtU =  U.t() @ U
        A = damping * I_M + (1.0 / M) * UtU # shape: (M, M)
        A_inv = torch.inverse(A)
        ones = torch.ones((M, 1), device=U.device, dtype=U.dtype)
        v = (ones - A_inv @ UtU.mean(dim=1, keepdim=True)) / (damping * M)  # shape: (M, 1)
        return v
    
    def precondition_gradient(self, U, v ):
        """根据预先计算好的系数 v 和 per-sample 梯度矩阵 U，计算预调节后的梯度。
           U 的形状为 (N, M)，返回预调节梯度形状为 (N, 1)，其中 N 为展开后参数维数。
           根据公式：pre_grad = (1/(damping*M)) * U @ v
        """
        pre_grad =  U @ v  # shape: (N, 1)
        return pre_grad
