import datetime
import sys
import os
import torch.distributed as dist
from trainning_kit.yaml_config_parser import merged_args_parser
from trainning_kit.trainer import Trainer

args = merged_args_parser()
print(f"Parsed arguments: {args}")
timeout = datetime.timedelta(seconds=30)
dist.init_process_group(backend='nccl', timeout=timeout)
print("Distributed process group initialized.")
print(f"World Size: {dist.get_world_size()}, World Rank: {dist.get_rank()} hostname {os.uname().nodename}")
if dist.get_rank() == 0:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

trainer = Trainer(args)
dist.barrier()  # 确保所有进程在开始前同步
trainer.train_and_evaluate()
dist.destroy_process_group()
trainer.print_after_train()
exit(0)
