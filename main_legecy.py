from copy import deepcopy
import datetime
import os
import trainning_kit.common as common
from trainning_kit.trainer import Trainer
import torch.distributed as dist
args = common.merged_args_parser()
timeout = datetime.timedelta(seconds=40)

dist.init_process_group(backend='nccl',timeout=timeout)
print("Distributed process group initialized.")
print(f"World size: {dist.get_world_size()}, Rank: {dist.get_rank()}")
if dist.get_rank() == 0:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mgr = Trainer(args)
dist.barrier()  # Ensure all processes are synchronized before starting training
mgr.train_and_evaluate()
dist.destroy_process_group() 